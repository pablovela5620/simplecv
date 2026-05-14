"""Batch-convert annotated ExoEgo sequences to RRD files.

Worker entrypoint is module-level so it can be re-imported by
``multiprocessing`` spawn workers without pickling closures.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from timeit import default_timer as timer

# Before workers import numpy: cap BLAS / OMP thread pools so 8+ processes
# don't all spin up 32-thread thread pools and thrash the kernel scheduler.
# Setting these in the parent propagates to ``spawn``-mode workers because
# environment is inherited at fork()/spawn time.
_BLAS_THREADS_DEFAULT: str = os.environ.get("SIMPLECV_BLAS_THREADS", "2")
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_var, _BLAS_THREADS_DEFAULT)

import numpy as np  # noqa: E402  - must come AFTER the env-var setup above
import rerun as rr  # noqa: E402
from tqdm import tqdm  # noqa: E402

from simplecv.apis.view_exoego import VisualizeConfig, visualize_exo_ego
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exoego.base_exoego import BaseExoEgoDatasetConfig, BaseExoEgoSequence
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.rerun_log_utils import RerunTyroConfig

np.set_printoptions(suppress=True)


@dataclass
class BatchConvertConfig:
    """Configuration for batch converting annotated sequences to RRD files."""

    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset factory capable of producing the annotated ``BaseExoEgoSequence``."""
    rrd_save_dir: Path = Path("data/exoego-forge-catalog")
    """Output directory that will receive the generated ``.rrd`` files. For
    long-running ingestion runs, ``/mnt/8tb/data/exoego-forge-catalog`` has
    more storage and is the preferred location on the workstation."""
    max_conversions: int | None = 5
    """Optional cap on how many sequences to convert; ``None`` processes all available episodes."""
    dry_run: bool = False
    """When ``True``, only report the sequences that would be converted without writing files."""
    force: bool = False
    """When ``True``, overwrite existing ``.rrd`` files instead of skipping them."""
    num_workers: int = 1
    """When >1, dispatch per-sequence ingestion to a process pool of this size.
    Sequences are independent so this scales nearly linearly until the box
    runs out of cores or memory."""
    log_labels: bool = True
    """Whether ``visualize_exo_ego`` runs the heavy ``log_exoego_batch``
    step that emits 3D keypoint streams (``coco133_xyz``) and per-camera
    2D projection streams (``coco133_uv``, ``KeypointConfidence``,
    pinhole intrinsics). These DO live in the GT catalog at
    ``data/exoego-forge-catalog/assembly101/all/`` — they're stored in
    the file's *blueprint* substore rather than the recording substore,
    which is why an earlier version of this tool incorrectly defaulted
    to ``False`` (the validator only inspected the recording store and
    missed ~217 k rows of GT keypoint data per sequence). ``True`` is
    the correct default; flip to ``False`` only if you've explicitly
    decided to ship keypoint-free RRDs."""


def _estimate_job_size(seq_cfg: BaseExoEgoDatasetConfig) -> int:
    """Best-effort heuristic for ordering jobs longest-first.

    Falls back to 0 (so order is preserved) when we can't easily estimate
    the work from disk; for Assembly101 we sum the input MP4 sizes which
    correlates almost linearly with downstream wall time.
    """
    sequence_name: str | None = getattr(seq_cfg, "sequence_name", None)
    root_directory: Path | None = getattr(seq_cfg, "root_directory", None)
    if sequence_name and root_directory is not None:
        candidate: Path = Path(root_directory) / "videos" / "av1-720-new" / sequence_name
        if candidate.is_dir():
            return sum(p.stat().st_size for p in candidate.iterdir() if p.is_file())
    return 0


def _pin_to_core_subset(worker_index: int, worker_count: int) -> None:
    """Pin the worker process to a disjoint subset of CPU cores.

    With many workers all sharing all cores, the kernel scheduler bounces
    threads, hurts cache locality, and amplifies BLAS contention. Pinning
    each worker to a 4-core slice (32 cores // 8 workers = 4 each) keeps
    each worker's threads sticky.
    """
    try:
        all_cores: set[int] = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return
    cores_sorted: list[int] = sorted(all_cores)
    if not cores_sorted or worker_count <= 0:
        return
    slice_size: int = max(len(cores_sorted) // worker_count, 1)
    start: int = (worker_index * slice_size) % len(cores_sorted)
    end: int = min(start + slice_size, len(cores_sorted))
    subset: set[int] = set(cores_sorted[start:end])
    if subset:
        try:
            os.sched_setaffinity(0, subset)
        except (AttributeError, OSError):
            pass


def _worker_init(worker_index: int, worker_count: int) -> None:
    _pin_to_core_subset(worker_index, worker_count)


# ``_INIT_INDEX_COUNTER`` is bumped in the parent and read by ``_pool_init``
# inside each worker on startup. Plain global is fine because each spawn
# worker re-imports this module from scratch.
_INIT_INDEX_COUNTER: list[int] = [0]


def _pool_init(worker_count: int) -> None:
    """ProcessPoolExecutor ``initializer`` hook. Assigns a CPU-core slice."""
    # Workers are spawned sequentially by the parent; this counter advances
    # within the WORKER process and starts at 0 because each spawn re-imports
    # the module. We can't share state easily across workers, so we just pin
    # to a slice keyed by os.getpid() % worker_count which is good enough.
    index: int = os.getpid() % max(worker_count, 1)
    _pin_to_core_subset(index, worker_count)


def _process_one_sequence(
    seq_cfg: BaseExoEgoDatasetConfig,
    rrd_save_path: Path,
    log_labels: bool = False,
) -> tuple[str, float]:
    """Build one ``BaseExoEgoSequence`` and write its RRD. Runs in a worker.

    Returning ``(sequence_label, wall_seconds)`` keeps the parent's logging
    tidy without pickling any heavy objects back across the pipe.
    """
    t0: float = timer()
    # If labels won't be emitted, skip ``load_labels`` entirely in the
    # dataset adapter — for Assembly101 this saves the ~0.3 s/seq spent
    # parsing the 96 MB landmarks JSON and building the
    # (n_frames, 133, 4) keypoint stack.
    if not log_labels and getattr(seq_cfg, "load_labels", False):
        from dataclasses import replace as dataclass_replace

        seq_cfg = dataclass_replace(seq_cfg, load_labels=False)
    sequence: BaseExoEgoSequence = seq_cfg.setup()
    identity: SequenceIdentity = sequence.sequence_identity
    rrd_save_path.parent.mkdir(parents=True, exist_ok=True)
    vc = VisualizeConfig(
        rr_config=RerunTyroConfig(
            application_id="exoego-forge",
            recording_id=identity.recording_id,
            save=rrd_save_path,
        ),
        dataset=seq_cfg,
        log_labels=log_labels,
    )
    rec: rr.RecordingStream = vc.rr_config.rec_stream
    rr.send_recording_name(identity.sequence_key, recording=rec)
    rec.send_property(
        "info",
        rr.AnyValues(
            sequence_key=identity.sequence_key,
            num_frames=len(sequence),
            has_ego=sequence.ego_sequence is not None,
            has_exo=sequence.exo_sequence is not None,
        ),
    )
    visualize_exo_ego(sequence, vc)
    return identity.sequence_key, timer() - t0


def _plan_sequences(
    config: BatchConvertConfig,
) -> list[tuple[BaseExoEgoDatasetConfig, Path, SequenceIdentity]]:
    """Walk the dataset and return per-sequence (config, rrd_path, identity).

    Uses the dataset's ``iter_sequence_configs`` classmethod when available
    so we don't pay the per-sequence construction cost twice (once here,
    once in the worker).
    """
    sequence_cls = config.dataset._target
    plan: list[tuple[BaseExoEgoDatasetConfig, Path, SequenceIdentity]] = []
    for idx, seq_cfg in enumerate(sequence_cls.iter_sequence_configs(config.dataset)):
        identity: SequenceIdentity = sequence_cls.sequence_identity_for_config(seq_cfg)
        rrd_save_path: Path = identity.rrd_path(config.rrd_save_dir)
        plan.append((seq_cfg, rrd_save_path, identity))
        if config.max_conversions is not None and idx + 1 >= config.max_conversions:
            break
    return plan


def main(config: BatchConvertConfig):
    start_time: float = timer()
    plan: list[tuple[BaseExoEgoDatasetConfig, Path, SequenceIdentity]] = _plan_sequences(config)

    # Duplicate-id guard works the same in both paths.
    seen_recording_ids: set[str] = set()
    seen_output_paths: set[Path] = set()
    runnable: list[tuple[BaseExoEgoDatasetConfig, Path, SequenceIdentity]] = []
    for seq_cfg, rrd_save_path, identity in plan:
        resolved_rrd_save_path: Path = rrd_save_path.expanduser().resolve()
        if identity.recording_id in seen_recording_ids:
            raise ValueError(f"Duplicate recording_id while converting sequences: {identity.recording_id}")
        if resolved_rrd_save_path in seen_output_paths:
            raise ValueError(f"Duplicate RRD output path while converting sequences: {rrd_save_path}")
        seen_recording_ids.add(identity.recording_id)
        seen_output_paths.add(resolved_rrd_save_path)

        if rrd_save_path.exists() and not config.force:
            tqdm.write(f"[skip-existing] {identity.sequence_key} -> {rrd_save_path}")
            continue
        if config.dry_run:
            tqdm.write(f"[dry-run] {identity.sequence_key} -> {rrd_save_path}")
            continue
        runnable.append((seq_cfg, rrd_save_path, identity))

    if config.num_workers > 1 and len(runnable) > 1:
        # ``spawn`` only — both ``fork`` and ``forkserver`` inherit rerun
        # module state from the parent and lose the per-worker
        # ``rr.Transform3D`` send_columns writes on a subset of sequences
        # (verified ~7/10 fails on forkserver, ~4/10 on fork).
        ctx = get_context("spawn")
        worker_count: int = min(config.num_workers, len(runnable))
        # Submit longest-running jobs first (estimated by source-data size) so
        # the heaviest sequence starts at t=0 instead of queueing behind smaller
        # jobs. ``Longest Processing Time First`` heuristic. Falls back to
        # natural order when no size estimate is available.
        runnable_sorted: list[tuple[BaseExoEgoDatasetConfig, Path, SequenceIdentity]] = sorted(
            runnable, key=lambda item: -_estimate_job_size(item[0])
        )
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=_pool_init,
            initargs=(worker_count,),
        ) as pool:
            futures = {
                pool.submit(_process_one_sequence, seq_cfg, rrd_save_path, config.log_labels): identity
                for seq_cfg, rrd_save_path, identity in runnable_sorted
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing sequences ({worker_count} workers)",
            ):
                identity = futures[fut]
                # Re-raise worker exceptions in the parent.
                key, secs = fut.result()
                tqdm.write(f"[done] {key} in {secs:.2f}s")
    else:
        for seq_cfg, rrd_save_path, identity in tqdm(runnable, desc="Processing sequences"):
            key, secs = _process_one_sequence(seq_cfg, rrd_save_path, config.log_labels)
            tqdm.write(f"[done] {key} in {secs:.2f}s")

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
