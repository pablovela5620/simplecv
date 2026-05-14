# Goal: Maximize `tools/batch_raw_to_rrd.py` Throughput on Assembly101

This document is the persistent **goal contract** for an autonomous coding
agent (either Codex or Claude) that optimizes the batch raw → RRD ingestion
pipeline. It follows the structure laid out in
[OpenAI's "Follow goals" use case](https://developers.openai.com/codex/use-cases/follow-goals)
but is agent-agnostic — keep it generic.

The agent is expected to read this file at every checkpoint, follow its
contract literally, and run **continuously until the deadline** without
needing further steering.

---

## 1. Objective

Make `python tools/batch_raw_to_rrd.py assembly101` produce the same `.rrd`
files it produces today, but **as fast as possible**, by changing only the
ingestion code paths (`simplecv/apis/batch_raw_to_rrd.py`,
`simplecv/apis/view_exoego.py`, and the assembly101 dataset adapters under
`simplecv/data/{ego,exo,exoego}/`).

You may not change the **observable content** of the resulting RRDs:
component columns, entity paths, timelines, and per-frame values must remain
equivalent to the ground-truth catalog under
`data/exoego-forge-catalog/assembly101/all/*.rrd` within the tolerances
defined in §4.

You may freely change *how* data gets in there (parallelism, vectorization,
batched `send_columns`, eliminating redundant decodes, avoiding round-trips
to disk, reusing video MP4 blobs, eager prefetch, etc.).

---

## 2. Stopping Condition (Hard Deadline)

Stop work at **07:00 AM America/Chicago (CDT)** on the date the run begins.

At every checkpoint, compute remaining time with:

```bash
date -d "$(date +%Y-%m-%d) 07:00 CDT" +%s
date +%s
```

If `now >= deadline`, the agent **must**:

1. Print the leaderboard and the absolute best configuration (§6).
2. Update `docs/optimize_batch_ingest_results.md` with the final entry.
3. Exit. Do not start a new experiment past the deadline.

While running, never assume a single experiment can be aborted "later" — set
a per-experiment wall-clock budget (start with 25 minutes, halve it if
remaining time < 1 h) and kill the run if it exceeds the budget.

---

## 3. Initial Resources (read these first, in order)

1. `tools/batch_raw_to_rrd.py` — the CLI entrypoint to optimize.
2. `simplecv/apis/batch_raw_to_rrd.py` — `BatchConvertConfig` + main loop.
3. `simplecv/apis/view_exoego.py` — `visualize_exo_ego`, the per-sequence
   logging routine that dominates wall time.
4. `simplecv/data/exoego/assembly101.py`,
   `simplecv/data/exo/assembly101_exo.py`,
   `simplecv/data/ego/assembly101_ego.py` — Assembly101 adapters.
5. `simplecv/rrd_query_utils.py` — `RRDQuerySession` for reading existing RRDs
   via the Rerun DataFusion / dataframe API
   (https://rerun.io/docs/howto/query-and-transform/get-data-out).
6. `data/exoego-forge-catalog/assembly101/all/*.rrd` — **10 ground-truth
   RRDs. Treat as read-only. Never write, move, or delete these.**

> The full Assembly101 source pile lives at
> `/mnt/8tb/data/assembly101-original/` (≈360 sequences). A full-corpus run
> takes several hours; do not run it as a benchmark.

---

## 4. Validation Method ("done" contract)

A candidate optimization is **valid** only if, for each of the 10 GT
sequences, the freshly-ingested RRD passes all of the following checks
against the corresponding GT file at
`data/exoego-forge-catalog/assembly101/all/<sequence>.rrd`:

1. **Entity-path set parity** — `set(schema.component_columns())` is equal,
   modulo a documented allow-list of cosmetic columns.
2. **Index parity** — for the `video_time` timeline, the index column length
   matches, and the first/last timestamps match to within 1 ns.
3. **Numeric parity (sampled)** — for at least three semantically critical
   entities — `/world/gt/coco133_xyz` positions, one
   `/world/exo/<cam>/pinhole/coco133_uv` 2D track, and one
   `/world/ego/<cam>/pinhole/coco133_uv` 2D track — values at five
   evenly-spaced frame indices match GT to **`atol=1e-5`** (float32 quantum).
4. **Static-component parity** — the camera intrinsics under each
   `pinhole` entity match exactly (no float drift allowed: these are
   recorded once and must be identical).
5. **MANO mesh parity (if present)** — vertex positions at three sampled
   frames match within `atol=1e-4`.

Use `simplecv/rrd_query_utils.RRDQuerySession.read_arrow` /
`read_pandas` for comparisons — do not reinvent loaders. Write the parity
check at `tools/validate_assembly101_rrd_parity.py` so it is reusable.

Any failure marks the experiment **invalid**, regardless of speed gain.

---

## 5. Per-Experiment Workflow

Each experiment is one full pass of the following loop. Treat it as
atomic; do not interleave.

1. **Branch the idea.** In `docs/optimize_batch_ingest_results.md`, append
   an entry with: hypothesis, files touched, expected speedup, risk.
2. **Implement** the change. Keep it self-contained: one experiment = one
   commit on the working branch.
3. **Benchmark.** Run on **the same 3-sequence subset** for fast feedback
   (use `--max-conversions 3`), with output to a scratch directory under
   `--rrd-save-dir /tmp/batch-bench/<exp-id>/`. Time it with:
   ```bash
   /usr/bin/time -f '%e' python tools/batch_raw_to_rrd.py \
       assembly101 \
       --rrd-save-dir /tmp/batch-bench/<exp-id> \
       --max-conversions 3 \
       --force
   ```
   Record `seconds_total` and `seconds_per_sequence`.
4. **Validate** with `tools/validate_assembly101_rrd_parity.py` against the
   10 GT RRDs **but only the 3 that you re-ingested** (intersect by
   sequence key). Record pass/fail per check.
5. **Promote.** If valid AND faster than current champion, also run a
   **10-sequence pass** against the same set of sequence keys the GT
   catalog used, validate against all 10, and update the leaderboard.
6. **Persist.** Append a results row to
   `docs/optimize_batch_ingest_results.md` regardless of outcome — failed
   experiments are still data.
7. **Clean.** Remove `/tmp/batch-bench/<exp-id>/` to reclaim disk.

If an experiment regresses or fails parity, **revert the working tree** to
the last known champion (`git restore` / `git checkout`) before starting
the next experiment.

---

## 6. Required Progress Logging

Create and continuously update `docs/optimize_batch_ingest_results.md` with
this schema (markdown table). Append-only; never rewrite history.

| exp_id | timestamp_cdt | hypothesis | files_touched | wall_seconds_3seq | wall_seconds_10seq | sec_per_seq | speedup_vs_baseline_pct | parity | champion | notes |
| ------ | ------------- | ---------- | ------------- | ----------------- | ------------------ | ----------- | ----------------------- | ------ | -------- | ----- |

Definitions:

- **baseline** — the wall-clock measured on `main` (or the starting commit)
  using the same 3-sequence subset. Measure this **first**, before any
  optimization. Store it as `exp_id = baseline`.
- **speedup_vs_baseline_pct** — `100 * (1 - new_wall_seconds /
  baseline_wall_seconds)`. Positive = faster.
- **parity** — `PASS` | `FAIL:<reason>` | `N/A` (for invalidated runs).
- **champion** — `Y` iff this experiment becomes the new fastest valid
  configuration; only one row should be `Y` at any time (clear the prior
  champion when surpassed).

At the bottom of the file, maintain a **Leaderboard** section with the top
five fastest *valid* experiments and a one-line summary of the current
champion's diff.

Print a compact status block to stdout at every checkpoint with:

```
[checkpoint] now=<HH:MM CDT> deadline=07:00 remaining=<HHhMMm>
  baseline_sec_per_seq=<x>
  champion_sec_per_seq=<y>  speedup=<z%>
  last_experiment=<exp_id> parity=<PASS|FAIL> wall=<s>
  experiments_total=<n> experiments_valid=<n_valid>
```

---

## 7. Configurable Save Directory

The current code hard-codes `data/exoego-forge-catalog`. Before doing any
benchmarking work, make the output location selectable via:

- CLI: an existing `--rrd-save-dir` flag is already exposed through
  `BatchConvertConfig.rrd_save_dir` — verify it works and is wired through
  `tools/batch_raw_to_rrd.py`.
- Sensible defaults: keep `data/exoego-forge-catalog` as the in-repo
  default but document **`/mnt/8tb/data/exoego-forge-catalog`** as the
  preferred location for the user's machine (more storage). Add a short
  paragraph to `README.md` (or a new `docs/storage.md`) capturing this.

All benchmark runs MUST write to `/tmp/batch-bench/<exp-id>/`. The shared
catalog directories on `/mnt/8tb` and `data/` are **read-only** during
optimization to keep GT pristine and benchmarks reproducible.

---

## 8. Ideas to Try (non-exhaustive, ordered roughly by expected impact)

These are seed ideas — explore freely, but do not skip the validation
contract. Cheap experiments first.

1. **MP4 blob reuse**: `view_exoego.py` reads the same MP4 once per video
   asset; verify it is not decoded twice. If it is, log the encoded blob
   directly via `rr.AssetVideo` from bytes instead of re-decoding.
2. **`compute_vertex_normals_batch`** has a Python `for k in range(n_faces)`
   loop accumulating into vertex normals. Replace with vectorized
   `np.add.at` / `scatter_add` or precompute on GPU. This is on the hot
   path for every MANO frame.
3. **`project_brown_conrady_*`** — profile per-call cost; reuse intrinsics
   matrices and avoid repeated `np.asarray` copies.
4. **Parallelism**: per-sequence work is embarrassingly parallel. Try
   `concurrent.futures.ProcessPoolExecutor` over `iter_dataset()` with
   `--max-conversions` partitioning. Watch for the global rerun
   `RecordingStream` — each child must get its own stream.
5. **Rerun chunk size**: tune `RERUN_FLUSH_TICK_SECS` or use `rec.flush()`
   strategically to avoid micro-flushes.
6. **NumPy contiguity**: confirm all `rr.send_columns` payloads are already
   `np.ascontiguousarray(..., dtype=...)` to skip implicit copies.
7. **Disk I/O**: write to a tmpfs / fast NVMe scratch first, `mv` at end.
8. **Lazy MANO**: skip `log_mano` if downstream catalog never reads it
   (only valid if GT does not require it — check parity first!).

Document every idea you actually try, even if you abandon it.

---

## 9. Etiquette

- Never `--no-verify` or skip pre-commit hooks.
- Never touch `data/exoego-forge-catalog/` or anything under
  `/mnt/8tb/data/assembly101-rrd/`.
- Never push to `main`. All work stays on the current branch.
- Profile before optimizing; cite `py-spy` / `cProfile` output in the
  results log for any non-trivial change.
- Prefer one self-contained commit per experiment with a message of the
  form `optimize(batch-ingest): <exp_id> — <hypothesis> [Δ=<sign><pct>%]`.

---

## 10. Starter Prompt (paste into the agent verbatim)

> You are an autonomous engineering agent. Your single goal is to maximize
> the ingestion throughput of `python tools/batch_raw_to_rrd.py assembly101`
> against the Assembly101 dataset, while preserving the observable content
> of the resulting `.rrd` files relative to the 10 ground-truth RRDs under
> `data/exoego-forge-catalog/assembly101/all/`.
>
> Read and follow `docs/optimize_batch_ingest_goal.md` literally — it is the
> contract. In particular:
>
> 1. Establish the baseline first (3-sequence wall time on the current
>    commit) and record it in `docs/optimize_batch_ingest_results.md`.
> 2. Build `tools/validate_assembly101_rrd_parity.py` using
>    `simplecv.rrd_query_utils.RRDQuerySession` and the dataframe API
>    described at https://rerun.io/docs/howto/query-and-transform/get-data-out.
>    The 10 GT RRDs are read-only.
> 3. Run experiments in a loop until **07:00 AM CDT today**. After each
>    experiment, append a row to `docs/optimize_batch_ingest_results.md`
>    and print the checkpoint status block defined in §6 of the goal doc.
>    On regressions or parity failures, revert to the last champion before
>    starting the next experiment.
> 4. All benchmark output goes to `/tmp/batch-bench/<exp-id>/`. The
>    user's preferred long-term storage is
>    `/mnt/8tb/data/exoego-forge-catalog/` — make sure `--rrd-save-dir` is
>    a first-class CLI flag and document this in the README.
> 5. The single deliverable is the **fastest valid configuration found
>    before the deadline**, captured as a single commit on the current
>    branch with `champion=Y` in the results log and a 1-line diff summary
>    at the bottom of `docs/optimize_batch_ingest_results.md`.
>
> Do not stop, ask for clarification, or change scope. If you finish the
> ideas list in §8 before the deadline, profile the current champion with
> `py-spy` and propose new experiments from the hottest stack frames. Stop
> exactly at 07:00 CDT and print the final leaderboard.
