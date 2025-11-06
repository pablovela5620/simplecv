from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm

from simplecv.apis.view_exoego import VisualizeConfig, visualize_exo_ego
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.rerun_log_utils import RerunTyroConfig

np.set_printoptions(suppress=True)


@dataclass
class BatchConvertConfig:
    """Configuration for batch converting annotated sequences to RRD files."""

    rrd_save_dir: Path
    """Output directory that will receive the generated ``.rrd`` files."""
    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset factory capable of producing the annotated ``BaseExoEgoSequence``."""
    max_conversions: int | None = None
    """Optional cap on how many sequences to convert; ``None`` processes all available episodes."""
    dry_run: bool = False
    """When ``True``, only report the sequences that would be converted without writing files."""
    force: bool = False
    """When ``True``, overwrite existing ``.rrd`` files instead of skipping them."""


def main(config: BatchConvertConfig):
    start_time: float = timer()
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()
    # create save directory if it does not exist
    if not config.dry_run:
        config.rrd_save_dir.mkdir(parents=True, exist_ok=True)
    for idx, current_exoego_sequence in enumerate(tqdm(exoego_sequence.iter_dataset(), desc="Processing sequences")):
        rrd_save_path: Path = config.rrd_save_dir / f"{idx:08d}.rrd"
        sequence_label: str = getattr(current_exoego_sequence.config, "sequence_name", rrd_save_path.stem)

        if rrd_save_path.exists() and not config.force:
            tqdm.write(f"[skip-existing] {sequence_label} -> {rrd_save_path}")
        elif config.dry_run:
            tqdm.write(f"[dry-run] {sequence_label} -> {rrd_save_path}")
        else:
            current_cfg = VisualizeConfig(
                rr_config=RerunTyroConfig(save=rrd_save_path), dataset=current_exoego_sequence.config
            )
            visualize_exo_ego(current_exoego_sequence, current_cfg)

        if config.max_conversions is not None and idx + 1 >= config.max_conversions:
            break

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
