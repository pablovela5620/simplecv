from __future__ import annotations

from typing import TYPE_CHECKING

import tyro

from simplecv.data.new_exoego.assembly_101_ego import EgoAssembly101Config
from simplecv.data.new_exoego.base_ego import BaseEgoDatasetConfig
from simplecv.data.new_exoego.ego_dex import EgoDexConfig
from simplecv.data.new_exoego.hocap_ego import EgoHocapConfig

# ───────────────────── registry → union ─────────────────── #
dataset_defaults = {
    "assembly101": EgoAssembly101Config(),
    "hocap": EgoHocapConfig(),
    "ego-dex": EgoDexConfig(),
}

if TYPE_CHECKING:  # for IDEs / mypy
    EgoDatasetUnion = BaseEgoDatasetConfig
else:
    EgoDatasetUnion = tyro.extras.subcommand_type_from_defaults(dataset_defaults, prefix_names=False)

AnnotatedEgoDatasetUnion = tyro.conf.OmitSubcommandPrefixes[EgoDatasetUnion]
