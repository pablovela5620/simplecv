from dataclasses import dataclass

from simplecv.configs.base_config import InstantiateConfig


@dataclass
class BaseExoEgoDatasetConfig(InstantiateConfig):
    load_labels: bool = True
