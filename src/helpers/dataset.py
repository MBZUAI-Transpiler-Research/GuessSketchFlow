from dataclasses import dataclass
from typing import List

@dataclass
class DatasetConfig:
    source_lang: str
    target_lang: str
    dataset_name: str
    split: str
    skip_files: List[str]


@dataclass
class DatasetInstance:
    instance_id: str
    source_lang: str
    target_lang: str
    source: str
    target: str
