from dataclasses import dataclass


@dataclass
class DatasetConfig:
    source_lang: str
    target_lang: str
    dataset_name: str
    split: str


@dataclass
class DatasetInstance:
    instance_id: str
    source_lang: str
    target_lang: srr
    source: str
    target: str
