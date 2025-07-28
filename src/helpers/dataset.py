from dataclasses import dataclass
from typing import List

# IMPORTANT: This must match the structure of the YAML config!
# If you add or remove fields here, update launch_spec.py and the config files too.

@dataclass
class DatasetConfig:
    source_lang: str
    target_lang: str
    dataset_name: str
    split: str
    skip_files: List[str]

# used as an input for guess method in the Guess class (see guess.py)
@dataclass
class DatasetInstance:
    instance_id: str
    source_lang: str
    target_lang: str
    source: str
    target: str