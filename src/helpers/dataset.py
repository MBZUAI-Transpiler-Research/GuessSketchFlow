from dataclasses import dataclass
from typing import List
import json

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


# IMPORTANT: This must match the structure of the YAML config!
# If you add or remove fields here, update launch_spec.py and the config files too.
@dataclass
class ClozeInstance:
    instance_id: str
    source_lang: str
    target_lang: str
    source_fn_code: str           # x86 function (or full cloze)
    target_fn_code: str # ARM function (or full cloze), for training/eval only
    source_fn_name: str           # e.g., "_func0", or "cloze"

def load_cloze_dataset(jsonl_path: str, source_lang: str, target_lang: str) -> List[ClozeInstance]:
    instances = []
    with open(jsonl_path, "r") as f:
        for line in f:
            row = json.loads(line)
            file_id = row["source"]
            source_fns = row[f"{source_lang}_fns"]
            target_fns = row.get(f"{target_lang}_fns", {})  # optional
            cloze = row[f"{source_lang}_cloze"]
            target_cloze = row.get(f"{target_lang}_cloze", None)

            # Add per-function examples
            for func_name, fn_code in source_fns.items():
                instances.append(ClozeInstance(
                    instance_id=f"{file_id}::{func_name}",
                    source_lang=source_lang,
                    target_lang=target_lang,
                    source_fn_code=fn_code,
                    target_fn_code=target_fns.get(func_name),
                    source_fn_name=func_name
                ))

            # Add cloze (full program) entry
            instances.append(ClozeInstance(
                instance_id=f"{file_id}::cloze",
                source_lang=source_lang,
                target_lang=target_lang,
                source_fn_code=cloze,
                target_fn_code=target_cloze,
                source_fn_name="cloze"
            ))
    return instances