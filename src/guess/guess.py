from typing import Dict, Optional

from src.helpers.model import Model, PredictionResult
from src.domain.models.QwenModel import QwenModel
from src.domain.models.BartLargeModel import BartLargeModel
from src.helpers.launch_spec import LaunchSpec
from src.helpers.dataset import DatasetInstance, ClozeInstance, load_cloze_dataset
from datasets import load_dataset

# This Guess class has three attributes: launch_spec, model, and dataset
#
# It stores the full config for reference (self.launch_spec)
# It loads a model based on the config's architecture (self.model)
# It loads a dataset from json rather than using HuggingFace’s datasets lib (self.dataset)
# Another piece of code loads the huggingface file and puts it in a JSON

# guess() does the following:
#
#  Iterates over the loaded dataset, which is a list of ClozeInstance objects
#  Skips files listed in the config
#  Sends it to the model's predict() method. Note: See QwenModel.py for more on this
#  Catches errors and saves None if prediction fails
#  Returns a dictionary mapping instance IDs to results


class Guess:
    def __init__(self, launch_spec: LaunchSpec) -> None:
        self.launch_spec = launch_spec

        if launch_spec.model_config.architecture == "qwen":
            print(f"Loading Qwen model: {launch_spec.model_config.name}")
            self.model: Model = QwenModel(model_name=launch_spec.model_config.name)
        elif launch_spec.model_config.architecture == "bart":
            print(f"Loading BART model: {launch_spec.model_config.name}")
            self.model: Model = BartLargeModel(model_name=launch_spec.model_config.name)
        else:
            raise ValueError(f"Unknown model type in {launch_spec.model_config.architecture}")

        # load_dataset("adpretko/reducedeval", split="train") gives Dataset object with rows {x86, arm, file} - can access examples via indexing (dataset[0])
        #self.dataset = load_dataset(launch_spec.dataset_config.dataset_name, split=launch_spec.dataset_config.split)
        self.dataset = load_cloze_dataset(
            jsonl_path=launch_spec.dataset_config.dataset_name,  # now a path to the .jsonl file
            source_lang=launch_spec.dataset_config.source_lang,
            target_lang=launch_spec.dataset_config.target_lang
        )

    def guess(self) -> Dict[str, Optional[PredictionResult]]:
        all_predictions = {}

        for instance in self.dataset:
            # Extract file_id from instance_id: "problem1::cloze" → "problem1"
            file_id = instance.instance_id.split("::")[0]
            if file_id in self.launch_spec.dataset_config.skip_files:
                continue

            try:
                prediction = self.model.predict(instance, self.launch_spec.model_config)
                all_predictions[instance.instance_id] = prediction
            except Exception as e:
                all_predictions[instance.instance_id] = None
                print(f"Error processing {instance.instance_id}: {e}")
                continue

        return all_predictions