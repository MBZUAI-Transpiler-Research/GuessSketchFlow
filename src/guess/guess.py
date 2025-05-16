from typing import Dict

from src.helpers.model import Model, PredictionResult, get_device
from src.domain.models.QwenModel import QwenModel
from src.domain.models.BartLargeModel import BartLargeModel
from src.helpers.launch_spec import LaunchSpec
from src.helpers.dataset import DatasetInstance
from datasets import load_dataset


class Guess:
    def __init__(self, launch_spec: LaunchSpec) -> None:
        self.launch_spec = launch_spec

        if launch_spec.model_config.architecture == "qwen":
            print(f"Loading Qwen model: {launch_spec.model_config.name}")
            self.model: Model = QwenModel(
                model_name=launch_spec.model_config.name, device=get_device()
            )
        elif launch_spec.model_config.architecture == "bart":
            print(f"Loading BART model: {launch_spec.model_config.name}")
            self.model: Model = BartLargeModel(
                model_name=launch_spec.model_config.name, device=get_device()
            )
        else:
            raise ValueError(
                f"Unknown model type in {launch_spec.model_config.architecture}"
            )

        self.dataset = load_dataset(
            launch_spec.dataset_config.dataset_name,
            split=launch_spec.dataset_config.split
        )

    def guess(self) -> Dict[str, PredictionResult]:
        all_predictions = {}

        for raw_instance in self.dataset:
            if raw_instance["file"] in self.launch_spec.dataset_config.skip_files:
                continue

            instance = DatasetInstance(
                instance_id=raw_instance["file"],
                source_lang=self.launch_spec.dataset_config.source_lang,
                target_lang=self.launch_spec.dataset_config.target_lang,
                source=raw_instance[self.launch_spec.dataset_config.source_lang],
                target=raw_instance[self.launch_spec.dataset_config.target_lang]
            )

            try:
                prediction = self.model.predict(instance, self.launch_spec.model_config)
                all_predictions[instance.instance_id] = prediction
            except Exception as e:
                all_predictions[instance.instance_id] = None
                print(f"Error processing {instance.instance_id}: {e}")
                continue

        return all_predictions

    def evaluate(self, predictions: Dict[str, Dict]) -> Dict[str, float]:
        return self.dataset.evaluate(predictions)
