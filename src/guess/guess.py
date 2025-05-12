"""Assembly code transpilation prediction module.

This module provides functionality for generating and managing assembly code
translations using configurable models. It supports multiple model types and
handles both function-level and cloze-style translations.
"""

from typing import List
from enum import Enum
from pathlib import Path
from typing import Dict

from src.config import Config
from src.helpers.model import Model, InferenceConfig, PredictionResult, get_device
from src.domain.models.QwenModel import QwenModel
from src.domain.models.BartLargeModel import BartLargeModel
from src.helpers.dataset import Dataset, DatasetConfig
from src.helpers.data_loader import DataLoader


class ConfigType(str, Enum):
    """Available configuration types for transpilation models.

    Each enum value corresponds to a launch spec file in the configs directory.
    The file path is constructed by appending '.yaml' to the enum value.
    """
    QWEN_X862ARM64 = "launch_spec_qwen"
    BART_RISC2ARM = "launch_spec_bart"
    QWEN_EULER_ARM2RISC = "launch_spec_qwen_euler_arm2risc"
    QWEN_EULER_RISC2ARM = "launch_spec_qwen_euler_risc2arm"

    def get_path(self) -> Path:
        """Get full path to the YAML configuration file.

        Returns:
            Path to the configuration file in the configs directory
        """
        return Path(__file__).parent.parent.parent / "configs" / f"{self.value}.yaml"


class Guess:
    """Handles generation of assembly code translations.

    This class manages the lifecycle of model initialization, data loading,
    and inference for assembly code translation. It supports both function-level
    translations and cloze-style predictions.

    The class uses a configuration-driven approach where model type, inference
    parameters, and data settings are loaded from YAML config files.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the Guess instance.

        Args:
            config_type: Type of configuration to use for model and inference

        Raises:
            ValueError: If the model type specified in config is not supported
        """
        self.config = config

        # Initialize model based on configuration
        model_name = self.config.model_name.lower()
        if "qwen" in model_name:
            model_name = model_name.split(":")[1]
            print(f"Loading Qwen model: {model_name}")
            self.model: Model = QwenModel(
                model_name=model_name, device=get_device()
            )
        elif "bart" in model_name:
            print(f"Loading BART model: {self.config.model_name}")
            self.model: Model = BartLargeModel(
                model_name=self.config.model_name, device=get_device()
            )
        else:
            raise ValueError(f"Unknown model type in {self.config.model_name}")

        self.is_enc_dec = self.config.is_enc_dec
        # Setup predictions directory
        self.predictions_folder = Path(
            self.config.artifacts_path
        ) / "predictions"
        self.predictions_folder.mkdir(parents=True, exist_ok=True)

        # Configure inference parameters
        self.inference_cfg = InferenceConfig(
            beam_size=self.config.inference_params.get("beam_size", 5),
            temperature=self.config.inference_params.get("temperature", 0.7),
            max_length=self.config.model_max_length,
            num_return_sequences=self.config.inference_params.get(
                "num_guesses", 5
            )
        )

        # Initialize dataset and data loader
        dataset_config = DatasetConfig(
            source_lang=self.config.source_lang,
            target_lang=self.config.target_lang,
            dataset_name=self.config.dataset_name,
            batch_size=self.config.inference_params.get("batch_size", 32),
            evaluation_metrics=["compilation"]
        )
        self.dataset = Dataset.from_config(dataset_config)

        # Configure data loader with chunking for encoder-decoder models
        self.data_loader = DataLoader.from_dataset(
            dataset=self.dataset,
        )

    def guess(self) -> Dict[str, PredictionResult]:
        """Generate translation candidates for all datapoints using batch processing.

        Returns:
            Dictionary mapping source files to their prediction results
        """
        all_predictions = {}

        # Process all batches
        for instance in self.data_loader.iter():
            try:
                prediction = self.model.predict(instance, self.inference_cfg)
                all_predictions[instance.instance_id] = prediction
            except Exception as e:
                all_predictions[instance.instance_id] = None
                print(f"Error processing {instance.instance_id}: {e}")
                continue

        return all_predictions

    def evaluate(self, predictions: Dict[str, Dict]) -> Dict[str, float]:
        return self.dataset.evaluate(predictions)
