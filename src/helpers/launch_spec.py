import os
import yaml
from dataclasses import dataclass, asdict

from .model import ModelConfig
from .dataset import DatasetConfig


@dataclass
class RunConfig:
    name: str


@dataclass
class LaunchSpec:
    run_config: RunConfig
    model_config: ModelConfig
    dataset_config: DatasetConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'LaunchSpec':
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Launch spec file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if 'run' not in config_dict:
            raise KeyError("'run' section missing from launch spec")
        run_config = RunConfig(
            name=config_dict['run'].get('name', '')
        )

        if 'model' not in config_dict:
            raise KeyError("'model' section missing from launch spec")

        model_dict = config_dict['model']
        model_config = ModelConfig(
            name=model_dict.get('name', ''),
            architecture=model_dict.get('architecture', ''),
            max_length=model_dict.get('max_length', 1024),
            num_beams=model_dict.get('num_beams', 1),
            num_return_sequences=model_dict.get('num_return_sequences', 1),
            temperature=model_dict.get('temperature', 1.0)
        )

        if 'dataset' not in config_dict:
            raise KeyError("'dataset' section missing from launch spec")

        dataset_dict = config_dict['dataset']
        dataset_config = DatasetConfig(
            source_lang=dataset_dict.get('source_lang', ''),
            target_lang=dataset_dict.get('target_lang', ''),
            dataset_name=dataset_dict.get('name', ''),
            split=dataset_dict.get('split', ''),
            skip_files=dataset_dict.get('skip_files', [])
        )

        return cls(
            run_config=run_config,
            model_config=model_config,
            dataset_config=dataset_config
        )

    def to_yaml(self, yaml_path: str) -> None:
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
