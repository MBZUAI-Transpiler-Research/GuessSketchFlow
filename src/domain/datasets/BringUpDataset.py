import os
import logging
import json
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.helpers.dataset import Dataset, DatasetInstance, AssemblyLanguage, DatasetConfig


@Dataset.register("bringup")
class BringUpDataset(Dataset):
    """Dataset for BringUp tasks

    The BringUp dataset consists of various C programming tasks that are compiled
    to both x86 and ARM64 assembly code. This dataset is used to evaluate the
    model's ability to translate between different assembly languages.
    """

    arch_suffix = {
        'x86': 'x86',
        'arm64': 'arm',
        'riscv': 'risc'
    }

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.dataset_name = 'UnixCommands'
        self.source_arch = config.source_lang
        self.target_arch = config.target_lang

        # Validate architectures
        valid_archs = {'x86', 'arm64', 'riscv'}
        if self.source_arch not in valid_archs or self.target_arch not in valid_archs:
            raise ValueError(
                f"Invalid architecture. Must be one of: {valid_archs}")

        # Load dataset metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load dataset metadata from source files and JSONL."""
        # Convert dataset name to Path and construct paths
        base_path = Path(self.dataset_name)

        # Load assembly files
        source_jsonl = Path('data/processed') / self.source_arch.upper() / \
            f'BringUp_{self.arch_suffix[self.source_arch]}.jsonl'
        target_jsonl = Path('data/processed') / self.target_arch.upper() / \
            f'BringUp_{self.arch_suffix[self.target_arch]}.jsonl'
        print(f"Source JSONL: {source_jsonl}")
        print(f"Target JSONL: {target_jsonl}")
        if not source_jsonl.exists() or not target_jsonl.exists():
            raise ValueError(f"Missing JSONL files for {self.dataset_name}")

        metadata = {
            'source': self._load_jsonl(source_jsonl),
            'target': self._load_jsonl(target_jsonl)
        }

        return metadata

    def _load_jsonl(self, file_path: Path, log_file: bool = False) -> List[Dict]:
        """Load entries from a JSONL file.

        Args:
            file_path: Path to the JSONL file
            log_file: Whether to log the file being loaded

        Returns:
            List of entries from the JSONL file
        """
        entries = []
        if log_file:
            self.logger.info(f"Loading JSONL file: {file_path}")
        print(f"Loading file: {file_path}")
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    entries.append(entry)
        print(f"Loaded {len(entries)} entries from {file_path}")
        return entries

    def load_data(self) -> List[DatasetInstance]:
        """Load all instances for inference."""
        instances = []

        # Create instances from all available data
        print(f"Source entries: {len(self.metadata['source'])}")
        print(f"Target entries: {len(self.metadata['target'])}")
        for idx in range(len(self.metadata['source'])):
            source_entry = self.metadata['source'][idx]
            target_entry = self.metadata['target'][idx]

            arch_map = {'x86': 'x86', 'arm64': 'arm', 'riscv': 'risc'}
            source_key = arch_map[self.source_arch]
            target_key = arch_map[self.target_arch]

            print(
                f"Creating instance {idx} with source key {source_key} and target key {target_key}"
            )
            print(f"Source entry keys: {source_entry.keys()}")
            print(f"Target entry keys: {target_entry.keys()}")

            instance = DatasetInstance(
                instance_id=f"{self.dataset_name}/{source_entry['source']}",
                source_lang=AssemblyLanguage(self.source_arch),
                target_lang=AssemblyLanguage(self.target_arch),
                source=source_entry[source_key],  # Contains assembly code
                target=target_entry[target_key],  # Contains assembly code
                metadata={
                    'source_file': source_entry['source'],
                    'source_output': source_entry.get(f'{source_key}_output', ''),
                    'target_output': target_entry.get(f'{target_key}_output', ''),
                }
            )
            instances.append(instance)

        print(f"Created {len(instances)} instances")
        self.logger.info(f"Loaded {len(instances)} instances")
        self._instances = instances
        return instances

    def evaluate(self, predictions: List[str], instances: List[DatasetInstance]) -> Dict[str, Any]:
        """Evaluate the model's predictions

        Args:
            predictions: List of predicted target assembly code
            instances: List of dataset instances

        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        if len(predictions) != len(instances):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) does not match number of instances ({len(instances)})")

        # Initialize metrics
        total = len(predictions)
        compile_success = 0
        per_problem_results = {}
        errors = {}

        # TODO: Implement proper evaluation by compiling the predicted assembly code
        # For now, we'll just count the number of non-empty predictions
        for i, (pred, instance) in enumerate(zip(predictions, instances)):
            problem_id = instance.instance_id

            if pred.strip():
                compile_success += 1
                per_problem_results[problem_id] = "success"
            else:
                per_problem_results[problem_id] = "failed"
                errors[problem_id] = "Empty prediction"

        # Calculate overall metrics
        compile_rate = compile_success / total if total > 0 else 0.0

        metrics = {
            'compile_rate': compile_rate,
            'per_problem_results': per_problem_results,
            'errors': errors
        }

        return metrics
