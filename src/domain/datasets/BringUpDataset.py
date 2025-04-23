import os
import logging
import json
import subprocess
from typing import List, Dict, Any, Optional

from src.helpers.dataset import Dataset, DatasetInstance, AssemblyLanguage, DatasetConfig


@Dataset.register("bringup")
class BringUpDataset(Dataset):
    """Dataset for BringUp tasks
    
    The BringUp dataset consists of various C programming tasks that are compiled
    to both x86 and ARM64 assembly code. This dataset is used to evaluate the
    model's ability to translate between different assembly languages.
    """
    
    def __init__(self, config: DatasetConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.dataset_path = "data/datasets/BringUp"
        self.logger = logging.getLogger(__name__)
        
        # Define paths for processed data
        self.x86_jsonl = os.path.join("data", "processed", "x86", "BringUp_x86.jsonl")
        self.arm64_jsonl = os.path.join("data", "processed", "ARM64", "BringUp_arm64.jsonl")
        
        # Validate architectures
        self.source_arch = kwargs.get("source_arch", "x86")
        self.target_arch = kwargs.get("target_arch", "arm64")
        
        if self.source_arch == self.target_arch:
            raise ValueError(f"Source and target architectures must be different.")
    
    def load_data(self) -> List[DatasetInstance]:
        """Load the BringUp dataset
        
        Returns:
            List[DatasetInstance]: List of dataset instances
        """
        instances = []
        
        # Check if JSONL files exist
        if not os.path.exists(self.x86_jsonl) or not os.path.exists(self.arm64_jsonl):
            self.logger.warning(f"JSONL files not found. Running compilation script...")
            self._compile_dataset()
        
        # Load data from JSONL files
        source_jsonl = self.x86_jsonl 
        target_jsonl = self.arm64_jsonl 
        
        source_data = {}
        target_data = {}
        
        # Load source data
        with open(source_jsonl, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    source_data[entry["source"]] = entry
        
        # Load target data
        with open(target_jsonl, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    target_data[entry["source"]] = entry
        
        # Create dataset instances
        for problem_id in source_data.keys():
            if problem_id in target_data:
                source_entry = source_data[problem_id]
                target_entry = target_data[problem_id]
                
                source_code = source_entry[self.source_arch]
                target_code = target_entry[self.target_arch]
                
                instance = DatasetInstance(
                    instance_id=problem_id,
                    source=source_code,
                    target=target_code,
                    source_lang=AssemblyLanguage(self.source_arch),
                    target_lang=AssemblyLanguage(self.target_arch),
                    metadata={
                        "problem_id": problem_id,
                        "source_arch": self.source_arch,
                        "target_arch": self.target_arch,
                    }
                )
                
                instances.append(instance)
        
        self.logger.info(f"Loaded {len(instances)} BringUp dataset instances")
        return instances
    
    def _compile_dataset(self) -> None:
        """Compile the BringUp dataset
        
        This method runs the compilation script to generate assembly code for
        both RISC-V and ARM architectures.
        """
        script_path = os.path.join(self.dataset_path, "compile_bringup.sh")
        
        if not os.path.exists(script_path):
            self.logger.error(f"Compilation script not found at {script_path}")
            raise FileNotFoundError(f"Compilation script not found at {script_path}")
        
        self.logger.info(f"Running compilation script: {script_path}")
        
        try:
            # Make the script executable
            os.chmod(script_path, 0o755)
            
            # Run the compilation script
            result = subprocess.run(
                ["bash", script_path],
                cwd=os.path.dirname(script_path),
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"Compilation script output:\n{result.stdout}")
            
            if result.stderr:
                self.logger.warning(f"Compilation script stderr:\n{result.stderr}")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Compilation script failed with exit code {e.returncode}")
            self.logger.error(f"Stdout: {e.stdout}")
            self.logger.error(f"Stderr: {e.stderr}")
            raise RuntimeError(f"Failed to compile BringUp dataset: {e}")
    
    def evaluate(self, predictions: List[str], instances: List[DatasetInstance]) -> Dict[str, Any]:
        """Evaluate the model's predictions
        
        Args:
            predictions: List of predicted target assembly code
            instances: List of dataset instances
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        if len(predictions) != len(instances):
            raise ValueError(f"Number of predictions ({len(predictions)}) does not match number of instances ({len(instances)})")
        
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
