import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .dataset import DatasetInstance
from typing import List, Optional, Union, Tuple
from transformers import PreTrainedTokenizer

## choose best available device, but be careful! this is set to cuda:1 and needs to be manually changed. NOT used in present iteration of code
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    elif torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

# IMPORTANT: This must match the structure of the YAML config!
# If you add or remove fields here, update launch_spec.py and the config files too.

@dataclass
class ModelConfig:
    name: str
    architecture: str
    max_length: int
    num_beams: int
    num_return_sequences: int
    temperature: float
    k: int

# Used as the return object of the predict method to amalgamate all data into one object
@dataclass
class PredictionResult:
    instance_id: str
    source: torch.Tensor
    pred: torch.Tensor
    alignments: List[List[int]]
    confidence: List[float]
    alt_tokens: Optional[List[List[Tuple[int, float]]]] = None

# Used to initialze each model (Bart or Qwen)
class Model(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer
    ):
        self.tokenizer = tokenizer

    @abstractmethod
    def predict(self, instance: DatasetInstance, config: ModelConfig) -> PredictionResult:
        pass