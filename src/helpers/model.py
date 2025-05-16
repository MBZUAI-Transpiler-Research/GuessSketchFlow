import torch
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .dataset import DatasetInstance
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    elif torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@dataclass
class ModelConfig:
    name: str
    architecture: str
    max_length: int
    num_beams: int
    num_return_sequences: int
    temperature: float


@dataclass
class PredictionResult:
    instance_id: str
    source: torch.Tensor
    pred: torch.Tensor
    alignments: List[List[int]]
    confidence: List[float]


class Model(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        device: Optional[Union[str, torch.device]] = "cpu"
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initialized {self.__class__.__name__} on device: {device}"
        )

    @abstractmethod
    def predict(self, instance: DatasetInstance, config: ModelConfig) -> PredictionResult:
        pass
