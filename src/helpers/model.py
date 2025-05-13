"""Model interface for assembly code transpilation.

This module defines the abstract base class for ML/AI models that perform
assembly code transpilation. It provides a generic interface for inference,
tokenization, decoding, and prompt preparation.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional, Union, TypeVar, Callable
from .dataset import DatasetInstance

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f'{func.__name__} executed in {elapsed_time:.2f} seconds')
        return result
    return wrapper


@dataclass
class InferenceConfig:
    """Configuration for model inference.

    Attributes:
        beam_size: Number of beams for beam search decoding
        temperature: Sampling temperature for generation
        max_length: Maximum length of generated sequence
        num_return_sequences: Number of sequences to return per input
    """
    beam_size: int = 2
    temperature: float = 0.7
    max_length: int = 15500
    num_return_sequences: int = 1
    


@dataclass
class PredictionResult:
    """Results from model inference.
    """
    instance_id: str
    source: [torch.Tensor]  # ids, seq
    pred: [torch.Tensor]  # ids, seq
    alignments: torch.Tensor
    confidence: torch.Tensor
    pred_dec: Optional[str] = None
    levenshtein_distance: Optional[int] = None


class Model(ABC):
    """Abstract base class for assembly transpilation models.

    This class defines the interface that all model implementations must follow.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, device: Optional[Union[str, torch.device]] = "cpu"):
        """Initialize model with tokenizer and device.

        Args:
            tokenizer: HuggingFace tokenizer for encoding/decoding
            device: device for inference (cpu/cuda or torch.device)
        """
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initialized {self.__class__.__name__} on device: {device}"
        )

    @abstractmethod
    @log_execution_time
    def decode(self, token_ids: torch.Tensor, **kwargs) -> Union[str, List[str]]:
        """Decode token IDs back to text.

        Args:
            token_ids: Tensor of token IDs to decode
            **kwargs: Additional decode args

        Returns:
            Decoded string or list of strings
        """
        self.logger.debug(f"Decoding tensor of shape: {token_ids.shape}")
        pass

    @abstractmethod
    def predict(self, instance: DatasetInstance, config: Optional[InferenceConfig] = None) -> PredictionResult:
        pass
