"""Implementation of Qwen-based assembly transpiler model."""

import logging
import time
from functools import wraps
from typing import Dict, List, Optional, Union, Callable, TypeVar
import warnings

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.helpers.dataset import DatasetInstance
from src.helpers.model import Model, InferenceConfig, PredictionResult

logger = logging.getLogger(__name__)

T = TypeVar('T')


def measure_time(description: str) -> Callable:
    """Decorator to measure execution time of methods.

    Args:
        description: Description of the operation being timed
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(
                f"{description} completed in {elapsed_time:.2f} seconds")
            return result
        return wrapper
    return decorator


class QwenModel(Model):
    """Concrete implementation of Qwen-based assembly transpiler model.

    This class implements the Model interface using the Qwen architecture,
    specifically optimized for assembly code translation tasks.

    Attributes:
        model: The underlying Qwen model
        tokenizer: Tokenizer for text processing
        device: Device the model is loaded on
    """

    def __init__(self, model_name: str, device: Optional[str] = "auto"):
        """Initialize Qwen model and tokenizer.

        Args:
            model_name: Huggingface model ID or local checkpoint path.
            device: device_map argument for loading models (usually "auto").
        """
        logger.info(
            f"Initializing QwenModel with {model_name} on device {device}"
        )

        start_time = time.time()
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).to(torch.device(device))

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Setup model
        device_resolved = self.model.device
        super().__init__(tokenizer, device=device_resolved)
        self.model.eval()

        elapsed_time = time.time() - start_time
        logger.info(
            f"Model initialization completed in {elapsed_time:.2f} seconds"
        )

    @measure_time("Prompt preparation")
    def prepare_prompt(self, source_code: str, source_lang: str, target_lang: str) -> str:
        """Prepare system/user prompt for chat-based inference.

        Args:
            source_code: Source assembly code
            source_lang: Source ISA name
            target_lang: Target ISA name

        Returns:
            Formatted chat prompt for the model
        """
        logger.debug(
            f"Preparing prompt for {source_lang} -> {target_lang} conversion")
        logger.debug(f"Source code length: {len(source_code)} characters")

        system_prompt = (
            f"You are a helpful coding assistant specialized in converting from {source_lang} to {target_lang} assembly."
        )
        user_prompt = (
            f"Convert the following {source_lang} assembly code to {target_lang} assembly:\n"
            f"```{source_lang.lower()}asm\n{source_code}```"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return chat_text

    @measure_time("Tokenization")
    def tokenize(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize input text and move tensors to model device.

        Args:
            text: Input text or list of texts to tokenize
            **kwargs: Additional tokenizer arguments

        Returns:
            Dict containing input_ids and attention_mask
        """
        logger.debug(f"Tokenizing text{'s' if isinstance(text, list) else ''} "
                     f"of length: {len(text) if isinstance(text, str) else len(text[0])}")

        tokens = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, **kwargs
        ).to(self.device)

        logger.debug(f"Tokenized shape: {tokens['input_ids'].shape}")
        return tokens

    @measure_time("Token decoding")
    def decode(self, token_ids: torch.Tensor, **kwargs) -> Union[str, List[str]]:
        """Decode generated tokens back to text.

        Args:
            token_ids: Tensor of token IDs to decode
            **kwargs: Additional decoding arguments

        Returns:
            Single string if batch size is 1, otherwise list of strings
        """
        decoded = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
        )
        if token_ids.shape[0] == 1:
            return decoded[0]
        return decoded

    # @measure_time("Model inference")
    def infer(
        self,
        input_tokens: Dict[str, torch.Tensor],
        config: Optional[InferenceConfig] = None,
        **kwargs
    ):
        """Perform inference on prepared input tokens.

        Args:
            input_tokens: Dict with input_ids and attention_mask
            config: Optional inference configuration
            **kwargs: Additional model-specific arguments

        Returns:
            InferenceResult with generated tokens and metadata
        """
        if config is None:
            config = InferenceConfig()

        logger.info(
            f"Starting inference with temperature={config.temperature}, "
            f"max_length={config.max_length}"
        )
        logger.info(
            f"Input shape: {input_tokens['input_ids'].shape}"
        )

        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=max(128, 2048 - input_tokens['input_ids'].shape[1]),
            temperature=config.temperature,
            do_sample=(config.temperature > 0),
            output_attentions=True,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        alignments = self.get_alignments(outputs, input_tokens.input_ids.shape[1])
        confidence = self.get_confidence(outputs)

        return (input_tokens.input_ids, outputs.sequences[:, input_tokens.input_ids.shape[1]:], alignments, confidence)

    def get_alignments(self, pred_outputs, prompt_len, top_k=10):
        attentions = [
            attn[-1].mean(dim=1)[:, 0]  # Mean over heads, then get last token's attention
            for attn in pred_outputs.attentions
        ]

        out_seq_len = pred_outputs.sequences.shape[-1] - prompt_len
        aligned_tokens = []

        for out_idx in range(out_seq_len):
            if out_idx >= len(attentions):
                break

            alignment = attentions[out_idx][0]
            top_indices = alignment.topk(top_k).indices.tolist()
            aligned_tokens.append(top_indices)

        return aligned_tokens

    def get_confidence(self, outputs):
        confidences = []
        scores = outputs.scores
        generated_tokens = outputs.sequences[:, -len(scores):]

        for step, (logits, tokens) in enumerate(zip(scores, generated_tokens.T)):
            probs = F.softmax(logits, dim=-1)
            batch_conf = probs[range(probs.size(0)), tokens]
            confidences.extend(batch_conf.tolist())
        
        return confidences

    def predict(
        self,
        instance: DatasetInstance,
        config: Optional[InferenceConfig] = None,
    ) -> PredictionResult:
        self.model.eval()

        with torch.no_grad():
            prompt = self.prepare_prompt(
                instance.source,
                instance.source_lang.value,
                instance.target_lang.value
            )
            print("Prompt:\n",prompt)
            tokenized_input = self.tokenize(prompt)

            pred = self.infer(
                tokenized_input, config
            )
            gc.collect()
        print("Decoded output:\n", self.decode(pred[1]))
        return PredictionResult(
            instance_id=instance.instance_id,
            source=pred[0],
            pred=pred[1],
            alignments=pred[2],
            confidence=pred[3],
            pred_dec=self.decode(pred[1])
        )
