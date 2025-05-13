import logging
import time
from typing import Dict, List, Optional, Union
import warnings

import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.helpers.dataset import DatasetInstance
from src.helpers.model import Model, InferenceConfig, PredictionResult

logger = logging.getLogger(__name__)


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
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(
            f"Initializing QwenModel with {model_name} on device {device}"
        )

        start_time = time.time()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)

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

    def infer(
        self,
        input_tokens: Dict[str, torch.Tensor],
        config: Optional[InferenceConfig] = None,
        **kwargs
    ):
        if config is None:
            config = InferenceConfig()

        logger.info(
            f"Starting inference with temperature={config.temperature}, "
            f"max_length={config.max_length}"
        )
        logger.info(
            f"Input shape: {input_tokens['input_ids'].shape}"
        )

        # Dynamically calculate max_new_tokens
        context_size = 15500
        input_token_count = input_tokens["input_ids"].shape[1]
        max_new_tokens = max(context_size - input_token_count, 1000)  # Ensure minimum

        # Fallback if input is still too large
        if input_token_count > context_size - 1000:
            logger.warning(f"Input exceeds safe context limit ({input_token_count} tokens). Truncating...")
            input_tokens["input_ids"] = input_tokens["input_ids"][:, :(context_size - 2000)]
            input_tokens["attention_mask"] = input_tokens["attention_mask"][:, :(context_size - 2000)]
            input_token_count = input_tokens["input_ids"].shape[1]
            max_new_tokens = max(context_size - input_token_count, 1000)

        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=config.temperature,
            num_beams=config.beam_size,
            num_return_sequences=config.num_return_sequences,
            do_sample=True,
            early_stopping=True,
            output_attentions=True,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        alignments = self.get_alignments(
            outputs, input_tokens.input_ids.shape[1])
        confidence = self.get_confidence(outputs)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (input_tokens.input_ids, outputs.sequences[:, input_tokens.input_ids.shape[1]:], alignments, confidence)

    # def get_alignments(self, pred_outputs, prompt_len, top_k=10):
    #     attentions = [
    #         # Mean over heads, then get last token's attention
    #         attn[-1].mean(dim=1)[:, 0]
    #         for attn in pred_outputs.attentions
    #     ]

    #     out_seq_len = pred_outputs.sequences.shape[-1] - prompt_len
    #     aligned_tokens = []

    #     for out_idx in range(out_seq_len):
    #         if out_idx >= len(attentions):
    #             break

    #         alignment = attentions[out_idx][0]
    #         top_indices = alignment.topk(top_k).indices.tolist()
    #         aligned_tokens.append(top_indices)

    #     return aligned_tokens

    def get_alignments(self, pred_outputs, prompt_len, top_k=5, batch_size=1):
        """
        Memory-efficient alignment extraction that handles large attention maps.

        Args:
            pred_outputs: Generation outputs with attention maps
            prompt_len: Length of the input prompt
            top_k: Number of top attention indices to keep
            batch_size: Number of attention maps to process at once

        Returns:
            List of top-k attended token indices for each output token
        """
        out_seq_len = pred_outputs.sequences.shape[-1] - prompt_len
        aligned_tokens = []

        for idx in range(len(pred_outputs.attentions)):
            if idx >= len(pred_outputs.attentions):
                continue

            try:
                # Get only the last layer
                attn = pred_outputs.attentions[idx][-1]

                # Move to CPU immediately and keep only what we need
                last_layer_attn = attn.mean(dim=1)[:, 0].detach().cpu()
                del attn

                if last_layer_attn.shape[1] > prompt_len:
                    relevant_attn = last_layer_attn[:, :prompt_len]
                else:
                    relevant_attn = last_layer_attn

                top_k_actual = min(top_k, relevant_attn.shape[1])
                top_indices = relevant_attn[0].topk(top_k_actual).indices.tolist()
                aligned_tokens.append(top_indices)

                del last_layer_attn
                del relevant_attn

            except Exception as e:
                print(f"Warning: Error processing attention at index {idx}: {e}")
                aligned_tokens.append([0] * top_k)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
    
    def edit_distance_assembly(self, gt: str, pred: str) -> int:
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        def preprocess(code):
            code = "\n".join(line.split(";")[0].strip() for line in code.split("\n"))
            code = "\n".join(line for line in code.split("\n") if line.strip())
            return code

        gt_processed = preprocess(gt)
        pred_processed = preprocess(pred)
        return levenshtein(gt_processed, pred_processed)

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
            tokenized_input = self.tokenize(prompt)

            pred = self.infer(
                tokenized_input, config
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        decoded = self.decode(pred[1])
        print("Decoded output:\n", decoded)
        distance = self.edit_distance_assembly(instance.target, decoded)
        print(f"Levenshtein distance: {distance}")

        return PredictionResult(
            instance_id=instance.instance_id,
            source=pred[0],
            pred=pred[1],
            alignments=pred[2],
            confidence=pred[3],
            pred_dec=decoded,
            levenshtein_distance=distance
        )
