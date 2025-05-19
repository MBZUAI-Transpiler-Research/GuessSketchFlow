import logging
import time
from typing import Dict, List, Optional, Union
import warnings

import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.helpers.dataset import DatasetInstance
from src.helpers.model import Model, ModelConfig, PredictionResult, get_device

logger = logging.getLogger(__name__)


class QwenModel(Model):
    def __init__(self, model_name: str, device: Optional[str] = None):
        if device is None:
            device = get_device()

        start_time = time.time()

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.bfloat16,
        # ).to(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        self.model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        device_resolved = self.model.device
        # super().__init__(tokenizer, device=device_resolved)
        super().__init__(tokenizer, device=device_resolved)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Model initialization completed in {elapsed_time:.2f} seconds"
        )

    def prepare_prompt(self, source_code: str, source_lang: str, target_lang: str) -> str:
        user_prompt = (
            f"Convert the following {source_lang} assembly code to {target_lang} assembly:\n"
            f"```{source_lang.lower()}asm\n{source_code}```"
        )
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return chat_text

    def tokenize(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        logger.debug(f"Tokenizing text{'s' if isinstance(text, list) else ''} "
                     f"of length: {len(text) if isinstance(text, str) else len(text[0])}")

        tokens = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, **kwargs
        ).to(self.device)

        logger.debug(f"Tokenized shape: {tokens['input_ids'].shape}")
        return tokens

    def decode(self, token_ids: torch.Tensor, **kwargs) -> Union[str, List[str]]:
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
        config: ModelConfig,
        **kwargs
    ):
        logger.info(
            f"Starting inference with temperature={config.temperature}, "
            f"max_length={config.max_length}"
        )
        logger.info(
            f"Input shape: {input_tokens['input_ids'].shape}"
        )

        # Dynamically calculate max_new_tokens
        context_size = 8000
        input_token_count = input_tokens["input_ids"].shape[1]
        max_new_tokens = max(
            context_size - input_token_count,
            1000
        )  # Ensure minimum

        # Fallback if input is still too large
        if input_token_count > context_size - 1000:
            logger.warning(
                f"Input exceeds safe context limit ({input_token_count} tokens). Truncating...")
            input_tokens["input_ids"] = input_tokens["input_ids"][:, :(
                context_size - 2000)]
            input_tokens["attention_mask"] = input_tokens["attention_mask"][:, :(
                context_size - 2000)]
            input_token_count = input_tokens["input_ids"].shape[1]
            max_new_tokens = max(context_size - input_token_count, 1000)
        print(f"Inferencing {max_new_tokens} ...")
        with torch.no_grad():
            outputs = self.model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens,
                temperature=config.temperature,
                num_beams=config.num_beams,
                num_return_sequences=config.num_return_sequences,
                do_sample=True,
                early_stopping=True,
                output_attentions=True,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        # import matplotlib.pyplot as plt
        # save 3 random attention maps as heat maps
        # for i in range(3):
        #     try:
        #         attn = outputs.attentions[i][-1]
        #         attn = attn.mean(dim=1)[:, 0].detach().cpu()
        #         attn = attn.squeeze(0).numpy()
        #         # Save the attention map as a heatmap
        #         plt.imshow(attn, cmap='hot', interpolation='nearest')
        #         plt.colorbar()
        #         plt.savefig(f"attention_map_{i}.png")
        #         plt.close()
        #     except Exception as e:
        #         print(f"Error saving attention map {i}: {e}")
        print("Finished Inferencing ...")
        alignments = self.get_alignments(
            outputs, input_tokens.input_ids.shape[1]
        )
        confidence = self.get_confidence(outputs)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (input_tokens.input_ids, outputs.sequences[:, input_tokens.input_ids.shape[1]:], alignments, confidence)

    def get_alignments(self, pred_outputs, prompt_len, top_k=5, batch_size=1):
        out_seq_len = pred_outputs.sequences.shape[-1] - prompt_len
        aligned_tokens = []

        for idx in range(len(pred_outputs.attentions)):
            if idx >= len(pred_outputs.attentions):
                continue

            try:
                attn = pred_outputs.attentions[idx][-1]

                last_layer_attn = attn.mean(dim=1)[:, 0].detach().cpu()
                del attn

                if last_layer_attn.shape[1] > prompt_len:
                    relevant_attn = last_layer_attn[:, :prompt_len]
                else:
                    relevant_attn = last_layer_attn

                top_k_actual = min(top_k, relevant_attn.shape[1])
                top_indices = relevant_attn[0].topk(
                    top_k_actual).indices.tolist()
                aligned_tokens.append(top_indices)

                del last_layer_attn
                del relevant_attn

            except Exception as e:
                print(
                    f"Warning: Error processing attention at index {idx}: {e}")
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

    def predict(
        self,
        instance: DatasetInstance,
        config: ModelConfig,
    ) -> PredictionResult:
        self.model.eval()

        with torch.no_grad():
            prompt = self.prepare_prompt(
                instance.source,
                instance.source_lang,
                instance.target_lang
            )
            tokenized_input = self.tokenize(prompt)

            pred = self.infer(
                tokenized_input, config
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = PredictionResult(
            instance_id=instance.instance_id,
            source=pred[0].detach().cpu()[0],
            pred=pred[1].detach().cpu()[0],
            alignments=pred[2],
            confidence=pred[3],
        )

        del pred
        del tokenized_input
        del prompt

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
