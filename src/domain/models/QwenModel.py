from typing import Dict, List, Optional, Union, Tuple
import warnings
import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from src.helpers.dataset import DatasetInstance, ClozeInstance
from src.helpers.model import Model, ModelConfig, PredictionResult
import pickle   

class QwenModel(Model):
    def __init__(self, model_name: str):

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
        self.model.eval() # ensures no gradients
        tokenizer = AutoTokenizer.from_pretrained(model_name) # sets self.tokenizer()
        super().__init__(tokenizer)

    # Builds a language-model-friendly prompt from the source code.
    # Uses Qwen's chat template format for better alignment with training behavior. Returns a formatted prompt string (not tokenized).
    def prepare_prompt(self, source_code: str, source_lang: str, target_lang: str) -> str:
        user_prompt = (
            f"Convert the following {source_lang} assembly code to {target_lang} assembly:\n"
            f"```{source_lang.lower()}asm\n{source_code}```"
        )
        messages = [{"role": "user", "content": user_prompt}]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) #tokenize separately with next function
        return chat_text

    #Returns a BatchEncoding object that includes: input_ids: the actual token IDs, attention_mask: 1s for tokens to attend to, 0s for padding
    def tokenize(self, text: Union[str, List[str]], **kwargs) -> BatchEncoding:
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, **kwargs)
        return tokens

    # Decodes token IDs back into string(s).Automatically handles both single sequences and batches. Skips special tokens (e.g., <|endoftext|>).
    def decode(self, token_ids: torch.Tensor, **kwargs) -> Union[str, List[str]]:
        if token_ids.dim() == 1:  # single sequence
            token_ids = token_ids.unsqueeze(0)
        decoded = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return decoded[0] if token_ids.size(0) == 1 else decoded

    def infer(self, input_tokens: Dict[str, torch.Tensor], config: ModelConfig, instance_id: Optional[str] = None, **kwargs):
        # Dynamically calculate max_new_tokens. Remember: Hugging Face's tokenizer always adds a batch dimension
        context_size = 8000
        input_token_count = input_tokens["input_ids"].shape[1] # input_tokens is shape [batch_size, sequence_length], so this gets length
        max_new_tokens = max(context_size - input_token_count, 1000)  # Ensure minimum

        # Fallback if input is still too large
        if input_token_count > context_size - 1000: #TRUE if max_new_tokens gets set to minimum
            input_tokens["input_ids"] = input_tokens["input_ids"][:, :(context_size - 2000)] #truncates inputs tokens to ensure at least 2000 new tokens
            input_tokens["attention_mask"] = input_tokens["attention_mask"][:, :(context_size - 2000)] #makes sure attention mask is same size as input_ids
            input_token_count = input_tokens["input_ids"].shape[1] #gives updated sequence length after truncating
            max_new_tokens = max(context_size - input_token_count, 1000) #updates to new value --- should be 2000
        print(f"Inferencing {instance_id} ...")
        with torch.no_grad():
            outputs = self.model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens,
                temperature=config.temperature,
                num_beams=config.num_beams,
                num_return_sequences=config.num_return_sequences,
                do_sample=False,
                early_stopping=True,
                output_attentions=True,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        input_len = input_tokens["input_ids"].shape[1]  # prompt length

        print("Finished Inferencing ...")
        alignments = self.get_alignments(outputs, input_len, config)
        confidence, alt_tokens = self.get_confidence(outputs, input_len)

        # Move tokens to CPU
        source_ids = input_tokens["input_ids"].detach().cpu()[0] #[0] removes the batch dimensions
        pred_ids = [
            outputs.sequences[i, input_len:].detach().cpu()
            for i in range(outputs.sequences.shape[0])
        ]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return PredictionResult(
            instance_id=instance_id,
            source=source_ids,
            pred=pred_ids,
            alignments=alignments,
            confidence=confidence,
            alt_tokens=alt_tokens
        )

    def get_alignments(self, pred_outputs, prompt_len, config: ModelConfig) -> List[List[List[int]]]:

        all_aligned = []
        num_beams = pred_outputs.sequences.shape[0]  # e.g., 3 if num_return_sequences=3
        num_output_tokens = len(pred_outputs.attentions)  # Number of tokens generated

        for beam_idx in range(num_beams):
            aligned_tokens = []

            for idx in range(num_output_tokens):  # One step per generated token
                try:
                    # Get the attention for the final layer at this generation step
                    attn = pred_outputs.attentions[idx][-1]  # shape: [batch, heads, 1, key_len] 
                    
                    # Select attention for this beam
                    attn_beam = attn[beam_idx]  # shape: [heads, 1, key_len]
                    
                    # Average across all heads → shape: [1, key_len]
                    avg = attn_beam.mean(dim=0)  # shape: [1, key_len]
                    
                    # Remove the query dim and restrict to original prompt
                    this_token_attn = avg[0, :prompt_len].detach().cpu()  # shape: [key_len]

                    # Top-k input tokens most attended by this output token
                    top_k = config.k
                    top_k_actual = min(top_k, this_token_attn.shape[0])  # avoid overflow on early steps
                    top_indices = this_token_attn.topk(top_k_actual).indices.tolist()

                    # Pad if fewer than k (edge case)
                    while len(top_indices) < top_k:
                        top_indices.append(0)

                    aligned_tokens.append(top_indices)

                    # Cleanup
                    del attn, attn_beam, avg, this_token_attn, top_indices

                except Exception as e:
                    print(f"Warning: Error in beam {beam_idx}, token {idx}: {e}")
                    aligned_tokens.append([0] * config.k)

                # Optional: clear GPU memory at each step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_aligned.append(aligned_tokens)

            # Optional: force garbage collection per beam
            gc.collect()

        return all_aligned  # shape: [num_beams][num_tokens][k]


    def get_confidence(self, outputs, input_len: int, top_k: int = 3) -> List[List[float]], List[List[List[Tuple[int, float]]]]]:

        num_beams = outputs.sequences.shape[0]
        num_steps = len(outputs.scores)

        confidence = []  # per-beam list of confidences for generated tokens
        alt_tokens = []  # per-beam list of top-k token predictions per step

        for beam_idx in range(num_beams):
            beam_conf = []
            beam_alt = []

            # Get the generated part only (strip prompt)
            generated_tokens = outputs.sequences[beam_idx][input_len:]

            assert len(generated_tokens) == num_steps, f"Mismatch: {len(generated_tokens)} tokens vs {num_steps} scores"

            for step_idx, token_id in enumerate(generated_tokens):
                logits = outputs.scores[step_idx][beam_idx]  # shape: [vocab_size]
                probs = F.softmax(logits, dim=-1)

                # Confidence of the chosen token
                token_conf = probs[token_id].item()
                beam_conf.append(token_conf)

                # Top-k alternative tokens
                topk = torch.topk(probs, k=top_k)
                alt = [(i.item(), p.item()) for i, p in zip(topk.indices, topk.values)]
                beam_alt.append(alt)

            confidence.append(beam_conf)
            alt_tokens.append(beam_alt)

        return confidence, alt_tokens


    def predict(self, instance: ClozeInstance, config: ModelConfig) -> PredictionResult:
        # 1. Ensure model is in evaluation mode (safe to omit if already set in __init__)
        self.model.eval()

        # 2. Disable gradient tracking for efficient inference
        with torch.no_grad():
            # 3. Prepare the prompt from the source code, tokenize it, and run inference, returning a PredictionResult
            #prompt = self.prepare_prompt(instance.source, instance.source_lang, instance.target_lang)
            prompt = self.prepare_prompt(instance.source_fn_code, instance.source_lang, instance.target_lang)
            tokenized_input = self.tokenize(prompt)
            device = next(self.model.parameters()).device # # Move to any CUDA device from CPU; HF handles multi-GPU dispatch internally
            tokenized_input = BatchEncoding({k: v.to(device) for k, v in tokenized_input.items()})
            result = self.infer(tokenized_input, config, instance.instance_id)

            # 4. Perform manual garbage collection and clear unused CUDA memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 5. Print the ground truth and the decoded prediction for inspection
        print(f"[{instance.instance_id}]\nGROUND TRUTH:\n{instance.target_fn_code}")
        for i, pred_ids in enumerate(result.pred):
            decoded = self.decode(pred_ids)
            print(f"BEAM {i}:\n{decoded}\n")

        # 6. Delete temporary variables and clean up memory again
        del tokenized_input, prompt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 7. Return the final prediction result
        return result