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
                do_sample=True,
                early_stopping=True,
                output_attentions=True,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        #Dump to pkl file
        with open("outputs_debug.pkl", "wb") as f:
            pickle.dump(outputs, f)

        input_len = input_tokens["input_ids"].shape[1]  # prompt length

        print("Finished Inferencing ...")
        alignments = self.get_alignments(outputs, input_len, config)
        confidence, alt_tokens = self.get_confidence(outputs, input_len)

        print("DEBUG SHAPE:", len(alt_tokens), len(alt_tokens[0]))
        print("DEBUG FIRST ENTRY:", alt_tokens[0][0])

        # Move tokens to CPU
        source_ids = input_tokens["input_ids"].detach().cpu()[0] #[0] removes the batch dimensions
        #pred_ids = outputs.sequences[:, input_tokens["input_ids"].shape[1]:].detach().cpu()[0]
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


    # def get_alignments(self, pred_outputs, prompt_len, config: ModelConfig):
    #     aligned_tokens = []

    #     for idx in range(len(pred_outputs.attentions)): #iterate through all the output tokens
    #         try:
    #             attn = pred_outputs.attentions[idx][-1]  # gets last layer for a specific output token
    #             # attn dimensions - step 0: [batch_size, num_heads, prompt_len, prompt_len]
    #             # step i > 0: [batch_size, num_heads, 1, prompt_len + i] due to caching 
    #             avg = attn.mean(dim=1)  # average across heads -> [batch_size, query_len, key_len]
    #             this_token_attn = avg[:, -1].detach().cpu()  # select last query -> [batch_size, key_len], need to detach so we can send to list later
    #             del attn, avg

    #             # Keep only attention over the original prompt tokens (exclude attention to generated outputs).
    #             relevant_attn = this_token_attn[:, :prompt_len]

    #             top_k = config.k
    #             top_k_actual = min(top_k, relevant_attn.shape[1]) # ensures we don't try to take more tokens than exist

    #             #relevant_attn has dimensions [1, seq_len] so this gets the single attention vector
    #             #we use topk to get the topk values for this idx step and put their positions into a list and append it at position idx to aligned_tokens
    #             top_indices = relevant_attn[0].topk(top_k_actual).indices.tolist()
    #             aligned_tokens.append(top_indices)

    #             del this_token_attn, relevant_attn

    #         except Exception as e:
    #             print(f"Warning: Error processing attention at index {idx}: {e}")
    #             aligned_tokens.append([0] * top_k) # just fill with 0's if there is an error

    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

    #     return aligned_tokens

    def get_confidence(self, outputs, prompt_len: int, n: int = 3) -> Tuple[List[List[float]], List[List[List[Tuple[int, float]]]]]:
        """
        Returns:
            confidences: List of shape [num_beams][num_generated_tokens]
            alt_tokens: List of shape [num_beams][num_generated_tokens][top_n]
        """
        scores = outputs.scores  # List of length = num_generated_tokens, each [num_beams, vocab_size]
        beam_indices = getattr(outputs, "beam_indices", None)

        num_beams = outputs.sequences.shape[0]
        num_generated_tokens = len(scores)

        # Only the generated portion (excludes prompt)
        generated_tokens = outputs.sequences[:, -num_generated_tokens:]  # [num_beams, num_generated_tokens]

        all_confidences = []
        all_alt_tokens = []

        for b in range(num_beams):
                beam_conf = []
                beam_alts = []

                for t in range(num_generated_tokens):
                    token_id = generated_tokens[b, t].item()
                    #token_id = outputs.sequences[b, prompt_len + t].item()

                    # if beam_indices is not None:
                    #     slot = beam_indices[b, prompt_len + t].item()
                    # else:
                    #     slot = b  # fallback
                    logits = scores[t][b]

                    #logits = scores[t][slot]
                    probs = F.softmax(logits, dim=-1)
                    
                    beam_conf.append(probs[token_id].item())
                    topk = probs.topk(n)
                    beam_alts.append([(i.item(), p.item()) for i, p in zip(topk.indices, topk.values)])

                all_confidences.append(beam_conf)
                all_alt_tokens.append(beam_alts)

        return all_confidences, all_alt_tokens


    ### COME BACK TO THIS 4 August 2025
    # def get_confidence(self, outputs, n: int = 3) -> Tuple[List[List[float]], List[List[List[Tuple[int, float]]]]]:
    #     """
    #     Returns:
    #         confidences: List of shape [num_beams][num_generated_tokens]
    #         alt_tokens: List of shape [num_beams][num_generated_tokens][top_n]
    #     """
    #     all_confidences = []
    #     all_alt_tokens = []

    #     scores = outputs.scores  # List of [num_beams, vocab_size], length = num_generated_tokens
    #     num_beams = outputs.sequences.shape[0]
    #     num_generated_tokens = len(scores)

    #     # Get just the generated portion of sequences (exclude prompt)
    #     generated_tokens = outputs.sequences[:, -num_generated_tokens:]  # shape: [num_beams, num_generated_tokens]

    #     for beam_idx in range(num_beams):
    #         confidences = []  # list of floats for this beam
    #         alt_tokens = []   # list of top-n alternatives for each token

    #         for step, token_id in enumerate(generated_tokens[beam_idx]):
    #             logits = scores[step]             # shape: [num_beams, vocab_size]
    #             probs = F.softmax(logits[beam_idx], dim=-1)  # shape: [vocab_size]

    #             # Confidence of the chosen token
    #             confidence = probs[token_id].item()
    #             confidences.append(confidence)

    #             # Top-n alternative tokens and their probabilities
    #             topk = probs.topk(n)
    #             alternatives = [(i.item(), p.item()) for i, p in zip(topk.indices, topk.values)]
    #             alt_tokens.append(alternatives)

    #         all_confidences.append(confidences)
    #         all_alt_tokens.append(alt_tokens)

    #     return all_confidences, all_alt_tokens  # shape: [num_beams][num_tokens][n]


    # def get_confidence(self, outputs, n: int = 3):
    #     confidences = []
    #     all_alt_tokens = []     # List of top-n alternatives for each token
    #     scores = outputs.scores
    #     generated_tokens = outputs.sequences[:, -len(scores):] #remember: outputs includes the inputs, so focus only on generated portion

    #     # len(scores) = num_generated_tokens, and each score tensor has shape [batch_size, vocab_size].
    #     # generated_tokens has shape [batch_size, num_generated_tokens], so its transpose has shape [num_generated_tokens, batch_size].
    #     # Since both scores and generated_tokens.T have num_generated_tokens as their first dimension, we can zip them together.
    #     for step, (logits, tokens) in enumerate(zip(scores, generated_tokens.T)):  # iterate over each generated token
    #         probs = F.softmax(logits, dim=-1)  # softmax across the vocab dimension (last dim), shape [batch_size, vocab_size]
    #         # tokens has shape [batch_size], where each entry is the token ID chosen/generated at this step.
    #         # probs[range(probs.size(0)), tokens] performs advanced indexing: for each row in the batch, select the probability of the chosen token.
    #         batch_conf = probs[range(probs.size(0)), tokens].detach().cpu()  # detach for tolist()
    #         confidences.extend(batch_conf.tolist())  # use extend to keep confidences as a flat list    

    #         # Top-n alternatives for each batch element
    #         alt_toks = []
    #         sorted_probs = probs.sort(descending=True)
    #         for tok_id, prob in zip(sorted_probs.indices[0, :n], sorted_probs.values[0, :n]):
    #             alt_toks.append((tok_id.item(), prob.item()))
    #         all_alt_tokens.append(alt_toks)

    #     return confidences, all_alt_tokens

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
        #print(f"[{instance.instance_id}]\nGROUND TRUTH:\n{instance.target_fn_code}\nPREDICTION:\n{self.decode(result.pred)}\n")
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


