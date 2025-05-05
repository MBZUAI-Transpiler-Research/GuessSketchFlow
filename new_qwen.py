from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import pickle
from src.helpers.model import PredictionResult
import gc
import torch.nn.functional as F

#dataset = load_dataset("ahmedheakl/asm2asm_bench_armv8_O0", split="train")
dataset = load_dataset("ahmedheakl/asm2asm_bringup_O0", split="train")
model_name = "ahmedheakl/ex19_qwen2.5-1.5b-1M-stack-16kcw"
print("Loading the model ...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded ...")

def inference(asm_x86: str) -> tuple:
    """
    Inference function that handles large inputs by truncating when necessary
    and ensures positive max_new_tokens.
    """
    # Calculate approximate token count - rough estimate based on whitespace
    estimated_token_count = len(asm_x86.split()) * 1.3  # Adjust multiplier as needed
    
    # If input is likely to exceed model context, truncate it
    max_input_tokens = 12000  # Leave room for generation within 16K context
    if estimated_token_count > max_input_tokens:
        print(f"Warning: Input likely exceeds token limit. Truncating from ~{int(estimated_token_count)} tokens.")
        # Truncate by lines - more readable than mid-sentence truncation
        lines = asm_x86.split("\n")
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = len(line.split()) * 1.3
            if current_tokens + line_tokens > max_input_tokens:
                break
            truncated_lines.append(line)
            current_tokens += line_tokens
            
        asm_x86 = "\n".join(truncated_lines)
        print(f"Truncated to ~{int(current_tokens)} tokens")

    messages = [
        {
            "role": "user",
            "content": f"""Convert the following x86 assembly code to ARM assembly:\n```x86asm\n{asm_x86}\n```""",
        },
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the input and check actual token count
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    input_tokens = len(model_inputs['input_ids'][0])
    
    # Calculate max_new_tokens, ensuring it's positive
    context_size = 15500  # Set below model's actual max to be safe
    max_new_tokens = max(context_size - input_tokens, 1000)  # Ensure minimum generation length
    
    # For very large inputs that still exceed limits, we need to truncate tokenized input
    if input_tokens > context_size - 1000:
        print(f"Input exceeds safe limit: {input_tokens} tokens. Truncating to {context_size - 2000} tokens.")
        # Truncate input_ids directly
        model_inputs['input_ids'] = model_inputs['input_ids'][:, :(context_size - 2000)]
        model_inputs['attention_mask'] = model_inputs['attention_mask'][:, :(context_size - 2000)]
        input_tokens = len(model_inputs['input_ids'][0])
        max_new_tokens = max(context_size - input_tokens, 1000)
    
    print(f"Input tokens: {input_tokens}, Max new tokens for generation: {max_new_tokens}")
    
    # Generate with memory-efficient settings
    generated_outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        num_beams=2,  # Reduce beam size to save memory
        num_return_sequences=1,
        output_attentions=True,                
        return_dict_in_generate=True,          
        output_scores=True,
        do_sample=True,  # Use greedy decoding to save memory
        early_stopping=True
    )
    
    confidence = get_confidence(generated_outputs)

    # Move to CPU immediately to free GPU memory
    generated_ids = [
        output_ids[len(input_ids):].detach().cpu()
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_outputs.sequences)
    ]
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Decode the output
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Extract ARM code
    try:
        pred_dec = (
            response
            .split("```armasm\n")[-1]
            .split("```")[0]
        )
    except:
        # Fallback in case the formatting is off
        pred_dec = response
    
    # Process alignments with small batch size
    try:
        alignments = get_alignments(
            generated_outputs, 
            model_inputs.input_ids.shape[1], 
            top_k=5,  # Reduce from 10 to 5 to save memory
            batch_size=1  # Process one attention map at a time
        )
    except Exception as e:
        print(f"Warning: Failed to extract alignments: {e}")
        alignments = []  # Return empty alignments in case of error
    
    # Free memory
    del generated_outputs
    del model_inputs
    del generated_ids
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return pred_dec, alignments, confidence

# def get_alignments(pred_outputs, prompt_len, top_k=10):
#     attentions = [
#         attn[-1].mean(dim=1)[:, 0]  # mean over heads, get attention to last token
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

def get_alignments(pred_outputs, prompt_len, top_k=5, batch_size=1):
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
    
    # Handle case where sequence is too long - take subset of attention maps
    max_attn_to_process = min(100, len(pred_outputs.attentions))
    process_indices = list(range(0, max_attn_to_process, max(1, max_attn_to_process // 100)))
    
    for idx in process_indices:
        if idx >= len(pred_outputs.attentions):
            continue
            
        # Process just this single attention map
        try:
            # Get only the last layer
            attn = pred_outputs.attentions[idx][-1]
            
            # Move to CPU immediately and keep only what we need
            last_layer_attn = attn.mean(dim=1)[:, 0].detach().cpu()
            del attn
            
            # Get top indices
            if last_layer_attn.shape[1] > prompt_len:
                # If attention is over full sequence, get attention to input only
                relevant_attn = last_layer_attn[:, :prompt_len]
            else:
                relevant_attn = last_layer_attn
                
            # Get top-k on CPU
            top_k_actual = min(top_k, relevant_attn.shape[1])
            top_indices = relevant_attn[0].topk(top_k_actual).indices.tolist()
            aligned_tokens.append(top_indices)
            
            # Free memory
            del last_layer_attn
            del relevant_attn
            
        except Exception as e:
            print(f"Warning: Error processing attention at index {idx}: {e}")
            aligned_tokens.append([0] * top_k)  # Placeholder
            
        # Force cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return aligned_tokens

def get_confidence(outputs):
    confidences = []
    scores = outputs.scores
    generated_tokens = outputs.sequences[:, -len(scores):]

    for step, (logits, tokens) in enumerate(zip(scores, generated_tokens.T)):
        probs = F.softmax(logits, dim=-1)
        batch_conf = probs[range(probs.size(0)), tokens]
        confidences.extend(batch_conf.tolist())
    
    return confidences

def edit_distance_assembly(gt: str, pred: str) -> int:
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
    distance = levenshtein(gt_processed, pred_processed)

    return distance



def main():
    print("Running")
    
    # Set PyTorch memory efficiency options
    if torch.cuda.is_available():
        # Enable TF32 math for faster processing
        torch.backends.cuda.matmul.allow_tf32 = True
        # Set expandable segments for better memory allocation
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    data = {
        "pred": [], 
        "gt": [], 
        "input": [], 
        "files": [],
        "ed": [],
        "alignment": [],
        "confidence": []
    }
    
    # Track successful conversions
    success_count = 0
    
    # Determine which examples to process (skip known problematic ones)
    extremely_large_files = ["anagram.s", "banner.s", "fft-int.s", "c-interp.s", "rho-factor.s", "satomi.s"]
    
    for i in tqdm(range(len(dataset))):
        try:
            example = dataset[i]
            file_name = example["file"]
            print(f"Processing file: {file_name}")
            
            # Skip known problematic files
            if file_name in extremely_large_files:
                print(f"Skipping {file_name} - known to be too large")
                continue
                
            # Get input and ground truth
            input_asm = example["x86"]
            gt = example["arm"]
            
            # Skip if input is extremely large (based on line count)
            line_count = len(input_asm.split("\n"))
            if line_count > 5000:
                print(f"Skipping {file_name} - too large ({line_count} lines)")
                continue
                
            # Attempt inference
            try:
                pred, alignments, confidence = inference(input_asm)
                success_count += 1
                
                # Store results
                data["pred"].append(pred)
                data["gt"].append(gt)
                data["input"].append(input_asm)
                data["files"].append(file_name)
                data["alignment"].append(alignments)
                data["confidence"].append(confidence)
                
                # Calculate edit distance
                distance = edit_distance_assembly(gt, pred)
                print(f"Edit distance: {distance}")
                data['ed'].append(distance)
                
                # Save after each successful conversion
                if success_count % 5 == 0:
                    print(f"Saving intermediate results after {success_count} successful examples")
                    with open(f"bringup_wAttention_progress.json", "w") as f:
                        json.dump(data, f, indent=4)
                
            except Exception as e:
                print(f"Error in processing {file_name}: {e}")
                continue

        except KeyboardInterrupt:
            print("Interrupted by user, saving current progress...")
            break
        except Exception as e:
            print(f"Error loading example {i}: {e}")
    
    # Save final results regardless of success count
    print(f"Successfully processed {success_count} examples out of {len(dataset)}")
    
    with open("bringup_wAttention.json", "w") as f:
        json.dump(data, f, indent=4)
    
    # Only process predictions if we have any successful conversions
    if success_count > 0:
        try:
            predictions = {}
            for i in range(len(data["input"])):
                input_x86 = data["input"][i]
                pred_arm = data["pred"][i]
                file_name = data["files"][i]
                instance_id = file_name
                
                # Create lightweight PredictionResult
                result = PredictionResult(
                    instance_id=instance_id,
                    source=input_x86,  # Skip tokenization 
                    pred=pred_arm,    # Skip tokenization
                    alignments=data["alignment"][i],
                    confidence=data["confidence"][i],
                )
                predictions[instance_id] = result
            
            # Save predictions
            with open("predictions.pkl", "wb") as f:
                pickle.dump(predictions, f)
            
            print(f"Saved {len(predictions)} predictions to predictions.pkl.")
            
        except Exception as e:
            print(f"Error saving predictions: {e}")
    else:
        print("No successful conversions to save.")

if __name__ == "__main__":
    main()