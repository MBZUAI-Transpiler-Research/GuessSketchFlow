from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import pickle
from src.helpers.model import PredictionResult


dataset = load_dataset("ahmedheakl/asm2asm_bench_armv8_O0", split="train")
model_name = "ahmedheakl/ex19_qwen2.5-1.5b-1M-stack-16kcw"
print("Loading the model ...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded ...")

def inference(asm_x86: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"""Convert the following x86 assembly code to ARM assembly:\n```x86asm\n{asm_x86}\n```""",
        },
    ]
    max_new_tokens = 32_000
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    max_new_tokens = max_new_tokens - len(model_inputs['input_ids'][0])
    generated_outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        num_return_sequences=1,
        output_attentions=True,                
        return_dict_in_generate=True,          
        output_scores=True                     
    )
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_outputs.sequences)
    ]

    response: str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    # Call alignment extractor
    alignments = get_alignments(generated_outputs, model_inputs.input_ids.shape[1])

    pred_dec = (
        response
        .split("```armasm\n")[-1]
        .split("```")[0]
        )
    
    return pred_dec, alignments

def get_alignments(pred_outputs, prompt_len, top_k=10):
    attentions = [
        attn[-1].mean(dim=1)[:, 0]  # mean over heads, get attention to last token
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
    data = {
        "pred": [], 
        "gt": [], 
        "input": [], 
        "files": [],
        "ed": [],
        "alignment": []
    }
    for i in tqdm(range(len(dataset))):
        example = dataset[i]
        try:
            example = dataset[i]
            gt = example["arm"]
            input_asm = example["x86"]
            pred, alignments = inference(input_asm)
            data["pred"].append(pred)
            data["gt"].append(gt)
            data["input"].append(input_asm)
            data["files"].append(example["file"])
            data["alignment"].append(alignments)
            distance = edit_distance_assembly(gt, pred)
            print(distance)
            data['ed'].append(distance)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in file {example['file']}: {e}")

    with open("eval_armv8_O0_b4.json", "w") as f:
        json.dump(data, f, indent=4)
    
    with open("eval_armv8_O0_b4.json", "r") as f:
        data = json.load(f)
    
    predictions = {}

    for i in range(len(data["input"])):
        input_x86 = data["input"][i]
        pred_arm = data["pred"][i]
        file_name = data["files"][i]  # example: eval/problem1/code.c

        instance_id = file_name 

        # Tokenize input x86 code
        tokenized_input = tokenizer(
            input_x86, return_tensors="pt", padding=True, truncation=True
        )
        source_tokens = tokenized_input["input_ids"]

        # Tokenize predicted arm code
        tokenized_pred = tokenizer(
            pred_arm, return_tensors="pt", padding=True, truncation=True
        )
        pred_tokens = tokenized_pred["input_ids"]

        alignments = data["alignment"][i]

        # Build PredictionResult
        result = PredictionResult(
            instance_id=instance_id,
            source=source_tokens,
            pred=pred_tokens,
            alignments=alignments,
            pred_dec=pred_arm
        )

        predictions[instance_id] = result

    # Save to predictions.pkl
    with open("predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    print(f"Saved {len(predictions)} predictions to predictions.pkl.")

    # Load the pickle
    with open("predictions.pkl", "rb") as f:
        predictions = pickle.load(f)

    # Now predictions is a dictionary: { instance_id : PredictionResult }

    # Inspect the keys
    print(list(predictions.keys())[:10])  # Print first 10 instance_ids

    # Pick one entry
    sample_id = list(predictions.keys())[0]
    sample_prediction = predictions[sample_id]

    print(f"Instance ID: {sample_prediction.instance_id}")
    print(f"Decoded Prediction (pred_dec):\n{sample_prediction.pred_dec[:500]}")  # Show first 500 chars
    print(f"Alignments (if any): {sample_prediction.alignments}")


if __name__ == "__main__":
    main()