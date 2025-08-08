import pickle
import torch
from transformers import AutoTokenizer

with open("predictions.pkl", "rb") as f:
    predictions = pickle.load(f)

print(len(predictions.keys()))

tokenizer = AutoTokenizer.from_pretrained("ahmedheakl/gg-armv8-O0")

#names = [f"problem{i}.x86.s::_func0" for i in range(1,165)]
names = ["problem100.x86.s::_func0"]

#name = "problem1.x86.s::_func0"

totals = []

beam_idx = 0  # Change to 1 for second beam
print("beam numer is",beam_idx)

for name in names:
    result = predictions[name]

    print("---------------------------------------------")
    print("name is", name)
    
    print("beam numer is",beam_idx)
    tokens = result.pred[beam_idx]  # List of token IDs (after prompt)
    confidences = result.confidence[beam_idx]  # List of floats
    alt_tokens = result.alt_tokens[beam_idx]  # List of top-k lists

    bob = tokenizer.decode(tokens)
    print("predictedoutput\n")
    print(bob)
    print("ground truth\n")
    print(tokenizer.decode(result.source))

    count = 0

    # Write everything to output_debug.txt
    with open("output_debug.txt", "w") as out:
        out.write(f"Problem Name: {name}\n\n")
        out.write(f"--- Tokens for BEAM {beam_idx} ---\n")

        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        for i, (tok_id, decoded) in enumerate(zip(tokens, decoded_tokens)):
            conf = confidences[i]
            alts = alt_tokens[i]
            in_topk = any(tok_id == alt_id for alt_id, _ in alts)
            alt_match_conf = next((p for alt_id, p in alts if alt_id == tok_id), None)

            out.write(f"\nToken {i}: {repr(decoded)}\n")
            out.write(f"  ID: {tok_id}\n")
            out.write(f"  Confidence: {conf:.6f}\n")
            out.write(f"  In top-k: {in_topk}\n")
            out.write(f"  Alt_tokens: {alts}\n")
            if in_topk:
                out.write(f"  Matched confidence in alt_tokens: {alt_match_conf:.6f}\n")
            else:
                out.write(f"  WARNING NOT in alt_tokens: {alts}\n")
                count += 1
                topk_ids = [tid for tid, _ in alts]
                #print(f"Token {i}: {decoded!r}")
                #print(f"  Actual token ID: {tok_id}")
                #print(f"  Top-k IDs: {topk_ids}")
                if i < len(tokens) - 1:
                    #print(f"  Next token ID: {tokens[i+1]}  -> {tokenizer.decode([tokens[i+1]])}")
                    continue
            decoded_alts = [(tokenizer.decode([tid]), prob) for tid, prob in alts]
            out.write(f"  Top-k: {decoded_alts}\n")

    totals.append([name, len(tokens), count])


for t in totals: print(t)