# # # # import pickle
# # # # import torch
# # # # import torch.nn.functional as F


# # # # # Load the outputs_debug.pkl
# # # # with open("outputs_debug.pkl", "rb") as f:
# # # #     outputs = pickle.load(f)

# # # # # # Check the type and keys/attributes of outputs
# # # # # print(f"Type of outputs: {type(outputs)}")
# # # # # if hasattr(outputs, "__dict__"):
# # # # #     print(f"Attributes: {outputs.__dict__.keys()}")
# # # # # else:
# # # # #     print(dir(outputs))

# # # # # # Check sequences and scores
# # # # # if hasattr(outputs, "sequences"):
# # # # #     print(f"Sequences shape: {outputs.sequences.shape}")
# # # # #     print(f"First few tokens: {outputs.sequences[0, :10]}")

# # # # # if hasattr(outputs, "scores"):
# # # # #     print(f"Number of score steps: {len(outputs.scores)}")
# # # # #     print(f"Shape of first score tensor: {outputs.scores[0].shape}")
# # # # #     print(f"First few values of first score tensor: {outputs.scores[0][0, :10]}")

# # # # # # Check attentions
# # # # # if hasattr(outputs, "attentions"):
# # # # #     print(f"Type of outputs.attentions: {type(outputs.attentions)}")
# # # # #     print(f"Length of outputs.attentions: {len(outputs.attentions)}")
# # # # #     if len(outputs.attentions) > 0:
# # # # #         print(f"Type of outputs.attentions[0]: {type(outputs.attentions[0])}")


# # # # # step = 0  # examine the first generation step
# # # # # logits = outputs.scores[step]  # shape [1, vocab_size]
# # # # # probs = torch.softmax(logits, dim=-1)
# # # # # top_probs, top_ids = probs.topk(10)

# # # # # print("Top 10 tokens for step 0:")
# # # # # for p, i in zip(top_probs[0], top_ids[0]):
# # # # #     print(f"  Token {i.item()} -> prob = {p.item():.5f}")
# # # # # 1, 10, 100, 101
# # # # scores = outputs.scores  # List of tensors, one per step
# # # # print(f"Total steps: {len(scores)}")

# # # # low_conf_steps = []

# # # # for step_idx, score_tensor in enumerate(scores):
# # # #     # Convert logits to probabilities
# # # #     probs = F.softmax(score_tensor[0], dim=-1)

# # # #     max_prob = probs.max().item()
# # # #     if max_prob < 1.0:
# # # #         # Get top 3 tokens and probs
# # # #         top_probs, top_tokens = torch.topk(probs, 3)
# # # #         low_conf_steps.append((step_idx, max_prob, list(zip(top_tokens.tolist(), top_probs.tolist()))))

# # # # print(f"Steps where max prob < 1.0: {len(low_conf_steps)}")
# # # # for step_idx, max_prob, top3 in low_conf_steps:
# # # #     print(f"  Step {step_idx}: max_prob = {max_prob:.6f}")
# # # #     for token_id, prob in top3:
# # # #         print(f"    Token {token_id} -> prob = {prob:.5f}")

# # # from transformers import AutoTokenizer
# # # import pickle

# # # # Load tokenizer (must match your model)
# # # tokenizer = AutoTokenizer.from_pretrained("ahmedheakl/gg-armv8-O0")

# # # # Load outputs
# # # with open("outputs_debug.pkl", "rb") as f:
# # #     outputs = pickle.load(f)

# # # # Get the step where max_prob < 1.0
# # # step = 195  # from your debug output
# # # scores = outputs.scores[step][0]
# # # probs = scores.softmax(dim=-1)

# # # # Top 2 tokens
# # # top_probs, top_ids = probs.topk(2)
# # # for p, t in zip(top_probs, top_ids):
# # #     print(f"Token {t.item()} = '{tokenizer.decode([t.item()])}' with prob {p.item():.5f}")

# # import pickle
# # from transformers import AutoTokenizer

# # # Load the outputs
# # with open("outputs_debug.pkl", "rb") as f:
# #     outputs = pickle.load(f)

# # # Load the tokenizer
# # tokenizer = AutoTokenizer.from_pretrained("ahmedheakl/gg-armv8-O0")

# # # Step we are inspecting (from debug logs)
# # step = 195

# # # Extract sequence tokens
# # seq = outputs.sequences[0].tolist()  # Single batch, get as list

# # # Define the window
# # start = max(0, step - 10)
# # end = min(len(seq), step + 11)

# # # Slice tokens around the uncertain step
# # context_tokens = seq[start:end]

# # # Decode context
# # decoded_context = tokenizer.decode(context_tokens, skip_special_tokens=True)

# # print(f"Tokens {start} to {end} around step {step}:")
# # print(context_tokens)
# # print("\nDecoded context:")
# # print(decoded_context)

# import pickle
# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer

# PKL_PATH = "outputs_debug.pkl"
# MODEL_NAME = "ahmedheakl/gg-armv8-O0"
# WINDOW = 1
# TOPK = 5

# with open(PKL_PATH, "rb") as f:
#     outputs = pickle.load(f)

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# seq = outputs.sequences[0]                    # [total_len]
# scores = outputs.scores                       # list of [1, vocab] logits, len == generated_len
# total_len = seq.shape[0]
# generated_len = len(scores)
# prompt_len = total_len - generated_len        # <-- critical offset

# print(f"total_len={total_len}, generated_len={generated_len}, prompt_len={prompt_len}")

# uncertain = []
# for step_idx, logits in enumerate(scores):
#     probs = F.softmax(logits[0], dim=-1)
#     max_prob, max_id = probs.max(dim=-1)
#     if max_prob.item() < 1.0:
#         uncertain.append((step_idx, max_prob.item()))

# print(f"Found {len(uncertain)} uncertain steps (<1.0 max prob).")

# for step_idx, max_prob in uncertain:
#     abs_idx = prompt_len + step_idx  # where that generated token sits in sequences
#     start = max(0, abs_idx - WINDOW)
#     end   = min(total_len, abs_idx + WINDOW + 1)

#     print("\n" + "="*80)
#     print(f"Step {step_idx} (abs token idx {abs_idx})  max_prob={max_prob:.6f}")
#     print(f"Token at abs_idx: id={seq[abs_idx].item()} "
#           f"-> '{tokenizer.decode([seq[abs_idx].item()], skip_special_tokens=True)}'")

#     # Top-K candidates at this step
#     probs = F.softmax(scores[step_idx][0], dim=-1)
#     top_probs, top_ids = probs.topk(TOPK)
#     print("\nTop candidates:")
#     for pid, p in zip(top_ids.tolist(), top_probs.tolist()):
#         print(f"  id={pid:<8} prob={p:.6f}  text='{tokenizer.decode([pid], skip_special_tokens=True)}'")

#     # Show window of tokens around it (IDs + per-token decode)
#     window_ids = seq[start:end].tolist()
#     print("\nWindow token IDs:", window_ids)
#     print("\nPer-token decode:")
#     for i, tid in enumerate(window_ids, start):
#         piece = tokenizer.decode([tid], skip_special_tokens=True)
#         marker = "<-- here" if i == abs_idx else ""
#         print(f"{i:6d}  id={tid:<8} text='{piece}' {marker}")

#     print("\nDecoded window:")
#     print(tokenizer.decode(window_ids, skip_special_tokens=True))

# import pickle
# import torch

# # Load the pickle file
# with open("outputs_debug.pkl", "rb") as f:
#     outputs = pickle.load(f)

# # Check the available attributes
# print("Available attributes in outputs:", outputs.keys() if hasattr(outputs, "keys") else dir(outputs))

# print(f"Outputs length is {outputs.sequences.shape[1]}")

# # Let's look at attentions
# attentions = outputs.attentions  # This should be a tuple
# print(f"Number of steps (len(outputs.attentions)): {len(attentions)}")

# # Example: Inspect the shape of attentions for the first generated step
# first_step = attentions[1]  # tuple of layers for step 0
# print(f"Type of first_step: {type(first_step)}, length (layers): {len(first_step)}")

# # Check last layer's attention for the first step
# last_layer_attn = first_step[-1]
# print(f"Shape of last_layer_attn for step 0: {last_layer_attn.shape}")
# # Should be [batch_size, num_heads, 1, seq_len]

# # Verify mean over heads and selection
# last_layer_attn_mean = last_layer_attn.mean(dim=1)  # -> [batch_size, 1, seq_len]
# print(f"Shape after mean(dim=1): {last_layer_attn_mean.shape}")
# squeezed = last_layer_attn_mean[:, 0]  # -> [batch_size, seq_len]
# print(f"Shape after selecting [:,0]: {squeezed.shape}")
# print(squeezed)

import pickle

# Load the outputs_debug.pkl file
with open("outputs_debug.pkl", "rb") as f:
    outputs = pickle.load(f)

# Inspect sequences shape
print(f"Shape of outputs.sequences: {outputs.sequences.shape}")

# Inspect scores
if hasattr(outputs, "scores"):
    print(f"Number of score tensors: {len(outputs.scores)}")
    for i, score in enumerate(outputs.scores[:5]):  # show first 5 for brevity
        print(f"Shape of scores[{i}]: {score.shape}")
else:
    print("No 'scores' attribute found in outputs.")

# Optionally inspect attentions
if hasattr(outputs, "attentions"):
    print(f"Number of attentions: {len(outputs.attentions)}")
    print(f"Example shape of attentions[0][-1]: {outputs.attentions[0][-1].shape}")
    print(f"Example shape of attentions[1][-1]: {outputs.attentions[1][-1].shape}")
    print(f"Example shape of attentions[2][-1]: {outputs.attentions[2][-1].shape}")