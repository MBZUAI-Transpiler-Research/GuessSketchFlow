import os
from datasets import load_dataset

# Create needed directories
os.makedirs("tmp_parse_inputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load dataset  
#dataset = load_dataset("ahmedheakl/gg-bench-armv8-O0", split="train")
#dataset = load_dataset("ahmedheakl/gg-bench-bringup-O0", split="train")
dataset = load_dataset("adpretko/reducedeval", split="train")

for row in dataset:
    basename = row["file"].replace("/code.c", "")  # e.g., "eval/problem42"
    tmp_prefix = f"tmp_parse_inputs/{basename}"

    # Ensure intermediate dirs (like eval/) exist inside tmp_parse_inputs/
    os.makedirs(os.path.dirname(tmp_prefix), exist_ok=True)

    x86_path = tmp_prefix + ".x86.s"
    arm_path = tmp_prefix + ".arm.s"

    with open(x86_path, "w") as f:
        f.write(row["x86"])
    with open(arm_path, "w") as f:
        f.write(row["arm"])