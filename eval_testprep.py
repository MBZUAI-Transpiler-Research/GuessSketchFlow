import json
import os

with open("humaneval_onlyguess.json") as f:
    predictions = json.load(f)

preds = predictions["pred"]
files = predictions["files"]

for pred, path in zip(preds, files):
    problem_dir = os.path.dirname(path)
    asm_path = os.path.join(problem_dir, "code.s")

    try:

        with open(asm_path, "w") as f:
            f.write(pred)
    except FileNotFoundError as e:
        print(f"Warning: Skipping: {e}")