import pickle
import json

import torch
from src.guess.guess import Guess
from src.helpers.launch_spec import LaunchSpec

with open("sketch_results.pkl", "rb") as f:
    sketch_results = pickle.load(f)

with open("predictions.pkl", "rb") as f:
    predictions = pickle.load(f)


launch_spec = LaunchSpec.from_yaml("configs/launch_spec_qwen.yaml")
guess = Guess(launch_spec)


def clean_pred(pred):
    return pred.split("```armasm\n")[-1].split("```")[0]


json_repr = {
    "files": [],
    "pred": [],
    "ed": [],
}

for instance_id, (pred, sketch, stats) in sketch_results.items():
    if pred is None:
        continue

    print(instance_id, stats)

    pred_dec = guess.model.tokenizer.decode(pred.pred)
    json_repr['files'].append(instance_id)
    json_repr['pred'].append(clean_pred(pred_dec))
    json_repr['ed'].append(0)


with open("sketch_results.json", "w") as f:
    json.dump(json_repr, f)


json_repr = {
    "files": [],
    "pred": [],
    "ed": [],
}

for instance_id, pred in predictions.items():
    if pred is None:
        continue

    pred_dec = guess.model.tokenizer.decode(pred.pred)
    json_repr['files'].append(instance_id)
    json_repr['pred'].append(clean_pred(pred_dec))
    json_repr['ed'].append(0)


with open("predictions_results.json", "w") as f:
    json.dump(json_repr, f)
