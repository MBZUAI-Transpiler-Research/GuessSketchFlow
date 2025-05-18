#!/usr/bin/env python3

import os
import pickle
import torch
from src.guess.guess import Guess
from src.helpers.launch_spec import LaunchSpec
from src.sketch.sketch import Sketch
from new_qwen import main as ahmed_model

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    launch_spec = LaunchSpec.from_yaml("configs/launch_spec_qwen.yaml")

    guess = Guess(launch_spec)
    predictions = guess.guess()
    print("Inference complete.")

    with open("predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    # with open("predictions.pkl", "rb") as f:
    #     predictions = pickle.load(f)

    # print("Loaded predictions from file.")

    failed_instances = []
    for instance_id in predictions.keys():
        if predictions[instance_id] is None:
            failed_instances.append(instance_id)
            
    print("Percentage of failed predictions: {}".format(len(failed_instances) / len(predictions)))

    sketch = Sketch(launch_spec=launch_spec, model=guess.model)
    sketch_results = sketch.sketch(predictions)

    with open("sketch_results.pkl", "wb") as f:
        pickle.dump(sketch_results, f)
