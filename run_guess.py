#!/usr/bin/env python3

import os
import pickle
import torch
from src.guess.guess import Guess
from src.helpers.launch_spec import LaunchSpec
#from src.sketch.sketch import Sketch

# IF you only want to run guess, just comment out the lines below pertaining to sketch. Same for running just sketch, but that assumes you already have the pkl output

# Start by creating an environment or activating an existing one. Conda init then conda activate "name" although you may need to use bash for this to work.
# After that do pip install -r requirements.txt inside the environment if you haven't already. This should have everything you need.
# NOTE: if you add anything to the code that requires pip install, add it here. Otherwise it will make duplicating this work hard in another environment

# Here is a very, very high level overview of guess:

# 1. Load model/config from `launch_spec_qwen.yaml` --- see launch_spec.py for notes on how this works
# 2. Initialize the `Guess` module --- be careful since it's autodeciding GPU's. 
# 3. Run `guess.guess()` to generate predictions. self.model.predict(instance, self.launch_spec.model_config) is called on each data instance, but details are in QwenModel.py
# 4. Save those predictions to `predictions.pkl` -- makes it easier to run sketch only without having to re-run guess

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  ##more precision that float 16 but faster than float 32. Good for inference, but maybe not training 
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  ##allocate memory dynamically, NOT upfront

    launch_spec = LaunchSpec.from_yaml("configs/launch_spec_qwen.yaml")

    guess = Guess(launch_spec)
    predictions = guess.guess()
    print("Inference complete.")

    with open("predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    with open("predictions.pkl", "rb") as f:
        predictions = pickle.load(f)

    print("Loaded predictions from file.")

    failed_instances = []
    for instance_id in predictions.keys():
        if predictions[instance_id] is None:
            failed_instances.append(instance_id)

    # print("Percentage of failed predictions: {}".format(
    #     len(failed_instances) / len(predictions)))

    # sketch = Sketch(launch_spec=launch_spec, model=guess.model)
    # sketch_results = sketch.sketch(predictions)

    # with open("sketch_results.pkl", "wb") as f:
    #     pickle.dump(sketch_results, f)
