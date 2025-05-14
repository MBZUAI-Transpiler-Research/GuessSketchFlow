#!/usr/bin/env python3
"""Entry point for running the Guess module."""

import pickle
from src.guess.guess import ConfigType, Guess, Config
from src.domain.datasets.UnixCommandDataset import UnixCommandDataset  # Register dataset
from src.domain.datasets.ProjectEulerDataset import ProjectEulerDataset
from src.domain.datasets.BringUpDataset import BringUpDataset
from src.helpers.launch_spec import LaunchSpec
from src.sketch.sketch import Sketch  # Register dataset
from new_qwen import main as ahmed_model

if __name__ == "__main__":
    launch_spec = LaunchSpec.from_yaml("configs/qwen_x862arm64.yaml")

    guess = Guess(launch_spec)
    predictions = guess.guess()
    print("Inference complete.")

    with open("predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    # with open("predictions.pkl", "rb") as f:
    #     predictions = pickle.load(f)

    print("Loaded predictions from file.")

    sketch = Sketch(config=config, model=guess.model)
    sketch_results = sketch.sketch(predictions)

    with open("sketch_results.pkl", "wb") as f:
        pickle.dump(sketch_results, f)
