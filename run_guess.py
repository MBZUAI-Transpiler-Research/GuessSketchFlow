#!/usr/bin/env python3
"""Entry point for running the Guess module."""

import pickle
from src.guess.guess import ConfigType, Guess, Config
from src.domain.datasets.UnixCommandDataset import UnixCommandDataset  # Register dataset
from src.domain.datasets.ProjectEulerDataset import ProjectEulerDataset
from src.domain.datasets.BringUpDataset import BringUpDataset
from src.sketch.sketch import Sketch  # Register dataset

if __name__ == "__main__":
    # Using QWEN model for ARM to x86 translation
    config = Config(ConfigType.QWEN_X862ARM64.get_path())
    guess = Guess(config=config)
    predictions = guess.guess()
    print("Inference complete.")

    with open("predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    metrics = guess.evaluate(predictions)
    print("Evaluation complete.")
    print(metrics)
    sketch = Sketch(config=config, model=guess.model)
    sketch.sketch(predictions)
