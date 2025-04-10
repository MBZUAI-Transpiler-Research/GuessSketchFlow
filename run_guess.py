#!/usr/bin/env python3
"""Entry point for running the Guess module."""

from src.guess.guess import ConfigType, Guess, Config
from src.domain.datasets.UnixCommandDataset import UnixCommandDataset  # Register dataset
from src.domain.datasets.ProjectEulerDataset import ProjectEulerDataset
from src.sketch.sketch import Sketch  # Register dataset

if __name__ == "__main__":
    # Using QWEN model for ARM to x86 translation
    config = Config(ConfigType.BART_RISC2ARM.get_path())
    guess = Guess(config=config)
    predictions = guess.guess()
    print("Inference complete.")
    metrics = guess.evaluate(predictions)
    print("Evaluation complete.")
    print(metrics)
    sketch = Sketch(config=config, model=guess.model)
    sketch.sketch(predictions)
