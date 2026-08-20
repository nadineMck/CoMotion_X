"""Utilities for deterministic experiments."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

