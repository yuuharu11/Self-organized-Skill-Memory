"""Runtime setup shared by the frozen-VLM light experiments."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_experiment(seed: int) -> None:
    """Seed all three RNGs and preserve the light runners' CPU thread limit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
