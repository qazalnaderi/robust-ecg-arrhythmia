"""Reproducibility utilities for model training."""

import random

import numpy as np
import torch


DEFAULT_SEED = 42


def set_global_seed(
    seed: int = DEFAULT_SEED,
) -> None:
    """Set random seeds used by Python, NumPy, and PyTorch."""

    if seed < 0:
        raise ValueError(
            "Seed must be non-negative."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False