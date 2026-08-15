import numpy as np
import pytest
import torch

from src.training.reproducibility import (
    set_global_seed,
)


def test_torch_randomness_is_reproducible():
    set_global_seed(42)

    first = torch.randn(5)

    set_global_seed(42)

    second = torch.randn(5)

    assert torch.equal(
        first,
        second,
    )


def test_numpy_randomness_is_reproducible():
    set_global_seed(42)

    first = np.random.rand(5)

    set_global_seed(42)

    second = np.random.rand(5)

    assert np.array_equal(
        first,
        second,
    )


def test_negative_seed_raises_error():
    with pytest.raises(ValueError):
        set_global_seed(-1)