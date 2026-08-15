import numpy as np
import pytest
import torch

from src.data.torch_dataset import (
    CLASS_TO_INDEX,
    ECGHeartbeatDataset,
)


def make_test_data():
    heartbeats = np.random.randn(
        4,
        256,
    ).astype(np.float32)

    labels = np.array(
        ["N", "S", "V", "F"]
    )

    return heartbeats, labels


def test_dataset_length():
    heartbeats, labels = make_test_data()

    dataset = ECGHeartbeatDataset(
        heartbeats,
        labels,
    )

    assert len(dataset) == 4


def test_dataset_input_shape():
    heartbeats, labels = make_test_data()

    dataset = ECGHeartbeatDataset(
        heartbeats,
        labels,
    )

    x, _ = dataset[0]

    assert x.shape == (
        1,
        256,
    )


def test_dataset_target_is_integer_tensor():
    heartbeats, labels = make_test_data()

    dataset = ECGHeartbeatDataset(
        heartbeats,
        labels,
    )

    _, target = dataset[0]

    assert target.dtype == torch.long


def test_class_mapping():
    assert CLASS_TO_INDEX == {
        "N": 0,
        "S": 1,
        "V": 2,
        "F": 3,
    }


def test_unknown_label_raises_error():
    heartbeats = np.zeros(
        (2, 256),
        dtype=np.float32,
    )

    labels = np.array(
        ["N", "Q"]
    )

    with pytest.raises(ValueError):
        ECGHeartbeatDataset(
            heartbeats,
            labels,
        )


def test_mismatched_lengths_raise_error():
    heartbeats = np.zeros(
        (3, 256),
        dtype=np.float32,
    )

    labels = np.array(
        ["N", "V"]
    )

    with pytest.raises(ValueError):
        ECGHeartbeatDataset(
            heartbeats,
            labels,
        )