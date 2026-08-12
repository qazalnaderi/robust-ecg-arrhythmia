import pytest
import torch

from src.models.cnn1d import ECGCNN1D


def test_cnn_output_shape():
    model = ECGCNN1D(
        num_classes=4
    )

    x = torch.randn(
        8,
        1,
        256,
    )

    logits = model(x)

    assert logits.shape == (
        8,
        4,
    )


def test_cnn_output_is_finite():
    model = ECGCNN1D()

    x = torch.randn(
        4,
        1,
        256,
    )

    logits = model(x)

    assert torch.all(
        torch.isfinite(logits)
    )


def test_cnn_accepts_different_batch_sizes():
    model = ECGCNN1D()

    x = torch.randn(
        3,
        1,
        256,
    )

    logits = model(x)

    assert logits.shape[0] == 3


def test_invalid_input_dimension_raises_error():
    model = ECGCNN1D()

    x = torch.randn(
        8,
        256,
    )

    with pytest.raises(ValueError):
        model(x)


def test_invalid_input_channel_count_raises_error():
    model = ECGCNN1D()

    x = torch.randn(
        8,
        2,
        256,
    )

    with pytest.raises(ValueError):
        model(x)


def test_invalid_number_of_classes_raises_error():
    with pytest.raises(ValueError):
        ECGCNN1D(
            num_classes=1
        )


def test_invalid_dropout_raises_error():
    with pytest.raises(ValueError):
        ECGCNN1D(
            dropout=1.5
        )