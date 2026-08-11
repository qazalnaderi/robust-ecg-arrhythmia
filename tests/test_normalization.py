import numpy as np
import pytest

from src.data.normalization import (
    normalize_heartbeats,
    zscore_normalize_heartbeat,
)


def test_single_heartbeat_has_zero_mean():
    heartbeat = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    normalized = zscore_normalize_heartbeat(
        heartbeat
    )

    assert np.isclose(
        normalized.mean(),
        0.0,
        atol=1e-6,
    )


def test_single_heartbeat_has_unit_standard_deviation():
    heartbeat = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    normalized = zscore_normalize_heartbeat(
        heartbeat
    )

    assert np.isclose(
        normalized.std(),
        1.0,
        atol=1e-6,
    )


def test_normalization_preserves_shape():
    heartbeat = np.arange(
        256,
        dtype=np.float32,
    )

    normalized = zscore_normalize_heartbeat(
        heartbeat
    )

    assert normalized.shape == heartbeat.shape


def test_constant_heartbeat_becomes_zero():
    heartbeat = np.ones(
        256,
        dtype=np.float32,
    )

    normalized = zscore_normalize_heartbeat(
        heartbeat
    )

    assert np.all(normalized == 0.0)


def test_batch_normalization_preserves_shape():
    heartbeats = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )

    normalized = normalize_heartbeats(
        heartbeats
    )

    assert normalized.shape == heartbeats.shape


def test_each_heartbeat_is_normalized_independently():
    heartbeats = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )

    normalized = normalize_heartbeats(
        heartbeats
    )

    assert np.allclose(
        normalized.mean(axis=1),
        0.0,
        atol=1e-6,
    )

    assert np.allclose(
        normalized.std(axis=1),
        1.0,
        atol=1e-6,
    )


def test_invalid_single_heartbeat_shape_raises_error():
    heartbeat = np.zeros(
        (2, 256),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        zscore_normalize_heartbeat(
            heartbeat
        )


def test_invalid_batch_shape_raises_error():
    heartbeats = np.zeros(
        256,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        normalize_heartbeats(
            heartbeats
        )