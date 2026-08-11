"""Normalization utilities for segmented ECG heartbeats."""

import numpy as np


DEFAULT_EPSILON = 1e-8


def zscore_normalize_heartbeat(
    heartbeat: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """
    Apply per-heartbeat z-score normalization.

    Each heartbeat is centered using its own mean and scaled using
    its own standard deviation.

    Parameters
    ----------
    heartbeat:
        One-dimensional ECG heartbeat window.

    epsilon:
        Small value used to safely handle nearly constant signals.

    Returns
    -------
    np.ndarray
        Normalized heartbeat with approximately zero mean and unit
        standard deviation.
    """

    heartbeat = np.asarray(
        heartbeat,
        dtype=np.float32,
    )

    if heartbeat.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional heartbeat, "
            f"got shape {heartbeat.shape}."
        )

    mean = heartbeat.mean()
    std = heartbeat.std()

    if std < epsilon:
        return np.zeros_like(
            heartbeat,
            dtype=np.float32,
        )

    return (
        (heartbeat - mean) / std
    ).astype(np.float32)


def normalize_heartbeats(
    heartbeats: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """
    Apply per-heartbeat z-score normalization to a batch.

    Parameters
    ----------
    heartbeats:
        ECG array with shape (number_of_heartbeats, heartbeat_length).

    Returns
    -------
    np.ndarray
        Normalized ECG array with the same shape as the input.
    """

    heartbeats = np.asarray(
        heartbeats,
        dtype=np.float32,
    )

    if heartbeats.ndim != 2:
        raise ValueError(
            "Expected a 2D heartbeat array, "
            f"got shape {heartbeats.shape}."
        )

    normalized = [
        zscore_normalize_heartbeat(
            heartbeat,
            epsilon=epsilon,
        )
        for heartbeat in heartbeats
    ]

    return np.stack(normalized)