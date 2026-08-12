"""Reproducible continuous ECG corruption using NSTDB noise."""

import hashlib

import numpy as np

from src.noise.mixing import (
    add_noise_at_snr,
    calculate_snr_db,
)
from src.noise.nstdb import VALID_NOISE_TYPES


DEFAULT_CORRUPTION_SEED = 20260812


def make_deterministic_seed(
    record_id: str,
    noise_type: str,
    base_seed: int = DEFAULT_CORRUPTION_SEED,
) -> int:
    """
    Create a stable random seed for one ECG record and noise type.

    The result is stable across Python sessions and machines.
    """

    key = f"{base_seed}:{record_id}:{noise_type}"

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def select_noise_variant(
    noise_record: np.ndarray,
    target_length: int,
    record_id: str,
    noise_type: str,
    base_seed: int = DEFAULT_CORRUPTION_SEED,
) -> tuple[np.ndarray, dict]:
    """
    Select a reproducible noise channel and circular start offset.

    The temporal structure of the NSTDB noise is preserved.
    """

    noise_record = np.asarray(
        noise_record,
        dtype=np.float64,
    )

    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(
            f"Unknown noise type: {noise_type}"
        )

    if noise_record.ndim != 2:
        raise ValueError(
            "Expected NSTDB noise with shape "
            "(samples, channels)."
        )

    if target_length <= 0:
        raise ValueError(
            "target_length must be greater than zero."
        )

    n_samples, n_channels = noise_record.shape

    if n_samples == 0 or n_channels == 0:
        raise ValueError(
            "Noise recording must not be empty."
        )

    seed = make_deterministic_seed(
        record_id=record_id,
        noise_type=noise_type,
        base_seed=base_seed,
    )

    rng = np.random.default_rng(seed)

    channel_index = int(
        rng.integers(0, n_channels)
    )

    start_offset = int(
        rng.integers(0, n_samples)
    )

    indices = (
        np.arange(target_length)
        + start_offset
    ) % n_samples

    noise_segment = noise_record[
        indices,
        channel_index,
    ]

    metadata = {
        "record_id": record_id,
        "noise_type": noise_type,
        "noise_channel": channel_index,
        "start_offset": start_offset,
        "seed": seed,
    }

    return noise_segment, metadata


def corrupt_ecg(
    clean_signal: np.ndarray,
    noise_record: np.ndarray,
    record_id: str,
    noise_type: str,
    target_snr_db: float,
    base_seed: int = DEFAULT_CORRUPTION_SEED,
) -> tuple[np.ndarray, dict]:
    """
    Corrupt a continuous ECG signal with reproducible NSTDB noise.
    """

    clean_signal = np.asarray(
        clean_signal,
        dtype=np.float64,
    )

    if clean_signal.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional clean ECG signal."
        )

    noise_segment, metadata = select_noise_variant(
        noise_record=noise_record,
        target_length=len(clean_signal),
        record_id=record_id,
        noise_type=noise_type,
        base_seed=base_seed,
    )

    noisy_signal, injected_noise = add_noise_at_snr(
        clean_signal=clean_signal,
        noise_signal=noise_segment,
        target_snr_db=target_snr_db,
    )

    achieved_snr_db = calculate_snr_db(
        clean_signal=clean_signal,
        noise_component=injected_noise,
    )

    metadata = {
        **metadata,
        "target_snr_db": float(target_snr_db),
        "achieved_snr_db": achieved_snr_db,
    }

    return noisy_signal, metadata