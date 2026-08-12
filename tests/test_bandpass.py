import numpy as np
import pytest

from src.signal_processing.bandpass import (
    bandpass_filter,
)


def test_bandpass_preserves_signal_shape():
    signal = np.sin(
        np.linspace(
            0,
            20 * np.pi,
            3600,
        )
    )

    filtered = bandpass_filter(
        signal=signal,
        sampling_rate=360.0,
    )

    assert filtered.shape == signal.shape


def test_bandpass_output_is_finite():
    signal = np.sin(
        np.linspace(
            0,
            20 * np.pi,
            3600,
        )
    )

    filtered = bandpass_filter(
        signal=signal,
        sampling_rate=360.0,
    )

    assert np.all(np.isfinite(filtered))


def test_bandpass_reduces_high_frequency_component():
    sampling_rate = 360.0
    duration_seconds = 10.0

    time = np.arange(
        0,
        duration_seconds,
        1.0 / sampling_rate,
    )

    low_frequency = np.sin(
        2 * np.pi * 5.0 * time
    )

    high_frequency = np.sin(
        2 * np.pi * 80.0 * time
    )

    mixed_signal = (
        low_frequency
        + high_frequency
    )

    filtered = bandpass_filter(
        signal=mixed_signal,
        sampling_rate=sampling_rate,
    )

    error_before = np.mean(
        (mixed_signal - low_frequency) ** 2
    )

    error_after = np.mean(
        (filtered - low_frequency) ** 2
    )

    assert error_after < error_before


def test_invalid_signal_shape_raises_error():
    signal = np.zeros(
        (2, 1000)
    )

    with pytest.raises(ValueError):
        bandpass_filter(
            signal=signal,
            sampling_rate=360.0,
        )


def test_invalid_cutoffs_raise_error():
    signal = np.zeros(3600)

    with pytest.raises(ValueError):
        bandpass_filter(
            signal=signal,
            sampling_rate=360.0,
            low_cutoff_hz=50.0,
            high_cutoff_hz=40.0,
        )