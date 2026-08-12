import numpy as np
import pytest

from src.signal_processing.wavelet import (
    estimate_noise_sigma,
    wavelet_denoise,
)


def test_wavelet_preserves_signal_shape():
    signal = np.sin(
        np.linspace(
            0,
            20 * np.pi,
            4096,
        )
    )

    denoised = wavelet_denoise(
        signal
    )

    assert denoised.shape == signal.shape


def test_wavelet_output_is_finite():
    signal = np.sin(
        np.linspace(
            0,
            20 * np.pi,
            4096,
        )
    )

    denoised = wavelet_denoise(
        signal
    )

    assert np.all(
        np.isfinite(denoised)
    )


def test_wavelet_reduces_high_frequency_noise():
    sampling_rate = 360.0
    duration_seconds = 12.0

    time = np.arange(
        0,
        duration_seconds,
        1.0 / sampling_rate,
    )

    clean_signal = np.sin(
        2 * np.pi * 5.0 * time
    )

    high_frequency_noise = (
        0.4
        * np.sin(
            2 * np.pi * 80.0 * time
        )
    )

    noisy_signal = (
        clean_signal
        + high_frequency_noise
    )

    denoised = wavelet_denoise(
        noisy_signal
    )

    error_before = np.mean(
        (noisy_signal - clean_signal) ** 2
    )

    error_after = np.mean(
        (denoised - clean_signal) ** 2
    )

    assert error_after < error_before


def test_noise_sigma_is_non_negative():
    coefficients = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    sigma = estimate_noise_sigma(
        coefficients
    )

    assert sigma >= 0.0


def test_invalid_signal_shape_raises_error():
    signal = np.zeros(
        (2, 4096)
    )

    with pytest.raises(ValueError):
        wavelet_denoise(
            signal
        )


def test_empty_signal_raises_error():
    signal = np.array([])

    with pytest.raises(ValueError):
        wavelet_denoise(
            signal
        )


def test_invalid_level_raises_error():
    signal = np.zeros(4096)

    with pytest.raises(ValueError):
        wavelet_denoise(
            signal,
            level=0,
        )