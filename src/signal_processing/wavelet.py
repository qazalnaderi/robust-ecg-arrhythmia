"""Wavelet denoising utilities for ECG signals."""

import numpy as np
import pywt


DEFAULT_WAVELET = "db4"
DEFAULT_LEVEL = 6
DEFAULT_THRESHOLD_MODE = "soft"

MAD_NORMALIZATION_CONSTANT = 0.6744897501960817


def estimate_noise_sigma(
    detail_coefficients: np.ndarray,
) -> float:
    """
    Estimate noise standard deviation using the median absolute deviation.

    The finest-scale detail coefficients are used because they contain
    much of the high-frequency content of the signal.
    """

    detail_coefficients = np.asarray(
        detail_coefficients,
        dtype=np.float64,
    )

    median = np.median(detail_coefficients)

    mad = np.median(
        np.abs(
            detail_coefficients - median
        )
    )

    return float(
        mad / MAD_NORMALIZATION_CONSTANT
    )


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    threshold_mode: str = DEFAULT_THRESHOLD_MODE,
) -> np.ndarray:
    """
    Denoise a one-dimensional ECG signal using wavelet thresholding.

    Parameters
    ----------
    signal:
        One-dimensional continuous ECG signal.

    wavelet:
        Wavelet family used for decomposition.

    level:
        Number of wavelet decomposition levels.

    threshold_mode:
        Thresholding strategy passed to PyWavelets.

    Returns
    -------
    np.ndarray
        Denoised ECG signal with the same length as the input.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    if signal.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional ECG signal, "
            f"got shape {signal.shape}."
        )

    if signal.size == 0:
        raise ValueError(
            "Signal must not be empty."
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            "ECG signal contains NaN or infinite values."
        )

    if level <= 0:
        raise ValueError(
            "Wavelet decomposition level must be greater than zero."
        )

    try:
        wavelet_object = pywt.Wavelet(
            wavelet
        )
    except ValueError as exc:
        raise ValueError(
            f"Unknown wavelet: {wavelet!r}."
        ) from exc

    max_level = pywt.dwt_max_level(
        data_len=signal.size,
        filter_len=wavelet_object.dec_len,
    )

    if level > max_level:
        raise ValueError(
            f"Requested wavelet level {level} exceeds "
            f"maximum useful level {max_level} "
            f"for signal length {signal.size}."
        )

    coefficients = pywt.wavedec(
        data=signal,
        wavelet=wavelet_object,
        mode="symmetric",
        level=level,
    )

    approximation = coefficients[0]
    detail_coefficients = coefficients[1:]

    finest_detail = detail_coefficients[-1]

    noise_sigma = estimate_noise_sigma(
        finest_detail
    )

    threshold = (
        noise_sigma
        * np.sqrt(
            2.0 * np.log(signal.size)
        )
    )

    thresholded_details = [
        pywt.threshold(
            detail,
            value=threshold,
            mode=threshold_mode,
        )
        for detail in detail_coefficients
    ]

    reconstructed = pywt.waverec(
        [
            approximation,
            *thresholded_details,
        ],
        wavelet=wavelet_object,
        mode="symmetric",
    )

    return reconstructed[
        : signal.size
    ]