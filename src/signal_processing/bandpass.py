"""Band-pass filtering utilities for ECG signals."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


DEFAULT_LOW_CUTOFF_HZ = 0.5
DEFAULT_HIGH_CUTOFF_HZ = 40.0
DEFAULT_FILTER_ORDER = 4


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    low_cutoff_hz: float = DEFAULT_LOW_CUTOFF_HZ,
    high_cutoff_hz: float = DEFAULT_HIGH_CUTOFF_HZ,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter to a 1D ECG signal.

    Parameters
    ----------
    signal:
        One-dimensional continuous ECG signal.

    sampling_rate:
        Sampling frequency in Hz.

    low_cutoff_hz:
        Lower band-pass cutoff frequency.

    high_cutoff_hz:
        Upper band-pass cutoff frequency.

    order:
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered ECG signal with the same shape as the input.
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

    if not np.isfinite(signal).all():
        raise ValueError(
            "ECG signal contains NaN or infinite values."
        )

    if sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be greater than zero."
        )

    nyquist = sampling_rate / 2.0

    if not (
        0 < low_cutoff_hz
        < high_cutoff_hz
        < nyquist
    ):
        raise ValueError(
            "Cutoff frequencies must satisfy "
            "0 < low < high < Nyquist frequency."
        )

    if order <= 0:
        raise ValueError(
            "Filter order must be greater than zero."
        )

    sos = butter(
        N=order,
        Wn=(
            low_cutoff_hz,
            high_cutoff_hz,
        ),
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )

    filtered_signal = sosfiltfilt(
        sos,
        signal,
    )

    return filtered_signal