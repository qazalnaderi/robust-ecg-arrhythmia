"""Utilities for scaling and injecting noise at a controlled SNR."""

import numpy as np


def _as_valid_1d_signal(
    signal: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Convert an input signal to a finite, non-empty
    one-dimensional float64 NumPy array.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    if signal.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional. "
            f"Got shape {signal.shape}."
        )

    if signal.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return signal


def signal_power(
    signal: np.ndarray,
) -> float:
    """
    Estimate AC signal power.

    The constant DC component is removed before
    calculating the mean squared amplitude.
    """

    signal = _as_valid_1d_signal(
        signal,
        name="Signal",
    )

    centered = (
        signal - np.mean(signal)
    )

    power = float(
        np.mean(centered ** 2)
    )

    if not np.isfinite(power):
        raise ValueError(
            "Signal power is not finite."
        )

    return power


def scale_noise_to_snr(
    clean_signal: np.ndarray,
    noise_signal: np.ndarray,
    target_snr_db: float,
) -> np.ndarray:
    """
    Scale a noise signal to achieve a requested SNR.

    SNR is defined using AC power after removal
    of the constant DC component:

        SNR_dB = 10 * log10(P_signal / P_noise)

    The clean signal and noise signal must be
    finite, non-empty, one-dimensional arrays
    with identical shapes.

    Parameters
    ----------
    clean_signal:
        Clean ECG signal.

    noise_signal:
        Noise signal to be scaled.

    target_snr_db:
        Requested signal-to-noise ratio in decibels.

    Returns
    -------
    np.ndarray
        Zero-mean scaled noise component.
    """

    clean_signal = _as_valid_1d_signal(
        clean_signal,
        name="Clean signal",
    )

    noise_signal = _as_valid_1d_signal(
        noise_signal,
        name="Noise signal",
    )

    if clean_signal.shape != noise_signal.shape:
        raise ValueError(
            "Clean signal and noise signal must have "
            f"the same shape. Got {clean_signal.shape} "
            f"and {noise_signal.shape}."
        )

    try:
        target_snr_db = float(
            target_snr_db
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "target_snr_db must be a real number."
        ) from exc

    if not np.isfinite(target_snr_db):
        raise ValueError(
            "target_snr_db must be finite."
        )

    # ---------------------------------------------------------
    # Clean ECG power
    # ---------------------------------------------------------

    clean_power = signal_power(
        clean_signal
    )

    if clean_power <= 0:
        raise ValueError(
            "Clean signal power must be greater than zero."
        )

    # ---------------------------------------------------------
    # Remove only the constant offset from NSTDB noise
    # ---------------------------------------------------------

    centered_noise = (
        noise_signal
        - np.mean(noise_signal)
    )

    noise_power = signal_power(
        centered_noise
    )

    if noise_power <= 0:
        raise ValueError(
            "Noise signal power must be greater than zero."
        )

    # ---------------------------------------------------------
    # Calculate amplitude scaling factor
    #
    # target_snr_db =
    #
    #   10 * log10(
    #       P_signal /
    #       (scale^2 * P_noise)
    #   )
    #
    # Therefore:
    #
    # scale =
    #
    #   sqrt(P_signal / P_noise)
    #   * 10^(-SNR_dB / 20)
    # ---------------------------------------------------------

    with np.errstate(
        over="ignore",
        under="ignore",
        divide="ignore",
        invalid="ignore",
    ):

        scale = (
            np.sqrt(
                clean_power / noise_power
            )
            * np.power(
                10.0,
                -target_snr_db / 20.0,
            )
        )

    if (
        not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            "Requested target SNR produced an invalid "
            "noise scaling factor."
        )

    scaled_noise = (
        centered_noise * scale
    )

    if not np.isfinite(
        scaled_noise
    ).all():
        raise ValueError(
            "Scaled noise contains NaN or infinite values."
        )

    return scaled_noise


def add_noise_at_snr(
    clean_signal: np.ndarray,
    noise_signal: np.ndarray,
    target_snr_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add scaled noise to a clean ECG signal.

    Parameters
    ----------
    clean_signal:
        Clean ECG signal.

    noise_signal:
        Noise signal to inject.

    target_snr_db:
        Requested SNR in decibels.

    Returns
    -------
    noisy_signal:
        Clean ECG plus scaled noise.

    scaled_noise:
        Actual zero-mean noise component injected
        into the ECG.
    """

    clean_signal = _as_valid_1d_signal(
        clean_signal,
        name="Clean signal",
    )

    scaled_noise = scale_noise_to_snr(
        clean_signal=clean_signal,
        noise_signal=noise_signal,
        target_snr_db=target_snr_db,
    )

    noisy_signal = (
        clean_signal + scaled_noise
    )

    if not np.isfinite(
        noisy_signal
    ).all():
        raise ValueError(
            "Noisy signal contains NaN or infinite values."
        )

    return (
        noisy_signal,
        scaled_noise,
    )


def calculate_snr_db(
    clean_signal: np.ndarray,
    noise_component: np.ndarray,
) -> float:
    """
    Calculate achieved SNR in decibels.

    Uses the same AC-power definition as
    ``scale_noise_to_snr``.
    """

    clean_signal = _as_valid_1d_signal(
        clean_signal,
        name="Clean signal",
    )

    noise_component = _as_valid_1d_signal(
        noise_component,
        name="Noise component",
    )

    if (
        clean_signal.shape
        != noise_component.shape
    ):
        raise ValueError(
            "Clean signal and noise component must have "
            f"the same shape. Got {clean_signal.shape} "
            f"and {noise_component.shape}."
        )

    clean_power = signal_power(
        clean_signal
    )

    noise_power = signal_power(
        noise_component
    )

    if clean_power <= 0:
        raise ValueError(
            "Clean signal power must be greater than zero."
        )

    if noise_power <= 0:
        raise ValueError(
            "Noise power must be greater than zero."
        )

    snr_db = float(
        10.0
        * np.log10(
            clean_power / noise_power
        )
    )

    if not np.isfinite(snr_db):
        raise ValueError(
            "Calculated SNR is not finite."
        )

    return snr_db