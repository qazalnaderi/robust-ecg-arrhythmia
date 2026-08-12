import numpy as np


def signal_power(signal: np.ndarray) -> float:
    """
    Estimate signal power using mean squared amplitude
    after removing the constant DC component.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(
            f"Expected a 1D signal, got shape {signal.shape}."
        )

    centered = signal - np.mean(signal)

    return float(np.mean(centered ** 2))


def scale_noise_to_snr(
    clean_signal: np.ndarray,
    noise_signal: np.ndarray,
    target_snr_db: float,
) -> np.ndarray:
    """
    Scale a noise signal to achieve a requested SNR.

    The clean signal and noise must have the same shape.
    """

    clean_signal = np.asarray(
        clean_signal,
        dtype=np.float64,
    )

    noise_signal = np.asarray(
        noise_signal,
        dtype=np.float64,
    )

    if clean_signal.shape != noise_signal.shape:
        raise ValueError(
            "Clean signal and noise signal must have "
            f"the same shape. Got {clean_signal.shape} "
            f"and {noise_signal.shape}."
        )

    clean_power = signal_power(clean_signal)

    # Remove only the constant offset from the noise.
    centered_noise = noise_signal - np.mean(noise_signal)

    noise_power = signal_power(centered_noise)

    if clean_power <= 0:
        raise ValueError(
            "Clean signal power must be greater than zero."
        )

    if noise_power <= 0:
        raise ValueError(
            "Noise signal power must be greater than zero."
        )

    target_linear_ratio = 10 ** (
        target_snr_db / 10.0
    )

    scale = np.sqrt(
        clean_power
        / (noise_power * target_linear_ratio)
    )

    return centered_noise * scale


def add_noise_at_snr(
    clean_signal: np.ndarray,
    noise_signal: np.ndarray,
    target_snr_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add scaled noise to a clean ECG signal.

    Returns
    -------
    noisy_signal:
        Clean ECG plus scaled noise.

    scaled_noise:
        The actual noise component that was injected.
    """

    clean_signal = np.asarray(
        clean_signal,
        dtype=np.float64,
    )

    scaled_noise = scale_noise_to_snr(
        clean_signal=clean_signal,
        noise_signal=noise_signal,
        target_snr_db=target_snr_db,
    )

    noisy_signal = clean_signal + scaled_noise

    return noisy_signal, scaled_noise


def calculate_snr_db(
    clean_signal: np.ndarray,
    noise_component: np.ndarray,
) -> float:
    """
    Calculate the achieved SNR in decibels.
    """

    clean_power = signal_power(clean_signal)
    noise_power = signal_power(noise_component)

    if noise_power <= 0:
        raise ValueError(
            "Noise power must be greater than zero."
        )

    return float(
        10.0 * np.log10(
            clean_power / noise_power
        )
    )