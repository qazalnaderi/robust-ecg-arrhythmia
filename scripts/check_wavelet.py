"""Smoke-check wavelet denoising on a real MIT-BIH ECG record."""

from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import welch

from src.data.segmentation import (
    DEFAULT_LEAD,
    get_lead_signal,
)
from src.signal_processing.wavelet import (
    DEFAULT_LEVEL,
    DEFAULT_WAVELET,
    wavelet_denoise,
)


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "100"


def band_power(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Estimate spectral power inside a frequency range."""

    mask = (
        (frequencies >= low_hz)
        & (frequencies < high_hz)
    )

    if not np.any(mask):
        return 0.0

    return float(
        np.trapezoid(
            power_spectrum[mask],
            frequencies[mask],
        )
    )


def main() -> None:
    record_path = DATA_DIR / TEST_RECORD

    record = wfdb.rdrecord(
        str(record_path)
    )

    signal = get_lead_signal(
        record,
        lead_name=DEFAULT_LEAD,
    )

    sampling_rate = float(record.fs)

    denoised = wavelet_denoise(
        signal
    )

    difference_rms = np.sqrt(
        np.mean(
            (signal - denoised) ** 2
        )
    )

    correlation = np.corrcoef(
        signal,
        denoised,
    )[0, 1]

    frequencies_before, psd_before = welch(
        signal,
        fs=sampling_rate,
        nperseg=2048,
    )

    frequencies_after, psd_after = welch(
        denoised,
        fs=sampling_rate,
        nperseg=2048,
    )

    high_power_before = band_power(
        frequencies_before,
        psd_before,
        40.0,
        sampling_rate / 2.0,
    )

    high_power_after = band_power(
        frequencies_after,
        psd_after,
        40.0,
        sampling_rate / 2.0,
    )

    print("=" * 70)
    print("MIT-BIH ECG WAVELET DENOISING CHECK")
    print("=" * 70)

    print(f"Record: {TEST_RECORD}")
    print(f"Sampling frequency: {sampling_rate} Hz")
    print(f"Available leads: {record.sig_name}")
    print(f"Selected lead: {DEFAULT_LEAD}")

    print(f"Wavelet: {DEFAULT_WAVELET}")
    print(f"Decomposition level: {DEFAULT_LEVEL}")

    print("-" * 70)

    print(f"Original shape: {signal.shape}")
    print(f"Denoised shape: {denoised.shape}")
    print(
        "All output finite: "
        f"{np.all(np.isfinite(denoised))}"
    )

    print("-" * 70)

    print("Original signal:")
    print(f"  Mean: {signal.mean():.6f}")
    print(f"  Std:  {signal.std():.6f}")
    print(f"  Min:  {signal.min():.6f}")
    print(f"  Max:  {signal.max():.6f}")

    print()

    print("Denoised signal:")
    print(f"  Mean: {denoised.mean():.6f}")
    print(f"  Std:  {denoised.std():.6f}")
    print(f"  Min:  {denoised.min():.6f}")
    print(f"  Max:  {denoised.max():.6f}")

    print("-" * 70)

    print(
        "RMS change introduced by Wavelet: "
        f"{difference_rms:.6f}"
    )

    print(
        "Correlation with original ECG: "
        f"{correlation:.6f}"
    )

    print("-" * 70)

    print("High-frequency power above 40 Hz:")

    print(
        f"  Before: {high_power_before:.8f}"
    )

    print(
        f"  After:  {high_power_after:.8f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()