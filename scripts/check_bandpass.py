"""Smoke-check ECG band-pass filtering on a real MIT-BIH record."""

from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import welch

from src.data.segmentation import (
    DEFAULT_LEAD,
    get_lead_signal,
)
from src.signal_processing.bandpass import (
    DEFAULT_HIGH_CUTOFF_HZ,
    DEFAULT_LOW_CUTOFF_HZ,
    bandpass_filter,
)


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "100"


def band_power(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Estimate spectral power inside one frequency range."""

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

    filtered = bandpass_filter(
        signal=signal,
        sampling_rate=sampling_rate,
    )

    frequencies_before, psd_before = welch(
        signal,
        fs=sampling_rate,
        nperseg=2048,
    )

    frequencies_after, psd_after = welch(
        filtered,
        fs=sampling_rate,
        nperseg=2048,
    )

    nyquist = sampling_rate / 2.0

    low_power_before = band_power(
        frequencies_before,
        psd_before,
        0.0,
        DEFAULT_LOW_CUTOFF_HZ,
    )

    low_power_after = band_power(
        frequencies_after,
        psd_after,
        0.0,
        DEFAULT_LOW_CUTOFF_HZ,
    )

    passband_power_before = band_power(
        frequencies_before,
        psd_before,
        DEFAULT_LOW_CUTOFF_HZ,
        DEFAULT_HIGH_CUTOFF_HZ,
    )

    passband_power_after = band_power(
        frequencies_after,
        psd_after,
        DEFAULT_LOW_CUTOFF_HZ,
        DEFAULT_HIGH_CUTOFF_HZ,
    )

    high_power_before = band_power(
        frequencies_before,
        psd_before,
        DEFAULT_HIGH_CUTOFF_HZ,
        nyquist,
    )

    high_power_after = band_power(
        frequencies_after,
        psd_after,
        DEFAULT_HIGH_CUTOFF_HZ,
        nyquist,
    )

    difference_rms = np.sqrt(
        np.mean((signal - filtered) ** 2)
    )

    print("=" * 70)
    print("MIT-BIH ECG BAND-PASS FILTER CHECK")
    print("=" * 70)

    print(f"Record: {TEST_RECORD}")
    print(f"Sampling frequency: {sampling_rate} Hz")
    print(f"Available leads: {record.sig_name}")
    print(f"Selected lead: {DEFAULT_LEAD}")

    print("-" * 70)

    print(f"Original shape: {signal.shape}")
    print(f"Filtered shape: {filtered.shape}")
    print(f"All output finite: {np.all(np.isfinite(filtered))}")

    print("-" * 70)

    print("Original signal:")
    print(f"  Mean: {signal.mean():.6f}")
    print(f"  Std:  {signal.std():.6f}")
    print(f"  Min:  {signal.min():.6f}")
    print(f"  Max:  {signal.max():.6f}")

    print()

    print("Filtered signal:")
    print(f"  Mean: {filtered.mean():.6f}")
    print(f"  Std:  {filtered.std():.6f}")
    print(f"  Min:  {filtered.min():.6f}")
    print(f"  Max:  {filtered.max():.6f}")

    print("-" * 70)

    print(f"RMS change introduced by filter: {difference_rms:.6f}")

    print("-" * 70)

    print("Spectral power comparison:")

    print(
        f"  Below {DEFAULT_LOW_CUTOFF_HZ} Hz:"
        f" {low_power_before:.8f}"
        f" -> {low_power_after:.8f}"
    )

    print(
        f"  {DEFAULT_LOW_CUTOFF_HZ}-{DEFAULT_HIGH_CUTOFF_HZ} Hz:"
        f" {passband_power_before:.8f}"
        f" -> {passband_power_after:.8f}"
    )

    print(
        f"  Above {DEFAULT_HIGH_CUTOFF_HZ} Hz:"
        f" {high_power_before:.8f}"
        f" -> {high_power_after:.8f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()