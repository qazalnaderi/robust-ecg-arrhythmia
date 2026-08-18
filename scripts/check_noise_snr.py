"""Verify NSTDB noise injection at controlled SNR levels."""

from pathlib import Path

import numpy as np
import wfdb

from src.data.segmentation import get_lead_signal
from src.noise.corruption import corrupt_ecg
from src.noise.nstdb import (
    VALID_NOISE_TYPES,
    load_noise_record,
)


DATA_DIR = Path("data/raw/mitdb")

TEST_RECORD = "100"
LEAD_NAME = "MLII"

TARGET_SNRS_DB = (
    18.0,
    12.0,
    6.0,
    0.0,
    -6.0,
)

SNR_TOLERANCE_DB = 1e-6


def independent_signal_power(
    signal: np.ndarray,
) -> float:
    """
    Independently calculate AC signal power.

    This intentionally does not call src.noise.mixing.signal_power
    so the verification is not circular.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    centered = (
        signal - np.mean(signal)
    )

    return float(
        np.mean(centered ** 2)
    )


def independent_snr_db(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
) -> float:
    """
    Independently calculate SNR from clean and noisy ECG.
    """

    injected_noise = (
        noisy_signal - clean_signal
    )

    clean_power = independent_signal_power(
        clean_signal
    )

    noise_power = independent_signal_power(
        injected_noise
    )

    if clean_power <= 0:
        raise RuntimeError(
            "Clean ECG has zero or negative power."
        )

    if noise_power <= 0:
        raise RuntimeError(
            "Injected noise has zero or negative power."
        )

    return float(
        10.0
        * np.log10(
            clean_power / noise_power
        )
    )


def main() -> None:

    print("=" * 88)
    print("NSTDB CONTROLLED-SNR VERIFICATION")
    print("=" * 88)

    # ---------------------------------------------------------
    # 1. Load sanity-only MIT-BIH record
    # ---------------------------------------------------------

    record_path = (
        DATA_DIR / TEST_RECORD
    )

    record = wfdb.rdrecord(
        str(record_path)
    )

    clean_signal = get_lead_signal(
        record,
        lead_name=LEAD_NAME,
    )

    clean_signal = np.asarray(
        clean_signal,
        dtype=np.float64,
    )

    print(
        f"Record: {TEST_RECORD}"
    )

    print(
        f"Lead: {LEAD_NAME}"
    )

    print(
        f"Sampling rate: {record.fs} Hz"
    )

    print(
        f"Samples: {len(clean_signal)}"
    )

    if not np.isfinite(
        clean_signal
    ).all():
        raise RuntimeError(
            "Clean ECG contains NaN or infinite values."
        )

    max_snr_error = 0.0

    # ---------------------------------------------------------
    # 2. Verify every NSTDB noise type
    # ---------------------------------------------------------

    for noise_type in VALID_NOISE_TYPES:

        print("\n" + "-" * 88)
        print(
            f"Noise type: {noise_type}"
        )
        print("-" * 88)

        noise_record, noise_fs = load_noise_record(
            noise_type
        )

        if float(noise_fs) != float(record.fs):
            raise RuntimeError(
                f"Sampling-rate mismatch for {noise_type}: "
                f"ECG={record.fs}, noise={noise_fs}"
            )

        reference_variant = None

        for target_snr_db in TARGET_SNRS_DB:

            noisy_signal, metadata = corrupt_ecg(
                clean_signal=clean_signal,
                noise_record=noise_record,
                record_id=TEST_RECORD,
                noise_type=noise_type,
                target_snr_db=target_snr_db,
            )

            # ---------------------------------------------
            # Independent SNR measurement
            # ---------------------------------------------

            independent_snr = independent_snr_db(
                clean_signal=clean_signal,
                noisy_signal=noisy_signal,
            )

            internal_snr = float(
                metadata["achieved_snr_db"]
            )

            target_error = abs(
                independent_snr
                - target_snr_db
            )

            internal_error = abs(
                independent_snr
                - internal_snr
            )

            max_snr_error = max(
                max_snr_error,
                target_error,
            )

            # ---------------------------------------------
            # Verify same noise segment across SNR levels
            # ---------------------------------------------

            current_variant = (
                metadata["noise_channel"],
                metadata["start_offset"],
                metadata["seed"],
            )

            if reference_variant is None:
                reference_variant = (
                    current_variant
                )

            elif (
                current_variant
                != reference_variant
            ):
                raise RuntimeError(
                    f"{noise_type} changed noise variant "
                    "between SNR levels."
                )

            # ---------------------------------------------
            # Hard checks
            # ---------------------------------------------

            if not np.isfinite(
                noisy_signal
            ).all():
                raise RuntimeError(
                    f"{noise_type} @ {target_snr_db:g} dB "
                    "contains invalid values."
                )

            if (
                target_error
                > SNR_TOLERANCE_DB
            ):
                raise RuntimeError(
                    f"SNR mismatch for {noise_type}: "
                    f"target={target_snr_db:.6f}, "
                    f"measured={independent_snr:.6f}"
                )

            if (
                internal_error
                > SNR_TOLERANCE_DB
            ):
                raise RuntimeError(
                    "Internal and independent SNR "
                    f"calculations disagree for {noise_type}."
                )

            print(
                f"Target={target_snr_db:>6.1f} dB | "
                f"Internal={internal_snr:>10.6f} dB | "
                f"Independent={independent_snr:>10.6f} dB | "
                f"Error={target_error:.2e} | "
                f"channel={metadata['noise_channel']} | "
                f"offset={metadata['start_offset']}"
            )

        # -----------------------------------------------------
        # 3. Explicit determinism check
        # -----------------------------------------------------

        first_signal, first_metadata = corrupt_ecg(
            clean_signal=clean_signal,
            noise_record=noise_record,
            record_id=TEST_RECORD,
            noise_type=noise_type,
            target_snr_db=6.0,
        )

        second_signal, second_metadata = corrupt_ecg(
            clean_signal=clean_signal,
            noise_record=noise_record,
            record_id=TEST_RECORD,
            noise_type=noise_type,
            target_snr_db=6.0,
        )

        deterministic_signal = np.array_equal(
            first_signal,
            second_signal,
        )

        deterministic_metadata = (
            first_metadata
            == second_metadata
        )

        print(
            f"\nDeterministic signal: "
            f"{deterministic_signal}"
        )

        print(
            f"Deterministic metadata: "
            f"{deterministic_metadata}"
        )

        if not deterministic_signal:
            raise RuntimeError(
                f"{noise_type} corruption is not deterministic."
            )

        if not deterministic_metadata:
            raise RuntimeError(
                f"{noise_type} metadata is not deterministic."
            )

    # ---------------------------------------------------------
    # 4. Final report
    # ---------------------------------------------------------

    print("\n" + "=" * 88)

    print(
        f"Maximum absolute SNR error: "
        f"{max_snr_error:.10f} dB"
    )

    print("\nControlled-SNR verification: PASS")

    print("=" * 88)


if __name__ == "__main__":
    main()