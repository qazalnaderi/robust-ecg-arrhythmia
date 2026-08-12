from pathlib import Path

import numpy as np
import wfdb

from src.noise.mixing import (
    add_noise_at_snr,
    calculate_snr_db,
)
from src.noise.nstdb import (
    VALID_NOISE_TYPES,
    load_noise_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MITDB_DIR = PROJECT_ROOT / "data" / "raw" / "mitdb"

TRAIN_RECORD = "101"

TARGET_SNRS = (
    18.0,
    6.0,
    0.0,
    -6.0,
)


def load_mlii_signal(
    record_id: str,
) -> np.ndarray:
    record = wfdb.rdrecord(
        str(MITDB_DIR / record_id)
    )

    if "MLII" not in record.sig_name:
        raise RuntimeError(
            f"Record {record_id} has no MLII lead."
        )

    lead_index = record.sig_name.index("MLII")

    return record.p_signal[:, lead_index]


def main() -> None:
    clean_signal = load_mlii_signal(
        TRAIN_RECORD
    )

    print("=" * 70)
    print("SNR MIXING VALIDATION")
    print("=" * 70)

    print(f"Development record: {TRAIN_RECORD}")
    print(f"Clean samples: {len(clean_signal)}")

    for noise_type in VALID_NOISE_TYPES:
        noise, noise_fs = load_noise_record(
            noise_type
        )

        # Use channel 0 only for this mathematical
        # validation step.
        noise_signal = noise[:, 0]

        if len(noise_signal) != len(clean_signal):
            raise RuntimeError(
                f"Length mismatch for {noise_type}: "
                f"{len(noise_signal)} vs "
                f"{len(clean_signal)}"
            )

        print(f"\nNoise type: {noise_type}")
        print(f"Noise sampling rate: {noise_fs} Hz")

        for target_snr in TARGET_SNRS:
            noisy_signal, injected_noise = (
                add_noise_at_snr(
                    clean_signal=clean_signal,
                    noise_signal=noise_signal,
                    target_snr_db=target_snr,
                )
            )

            achieved_snr = calculate_snr_db(
                clean_signal,
                injected_noise,
            )

            difference = abs(
                achieved_snr - target_snr
            )

            print(
                f"  Target: {target_snr:>5.1f} dB | "
                f"Achieved: {achieved_snr:>8.4f} dB | "
                f"Error: {difference:.6f}"
            )

            if not np.isclose(
                achieved_snr,
                target_snr,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "SNR validation failed."
                )

    print("\nSNR validation: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()