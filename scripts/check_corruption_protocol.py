from pathlib import Path

import numpy as np
import wfdb

from src.data.segmentation import get_lead_signal
from src.noise.corruption import corrupt_ecg
from src.noise.nstdb import (
    VALID_NOISE_TYPES,
    load_noise_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MITDB_DIR = PROJECT_ROOT / "data" / "raw" / "mitdb"

DEVELOPMENT_RECORD = "101"

CHECK_SNRS = (
    18.0,
    0.0,
)


def main() -> None:
    record = wfdb.rdrecord(
        str(MITDB_DIR / DEVELOPMENT_RECORD)
    )

    clean_signal = get_lead_signal(
        record,
        lead_name="MLII",
    )

    print("=" * 70)
    print("CORRUPTION PROTOCOL CHECK")
    print("=" * 70)

    print(
        f"Development record: "
        f"{DEVELOPMENT_RECORD}"
    )

    print(
        f"ECG sampling rate: "
        f"{record.fs} Hz"
    )

    for noise_type in VALID_NOISE_TYPES:
        noise_record, noise_fs = (
            load_noise_record(noise_type)
        )

        if float(record.fs) != float(noise_fs):
            raise RuntimeError(
                f"Sampling-rate mismatch: "
                f"ECG={record.fs}, "
                f"noise={noise_fs}"
            )

        metadata_by_snr = []

        print(f"\nNoise type: {noise_type}")

        for snr_db in CHECK_SNRS:
            noisy_signal, metadata = corrupt_ecg(
                clean_signal=clean_signal,
                noise_record=noise_record,
                record_id=DEVELOPMENT_RECORD,
                noise_type=noise_type,
                target_snr_db=snr_db,
            )

            metadata_by_snr.append(metadata)

            if not np.all(
                np.isfinite(noisy_signal)
            ):
                raise RuntimeError(
                    "Non-finite values found "
                    "in corrupted ECG."
                )

            print(
                f"  SNR={snr_db:>5.1f} dB | "
                f"channel={metadata['noise_channel']} | "
                f"offset={metadata['start_offset']} | "
                f"achieved="
                f"{metadata['achieved_snr_db']:.4f} dB"
            )

        first = metadata_by_snr[0]
        second = metadata_by_snr[1]

        if (
            first["noise_channel"]
            != second["noise_channel"]
        ):
            raise RuntimeError(
                "Noise channel changed across SNRs."
            )

        if (
            first["start_offset"]
            != second["start_offset"]
        ):
            raise RuntimeError(
                "Noise offset changed across SNRs."
            )

    print("\nReproducible corruption: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()