from pathlib import Path

import numpy as np

from src.data.normalization import normalize_heartbeats
from src.data.segmentation import segment_record
from src.noise.heartbeat_pipeline import (
    build_noisy_heartbeats,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MITDB_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mitdb"
)

RECORD_ID = "101"

NOISE_TYPE = "ma"
SNR_DB = 6.0


def main() -> None:
    record_path = MITDB_DIR / RECORD_ID

    print("=" * 70)
    print("NOISY HEARTBEAT PIPELINE CHECK")
    print("=" * 70)

    # Clean heartbeat pipeline from Rozhin's code.
    clean_X, clean_y, clean_metadata = (
        segment_record(record_path)
    )

    clean_X = normalize_heartbeats(
        clean_X
    )

    # Noisy pipeline.
    noisy_X, noisy_y, noisy_metadata, corruption = (
        build_noisy_heartbeats(
            record_path=record_path,
            noise_type=NOISE_TYPE,
            target_snr_db=SNR_DB,
        )
    )

    print(f"Record: {RECORD_ID}")
    print(
        f"Noise: {NOISE_TYPE} @ {SNR_DB} dB"
    )

    print()
    print(f"Clean shape: {clean_X.shape}")
    print(f"Noisy shape: {noisy_X.shape}")

    # -------------------------
    # Check 1: same shape
    # -------------------------
    if clean_X.shape != noisy_X.shape:
        raise RuntimeError(
            "Clean and noisy heartbeat shapes differ."
        )

    # -------------------------
    # Check 2: same labels
    # -------------------------
    if not np.array_equal(
        clean_y,
        noisy_y,
    ):
        raise RuntimeError(
            "Clean and noisy labels differ."
        )

    # -------------------------
    # Check 3: same beat positions
    # -------------------------
    clean_samples = [
        item["annotation_sample"]
        for item in clean_metadata
    ]

    noisy_samples = [
        item["annotation_sample"]
        for item in noisy_metadata
    ]

    if clean_samples != noisy_samples:
        raise RuntimeError(
            "Heartbeat annotation positions differ."
        )

    # -------------------------
    # Check 4: corruption actually changed ECG
    # -------------------------
    if np.allclose(
        clean_X,
        noisy_X,
    ):
        raise RuntimeError(
            "Noisy heartbeats are identical to clean heartbeats."
        )

    # -------------------------
    # Check 5: finite values
    # -------------------------
    if not np.all(
        np.isfinite(noisy_X)
    ):
        raise RuntimeError(
            "Non-finite values found."
        )

    print()
    print(
        f"Heartbeats: {len(noisy_y)}"
    )

    print(
        f"Achieved continuous SNR: "
        f"{corruption['achieved_snr_db']:.4f} dB"
    )

    print(
        f"Noise channel: "
        f"{corruption['noise_channel']}"
    )

    print(
        f"Noise offset: "
        f"{corruption['start_offset']}"
    )

    print()
    print("Labels unchanged: PASS")
    print("Annotation alignment: PASS")
    print("Noisy signal differs from clean: PASS")
    print("Finite normalized heartbeats: PASS")

    print()
    print("NOISY HEARTBEAT PIPELINE: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()