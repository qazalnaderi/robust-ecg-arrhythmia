"""Check alignment between segmented ECG beats and RR features."""

from pathlib import Path

import numpy as np
import wfdb

from src.data.aami import map_to_aami
from src.data.rr_features import compute_rr_features
from src.data.segmentation import segment_record


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "100"

CORE_CLASSES = {
    "N",
    "S",
    "V",
    "F",
}


def main() -> None:
    record_path = DATA_DIR / TEST_RECORD

    # ---------------------------------------------------------
    # 1. Read original annotations
    # ---------------------------------------------------------

    header = wfdb.rdheader(
        str(record_path)
    )

    annotation = wfdb.rdann(
        str(record_path),
        extension="atr",
    )

    # ---------------------------------------------------------
    # 2. Build the complete heartbeat sequence for RR
    # ---------------------------------------------------------

    rr_beat_samples = []

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol,
    ):
        if map_to_aami(symbol) is not None:
            rr_beat_samples.append(sample)

    rr_beat_samples = np.asarray(
        rr_beat_samples,
        dtype=np.int64,
    )

    rr_features = compute_rr_features(
        beat_samples=rr_beat_samples,
        sampling_rate=header.fs,
    )

    # Map annotation sample -> RR feature row
    rr_by_sample = {
        int(sample): features
        for sample, features in zip(
            rr_beat_samples,
            rr_features,
        )
    }

    # ---------------------------------------------------------
    # 3. Build the actual segmented core dataset
    # ---------------------------------------------------------

    X, y, metadata = segment_record(
        record_path=record_path
    )

    # ---------------------------------------------------------
    # 4. Verify one-to-one alignment
    # ---------------------------------------------------------

    missing_samples = []

    aligned_rr = []

    for beat_metadata in metadata:
        annotation_sample = int(
            beat_metadata["annotation_sample"]
        )

        if annotation_sample not in rr_by_sample:
            missing_samples.append(
                annotation_sample
            )
            continue

        aligned_rr.append(
            rr_by_sample[annotation_sample]
        )

    aligned_rr = np.asarray(
        aligned_rr,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # 5. Report
    # ---------------------------------------------------------

    print("=" * 72)
    print("RR / ECG SEGMENT ALIGNMENT CHECK")
    print("=" * 72)

    print(f"Record: {TEST_RECORD}")
    print(f"RR heartbeat sequence: {len(rr_beat_samples)}")
    print(f"Segmented core beats: {len(X)}")
    print(f"Labels: {len(y)}")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Aligned RR rows: {len(aligned_rr)}")
    print(f"Missing RR matches: {len(missing_samples)}")

    if missing_samples:
        print(
            "Missing annotation samples:",
            missing_samples[:20],
        )

    print("\nFirst 10 aligned beats:")

    for index in range(
        min(10, len(metadata))
    ):
        sample = metadata[index][
            "annotation_sample"
        ]

        symbol = metadata[index][
            "original_symbol"
        ]

        label = y[index]

        rr = rr_by_sample[
            int(sample)
        ]

        print(
            f"{index:02d}: "
            f"sample={sample}, "
            f"symbol={symbol}, "
            f"class={label}, "
            f"pre_rr={rr[0]:.4f}, "
            f"post_rr={rr[1]:.4f}"
        )

    # ---------------------------------------------------------
    # 6. Hard sanity checks
    # ---------------------------------------------------------

    if len(missing_samples) != 0:
        raise RuntimeError(
            "Some segmented beats could not be matched "
            "to RR features."
        )

    if len(aligned_rr) != len(X):
        raise RuntimeError(
            "RR feature count does not match segmented beat count."
        )

    print("\nAlignment check: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()