"""Smoke-check heartbeat segmentation on one real MIT-BIH record."""

from collections import Counter
from pathlib import Path

import wfdb

from src.data.aami import map_to_aami
from src.data.segmentation import (
    CORE_CLASSES,
    DEFAULT_LEAD,
    extract_heartbeat_window,
    get_lead_signal,
    segment_record,
)


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "114"

def main() -> None:
    record_path = DATA_DIR / TEST_RECORD

    # Read basic record metadata.
    record = wfdb.rdrecord(
        str(record_path)
    )

    # Read heartbeat annotations.
    annotation = wfdb.rdann(
        str(record_path),
        extension="atr",
    )

    # Select the requested ECG lead by name.
    signal = get_lead_signal(
        record,
        lead_name=DEFAULT_LEAD,
    )

    # Run the real segmentation pipeline.
    X, y, metadata = segment_record(
        record_path=record_path,
        lead_name=DEFAULT_LEAD,
    )

    class_counts = Counter(y)

    # Count all core AAMI annotations before segmentation
    # and identify beats skipped because a full window
    # cannot be extracted near the signal boundaries.
    mapped_core_beats = 0
    boundary_skipped = []

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol,
    ):
        aami_class = map_to_aami(symbol)

        if aami_class not in CORE_CLASSES:
            continue

        mapped_core_beats += 1

        window = extract_heartbeat_window(
            signal=signal,
            center_sample=int(sample),
        )

        if window is None:
            boundary_skipped.append(
                {
                    "sample": int(sample),
                    "symbol": symbol,
                    "aami_class": aami_class,
                }
            )

    print("=" * 70)
    print("MIT-BIH HEARTBEAT SEGMENTATION CHECK")
    print("=" * 70)

    print(f"Record: {TEST_RECORD}")
    print(f"Sampling frequency: {record.fs} Hz")
    print(f"Available leads: {record.sig_name}")
    print(f"Selected lead: {DEFAULT_LEAD}")

    print("-" * 70)

    print(f"Segment array shape: {X.shape}")
    print(f"Number of labels: {len(y)}")
    print(f"Number of metadata rows: {len(metadata)}")

    print(f"Mapped core annotations: {mapped_core_beats}")
    print(f"Boundary-skipped beats: {len(boundary_skipped)}")

    for beat in boundary_skipped:
        print(f"  Skipped: {beat}")

    print("-" * 70)

    print("Segmented class counts:")

    for class_name in CORE_CLASSES:
        print(
            f"  {class_name}: "
            f"{class_counts.get(class_name, 0)}"
        )

    print("-" * 70)

    if len(X) > 0:
        print(
            f"Heartbeat length: "
            f"{X.shape[1]} samples"
        )

        print(
            "First heartbeat metadata:",
            metadata[0],
        )

    print("=" * 70)


if __name__ == "__main__":
    main()