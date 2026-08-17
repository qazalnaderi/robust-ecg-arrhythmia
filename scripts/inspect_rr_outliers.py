"""Inspect unusually long RR intervals in validation records."""

from collections import Counter
from pathlib import Path

import wfdb

from src.data.aami import map_to_aami
from src.data.splits import VALIDATION_RECORDS


DATA_DIR = Path("data/raw/mitdb")

RR_THRESHOLD_SECONDS = 3.0


def main() -> None:

    print("=" * 72)
    print("VALIDATION RR OUTLIER INSPECTION")
    print("=" * 72)

    total_outliers = 0

    for record_id in VALIDATION_RECORDS:

        record_path = DATA_DIR / record_id

        header = wfdb.rdheader(
            str(record_path)
        )

        annotation = wfdb.rdann(
            str(record_path),
            extension="atr",
        )

        # -----------------------------------------------------
        # Keep indices of actual heartbeat annotations
        # -----------------------------------------------------

        beat_annotation_indices = []

        for annotation_index, symbol in enumerate(
            annotation.symbol
        ):
            if map_to_aami(symbol) is not None:
                beat_annotation_indices.append(
                    annotation_index
                )

        record_outliers = []

        # -----------------------------------------------------
        # Inspect consecutive heartbeat annotations
        # -----------------------------------------------------

        for beat_index in range(
            len(beat_annotation_indices) - 1
        ):

            previous_index = (
                beat_annotation_indices[beat_index]
            )

            next_index = (
                beat_annotation_indices[beat_index + 1]
            )

            previous_sample = int(
                annotation.sample[previous_index]
            )

            next_sample = int(
                annotation.sample[next_index]
            )

            rr_seconds = (
                next_sample - previous_sample
            ) / header.fs

            if rr_seconds <= RR_THRESHOLD_SECONDS:
                continue

            previous_symbol = (
                annotation.symbol[previous_index]
            )

            next_symbol = (
                annotation.symbol[next_index]
            )

            # Raw annotation symbols that occurred between
            # the two heartbeat annotations.
            symbols_between = annotation.symbol[
                previous_index + 1:
                next_index
            ]

            symbol_counts = Counter(
                symbols_between
            )

            record_outliers.append(
                {
                    "rr_seconds": rr_seconds,
                    "previous_sample": previous_sample,
                    "previous_symbol": previous_symbol,
                    "previous_class": map_to_aami(
                        previous_symbol
                    ),
                    "next_sample": next_sample,
                    "next_symbol": next_symbol,
                    "next_class": map_to_aami(
                        next_symbol
                    ),
                    "symbols_between": dict(
                        symbol_counts
                    ),
                }
            )

        if not record_outliers:
            continue

        record_outliers.sort(
            key=lambda item: item["rr_seconds"],
            reverse=True,
        )

        total_outliers += len(
            record_outliers
        )

        print(f"\nRecord {record_id}")
        print("-" * 72)

        for index, outlier in enumerate(
            record_outliers,
            start=1,
        ):

            print(
                f"\nOutlier {index}"
            )

            print(
                f"RR interval: "
                f"{outlier['rr_seconds']:.4f} s"
            )

            print(
                "Previous beat: "
                f"sample={outlier['previous_sample']}, "
                f"symbol={outlier['previous_symbol']}, "
                f"class={outlier['previous_class']}"
            )

            print(
                "Next beat: "
                f"sample={outlier['next_sample']}, "
                f"symbol={outlier['next_symbol']}, "
                f"class={outlier['next_class']}"
            )

            print(
                "Raw annotation symbols between beats: "
                f"{outlier['symbols_between']}"
            )

    print("\n" + "=" * 72)

    print(
        f"Total RR intervals > "
        f"{RR_THRESHOLD_SECONDS:.1f} s: "
        f"{total_outliers}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()