"""Build the reproducible NSTDB corruption manifest for development data."""

import csv
from pathlib import Path

from src.data.splits import (
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
)
from src.noise.corruption import select_noise_variant
from src.noise.nstdb import (
    VALID_NOISE_TYPES,
    load_noise_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "development_corruption_manifest.csv"
)


def main() -> None:
    rows = []

    development_splits = {
        "train": TRAIN_RECORDS,
        "validation": VALIDATION_RECORDS,
    }

    # Load each NSTDB recording once.
    noise_records = {
        noise_type: load_noise_record(noise_type)[0]
        for noise_type in VALID_NOISE_TYPES
    }

    for split_name, record_ids in development_splits.items():

        for record_id in record_ids:

            for noise_type in VALID_NOISE_TYPES:

                noise_record = noise_records[
                    noise_type
                ]

                # We only need the deterministic selection metadata here.
                _, metadata = select_noise_variant(
                    noise_record=noise_record,
                    target_length=1,
                    record_id=record_id,
                    noise_type=noise_type,
                )

                rows.append(
                    {
                        "split": split_name,
                        "record_id": record_id,
                        "noise_type": noise_type,
                        "noise_channel": metadata[
                            "noise_channel"
                        ],
                        "start_offset": metadata[
                            "start_offset"
                        ],
                        "seed": metadata[
                            "seed"
                        ],
                    }
                )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 70)
    print("DEVELOPMENT CORRUPTION MANIFEST")
    print("=" * 70)

    print(
        f"Train records: {len(TRAIN_RECORDS)}"
    )

    print(
        f"Validation records: "
        f"{len(VALIDATION_RECORDS)}"
    )

    print(
        f"Noise types: "
        f"{len(VALID_NOISE_TYPES)}"
    )

    print(
        f"Manifest rows: {len(rows)}"
    )

    print()
    print(f"Saved to:")
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()