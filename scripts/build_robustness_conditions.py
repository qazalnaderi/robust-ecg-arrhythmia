import csv
from pathlib import Path

from configs.robustness_protocol import (
    EVALUATION_SNRS_DB,
    NOISE_TYPES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "robustness_conditions.csv"
)


def main() -> None:
    rows = [
        {
            "condition": "clean",
            "noise_type": "clean",
            "snr_db": "",
        }
    ]

    for noise_type in NOISE_TYPES:
        for snr_db in EVALUATION_SNRS_DB:
            rows.append(
                {
                    "condition": (
                        f"{noise_type}_{snr_db:g}db"
                    ),
                    "noise_type": noise_type,
                    "snr_db": snr_db,
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

    print("=" * 60)
    print("ROBUSTNESS CONDITIONS")
    print("=" * 60)

    print(f"Total conditions: {len(rows)}")

    for row in rows:
        print(row["condition"])

    print()
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()