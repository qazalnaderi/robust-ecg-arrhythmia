"""Audit class distributions across train, validation, and final-test splits."""

from pathlib import Path
import csv

from src.data.aami import AAMI_CLASSES
from src.data.splits import (
    DS1_RECORDS,
    DS2_RECORDS,
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RECORD_AUDIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "record_audit.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "split_class_distribution.csv"
)


def load_record_audit() -> dict[str, dict]:
    """Load record-level class counts from the MIT-BIH audit table."""

    if not RECORD_AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Record audit table not found: {RECORD_AUDIT_PATH}"
        )

    records = {}

    with RECORD_AUDIT_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            record_id = row["record_id"]

            records[record_id] = row

    return records


def aggregate_split(
    record_ids: tuple[str, ...],
    records: dict[str, dict],
    split_name: str,
) -> dict:
    """Aggregate heartbeat counts for a group of MIT-BIH records."""

    result = {
        "split": split_name,
        "number_of_records": len(record_ids),
    }

    for class_name in AAMI_CLASSES:
        result[class_name] = sum(
            int(records[record_id][class_name])
            for record_id in record_ids
        )

    result["mapped_beats"] = sum(
        int(records[record_id]["mapped_beats"])
        for record_id in record_ids
    )

    return result


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Write aggregated split statistics to CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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


def main() -> None:
    records = load_record_audit()

    split_rows = [
        aggregate_split(
            TRAIN_RECORDS,
            records,
            "train",
        ),
        aggregate_split(
            VALIDATION_RECORDS,
            records,
            "validation",
        ),
        aggregate_split(
            DS1_RECORDS,
            records,
            "ds1_development_total",
        ),
        aggregate_split(
            DS2_RECORDS,
            records,
            "ds2_final_test",
        ),
    ]

    print("=" * 70)
    print("MIT-BIH SPLIT AUDIT")
    print("=" * 70)

    for row in split_rows:
        print(
            f"\nSplit: {row['split']} "
            f"({row['number_of_records']} records)"
        )

        for class_name in AAMI_CLASSES:
            print(
                f"  {class_name}: "
                f"{row[class_name]}"
            )

        print(
            f"  Mapped beats: "
            f"{row['mapped_beats']}"
        )

    write_csv(
        OUTPUT_PATH,
        split_rows,
    )

    print("\nSaved table:")
    print(OUTPUT_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()