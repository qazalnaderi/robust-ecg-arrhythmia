from collections import Counter
from pathlib import Path
import csv

import wfdb

from src.data.aami import AAMI_CLASSES, map_to_aami


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mitdb"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

# Records dominated by paced beats.
# We audit them, but they will later be excluded from the core AAMI experiment.
PACED_EXCLUDED_RECORDS = {"102", "104", "107", "217"}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dictionaries to a CSV file."""

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    rows: list[dict],
    scope_name: str,
) -> dict:
    """Aggregate AAMI counts over a group of records."""

    result = {
        "scope": scope_name,
        "number_of_records": len(rows),
    }

    for class_name in AAMI_CLASSES:
        result[class_name] = sum(
            row[class_name]
            for row in rows
        )

    result["mapped_beats"] = sum(
        row["mapped_beats"]
        for row in rows
    )

    result["all_annotations"] = sum(
        row["number_of_annotations"]
        for row in rows
    )

    result["unmapped_annotations"] = sum(
        row["unmapped_annotations"]
        for row in rows
    )

    return result


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"MIT-BIH directory not found: {DATA_DIR}"
        )

    header_files = sorted(
        DATA_DIR.glob("*.hea"),
        key=lambda path: int(path.stem),
    )

    if len(header_files) != 48:
        raise RuntimeError(
            f"Expected 48 MIT-BIH records, found {len(header_files)}."
        )

    record_rows = []

    total_unmapped_symbols = Counter()

    print("=" * 70)
    print("MIT-BIH DATASET AUDIT")
    print("=" * 70)

    for header_file in header_files:
        record_id = header_file.stem
        record_path = DATA_DIR / record_id

        # Header only: enough for metadata.
        # We do not need to load the full ECG signal for this audit.
        header = wfdb.rdheader(
            str(record_path)
        )

        annotation = wfdb.rdann(
            str(record_path),
            extension="atr",
        )

        class_counts = Counter({
            class_name: 0
            for class_name in AAMI_CLASSES
        })

        unmapped_symbols = Counter()

        for symbol in annotation.symbol:
            aami_class = map_to_aami(symbol)

            if aami_class is None:
                unmapped_symbols[symbol] += 1
            else:
                class_counts[aami_class] += 1

        mapped_beats = sum(
            class_counts.values()
        )

        unmapped_count = sum(
            unmapped_symbols.values()
        )

        total_unmapped_symbols.update(
            unmapped_symbols
        )

        duration_minutes = (
            header.sig_len / header.fs / 60
        )

        row = {
            "record_id": record_id,
            "sampling_rate_hz": header.fs,
            "number_of_channels": header.n_sig,
            "signal_names": "|".join(header.sig_name),
            "signal_length": header.sig_len,
            "duration_minutes": round(duration_minutes, 3),
            "number_of_annotations": len(annotation.sample),
            "N": class_counts["N"],
            "S": class_counts["S"],
            "V": class_counts["V"],
            "F": class_counts["F"],
            "Q": class_counts["Q"],
            "mapped_beats": mapped_beats,
            "unmapped_annotations": unmapped_count,
            "paced_excluded": (
                record_id in PACED_EXCLUDED_RECORDS
            ),
        }

        record_rows.append(row)

        print(
            f"Record {record_id}: "
            f"N={class_counts['N']} "
            f"S={class_counts['S']} "
            f"V={class_counts['V']} "
            f"F={class_counts['F']} "
            f"Q={class_counts['Q']}"
        )

    core_rows = [
        row
        for row in record_rows
        if not row["paced_excluded"]
    ]

    distribution_rows = [
        aggregate_rows(
            record_rows,
            "all_48_records",
        ),
        aggregate_rows(
            core_rows,
            "core_44_non_paced_records",
        ),
    ]

    unmapped_rows = [
        {
            "symbol": symbol,
            "count": count,
        }
        for symbol, count
        in total_unmapped_symbols.most_common()
    ]

    write_csv(
        RESULTS_DIR / "record_audit.csv",
        record_rows,
    )

    write_csv(
        RESULTS_DIR / "class_distribution.csv",
        distribution_rows,
    )

    write_csv(
        RESULTS_DIR / "unmapped_annotations.csv",
        unmapped_rows,
    )

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(f"Records audited: {len(record_rows)}")
    print(f"Core non-paced records: {len(core_rows)}")

    for row in distribution_rows:
        print(f"\nScope: {row['scope']}")

        for class_name in AAMI_CLASSES:
            print(
                f"  {class_name}: "
                f"{row[class_name]}"
            )

        print(
            f"  Mapped beats: "
            f"{row['mapped_beats']}"
        )

        print(
            f"  Unmapped annotations: "
            f"{row['unmapped_annotations']}"
        )

    print("\nUnmapped annotation symbols:")

    for symbol, count in total_unmapped_symbols.most_common():
        print(
            f"  {repr(symbol)}: {count}"
        )

    print("\nSaved tables:")
    print(
        RESULTS_DIR / "record_audit.csv"
    )
    print(
        RESULTS_DIR / "class_distribution.csv"
    )
    print(
        RESULTS_DIR / "unmapped_annotations.csv"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()