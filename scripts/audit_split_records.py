"""Audit class support for each record in a development split."""

import argparse
from pathlib import Path

import torch

from src.data.splits import (
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
)
from src.data.torch_dataset import (
    CLASS_NAMES,
    build_dataset_from_records,
)


DATA_DIR = Path("data/raw/mitdb")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Audit class support by ECG record."
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        required=True,
        help="Choose which development split to audit.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------
    # 1. Select records
    # ---------------------------------------------------------

    if args.split == "train":
        record_ids = TRAIN_RECORDS
    else:
        record_ids = VALIDATION_RECORDS

    print("=" * 72)
    print(f"{args.split.upper()} RECORD CLASS AUDIT")
    print("=" * 72)

    # Total number of beats from each class
    total_counts = torch.zeros(
        len(CLASS_NAMES),
        dtype=torch.long,
    )

    # Number of different records that contain each class
    records_with_class = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    # ---------------------------------------------------------
    # 2. Audit every record separately
    # ---------------------------------------------------------

    for record_id in record_ids:
        dataset = build_dataset_from_records(
            record_ids=(record_id,),
            data_dir=DATA_DIR,
        )

        counts = torch.bincount(
            dataset.targets,
            minlength=len(CLASS_NAMES),
        )

        total_counts += counts

        for class_index, class_name in enumerate(CLASS_NAMES):
            if counts[class_index].item() > 0:
                records_with_class[class_name] += 1

        print(f"\nRecord {record_id}")
        print("-" * 32)

        for class_index, class_name in enumerate(CLASS_NAMES):
            print(
                f"{class_name}: "
                f"{counts[class_index].item()}"
            )

        print(f"Total: {len(dataset)}")

    # ---------------------------------------------------------
    # 3. Split-level summary
    # ---------------------------------------------------------

    print("\n" + "=" * 72)
    print(f"{args.split.upper()} TOTAL")
    print("=" * 72)

    for class_index, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name}: "
            f"{total_counts[class_index].item()}"
        )

    print(
        f"Total beats: "
        f"{total_counts.sum().item()}"
    )

    # ---------------------------------------------------------
    # 4. Patient/record diversity of each class
    # ---------------------------------------------------------

    print("\nRecords containing each class:")

    for class_name in CLASS_NAMES:
        print(
            f"{class_name}: "
            f"{records_with_class[class_name]} "
            f"of {len(record_ids)} records"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()