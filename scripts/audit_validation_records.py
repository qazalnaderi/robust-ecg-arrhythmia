"""Audit class support for each validation record."""

from pathlib import Path

import torch

from src.data.splits import VALIDATION_RECORDS
from src.data.torch_dataset import (
    CLASS_NAMES,
    build_dataset_from_records,
)


DATA_DIR = Path("data/raw/mitdb")


def main() -> None:
    print("=" * 72)
    print("VALIDATION RECORD CLASS AUDIT")
    print("=" * 72)

    total_counts = torch.zeros(
        len(CLASS_NAMES),
        dtype=torch.long,
    )

    for record_id in VALIDATION_RECORDS:
        dataset = build_dataset_from_records(
            record_ids=(record_id,),
            data_dir=DATA_DIR,
        )

        counts = torch.bincount(
            dataset.targets,
            minlength=len(CLASS_NAMES),
        )

        total_counts += counts

        print(f"\nRecord {record_id}")
        print("-" * 32)

        for class_index, class_name in enumerate(CLASS_NAMES):
            print(
                f"{class_name}: "
                f"{counts[class_index].item()}"
            )

        print(
            f"Total: {len(dataset)}"
        )

    print("\n" + "=" * 72)
    print("VALIDATION TOTAL")
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

    print("=" * 72)


if __name__ == "__main__":
    main()