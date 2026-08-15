"""Check real train/validation ECG datasets before model training."""

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
from src.training.weights import compute_class_weights


DATA_DIR = Path("data/raw/mitdb")


def print_distribution(
    name: str,
    targets: torch.Tensor,
) -> None:
    """Print class distribution for one dataset."""

    counts = torch.bincount(
        targets,
        minlength=len(CLASS_NAMES),
    )

    print(name)

    for class_name, count in zip(
        CLASS_NAMES,
        counts.tolist(),
    ):
        print(
            f"  {class_name}: {count}"
        )

    print(
        f"  Total: {len(targets)}"
    )


def main() -> None:
    print("=" * 70)
    print("MIT-BIH TRAIN / VALIDATION DATA CHECK")
    print("=" * 70)

    print(
        f"Train records: {len(TRAIN_RECORDS)}"
    )

    print(
        f"Validation records: {len(VALIDATION_RECORDS)}"
    )

    print("\nBuilding training dataset...")

    train_dataset = build_dataset_from_records(
        record_ids=TRAIN_RECORDS,
        data_dir=DATA_DIR,
    )

    print("Building validation dataset...")

    validation_dataset = build_dataset_from_records(
        record_ids=VALIDATION_RECORDS,
        data_dir=DATA_DIR,
    )

    print("-" * 70)

    print_distribution(
        "Training distribution:",
        train_dataset.targets,
    )

    print()

    print_distribution(
        "Validation distribution:",
        validation_dataset.targets,
    )

    print("-" * 70)

    class_weights = compute_class_weights(
        targets=train_dataset.targets,
        num_classes=len(CLASS_NAMES),
    )

    print("Training class weights:")

    for class_name, weight in zip(
        CLASS_NAMES,
        class_weights.tolist(),
    ):
        print(
            f"  {class_name}: {weight:.6f}"
        )

    print("-" * 70)

    x, y = train_dataset[0]

    print(
        f"One training input shape: {tuple(x.shape)}"
    )

    print(
        f"One training target: {y.item()}"
    )

    print(
        f"Input dtype: {x.dtype}"
    )

    print(
        f"Target dtype: {y.dtype}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()