"""Smoke-check evaluation pipeline on real MIT-BIH validation data."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.splits import VALIDATION_RECORDS
from src.data.torch_dataset import (
    CLASS_NAMES,
    build_dataset_from_records,
)
from src.evaluation.evaluator import evaluate_model
from src.models.cnn1d import ECGCNN1D


DATA_DIR = Path("data/raw/mitdb")
BATCH_SIZE = 256


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("REAL ECG EVALUATION PIPELINE CHECK")
    print("=" * 70)

    print(f"Device: {device}")
    print("Building validation dataset...")

    validation_dataset = build_dataset_from_records(
        record_ids=VALIDATION_RECORDS,
        data_dir=DATA_DIR,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = ECGCNN1D(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    metrics = evaluate_model(
        model=model,
        dataloader=validation_loader,
        device=device,
    )

    print("-" * 70)

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Macro-F1: "
        f"{metrics['macro_f1']:.6f}"
    )

    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.6f}"
    )

    print("-" * 70)

    print("Per-class metrics:")

    for class_name in CLASS_NAMES:
        values = metrics["per_class"][class_name]

        print(
            f"{class_name}: "
            f"precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, "
            f"f1={values['f1']:.4f}, "
            f"support={values['support']}"
        )

    print("-" * 70)

    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    print("=" * 70)
    print(
        "NOTE: Model is untrained; metric values "
        "are not performance results."
    )


if __name__ == "__main__":
    main()