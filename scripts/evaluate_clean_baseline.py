"""Evaluate a clean-baseline checkpoint on validation data."""

import argparse
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
RESULTS_ROOT = Path("results/clean_baseline")
BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a clean ECG baseline checkpoint."
    )

    parser.add_argument(
        "--loss-weighting",
        choices=("weighted", "unweighted"),
        required=True,
        help="Choose which clean-baseline checkpoint to evaluate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = (
        RESULTS_ROOT
        / args.loss_weighting
        / "best_model.pt"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("CLEAN BASELINE VALIDATION EVALUATION")
    print("=" * 72)

    print(f"Loss weighting: {args.loss_weighting}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # ---------------------------------------------------------
    # 1. Load validation data
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 2. Load model checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = ECGCNN1D(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ---------------------------------------------------------
    # 3. Evaluate model
    # ---------------------------------------------------------

    metrics = evaluate_model(
        model=model,
        dataloader=validation_loader,
        device=device,
    )

    # ---------------------------------------------------------
    # 4. Report results
    # ---------------------------------------------------------

    print("-" * 72)

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Stored Macro-F1: "
        f"{checkpoint['validation_macro_f1']:.6f}"
    )

    print(
        f"Recomputed Macro-F1: "
        f"{metrics['macro_f1']:.6f}"
    )

    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.6f}"
    )

    print("-" * 72)

    print("Per-class validation performance:")

    for class_name in CLASS_NAMES:
        values = metrics["per_class"][class_name]

        print(
            f"{class_name}: "
            f"precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, "
            f"f1={values['f1']:.4f}, "
            f"support={values['support']}"
        )

    print("-" * 72)

    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    print("=" * 72)


if __name__ == "__main__":
    main()