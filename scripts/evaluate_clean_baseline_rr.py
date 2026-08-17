"""Evaluate the clean ECG + RR baseline checkpoint."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.rr_normalization import (
    standardize_rr_features,
)
from src.data.splits import (
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
)
from src.data.torch_dataset import (
    CLASS_NAMES,
    build_dataset_with_rr_from_records,
)
from src.evaluation.rr_evaluator import (
    evaluate_rr_model,
)
from src.models.cnn1d_rr import (
    ECGCNN1DWithRR,
)


DATA_DIR = Path("data/raw/mitdb")

RESULTS_ROOT = Path(
    "results/clean_baseline_rr"
)

BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a clean ECG + RR baseline checkpoint."
        )
    )

    parser.add_argument(
        "--loss-weighting",
        choices=(
            "weighted",
            "sqrt_weighted",
            "unweighted",
        ),
        required=True,
        help="Choose which ECG + RR checkpoint to evaluate.",
    )

    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
        ),
        default="validation",
        help="Choose which development split to evaluate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------
    # 1. Select split
    # ---------------------------------------------------------

    if args.split == "train":
        record_ids = TRAIN_RECORDS
    else:
        record_ids = VALIDATION_RECORDS

    checkpoint_path = (
        RESULTS_ROOT
        / args.loss_weighting
        / "best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ---------------------------------------------------------
    # 2. Load checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # ---------------------------------------------------------
    # 3. Build requested dataset
    # ---------------------------------------------------------

    dataset = build_dataset_with_rr_from_records(
        record_ids=record_ids,
        data_dir=DATA_DIR,
    )

    # ---------------------------------------------------------
    # 4. Apply TRAIN normalization stored in checkpoint
    # ---------------------------------------------------------

    rr_mean = checkpoint[
        "rr_mean"
    ].to(
        dataset.rr_features.device
    )

    rr_std = checkpoint[
        "rr_std"
    ].to(
        dataset.rr_features.device
    )

    dataset.rr_features = (
        standardize_rr_features(
            rr_features=dataset.rr_features,
            mean=rr_mean,
            std=rr_std,
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # 5. Restore model
    # ---------------------------------------------------------

    model = ECGCNN1DWithRR(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ---------------------------------------------------------
    # 6. Evaluate
    # ---------------------------------------------------------

    metrics = evaluate_rr_model(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    # ---------------------------------------------------------
    # 7. Report
    # ---------------------------------------------------------

    print("=" * 72)

    print(
        f"CLEAN ECG + RR {args.split.upper()} EVALUATION"
    )

    print("=" * 72)

    print(
        f"Loss weighting: {args.loss_weighting}"
    )

    print(
        f"Evaluation split: {args.split}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print("-" * 72)

    print(
        f"Best epoch: {checkpoint['epoch']}"
    )

    print(
        "Checkpoint Validation Macro-F1: "
        f"{checkpoint['validation_macro_f1']:.6f}"
    )

    print(
        f"{args.split.capitalize()} Macro-F1: "
        f"{metrics['macro_f1']:.6f}"
    )

    print(
        "Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.6f}"
    )

    print("-" * 72)

    print(
        f"Per-class {args.split} performance:"
    )

    for class_name in CLASS_NAMES:

        values = metrics[
            "per_class"
        ][class_name]

        print(
            f"{class_name}: "
            f"precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, "
            f"f1={values['f1']:.4f}, "
            f"support={values['support']}"
        )

    print("-" * 72)

    print("Confusion matrix:")

    print(
        metrics["confusion_matrix"]
    )

    print("=" * 72)


if __name__ == "__main__":
    main()