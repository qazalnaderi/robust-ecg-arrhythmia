"""Evaluate the clean ECG + relative-RR baseline."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.relative_rr import (
    make_relative_rr_features,
)
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

CHECKPOINT_PATH = Path(
    "results/clean_baseline_relative_rr/"
    "sqrt_weighted/best_model.pt"
)

BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
        ),
        default="validation",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.split == "train":
        record_ids = TRAIN_RECORDS
    else:
        record_ids = VALIDATION_RECORDS

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    dataset = build_dataset_with_rr_from_records(
        record_ids=record_ids,
        data_dir=DATA_DIR,
    )

    # Raw RR -> relative RR
    relative_rr = make_relative_rr_features(
        dataset.rr_features.cpu().numpy()
    )

    dataset.rr_features = torch.from_numpy(
        relative_rr
    )

    # Apply TRAIN statistics stored in checkpoint.
    dataset.rr_features = (
        standardize_rr_features(
            rr_features=dataset.rr_features,
            mean=checkpoint["rr_mean"],
            std=checkpoint["rr_std"],
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = ECGCNN1DWithRR(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    metrics = evaluate_rr_model(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    print("=" * 72)

    print(
        f"RELATIVE RR {args.split.upper()} EVALUATION"
    )

    print("=" * 72)

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
    print(metrics["confusion_matrix"])

    print("=" * 72)


if __name__ == "__main__":
    main()