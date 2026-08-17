"""Evaluate the clean baseline separately on each validation record."""

import csv
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

CHECKPOINT_PATH = Path(
    "results/clean_baseline/sqrt_weighted/best_model.pt"
)

OUTPUT_PATH = Path(
    "results/tables/validation_per_record_metrics.csv"
)

BATCH_SIZE = 256


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ---------------------------------------------------------
    # 1. Load the frozen candidate checkpoint once
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = ECGCNN1D(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("=" * 72)
    print("PER-RECORD VALIDATION EVALUATION")
    print("=" * 72)

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Best epoch: {checkpoint['epoch']}"
    )

    print(
        f"Stored validation Macro-F1: "
        f"{checkpoint['validation_macro_f1']:.6f}"
    )

    rows = []

    # ---------------------------------------------------------
    # 2. Evaluate every validation patient separately
    # ---------------------------------------------------------

    for record_id in VALIDATION_RECORDS:

        dataset = build_dataset_from_records(
            record_ids=(record_id,),
            data_dir=DATA_DIR,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        metrics = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
        )

        print("\n" + "-" * 72)
        print(f"Record {record_id}")
        print("-" * 72)

        print(
            f"Macro-F1: "
            f"{metrics['macro_f1']:.6f}"
        )

        print(
            f"Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.6f}"
        )

        for class_name in CLASS_NAMES:
            values = metrics["per_class"][
                class_name
            ]

            print(
                f"{class_name}: "
                f"precision={values['precision']:.4f}, "
                f"recall={values['recall']:.4f}, "
                f"f1={values['f1']:.4f}, "
                f"support={values['support']}"
            )

        print("Confusion matrix:")
        print(metrics["confusion_matrix"])

        # -----------------------------------------------------
        # 3. Save compact metrics for later analysis
        # -----------------------------------------------------

        row = {
            "record_id": record_id,
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": (
                metrics["balanced_accuracy"]
            ),
        }

        for class_name in CLASS_NAMES:
            values = metrics["per_class"][
                class_name
            ]

            row[f"{class_name}_precision"] = (
                values["precision"]
            )

            row[f"{class_name}_recall"] = (
                values["recall"]
            )

            row[f"{class_name}_f1"] = (
                values["f1"]
            )

            row[f"{class_name}_support"] = (
                values["support"]
            )

        rows.append(row)

    # ---------------------------------------------------------
    # 4. Save results automatically
    # ---------------------------------------------------------

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

    print("\n" + "=" * 72)

    print(
        f"Saved per-record metrics to: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()