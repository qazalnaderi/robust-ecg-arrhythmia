"""Train the clean ECG baseline model."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.splits import (
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
)
from src.data.torch_dataset import (
    CLASS_NAMES,
    build_dataset_from_records,
)
from src.evaluation.evaluator import evaluate_model
from src.models.cnn1d import ECGCNN1D
from src.training.engine import (
    evaluate_loss,
    train_one_epoch,
)
from src.training.reproducibility import (
    DEFAULT_SEED,
    set_global_seed,
)
from src.training.weights import compute_class_weights


DATA_DIR = Path("data/raw/mitdb")
RESULTS_ROOT = Path("results/clean_baseline")

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train the clean ECG baseline."
    )

    parser.add_argument(
        "--loss-weighting",
        choices=("weighted", "unweighted"),
        default="weighted",
        help="Choose weighted or unweighted cross-entropy loss.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loss_weighting = args.loss_weighting

    # Separate folder for each experiment.
    output_dir = (
        RESULTS_ROOT
        / loss_weighting
    )

    checkpoint_path = (
        output_dir
        / "best_model.pt"
    )

    # ---------------------------------------------------------
    # 1. Reproducibility
    # ---------------------------------------------------------

    set_global_seed(
        DEFAULT_SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("CLEAN ECG BASELINE TRAINING")
    print("=" * 72)

    print(f"Device: {device}")
    print(f"Seed: {DEFAULT_SEED}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Loss weighting: {loss_weighting}")

    # ---------------------------------------------------------
    # 2. Build datasets
    # ---------------------------------------------------------

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

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        "Validation samples: "
        f"{len(validation_dataset)}"
    )

    # ---------------------------------------------------------
    # 3. DataLoaders
    # ---------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        DEFAULT_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # 4. Model
    # ---------------------------------------------------------

    model = ECGCNN1D(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    # ---------------------------------------------------------
    # 5. Loss
    # ---------------------------------------------------------

    if loss_weighting == "weighted":

        class_weights = compute_class_weights(
            targets=train_dataset.targets,
            num_classes=len(CLASS_NAMES),
        ).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        print("\nClass weights:")

        for class_name, weight in zip(
            CLASS_NAMES,
            class_weights.tolist(),
        ):
            print(
                f"  {class_name}: {weight:.6f}"
            )

    else:

        criterion = nn.CrossEntropyLoss()

        print(
            "\nUsing unweighted CrossEntropyLoss."
        )

    # ---------------------------------------------------------
    # 6. Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------------------------------------
    # 7. Training
    # ---------------------------------------------------------

    best_macro_f1 = -1.0
    best_epoch = -1

    print("\nStarting training...")
    print("-" * 72)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = evaluate_loss(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        metrics = evaluate_model(
            model=model,
            dataloader=validation_loader,
            device=device,
        )

        macro_f1 = metrics[
            "macro_f1"
        ]

        balanced_accuracy = metrics[
            "balanced_accuracy"
        ]

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Macro-F1: {macro_f1:.4f} | "
            f"Bal Acc: {balanced_accuracy:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "validation_macro_f1": (
                        macro_f1
                    ),
                    "validation_balanced_accuracy": (
                        balanced_accuracy
                    ),
                    "seed": DEFAULT_SEED,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": (
                        LEARNING_RATE
                    ),
                    "class_names": CLASS_NAMES,
                    "loss_weighting": (
                        loss_weighting
                    ),
                },
                checkpoint_path,
            )

            print(
                "  -> Saved new best checkpoint."
            )

    # ---------------------------------------------------------
    # 8. Final report
    # ---------------------------------------------------------

    print("-" * 72)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        "Best validation Macro-F1: "
        f"{best_macro_f1:.6f}"
    )

    print(
        "Checkpoint: "
        f"{checkpoint_path}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()