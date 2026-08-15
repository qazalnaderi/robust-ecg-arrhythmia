"""Smoke-check the training pipeline on real MIT-BIH data."""

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
from src.models.cnn1d import ECGCNN1D
from src.training.engine import (
    evaluate_loss,
    train_one_epoch,
)
from src.training.weights import compute_class_weights


DATA_DIR = Path("data/raw/mitdb")

BATCH_SIZE = 256
LEARNING_RATE = 1e-3


def main() -> None:
    # ---------------------------------------------------------
    # 1. Choose CPU or GPU
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("REAL ECG TRAINING PIPELINE CHECK")
    print("=" * 70)

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 2. Build real train / validation datasets
    # ---------------------------------------------------------

    print("Building training dataset...")

    train_dataset = build_dataset_from_records(
        record_ids=TRAIN_RECORDS,
        data_dir=DATA_DIR,
    )

    print("Building validation dataset...")

    validation_dataset = build_dataset_from_records(
        record_ids=VALIDATION_RECORDS,
        data_dir=DATA_DIR,
    )

    # ---------------------------------------------------------
    # 3. DataLoaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
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
    # 5. Class-weighted loss
    # ---------------------------------------------------------

    class_weights = compute_class_weights(
        targets=train_dataset.targets,
        num_classes=len(CLASS_NAMES),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ---------------------------------------------------------
    # 6. Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------------------------------------
    # 7. Validation BEFORE training
    # ---------------------------------------------------------

    initial_validation_loss = evaluate_loss(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=device,
    )

    print("-" * 70)

    print(
        "Validation loss before training: "
        f"{initial_validation_loss:.6f}"
    )

    # ---------------------------------------------------------
    # 8. Train one real epoch
    # ---------------------------------------------------------

    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # ---------------------------------------------------------
    # 9. Validation AFTER training
    # ---------------------------------------------------------

    validation_loss = evaluate_loss(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"Training loss after epoch 1: "
        f"{train_loss:.6f}"
    )

    print(
        f"Validation loss after epoch 1: "
        f"{validation_loss:.6f}"
    )

    print("-" * 70)

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(validation_loader)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()