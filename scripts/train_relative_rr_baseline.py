"""Train the clean ECG + relative-RR baseline model."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.relative_rr import (
    RELATIVE_RR_FEATURE_NAMES,
    make_relative_rr_features,
)
from src.data.rr_normalization import (
    fit_rr_standardizer,
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
from src.evaluation.rr_evaluator import evaluate_rr_model
from src.models.cnn1d_rr import ECGCNN1DWithRR
from src.training.reproducibility import (
    DEFAULT_SEED,
    set_global_seed,
)
from src.training.rr_engine import (
    evaluate_rr_loss,
    train_rr_one_epoch,
)
from src.training.weights import (
    compute_sqrt_class_weights,
)


DATA_DIR = Path("data/raw/mitdb")

OUTPUT_DIR = Path(
    "results/clean_baseline_relative_rr/sqrt_weighted"
)

CHECKPOINT_PATH = (
    OUTPUT_DIR / "best_model.pt"
)

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15


def convert_dataset_to_relative_rr(
    dataset,
) -> None:
    """Replace raw RR features with relative RR features."""

    raw_rr = (
        dataset.rr_features
        .cpu()
        .numpy()
    )

    relative_rr = make_relative_rr_features(
        raw_rr
    )

    dataset.rr_features = torch.from_numpy(
        relative_rr
    )


def main() -> None:

    # ---------------------------------------------------------
    # 1. Reproducibility
    # ---------------------------------------------------------

    set_global_seed(DEFAULT_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("CLEAN ECG + RELATIVE RR BASELINE TRAINING")
    print("=" * 72)

    print(f"Device: {device}")
    print(f"Seed: {DEFAULT_SEED}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("Loss weighting: sqrt_weighted")
    print(
        f"RR representation: "
        f"{RELATIVE_RR_FEATURE_NAMES}"
    )

    # ---------------------------------------------------------
    # 2. Build exactly the same train/validation split
    # ---------------------------------------------------------

    print("\nBuilding training dataset...")

    train_dataset = build_dataset_with_rr_from_records(
        record_ids=TRAIN_RECORDS,
        data_dir=DATA_DIR,
    )

    print("Building validation dataset...")

    validation_dataset = build_dataset_with_rr_from_records(
        record_ids=VALIDATION_RECORDS,
        data_dir=DATA_DIR,
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    # ---------------------------------------------------------
    # 3. Raw RR -> Relative RR
    # ---------------------------------------------------------

    convert_dataset_to_relative_rr(
        train_dataset
    )

    convert_dataset_to_relative_rr(
        validation_dataset
    )

    # ---------------------------------------------------------
    # 4. Fit normalization ONLY on TRAIN relative RR
    # ---------------------------------------------------------

    rr_mean, rr_std = fit_rr_standardizer(
        train_dataset.rr_features
    )

    print("\nTrain relative-RR normalization statistics:")

    for index, feature_name in enumerate(
        RELATIVE_RR_FEATURE_NAMES
    ):
        print(
            f"{feature_name}: "
            f"mean={rr_mean[index].item():.6f}, "
            f"std={rr_std[index].item():.6f}"
        )

    train_dataset.rr_features = (
        standardize_rr_features(
            rr_features=train_dataset.rr_features,
            mean=rr_mean,
            std=rr_std,
        )
    )

    validation_dataset.rr_features = (
        standardize_rr_features(
            rr_features=validation_dataset.rr_features,
            mean=rr_mean,
            std=rr_std,
        )
    )

    # ---------------------------------------------------------
    # 5. DataLoaders
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
    # 6. EXACT SAME ECG + RR MODEL
    # ---------------------------------------------------------

    model = ECGCNN1DWithRR(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    # ---------------------------------------------------------
    # 7. EXACT SAME sqrt weighting
    # ---------------------------------------------------------

    class_weights = compute_sqrt_class_weights(
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

    # ---------------------------------------------------------
    # 8. EXACT SAME optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------------------------------------
    # 9. Training
    # ---------------------------------------------------------

    best_macro_f1 = -1.0
    best_epoch = -1

    print("\nStarting training...")
    print("-" * 72)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        train_loss = train_rr_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = evaluate_rr_loss(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        metrics = evaluate_rr_model(
            model=model,
            dataloader=validation_loader,
            device=device,
        )

        macro_f1 = metrics["macro_f1"]

        balanced_accuracy = (
            metrics["balanced_accuracy"]
        )

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
                        "sqrt_weighted"
                    ),

                    "rr_representation": (
                        "relative"
                    ),

                    "rr_feature_names": (
                        RELATIVE_RR_FEATURE_NAMES
                    ),

                    # Fitted ONLY on training relative RR.
                    "rr_mean": rr_mean.cpu(),
                    "rr_std": rr_std.cpu(),
                },
                CHECKPOINT_PATH,
            )

            print(
                "  -> Saved new best checkpoint."
            )

    print("-" * 72)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        "Best validation Macro-F1: "
        f"{best_macro_f1:.6f}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()