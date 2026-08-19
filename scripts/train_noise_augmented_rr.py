"""Train the ECG + RR model with controlled noise augmentation for RQ4."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.noise_augmented_dataset import (
    TRAIN_NOISE_TYPES,
    TRAIN_SNR_LEVELS_DB,
    UNSEEN_SNR_LEVELS_DB,
    build_noise_augmented_training_dataset,
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
from src.evaluation.rr_evaluator import (
    evaluate_rr_model,
)
from src.models.cnn1d_rr import (
    ECGCNN1DWithRR,
)
from src.training.paired_clean_noisy_sampler import (
    PairedCleanNoisyEpochSampler,
)
from src.training.reproducibility import (
    DEFAULT_SEED,
    set_global_seed,
)
from src.training.rr_engine import (
    evaluate_rr_loss,
    train_rr_one_epoch,
)
from src.training.weights import (
    compute_class_weights,
    compute_sqrt_class_weights,
)


DATA_DIR = Path("data/raw/mitdb")
RESULTS_ROOT = Path(
    "results/noise_augmented_rr"
)

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the ECG + RR model with controlled "
            "50/50 clean-noisy augmentation."
        )
    )

    parser.add_argument(
        "--loss-weighting",
        choices=(
            "weighted",
            "sqrt_weighted",
            "unweighted",
        ),
        default="sqrt_weighted",
        help="Choose cross-entropy loss weighting.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loss_weighting = args.loss_weighting

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
    print("RQ4 NOISE-AUGMENTED ECG + RR TRAINING")
    print("=" * 72)

    print(f"Device: {device}")
    print(f"Seed: {DEFAULT_SEED}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Loss weighting: {loss_weighting}")
    print(f"Train noise types: {TRAIN_NOISE_TYPES}")
    print(f"Train SNRs: {TRAIN_SNR_LEVELS_DB}")
    print(f"Reserved unseen SNRs: {UNSEEN_SNR_LEVELS_DB}")

    # ---------------------------------------------------------
    # 2. Build datasets
    # ---------------------------------------------------------

    print(
        "\nBuilding clean TRAIN reference dataset..."
    )

    # We keep a clean TRAIN reference so RR normalization and
    # class weights exactly follow the clean-baseline protocol.
    clean_train_reference = (
        build_dataset_with_rr_from_records(
            record_ids=TRAIN_RECORDS,
            data_dir=DATA_DIR,
        )
    )

    print(
        "Building RQ4 augmented TRAIN pool..."
    )

    train_dataset, augmentation_audit = (
        build_noise_augmented_training_dataset(
            data_dir=DATA_DIR,
            record_ids=TRAIN_RECORDS,
            return_audit=True,
        )
    )

    clean_train_count = int(
        augmentation_audit[
            "clean_count"
        ]
    )

    noisy_train_count = int(
        augmentation_audit[
            "noisy_count"
        ]
    )

    print(
        "\nRQ4 training pool:"
    )

    print(
        f"  Clean pool: {clean_train_count}"
    )

    print(
        f"  Noisy pool: {noisy_train_count}"
    )

    print(
        f"  Total pool: {len(train_dataset)}"
    )

    print(
        f"  Samples used per epoch: {clean_train_count}"
    )

    print(
        "  Per-epoch composition: 50% clean / 50% noisy"
    )

    print(
        "Building clean VALIDATION dataset..."
    )

    validation_dataset = (
        build_dataset_with_rr_from_records(
            record_ids=VALIDATION_RECORDS,
            data_dir=DATA_DIR,
        )
    )

    print(
        f"Clean TRAIN reference samples: "
        f"{len(clean_train_reference)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    # ---------------------------------------------------------
    # 3. Fit RR normalization on CLEAN TRAIN ONLY
    # ---------------------------------------------------------

    rr_mean, rr_std = fit_rr_standardizer(
        clean_train_reference.rr_features
    )

    print("\nTRAIN-only RR normalization statistics:")

    print(
        "RR mean: "
        f"{rr_mean.tolist()}"
    )

    print(
        "RR std: "
        f"{rr_std.tolist()}"
    )

    train_dataset.rr_features = (
        standardize_rr_features(
            rr_features=(
                train_dataset.rr_features
            ),
            mean=rr_mean,
            std=rr_std,
        )
    )

    validation_dataset.rr_features = (
        standardize_rr_features(
            rr_features=(
                validation_dataset.rr_features
            ),
            mean=rr_mean,
            std=rr_std,
        )
    )

    # ---------------------------------------------------------
    # 4. DataLoaders
    # ---------------------------------------------------------

    train_sampler = (
        PairedCleanNoisyEpochSampler(
            clean_count=clean_train_count,
            noisy_count=noisy_train_count,
            seed=DEFAULT_SEED,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # 5. Model
    # ---------------------------------------------------------

    model = ECGCNN1DWithRR(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    # ---------------------------------------------------------
    # 6. Loss
    # ---------------------------------------------------------
    # Use CLEAN TRAIN labels so class-weight computation exactly
    # matches the clean baseline protocol.

    if loss_weighting == "weighted":

        class_weights = compute_class_weights(
            targets=(
                clean_train_reference.targets
            ),
            num_classes=len(CLASS_NAMES),
        ).to(device)

    elif loss_weighting == "sqrt_weighted":

        class_weights = (
            compute_sqrt_class_weights(
                targets=(
                    clean_train_reference.targets
                ),
                num_classes=len(CLASS_NAMES),
            ).to(device)
        )

    else:
        class_weights = None

    if class_weights is not None:

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        print("\nClass weights:")

        for class_name, weight in zip(
            CLASS_NAMES,
            class_weights.tolist(),
        ):
            print(
                f"  {class_name}: "
                f"{weight:.6f}"
            )

    else:

        criterion = nn.CrossEntropyLoss()

        print(
            "\nUsing unweighted CrossEntropyLoss."
        )

    # ---------------------------------------------------------
    # 7. Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------------------------------------
    # 8. Training
    # ---------------------------------------------------------

    best_macro_f1 = -1.0
    best_epoch = -1

    print("\nStarting RQ4 training...")
    print("-" * 72)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        # Reproducibly choose a new 50/50 clean/noisy subset
        # while keeping exactly one version of every underlying
        # training heartbeat in each epoch.
        train_sampler.set_epoch(
            epoch
        )

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

                    "rr_mean": (
                        rr_mean.cpu()
                    ),

                    "rr_std": (
                        rr_std.cpu()
                    ),

                    # RQ4 reproducibility metadata.
                    "training_mode": (
                        "paired_50_50_noise_augmentation"
                    ),

                    "clean_train_count": (
                        clean_train_count
                    ),

                    "noisy_train_count": (
                        noisy_train_count
                    ),

                    "epoch_sample_count": (
                        clean_train_count
                    ),

                    "train_noise_types": (
                        TRAIN_NOISE_TYPES
                    ),

                    "train_snr_levels_db": (
                        TRAIN_SNR_LEVELS_DB
                    ),

                    "unseen_snr_levels_db": (
                        UNSEEN_SNR_LEVELS_DB
                    ),
                },
                checkpoint_path,
            )

            print(
                "  -> Saved new best RQ4 checkpoint."
            )

    # ---------------------------------------------------------
    # 9. Final report
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
