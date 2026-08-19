"""Multi-seed reproducibility study for the two frozen final training strategies.

This script trains exactly two configurations on the development split:

1. clean_rr
   Clean ECG + raw RR features.

2. noise_augmented_rr
   The same ECG + raw RR model, but each epoch contains exactly
   50% clean and 50% noisy ECG while keeping the same epoch size
   as the clean baseline.

No Final Test records are used.

Fixed protocol:
- patient-independent TRAIN / VALIDATION split
- ECGCNN1DWithRR
- sqrt-weighted cross entropy
- Adam, lr=1e-3
- batch size=256
- 15 epochs
- RR standardization fitted on CLEAN TRAIN only
- model selection by clean Validation Macro-F1
- seeds: 42, 123, 2026
"""

import csv
import gc
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

RESULTS_ROOT = Path(
    "results/reproducibility"
)

SUMMARY_PATH = Path(
    "results/tables/reproducibility_training_summary.csv"
)

SEEDS = (
    42,
    123,
    2026,
)

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15


def save_summary(rows: list[dict]) -> None:
    """Save current reproducibility results after every completed run."""

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open(
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


def build_development_datasets():
    """Build the clean and augmented development datasets once."""

    print("=" * 88)
    print("BUILDING DEVELOPMENT DATASETS")
    print("=" * 88)

    print("\nBuilding CLEAN TRAIN dataset...")

    clean_train = build_dataset_with_rr_from_records(
        record_ids=TRAIN_RECORDS,
        data_dir=DATA_DIR,
    )

    print(
        f"Clean TRAIN samples: {len(clean_train)}"
    )

    print("\nBuilding CLEAN VALIDATION dataset...")

    validation = build_dataset_with_rr_from_records(
        record_ids=VALIDATION_RECORDS,
        data_dir=DATA_DIR,
    )

    print(
        f"Validation samples: {len(validation)}"
    )

    # ---------------------------------------------------------
    # RR normalization is fitted once on CLEAN TRAIN only.
    # ---------------------------------------------------------

    rr_mean, rr_std = fit_rr_standardizer(
        clean_train.rr_features
    )

    print("\nTRAIN-only RR normalization:")
    print(f"  mean = {rr_mean.tolist()}")
    print(f"  std  = {rr_std.tolist()}")

    # ---------------------------------------------------------
    # Build the deterministic augmentation pool once.
    # ---------------------------------------------------------

    print("\nBuilding RQ4 augmented TRAIN pool...")

    augmented_train, augmentation_audit = (
        build_noise_augmented_training_dataset(
            data_dir=DATA_DIR,
            record_ids=TRAIN_RECORDS,
            return_audit=True,
        )
    )

    clean_count = int(
        augmentation_audit[
            "clean_count"
        ]
    )

    noisy_count = int(
        augmentation_audit[
            "noisy_count"
        ]
    )

    if clean_count != len(
        clean_train
    ):
        raise RuntimeError(
            "Augmented clean-count does not match "
            "the clean TRAIN dataset."
        )

    if clean_count != noisy_count:
        raise RuntimeError(
            "RQ4 augmentation must have one noisy "
            "counterpart per clean heartbeat."
        )

    print(
        f"Augmented pool: "
        f"{clean_count} clean + {noisy_count} noisy"
    )

    print(
        f"Samples used per augmented epoch: "
        f"{clean_count}"
    )

    # ---------------------------------------------------------
    # Apply exactly the same TRAIN-derived RR standardization.
    # ---------------------------------------------------------

    clean_train.rr_features = standardize_rr_features(
        rr_features=clean_train.rr_features,
        mean=rr_mean,
        std=rr_std,
    )

    validation.rr_features = standardize_rr_features(
        rr_features=validation.rr_features,
        mean=rr_mean,
        std=rr_std,
    )

    augmented_train.rr_features = standardize_rr_features(
        rr_features=augmented_train.rr_features,
        mean=rr_mean,
        std=rr_std,
    )

    # ---------------------------------------------------------
    # Same sqrt class weights for both configurations.
    # ---------------------------------------------------------

    class_weights = compute_sqrt_class_weights(
        targets=clean_train.targets,
        num_classes=len(CLASS_NAMES),
    )

    print("\nFrozen sqrt class weights:")

    for class_name, weight in zip(
        CLASS_NAMES,
        class_weights.tolist(),
    ):
        print(
            f"  {class_name}: {weight:.6f}"
        )

    return {
        "clean_train": clean_train,
        "augmented_train": augmented_train,
        "validation": validation,
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "class_weights": class_weights,
        "clean_count": clean_count,
        "noisy_count": noisy_count,
    }


def make_clean_train_loader(
    dataset,
    seed: int,
):
    """Clean baseline: all clean beats once per epoch, shuffled."""

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


def make_augmented_train_loader(
    dataset,
    clean_count: int,
    noisy_count: int,
    seed: int,
):
    """RQ4: same epoch size, exactly 50% clean and 50% noisy."""

    sampler = PairedCleanNoisyEpochSampler(
        clean_count=clean_count,
        noisy_count=noisy_count,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
    )

    return loader, sampler


def train_one_run(
    *,
    configuration: str,
    seed: int,
    datasets: dict,
    device: torch.device,
) -> dict:
    """Train one frozen configuration at one seed."""

    set_global_seed(
        seed
    )

    output_dir = (
        RESULTS_ROOT
        / configuration
        / f"seed_{seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "best_model.pt"
    )

    validation_loader = DataLoader(
        datasets[
            "validation"
        ],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    train_sampler = None

    if configuration == "clean_rr":

        train_loader = make_clean_train_loader(
            dataset=datasets[
                "clean_train"
            ],
            seed=seed,
        )

        training_mode = (
            "clean_ecg_raw_rr"
        )

    elif configuration == "noise_augmented_rr":

        (
            train_loader,
            train_sampler,
        ) = make_augmented_train_loader(
            dataset=datasets[
                "augmented_train"
            ],
            clean_count=datasets[
                "clean_count"
            ],
            noisy_count=datasets[
                "noisy_count"
            ],
            seed=seed,
        )

        training_mode = (
            "paired_50_50_noise_augmentation"
        )

    else:
        raise ValueError(
            f"Unknown configuration: {configuration}"
        )

    model = ECGCNN1DWithRR(
        num_classes=len(
            CLASS_NAMES
        )
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=datasets[
            "class_weights"
        ].to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = -1.0
    best_balanced_accuracy = -1.0
    best_epoch = -1

    print("\n" + "=" * 88)

    print(
        f"TRAINING: {configuration} | seed={seed}"
    )

    print("=" * 88)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        if train_sampler is not None:
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

        macro_f1 = float(
            metrics[
                "macro_f1"
            ]
        )

        balanced_accuracy = float(
            metrics[
                "balanced_accuracy"
            ]
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Macro-F1: {macro_f1:.4f} | "
            f"Bal Acc: {balanced_accuracy:.4f}"
        )

        if macro_f1 > best_macro_f1:

            best_macro_f1 = (
                macro_f1
            )

            best_balanced_accuracy = (
                balanced_accuracy
            )

            best_epoch = (
                epoch
            )

            checkpoint = {
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
                "seed": seed,
                "batch_size": BATCH_SIZE,
                "learning_rate": (
                    LEARNING_RATE
                ),
                "num_epochs": (
                    NUM_EPOCHS
                ),
                "class_names": (
                    CLASS_NAMES
                ),
                "loss_weighting": (
                    "sqrt_weighted"
                ),
                "rr_mean": (
                    datasets[
                        "rr_mean"
                    ].cpu()
                ),
                "rr_std": (
                    datasets[
                        "rr_std"
                    ].cpu()
                ),
                "training_mode": (
                    training_mode
                ),
            }

            if (
                configuration
                == "noise_augmented_rr"
            ):

                checkpoint.update(
                    {
                        "clean_train_count": (
                            datasets[
                                "clean_count"
                            ]
                        ),
                        "noisy_train_count": (
                            datasets[
                                "noisy_count"
                            ]
                        ),
                        "epoch_sample_count": (
                            datasets[
                                "clean_count"
                            ]
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
                    }
                )

            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print(
                "  -> Saved new best checkpoint."
            )

    print("-" * 88)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best Validation Macro-F1: "
        f"{best_macro_f1:.6f}"
    )

    print(
        f"Best Validation Balanced Accuracy: "
        f"{best_balanced_accuracy:.6f}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    return {
        "configuration": (
            configuration
        ),
        "seed": (
            seed
        ),
        "best_epoch": (
            best_epoch
        ),
        "validation_macro_f1": (
            best_macro_f1
        ),
        "validation_balanced_accuracy": (
            best_balanced_accuracy
        ),
        "checkpoint": str(
            checkpoint_path
        ),
    }


def main() -> None:

    print("=" * 88)
    print("MULTI-SEED REPRODUCIBILITY STUDY")
    print("=" * 88)

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        "Configurations: clean_rr, noise_augmented_rr"
    )

    print(
        "Final Test: NOT USED"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    datasets = (
        build_development_datasets()
    )

    rows = []

    for configuration in (
        "clean_rr",
        "noise_augmented_rr",
    ):

        for seed in SEEDS:

            result = train_one_run(
                configuration=(
                    configuration
                ),
                seed=seed,
                datasets=datasets,
                device=device,
            )

            rows.append(
                result
            )

            save_summary(
                rows
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print("REPRODUCIBILITY TRAINING COMPLETE")
    print("=" * 88)

    print(
        f"Summary saved to: {SUMMARY_PATH}"
    )

    # Human-readable aggregate
    for configuration in (
        "clean_rr",
        "noise_augmented_rr",
    ):

        values = [
            row[
                "validation_macro_f1"
            ]
            for row in rows
            if row[
                "configuration"
            ] == configuration
        ]

        tensor = torch.tensor(
            values,
            dtype=torch.float64,
        )

        mean = float(
            torch.mean(
                tensor
            )
        )

        # sample standard deviation for 3-seed reporting
        std = float(
            torch.std(
                tensor,
                unbiased=True,
            )
        )

        print(
            f"{configuration}: "
            f"Macro-F1 = {mean:.6f} ± {std:.6f}"
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
