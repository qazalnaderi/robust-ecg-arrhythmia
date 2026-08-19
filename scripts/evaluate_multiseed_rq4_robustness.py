"""Multi-seed robustness evaluation for the two final training strategies.

This script evaluates all reproducibility checkpoints on the SAME
patient-independent Validation split and the SAME NSTDB noise conditions.

Configurations:
1. clean_rr
2. noise_augmented_rr

Seeds:
42, 123, 2026

Noise conditions:
bw / ma / em at 18, 12, 6, 0, -6 dB

Seen by the noise-augmented model during training:
18, 6, -6 dB

Reserved unseen intensities:
12, 0 dB

Final Test is NOT used.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.rr_normalization import standardize_rr_features
from src.data.splits import VALIDATION_RECORDS
from src.data.torch_dataset import (
    CLASS_NAMES,
    ECGRRHeartbeatDataset,
    build_dataset_with_rr_from_records,
)
from src.evaluation.rr_evaluator import evaluate_rr_model
from src.models.cnn1d_rr import ECGCNN1DWithRR
from src.noise.heartbeat_pipeline import build_noisy_heartbeats
from src.noise.nstdb import VALID_NOISE_TYPES


DATA_DIR = Path("data/raw/mitdb")

RESULTS_ROOT = Path(
    "results/reproducibility"
)

RAW_OUTPUT = Path(
    "results/tables/rq4_multiseed_robustness_raw.csv"
)

SUMMARY_OUTPUT = Path(
    "results/tables/rq4_multiseed_robustness_summary.csv"
)

PER_CLASS_RAW_OUTPUT = Path(
    "results/tables/rq4_multiseed_per_class_raw.csv"
)

PER_CLASS_SUMMARY_OUTPUT = Path(
    "results/tables/rq4_multiseed_per_class_summary.csv"
)

SEEDS = (
    42,
    123,
    2026,
)

CONFIGURATIONS = (
    "clean_rr",
    "noise_augmented_rr",
)

SNR_LEVELS_DB = (
    18.0,
    12.0,
    6.0,
    0.0,
    -6.0,
)

SEEN_SNRS = {
    18.0,
    6.0,
    -6.0,
}

UNSEEN_SNRS = {
    12.0,
    0.0,
}

BATCH_SIZE = 256


def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Save a list of dictionaries to CSV."""

    if not rows:
        raise RuntimeError(
            f"No rows available for {path}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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


def checkpoint_path(
    configuration: str,
    seed: int,
) -> Path:
    """Return one reproducibility checkpoint path."""

    return (
        RESULTS_ROOT
        / configuration
        / f"seed_{seed}"
        / "best_model.pt"
    )


def load_runs(
    device: torch.device,
) -> list[dict]:
    """Load all six frozen reproducibility checkpoints."""

    runs = []

    print("\nLoading reproducibility checkpoints...")

    for configuration in CONFIGURATIONS:

        for seed in SEEDS:

            path = checkpoint_path(
                configuration,
                seed,
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint: {path}"
                )

            checkpoint = torch.load(
                path,
                map_location=device,
                weights_only=False,
            )

            stored_seed = int(
                checkpoint[
                    "seed"
                ]
            )

            if stored_seed != seed:
                raise RuntimeError(
                    f"Seed mismatch in {path}: "
                    f"expected {seed}, found {stored_seed}"
                )

            model = ECGCNN1DWithRR(
                num_classes=len(
                    CLASS_NAMES
                )
            ).to(device)

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            model.eval()

            runs.append(
                {
                    "configuration": (
                        configuration
                    ),
                    "seed": (
                        seed
                    ),
                    "checkpoint_path": (
                        path
                    ),
                    "checkpoint": (
                        checkpoint
                    ),
                    "model": (
                        model
                    ),
                }
            )

            print(
                f"  Loaded {configuration} "
                f"seed={seed}"
            )

    return runs


def build_rr_reference_cache() -> dict:
    """Build raw RR and label references once for Validation."""

    cache = {}

    print("\nBuilding Validation RR reference cache...")

    for record_id in VALIDATION_RECORDS:

        dataset = build_dataset_with_rr_from_records(
            record_ids=(record_id,),
            data_dir=DATA_DIR,
        )

        labels = np.asarray(
            [
                CLASS_NAMES[index]
                for index in dataset.targets.tolist()
            ]
        )

        rr_features = (
            dataset.rr_features
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        cache[
            record_id
        ] = {
            "labels": labels,
            "rr_features": rr_features,
        }

        print(
            f"  {record_id}: "
            f"{len(labels)} beats"
        )

    return cache


def build_noisy_condition_arrays(
    *,
    noise_type: str,
    target_snr_db: float,
    rr_cache: dict,
):
    """Build one noisy Validation condition once, shared by all six models."""

    all_beats = []
    all_rr = []
    all_labels = []

    for record_id in VALIDATION_RECORDS:

        (
            noisy_beats,
            noisy_labels,
            heartbeat_metadata,
            corruption_metadata,
        ) = build_noisy_heartbeats(
            record_path=(
                DATA_DIR / record_id
            ),
            noise_type=noise_type,
            target_snr_db=target_snr_db,
        )

        noisy_beats = np.asarray(
            noisy_beats,
            dtype=np.float32,
        )

        noisy_labels = np.asarray(
            noisy_labels
        )

        reference = rr_cache[
            record_id
        ]

        reference_rr = reference[
            "rr_features"
        ]

        reference_labels = reference[
            "labels"
        ]

        if len(noisy_beats) != len(
            reference_rr
        ):
            raise RuntimeError(
                "ECG/RR count mismatch for "
                f"{record_id}, "
                f"{noise_type}@{target_snr_db:g} dB."
            )

        if not np.array_equal(
            noisy_labels,
            reference_labels,
        ):
            raise RuntimeError(
                "Label alignment changed for "
                f"{record_id}, "
                f"{noise_type}@{target_snr_db:g} dB."
            )

        if len(
            heartbeat_metadata
        ) != len(
            noisy_beats
        ):
            raise RuntimeError(
                "Heartbeat metadata mismatch for "
                f"{record_id}."
            )

        achieved_snr = float(
            corruption_metadata[
                "achieved_snr_db"
            ]
        )

        if not np.isclose(
            achieved_snr,
            target_snr_db,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Achieved SNR mismatch for "
                f"{record_id}, "
                f"{noise_type}@{target_snr_db:g} dB: "
                f"{achieved_snr:.6f}"
            )

        if not np.isfinite(
            noisy_beats
        ).all():
            raise RuntimeError(
                "Non-finite ECG values found."
            )

        all_beats.append(
            noisy_beats
        )

        all_rr.append(
            reference_rr
        )

        all_labels.append(
            noisy_labels
        )

    return (
        np.concatenate(
            all_beats,
            axis=0,
        ),
        np.concatenate(
            all_rr,
            axis=0,
        ),
        np.concatenate(
            all_labels,
            axis=0,
        ),
    )


def make_dataset(
    *,
    heartbeats: np.ndarray,
    rr_features: np.ndarray,
    labels: np.ndarray,
    rr_mean: torch.Tensor,
    rr_std: torch.Tensor,
) -> ECGRRHeartbeatDataset:
    """Build one fresh dataset so RR normalization cannot leak across runs."""

    dataset = ECGRRHeartbeatDataset(
        heartbeats=heartbeats,
        rr_features=rr_features.copy(),
        labels=labels.copy(),
    )

    dataset.rr_features = standardize_rr_features(
        rr_features=dataset.rr_features,
        mean=rr_mean.cpu(),
        std=rr_std.cpu(),
    )

    return dataset


def evaluate_dataset(
    *,
    model,
    dataset,
    device,
) -> dict:
    """Evaluate one model on one frozen dataset."""

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return evaluate_rr_model(
        model=model,
        dataloader=loader,
        device=device,
    )


def exposure_label(
    target_snr_db,
) -> str:
    """Classify one condition as clean, seen-SNR, or unseen-SNR."""

    if target_snr_db is None:
        return "clean"

    snr = float(
        target_snr_db
    )

    if snr in SEEN_SNRS:
        return "seen_snr"

    if snr in UNSEEN_SNRS:
        return "unseen_snr"

    raise RuntimeError(
        f"Unexpected SNR level: {snr}"
    )


def append_result(
    *,
    raw_rows: list[dict],
    per_class_rows: list[dict],
    configuration: str,
    seed: int,
    noise_type: str,
    target_snr_db,
    metrics: dict,
) -> None:
    """Append overall and class-wise results for one evaluation."""

    exposure = exposure_label(
        target_snr_db
    )

    condition = (
        "clean"
        if target_snr_db is None
        else f"{noise_type}_{float(target_snr_db):g}db"
    )

    raw_rows.append(
        {
            "configuration": (
                configuration
            ),
            "seed": (
                seed
            ),
            "condition": (
                condition
            ),
            "noise_type": (
                noise_type
            ),
            "target_snr_db": (
                ""
                if target_snr_db is None
                else float(
                    target_snr_db
                )
            ),
            "exposure": (
                exposure
            ),
            "macro_f1": float(
                metrics[
                    "macro_f1"
                ]
            ),
            "balanced_accuracy": float(
                metrics[
                    "balanced_accuracy"
                ]
            ),
        }
    )

    for class_name in CLASS_NAMES:

        values = metrics[
            "per_class"
        ][
            class_name
        ]

        per_class_rows.append(
            {
                "configuration": (
                    configuration
                ),
                "seed": (
                    seed
                ),
                "condition": (
                    condition
                ),
                "noise_type": (
                    noise_type
                ),
                "target_snr_db": (
                    ""
                    if target_snr_db is None
                    else float(
                        target_snr_db
                    )
                ),
                "exposure": (
                    exposure
                ),
                "class_name": (
                    class_name
                ),
                "precision": float(
                    values[
                        "precision"
                    ]
                ),
                "recall": float(
                    values[
                        "recall"
                    ]
                ),
                "f1": float(
                    values[
                        "f1"
                    ]
                ),
                "support": int(
                    values[
                        "support"
                    ]
                ),
            }
        )


def evaluate_clean(
    *,
    runs: list[dict],
    device: torch.device,
    raw_rows: list[dict],
    per_class_rows: list[dict],
) -> None:
    """Evaluate clean Validation and verify every checkpoint."""

    print("\n" + "=" * 88)
    print("CLEAN VALIDATION")
    print("=" * 88)

    for run in runs:

        # Fresh clean dataset for every checkpoint because
        # RR standardization is applied in-place.
        dataset = build_dataset_with_rr_from_records(
            record_ids=VALIDATION_RECORDS,
            data_dir=DATA_DIR,
        )

        dataset.rr_features = standardize_rr_features(
            rr_features=dataset.rr_features,
            mean=run[
                "checkpoint"
            ][
                "rr_mean"
            ].cpu(),
            std=run[
                "checkpoint"
            ][
                "rr_std"
            ].cpu(),
        )

        metrics = evaluate_dataset(
            model=run[
                "model"
            ],
            dataset=dataset,
            device=device,
        )

        stored_macro_f1 = float(
            run[
                "checkpoint"
            ][
                "validation_macro_f1"
            ]
        )

        if not np.isclose(
            metrics[
                "macro_f1"
            ],
            stored_macro_f1,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Clean checkpoint verification failed for "
                f"{run['configuration']} seed={run['seed']}: "
                f"current={metrics['macro_f1']:.6f}, "
                f"stored={stored_macro_f1:.6f}"
            )

        append_result(
            raw_rows=raw_rows,
            per_class_rows=per_class_rows,
            configuration=run[
                "configuration"
            ],
            seed=run[
                "seed"
            ],
            noise_type="clean",
            target_snr_db=None,
            metrics=metrics,
        )

        print(
            f"{run['configuration']:<20} "
            f"seed={run['seed']:<4} "
            f"Macro-F1={metrics['macro_f1']:.6f} "
            f"| checkpoint PASS"
        )


def build_overall_summary(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate by condition and compute paired seed differences."""

    summary_rows = []

    condition_columns = [
        "condition",
        "noise_type",
        "target_snr_db",
        "exposure",
    ]

    unique_conditions = (
        raw_df[
            condition_columns
        ]
        .drop_duplicates()
    )

    for _, condition_row in unique_conditions.iterrows():

        condition = condition_row[
            "condition"
        ]

        condition_df = raw_df[
            raw_df[
                "condition"
            ]
            == condition
        ]

        clean_df = (
            condition_df[
                condition_df[
                    "configuration"
                ]
                == "clean_rr"
            ]
            .sort_values(
                "seed"
            )
        )

        augmented_df = (
            condition_df[
                condition_df[
                    "configuration"
                ]
                == "noise_augmented_rr"
            ]
            .sort_values(
                "seed"
            )
        )

        if clean_df[
            "seed"
        ].tolist() != list(
            SEEDS
        ):
            raise RuntimeError(
                f"Missing clean seeds for {condition}."
            )

        if augmented_df[
            "seed"
        ].tolist() != list(
            SEEDS
        ):
            raise RuntimeError(
                f"Missing augmented seeds for {condition}."
            )

        clean_values = clean_df[
            "macro_f1"
        ].to_numpy(
            dtype=float
        )

        augmented_values = augmented_df[
            "macro_f1"
        ].to_numpy(
            dtype=float
        )

        paired_difference = (
            augmented_values
            - clean_values
        )

        summary_rows.append(
            {
                "condition": (
                    condition
                ),
                "noise_type": (
                    condition_row[
                        "noise_type"
                    ]
                ),
                "target_snr_db": (
                    condition_row[
                        "target_snr_db"
                    ]
                ),
                "exposure": (
                    condition_row[
                        "exposure"
                    ]
                ),
                "clean_macro_f1_mean": float(
                    np.mean(
                        clean_values
                    )
                ),
                "clean_macro_f1_std": float(
                    np.std(
                        clean_values,
                        ddof=1,
                    )
                ),
                "noise_aug_macro_f1_mean": float(
                    np.mean(
                        augmented_values
                    )
                ),
                "noise_aug_macro_f1_std": float(
                    np.std(
                        augmented_values,
                        ddof=1,
                    )
                ),
                "paired_difference_mean": float(
                    np.mean(
                        paired_difference
                    )
                ),
                "paired_difference_std": float(
                    np.std(
                        paired_difference,
                        ddof=1,
                    )
                ),
                "noise_aug_better_seeds": int(
                    np.sum(
                        paired_difference
                        > 0.0
                    )
                ),
                "total_seeds": len(
                    SEEDS
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


def build_per_class_summary(
    per_class_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate class-wise F1 and paired seed differences."""

    summary_rows = []

    keys = (
        per_class_df[
            [
                "condition",
                "noise_type",
                "target_snr_db",
                "exposure",
                "class_name",
            ]
        ]
        .drop_duplicates()
    )

    for _, row in keys.iterrows():

        subset = per_class_df[
            (
                per_class_df[
                    "condition"
                ]
                == row[
                    "condition"
                ]
            )
            &
            (
                per_class_df[
                    "class_name"
                ]
                == row[
                    "class_name"
                ]
            )
        ]

        clean_df = (
            subset[
                subset[
                    "configuration"
                ]
                == "clean_rr"
            ]
            .sort_values(
                "seed"
            )
        )

        augmented_df = (
            subset[
                subset[
                    "configuration"
                ]
                == "noise_augmented_rr"
            ]
            .sort_values(
                "seed"
            )
        )

        clean_values = clean_df[
            "f1"
        ].to_numpy(
            dtype=float
        )

        augmented_values = augmented_df[
            "f1"
        ].to_numpy(
            dtype=float
        )

        if len(clean_values) != len(
            SEEDS
        ) or len(augmented_values) != len(
            SEEDS
        ):
            raise RuntimeError(
                "Missing class-wise seed results for "
                f"{row['condition']} / {row['class_name']}."
            )

        paired_difference = (
            augmented_values
            - clean_values
        )

        summary_rows.append(
            {
                "condition": (
                    row[
                        "condition"
                    ]
                ),
                "noise_type": (
                    row[
                        "noise_type"
                    ]
                ),
                "target_snr_db": (
                    row[
                        "target_snr_db"
                    ]
                ),
                "exposure": (
                    row[
                        "exposure"
                    ]
                ),
                "class_name": (
                    row[
                        "class_name"
                    ]
                ),
                "clean_f1_mean": float(
                    np.mean(
                        clean_values
                    )
                ),
                "clean_f1_std": float(
                    np.std(
                        clean_values,
                        ddof=1,
                    )
                ),
                "noise_aug_f1_mean": float(
                    np.mean(
                        augmented_values
                    )
                ),
                "noise_aug_f1_std": float(
                    np.std(
                        augmented_values,
                        ddof=1,
                    )
                ),
                "paired_difference_mean": float(
                    np.mean(
                        paired_difference
                    )
                ),
                "paired_difference_std": float(
                    np.std(
                        paired_difference,
                        ddof=1,
                    )
                ),
                "noise_aug_better_seeds": int(
                    np.sum(
                        paired_difference
                        > 0.0
                    )
                ),
                "total_seeds": len(
                    SEEDS
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


def print_exposure_summary(
    summary_df: pd.DataFrame,
) -> None:
    """Print simple human-readable RQ4 conclusions."""

    print("\n" + "=" * 88)
    print("MULTI-SEED RQ4 ROBUSTNESS SUMMARY")
    print("=" * 88)

    clean_row = summary_df[
        summary_df[
            "exposure"
        ]
        == "clean"
    ].iloc[0]

    print(
        "Clean:"
    )

    print(
        f"  Clean-trained: "
        f"{clean_row['clean_macro_f1_mean']:.6f} "
        f"± {clean_row['clean_macro_f1_std']:.6f}"
    )

    print(
        f"  Noise-augmented: "
        f"{clean_row['noise_aug_macro_f1_mean']:.6f} "
        f"± {clean_row['noise_aug_macro_f1_std']:.6f}"
    )

    print(
        f"  Paired difference: "
        f"{clean_row['paired_difference_mean']:+.6f}"
    )

    for exposure in (
        "seen_snr",
        "unseen_snr",
    ):

        subset = summary_df[
            summary_df[
                "exposure"
            ]
            == exposure
        ]

        mean_clean = float(
            subset[
                "clean_macro_f1_mean"
            ].mean()
        )

        mean_augmented = float(
            subset[
                "noise_aug_macro_f1_mean"
            ].mean()
        )

        mean_difference = float(
            subset[
                "paired_difference_mean"
            ].mean()
        )

        conditions_positive_all_seeds = int(
            (
                subset[
                    "noise_aug_better_seeds"
                ]
                == len(
                    SEEDS
                )
            ).sum()
        )

        conditions_positive_mean = int(
            (
                subset[
                    "paired_difference_mean"
                ]
                > 0.0
            ).sum()
        )

        print(
            f"\n{exposure}:"
        )

        print(
            f"  Mean clean-trained Macro-F1 "
            f"across conditions: {mean_clean:.6f}"
        )

        print(
            f"  Mean noise-augmented Macro-F1 "
            f"across conditions: {mean_augmented:.6f}"
        )

        print(
            f"  Mean paired difference: "
            f"{mean_difference:+.6f}"
        )

        print(
            f"  Conditions with positive mean: "
            f"{conditions_positive_mean}/{len(subset)}"
        )

        print(
            f"  Conditions improved in all 3 seeds: "
            f"{conditions_positive_all_seeds}/{len(subset)}"
        )

    print("=" * 88)


def main() -> None:

    print("=" * 88)
    print("MULTI-SEED RQ4 ROBUSTNESS EVALUATION")
    print("=" * 88)

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        "Configurations: clean_rr vs noise_augmented_rr"
    )

    print(
        "Split: Validation only"
    )

    print(
        "Final Test: NOT USED"
    )

    if SEEN_SNRS & UNSEEN_SNRS:
        raise RuntimeError(
            "Seen and unseen SNR sets overlap."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    runs = load_runs(
        device
    )

    rr_cache = build_rr_reference_cache()

    raw_rows = []
    per_class_rows = []

    # ---------------------------------------------------------
    # Clean condition
    # ---------------------------------------------------------

    evaluate_clean(
        runs=runs,
        device=device,
        raw_rows=raw_rows,
        per_class_rows=per_class_rows,
    )

    # ---------------------------------------------------------
    # All noisy conditions
    # ---------------------------------------------------------

    total_conditions = (
        len(VALID_NOISE_TYPES)
        * len(SNR_LEVELS_DB)
    )

    condition_index = 0

    for noise_type in VALID_NOISE_TYPES:

        for target_snr_db in SNR_LEVELS_DB:

            condition_index += 1

            print("\n" + "=" * 88)

            print(
                f"Noise condition "
                f"{condition_index}/{total_conditions}: "
                f"{noise_type} @ "
                f"{target_snr_db:g} dB "
                f"[{exposure_label(target_snr_db)}]"
            )

            print("=" * 88)

            (
                heartbeats,
                rr_features,
                labels,
            ) = build_noisy_condition_arrays(
                noise_type=noise_type,
                target_snr_db=target_snr_db,
                rr_cache=rr_cache,
            )

            for run in runs:

                dataset = make_dataset(
                    heartbeats=heartbeats,
                    rr_features=rr_features,
                    labels=labels,
                    rr_mean=run[
                        "checkpoint"
                    ][
                        "rr_mean"
                    ],
                    rr_std=run[
                        "checkpoint"
                    ][
                        "rr_std"
                    ],
                )

                metrics = evaluate_dataset(
                    model=run[
                        "model"
                    ],
                    dataset=dataset,
                    device=device,
                )

                append_result(
                    raw_rows=raw_rows,
                    per_class_rows=per_class_rows,
                    configuration=run[
                        "configuration"
                    ],
                    seed=run[
                        "seed"
                    ],
                    noise_type=noise_type,
                    target_snr_db=target_snr_db,
                    metrics=metrics,
                )

                print(
                    f"  {run['configuration']:<20} "
                    f"seed={run['seed']:<4} "
                    f"Macro-F1="
                    f"{metrics['macro_f1']:.6f}"
                )

    # ---------------------------------------------------------
    # Save raw results first
    # ---------------------------------------------------------

    save_csv(
        RAW_OUTPUT,
        raw_rows,
    )

    save_csv(
        PER_CLASS_RAW_OUTPUT,
        per_class_rows,
    )

    # ---------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------

    raw_df = pd.DataFrame(
        raw_rows
    )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    summary_df = build_overall_summary(
        raw_df
    )

    per_class_summary_df = (
        build_per_class_summary(
            per_class_df
        )
    )

    SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    per_class_summary_df.to_csv(
        PER_CLASS_SUMMARY_OUTPUT,
        index=False,
    )

    print_exposure_summary(
        summary_df
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {RAW_OUTPUT}"
    )

    print(
        f"  {SUMMARY_OUTPUT}"
    )

    print(
        f"  {PER_CLASS_RAW_OUTPUT}"
    )

    print(
        f"  {PER_CLASS_SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()
