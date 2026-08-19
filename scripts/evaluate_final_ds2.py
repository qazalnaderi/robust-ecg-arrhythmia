"""Final held-out DS2 evaluation — corrected v2.

This script evaluates the six already-frozen reproducibility checkpoints on:
- clean DS2
- bw / ma / em at 18, 12, 6, 0, -6 dB

IMPORTANT:
- No training happens here.
- No model selection happens here.
- No tuning is allowed after viewing Final Test results.
- Clean evaluation uses the dataset object directly, exactly like the
  already-successful multi-seed Validation evaluator. It never accesses
  a nonexistent `dataset.heartbeats` attribute.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.rr_normalization import standardize_rr_features
from src.data.splits import (
    DS2_RECORDS,
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
    SANITY_RECORDS,
    PACED_EXCLUDED_RECORDS,
)
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
RESULTS_ROOT = Path("results/reproducibility")

RAW_OUTPUT = Path(
    "results/final_test/final_multiseed_raw.csv"
)
SUMMARY_OUTPUT = Path(
    "results/final_test/final_multiseed_summary.csv"
)
PER_CLASS_RAW_OUTPUT = Path(
    "results/final_test/final_multiseed_per_class_raw.csv"
)
PER_CLASS_SUMMARY_OUTPUT = Path(
    "results/final_test/final_multiseed_per_class_summary.csv"
)

SEEDS = (42, 123, 2026)

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


def verify_final_split() -> None:
    final_set = set(
        DS2_RECORDS
    )

    forbidden = (
        set(TRAIN_RECORDS)
        | set(VALIDATION_RECORDS)
        | set(SANITY_RECORDS)
        | set(PACED_EXCLUDED_RECORDS)
    )

    overlap = (
        final_set
        & forbidden
    )

    if overlap:
        raise RuntimeError(
            "Final Test overlaps development records: "
            f"{sorted(overlap)}"
        )

    if len(
        DS2_RECORDS
    ) != 20:
        raise RuntimeError(
            "Expected 20 Final Test records, "
            f"found {len(DS2_RECORDS)}."
        )

    print("Final split integrity: PASS")
    print(
        f"Final Test records: {DS2_RECORDS}"
    )


def checkpoint_path(
    configuration: str,
    seed: int,
) -> Path:
    return (
        RESULTS_ROOT
        / configuration
        / f"seed_{seed}"
        / "best_model.pt"
    )


def load_runs(
    device: torch.device,
) -> list[dict]:
    runs = []

    print(
        "\nLoading frozen checkpoints..."
    )

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

            if int(
                checkpoint["seed"]
            ) != seed:
                raise RuntimeError(
                    f"Seed mismatch in {path}."
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
                    "configuration": configuration,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "model": model,
                }
            )

            print(
                f"  Loaded {configuration} "
                f"seed={seed}"
            )

    if len(runs) != 6:
        raise RuntimeError(
            f"Expected 6 runs, got {len(runs)}."
        )

    return runs


def build_rr_reference_cache() -> dict:
    """Raw RR + label references, record by record."""

    cache = {}

    print(
        "\nBuilding Final Test RR reference cache..."
    )

    for record_id in DS2_RECORDS:
        dataset = build_dataset_with_rr_from_records(
            record_ids=(record_id,),
            data_dir=DATA_DIR,
        )

        labels = np.asarray(
            [
                CLASS_NAMES[index]
                for index
                in dataset.targets.tolist()
            ]
        )

        rr_features = (
            dataset.rr_features
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        cache[record_id] = {
            "labels": labels,
            "rr_features": rr_features,
        }

        print(
            f"  {record_id}: "
            f"{len(labels)} beats"
        )

    return cache


def build_noisy_final_arrays(
    *,
    noise_type: str,
    target_snr_db: float,
    rr_cache: dict,
):
    """Build one Final Test corruption condition once."""

    all_beats = []
    all_rr = []
    all_labels = []

    for record_id in DS2_RECORDS:
        (
            noisy_beats,
            noisy_labels,
            heartbeat_metadata,
            corruption_metadata,
        ) = build_noisy_heartbeats(
            record_path=(
                DATA_DIR
                / record_id
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
                "Heartbeat metadata count mismatch for "
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
                f"{record_id}: "
                f"target={target_snr_db}, "
                f"achieved={achieved_snr}"
            )

        if not np.isfinite(
            noisy_beats
        ).all():
            raise RuntimeError(
                "Non-finite noisy ECG values found."
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


def make_noisy_dataset(
    *,
    heartbeats: np.ndarray,
    rr_features: np.ndarray,
    labels: np.ndarray,
    rr_mean: torch.Tensor,
    rr_std: torch.Tensor,
) -> ECGRRHeartbeatDataset:
    """Fresh noisy dataset for one checkpoint."""

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
        f"Unexpected SNR: {snr}"
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
    condition = (
        "clean"
        if target_snr_db is None
        else (
            f"{noise_type}_"
            f"{float(target_snr_db):g}db"
        )
    )

    exposure = exposure_label(
        target_snr_db
    )

    raw_rows.append(
        {
            "configuration": configuration,
            "seed": seed,
            "condition": condition,
            "noise_type": noise_type,
            "target_snr_db": (
                ""
                if target_snr_db is None
                else float(
                    target_snr_db
                )
            ),
            "exposure": exposure,
            "macro_f1": float(
                metrics["macro_f1"]
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
        ][class_name]

        per_class_rows.append(
            {
                "configuration": configuration,
                "seed": seed,
                "condition": condition,
                "noise_type": noise_type,
                "target_snr_db": (
                    ""
                    if target_snr_db is None
                    else float(
                        target_snr_db
                    )
                ),
                "exposure": exposure,
                "class_name": class_name,
                "precision": float(
                    values["precision"]
                ),
                "recall": float(
                    values["recall"]
                ),
                "f1": float(
                    values["f1"]
                ),
                "support": int(
                    values["support"]
                ),
            }
        )


def evaluate_clean_final(
    *,
    runs: list[dict],
    device: torch.device,
    raw_rows: list[dict],
    per_class_rows: list[dict],
) -> None:
    """Evaluate clean DS2 directly from the dataset object.

    This deliberately mirrors the clean Validation path that already worked.
    """

    print(
        "\n" + "=" * 88
    )
    print(
        "FINAL CONDITION: CLEAN"
    )
    print(
        "=" * 88
    )

    for run in runs:
        # Fresh dataset because RR standardization modifies rr_features.
        dataset = build_dataset_with_rr_from_records(
            record_ids=DS2_RECORDS,
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
            model=run["model"],
            dataset=dataset,
            device=device,
        )

        append_result(
            raw_rows=raw_rows,
            per_class_rows=per_class_rows,
            configuration=run[
                "configuration"
            ],
            seed=run["seed"],
            noise_type="clean",
            target_snr_db=None,
            metrics=metrics,
        )

        print(
            f"  {run['configuration']:<20} "
            f"seed={run['seed']:<4} "
            f"Macro-F1="
            f"{metrics['macro_f1']:.6f}"
        )


def build_summary(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    keys = (
        raw_df[
            [
                "condition",
                "noise_type",
                "target_snr_db",
                "exposure",
            ]
        ]
        .drop_duplicates()
    )

    for _, key in keys.iterrows():
        subset = raw_df[
            raw_df[
                "condition"
            ]
            == key[
                "condition"
            ]
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

        aug_df = (
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

        if clean_df[
            "seed"
        ].tolist() != list(
            SEEDS
        ):
            raise RuntimeError(
                f"Missing clean seeds for "
                f"{key['condition']}."
            )

        if aug_df[
            "seed"
        ].tolist() != list(
            SEEDS
        ):
            raise RuntimeError(
                f"Missing augmented seeds for "
                f"{key['condition']}."
            )

        clean_values = clean_df[
            "macro_f1"
        ].to_numpy(
            dtype=float
        )
        aug_values = aug_df[
            "macro_f1"
        ].to_numpy(
            dtype=float
        )

        difference = (
            aug_values
            - clean_values
        )

        rows.append(
            {
                "condition": key[
                    "condition"
                ],
                "noise_type": key[
                    "noise_type"
                ],
                "target_snr_db": key[
                    "target_snr_db"
                ],
                "exposure": key[
                    "exposure"
                ],
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
                        aug_values
                    )
                ),
                "noise_aug_macro_f1_std": float(
                    np.std(
                        aug_values,
                        ddof=1,
                    )
                ),
                "paired_difference_mean": float(
                    np.mean(
                        difference
                    )
                ),
                "paired_difference_std": float(
                    np.std(
                        difference,
                        ddof=1,
                    )
                ),
                "noise_aug_better_seeds": int(
                    np.sum(
                        difference
                        > 0.0
                    )
                ),
                "total_seeds": len(
                    SEEDS
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_per_class_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    keys = (
        df[
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

    for _, key in keys.iterrows():
        subset = df[
            (
                df[
                    "condition"
                ]
                == key[
                    "condition"
                ]
            )
            &
            (
                df[
                    "class_name"
                ]
                == key[
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

        aug_df = (
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
        aug_values = aug_df[
            "f1"
        ].to_numpy(
            dtype=float
        )

        if (
            len(clean_values)
            != len(SEEDS)
            or len(aug_values)
            != len(SEEDS)
        ):
            raise RuntimeError(
                "Missing class-wise seed results for "
                f"{key['condition']} / "
                f"{key['class_name']}."
            )

        difference = (
            aug_values
            - clean_values
        )

        rows.append(
            {
                "condition": key[
                    "condition"
                ],
                "noise_type": key[
                    "noise_type"
                ],
                "target_snr_db": key[
                    "target_snr_db"
                ],
                "exposure": key[
                    "exposure"
                ],
                "class_name": key[
                    "class_name"
                ],
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
                        aug_values
                    )
                ),
                "noise_aug_f1_std": float(
                    np.std(
                        aug_values,
                        ddof=1,
                    )
                ),
                "paired_difference_mean": float(
                    np.mean(
                        difference
                    )
                ),
                "paired_difference_std": float(
                    np.std(
                        difference,
                        ddof=1,
                    )
                ),
                "noise_aug_better_seeds": int(
                    np.sum(
                        difference
                        > 0.0
                    )
                ),
                "total_seeds": len(
                    SEEDS
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def print_final_summary(
    summary_df: pd.DataFrame,
) -> None:
    print(
        "\n" + "=" * 88
    )
    print(
        "FINAL TEST SUMMARY"
    )
    print(
        "=" * 88
    )

    clean_row = summary_df[
        summary_df[
            "exposure"
        ]
        == "clean"
    ].iloc[0]

    print("Clean DS2:")
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
        f"  Difference: "
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

        print(
            f"\n{exposure}:"
        )
        print(
            "  Mean clean-trained Macro-F1: "
            f"{subset['clean_macro_f1_mean'].mean():.6f}"
        )
        print(
            "  Mean noise-augmented Macro-F1: "
            f"{subset['noise_aug_macro_f1_mean'].mean():.6f}"
        )
        print(
            "  Mean paired difference: "
            f"{subset['paired_difference_mean'].mean():+.6f}"
        )
        print(
            "  Conditions with positive mean: "
            f"{int((subset['paired_difference_mean'] > 0).sum())}"
            f"/{len(subset)}"
        )
        print(
            "  Conditions improved in all 3 seeds: "
            f"{int((subset['noise_aug_better_seeds'] == 3).sum())}"
            f"/{len(subset)}"
        )


def main() -> None:
    print(
        "=" * 88
    )
    print(
        "FINAL HELD-OUT DS2 EVALUATION — V2"
    )
    print(
        "=" * 88
    )
    print(
        "No training. No tuning. Frozen checkpoints only."
    )

    verify_final_split()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(
        f"\nDevice: {device}"
    )

    runs = load_runs(
        device
    )

    rr_cache = build_rr_reference_cache()

    raw_rows = []
    per_class_rows = []

    # Clean DS2 — direct dataset evaluation.
    evaluate_clean_final(
        runs=runs,
        device=device,
        raw_rows=raw_rows,
        per_class_rows=per_class_rows,
    )

    # Noisy DS2.
    total_conditions = (
        len(
            VALID_NOISE_TYPES
        )
        * len(
            SNR_LEVELS_DB
        )
    )

    condition_index = 0

    for noise_type in VALID_NOISE_TYPES:
        for target_snr_db in SNR_LEVELS_DB:
            condition_index += 1

            print(
                "\n" + "=" * 88
            )
            print(
                f"FINAL CONDITION "
                f"{condition_index}/{total_conditions}: "
                f"{noise_type} @ "
                f"{target_snr_db:g} dB "
                f"[{exposure_label(target_snr_db)}]"
            )
            print(
                "=" * 88
            )

            (
                heartbeats,
                rr_features,
                labels,
            ) = build_noisy_final_arrays(
                noise_type=noise_type,
                target_snr_db=target_snr_db,
                rr_cache=rr_cache,
            )

            for run in runs:
                dataset = make_noisy_dataset(
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

    save_csv(
        RAW_OUTPUT,
        raw_rows,
    )
    save_csv(
        PER_CLASS_RAW_OUTPUT,
        per_class_rows,
    )

    raw_df = pd.DataFrame(
        raw_rows
    )
    per_class_df = pd.DataFrame(
        per_class_rows
    )

    summary_df = build_summary(
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

    print_final_summary(
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

    print(
        "\nFINAL TEST HAS NOW BEEN EVALUATED."
    )
    print(
        "Do not tune or retrain based on these results."
    )


if __name__ == "__main__":
    main()
