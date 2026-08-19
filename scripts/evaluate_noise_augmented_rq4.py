"""RQ4 evaluation: compare clean-trained and noise-augmented ECG+RR models.

The clean-trained model's previously saved RQ1 results are used as the
reference. The newly trained noise-augmented model is evaluated on the
same clean Validation split and the same NSTDB noise conditions.

Final Test is not used.
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

AUGMENTED_CHECKPOINT_PATH = Path(
    "results/noise_augmented_rr/sqrt_weighted/best_model.pt"
)

RQ1_REFERENCE_PATH = Path(
    "results/tables/rq1_noise_robustness_summary.csv"
)

SUMMARY_OUTPUT = Path(
    "results/tables/rq4_noise_augmented_comparison.csv"
)

PER_CLASS_OUTPUT = Path(
    "results/tables/rq4_noise_augmented_per_class.csv"
)

BATCH_SIZE = 256

SNR_LEVELS_DB = (
    18.0,
    12.0,
    6.0,
    0.0,
    -6.0,
)


def save_csv(path: Path, rows: list[dict]) -> None:
    """Save rows to CSV."""

    if not rows:
        raise RuntimeError(
            f"No rows were generated for {path}."
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


def standardize_dataset_rr(
    dataset,
    rr_mean,
    rr_std,
) -> None:
    """Apply the RR statistics stored in the RQ4 checkpoint."""

    dataset.rr_features = standardize_rr_features(
        rr_features=dataset.rr_features,
        mean=rr_mean,
        std=rr_std,
    )


def build_rr_reference_cache() -> dict:
    """Build clean RR/label references for every Validation record."""

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

        cache[record_id] = {
            "labels": labels,
            "rr_features": rr_features,
        }

        print(
            f"  {record_id}: {len(labels)} beats"
        )

    return cache


def build_noisy_validation_dataset(
    noise_type: str,
    target_snr_db: float,
    rr_cache: dict,
) -> ECGRRHeartbeatDataset:
    """Build one noisy Validation condition with annotation-based RR kept fixed."""

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
            record_path=DATA_DIR / record_id,
            noise_type=noise_type,
            target_snr_db=target_snr_db,
        )

        reference = rr_cache[record_id]

        noisy_beats = np.asarray(
            noisy_beats,
            dtype=np.float32,
        )

        noisy_labels = np.asarray(
            noisy_labels
        )

        reference_rr = reference[
            "rr_features"
        ]

        reference_labels = reference[
            "labels"
        ]

        if len(noisy_beats) != len(reference_rr):
            raise RuntimeError(
                "ECG/RR length mismatch for "
                f"{record_id}, {noise_type}@{target_snr_db:g} dB."
            )

        if len(heartbeat_metadata) != len(noisy_beats):
            raise RuntimeError(
                "Heartbeat metadata length mismatch for "
                f"{record_id}, {noise_type}@{target_snr_db:g} dB."
            )

        if not np.array_equal(
            noisy_labels,
            reference_labels,
        ):
            raise RuntimeError(
                "Label alignment changed for "
                f"{record_id}, {noise_type}@{target_snr_db:g} dB."
            )

        if not np.isfinite(
            noisy_beats
        ).all():
            raise RuntimeError(
                "Non-finite noisy ECG values detected."
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
                "Achieved SNR does not match target for "
                f"{record_id}, {noise_type}@{target_snr_db:g} dB: "
                f"{achieved_snr:.6f}"
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

    heartbeats = np.concatenate(
        all_beats,
        axis=0,
    )

    rr_features = np.concatenate(
        all_rr,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    return ECGRRHeartbeatDataset(
        heartbeats=heartbeats,
        rr_features=rr_features,
        labels=labels,
    )


def get_reference_row(
    rq1_df: pd.DataFrame,
    noise_type: str,
    target_snr_db,
) -> pd.Series:
    """Get exactly one matching clean-model RQ1 result."""

    if noise_type == "clean":

        rows = rq1_df[
            rq1_df["condition"]
            == "clean"
        ]

    else:

        numeric_snr = pd.to_numeric(
            rq1_df[
                "target_snr_db"
            ],
            errors="coerce",
        )

        rows = rq1_df[
            (
                rq1_df[
                    "noise_type"
                ]
                == noise_type
            )
            &
            (
                np.isclose(
                    numeric_snr,
                    float(
                        target_snr_db
                    ),
                    atol=1e-9,
                )
            )
        ]

    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one RQ1 reference row for "
            f"{noise_type} @ {target_snr_db}; got {len(rows)}."
        )

    return rows.iloc[0]


def classify_exposure(
    target_snr_db,
    seen_snrs: set[float],
    unseen_snrs: set[float],
) -> str:
    """Label each condition as clean, seen intensity, or unseen intensity."""

    if target_snr_db is None:
        return "clean"

    snr = float(
        target_snr_db
    )

    if snr in seen_snrs:
        return "seen_snr"

    if snr in unseen_snrs:
        return "unseen_snr"

    return "unexpected"


def evaluate_dataset(
    model,
    dataset,
    rr_mean,
    rr_std,
    device,
) -> dict:
    """Standardize RR and evaluate one dataset."""

    standardize_dataset_rr(
        dataset=dataset,
        rr_mean=rr_mean,
        rr_std=rr_std,
    )

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


def append_comparison_rows(
    *,
    summary_rows: list[dict],
    per_class_rows: list[dict],
    rq1_row: pd.Series,
    new_metrics: dict,
    condition: str,
    noise_type: str,
    target_snr_db,
    exposure: str,
) -> None:
    """Append overall and class-wise old-vs-new comparison rows."""

    old_macro_f1 = float(
        rq1_row[
            "macro_f1"
        ]
    )

    old_balanced_accuracy = float(
        rq1_row[
            "balanced_accuracy"
        ]
    )

    new_macro_f1 = float(
        new_metrics[
            "macro_f1"
        ]
    )

    new_balanced_accuracy = float(
        new_metrics[
            "balanced_accuracy"
        ]
    )

    summary_rows.append(
        {
            "condition": condition,
            "noise_type": noise_type,
            "target_snr_db": (
                ""
                if target_snr_db is None
                else float(target_snr_db)
            ),
            "exposure": exposure,
            "clean_trained_macro_f1": old_macro_f1,
            "noise_augmented_macro_f1": new_macro_f1,
            "macro_f1_difference": (
                new_macro_f1
                - old_macro_f1
            ),
            "clean_trained_balanced_accuracy": (
                old_balanced_accuracy
            ),
            "noise_augmented_balanced_accuracy": (
                new_balanced_accuracy
            ),
            "balanced_accuracy_difference": (
                new_balanced_accuracy
                - old_balanced_accuracy
            ),
        }
    )

    for class_name in CLASS_NAMES:

        old_class_f1_column = (
            f"{class_name}_f1"
        )

        if old_class_f1_column not in rq1_row.index:
            raise RuntimeError(
                "RQ1 summary is missing per-class column "
                f"{old_class_f1_column}."
            )

        old_class_f1 = float(
            rq1_row[
                old_class_f1_column
            ]
        )

        new_class_metrics = (
            new_metrics[
                "per_class"
            ][
                class_name
            ]
        )

        new_class_f1 = float(
            new_class_metrics[
                "f1"
            ]
        )

        per_class_rows.append(
            {
                "condition": condition,
                "noise_type": noise_type,
                "target_snr_db": (
                    ""
                    if target_snr_db is None
                    else float(target_snr_db)
                ),
                "exposure": exposure,
                "class_name": class_name,
                "clean_trained_f1": old_class_f1,
                "noise_augmented_f1": new_class_f1,
                "f1_difference": (
                    new_class_f1
                    - old_class_f1
                ),
                "noise_augmented_precision": float(
                    new_class_metrics[
                        "precision"
                    ]
                ),
                "noise_augmented_recall": float(
                    new_class_metrics[
                        "recall"
                    ]
                ),
                "support": int(
                    new_class_metrics[
                        "support"
                    ]
                ),
            }
        )


def main() -> None:

    print("=" * 88)
    print("RQ4 NOISE-AUGMENTED ROBUSTNESS COMPARISON")
    print("=" * 88)

    print(
        "Split: Validation only"
    )

    print(
        "Old model: clean-trained ECG + raw RR"
    )

    print(
        "New model: 50/50 clean-noisy training"
    )

    print(
        "Final Test: NOT USED"
    )

    if not AUGMENTED_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing RQ4 checkpoint: {AUGMENTED_CHECKPOINT_PATH}"
        )

    if not RQ1_REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing RQ1 reference table: {RQ1_REFERENCE_PATH}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    checkpoint = torch.load(
        AUGMENTED_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    rr_mean = checkpoint[
        "rr_mean"
    ].cpu()

    rr_std = checkpoint[
        "rr_std"
    ].cpu()

    seen_snrs = {
        float(value)
        for value in checkpoint[
            "train_snr_levels_db"
        ]
    }

    unseen_snrs = {
        float(value)
        for value in checkpoint[
            "unseen_snr_levels_db"
        ]
    }

    if seen_snrs & unseen_snrs:
        raise RuntimeError(
            "RQ4 checkpoint contains overlapping seen/unseen SNRs."
        )

    print(
        f"Seen SNRs during Train: "
        f"{sorted(seen_snrs, reverse=True)}"
    )

    print(
        f"Reserved unseen SNRs: "
        f"{sorted(unseen_snrs, reverse=True)}"
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

    rq1_df = pd.read_csv(
        RQ1_REFERENCE_PATH
    )

    rr_cache = (
        build_rr_reference_cache()
    )

    summary_rows = []
    per_class_rows = []

    # =========================================================
    # 1. Clean Validation
    # =========================================================

    print("\n" + "-" * 88)
    print("Evaluating CLEAN Validation")
    print("-" * 88)

    clean_dataset = (
        build_dataset_with_rr_from_records(
            record_ids=VALIDATION_RECORDS,
            data_dir=DATA_DIR,
        )
    )

    clean_metrics = evaluate_dataset(
        model=model,
        dataset=clean_dataset,
        rr_mean=rr_mean,
        rr_std=rr_std,
        device=device,
    )

    checkpoint_validation_f1 = float(
        checkpoint[
            "validation_macro_f1"
        ]
    )

    if not np.isclose(
        clean_metrics[
            "macro_f1"
        ],
        checkpoint_validation_f1,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Current clean RQ4 evaluation does not match "
            "the saved checkpoint validation score. "
            f"Current={clean_metrics['macro_f1']:.6f}, "
            f"checkpoint={checkpoint_validation_f1:.6f}"
        )

    clean_reference = get_reference_row(
        rq1_df=rq1_df,
        noise_type="clean",
        target_snr_db=None,
    )

    append_comparison_rows(
        summary_rows=summary_rows,
        per_class_rows=per_class_rows,
        rq1_row=clean_reference,
        new_metrics=clean_metrics,
        condition="clean",
        noise_type="clean",
        target_snr_db=None,
        exposure="clean",
    )

    print(
        "Clean-trained Macro-F1: "
        f"{float(clean_reference['macro_f1']):.6f}"
    )

    print(
        "Noise-augmented Macro-F1: "
        f"{clean_metrics['macro_f1']:.6f}"
    )

    print(
        "Difference: "
        f"{clean_metrics['macro_f1'] - float(clean_reference['macro_f1']):+.6f}"
    )

    # =========================================================
    # 2. All NSTDB noise conditions
    # =========================================================

    total_conditions = (
        len(VALID_NOISE_TYPES)
        * len(SNR_LEVELS_DB)
    )

    condition_index = 0

    for noise_type in VALID_NOISE_TYPES:

        for target_snr_db in SNR_LEVELS_DB:

            condition_index += 1

            exposure = classify_exposure(
                target_snr_db=target_snr_db,
                seen_snrs=seen_snrs,
                unseen_snrs=unseen_snrs,
            )

            if exposure == "unexpected":
                raise RuntimeError(
                    f"SNR {target_snr_db} is neither "
                    "seen nor reserved unseen."
                )

            print("\n" + "-" * 88)

            print(
                f"Condition {condition_index}/{total_conditions}: "
                f"{noise_type} @ {target_snr_db:g} dB "
                f"[{exposure}]"
            )

            print("-" * 88)

            noisy_dataset = (
                build_noisy_validation_dataset(
                    noise_type=noise_type,
                    target_snr_db=target_snr_db,
                    rr_cache=rr_cache,
                )
            )

            metrics = evaluate_dataset(
                model=model,
                dataset=noisy_dataset,
                rr_mean=rr_mean,
                rr_std=rr_std,
                device=device,
            )

            rq1_row = get_reference_row(
                rq1_df=rq1_df,
                noise_type=noise_type,
                target_snr_db=target_snr_db,
            )

            condition_name = (
                f"{noise_type}_{target_snr_db:g}db"
            )

            append_comparison_rows(
                summary_rows=summary_rows,
                per_class_rows=per_class_rows,
                rq1_row=rq1_row,
                new_metrics=metrics,
                condition=condition_name,
                noise_type=noise_type,
                target_snr_db=target_snr_db,
                exposure=exposure,
            )

            old_macro_f1 = float(
                rq1_row[
                    "macro_f1"
                ]
            )

            new_macro_f1 = float(
                metrics[
                    "macro_f1"
                ]
            )

            print(
                f"Clean-trained:   {old_macro_f1:.6f}"
            )

            print(
                f"Noise-augmented: {new_macro_f1:.6f}"
            )

            print(
                f"Difference:      "
                f"{new_macro_f1 - old_macro_f1:+.6f}"
            )

    # =========================================================
    # 3. Save comparison tables
    # =========================================================

    save_csv(
        path=SUMMARY_OUTPUT,
        rows=summary_rows,
    )

    save_csv(
        path=PER_CLASS_OUTPUT,
        rows=per_class_rows,
    )

    # =========================================================
    # 4. Simple seen/unseen summary
    # =========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    print("\n" + "=" * 88)
    print("RQ4 SUMMARY")
    print("=" * 88)

    clean_row = summary_df[
        summary_df[
            "exposure"
        ]
        == "clean"
    ].iloc[0]

    print(
        "Clean Macro-F1 difference: "
        f"{float(clean_row['macro_f1_difference']):+.6f}"
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

        mean_old = float(
            subset[
                "clean_trained_macro_f1"
            ].mean()
        )

        mean_new = float(
            subset[
                "noise_augmented_macro_f1"
            ].mean()
        )

        mean_difference = float(
            subset[
                "macro_f1_difference"
            ].mean()
        )

        improved_conditions = int(
            (
                subset[
                    "macro_f1_difference"
                ]
                > 0.0
            ).sum()
        )

        total = len(
            subset
        )

        print(
            f"\n{exposure}:"
        )

        print(
            f"  Mean clean-trained Macro-F1: "
            f"{mean_old:.6f}"
        )

        print(
            f"  Mean noise-augmented Macro-F1: "
            f"{mean_new:.6f}"
        )

        print(
            f"  Mean difference: "
            f"{mean_difference:+.6f}"
        )

        print(
            f"  Improved conditions: "
            f"{improved_conditions}/{total}"
        )

    print(
        f"\nSaved summary: {SUMMARY_OUTPUT}"
    )

    print(
        f"Saved per-class table: {PER_CLASS_OUTPUT}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
