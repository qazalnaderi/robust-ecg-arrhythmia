"""RQ3 benchmark: evaluate denoising under realistic NSTDB artifacts."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.rr_normalization import (
    standardize_rr_features,
)
from src.data.splits import (
    VALIDATION_RECORDS,
)
from src.data.torch_dataset import (
    CLASS_NAMES,
    ECGRRHeartbeatDataset,
    build_dataset_with_rr_from_records,
)
from src.evaluation.rr_evaluator import (
    evaluate_rr_model,
)
from src.models.cnn1d_rr import (
    ECGCNN1DWithRR,
)
from src.noise.heartbeat_pipeline import (
    build_denoised_heartbeats,
)
from src.noise.nstdb import (
    VALID_NOISE_TYPES,
)


DATA_DIR = Path(
    "data/raw/mitdb"
)

CHECKPOINT_PATH = Path(
    "results/clean_baseline_rr/"
    "sqrt_weighted/best_model.pt"
)

RQ1_PATH = Path(
    "results/tables/"
    "rq1_noise_robustness_summary.csv"
)

SUMMARY_OUTPUT = Path(
    "results/tables/"
    "rq3_denoising_summary.csv"
)

PER_CLASS_OUTPUT = Path(
    "results/tables/"
    "rq3_denoising_per_class.csv"
)

BATCH_SIZE = 256

SNR_LEVELS_DB = (
    18.0,
    12.0,
    6.0,
    0.0,
    -6.0,
)

DENOISING_METHODS = (
    "none",
    "bandpass",
    "wavelet",
)


def standardize_dataset_rr(
    dataset,
    rr_mean,
    rr_std,
):
    """Apply RR statistics learned from TRAIN only."""

    dataset.rr_features = (
        standardize_rr_features(
            rr_features=dataset.rr_features,
            mean=rr_mean,
            std=rr_std,
        )
    )


def build_rr_reference_cache():
    """
    Cache clean annotation-based RR features for Validation.

    RR timing remains unchanged because RQ3 studies
    waveform denoising, not timing corruption.
    """

    cache = {}

    print(
        "\nBuilding RR reference cache..."
    )

    for record_id in VALIDATION_RECORDS:

        dataset = (
            build_dataset_with_rr_from_records(
                record_ids=(record_id,),
                data_dir=DATA_DIR,
            )
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


def build_condition_dataset(
    noise_type,
    target_snr_db,
    denoising_method,
    rr_cache,
):
    """Build one complete noisy Validation condition."""

    all_beats = []
    all_rr = []
    all_labels = []

    for record_id in VALIDATION_RECORDS:

        (
            beats,
            labels,
            metadata,
            corruption_metadata,
        ) = build_denoised_heartbeats(
            record_path=(
                DATA_DIR / record_id
            ),
            noise_type=noise_type,
            target_snr_db=target_snr_db,
            denoising_method=(
                denoising_method
            ),
        )

        reference = rr_cache[
            record_id
        ]

        reference_labels = reference[
            "labels"
        ]

        reference_rr = reference[
            "rr_features"
        ]

        # ---------------------------------------------
        # Scientific alignment checks
        # ---------------------------------------------

        if len(beats) != len(
            reference_rr
        ):
            raise RuntimeError(
                f"ECG/RR mismatch for "
                f"{record_id}, "
                f"{noise_type}, "
                f"{target_snr_db}, "
                f"{denoising_method}."
            )

        if not np.array_equal(
            labels,
            reference_labels,
        ):
            raise RuntimeError(
                f"Label alignment changed for "
                f"{record_id}."
            )

        if len(metadata) != len(
            beats
        ):
            raise RuntimeError(
                f"Metadata mismatch for "
                f"{record_id}."
            )

        if not np.isfinite(
            beats
        ).all():
            raise RuntimeError(
                "Non-finite ECG values detected."
            )

        if not np.isfinite(
            reference_rr
        ).all():
            raise RuntimeError(
                "Non-finite RR values detected."
            )

        all_beats.append(
            beats
        )

        all_rr.append(
            reference_rr
        )

        all_labels.append(
            labels
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


def save_csv(
    path,
    rows,
):

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


def get_rq1_reference(
    rq1_df,
    noise_type,
    target_snr_db,
):
    """Load the original no-denoising RQ1 result."""

    rows = rq1_df[
        (
            rq1_df["noise_type"]
            == noise_type
        )
        &
        (
            np.isclose(
                pd.to_numeric(
                    rq1_df[
                        "target_snr_db"
                    ],
                    errors="coerce",
                ),
                target_snr_db,
            )
        )
    ]

    if len(rows) != 1:
        raise RuntimeError(
            "Could not find unique RQ1 "
            f"reference for {noise_type} "
            f"@ {target_snr_db} dB."
        )

    return float(
        rows.iloc[0][
            "macro_f1"
        ]
    )


def main():

    print("=" * 88)
    print("RQ3 DENOISING BENCHMARK")
    print("=" * 88)

    print(
        "Split: Validation only"
    )

    print(
        "Model: frozen ECG + raw RR"
    )

    print(
        "Methods: none / bandpass / wavelet"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # =========================================================
    # Load frozen model
    # =========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    rr_mean = checkpoint[
        "rr_mean"
    ].cpu()

    rr_std = checkpoint[
        "rr_std"
    ].cpu()

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

    # =========================================================
    # Verify clean baseline before RQ3
    # =========================================================

    print(
        "\nChecking frozen clean baseline..."
    )

    clean_dataset = (
        build_dataset_with_rr_from_records(
            record_ids=(
                VALIDATION_RECORDS
            ),
            data_dir=DATA_DIR,
        )
    )

    standardize_dataset_rr(
        clean_dataset,
        rr_mean,
        rr_std,
    )

    clean_loader = DataLoader(
        clean_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    clean_metrics = evaluate_rr_model(
        model=model,
        dataloader=clean_loader,
        device=device,
    )

    checkpoint_clean_f1 = float(
        checkpoint[
            "validation_macro_f1"
        ]
    )

    print(
        f"Current clean Macro-F1: "
        f"{clean_metrics['macro_f1']:.6f}"
    )

    print(
        f"Checkpoint Macro-F1:     "
        f"{checkpoint_clean_f1:.6f}"
    )

    if not np.isclose(
        clean_metrics["macro_f1"],
        checkpoint_clean_f1,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Clean baseline changed. "
            "Do not continue RQ3."
        )

    print(
        "Clean baseline consistency: PASS"
    )

    # =========================================================
    # References
    # =========================================================

    if not RQ1_PATH.exists():
        raise FileNotFoundError(
            f"Missing RQ1 table: "
            f"{RQ1_PATH}"
        )

    rq1_df = pd.read_csv(
        RQ1_PATH
    )

    rr_cache = (
        build_rr_reference_cache()
    )

    summary_rows = []
    per_class_rows = []

    total_conditions = (
        len(VALID_NOISE_TYPES)
        * len(SNR_LEVELS_DB)
    )

    condition_number = 0

    # =========================================================
    # Main RQ3 experiment
    # =========================================================

    for noise_type in VALID_NOISE_TYPES:

        for target_snr_db in SNR_LEVELS_DB:

            condition_number += 1

            print("\n" + "=" * 88)

            print(
                f"Condition "
                f"{condition_number}/"
                f"{total_conditions}: "
                f"{noise_type} @ "
                f"{target_snr_db:g} dB"
            )

            print("=" * 88)

            method_results = {}

            for method in DENOISING_METHODS:

                print(
                    f"\nEvaluating: {method}"
                )

                dataset = (
                    build_condition_dataset(
                        noise_type=(
                            noise_type
                        ),
                        target_snr_db=(
                            target_snr_db
                        ),
                        denoising_method=(
                            method
                        ),
                        rr_cache=rr_cache,
                    )
                )

                standardize_dataset_rr(
                    dataset,
                    rr_mean,
                    rr_std,
                )

                loader = DataLoader(
                    dataset,
                    batch_size=(
                        BATCH_SIZE
                    ),
                    shuffle=False,
                    num_workers=0,
                )

                metrics = (
                    evaluate_rr_model(
                        model=model,
                        dataloader=(
                            loader
                        ),
                        device=device,
                    )
                )

                method_results[
                    method
                ] = metrics

                print(
                    f"  Macro-F1 = "
                    f"{metrics['macro_f1']:.6f}"
                )

            # ---------------------------------------------
            # RQ1 backward consistency
            # ---------------------------------------------

            rq1_macro_f1 = (
                get_rq1_reference(
                    rq1_df=(
                        rq1_df
                    ),
                    noise_type=(
                        noise_type
                    ),
                    target_snr_db=(
                        target_snr_db
                    ),
                )
            )

            rq3_none_f1 = (
                method_results[
                    "none"
                ]["macro_f1"]
            )

            if not np.isclose(
                rq1_macro_f1,
                rq3_none_f1,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "RQ3 none condition does "
                    "not reproduce RQ1. "
                    f"{noise_type} "
                    f"{target_snr_db:g} dB: "
                    f"RQ1={rq1_macro_f1:.6f}, "
                    f"RQ3={rq3_none_f1:.6f}"
                )

            print(
                "  RQ1 consistency: PASS"
            )

            baseline_f1 = (
                method_results[
                    "none"
                ]["macro_f1"]
            )

            # ---------------------------------------------
            # Save method-level results
            # ---------------------------------------------

            for method in DENOISING_METHODS:

                metrics = (
                    method_results[
                        method
                    ]
                )

                macro_recovery = (
                    metrics["macro_f1"]
                    - baseline_f1
                )

                summary_rows.append(
                    {
                        "noise_type": (
                            noise_type
                        ),
                        "target_snr_db": (
                            target_snr_db
                        ),
                        "method": (
                            method
                        ),
                        "macro_f1": (
                            metrics[
                                "macro_f1"
                            ]
                        ),
                        "macro_f1_recovery_vs_none": (
                            macro_recovery
                        ),
                        "balanced_accuracy": (
                            metrics[
                                "balanced_accuracy"
                            ]
                        ),
                    }
                )

                for class_name in CLASS_NAMES:

                    class_metrics = (
                        metrics[
                            "per_class"
                        ][class_name]
                    )

                    baseline_class_f1 = (
                        method_results[
                            "none"
                        ][
                            "per_class"
                        ][
                            class_name
                        ][
                            "f1"
                        ]
                    )

                    per_class_rows.append(
                        {
                            "noise_type": (
                                noise_type
                            ),
                            "target_snr_db": (
                                target_snr_db
                            ),
                            "method": (
                                method
                            ),
                            "class_name": (
                                class_name
                            ),
                            "precision": (
                                class_metrics[
                                    "precision"
                                ]
                            ),
                            "recall": (
                                class_metrics[
                                    "recall"
                                ]
                            ),
                            "f1": (
                                class_metrics[
                                    "f1"
                                ]
                            ),
                            "support": (
                                class_metrics[
                                    "support"
                                ]
                            ),
                            "f1_recovery_vs_none": (
                                class_metrics[
                                    "f1"
                                ]
                                - baseline_class_f1
                            ),
                        }
                    )

            # ---------------------------------------------
            # Quick condition summary
            # ---------------------------------------------

            print(
                "\nRecovery vs no denoising:"
            )

            for method in (
                "bandpass",
                "wavelet",
            ):

                recovery = (
                    method_results[
                        method
                    ]["macro_f1"]
                    - baseline_f1
                )

                print(
                    f"  {method:<9}: "
                    f"{recovery:+.6f}"
                )

    # =========================================================
    # Save
    # =========================================================

    save_csv(
        SUMMARY_OUTPUT,
        summary_rows,
    )

    save_csv(
        PER_CLASS_OUTPUT,
        per_class_rows,
    )

    print("\n" + "=" * 88)

    print(
        "RQ3 BENCHMARK COMPLETE"
    )

    print(
        f"Summary: "
        f"{SUMMARY_OUTPUT}"
    )

    print(
        f"Per-class: "
        f"{PER_CLASS_OUTPUT}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()