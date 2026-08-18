"""Benchmark the frozen ECG + RR model under NSTDB noise."""

import csv
from pathlib import Path

import numpy as np
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
    build_noisy_heartbeats,
)
from src.noise.nstdb import (
    VALID_NOISE_TYPES,
)


DATA_DIR = Path("data/raw/mitdb")

CHECKPOINT_PATH = Path(
    "results/clean_baseline_rr/"
    "sqrt_weighted/best_model.pt"
)

SUMMARY_OUTPUT = Path(
    "results/tables/"
    "rq1_noise_robustness_summary.csv"
)

PER_CLASS_OUTPUT = Path(
    "results/tables/"
    "rq2_noise_robustness_per_class.csv"
)

CORRUPTION_OUTPUT = Path(
    "results/tables/"
    "noise_corruption_metadata.csv"
)

BATCH_SIZE = 256

SNR_LEVELS_DB = (
    18.0,
    12.0,
    6.0,
    0.0,
    -6.0,
)


def standardize_dataset_rr(
    dataset: ECGRRHeartbeatDataset,
    rr_mean: torch.Tensor,
    rr_std: torch.Tensor,
) -> None:
    """Apply frozen TRAIN RR normalization statistics."""

    dataset.rr_features = (
        standardize_rr_features(
            rr_features=dataset.rr_features,
            mean=rr_mean,
            std=rr_std,
        )
    )


def build_rr_reference_cache(
) -> dict[str, dict]:
    """
    Build clean RR references for each validation record.

    Only RR features and labels are cached.
    ECG waveforms will later be replaced by noisy versions.
    """

    cache = {}

    print("\nBuilding clean RR reference cache...")

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
            .cpu()
            .numpy()
            .copy()
        )

        cache[record_id] = {
            "rr_features": rr_features,
            "labels": labels,
        }

        print(
            f"  Record {record_id}: "
            f"{len(labels)} beats"
        )

    return cache


def build_noisy_validation_dataset(
    noise_type: str,
    target_snr_db: float,
    rr_reference_cache: dict[str, dict],
) -> tuple[
    ECGRRHeartbeatDataset,
    list[dict],
]:
    """
    Build one complete noisy validation dataset.

    ECG waveforms are corrupted.
    RR features remain based on clean annotation timing.
    """

    all_heartbeats = []
    all_rr_features = []
    all_labels = []

    corruption_rows = []

    for record_id in VALIDATION_RECORDS:

        (
            noisy_heartbeats,
            noisy_labels,
            heartbeat_metadata,
            corruption_metadata,
        ) = build_noisy_heartbeats(
            record_path=DATA_DIR / record_id,
            noise_type=noise_type,
            target_snr_db=target_snr_db,
        )

        reference = rr_reference_cache[
            record_id
        ]

        reference_rr = reference[
            "rr_features"
        ]

        reference_labels = reference[
            "labels"
        ]

        # -----------------------------------------------------
        # Hard alignment checks
        # -----------------------------------------------------

        if (
            len(noisy_heartbeats)
            != len(reference_rr)
        ):
            raise RuntimeError(
                "Noisy ECG / RR length mismatch for "
                f"record {record_id}: "
                f"ECG={len(noisy_heartbeats)}, "
                f"RR={len(reference_rr)}"
            )

        if (
            len(noisy_labels)
            != len(reference_labels)
        ):
            raise RuntimeError(
                "Noisy / clean label count mismatch "
                f"for record {record_id}."
            )

        if not np.array_equal(
            noisy_labels,
            reference_labels,
        ):
            raise RuntimeError(
                "Heartbeat label alignment changed "
                f"for record {record_id}."
            )

        if (
            len(heartbeat_metadata)
            != len(noisy_heartbeats)
        ):
            raise RuntimeError(
                "Heartbeat metadata length mismatch "
                f"for record {record_id}."
            )

        all_heartbeats.append(
            noisy_heartbeats
        )

        all_rr_features.append(
            reference_rr
        )

        all_labels.append(
            noisy_labels
        )

        corruption_rows.append(
            {
                "record_id": record_id,
                "noise_type": noise_type,
                "target_snr_db": (
                    target_snr_db
                ),
                "achieved_snr_db": (
                    corruption_metadata[
                        "achieved_snr_db"
                    ]
                ),
                "noise_channel": (
                    corruption_metadata[
                        "noise_channel"
                    ]
                ),
                "start_offset": (
                    corruption_metadata[
                        "start_offset"
                    ]
                ),
                "seed": (
                    corruption_metadata[
                        "seed"
                    ]
                ),
            }
        )

    heartbeats = np.concatenate(
        all_heartbeats,
        axis=0,
    )

    rr_features = np.concatenate(
        all_rr_features,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    dataset = ECGRRHeartbeatDataset(
        heartbeats=heartbeats,
        rr_features=rr_features,
        labels=labels,
    )

    return (
        dataset,
        corruption_rows,
    )


def make_summary_row(
    condition: str,
    noise_type: str,
    target_snr_db,
    metrics: dict,
    clean_metrics: dict,
) -> dict:
    """Convert metrics into one summary-table row."""

    row = {
        "condition": condition,
        "noise_type": noise_type,
        "target_snr_db": target_snr_db,
        "macro_f1": metrics[
            "macro_f1"
        ],
        "macro_f1_delta": (
            metrics["macro_f1"]
            - clean_metrics["macro_f1"]
        ),
        "balanced_accuracy": (
            metrics["balanced_accuracy"]
        ),
        "balanced_accuracy_delta": (
            metrics["balanced_accuracy"]
            - clean_metrics[
                "balanced_accuracy"
            ]
        ),
    }

    for class_name in CLASS_NAMES:

        class_metrics = metrics[
            "per_class"
        ][class_name]

        clean_class_metrics = (
            clean_metrics[
                "per_class"
            ][class_name]
        )

        row[
            f"{class_name}_f1"
        ] = class_metrics["f1"]

        row[
            f"{class_name}_f1_delta"
        ] = (
            class_metrics["f1"]
            - clean_class_metrics["f1"]
        )

    return row


def make_per_class_rows(
    condition: str,
    noise_type: str,
    target_snr_db,
    metrics: dict,
    clean_metrics: dict,
) -> list[dict]:
    """Build long-format class-wise result rows."""

    rows = []

    for class_name in CLASS_NAMES:

        values = metrics[
            "per_class"
        ][class_name]

        clean_values = clean_metrics[
            "per_class"
        ][class_name]

        rows.append(
            {
                "condition": condition,
                "noise_type": noise_type,
                "target_snr_db": (
                    target_snr_db
                ),
                "class_name": class_name,
                "precision": values[
                    "precision"
                ],
                "recall": values[
                    "recall"
                ],
                "f1": values[
                    "f1"
                ],
                "support": values[
                    "support"
                ],
                "f1_delta_from_clean": (
                    values["f1"]
                    - clean_values["f1"]
                ),
            }
        )

    return rows


def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Save rows to CSV."""

    if not rows:
        raise ValueError(
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


def main() -> None:

    print("=" * 88)
    print("RQ1 / RQ2 NSTDB NOISE ROBUSTNESS BENCHMARK")
    print("=" * 88)

    print(
        "Evaluation split: VALIDATION only"
    )

    print(
        "Frozen model: ECG + raw RR"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ---------------------------------------------------------
    # 1. Load frozen model
    # ---------------------------------------------------------

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
        num_classes=len(CLASS_NAMES)
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ---------------------------------------------------------
    # 2. CLEAN reference evaluation
    # ---------------------------------------------------------

    print(
        "\nEvaluating clean validation reference..."
    )

    clean_dataset = (
        build_dataset_with_rr_from_records(
            record_ids=VALIDATION_RECORDS,
            data_dir=DATA_DIR,
        )
    )

    standardize_dataset_rr(
        dataset=clean_dataset,
        rr_mean=rr_mean,
        rr_std=rr_std,
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

    stored_clean_macro_f1 = float(
        checkpoint[
            "validation_macro_f1"
        ]
    )

    clean_difference = abs(
        clean_metrics["macro_f1"]
        - stored_clean_macro_f1
    )

    if clean_difference > 1e-6:
        raise RuntimeError(
            "Clean validation result does not "
            "match the frozen checkpoint. "
            f"Checkpoint={stored_clean_macro_f1:.6f}, "
            f"Current={clean_metrics['macro_f1']:.6f}"
        )

    print(
        "Clean Macro-F1: "
        f"{clean_metrics['macro_f1']:.6f}"
    )

    print(
        "Clean result matches frozen checkpoint: PASS"
    )

    # ---------------------------------------------------------
    # 3. Prepare RR references
    # ---------------------------------------------------------

    rr_reference_cache = (
        build_rr_reference_cache()
    )

    summary_rows = []

    per_class_rows = []

    corruption_rows = []

    # Clean row
    summary_rows.append(
        make_summary_row(
            condition="clean",
            noise_type="clean",
            target_snr_db="",
            metrics=clean_metrics,
            clean_metrics=clean_metrics,
        )
    )

    per_class_rows.extend(
        make_per_class_rows(
            condition="clean",
            noise_type="clean",
            target_snr_db="",
            metrics=clean_metrics,
            clean_metrics=clean_metrics,
        )
    )

    # ---------------------------------------------------------
    # 4. Main RQ1 / RQ2 benchmark
    # ---------------------------------------------------------

    total_conditions = (
        len(VALID_NOISE_TYPES)
        * len(SNR_LEVELS_DB)
    )

    condition_index = 0

    for noise_type in VALID_NOISE_TYPES:

        for target_snr_db in SNR_LEVELS_DB:

            condition_index += 1

            print("\n" + "-" * 88)

            print(
                f"Condition "
                f"{condition_index}/{total_conditions}: "
                f"{noise_type} @ "
                f"{target_snr_db:g} dB"
            )

            print("-" * 88)

            (
                noisy_dataset,
                condition_corruption_rows,
            ) = build_noisy_validation_dataset(
                noise_type=noise_type,
                target_snr_db=target_snr_db,
                rr_reference_cache=(
                    rr_reference_cache
                ),
            )

            # Apply the exact RR normalization
            # stored in the frozen checkpoint.
            standardize_dataset_rr(
                dataset=noisy_dataset,
                rr_mean=rr_mean,
                rr_std=rr_std,
            )

            noisy_loader = DataLoader(
                noisy_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )

            metrics = evaluate_rr_model(
                model=model,
                dataloader=noisy_loader,
                device=device,
            )

            condition_name = (
                f"{noise_type}_"
                f"{target_snr_db:g}db"
            )

            summary_rows.append(
                make_summary_row(
                    condition=condition_name,
                    noise_type=noise_type,
                    target_snr_db=(
                        target_snr_db
                    ),
                    metrics=metrics,
                    clean_metrics=clean_metrics,
                )
            )

            per_class_rows.extend(
                make_per_class_rows(
                    condition=condition_name,
                    noise_type=noise_type,
                    target_snr_db=(
                        target_snr_db
                    ),
                    metrics=metrics,
                    clean_metrics=clean_metrics,
                )
            )

            corruption_rows.extend(
                condition_corruption_rows
            )

            print(
                f"Macro-F1: "
                f"{metrics['macro_f1']:.6f}"
            )

            print(
                f"Delta from clean: "
                f"{metrics['macro_f1'] - clean_metrics['macro_f1']:+.6f}"
            )

            print(
                "Per-class F1:"
            )

            for class_name in CLASS_NAMES:

                class_f1 = metrics[
                    "per_class"
                ][class_name]["f1"]

                print(
                    f"  {class_name}: "
                    f"{class_f1:.4f}"
                )

    # ---------------------------------------------------------
    # 5. Save results
    # ---------------------------------------------------------

    save_csv(
        path=SUMMARY_OUTPUT,
        rows=summary_rows,
    )

    save_csv(
        path=PER_CLASS_OUTPUT,
        rows=per_class_rows,
    )

    save_csv(
        path=CORRUPTION_OUTPUT,
        rows=corruption_rows,
    )

    print("\n" + "=" * 88)

    print("BENCHMARK COMPLETE")

    print(
        f"Summary: {SUMMARY_OUTPUT}"
    )

    print(
        f"Per-class results: "
        f"{PER_CLASS_OUTPUT}"
    )

    print(
        f"Corruption metadata: "
        f"{CORRUPTION_OUTPUT}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()