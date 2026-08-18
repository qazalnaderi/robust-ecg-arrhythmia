"""Clean-signal control for the RQ3 denoising experiment.

Question:
Does post-hoc band-pass or wavelet preprocessing change
classification performance even when the ECG is clean?

Validation only. Frozen ECG + raw-RR classifier.
"""

import csv
from pathlib import Path

import numpy as np
import torch
import wfdb
from torch.utils.data import DataLoader

from src.data.aami import map_to_aami
from src.data.normalization import normalize_heartbeats
from src.data.rr_normalization import (
    standardize_rr_features,
)
from src.data.segmentation import (
    CORE_CLASSES,
    DEFAULT_LEAD,
    extract_heartbeat_window,
    get_lead_signal,
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
from src.signal_processing.bandpass import (
    bandpass_filter,
)
from src.signal_processing.wavelet import (
    wavelet_denoise,
)


DATA_DIR = Path(
    "data/raw/mitdb"
)

CHECKPOINT_PATH = Path(
    "results/clean_baseline_rr/"
    "sqrt_weighted/best_model.pt"
)

SUMMARY_OUTPUT = Path(
    "results/tables/"
    "rq3_clean_filter_control.csv"
)

PER_CLASS_OUTPUT = Path(
    "results/tables/"
    "rq3_clean_filter_control_per_class.csv"
)

BATCH_SIZE = 256

METHODS = (
    "none",
    "bandpass",
    "wavelet",
)


def standardize_dataset_rr(
    dataset,
    rr_mean,
    rr_std,
):
    """Apply RR normalization learned from TRAIN only."""

    dataset.rr_features = (
        standardize_rr_features(
            rr_features=dataset.rr_features,
            mean=rr_mean,
            std=rr_std,
        )
    )


def process_clean_signal(
    signal: np.ndarray,
    sampling_rate: float,
    method: str,
) -> np.ndarray:
    """Apply one RQ3 preprocessing method to clean ECG."""

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    if method == "none":
        return signal.copy()

    if method == "bandpass":
        return bandpass_filter(
            signal=signal,
            sampling_rate=sampling_rate,
        )

    if method == "wavelet":
        return wavelet_denoise(
            signal=signal,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


def extract_processed_heartbeats(
    record_id: str,
    method: str,
):
    """Process continuous clean ECG, then extract heartbeats."""

    record_path = (
        DATA_DIR / record_id
    )

    record = wfdb.rdrecord(
        str(record_path)
    )

    annotation = wfdb.rdann(
        str(record_path),
        extension="atr",
    )

    clean_signal = get_lead_signal(
        record,
        lead_name=DEFAULT_LEAD,
    )

    processed_signal = process_clean_signal(
        signal=clean_signal,
        sampling_rate=float(record.fs),
        method=method,
    )

    if processed_signal.shape != np.asarray(
        clean_signal
    ).shape:

        raise RuntimeError(
            f"{method} changed ECG length "
            f"for record {record_id}."
        )

    if not np.isfinite(
        processed_signal
    ).all():

        raise RuntimeError(
            f"{method} produced invalid values "
            f"for record {record_id}."
        )

    heartbeats = []
    labels = []
    samples = []

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol,
    ):

        aami_class = map_to_aami(
            symbol
        )

        if aami_class not in CORE_CLASSES:
            continue

        heartbeat = extract_heartbeat_window(
            signal=processed_signal,
            center_sample=int(sample),
        )

        if heartbeat is None:
            continue

        heartbeats.append(
            heartbeat
        )

        labels.append(
            aami_class
        )

        samples.append(
            int(sample)
        )

    if not heartbeats:
        raise RuntimeError(
            f"No usable heartbeats for "
            f"record {record_id}."
        )

    heartbeats = np.stack(
        heartbeats
    )

    heartbeats = normalize_heartbeats(
        heartbeats
    )

    return (
        heartbeats,
        np.asarray(labels),
        np.asarray(
            samples,
            dtype=np.int64,
        ),
    )


def build_rr_reference_cache():
    """Build clean RR references using the existing trusted pipeline."""

    cache = {}

    print(
        "\nBuilding clean RR references..."
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

        # Build annotation/sample reference separately.
        _, direct_labels, samples = (
            extract_processed_heartbeats(
                record_id=record_id,
                method="none",
            )
        )

        if not np.array_equal(
            labels,
            direct_labels,
        ):
            raise RuntimeError(
                f"Reference label mismatch "
                f"for record {record_id}."
            )

        if len(rr_features) != len(
            labels
        ):
            raise RuntimeError(
                f"RR/label mismatch "
                f"for record {record_id}."
            )

        cache[record_id] = {
            "labels": labels,
            "rr_features": rr_features,
            "samples": samples,
        }

        print(
            f"  {record_id}: "
            f"{len(labels)} beats"
        )

    return cache


def build_filtered_validation_dataset(
    method: str,
    rr_cache: dict,
):
    """Build Validation ECG after one clean preprocessing method."""

    all_beats = []
    all_rr = []
    all_labels = []

    for record_id in VALIDATION_RECORDS:

        (
            beats,
            labels,
            samples,
        ) = extract_processed_heartbeats(
            record_id=record_id,
            method=method,
        )

        reference = rr_cache[
            record_id
        ]

        if not np.array_equal(
            labels,
            reference["labels"],
        ):
            raise RuntimeError(
                f"Labels changed under {method} "
                f"for record {record_id}."
            )

        if not np.array_equal(
            samples,
            reference["samples"],
        ):
            raise RuntimeError(
                f"Heartbeat positions changed "
                f"under {method} "
                f"for record {record_id}."
            )

        if len(beats) != len(
            reference["rr_features"]
        ):
            raise RuntimeError(
                f"ECG/RR alignment failed under "
                f"{method} for record {record_id}."
            )

        all_beats.append(
            beats
        )

        all_rr.append(
            reference[
                "rr_features"
            ]
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
    path: Path,
    rows: list[dict],
):
    """Save result rows."""

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


def main():

    print("=" * 88)
    print("RQ3 CLEAN DENOISING CONTROL")
    print("=" * 88)

    print(
        "Question: Does denoising itself shift "
        "clean classification performance?"
    )

    print(
        "Split: Validation only"
    )

    print(
        "Model: Frozen ECG + raw RR"
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
    # Load frozen classifier
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
    # RR + alignment references
    # =========================================================

    rr_cache = (
        build_rr_reference_cache()
    )

    summary_rows = []
    per_class_rows = []

    method_results = {}

    # =========================================================
    # Evaluate all three clean conditions
    # =========================================================

    for method in METHODS:

        print(
            "\n" + "-" * 88
        )

        print(
            f"Evaluating clean + {method}"
        )

        print(
            "-" * 88
        )

        dataset = (
            build_filtered_validation_dataset(
                method=method,
                rr_cache=rr_cache,
            )
        )

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

        metrics = evaluate_rr_model(
            model=model,
            dataloader=loader,
            device=device,
        )

        method_results[
            method
        ] = metrics

        print(
            f"Macro-F1: "
            f"{metrics['macro_f1']:.6f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.6f}"
        )

        print(
            "Per-class F1:"
        )

        for class_name in CLASS_NAMES:

            print(
                f"  {class_name}: "
                f"{metrics['per_class'][class_name]['f1']:.4f}"
            )

    # =========================================================
    # Verify original clean result
    # =========================================================

    original_clean_f1 = (
        method_results[
            "none"
        ]["macro_f1"]
    )

    checkpoint_clean_f1 = float(
        checkpoint[
            "validation_macro_f1"
        ]
    )

    if not np.isclose(
        original_clean_f1,
        checkpoint_clean_f1,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Clean none condition no longer "
            "matches the frozen checkpoint. "
            f"Current={original_clean_f1:.6f}, "
            f"checkpoint={checkpoint_clean_f1:.6f}"
        )

    print(
        "\nClean baseline consistency: PASS"
    )

    # =========================================================
    # Build tables
    # =========================================================

    for method in METHODS:

        metrics = (
            method_results[
                method
            ]
        )

        summary_rows.append(
            {
                "method": method,
                "macro_f1": (
                    metrics[
                        "macro_f1"
                    ]
                ),
                "macro_f1_delta_from_clean": (
                    metrics[
                        "macro_f1"
                    ]
                    - original_clean_f1
                ),
                "balanced_accuracy": (
                    metrics[
                        "balanced_accuracy"
                    ]
                ),
                "balanced_accuracy_delta_from_clean": (
                    metrics[
                        "balanced_accuracy"
                    ]
                    - method_results[
                        "none"
                    ][
                        "balanced_accuracy"
                    ]
                ),
            }
        )

        for class_name in CLASS_NAMES:

            values = metrics[
                "per_class"
            ][class_name]

            clean_values = (
                method_results[
                    "none"
                ][
                    "per_class"
                ][class_name]
            )

            per_class_rows.append(
                {
                    "method": method,
                    "class_name": (
                        class_name
                    ),
                    "precision": (
                        values[
                            "precision"
                        ]
                    ),
                    "recall": (
                        values[
                            "recall"
                        ]
                    ),
                    "f1": (
                        values[
                            "f1"
                        ]
                    ),
                    "support": (
                        values[
                            "support"
                        ]
                    ),
                    "f1_delta_from_clean": (
                        values[
                            "f1"
                        ]
                        - clean_values[
                            "f1"
                        ]
                    ),
                }
            )

    save_csv(
        SUMMARY_OUTPUT,
        summary_rows,
    )

    save_csv(
        PER_CLASS_OUTPUT,
        per_class_rows,
    )

    # =========================================================
    # Simple interpretation-ready output
    # =========================================================

    print(
        "\n" + "=" * 88
    )

    print(
        "CHANGE CAUSED BY FILTERING CLEAN ECG"
    )

    print(
        "=" * 88
    )

    for method in (
        "bandpass",
        "wavelet",
    ):

        delta = (
            method_results[
                method
            ]["macro_f1"]
            - original_clean_f1
        )

        print(
            f"{method:<9}: "
            f"{delta:+.6f} Macro-F1"
        )

    print(
        "\nSaved:"
    )

    print(
        f"  {SUMMARY_OUTPUT}"
    )

    print(
        f"  {PER_CLASS_OUTPUT}"
    )

    print(
        "\nRQ3 clean control: COMPLETE"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()