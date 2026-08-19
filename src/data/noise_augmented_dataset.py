"""Noise-augmented ECG + RR training dataset for RQ4.

The purpose of this module is to build a controlled training set
containing:

    50% clean ECG beats
    50% NSTDB-corrupted ECG beats

Only TRAIN_RECORDS are allowed.

Noise augmentation uses:
    noise types: bw, ma, em
    training SNRs: 18, 6, -6 dB

The intermediate SNRs 12 and 0 dB are intentionally excluded from
training so that they can later be used to study generalization to
unseen noise intensities.

RR features are NOT corrupted. They remain derived from reference
annotation timing because RQ4 studies robustness to ECG waveform
artifacts.
"""

from collections import Counter
from pathlib import Path

import numpy as np

from src.data.splits import TRAIN_RECORDS
from src.data.torch_dataset import (
    CLASS_NAMES,
    ECGRRHeartbeatDataset,
    build_dataset_with_rr_from_records,
)
from src.noise.heartbeat_pipeline import (
    build_noisy_heartbeats,
)


# ---------------------------------------------------------------------
# Frozen RQ4 augmentation design
# ---------------------------------------------------------------------

TRAIN_NOISE_TYPES = (
    "bw",
    "ma",
    "em",
)

TRAIN_SNR_LEVELS_DB = (
    18.0,
    6.0,
    -6.0,
)

UNSEEN_SNR_LEVELS_DB = (
    12.0,
    0.0,
)

AUGMENTATION_CONDITIONS = tuple(
    (
        noise_type,
        snr_db,
    )
    for noise_type in TRAIN_NOISE_TYPES
    for snr_db in TRAIN_SNR_LEVELS_DB
)

EXPECTED_CONDITION_COUNT = 9


def _validate_record_ids(
    record_ids,
) -> tuple[str, ...]:
    """Guarantee that RQ4 augmentation uses training patients only."""

    record_ids = tuple(
        str(record_id)
        for record_id in record_ids
    )

    if not record_ids:
        raise ValueError(
            "At least one training record is required."
        )

    allowed_records = set(
        TRAIN_RECORDS
    )

    invalid_records = [
        record_id
        for record_id in record_ids
        if record_id not in allowed_records
    ]

    if invalid_records:
        raise ValueError(
            "Noise augmentation is restricted to TRAIN_RECORDS. "
            f"Invalid records: {invalid_records}"
        )

    return record_ids


def _dataset_to_numpy(
    dataset: ECGRRHeartbeatDataset,
):
    """
    Convert an existing clean ECG+RR dataset into NumPy arrays.

    This intentionally uses the public dataset interface instead of
    relying on internal storage details.
    """

    ecg_beats = []
    rr_features = []
    labels = []

    for index in range(
        len(dataset)
    ):

        ecg, rr, target = dataset[
            index
        ]

        ecg = (
            ecg
            .detach()
            .cpu()
            .numpy()
        )

        rr = (
            rr
            .detach()
            .cpu()
            .numpy()
        )

        target_index = int(
            target
        )

        # Dataset items have ECG shape [1, 256].
        # ECGRRHeartbeatDataset expects heartbeats without the
        # channel dimension when constructed from arrays.
        if (
            ecg.ndim == 2
            and ecg.shape[0] == 1
        ):
            ecg = ecg[0]

        if ecg.ndim != 1:
            raise RuntimeError(
                "Unexpected clean ECG shape at dataset index "
                f"{index}: {ecg.shape}"
            )

        if rr.ndim != 1:
            raise RuntimeError(
                "Unexpected RR feature shape at dataset index "
                f"{index}: {rr.shape}"
            )

        if (
            target_index < 0
            or target_index >= len(
                CLASS_NAMES
            )
        ):
            raise RuntimeError(
                f"Invalid target index: {target_index}"
            )

        ecg_beats.append(
            ecg
        )

        rr_features.append(
            rr
        )

        labels.append(
            CLASS_NAMES[
                target_index
            ]
        )

    if not ecg_beats:
        raise RuntimeError(
            "Clean training dataset is empty."
        )

    ecg_beats = np.stack(
        ecg_beats
    ).astype(
        np.float32,
        copy=False,
    )

    rr_features = np.stack(
        rr_features
    ).astype(
        np.float32,
        copy=False,
    )

    labels = np.asarray(
        labels
    )

    if not np.isfinite(
        ecg_beats
    ).all():
        raise RuntimeError(
            "Clean ECG contains NaN or infinite values."
        )

    if not np.isfinite(
        rr_features
    ).all():
        raise RuntimeError(
            "Clean RR features contain NaN or infinite values."
        )

    return (
        ecg_beats,
        rr_features,
        labels,
    )


def _record_condition_offset(
    record_id: str,
) -> int:
    """
    Return a deterministic condition offset for one record.

    Rotating the starting augmentation condition prevents every
    patient from starting with exactly the same condition while
    keeping the whole experiment fully reproducible.
    """

    try:
        numeric_id = int(
            record_id
        )

    except ValueError:
        numeric_id = sum(
            ord(character)
            for character in record_id
        )

    return (
        numeric_id
        % len(
            AUGMENTATION_CONDITIONS
        )
    )


def _build_noisy_record_copy(
    *,
    record_id: str,
    data_dir: Path,
    clean_labels: np.ndarray,
):
    """
    Build exactly one noisy counterpart for every clean beat.

    Nine complete continuous-signal corruption conditions are first
    generated for the record. Each heartbeat is then deterministically
    assigned to one of those conditions.

    This gives every clean heartbeat one noisy counterpart without
    expanding the training set nine-fold.
    """

    number_of_beats = len(
        clean_labels
    )

    if number_of_beats == 0:
        raise RuntimeError(
            f"Record {record_id} contains no training beats."
        )

    condition_outputs = {}

    corruption_metadata = {}

    # -----------------------------------------------------------------
    # Generate all nine continuous corruption conditions.
    # -----------------------------------------------------------------

    for (
        noise_type,
        snr_db,
    ) in AUGMENTATION_CONDITIONS:

        (
            noisy_beats,
            noisy_labels,
            heartbeat_metadata,
            condition_corruption,
        ) = build_noisy_heartbeats(
            record_path=(
                data_dir
                / record_id
            ),
            noise_type=noise_type,
            target_snr_db=snr_db,
        )

        noisy_beats = np.asarray(
            noisy_beats,
            dtype=np.float32,
        )

        noisy_labels = np.asarray(
            noisy_labels
        )

        # -------------------------------------------------------------
        # Hard alignment checks.
        # -------------------------------------------------------------

        if len(
            noisy_beats
        ) != number_of_beats:

            raise RuntimeError(
                "Clean/noisy heartbeat count mismatch for "
                f"record {record_id}, "
                f"{noise_type}@{snr_db:g} dB: "
                f"clean={number_of_beats}, "
                f"noisy={len(noisy_beats)}"
            )

        if len(
            heartbeat_metadata
        ) != number_of_beats:

            raise RuntimeError(
                "Heartbeat metadata count mismatch for "
                f"record {record_id}, "
                f"{noise_type}@{snr_db:g} dB."
            )

        if not np.array_equal(
            noisy_labels,
            clean_labels,
        ):
            raise RuntimeError(
                "Noise augmentation changed heartbeat labels for "
                f"record {record_id}, "
                f"{noise_type}@{snr_db:g} dB."
            )

        if not np.isfinite(
            noisy_beats
        ).all():

            raise RuntimeError(
                "Noise augmentation produced invalid ECG values for "
                f"record {record_id}, "
                f"{noise_type}@{snr_db:g} dB."
            )

        condition_outputs[
            (
                noise_type,
                snr_db,
            )
        ] = noisy_beats

        corruption_metadata[
            (
                noise_type,
                snr_db,
            )
        ] = condition_corruption

    # -----------------------------------------------------------------
    # Deterministically assign every beat to ONE augmentation condition.
    # -----------------------------------------------------------------

    offset = _record_condition_offset(
        record_id
    )

    condition_indices = (
        np.arange(
            number_of_beats,
            dtype=np.int64,
        )
        + offset
    ) % len(
        AUGMENTATION_CONDITIONS
    )

    first_condition = (
        AUGMENTATION_CONDITIONS[
            0
        ]
    )

    heartbeat_shape = (
        condition_outputs[
            first_condition
        ].shape[1:]
    )

    selected_noisy_beats = np.empty(
        (
            number_of_beats,
            *heartbeat_shape,
        ),
        dtype=np.float32,
    )

    selected_noise_types = np.empty(
        number_of_beats,
        dtype=object,
    )

    selected_snr_levels = np.empty(
        number_of_beats,
        dtype=np.float32,
    )

    condition_counts = Counter()

    for condition_index, condition in enumerate(
        AUGMENTATION_CONDITIONS
    ):

        mask = (
            condition_indices
            == condition_index
        )

        if not np.any(
            mask
        ):
            continue

        (
            noise_type,
            snr_db,
        ) = condition

        selected_noisy_beats[
            mask
        ] = condition_outputs[
            condition
        ][
            mask
        ]

        selected_noise_types[
            mask
        ] = noise_type

        selected_snr_levels[
            mask
        ] = snr_db

        condition_counts[
            f"{noise_type}@{snr_db:g}"
        ] += int(
            np.sum(
                mask
            )
        )

    if not np.isfinite(
        selected_noisy_beats
    ).all():

        raise RuntimeError(
            f"Selected noisy ECG contains invalid values "
            f"for record {record_id}."
        )

    return {
        "heartbeats": selected_noisy_beats,
        "noise_types": selected_noise_types,
        "snr_levels_db": selected_snr_levels,
        "condition_counts": dict(
            condition_counts
        ),
        "corruption_metadata": (
            corruption_metadata
        ),
    }


def build_noise_augmented_training_dataset(
    *,
    data_dir,
    record_ids=TRAIN_RECORDS,
    return_audit: bool = False,
):
    """
    Build the frozen RQ4 training dataset.

    Final composition:

        N clean beats
        +
        N noisy counterparts

    Therefore:

        50% clean
        50% noisy

    Parameters
    ----------
    data_dir:
        MIT-BIH directory, e.g. ``data/raw/mitdb``.

    record_ids:
        Must be a subset of TRAIN_RECORDS.

    return_audit:
        If True, returns ``(dataset, audit)``.
        Otherwise returns only the dataset.

    Notes
    -----
    RR features are duplicated unchanged between each clean heartbeat
    and its noisy counterpart. Only the ECG waveform is corrupted.
    """

    data_dir = Path(
        data_dir
    )

    record_ids = (
        _validate_record_ids(
            record_ids
        )
    )

    if len(
        AUGMENTATION_CONDITIONS
    ) != EXPECTED_CONDITION_COUNT:

        raise RuntimeError(
            "Unexpected RQ4 augmentation condition count: "
            f"{len(AUGMENTATION_CONDITIONS)}"
        )

    # Explicit scientific safeguard:
    # unseen evaluation SNR levels must never enter augmentation.
    train_snr_set = set(
        TRAIN_SNR_LEVELS_DB
    )

    unseen_snr_set = set(
        UNSEEN_SNR_LEVELS_DB
    )

    if (
        train_snr_set
        & unseen_snr_set
    ):
        raise RuntimeError(
            "Seen and unseen RQ4 SNR levels overlap."
        )

    all_clean_beats = []
    all_noisy_beats = []

    all_clean_rr = []
    all_noisy_rr = []

    all_clean_labels = []
    all_noisy_labels = []

    total_condition_counts = Counter()

    per_record_audit = {}

    print(
        "=" * 80
    )

    print(
        "BUILDING RQ4 NOISE-AUGMENTED TRAINING DATASET"
    )

    print(
        "=" * 80
    )

    print(
        f"Training records: {len(record_ids)}"
    )

    print(
        f"Noise types: {TRAIN_NOISE_TYPES}"
    )

    print(
        f"Training SNRs: {TRAIN_SNR_LEVELS_DB}"
    )

    print(
        f"Reserved unseen SNRs: {UNSEEN_SNR_LEVELS_DB}"
    )

    # =================================================================
    # Process each TRAIN patient independently.
    # =================================================================

    for record_id in record_ids:

        print(
            f"\nRecord {record_id}"
        )

        # -------------------------------------------------------------
        # Trusted clean ECG + raw-RR pipeline.
        # -------------------------------------------------------------

        clean_dataset = (
            build_dataset_with_rr_from_records(
                record_ids=(
                    record_id,
                ),
                data_dir=data_dir,
            )
        )

        (
            clean_beats,
            clean_rr,
            clean_labels,
        ) = _dataset_to_numpy(
            clean_dataset
        )

        # -------------------------------------------------------------
        # One noisy counterpart for every clean beat.
        # -------------------------------------------------------------

        noisy_result = (
            _build_noisy_record_copy(
                record_id=record_id,
                data_dir=data_dir,
                clean_labels=clean_labels,
            )
        )

        noisy_beats = (
            noisy_result[
                "heartbeats"
            ]
        )

        # RR timing side-information stays unchanged.
        noisy_rr = clean_rr.copy()

        noisy_labels = (
            clean_labels.copy()
        )

        if noisy_beats.shape != clean_beats.shape:
            raise RuntimeError(
                f"Clean/noisy ECG shape mismatch for "
                f"record {record_id}: "
                f"{clean_beats.shape} vs "
                f"{noisy_beats.shape}"
            )

        if noisy_rr.shape != clean_rr.shape:
            raise RuntimeError(
                f"Clean/noisy RR shape mismatch for "
                f"record {record_id}."
            )

        if not np.array_equal(
            noisy_labels,
            clean_labels,
        ):
            raise RuntimeError(
                f"Clean/noisy labels differ for "
                f"record {record_id}."
            )

        all_clean_beats.append(
            clean_beats
        )

        all_noisy_beats.append(
            noisy_beats
        )

        all_clean_rr.append(
            clean_rr
        )

        all_noisy_rr.append(
            noisy_rr
        )

        all_clean_labels.append(
            clean_labels
        )

        all_noisy_labels.append(
            noisy_labels
        )

        condition_counts = (
            noisy_result[
                "condition_counts"
            ]
        )

        total_condition_counts.update(
            condition_counts
        )

        per_record_audit[
            record_id
        ] = {
            "clean_beats": int(
                len(
                    clean_beats
                )
            ),
            "noisy_beats": int(
                len(
                    noisy_beats
                )
            ),
            "condition_counts": (
                condition_counts
            ),
        }

        print(
            f"  clean={len(clean_beats)} "
            f"| noisy={len(noisy_beats)}"
        )

    # =================================================================
    # Merge all training patients.
    # =================================================================

    clean_beats = np.concatenate(
        all_clean_beats,
        axis=0,
    )

    noisy_beats = np.concatenate(
        all_noisy_beats,
        axis=0,
    )

    clean_rr = np.concatenate(
        all_clean_rr,
        axis=0,
    )

    noisy_rr = np.concatenate(
        all_noisy_rr,
        axis=0,
    )

    clean_labels = np.concatenate(
        all_clean_labels,
        axis=0,
    )

    noisy_labels = np.concatenate(
        all_noisy_labels,
        axis=0,
    )

    clean_count = len(
        clean_beats
    )

    noisy_count = len(
        noisy_beats
    )

    if clean_count != noisy_count:
        raise RuntimeError(
            "RQ4 must contain exactly one noisy counterpart "
            "per clean training heartbeat."
        )

    # -----------------------------------------------------------------
    # Final 50/50 dataset.
    # -----------------------------------------------------------------

    augmented_beats = np.concatenate(
        (
            clean_beats,
            noisy_beats,
        ),
        axis=0,
    )

    augmented_rr = np.concatenate(
        (
            clean_rr,
            noisy_rr,
        ),
        axis=0,
    )

    augmented_labels = np.concatenate(
        (
            clean_labels,
            noisy_labels,
        ),
        axis=0,
    )

    if not (
        len(
            augmented_beats
        )
        == len(
            augmented_rr
        )
        == len(
            augmented_labels
        )
    ):
        raise RuntimeError(
            "Final augmented ECG/RR/label lengths do not match."
        )

    if not np.isfinite(
        augmented_beats
    ).all():
        raise RuntimeError(
            "Final augmented ECG contains invalid values."
        )

    if not np.isfinite(
        augmented_rr
    ).all():
        raise RuntimeError(
            "Final augmented RR contains invalid values."
        )

    dataset = ECGRRHeartbeatDataset(
        heartbeats=augmented_beats,
        rr_features=augmented_rr,
        labels=augmented_labels,
    )

    # =================================================================
    # Audit information
    # =================================================================

    original_class_counts = Counter(
        clean_labels.tolist()
    )

    final_class_counts = Counter(
        augmented_labels.tolist()
    )

    audit = {
        "record_ids": tuple(
            record_ids
        ),
        "clean_count": int(
            clean_count
        ),
        "noisy_count": int(
            noisy_count
        ),
        "total_count": int(
            len(
                dataset
            )
        ),
        "clean_fraction": float(
            clean_count
            / len(
                dataset
            )
        ),
        "noisy_fraction": float(
            noisy_count
            / len(
                dataset
            )
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
        "condition_counts": dict(
            total_condition_counts
        ),
        "original_class_counts": dict(
            original_class_counts
        ),
        "final_class_counts": dict(
            final_class_counts
        ),
        "per_record": (
            per_record_audit
        ),
    }

    print(
        "\n" + "=" * 80
    )

    print(
        "RQ4 AUGMENTED DATASET BUILT"
    )

    print(
        "=" * 80
    )

    print(
        f"Clean samples: {clean_count}"
    )

    print(
        f"Noisy samples: {noisy_count}"
    )

    print(
        f"Total samples: {len(dataset)}"
    )

    print(
        f"Clean fraction: "
        f"{audit['clean_fraction']:.3f}"
    )

    print(
        f"Noisy fraction: "
        f"{audit['noisy_fraction']:.3f}"
    )

    print(
        "\nNoise-condition counts:"
    )

    for (
        noise_type,
        snr_db,
    ) in AUGMENTATION_CONDITIONS:

        condition_name = (
            f"{noise_type}@{snr_db:g}"
        )

        print(
            f"  {condition_name:<8}: "
            f"{total_condition_counts[condition_name]}"
        )

    print(
        "\nOriginal TRAIN class counts:"
    )

    for class_name in CLASS_NAMES:
        print(
            f"  {class_name}: "
            f"{original_class_counts[class_name]}"
        )

    print(
        "\nAugmented TRAIN class counts:"
    )

    for class_name in CLASS_NAMES:
        print(
            f"  {class_name}: "
            f"{final_class_counts[class_name]}"
        )

    print(
        "=" * 80
    )

    if return_audit:
        return (
            dataset,
            audit,
        )

    return dataset