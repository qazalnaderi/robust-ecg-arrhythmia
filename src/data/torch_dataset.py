"""PyTorch dataset utilities for segmented ECG heartbeats."""

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset

from src.data.aami import map_to_aami
from src.data.normalization import normalize_heartbeats
from src.data.rr_features import compute_rr_features
from src.data.segmentation import segment_record


CLASS_NAMES = (
    "N",
    "S",
    "V",
    "F",
)

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


class ECGHeartbeatDataset(Dataset):
    """PyTorch dataset for normalized ECG heartbeat windows."""

    def __init__(
        self,
        heartbeats: np.ndarray,
        labels: np.ndarray,
    ) -> None:

        heartbeats = np.asarray(
            heartbeats,
            dtype=np.float32,
        )

        labels = np.asarray(labels)

        if heartbeats.ndim != 2:
            raise ValueError(
                "Expected heartbeats with shape "
                "(number_of_beats, heartbeat_length)."
            )

        if labels.ndim != 1:
            raise ValueError(
                "Expected one-dimensional labels."
            )

        if len(heartbeats) != len(labels):
            raise ValueError(
                "Heartbeats and labels must have the same length."
            )

        unknown_labels = set(labels) - set(CLASS_NAMES)

        if unknown_labels:
            raise ValueError(
                f"Unknown ECG labels: {sorted(unknown_labels)}"
            )

        self.inputs = torch.from_numpy(
            heartbeats
        ).unsqueeze(1)

        self.targets = torch.tensor(
            [
                CLASS_TO_INDEX[label]
                for label in labels
            ],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return (
            self.inputs[index],
            self.targets[index],
        )


class ECGRRHeartbeatDataset(Dataset):
    """
    PyTorch dataset containing ECG heartbeat windows,
    RR-interval features, and class labels.
    """

    def __init__(
        self,
        heartbeats: np.ndarray,
        rr_features: np.ndarray,
        labels: np.ndarray,
    ) -> None:

        heartbeats = np.asarray(
            heartbeats,
            dtype=np.float32,
        )

        rr_features = np.asarray(
            rr_features,
            dtype=np.float32,
        )

        labels = np.asarray(labels)

        if heartbeats.ndim != 2:
            raise ValueError(
                "Expected heartbeats with shape "
                "(number_of_beats, heartbeat_length)."
            )

        if rr_features.ndim != 2:
            raise ValueError(
                "Expected RR features with shape "
                "(number_of_beats, number_of_rr_features)."
            )

        if rr_features.shape[1] != 4:
            raise ValueError(
                "Expected exactly 4 RR features per heartbeat."
            )

        if labels.ndim != 1:
            raise ValueError(
                "Expected one-dimensional labels."
            )

        if not (
            len(heartbeats)
            == len(rr_features)
            == len(labels)
        ):
            raise ValueError(
                "Heartbeats, RR features, and labels "
                "must have the same length."
            )

        if not np.isfinite(rr_features).all():
            raise ValueError(
                "RR features contain NaN or infinite values."
            )

        unknown_labels = set(labels) - set(CLASS_NAMES)

        if unknown_labels:
            raise ValueError(
                f"Unknown ECG labels: {sorted(unknown_labels)}"
            )

        self.inputs = torch.from_numpy(
            heartbeats
        ).unsqueeze(1)

        self.rr_features = torch.from_numpy(
            rr_features
        )

        self.targets = torch.tensor(
            [
                CLASS_TO_INDEX[label]
                for label in labels
            ],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        return (
            self.inputs[index],
            self.rr_features[index],
            self.targets[index],
        )


def build_dataset_from_records(
    record_ids: Sequence[str],
    data_dir: str | Path,
) -> ECGHeartbeatDataset:
    """Build one normalized heartbeat dataset from MIT-BIH records."""

    data_dir = Path(data_dir)

    all_heartbeats = []
    all_labels = []

    for record_id in record_ids:

        heartbeats, labels, _ = segment_record(
            record_path=data_dir / record_id
        )

        if len(heartbeats) == 0:
            continue

        heartbeats = normalize_heartbeats(
            heartbeats
        )

        all_heartbeats.append(
            heartbeats
        )

        all_labels.append(
            labels
        )

    if not all_heartbeats:
        raise ValueError(
            "No heartbeats were extracted from the requested records."
        )

    heartbeats = np.concatenate(
        all_heartbeats,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    return ECGHeartbeatDataset(
        heartbeats=heartbeats,
        labels=labels,
    )


def build_dataset_with_rr_from_records(
    record_ids: Sequence[str],
    data_dir: str | Path,
) -> ECGRRHeartbeatDataset:
    """
    Build a normalized ECG heartbeat dataset with aligned RR features.

    RR intervals are calculated from the complete heartbeat sequence
    of each record. Intervals crossing ventricular flutter episodes
    are marked invalid and excluded from RR statistics.

    The resulting RR vectors are then aligned with the segmented
    N/S/V/F heartbeats using annotation sample locations.
    """

    data_dir = Path(data_dir)

    all_heartbeats = []
    all_rr_features = []
    all_labels = []

    ventricular_flutter_symbols = {
        "[",
        "!",
        "]",
    }

    for record_id in record_ids:

        record_path = data_dir / record_id

        # -----------------------------------------------------
        # 1. Segment the core N/S/V/F heartbeat windows
        # -----------------------------------------------------

        heartbeats, labels, metadata = segment_record(
            record_path=record_path
        )

        if len(heartbeats) == 0:
            continue

        heartbeats = normalize_heartbeats(
            heartbeats
        )

        # -----------------------------------------------------
        # 2. Read complete annotation sequence
        # -----------------------------------------------------

        header = wfdb.rdheader(
            str(record_path)
        )

        annotation = wfdb.rdann(
            str(record_path),
            extension="atr",
        )

        rr_beat_samples = []
        rr_annotation_indices = []

        for annotation_index, (
            sample,
            symbol,
        ) in enumerate(
            zip(
                annotation.sample,
                annotation.symbol,
            )
        ):

            if map_to_aami(symbol) is not None:

                rr_beat_samples.append(
                    int(sample)
                )

                rr_annotation_indices.append(
                    annotation_index
                )

        rr_beat_samples = np.asarray(
            rr_beat_samples,
            dtype=np.int64,
        )

        if len(rr_beat_samples) < 2:
            raise RuntimeError(
                f"Record {record_id} does not contain "
                "enough heartbeat annotations for RR features."
            )

        # -----------------------------------------------------
        # 3. Identify invalid RR intervals
        # -----------------------------------------------------

        valid_rr_mask = np.ones(
            len(rr_beat_samples) - 1,
            dtype=bool,
        )

        for rr_index in range(
            len(rr_annotation_indices) - 1
        ):

            previous_annotation_index = (
                rr_annotation_indices[
                    rr_index
                ]
            )

            next_annotation_index = (
                rr_annotation_indices[
                    rr_index + 1
                ]
            )

            symbols_between = annotation.symbol[
                previous_annotation_index + 1:
                next_annotation_index
            ]

            crosses_ventricular_flutter = any(
                symbol in ventricular_flutter_symbols
                for symbol in symbols_between
            )

            if crosses_ventricular_flutter:

                valid_rr_mask[
                    rr_index
                ] = False

        # -----------------------------------------------------
        # 4. Compute RR features using only valid intervals
        # -----------------------------------------------------

        rr_features = compute_rr_features(
            beat_samples=rr_beat_samples,
            sampling_rate=header.fs,
            valid_rr_mask=valid_rr_mask,
        )

        rr_by_sample = {
            int(sample): feature_vector
            for sample, feature_vector in zip(
                rr_beat_samples,
                rr_features,
            )
        }

        # -----------------------------------------------------
        # 5. Align RR vectors with segmented core beats
        # -----------------------------------------------------

        aligned_rr_features = []

        for beat_metadata in metadata:

            annotation_sample = int(
                beat_metadata["annotation_sample"]
            )

            if annotation_sample not in rr_by_sample:

                raise RuntimeError(
                    "Could not align RR features for "
                    f"record {record_id}, "
                    f"annotation sample {annotation_sample}."
                )

            aligned_rr_features.append(
                rr_by_sample[
                    annotation_sample
                ]
            )

        aligned_rr_features = np.asarray(
            aligned_rr_features,
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # 6. Record-level consistency check
        # -----------------------------------------------------

        if not (
            len(heartbeats)
            == len(labels)
            == len(aligned_rr_features)
        ):

            raise RuntimeError(
                "ECG/RR/label length mismatch in "
                f"record {record_id}."
            )

        all_heartbeats.append(
            heartbeats
        )

        all_rr_features.append(
            aligned_rr_features
        )

        all_labels.append(
            labels
        )

    # ---------------------------------------------------------
    # 7. Combine all requested records
    # ---------------------------------------------------------

    if not all_heartbeats:
        raise ValueError(
            "No heartbeats were extracted from "
            "the requested records."
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

    return ECGRRHeartbeatDataset(
        heartbeats=heartbeats,
        rr_features=rr_features,
        labels=labels,
    )