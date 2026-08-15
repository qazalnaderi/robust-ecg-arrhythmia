"""PyTorch dataset utilities for segmented ECG heartbeats."""

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.normalization import normalize_heartbeats
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