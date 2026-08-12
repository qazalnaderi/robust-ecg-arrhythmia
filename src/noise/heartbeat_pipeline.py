"""Build normalized noisy heartbeat windows from MIT-BIH records."""

from pathlib import Path

import numpy as np
import wfdb

from src.data.aami import map_to_aami
from src.data.normalization import normalize_heartbeats
from src.data.segmentation import (
    CORE_CLASSES,
    DEFAULT_LEAD,
    extract_heartbeat_window,
    get_lead_signal,
)
from src.noise.corruption import corrupt_ecg
from src.noise.nstdb import load_noise_record


def build_noisy_heartbeats(
    record_path: str | Path,
    noise_type: str,
    target_snr_db: float,
    lead_name: str = DEFAULT_LEAD,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """
    Build normalized heartbeat windows from a continuously corrupted ECG.

    Returns
    -------
    X:
        Normalized noisy heartbeats with shape (n_beats, 256).

    y:
        AAMI labels (N, S, V, F).

    metadata:
        Metadata for each heartbeat.

    corruption_metadata:
        Information about the applied NSTDB corruption.
    """

    record_path = Path(record_path)

    # Load ECG and annotations.
    record = wfdb.rdrecord(
        str(record_path)
    )

    annotation = wfdb.rdann(
        str(record_path),
        extension="atr",
    )

    # Select MLII by name, not channel number.
    clean_signal = get_lead_signal(
        record,
        lead_name=lead_name,
    )

    # Load real NSTDB noise.
    noise_record, noise_fs = load_noise_record(
        noise_type
    )

    # ECG and noise must have the same sampling rate.
    if float(record.fs) != float(noise_fs):
        raise ValueError(
            f"Sampling-rate mismatch: "
            f"ECG={record.fs}, noise={noise_fs}"
        )

    # Corrupt the full continuous ECG first.
    noisy_signal, corruption_metadata = corrupt_ecg(
        clean_signal=clean_signal,
        noise_record=noise_record,
        record_id=record_path.name,
        noise_type=noise_type,
        target_snr_db=target_snr_db,
    )

    segments = []
    labels = []
    metadata = []

    # Use the original MIT-BIH beat annotations.
    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol,
    ):
        aami_class = map_to_aami(symbol)

        if aami_class not in CORE_CLASSES:
            continue

        heartbeat = extract_heartbeat_window(
            signal=noisy_signal,
            center_sample=int(sample),
        )

        if heartbeat is None:
            continue

        segments.append(heartbeat)
        labels.append(aami_class)

        metadata.append(
            {
                "record_id": record_path.name,
                "annotation_sample": int(sample),
                "original_symbol": symbol,
                "aami_class": aami_class,
                "noise_type": noise_type,
                "snr_db": float(target_snr_db),
            }
        )

    if not segments:
        raise RuntimeError(
            f"No usable heartbeats found in "
            f"record {record_path.name}."
        )

    X = np.stack(segments)

    # Same normalization used by the clean pipeline.
    X = normalize_heartbeats(X)

    y = np.asarray(labels)

    return (
        X,
        y,
        metadata,
        corruption_metadata,
    )