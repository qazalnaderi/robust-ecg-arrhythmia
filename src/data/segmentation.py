"""Heartbeat segmentation utilities for the MIT-BIH Arrhythmia Database."""

from pathlib import Path

import numpy as np
import wfdb

from src.data.aami import map_to_aami


CORE_CLASSES = ("N", "S", "V", "F")

DEFAULT_PRE_SAMPLES = 128
DEFAULT_POST_SAMPLES = 128
DEFAULT_LEAD = "MLII"


def extract_heartbeat_window(
    signal: np.ndarray,
    center_sample: int,
    pre_samples: int = DEFAULT_PRE_SAMPLES,
    post_samples: int = DEFAULT_POST_SAMPLES,
) -> np.ndarray | None:
    """
    Extract a fixed-length ECG window around one heartbeat annotation.

    Returns None when the requested window extends outside the signal.
    """

    start = center_sample - pre_samples
    end = center_sample + post_samples

    if start < 0 or end > len(signal):
        return None

    return signal[start:end].copy()


def get_lead_signal(
    record: wfdb.Record,
    lead_name: str = DEFAULT_LEAD,
) -> np.ndarray:
    """
    Return one ECG lead by its signal name.

    Using the lead name instead of a fixed channel index avoids problems
    in records whose channel order differs from the usual MIT-BIH layout.
    """

    if lead_name not in record.sig_name:
        raise ValueError(
            f"Lead {lead_name!r} not found. "
            f"Available leads: {record.sig_name}"
        )

    lead_index = record.sig_name.index(lead_name)

    return record.p_signal[:, lead_index]


def segment_record(
    record_path: str | Path,
    lead_name: str = DEFAULT_LEAD,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Extract core AAMI heartbeat segments from one MIT-BIH record.

    Returns
    -------
    X:
        ECG heartbeat windows with shape (number_of_beats, 256).

    y:
        AAMI labels corresponding to each heartbeat.

    metadata:
        Record ID, annotation sample, and original annotation symbol
        for each extracted heartbeat.
    """

    record_path = Path(record_path)

    record = wfdb.rdrecord(
        str(record_path)
    )

    annotation = wfdb.rdann(
        str(record_path),
        extension="atr",
    )

    signal = get_lead_signal(
        record,
        lead_name=lead_name,
    )

    segments = []
    labels = []
    metadata = []

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol,
    ):
        aami_class = map_to_aami(symbol)

        if aami_class not in CORE_CLASSES:
            continue

        window = extract_heartbeat_window(
            signal=signal,
            center_sample=int(sample),
        )

        if window is None:
            continue

        segments.append(window)
        labels.append(aami_class)

        metadata.append(
            {
                "record_id": record_path.name,
                "annotation_sample": int(sample),
                "original_symbol": symbol,
                "aami_class": aami_class,
            }
        )

    if not segments:
        return (
            np.empty((0, DEFAULT_PRE_SAMPLES + DEFAULT_POST_SAMPLES)),
            np.empty((0,), dtype=str),
            [],
        )

    return (
        np.stack(segments),
        np.asarray(labels),
        metadata,
    )