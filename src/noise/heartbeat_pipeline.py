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
from src.signal_processing.bandpass import bandpass_filter
from src.signal_processing.wavelet import wavelet_denoise


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


VALID_DENOISING_METHODS = (
    "none",
    "bandpass",
    "wavelet",
)


def _apply_denoising(
    signal: np.ndarray,
    sampling_rate: float,
    method: str,
) -> np.ndarray:
    """Apply one frozen denoising method to continuous ECG."""

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    if signal.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional ECG signal."
        )

    if signal.size == 0:
        raise ValueError(
            "ECG signal must not be empty."
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            "ECG signal contains NaN or infinite values."
        )

    if method not in VALID_DENOISING_METHODS:
        raise ValueError(
            f"Unknown denoising method: {method}. "
            f"Expected one of {VALID_DENOISING_METHODS}."
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

    raise RuntimeError(
        "Unexpected denoising branch."
    )


def build_denoised_heartbeats(
    record_path,
    noise_type: str,
    target_snr_db: float,
    denoising_method: str,
    lead_name: str = DEFAULT_LEAD,
):
    """
    Corrupt continuous ECG, optionally denoise it,
    then extract and normalize aligned heartbeat windows.

    Noise is always injected BEFORE denoising and segmentation.
    """

    record_path = Path(
        record_path
    )

    record_id = record_path.name

    # ---------------------------------------------------------
    # 1. Load ECG and reference annotations
    # ---------------------------------------------------------

    record = wfdb.rdrecord(
        str(record_path)
    )

    annotation = wfdb.rdann(
        str(record_path),
        extension="atr",
    )

    clean_signal = get_lead_signal(
        record,
        lead_name=lead_name,
    )

    clean_signal = np.asarray(
        clean_signal,
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # 2. Load NSTDB artifact
    # ---------------------------------------------------------

    noise_record, noise_fs = load_noise_record(
        noise_type
    )

    if float(noise_fs) != float(record.fs):
        raise RuntimeError(
            "Sampling-rate mismatch: "
            f"ECG={record.fs}, noise={noise_fs}"
        )

    # ---------------------------------------------------------
    # 3. Inject controlled noise into CONTINUOUS ECG
    # ---------------------------------------------------------

    noisy_signal, corruption_metadata = corrupt_ecg(
        clean_signal=clean_signal,
        noise_record=noise_record,
        record_id=record_id,
        noise_type=noise_type,
        target_snr_db=target_snr_db,
    )

    # ---------------------------------------------------------
    # 4. Apply frozen denoising method
    # ---------------------------------------------------------

    processed_signal = _apply_denoising(
        signal=noisy_signal,
        sampling_rate=float(record.fs),
        method=denoising_method,
    )

    if processed_signal.shape != clean_signal.shape:
        raise RuntimeError(
            "Denoising changed continuous ECG length."
        )

    if not np.isfinite(processed_signal).all():
        raise RuntimeError(
            "Denoising produced NaN or infinite values."
        )

    # ---------------------------------------------------------
    # 5. Extract exactly the same core-class heartbeats
    # ---------------------------------------------------------

    heartbeats = []
    labels = []
    metadata = []

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

        metadata.append(
        {
            "record_id": record_id,
            "annotation_sample": int(sample),
            "original_symbol": symbol,
            "aami_class": aami_class,
            "noise_type": noise_type,
            "snr_db": float(target_snr_db),
            "denoising_method": denoising_method,
        }
    )

    if not heartbeats:
        raise RuntimeError(
            f"No usable heartbeats extracted from record {record_id}."
        )

    heartbeats = np.stack(
        heartbeats
    )

    labels = np.asarray(
        labels
    )

    # ---------------------------------------------------------
    # 6. Exact classifier preprocessing
    # ---------------------------------------------------------

    normalized_heartbeats = normalize_heartbeats(
        heartbeats
    )

    if not np.isfinite(
        normalized_heartbeats
    ).all():

        raise RuntimeError(
            "Heartbeat normalization produced invalid values."
        )

    return (
        normalized_heartbeats,
        labels,
        metadata,
        corruption_metadata,
    )