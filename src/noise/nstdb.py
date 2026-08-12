from pathlib import Path

import numpy as np
import wfdb


DEFAULT_NSTDB_DIR = Path("data/raw/nstdb")

VALID_NOISE_TYPES = ("bw", "ma", "em")


def load_noise_record(
    noise_type: str,
    data_dir: Path = DEFAULT_NSTDB_DIR,
) -> tuple[np.ndarray, float]:
    """
    Load one NSTDB noise recording.

    Parameters
    ----------
    noise_type:
        One of: "bw", "ma", "em".

    data_dir:
        Directory containing the NSTDB files.

    Returns
    -------
    signal:
        Noise recording with shape (samples, channels).

    sampling_rate:
        Sampling frequency in Hz.
    """

    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(
            f"Unknown noise type: {noise_type}. "
            f"Expected one of {VALID_NOISE_TYPES}."
        )

    record_path = data_dir / noise_type

    if not record_path.with_suffix(".hea").exists():
        raise FileNotFoundError(
            f"NSTDB record not found: {record_path}"
        )

    record = wfdb.rdrecord(str(record_path))

    return record.p_signal, float(record.fs)