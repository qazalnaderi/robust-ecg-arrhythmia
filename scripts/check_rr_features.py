"""Sanity check RR-interval features on the reserved record 100."""

from pathlib import Path

import numpy as np
import wfdb

from src.data.aami import map_to_aami
from src.data.rr_features import (
    RR_FEATURE_NAMES,
    compute_rr_features,
)


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "100"


def main() -> None:
    record_path = DATA_DIR / TEST_RECORD

    header = wfdb.rdheader(
        str(record_path)
    )

    annotation = wfdb.rdann(
        str(record_path),
        extension="atr",
    )

    # ---------------------------------------------------------
    # Keep heartbeat annotations only
    # ---------------------------------------------------------

    beat_samples = []

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol,
    ):
        if map_to_aami(symbol) is not None:
            beat_samples.append(sample)

    beat_samples = np.asarray(
        beat_samples,
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # Compute RR features
    # ---------------------------------------------------------

    features = compute_rr_features(
        beat_samples=beat_samples,
        sampling_rate=header.fs,
    )

    print("=" * 72)
    print("RR FEATURE SANITY CHECK")
    print("=" * 72)

    print(f"Record: {TEST_RECORD}")
    print(f"Sampling rate: {header.fs} Hz")
    print(f"Heartbeat annotations: {len(beat_samples)}")
    print(f"Feature matrix shape: {features.shape}")

    print("\nFeature names:")
    print(RR_FEATURE_NAMES)

    print("\nFirst 10 feature rows:")

    for index in range(
        min(10, len(features))
    ):
        print(
            f"{index:02d}: "
            f"pre={features[index, 0]:.4f}, "
            f"post={features[index, 1]:.4f}, "
            f"avg={features[index, 2]:.4f}, "
            f"local={features[index, 3]:.4f}"
        )

    print("\nFeature ranges:")

    for feature_index, feature_name in enumerate(
        RR_FEATURE_NAMES
    ):
        values = features[:, feature_index]

        print(
            f"{feature_name}: "
            f"min={values.min():.4f}, "
            f"mean={values.mean():.4f}, "
            f"max={values.max():.4f}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()