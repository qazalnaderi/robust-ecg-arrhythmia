"""Smoke-check ECG heartbeat normalization on a real MIT-BIH record."""

from pathlib import Path

import numpy as np

from src.data.normalization import normalize_heartbeats
from src.data.segmentation import segment_record


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "100"


def main() -> None:
    record_path = DATA_DIR / TEST_RECORD

    X, y, metadata = segment_record(
        record_path=record_path
    )

    X_normalized = normalize_heartbeats(X)

    print("=" * 70)
    print("MIT-BIH ECG NORMALIZATION CHECK")
    print("=" * 70)

    print(f"Record: {TEST_RECORD}")

    print("-" * 70)

    print(f"Original shape: {X.shape}")
    print(f"Normalized shape: {X_normalized.shape}")

    print("-" * 70)

    if len(X) > 0:
        first_original = X[0]
        first_normalized = X_normalized[0]

        print("First heartbeat before normalization:")
        print(f"  Mean: {first_original.mean():.6f}")
        print(f"  Std:  {first_original.std():.6f}")
        print(f"  Min:  {first_original.min():.6f}")
        print(f"  Max:  {first_original.max():.6f}")

        print()

        print("First heartbeat after normalization:")
        print(f"  Mean: {first_normalized.mean():.6f}")
        print(f"  Std:  {first_normalized.std():.6f}")
        print(f"  Min:  {first_normalized.min():.6f}")
        print(f"  Max:  {first_normalized.max():.6f}")

    print("-" * 70)

    means = X_normalized.mean(axis=1)
    stds = X_normalized.std(axis=1)

    print(
        "Maximum absolute heartbeat mean: "
        f"{np.max(np.abs(means)):.8f}"
    )

    print(
        "Mean normalized heartbeat std: "
        f"{np.mean(stds):.8f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()