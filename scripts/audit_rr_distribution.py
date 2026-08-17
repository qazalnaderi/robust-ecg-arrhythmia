"""Audit RR feature distributions in train and validation splits."""

from pathlib import Path

import numpy as np
import torch

from src.data.rr_features import RR_FEATURE_NAMES
from src.data.rr_normalization import (
    fit_rr_standardizer,
    standardize_rr_features,
)
from src.data.splits import (
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
)
from src.data.torch_dataset import (
    build_dataset_with_rr_from_records,
)


DATA_DIR = Path("data/raw/mitdb")


def print_distribution(
    name: str,
    values: torch.Tensor,
) -> None:
    """Print percentile and outlier statistics."""

    values_np = values.detach().cpu().numpy()

    percentiles = (
        0,
        1,
        5,
        50,
        95,
        99,
        100,
    )

    print(f"\n{name}")
    print("-" * 72)

    for feature_index, feature_name in enumerate(
        RR_FEATURE_NAMES
    ):
        feature_values = values_np[
            :,
            feature_index,
        ]

        q = np.percentile(
            feature_values,
            percentiles,
        )

        print(f"\n{feature_name}:")

        print(
            f"  min={q[0]:.4f}"
        )
        print(
            f"  p01={q[1]:.4f}"
        )
        print(
            f"  p05={q[2]:.4f}"
        )
        print(
            f"  median={q[3]:.4f}"
        )
        print(
            f"  p95={q[4]:.4f}"
        )
        print(
            f"  p99={q[5]:.4f}"
        )
        print(
            f"  max={q[6]:.4f}"
        )


def print_standardized_outliers(
    name: str,
    values: torch.Tensor,
) -> None:
    """Count large standardized RR values."""

    print(f"\n{name} standardized outliers")
    print("-" * 72)

    for feature_index, feature_name in enumerate(
        RR_FEATURE_NAMES
    ):
        feature_values = values[
            :,
            feature_index,
        ].abs()

        above_3 = (
            feature_values > 3
        ).sum().item()

        above_5 = (
            feature_values > 5
        ).sum().item()

        above_10 = (
            feature_values > 10
        ).sum().item()

        maximum = feature_values.max().item()

        print(
            f"{feature_name}: "
            f"|z|>3: {above_3}, "
            f"|z|>5: {above_5}, "
            f"|z|>10: {above_10}, "
            f"max|z|={maximum:.4f}"
        )


def main() -> None:

    print("=" * 72)
    print("RR DISTRIBUTION AUDIT")
    print("=" * 72)

    print("Building training dataset...")

    train_dataset = build_dataset_with_rr_from_records(
        record_ids=TRAIN_RECORDS,
        data_dir=DATA_DIR,
    )

    print("Building validation dataset...")

    validation_dataset = build_dataset_with_rr_from_records(
        record_ids=VALIDATION_RECORDS,
        data_dir=DATA_DIR,
    )

    # ---------------------------------------------------------
    # Fit ONLY on training data
    # ---------------------------------------------------------

    rr_mean, rr_std = fit_rr_standardizer(
        train_dataset.rr_features
    )

    standardized_train = standardize_rr_features(
        rr_features=train_dataset.rr_features,
        mean=rr_mean,
        std=rr_std,
    )

    standardized_validation = standardize_rr_features(
        rr_features=validation_dataset.rr_features,
        mean=rr_mean,
        std=rr_std,
    )

    # ---------------------------------------------------------
    # Raw distributions
    # ---------------------------------------------------------

    print_distribution(
        "TRAIN RAW RR (seconds)",
        train_dataset.rr_features,
    )

    print_distribution(
        "VALIDATION RAW RR (seconds)",
        validation_dataset.rr_features,
    )

    # ---------------------------------------------------------
    # Standardized extreme values
    # ---------------------------------------------------------

    print_standardized_outliers(
        "TRAIN",
        standardized_train,
    )

    print_standardized_outliers(
        "VALIDATION",
        standardized_validation,
    )

    print("\n" + "=" * 72)
    print("RR distribution audit complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()