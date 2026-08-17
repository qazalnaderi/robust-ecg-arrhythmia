"""Check train-only RR feature standardization."""

from pathlib import Path

import torch

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


def main() -> None:

    print("=" * 72)
    print("RR NORMALIZATION CHECK")
    print("=" * 72)

    # ---------------------------------------------------------
    # 1. Build training RR dataset
    # ---------------------------------------------------------

    print("Building training dataset...")

    train_dataset = build_dataset_with_rr_from_records(
        record_ids=TRAIN_RECORDS,
        data_dir=DATA_DIR,
    )

    # ---------------------------------------------------------
    # 2. Fit normalization ONLY on training data
    # ---------------------------------------------------------

    rr_mean, rr_std = fit_rr_standardizer(
        train_dataset.rr_features
    )

    print("\nTraining RR statistics:")

    for index in range(
        train_dataset.rr_features.shape[1]
    ):
        print(
            f"Feature {index}: "
            f"mean={rr_mean[index].item():.6f}, "
            f"std={rr_std[index].item():.6f}"
        )

    standardized_train = standardize_rr_features(
        rr_features=train_dataset.rr_features,
        mean=rr_mean,
        std=rr_std,
    )

    print("\nStandardized TRAIN statistics:")

    train_mean = standardized_train.mean(
        dim=0
    )

    train_std = standardized_train.std(
        dim=0,
        unbiased=False,
    )

    for index in range(
        standardized_train.shape[1]
    ):
        print(
            f"Feature {index}: "
            f"mean={train_mean[index].item():.6f}, "
            f"std={train_std[index].item():.6f}"
        )

    # ---------------------------------------------------------
    # 3. Apply TRAIN statistics to validation
    # ---------------------------------------------------------

    print("\nBuilding validation dataset...")

    validation_dataset = build_dataset_with_rr_from_records(
        record_ids=VALIDATION_RECORDS,
        data_dir=DATA_DIR,
    )

    standardized_validation = standardize_rr_features(
        rr_features=validation_dataset.rr_features,
        mean=rr_mean,
        std=rr_std,
    )

    print("\nStandardized VALIDATION statistics:")
    print(
        "Note: these are not expected to be "
        "exactly mean=0 and std=1."
    )

    validation_mean = (
        standardized_validation.mean(dim=0)
    )

    validation_std = (
        standardized_validation.std(
            dim=0,
            unbiased=False,
        )
    )

    for index in range(
        standardized_validation.shape[1]
    ):
        print(
            f"Feature {index}: "
            f"mean={validation_mean[index].item():.6f}, "
            f"std={validation_std[index].item():.6f}"
        )

    # ---------------------------------------------------------
    # 4. Hard checks
    # ---------------------------------------------------------

    expected_zero = torch.zeros_like(
        train_mean
    )

    expected_one = torch.ones_like(
        train_std
    )

    if not torch.allclose(
        train_mean,
        expected_zero,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Standardized training RR means are not near zero."
        )

    if not torch.allclose(
        train_std,
        expected_one,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Standardized training RR standard deviations "
            "are not near one."
        )

    if not torch.isfinite(
        standardized_validation
    ).all():
        raise RuntimeError(
            "Validation RR features contain invalid values."
        )

    print("\nRR normalization check: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()