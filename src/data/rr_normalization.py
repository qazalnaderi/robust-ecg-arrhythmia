"""Training-only standardization utilities for RR features."""

import torch


def fit_rr_standardizer(
    rr_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate RR feature mean and standard deviation.

    The function should be fitted on the training split only.
    """

    if rr_features.ndim != 2:
        raise ValueError(
            "RR features must have shape "
            "(number_of_beats, number_of_features)."
        )

    if rr_features.shape[0] < 2:
        raise ValueError(
            "At least two RR samples are required."
        )

    mean = rr_features.mean(
        dim=0
    )

    std = rr_features.std(
        dim=0,
        unbiased=False,
    )

    if torch.any(std <= 0):
        raise ValueError(
            "RR feature standard deviation must be positive."
        )

    return mean, std


def standardize_rr_features(
    rr_features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Standardize RR features using precomputed statistics."""

    if rr_features.ndim != 2:
        raise ValueError(
            "RR features must be two-dimensional."
        )

    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(
            "RR mean and std must be one-dimensional."
        )

    if (
        rr_features.shape[1] != len(mean)
        or len(mean) != len(std)
    ):
        raise ValueError(
            "RR feature dimensions do not match "
            "the supplied mean and std."
        )

    if torch.any(std <= 0):
        raise ValueError(
            "RR standard deviation must be positive."
        )

    standardized = (
        rr_features - mean
    ) / std

    if not torch.isfinite(
        standardized
    ).all():
        raise ValueError(
            "Standardized RR features contain invalid values."
        )

    return standardized