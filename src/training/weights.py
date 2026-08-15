"""Class-weight utilities for imbalanced ECG classification."""

import torch


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training labels.

    Weight for each class:
        total_samples / (num_classes * class_count)
    """

    if targets.ndim != 1:
        raise ValueError(
            "Targets must be one-dimensional."
        )

    if num_classes <= 1:
        raise ValueError(
            "num_classes must be greater than one."
        )

    counts = torch.bincount(
        targets,
        minlength=num_classes,
    ).float()

    if torch.any(counts == 0):
        raise ValueError(
            "Every class must have at least one training sample."
        )

    total_samples = counts.sum()

    weights = (
        total_samples
        / (num_classes * counts)
    )

    return weights