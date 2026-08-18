"""Relative RR-interval feature transformation."""

import numpy as np


RELATIVE_RR_FEATURE_NAMES = (
    "pre_over_average",
    "post_over_average",
    "pre_over_local",
    "post_over_local",
)


def make_relative_rr_features(
    rr_features: np.ndarray,
) -> np.ndarray:
    """
    Convert raw RR features into patient-relative timing features.

    Expected input columns:
        0. pre_rr
        1. post_rr
        2. average_rr
        3. local_average_rr
    """

    rr_features = np.asarray(
        rr_features,
        dtype=np.float32,
    )

    if rr_features.ndim != 2:
        raise ValueError(
            "RR features must be two-dimensional."
        )

    if rr_features.shape[1] != 4:
        raise ValueError(
            "Expected exactly four raw RR features."
        )

    pre_rr = rr_features[:, 0]
    post_rr = rr_features[:, 1]
    average_rr = rr_features[:, 2]
    local_rr = rr_features[:, 3]

    if np.any(average_rr <= 0):
        raise ValueError(
            "average_rr must be positive."
        )

    if np.any(local_rr <= 0):
        raise ValueError(
            "local_average_rr must be positive."
        )

    relative_features = np.column_stack(
        (
            pre_rr / average_rr,
            post_rr / average_rr,
            pre_rr / local_rr,
            post_rr / local_rr,
        )
    ).astype(np.float32)

    if not np.isfinite(
        relative_features
    ).all():
        raise ValueError(
            "Relative RR features contain invalid values."
        )

    return relative_features