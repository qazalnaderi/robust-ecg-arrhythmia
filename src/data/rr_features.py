"""RR-interval feature extraction for ECG heartbeat sequences."""

import numpy as np


RR_FEATURE_NAMES = (
    "pre_rr",
    "post_rr",
    "average_rr",
    "local_average_rr",
)


def compute_rr_features(
    beat_samples: np.ndarray,
    sampling_rate: float,
    valid_rr_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute RR-interval features for a sequence of heartbeat locations.

    Parameters
    ----------
    beat_samples:
        One-dimensional array containing heartbeat annotation
        sample locations in chronological order.

    sampling_rate:
        ECG sampling frequency in Hz.

    valid_rr_mask:
        Optional boolean array of shape (n_beats - 1,).
        False marks an interval that must not be interpreted as
        a physiological beat-to-beat RR interval, for example
        an interval spanning a ventricular flutter episode.

    Returns
    -------
    np.ndarray
        Array of shape (n_beats, 4) containing:

        1. pre-RR interval
        2. post-RR interval
        3. recording-average valid RR interval
        4. local-average valid RR interval

        All values are expressed in seconds.
    """

    beat_samples = np.asarray(
        beat_samples,
        dtype=np.float64,
    )

    if beat_samples.ndim != 1:
        raise ValueError(
            "beat_samples must be one-dimensional."
        )

    if len(beat_samples) < 2:
        raise ValueError(
            "At least two heartbeat locations are required."
        )

    if sampling_rate <= 0:
        raise ValueError(
            "sampling_rate must be positive."
        )

    # ---------------------------------------------------------
    # 1. Raw RR intervals
    # ---------------------------------------------------------

    rr_intervals = (
        np.diff(beat_samples)
        / sampling_rate
    )

    if np.any(rr_intervals <= 0):
        raise ValueError(
            "Heartbeat locations must be strictly increasing."
        )

    # ---------------------------------------------------------
    # 2. Define which RR intervals are physiologically valid
    # ---------------------------------------------------------

    if valid_rr_mask is None:
        valid_rr_mask = np.ones(
            len(rr_intervals),
            dtype=bool,
        )
    else:
        valid_rr_mask = np.asarray(
            valid_rr_mask,
            dtype=bool,
        )

        if valid_rr_mask.ndim != 1:
            raise ValueError(
                "valid_rr_mask must be one-dimensional."
            )

        if len(valid_rr_mask) != len(rr_intervals):
            raise ValueError(
                "valid_rr_mask must have one entry for "
                "each RR interval."
            )

    if not np.any(valid_rr_mask):
        raise ValueError(
            "No valid RR intervals are available."
        )

    # Never let invalid intervals affect the recording average.
    average_rr = float(
        np.mean(
            rr_intervals[valid_rr_mask]
        )
    )

    # ---------------------------------------------------------
    # 3. Allocate output
    # ---------------------------------------------------------

    features = np.empty(
        (len(beat_samples), 4),
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # 4. Compute features for every heartbeat
    # ---------------------------------------------------------

    for beat_index in range(
        len(beat_samples)
    ):

        # Previous RR interval
        if (
            beat_index > 0
            and valid_rr_mask[
                beat_index - 1
            ]
        ):
            pre_rr = rr_intervals[
                beat_index - 1
            ]
        else:
            pre_rr = average_rr

        # Following RR interval
        if (
            beat_index
            < len(beat_samples) - 1
            and valid_rr_mask[
                beat_index
            ]
        ):
            post_rr = rr_intervals[
                beat_index
            ]
        else:
            post_rr = average_rr

        # -----------------------------------------------------
        # Local RR context
        #
        # Use up to 5 RR intervals before and 5 after,
        # but exclude intervals marked as invalid.
        # -----------------------------------------------------

        start = max(
            0,
            beat_index - 5,
        )

        end = min(
            len(rr_intervals),
            beat_index + 5,
        )

        local_rr = rr_intervals[
            start:end
        ]

        local_valid_mask = valid_rr_mask[
            start:end
        ]

        valid_local_rr = local_rr[
            local_valid_mask
        ]

        if len(valid_local_rr) > 0:
            local_average_rr = float(
                np.mean(valid_local_rr)
            )
        else:
            local_average_rr = average_rr

        features[beat_index] = (
            pre_rr,
            post_rr,
            average_rr,
            local_average_rr,
        )

    return features