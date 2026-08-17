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

    Returns
    -------
    np.ndarray
        Array of shape (n_beats, 4) containing:

        1. pre-RR interval
        2. post-RR interval
        3. recording-average RR interval
        4. local-average RR interval

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
    # 1. RR sequence
    # ---------------------------------------------------------

    rr_intervals = (
        np.diff(beat_samples) / sampling_rate
    )

    if np.any(rr_intervals <= 0):
        raise ValueError(
            "Heartbeat locations must be strictly increasing."
        )

    average_rr = float(
        np.mean(rr_intervals)
    )

    # ---------------------------------------------------------
    # 2. Allocate output
    # ---------------------------------------------------------

    features = np.empty(
        (len(beat_samples), 4),
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # 3. Compute features for every heartbeat
    # ---------------------------------------------------------

    for beat_index in range(len(beat_samples)):

        # Previous RR interval.
        # The first beat has no previous heartbeat, so we use
        # the recording average as a neutral fallback.
        if beat_index > 0:
            pre_rr = rr_intervals[
                beat_index - 1
            ]
        else:
            pre_rr = average_rr

        # Following RR interval.
        # The final beat has no next heartbeat.
        if beat_index < len(beat_samples) - 1:
            post_rr = rr_intervals[
                beat_index
            ]
        else:
            post_rr = average_rr

        # Local rhythm context:
        # use up to 5 RR intervals before and 5 after
        # the current heartbeat.
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

        if len(local_rr) > 0:
            local_average_rr = float(
                np.mean(local_rr)
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