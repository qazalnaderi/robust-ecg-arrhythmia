import numpy as np

from src.data.segmentation import (
    DEFAULT_POST_SAMPLES,
    DEFAULT_PRE_SAMPLES,
    extract_heartbeat_window,
)


def test_extract_heartbeat_window_has_expected_length():
    signal = np.arange(1000)

    window = extract_heartbeat_window(
        signal=signal,
        center_sample=500,
    )

    assert window is not None
    assert len(window) == (
        DEFAULT_PRE_SAMPLES
        + DEFAULT_POST_SAMPLES
    )


def test_r_peak_is_aligned_at_expected_position():
    signal = np.arange(1000)

    window = extract_heartbeat_window(
        signal=signal,
        center_sample=500,
    )

    assert window is not None

    assert (
        window[DEFAULT_PRE_SAMPLES]
        == signal[500]
    )


def test_window_is_skipped_near_signal_start():
    signal = np.arange(1000)

    window = extract_heartbeat_window(
        signal=signal,
        center_sample=50,
    )

    assert window is None


def test_window_is_skipped_near_signal_end():
    signal = np.arange(1000)

    window = extract_heartbeat_window(
        signal=signal,
        center_sample=950,
    )

    assert window is None