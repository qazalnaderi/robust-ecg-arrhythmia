import numpy as np
import pytest

from src.evaluation.metrics import (
    CLASS_NAMES,
    compute_classification_metrics,
)


def test_perfect_predictions_have_perfect_scores():
    y_true = np.array(
        [0, 1, 2, 3]
    )

    y_pred = np.array(
        [0, 1, 2, 3]
    )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    assert metrics["macro_f1"] == pytest.approx(
        1.0
    )

    assert metrics[
        "balanced_accuracy"
    ] == pytest.approx(
        1.0
    )


def test_per_class_metrics_are_returned():
    y_true = np.array(
        [0, 1, 2, 3]
    )

    y_pred = np.array(
        [0, 1, 2, 3]
    )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    assert set(
        metrics["per_class"].keys()
    ) == set(CLASS_NAMES)


def test_confusion_matrix_has_expected_shape():
    y_true = np.array(
        [0, 1, 2, 3]
    )

    y_pred = np.array(
        [0, 1, 2, 3]
    )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    assert metrics[
        "confusion_matrix"
    ].shape == (
        4,
        4,
    )


def test_support_counts_true_samples():
    y_true = np.array(
        [0, 0, 0, 1, 2, 3]
    )

    y_pred = np.array(
        [0, 0, 1, 1, 2, 3]
    )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    assert (
        metrics["per_class"]["N"]["support"]
        == 3
    )

    assert (
        metrics["per_class"]["S"]["support"]
        == 1
    )


def test_mismatched_lengths_raise_error():
    y_true = np.array(
        [0, 1, 2]
    )

    y_pred = np.array(
        [0, 1]
    )

    with pytest.raises(ValueError):
        compute_classification_metrics(
            y_true,
            y_pred,
        )


def test_empty_arrays_raise_error():
    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(ValueError):
        compute_classification_metrics(
            y_true,
            y_pred,
        )


def test_invalid_dimensions_raise_error():
    y_true = np.zeros(
        (2, 2)
    )

    y_pred = np.zeros(4)

    with pytest.raises(ValueError):
        compute_classification_metrics(
            y_true,
            y_pred,
        )