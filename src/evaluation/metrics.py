"""Evaluation metrics for multi-class ECG classification."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


CLASS_NAMES = (
    "N",
    "S",
    "V",
    "F",
)

CLASS_INDICES = np.arange(
    len(CLASS_NAMES)
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Compute core ECG classification metrics.

    Returns per-class precision, recall, F1,
    support, confusion matrix, Macro-F1,
    and balanced accuracy.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if y_pred.ndim != 1:
        raise ValueError(
            "y_pred must be one-dimensional."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Evaluation arrays must not be empty."
        )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_INDICES,
            average=None,
            zero_division=0,
        )
    )

    macro_f1 = float(
        np.mean(f1)
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_INDICES,
    )

    per_class = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        per_class[class_name] = {
            "precision": float(
                precision[index]
            ),
            "recall": float(
                recall[index]
            ),
            "f1": float(
                f1[index]
            ),
            "support": int(
                support[index]
            ),
        }

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }