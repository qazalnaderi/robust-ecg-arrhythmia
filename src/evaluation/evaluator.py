"""Model evaluation utilities for ECG classification."""

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    compute_classification_metrics,
)


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect true and predicted labels from a model.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        y_true and y_pred arrays.
    """

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            true_labels.append(
                targets.cpu().numpy()
            )

            predicted_labels.append(
                predictions.cpu().numpy()
            )

    if not true_labels:
        raise ValueError(
            "Evaluation dataloader is empty."
        )

    y_true = np.concatenate(
        true_labels
    )

    y_pred = np.concatenate(
        predicted_labels
    )

    return y_true, y_pred


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate an ECG model using the project's classification metrics.
    """

    y_true, y_pred = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    return compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )