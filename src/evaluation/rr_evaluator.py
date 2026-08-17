"""Evaluation utilities for ECG + RR classification models."""

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from src.data.torch_dataset import CLASS_NAMES


def evaluate_rr_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate an ECG + RR classifier."""

    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():

        for ecg, rr_features, targets in dataloader:

            ecg = ecg.to(device)
            rr_features = rr_features.to(device)

            logits = model(
                ecg=ecg,
                rr_features=rr_features,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_targets.append(
                targets.cpu()
            )

            all_predictions.append(
                predictions.cpu()
            )

    if not all_targets:
        raise ValueError(
            "Evaluation dataloader is empty."
        )

    targets = torch.cat(
        all_targets
    ).numpy()

    predictions = torch.cat(
        all_predictions
    ).numpy()

    labels = np.arange(
        len(CLASS_NAMES)
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        targets,
        predictions,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=labels,
            zero_division=0,
        )
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

    cm = confusion_matrix(
        targets,
        predictions,
        labels=labels,
    )

    return {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "per_class": per_class,
        "confusion_matrix": cm,
    }