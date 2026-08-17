"""Training utilities for ECG + RR classification models."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_rr_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train an ECG + RR model for one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for ecg, rr_features, targets in dataloader:

        ecg = ecg.to(device)
        rr_features = rr_features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(
            ecg=ecg,
            rr_features=rr_features,
        )

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Training dataloader is empty."
        )

    return total_loss / total_samples


def evaluate_rr_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate ECG + RR loss without updating model weights."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for ecg, rr_features, targets in dataloader:

            ecg = ecg.to(device)
            rr_features = rr_features.to(device)
            targets = targets.to(device)

            logits = model(
                ecg=ecg,
                rr_features=rr_features,
            )

            loss = criterion(
                logits,
                targets,
            )

            batch_size = targets.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Evaluation dataloader is empty."
        )

    return total_loss / total_samples