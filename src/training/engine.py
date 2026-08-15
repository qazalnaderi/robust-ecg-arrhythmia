"""Training and validation utilities for ECG classification."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one complete epoch.

    Returns
    -------
    float
        Mean training loss over all samples.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()

        optimizer.step()

        batch_size = inputs.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Training dataloader is empty."
        )

    return total_loss / total_samples


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate mean loss without updating model parameters.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            loss = criterion(
                logits,
                targets,
            )

            batch_size = inputs.shape[0]

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Evaluation dataloader is empty."
        )

    return total_loss / total_samples