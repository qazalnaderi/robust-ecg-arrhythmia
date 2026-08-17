"""1D CNN with RR-interval features for ECG classification."""

import torch
from torch import nn

from src.models.cnn1d import (
    ECGCNN1D,
    DEFAULT_DROPOUT,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_NUM_CLASSES,
)


DEFAULT_NUM_RR_FEATURES = 4
ECG_FEATURE_DIM = 128


class ECGCNN1DWithRR(nn.Module):
    """
    ECG classifier combining CNN morphology features
    with standardized RR-interval features.

    Expected inputs
    ---------------
    ecg:
        Shape (batch_size, 1, heartbeat_length)

    rr_features:
        Shape (batch_size, 4)

    Output
    ------
    Raw class logits with shape
    (batch_size, num_classes).
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        input_channels: int = DEFAULT_INPUT_CHANNELS,
        num_rr_features: int = DEFAULT_NUM_RR_FEATURES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()

        if num_rr_features <= 0:
            raise ValueError(
                "num_rr_features must be positive."
            )

        # -----------------------------------------------------
        # Reuse the exact CNN backbone from the clean baseline.
        # -----------------------------------------------------

        baseline_model = ECGCNN1D(
            num_classes=num_classes,
            input_channels=input_channels,
            dropout=dropout,
        )

        self.input_channels = input_channels
        self.num_rr_features = num_rr_features

        self.features = baseline_model.features
        self.pool = baseline_model.pool

        # -----------------------------------------------------
        # ECG representation (128) + RR features (4)
        # -----------------------------------------------------

        fused_feature_dim = (
            ECG_FEATURE_DIM
            + num_rr_features
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                in_features=fused_feature_dim,
                out_features=num_classes,
            ),
        )

    def forward(
        self,
        ecg: torch.Tensor,
        rr_features: torch.Tensor,
    ) -> torch.Tensor:
        """Run one ECG + RR forward pass."""

        if ecg.ndim != 3:
            raise ValueError(
                "Expected ECG shape "
                "(batch, channels, length), "
                f"got {tuple(ecg.shape)}."
            )

        if ecg.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} ECG channel(s), "
                f"got {ecg.shape[1]}."
            )

        if rr_features.ndim != 2:
            raise ValueError(
                "Expected RR shape "
                "(batch, rr_features), "
                f"got {tuple(rr_features.shape)}."
            )

        if rr_features.shape[1] != self.num_rr_features:
            raise ValueError(
                f"Expected {self.num_rr_features} RR features, "
                f"got {rr_features.shape[1]}."
            )

        if ecg.shape[0] != rr_features.shape[0]:
            raise ValueError(
                "ECG and RR batch sizes must match."
            )

        # ECG morphology branch
        ecg_features = self.features(
            ecg
        )

        ecg_features = self.pool(
            ecg_features
        )

        ecg_features = torch.flatten(
            ecg_features,
            start_dim=1,
        )

        # Combine morphology and rhythm information
        fused_features = torch.cat(
            (
                ecg_features,
                rr_features,
            ),
            dim=1,
        )

        logits = self.classifier(
            fused_features
        )

        return logits