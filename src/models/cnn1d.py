"""Compact 1D CNN baseline for ECG heartbeat classification."""

import torch
from torch import nn


DEFAULT_NUM_CLASSES = 4
DEFAULT_INPUT_CHANNELS = 1
DEFAULT_DROPOUT = 0.3


class ECGCNN1D(nn.Module):
    """
    Compact 1D CNN for classifying segmented ECG heartbeats.

    Expected input shape:
        (batch_size, 1, heartbeat_length)

    Output:
        Raw class logits with shape:
        (batch_size, num_classes)
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        input_channels: int = DEFAULT_INPUT_CHANNELS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than one."
            )

        if input_channels <= 0:
            raise ValueError(
                "input_channels must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        self.input_channels = input_channels

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=128,
                out_features=num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run one forward pass through the network."""

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, channels, length), "
                f"got {tuple(x.shape)}."
            )

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channel(s), "
                f"got {x.shape[1]}."
            )

        x = self.features(x)

        x = self.pool(x)

        logits = self.classifier(x)

        return logits