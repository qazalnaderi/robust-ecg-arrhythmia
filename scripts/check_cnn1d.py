"""Smoke-check the 1D CNN using real segmented MIT-BIH heartbeats."""

from pathlib import Path

import torch

from src.data.normalization import normalize_heartbeats
from src.data.segmentation import segment_record
from src.models.cnn1d import ECGCNN1D


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "100"
BATCH_SIZE = 8

CLASS_NAMES = (
    "N",
    "S",
    "V",
    "F",
)


def count_trainable_parameters(
    model: torch.nn.Module,
) -> int:
    """Count trainable model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    record_path = DATA_DIR / TEST_RECORD

    # ---------------------------------------------------------
    # 1. Extract real ECG heartbeat windows
    # ---------------------------------------------------------

    X, y, metadata = segment_record(
        record_path=record_path
    )

    # ---------------------------------------------------------
    # 2. Normalize every heartbeat independently
    # ---------------------------------------------------------

    X_normalized = normalize_heartbeats(X)

    # ---------------------------------------------------------
    # 3. Convert NumPy ECG data to a PyTorch Tensor
    # ---------------------------------------------------------

    batch = torch.from_numpy(
        X_normalized[:BATCH_SIZE]
    ).float()

    # Current shape:
    #
    # (batch, heartbeat_length)
    #
    # Example:
    # (8, 256)

    batch = batch.unsqueeze(1)

    # New shape:
    #
    # (batch, channels, heartbeat_length)
    #
    # Example:
    # (8, 1, 256)

    # ---------------------------------------------------------
    # 4. Create the CNN
    # ---------------------------------------------------------

    model = ECGCNN1D(
        num_classes=len(CLASS_NAMES)
    )

    model.eval()

    # ---------------------------------------------------------
    # 5. Forward pass
    # ---------------------------------------------------------

    with torch.no_grad():
        logits = model(batch)

    predicted_indices = torch.argmax(
        logits,
        dim=1,
    )

    predicted_classes = [
        CLASS_NAMES[index]
        for index in predicted_indices.tolist()
    ]

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("=" * 70)
    print("MIT-BIH 1D CNN FORWARD-PASS CHECK")
    print("=" * 70)

    print(f"Record: {TEST_RECORD}")
    print(f"Total segmented beats: {len(X)}")

    print("-" * 70)

    print(
        f"Segmented array shape: "
        f"{X.shape}"
    )

    print(
        f"Normalized array shape: "
        f"{X_normalized.shape}"
    )

    print(
        f"CNN input tensor shape: "
        f"{tuple(batch.shape)}"
    )

    print(
        f"CNN output tensor shape: "
        f"{tuple(logits.shape)}"
    )

    print("-" * 70)

    print(
        "Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )

    print("-" * 70)

    print("True labels:")
    print(y[:BATCH_SIZE].tolist())

    print()

    print("Untrained model predictions:")
    print(predicted_classes)

    print("=" * 70)


if __name__ == "__main__":
    main()