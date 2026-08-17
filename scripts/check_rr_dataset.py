"""Sanity check the ECG + RR PyTorch dataset."""

from pathlib import Path

import torch

from src.data.torch_dataset import (
    CLASS_NAMES,
    build_dataset_with_rr_from_records,
)


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "100"


def main() -> None:

    dataset = build_dataset_with_rr_from_records(
        record_ids=(TEST_RECORD,),
        data_dir=DATA_DIR,
    )

    print("=" * 72)
    print("ECG + RR DATASET SANITY CHECK")
    print("=" * 72)

    print(f"Record: {TEST_RECORD}")
    print(f"Dataset size: {len(dataset)}")
    print(f"ECG tensor shape: {dataset.inputs.shape}")
    print(f"RR tensor shape: {dataset.rr_features.shape}")
    print(f"Target tensor shape: {dataset.targets.shape}")

    ecg, rr, target = dataset[0]

    print("\nFirst sample:")
    print(f"ECG shape: {ecg.shape}")
    print(f"RR shape: {rr.shape}")
    print(f"RR values: {rr.tolist()}")
    print(
        f"Target: {target.item()} "
        f"({CLASS_NAMES[target.item()]})"
    )

    print("\nChecks:")

    same_length = (
        len(dataset.inputs)
        == len(dataset.rr_features)
        == len(dataset.targets)
    )

    finite_ecg = torch.isfinite(
        dataset.inputs
    ).all().item()

    finite_rr = torch.isfinite(
        dataset.rr_features
    ).all().item()

    print(
        f"Matching sample counts: {same_length}"
    )

    print(
        f"Finite ECG values: {finite_ecg}"
    )

    print(
        f"Finite RR values: {finite_rr}"
    )

    if not same_length:
        raise RuntimeError(
            "Dataset components have different lengths."
        )

    if not finite_ecg:
        raise RuntimeError(
            "ECG dataset contains invalid values."
        )

    if not finite_rr:
        raise RuntimeError(
            "RR dataset contains invalid values."
        )

    print("\nDataset check: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()