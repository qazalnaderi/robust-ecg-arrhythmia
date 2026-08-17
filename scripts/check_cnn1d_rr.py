"""Sanity check the ECG + RR CNN using reserved record 100."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.rr_normalization import (
    fit_rr_standardizer,
    standardize_rr_features,
)
from src.data.splits import TRAIN_RECORDS
from src.data.torch_dataset import (
    CLASS_NAMES,
    build_dataset_with_rr_from_records,
)
from src.models.cnn1d_rr import ECGCNN1DWithRR


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "100"
BATCH_SIZE = 32


def main() -> None:

    print("=" * 72)
    print("ECG + RR CNN SANITY CHECK")
    print("=" * 72)

    # ---------------------------------------------------------
    # 1. Build the sanity-only Record 100 dataset
    # ---------------------------------------------------------

    print("Building sanity dataset...")

    dataset = build_dataset_with_rr_from_records(
        record_ids=(TEST_RECORD,),
        data_dir=DATA_DIR,
    )

    # ---------------------------------------------------------
    # 2. Fit RR normalization statistics on TRAIN ONLY
    # ---------------------------------------------------------

    print("Building training dataset for RR normalization...")

    train_dataset = build_dataset_with_rr_from_records(
        record_ids=TRAIN_RECORDS,
        data_dir=DATA_DIR,
    )

    rr_mean, rr_std = fit_rr_standardizer(
        train_dataset.rr_features
    )

    # Apply TRAIN statistics to Record 100.
    dataset.rr_features = standardize_rr_features(
        rr_features=dataset.rr_features,
        mean=rr_mean,
        std=rr_std,
    )

    # ---------------------------------------------------------
    # 3. Create DataLoader
    # ---------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # 4. Create ECG + RR model
    # ---------------------------------------------------------

    model = ECGCNN1DWithRR(
        num_classes=len(CLASS_NAMES)
    )

    model.eval()

    # ---------------------------------------------------------
    # 5. Get one real batch
    # ---------------------------------------------------------

    ecg, rr_features, targets = next(
        iter(dataloader)
    )

    # ---------------------------------------------------------
    # 6. Forward pass
    # ---------------------------------------------------------

    with torch.no_grad():
        logits = model(
            ecg=ecg,
            rr_features=rr_features,
        )

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    # ---------------------------------------------------------
    # 7. Report shapes and basic validity
    # ---------------------------------------------------------

    print(f"\nRecord: {TEST_RECORD}")

    print(
        f"ECG batch shape: {ecg.shape}"
    )

    print(
        f"RR batch shape: {rr_features.shape}"
    )

    print(
        f"Target shape: {targets.shape}"
    )

    print(
        f"Logit shape: {logits.shape}"
    )

    print(
        f"Prediction shape: {predictions.shape}"
    )

    print(
        f"Finite ECG values: "
        f"{torch.isfinite(ecg).all().item()}"
    )

    print(
        f"Finite RR values: "
        f"{torch.isfinite(rr_features).all().item()}"
    )

    print(
        f"Finite logits: "
        f"{torch.isfinite(logits).all().item()}"
    )

    # ---------------------------------------------------------
    # 8. Hard checks
    # ---------------------------------------------------------

    expected_ecg_shape = (
        BATCH_SIZE,
        1,
        256,
    )

    expected_rr_shape = (
        BATCH_SIZE,
        4,
    )

    expected_logit_shape = (
        BATCH_SIZE,
        len(CLASS_NAMES),
    )

    if tuple(ecg.shape) != expected_ecg_shape:
        raise RuntimeError(
            f"Unexpected ECG shape: {tuple(ecg.shape)}"
        )

    if tuple(rr_features.shape) != expected_rr_shape:
        raise RuntimeError(
            f"Unexpected RR shape: {tuple(rr_features.shape)}"
        )

    if tuple(logits.shape) != expected_logit_shape:
        raise RuntimeError(
            f"Unexpected logits shape: {tuple(logits.shape)}"
        )

    if not torch.isfinite(ecg).all():
        raise RuntimeError(
            "ECG batch contains invalid values."
        )

    if not torch.isfinite(rr_features).all():
        raise RuntimeError(
            "RR batch contains invalid values."
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Model produced invalid logits."
        )

    print("\nECG + RR CNN check: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()