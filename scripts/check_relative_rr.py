"""Sanity check relative RR features."""

from pathlib import Path

import numpy as np

from src.data.relative_rr import (
    RELATIVE_RR_FEATURE_NAMES,
    make_relative_rr_features,
)
from src.data.torch_dataset import (
    build_dataset_with_rr_from_records,
)


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "100"


def main() -> None:

    dataset = build_dataset_with_rr_from_records(
        record_ids=(TEST_RECORD,),
        data_dir=DATA_DIR,
    )

    raw_rr = (
        dataset.rr_features
        .cpu()
        .numpy()
    )

    relative_rr = make_relative_rr_features(
        raw_rr
    )

    print("=" * 72)
    print("RELATIVE RR SANITY CHECK")
    print("=" * 72)

    print(
        f"Input shape: {raw_rr.shape}"
    )

    print(
        f"Output shape: {relative_rr.shape}"
    )

    print(
        f"Features: {RELATIVE_RR_FEATURE_NAMES}"
    )

    print("\nFirst 5 rows:")

    for index in range(5):
        print(
            f"{index:02d}: "
            f"{relative_rr[index].tolist()}"
        )

    print(
        "\nFinite values: "
        f"{np.isfinite(relative_rr).all()}"
    )

    if relative_rr.shape != raw_rr.shape:
        raise RuntimeError(
            "Relative RR shape mismatch."
        )

    if not np.isfinite(
        relative_rr
    ).all():
        raise RuntimeError(
            "Invalid relative RR values."
        )

    print("\nRelative RR check: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()