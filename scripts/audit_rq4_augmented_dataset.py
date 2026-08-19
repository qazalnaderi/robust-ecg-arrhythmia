"""Audit the RQ4 noise-augmented training dataset.

This is a scientific/reproducibility audit, not a temporary contract test.

Checks:
- TRAIN patients only
- exact 50/50 clean/noisy composition
- one noisy counterpart per clean heartbeat
- RR features preserved between clean/noisy counterparts
- labels preserved between clean/noisy counterparts
- class distribution exactly doubled
- only 18, 6, -6 dB used during training
- 12 and 0 dB remain unseen
- nine augmentation conditions are approximately balanced
"""

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.data.noise_augmented_dataset import (
    AUGMENTATION_CONDITIONS,
    TRAIN_NOISE_TYPES,
    TRAIN_SNR_LEVELS_DB,
    UNSEEN_SNR_LEVELS_DB,
    build_noise_augmented_training_dataset,
)
from src.data.splits import TRAIN_RECORDS
from src.data.torch_dataset import CLASS_NAMES


DATA_DIR = Path(
    "data/raw/mitdb"
)


def main() -> None:

    print("=" * 88)
    print("RQ4 NOISE-AUGMENTED TRAINING DATASET AUDIT")
    print("=" * 88)

    # -----------------------------------------------------------------
    # 1. Build the real RQ4 training dataset
    # -----------------------------------------------------------------

    dataset, audit = (
        build_noise_augmented_training_dataset(
            data_dir=DATA_DIR,
            record_ids=TRAIN_RECORDS,
            return_audit=True,
        )
    )

    clean_count = audit[
        "clean_count"
    ]

    noisy_count = audit[
        "noisy_count"
    ]

    total_count = audit[
        "total_count"
    ]

    print("\n" + "=" * 88)
    print("AUDIT CHECKS")
    print("=" * 88)

    # -----------------------------------------------------------------
    # 2. TRAIN patients only
    # -----------------------------------------------------------------

    actual_records = set(
        audit["record_ids"]
    )

    expected_records = set(
        TRAIN_RECORDS
    )

    train_records_ok = (
        actual_records
        == expected_records
    )

    print(
        f"TRAIN records only: "
        f"{train_records_ok}"
    )

    if not train_records_ok:
        raise RuntimeError(
            "RQ4 dataset does not contain exactly TRAIN_RECORDS."
        )

    # -----------------------------------------------------------------
    # 3. Exact 50/50 clean-noisy composition
    # -----------------------------------------------------------------

    counts_match = (
        clean_count
        == noisy_count
    )

    total_matches = (
        total_count
        == clean_count + noisy_count
        == len(dataset)
    )

    fractions_ok = (
        np.isclose(
            audit["clean_fraction"],
            0.5,
        )
        and np.isclose(
            audit["noisy_fraction"],
            0.5,
        )
    )

    print(
        f"Clean/noisy counts equal: "
        f"{counts_match}"
    )

    print(
        f"Dataset length consistent: "
        f"{total_matches}"
    )

    print(
        f"50/50 composition: "
        f"{fractions_ok}"
    )

    if not counts_match:
        raise RuntimeError(
            "Clean/noisy counts are not equal."
        )

    if not total_matches:
        raise RuntimeError(
            "Dataset total length is inconsistent."
        )

    if not fractions_ok:
        raise RuntimeError(
            "RQ4 dataset is not exactly 50/50 clean/noisy."
        )

    # -----------------------------------------------------------------
    # 4. Verify seen / unseen SNR separation
    # -----------------------------------------------------------------

    train_snrs = set(
        float(value)
        for value in TRAIN_SNR_LEVELS_DB
    )

    unseen_snrs = set(
        float(value)
        for value in UNSEEN_SNR_LEVELS_DB
    )

    overlap = (
        train_snrs
        & unseen_snrs
    )

    snr_separation_ok = (
        len(overlap)
        == 0
    )

    print(
        f"Seen/unseen SNR separation: "
        f"{snr_separation_ok}"
    )

    print(
        f"  Seen during Train: "
        f"{sorted(train_snrs, reverse=True)}"
    )

    print(
        f"  Reserved unseen: "
        f"{sorted(unseen_snrs, reverse=True)}"
    )

    if not snr_separation_ok:
        raise RuntimeError(
            "Seen and unseen RQ4 SNR levels overlap."
        )

    # -----------------------------------------------------------------
    # 5. Check augmentation condition counts
    # -----------------------------------------------------------------

    condition_counts = audit[
        "condition_counts"
    ]

    expected_condition_names = {
        f"{noise_type}@{snr_db:g}"
        for (
            noise_type,
            snr_db,
        ) in AUGMENTATION_CONDITIONS
    }

    actual_condition_names = set(
        condition_counts.keys()
    )

    condition_names_ok = (
        actual_condition_names
        == expected_condition_names
    )

    print(
        f"Exactly 9 expected noise conditions: "
        f"{condition_names_ok}"
    )

    if not condition_names_ok:
        raise RuntimeError(
            "Unexpected RQ4 augmentation conditions."
        )

    condition_total = sum(
        condition_counts.values()
    )

    condition_total_ok = (
        condition_total
        == noisy_count
    )

    print(
        f"Condition counts sum to noisy count: "
        f"{condition_total_ok}"
    )

    if not condition_total_ok:
        raise RuntimeError(
            "Noise-condition counts do not sum "
            "to total noisy samples."
        )

    condition_values = list(
        condition_counts.values()
    )

    min_condition = min(
        condition_values
    )

    max_condition = max(
        condition_values
    )

    condition_spread = (
        max_condition
        - min_condition
    )

    # Because assignment is round-robin within each patient,
    # each patient can contribute at most one extra beat to
    # some conditions. Therefore a spread larger than the
    # number of training records would indicate a bug.
    balance_tolerance = len(
        TRAIN_RECORDS
    )

    conditions_balanced = (
        condition_spread
        <= balance_tolerance
    )

    print(
        f"Noise conditions approximately balanced: "
        f"{conditions_balanced}"
    )

    print(
        f"  Min condition count: "
        f"{min_condition}"
    )

    print(
        f"  Max condition count: "
        f"{max_condition}"
    )

    print(
        f"  Spread: "
        f"{condition_spread}"
    )

    if not conditions_balanced:
        raise RuntimeError(
            "RQ4 augmentation conditions are unexpectedly imbalanced."
        )

    # -----------------------------------------------------------------
    # 6. Class counts must exactly double
    # -----------------------------------------------------------------

    original_counts = Counter(
        audit[
            "original_class_counts"
        ]
    )

    final_counts = Counter(
        audit[
            "final_class_counts"
        ]
    )

    class_counts_ok = True

    print(
        "\nClass-count verification:"
    )

    for class_name in CLASS_NAMES:

        original = int(
            original_counts[
                class_name
            ]
        )

        final = int(
            final_counts[
                class_name
            ]
        )

        expected_final = (
            2 * original
        )

        passed = (
            final
            == expected_final
        )

        class_counts_ok = (
            class_counts_ok
            and passed
        )

        print(
            f"  {class_name}: "
            f"{original} -> {final} "
            f"(expected {expected_final}) "
            f"| PASS={passed}"
        )

    if not class_counts_ok:
        raise RuntimeError(
            "Augmentation changed class proportions."
        )

    # -----------------------------------------------------------------
    # 7. ECG / RR / target interface checks
    # -----------------------------------------------------------------

    indices_to_check = (
        0,
        clean_count // 2,
        clean_count - 1,
        clean_count,
        clean_count + (
            clean_count // 2
        ),
        total_count - 1,
    )

    interface_ok = True

    for index in indices_to_check:

        ecg, rr, target = dataset[
            index
        ]

        if ecg.ndim != 2:
            interface_ok = False

        if rr.ndim != 1:
            interface_ok = False

        if not torch.isfinite(
            ecg
        ).all():
            interface_ok = False

        if not torch.isfinite(
            rr
        ).all():
            interface_ok = False

        target_value = int(
            target
        )

        if not (
            0
            <= target_value
            < len(CLASS_NAMES)
        ):
            interface_ok = False

    print(
        f"\nDataset item interface valid: "
        f"{interface_ok}"
    )

    if not interface_ok:
        raise RuntimeError(
            "RQ4 dataset item interface is invalid."
        )

    # -----------------------------------------------------------------
    # 8. Critical paired clean/noisy alignment check
    #
    # Dataset layout:
    #
    #   [all clean samples]
    #   [all noisy counterparts]
    #
    # Therefore sample i and sample i + clean_count must have
    # identical RR and label.
    # -----------------------------------------------------------------

    rr_alignment_ok = True
    label_alignment_ok = True

    max_rr_difference = 0.0

    for index in range(
        clean_count
    ):

        (
            clean_ecg,
            clean_rr,
            clean_target,
        ) = dataset[
            index
        ]

        (
            noisy_ecg,
            noisy_rr,
            noisy_target,
        ) = dataset[
            index
            + clean_count
        ]

        rr_difference = float(
            torch.max(
                torch.abs(
                    clean_rr
                    - noisy_rr
                )
            )
        )

        max_rr_difference = max(
            max_rr_difference,
            rr_difference,
        )

        if not torch.equal(
            clean_rr,
            noisy_rr,
        ):
            rr_alignment_ok = False
            break

        if int(
            clean_target
        ) != int(
            noisy_target
        ):
            label_alignment_ok = False
            break

    print(
        f"RR preserved clean -> noisy: "
        f"{rr_alignment_ok}"
    )

    print(
        f"Label preserved clean -> noisy: "
        f"{label_alignment_ok}"
    )

    print(
        f"Maximum paired RR difference: "
        f"{max_rr_difference:.12f}"
    )

    if not rr_alignment_ok:
        raise RuntimeError(
            "RR features changed between clean "
            "and noisy counterparts."
        )

    if not label_alignment_ok:
        raise RuntimeError(
            "Labels changed between clean "
            "and noisy counterparts."
        )

    # -----------------------------------------------------------------
    # 9. Per-record consistency
    # -----------------------------------------------------------------

    per_record_ok = True

    for (
        record_id,
        values,
    ) in audit[
        "per_record"
    ].items():

        if (
            values[
                "clean_beats"
            ]
            != values[
                "noisy_beats"
            ]
        ):
            per_record_ok = False

    print(
        f"One noisy counterpart per beat "
        f"within every patient: "
        f"{per_record_ok}"
    )

    if not per_record_ok:
        raise RuntimeError(
            "Per-record clean/noisy counts differ."
        )

    # -----------------------------------------------------------------
    # Final human-readable summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 88)
    print("RQ4 AUGMENTED DATASET SUMMARY")
    print("=" * 88)

    print(
        f"Clean samples : {clean_count}"
    )

    print(
        f"Noisy samples : {noisy_count}"
    )

    print(
        f"Total samples : {total_count}"
    )

    print(
        "\nNoise-condition distribution:"
    )

    for (
        noise_type,
        snr_db,
    ) in AUGMENTATION_CONDITIONS:

        name = (
            f"{noise_type}@{snr_db:g}"
        )

        print(
            f"  {name:<8}: "
            f"{condition_counts[name]}"
        )

    print(
        "\nAll RQ4 dataset audit checks: PASS"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()