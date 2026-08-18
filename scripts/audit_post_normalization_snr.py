"""Audit how heartbeat normalization changes effective noise severity."""

import csv
from pathlib import Path

import numpy as np
import wfdb

from src.data.aami import map_to_aami
from src.data.normalization import normalize_heartbeats
from src.data.segmentation import (
    CORE_CLASSES,
    DEFAULT_LEAD,
    extract_heartbeat_window,
    get_lead_signal,
)
from src.noise.corruption import corrupt_ecg
from src.noise.mixing import signal_power
from src.noise.nstdb import (
    VALID_NOISE_TYPES,
    load_noise_record,
)


DATA_DIR = Path("data/raw/mitdb")

OUTPUT_PATH = Path(
    "results/tables/post_normalization_snr_audit.csv"
)

TEST_RECORD = "100"

LEAD_NAME = DEFAULT_LEAD

TARGET_SNRS_DB = (
    18.0,
    12.0,
    6.0,
    0.0,
    -6.0,
)

POWER_EPSILON = 1e-12


def calculate_pair_snr_db(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
) -> float:
    """
    Calculate SNR between a clean signal and its noisy version.

    The noise component is defined as:

        noisy_signal - clean_signal
    """

    clean_signal = np.asarray(
        clean_signal,
        dtype=np.float64,
    )

    noisy_signal = np.asarray(
        noisy_signal,
        dtype=np.float64,
    )

    if clean_signal.shape != noisy_signal.shape:
        raise ValueError(
            "Clean and noisy signals must have identical shapes."
        )

    noise_component = (
        noisy_signal - clean_signal
    )

    clean_power = signal_power(
        clean_signal
    )

    noise_power = signal_power(
        noise_component
    )

    if clean_power <= POWER_EPSILON:
        return np.nan

    if noise_power <= POWER_EPSILON:
        return np.inf

    return float(
        10.0
        * np.log10(
            clean_power / noise_power
        )
    )


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    """Return robust summary statistics."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite_values = values[
        np.isfinite(values)
    ]

    if len(finite_values) == 0:
        raise RuntimeError(
            "No finite SNR measurements were available."
        )

    return {
        "median": float(
            np.median(finite_values)
        ),
        "p05": float(
            np.percentile(
                finite_values,
                5,
            )
        ),
        "p95": float(
            np.percentile(
                finite_values,
                95,
            )
        ),
        "mean": float(
            np.mean(finite_values)
        ),
    }


def extract_aligned_clean_noisy_beats(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    annotation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract exactly aligned clean and noisy heartbeat windows.
    """

    clean_beats = []
    noisy_beats = []
    labels = []

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol,
    ):

        aami_class = map_to_aami(
            symbol
        )

        if aami_class not in CORE_CLASSES:
            continue

        clean_beat = extract_heartbeat_window(
            signal=clean_signal,
            center_sample=int(sample),
        )

        noisy_beat = extract_heartbeat_window(
            signal=noisy_signal,
            center_sample=int(sample),
        )

        # Both signals use identical length and annotation positions,
        # so either both should exist or both should fail near an edge.
        if (
            clean_beat is None
            or noisy_beat is None
        ):
            continue

        clean_beats.append(
            clean_beat
        )

        noisy_beats.append(
            noisy_beat
        )

        labels.append(
            aami_class
        )

    if not clean_beats:
        raise RuntimeError(
            "No aligned heartbeat windows were extracted."
        )

    return (
        np.stack(clean_beats),
        np.stack(noisy_beats),
        np.asarray(labels),
    )


def main() -> None:

    print("=" * 88)
    print("POST-NORMALIZATION NOISE SEVERITY AUDIT")
    print("=" * 88)

    # ---------------------------------------------------------
    # 1. Load reserved sanity record
    # ---------------------------------------------------------

    record_path = (
        DATA_DIR / TEST_RECORD
    )

    record = wfdb.rdrecord(
        str(record_path)
    )

    annotation = wfdb.rdann(
        str(record_path),
        extension="atr",
    )

    clean_signal = get_lead_signal(
        record,
        lead_name=LEAD_NAME,
    )

    clean_signal = np.asarray(
        clean_signal,
        dtype=np.float64,
    )

    print(f"Record: {TEST_RECORD}")
    print(f"Lead: {LEAD_NAME}")
    print(f"Sampling rate: {record.fs} Hz")
    print(f"Continuous samples: {len(clean_signal)}")

    rows = []

    post_medians_by_noise = {
        noise_type: []
        for noise_type in VALID_NOISE_TYPES
    }

    # ---------------------------------------------------------
    # 2. Test every noise type and severity
    # ---------------------------------------------------------

    for noise_type in VALID_NOISE_TYPES:

        noise_record, noise_fs = load_noise_record(
            noise_type
        )

        if float(noise_fs) != float(record.fs):
            raise RuntimeError(
                f"Sampling-rate mismatch for {noise_type}: "
                f"ECG={record.fs}, noise={noise_fs}"
            )

        print("\n" + "-" * 88)
        print(f"Noise type: {noise_type}")
        print("-" * 88)

        for target_snr_db in TARGET_SNRS_DB:

            # -------------------------------------------------
            # Continuous corruption
            # -------------------------------------------------

            noisy_signal, corruption_metadata = corrupt_ecg(
                clean_signal=clean_signal,
                noise_record=noise_record,
                record_id=TEST_RECORD,
                noise_type=noise_type,
                target_snr_db=target_snr_db,
            )

            # -------------------------------------------------
            # Extract same heartbeat windows from both signals
            # -------------------------------------------------

            (
                clean_beats,
                noisy_beats,
                labels,
            ) = extract_aligned_clean_noisy_beats(
                clean_signal=clean_signal,
                noisy_signal=noisy_signal,
                annotation=annotation,
            )

            # -------------------------------------------------
            # Beat-level local SNR BEFORE normalization
            #
            # Note:
            # This does not have to equal the global target SNR.
            # NSTDB noise is non-stationary, so different beats
            # naturally receive different local noise severity.
            # -------------------------------------------------

            pre_snr_values = np.asarray(
                [
                    calculate_pair_snr_db(
                        clean_beat,
                        noisy_beat,
                    )
                    for clean_beat, noisy_beat
                    in zip(
                        clean_beats,
                        noisy_beats,
                    )
                ],
                dtype=np.float64,
            )

            # -------------------------------------------------
            # Apply exact classifier preprocessing
            # -------------------------------------------------

            normalized_clean = normalize_heartbeats(
                clean_beats
            )

            normalized_noisy = normalize_heartbeats(
                noisy_beats
            )

            # -------------------------------------------------
            # Effective perturbation SNR AT MODEL INPUT
            # -------------------------------------------------

            post_snr_values = np.asarray(
                [
                    calculate_pair_snr_db(
                        clean_beat,
                        noisy_beat,
                    )
                    for clean_beat, noisy_beat
                    in zip(
                        normalized_clean,
                        normalized_noisy,
                    )
                ],
                dtype=np.float64,
            )

            pre_summary = summarize(
                pre_snr_values
            )

            post_summary = summarize(
                post_snr_values
            )

            normalization_shift = (
                post_summary["median"]
                - pre_summary["median"]
            )

            post_medians_by_noise[
                noise_type
            ].append(
                post_summary["median"]
            )

            row = {
                "record_id": TEST_RECORD,
                "noise_type": noise_type,
                "target_snr_db": target_snr_db,
                "global_achieved_snr_db": (
                    corruption_metadata[
                        "achieved_snr_db"
                    ]
                ),
                "n_beats": len(labels),

                "pre_median_snr_db": (
                    pre_summary["median"]
                ),
                "pre_p05_snr_db": (
                    pre_summary["p05"]
                ),
                "pre_p95_snr_db": (
                    pre_summary["p95"]
                ),

                "post_median_snr_db": (
                    post_summary["median"]
                ),
                "post_p05_snr_db": (
                    post_summary["p05"]
                ),
                "post_p95_snr_db": (
                    post_summary["p95"]
                ),

                "normalization_median_shift_db": (
                    normalization_shift
                ),
            }

            rows.append(row)

            print(
                f"Target={target_snr_db:>6.1f} dB | "
                f"Global={corruption_metadata['achieved_snr_db']:>7.3f} | "
                f"Local pre median={pre_summary['median']:>8.3f} | "
                f"Post median={post_summary['median']:>8.3f} | "
                f"Shift={normalization_shift:>+8.3f} dB"
            )

    # ---------------------------------------------------------
    # 3. Check whether severity ordering survives preprocessing
    # ---------------------------------------------------------

    print("\n" + "=" * 88)
    print("SEVERITY ORDERING AFTER NORMALIZATION")
    print("=" * 88)

    all_orderings_preserved = True

    for noise_type in VALID_NOISE_TYPES:

        medians = post_medians_by_noise[
            noise_type
        ]

        # TARGET_SNRS_DB goes from high SNR / mild noise
        # to low SNR / severe noise.
        #
        # We therefore expect post-normalization effective
        # SNR to also decrease as corruption becomes stronger.
        preserved = all(
            medians[index]
            > medians[index + 1]
            for index in range(
                len(medians) - 1
            )
        )

        all_orderings_preserved = (
            all_orderings_preserved
            and preserved
        )

        print(
            f"{noise_type}: "
            f"{[round(value, 3) for value in medians]} "
            f"-> preserved={preserved}"
        )

    # ---------------------------------------------------------
    # 4. Save machine-readable result
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 88)

    print(
        f"Saved audit table to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "All severity orderings preserved: "
        f"{all_orderings_preserved}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()