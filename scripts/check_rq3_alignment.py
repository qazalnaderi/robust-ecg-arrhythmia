"""Integration check for the RQ3 denoising pipeline.

Uses sanity-only MIT-BIH record 100.
No validation or final-test records are involved.
"""

from pathlib import Path

import numpy as np

from src.noise.heartbeat_pipeline import (
    build_denoised_heartbeats,
    build_noisy_heartbeats,
)


DATA_DIR = Path("data/raw/mitdb")

RECORD_ID = "100"

NOISE_TYPE = "ma"
TARGET_SNR_DB = 0.0

METHODS = (
    "none",
    "bandpass",
    "wavelet",
)


def get_samples(metadata):
    return np.asarray(
        [
            row["annotation_sample"]
            for row in metadata
        ],
        dtype=np.int64,
    )


def check_corruption_metadata(
    reference: dict,
    candidate: dict,
    method: str,
) -> None:
    """Confirm that every method starts from the same corruption."""

    keys = (
        "noise_channel",
        "start_offset",
        "seed",
    )

    for key in keys:

        if reference[key] != candidate[key]:
            raise RuntimeError(
                f"Corruption changed for {method}: "
                f"{key}: "
                f"{reference[key]} != {candidate[key]}"
            )

    if not np.isclose(
        reference["achieved_snr_db"],
        candidate["achieved_snr_db"],
        atol=1e-9,
    ):
        raise RuntimeError(
            f"Achieved SNR changed for {method}."
        )


def main() -> None:

    print("=" * 80)
    print("RQ3 DENOISING ALIGNMENT CHECK")
    print("=" * 80)

    record_path = (
        DATA_DIR / RECORD_ID
    )

    # ---------------------------------------------------------
    # Existing RQ1 noisy pipeline = reference
    # ---------------------------------------------------------

    (
        reference_beats,
        reference_labels,
        reference_metadata,
        reference_corruption,
    ) = build_noisy_heartbeats(
        record_path=record_path,
        noise_type=NOISE_TYPE,
        target_snr_db=TARGET_SNR_DB,
    )

    reference_samples = get_samples(
        reference_metadata
    )

    print(
        f"Record: {RECORD_ID}"
    )

    print(
        f"Noise: {NOISE_TYPE} @ "
        f"{TARGET_SNR_DB:g} dB"
    )

    print(
        f"Reference beats: "
        f"{len(reference_beats)}"
    )

    print(
        f"Reference achieved SNR: "
        f"{reference_corruption['achieved_snr_db']:.6f} dB"
    )

    print("\nChecking methods...")

    # ---------------------------------------------------------
    # Test all RQ3 branches
    # ---------------------------------------------------------

    outputs = {}

    for method in METHODS:

        (
            beats,
            labels,
            metadata,
            corruption,
        ) = build_denoised_heartbeats(
            record_path=record_path,
            noise_type=NOISE_TYPE,
            target_snr_db=TARGET_SNR_DB,
            denoising_method=method,
        )

        samples = get_samples(
            metadata
        )

        # -----------------------------------------------------
        # Beat count
        # -----------------------------------------------------

        if len(beats) != len(
            reference_beats
        ):
            raise RuntimeError(
                f"Beat-count mismatch for {method}: "
                f"{len(beats)} vs "
                f"{len(reference_beats)}"
            )

        # -----------------------------------------------------
        # Labels
        # -----------------------------------------------------

        if not np.array_equal(
            labels,
            reference_labels,
        ):
            raise RuntimeError(
                f"Label alignment changed for {method}."
            )

        # -----------------------------------------------------
        # Annotation/sample positions
        # -----------------------------------------------------

        if not np.array_equal(
            samples,
            reference_samples,
        ):
            raise RuntimeError(
                f"Sample alignment changed for {method}."
            )

        # -----------------------------------------------------
        # Same initial corruption
        # -----------------------------------------------------

        check_corruption_metadata(
            reference=reference_corruption,
            candidate=corruption,
            method=method,
        )

        # -----------------------------------------------------
        # Shape and numerical validity
        # -----------------------------------------------------

        if beats.shape != reference_beats.shape:
            raise RuntimeError(
                f"Heartbeat shape changed for {method}: "
                f"{beats.shape} vs "
                f"{reference_beats.shape}"
            )

        if not np.isfinite(
            beats
        ).all():

            raise RuntimeError(
                f"Invalid values found for {method}."
            )

        outputs[method] = beats

        print(
            f"  {method:<9} "
            f"beats={len(beats):>5} | "
            f"labels aligned=True | "
            f"samples aligned=True | "
            f"finite=True"
        )

    # ---------------------------------------------------------
    # Critical backward-compatibility check
    #
    # denoising_method='none' must reproduce the existing
    # RQ1 noisy pipeline.
    # ---------------------------------------------------------

    none_difference = np.max(
        np.abs(
            outputs["none"]
            - reference_beats
        )
    )

    print(
        "\nMaximum difference:"
    )

    print(
        f"  existing noisy pipeline "
        f"vs RQ3 none = "
        f"{none_difference:.12f}"
    )

    if not np.allclose(
        outputs["none"],
        reference_beats,
        rtol=1e-7,
        atol=1e-8,
    ):
        raise RuntimeError(
            "RQ3 method='none' does not reproduce "
            "the existing RQ1 noisy pipeline."
        )

    # ---------------------------------------------------------
    # Confirm denoisers actually do something
    # ---------------------------------------------------------

    bandpass_change = float(
        np.mean(
            np.abs(
                outputs["bandpass"]
                - outputs["none"]
            )
        )
    )

    wavelet_change = float(
        np.mean(
            np.abs(
                outputs["wavelet"]
                - outputs["none"]
            )
        )
    )

    print(
        "\nMean absolute model-input change:"
    )

    print(
        f"  Band-pass vs none = "
        f"{bandpass_change:.8f}"
    )

    print(
        f"  Wavelet   vs none = "
        f"{wavelet_change:.8f}"
    )

    if bandpass_change <= 0.0:
        raise RuntimeError(
            "Band-pass produced no change."
        )

    if wavelet_change <= 0.0:
        raise RuntimeError(
            "Wavelet produced no change."
        )

    print("\n" + "=" * 80)

    print(
        "RQ3 denoising alignment check: PASS"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()