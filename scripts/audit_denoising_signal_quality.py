"""Signal-level sanity audit for ECG denoising methods.

This script uses reserved sanity record 100 only.
It does NOT use training, validation, or final-test performance.

Goal:
1. Measure how much each denoiser changes clean ECG.
2. Measure whether each denoiser makes noisy ECG closer to clean ECG
   at the actual normalized model input.
"""

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
from src.noise.nstdb import (
    VALID_NOISE_TYPES,
    load_noise_record,
)
from src.signal_processing.bandpass import (
    bandpass_filter,
)
from src.signal_processing.wavelet import (
    wavelet_denoise,
)


DATA_DIR = Path("data/raw/mitdb")

RECORD_ID = "100"

OUTPUT_PATH = Path(
    "results/tables/denoising_signal_audit.csv"
)

SNR_LEVELS_DB = (
    18.0,
    12.0,
    6.0,
    0.0,
    -6.0,
)

EPSILON = 1e-12


def extract_core_beats(
    signal: np.ndarray,
    annotation,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Extract aligned core-class heartbeat windows."""

    beats = []
    labels = []
    samples = []

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol,
    ):

        aami_class = map_to_aami(
            symbol
        )

        if aami_class not in CORE_CLASSES:
            continue

        heartbeat = extract_heartbeat_window(
            signal=signal,
            center_sample=int(sample),
        )

        if heartbeat is None:
            continue

        beats.append(
            heartbeat
        )

        labels.append(
            aami_class
        )

        samples.append(
            int(sample)
        )

    if not beats:
        raise RuntimeError(
            "No usable heartbeats were extracted."
        )

    return (
        np.stack(beats),
        np.asarray(labels),
        np.asarray(samples),
    )


def effective_input_snr_db(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    """
    Measure how close one normalized heartbeat is to
    the clean normalized reference heartbeat.

    Higher value = closer to clean.
    """

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    candidate = np.asarray(
        candidate,
        dtype=np.float64,
    )

    if reference.shape != candidate.shape:
        raise ValueError(
            "Reference and candidate must have "
            "the same shape."
        )

    difference = (
        candidate - reference
    )

    reference_ac = (
        reference
        - np.mean(reference)
    )

    difference_ac = (
        difference
        - np.mean(difference)
    )

    reference_power = np.mean(
        reference_ac ** 2
    )

    difference_power = np.mean(
        difference_ac ** 2
    )

    if reference_power <= EPSILON:
        return np.nan

    if difference_power <= EPSILON:
        return np.inf

    return float(
        10.0
        * np.log10(
            reference_power
            / difference_power
        )
    )


def heartbeat_correlation(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    """Calculate Pearson correlation for one heartbeat."""

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    candidate = np.asarray(
        candidate,
        dtype=np.float64,
    )

    reference_std = np.std(
        reference
    )

    candidate_std = np.std(
        candidate
    )

    if (
        reference_std <= EPSILON
        or candidate_std <= EPSILON
    ):
        return np.nan

    return float(
        np.corrcoef(
            reference,
            candidate,
        )[0, 1]
    )


def summarize_pair(
    clean_beats: np.ndarray,
    candidate_beats: np.ndarray,
) -> dict[str, float]:
    """Summarize model-input similarity."""

    if clean_beats.shape != candidate_beats.shape:
        raise RuntimeError(
            "Heartbeat arrays are not aligned."
        )

    snr_values = np.asarray(
        [
            effective_input_snr_db(
                clean_beat,
                candidate_beat,
            )
            for clean_beat, candidate_beat
            in zip(
                clean_beats,
                candidate_beats,
            )
        ],
        dtype=np.float64,
    )

    correlation_values = np.asarray(
        [
            heartbeat_correlation(
                clean_beat,
                candidate_beat,
            )
            for clean_beat, candidate_beat
            in zip(
                clean_beats,
                candidate_beats,
            )
        ],
        dtype=np.float64,
    )

    finite_snr = snr_values[
        np.isfinite(snr_values)
    ]

    finite_corr = correlation_values[
        np.isfinite(correlation_values)
    ]

    if len(finite_snr) == 0:
        raise RuntimeError(
            "No finite SNR values were produced."
        )

    if len(finite_corr) == 0:
        raise RuntimeError(
            "No finite correlation values were produced."
        )

    return {
        "median_input_snr_db": float(
            np.median(finite_snr)
        ),
        "p05_input_snr_db": float(
            np.percentile(
                finite_snr,
                5,
            )
        ),
        "p95_input_snr_db": float(
            np.percentile(
                finite_snr,
                95,
            )
        ),
        "median_correlation": float(
            np.median(
                finite_corr
            )
        ),
    }


def verify_alignment(
    reference_labels: np.ndarray,
    reference_samples: np.ndarray,
    labels: np.ndarray,
    samples: np.ndarray,
    name: str,
) -> None:
    """Fail loudly if heartbeat alignment changes."""

    if not np.array_equal(
        reference_labels,
        labels,
    ):
        raise RuntimeError(
            f"Label alignment changed for {name}."
        )

    if not np.array_equal(
        reference_samples,
        samples,
    ):
        raise RuntimeError(
            f"Annotation alignment changed for {name}."
        )


def save_csv(
    rows: list[dict],
) -> None:

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


def main() -> None:

    print("=" * 88)
    print("DENOISING SIGNAL-QUALITY AUDIT")
    print("=" * 88)

    record_path = (
        DATA_DIR / RECORD_ID
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
        lead_name=DEFAULT_LEAD,
    )

    clean_signal = np.asarray(
        clean_signal,
        dtype=np.float64,
    )

    print(
        f"Record: {RECORD_ID}"
    )

    print(
        f"Lead: {DEFAULT_LEAD}"
    )

    print(
        f"Sampling rate: {record.fs} Hz"
    )

    print(
        f"Samples: {len(clean_signal)}"
    )

    # ---------------------------------------------------------
    # Clean reference
    # ---------------------------------------------------------

    (
        clean_beats,
        clean_labels,
        clean_samples,
    ) = extract_core_beats(
        clean_signal,
        annotation,
    )

    normalized_clean = normalize_heartbeats(
        clean_beats
    )

    print(
        f"Usable beats: {len(clean_beats)}"
    )

    # ---------------------------------------------------------
    # Apply denoisers to CLEAN ECG first.
    #
    # This tells us how much the denoiser itself changes
    # an already-clean model input.
    # ---------------------------------------------------------

    clean_bandpass_signal = (
        bandpass_filter(
            signal=clean_signal,
            sampling_rate=float(
                record.fs
            ),
        )
    )

    clean_wavelet_signal = (
        wavelet_denoise(
            signal=clean_signal
        )
    )

    (
        clean_bandpass_beats,
        bandpass_labels,
        bandpass_samples,
    ) = extract_core_beats(
        clean_bandpass_signal,
        annotation,
    )

    verify_alignment(
        clean_labels,
        clean_samples,
        bandpass_labels,
        bandpass_samples,
        "clean band-pass",
    )

    (
        clean_wavelet_beats,
        wavelet_labels,
        wavelet_samples,
    ) = extract_core_beats(
        clean_wavelet_signal,
        annotation,
    )

    verify_alignment(
        clean_labels,
        clean_samples,
        wavelet_labels,
        wavelet_samples,
        "clean wavelet",
    )

    normalized_clean_bandpass = (
        normalize_heartbeats(
            clean_bandpass_beats
        )
    )

    normalized_clean_wavelet = (
        normalize_heartbeats(
            clean_wavelet_beats
        )
    )

    bandpass_clean_summary = summarize_pair(
        normalized_clean,
        normalized_clean_bandpass,
    )

    wavelet_clean_summary = summarize_pair(
        normalized_clean,
        normalized_clean_wavelet,
    )

    rows = []

    for method, summary in (
        (
            "bandpass",
            bandpass_clean_summary,
        ),
        (
            "wavelet",
            wavelet_clean_summary,
        ),
    ):

        rows.append(
            {
                "record_id": RECORD_ID,
                "noise_type": "clean",
                "target_snr_db": "",
                "method": method,
                "median_input_snr_db": (
                    summary[
                        "median_input_snr_db"
                    ]
                ),
                "p05_input_snr_db": (
                    summary[
                        "p05_input_snr_db"
                    ]
                ),
                "p95_input_snr_db": (
                    summary[
                        "p95_input_snr_db"
                    ]
                ),
                "median_correlation": (
                    summary[
                        "median_correlation"
                    ]
                ),
                "improvement_vs_no_denoising_db": "",
            }
        )

    print("\nCLEAN-SIGNAL CONTROL")

    print(
        "Band-pass:"
        f" input SNR="
        f"{bandpass_clean_summary['median_input_snr_db']:.3f} dB"
        f" | median correlation="
        f"{bandpass_clean_summary['median_correlation']:.5f}"
    )

    print(
        "Wavelet:"
        f" input SNR="
        f"{wavelet_clean_summary['median_input_snr_db']:.3f} dB"
        f" | median correlation="
        f"{wavelet_clean_summary['median_correlation']:.5f}"
    )

    # ---------------------------------------------------------
    # Noisy conditions
    # ---------------------------------------------------------

    for noise_type in VALID_NOISE_TYPES:

        print("\n" + "-" * 88)
        print(
            f"Noise type: {noise_type}"
        )
        print("-" * 88)

        noise_record, noise_fs = (
            load_noise_record(
                noise_type
            )
        )

        if float(noise_fs) != float(
            record.fs
        ):
            raise RuntimeError(
                "ECG / NSTDB sampling-rate mismatch."
            )

        for target_snr_db in SNR_LEVELS_DB:

            noisy_signal, _ = corrupt_ecg(
                clean_signal=clean_signal,
                noise_record=noise_record,
                record_id=RECORD_ID,
                noise_type=noise_type,
                target_snr_db=target_snr_db,
            )

            bandpass_signal = (
                bandpass_filter(
                    signal=noisy_signal,
                    sampling_rate=float(
                        record.fs
                    ),
                )
            )

            wavelet_signal = (
                wavelet_denoise(
                    signal=noisy_signal
                )
            )

            (
                noisy_beats,
                noisy_labels,
                noisy_samples,
            ) = extract_core_beats(
                noisy_signal,
                annotation,
            )

            (
                bandpass_beats,
                bp_labels,
                bp_samples,
            ) = extract_core_beats(
                bandpass_signal,
                annotation,
            )

            (
                wavelet_beats,
                wt_labels,
                wt_samples,
            ) = extract_core_beats(
                wavelet_signal,
                annotation,
            )

            verify_alignment(
                clean_labels,
                clean_samples,
                noisy_labels,
                noisy_samples,
                "noisy ECG",
            )

            verify_alignment(
                clean_labels,
                clean_samples,
                bp_labels,
                bp_samples,
                "band-pass ECG",
            )

            verify_alignment(
                clean_labels,
                clean_samples,
                wt_labels,
                wt_samples,
                "wavelet ECG",
            )

            normalized_noisy = (
                normalize_heartbeats(
                    noisy_beats
                )
            )

            normalized_bandpass = (
                normalize_heartbeats(
                    bandpass_beats
                )
            )

            normalized_wavelet = (
                normalize_heartbeats(
                    wavelet_beats
                )
            )

            no_denoising_summary = (
                summarize_pair(
                    normalized_clean,
                    normalized_noisy,
                )
            )

            bandpass_summary = (
                summarize_pair(
                    normalized_clean,
                    normalized_bandpass,
                )
            )

            wavelet_summary = (
                summarize_pair(
                    normalized_clean,
                    normalized_wavelet,
                )
            )

            baseline_snr = (
                no_denoising_summary[
                    "median_input_snr_db"
                ]
            )

            summaries = {
                "none": (
                    no_denoising_summary
                ),
                "bandpass": (
                    bandpass_summary
                ),
                "wavelet": (
                    wavelet_summary
                ),
            }

            print(
                f"\nTarget {target_snr_db:>5.1f} dB"
            )

            for method, summary in (
                summaries.items()
            ):

                improvement = (
                    summary[
                        "median_input_snr_db"
                    ]
                    - baseline_snr
                )

                rows.append(
                    {
                        "record_id": RECORD_ID,
                        "noise_type": (
                            noise_type
                        ),
                        "target_snr_db": (
                            target_snr_db
                        ),
                        "method": method,
                        "median_input_snr_db": (
                            summary[
                                "median_input_snr_db"
                            ]
                        ),
                        "p05_input_snr_db": (
                            summary[
                                "p05_input_snr_db"
                            ]
                        ),
                        "p95_input_snr_db": (
                            summary[
                                "p95_input_snr_db"
                            ]
                        ),
                        "median_correlation": (
                            summary[
                                "median_correlation"
                            ]
                        ),
                        "improvement_vs_no_denoising_db": (
                            improvement
                        ),
                    }
                )

                print(
                    f"  {method:<9}"
                    f" effective SNR="
                    f"{summary['median_input_snr_db']:>8.3f} dB"
                    f" | improvement="
                    f"{improvement:>+8.3f} dB"
                    f" | corr="
                    f"{summary['median_correlation']:.5f}"
                )

    save_csv(
        rows
    )

    print("\n" + "=" * 88)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print(
        "Denoising signal-quality audit: COMPLETE"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()