from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb

from src.data.segmentation import get_lead_signal
from src.noise.corruption import corrupt_ecg
from src.noise.nstdb import (
    VALID_NOISE_TYPES,
    load_noise_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MITDB_DIR = PROJECT_ROOT / "data" / "raw" / "mitdb"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

RECORD_ID = "101"
LEAD_NAME = "MLII"

SNR_DB = 6.0

# فقط یک قسمت کوتاه ECG را برای مشاهده رسم می‌کنیم.
START_SECONDS = 300
DURATION_SECONDS = 8


def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Load clean ECG
    # -------------------------
    record = wfdb.rdrecord(
        str(MITDB_DIR / RECORD_ID)
    )

    clean_signal = get_lead_signal(
        record,
        lead_name=LEAD_NAME,
    )

    fs = float(record.fs)

    # -------------------------
    # Define viewing window
    # -------------------------
    start_sample = int(
        START_SECONDS * fs
    )

    end_sample = int(
        (START_SECONDS + DURATION_SECONDS) * fs
    )

    if end_sample > len(clean_signal):
        raise RuntimeError(
            "Requested plotting window is outside "
            "the ECG recording."
        )

    time_axis = (
        np.arange(start_sample, end_sample)
        / fs
    )

    signals_to_plot = [
        (
            "Clean ECG",
            clean_signal[
                start_sample:end_sample
            ],
        )
    ]

    # -------------------------
    # Generate noisy versions
    # -------------------------
    for noise_type in VALID_NOISE_TYPES:

        noise_record, noise_fs = (
            load_noise_record(noise_type)
        )

        if float(noise_fs) != fs:
            raise RuntimeError(
                f"Sampling-rate mismatch for "
                f"{noise_type}: "
                f"ECG={fs}, noise={noise_fs}"
            )

        noisy_signal, metadata = corrupt_ecg(
            clean_signal=clean_signal,
            noise_record=noise_record,
            record_id=RECORD_ID,
            noise_type=noise_type,
            target_snr_db=SNR_DB,
        )

        print(
            f"{noise_type}: "
            f"channel={metadata['noise_channel']}, "
            f"offset={metadata['start_offset']}, "
            f"SNR={metadata['achieved_snr_db']:.4f} dB"
        )

        signals_to_plot.append(
            (
                f"{noise_type.upper()} @ {SNR_DB:g} dB",
                noisy_signal[
                    start_sample:end_sample
                ],
            )
        )

    # -------------------------
    # Plot
    # -------------------------
    fig, axes = plt.subplots(
        len(signals_to_plot),
        1,
        figsize=(12, 9),
        sharex=True,
        sharey=True
    )

    for axis, (title, signal) in zip(
        axes,
        signals_to_plot,
    ):
        axis.plot(
            time_axis,
            signal,
            linewidth=1.0,
        )

        axis.set_title(title)
        axis.set_ylabel("Amplitude (mV)")
        axis.grid(
            alpha=0.2,
        )

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(
        f"MIT-BIH Record {RECORD_ID} — "
        f"Clean vs NSTDB Noise",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        / "noise_examples_record_101.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Figure saved to:")
    print(output_path)


if __name__ == "__main__":
    main()