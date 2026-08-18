"""Plot RQ1/RQ2 noise-robustness results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATH = Path(
    "results/tables/rq1_noise_robustness_summary.csv"
)

PER_CLASS_PATH = Path(
    "results/tables/rq2_noise_robustness_per_class.csv"
)

OUTPUT_DIR = Path(
    "results/figures"
)

SNR_ORDER = [
    18.0,
    12.0,
    6.0,
    0.0,
    -6.0,
]

NOISE_LABELS = {
    "bw": "Baseline Wander",
    "ma": "Muscle Artifact",
    "em": "Electrode Motion",
}


def plot_macro_f1() -> None:
    """Plot overall Macro-F1 degradation versus SNR."""

    df = pd.read_csv(
        SUMMARY_PATH
    )

    clean_row = df[
        df["condition"] == "clean"
    ].iloc[0]

    clean_macro_f1 = float(
        clean_row["macro_f1"]
    )

    noisy_df = df[
        df["noise_type"].isin(
            NOISE_LABELS
        )
    ].copy()

    noisy_df[
        "target_snr_db"
    ] = pd.to_numeric(
        noisy_df["target_snr_db"]
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for noise_type, label in NOISE_LABELS.items():

        subset = noisy_df[
            noisy_df["noise_type"]
            == noise_type
        ]

        subset = (
            subset
            .set_index("target_snr_db")
            .loc[SNR_ORDER]
            .reset_index()
        )

        ax.plot(
            subset["target_snr_db"],
            subset["macro_f1"],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.axhline(
        clean_macro_f1,
        linestyle="--",
        linewidth=1.5,
        label=f"Clean ({clean_macro_f1:.3f})",
    )

    ax.set_xlabel(
        "Injected SNR (dB)"
    )

    ax.set_ylabel(
        "Macro-F1"
    )

    ax.set_title(
        "ECG Arrhythmia Classification Robustness to NSTDB Noise"
    )

    # High SNR on the left, severe noise on the right.
    ax.invert_xaxis()

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "rq1_macro_f1_vs_snr.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def plot_per_class_f1() -> None:
    """Plot class-wise F1 degradation versus SNR."""

    df = pd.read_csv(
        PER_CLASS_PATH
    )

    noisy_df = df[
        df["noise_type"].isin(
            NOISE_LABELS
        )
    ].copy()

    noisy_df[
        "target_snr_db"
    ] = pd.to_numeric(
        noisy_df["target_snr_db"]
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
        sharex=True,
    )

    axes = axes.flatten()

    class_names = [
        "N",
        "S",
        "V",
        "F",
    ]

    for ax, class_name in zip(
        axes,
        class_names,
    ):

        class_df = noisy_df[
            noisy_df["class_name"]
            == class_name
        ]

        clean_f1 = float(
            df[
                (df["condition"] == "clean")
                & (
                    df["class_name"]
                    == class_name
                )
            ]["f1"].iloc[0]
        )

        for noise_type, label in NOISE_LABELS.items():

            subset = class_df[
                class_df["noise_type"]
                == noise_type
            ]

            subset = (
                subset
                .set_index("target_snr_db")
                .loc[SNR_ORDER]
                .reset_index()
            )

            ax.plot(
                subset["target_snr_db"],
                subset["f1"],
                marker="o",
                linewidth=2,
                label=label,
            )

        ax.axhline(
            clean_f1,
            linestyle="--",
            linewidth=1.2,
            label="Clean",
        )

        ax.set_title(
            f"Class {class_name}"
        )

        ax.set_ylabel(
            "F1-score"
        )

        ax.invert_xaxis()

        ax.grid(
            alpha=0.3
        )

    for ax in axes[2:]:
        ax.set_xlabel(
            "Injected SNR (dB)"
        )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
    )

    fig.suptitle(
        "Class-wise Robustness under NSTDB Noise",
        y=1.06,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "rq2_per_class_f1_vs_snr.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_macro_f1()
    plot_per_class_f1()

    print(
        "\nNoise robustness figures: PASS"
    )


if __name__ == "__main__":
    main()