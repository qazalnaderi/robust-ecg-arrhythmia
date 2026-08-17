"""Visual sanity check of segmented validation heartbeats."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.normalization import normalize_heartbeats
from src.data.segmentation import segment_record
from src.data.splits import VALIDATION_RECORDS
from src.data.torch_dataset import CLASS_NAMES


DATA_DIR = Path("data/raw/mitdb")
FIGURE_DIR = Path("results/figures")

SAMPLES_PER_CLASS = 4
RANDOM_SEED = 42


def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_heartbeats = []
    all_labels = []
    all_metadata = []

    # ---------------------------------------------------------
    # 1. Collect real validation heartbeats
    # ---------------------------------------------------------

    for record_id in VALIDATION_RECORDS:
        X, y, metadata = segment_record(
            record_path=DATA_DIR / record_id
        )

        all_heartbeats.append(X)
        all_labels.append(y)
        all_metadata.extend(metadata)

    X = np.concatenate(
        all_heartbeats,
        axis=0,
    )

    y = np.concatenate(
        all_labels,
        axis=0,
    )

    # ---------------------------------------------------------
    # 2. Normalize using the same pipeline used for training
    # ---------------------------------------------------------

    X_normalized = normalize_heartbeats(X)

    # ---------------------------------------------------------
    # 3. Select deterministic examples from every class
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    selected_indices = {}

    for class_name in CLASS_NAMES:
        class_indices = np.where(
            y == class_name
        )[0]

        if len(class_indices) < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Not enough examples for class {class_name}."
            )

        selected_indices[class_name] = rng.choice(
            class_indices,
            size=SAMPLES_PER_CLASS,
            replace=False,
        )

    # ---------------------------------------------------------
    # 4. Plot normalized heartbeat windows
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        len(CLASS_NAMES),
        SAMPLES_PER_CLASS,
        figsize=(14, 10),
        sharex=True,
    )

    for row_index, class_name in enumerate(CLASS_NAMES):

        for column_index, sample_index in enumerate(
            selected_indices[class_name]
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            heartbeat = X_normalized[
                sample_index
            ]

            metadata = all_metadata[
                sample_index
            ]

            axis.plot(heartbeat)

            # Annotation is at the center of the
            # 256-sample heartbeat window.
            axis.axvline(
                x=128,
                linestyle="--",
                linewidth=1,
            )

            axis.set_title(
                f"{class_name} | "
                f"record {metadata['record_id']}\n"
                f"symbol {metadata['original_symbol']}"
            )

            if column_index == 0:
                axis.set_ylabel(
                    f"Class {class_name}"
                )

    for axis in axes[-1]:
        axis.set_xlabel("Sample")

    fig.suptitle(
        "Validation Heartbeat Segmentation Sanity Check"
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        / "validation_heartbeat_samples.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("=" * 72)
    print("VALIDATION HEARTBEAT VISUAL CHECK")
    print("=" * 72)

    print(f"Total validation beats: {len(X)}")

    for class_name in CLASS_NAMES:
        print(f"\n{class_name} examples:")

        for index in selected_indices[class_name]:
            metadata = all_metadata[index]

            print(
                f"  record={metadata['record_id']}, "
                f"sample={metadata['annotation_sample']}, "
                f"symbol={metadata['original_symbol']}"
            )

    print()
    print(f"Figure saved to: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()