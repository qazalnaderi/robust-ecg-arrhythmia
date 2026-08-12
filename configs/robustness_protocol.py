
"""Core robustness experiment protocol."""

NOISE_TYPES = (
    "bw",
    "ma",
    "em",
)

EVALUATION_SNRS_DB = (
    18.0,
    12.0,
    6.0,
    0.0,
    -6.0,
)

# Used later for noise-augmented training.
TRAIN_AUGMENTATION_SNRS_DB = (
    24.0,
    12.0,
    6.0,
)

# These intensities are deliberately not used
# during noise-augmented training.
UNSEEN_EVALUATION_SNRS_DB = (
    18.0,
    0.0,
    -6.0,
)

PRIMARY_METRIC = "macro_f1"

SECONDARY_METRICS = (
    "balanced_accuracy",
    "precision_per_class",
    "recall_per_class",
    "f1_per_class",
)