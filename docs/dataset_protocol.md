# MIT-BIH Dataset Protocol

## Dataset

The core classification dataset is the MIT-BIH Arrhythmia Database.

The complete database contains 48 ECG records. All 48 records are included
in the initial dataset audit.

## AAMI Mapping

MIT-BIH beat annotations are mapped into the five commonly used AAMI
superclasses:

- N: Normal / non-ectopic
- S: Supraventricular ectopic
- V: Ventricular ectopic
- F: Fusion
- Q: Unknown / paced-related

Non-beat annotations are not assigned to an AAMI heartbeat class.

## Paced Records

The following paced-heavy records are excluded from the core
classification experiment:

- 102
- 104
- 107
- 217

They remain included in the full dataset audit for transparency.

This leaves 44 records for the core experiment.

## Dataset Audit

The audit of the 44 non-paced records produced:

| Class | Beats |
|---|---:|
| N | 90,125 |
| S | 2,781 |
| V | 7,009 |
| F | 803 |
| Q | 15 |

Total mapped beats: 100,733.

## Core Classification Task

The primary classification task is defined as:

N / S / V / F

Q is excluded from the core task because only 15 Q beats remain after
excluding the paced-heavy records, which is insufficient for reliable
patient-independent model training and evaluation.

The AAMI mapping itself still retains Q; Q is excluded only at the
classification-task level.

## Data Splitting

Data splitting must be performed at the record/patient level, not at the
individual beat level.

Records assigned to the final test set must not be used during model
development, hyperparameter selection, or repeated evaluation.

The exact train/validation/test record lists will be finalized separately.

## Primary Metric

Macro-F1 is the primary classification metric because the audited class
distribution is strongly imbalanced.

Secondary metrics will include:

- Balanced Accuracy
- Per-class Precision
- Per-class Recall
- Per-class F1
- Confusion Matrix