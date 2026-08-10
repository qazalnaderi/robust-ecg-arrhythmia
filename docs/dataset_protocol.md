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


### Patient-independent DS1/DS2 split

The core dataset contains 44 non-paced MIT-BIH records.

A commonly used DS1/DS2 protocol assigns 22 records to each split.
However, records 201 and 202 originate from the same subject.

To enforce strict patient independence between development and final
evaluation, both records are assigned to DS1 in this project.

Therefore:

- DS1: 23 records, used for training and validation
- DS2: 21 records, reserved for final testing

DS2 must not be used for model selection or hyperparameter tuning.


### Train/Validation Split

The DS1 development set is further divided at the record level into
training and validation subsets.

Validation records:

- 108
- 114
- 205
- 207
- 223

The remaining 18 DS1 records are used for training.

The split is performed at the record level rather than the heartbeat level
to avoid subject leakage between training and validation.

The validation subset was selected using the record-level dataset audit to
retain representation of the N, S, V, and F classes while preserving enough
rare F-class beats for training.

Records 201 and 202 originate from the same subject and are both retained in
the training subset.

DS2 remains reserved for final evaluation and must not be used for model
selection or hyperparameter tuning.

## Primary Metric

Macro-F1 is the primary classification metric because the audited class
distribution is strongly imbalanced.

Secondary metrics will include:

- Balanced Accuracy
- Per-class Precision
- Per-class Recall
- Per-class F1
- Confusion Matrix