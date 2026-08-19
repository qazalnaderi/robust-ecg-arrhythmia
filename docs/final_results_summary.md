# Final Research Results — Noise-Robust ECG Arrhythmia Classification

## Project title

**Noise-Robust Multi-Class ECG Arrhythmia Classification under Realistic Motion Artifacts**

---

## 1. Experimental setting

The project studies patient-independent four-class ECG beat classification under realistic physiological signal artifacts.

### Classification classes

- **N** — Normal and bundle branch beats
- **S** — Supraventricular ectopic beats
- **V** — Ventricular ectopic beats
- **F** — Fusion beats

The AAMI Q class was excluded from the core four-class experiment because only 15 Q beats remained after the paced-heavy records were excluded.

### Input to the final classifier

Each prediction uses:

1. a **256-sample ECG heartbeat window**, processed by a 1D CNN;
2. four raw RR timing features:
   - `pre_rr`
   - `post_rr`
   - `average_rr`
   - `local_average_rr`

The RR features are standardized using statistics fitted on the training split only.

### Data protocol

The core experiment uses the MIT-BIH Arrhythmia Database and realistic artifact signals from the MIT-BIH Noise Stress Test Database.

Patient-independent development and test partitions were used.

- paced-heavy records `102, 104, 107, 217` were excluded from the core experiment;
- record `100` was reserved for sanity/development checks because it had been inspected during early preprocessing;
- the final DS2 test set remained untouched until all development decisions had been frozen.

### Primary metric

**Macro-F1** is the primary metric because the four classes are severely imbalanced.

---

# 2. Research questions

## RQ1 — How strongly does realistic ECG noise affect patient-independent arrhythmia classification?

### Development result

The frozen clean-trained ECG + Raw RR model achieved:

- clean Validation Macro-F1: **0.403**

At the most severe evaluated noise level, **-6 dB**:

- baseline wander (`bw`): **0.336**
- muscle artifact (`ma`): **0.277**
- electrode motion (`em`): **0.251**

Approximate degradation relative to clean Validation performance:

- `bw`: ~17%
- `ma`: ~31%
- `em`: ~38%

### Answer to RQ1

Realistic artifacts substantially degrade arrhythmia classification as SNR decreases.

Among the three evaluated artifact types, electrode-motion artifact caused the largest overall degradation, followed by muscle artifact, while baseline wander was generally less destructive.

**Conclusion:** the clean-trained model is not inherently robust to realistic severe ECG corruption.

---

## RQ2 — Which arrhythmia classes are most vulnerable to realistic artifacts?

### Development observations

On clean Validation:

- N F1 ≈ **0.948**
- V F1 ≈ **0.598**
- S F1 ≈ **0.053**
- F F1 ≈ **0.014**

The N class remained comparatively stable under corruption.

The V class showed meaningful clean performance but degraded strongly as noise severity increased.

S and F already had weak clean performance. Therefore, changes under noise cannot be interpreted as reliable estimates of noise vulnerability for these two classes.

### Final-test evidence after noise augmentation

Across the 15 noisy Final Test conditions, mean class-wise F1 was approximately:

| Class | Clean-trained | Noise-augmented | Difference |
|---|---:|---:|---:|
| N | 0.907 | 0.964 | +0.057 |
| S | 0.113 | 0.115 | +0.002 |
| V | 0.693 | 0.854 | +0.161 |
| F | 0.006 | 0.002 | -0.004 |

The strongest reproducible robustness gain occurred for **V**, followed by **N**.

Examples at -6 dB:

- V under baseline wander: **0.337 → 0.806**
- V under muscle artifact: **0.280 → 0.701**
- V under electrode motion: **0.292 → 0.661**

### Answer to RQ2

The ventricular class V is a major noise-sensitive class in the clean-trained model and benefits strongly from robustness training.

The N class is comparatively robust and becomes even more stable after noise augmentation.

The S and F classes remain difficult because their baseline classification performance is already poor, particularly for F.

**Conclusion:** class-wise robustness must be interpreted together with baseline class performance and class support.

---

## RQ3 — Can post-hoc denoising recover classifier performance under noise?

Two denoising methods were evaluated without retraining the classifier:

- Butterworth band-pass filtering, 0.5–40 Hz, fourth order, zero-phase offline filtering;
- wavelet denoising using `db4`, level 6, soft thresholding.

### Clean Validation control

Macro-F1:

- no denoising: **0.403**
- band-pass: **0.282**
- wavelet: **0.271**

Thus, applying post-hoc denoising to otherwise clean model inputs produced a large classification-performance drop.

### Noisy Validation result

Denoising did not consistently recover classifier performance.

A meaningful exception was severe muscle artifact at -6 dB:

- no denoising: **0.277**
- band-pass: **0.296**
- wavelet: **0.316**

Wavelet denoising therefore provided a modest recovery under this specific severe artifact condition.

However, improvements in waveform similarity did not consistently translate to improvements in classification.

### Answer to RQ3

Post-hoc denoising is not a reliable general robustness strategy for this classifier.

A key interpretation is **distribution shift**: the classifier was trained on unfiltered ECG morphology, so filtering can alter discriminative waveform characteristics even when the signal visually appears cleaner.

**Conclusion:** cleaner signals do not necessarily imply better classifier predictions.

---

## RQ4 — Does multi-SNR noise augmentation improve robustness, including at unseen noise intensities?

A second ECG + Raw RR model was trained using paired clean/noisy augmentation.

### Frozen augmentation protocol

Noise types:

- baseline wander
- muscle artifact
- electrode motion

SNRs used during augmentation training:

- **18 dB**
- **6 dB**
- **-6 dB**

Reserved unseen SNR intensities:

- **12 dB**
- **0 dB**

Each augmented epoch used the same number of optimization samples as the clean baseline:

- 50% clean
- 50% noisy

This prevents the augmented model from receiving twice as many optimizer updates as the clean baseline.

---

# 3. Multi-seed reproducibility

Both frozen strategies were trained using:

- seed 42
- seed 123
- seed 2026

### Clean Validation Macro-F1

Clean-trained ECG + Raw RR:

- 0.4030
- 0.3865
- 0.3647

Mean ± sample SD:

**0.3847 ± 0.0192**

Noise-augmented ECG + Raw RR:

- 0.3456
- 0.3592
- 0.3903

Mean ± sample SD:

**0.3650 ± 0.0229**

Therefore, the clean-trained model had better average clean Validation performance, although the direction of the difference was seed-sensitive.

### Multi-seed robustness on Validation

The key severe-artifact gains were reproducible.

Mean Noise-Augmented minus Clean-Trained Macro-F1:

- muscle artifact at 0 dB: **+0.014**
- muscle artifact at -6 dB: **+0.043**
- electrode motion at 0 dB: **+0.021**
- electrode motion at -6 dB: **+0.056**

For all four conditions above, the noise-augmented model improved in **3/3 seeds**.

Baseline-wander improvement was weaker.

### Development conclusion

Noise augmentation did not provide uniform gains across all operating conditions.

Instead, it produced its clearest and most reproducible benefit under severe muscle and electrode-motion artifacts.

---

# 4. Final held-out test

After all model-development decisions were frozen, the untouched DS2 Final Test set was evaluated once.

No tuning or retraining was performed after viewing Final Test results.

## Clean Final Test

Macro-F1, mean ± SD across three seeds:

- Clean-trained: **0.4767 ± 0.0208**
- Noise-augmented: **0.4898 ± 0.0268**

Difference:

**+0.0132**

The Final Test therefore did not reproduce the modest clean-performance penalty observed on Validation.

This should be described as **preserved clean-test performance**, rather than claiming a definitive clean-performance improvement.

---

## Severe-noise Final Test

At -6 dB:

| Artifact | Clean-trained | Noise-augmented | Difference |
|---|---:|---:|---:|
| Baseline wander | 0.3570 | 0.4798 | +0.1228 |
| Muscle artifact | 0.3196 | 0.4517 | +0.1321 |
| Electrode motion | 0.2945 | 0.4441 | +0.1495 |

The strongest absolute improvement occurred under severe electrode-motion artifact.

Approximate degradation from clean Final Test performance to -6 dB:

| Artifact | Clean-trained drop | Noise-augmented drop |
|---|---:|---:|
| Baseline wander | ~25% | ~2% |
| Muscle artifact | ~33% | ~8% |
| Electrode motion | ~38% | ~9% |

---

## Unseen-SNR generalization

For SNR intensities that were not used during noise-augmented training, 12 dB and 0 dB:

- Clean-trained mean Macro-F1: **0.4452**
- Noise-augmented mean Macro-F1: **0.4893**
- Difference: **+0.0441**

### Answer to RQ4

Multi-SNR noise augmentation substantially improved robustness on held-out patients, particularly under severe muscle and electrode-motion artifacts.

The benefit also generalized to SNR intensities not used during training.

On the Final Test, this robustness gain was obtained while preserving clean-test performance.

---

# 5. Main findings

1. **Realistic ECG artifacts substantially reduce arrhythmia-classification performance.**

2. **Artifact type matters.** Electrode motion and muscle artifact are generally more destructive than baseline wander.

3. **Ventricular beats are especially important in the robustness analysis.** V had useful baseline classification performance but degraded strongly under noise.

4. **Post-hoc denoising was not a general solution.** Band-pass and wavelet preprocessing often reduced classification performance because filtering changed the model-input distribution.

5. **Noise augmentation was substantially more effective than post-hoc denoising for severe artifacts.**

6. **The robustness improvement was reproducible across multiple random seeds.**

7. **The final held-out patient test confirmed the major robustness trend.**

8. **Robustness gains generalized to unseen SNR intensities.**

---

# 6. Limitations

## Severe class imbalance

The dataset is strongly imbalanced.

The F class has particularly poor patient diversity and very low effective support, making robust learning difficult.

## Weak S and F classification

The project does not solve reliable S or F recognition.

The low baseline F1 for these classes limits how confidently their noise robustness can be interpreted.

## Limited model family

The study intentionally uses a compact 1D CNN rather than a large model zoo.

This keeps the project focused on robustness methodology rather than architecture search.

## Limited seed count

Three random seeds provide a basic reproducibility check but do not support strong statistical significance claims.

## Fixed beat segmentation

The study assumes annotated beat locations and evaluates classification robustness after corruption.

It does not study the upstream problem of R-peak detection under noise.

## Dataset scope

The core arrhythmia conclusions are specific to the MIT-BIH Arrhythmia Database with NSTDB-based corruption.

External clinical validation on other arrhythmia datasets was not performed.

---

# 7. Final interpretation

The project should not be described as building a state-of-the-art arrhythmia classifier.

Its contribution is the experimental robustness analysis.

A suitable high-level conclusion is:

> A patient-independent ECG classifier that performs reasonably on clean ventricular beats can degrade sharply under realistic motion artifacts. Post-hoc signal denoising does not reliably restore classification performance, whereas multi-SNR artifact augmentation substantially improves robustness under severe baseline wander, muscle artifact, and electrode motion. The improvement is reproducible across random seeds, generalizes to unseen SNR intensities, and is confirmed on a held-out patient test set. Remaining limitations are concentrated in the highly imbalanced S and F classes.

---

# 8. Status

**Experimental phase: complete**

Frozen after Final Test:

- data split
- ECG preprocessing
- RR features
- CNN architecture
- class weighting
- augmentation protocol
- training seeds
- SNR evaluation levels
- model checkpoints
- Final Test results

No further model tuning should be performed using the Final Test results.

Next steps:

1. create publication-style final figures;
2. create compact final result tables;
3. write README / Methods / Results / Discussion;
4. prepare CV bullets and GitHub presentation.
