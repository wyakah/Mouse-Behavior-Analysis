# Pose and behavior distinction

The automatic queue now collects 10 Hz side-view SuperAnimal pose alongside the
existing video scores. Nose, neck, mid-back, tail base and four paws appear in the
live/final annotation. `pose-evidence.json` and each score window preserve usable
coordinates, missingness, normalized head-paw distance and forepaw speed. A
bounded temporal jump check rejects implausible changes; it does not fill hidden
paws or label inferred contact as observed. This is a lightweight temporal check,
not a reproduction of Lightning Pose's trained ensemble.

The current score policy remains explicit in each export. A pose coordinate or
an upright body does not automatically change grooming into rearing. Contact with
a nonfood target is not yet measured reliably, so no geometric gnawing veto is
invented. The first 1200 seconds remain the analysis window. Model-loading failure
retains video scoring and records the pose error in the result artifact.

## Published LabGym comparator

`scripts/benchmark_stereotypy_labgym.py` runs the actual downloaded 20-class
SavedModel using legacy Keras in `.labgym-env`. It calls the installed LabGym
2.2.2 functions for contours, internal motion patterns and 14-frame animations.
The pattern uses internal contours, std=50; animation is 64-square grayscale,
background-free; pattern is 64-square BGR. Names come from the released CSV.
Model weights, variables, parameter CSV, source video and adapter hashes are
recorded. The adapter supplies its own 32-frame percentile background and rejects
empty or implausibly sized contours. It is not an exact reproduction of the
LabGym GUI's background estimator.

Both native animal/background modes are tested on external validation:
1 (dark animal, inverted appearance) and 2 (absolute foreground difference,
original appearance). Selection uses validation grooming/rearing macro F1 only.
The direct model keeps its 20 labels. Chewing is not nonfood gnawing, and foraging
is not digging. `check_labgym_native_inputs.py` checks released example inputs;
those are training examples, not a held-out accuracy benchmark.

## Joint classifier experiment

`scripts/stereotypy_distinction_external.py` uses existing MIT human-labeled
windows, original date groups and original splits. Deterministic class budgets
select 240 training and 120 validation windows; all 266 original test windows are
retained. The test was inspected in earlier work and remains a legacy diagnostic.
No supplied mouse contributes a behavior label.

`scripts/train_stereotypy_distinction.py` compares the frozen baseline, native
LabGym, a jointly trained image/motion model and a pose-fusion model. The two fitted
models predict grooming/rearing/other jointly. Native probabilities and frozen
baseline probabilities form the image features; pose adds usable-nose fraction,
head-paw distance, forepaw speed and upright evidence with missingness indicators.
External pose uses nine ordered frames at 10 Hz per two-second reference window.
Model regularization is chosen on validation; selection is saved before test
metrics are calculated. The existing six-behavior classifier is not overwritten. Joint probabilities
have not been calibrated to the prevalence in local recordings.
This experiment cannot train digging, jumping, circling or nonfood gnawing from
MIT's unsupported labels.

Local comparison reuses verified raw 5 Hz pose caches for the earlier five mice
and computes a new 10 Hz pose pass for 756. Temporal evidence is recomputed using
the same shared code. Reuse and sampling rates are recorded per mouse. This saves
repeating unchanged neural inference without claiming new tracking measurements.
The report distinguishes behavior changes, evidence availability, externally
measured accuracy and unavailable local accuracy.

## Run

```bash
.dlc-env/bin/python -m venv .labgym-env
.labgym-env/bin/python -m pip install -r requirements-labgym.txt
.labgym-env/bin/python -m pip install LabGym==2.2.2 --no-deps
.labgym-env/bin/python scripts/stereotypy_distinction_external.py --backend labgym
.labgym-env/bin/python scripts/stereotypy_distinction_external.py --backend labgym --animal-vs-bg 2
.dlc-env/bin/python scripts/stereotypy_distinction_external.py --backend pose
.dlc-env/bin/python scripts/train_stereotypy_distinction.py
```

Feature and model commands refuse to overwrite completed experiment artifacts.
Use a new versioned output directory for later experiments. No local accuracy
claim or supervised local fine-tuning is possible without independent references.
That requirement concerns development validation; the batch user flow stays fully
automatic.

## Measured outcome of this run

On the same 266 legacy MIT test windows, the existing scorer achieved grooming
F1 0.6364 and rearing F1 0.6341. Pose fusion achieved 0.6292 and 0.6237.
Rearing recall rose from 0.5417 to 0.6042 while precision fell from 0.7647
to 0.6444. The candidate was not promoted. Original-image LabGym preprocessing
was selected on validation (priority macro F1 0.414 vs 0.349 inverted), but the
native model still performed poorly on the legacy test.

All six uploads (710, 711, 712, 743, 745, 756) received 18 two-second candidate
checks: 12 systematic and six high-gnawing-score examples, within the first
1200 seconds. These 108 clips total 216 seconds; they are not full candidate
behavior totals or locally labeled accuracy measurements. Full-recording
pose evidence covers all six (verified cache reuse for five; 12,000 new 10 Hz
pose samples for 756). The full 20-minute756 review has all 36,000 source frames
and retains the previous cumulative behavior totals. Initial native/inverted
full runs for 710 and 711 are retained as failed-preprocessing diagnostics;
further full native runs were stopped after validation selected original-image
preprocessing and the joint candidate failed the benchmark.

Open `/reports/stereotypy-distinction/index.html` for the benchmark, source-video
replay, clip comparison CSV and full 756 anatomical overlay.
