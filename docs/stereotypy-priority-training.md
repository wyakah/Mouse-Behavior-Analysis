# Priority behavior training: expanded data and stronger image features

This second experiment targets grooming, digging, and rearing. It keeps the first
six-head experiment intact. Nonfood gnawing, jumping, and circling do not gain new
validated classifiers from this run. Experimental scores never become accepted
behavior annotations or Excel totals automatically.

## Completed run: target not met

The external test was opened after validation-only selection across six model
designs. Grooming uses the shared temporal model; digging and rearing use
specialized binary temporal heads. The held-out external-file test exposed a
large validation-to-test gap:

| CBAS published test files, 2 Hz | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Grooming | 45.0% | 64.5% | 53.0% | 67.5% |
| Digging | 54.2% | 31.7% | 40.0% | 64.4% |
| Rearing | 62.1% | 25.0% | 35.7% | 62.4% |

Ordinary accuracy was 68.9%, 90.6%, and 98.3%, respectively. The latter two
numbers **do not meet acceptance** because many actual events were missed.
No class is accepted for automatic measurement. Local accuracy remains unknown.
The precise cause of the external generalization gap remains unresolved; the
stronger validation results must not be substituted for these test results.

On the **same 266 MIT two-second windows** used in the first experiment,
grooming F1 increased from 56.8% to 63.6%, and rearing from 38.1% to 63.4%.
These are previously observed test windows, retained as a diagnostic; MIT has no
usable digging label. New window scores average the four sampled-frame scores
using thresholds selected beforehand on frame-validation data.

The complete cache contains 47,800 sampled images: 5,962 CBAS training samples,
9,645 CBAS validation, 8,856 CBAS test; 4,096 MIT training, 1,188 MIT validation,
1,064 MIT legacy test; 15,948 LabGym training; and 1,041 local development samples.
There are 4,021 source files and no duplicate source-byte hashes. Training uses
only 26,006 labeled training samples; the supplied mice never supply behavior
labels. These are sampled images, not a claim that every acquired frame trained.

A further control using the published CBAS DINOv2-base checkpoint was prepared
but stopped before completion when the user requested wrap-up. **No control
metrics are available.** Its script, `benchmark_cbas_reference.py`, records two
existing author code paths (vector versus scalar centering) as diagnostics;
neither is an accepted local model. The compatible head retains the upstream
MIT license in `third_party/cbas/LICENSE`.

Open `/reports/stereotypy-priority/index.html` for the measured results,
source-level breakdown, model fingerprints, and timestamp-aligned candidate
reviews for 745 and 729. Predictions have not changed accepted annotations.
For additional 20-minute recordings, reserve mouse IDs for independent
validation before using the remaining recordings for training and correction.

## New sources actually acquired

- **CBAS:** 20 videos (260,123 frames; 7.23 hours) and 20 human annotation tables from the authors' public
  training/testing release. The nine source classes include explicit grooming,
  digging, and rearing labels. The published videos are 256 × 256 at 10 Hz and
  concatenate labeled examples. We retain the four published test files, reserve
  training files 3, 9, and 14 for validation, and use the other 13 for training.
  The source ethogram uses distinct categories, whereas local labels can overlap;
  supported rears, climbing, and upright grooming require explicit local review.
  Original animal and segment identities are not supplied in these files. This
  is file-disjoint evaluation, not a demonstrated independent-mouse test.
  [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12146638/),
  [published code](https://github.com/jones-lab-tamu/CBAS/tree/v2-stable),
  [author's data](https://drive.google.com/drive/folders/1rvIjUgC48ZhaxV1Q6eLpK0Cz60quJirN).
- **LabGym:** 5,741 short curated mouse example clips downloaded; 3,987 have usable
  labels for at least one priority head. Body/face grooming map to grooming;
  rearing up and standing away from the wheel map to rearing. Foraging includes
  multiple activities and **never maps to digging**. Grooming clips have unknown
  rearing status rather than an invented negative. All clips train only because
  original mouse/recording identities are unavailable. They are segmented mouse
  images, unlike the full-cage CBAS and local inputs. The AVI playback rate is not
  used as a behavioral duration authority.
  [Model zoo and example download](https://github.com/umyelab/LabGym/blob/master/LabGym_Zoo.md),
  [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10088092/).
- **MIT:** retain the existing side-view recordings and annotation-group-1 labels
  for grooming/rearing. Preserve date grouping and the original validation set.
  The previous test set has already been inspected; it is reported as a **legacy
  diagnostic**, not a newly sealed independent test.
  [Dataset](https://cbmm.mit.edu/mouse-dataset).

The CBAS and LabGym pretrained heads were also retrieved for inspection. Their
weights are not the fitted models in this experiment. In particular, LabGym's
foraging output is not a pure digging classifier. The DINOv2-base + LSTM CBAS
published head is incompatible with this experiment's smaller, differently
preprocessed DINOv2-small features; substituting the weights would be invalid.

The separate CBAS `test_sets.zip` metadata contains original recording paths
for additional annotated instances across four experimental cohorts. It does not
provide a mapping back to the concatenated clips used here, so it does not
establish animal-disjointness for this run. Those extra recordings have not
been silently counted as training data.

Additional routes identified include the public DeepEthogram dataset collection
(linked in its [official README](https://github.com/jbohnslav/deepethogram#data))
and [JABS](https://doi.org/10.7554/eLife.107259.2). These are leads, not extra
training examples silently counted in this experiment. Viewpoint, ethogram,
independent source IDs, and actual labeled positive support matter more than the
number of download links.

## Model and efficiency

The encoder is frozen `facebook/dinov2-small`, a 384-dimensional self-supervised
image representation. Each sampled frame supplies a full-cage view and a closer
foreground mouse view (768 combined features), plus seven geometry values.
Images are grayscale, aspect-preserving 224-pixel letterboxes with ImageNet
normalization. The foreground crop uses a per-video background estimate; it is
not a validated anatomical tracker. A missing component falls back to the cage
view and carries a missing-geometry flag. No detection score is called accuracy.

The local GPU benchmark favored float32 batches of 16 over float16 batches of
32/64. Features are cached once, bound to source bytes, annotations, crop and
sampling recipe. For CBAS training only, each source class contributes up to 12
seeded center samples with two neighboring 2 Hz samples on either side. This
reduces redundant GPU work without choosing easy validation/test examples.
All CBAS validation/test files retain regular 2 Hz sampling across their full
length, including transitions and negatives. MIT supplies four frames per
previously selected two-second window; LabGym supplies four ordered images per
curated example. The two identical LabGym image views are encoded only once.

The motion ablation computes local optical flow from native neighboring frames,
not just differences between 2 Hz samples. Spatial flow histograms and magnitude
statistics describe movement around the mouse. Missing edge/gap motion carries an
explicit unavailable flag. LabGym uses ordered neighboring frames without
assuming its AVI playback rate is the acquisition rate. This follows the general
appearance-plus-motion strategy described by
[DeepEthogram](https://elifesciences.org/articles/63377); it is not a reproduction
of that software's trained flow generator. Motion features run on CPU while the
GPU finishes the visual cache.

Normalization and separate PCA projections (128 appearance components and 32
motion components) fit a seeded sample of up to 4,000 training frames per dataset,
reducing domination by the much larger LabGym clip collection. All validation,
test, and local development frames are excluded from these fits. Temporal context
uses seven sampled frames, with edge padding at video boundaries, detected cuts,
and sparse sampling gaps. It never reaches into another file. Camera cuts within
concatenated source videos may still be missed. Temporal context and movement
statistics feed an appearance-only and motion-augmented logistic model, matching
tree ensembles, and a learned temporal convolutional network. Tree feature
subsampling uses the square root of feature count to reduce fitting cost and
correlated splits. Unknown labels do not contribute a classification loss. A final validation-only
comparison fits regularized binary temporal heads, selecting an epoch separately
for each behavior. The shared model runs for at most 40 epochs (early stopping
after ten unimproved epochs, with a 20-epoch minimum); specialized heads run for
at most 30 (seven unimproved epochs, 12-epoch minimum). The pre-optimization
selection is retained before updating the selected models.
Dataset and positive/negative weighting reduce domination by a large source or
frequent negatives. Per-class model and threshold selection maximizes mean
validation F1 with equal weight for each supported camera dataset.

A separate `--evaluate` command opens test metrics only after the selected models
and thresholds have been saved. Training refuses to overwrite a run whose test
report already exists. Any subsequent optimization must disclose test reuse and
use a fresh protocol. Repeatedly tuning against the test until it exceeds 90%
would not demonstrate generalization.

## Meaning of the 90% target

The acceptance screen requires **greater than 90% precision, recall, balanced
accuracy, and ordinary accuracy** for each priority behavior. This prevents a
classifier that misses rare rearing events from passing through abundant true
negatives. F1, average precision, class support and confusion matrices are also
reported, separately by source and camera dataset. Samples within a recording
are correlated; frame count is not an independent-mouse sample size.

Passing an external benchmark does not accept the local assay. Mice 745 and 729
were used for unlabeled development in the first run and have no authoritative
behavior labels. Their scores remain suggestions, with no accepted totals.
Local acceptance needs expert-reviewed positive and confounding-negative
intervals from multiple mice, with at least one reserved mouse group excluded
from training and model selection. Duration and bout-boundary error must also be
checked at source frame rate; a 2 Hz frame benchmark cannot establish them.

The app's stereotypy analysis window already defaults to the actual full source
duration, so future 20-minute recordings are not truncated to the three-chamber
10-minute window. Those experimental recordings have not yet been supplied.
They should remain separate from training and local model selection if they are
to provide unbiased experimental measurements.

## Reproduce

Use the optional training environment; the Flask application still needs no
transformer dependencies. Raw videos, annotations, extracted features and model
weights stay in ignored local directories. Public source IDs and audited CBAS
SHA-256 fingerprints are in `docs/data/stereotypy-priority-sources.json`.
The downloader records LabGym file hashes on acquisition. Data redistribution
rights are not presumed from a software license; raw data and weights are not
included in the public repository.

```bash
.venv/bin/python scripts/download_stereotypy_priority.py
.dlc-env/bin/python scripts/stereotypy_priority_features.py --only cbas mit labgym local
.dlc-env/bin/python scripts/stereotypy_motion_features.py
.dlc-env/bin/python scripts/train_stereotypy_priority.py --epochs 40
.dlc-env/bin/python scripts/optimize_stereotypy_priority.py
.dlc-env/bin/python scripts/train_stereotypy_priority.py --evaluate
.dlc-env/bin/python scripts/review_stereotypy_priority.py
.dlc-env/bin/python scripts/report_stereotypy_priority.py
.dlc-env/bin/python -m pytest tests/test_stereotypy_priority_training.py -q
```

The feature extractor currently builds on the first experiment's MIT/local
manifest and source timestamp indices. This is a reproducible research training
workflow, not a claim that arbitrary uploaded videos can provide labels.
Results and selected classifiers are under `outputs/stereotypy-training/v2/`.
