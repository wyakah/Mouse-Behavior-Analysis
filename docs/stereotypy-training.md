# First six-behavior training experiment

This workflow trains **experimental temporal video classifiers**. It does not
fine-tune DeepLabCut, validate the body proposals, or turn predictions into scored
behavior intervals. The frozen ImageNet image backbone is separate from the DLC
pose model tested in the earlier [feasibility study](stereotypy-feasibility-745-729.md).

The initial run uses 32 external recordings and the two supplied recordings,
745 and 729. All six output heads receive training signal, but the evidence is
uneven: three classes have explicit window labels and independent external test
groups; three have only a positive demonstration recording and derived negatives.
**No class has independent accuracy measurements on the user's cage.**

The completed balanced run selected epoch 23 of 40 by validation average
precision. Its external test results at a fixed 0.5 threshold are:

| Behavior | Test windows / positive | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|---:|
| Grooming | 266 / 48 | 0.574 | 0.563 | 0.568 | 0.572 |
| Digging | 109 / 39 | 0.326 | 0.359 | 0.341 | 0.368 |
| Rearing | 266 / 48 | 0.800 | 0.250 | 0.381 | 0.609 |
| Nonfood gnawing, jumping, circling | No independent positive test recording | — | — | — | — |

These are weak baseline results, not evidence of the requested 95–99% reliability.
Rearing's relatively high precision accompanies substantial missed positives.
Digging transfer is especially poor. None of the six heads is accepted for
automatic measurement. Checkpoint SHA-256:
`231ffb9b2a854e2647f5b45e63554aad793a122653e95e6380ef4b63b0c7c12f`.

## Sources and label mapping

| Source | Local material | Training supervision | Transfer limitations |
|---|---|---|---|
| [MIT mouse behavior dataset](https://cbmm.mit.edu/mouse-dataset) | 12 full side-view recordings; annotator groups 1 and 2 downloaded | Group 1 `groom` → grooming; `rear` → rearing. Other source categories are negatives for those two source-defined classes. | Source ethogram is mutually exclusive; the local assay permits overlap. MIT animal identities are not available in the archive. |
| [Winters et al., OSF RWHTD](https://osf.io/rwhtd/) | 17 matched videos and `targets_inserted` CSVs; 27,684 rows | Human digging column, named either `digging` or `diggingging`. Preserve both column names in provenance. **Do not use `machine_results`.** | Overhead maternal/pup-retrieval context differs from the local side view. |
| [Stanford Mouse Ethogram](https://med.stanford.edu/mousebehavior/ethogram/ethogram-index.html) | One official video each for jumping, circling and bar-mouthing | A positive **whole-video bag** for its behavior. Window labels remain unknown. MIT `rest` intervals supply explicitly derived negatives. | One positive recording per class cannot support independent validation. Camera, coat and background are confounded with class. Bar-mouthing is narrower than all nonfood gnawing; the circling definition is broader than the local completed-turn rule. |
| User recordings 745 and 729 | Existing cage crops and original source indices | No behavior labels. Train the feature adapter with denoising reconstruction and include these windows in normalization statistics. | This is unsupervised adaptation, not local validation or human ground truth. |

Stanford entries are `1_dgatnabm` (jumping), `1_ggs8x7g2` (circling), and
`1_5mzyurjp` (bar-mouthing). Retrieval uses the same anonymous public Kaltura
session as the embedded website player; no author requests or credentialed access
are submitted. A demonstration title never becomes a label for every frame.

Data copies and learned weights remain local. The MIT page links a research-only
**software** agreement; a separate data redistribution grant was not established.
The OSF node's license field is null, and Stanford media redistribution terms were
not established. Do not bundle these recordings, annotations, or weights into the
public repository. The downloader records source URLs, archive members, media
identifiers and SHA-256 hashes without retaining temporary media session tokens.

## How this run learns

1. Use fixed cage-focused crops. Convert to grayscale, replicate to three
   channels and letterbox to 224 pixels without distorting aspect ratio. Remove
   the Stanford caption area and OSF bottom frame-counter strip. Background and
   camera cues remain possible confounders, especially for the weak classes.
2. Select two-second windows and sample 16 ordered frames from each. MIT windows
   fit entirely inside an annotator-1 interval with 0.2-second boundary margins;
   cap sampling at 24 windows per source category per recording. OSF windows must
   have one constant human digging label throughout. Transition windows and short
   events are excluded, so evaluation is biased toward stable, longer behavior.
3. Decode source videos sequentially. External labels use zero-based frame
   indices, converting MIT one-based inclusive bounds explicitly. OSF decoded
   counts must match the annotation rows. Local sampling looks up the existing
   source PTS index, whose video hash must match the original.
4. Extract 960-dimensional image features with frozen
   [Torchvision MobileNetV3 Large ImageNet V2 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html).
   Cache features against source hashes, crop/preprocessing and exact window
   definitions. Full decoding does **not** mean every frame was used for learning.
5. Fit a 128-dimensional projection, two temporal convolution layers and six
   independent output heads. Weighted binary cross-entropy sees only explicitly
   known labels; unknown labels produce zero classification gradient. Weak
   positive videos use max-over-window multiple-instance loss plus a balanced
   loss on derived resting negatives. This can memorize a single example and
   cannot establish behavior specificity.
6. Fit a denoising reconstruction adapter jointly, using both user recordings
   without class labels. Normalization uses only training and adaptation data.
   No external validation or test feature enters these updates.
7. Fit 40 epochs with fixed seed 20260905. Select the checkpoint by validation
   macro average precision for grooming, digging and rearing. Report test
   precision, recall, F1 and average precision; use a fixed 0.5 threshold, not one
   chosen from the test set. No calibration claim is made.

The balanced weak-head loss supersedes the preliminary fitting pass, which
overweighted the positive demonstration bags. This is a training-objective fix;
test measurements are not used to choose the objective or checkpoint.

## Splits and scope of the measurements

- MIT recordings from a shared date stay together, including A/B/F variants.
  Validation: 20080422 and the agouti 20080229 group. Test: 20080424 and 20080428.
  Other dates train. Date grouping is **not proven animal-level independence**.
- OSF recordings stay together by mother ID. Mother13 is validation; Mother17
  and Mother19 are test; other mothers train.
- Stanford demonstration bags train only. There is no independent positive test
  recording for gnawing, jumping or circling.
- Both local mice are adaptation data. Predictions on them are a development
  review, not a held-out test. Reusing these mice later for evaluation would
  require an explicitly described transductive evaluation, not an unseen-mouse claim.

The initial manifest contains **2,335 windows**: 1,368 training, 333 validation,
375 test and 259 unlabeled adaptation windows. The extractor decoded 1,174,241
frames and selected 37,360 unique frames. Two-second local windows cover 518 s
of the approximately 519.793 s supplied; the final partial windows remain
unscored. Original timestamp gaps also remain unknown. No ten-minute duration is
invented for these shorter stereotypy files.

The results measure selected external **window classification**, not all-frame
accuracy, event boundaries, bout counts, circling revolutions or duration error.
Two-second, 8 Hz sampling is insufficient to establish precise takeoff/landing
boundaries or guarantee that a completed circle is visible. No threshold in this
run authorizes automatic assay measurements, including a high model score.

## Reproduce locally

Use the existing `.dlc-env` or create a separate Python 3.12 environment and install
`requirements-stereotypy-training.txt`. The lightweight web app does not need
torch. Raw sources occupy roughly 1.3 GB; leave additional space for caches and
outputs. Downloads can resume by skipping completed local files.

```bash
.venv/bin/python scripts/download_stereotypy_training.py
.venv/bin/python scripts/prepare_stereotypy_training.py
.dlc-env/bin/python scripts/train_stereotypy_video.py --epochs 40
.venv/bin/python scripts/report_stereotypy_training.py
.dlc-env/bin/python -m pytest tests/test_stereotypy_training.py -q
```

The defaults expect the existing original videos and source indices from the
feasibility pass. This first-run preparation script is specific to these datasets
and the two supplied recordings; it is not yet a general uploaded-video trainer.
New datasets need explicit label mappings and source groups, not renamed columns
or assumed negatives.

Open `/reports/stereotypy-training/index.html` on the local app. Play either cage
review beside the six model scores, select a behavior and inspect candidate clips.
CSV downloads are explicitly **model scores**, not behavior measurements.
The normal reviewer remains available for authoritative annotations and exports.

Saved under `outputs/stereotypy-training/v1/`:

- `dataset.json`: source hashes, annotations, crops, per-window labels, authorities
  and splits. Positive bags and unlabeled local data remain distinct.
- `features/`, `extraction.json`: cached features and decode/sample provenance.
- `temporal-model.pt`: fitted weights, normalization, class order, selected epoch,
  source-code hashes and `production_ready: false`.
- `training-report.json`: learning history, per-class support, external metrics,
  unavailable local/rare-class accuracy and checkpoint hash.
- `window-predictions.npz`, `745_stereotypy-proposals.json`,
  `729_stereotypy-proposals.json`: experimental scores with source/model hashes.
  `behavior_totals` is null. None of these files changes saved annotations.

## Acceptance work still required

The next useful labels are local, expert-reviewed positive and negative clips,
including confounders: feeding versus gnawing, sniffing versus digging, rearing
versus jumping, and partial turns versus completed circles. Collect multiple
positive recordings and independent mice for each rare behavior; one video per
class is insufficient. Verify the local positive evidence can actually be seen
through the bedding and hopper before assigning a label.

Train with those labels, preserve a new-mouse test set, and compare per-class
precision/recall, duration agreement and event-count error against independent
human scoring. Address boundary timing at native frame rate. These tests are what
could support a future accuracy claim; successful optimization, model likelihood,
and body-proposal coverage cannot.
