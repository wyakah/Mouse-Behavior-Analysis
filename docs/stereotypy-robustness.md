# Stereotypy robustness comparison

This development pass compares the same five mice (710, 711, 712, 743, 745),
starting at source time zero and ending at 1200 seconds. It preserves the original
five-mouse report, models, source videos and manual annotations. Mouse 756 is not
silently added. No group statistics are calculated.

## Implemented evidence

- Reproduce both frozen video classifiers from their existing cached features,
  verifying original source identity and selected model hashes. A changed score
  beyond 1e-5 aborts the comparison. Feature extraction is reused, not represented
  as new training.
- Decode all 36,000 source frames per mouse. Extract full-rate Farneback motion
  and infer side-view SuperAnimal Quadruped HRNet-W32 pose at 5 Hz. Review video
  retains every source frame and its timestamp; pose is held between samples.
- Add a stationary dark-body appearance proposal, separate from the prior
  background-subtraction proposal. It is **untrained**, excludes a narrow cage
  border/left fixture region, and can miss the mouse near fixtures or mistake
  dark objects for it. The cached learned quadruped detector returned no mouse
  in four feasibility frames and was not adopted as a reliable locator.
- Preserve eight task landmarks: nose, neck base, mid-back, tail base, two forepaws,
  two hindpaws. The checkpoint predicts a larger generic skeleton internally.
  Mid-back is a surface landmark, not center of mass. A missing landmark remains
  unknown; confidence >=0.6 is availability, not calibrated accuracy.
- Extract body-axis orientation, normalized head–paw distance, bedding proximity,
  pose speed, angular speed and ascent, with missingness indicators. A separate
  body-box center and its velocity remain explicitly geometric proxies, not
  anatomical center-of-mass estimates. Fine paw and
  mouth contact cannot be recovered when occluded. Speeds reset over gaps >.25 s.
- Flag low static cage-edge agreement against the initial view. This deliberately
  conservative check flags camera movement/reframing as well as footage outside
  the cage. It does **not** prove that every flagged frame is unusable. Automatic
  remapping or stabilization is not validated. Mouse 743 remains withheld under
  its prior visual audit, even if a new proposal is available.
- Surface video/pose rearing disagreements, upright pose with low video score,
  and rapid ascent/motion for review. These are exploratory review rules,
  **not fitted six-behavior fusion predictions**. Grooming/digging/gnawing and
  circling do not have validated geometric veto rules. Behaviors may overlap.
- Compare raw input, fixed gamma .8 and mild CLAHE (1.5,8x8) on the same 12
  systematic frames per mouse with the same crop. Preserve original images.
  Confidence changes do not select a preprocessing winner; raw stays default.

## Local labeling and training prerequisites

`/reports/stereotypy-robustness/labeling.html` presents 60 raw cage images without
model predictions. Click visible landmarks, explicitly mark hidden points as
occluded, and export JSON. Browser-local persistence is a convenience, not an
independent permanent annotation store. The existing `/stereotypy` interval
editor supports present, absent, ambiguous and unobservable labels for all six
behaviors. Whole mice must be assigned to disjoint training/validation/test sets
before fitting. These five mice have already been inspected during development;
a fresh independently scored set is preferable for the final accuracy claim.

`evaluate_stereotypy_pose_reference.py` compares raw/gamma/CLAHE landmark errors
against corrected visible points, reporting availability separately. It refuses
an unlabeled reference.

`export_stereotypy_pose_reference.py` writes DeepLabCut HDF/CSV only for fully
reviewed frames with explicit human provenance, preserving hidden points as NaN.
It checks source hashes, coordinate bounds and disjoint train/validation mice.
The export is preparation for side-view pose fine-tuning; **no fine-tuning was
performed in this run**, because no corrected human landmarks exist.

`train_stereotypy_fusion.py` provides a conservative supervised baseline combining
video scores, pose, full-rate motion and missingness. It requires explicit human
positive and negative intervals in all splits for each behavior, uses training-
only imputation/scaling, selects thresholds on validation mice and evaluates on
separate test mice. Classes without reference support stay unavailable. Labels
must never be copied from predictions. Its model is not automatically promoted
into experimental scoring. The future pose model must also be trained without
using validation/test mice before generating fusion features.

Example commands (replace IDs with a prospectively chosen split):

```bash
.dlc-env/bin/python scripts/stereotypy_robustness_run.py
.dlc-env/bin/python scripts/report_stereotypy_robustness.py
.dlc-env/bin/python scripts/export_stereotypy_pose_reference.py reference.json --out labeling/stereotypy-pose-export --train TRAIN_MOUSE_IDS --validation VALIDATION_MOUSE_IDS
.dlc-env/bin/python scripts/train_stereotypy_fusion.py --reference reference.json --train TRAIN_MOUSE_IDS --validation VALIDATION_MOUSE_IDS --test TEST_MOUSE_IDS
```

`export_stereotypy_behavior_reference.py --out reference.json` collects the
latest explicitly human-scored revisions from those five review sessions and
checks source hashes. It refuses empty sessions.

The reference JSON's `behaviors` list uses the existing interval schema plus
`mouse_id` and `source_sha256`: `behavior`, `label`, `start_s`, `end_s`, optional `note`. Unlabeled time
is not negative. Source gaps and unreliable body intervals cannot supply training
examples. Boundary samples without full human coverage are excluded. Duration-
weighted precision, recall, F1, balanced accuracy and duration error are reported.
The same eligible test intervals and original thresholds evaluate the baseline.
Report coverage separately: evaluation restricted to observable intervals can
otherwise hide systematic failures.

## What the comparison can establish

The report shows body/nose/axis availability, scene-flag time, disagreements,
paired enhancement availability and raw versus scene/body-eligible candidate
seconds for six behaviors. For 743 these are **diagnostic values**, not released
experimental scores. Fewer candidate seconds is not an accuracy improvement.

There are currently zero independent local reference labels. Local precision,
recall, landmark error, accuracy improvement and accepted behavior totals are
therefore unavailable. Existing external classifier benchmarks are unchanged;
this pass does not claim a newly validated model or the requested >90% target.
A larger ensemble cannot compensate for wrong anatomy or ambiguous behavior
labels. Promotion requires blinded reference evaluation and inspection of the
new failure cases.
