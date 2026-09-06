# Five-mouse, twenty-minute stereotypy pilot

Analyze `710`, `711`, `712`, `743`, and `745` from the local `stereotypy_videos`
folder. The first 1,200 seconds are used consistently. All five source files have
at least that duration, 30 frames per second, and no source timestamp gaps.
Each resulting review contains 36,000 source frames. Source files remain intact.
Although filenames say 540p, decoded dimensions are 536–572 × 286–324 pixels.

## What the outputs mean

The report contains **unreviewed model-candidate durations**, not validated
behavioral measurements. The models did not meet their external validation
target. No local accuracy estimate is available without independent reference
labels. Candidate segments must not be interpreted as verified behavioral bouts,
jump counts, or completed circles. Zero candidates do not establish absence.

- Grooming, digging and rearing use the frozen priority models from the second
  training experiment. The exact visual normalization, PCA, temporal context,
  model weights and previously selected thresholds are reused.
- Gnawing, jumping and circling use the earlier weak six-head model. Those heads
  have no independent positive validation; their 0.50 threshold is an exploratory
  default, not a validated operating point.
- Priority inference samples at 2 Hz with native ±0.1-second optical flow.
  Weak inference samples 16 frames per two-second window. Scores are held until
  the next sample. Every source frame is included in the annotated video.
- Source gaps are excluded from candidate duration and split segments. Time
  beyond 1,200 seconds cannot enter inference context, motion, or review output.
- Behaviors may overlap. No minimum bout filter, smoothing, statistical tests,
  pseudo-labeling, model fitting or accepted annotations are introduced.
- The report also includes duration sensitivity at threshold ±0.10. These ranges
  are not confidence intervals.

Mouse 745 was previously used as an unlabeled development animal. It must not
be treated as an independent validation animal. The other recordings also need
an explicit reference-label protocol before any accuracy claim.

## Cage and timing checks

The cage was inspected at 0, 60, 300, 600, 900 and 1,190 seconds. An initial tighter
crop clipped a tall upright posture in mouse 710. That test was stopped and its
predictions and audit evidence retained in `outputs/stereotypy-five-video-initial-crop`. Final inference
uses expanded upper bounds with additional headroom. Source timing indexes are
reused only after verifying file timestamps, size and SHA-256.

To make room for the batch, the redundant initial-crop feature caches and video
were removed, along with the unused DINOv2-base download from the stopped
reference control. Selected DINOv2-small and MobileNet weights, trained heads,
source recordings and training data were preserved. The optional base-model
control would need its cached backbone downloaded again before use.

The full source timing index stays intact. `stereotypy.window` defines the common
first-20-minute policy and clips only the analysis derivative. New manual sessions
also default to this window, reject shifted or longer windows, and retain an
immutable revision history. Short sources use their actual duration.

## Reproduce locally

Use the optional training environment and the previously downloaded model
weights. No network model download is required during inference.

```bash
.dlc-env/bin/python scripts/analyze_stereotypy_batch.py
.venv/bin/python scripts/report_stereotypy_batch.py
```

The analysis script supports `--only 710` and `--phase features`, `predict`, or
`render`. The five source-specific cage bounds are recorded in the script and
in each result. Reusing a result directory with changed inputs is rejected.
The source-specific pilot is not a general-purpose automatic cage mapper.

The report generator writes CSV exports, a local review page, and empty manual
review sessions. Existing sessions are never overwritten. It refuses to publish
an incomplete five-video review. It also requires the local
`qualitative-audit.json` with a reviewed `localization_approved` decision for each
mouse; those decisions must follow visual inspection rather than be fabricated
from detector coverage. The optional Excel builder consumes the report
JSON using the bundled spreadsheet runtime; no statistics are included.

Open `http://127.0.0.1:8765/reports/stereotypy-five-video/index.html` after the run.
The report links back to the measured external validation results and retains
source and selected-model fingerprints.

## Final quality-filtered outputs

All five annotated videos were verified against all 36,000 source timestamps,
including the exact 1,200-second endpoint. Four recordings retain provisional
candidate totals after excluding unusable body proposals. **Mouse 743 fails the
whole-recording localization audit and has all six totals withheld.**

| Mouse | Eligible seconds | Grooming candidates (s) | Digging candidates (s) | Rearing candidates (s) |
|---|---:|---:|---:|---:|
| 710 | 1199.9 | 4.5 | 61.0 | 51.0 |
| 711 | 1199.9 | 9.5 | 21.0 | 147.4 |
| 712 | 1197.3 | 1.0 | 6.0 | 60.5 |
| 743 | 0.0 | Withheld | Withheld | Withheld |
| 745 | 1200.0 | 0.0 | 6.0 | 117.5 |

The four retained recordings have zero weak-head gnawing, jumping, and circling
candidates. This does not establish absence: repeated upright movements and
possible leaping are visible in the mouse 712 systematic sample. Raw model
outputs remain available, including the rejected outputs for 743.

The qualitative audit inspected 12 systematically spaced body frames per mouse
and ordered frame samples from selected six-second contexts. It identifies
failure modes, not an accuracy benchmark. Mouse 743 includes camera-away footage
near 84 seconds that generated false behavior predictions; its later body boxes
often capture only a small part of the stationary mouse. Other reviewed examples
show drinking/grooming and upright-movement/digging confusions. No local
precision, recall, duration accuracy or 90% claim is justified.

The observability gate excludes ambiguous, edge, missing and camera-motion
proposals. A failed whole-recording audit excludes the entire window. Unknown
time does not become zero behavior; Excel candidate fractions use eligible
time. The burned-in review-video scores remain raw diagnostics, with rejection
and excluded intervals clearly marked in the report. Manual sessions start with
no accepted annotations. Neither models nor thresholds were fitted on this batch.

The completed report is limited to the original five IDs. The source selector
now rejects duplicates/missing IDs and ignores unselected files that arrive in
the folder while work is in progress. About 1.68 GiB of disposable older UI-test
media and frame tables were also removed while retaining their summaries/logs.
