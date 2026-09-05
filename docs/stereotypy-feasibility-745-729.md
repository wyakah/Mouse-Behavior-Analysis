# Side-view automation: recordings 745 and 729

Assessment date: 5 September 2026. This is a feasibility study on two independent
mice, not a validated behavioral classifier or a treatment comparison.

## What the recordings support

Both originals are 3840 × 2160 HEVC at approximately 30 fps, with no rotation
metadata. Recording 745 lasts 280.263 s; 729 lasts 239.530 s. These are the full
recordings supplied, so the Three Chamber ten-minute window is not applied.

The cage is front-facing, with a slight view onto the bedding. A single dark
mouse contrasts with the background. The view contains tinted, scratched plastic,
an overhanging hopper, a left-side fixture and bedding that can conceal the feet.
745 includes substantially more surrounding bench and has a smaller mouse in the
original image. Sampling across both recordings shows changes in body orientation
and upright posture. These observations do not establish behavior durations.

Recording-specific bounds in **unrotated original pixels**, `[left, top, right, bottom]`:

| Recording | Cage crop | Bedding reference y | Original pixels retained |
|---|---|---|---|
| 745 | [1200, 245, 3120, 1090] | 865 | 19.6% |
| 729 | [300, 340, 3210, 1590] | 1270 | 43.9% |

The reference is a visible bedding level, not a physical floor plane. It changes
with depth and bedding displacement. We do not infer centimeters, 3D trajectories,
or jump height from it. The full cage remains visible, including the upper space
used for rearing; a moving, tight mouse crop alone would discard this context.

## Actual pretrained experiment

We ran installed DeepLabCut 3.0.1 with **SuperAnimal-Quadruped HRNet-W32** and its
FasterRCNN ResNet50 FPN v2 detector. This model explicitly targets side-view
quadrupeds, unlike the Three Chamber TopViewMouse model. The official model zoo
also documents transfer learning, adaptation and domain-shift failure modes.
[DeepLabCut documentation](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html)

The test uses 24 evenly spaced source frames per mouse. Frames are cage-cropped,
resized with recorded transforms and packed into a 48-frame diagnostic video.
**That diagnostic video is temporally discontinuous and must never be scored for
behavior.** Its frame manifest retains each original PTS, time base and crop.

| Diagnostic, cutoff 0.6 | 745 / 24 frames | 729 / 24 frames |
|---|---:|---:|
| Default detector: any landmark above cutoff | 4 | 4 |
| Default detector: nose above cutoff | 2 | 3 |
| Foreground-conditioned mouse box available | 24 | 24 |
| Conditioned pose: nose above cutoff | 8 | 13 |
| Conditioned pose: left forepaw above cutoff | 8 | 9 |
| Conditioned pose: right forepaw above cutoff | 8 | 14 |

Conditioning improves availability, but likelihood is neither a calibrated
probability of correctness nor evidence that the visible anatomical location is
right. Inspection of all 48 overlay thumbnails shows boxes following the mouse,
sometimes including the tail or an adjacent fixture; it also shows incomplete
and questionable anatomical predictions. There is no human coordinate ground truth
from which to claim 95–99% accuracy. **Neither pretrained variant is accepted for
automatic behavior scoring. No weights were fine-tuned.**

The model has 39 generic quadruped points, including anatomies irrelevant to mice.
Do not train on all of these automatically. A first side-view specialization
should use nose, neck base, mid-back, tail base and four paws. Ear bases are an
optional additional cue for orientation when visible. The three-point overhead
schema is sufficient for some chamber measurements, but not for identifying
head–paw interaction or hindpaw takeoff. Hidden points must remain unlabeled or
explicitly occluded, not guessed from a silhouette.

## Implemented full-recording review pass

1. Decode and index every source frame. Preserve PTS and declared durations.
   A declared duration exceeding the following PTS by at most two source clock
   ticks (and no more than 10% of a frame interval) is capped and recorded in
   `duration_corrections`. This bounded policy handles the one- and two-tick
   discrepancies measured in these MOV files. Larger overlaps and
   non-increasing timestamps fail; source gaps remain explicit unknown time.
   The supplied MOV clock is 1/600 s. No frame presentation time is moved.
2. Save a versioned cage crop and bedding reference. Numeric fields supplement
   drag selection. A new mapping invalidates the current review selection while
   retaining prior runs and annotations.
3. Construct a fixed 90th-percentile background from 32 time-spaced cage frames.
   It is computed offline over the recording, not adapted onto a stationary mouse.
4. Generate experimental dark-foreground components, excluding very small and
   excessively large components. Competing components, crop-edge contact, missing
   detections and sufficiently strong camera displacement are marked separately.
   This is an untrained proposal method tailored to a fixed camera and a dark
   mouse on a contrasting background; it is not a universal mouse detector.
5. Retain per-frame bounding box coordinates, timestamps, aspect ratio, local
   image change and centroid speed in cage-width units. A centroid is explicitly
   not an anatomical center or nose. Speed is suppressed across large time steps.
6. Render every source frame into a cage-focused H.264 review with original
   presentation times. Source frame controls remain authoritative at boundaries.
7. Suggest six-second review windows from systematic sampling, upright posture,
   uncertain localization and stationary movement. These cues are not behavior
   classifications, probabilities or scored bouts. Unreviewed behavior remains
   unknown, never zero. Feature CSV and provenance JSON are downloadable.

The full-pass overlay uses foreground proposals, **not** the rejected pretrained
pose model. Its coverage percentage describes availability of a proposed body box.
It cannot establish how often the box is correct. Camera-change flags use phase
correlation in the cage's upper band; they are a QC heuristic, not validated
stabilization. A camera change still needs inspection and possibly a new mapping.

Full-source results:

| Recording | Decoded / rendered frames | Body-proposed seconds | Proposal availability | Other localization states |
|---|---:|---:|---:|---|
| 745 | 8,402 / 8,402 | 280.092 | 99.94% | No ambiguous, edge or missing frame proposals |
| 729 | 7,186 / 7,186 | 237.308 | 99.07% | 1.667 s ambiguous; 0.533 s touching the crop edge |

The remaining duration includes declared timing gaps: approximately 0.172 s in
745 and 0.022 s in 729. These short timestamp/duration discrepancies are retained
conservatively as timing uncertainty, not evidence that the animal disappeared.
They can split/censor manually scored events, so counts need timing review as well
as behavior validation. Both rendered videos were decoded again: frame counts
match and maximum source-versus-review PTS error was 0.0 s. Final encoded packet
durations were corrected by lossless remux without changing pixels or PTS; source
intervals remain authoritative. No grooming, digging, gnawing, rearing, jumping
or circling totals were generated from these image heuristics.

## Behavior definitions and evidence requirements

The user requested all six classes. `side-view-draft-2` adds jumping and circling
without changing previous annotations; their unmarked time remains unreviewed.
The definitions remain a draft for lab agreement. Normal occurrences of these
actions alone do not establish pathological stereotypy.

| Class | Necessary evidence | Main confounders / limitations |
|---|---|---|
| Grooming | Repeated self-directed paw/head or mouth/fur interaction over time | Feeding, drinking, scratching; mouth or paws hidden by body |
| Digging | Repeated forepaw scraping with bedding displacement | Sniffing, walking and moving nest material; bedding hides paws |
| Nonfood gnawing | Visible repeated oral motion against an identified nonfood object | Contact/proximity alone is insufficient; hopper contact may be feeding |
| Rearing | Raised forequarters with forepaws off the substrate | Distinguish supported/unsupported if the lab needs it; not automatically a jump |
| Jumping | Visible loss of hindpaw support followed by landing | Rearing, climbing and occluded takeoff; 30 fps limits brief-event boundaries |
| Circling | A completed roughly 360° locomotor turn across consecutive frames | Partial turns and back-and-forth travel; a single side view can make completion unobservable |

For repeated jumping, mark each visible takeoff/landing interval and the intervening
known-negative time when individual jump counts are needed. For circling, record
direction and repeated-sequence context in the note. A generic positive interval
counts an observed segment; it does not count multiple revolutions inside that
interval. Automated revolution counts require a separately validated event model.

## Established strategies and their role here

- **Side-view DLC + SimBA:** Correia et al. used manually labeled side-view
  landmarks and supervised grooming classification. Their comparison illustrates
  why total duration and bout counts require separate agreement checks. This
  supports the proposed pipeline, not transfer of their accuracy to our cages.
  [Correia et al., 2024](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2024.1340357/full)
- **Temporal pose classifiers:** SimBA uses engineered pose features and supervised
  learning. Our intended first trained classifier uses independent behavior heads,
  temporal pose/motion features and an auditable random-forest baseline. This is
  an engineering choice inspired by that approach, not a reimplementation claim.
  [Goodwin et al., 2024](https://www.nature.com/articles/s41593-024-01649-9)
- **Appearance plus movement:** DeepEthogram demonstrates supervised behavior
  recognition from video pixels. It is an appropriate comparison if head/paw pose
  remains unreliable, particularly for self-directed movements. It still requires
  relevant behavior labels and validation on this setup.
  [Bohnslav et al., 2021](https://elifesciences.org/articles/63377)
- **Active review:** A-SOiD provides precedent for concentrating expert labeling
  effort on informative examples. Our systematic + uncertainty queue is currently
  a simple sampling heuristic, not an installed A-SOiD model or uncertainty-calibrated
  active learner. Retain systematic windows to avoid evaluating only easy positives.
  [Tillmann et al., 2024](https://www.nature.com/articles/s41592-024-02200-1)
- **Digging precedent:** A DLC/SimBA pup-retrieval study includes a digging classifier.
  Maternal assay context differs from these cages, so its weights are not assumed
  valid here. [Winters et al., 2022](https://www.nature.com/articles/s41598-022-05641-w.pdf)

No published side-view classifier covering all six behaviors in these exact cages
was verified in this bounded search. We have not bundled third-party classifiers
or redistributed model weights. Local research weights retain upstream terms.

## Next training and acceptance cycle

1. Have a lab scorer define onset/offset rules and review visibility in both real
   recordings. Label positives, explicit negatives and ambiguous/unobservable time
   independently for each of the six behaviors. Do not train against unreviewed time.
2. Correct a diverse pose set covering both cages, horizontal/upright poses, turns,
   front/back-facing mice and fixture/bedding occlusions. Start with the diagnostic
   frames and the full-pass review windows; expand based on visible errors rather
   than claiming a fixed number guarantees accuracy.
3. Fine-tune the side-view pose model; compare raw pretrained, conditioned and
   corrected versions on **unseen animals**, with visible-point pixel errors and
   localization misses. Do not validate using neighboring frames from training mice.
4. Train temporal classifiers on confirmed intervals. Compare pose-only with
   pose + local motion; consider pixel classifiers for poorly resolved paw actions.
   Genotype, sex, mouse ID and timestamps identifying experimental groups are not
   classifier inputs. Split by animal before feature normalization or threshold tuning.
5. Validate each behavior independently on additional mice with blinded human
   scoring: precision/recall, false positives among confounders, duration bias and
   absolute error, one-to-one event matching, boundary errors and unknown coverage.
   Visible positives on which the algorithm abstains count as misses in end-to-end
   sensitivity. Report human-observable coverage separately.
6. Keep duration readiness and event-count readiness separate. Two mice are useful
   for feasibility, not credible population-level accuracy or genotype statistics.
   Any desired 95–99% criterion needs a defined denominator and independent data;
   model confidence and proposal coverage cannot substitute for it.

For future recordings, the most useful improvements are a fixed camera, cage-filling
framing similar to 729, consistent exposure/focus, reduced glare and a clearer view
of feet and mouth where compatible with the assay. A synchronized second view would
help distinguish hidden paw contact and completed turns. Existing originals remain
usable for assisted review; cropping cannot recover anatomy hidden behind objects.

## Reproduce locally

```bash
.venv/bin/python scripts/stereotypy_cage_pilot.py
.dlc-env/bin/python scripts/stereotypy_pose_pilot.py
.dlc-env/bin/python scripts/stereotypy_conditioned_pose.py
```

Generated recordings, weights, source indices and results stay in ignored local
directories. The scripts and methods are tracked. `source-index.json` is reused only
when the source stat and hash checks remain valid. Per-run manifests record method,
parameters, mapping, source identity and implementation hashes. Original behavior
exports remain separate and preserve the annotation revision history.
