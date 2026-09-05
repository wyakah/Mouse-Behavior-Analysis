# Stereotypy: side-view research and implementation plan

Prepared 5 September 2026. Status: research complete for the initial architecture; manual-scoring foundation implemented; experimental videos and recognition validation pending.

## Decision

Keep the familiar **Recordings → Review → Measurements/export** flow in the existing application. Replace chamber geometry with independently scored behavior intervals and visibility. Start automation with total self-grooming duration, then evaluate digging and nonfood gnawing separately. Keep all three available manually, with optional rearing. Neither ordinary grooming nor digging alone establishes pathological stereotypy.

The preferred first trainable backend is side-view DeepLabCut pose plus an independently implemented scikit-learn temporal random forest. This is an engineering choice based on the evidence and existing installation, not a reproduction or validation of a published pipeline. Maintain a neutral prediction interface so a pixel model can replace it if visible mouth/paw motion is not captured reliably by pose.

The user requested a branch, research, planning, and initial development before supplying real recordings. The supplied `Mouse_Behavior_Codex_Build_Guide.md` was used as background; its embedded “copy this brief” instructions were not treated as an additional user request. The still image mentioned inside that guide was unavailable and was not inspected. No side-view experimental recording has been assessed in this task.

## Research findings and build consequences

Primary papers, author repositories, model pages, and official documentation were checked on 5 September 2026. Publication years below refer to the publications, not search-engine crawl dates. The search covered side-view grooming, digging, nonfood gnawing, newer repetitive-behavior models, manual coding tools, and reusable model artifacts.

| Approach | What the evidence supports | Decision for this project |
| --- | --- | --- |
| Correia et al., 2024, DLC/SimBA vs manual vs HomeCageScan | Direct side-view grooming precedent. Duration agreed better with manual scores than bout counts; single-side occlusion remained a limitation. | Prioritize grooming seconds. Validate event counts independently. Do not reuse paper thresholds as universal defaults. [Paper](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2024.1340357/full) |
| Geuther et al., 2021, JAX MouseGrooming | A published pixel-based grooming model with author code and downloadable annotated data/container. Developed for single-mouse overhead open-field footage. | Useful reference or later benchmark, not a ready side-view detector. Do not square-resize these recordings merely to match its inputs. [Paper](https://elifesciences.org/articles/63207), [author repository](https://github.com/KumarLabJax/MouseGrooming), [dataset](https://zenodo.org/records/4646088) |
| Winters et al., 2022 | Published DLC/SimBA digging classification in a dam/pup-retrieval assay. | Establishes a precedent for digging labels, not transfer to arbitrary single-mouse side-view cages. Full HTML retrieval failed; publisher PDF/search-visible methods were available. [Publisher PDF](https://www.nature.com/articles/s41598-022-05641-w.pdf) |
| DeepLabCut SuperAnimal | Official documentation distinguishes TopViewMouse from Quadruped; Quadruped includes side-view rodents. Both return body points. | Trial Quadruped only after checking actual footage and model terms. Existing top-view three-point weights and the current arena localizer are unsuitable defaults. [Model Zoo](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html) |
| DeepEthogram | Published behavior classification from video appearance/motion without a required pose stage. Current license limits internal academic/noncommercial use and redistribution. | Pixel-method precedent and possible separately arranged research benchmark; do not copy or bundle its implementation into this module. [Paper](https://elifesciences.org/articles/63377), [current license](https://github.com/jbohnslav/deepethogram/blob/master/license.txt) |
| CBAS, 2025 paper / current author repository | A newer behavior-labeling/training/analysis suite. The repository distinguishes the published `v2-stable` path from a newer beta interface and discusses streamed video encoding. | Add to the pixel-backend shortlist. Verify the exact stable model's camera/labels, dependencies, and weight rights before a trial. An MIT repository badge does not clear all bundled dependencies or weights. The full PMC paper returned a browser challenge, so no unverified numerical performance is claimed here. [Author repository](https://github.com/jones-lab-tamu/CBAS), [paper record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12146638/) |
| Augustine et al., 2025, YOLOv11 | Repetitive-behavior classification precedent in deer mice, including grooming/rearing/jumping categories. | Species and label mismatch; not evidence for validated house-mouse digging/gnawing in this setup. [PubMed](https://pubmed.ncbi.nlm.nih.gov/40368233/) |
| BORIS | Established manual state-event coding, editable ethograms, timing controls, and CSV event exports. | Use its workflow as a usability reference. Add a declared BORIS CSV adapter later if existing lab annotations use it. No BORIS code is copied. [Official tool](https://www.boris.unito.it/), [event export documentation](https://www.boris.unito.it/user_guide/export_events/) |

No suitably licensed, ready-to-use side-view nonfood-gnawing classifier was verified in this search. That is a bounded finding, not proof none exists. Bar-mouthing specifically involves cage bars; it is not synonymous with contact near an unidentified wall or object. Keep object identity in annotation notes and decide whether a separate bar-mouthing label is needed after examples. [Stanford ethogram](https://med.stanford.edu/mousebehavior/ethogram/active-behaviors/abnormal-behaviors/stereotypies/bar-mouthing.html)

## Repository integration and isolation

- Branch `stereotypy` starts at local `main` commit `30c4c66` (not a fetched remote revision), in `/Users/yakahwil/Downloads/Three-Chamber-stereotypy`.
- The original checkout remains on `main`; its active statistics/batch changes belong to the other task. No checkout switch, stash, reset, environment installation, video copying, or model job was performed there.
- Existing architecture: Python/Flask, plain HTML/CSS/JavaScript, PyAV timestamp decoding, local video upload/stream routes, one executor, batch state, and CSV exports. There is no need for React, a second service framework, or hosting.
- Reused: local upload and video streaming, the app's loopback/origin checks, job executor/status route, hashing utility, and visual conventions.
- Kept separate: behavior definitions, annotation/measurement engine, source timing index, side-view routes, session files, outputs, and eventual model adapters. No cup circle, chamber calibration, fixed EthoVision crop, first-600-second preparation, or top-view detector is invoked by this flow.
- Runtime observed: macOS 26.1 on arm64, Python 3.12.14, PyAV 18.1.0. Existing lockfile records DLC 3.0.1, PyTorch 2.14.0, scikit-learn 1.9.0. These are environment records, not a new compatibility guarantee; GPU throughput and memory have not been benchmarked for side-view inference.
- The repository's `CONTRIBUTING.md` explicitly says no open-source license has been selected. Preserve that status. Release licensing is a later decision, not an implicit consequence of the guide.

## Proposed operational labels

`side-view-draft-1` is a proposed lab ethogram, awaiting examples and agreement. Onset is the first visible qualifying movement; offset is the exclusive end of the last qualifying movement. Initially retain short actions.

| Label | Visible evidence | Important competing actions |
| --- | --- | --- |
| Grooming | Self-directed paw washing, face/head sweeps, licking or nibbling own fur | Feeding, drinking, isolated scratching, curled resting, sniffing |
| Digging | Repeated forepaw scraping with bedding displacement | Walking through bedding, sniffing, nest carrying, head pushing; whether push-digging belongs here needs lab review |
| Nonfood gnawing | Repeated oral movement contacting an identified nonfood object | Eating, drinking, brief investigation, wall scratching; proximity does not establish contact |
| Rearing | Forequarters raised with forepaws off the substrate | Optional posture that may overlap target behavior |

Each behavior has independent `present`, `absent`, `unobservable`, `ambiguous`, or `unreviewed` intervals. Only explicit present/absent intervals are potential supervised labels. Scoring grooming does not imply digging negatives. Occluded mouths/paws remain unknown for the relevant behavior. Conflicting oral labels should be reviewed; the current manual module permits cross-behavior overlap and does not automatically adjudicate it.

## Workflow and implementation status

| Milestone | Deliverable / acceptance | Current state |
| --- | --- | --- |
| 0. Inspect and research | Isolated branch, evidence matrix, code integration boundary, sample plan | Complete; no real side-view samples yet |
| 1a. Manual foundation | Original video timing/hash, explicit window, source-frame inspection, interval edits/undo, revision history, unknown denominators, CSV package | Implemented and tested with synthetic data |
| 1b. Lab-ready reviewer | Real-video timing/playback check, accepted ethogram, bounded clip queue, zoomable timeline, robust persistent/cancellable indexing, labeler session timing | Next after first sample inspection; can harden independent UI/storage pieces earlier |
| 2. Side-view pose pilot | DLC CSV/HDF5 adapter, frame-map validation, visibility/jump report, overlays, grouped split manifest | Planned; no side-view inference started |
| 3. Grooming baseline | Deterministic temporal features, grouped training, immutable predictions, assisted corrections, locked duration evaluation | Planned; needs local labels |
| 4. Other actions | Independent digging model and validation; gnawing only where visible and sufficiently represented | Planned; retain manual scoring when a class fails |
| 5. Batch/release preparation | Queue/resume/cancel, frozen model/run cards, reproducible environment, approved demo and licenses | Planned; no publishing requested |

The manual foundation is intentionally useful before automation. It stores normalized source PTS, frame durations, content hash, camera dimensions, codec, rotation, audio presence, anonymous IDs, explicit window, and annotation snapshots. The exact-frame viewer decodes the requested source PTS. Playback marking uses the player's current time; source-frame inspection is available for boundary precision.

Verification on 5 September: 104 Python tests and 3 existing JavaScript player tests pass; JavaScript syntax checks pass. The new tests cover constant/unequal durations, a real encoded nonzero-PTS/gap fixture, unknown/absence separation, cross-behavior overlap, gap merging/censoring, source identity changes, exact-frame access, stale revision rejection, and immutable re-exports. Browser inspection exercised session creation, interval entry, saving, and source-frame boundary controls on the clearly marked synthetic demo; no browser warnings/errors were recorded during those checks. These checks do not validate animal behavior recognition.

Unlabeled time remains unreviewed. Declared source gaps override positive annotations. All-unknown behavior has blank measured duration, percentage, and event count; explicitly reviewed absence has zero duration. Exported summaries include scored/unknown seconds and coverage. Bouts have separate active and span duration plus left/right censoring. Known-negative gaps can be merged explicitly; unknown gaps never merge. A wholly unreviewed behavior cannot contribute a zero to group analysis.

Every save creates a revision with annotator and time; updates use optimistic revision checks to prevent stale-tab overwrites. A single atomic session file holds history. Every export writes a new ZIP under `outputs/stereotypy/<session-id>/`, including `summary.csv`, `bouts.csv`, `annotations.csv`, and a manifest containing timing, definitions, and revision history. No predictions file is fabricated; `model_required` and `predictions_available=false` are explicit.

### Current implementation limits

- No trained side-view detector, pose adapter, model inference, active learning, group statistics, or recognition-accuracy claim.
- Timeline overview and source-frame controls work; timeline zoom, review-clip queue, conflict adjudication, and analyst-time tracking remain planned.
- The indexer decodes in the background without retaining image arrays, but stores an O(frame-count) JSON timing index and sends it to the browser. Paginated/compact indexes are needed before long continuous recordings.
- Indexing jobs currently use the existing in-memory executor/status convention. Completed sessions persist; a restarted server cannot resume an unfinished index. Cancellation and persistent job recovery are outstanding.
- Files lacking valid frame durations/timestamps or failing complete decoding are rejected, not repaired or nominal-fps scored. Declared durations cannot reveal a freeze hidden by the encoder as ordinary frames. Container tail/edit-list behavior and browser clock alignment still need checks on the actual camera exports; exported duration is the decoded source interval, not an assertion about an unseen acquisition tail.
- Browser playback depends on codec support; exact source-image inspection is the fallback. Rotated and nonzero-start recordings need real-browser timing/orientation review before claiming general camera compatibility.
- Updating the analysis window cannot silently discard existing annotations; edit/remove those intervals first. Anonymous identifiers are required but not automatically deduplicated across sessions; split/dedup enforcement belongs to the training milestone.
- Records are for one local app process. Independent blinded scorers should use separate scoring copies until multi-annotator adjudication support is implemented. Do not run multiple servers against the same session directory.

## Plan when experimental videos arrive

1. Inspect a small set of original, untrimmed examples first. Obtain anonymous animal/session mapping, approximate positive examples if known, and the intended analysis window. Include ordinary negative/competing behavior, turns, occlusion, different lighting/cage positions, and each target action that actually occurs. Do not select only striking positives.
2. Check frame rate/PTS, native mouth/paw detail, reflections, bedding, bars/objects, external labels, camera movement, and whether a single mouse is visible. Produce a per-behavior observability report before choosing landmarks or resizing. Keep full source resolution for the feasibility judgment.
3. Agree on the draft ethogram with positive, negative, and ambiguous examples. Determine gnawing target/object and whether bar-mouthing or scratching needs its own output. Confirm whether this assay is single-mouse and whether analysis spans whole recordings or a protocol-specific window.
4. Split animals before model adaptation. Keep repeated sessions, crops, pose labels, and augmented clips from an animal together. Hash/deduplicate source videos. Reserve a locked test set and do not use its corrections or poses for the current training cycle. Keep genotype/treatment separate and hidden from labelers and features. [Grouped-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
5. Proposed feasibility budget, not a statistical adequacy claim: aim for 8–12 distinct animals if available; a 12-animal example is 6 development-training / 3 tuning / 3 locked test. Start with 200–400 diverse pose frames and 10–20 minutes of completely scored short behavior clips from training animals. Adapt the budget to positive-event diversity, scorer agreement, and learning curves. A smaller set can establish feasibility only.
6. Select random early/middle/late clips plus human-identified positives and confounders. Reserve full held-out sessions at natural prevalence for final comparison. Add a second blinded scorer on overlapping material; resolve definitions before freezing reference labels. Exact filenames/timestamps can only be selected after the files exist.
7. Trial a short permitted side-view pose model. Candidate landmarks: nose, ears, mid-back, forepaws, hindpaws, optionally shoulders/tail base. Preserve anatomical identity through turns; never relabel the visible paw as a fixed left paw. Review missingness, jumps, orientation, and motion blur.
8. Train the grooming baseline only when usable labels exist. Escalate to a pixel model if pose systematically loses visible fine motion or confuses grooming and feeding despite adequate labels. Compare methods on the same development animals and human-review budget; final testing stays locked.

## Future analysis contracts

- **Pose:** immutable raw tracks, source hash/frame mapping, confidence, body-part schema, crop map, backend/version; mismatched frame counts or coordinate systems fail explicitly.
- **Features:** body-relative distances/angles, head-to-paw motion, posture, temporal velocity/dispersion, quality/missingness features; use elapsed seconds for windows. Fit any imputation/selection only within training folds. Do not interpolate hidden paw movements into evidence.
- **Classifier:** per-behavior binary scores plus validity/unknown reason, from an independent sklearn pipeline. Class weighting and threshold tuning happen on development animals. Store class order, thresholds, feature/pose/ethogram versions and preprocessing together. Do not import arbitrary pickle/joblib files from users.
- **Predictions:** source frame ID, interval start/end, behavior, raw score and score type, state, unknown reason, observability, model ID, run ID. Raw predictions are immutable; human corrections form a separate overlay. No calibrated-probability claim unless calibration is evaluated.
- **Model card:** training/split manifest hashes, code commit, dependency versions, random seed, initialization/data/weight rights, allowed use, camera/setup domain, thresholds, segmentation configuration, validation report, per-behavior readiness.
- **Batch integration:** re-use job presentation and local exports, but dispatch by assay. Replace top-view stages with indexing → pose/appearance features → temporal prediction → review → measurements. Do not trigger the existing arena localizer. Persist/cancel/resume with frozen run manifests before unattended use.

## Validation gates

Evaluate each behavior and endpoint independently. Software tests establish timing and export mechanics only.

- **Recognition:** per-behavior precision/recall/F1 and PR curves at natural prevalence; explicit confusions with feeding, drinking, scratching, nesting, and immobility.
- **Duration:** per-animal signed bias and absolute error in seconds, manual-vs-predicted agreement/limits of agreement. Correlation alone is insufficient.
- **Events:** one-to-one interval matching, declared IoU rule, event precision/recall/F1, splits/merges and onset/offset errors. Report censored observed segments separately from complete bouts.
- **Coverage:** human-observable and algorithmically scored fractions; unknown reason distribution. Operational recall counts algorithmic abstentions on human-visible positives as misses, so abstention cannot inflate apparent performance.
- **Uncertainty:** animal-level resampling when sample size permits, with numbers of animals, sessions, positive bouts and positive seconds. Missing test positives mean recall is not estimable.
- **Utility:** baseline manual time versus labeling+correction+audit time, plus compute time on named hardware. An accurate classifier that requires almost full manual review has not met the workflow goal.

Discuss and predeclare tolerances before the locked evaluation. The guide's candidate 0.85 precision/operational recall, 0.80 event F1, and `max(5 s, 10% of reference positive time)` duration tolerance are engineering discussion points, not published standards or agreed lab criteria. Rare gnawing may demand much tighter absolute error. Readiness states should distinguish manual only, training needed, experimental, duration validated, and bouts validated; success for grooming must not promote another label.

## Dependency/model provenance decisions

Current upstream terms were inspected where available; release-specific audits remain necessary before distribution. This work adds no new package or model dependency.

| Item | Current evidence / handling |
| --- | --- |
| Existing application | No license selected, per repository contributing guide; unchanged |
| SimBA | Modified academic/research-use license; no copied code or bundled dependency. [License](https://raw.githubusercontent.com/sgoldenlab/simba/master/LICENSE) |
| DeepEthogram | Internal academic/noncommercial license restricts redistribution and derivatives; no copied code or package. [License](https://github.com/jbohnslav/deepethogram/blob/master/license.txt) |
| DLC/SuperAnimal | Distinguish core software terms from research/noncommercial pretrained weights; no weights fetched in this task. [Official README](https://raw.githubusercontent.com/DeepLabCut/DeepLabCut/main/README.md) |
| CBAS | Repository advertises MIT; exact license file retrieval failed in this check, so transitive code/weight terms remain unverified. Candidate only. [Repository](https://github.com/jones-lab-tamu/CBAS) |
| JAX assets | Code, annotations, container and weights require separate provenance checks; no multi-GB download initiated. [Resource record](https://zenodo.org/records/4646088) |

The next consequential scientific decision is whether the real side view resolves the actions the lab intends to measure. The reviewer and export contracts are built to remain useful if one or more automatic detectors never meets that standard.
