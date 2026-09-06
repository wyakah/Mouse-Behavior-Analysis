# Mouse Behavior Analysis

Behavior Studio is a local desktop tool for Three Chamber and Stereotypy mouse-behavior analysis. Upload recordings, enter mouse metadata, run analysis with a live annotated preview, and export Excel results and review videos.

**Status: research prototype.** Tracking accuracy has not been validated to 95–99%. Confidence coverage and correct landmark placement are different measures. Review the annotated videos before interpreting results.

## Download and install

[**Download the Windows x64 installer (v0.1.4)**](https://github.com/wyakah/Mouse-Behavior-Analysis/releases/tag/v0.1.4)

[**Download the Apple Silicon Mac installer (v0.1.3)**](https://github.com/wyakah/Mouse-Behavior-Analysis/releases/tag/v0.1.3)

| Platform | Availability |
| --- | --- |
| Apple Silicon Mac (M1 or later), macOS 14+ | Development installer available; about 1.2 GB download / 2.2 GB installed |
| Intel Mac | Not built or tested |
| Windows 10/11 x64 | Development installer available; bundled CPU inference runtime |
| Linux | No desktop installer |

### Mac

1. Open the Mac release above and download `Behavior-Studio-0.1.3-apple-silicon.dmg`.
2. Open the disk image and drag **Behavior Studio** into **Applications**.
3. Open **Behavior Studio**. Python and the models are included; no terminal or separate installation is required.
4. Choose **Three Chamber** or **Stereotypy**, upload videos, and enter mouse ID, sex, and genotype. For Three Chamber, choose the stranger mouse’s left/right position for each recording. Three Chamber additionally requires checking its cup zones and chamber geometry.
5. Run analysis and export the results. Three Chamber scores the first 10 minutes; Stereotypy scores the first 20 minutes, or the available duration if shorter.

This development build has a local ad-hoc signature, **not an Apple Developer ID signature or notarization**. macOS may block opening it. Only proceed if you trust this repository and have verified the release checksum; Apple’s [instructions for opening an app from an unidentified developer](https://support.apple.com/en-us/102445) explain the per-app controls. Do not disable Gatekeeper globally.

Recordings and results stay on your computer in `~/Library/Application Support/com.wyakah.behaviorstudio/workspace/`. Exported downloads go to Downloads. The app works offline after installation. Its bundled SuperAnimal models are limited to academic, non-commercial use under their [upstream notices](desktop/THIRD_PARTY_NOTICES.md).

### Windows

Download `Behavior-Studio-0.1.4-windows-x64-setup.exe` from the Windows release. Run the installer, then open **Behavior Studio** from the Start menu. It installs for your Windows user account and includes Python, the CPU inference libraries, both assay models, and the Microsoft C++ runtime.

This is an **unsigned development installer** for Windows 10/11 x64. Windows may show an unknown-publisher or SmartScreen prompt. Verify the published SHA-256 checksum and only proceed if you trust this repository. The Microsoft WebView2 bootstrapper needs internet if WebView2 is not already installed; analysis and bundled models run locally afterward. No separate Python or DeepLabCut installation is needed. The first launch takes longer while the engine is unpacked; later launches reuse that local copy.

The installer is tested on a GitHub-hosted Windows Server 2022 runner, including both analysis pipelines, installed-runtime exports, and native startup/quit. Synthetic smoke videos test software operation, not tracking or behavior accuracy. GPU acceleration is not enabled in this Windows CPU build.

### Why is the installer large?

The Mac build packages the entire tested analysis environment: about **1.8 GB for Python, native libraries, PyTorch, DeepLabCut and pose weights**, plus **362 MB for the remaining models**. The web interface and native window are a small part. Compression reduces the installed 2.2 GB application to approximately 1.2 GB. The Windows installer is about **1.1 GB** and unpacks its engine on first launch. There are no laboratory recordings in either installer.

A smaller future installer can download assay-specific models and omit unused runtime components. That reduces the initial download; models and inference libraries still need local disk space. The Windows build bundles a separate x64 CPU runtime and uses native Windows process management.

See [desktop build instructions](desktop/README.md) and [validation results](desktop/VALIDATION.md).

## What it does

- **One workflow for one video or a batch:** Choose test → Videos → Test setup/review → Results. Both tests share the same upload and mouse metadata screen, with up to 20 videos per setup.
- **Mouse metadata and statistics:** mouse ID, sex, and genotype; descriptive summaries, genotype comparisons, within-sex comparisons, and genotype × sex ANOVA in Excel. One independent mouse per sample ID.
- **Test selection:** choose Three Chamber for tracking and group statistics, or Stereotypy for automatic side-view behavior estimates. Stereotypy models remain experimental; automation does not establish accuracy.
- **Matching cup circles:** one shared diameter across both cups and all recordings, with independently movable centers.
- **Visual region review:** drag cup circles, floor corners, and chamber dividers; confirm each recording before analysis.
- **Automatic processing:** score the first 600 seconds, or the available duration for shorter recordings. Originals are retained.
- **Buffered analysis video:** watch smooth annotated footage with every frame in order, landmarks, cumulative measurements, processing stages, and the recording queue. Previews start collapsed; click to watch or expand into Focus view while analysis continues. Hidden previews pause playback and segment downloads.
- **Subject tracking:** background-based body localization followed by actual DeepLabCut landmark inference.
- **Three-point labeling:** nose, body center, and tail base. Only manually reviewed frames enter a training export.
- **Reviewable outputs:** combined workbook, per-frame measurements, bouts, geometry, provenance, and annotated video.

The current camera profile targets **1024 × 768 EthoVision exports** with a fixed arena crop. It is not a general-purpose video tracker.

## Run from source

The UI and scoring engine use Python 3.12. They are separate from the heavier DeepLabCut environment.

```bash
git clone https://github.com/wyakah/Mouse-Behavior-Analysis.git
cd three-chamber
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8765**. On macOS, subsequent launches can use `Launch Three Chamber.command`. The server listens on loopback; it is not configured as a public web service.

A fresh checkout opens with an empty recording queue. Add your videos through the app. Recordings, labels, trained weights, and generated results are intentionally not stored in Git.

**Additional requirements for full analysis:** install DeepLabCut and supply the model artifacts described below. Cloning the source alone does not install the trained stereotypy heads. Use the desktop release for a complete packaged installation.

### DeepLabCut setup

The development installation uses **DeepLabCut 3.0.1** with PyTorch in `.dlc-env`. See the [official installation guide](https://deeplabcut.github.io/DeepLabCut/docs/installation.html) for platform-specific prerequisites.

```bash
python3.12 -m venv .dlc-env
.dlc-env/bin/python -m pip install 'deeplabcut==3.0.1'
.dlc-env/bin/python -m pip install -r requirements-dlc-preview.txt
```

`requirements-dlc.lock.txt` records the complete development environment; it is an environment snapshot, not a universal cross-platform lockfile. The UI environment snapshot is `requirements.lock.txt`.

By default, inference runs with `.dlc-env/bin/python`. Set `DLC_PYTHON` to use another local DLC installation. The initial pretrained run may download SuperAnimal weights. Weights are not included in this repository. Apple Silicon MPS was exercised during development; the inference script falls back to CPU when MPS is unavailable. CUDA selection has not been implemented in the current automatic path.

### Excel export

The production Excel exporter uses `openpyxl`, included in `requirements.txt` and the desktop runtime. Neither Node nor Codex is required for analysis exports. Older research/reporting scripts using the Codex Artifact Tool remain in `scripts/`; they are not part of the automatic export path.

## Using the app

1. **Choose test:** select Three Chamber, select Stereotypy, or continue your saved Three Chamber setup. Stereotypy opens an automatic video queue; **Choose test** returns to the shared selection screen.
2. **Videos:** upload up to 20 recordings and enter a unique mouse ID, sex, and genotype in the shared table. Missing sex/genotype remain explicit. Click **Continue**. Three Chamber’s optional Excel comparisons are in the collapsed **Excel statistics** section.
3. **Regions:** check the social/novel cup side. Move the left and right circles over their cup regions; resize either handle to change the common diameter. Check the floor and chamber dividers. Confirm each recording, then click **Analyze**.
4. **Results:** follow compact progress as recordings process one at a time; click the thumbnail to watch annotated footage. Completed videos become available immediately. When the batch finishes, download the combined Excel table and inspect the annotated videos. Filter results by sample ID, sex, or genotype. Completed previews are also click-to-play; detailed exports and previous analyses are collapsible.

The live viewer plays continuous annotated video after a small starting buffer, once model preparation and localization finish. Each frame retains its source timestamp. **Watching** and **Video ready through** distinguish playback from processing: slow analysis may buffer, while fast analysis runs ahead. Pause, replay available footage, hide/show, or use Focus view; processing continues independently. **Watch active recording** switches to the current job without forcing you away from an earlier video.

Nose, body-center, and tail-base markers share a renderer with the final review video. Solid markers are accepted, dashed markers are uncertain, and missing points are absent. Three Chamber previews show mouse ID, sex, genotype, cumulative chamber and cup-zone seconds, Chamber Stranger Interaction %, and Zone Stranger Interaction % using the same frame classifications as final scoring. Eight cumulative cards use four columns and two rows. Center Zone is N/A because only left and right cup zones are defined. Stranger/Object labels sit inside the top center of their chambers; chamber labels in the video, results table, and Excel include each role. Mixed-side batches retain per-mouse role labels in Excel. Chamber Stranger Interaction % uses stranger-side chamber occupancy; Zone Stranger Interaction % uses stranger nose-in-zone time. Both divide by the sum of all three chamber times. Missing noses remain unscored; `tracking_quality.json` records their gaps and body-in-cup-zone context for automatic diagnostics. Browser streaming requires Media Source Extensions with H.264 support. If preview encoding or playback fails, tracking and final exports remain available. See the [buffered video design and verification notes](docs/live-analysis-plan.md).

Changing sample metadata or statistical settings preserves region confirmations. Changing the shared diameter or likelihood cutoff clears all region confirmations. Coordinates and reference-frame selection are under **Precise placement & reference image**; the cutoff is under **Advanced tracking settings**. Camera scale and framing must remain consistent across a batch. Matching image resolution alone does not establish matching physical scale.

## Measurement definitions

| Measurement | Definition |
| --- | --- |
| Chamber Stranger Interaction % | `100 × stranger-side chamber seconds / (left + center + right chamber seconds)`. |
| Zone Stranger Interaction % | `100 × stranger nose-in-zone seconds / (left + center + right chamber seconds)`. Unknown/outside chamber time is excluded; missing data or a zero denominator is blank. |
| Cup-zone time | Accepted subject nose inside the selected circle, including its boundary. Circles cannot overlap or extend beyond the reviewed floor. |
| Chamber occupancy | Accepted body center in the left, center, or right partition. A divider boundary belongs to the chamber on its right. |
| Unknown chamber time | Body center unavailable or below the likelihood cutoff. |
| Outside chamber time | Accepted body center outside the reviewed floor. |
| Time weighting | Actual video presentation timestamps, clipped to the analysis window. |
| Preference index | `(target cup time − other cup time) / (target cup time + other cup time)`; unavailable if the target is unspecified or the denominator is zero. |

Low-confidence or missing landmarks are not interpolated. Left, center, right, unknown, and outside chamber times sum to the analyzed duration. Occupancy lower/upper bounds only allocate unknown tracking time; they do not cover identity, landmark, or geometry errors.

**No physical dimensions are required for selected-circle scoring.** A circle in video pixels does not establish a 1 cm distance, and nose-in-zone time is a proximity proxy rather than confirmed sniffing. The assay reference is [Zahran et al. (2024)](https://doi.org/10.1016/j.heliyon.2024.e36352); the selected-circle method is a deliberate measurement variation, not an exact reproduction of its calibrated proximity criterion.

## Outputs

Each completed recording has a folder under `outputs/`:

| File | Contents |
| --- | --- |
| `summary.csv`, `summary.json` | Durations, coverage, preference, scoring settings, and method information |
| `frames.csv` | Frame coordinates, likelihoods, zone flags, chamber assignments, and duration weights |
| `bouts.csv` | Contiguous cup-zone events with start times, durations, and exclusive end-frame indices |
| `calibration.json` | Region settings used for the run |
| `manifest.json` | Source/track hashes, method version, and provenance |
| `review.mp4` | Silent H.264 annotated video retaining source frame timestamps |

The combined workbook is `outputs/<batch-id>/results.xlsx`, with **Results** (including sample ID, sex, genotype, and test), **Setup**, **Bouts**, **Groups**, and **Statistics notes** sheets. Selected tests add **Comparisons** and/or **ANOVA**; omitted observations or metadata add **Exclusions**. Failed or unavailable measurements stay blank. A failed recording does not prevent subsequent recordings from being analyzed.

The batch draft and job state are stored in `batches/`. A server restart marks an interrupted job for retry; it does not resume training or inference automatically. Existing tracking predictions are reused only when their manifest matches the working video's content hash.

## Group statistics

Select outcomes before running the batch: social/novel cup time, other cup time, preference index, chamber occupancy, or left/right cup times. Target-relative measures follow each recording's reviewed cup side, allowing left/right counterbalancing. Do not combine different test phases or experimental conditions without accounting for the study design.

- **Group summaries only** (default): valid n, missing n, mean, sample SD, SEM, and unadjusted 95% t confidence intervals, pooled by genotype and split by sex.
- **Compare genotypes**: two-sided pairwise Welch t-tests, pooling sexes.
- **Within each sex**: separate genotype Welch comparisons for female and male mice.
- **Genotype × sex**: OLS with sum contrasts and Type III two-way ANOVA, including interaction and partial eta squared. Requires both sexes in each genotype.
- **All comparisons**: all of the above, with one Holm correction family across every estimable selected outcome, pairwise test, and ANOVA term. Alpha can be 0.05 or 0.01. Confidence intervals remain unadjusted 95% intervals.

Each sample ID is one independent mouse; frames and bouts are never treated as replicates. Tests require matching analyzed durations and at least two valid mice per tested group (or per factorial cell). This minimum allows calculation; it does not establish adequate power. Missing metadata, unavailable outcomes, zero residual variance, and insufficient groups produce documented reasons instead of invented p-values. No automatic outlier or low-coverage filter is applied, except that an outcome with no scoreable observations is unavailable.

The factorial model assumes independent mice, approximately normal residuals, and constant residual variance. Within-sex significance alone does not establish a difference between sexes; that question requires the interaction. Tracking quality and statistical assumptions still require review.

Exports are **fixed reports**. Editing Excel cells does not recalculate tests. Change setup and rerun the batch to generate a new report; valid cached predictions are reused. The saved `batches/<batch-id>/statistics.json` records settings, individual observations, groups, exclusions, computed tests, software versions, and methods.

Methods: [SciPy Welch t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html), [statsmodels ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html), and [Holm adjustment](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html). See [workflow and statistics design](docs/workflow-statistics-plan.md).

## Labeling and model development

Open **Model tools → Review training labels** (or `/labeling`). Label the free subject's **nose tip**, **body center excluding the tail**, and **tail base**. Mark hidden landmarks as not visible rather than guessing their positions. Existing five-point annotations are accepted; new training exports use three points and retain an original-annotation audit copy.

Fine-tuning is an explicit separate operation; saving labels does not update model weights or change the automatic batch model. See [model development](docs/model-development.md) for the current sample-specific queue and split requirements, training commands, and validation limits.

## Development

### Automatic stereotypy

Open `/stereotypy`, upload up to 20 side-view recordings, enter mouse ID, sex and genotype, then click **Analyze videos**. Recordings process sequentially through the frozen six-behavior models. The first 1,200 seconds (or the full shorter source) are included. No annotation session, annotator ID, cage drawing or manual interval scoring is required. The full field of view is retained to avoid cropping out rearing and jumping. At each source interval the highest threshold-qualified behavior score wins; exact ties or unreliable localization are unknown, and intervals with no qualifying behavior are other activity. Head scores are uncalibrated, so this exclusive decision rule is not an accuracy improvement claim.

The results page restores running progress after refresh and provides six provisional exclusive candidate durations, other-activity and unknown time, and the shared collapsible live-video player with running results, combined Excel, scores CSV and interval CSV. Individual failures do not stop the remaining queue. Failed or interrupted batches can be retried. After source indexing and model warmup, inference and rendering advance in eight-second sections. Each section includes the temporal model’s three future samples before its scores are published. Two-second video segments become available while later sections are still being analyzed. Single-video previews open automatically; batch previews start collapsed. Video overlays show cumulative seconds at playback time; result cards show totals through the latest processed time. Exports use the same exclusive intervals, with no double-counted behavior time. Earlier overlapping results remain marked legacy. Three Chamber’s scored overlays also show cumulative chamber and cup-zone time. No statistics are performed.

Automatic scoring requires the locally installed `.dlc-env`, cached DINOv2/MobileNet weights and frozen v1/v2 model artifacts in `outputs/stereotypy-training`. Missing models produce an explicit installation error, never a manual scoring fallback. Models have not met the accuracy target. Unreliable body proposals are excluded automatically; these gates are not accuracy validation. The [training experiments](docs/stereotypy-priority-training.md) and [robustness diagnostics](docs/stereotypy-robustness.md) document limitations. Historical annotation data remain on disk for provenance but are not part of the product flow.

### Checks

```bash
.venv/bin/python -m pytest -q
node --test tests/*.test.cjs
```

Tests generate synthetic video and tracking fixtures. No sample recordings, GPU, model downloads, or Excel runtime are required. GitHub Actions runs the Python suite and JavaScript syntax checks.

Project layout:

```text
app.py                 Flask app and local API
threechamber/          Scoring, batch jobs, localization, and label export
static/                Analysis and labeling interfaces
profiles/              Camera profile (review before use on another setup)
scripts/               Inference, training, inspection, and export utilities
tests/                 Synthetic scoring, pipeline, and provenance tests
examples/              Example linked-circle configuration
docs/                  Architecture and model-development notes
```

For headless scoring with an aligned single-subject tracking table:

```bash
.venv/bin/python -m threechamber \
  --video videos/trial.mp4 \
  --tracks tracks/trial.csv \
  --calibration examples/linked-circles.json \
  --output outputs/trial
```

Review and set `confirmed` in your own configuration first. The example is provisional geometry, not a calibration for an arbitrary video. The older manual/calibrated interface is retained at `/advanced` for compatibility.

## Current limits

- Nose and body-center accuracy remain unvalidated; no 95–99% correctness claim is supported.
- Cup occlusion, reflections, shadows, and small snout size can produce missing or incorrect predictions.
- The fixed camera profile and localizer require validation for a different camera setup.
- Training utilities still use the original sample IDs for their development split; final evaluation requires independent animals/sessions.
- Excel export currently depends on the bundled Codex runtime.
- CI tests software behavior with synthetic fixtures; they do not establish scientific tracking accuracy.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions and [docs/architecture.md](docs/architecture.md) for the data flow.

## Shared recording setup

Three Chamber and Stereotypy use the same compact upload component and mouse table. File selection beyond the remaining 20-video capacity is rejected before uploading; saved drafts and setup validation also enforce the limit. Editing metadata does not reset Three Chamber region reviews.

Stereotypy saves its upload draft separately and restores persisted automatic batch progress. All exported rows include mouse ID, sex and genotype. Analysis continues while the page is closed.

### Stereotypy pose and motion comparison

See [the robustness workflow](docs/stereotypy-robustness.md) for the separate five-mouse pose, motion, scene and lighting comparison, raw-frame landmark editor, and guarded human-label export/fusion training. These are development diagnostics; local behavior accuracy is not established.

Pose and behavior distinction experiments, automatic anatomical overlays, and the
measured comparison are documented in [stereotypy distinction](docs/stereotypy-distinction.md).

## Desktop application

An offline Apple Silicon Mac desktop build is available in [desktop/README.md](desktop/README.md). It packages the interface, Python engine, and existing models; recordings and results stay in the user’s application workspace. The development build is not yet signed or notarized for public distribution.
