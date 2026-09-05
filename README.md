# Three Chamber

A local tool for analyzing three-chamber mouse-behavior recordings from EthoVision. Track the free subject with DeepLabCut, measure nose time inside matching cup zones and occupancy in all three chambers, then export a combined Excel table and annotated review videos.

**Status: research prototype.** Tracking accuracy has not been validated to 95–99%. Confidence coverage and correct landmark placement are different measures. Review the annotated videos before interpreting results.

## What it does

- **One workflow for one video or a batch:** Videos → Review regions → Results.
- **Matching cup circles:** one shared diameter across both cups and all recordings, with independently movable centers.
- **Visual region review:** drag cup circles, floor corners, and chamber dividers; confirm each recording before analysis.
- **Automatic processing:** score the first 600 seconds, or the available duration for shorter recordings. Originals are retained.
- **Buffered analysis video:** watch smooth annotated footage with every frame in order, landmark confidence, processing stages, and the recording queue. Hide the preview or expand it into Focus view while analysis continues.
- **Subject tracking:** background-based body localization followed by actual DeepLabCut landmark inference.
- **Three-point labeling:** nose, body center, and tail base. Only manually reviewed frames enter a training export.
- **Reviewable outputs:** combined workbook, per-frame measurements, bouts, geometry, provenance, and annotated video.

The current camera profile targets **1024 × 768 EthoVision exports** with a fixed arena crop. It is not a general-purpose video tracker.

## Quick start

The UI and scoring engine use Python 3.12. They are separate from the heavier DeepLabCut environment.

```bash
git clone https://github.com/wyakah/three-chamber.git
cd three-chamber
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8765**. On macOS, subsequent launches can use `Launch Three Chamber.command`. The server listens on loopback; it is not configured as a public web service.

A fresh checkout opens with an empty recording queue. Add your videos through the app. Recordings, labels, trained weights, and generated results are intentionally not stored in Git.

**Additional requirements for full analysis:** install DeepLabCut as described below. The current Excel exporter also requires the Codex Artifact Tool runtime described under [Excel export](#excel-export). You can open the UI and run the automated tests without either of those runtimes.

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

The current workbook builder, `scripts/export_batch.mjs`, uses `@oai/artifact-tool` from the **bundled Codex runtime**. This is not installed by `pip` and no public npm installation is assumed.

The default dependency root is:

```text
~/.cache/codex-runtimes/codex-primary-runtime/dependencies
```

To use another installed copy:

```bash
export CODEX_WORKSPACE_DEPENDENCIES=/absolute/path/to/dependencies
```

That directory must contain `node/bin/node` and `node/node_modules/@oai/artifact-tool`. If the runtime is unavailable, analysis CSVs and review videos are retained and the batch reports an Excel-export failure. A standalone Excel backend is a portability improvement still to be implemented.

## Using the app

1. **Videos:** add one or more recordings, or select existing workspace recordings. The app automatically chooses available prepared trial copies.
2. **Review regions:** check the recording ID and social/novel cup side. Move the left and right circles over their cup regions; resize either handle to change the common diameter. Check the floor and chamber dividers. Confirm each recording, then click **Analyze**.
3. **Results:** watch the live arena preview as recordings process one at a time. Completed videos become available immediately. When the batch finishes, download the combined Excel table and inspect the annotated videos. Detailed exports appear beneath each result.

The live viewer plays continuous annotated video after a small starting buffer, once model preparation and localization finish. Each frame retains its source timestamp. **Watching** and **Video ready through** distinguish playback from processing: slow analysis may buffer, while fast analysis runs ahead. Pause, replay available footage, hide/show, or use Focus view; processing continues independently. **Watch active recording** switches to the current job without forcing you away from an earlier video.

Nose, body-center, and tail-base markers share a renderer with the final review video. Solid markers are accepted, dashed markers are uncertain, and missing points are absent. Confidence values belong to the displayed frame. The tracking preview identifies its provisional stage; the final scored review adds chamber and cup-zone state. Browser streaming requires Media Source Extensions with H.264 support. If preview encoding or playback fails, tracking and final exports remain available. See the [buffered video design and verification notes](docs/live-analysis-plan.md).

Changing the shared diameter or likelihood cutoff clears all region confirmations. Coordinates and reference-frame selection are under **Precise placement & reference image**; the cutoff is under **Advanced tracking settings**. Camera scale and framing must remain consistent across a batch. Matching image resolution alone does not establish matching physical scale.

## Measurement definitions

| Measurement | Definition |
| --- | --- |
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

The combined workbook is `outputs/<batch-id>/results.xlsx`, with **Results**, **Setup**, and **Bouts** sheets. Failed or unavailable measurements stay blank. A failed recording does not prevent subsequent recordings from being analyzed.

The batch draft and job state are stored in `batches/`. A server restart marks an interrupted job for retry; it does not resume training or inference automatically. Existing tracking predictions are reused only when their manifest matches the working video's content hash.

## Labeling and model development

Open **Model tools → Review training labels** (or `/labeling`). Label the free subject's **nose tip**, **body center excluding the tail**, and **tail base**. Mark hidden landmarks as not visible rather than guessing their positions. Existing five-point annotations are accepted; new training exports use three points and retain an original-annotation audit copy.

Fine-tuning is an explicit separate operation; saving labels does not update model weights or change the automatic batch model. See [model development](docs/model-development.md) for the current sample-specific queue and split requirements, training commands, and validation limits.

## Development

```bash
.venv/bin/python -m pytest -q
node --test tests/stream-player.test.cjs
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
