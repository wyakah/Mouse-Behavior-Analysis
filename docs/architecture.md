# Architecture

## Analysis path

1. Flask stores uploads under `videos/` and reads their metadata.
2. The browser seeds regions from the saved camera profile, then collects per-recording review. A shared circle diameter and likelihood cutoff apply to the whole batch.
3. The server validates IDs, duplicate files, framing, confirmation state, floor geometry, and circle containment/non-overlap before enqueueing an immutable batch snapshot.
4. A single worker processes each recording sequentially. It prepares the first 600 seconds, locates the subject, and runs DeepLabCut. Verified matching predictions can be reused.
5. The scoring engine combines full-source landmark coordinates with source presentation timestamps. It measures circle membership and chamber occupancy and exports a review video with the same frame timeline.
6. Recording outputs persist individually. The workbook aggregates summaries, configuration, provenance, and bouts. Errors are retained per recording.

## Live monitoring

An optional publisher replaces `batches/<id>/live/snapshot.json` atomically with one cropped JPEG and matching frame metadata. Localization and review publish from their existing decode loops; DLC pairs its prediction-writer callback with a bounded, thread-safe cache of arena crops sized for its inference prefetch queue. There is no second video decode for the preview. Encoding is throttled to two updates per second, with a final-frame update at the end of a pass.

The live API omits unchanged images and rejects snapshots belonging to a different recording. The browser polls without overlapping requests, guards against stale responses, and stops image transfers when the preview is hidden. Completed result videos stay mounted across progress updates. Missing or failed preview telemetry cannot change measurements or terminate scoring. The [live analysis plan](live-analysis-plan.md) documents the experience and verification.

## Coordinate systems

- Full-source video pixels are authoritative for landmarks and cup circles.
- The arena crop is a display/inference optimization; predicted coordinates are mapped back before scoring.
- A projective floor mapping defines chamber partitions from four reviewed corners and divider fractions.
- Circle scoring uses exact Euclidean radius in original pixels, including the boundary; it does not use a polygon approximation or infer centimeters.
- Optional legacy calibrated scoring remains in the engine and manual interface.

## Local files

| Directory | Role | Stored in Git? |
| --- | --- | --- |
| `profiles/` | Camera framing and provisional geometry | Yes |
| `videos/`, `prepared/` | Original and working video | No |
| `configs/`, `batches/` | Per-recording setups and saved batch state | No |
| `labeling/` | Native frames, annotations, histories, exports | No |
| `dlc-projects/`, `.cache/`, `.dlc-env/` | Models, downloads, runtime | No |
| `outputs/`, `reports/`, `research/` | Generated data and local research work | No |

The server binds to loopback and checks request host/origin. It is a single-user local tool, without production authentication or concurrent multi-user draft editing. Keep it on the local machine.

## Portability

Core tests and scoring require the Python UI environment. Actual inference additionally requires the separate DLC environment and downloaded weights. The workbook builder uses the external Codex Artifact Tool runtime; the application reports an export failure while retaining video outputs when that runtime is missing. Model and workbook runtimes are not vendored.

The `scripts/` directory includes earlier sample-specific experiments and report builders. Those require their original local inputs; they are not part of clean-checkout CI.
