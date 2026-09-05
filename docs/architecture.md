# Architecture

## Analysis path

1. Flask stores uploads under `videos/` and reads their metadata.
2. The browser seeds regions from the saved camera profile, then collects per-recording review. A shared circle diameter and likelihood cutoff apply to the whole batch.
3. The server validates IDs, duplicate files, framing, confirmation state, floor geometry, and circle containment/non-overlap before enqueueing an immutable batch snapshot.
4. A single worker processes each recording sequentially. It prepares the first 600 seconds, locates the subject, and runs DeepLabCut. Verified matching predictions can be reused.
5. The scoring engine combines full-source landmark coordinates with source presentation timestamps. It measures circle membership and chamber occupancy and exports a review video with the same frame timeline.
6. Recording outputs persist individually. The workbook aggregates summaries, configuration, provenance, and bouts. Errors are retained per recording.

## Live monitoring

The inference prediction hook pairs every frame with its original arena crop and sends it to a bounded video-encoding worker. The worker publishes immutable fragmented-MP4 segments and an atomic per-recording manifest under `batches/<id>/live/video/<index>/`. The browser appends these segments to one native video element, preserving source time offsets. Playback, buffering, and pausing never control analysis scheduling.

One annotation renderer serves both streamed footage and the final review export. Completed streams remain available while the batch advances. Cached predictions create their stream during review rendering. Metadata-only telemetry supplies current processing stages independently of the watched recording. Encoder overload, interruptions, and unavailable browser support have explicit failure states; they cannot change scientific tables or stop inference. See the [buffered video design](live-analysis-plan.md) for lifecycle, resource, and verification details.

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
