# Buffered annotated analysis video

## Product contract

The Results screen plays continuous annotated footage while analysis proceeds independently. It retains every frame in source order, with the original relative presentation timestamps. Playback starts once at least three seconds of footage are buffered (normally two approximately two-second segments), or immediately when a shorter completed clip is available.

Model preparation and subject localization happen first. The initial loading state is explicit. A fixed wall-clock delay is not promised: slow inference can cause buffering; faster processing increases the amount of footage available ahead of playback. The player shows **Watching** and **Video ready through** separately from current-pass processing progress.

## Experience

- Native video controls: pause, play, replay buffered footage, and expand. Hide/show preserves position; Focus view enlarges the arena.
- Matching cup circles and reviewed chamber boundaries; nose, center, and tail-base overlays share one renderer with the final review export.
- Solid points are accepted by the confidence/validity rules. Dashed points are uncertain; missing points are absent. Frame-specific likelihoods are burned into the video so they cannot drift ahead of playback.
- During inference the caption identifies tracking preview. The final scored review additionally shows chamber and cup-zone state. Behavioral totals are finalized after scoring.
- Processing another recording never replaces footage the user is still watching. **Watch active recording** switches explicitly; reaching the end automatically advances to the active recording, if one exists. Queue buttons allow replay of available recordings.
- Batch completion preserves playback. Completed result cards and final downloads remain available below.
- Missing encoders, failed previews, interrupted processing, unavailable browser support, and reconnecting transport have explicit states. Preview failure never substitutes invented frames or changes scientific results.

## Architecture

1. DLC's prediction-writer hook pairs each prediction with the corresponding original arena crop. Its thread-safe cache covers the queued, active, and producer inference batches.
2. A dedicated encoding worker uses a bounded 64-frame queue. It writes H.264 baseline fragmented-MP4 segments with source-derived frame durations. Each immutable segment is published only after closing; an atomic manifest lists contiguous available segments.
3. Frames are encoded relative to their segment's first timestamp. The browser applies each segment's source offset using Media Source Extensions. This preserves variable source intervals across boundaries.
4. The UI begins after a small buffer and appends segments without replacing the video element. Transport requests do not overlap. Recording changes abort stale requests; hiding finishes an in-flight append before suspending further transfers, avoiding duplicated segments.
5. Each recording retains its own manifest and segments under `batches/<id>/live/video/<index>/`. A complete inference stream stays available during scoring/export. Cached predictions produce the stream during review rendering instead.
6. `AnnotationRenderer` draws both the stream and final review. The original final MP4 and scientific table exports remain separate, so an interrupted preview cannot invalidate them.
7. Existing live telemetry supplies processing stages and frame counts without JPEG images. The sampled-image canvas and detached likelihood cards have been removed from the interface.

## Resource and failure behavior

No additional source decode occurs during inference. The preview encoder runs independently of playback speed. If its bounded queue fills, the preview reports failure rather than silently dropping frames or blocking inference. A prematurely ended frame sequence is marked failed. Final review rendering can provide a new stream when an earlier preview failed.

Segments persist locally for replay and are excluded from Git. One browser player holds one recording at a time; changing recordings releases its media buffers. A ten-minute sample produced approximately 72 MiB of preview segments. Browser codec support is checked before opening the stream; the final downloadable MP4 remains available if streaming is unsupported.

## Dependencies

The UI environment requires PyAV 18. The separate DeepLabCut environment also needs `requirements-dlc-preview.txt`. Missing preview imports warn and preserve normal inference; existing tracking weights and inference batches are unchanged.

## Verification · September 5, 2026

- **82 Python tests and three JavaScript playback lifecycle tests pass.** JavaScript syntax checks also pass.
- Python tests cover segment publication before completion, variable frame rates, exact timestamps and durations, shared crop/source drawing, missing/uncertain landmarks, scientific output equivalence, bounded-queue overload, interrupted streams, and API path/restart behavior.
- JavaScript lifecycle tests cover short-clip buffering, hiding during an append without duplicate segments, and preserving manually paused playback across status refreshes.
- Two fresh DeepLabCut sample runs produced **448 frames each**, across ten playable segments per recording. Every streamed frame timestamp matched its source within one microsecond. Raw poses and selected tracking tables matched the prior inference output exactly.
- A full **600-second, 13,482-frame** sample using content-matched predictions produced **300 segments**. Every streamed and final-review frame retained its timestamp. All prior frame measurements and the summary matched exactly; Excel export succeeded.
- Browser checks in the in-app browser exercised actual H.264 playback, replay of a completed recording during the next recording's inference, automatic queue advance, Focus view, hide/show at the same playback position, and playback continuing after analysis completion.
- A standalone warm encoding check drew and encoded 500 repetitions of an arena frame in approximately 0.55 seconds. One warm 448-frame DLC run with streaming took 14.4 seconds. These are local observations, not latency or throughput guarantees.

Validation used isolated local workspaces. Test recordings, predictions, and generated media are excluded from Git. Tracking accuracy remains scientifically unvalidated; this feature changes review and monitoring, not model weights.
