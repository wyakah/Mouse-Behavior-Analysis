# Live analysis preview

## Product contract

The Results screen becomes the live workspace when a batch starts. It shows the actual current recording, actual processing pass, the latest completed frame available from that pass, and the sequential queue. Completed measurements and review videos remain accessible throughout the batch.

A preview is not real-time video playback. It samples completed work at up to 2 Hz; analysis and export still process all frames. No synthetic mouse motion, interpolated landmarks, guessed progress percentages, or provisional behavioral totals are displayed.

## Experience

- Dark arena viewer with matched cup circles, reviewed chamber boundaries, and three color-coded landmarks.
- Localization shows only a labeled candidate box. It never masquerades as nose/body-center tracking.
- Accepted points are solid; low-confidence finite points use an uncertainty marker; missing points are not drawn. Confidence badges explain each state.
- Source timestamp, frame count, current-pass progress, batch completion count, and actual stage transitions.
- Hide/show preview without canceling work; optional expanded viewer. Reduced-motion support and readable narrow layouts.
- Completed videos remain mounted while progress refreshes. History inspection cannot replace or cancel the running job.
- Clear preparing, waiting-for-frame, stale connection, failed, interrupted, cached-track, and completed states.

## Architecture

1. An optional, best-effort publisher atomically replaces a bounded JSON snapshot containing one JPEG and its exact-frame metadata. Preview failure cannot fail scientific scoring.
2. Localization publishes a source frame and its actual candidate box from its decode loop.
3. DLC's existing per-frame prediction-writer hook publishes postprocessed predictions paired with a bounded cache of original frames. Inference batch size and prediction behavior are retained.
4. The scoring/review path publishes the original frame plus accepted scoring landmarks. Source timestamps and exported measurements remain unchanged.
5. The batch API exposes snapshots only for the selected local batch, without caching. Client requests are throttled and non-overlapping, with stale-response guards.
6. The UI updates the viewer and progress in place. It adds completed cards once instead of rebuilding their video elements on every poll.

## Verification

- Unit tests: snapshot throttling, matching image/metadata identity, confidence handling, failure isolation, bounded frame retention, and DLC prediction passthrough.
- Pipeline tests: preview on/off result equivalence, complete frame/PTS preservation, two-recording transition, failure continuation, missing/stale snapshots, and restart state.
- Real-data checks: existing sample scoring with live preview plus a fresh short DLC run to exercise actual inference callbacks.
- Browser checks: live stages, queue transitions, hide/show and expand, completed-video continuity, narrow layout, and final exports.
- Performance: record publication count/payload sizes and compare representative preview-on/off processing. Preview does not decode the source a second time during inference or queue frames for streaming.

## Scope

No new setup step, live scrubbing, manually paused inference, or continuously changing behavioral totals. The current model and scientific validation status are unchanged.

## Verification results · September 5, 2026

- **74 automated tests pass**, including preview failure isolation, exact image/prediction pairing, source-timestamp preservation, recording transitions, and stale API responses. JavaScript syntax checks pass.
- Fresh DeepLabCut runs on two native 20-second sample clips produced 448 predictions each, with live tracking snapshots from frame 0 through 447. Both recordings completed scoring, review videos, and the combined workbook. Largest observed snapshot was 38,407 bytes.
- Preview enabled versus disabled on the same 448-frame input produced **exactly identical raw poses and selected tracking tables**. Measured inference/export-script time was 14.23 seconds enabled versus 14.05 seconds disabled. This single-machine comparison is indicative, not a general performance guarantee.
- A complete 600-second sample using its existing content-matched predictions produced **13,482 scored frames**, exactly matching prior measurements and summary. All annotated-video timestamps matched the source within one microsecond. The full recording emitted observed preview frames from 0 through 13,481 and exported the workbook successfully.
- Browser review exercised actual DLC overlays, uncertain landmarks, sequential queue transitions, hide/show, Focus view, the narrow layout, and compact completion state. A completed review video continued from the start to 17.9 seconds across progress refreshes while the full recording was still processing.
- Real inference exposed a prefetch-buffer issue missed by the initial fixtures. The corrected cache retains arena crops across DLC's queued, active, and producer batches, with explicit regression coverage and a lock for producer/consumer access.

Validation runs used isolated local workspaces. Sample media, predictions, and generated test artifacts are excluded from Git.
