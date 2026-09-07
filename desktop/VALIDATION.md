# Development build validation — 2026-09-06

Apple Silicon Mac, version 0.1.0. Packaging retains the existing scoring models and thresholds.

- Bundled Python runtime: **187 tests passed**.
- Development Python runtime: **177 passed, 3 skipped**.
- JavaScript interface tests: **8 passed**.
- Fresh isolated workspace, HTTP authentication, video upload, model readiness, sequential worker, live MP4 segments, annotated output, and downloaded Excel workbook exercised with both assays.
- Stereotypy: a 12-second excerpt from mouse 756 completed in 76.5 seconds, including first-use model/font loading; live segments observed.
- Three Chamber: a 4-second excerpt from mouse 675 completed in 20.2 seconds; live segments observed.
- Final disk-image CRC verification passed. App signature verification passed both before and after launch from `desktop/dist/`.
- Native Tauri window opened the test-selection screen from its own application bundle. Normal Quit terminated its private local service.
- Workspace installation/reinstallation tests preserve recordings and configuration. Interrupted-job recovery preserves completed entries. Session endpoints reject requests without the private session cookie. Excel metadata is stored as text, including strings beginning with `=`.

These are packaging and functional smoke checks, not new accuracy benchmarks or clean-machine certification. Full 20-minute runtime performance, native upload/download interaction, and installer behavior on other Macs still need release validation. Developer ID signing, notarization, Windows support, and automatic updates are not part of this development build.

Local smoke reports and workbooks are retained under `.cache/desktop-smoke/` and are excluded from Git. Reproduce the HTTP tests with `desktop/smoke_test.py`; see its `--help` output.

## Version 0.1.1 — stranger interaction

- 191 bundled-runtime Python tests and 9 JavaScript tests passed.
- Explicit left/right setup, legacy target-side compatibility, timestamp-weighted stranger scoring, missing-data handling, zero denominator, and Excel cell values covered.
- Packaged Three Chamber HTTP upload → tracking → scoring → live segments → review video → Excel test completed on a four-second mouse 675 excerpt in 40.2 seconds. The selected right-side numerator and summed chamber-time denominator matched the exported percentage.
- Prior result artifacts are retained; rerun an analysis to produce the new columns and cumulative video overlay. Historical on-screen results can derive the percentage when a target side and valid times were recorded.

## Version 0.1.2 — cumulative live preview

- Packaged runtime: 193 tests passed. Development runtime: 183 passed, 3 skipped. JavaScript: 9 passed.
- Live pose frames and final results share geometry/classification rules. Regression tests check cumulative SI %, variable frame durations, missing noses, clipped scoring windows, and repeated rendering.
- A real four-second Three Chamber HTTP upload completed in 52.3 seconds with tracking-stage video segments, metadata in the exported configuration, an annotated review, tracking-quality diagnostics, and an Excel download. Visually checked the tracking-stage MP4 overlay.
- Overlay-only classification/rendering sustained approximately 244 frames/second on this development Mac; this is not end-to-end model throughput.
- Automatic diagnostics retain missing-nose gaps and body-in-circle context without imputing nose interaction. Existing reviewed data are insufficient to validate anatomical accuracy; model weights were not changed.

## Version 0.1.3 — chamber role labels

- Restores four cumulative cards and spells out Stranger Interaction % in the fourth card.
- Object/Stranger labels are centered at the top inside their rectified chambers. Chamber roles also appear in result tables and Excel; mixed-side batches retain correct per-mouse roles.
- 194 packaged Python tests, 184 development Python tests (3 skipped), and 9 JavaScript tests passed. Excel regression checks cover uniform and mixed stranger-side batches.
- Existing saved review videos and buffered previews can be regenerated from their scored frame tables with `scripts/refresh_three_chamber_reviews.py`; previous videos are retained in a local backup. This does not rerun inference or modify scores.

## Version 0.1.4 — Windows x64

- Public repository renamed to `wyakah/Mouse-Behavior-Analysis`; prior GitHub links redirect.
- Windows Server 2022 CI: 40 targeted engine tests passed. Synthetic six-second videos exercised both assays before and after installation, including real bundled model inference, live MP4 segments, annotated output, and Excel downloads.
- NSIS per-user installation completed. Native first launch unpacked its verified engine archive, opened the main Behavior Studio window, and exited cleanly on close.
- Bundles an x64 Python 3.12 CPU runtime, application-local Microsoft C++ redistributable DLLs, and checksum-pinned inference weights. Python/DeepLabCut need no separate installation. WebView2 uses Microsoft's bootstrapper when absent.
- Windows workspace links use junctions, locking uses Windows file locking, process checks use the Windows API, and quit terminates the owned worker tree.
- Mac launcher compilation and 184 local Python tests (3 skipped) still pass. The existing Mac v0.1.3 installer remains available; this release adds Windows.
- The executable is unsigned. Tests establish build/install/functionality on the CI runner, not behavioral accuracy, clean-machine certification across all Windows versions, or GPU acceleration.

Build and installed-app evidence: https://github.com/wyakah/Mouse-Behavior-Analysis/actions/runs/34045345552


## Version 0.1.5 — separate chamber and zone interaction

- Eight cumulative cards in a four-column, two-row layout: three chamber times and chamber interaction percentage, followed by left/right zone times, Center Zone N/A, and zone interaction percentage.
- Both percentages divide by the summed observed chamber times. Chamber percentage uses stranger-side occupancy; zone percentage uses accepted nose-in-zone time. Missing nose observations do not suppress chamber occupancy. The legacy percentage field retains its zone-based meaning.
- Development Python: 185 passed, 3 skipped. JavaScript: 10 passed. Mac bundled runtime: all 41 targeted analysis, social-metric, streaming, and desktop tests passed.
- Regression checks cover irregular timestamps, clipped windows, missing noses, repeated preview rendering, live/final parity, Excel column order and mixed stranger-side roles. Saved mouse durations were independently checked against their per-frame classifications.
- Mac application signature and DMG checksum verification passed. The bundle manifest includes and verifies the new shared UI metric helper, renderer, scoring calculations, and workbook exporter.
- Existing local review videos, stream segments, summary CSV/JSON, and batch Excel workbooks were regenerated from saved scored frames. No model weights or nose classifications changed; this is not a new tracking-accuracy benchmark.
- Windows CI passed engine tests, real-model synthetic smoke checks for both assays before and after installation, and native first-launch/clean-close checks. Verified installer publication is recorded in [the v0.1.5 build](https://github.com/wyakah/Mouse-Behavior-Analysis/actions/runs/34046841760).


## Version 0.1.6 — selectable storage and large uploads

- Confirmed the reported upload failures in native-app logs: `ENOSPC` while staging multipart data in the system temporary folder and while saving uploaded videos. Five truncated copies were also blocking video inventory; the new inventory skips unreadable files and preserves them.
- Both assays support 150-recording setups. Shared UI tests cover 150/151 boundaries and eight sequential uploads in bounded binary chunks. No whole-video string conversion is used; each chunk is at most 16 MB. The former 4 GB request cap is removed.
- Storage selection persists between launches. Recording, staging, preparation, tracking, preview, and result paths use the selected workspace. Prior workspaces are preserved, running work blocks location changes, and stale tabs cannot write into a new workspace.
- 195 development Python tests passed (3 skipped); 11 JavaScript tests passed. The bundled Mac runtime passed all 54 targeted engine, storage, upload, and desktop tests. Tests include JSON disk-full errors, selected-drive multipart staging, chunk rollback, cancellation after disconnection, and unreadable legacy uploads.
- Real-model HTTP checks used a separate selected folder: a four-second Three Chamber clip completed in 56.5 seconds and a six-second stereotypy clip in 26.1 seconds. Both produced live segments, annotated output, and Excel in the selected workspace; its original workspace received no videos. Restart restored the selected location and model readiness.
- These selected-folder checks ran on the local filesystem. Model-copy fallback for filesystems without symbolic links is separately covered by a regression test; physical removable-drive hardware was not part of this run.
- Mac app signature and DMG verification passed; packaged program hashes match the final source. Existing scoring models and measurement definitions are unchanged.
- Windows CI passed the targeted tests, pre/post-install real-model smoke checks for both assays, and native first-launch/clean-close checks. [Final Windows build and installer validation](https://github.com/wyakah/Mouse-Behavior-Analysis/actions/runs/34081842283).
