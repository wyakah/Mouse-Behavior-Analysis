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
