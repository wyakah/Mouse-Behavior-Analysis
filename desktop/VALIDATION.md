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
