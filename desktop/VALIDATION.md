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
