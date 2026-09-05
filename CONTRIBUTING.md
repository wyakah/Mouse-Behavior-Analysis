# Contributing

Use Python 3.12 and the dependencies in `requirements.txt`. Run `.venv/bin/python -m pytest -q` before submitting changes. When editing the browser code, also run `node --check` on the changed JavaScript files and inspect the affected flow in the local app.

Preserve these invariants:

- Keep source videos and frame timestamps intact.
- Keep missing/low-confidence observations explicit; do not silently interpolate.
- Distinguish detection coverage from validated correctness.
- Keep cup circles equal in size and review every recording after shared-setting changes.
- Retain provenance, reviewed annotations, and per-recording errors.
- Use synthetic test fixtures; do not commit recordings, labels, weights, credentials, or generated results.

Describe the observable change and relevant validation in pull requests. Scientific accuracy improvements need evaluation against reviewed, held-out data in addition to software tests.

This is a private research project. No open-source license has been selected. Third-party dependencies and pretrained weights remain subject to their respective terms.
