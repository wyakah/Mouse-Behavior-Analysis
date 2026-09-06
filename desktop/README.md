# Behavior Studio desktop

An offline Apple Silicon Mac application for Three Chamber and Stereotypy. The Tauri window starts a bundled Python engine on a private, dynamically allocated loopback port. No terminal, system Python, Node, or model download is needed on the destination Mac.

## Build

Requires macOS 14+, Apple Silicon, Xcode Command Line Tools, Rust stable, Node 22+, and the existing tested DeepLabCut environment and model files. The Python base must be a relocatable `python-build-standalone` 3.12 distribution; copying a virtual environment alone is not sufficient.

```sh
python desktop/build_payload.py --python-base /path/to/standalone/python
cd desktop
npm ci
./package_macos.sh
```

The signed development app and disk image are generated under `desktop/dist/`. The packaging script applies a local ad-hoc signature, verifies it, and writes a SHA-256 checksum. This does not replace Developer ID signing or notarization. `--refresh` refreshes program files and models in an existing payload. Delete the generated runtime and rebuild to change its Python dependencies. All build outputs are ignored by Git.

## Storage and lifecycle

User data lives in `~/Library/Application Support/com.wyakah.behaviorstudio/workspace/`. Recordings, setup, queues, results, and logs persist across launches and app replacements. Program files are deployed from a hash manifest into that private workspace because the existing analysis workers use workspace-relative paths. Runtime and model links point into the application bundle. Only one process can own a workspace.

A new authenticated loopback session is created each launch. The server binds only to 127.0.0.1, uses an HttpOnly SameSite cookie, and retains the existing origin guard. The web interface has no Tauri filesystem or shell permissions. The shell owns the service process group; quitting terminates its analysis children. Downloads go to the user’s Downloads folder with non-overwriting filenames. Interrupted batches retain completed results and can be rerun from the existing queue. Automatic checkpoint continuation is not implemented.

The first build includes model weights, avoiding an external hosting dependency. Model package downloading and an updater are future release work; do not configure unsigned remote runtime downloads. `manifest.json` records hashes and model versions. The scoring models and thresholds are unchanged by packaging.

## Release status

This is a local development release. Distribution to other Macs requires a Developer ID signature and Apple notarization, plus review of third-party runtime/model redistribution terms. Tauri signing credentials are supplied by the release operator; never commit credentials. The public development installer is on [GitHub Releases](https://github.com/wyakah/three-chamber/releases/tag/v0.1.3). The build script itself does not publish releases or configure automatic updates. Windows and Intel Mac bundles are not yet supported.

Validate each release on a clean Mac, including native video upload, playback, downloads, an end-to-end analysis, process cleanup, and restart persistence. A successful build on the development Mac alone is not clean-machine certification.
