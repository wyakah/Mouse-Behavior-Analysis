#!/bin/bash
# Run after build_payload.py. Produces a local, ad-hoc-signed developer installer.
set -euo pipefail
cd "$(dirname "$0")"
npm run build -- --no-bundle
./node_modules/.bin/tauri bundle --bundles app
mkdir -p dist
staging="$(mktemp -d "${TMPDIR:-/tmp}/behavior-studio-dmg.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
ditto 'src-tauri/target/release/bundle/macos/Behavior Studio.app' "$staging/Behavior Studio.app"
codesign --force --deep --sign "${BEHAVIOR_SIGNING_IDENTITY:--}" "$staging/Behavior Studio.app"
codesign --verify --deep --strict "$staging/Behavior Studio.app"
ln -s /Applications "$staging/Applications"
hdiutil create -ov -format UDZO -volname 'Behavior Studio' -srcfolder "$staging" 'dist/Behavior-Studio-0.1.2-apple-silicon.dmg'
# Some macOS versions leave the newly created image attached after compression.
python3 - <<'PYIMAGE'
import pathlib, plistlib, subprocess
image=pathlib.Path('dist/Behavior-Studio-0.1.2-apple-silicon.dmg').resolve()
info=plistlib.loads(subprocess.check_output(['hdiutil','info','-plist']))
for record in info.get('images',[]):
    if pathlib.Path(record.get('image-path','')).resolve()==image:
        devices=[e['dev-entry'] for e in record.get('system-entities',[]) if e.get('dev-entry')]
        if devices:subprocess.run(['hdiutil','detach',devices[0]],check=True)
PYIMAGE
hdiutil verify 'dist/Behavior-Studio-0.1.2-apple-silicon.dmg'
(cd dist && shasum -a 256 'Behavior-Studio-0.1.2-apple-silicon.dmg' > 'Behavior-Studio-0.1.2-apple-silicon.dmg.sha256')
# Keep the signed app available for direct local testing as well.
ditto "$staging/Behavior Studio.app" 'dist/Behavior Studio.app'
