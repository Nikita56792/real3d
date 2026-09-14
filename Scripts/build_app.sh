#!/bin/bash
# Builds dist/Real3D.app (and dist/Real3D.dmg unless NO_DMG=1) with Command Line Tools only.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="${VERSION:-1.0.0}"
BUILD_NUM="${BUILD_NUM:-$(date +%Y%m%d%H%M)}"
BUNDLE_ID="com.real3d.app"
APP=dist/Real3D.app

./Scripts/gen_shader_source.sh
swift build -c release --product Real3D 2>&1 | grep -E "error|warning: unre|Build complete|Compiling" || true
test -x .build/release/Real3D

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/Models"
cp .build/release/Real3D "$APP/Contents/MacOS/Real3D"
cp Resources/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
printf 'APPL????' > "$APP/Contents/PkgInfo"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>Real3D</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleIdentifier</key><string>${BUNDLE_ID}</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>Real3D</string>
  <key>CFBundleDisplayName</key><string>Real3D</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${BUILD_NUM}</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.video</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
  <key>NSScreenCaptureUsageDescription</key><string>Real3D captures a display or window to convert it to stereoscopic 3D.</string>
  <key>LSRequiresNativeExecution</key><true/>
</dict>
</plist>
PLIST

echo "Compiling bundled models…"
swift Scripts/compile_models.swift models/bundled "$APP/Contents/Resources/Models"

# Sign with an Apple Development identity if present (stable TCC identity), else ad-hoc.
IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null | grep -m1 'Apple Development' | sed -E 's/^ *[0-9]+\) ([0-9A-F]+) .*/\1/' || true)"
if [ -n "${IDENTITY}" ]; then
  echo "Signing with Apple Development identity ${IDENTITY}"
  codesign --force --deep --timestamp=none --sign "${IDENTITY}" "$APP"
else
  echo "Signing ad-hoc"
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict "$APP" && echo "Signature OK"
du -sh "$APP"

if [ "${NO_DMG:-0}" != "1" ]; then
  echo "Creating DMG…"
  STAGING=dist/dmg
  rm -rf "$STAGING" dist/Real3D.dmg
  mkdir -p "$STAGING"
  cp -R "$APP" "$STAGING/"
  ln -s /Applications "$STAGING/Applications"
  hdiutil create -volname "Real3D" -srcfolder "$STAGING" -ov -format UDZO -quiet dist/Real3D.dmg
  rm -rf "$STAGING"
  ls -lh dist/Real3D.dmg
fi
echo "Done: $APP"
