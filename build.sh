#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
app="$stage/HF to Ollama.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
swiftc -module-cache-path "$stage/ModuleCache" Sources/main.swift Sources/Models.swift -o "$app/Contents/MacOS/HFToOllama"
cp assets/Info.plist "$app/Contents/"
cp assets/AppIcon.icns backend/*.py "$app/Contents/Resources/"
xattr -cr "$app"
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"
ditto --norsrc "$app" 'dist/HF to Ollama.app'
ditto -c -k --norsrc --keepParent "$app" 'dist/HF to Ollama.zip'
