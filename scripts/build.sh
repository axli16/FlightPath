#!/usr/bin/env bash
# =============================================================================
# build.sh — FlightPath Linux build script
# Usage: bash scripts/build.sh [--debug]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
BUILD_TYPE="Release"

# Allow --debug flag
for arg in "$@"; do
  if [[ "$arg" == "--debug" ]]; then
    BUILD_TYPE="Debug"
  fi
done

echo "========================================"
echo "  FlightPath Build Script (Linux)"
echo "  Build type : $BUILD_TYPE"
echo "  Project    : $PROJECT_ROOT"
echo "  Build dir  : $BUILD_DIR"
echo "========================================"

# Create and enter build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo ""
echo "[1/2] Configuring with CMake..."
cmake "$PROJECT_ROOT" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# Build using all available CPU cores
echo ""
echo "[2/2] Building..."
cmake --build . --config "$BUILD_TYPE" -- -j"$(nproc)"

BINARY="$BUILD_DIR/bin/FlightPath"
echo ""
if [[ -f "$BINARY" ]]; then
  echo "✓ Build succeeded!"
  echo "  Binary: $BINARY"
  echo ""
  echo "Run headless (SSH / no display):"
  echo "  $BINARY data/dashcam.mp4 --cuda --no-display --output result.mp4"
  echo ""
  echo "Run with X11 forwarding (ssh -X):"
  echo "  $BINARY data/dashcam.mp4 --cuda"
else
  echo "✗ Build failed — binary not found at expected path."
  exit 1
fi
