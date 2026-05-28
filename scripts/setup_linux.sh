#!/usr/bin/env bash
# =============================================================================
# setup_linux.sh — FlightPath dependency checker & installer (Linux)
# Supports Ubuntu/Debian and RHEL/CentOS/Fedora family distros.
# Run once on the Linux server before building.
# Usage: bash scripts/setup_linux.sh
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
NC='\033[0m' # No Color

ok()   { echo -e "${GRN}[✓]${NC} $*"; }
warn() { echo -e "${YLW}[!]${NC} $*"; }
fail() { echo -e "${RED}[✗]${NC} $*"; }

echo "========================================"
echo "  FlightPath — Linux Setup Checker"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Detect package manager
# ---------------------------------------------------------------------------
if command -v apt-get &>/dev/null; then
  PKG_MANAGER="apt"
elif command -v dnf &>/dev/null; then
  PKG_MANAGER="dnf"
elif command -v yum &>/dev/null; then
  PKG_MANAGER="yum"
else
  warn "Could not detect package manager. Install dependencies manually."
  PKG_MANAGER="unknown"
fi
ok "Package manager: ${PKG_MANAGER}"

install_pkg() {
  local pkg="$1"
  echo "  → Installing $pkg..."
  case "$PKG_MANAGER" in
    apt) sudo apt-get install -y "$pkg" ;;
    dnf) sudo dnf install -y "$pkg" ;;
    yum) sudo yum install -y "$pkg" ;;
    *)   fail "Cannot auto-install $pkg. Please install it manually." ;;
  esac
}

# ---------------------------------------------------------------------------
# 2. Check CMake (>= 3.15)
# ---------------------------------------------------------------------------
echo ""
echo "--- Checking CMake ---"
if command -v cmake &>/dev/null; then
  CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
  ok "cmake $CMAKE_VER found"
else
  warn "cmake not found. Installing..."
  case "$PKG_MANAGER" in
    apt) install_pkg "cmake" ;;
    dnf|yum) install_pkg "cmake" ;;
    *) fail "Install cmake >= 3.15 manually from https://cmake.org" ;;
  esac
fi

# ---------------------------------------------------------------------------
# 3. Check C++ compiler
# ---------------------------------------------------------------------------
echo ""
echo "--- Checking C++ Compiler ---"
if command -v g++ &>/dev/null; then
  GXX_VER=$(g++ --version | head -1)
  ok "g++ found: $GXX_VER"
else
  warn "g++ not found. Installing..."
  case "$PKG_MANAGER" in
    apt) install_pkg "build-essential" ;;
    dnf|yum) install_pkg "gcc-c++" ;;
    *) fail "Install g++ (GCC C++) manually." ;;
  esac
fi

# ---------------------------------------------------------------------------
# 4. Check OpenCV
# ---------------------------------------------------------------------------
echo ""
echo "--- Checking OpenCV ---"
if pkg-config --exists opencv4 2>/dev/null; then
  OCV_VER=$(pkg-config --modversion opencv4)
  ok "OpenCV $OCV_VER found via pkg-config"
elif pkg-config --exists opencv 2>/dev/null; then
  OCV_VER=$(pkg-config --modversion opencv)
  ok "OpenCV $OCV_VER found via pkg-config"
elif command -v opencv_version &>/dev/null; then
  ok "OpenCV $(opencv_version) found"
else
  warn "OpenCV not found."
  echo ""
  echo "  Option A — Install pre-built OpenCV (no CUDA support):"
  case "$PKG_MANAGER" in
    apt) echo "    sudo apt-get install -y libopencv-dev" ;;
    dnf) echo "    sudo dnf install -y opencv-devel" ;;
    yum) echo "    sudo yum install -y opencv-devel" ;;
  esac
  echo ""
  echo "  Option B — Build OpenCV from source WITH CUDA support (recommended):"
  echo "    See: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html"
  echo "    Key CMake flags: -DWITH_CUDA=ON -DCUDA_ARCH_BIN=<your GPU arch>"
  echo "    Example for RTX 30xx: -DCUDA_ARCH_BIN=8.6"
  echo ""
  echo "  Run this script again after installing OpenCV."
  exit 1
fi

# ---------------------------------------------------------------------------
# 5. Check CUDA
# ---------------------------------------------------------------------------
echo ""
echo "--- Checking CUDA ---"
if command -v nvcc &>/dev/null; then
  CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
  ok "CUDA $CUDA_VER found (nvcc)"
elif [[ -f /usr/local/cuda/version.txt ]]; then
  ok "CUDA found at /usr/local/cuda"
elif [[ -f /usr/local/cuda/version.json ]]; then
  ok "CUDA found at /usr/local/cuda"
else
  warn "CUDA not found in PATH. If your OpenCV was built with CUDA support,"
  warn "make sure /usr/local/cuda/bin is in your PATH:"
  echo "    export PATH=\$PATH:/usr/local/cuda/bin"
  echo "    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
fi

# Check nvidia-smi
if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  ok "GPU detected: $GPU_INFO"
else
  warn "nvidia-smi not found — cannot detect GPU. Install NVIDIA drivers if using --cuda."
fi

# ---------------------------------------------------------------------------
# 6. Check model files
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"

echo ""
echo "--- Checking Model Files ---"
mkdir -p "$MODELS_DIR"

check_model() {
  local file="$1"
  local url="$2"
  local min_bytes="$3"
  local path="$MODELS_DIR/$file"
  if [[ -f "$path" ]] && [[ $(stat -c%s "$path") -ge $min_bytes ]]; then
    ok "$file present ($(du -sh "$path" | cut -f1))"
  else
    warn "$file missing or incomplete. Downloading..."
    wget -q --show-progress -O "$path" "$url" || \
      { fail "Download failed. Get $file manually from the README."; }
    ok "$file downloaded."
  fi
}

check_model "yolov4.weights" \
  "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights" \
  245000000

check_model "yolov4.cfg" \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg" \
  10000

check_model "coco.names" \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names" \
  500

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Setup check complete!"
echo "  Next step: bash scripts/build.sh"
echo "========================================"
