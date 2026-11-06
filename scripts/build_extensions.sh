#!/usr/bin/env bash
# Usage: scripts/build_extensions.sh [python]
set -e
#export NVCC_APPEND_FLAGS="-std=c++20"
PY=${1:-python}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
$PY -m pip install "$HERE/submodules/i3dgs/submodules/simple-knn" --no-build-isolation --no-deps
$PY -m pip install "$HERE/submodules/i3dgs/submodules/diff-gaussian-rasterization" --no-build-isolation --no-deps
$PY -m pip install "$HERE/submodules/faster-beta-splatting" --no-build-isolation --no-deps
$PY -m pip install "git+https://github.com/rahul-goel/fused-ssim" --no-build-isolation --no-deps
$PY -m pip install "git+https://github.com/nv-tlabs/ppisp.git" --no-build-isolation --no-deps
