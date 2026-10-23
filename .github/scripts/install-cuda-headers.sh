#!/usr/bin/env bash
# Install what building the root crate's `cuda` feature needs without a CUDA
# toolkit: the CUDA 13 headers (`cuda.h`, `curand.h` and their includes) from
# NVIDIA's pinned pip wheels, which `cuda-bindings`' build script runs bindgen
# over, and libclang, which bindgen loads at run time. Needs uv on PATH.
#
# Without an argument (Ubuntu GitHub Actions runner, whose image's clang
# packages already ship libclang): headers in $RUNNER_TEMP and CUDA_HOME
# appended to GITHUB_ENV. With a directory (maturin-action's manylinux
# container, as root): libclang from dnf, headers in that directory, and
# CUDA_HOME printed for the caller to export:
#
#   CUDA_HOME="$(bash .github/scripts/install-cuda-headers.sh /tmp/cuda-headers)"
#   export CUDA_HOME
set -euo pipefail

if [[ $# -eq 0 ]]; then
  root="$RUNNER_TEMP/cuda-headers"
else
  dnf install -y --quiet clang-devel >&2
  root="$1"
fi
uv pip install --quiet --target "$root" \
  nvidia-cuda-runtime==13.4.92 nvidia-cuda-crt==13.4.92 nvidia-curand==10.4.4.72 >&2
if [[ $# -eq 0 ]]; then
  echo "CUDA_HOME=$root/nvidia/cu13" >> "$GITHUB_ENV"
else
  echo "$root/nvidia/cu13"
fi
