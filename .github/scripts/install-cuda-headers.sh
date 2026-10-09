#!/usr/bin/env bash
# Install what building the root crate's `cuda` feature needs without a
# CUDA toolkit: the build script of `cuda-bindings` (cuda-core's bindings)
# runs bindgen over CUDA 13's driver `cuda.h` and cuRAND's `curand.h`, so it
# needs those headers and libclang (the crate itself loads the driver at run
# time). The headers come from NVIDIA's pinned pip wheels: `cuda.h` and the
# `cuda_runtime.h` that `curand.h` includes from nvidia-cuda-runtime, their
# `crt/` headers from nvidia-cuda-crt, `curand.h` from nvidia-curand.
# libclang comes with clang's resource headers (`stddef.h`). Needs uv on
# PATH.
#
# Without an argument, on an Ubuntu GitHub Actions runner: libclang from
# apt's libclang-dev, the headers in $RUNNER_TEMP, and CUDA_HOME appended to
# GITHUB_ENV. With a directory, in maturin-action's manylinux container
# (AlmaLinux, as root; `before-script-linux`): libclang from dnf's
# clang-devel, the headers in that directory, and CUDA_HOME printed for the
# caller to export:
#
#   CUDA_HOME="$(bash .github/scripts/install-cuda-headers.sh /tmp/cuda-headers)"
#   export CUDA_HOME
set -euo pipefail

if [[ $# -eq 0 ]]; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends libclang-dev
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
