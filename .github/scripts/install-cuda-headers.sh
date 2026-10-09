#!/usr/bin/env bash
# Install what building the root crate's `cuda` feature needs on a Linux
# runner without a CUDA toolkit: the build script of `cuda-bindings`
# (cuda-core's bindings) runs bindgen over CUDA 13's driver `cuda.h` and
# cuRAND's `curand.h`, so it needs those headers and libclang (the crate
# itself loads the driver at run time). The headers come from NVIDIA's
# pinned pip wheels: `cuda.h` and the `cuda_runtime.h` that `curand.h`
# includes from nvidia-cuda-runtime, their `crt/` headers from
# nvidia-cuda-crt, `curand.h` from nvidia-curand. apt's libclang-dev
# provides libclang with clang's resource headers (`stddef.h`). Run as a
# GitHub Actions step after uv is on PATH (it appends CUDA_HOME to
# GITHUB_ENV).
set -euo pipefail

sudo apt-get update
sudo apt-get install -y --no-install-recommends libclang-dev

root="$RUNNER_TEMP/cuda-headers"
uv pip install --quiet --target "$root" \
  nvidia-cuda-runtime==13.4.92 nvidia-cuda-crt==13.4.92 nvidia-curand==10.4.4.72
echo "CUDA_HOME=$root/nvidia/cu13" >> "$GITHUB_ENV"
