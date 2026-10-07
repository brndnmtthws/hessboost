#!/usr/bin/env bash
# Compiles the kernels to PTX with cuda-oxide and installs the results as
# ../src/backend/cuda/training.ptx and prediction.ptx, which hessboost
# embeds. Those files are committed; CI runs this script on x86_64 Linux
# and fails if it changes them (the output is byte-for-byte reproducible per
# host architecture).
#
# Usage (from cuda-kernels/, where rust-toolchain.toml selects cuda-oxide's
# nightly; mise.toml provides it and cargo-oxide for mise users):
#   ./build.sh
set -euo pipefail

cd "$(dirname "$0")"
out="$PWD/target/ptx"
ptx="$out/hessboost_cuda_kernels.ptx"

# Build the kernels of cargo feature $1 into ../src/backend/cuda/$2.
module() {
  rm -f "$ptx"
  # sm_75 (Turing) is the oldest architecture the backend runs on; the
  # driver JIT-compiles the PTX for the device's own. `--no-fmad`: no
  # floating-point contraction, so every operation is the CPU's single IEEE
  # operation.
  CUDA_OXIDE_PTX_DIR="$out" cargo oxide build --arch sm_75 --no-fmad --features "$1"
  # A libdevice call (`__nv_*`, e.g. from `f32::mul_add`) makes cuda-oxide
  # emit NVVM IR for a CUDA toolkit to finish instead of PTX.
  if [ ! -f "$ptx" ]; then
    echo "error: cuda-oxide wrote no PTX for '$1'; does a kernel call libdevice?" >&2
    exit 1
  fi
  # Bit-for-bit CPU parity rests on every floating-point instruction being
  # the IEEE one: refuse approximate or flush-to-zero instructions, and adds,
  # subtractions and multiplications without an explicit rounding mode,
  # which ptxas may fuse into FMAs.
  if grep -nE '\.(approx|ftz)\b|\b(add|sub|mul|mad)\.f(32|64)\b|\bdiv\.full\b|__nv_' "$ptx"; then
    echo "error: the '$1' PTX has a non-IEEE floating-point instruction" >&2
    exit 1
  fi
  cp "$ptx" "../src/backend/cuda/$2"
}

module train training.ptx
module predict prediction.ptx
# rustc's crate hashes, which name the PTX's shared-memory symbols, depend on
# the host architecture.
if [ "$(uname -m)" != x86_64 ]; then
  echo "note: built on $(uname -m): commit an x86_64 Linux build instead" \
    "(CI uploads one when its check fails)" >&2
fi
