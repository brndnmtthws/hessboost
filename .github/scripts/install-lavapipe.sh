#!/bin/sh
# Install Mesa's lavapipe, a software Vulkan adapter, so the wgpu backend's
# tests (the crate's and python/tests/test_gpu.py) run their GPU paths on a
# Linux runner without a GPU, and set HESSBOOST_REQUIRE_WGPU so they fail
# instead of skipping when the backend is unavailable. Run as a GitHub
# Actions step (it appends to GITHUB_ENV).
set -eu

sudo apt-get update
sudo apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1
echo "HESSBOOST_REQUIRE_WGPU=1" >> "$GITHUB_ENV"
