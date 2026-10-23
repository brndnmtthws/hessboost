#!/bin/sh
# Install Mesa's lavapipe, a software Vulkan adapter, so the wgpu backend's
# tests (the crate's and python/tests/test_gpu.py) run their GPU paths on a
# Linux runner without a GPU, and set HESSBOOST_REQUIRE_WGPU so they fail
# instead of skipping when the backend is unavailable. Run as a GitHub
# Actions step (it appends to GITHUB_ENV), through .github/actions/lavapipe,
# which caches DEBS per runner image.
#
# Usage: install-lavapipe.sh DEBS
#
# DEBS is a directory of the packages to install. A missing or empty one is
# filled from the Ubuntu archive first. The archive's mirrors at times serve
# a runner at a few kB/s, which stalled jobs for minutes; cached, each runner
# image downloads the packages once.
set -eu

debs="$1"
if ! ls "$debs"/*.deb > /dev/null 2>&1; then
  # Clean first, so the archive cache holds only this install's downloads.
  sudo apt-get clean
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends --download-only mesa-vulkan-drivers libvulkan1
  mkdir -p "$debs"
  # No match when the image already has every package.
  cp /var/cache/apt/archives/*.deb "$debs"/ 2> /dev/null || true
fi
# apt resolved this set on the same runner image, so dpkg needs no package
# lists (or network) to install it.
if ls "$debs"/*.deb > /dev/null 2>&1; then
  sudo dpkg -i "$debs"/*.deb
fi
echo "HESSBOOST_REQUIRE_WGPU=1" >> "$GITHUB_ENV"
