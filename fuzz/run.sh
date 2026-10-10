#!/usr/bin/env bash
# Runs every fuzz target for a fixed wall time: a smoke test, not a fuzzing
# campaign. Seeds come from the models each release saved
# (tests/data/saved/*/) and the XGBoost and LightGBM saves the importer tests
# use, so new formats and releases are picked up without checked-in copies,
# plus the inputs in fixed-seeds/ that reach paths the fuzzer is slow to find
# (e.g. a gblinear case that trains, LightGBM zero-as-missing and `inf`
# splits).
#
# Usage (from fuzz/, where mise provides nightly Rust and cargo-fuzz):
#   ./run.sh [seconds-per-target] [target...]
set -euo pipefail

cd "$(dirname "$0")"
seconds="${1:-30}"
shift || true
data=../tests/data

seed() {
  mkdir -p "seeds/$1"
}

rm -rf seeds
mkdir seeds
cp -R fixed-seeds/. seeds/
seed native-model
seed json-model
seed compact-model
seed xgboost-json-model
seed xgboost-ubjson-model
seed lightgbm-model
seed loaders
seed diffusion-model
seed forest-model
for version in "$data"/saved/*/; do
  v="$(basename "$version")"
  for file in "$version"*.bin; do
    # Leading 1: the target seals the section table (the container minus
    # its 5-byte header and 8-byte checksum) itself; see native-model.rs.
    { printf '\x01'; zstd -dcq "$file" | tail -c +6 | head -c -8; } \
      > "seeds/native-model/$v-$(basename "$file" .bin)"
    # Leading 0: the raw, compressed file.
    { printf '\x00'; cat "$file"; } > "seeds/native-model/$v-$(basename "$file" .bin)-raw"
  done
  # The GBDT JSON saves. The forest and diffusion JSON mirrors are stored
  # zstd-compressed (`.json.zst`) and are not documents this target parses;
  # the forest and diffusion targets take their JSON seeds from fixed-seeds.
  for file in "$version"*.json; do
    cp "$file" "seeds/json-model/$v-$(basename "$file")"
  done
  for file in "$version"*.hbtd; do
    cp "$file" "seeds/compact-model/$v-$(basename "$file")"
  done
done
cp "$data"/xgboost-*.json seeds/xgboost-json-model/
cp "$data"/xgboost-*.ubj seeds/xgboost-ubjson-model/
cp "$data"/lightgbm-*.txt seeds/lightgbm-model/
# The first byte selects the CSV options (see loaders.rs).
printf '\x00# comment\n1 0:1.5 3:-2\n0 1:0.25\n\n2 2:1e3 0:7\n' > seeds/loaders/libsvm
printf '\x09y,a,b\n1,2.5,\n0,,-3\n1,4,5\n' > seeds/loaders/csv-header-label0
printf '\x23a;b;c\n1;NA;3\n4;5;NA\n' > seeds/loaders/csv-semicolon-na

# cargo-fuzz defaults to the target it was built for; the prebuilt Linux
# binary is a musl build, whose static libc the sanitizers cannot use.
host="$(rustc -vV | sed -n 's/^host: //p')"
# Build first, with the flags `cargo fuzz run` uses, so the runs below only
# check freshness: one build of every target compiles their binaries in
# parallel.
if [ $# -eq 0 ]; then
  mapfile -t targets < <(cargo fuzz list)
  cargo fuzz build --target "$host"
else
  targets=("$@")
  for target in "${targets[@]}"; do
    cargo fuzz build --target "$host" "$target"
  done
fi

# One libFuzzer process per target, up to one per CPU (FUZZ_JOBS overrides;
# macOS has no `nproc`) at a time. Each writes logs/<target>.log, printed
# when it ends, and its exit status to logs/<target>.status.
jobs="${FUZZ_JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}"
rm -rf logs
mkdir logs
fuzz() {
  local target="$1" status=0
  local corpora=("corpus/$target")
  mkdir -p "${corpora[0]}"
  if [ -d "seeds/$target" ]; then
    corpora+=("seeds/$target")
  fi
  # `-max_total_time` does not interrupt a stuck input, so `-timeout` bounds
  # each input too; libFuzzer's default 2 GB RSS limit catches runaway
  # allocations.
  cargo fuzz run --target "$host" "$target" "${corpora[@]}" \
    -- -max_total_time="$seconds" -timeout=10 > "logs/$target.log" 2>&1 || status=$?
  echo "$status" > "logs/$target.status"
  printf '== %s (exit %s)\n%s\n' "$target" "$status" "$(cat "logs/$target.log")"
}
running=0
for target in "${targets[@]}"; do
  if [ "$running" -ge "$jobs" ]; then
    wait -n
    running=$((running - 1))
  fi
  fuzz "$target" &
  running=$((running + 1))
done
wait

failed=()
for target in "${targets[@]}"; do
  if [ "$(cat "logs/$target.status")" != 0 ]; then
    failed+=("$target")
  fi
done
if [ ${#failed[@]} -ne 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
