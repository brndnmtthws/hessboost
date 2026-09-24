#!/usr/bin/env bash
# Runs every fuzz target for a fixed wall time: a smoke test, not a fuzzing
# campaign. Seeds come from the models each release saved
# (tests/data/saved/*/) and the XGBoost saves the importer tests use, so new
# formats and releases are picked up without checked-in copies.
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
seed native-model
seed json-model
seed compact-model
seed xgboost-json-model
seed xgboost-ubjson-model
seed loaders
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
  for file in "$version"*.json; do
    cp "$file" "seeds/json-model/$v-$(basename "$file")"
  done
  for file in "$version"*.hbtd; do
    cp "$file" "seeds/compact-model/$v-$(basename "$file")"
  done
done
cp "$data"/xgboost-*.json seeds/xgboost-json-model/
cp "$data"/xgboost-*.ubj seeds/xgboost-ubjson-model/
# The first byte selects the CSV options (see loaders.rs).
printf '\x00# comment\n1 0:1.5 3:-2\n0 1:0.25\n\n2 2:1e3 0:7\n' > seeds/loaders/libsvm
printf '\x09y,a,b\n1,2.5,\n0,,-3\n1,4,5\n' > seeds/loaders/csv-header-label0
printf '\x23a;b;c\n1;NA;3\n4;5;NA\n' > seeds/loaders/csv-semicolon-na

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  mapfile -t targets < <(cargo fuzz list)
fi
for target in "${targets[@]}"; do
  corpora=("corpus/$target")
  mkdir -p "${corpora[0]}"
  if [ -d "seeds/$target" ]; then
    corpora+=("seeds/$target")
  fi
  # `-max_total_time` does not interrupt a stuck input, so `-timeout` bounds
  # each input too; libFuzzer's default 2 GB RSS limit catches runaway
  # allocations.
  cargo fuzz run "$target" "${corpora[@]}" -- -max_total_time="$seconds" -timeout=10
done
