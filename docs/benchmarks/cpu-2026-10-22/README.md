# CPU training and small-batch inference, 2026-10-22

Baseline: `f6c5d8f66fa9d0ce55d6990efa4f1aa14458000e` (v0.4.1).
Candidate: this change's selected-feature histograms and small-batch RowForest
routing. No SIMD or floating-point arithmetic change. Both binaries use the
same benchmark source and dependency resolution, default features, release
profile (`opt-level=3`, thin LTO, one codegen unit), no extra RUSTFLAGS.
Rust 1.99.1 (58cf8f9c3), LLVM 23.1.1, x86_64-unknown-linux-gnu.

Host exposes AMD EPYC 9V74, AVX2/FMA/AVX-512 and nine logical CPUs, Linux
6.18.55. This is a shared virtualized cloud environment, not a dedicated
80-core benchmark host. No affinity or frequency controls. Hardware and
binary hashes are in metadata.json. Do not extrapolate to other CPUs or GPUs.

## Method

The reproducible example and ABBA runner are documented in
[the script guide](../../../scripts/README.md#cpu-baselinecandidate-comparison).
Each workload/thread combination ran baseline, candidate, candidate, baseline
serially. Each process discarded one warmup and recorded five samples: ten
samples per binary. No concurrent compilation or benchmark jobs. Small-batch
prediction samples each average 1,000 API calls; larger batches time one call.
All prediction and serialized JSON model checksums match between binaries,
repeats and thread counts. Checksums validate equivalence, not predictive
quality on real-world data.

Data: deterministic LCG seed `0x2545f4914f6cdd1d`, uniform features; labels
sum the first five features weighted 1..5 plus uniform noise in [0,0.1).
Missing-data case replaces 60% of dense entries by NaN (not CSR).
Hist training, 256 bins, eta=0.1, depth=6, defaults otherwise. Wide cases:
20,000 x 128, 30 rounds, colsample_bytree=0.25 or 0.8. Full-feature control:
50,000 x 20, 50 rounds. Missing:20,000 x30,30 rounds,colsample=0.25.
Small-batch models:10,000 x30 training rows,100 trees; large batch:
100,000 x30 training/prediction rows,100 trees.

Warm training reuses a DMatrix and its cached bins. Cold includes fresh matrix
construction, labels, binning and training. Prediction excludes training and
lazy layout initialization. Fit timers exclude model destruction, subsequent
prediction and checksum validation.

## Initial paired results

Times are median [min, max], in milliseconds for training/large batch and
microseconds for small prediction batches. Speedup = baseline / candidate.

| Case | Threads | Unit | Baseline | Candidate | Speedup |
|---|---:|---|---:|---:|---:|
| train_wide_sampled | 1 | ms | 262.256 [243.732, 296.717] | 106.790 [103.659, 115.724] | 2.46x |
| train_wide_sampled | 4 | ms | 144.948 [120.125, 179.682] | 71.244 [54.839, 74.593] | 2.03x |
| train_wide_sampled_cold | 1 | ms | 347.468 [333.180, 381.961] | 204.651 [198.327, 218.785] | 1.70x |
| train_wide_sampled_cold | 4 | ms | 164.296 [148.515, 183.525] | 101.252 [94.321, 127.469] | 1.62x |
| train_wide_80pct | 1 | ms | 383.401 [350.276, 408.260] | 330.907 [316.307, 349.628] | 1.16x |
| train_wide_80pct | 4 | ms | 168.875 [140.904, 204.313] | 179.013 [135.549, 191.941] | 0.94x |
| train_full | 1 | ms | 164.605 [155.777, 176.892] | 162.190 [159.897, 168.275] | 1.01x |
| train_full | 4 | ms | 125.356 [102.177, 160.775] | 129.070 [93.382, 154.134] | 0.97x |
| train_missing_sampled | 1 | ms | 68.942 [64.223, 74.638] | 67.015 [63.781, 70.482] | 1.03x |
| train_missing_sampled | 4 | ms | 65.771 [57.453, 95.141] | 62.419 [55.838, 71.554] | 1.05x |
| predict_1 | 1 | us | 1.222 [1.201, 1.539] | 0.699 [0.675, 0.885] | 1.75x |
| predict_1 | 4 | us | 1.233 [1.199, 1.463] | 0.696 [0.667, 0.803] | 1.77x |
| predict_8 | 1 | us | 9.995 [9.201, 10.824] | 5.308 [5.021, 5.696] | 1.88x |
| predict_8 | 4 | us | 9.100 [9.017, 9.361] | 5.176 [4.978, 6.308] | 1.76x |
| predict_15 | 1 | us | 18.492 [16.801, 18.923] | 9.919 [9.447, 10.881] | 1.86x |
| predict_15 | 4 | us | 18.585 [17.014, 24.034] | 9.870 [9.489, 10.419] | 1.88x |
| predict_16 | 1 | us | 7.521 [7.491, 7.992] | 7.546 [7.501, 7.852] | 1.00x |
| predict_16 | 4 | us | 7.506 [7.421, 20.991] | 7.496 [7.401, 7.821] | 1.00x |
| predict_batch | 1 | ms | 38.878 [38.166, 43.000] | 37.951 [37.325, 38.684] | 1.02x |
| predict_batch | 4 | ms | 10.485 [9.686, 13.325] | 12.851 [9.853, 20.071] | 0.82x |

## Noisy controls: larger follow-up

The first four-thread large-batch and 80%-feature runs showed overlapping,
noisy slowdowns. Repeated those plus full-feature training with 25 samples
per process (50/binary), same ABBA order. Both complete sets are retained;
the initial results are not discarded. These controls do not establish a
large-batch inference improvement, and the broad-pool training result is
less stable than the 25%-feature case.

| Case (4 threads) | Baseline ms [min,max] | Candidate ms [min,max] | Speedup |
|---|---:|---:|---:|
| predict_batch | 10.781 [9.555, 19.093] | 10.253 [9.416, 15.482] | 1.05x |
| train_wide_80pct | 172.930 [139.067, 239.517] | 143.460 [124.720, 189.011] | 1.21x |
| train_full | 118.965 [99.793, 164.300] | 117.436 [100.599, 156.572] | 1.01x |

Main findings: sampled-wide warm training 2.03–2.46x; cold training
1.62–1.70x; 1/8/15-row prediction batches 1.75–1.88x. Full-feature training,
16-row batches and large-batch inference are controls, not claimed wins.
Sparse histogram construction keeps the old full-build fallback; only sibling
subtraction can avoid unused bins there. Neither GPU speed nor Arm performance
was measured. Single-row APIs already used RowForest and are not claimed to
improve. Column sampling is opt-in; default full-feature training is unchanged.

An eight-state x86 batch-walk experiment was rejected after a single-thread
regression and is not part of this change.

## Validation

- `cargo clippy --all-targets --all-features -- -D warnings`: passed.
- `RAYON_NUM_THREADS=4 cargo test --profile ci --all-features`: 963 passed,
  five intentionally ignored tests (XGBoost/LightGBM fixtures, release-model
  generation and the 1-GiB allocation test). Includes doctests.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`: passed.
- Root, fuzz and CUDA-kernel rustfmt checks: passed.
- Ruff check/format and ty check on the new runner: passed.
- Independent correctness review found no blockers.

CUDA headers were installed for compilation; device-dependent GPU tests
skip when no usable device exists. No real GPU execution, cross-platform
runtime test, Python-extension rebuild or external XGBoost/LightGBM parity
fixture generation was performed locally. Existing CPU determinism and row
prediction tests plus the new bitwise/reference tests passed. CI remains a
separate check of the published revision.
