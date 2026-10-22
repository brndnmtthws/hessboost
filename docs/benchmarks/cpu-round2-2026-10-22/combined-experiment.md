# Rejected combined training/inference experiment, 2026-10-22

Baseline: `d7b6a1e053fb2e4e9ac8380f9aa84a052a1ddf62` (main after #177).
Experimental candidate: complete dense-block binning and 32-feature prediction-key tiles for all widths.
This combination is NOT the shipped patch. The training specialization was rejected
after repeated end-to-end cold-training slowdowns; narrow inference retains its
original staging in the final patch. See [the final report](README.md).
No floating-point arithmetic, tree traversal, model format, or public API changes.

## Environment and method

AMD EPYC 9V74 exposed by a shared virtualized cloud workspace, nine logical
CPUs, AVX2/FMA/AVX-512, Linux 6.18.55. Not a dedicated host; no CPU affinity
or frequency controls. Rust 1.99.1 (58cf8f9c3), LLVM 23.1.1,
x86_64-unknown-linux-gnu. Default features, release opt-level=3, thin LTO,
one codegen unit, no extra RUSTFLAGS; same Cargo.lock resolution and benchmark
source for both binaries. Rayon uses exactly 1 or 4 threads. Hardware details,
binary SHA-256s, workload arguments, timestamps and individual samples are retained in
each run directory. No competing builds/tests/benchmarks were launched during
timing. This does not exclude load elsewhere on the shared host.

The serial ABBA runner launches baseline/candidate/candidate/baseline for each
case/thread combination, discards one warmup per process, then records 20
samples per process (40 per binary) in `paired-confirm`. `paired-initial` uses
5 (10 per binary). `regression-check` and `same-binary-control` use 20. These
are repeated timings within two processes per binary, not 40 independent hosts.
Model JSON and prediction-bit checksums match across binaries, repeats and
thread counts. Phase cases compare bin/cut hashes instead; their legacy
`model_checksum` field is the cut hash. No statistical significance claim.

Data are synthetic deterministic LCG seed `0x2545f4914f6cdd1d`: uniform
features; labels weight the first five features by 1..5 plus uniform noise
in [0,0.1). Missing cases replace 60% of dense cells with NaN, not CSR.
Hist training uses 256 bins, eta=0.1, default other parameters. Full workload
shapes, tree counts/depths, column sampling and training rows are in
`scripts/bench_cpu.py` and every raw record. Width-heavy synthetic workloads
do not establish gains on real-world models or other CPUs/GPUs.

## Confirmation measurements

All times below are milliseconds, median [minimum, maximum], 40 samples per
binary. Speedup is baseline median / candidate median, not percent time saved.

| Case | Threads | Baseline ms | Candidate ms | Speedup |
|---|---:|---:|---:|---:|
| bins_dense | 1 | 27.7891 [26.2319, 56.2002] | 25.4436 [23.9398, 29.1378] | 1.092× |
| bins_dense | 4 | 9.2419 [7.2756, 19.0368] | 8.4801 [6.7524, 10.6788] | 1.090× |
| bins_wide | 1 | 24.1113 [22.9011, 43.7438] | 21.8620 [20.8379, 23.8647] | 1.103× |
| bins_wide | 4 | 7.6614 [6.4886, 10.6562] | 6.5275 [5.7543, 8.9705] | 1.174× |
| bins_missing | 1 | 13.2435 [12.2834, 15.6088] | 13.0595 [12.2057, 14.5146] | 1.014× |
| bins_missing | 4 | 4.3746 [3.5443, 5.7846] | 4.3282 [3.6239, 10.4297] | 1.011× |
| train_cold_one_round | 1 | 129.5033 [116.6641, 155.0573] | 121.3183 [115.5974, 166.4502] | 1.067× |
| train_cold_one_round | 4 | 42.7896 [36.4109, 49.1319] | 41.1565 [34.4148, 56.3530] | 1.040× |
| train_cold_full | 1 | 286.9237 [278.8838, 384.8368] | 295.7525 [282.5458, 321.8239] | 0.970× |
| train_cold_full | 4 | 132.8680 [109.7464, 171.1072] | 133.0250 [116.8209, 281.9903] | 0.999× |
| train_cold_missing | 1 | 165.1664 [159.4424, 179.4078] | 173.4439 [165.1284, 255.5938] | 0.952× |
| train_cold_missing | 4 | 96.4495 [77.4864, 155.4605] | 94.5487 [78.7026, 114.8127] | 1.020× |
| train_full | 1 | 165.6775 [157.6499, 194.8532] | 172.0876 [159.2548, 187.6144] | 0.963× |
| train_full | 4 | 114.8946 [100.9788, 159.4565] | 142.7834 [102.9897, 172.8752] | 0.805× |
| train_wide_sampled | 1 | 109.8071 [104.3543, 130.1887] | 111.9727 [104.4215, 117.9527] | 0.981× |
| train_wide_sampled | 4 | 72.0877 [58.3536, 91.2868] | 82.3462 [64.0898, 131.5659] | 0.875× |
| predict_wide_256 | 1 | 3.2986 [2.9341, 6.9702] | 2.4353 [2.2820, 3.1407] | 1.354× |
| predict_wide_256 | 4 | 1.3230 [0.8597, 2.0274] | 0.9338 [0.7130, 1.1415] | 1.417× |
| predict_wide_1024 | 1 | 20.3468 [19.1763, 22.0387] | 11.1141 [9.6716, 21.4932] | 1.831× |
| predict_wide_1024 | 4 | 6.9398 [5.5600, 7.6125] | 4.1203 [3.0727, 5.1657] | 1.684× |
| predict_wide_deep | 1 | 7.5489 [6.7569, 11.3479] | 6.3914 [5.9502, 7.2610] | 1.181× |
| predict_wide_deep | 4 | 2.6008 [2.4620, 2.8400] | 2.2885 [1.6991, 3.1743] | 1.136× |
| predict_batch | 1 | 39.4719 [37.4536, 58.1800] | 38.5606 [37.0638, 56.4451] | 1.024× |
| predict_batch | 4 | 10.2104 [9.5941, 16.7395] | 10.0751 [9.4348, 13.6655] | 1.013× |
| predict_16 | 1 | 0.0079 [0.0074, 0.0262] | 0.0076 [0.0074, 0.0094] | 1.041× |
| predict_16 | 4 | 0.0080 [0.0075, 0.0104] | 0.0077 [0.0074, 0.0091] | 1.039× |
| predict_64 | 1 | 0.0257 [0.0245, 0.0647] | 0.0247 [0.0244, 0.0293] | 1.038× |
| predict_64 | 4 | 0.0249 [0.0244, 0.0284] | 0.0247 [0.0244, 0.0274] | 1.010× |
| predict_256 | 1 | 0.1002 [0.0954, 0.1080] | 0.1025 [0.0948, 0.1273] | 0.978× |
| predict_256 | 4 | 0.1002 [0.0967, 0.1235] | 0.0976 [0.0955, 0.1247] | 1.027× |

## Interpretation

- Wide/shallow 10,000×1,024 inference (10 depth-4 trees) is the clearest result:
  20.347→11.114 ms at one thread (1.831×, 45.4% less time), and
  6.940→4.120 ms at four (1.684×, 40.6% less time). The earlier independent
  sweep measured 1.915×/1.700×, so this is not a single favorable timing.
- Complete dense bin assignment improves 1.09–1.17× in the confirmation sweep
  (8.2–14.8% less time). It removes per-row cursor allocation/increments and
  per-cell missing checks only when existing row offsets prove a block complete.
  Missing/CSR behavior is unchanged. This is a training preparation phase,
  not a claim that overall training improves by that amount.
- Overall cold and warmed training do not show a reliable speedup. Some confirmation
  sweep controls looked slower, notably warmed full training at four threads
  (0.805×), but a separately repeated check returned 0.995×; one-thread cold
  missing training moved from 0.952× to 0.992×. The same-binary A/A control
  itself ranged from 0.967× to 1.042× across these cases. Do not attribute
  small unchanged-path shifts to this patch or hide the unfavorable samples.
  The sampled warm-training four-thread case later returned 1.018× in
  `combined-final`, but this does not resolve the cold-training concern.
- At 256 features or with deeper/more trees, benefits vary much more with
  traversal cost and host noise. Narrow and small-batch controls are broadly
  unchanged; no gain is claimed there. Prediction allocation and key staging
  are inside the timer; model training and matrix construction are outside.
- Cold training includes fresh matrix construction, cuts and bins. Warm training
  reuses cached bins. `bins` includes cloning precomputed cuts but not constructing
  them. The bins/cuts result is destroyed outside its timer.

## Investigation and rejected alternatives

Initial one-thread phase medians on 100,000×30 were about 77 ms for cuts and
27 ms for bin assignment (`baseline-phases.jsonl`). This localized preparation
cost before changing it. Manual isolated `fill_lanes` profiling on 256 rows × 1,024
features measured about 436→240 µs; the retained end-to-end measurements
above are the primary evidence. The tile keeps a 4 KiB destination region hot
across sixteen rows and replaces heap scratch with 256 bytes of stack scratch.

Two alternatives were explicitly rejected:

1. Prioritizing a higher RowForest cutoff. The existing batch path already
   beat repeated public single-row prediction at every tested size 16–256.
   `baseline-crossover.jsonl` and `examples/bench_row_crossover.rs` retain the
   test. Both layouts were warmed; row calls reused output storage. Repeated
   public row calls include validation/call overhead; this is evidence against
   prioritizing the idea, not proof that a modified internal cutoff cannot win.
2. Four 8-bit radix passes instead of the existing three 11-bit passes. Although
   bucket storage shrank, one-thread cuts slowed 77.005→88.101 ms (14.4% more
   time), and one-round cold training slowed too. `radix-only` compares the
   retained candidate with and without this experiment. The rejected change
   is preserved only as `radix8-rejected.patch`, not applied to production code.

## Reproduce

Use the same `examples/bench_cpu.rs` and `scripts/bench_cpu.py` from this PR
on baseline and candidate. Build in separate target directories, or touch
`src/lib.rs` to force rebuilding when sharing a target directory across
worktrees, and copy each executable before building the next revision.

```sh
cargo build --release --example bench_cpu
cp target/release/examples/bench_cpu /tmp/candidate-bench_cpu
# Repeat on d7b6a1e with this PR’s benchmark example to obtain baseline.
python scripts/bench_cpu.py --baseline /tmp/baseline-bench_cpu \
  --candidate /tmp/candidate-bench_cpu --output /tmp/confirmation \
  --threads 1 4 --repeats 20 --cases \
  bins_dense bins_wide bins_missing train_cold_one_round train_cold_full \
  train_cold_missing train_full train_wide_sampled predict_wide_256 \
  predict_wide_1024 predict_wide_deep predict_batch predict_16 predict_64 predict_256
RAYON_NUM_THREADS=1 cargo run --release --example bench_row_crossover -- 30 100 6 5 100
cargo test --release --lib tree::compact::tests::profile_fill_lanes -- --ignored --exact --nocapture
```

Do not run these commands concurrently with builds, tests or other benchmarks.
See the [script guide](../../../scripts/README.md#cpu-baselinecandidate-comparison)
for operation semantics. Existing scalar-key and generic-binning oracles test
exact output across tile/block boundaries, special floating-point bit patterns,
row views, missing values, categorical features, bin widths and parallel dispatch.

## Local validation

- `cargo test --profile ci --all-features -- --test-threads=4`:
  965 passed including 45 doctests; six ignored tests remain opt-in.
- `cargo clippy --all-targets --all-features -- -D warnings` passed.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` passed.
- Rust formatting checks passed for root, fuzz and cuda-kernels manifests.
- Ruff 0.16.9 check/format passed for the benchmark runner; `git diff --check` passed.
- Independent read-only review found no correctness blockers and audited stored
  summaries against raw samples. GPU hardware was not benchmarked; tests that
  require unavailable devices may return early. No real-data quality claim.

## Final combined-candidate decision

After lint-only cleanup, `combined-final` remeasured the rebuilt combination.
One-thread cold 20-round dense training was 297.522→309.558 ms (0.961×),
following 0.970× in the earlier confirmation sweep. Both candidate process
medians exceeded both baseline process medians. Exact-case A/A controls
(`same-binary-final-controls`) quantify host noise but cannot exonerate this
repeat direction. Faster bin assignment alone was not sufficient evidence to
ship it. The dense-binning specialization was removed completely.

The final inference change also leaves ≤64-feature input on the original
staging path. The combined patch is preserved as `combined-rejected.patch`;
apply it only to a fresh baseline checkout with the common benchmark harness
to reproduce this historical experiment. Apply `radix8-rejected.patch` on top
of that combination to reproduce the separate radix experiment.

The last combined binary is SHA-256
`e8d9526e86547a512b18b573c33bd0e15f6dcb203eb1374d5acf1ba1eecff85e`.
A preliminary final-verification run overlapped a formatting check and was
discarded; `combined-final` is the clean, serial replacement, not that run.

Archived patches use zero context; apply with `git apply --unidiff-zero`
only on the stated baseline (and the radix patch after the combined patch).
