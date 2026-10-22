# Wide-batch inference optimization, 2026-10-22

Baseline: `d7b6a1e053fb2e4e9ac8380f9aa84a052a1ddf62` (main after #177).
Final candidate: 32-feature tiles when staging prediction keys for inputs
wider than 64 features. Narrower inputs retain the original staging routine.
No floating-point arithmetic, tree traversal, model format, public API, or
training implementation changes. The 64-feature boundary is conservative,
not a claim to have found the optimal cutoff for every CPU.

The tile keeps a 4 KiB destination region hot across sixteen rows and uses
256 bytes of stack scratch instead of a feature-wide heap allocation. Scalar
oracle tests cover feature/tile boundaries, incomplete row groups, repeated
buffer use, signed zeros, infinities, NaN payloads and arbitrary float bits.

## Environment and method

Shared virtualized cloud workspace exposing AMD EPYC 9V74, nine logical CPUs,
AVX2/FMA/AVX-512 and Linux 6.18.55. This is not a dedicated benchmark host.
No affinity or frequency controls; no competing builds/tests/benchmarks were
launched during the retained measurements. Load elsewhere on the host is
not controlled. Rust 1.99.1 (58cf8f9c3), LLVM 23.1.1,
x86_64-unknown-linux-gnu; default features, opt-level=3, thin LTO, one codegen
unit, no extra RUSTFLAGS. Both binaries use the same benchmark source and
dependency resolution. Rayon uses exactly one or four threads.

For each workload/thread pair the serial runner launches baseline, candidate,
candidate, baseline. Each process discards one warmup and records 20 samples,
giving 40 per binary. Samples within each of two processes are correlated;
they are not 40 independent machines. Raw samples, workload arguments, hashes,
host information and timestamps are in `inference-only-results`. All model
JSON and prediction-bit checksums match across binaries, samples and threads.

Data: deterministic LCG seed `0x2545f4914f6cdd1d`, uniform features; labels
weight the first five features by 1..5 plus uniform noise in [0,0.1). Training
uses hist, 256 bins, eta=0.1, defaults otherwise. Full shapes/tree counts/depths
are recorded in the raw JSON and `scripts/bench_cpu.py`. Models are synthetic;
this is not a predictive-quality study or a general hardware speedup claim.

Prediction timings include API allocation and key staging, but exclude model
training and matrix construction. Batches below 16 rows average 1,000 calls
per sample; 16–1,023 rows average 100; larger batches time one call. Cold
training includes a fresh matrix, cuts and bins, not a cold CPU cache. Warm
training reuses cached bins. Checksums/model serialization/destruction are
outside fit timing.

## Final retained patch: all measurements

Times are milliseconds, median [minimum, maximum], 40 samples per binary.
Speedup is baseline median / candidate median.

| Case | Threads | Baseline ms | Candidate ms | Speedup |
|---|---:|---:|---:|---:|
| train_cold_full | 1 | 319.5520 [282.9771, 370.9461] | 321.0378 [281.3218, 359.6761] | 0.995× |
| train_cold_full | 4 | 147.5177 [121.0231, 193.3218] | 144.0246 [108.8890, 171.8135] | 1.024× |
| train_full | 1 | 166.9332 [160.1758, 194.1812] | 170.3838 [159.6145, 216.7278] | 0.980× |
| train_full | 4 | 138.1684 [102.5584, 195.1562] | 122.6788 [96.9716, 157.0449] | 1.126× |
| train_wide_sampled | 1 | 110.2044 [103.7043, 137.1700] | 110.9600 [101.9870, 138.4530] | 0.993× |
| train_wide_sampled | 4 | 73.9096 [55.3507, 116.3658] | 86.0088 [53.3288, 142.1222] | 0.859× |
| predict_wide_256 | 1 | 3.0645 [2.9381, 9.1823] | 2.4187 [2.2826, 2.9203] | 1.267× |
| predict_wide_256 | 4 | 1.2716 [0.8843, 2.3255] | 1.3932 [0.9600, 2.1950] | 0.913× |
| predict_wide_1024 | 1 | 20.5115 [19.4181, 22.0287] | 10.0479 [9.6556, 10.9474] | 2.041× |
| predict_wide_1024 | 4 | 5.5629 [5.1773, 5.9533] | 3.4745 [2.6385, 3.9603] | 1.601× |
| predict_wide_deep | 1 | 7.3505 [6.6535, 9.2779] | 6.1353 [5.9257, 6.7371] | 1.198× |
| predict_wide_deep | 4 | 3.7946 [2.6105, 4.9661] | 3.2176 [1.7092, 11.6121] | 1.179× |
| predict_batch | 1 | 39.0002 [37.0752, 45.4031] | 37.9261 [37.0446, 40.4757] | 1.028× |
| predict_batch | 4 | 10.2949 [9.6118, 15.8888] | 9.8294 [9.3361, 12.9833] | 1.047× |
| predict_16 | 1 | 0.0076 [0.0074, 0.0098] | 0.0076 [0.0074, 0.0092] | 1.003× |
| predict_16 | 4 | 0.0076 [0.0074, 0.0114] | 0.0076 [0.0074, 0.0090] | 1.000× |
| predict_256 | 1 | 0.0990 [0.0953, 0.1311] | 0.0986 [0.0958, 0.1057] | 1.005× |
| predict_256 | 4 | 0.1004 [0.0961, 0.1541] | 0.0984 [0.0949, 0.1615] | 1.020× |
| predict_1 | 1 | 0.0007 [0.0007, 0.0024] | 0.0007 [0.0007, 0.0009] | 1.005× |
| predict_1 | 4 | 0.0007 [0.0007, 0.0009] | 0.0007 [0.0007, 0.0011] | 0.972× |
| predict_15 | 1 | 0.0098 [0.0093, 0.0141] | 0.0095 [0.0094, 0.0142] | 1.032× |
| predict_15 | 4 | 0.0100 [0.0095, 0.0113] | 0.0109 [0.0096, 0.0151] | 0.915× |

## What the results support

- 10,000×1,024 prediction, 10 depth-4 trees, 1 thread(s): 20.512→10.048 ms, 2.041× or 51.0% less time.
- 10,000×1,024 prediction, 10 depth-4 trees, 4 thread(s): 5.563→3.474 ms, 1.601× or 37.5% less time.
- Earlier combined-candidate sweeps independently showed the same large
  wide/shallow inference effect (1.83–2.06× at one thread; 1.68–1.80× at four).
  Those are supporting experiments, not measurements of the final binary.
- Gains depend strongly on feature width and traversal cost. The full table
  deliberately retains slower/noisy cases. No overall training, small-batch,
  narrow-input or universal no-regression claim is made. In particular,
  four-thread controls fluctuate substantially on this shared host.

## Threaded variability checks

Two unfavorable four-thread observations were rechecked with 30 samples per
process (60 per binary), retaining both runs. Sampled warm training changed
from 0.859× in the main sweep to 1.065× in `threaded-recheck`; wide 256-feature
shallow inference changed from 0.913× to 1.720×. The corresponding baseline
versus itself run (`same-binary-threaded`) measured 1.085× and 0.775×. This is
substantial shared-host variation, not evidence of a real 22.5% self-regression.
The rerun does not erase the unfavorable main-sweep samples. It limits what
can responsibly be claimed, especially for short, multi-threaded workloads.

## Training investigation: rejected, not shipped

Profiling cold 100,000×30 data preparation found roughly 77 ms of cuts and
27 ms of bin assignment at one thread. A complete dense-block specialization
gave isolated bin assignment a 1.09–1.20× speedup, but repeated cold 20-round training
was about 3–4% slower in two sweeps. It was removed rather than promoting
a phase-only win into an end-to-end claim. Final cold-training control medians
are 0.995×/1.024× at one/four threads; no training code change remains.

Four 8-bit radix passes were also rejected: cuts slowed about 14% compared
with the existing three 11-bit passes. Repeated public single-row calls lost
to the existing batch path at 16–256 rows; this discouraged prioritizing a
higher RowForest cutoff, but does not directly benchmark a modified internal
cutoff because public calls include validation/call overhead.

See [the combined experiment report](combined-experiment.md) for its full
table, unfavorable results, raw-data provenance and reproducible rejected
patches. `combined-rejected.patch` applies to a fresh baseline checkout; the
radix patch applies on top. None of those training changes is in production.

## Reproduce

Copy this PR’s benchmark examples and runner onto the baseline revision as
well as the candidate. Build in separate target directories, or touch
`src/lib.rs` to force recompilation when sharing a target directory between
worktrees. Copy each executable before building the other revision.

```sh
cargo build --release --example bench_cpu
cp target/release/examples/bench_cpu /tmp/candidate-bench_cpu
# Repeat on d7b6a1e with the common benchmark example for baseline.
python scripts/bench_cpu.py --baseline /tmp/baseline-bench_cpu \
  --candidate /tmp/candidate-bench_cpu --output /tmp/inference-results \
  --threads 1 4 --repeats 20 --cases \
  train_cold_full train_full train_wide_sampled predict_wide_256 \
  predict_wide_1024 predict_wide_deep predict_batch predict_16 predict_256 \
  predict_1 predict_15
```

Do not run alongside builds, tests, formatters or other benchmarks. The
[script guide](../../../scripts/README.md#cpu-baselinecandidate-comparison)
describes all operation semantics. `baseline-lanes.log` and
`candidate-lanes.log` are exploratory pre-guard staging microbenchmarks, not
the final end-to-end measurements. `baseline-crossover.jsonl` is reproduced
with `RAYON_NUM_THREADS=1 cargo run --release --example bench_row_crossover -- 30 100 6 5 100`.

Final benchmark binary SHA-256:
`1e3dca52509f2432bf7a7f0e71c73233f7d2c7a78d9ddafe2547b20c7e71b116`.
Baseline SHA-256:
`45a79b249e3acb732c33b098588de9eb3dadb909b29c956e4f6d7c181432ec9f`.

## Validation of the retained code

- `cargo test --profile ci --all-features -- --test-threads=4`: 964 passed,
  including 45 doctests; six ignored tests remain opt-in.
- `cargo clippy --all-targets --all-features -- -D warnings` passed.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` passed.
- Rust formatting checks passed for root, fuzz and cuda-kernels manifests.
- Ruff 0.16.9 check/format passed for the benchmark runner.
- Independent read-only correctness review and raw-summary audit completed.

GPU hardware was not benchmarked; runtime-conditional GPU tests may return
early without a device. No real-data model-quality claim is made.

Archived patches use zero context; apply with `git apply --unidiff-zero`
only on the stated baseline (and the radix patch after the combined patch).
