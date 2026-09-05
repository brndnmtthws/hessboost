#!/usr/bin/env python3
"""Compare two compiled Criterion executables in baseline/optimized/optimized/baseline order."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for results")
    parser.add_argument("--filter", default=".", help="Criterion benchmark-name regex")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=float, default=0.5)
    parser.add_argument("--measurement", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    if args.threads < 1 or args.samples < 10 or args.resamples < 1:
        parser.error("threads/resamples must be positive and samples must be at least 10")
    if args.warmup <= 0 or args.measurement <= 0:
        parser.error("warmup and measurement must be positive")
    executables = {name: path.resolve(strict=True) for name, path in
                   [("baseline", args.baseline), ("optimized", args.optimized)]}
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "platform": platform.platform(),
        "rayon_threads": args.threads,
        "filter": args.filter,
        "warmup_seconds": args.warmup,
        "requested_measurement_seconds": args.measurement,
        "requested_samples": args.samples,
        "resamples": args.resamples,
        "executable_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in executables.items()},
        "run_started_at": {},
    }
    estimates, raw = {}, {}
    for name, repeat in [("baseline", 1), ("optimized", 1), ("optimized", 2), ("baseline", 2)]:
        label = f"{name}-{repeat}"
        folder = args.output / label
        folder.mkdir()
        metadata["run_started_at"][label] = datetime.now(timezone.utc).isoformat()
        command = [str(executables[name]), "--bench", args.filter, "--noplot",
                   "--warm-up-time", str(args.warmup), "--measurement-time", str(args.measurement),
                   "--sample-size", str(args.samples), "--nresamples", str(args.resamples)]
        env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads),
                   CRITERION_HOME=str((folder / "criterion").resolve()))
        with (folder / "stdout.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        estimates[label] = {
            str(path.parent.parent.relative_to(folder / "criterion")): json.loads(path.read_text())
            for path in folder.glob("criterion/**/new/estimates.json")
        }
        if not estimates[label]:
            raise RuntimeError(f"No benchmarks matched {args.filter!r}")
        raw[label] = {
            "command": command,
            "stdout": (folder / "stdout.log").read_text(),
            "files": {str(path.relative_to(folder)): json.loads(path.read_text())
                      for path in folder.glob("criterion/**/new/*.json")},
        }
        print(f"{label}: {len(estimates[label])} cases", flush=True)
    cases = set(estimates["baseline-1"])
    if any(set(run) != cases for run in estimates.values()):
        raise RuntimeError("The executables produced different benchmark cases")
    results = {}
    for case in sorted(cases):
        medians = {label: run[case]["median"] for label, run in estimates.items()}
        baseline = [medians[f"baseline-{i}"]["point_estimate"] for i in [1, 2]]
        optimized = [medians[f"optimized-{i}"]["point_estimate"] for i in [1, 2]]
        results[case] = {
            "baseline_median_ms": sum(baseline) / 2e6,
            "optimized_median_ms": sum(optimized) / 2e6,
            "elapsed_time_reduction_percent": 100 * (1 - sum(optimized) / sum(baseline)),
            "paired_reduction_percent": [100 * (1 - o / b) for b, o in zip(baseline, optimized)],
            "run_medians_ns": medians,
        }
    (args.output / "comparison.json").write_text(
        json.dumps({"metadata": metadata, "benchmarks": results}, indent=2) + "\n")
    (args.output / "samples.json.gz").write_bytes(
        gzip.compress(json.dumps(raw, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
