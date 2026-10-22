#!/usr/bin/env python3
"""Compare two bench_cpu binaries serially in ABBA order; retain raw timings."""

import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

# operation, measured rows, features, rounds, depth, colsample, layout, training rows
CASES = {
    "train_wide_sampled": ("train", 20000, 128, 30, 6, 0.25, "dense", 20000),
    "train_wide_sampled_cold": ("train-cold", 20000, 128, 30, 6, 0.25, "dense", 20000),
    "train_wide_80pct": ("train", 20000, 128, 30, 6, 0.8, "dense", 20000),
    "train_full": ("train", 50000, 20, 50, 6, 1, "dense", 50000),
    "train_missing_sampled": ("train", 20000, 30, 30, 6, 0.25, "sparse", 20000),
    "predict_1": ("predict", 1, 30, 100, 6, 1, "dense", 10000),
    "predict_8": ("predict", 8, 30, 100, 6, 1, "dense", 10000),
    "predict_15": ("predict", 15, 30, 100, 6, 1, "dense", 10000),
    "predict_16": ("predict", 16, 30, 100, 6, 1, "dense", 10000),
    "predict_batch": ("predict", 100000, 30, 100, 6, 1, "dense", 100000),
    "train_cold_full": ("train-cold", 100000, 30, 20, 6, 1, "dense", 100000),
    "train_cold_one_round": ("train-cold", 100000, 30, 1, 6, 1, "dense", 100000),
    "train_cold_missing": ("train-cold", 50000, 30, 20, 6, 1, "sparse", 50000),
    "bins_dense": ("bins", 100000, 30, 1, 6, 1, "dense", 100000),
    "bins_wide": ("bins", 20000, 128, 1, 6, 1, "dense", 20000),
    "bins_missing": ("bins", 100000, 30, 1, 6, 1, "sparse", 100000),
    "cuts_dense": ("cuts", 100000, 30, 1, 6, 1, "dense", 100000),
    "predict_wide_256": ("predict", 10000, 256, 10, 4, 1, "dense", 10000),
    "predict_wide_1024": ("predict", 10000, 1024, 10, 4, 1, "dense", 10000),
    "predict_wide_deep": ("predict", 10000, 256, 100, 6, 1, "dense", 10000),
    "predict_64": ("predict", 64, 30, 100, 6, 1, "dense", 10000),
    "predict_256": ("predict", 256, 30, 100, 6, 1, "dense", 10000),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if args.repeats <= 0 or any(t <= 0 for t in args.threads):
        parser.error("repeats and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {name: getattr(args, name).resolve() for name in ("baseline", "candidate")}
    metadata = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": platform.platform(),
        "cpuinfo": Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else None,
        "binaries": {
            k: {"path": str(v), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()}
            for k, v in binaries.items()
        },
        "order": ["baseline", "candidate", "candidate", "baseline"],
        "repeats_per_process": args.repeats,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summaries = []
    with (args.output / "samples.jsonl").open("w") as raw:
        for case in args.cases:
            op, rows, features, rounds, depth, colsample, layout, train_rows = CASES[case]
            expected = None
            for threads in args.threads:
                timings = {name: [] for name in binaries}
                for batch, name in enumerate(metadata["order"]):
                    cmd = [
                        str(binaries[name]),
                        op,
                        rows,
                        features,
                        rounds,
                        depth,
                        colsample,
                        args.repeats,
                        layout,
                        train_rows,
                    ]
                    result = subprocess.run(
                        [str(x) for x in cmd],
                        check=True,
                        capture_output=True,
                        text=True,
                        env={**os.environ, "RAYON_NUM_THREADS": str(threads)},
                    )
                    record = json.loads(result.stdout)
                    identity = (record["checksum"], record["model_checksum"])
                    if expected is None:
                        expected = identity
                    if identity != expected:
                        raise RuntimeError(
                            f"bitwise checksum mismatch in {case}: {identity} != {expected}"
                        )
                    record.update(case=case, engine=name, batch=batch)
                    raw.write(json.dumps(record) + "\n")
                    raw.flush()
                    timings[name].extend(record["seconds"])
                summary = {"case": case, "threads": threads}
                for name, values in timings.items():
                    summary[name] = {
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "samples": len(values),
                    }
                summary["speedup"] = summary["baseline"]["median"] / summary["candidate"]["median"]
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
