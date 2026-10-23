#!/usr/bin/env python3
"""Compare two bench_quantile_round3 binaries serially in ABBA order; retain raw timings."""

import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

# operation, rows, features, rounds, depth, layout, weighting, training mode
CASES = {
    "cuts_dense": ("cuts", 100_000, 30, 1, 6, "dense", "weighted", "approx-logistic"),
    "cuts_duplicates": ("cuts", 100_000, 30, 1, 6, "duplicates", "weighted", "approx-logistic"),
    "cuts_card64": ("cuts", 100_000, 30, 1, 6, "cardinality-64", "weighted", "approx-logistic"),
    "cuts_card256": ("cuts", 100_000, 30, 1, 6, "cardinality-256", "weighted", "approx-logistic"),
    "cuts_card1024": ("cuts", 100_000, 30, 1, 6, "cardinality-1024", "weighted", "approx-logistic"),
    "cuts_ordered": ("cuts", 100_000, 30, 1, 6, "ordered", "weighted", "approx-logistic"),
    "cuts_descending": ("cuts", 100_000, 30, 1, 6, "descending", "weighted", "approx-logistic"),
    "cuts_nearly_ordered": (
        "cuts",
        100_000,
        30,
        1,
        6,
        "nearly-ordered",
        "weighted",
        "approx-logistic",
    ),
    "cuts_two_runs": ("cuts", 100_000, 30, 1, 6, "two-runs", "weighted", "approx-logistic"),
    "cuts_organ_pipe": ("cuts", 100_000, 30, 1, 6, "organ-pipe", "weighted", "approx-logistic"),
    "cuts_constant": ("cuts", 100_000, 30, 1, 6, "constant", "weighted", "approx-logistic"),
    "cuts_ordered_lowcard": (
        "cuts",
        100_000,
        30,
        1,
        6,
        "ordered-lowcard",
        "weighted",
        "approx-logistic",
    ),
    "cuts_small": ("cuts", 1_024, 30, 1, 6, "dense", "weighted", "approx-logistic"),
    "cuts_threshold": ("cuts", 2_048, 30, 1, 6, "dense", "weighted", "approx-logistic"),
    "cuts_medium": ("cuts", 4_096, 30, 1, 6, "dense", "weighted", "approx-logistic"),
    "approx_dense": ("train-cold", 30_000, 30, 10, 6, "dense", "weighted", "approx-logistic"),
    "approx_unit": ("train-cold", 30_000, 30, 10, 6, "dense", "unit", "approx-logistic"),
    "approx_missing": ("train-cold", 30_000, 30, 10, 6, "sparse", "weighted", "approx-logistic"),
    "approx_duplicates": (
        "train-cold",
        30_000,
        30,
        10,
        6,
        "duplicates",
        "weighted",
        "approx-logistic",
    ),
    "approx_card64": (
        "train-cold",
        30_000,
        30,
        10,
        6,
        "cardinality-64",
        "weighted",
        "approx-logistic",
    ),
    "approx_card256": (
        "train-cold",
        30_000,
        30,
        10,
        6,
        "cardinality-256",
        "weighted",
        "approx-logistic",
    ),
    "approx_card1024": (
        "train-cold",
        30_000,
        30,
        10,
        6,
        "cardinality-1024",
        "weighted",
        "approx-logistic",
    ),
    "approx_ordered": ("train-cold", 30_000, 30, 10, 6, "ordered", "weighted", "approx-logistic"),
    "approx_descending": (
        "train-cold",
        30_000,
        30,
        10,
        6,
        "descending",
        "weighted",
        "approx-logistic",
    ),
    "approx_nearly_ordered": (
        "train-cold",
        30_000,
        30,
        10,
        6,
        "nearly-ordered",
        "weighted",
        "approx-logistic",
    ),
    "approx_two_runs": ("train-cold", 30_000, 30, 10, 6, "two-runs", "weighted", "approx-logistic"),
    "approx_organ_pipe": (
        "train-cold",
        30_000,
        30,
        10,
        6,
        "organ-pipe",
        "weighted",
        "approx-logistic",
    ),
    "hist_control": ("train-cold", 30_000, 30, 10, 6, "dense", "weighted", "hist-logistic"),
    "approx_square_control": (
        "train-cold",
        30_000,
        30,
        10,
        6,
        "dense",
        "weighted",
        "approx-square",
    ),
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
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
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
            op, rows, features, rounds, depth, layout, weighting, mode = CASES[case]
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
                        args.repeats,
                        layout,
                        weighting,
                        mode,
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
                    record.update(
                        case=case, engine=name, batch=batch, command=[str(x) for x in cmd]
                    )
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
