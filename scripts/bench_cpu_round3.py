#!/usr/bin/env python3
"""Compare already-built bench_cpu_round3 binaries serially in ABBA order.

Retain raw JSONL per-call timings and fail on any model, input, prediction-bit,
or configuration mismatch, including across thread counts. The Rust harness
also checks every prediction bit against a plain scalar tree walk. Examples:

python scripts/bench_cpu_round3.py --baseline /tmp/base --candidate /tmp/new \
    --output /tmp/round3 --threads 1 4 --repeats 7
python scripts/bench_cpu_round3.py --baseline /tmp/base --candidate /tmp/new \
    --output /tmp/round3-cold --cold-layout --cases w1024_p1_d4_t10_b256

Build both binaries with the exact same harness and release/toolchain flags;
this runner does not build, change source, or train a model. Binary SHA-256,
host CPU details, process order, and commands are saved with the raw results.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

ORDER = ("baseline", "candidate", "candidate", "baseline")


def case_catalog(binary):
    result = subprocess.run([str(binary), "--list"], check=True, capture_output=True, text=True)
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--api", choices=["matrix", "rows", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--cold-layout", action="store_true")
    parser.add_argument("--baseline-revision")
    parser.add_argument("--candidate-revision")
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup < 0 or any(t <= 0 for t in args.threads):
        parser.error("repeats/threads must be positive; warmup must be nonnegative")
    if args.iterations is not None and args.iterations <= 0:
        parser.error("iterations must be positive")
    if args.cold_layout and args.iterations not in (None, 1):
        parser.error("--cold-layout uses exactly one call per sample")
    return args, parser


def command(binary, case, args):
    result = [
        str(binary),
        "--case",
        case,
        "--api",
        args.api,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
    ]
    if args.iterations is not None:
        result.extend(["--iterations", str(args.iterations)])
    if args.cold_layout:
        result.append("--cold-layout")
    return result


def identity(record):
    # All descriptive metadata must match. Thread counts intentionally vary;
    # timing is the only remaining permitted difference.
    return {key: value for key, value in record.items() if key not in ("seconds", "threads")}


def timing_summary(values):
    return {
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "samples": len(values),
    }


def main():
    args, parser = parse_args()
    binaries = {name: getattr(args, name).resolve() for name in ("baseline", "candidate")}
    catalogs = {name: case_catalog(binary) for name, binary in binaries.items()}
    if catalogs["baseline"] != catalogs["candidate"]:
        raise RuntimeError("baseline and candidate have different case catalogs")
    available = [record["case"] for record in catalogs["baseline"]]
    cases = args.cases if args.cases is not None else available
    if len(set(cases)) != len(cases) or any(case not in available for case in cases):
        parser.error(f"cases must be distinct members of: {', '.join(available)}")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": platform.platform(),
        "cpuinfo": Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else None,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "binaries": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "revision": getattr(args, f"{name}_revision"),
            }
            for name, path in binaries.items()
        },
        "case_catalog": catalogs["baseline"],
        "selected_cases": cases,
        "threads": args.threads,
        "order": ORDER,
        "warmup_samples": args.warmup,
        "measured_samples_per_process": args.repeats,
        "layout_cache": "cold" if args.cold_layout else "warm",
        "inherited_env": {
            name: os.environ.get(name)
            for name in (
                "RUSTFLAGS",
                "RAYON_NUM_THREADS",
                "CARGO_PROFILE_RELEASE_LTO",
                "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
            )
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    expected = {}
    summaries = []
    apis = ["matrix", "rows"] if args.api == "both" else [args.api]
    with (args.output / "samples.jsonl").open("w") as raw:
        for case in cases:
            for threads in args.threads:
                timings = {api: {name: [] for name in binaries} for api in apis}
                for batch, name in enumerate(ORDER):
                    cmd = command(binaries[name], case, args)
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        env={**os.environ, "RAYON_NUM_THREADS": str(threads)},
                    )
                    records = [
                        json.loads(line) for line in result.stdout.splitlines() if line.strip()
                    ]
                    if [record["api"] for record in records] != apis:
                        raise RuntimeError(f"unexpected API records for {name}, {case}")
                    for record in records:
                        api = record["api"]
                        if record["case"] != case or record["threads"] != threads:
                            raise RuntimeError(
                                f"unexpected case/thread metadata for {name}, {case}"
                            )
                        actual = identity(record)
                        key = (case, api)
                        reference = expected.setdefault(key, actual)
                        if actual != reference:
                            changed = [
                                field for field in actual if actual[field] != reference.get(field)
                            ]
                            raise RuntimeError(
                                f"bitwise/configuration mismatch in {name}, {case}, {api}: "
                                f"{changed}"
                            )
                        timings[api][name].extend(record["seconds"])
                        record.update(engine=name, batch=batch, command=cmd)
                        raw.write(json.dumps(record) + "\n")
                        raw.flush()
                for api in apis:
                    summary = {
                        "case": case,
                        "api": api,
                        "threads": threads,
                        "layout_cache": metadata["layout_cache"],
                    }
                    for name, values in timings[api].items():
                        summary[name] = timing_summary(values)
                    baseline = summary["baseline"]["median_seconds"]
                    candidate = summary["candidate"]["median_seconds"]
                    summary["speedup"] = baseline / candidate
                    summary["change_percent"] = (candidate / baseline - 1) * 100
                    summaries.append(summary)
                    print(json.dumps(summary), flush=True)
                    (args.output / "summary.json").write_text(
                        json.dumps(summaries, indent=2) + "\n"
                    )


if __name__ == "__main__":
    main()
