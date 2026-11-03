#!/usr/bin/env python3
"""Prepare four chronological forecasting tasks from the public UCI archives.

Run ``python scripts/prepare_forecasting.py --output fixtures/research-real``.
Only the standard library is needed, including the system timezone database.
Downloads are retained in OUTPUT/raw; prepared JSON and provenance JSONL on
stdout are deterministic for the pinned source archives.

Source observations become available at the end of their recorded hour. Bike
uses America/New_York and Beijing uses Asia/Shanghai. Bike does not distinguish
folds of the repeated autumn hour: those ambiguous observations are omitted,
not assigned an invented offset. Missing labels and timestamp gaps remain in
an exact-time lookup; they never turn an hourly lag into a previous-row lag.
Target calendar features describe the forecast source hour, one elapsed hour
before label availability. No future observed weather or target components are
used. Weather stays in the source units, including Bike's supplied normalized
values; no full-data feature statistics are estimated.

Splits use 50/20/10/20 percent of the usable decision-time span, not row counts.
Labels at or beyond the next split boundary are purged before computing the
training-only RMS label scale. The four documents follow the input contract of
examples/research_forecasting.rs; stdout also records the clock assumptions,
ambiguous-hour omissions, actual boundaries, row counts, and output hashes.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.request
import zipfile
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from zoneinfo import ZoneInfo

HOUR = 3600
LAGS = (0, 1, 24, 168)
HORIZONS = (1, 6)
SPLIT_NAMES = ("train", "validation", "calibration", "test")
LICENSE = "CC BY 4.0; https://creativecommons.org/licenses/by/4.0/"
MISSING = frozenset(("", "NA"))


@dataclass(frozen=True)
class Source:
    """Fixed schema and provenance from an inspected primary-source archive."""

    name: str
    url: str
    sha256: str
    member: str
    timezone: str
    citation: str
    fields: tuple[str, ...]
    target: str
    numeric: tuple[str, ...]
    category: str
    categories: tuple[str, ...]


SOURCES = (
    Source(
        name="bike",
        url="https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        sha256="b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
        member="hour.csv",
        timezone="America/New_York",
        citation=(
            "Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5W894. "
            "Fanaee-T, H. and Gama, J. (2013). Event labeling combining ensemble detectors "
            "and background knowledge. https://doi.org/10.1007/s13748-013-0040-3."
        ),
        fields=(
            "instant",
            "dteday",
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "casual",
            "registered",
            "cnt",
        ),
        target="cnt",
        numeric=("temp", "atemp", "hum", "windspeed"),
        category="weathersit",
        categories=("1", "2", "3", "4"),
    ),
    Source(
        name="beijing",
        url="https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
        sha256="40cdd29c24918e855133dce280ff59779bb0b5852898e05bafcf422a4cc0a175",
        member="PRSA_data_2010.1.1-2014.12.31.csv",
        timezone="Asia/Shanghai",
        citation=(
            "Chen, S. (2015). Beijing PM2.5 [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5JS49."
        ),
        fields=(
            "No",
            "year",
            "month",
            "day",
            "hour",
            "pm2.5",
            "DEWP",
            "TEMP",
            "PRES",
            "cbwd",
            "Iws",
            "Is",
            "Ir",
        ),
        target="pm2.5",
        numeric=("DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"),
        category="cbwd",
        # These literal source codes are not ordinal or inferred compass angles.
        categories=("NE", "NW", "SE", "cv"),
    ),
)


@dataclass(frozen=True)
class Observation:
    label: float | None
    numeric: tuple[float | None, ...]
    category: str | None


@dataclass
class RawData:
    observations: dict[int, Observation]
    source_rows: int
    missing_labels: int
    ambiguous_source_hours: list[str]


class Split(TypedDict):
    x: list[float]
    y: list[float]
    decision_at: list[int]
    label_available_at: list[int]


class Prepared(TypedDict):
    dataset: str
    horizon_hours: int
    source_url: str
    source_sha256: str
    license: str
    citation: str
    feature_names: list[str]
    label_scale: float
    splits: dict[str, Split]


def download(source: Source, output: Path) -> bytes:
    """Cache only the exact inspected UCI archive, not an unpinned replacement."""
    raw_path = output / "raw" / f"{source.name}.zip"
    if raw_path.exists():
        payload = raw_path.read_bytes()
    else:
        with urllib.request.urlopen(source.url, timeout=120) as response:
            payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != source.sha256:
        raise ValueError(
            f"{source.name}: archive SHA256 {digest} differs from pinned {source.sha256}"
        )
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payload)
    return payload


def numeric_value(value: str, context: str) -> float | None:
    if value in MISSING:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: nonfinite numeric value {value!r}")
    return result


def source_hour(source: Source, row: dict[str, str], timezone: ZoneInfo) -> datetime:
    if source.name == "bike":
        year, month, day = (int(part) for part in row["dteday"].split("-"))
        hour = int(row["hr"])
    else:
        year, month, day, hour = (int(row[name]) for name in ("year", "month", "day", "hour"))
    return datetime(year, month, day, hour, tzinfo=timezone)


def hour_instants(hour: datetime) -> set[int]:
    """Find both valid folds; zero means a nonexistent civil hour, two ambiguous."""
    instants = set()
    for fold in (0, 1):
        timestamp = int(hour.replace(fold=fold).timestamp())
        restored = datetime.fromtimestamp(timestamp, tz=hour.tzinfo)
        if restored.replace(fold=0) == hour.replace(fold=0):
            instants.add(timestamp)
    return instants


def read_observations(source: Source, payload: bytes) -> RawData:
    timezone = ZoneInfo(source.timezone)
    data = RawData({}, 0, 0, [])
    previous_hour: datetime | None = None
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist().count(source.member) != 1:
            raise ValueError(f"{source.name}: expected exactly one {source.member}")
        with (
            archive.open(source.member) as binary,
            io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text,
        ):
            reader = csv.DictReader(text)
            fields = reader.fieldnames
            if fields is None or len(set(fields)) != len(fields):
                raise ValueError(f"{source.name}: missing or duplicate CSV header")
            missing = set(source.fields).difference(fields)
            if missing:
                raise ValueError(f"{source.name}: missing CSV columns {sorted(missing)}")
            for row in reader:
                data.source_rows += 1
                context = f"{source.name}: CSV row {reader.line_num}"
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{context}: malformed CSV row")
                hour = source_hour(source, row, timezone)
                if previous_hour is not None and hour <= previous_hour:
                    raise ValueError(f"{context}: duplicate or out-of-order source hour")
                previous_hour = hour
                label = numeric_value(row[source.target], context)
                if label is None:
                    data.missing_labels += 1
                elif label < 0:
                    raise ValueError(f"{context}: negative {source.target}")
                numeric = tuple(numeric_value(row[name], context) for name in source.numeric)
                category = row[source.category]
                if category in MISSING:
                    category = None
                elif category not in source.categories:
                    raise ValueError(f"{context}: unknown {source.category} code {category!r}")
                instants = hour_instants(hour)
                if not instants:
                    raise ValueError(f"{context}: nonexistent local source hour {hour.isoformat()}")
                if len(instants) != 1:
                    # The archive provides no fold flag: neither possible instant is guessed.
                    data.ambiguous_source_hours.append(hour.strftime("%Y-%m-%dT%H:00:00"))
                    continue
                available_at = instants.pop() + HOUR
                if available_at in data.observations:
                    raise ValueError(f"{context}: duplicate absolute availability timestamp")
                data.observations[available_at] = Observation(label, numeric, category)
    if not data.observations:
        raise ValueError(f"{source.name}: no unambiguous observations")
    return data


def feature_names(source: Source) -> list[str]:
    names = []
    for name in source.numeric:
        names.extend((f"origin.{name}", f"origin.{name}_missing"))
    for lag in LAGS:
        names.extend((f"target_lag{lag}h", f"target_lag{lag}h_missing"))
    names.extend(f"origin.{source.category}={category}" for category in source.categories)
    names.append(f"origin.{source.category}_missing")
    names.append("target.year")
    names.extend(f"target.month={month:02d}" for month in range(1, 13))
    names.extend(f"target.weekday={weekday}" for weekday in range(7))
    names.extend(f"target.hour={hour:02d}" for hour in range(24))
    return names


def append_numeric(features: list[float], value: float | None) -> None:
    features.extend((0.0 if value is None else value, float(value is None)))


def features_at(
    source: Source,
    observations: dict[int, Observation],
    decision_at: int,
    label_at: int,
    timezone: ZoneInfo,
) -> list[float]:
    observed = observations[decision_at]
    features: list[float] = []
    for value in observed.numeric:
        append_numeric(features, value)
    for lag in LAGS:
        past = observations.get(decision_at - lag * HOUR)
        append_numeric(features, None if past is None else past.label)
    features.extend(float(observed.category == category) for category in source.categories)
    features.append(float(observed.category is None))
    # Calendar is computable at issuance; never read any future observation's features.
    target_hour = datetime.fromtimestamp(label_at - HOUR, tz=timezone)
    features.append(float(target_hour.year))
    features.extend(float(target_hour.month == month) for month in range(1, 13))
    features.extend(float(target_hour.weekday() == weekday) for weekday in range(7))
    features.extend(float(target_hour.hour == hour) for hour in range(24))
    return features


def prepare(source: Source, data: RawData, horizon: int) -> tuple[Prepared, dict[str, object]]:
    observations = data.observations
    delay = horizon * HOUR
    # Retain missing-label observations in the lookup: they can still be origins.
    usable = [
        decision_at
        for decision_at in sorted(observations)
        if (target := observations.get(decision_at + delay)) is not None
        and target.label is not None
    ]
    if len(usable) < 2:
        raise ValueError(f"{source.name}-h{horizon}: insufficient usable origins")
    first, last = usable[0], usable[-1]
    span = last - first
    boundaries = [first + span * fraction // 10 for fraction in (5, 7, 8)]
    if not first < boundaries[0] < boundaries[1] < boundaries[2] < last:
        raise ValueError(f"{source.name}-h{horizon}: collapsed split boundaries")
    splits: dict[str, Split] = {
        name: {"x": [], "y": [], "decision_at": [], "label_available_at": []}
        for name in SPLIT_NAMES
    }
    purged = dict.fromkeys(SPLIT_NAMES, 0)
    names = feature_names(source)
    timezone = ZoneInfo(source.timezone)
    for decision_at in usable:
        split_index = bisect_right(boundaries, decision_at)
        name = SPLIT_NAMES[split_index]
        label_at = decision_at + delay
        if split_index < len(boundaries) and label_at >= boundaries[split_index]:
            purged[name] += 1
            continue
        label = observations[label_at].label
        if label is None:
            raise ValueError("usable origin unexpectedly has a missing future label")
        features = features_at(source, observations, decision_at, label_at, timezone)
        if len(features) != len(names) or not all(math.isfinite(value) for value in features):
            raise ValueError(f"{source.name}-h{horizon}: invalid feature row")
        split = splits[name]
        split["x"].extend(features)
        split["y"].append(label)
        split["decision_at"].append(decision_at)
        split["label_available_at"].append(label_at)
    for index, name in enumerate(SPLIT_NAMES):
        split = splits[name]
        if not split["y"]:
            raise ValueError(f"{source.name}-h{horizon}: empty {name} split")
        if index:
            previous = splits[SPLIT_NAMES[index - 1]]
            if previous["label_available_at"][-1] >= split["decision_at"][0]:
                raise ValueError(f"{source.name}-h{horizon}: label leakage before {name}")
    train_labels = splits["train"]["y"]
    label_scale = math.sqrt(math.fsum(value * value for value in train_labels) / len(train_labels))
    if not math.isfinite(label_scale) or label_scale <= 0:
        raise ValueError(f"{source.name}-h{horizon}: training RMS must be finite and positive")
    for split in splits.values():
        split["y"] = [value / label_scale for value in split["y"]]
        if not all(math.isfinite(value) for value in split["y"]):
            raise ValueError(f"{source.name}-h{horizon}: nonfinite scaled label")
    document: Prepared = {
        "dataset": source.name,
        "horizon_hours": horizon,
        "source_url": source.url,
        "source_sha256": source.sha256,
        "license": LICENSE,
        "citation": source.citation,
        "feature_names": names,
        "label_scale": label_scale,
        "splits": splits,
    }
    metadata: dict[str, object] = {
        "dataset": source.name,
        "horizon_hours": horizon,
        "source_url": source.url,
        "source_sha256": source.sha256,
        "source_member": source.member,
        "license": LICENSE,
        "citation": source.citation,
        "source_timezone": source.timezone,
        "timestamp_units": "absolute POSIX seconds",
        "availability": "recorded source-hour start plus 3600 elapsed seconds",
        "target_calendar": "label_available_at minus 3600 seconds, in source timezone",
        "ambiguous_hour_policy": "omit both possible instants; no invented fold",
        "ambiguous_source_hours_omitted": data.ambiguous_source_hours,
        "source_rows": data.source_rows,
        "source_missing_labels": data.missing_labels,
        "lookup_rows": len(observations),
        "usable_origin_rows": len(usable),
        "origins_without_future_label": len(observations) - len(usable),
        "missing_feature_policy": "zero plus per-feature missingness indicator; no imputation fit",
        "lag_hours": LAGS,
        "weekday_convention": "Monday=0, Sunday=6",
        "label_scale": label_scale,
        "label_scale_fit": "RMS of purged training labels only; features retain source units",
        "feature_count": len(names),
        "usable_decision_range": [first, last],
        "split_boundary_rule": "first + floor((last-first)*fraction), fractions 0.5,0.7,0.8",
        "split_boundaries": dict(zip(SPLIT_NAMES[1:], boundaries, strict=True)),
        "purge_rule": "label_available_at strictly less than next split boundary",
        "splits": {
            name: {
                "rows": len(split["y"]),
                "purged_rows": purged[name],
                "decision_range": [split["decision_at"][0], split["decision_at"][-1]],
                "label_range": [split["label_available_at"][0], split["label_available_at"][-1]],
            }
            for name, split in splits.items()
        },
    }
    return document, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures" / "research-real",
        help="directory for four prepared JSON files and cached raw UCI archives",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        data = read_observations(source, download(source, args.output))
        for horizon in HORIZONS:
            document, metadata = prepare(source, data, horizon)
            path = args.output / f"{source.name}-h{horizon}.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(document, handle, separators=(",", ":"), allow_nan=False)
                handle.write("\n")
            with path.open("rb") as handle:
                metadata["prepared_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
            metadata["prepared_path"] = str(path)
            print(json.dumps(metadata, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
