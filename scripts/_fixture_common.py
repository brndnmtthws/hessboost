from __future__ import annotations

import json
import zlib
from os import PathLike
from typing import Any

import numpy as np


def seed_for(base: int, name: str) -> int:
    return base ^ zlib.crc32(name.encode())


def json_floats(values: Any, dtype: Any) -> list:
    """Flatten an array at the requested precision and map NaN to JSON null."""
    flat = np.asarray(values, dtype=dtype).ravel()
    out = flat.astype(np.float64).astype(object)
    out[np.isnan(flat)] = None
    return out.tolist()


def dense_f32(values: list, rows: int, cols: int) -> np.ndarray:
    dense = np.array([np.nan if value is None else value for value in values], dtype=np.float32)
    return dense.reshape(rows, cols)


def write_json(path: str | PathLike[str], value: Any) -> None:
    with open(path, "w") as output:
        json.dump(value, output, separators=(",", ":"))
