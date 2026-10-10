"""Shared data and helpers for the hessboost test suite."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, overload

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from hessboost.diffusion import DiffusionModel
from hessboost.diffusion.forest import ForestModel

# Models the Rust crate's releases saved, with the margins they recorded: one
# directory per distinguishable writer output (repository checkout only).
SAVED_MODELS = Path(__file__).resolve().parents[2] / "tests" / "data" / "saved"


def saved_models_dir() -> Path:
    """The Rust crate's saved-model directory; skips the calling test when
    it is not available (for example, when testing an unpacked sdist)."""
    if not SAVED_MODELS.is_dir():
        pytest.skip(f"saved models not found at {SAVED_MODELS} (not a repository checkout)")
    return SAVED_MODELS


def regression(
    rows: int = 400, features: int = 5, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Features and a noisy linear target."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, features))
    y = x[:, 0] - 2.0 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3] + rng.normal(0.0, 0.1, rows)
    return x, y


def classes(
    rows: int = 400, n_classes: int = 2, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Features and ``n_classes`` separable-ish class labels."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 4))
    score = x[:, 0] + 0.5 * x[:, 1] + rng.normal(0.0, 0.3, rows)
    edges = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    return x, np.searchsorted(edges, score).astype(np.int64)


def additive(
    rows: int, seed: int, *, features: int, noise: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``features`` (2 or 3) uniform features and an additive target,
    ``sin(6 x0) + (x1 - 0.5)^2 [+ (x2 > 0.5)]`` plus Gaussian ``noise``."""
    rng = np.random.default_rng(seed)
    x = rng.random((rows, features))
    y = np.sin(6 * x[:, 0]) + (x[:, 1] - 0.5) ** 2
    if features > 2:
        y = y + (x[:, 2] > 0.5)
    return x, y + noise * rng.standard_normal(rows)


def rmse(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """The root mean squared difference of ``a`` and ``b``, in ``float64``."""
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - b) ** 2)))


@overload
def reloaded(model: DiffusionModel, tmp_path: Path, suffix: str) -> list[DiffusionModel]: ...
@overload
def reloaded(model: ForestModel, tmp_path: Path, suffix: str) -> list[ForestModel]: ...
def reloaded(model: DiffusionModel | ForestModel, tmp_path: Path, suffix: str) -> list[Any]:
    """``model`` read back from each of its encodings: binary and JSON
    bytes (detected and named), a binary file named with ``suffix`` and a
    JSON file (detected and named), and a pickle."""
    cls = type(model)
    binary, text = model.to_bytes(), model.to_bytes("json")
    model.save(tmp_path / f"model{suffix}")
    model.save(tmp_path / "model.json", "json")
    assert (tmp_path / "model.json").read_bytes() == text
    return [
        cls.from_bytes(binary),
        cls.from_bytes(text),
        cls.from_bytes(bytearray(binary), "binary"),
        cls.from_bytes(memoryview(text), "json"),
        cls.load(tmp_path / f"model{suffix}"),
        cls.load(tmp_path / "model.json"),
        cls.load(tmp_path / "model.json", "json"),
        pickle.loads(pickle.dumps(model)),
    ]


def frame(rows: int = 400, seed: int = 0) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """A frame of a five-category ``color`` (a large effect), a numeric
    ``size``, a nullable integer ``count`` and a boolean ``flag``, with its
    target."""
    rng = np.random.default_rng(seed)
    colors = np.array(["red", "green", "blue", "cyan", "plum"])
    color = colors[rng.integers(0, 5, rows)]
    effect = {"red": 0.0, "green": 3.0, "blue": -2.0, "cyan": 1.0, "plum": 5.0}
    size = rng.normal(size=rows)
    y = np.array([effect[c] for c in color]) + size
    df = pd.DataFrame(
        {
            "color": pd.Categorical(color),
            "size": size,
            "count": pd.array(rng.integers(0, 5, rows), dtype="Int64"),
            "flag": rng.random(rows) < 0.5,
        }
    )
    return df, y


def reorder_colors(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with the same ``color`` values but their categories listed in
    another order, so their codes differ."""
    reordered = df.copy()
    reordered["color"] = reordered["color"].cat.reorder_categories(
        ["plum", "cyan", "blue", "green", "red"]
    )
    assert not np.array_equal(reordered["color"].cat.codes, df["color"].cat.codes)
    return reordered


def numpy_codes(df: pd.DataFrame) -> NDArray[np.float64]:
    """A :func:`frame` as plain numbers, ``color`` as its category codes
    (train with ``feature_types=FRAME_TYPES``)."""
    return np.column_stack(
        [df["color"].cat.codes, df["size"], df["count"].astype(float), df["flag"]]
    ).astype(np.float64)


FRAME_TYPES = ["c", "q", "q", "q"]
