"""Shared data and helpers for the hessboost test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

# Models saved by each release of the Rust crate, with the margins it
# recorded (repository checkout only).
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
