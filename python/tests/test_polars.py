"""polars frames: names, null and categorical columns, re-coding to a
model's categories, and the scikit-learn estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import hessboost
from conftest import frame
from hessboost import DMatrix, HessboostError

COLORS = ["plum", "cyan", "blue", "green", "red"]


def polars_frame(df: pd.DataFrame, color: pl.DataType | None = None) -> pl.DataFrame:
    """A :func:`conftest.frame` as polars, ``color`` as ``Categorical`` (or
    ``color``)."""
    return pl.DataFrame(
        {
            "color": pl.Series(df["color"].astype(str).tolist(), dtype=color or pl.Categorical()),
            "size": df["size"].to_numpy(),
            "count": pl.Series(
                [None if pd.isna(v) else int(v) for v in df["count"]], dtype=pl.Int64
            ),
            "flag": df["flag"].to_numpy(),
        }
    )


def test_polars_frames_train_as_the_equal_pandas_frame() -> None:
    # Codes of a Categorical index a pool every Categorical shares, so other
    # values seen first must not change them.
    pl.Series(["zzz", "aaa", "green"], dtype=pl.Categorical)
    df, y = frame()
    df.loc[3, "count"] = pd.NA
    params = {"max_depth": 3}
    expected = hessboost.train(params, DMatrix(df, y), 20)
    for color in [pl.Categorical(), pl.Enum(sorted(COLORS))]:
        polars = polars_frame(df, color)
        matrix = DMatrix(polars, y)
        assert matrix.feature_names == ["color", "size", "count", "flag"]
        assert matrix.feature_types == ["c", "q", "q", "q"]
        booster = hessboost.train(params, matrix, 20)
        assert booster.save_raw() == expected.save_raw()
        np.testing.assert_array_equal(booster.predict(polars), expected.predict(df))


def test_polars_frames_are_recoded_to_the_model_categories() -> None:
    df, y = frame()
    booster = hessboost.train({"max_depth": 3}, DMatrix(polars_frame(df), y), 20)
    expected = booster.predict(df)
    # An Enum listing the categories in another order has other codes.
    reordered = polars_frame(df, pl.Enum(COLORS))
    np.testing.assert_array_equal(booster.predict(reordered), expected)
    with pytest.raises(HessboostError, match="categories of feature 'color' differ"):
        booster.predict(DMatrix(reordered))
    # A pandas model reads polars frames and the other way round.
    from_pandas = hessboost.train({"max_depth": 3}, DMatrix(df, y), 20)
    np.testing.assert_array_equal(from_pandas.predict(reordered), expected)
    # An unseen category is missing.
    head = polars_frame(df.head(3))
    unseen = head.with_columns(pl.Series("color", ["mauve"] * 3, dtype=pl.Categorical))
    missing = head.with_columns(pl.Series("color", [None] * 3, dtype=pl.Categorical))
    np.testing.assert_array_equal(booster.predict(unseen), booster.predict(missing))


def test_polars_frames_need_numeric_or_categorical_columns() -> None:
    df, y = frame(rows=20)
    polars = polars_frame(df)
    with pytest.raises(TypeError, match="column 'name' has dtype String"):
        DMatrix(polars.with_columns(name=pl.lit("x")), y)
    with pytest.raises(HessboostError, match="column 'color' is categorical"):
        DMatrix(polars, y, enable_categorical=False)


def test_estimators_take_polars_frames() -> None:
    from hessboost.sklearn import HessboostRegressor

    df, y = frame()
    polars = polars_frame(df)
    model = HessboostRegressor(n_estimators=20, max_depth=3).fit(
        polars, y, eval_set=[(polars_frame(df, pl.Enum(COLORS)), y)]
    )
    assert list(model.feature_names_in_) == ["color", "size", "count", "flag"]
    reference = HessboostRegressor(n_estimators=20, max_depth=3).fit(df, y)
    np.testing.assert_array_equal(model.predict(polars), reference.predict(df))
    with pytest.raises(ValueError, match="feature names"):
        model.predict(polars.select(["size", "color", "count", "flag"]))
