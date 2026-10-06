"""polars frames (1.x and 2.x): names, null and categorical columns,
re-coding to a model's categories, LazyFrames, metadata taken from a
frame's columns, and the scikit-learn estimators."""

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
    pytest.importorskip("sklearn")  # the musllinux wheel is tested without it
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


def labelled_frame() -> tuple[pl.DataFrame, pd.DataFrame]:
    """A :func:`polars_frame` with its target ``y``, a weight ``w`` and a
    sorted query id ``qid`` as columns, and the pandas frame it came from."""
    df, y = frame()
    rng = np.random.default_rng(1)
    full = polars_frame(df).with_columns(
        y=pl.Series(y),
        w=pl.Series(rng.uniform(0.5, 2.0, len(y))),
        qid=pl.Series(np.repeat(np.arange(len(y) // 20), 20)),
    )
    return full, df


def test_lazyframe_takes_labels_and_weights_from_its_columns() -> None:
    full, df = labelled_frame()
    features = ["color", "size", "count", "flag"]
    params = {"max_depth": 3}
    # One collect yields the features, labels and weights (polars 2.0
    # collects on the streaming engine).
    lazy = DMatrix(full.lazy().drop("qid"), label="y", weight="w")
    assert lazy.feature_names == features
    arrays = DMatrix(full.select(features), label=full["y"], weight=full["w"])
    np.testing.assert_array_equal(lazy.get_label(), arrays.get_label())
    np.testing.assert_array_equal(lazy.get_weight(), arrays.get_weight())
    booster = hessboost.train(params, lazy, 20)
    assert booster.save_raw() == hessboost.train(params, arrays, 20).save_raw()
    # Predictions align with the collected rows.
    np.testing.assert_array_equal(
        booster.predict(full.lazy().select(features)), booster.predict(df)
    )
    # Query ids by name, numeric or string, give the groups.
    ranked = DMatrix(full.lazy().drop("w"), label="y", qid="qid")
    assert ranked.feature_names == features
    assert ranked.get_group().tolist() == [20] * 20
    strings = full.with_columns(pl.col("qid").cast(pl.String).str.zfill(3))
    assert DMatrix(strings, label="y", qid="qid").get_group().tolist() == [20] * 20


def test_lazyframe_refuses_metadata_it_cannot_align() -> None:
    full, _ = labelled_frame()
    lazy = full.lazy()
    y = full["y"].to_numpy()
    with pytest.raises(HessboostError, match="label is an array, but data is a polars LazyFrame"):
        DMatrix(lazy, label=y)
    with pytest.raises(HessboostError, match="weight is an array"):
        DMatrix(lazy, label="y", weight=full["w"])
    with pytest.raises(HessboostError, match=r"group is an array.*qid="):
        DMatrix(lazy, label="y", group=[20] * 20)
    # Unlabelled data (a prediction frame) has nothing to align.
    assert DMatrix(lazy.drop(["y", "w", "qid"])).num_row() == 400


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_frame_columns_name_the_metadata(library: str) -> None:
    full, df = labelled_frame()
    data: pl.DataFrame | pd.DataFrame
    labelled: pl.DataFrame | pd.DataFrame
    unlabelled: pl.DataFrame | pd.DataFrame
    if library == "pandas":
        data = df.assign(y=full["y"].to_numpy(), w=full["w"].to_numpy(), qid=full["qid"].to_numpy())
        labelled = data.drop(columns=["w", "qid"])
        unlabelled = labelled.drop(columns=["y"])
    else:
        data = full
        labelled = full.drop(["w", "qid"])
        unlabelled = labelled.drop("y")
    features = ["color", "size", "count", "flag"]
    matrix = DMatrix(data, label="y", weight="w", qid="qid")
    assert matrix.feature_names == features
    assert matrix.feature_types == ["c", "q", "q", "q"]
    np.testing.assert_array_equal(matrix.get_label(), full["y"].to_numpy().astype(np.float32))
    np.testing.assert_array_equal(matrix.get_weight(), full["w"].to_numpy().astype(np.float32))
    assert matrix.get_group().tolist() == [20] * 20
    # Several names make a label matrix or per-output margins; the bounds
    # come by name too. Columns not taken stay features.
    two = DMatrix(data, label=["y", "w"], base_margin=["w", "y"])
    assert two.feature_names == [*features, "qid"]
    assert two.get_label().shape == (400, 2)
    np.testing.assert_array_equal(two.get_label()[:, 1], two.get_base_margin()[:, 0])
    bounded = DMatrix(data, label_lower_bound="y", label_upper_bound="w")
    lower, upper = bounded.get_label_bounds()
    np.testing.assert_array_equal(lower, full["y"].to_numpy().astype(np.float32))
    np.testing.assert_array_equal(upper, full["w"].to_numpy().astype(np.float32))
    with pytest.raises(HessboostError, match="label column 'price' is not a column"):
        DMatrix(data, label="price")
    with pytest.raises(HessboostError, match="weight takes one column name"):
        DMatrix(data, label="y", weight=["w", "y"])
    with pytest.raises(HessboostError, match=r"group is not one value per row.*qid="):
        DMatrix(data, label="y", group="qid")
    with pytest.raises(TypeError, match="label column 'color' has dtype"):
        DMatrix(data, label="color")
    # Prediction entry points resolve names the same way, and refuse them
    # for data without columns.
    booster = hessboost.train({"max_depth": 3}, matrix, 10)
    from hessboost.conformal import SplitConformal

    conformal = SplitConformal.calibrate(booster, labelled, label="y", alpha=0.2)
    assert conformal.predict_interval(unlabelled).shape == (400, 2)
    with pytest.raises(TypeError, match="names a column, but data is a DMatrix"):
        SplitConformal.calibrate(booster, matrix, label="y", alpha=0.2)
    with pytest.raises(TypeError, match="names a column, but data is a ndarray"):
        DMatrix(np.zeros((4, 2)), label="y")


def test_polars_dtypes_are_read_as_numbers_or_refused_with_a_hint() -> None:
    full, _ = labelled_frame()
    features, y = full.drop(["y", "w", "qid"]), full["y"]
    params = {"max_depth": 3}
    expected = hessboost.train(params, DMatrix(features, label=y), 10).predict(features)
    # Null (nothing but missing) and Decimal columns are numeric.
    widened = features.with_columns(
        pl.lit(None).alias("empty"), pl.col("size").cast(pl.Decimal(12, 6)).alias("size")
    )
    matrix = DMatrix(widened, label=y)
    assert matrix.feature_types == ["c", "q", "q", "q", "q"]
    np.testing.assert_allclose(
        hessboost.train(params, matrix, 10).predict(widened), expected, rtol=1e-5
    )
    with pytest.raises(TypeError, match=r"column 'when' has dtype Datetime.*\.dt\.epoch\(\)"):
        DMatrix(features.with_columns(when=pl.datetime(2026, 10, 6)), label=y)
    if hasattr(pl, "Extension"):  # polars 2.0 loads unknown Arrow extension types as this
        extended = features.with_columns(pl.col("count").cast(pl.Extension("acme:count", pl.Int64)))
        if isinstance(extended.schema["count"], pl.Extension):  # 1.x casts keep the storage
            with pytest.raises(
                TypeError, match=r"column 'count' has dtype Extension.*\.ext\.storage"
            ):
                DMatrix(extended, label=y)
            storage = extended.with_columns(pl.col("count").ext.storage())
            np.testing.assert_array_equal(
                hessboost.train(params, DMatrix(storage, label=y), 10).predict(storage), expected
            )


def test_regex_looking_column_names_are_never_misread() -> None:
    # Columns are addressed by position: `pl.col("^a$")` would be a regex.
    odd = pl.DataFrame({"^a$": [1.0, 2.0, 3.0, 4.0], "b": [3.0, 4.0, 5.0, 6.0], "y": [0, 1, 0, 1]})
    if odd.select(pl.nth(0)).width == 0:  # polars 1.0 itself drops the column from a select
        with pytest.raises(HessboostError, match="reads as a regex"):
            DMatrix(odd, label="y")
        return
    matrix = DMatrix(odd, label="y")
    assert matrix.feature_names == ["^a$", "b"]
    booster = hessboost.train({"max_depth": 1}, matrix, 2)
    np.testing.assert_array_equal(
        booster.predict(odd[:, [0, 1]]), booster.predict(odd[:, [0, 1]].to_numpy())
    )


def test_estimators_collect_lazyframes_only_without_aligned_arrays() -> None:
    pytest.importorskip("sklearn")
    from hessboost.sklearn import HessboostRegressor

    full, _ = labelled_frame()
    features = full.select(["color", "size", "count", "flag"])
    y = full["y"].to_numpy()
    with pytest.raises(HessboostError, match="X is a polars LazyFrame while y is a separate array"):
        HessboostRegressor(n_estimators=10).fit(features.lazy(), y)
    model = HessboostRegressor(n_estimators=10).fit(features, y)
    with pytest.raises(HessboostError, match="X is a polars LazyFrame"):
        HessboostRegressor(n_estimators=10).fit(features, y, eval_set=[(features.lazy(), y)])
    np.testing.assert_array_equal(model.predict(features.lazy()), model.predict(features))
    # A base_margin array is aligned with X's rows too: it comes with a
    # DataFrame, not a LazyFrame. The booster reads one from a column.
    margin = full["w"].to_numpy()
    booster = model.get_booster()
    np.testing.assert_array_equal(
        model.predict(features, base_margin=margin), booster.predict(features, base_margin=margin)
    )
    with pytest.raises(HessboostError, match="X is a polars LazyFrame while base_margin"):
        model.predict(features.lazy(), base_margin=margin)
    np.testing.assert_array_equal(
        booster.predict(full.lazy().drop(["y", "qid"]), base_margin="w"),
        booster.predict(features, base_margin=margin),
    )
