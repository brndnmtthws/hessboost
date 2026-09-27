"""Conditional diffusion and flow matching (``hessboost.diffusion``)."""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import hessboost
from conftest import frame, reorder_colors
from hessboost import DMatrix, HessboostError, ModelFormatError
from hessboost.diffusion import (
    DiffusionModel,
    DiffusionParams,
    EarlyStopping,
    Edm,
    FlowMatching,
    LogNoiseNormal,
    Residualizer,
    Score,
    SubVariancePreserving,
    VarianceExploding,
    VariancePreserving,
    crps,
    mean,
    quantiles,
)


def bimodal(rows: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``y = ±(1 + x) + noise``: two modes whose gap grows with ``x``."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(rows, 1))
    sign = np.where(rng.random(rows) < 0.5, -1.0, 1.0)
    return x, sign * (1.0 + x[:, 0]) + rng.normal(0.0, 0.05, rows)


def quick(params: DiffusionParams, **changes: Any) -> DiffusionParams:
    """A small, fast variant of ``params``."""
    return dataclasses.replace(params, n_repeats=8, num_boost_round=60, **changes)


@pytest.fixture(scope="module")
def fitted() -> tuple[DiffusionModel, NDArray[np.float64]]:
    x, y = bimodal(400, 0)
    return DiffusionModel.fit(quick(DiffusionParams.flow_matching()), x, y), x


def test_constructor_defaults_are_the_default_preset() -> None:
    params = DiffusionParams()
    assert params == DiffusionParams.default()
    assert params.method == Score()
    assert DiffusionParams.flow_matching().method == FlowMatching()
    assert DiffusionParams.flow_matching().n_steps == 5
    treeffuser = DiffusionParams.treeffuser()
    assert treeffuser.method == Score(
        parameterization="noise", noise_level_feature=False, time_sampling="uniform"
    )
    assert treeffuser.residualizer is None
    assert params.early_stopping == EarlyStopping(rounds=50, eval_fraction=0.1)
    # LightGBM's defaults, as XGBoost parameters hessboost.train would read.
    assert params.training["grow_policy"] == "lossguide"
    assert params.training["max_leaves"] == 31
    assert Residualizer().training["eta"] == 0.05


def test_samples_recover_both_modes(fitted: tuple[DiffusionModel, NDArray[np.float64]]) -> None:
    model, _ = fitted
    assert (model.n_features, model.n_outputs, model.is_residualized) == (1, 1, True)
    draws = model.sample(np.array([[0.2], [0.8]]), 300, seed=1)
    assert draws.shape == (2, 300, 1)
    assert draws.dtype == np.float32
    for row, x in enumerate([0.2, 0.8]):
        values = draws[row, :, 0]
        upper = np.sum(np.abs(values - (1 + x)) < 0.5)
        lower = np.sum(np.abs(values + (1 + x)) < 0.5)
        assert upper > 70, (upper, lower)
        assert lower > 70, (upper, lower)
        assert upper + lower > 220


def test_vector_labels_are_sampled_jointly() -> None:
    rng = np.random.default_rng(3)
    u = np.where(rng.random(300) < 0.5, -1.0, 1.0)
    x = rng.uniform(size=(300, 2))
    params = quick(DiffusionParams.treeffuser(), early_stopping=None)
    model = DiffusionModel.fit(params, DMatrix(x, np.column_stack([u, -u])))
    assert model.n_outputs == 2
    draws = model.sample(x[:3], 100, seed=0)
    assert draws.shape == (3, 100, 2)
    together = np.abs(draws[..., 0] + draws[..., 1]) < 0.5
    assert together.mean() > 0.8
    assert quantiles(draws, [0.1, 0.5, 0.9]).shape == (3, 3, 2)
    assert mean(draws).shape == crps(draws, np.column_stack([u, -u])[:3]).shape == (3, 2)


def test_draws_are_deterministic_per_seed(
    fitted: tuple[DiffusionModel, NDArray[np.float64]],
) -> None:
    model, x = fitted
    draws = model.sample(x[:5], 30, seed=7)
    np.testing.assert_array_equal(model.sample(x[:5], 30, seed=7), draws)
    # Fewer samples are a prefix of each row's draws; other seeds differ.
    np.testing.assert_array_equal(model.sample(x[:5], 10, seed=7), draws[:, :10])
    assert not np.array_equal(model.sample(x[:5], 30, seed=8), draws)
    # A DMatrix reads the same features.
    np.testing.assert_array_equal(model.sample(DMatrix(x[:5]), 30, seed=7), draws)


def test_fitting_is_deterministic() -> None:
    x, y = bimodal(200, 4)
    params = quick(DiffusionParams.treeffuser(), seed=3)
    first = DiffusionModel.fit(params, x, y).to_bytes()
    assert DiffusionModel.fit(params, x, y).to_bytes() == first
    assert DiffusionModel.fit(dataclasses.replace(params, seed=4), x, y).to_bytes() != first


def test_every_format_round_trips_bit_for_bit(
    fitted: tuple[DiffusionModel, NDArray[np.float64]], tmp_path: Path
) -> None:
    model, x = fitted
    draws = model.sample(x[:4], 25, seed=2)
    binary, text = model.to_bytes(), model.to_json()
    model.save_binary(tmp_path / "model.hbdm")
    model.save_json(tmp_path / "model.json")
    restored = [
        DiffusionModel.from_bytes(binary),
        DiffusionModel.from_json(text),
        DiffusionModel.load_binary(tmp_path / "model.hbdm"),
        DiffusionModel.load_json(tmp_path / "model.json"),
        pickle.loads(pickle.dumps(model)),
    ]
    for other in restored:
        np.testing.assert_array_equal(other.sample(x[:4], 25, seed=2), draws)
        assert other.to_bytes() == binary
        assert other.method == model.method
        assert other.n_steps == model.n_steps


def test_n_steps_swaps_in_a_copy(fitted: tuple[DiffusionModel, NDArray[np.float64]]) -> None:
    model, x = fitted
    copy = pickle.loads(pickle.dumps(model))
    copy.n_steps = 12
    assert (copy.n_steps, model.n_steps) == (12, 5)
    assert not np.array_equal(copy.sample(x[:2], 10), model.sample(x[:2], 10))
    assert DiffusionModel.from_bytes(copy.to_bytes()).n_steps == 12
    with pytest.raises(HessboostError, match="n_steps"):
        copy.n_steps = 0


def test_summaries_match_numpy(fitted: tuple[DiffusionModel, NDArray[np.float64]]) -> None:
    model, x = fitted
    draws = model.sample(x[:6], 50, seed=0)
    np.testing.assert_allclose(mean(draws), draws.astype(np.float64).mean(axis=1), rtol=1e-12)
    levels = [0.0, 0.1, 0.5, 0.95, 1.0]
    expected = np.quantile(draws.astype(np.float64), levels, axis=1).transpose(1, 0, 2)
    np.testing.assert_allclose(quantiles(draws, levels), expected, rtol=1e-12)
    y = np.linspace(-2, 2, 6, dtype=np.float32)
    d = draws[:, :, 0].astype(np.float64)
    spread = np.abs(d - y[:, None].astype(np.float64)).mean(axis=1)
    pairwise = np.abs(d[:, :, None] - d[:, None, :]).mean(axis=(1, 2))
    np.testing.assert_allclose(crps(draws, y)[:, 0], spread - pairwise / 2, rtol=1e-9)


def test_frames_are_recoded_to_the_training_categories() -> None:
    df, y = frame(300, 1)
    params = quick(DiffusionParams.treeffuser(), early_stopping=None)
    model = DiffusionModel.fit(params, df, y)
    assert model.feature_names == list(df.columns)
    draws = model.sample(df.iloc[:5], 20, seed=1)
    np.testing.assert_array_equal(model.sample(reorder_colors(df).iloc[:5], 20, seed=1), draws)
    renamed = df.rename(columns={"size": "width"})
    with pytest.raises(HessboostError, match="feature names differ"):
        model.sample(renamed, 5)
    # Pickles keep the schema.
    assert pickle.loads(pickle.dumps(model)).feature_names == list(df.columns)


def test_invalid_configurations_are_refused() -> None:
    with pytest.raises(HessboostError, match="n_repeats"):
        DiffusionParams(n_repeats=0)
    with pytest.raises(HessboostError, match="sde"):
        DiffusionParams(method=Score(sde=VarianceExploding(sigma_min=1.0, sigma_max=0.5)))
    with pytest.raises(HessboostError, match="sde"):
        DiffusionParams(method=Score(sde=SubVariancePreserving(beta_min=-1.0)))
    with pytest.raises(HessboostError, match="eval_fraction"):
        DiffusionParams(early_stopping=EarlyStopping(eval_fraction=1.5))
    with pytest.raises(HessboostError, match="folds"):
        DiffusionParams(residualizer=Residualizer(folds=1))
    with pytest.raises(HessboostError, match="max_dept"):
        DiffusionParams(training={"max_dept": 3})
    with pytest.raises(HessboostError, match="squarederror"):
        DiffusionParams(training={"objective": "reg:absoluteerror"})
    with pytest.raises(HessboostError, match="parameterization"):
        Score(parameterization="edm")  # ty: ignore[invalid-argument-type]
    with pytest.raises(HessboostError, match="solver"):
        FlowMatching(solver="rk4")  # ty: ignore[invalid-argument-type]
    with pytest.raises(HessboostError, match="n_steps"):
        DiffusionParams(n_steps=-1)
    # A training mapping from scratch starts at XGBoost's defaults.
    assert DiffusionParams(training={"max_depth": 3}).training == {"max_depth": 3}
    assert Edm(sigma_data=2) == Edm(sigma_data=2.0)
    assert LogNoiseNormal(mean=0).mean == 0.0
    assert FlowMatching(path=VariancePreserving(beta_max=10)).path != FlowMatching().path


def test_wrong_types_raise_type_error() -> None:
    with pytest.raises(TypeError, match="method"):
        DiffusionParams(method="score")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="n_steps"):
        DiffusionParams(n_steps="5")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="n_repeats"):
        DiffusionParams(n_repeats=True)
    with pytest.raises(TypeError, match="training"):
        DiffusionParams(training=[("eta", 0.1)])  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="sde"):
        Score(sde="variance_exploding")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="sigma_min"):
        VarianceExploding(sigma_min="0.01")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="early_stopping"):
        DiffusionParams(early_stopping=(50, 0.1))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="DiffusionParams"):
        DiffusionModel.fit({"n_steps": 5}, *bimodal(10, 0))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        DiffusionModel()
    with pytest.raises(TypeError):
        DiffusionModel.from_json(b"{}")  # ty: ignore[invalid-argument-type]


def test_unsupported_data_is_refused(fitted: tuple[DiffusionModel, NDArray[np.float64]]) -> None:
    model, x = fitted
    x_train, y_train = bimodal(100, 5)
    params = quick(DiffusionParams.treeffuser())
    with pytest.raises(HessboostError, match="labels"):
        DiffusionModel.fit(params, x_train)
    with pytest.raises(HessboostError, match="weights"):
        DiffusionModel.fit(params, DMatrix(x_train, y_train, weight=np.ones(100)))
    with pytest.raises(HessboostError, match="80"):
        DiffusionModel.fit(quick(DiffusionParams()), x_train[:50], y_train[:50])
    with pytest.raises(HessboostError, match="feature count"):
        model.sample(np.ones((3, 2)), 5)
    with pytest.raises(HessboostError, match="n_samples"):
        model.sample(x[:3], 0)
    with pytest.raises(HessboostError, match="base margins"):
        model.sample(DMatrix(x[:3], base_margin=np.zeros(3)), 5)
    draws = model.sample(x[:3], 5)
    with pytest.raises(HessboostError, match="levels"):
        quantiles(draws, [1.5])
    with pytest.raises(HessboostError, match="y must be"):
        crps(draws, np.zeros(4))
    with pytest.raises(HessboostError, match="samples"):
        mean(draws[:, :, 0])
    with pytest.raises(HessboostError, match="finite"):
        mean(np.full((1, 2, 1), np.nan))


def test_summaries_of_no_rows_are_empty() -> None:
    # Zero rows leave the sample count unbounded by the array's size: the
    # summaries must not allocate by it.
    empty = np.empty((0, 2**40, 1), dtype=np.float32)
    assert mean(empty).shape == (0, 1)
    assert quantiles(empty, [0.1, 0.9]).shape == (0, 2, 1)
    assert crps(empty, np.empty((0, 1))).shape == (0, 1)


def test_damaged_models_are_refused(
    fitted: tuple[DiffusionModel, NDArray[np.float64]], tmp_path: Path
) -> None:
    model, _ = fitted
    binary = model.to_bytes()
    with pytest.raises(ModelFormatError):
        DiffusionModel.from_bytes(binary[:-8])
    with pytest.raises(ModelFormatError):
        DiffusionModel.from_json(model.to_json()[:-10])
    with pytest.raises(ModelFormatError):
        DiffusionModel.from_bytes(hessboost.train({}, DMatrix(*bimodal(20, 0)), 2).save_raw())
    with pytest.raises(FileNotFoundError):
        DiffusionModel.load_binary(tmp_path / "missing.hbdm")
