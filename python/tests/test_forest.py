"""ForestFlow and ForestDiffusion (``hessboost.diffusion.forest``)."""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from hessboost import DMatrix, HessboostError, ModelFormatError
from hessboost.diffusion import DiffusionModel, DiffusionParams
from hessboost.diffusion.forest import Diffusion, ForestModel, ForestParams, Repaint

KINDS: list[Any] = ["continuous", "integer", "categorical"]


def table(rows: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A continuous column, a binary one that follows it, a three-category
    one, and a class label."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=rows)
    b = (a + rng.normal(0, 0.3, rows) > 0).astype(np.float64)
    c = rng.choice([2.0, 5.0, 7.0], size=rows)
    return np.column_stack([a, b, c]), (rng.random(rows) < 0.4).astype(np.float64)


def quick(params: ForestParams, **changes: Any) -> ForestParams:
    return dataclasses.replace(
        params, n_t=6, duplicate_k=6, num_boost_round=15, column_kinds=KINDS, **changes
    )


@pytest.fixture(scope="module")
def conditional() -> tuple[ForestModel, NDArray[np.float64], NDArray[np.float64]]:
    x, y = table(200, 0)
    return ForestModel.fit(quick(ForestParams.diffusion()), x, y), x, y


def test_constructor_defaults_are_the_default_preset() -> None:
    assert ForestParams() == ForestParams.default()
    assert ForestParams().method == "flow"
    assert ForestParams.diffusion().method == Diffusion(beta_min=0.1, beta_max=8.0)
    training = ForestParams().training
    assert (training["max_depth"], training["eta"], training["lambda"]) == (7, 0.3, 0.0)
    assert (ForestParams().n_t, ForestParams().duplicate_k) == (50, 100)


def test_generated_rows_follow_the_column_kinds() -> None:
    x, _ = table(200, 1)
    model = ForestModel.fit(quick(ForestParams()), x)
    assert (model.method, model.n_t, model.n_columns) == ("flow", 6, 3)
    assert model.classes.size == 0
    values, labels = model.generate(300, seed=2)
    assert values.shape == (300, 3)
    assert values.dtype == np.float32
    assert labels is None
    assert set(np.unique(values[:, 1])) <= {0.0, 1.0}
    assert set(np.unique(values[:, 2])) <= {2.0, 5.0, 7.0}
    assert x[:, 0].min() <= values[:, 0].min() and values[:, 0].max() <= x[:, 0].max()
    # The binary column follows the continuous one, as in the data.
    assert np.corrcoef(values[:, 0], values[:, 1])[0, 1] > 0.3


def test_class_conditional_generation(
    conditional: tuple[ForestModel, NDArray[np.float64], NDArray[np.float64]],
) -> None:
    model, _, _ = conditional
    np.testing.assert_array_equal(model.classes, [0.0, 1.0])
    values, labels = model.generate(50, seed=0)
    assert labels is not None and labels.shape == (50,)
    assert set(np.unique(labels)) <= {0.0, 1.0}
    rows = model.generate_for_labels([1, 0, 1], seed=4)
    assert rows.shape == (3, 3)
    np.testing.assert_array_equal(model.generate_for_labels([1, 0, 1], seed=4), rows)


def test_draws_are_deterministic_per_seed(
    conditional: tuple[ForestModel, NDArray[np.float64], NDArray[np.float64]],
) -> None:
    model, _, _ = conditional
    values, labels = model.generate(40, seed=3)
    again, again_labels = model.generate(40, seed=3)
    np.testing.assert_array_equal(again, values)
    np.testing.assert_array_equal(again_labels, labels)
    assert not np.array_equal(model.generate(40, seed=4)[0], values)
    x, y = table(200, 0)
    params = quick(ForestParams.diffusion())
    assert ForestModel.fit(params, x, y).to_bytes() == model.to_bytes()


def test_imputation_keeps_observed_entries(
    conditional: tuple[ForestModel, NDArray[np.float64], NDArray[np.float64]],
) -> None:
    model, x, y = conditional
    holes = x[:8].copy()
    holes[0, 0] = holes[1, 2] = holes[2, 1] = np.nan
    missing = np.isnan(holes)
    filled = model.impute(holes, y[:8], n_imputations=4, seed=1)
    assert filled.shape == (4, 8, 3)
    assert not np.isnan(filled).any()
    for imputation in filled:
        np.testing.assert_allclose(imputation[~missing], holes[~missing].astype(np.float32))
    assert set(np.unique(filled[:, 1, 2])) <= {2.0, 5.0, 7.0}
    assert set(np.unique(filled[:, 2, 1])) <= {0.0, 1.0}
    # A labelled DMatrix, RePaint, and determinism per seed.
    repainted = model.impute(
        DMatrix(holes, y[:8]), n_imputations=2, repaint=Repaint(resample=2, jump=0.5), seed=1
    )
    np.testing.assert_array_equal(
        model.impute(holes, y[:8], n_imputations=2, repaint=Repaint(2, 0.5), seed=1), repainted
    )
    np.testing.assert_array_equal(
        model.impute(holes, y[:8], n_imputations=4, seed=1), filled
    )


def test_every_format_round_trips_bit_for_bit(
    conditional: tuple[ForestModel, NDArray[np.float64], NDArray[np.float64]], tmp_path: Path
) -> None:
    model, x, y = conditional
    values, _ = model.generate(30, seed=5)
    holes = x[:4].copy()
    holes[:, 0] = np.nan
    filled = model.impute(holes, y[:4], n_imputations=2, seed=5)
    binary = model.to_bytes()
    model.save_binary(tmp_path / "model.hbff")
    model.save_json(tmp_path / "model.json")
    restored = [
        ForestModel.from_bytes(binary),
        ForestModel.from_json(model.to_json()),
        ForestModel.load_binary(tmp_path / "model.hbff"),
        ForestModel.load_json(tmp_path / "model.json"),
        pickle.loads(pickle.dumps(model)),
    ]
    for other in restored:
        np.testing.assert_array_equal(other.generate(30, seed=5)[0], values)
        np.testing.assert_array_equal(other.impute(holes, y[:4], n_imputations=2, seed=5), filled)
        assert other.to_bytes() == binary
        assert other.method == model.method


def test_invalid_configurations_are_refused() -> None:
    with pytest.raises(HessboostError, match="n_t"):
        ForestParams(n_t=1)
    with pytest.raises(HessboostError, match="duplicate_k"):
        ForestParams(duplicate_k=0)
    with pytest.raises(HessboostError, match="method"):
        ForestParams(method=Diffusion(beta_min=2.0, beta_max=1.0))
    with pytest.raises(HessboostError, match="column kind"):
        ForestParams(column_kinds=["ordinal"])  # type: ignore[list-item]
    with pytest.raises(HessboostError, match="squarederror"):
        ForestParams(training={"objective": "reg:absoluteerror"})
    with pytest.raises(HessboostError, match="method"):
        ForestParams(method="diffusion")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="method"):
        ForestParams(method=DiffusionParams())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="column_kinds"):
        ForestParams(column_kinds="continuous")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="n_t"):
        ForestParams(n_t=2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="beta_max"):
        Diffusion(beta_max="8")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ForestParams"):
        ForestModel.fit(DiffusionParams(), np.ones((4, 2)))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ForestModel()


def test_unsupported_requests_are_refused(
    conditional: tuple[ForestModel, NDArray[np.float64], NDArray[np.float64]],
) -> None:
    model, x, y = conditional
    params = quick(ForestParams())
    with pytest.raises(HessboostError, match="column_kinds"):
        ForestModel.fit(params, x[:, :2])
    with pytest.raises(HessboostError, match="weights"):
        ForestModel.fit(params, DMatrix(x, weight=np.ones(len(x))))
    flow = ForestModel.fit(params, x)
    with pytest.raises(HessboostError, match="Diffusion"):
        flow.impute(x[:3])
    with pytest.raises(HessboostError, match="labels"):
        flow.generate_for_labels([0.0])
    with pytest.raises(HessboostError, match="n_rows"):
        model.generate(0)
    with pytest.raises(HessboostError, match="label"):
        model.generate_for_labels([3.0])
    with pytest.raises(HessboostError, match="labels"):
        model.impute(x[:3])
    with pytest.raises(HessboostError, match="column count"):
        model.impute(x[:3, :2], y[:3])
    with pytest.raises(HessboostError, match="n_imputations"):
        model.impute(x[:3], y[:3], n_imputations=0)
    with pytest.raises(HessboostError, match="jump"):
        model.impute(x[:3], y[:3], repaint=Repaint(jump=2.0))
    with pytest.raises(TypeError, match="repaint"):
        model.impute(x[:3], y[:3], repaint=(5, 0.1))  # type: ignore[arg-type]


def test_imputation_refuses_label_matrices(
    conditional: tuple[ForestModel, NDArray[np.float64], NDArray[np.float64]],
) -> None:
    # Two columns of valid classes, which read flat would misassign rows.
    model, x, y = conditional
    two_columns = np.column_stack([y[:3], 1.0 - y[:3]])
    with pytest.raises(HessboostError, match="label matrices"):
        model.impute(x[:3], two_columns)
    with pytest.raises(HessboostError, match="label matrices"):
        model.impute(DMatrix(x[:3], label=two_columns))


def test_damaged_models_are_refused(
    conditional: tuple[ForestModel, NDArray[np.float64], NDArray[np.float64]], tmp_path: Path
) -> None:
    model, x, y = conditional
    with pytest.raises(ModelFormatError):
        ForestModel.from_bytes(model.to_bytes()[:-8])
    with pytest.raises(ModelFormatError):
        ForestModel.from_json(model.to_json()[:-10])
    diffusion = DiffusionModel.fit(
        dataclasses.replace(DiffusionParams.treeffuser(), n_repeats=2, num_boost_round=3),
        x,
        y,
    )
    with pytest.raises(ModelFormatError):
        ForestModel.from_bytes(diffusion.to_bytes())
    with pytest.raises(OSError):
        ForestModel.load_json(tmp_path / "missing.json")
