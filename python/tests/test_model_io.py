"""Model I/O in every format, the Rust crate's saved models, pickling,
copying, slicing, and prediction layouts (SHAP, leaves)."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import pytest
from conftest import classes, regression, saved_models_dir

import hessboost
from hessboost import Booster, DMatrix, HessboostError, ModelFormatError


@pytest.fixture(scope="module")
def trained() -> tuple[Booster, np.ndarray]:
    x, y = regression()
    booster = hessboost.train({"max_depth": 4}, DMatrix(x, y, feature_names=list("abcde")), 12)
    return booster, x


@pytest.mark.parametrize(
    ("name", "format"),
    [
        ("model.bin", "binary"),
        ("model.hbm", "binary"),
        ("model.json", "json"),
        ("model.ubj", "xgboost-ubjson"),
    ],
)
def test_save_model_picks_the_format_by_extension(
    tmp_path: Path, trained: tuple[Booster, np.ndarray], name: str, format: hessboost.ModelFormat
) -> None:
    booster, x = trained
    path = tmp_path / name
    booster.save_model(path)
    assert path.read_bytes() == booster.save_raw(format)
    loaded = Booster(path)
    np.testing.assert_array_equal(loaded.predict(x), booster.predict(x))
    assert loaded.feature_names is None


@pytest.mark.parametrize("format", ["binary", "json", "xgboost-json", "xgboost-ubjson"])
def test_every_format_round_trips_predictions_and_is_detected(
    tmp_path: Path, trained: tuple[Booster, np.ndarray], format: hessboost.ModelFormat
) -> None:
    booster, x = trained
    raw = booster.save_raw(format)
    for loaded in [Booster(raw), Booster(bytearray(raw)), Booster(memoryview(raw))]:
        np.testing.assert_array_equal(loaded.predict(x), booster.predict(x))
        np.testing.assert_array_equal(
            loaded.predict(x, pred_contribs=True), booster.predict(x, pred_contribs=True)
        )
    explicit = Booster()
    explicit.load_model(raw, format=format)
    assert explicit.save_raw(format) == raw
    path = tmp_path / "model.any"
    booster.save_model(path, format=format)
    assert Booster(str(path)).save_raw(format) == raw


def test_xgboost_json_is_an_xgboost_document(trained: tuple[Booster, np.ndarray]) -> None:
    import json

    booster, _ = trained
    document = json.loads(booster.save_raw("xgboost-json"))
    assert document["version"][0] == 3
    assert document["learner"]["objective"]["name"] == "reg:squarederror"
    trees = document["learner"]["gradient_booster"]["model"]["trees"]
    assert len(trees) == 12


def test_formats_refuse_what_they_cannot_hold() -> None:
    x, y = regression(rows=100)
    linear = hessboost.train({"booster": "gblinear"}, DMatrix(x, y), 3)
    with pytest.raises(ModelFormatError):
        linear.save_raw("xgboost-json")
    np.testing.assert_array_equal(Booster(linear.save_raw()).predict(x), linear.predict(x))
    with pytest.raises(ValueError, match="unknown model format"):
        linear.save_raw("pickle")  # type: ignore[arg-type]


def test_corrupt_and_missing_models_raise(tmp_path: Path, trained: tuple[Booster, np.ndarray]) -> None:
    booster, _ = trained
    raw = bytearray(booster.save_raw())
    raw[len(raw) // 2] ^= 0xFF
    with pytest.raises(ModelFormatError):
        Booster(bytes(raw))
    with pytest.raises(ModelFormatError):
        Booster(b"")
    with pytest.raises(ModelFormatError):
        Booster(b'{"learner": {}}')
    with pytest.raises(ModelFormatError):
        Booster(b"{\x00\x00")
    with pytest.raises(FileNotFoundError):
        Booster(tmp_path / "absent.bin")
    empty = Booster()
    assert repr(empty) == "Booster(empty)"
    with pytest.raises(HessboostError, match="holds no model"):
        empty.predict(np.zeros((1, 5)))


def saved_features(rows: int = 160) -> np.ndarray:
    """``tests/common/mod.rs``'s ``four_features`` rows, in float32 as the
    Rust tests compute them."""
    i = np.arange(rows)

    def ratio(numerator: np.ndarray, denominator: int) -> np.ndarray:
        return numerator.astype(np.float32) / np.float32(denominator)

    d = ratio((i * 29) % 83, 83)
    d[i % 7 == 0] = np.nan
    return np.column_stack(
        [ratio((i * 37) % 101, 101), ratio((i * 53) % 97, 97), ratio((i * 11) % 89, 89), d]
    )


def test_models_saved_by_the_rust_crate_predict_its_recorded_margins() -> None:
    root = saved_models_dir()
    cases = sorted(root.glob("*/*.margins"))
    assert len(cases) >= 14
    for margins in cases:
        x = saved_features()
        if margins.stem == "categorical_splits":
            x[:, 0] = np.arange(160) % 5
        expected = np.frombuffer(margins.read_bytes(), dtype="<f4")
        for suffix in (".bin", ".json"):
            booster = Booster(margins.with_suffix(suffix))
            got = booster.predict(x, output_margin=True).reshape(-1)
            assert got.tobytes() == expected.tobytes(), f"{margins.stem}{suffix}"


def test_pickle_keeps_the_model_and_its_python_metadata(trained: tuple[Booster, np.ndarray]) -> None:
    booster, x = trained
    booster = booster.copy()
    booster.best_score = 1.5
    restored = pickle.loads(pickle.dumps(booster))
    assert type(restored) is Booster
    assert restored.feature_names == list("abcde")
    assert restored.best_score == 1.5
    assert restored.save_raw() == booster.save_raw()
    empty = pickle.loads(pickle.dumps(Booster()))
    assert repr(empty) == "Booster(empty)"


def test_native_bytes_round_trip_exactly(trained: tuple[Booster, np.ndarray]) -> None:
    booster, _ = trained
    raw = booster.save_raw()
    assert Booster(raw).save_raw() == raw
    text = booster.save_raw("json")
    assert Booster(text).save_raw("json") == text


def test_copies_are_independent(trained: tuple[Booster, np.ndarray]) -> None:
    booster, x = trained
    for duplicate in [booster.copy(), copy.copy(booster), copy.deepcopy(booster)]:
        duplicate.feature_names = ["p", "q", "r", "s", "t"]
        duplicate.load_model(booster[:2].save_raw())
        assert booster.feature_names == list("abcde")
        assert booster.num_boosted_rounds() == 12


def test_slicing_selects_iterations(trained: tuple[Booster, np.ndarray]) -> None:
    booster, x = trained
    assert booster[3:7].num_boosted_rounds() == 4
    assert booster[::3].num_boosted_rounds() == 4
    assert booster[-2:].num_boosted_rounds() == 2
    assert booster[5].num_boosted_rounds() == 1
    np.testing.assert_array_equal(
        booster[3:7].predict(x, output_margin=True),
        booster.predict(x, output_margin=True, iteration_range=(3, 7)),
    )
    np.testing.assert_array_equal(booster[-1].predict(x), booster[11].predict(x))
    assert booster[3:7].feature_names == list("abcde")
    with pytest.raises(IndexError):
        booster[12]
    with pytest.raises(HessboostError):
        booster[5:5]
    with pytest.raises(HessboostError, match="positive step"):
        booster[::-1]
    with pytest.raises(TypeError):
        booster["a"]  # type: ignore[call-overload]


def test_shap_contributions_sum_to_the_margin(trained: tuple[Booster, np.ndarray]) -> None:
    booster, x = trained
    contribs = booster.predict(x, pred_contribs=True)
    assert contribs.shape == (400, 6)
    margin = booster.predict(x, output_margin=True)
    np.testing.assert_allclose(contribs.sum(axis=1), margin, atol=1e-4)
    interactions = booster.predict(x[:20], pred_interactions=True)
    assert interactions.shape == (20, 6, 6)
    np.testing.assert_allclose(interactions.sum(axis=2), contribs[:20], atol=1e-4)
    early = booster.predict(x, pred_contribs=True, iteration_range=(0, 4))
    np.testing.assert_allclose(
        early.sum(axis=1), booster.predict(x, output_margin=True, iteration_range=(0, 4)), atol=1e-4
    )


def test_multiclass_layouts() -> None:
    x, y = classes(n_classes=3)
    booster = hessboost.train(
        {"objective": "multi:softprob", "num_class": 3, "num_parallel_tree": 2}, DMatrix(x, y), 4
    )
    contribs = booster.predict(x, pred_contribs=True)
    assert contribs.shape == (400, 3, 5)
    np.testing.assert_allclose(
        contribs.sum(axis=2), booster.predict(x, output_margin=True), atol=1e-4
    )
    assert booster.predict(x[:3], pred_interactions=True).shape == (3, 3, 5, 5)
    leaves = booster.predict(x, pred_leaf=True)
    assert leaves.shape == (400, 4 * 3 * 2)
    assert leaves.dtype == np.int32
    assert booster.predict(x, pred_leaf=True, iteration_range=(0, 1)).shape == (400, 6)
    with pytest.raises(HessboostError, match="exclusive"):
        booster.predict(x, pred_leaf=True, pred_contribs=True)
    assert repr(booster) == "Booster(objective='multi:softprob', rounds=4, features=4, outputs=3)"


def test_feature_importance(trained: tuple[Booster, np.ndarray]) -> None:
    booster, _ = trained
    weight = booster.get_score()
    assert set(weight) <= set("abcde") and "b" in weight
    assert all(isinstance(value, float) for value in weight.values())
    total = booster.get_score("total_gain")
    average = booster.get_score("gain")
    for name in weight:
        assert total[name] == pytest.approx(average[name] * weight[name])
    assert set(booster.get_score("cover")) == set(weight)
    unnamed = Booster(booster.save_raw())
    assert set(unnamed.get_score()) <= {"f0", "f1", "f2", "f3", "f4"}
    with pytest.raises(ValueError, match="importance_type"):
        booster.get_score("split")  # type: ignore[arg-type]
