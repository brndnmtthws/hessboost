"""The compact (``HBTD``) model: bit-identical predictions to its booster,
persistence, the feature schema it keeps, and its refusals."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hessboost
from conftest import classes, regression
from hessboost import CompactModel, DMatrix, HessboostError, ModelFormatError


def test_compact_predicts_what_the_booster_predicts(tmp_path: Path) -> None:
    x, y = regression()
    dtrain = DMatrix(x[:300], y[:300], feature_names=list("abcde"))
    dvalid = DMatrix(x[300:], y[300:], feature_names=list("abcde"))
    booster = hessboost.train(
        {"max_depth": 4, "eta": 0.8, "toad_penalty_feature": 1.0, "toad_penalty_threshold": 0.5},
        dtrain,
        200,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=3,
        verbose_eval=False,
    )
    assert booster.best_iteration is not None
    compact = booster.to_compact()
    # Only the trees default prediction uses (through best_iteration).
    assert compact.num_trees() == booster.best_iteration + 1
    np.testing.assert_array_equal(compact.predict(x), booster.predict(x))
    np.testing.assert_array_equal(
        compact.predict(x, output_margin=True), booster.predict(x, output_margin=True)
    )
    assert compact.feature_names == list("abcde")
    assert compact.num_features() == 5
    assert set(compact.used_features) <= set(range(5))

    report = booster.size_report()
    assert report.compact_bytes == compact.size_bytes == len(compact.save_raw())
    assert report.compact_bytes < report.native_bytes
    assert report.trees == compact.num_trees()
    assert report.compression_ratio > 1

    path = tmp_path / "model.hbtd"
    compact.save_model(path)
    loaded = CompactModel(path)
    assert loaded.feature_names is None
    np.testing.assert_array_equal(loaded.predict(x), booster.predict(x))
    assert CompactModel(compact.save_raw()).save_raw() == compact.save_raw()
    restored = pickle.loads(pickle.dumps(compact))
    assert restored.feature_names == list("abcde")
    np.testing.assert_array_equal(restored.predict(x), booster.predict(x))


def test_compact_keeps_the_boosters_categories() -> None:
    rng = np.random.default_rng(1)
    colors = pd.Categorical(rng.choice(["red", "green", "blue"], 300))
    frame = pd.DataFrame({"color": colors, "x": rng.normal(size=300)})
    y = (frame["color"] == "red").to_numpy(float) + 0.1 * frame["x"].to_numpy()
    booster = hessboost.train({"max_depth": 3}, DMatrix(frame, y), 10)
    compact = booster.to_compact()
    # The same values in another category order are re-coded to the model's.
    reordered = frame.assign(color=frame["color"].cat.reorder_categories(["red", "green", "blue"]))
    np.testing.assert_array_equal(compact.predict(reordered), booster.predict(frame))
    with pytest.raises(HessboostError, match="feature names differ"):
        compact.predict(frame.rename(columns={"x": "z"}))


def test_multiclass_compact_and_refusals() -> None:
    x, y = classes(n_classes=3)
    booster = hessboost.train({"objective": "multi:softprob", "num_class": 3}, DMatrix(x, y), 5)
    compact = booster.to_compact()
    assert compact.num_outputs == 3
    np.testing.assert_array_equal(compact.predict(x), booster.predict(x))
    linear = hessboost.train({"booster": "gblinear"}, DMatrix(x, y), 3)
    with pytest.raises(HessboostError):
        linear.to_compact()
    with pytest.raises(ModelFormatError):
        CompactModel(booster.save_raw())
