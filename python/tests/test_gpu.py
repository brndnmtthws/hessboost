"""GPU batch prediction on Metal (macOS only)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import hessboost
from conftest import regression
from hessboost import Booster, DMatrix, GpuModel, HessboostError

NEEDS_GPU = pytest.mark.skipif(
    not GpuModel.available(),
    reason="no Metal device (off macOS, or a machine with no GPU)",
)


@pytest.fixture(scope="module")
def trained() -> tuple[Booster, np.ndarray]:
    x, y = regression()
    booster = hessboost.train({"max_depth": 4}, DMatrix(x, y), 12, verbose_eval=False)
    return booster, x


def test_available_reports_a_device_name(trained: tuple[Booster, np.ndarray]) -> None:
    assert GpuModel.available() == (GpuModel.device_name() is not None)
    if sys.platform == "darwin" and GpuModel.available():
        assert GpuModel.device_name()
    trained[0].to_gpu()


@NEEDS_GPU
def test_gpu_predicts_bit_identically(trained: tuple[Booster, np.ndarray]) -> None:
    booster, x = trained
    gpu = booster.to_gpu()
    assert isinstance(gpu.booster, Booster)
    np.testing.assert_array_equal(gpu.predict(x), booster.predict(x))
    np.testing.assert_array_equal(
        gpu.predict(x, output_margin=True), booster.predict(x, output_margin=True)
    )
    half = booster.num_boosted_rounds() // 2
    np.testing.assert_array_equal(
        gpu.predict(x, iteration_range=(0, half)),
        booster.predict(x, iteration_range=(0, half)),
    )
    with pytest.raises(TypeError):
        GpuModel()


@NEEDS_GPU
def test_gpu_predict_matches_cpu_on_every_objective() -> None:
    x, y = regression()
    base = {"max_depth": 3, "seed": 1}
    cases = [
        ({"objective": "reg:squarederror"}, y),
        ({"objective": "binary:logistic"}, (y > y.mean()).astype(int)),
        ({"objective": "multi:softprob", "num_class": 3}, (y * 3 % 3).astype(int)),
    ]
    for params, labels in cases:
        booster = hessboost.train({**base, **params}, DMatrix(x, labels), 5)
        gpu = booster.to_gpu()
        np.testing.assert_array_equal(gpu.predict(x), booster.predict(x))
        np.testing.assert_array_equal(
            gpu.predict(x, output_margin=True), booster.predict(x, output_margin=True)
        )


@NEEDS_GPU
def test_gpu_refuses_unsupported_models() -> None:
    x, y = regression(rows=100)
    linear = hessboost.train({"booster": "gblinear"}, DMatrix(x, y), 3, verbose_eval=False)
    with pytest.raises(HessboostError):
        linear.to_gpu()
    tree = hessboost.train({"linear_tree": True}, DMatrix(x, y), 3, verbose_eval=False)
    with pytest.raises(HessboostError):
        tree.to_gpu()
