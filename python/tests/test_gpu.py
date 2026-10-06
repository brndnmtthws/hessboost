"""GPU training and batch prediction: Metal (macOS) and wgpu (Vulkan, Metal,
DirectX 12). Tests that need a device skip without one; CI's Linux runners
install Mesa's lavapipe, a software Vulkan adapter, and set
``HESSBOOST_REQUIRE_WGPU``, so the wgpu tests run there without a GPU."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, TypeAlias

import numpy as np
import pytest

import hessboost
from conftest import regression
from hessboost import Booster, DMatrix, GpuModel, HessboostError, IncompatibleModelError

Device: TypeAlias = Literal["metal", "wgpu"]
DEVICES: list[Device] = ["metal", "wgpu"]


def needs(device: Device) -> None:
    """Skips the calling test when ``device`` cannot predict here (the guard
    below fails on a wgpu adapter that only trains)."""
    if not GpuModel.available(device):
        pytest.skip(f"no {device} GPU here")


@pytest.fixture(scope="module")
def trained() -> tuple[Booster, np.ndarray]:
    x, y = regression()
    booster = hessboost.train({"max_depth": 4}, DMatrix(x, y), 12, verbose_eval=False)
    return booster, x


def test_wgpu_is_available_or_the_machine_has_no_adapter(
    trained: tuple[Booster, np.ndarray],
) -> None:
    """A kernel-compile failure, or an adapter that fails the addition-order
    probe (it trains but cannot predict), is never a reason to skip the wgpu
    tests: without this guard they would pass vacuously while the backend is
    broken. ``HESSBOOST_REQUIRE_WGPU`` turns a missing adapter into a
    failure too. Without an adapter, ``device="wgpu"`` training is refused
    rather than run on the CPU."""
    if GpuModel.available("wgpu"):
        return
    with pytest.raises(HessboostError) as refused:
        trained[0].to_gpu("wgpu")
    reason = str(refused.value)
    assert "HESSBOOST_REQUIRE_WGPU" not in os.environ, (
        f"HESSBOOST_REQUIRE_WGPU is set but wgpu prediction is unavailable: {reason}"
    )
    assert reason.startswith("GPU backend error: no wgpu adapter"), (
        f"wgpu prediction is unavailable: {reason}"
    )
    assert GpuModel.device_name("wgpu") is None
    x, y = regression(rows=100)
    with pytest.raises(HessboostError, match="no wgpu adapter"):
        hessboost.train({"device": "wgpu"}, DMatrix(x, y), 1)


@pytest.mark.parametrize("device", DEVICES)
def test_a_device_that_predicts_has_a_name(device: Device) -> None:
    needs(device)
    assert GpuModel.device_name(device)


def test_default_device_is_metal_on_macos_and_wgpu_elsewhere(
    trained: tuple[Booster, np.ndarray],
) -> None:
    default: Device = "metal" if sys.platform == "darwin" else "wgpu"
    assert GpuModel.available() == GpuModel.available(default)
    assert GpuModel.device_name() == GpuModel.device_name(default)
    if GpuModel.available():
        assert trained[0].to_gpu().device == default


def test_unknown_devices_are_refused(trained: tuple[Booster, np.ndarray]) -> None:
    with pytest.raises(HessboostError, match='unknown GPU device "cuda"'):
        trained[0].to_gpu("cuda")  # ty: ignore[invalid-argument-type]
    with pytest.raises(HessboostError, match='unknown GPU device "cpu"'):
        GpuModel.available("cpu")  # ty: ignore[invalid-argument-type]
    with pytest.raises(HessboostError, match='unknown GPU device "Metal"'):
        GpuModel.device_name("Metal")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="to_gpu"):
        GpuModel()


@pytest.mark.skipif(sys.platform == "darwin", reason="Metal is compiled into the macOS wheels")
def test_metal_is_refused_off_macos(trained: tuple[Booster, np.ndarray]) -> None:
    assert not GpuModel.available("metal")
    assert GpuModel.device_name("metal") is None
    with pytest.raises(HessboostError, match="macOS"):
        trained[0].to_gpu("metal")


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_predicts_bit_identically(device: Device, trained: tuple[Booster, np.ndarray]) -> None:
    needs(device)
    booster, x = trained
    gpu = booster.to_gpu(device)
    assert gpu.device == device
    assert gpu.booster is booster
    np.testing.assert_array_equal(gpu.predict(x), booster.predict(x))
    np.testing.assert_array_equal(
        gpu.predict(x, output_margin=True), booster.predict(x, output_margin=True)
    )
    half = booster.num_boosted_rounds() // 2
    np.testing.assert_array_equal(
        gpu.predict(x, iteration_range=(0, half)),
        booster.predict(x, iteration_range=(0, half)),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_predict_matches_cpu_on_every_objective(device: Device) -> None:
    needs(device)
    x, y = regression()
    base = {"max_depth": 3, "seed": 1}
    cases = [
        ({"objective": "reg:squarederror"}, y),
        ({"objective": "binary:logistic"}, (y > y.mean()).astype(int)),
        ({"objective": "multi:softprob", "num_class": 3}, (y * 3 % 3).astype(int)),
    ]
    for params, labels in cases:
        booster = hessboost.train({**base, **params}, DMatrix(x, labels), 5)
        gpu = booster.to_gpu(device)
        np.testing.assert_array_equal(gpu.predict(x), booster.predict(x))
        np.testing.assert_array_equal(
            gpu.predict(x, output_margin=True), booster.predict(x, output_margin=True)
        )


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_predicts_from_concurrent_threads(
    device: Device, trained: tuple[Booster, np.ndarray]
) -> None:
    """Predictions run without the GIL, so threads share one GPU model."""
    needs(device)
    booster, x = trained
    gpu = booster.to_gpu(device)
    expected = booster.predict(x)
    with ThreadPoolExecutor(4) as pool:
        for predictions in pool.map(lambda _: gpu.predict(x), range(16)):
            np.testing.assert_array_equal(predictions, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_training_reproduces_cpu_training(device: Device) -> None:
    """Nodes of at least 8,192 rows build their histograms on the GPU; the
    model is the CPU's bit for bit."""
    needs(device)
    x, y = regression(rows=20_000, features=8)
    dtrain = DMatrix(x, y)
    params = {"tree_method": "hist", "max_depth": 5, "seed": 1}
    cpu = hessboost.train(params, dtrain, 5)
    gpu = hessboost.train({**params, "device": device}, dtrain, 5)
    assert gpu.save_raw() == cpu.save_raw()


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_refuses_models_without_a_tree_forest(device: Device) -> None:
    x, y = regression(rows=100)
    # Without a GPU, the device's own refusal comes first.
    error = IncompatibleModelError if GpuModel.available(device) else HessboostError
    for params in ({"booster": "gblinear"}, {"linear_tree": True}):
        model = hessboost.train(params, DMatrix(x, y), 3, verbose_eval=False)
        with pytest.raises(error):
            model.to_gpu(device)
