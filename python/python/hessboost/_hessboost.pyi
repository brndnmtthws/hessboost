"""Native extension module. Import from :mod:`hessboost` instead."""

from collections.abc import Mapping
from typing import Any, final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Booster",
    "ConformalizedQuantile",
    "DMatrix",
    "Distributions",
    "Params",
    "SplitConformal",
    "cv",
    "forward_chaining",
    "k_fold",
    "purged_forward",
    "train",
]

@final
class DMatrix:
    @staticmethod
    def dense(data: NDArray[np.float32], missing: float, info: Mapping[str, object]) -> DMatrix: ...
    @staticmethod
    def csr(
        csr: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]],
        n_cols: int,
        info: Mapping[str, object],
    ) -> DMatrix: ...
    def with_info(self, info: Mapping[str, object]) -> DMatrix: ...
    def select_rows(self, rows: NDArray[np.int64]) -> DMatrix: ...
    @property
    def num_row(self) -> int: ...
    @property
    def num_col(self) -> int: ...
    @property
    def label(self) -> NDArray[np.float32] | None: ...
    @property
    def weight(self) -> NDArray[np.float32] | None: ...
    @property
    def base_margin(self) -> NDArray[np.float32] | None: ...
    @property
    def label_lower_bound(self) -> NDArray[np.float32] | None: ...
    @property
    def label_upper_bound(self) -> NDArray[np.float32] | None: ...
    @property
    def feature_weights(self) -> NDArray[np.float32] | None: ...
    @property
    def group(self) -> list[int] | None: ...
    @property
    def categorical(self) -> list[int]: ...

@final
class Params:
    def __new__(cls, params: Mapping[str, Any]) -> Params: ...
    def to_dict(self) -> dict[str, Any]: ...

@final
class Distributions:
    """The conditional distributions a ``dist:*`` model predicts, one per row
    (returned by ``Booster.predict_distribution``).

    Every summary is vectorized over the rows and returns a ``float64``
    array; methods taking ``y`` expect one value per row.
    """

    @property
    def family(self) -> str:
        """The objective naming the family, e.g. ``"dist:normal"``."""
    @property
    def param_names(self) -> list[str]:
        """The natural parameters' names, in column order of ``params``."""
    @property
    def params(self) -> NDArray[np.float64]:
        """The natural parameters, ``(rows, len(param_names))``."""
    def mean(self) -> NDArray[np.float64]:
        """The mean of every row's distribution, ``(rows,)``."""
    def variance(self) -> NDArray[np.float64]:
        """The variance of every row's distribution, ``(rows,)``."""
    def std(self) -> NDArray[np.float64]:
        """The standard deviation of every row's distribution, ``(rows,)``."""
    def quantile(self, q: float) -> NDArray[np.float64]:
        """The ``q``-quantile of every row's distribution, ``(rows,)``."""
    def interval(self, coverage: float) -> NDArray[np.float64]:
        """The central interval holding probability ``coverage``, ``(rows, 2)``."""
    def cdf(self, y: ArrayLike) -> NDArray[np.float64]:
        """``P(Y <= y)`` for every row, ``(rows,)``."""
    def log_prob(self, y: ArrayLike) -> NDArray[np.float64]:
        """The log density (log mass for count families) at ``y``, ``(rows,)``."""
    def crps(self, y: ArrayLike) -> NDArray[np.float64]:
        """The continuous ranked probability score of ``y``, ``(rows,)``."""
    def __len__(self) -> int:
        """The number of rows."""

@final
class Booster:
    @staticmethod
    def load(data: bytes, format: str) -> Booster: ...
    def save(self, format: str) -> bytes: ...
    def predict(
        self, data: DMatrix, kind: str, iteration_range: tuple[int, int] | None = None
    ) -> NDArray[np.float32]: ...
    def predict_leaf(
        self, data: DMatrix, iteration_range: tuple[int, int] | None = None
    ) -> NDArray[np.int32]: ...
    def predict_distribution(
        self, data: DMatrix, iteration_range: tuple[int, int] | None = None
    ) -> Distributions: ...
    def feature_importance(self, importance_type: str) -> dict[int, float]: ...
    def slice(self, begin: int, end: int, step: int) -> Booster: ...
    @property
    def objective(self) -> str: ...
    @property
    def num_features(self) -> int: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def num_targets(self) -> int: ...
    @property
    def num_boosted_rounds(self) -> int: ...
    @property
    def num_trees(self) -> int: ...
    @property
    def num_parallel_tree(self) -> int: ...
    @property
    def best_iteration(self) -> int | None: ...
    @property
    def base_margins(self) -> list[float]: ...
    @property
    def vector_leaves(self) -> bool: ...

@final
class SplitConformal:
    @staticmethod
    def calibrate(booster: Booster, calibration: DMatrix, alpha: float) -> SplitConformal: ...
    def predict_interval(self, data: DMatrix) -> NDArray[np.float32]: ...
    @property
    def half_width(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def n_calibration(self) -> int: ...

@final
class ConformalizedQuantile:
    @staticmethod
    def calibrate(
        models: tuple[Booster, Booster], calibration: DMatrix, alpha: float
    ) -> ConformalizedQuantile: ...
    @staticmethod
    def calibrate_outputs(
        booster: Booster, outputs: tuple[int, int], calibration: DMatrix, alpha: float
    ) -> ConformalizedQuantile: ...
    @staticmethod
    def calibrate_distribution(
        booster: Booster, calibration: DMatrix, alpha: float
    ) -> ConformalizedQuantile: ...
    def predict_interval(self, data: DMatrix) -> NDArray[np.float32]: ...
    @property
    def correction(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def n_calibration(self) -> int: ...

def train(request: Mapping[str, object]) -> tuple[Booster, float | None]: ...
def cv(
    params: Params,
    data: DMatrix,
    num_boost_round: int,
    folds: list[tuple[list[int], list[int]]],
    early_stopping_rounds: int | None = None,
) -> list[tuple[str, list[float], list[float]]]: ...
def k_fold(
    n_rows: int, nfold: int, seed: int
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...
def forward_chaining(
    n_rows: int, n_splits: int, gap: int
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...
def purged_forward(
    decision_at: NDArray[np.int64],
    label_end: NDArray[np.int64],
    validation_fraction: float,
    blocks: int,
    min_train: int,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...
