"""Distribution-free prediction intervals by split conformal prediction.

Both calibrators wrap an already trained :class:`~hessboost.Booster` and
need a **calibration set**: labelled rows that were not used to train it.
With ``n`` calibration rows and miscoverage ``alpha``, the correction ``Q``
is the ``k``-th smallest calibration score, ``k = ceil((n + 1)(1 -
alpha))``. If the calibration rows and a test row are exchangeable, the
interval covers the test label with probability at least ``1 - alpha``
(marginally, over calibration and test draws; not conditionally on ``x``).
When ``alpha < 1 / (n + 1)`` the set is too small and every interval is
``(-inf, inf)``. Calibration sets with non-uniform weights are refused.

* :class:`SplitConformal`: symmetric intervals ``[f(x) - Q, f(x) + Q]``
  around a point model from absolute residuals.
* :class:`ConformalizedQuantile`: conformalized quantile regression (CQR,
  Romano et al., 2019) around a lower/upper quantile band, which keeps the
  band's input-dependent width::

      from hessboost.conformal import ConformalizedQuantile

      params = {"objective": "reg:quantileerror", "quantile_alpha": [0.05, 0.95]}
      band = hessboost.train(params, hessboost.DMatrix(X_train, y_train), 200)
      cqr = ConformalizedQuantile.calibrate_outputs(band, X_cal, y_cal, alpha=0.1)
      lower, upper = cqr.predict_interval(X_test).T
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _data, _hessboost
from hessboost._core import _RECODE_HINT, Booster, DMatrix, _check_schema
from hessboost._exceptions import HessboostError

__all__ = ["ConformalizedQuantile", "SplitConformal"]


_Models: TypeAlias = tuple[tuple[Booster, str], ...]
"""The models reading one matrix, each with its name in errors."""


def _matrix(
    models: _Models, data: object, label: ArrayLike | None, *, calibration: bool
) -> _hessboost.DMatrix:
    """``data`` as a matrix every one of ``models`` reads, labelled by
    ``label`` (a calibration matrix needs labels): other input converted
    with its pandas categories re-coded to the models' (they agree wherever
    both record them), and checked against each model, since a side without
    recorded categories matches anything."""
    if isinstance(data, DMatrix):
        matrix = data
    else:
        if calibration and label is None:
            raise HessboostError("calibration needs labels: pass label= or a labelled DMatrix")
        categories: _data.Categories = {}
        for model, _ in reversed(models):
            categories.update(model._categories)
        info = {} if label is None else _data.info(label=label)
        matrix = DMatrix._coded(data, categories, np.nan, info)
        label = None
    for model, name in models:
        _check_schema(model, matrix, "the data", f"{name}'s", hint=_RECODE_HINT)
    if label is None:
        return matrix._core
    return matrix._core.with_info({**_data.info(label=label), "categorical": None})


def _check_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating)):
        raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
    return float(alpha)


class SplitConformal:
    """Split-conformal intervals ``[f(x) - Q, f(x) + Q]`` around a
    single-output model ``f``, from the absolute residuals ``|y - f(x)|`` of
    a calibration set. Build one with :meth:`calibrate`."""

    __module__ = "hessboost.conformal"

    _core: _hessboost.SplitConformal
    _models: _Models

    def __init__(self) -> None:
        raise TypeError("use SplitConformal.calibrate(...)")

    @classmethod
    def calibrate(
        cls, booster: Booster, data: object, label: ArrayLike | None = None, *, alpha: float
    ) -> Self:
        """Calibrates ``booster`` on ``data`` (a :class:`~hessboost.DMatrix`,
        whose labels are used unless ``label`` is given, or anything it
        accepts) at miscoverage ``alpha`` in ``(0, 1)``.

        Raises:
            HessboostError: ``alpha`` is out of range, the model has several
                outputs, or the calibration rows are unlabelled or weighted
                non-uniformly.
        """
        self = object.__new__(cls)
        self._models = ((booster, "the model"),)
        self._core = _hessboost.SplitConformal.calibrate(
            booster._model,
            _matrix(self._models, data, label, calibration=True),
            _check_alpha(alpha),
        )
        return self

    def predict_interval(self, data: object) -> NDArray[np.float32]:
        """``(rows, 2)`` ``[lower, upper]`` intervals for every row of
        ``data``."""
        return self._core.predict_interval(_matrix(self._models, data, None, calibration=False))

    @property
    def half_width(self) -> float:
        """The calibrated half-width ``Q`` (``inf`` when the calibration set
        is too small for ``alpha``)."""
        return self._core.half_width

    @property
    def alpha(self) -> float:
        """The miscoverage level."""
        return self._core.alpha

    @property
    def n_calibration(self) -> int:
        """The number of calibration rows."""
        return self._core.n_calibration

    def __repr__(self) -> str:
        return (
            f"SplitConformal(alpha={self.alpha}, half_width={self.half_width}, "
            f"n_calibration={self.n_calibration})"
        )


class ConformalizedQuantile:
    """Conformalized quantile regression: a band ``[q_lo(x), q_hi(x)]``
    adjusted to ``[q_lo(x) - Q, q_hi(x) + Q]`` from the calibration scores
    ``max(q_lo(x) - y, y - q_hi(x))``. ``Q`` is negative when the band
    over-covers, so CQR also tightens a band that is too wide.

    Build one from two quantile models (:meth:`calibrate`), two outputs of
    one multi-quantile model (:meth:`calibrate_outputs`), or a ``dist:*``
    model's central quantiles (:meth:`calibrate_distribution`).
    """

    __module__ = "hessboost.conformal"

    _core: _hessboost.ConformalizedQuantile
    _models: _Models

    def __init__(self) -> None:
        raise TypeError("use ConformalizedQuantile.calibrate(...) or its siblings")

    @classmethod
    def _calibrate(
        cls,
        models: _Models,
        data: object,
        label: ArrayLike | None,
        build: Callable[[_hessboost.DMatrix], _hessboost.ConformalizedQuantile],
    ) -> Self:
        self = object.__new__(cls)
        self._models = models
        self._core = build(_matrix(models, data, label, calibration=True))
        return self

    @classmethod
    def calibrate(
        cls,
        lower: Booster,
        upper: Booster,
        data: object,
        label: ArrayLike | None = None,
        *,
        alpha: float,
    ) -> Self:
        """A band from two single-output models, ``lower`` (e.g. trained at
        quantile ``alpha / 2``) and ``upper`` (at ``1 - alpha / 2``). Both
        read the same matrices, so they must have the same features
        (names, categorical features, and categories in the same order).

        Raises:
            HessboostError: The models' features differ.
        """
        _check_schema(lower, upper, "upper", "lower's")
        return cls._calibrate(
            ((lower, "lower"), (upper, "upper")),
            data,
            label,
            lambda matrix: _hessboost.ConformalizedQuantile.calibrate(
                (lower._model, upper._model), matrix, _check_alpha(alpha)
            ),
        )

    @classmethod
    def calibrate_outputs(
        cls,
        booster: Booster,
        data: object,
        label: ArrayLike | None = None,
        *,
        alpha: float,
        outputs: tuple[int, int] = (0, 1),
    ) -> Self:
        """A band from the ``outputs = (lower, upper)`` outputs of one model,
        e.g. ``reg:quantileerror`` with ``quantile_alpha=[alpha / 2, 1 -
        alpha / 2]``."""
        return cls._calibrate(
            ((booster, "the model"),),
            data,
            label,
            lambda matrix: _hessboost.ConformalizedQuantile.calibrate_outputs(
                booster._model, (int(outputs[0]), int(outputs[1])), matrix, _check_alpha(alpha)
            ),
        )

    @classmethod
    def calibrate_distribution(
        cls,
        booster: Booster,
        data: object,
        label: ArrayLike | None = None,
        *,
        alpha: float,
    ) -> Self:
        """A band from a ``dist:*`` model's ``alpha / 2`` and ``1 - alpha /
        2`` quantiles, restoring finite-sample coverage whether or not the
        distribution is well specified."""
        return cls._calibrate(
            ((booster, "the model"),),
            data,
            label,
            lambda matrix: _hessboost.ConformalizedQuantile.calibrate_distribution(
                booster._model, matrix, _check_alpha(alpha)
            ),
        )

    def predict_interval(self, data: object) -> NDArray[np.float32]:
        """``(rows, 2)`` ``[lower, upper]`` intervals for every row of
        ``data``; a row whose adjusted bounds cross is returned as is (the
        empty set)."""
        return self._core.predict_interval(_matrix(self._models, data, None, calibration=False))

    @property
    def correction(self) -> float:
        """The calibrated correction ``Q`` (``inf`` when the calibration set
        is too small for ``alpha``)."""
        return self._core.correction

    @property
    def alpha(self) -> float:
        """The miscoverage level."""
        return self._core.alpha

    @property
    def n_calibration(self) -> int:
        """The number of calibration rows."""
        return self._core.n_calibration

    def __repr__(self) -> str:
        return (
            f"ConformalizedQuantile(alpha={self.alpha}, correction={self.correction}, "
            f"n_calibration={self.n_calibration})"
        )
