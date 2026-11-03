"""Held-out conformal intervals and online score calibration.

The two held-out calibrators wrap an already trained :class:`~hessboost.Booster` and
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

:class:`OnlineCalibrator` instead learns a score threshold from sequential
feedback, without owning a model. Its forecast receipts retain the original
threshold and features, so delayed updates cannot accidentally use newer
predictions. It does not inherit the held-out finite-sample guarantee above.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _hessboost
from hessboost._booster import Booster
from hessboost._exceptions import HessboostError
from hessboost._matrix import _check_schema, _matrix_for

__all__ = ["CalibrationForecast", "ConformalizedQuantile", "OnlineCalibrator", "SplitConformal"]


type _Models = tuple[tuple[Booster, str], ...]
"""The models reading one matrix, each with its name in errors
(``"the model's"``)."""


def _matrix(models: _Models, data: object, label: ArrayLike | None = None) -> _hessboost.DMatrix:
    """``data`` as every one of ``models`` reads it, labelled by ``label``
    (:func:`~hessboost._matrix._matrix_for`)."""
    return _matrix_for(data, models, label=label)._core


def _calibration(models: _Models, data: object, label: ArrayLike | None) -> _hessboost.DMatrix:
    """As :func:`_matrix`, for labelled rows (other input than a
    :class:`~hessboost.DMatrix` needs ``label``)."""
    return _matrix_for(data, models, label=label, require_label=True)._core


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
        self._models = ((booster, "the model's"),)
        self._core = _hessboost.SplitConformal.calibrate(
            booster._model,
            _calibration(self._models, data, label),
            _check_alpha(alpha),
        )
        return self

    def predict_interval(self, data: object) -> NDArray[np.float32]:
        """``(rows, 2)`` ``[lower, upper]`` intervals for every row of
        ``data``."""
        return self._core.predict_interval(_matrix(self._models, data))

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
        self._core = build(_calibration(models, data, label))
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
            ((lower, "lower's"), (upper, "upper's")),
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
            ((booster, "the model's"),),
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
            ((booster, "the model's"),),
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
        return self._core.predict_interval(_matrix(self._models, data))

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


def _online_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _online_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise HessboostError(f"{name} must be >= 0, got {value}")
    return int(value)


@dataclass(frozen=True, init=False, slots=True)
class CalibrationForecast:
    """An immutable issued threshold from :meth:`OnlineCalibrator.predict`.

    Return this receipt to its creating calibrator exactly once, in issue
    order, when its label becomes observable. Receipts are not picklable.
    """

    _core: _hessboost.CalibrationForecast = field(repr=False)

    def __init__(self) -> None:
        raise TypeError("use OnlineCalibrator.predict(...)")

    @classmethod
    def _from_core(cls, core: _hessboost.CalibrationForecast) -> Self:
        result = object.__new__(cls)
        object.__setattr__(result, "_core", core)
        return result

    @property
    def threshold(self) -> float:
        """Issued ``q`` defining the prediction set ``score(x, y) <= q``."""
        return self._core.threshold

    @property
    def index(self) -> int:
        """Zero-based issue index within the creating calibrator."""
        return self._core.index


class OnlineCalibrator:
    """Fixed-step scalar or linear online quantile tracking of scores.

    Tracks the ``1 - alpha`` quantile without owning or retraining a model.
    Set ``lags`` to use the most recently observed scores (newest first,
    initially zero), and ``covariates`` to add forecast-time features. The
    coefficients start at zero and are L2-projected to ``coefficient_bound``
    after every update; the intercept starts at ``initial_threshold``.

    For a point prediction ``p``, report ``[p - q, p + q]`` and later update
    with ``abs(y - p)``. For a quantile band ``[lo, hi]``, report
    ``[lo - q, hi + q]`` and update with ``max(lo - y, y - hi)``. Retain
    that original point or band until its label arrives. Negative thresholds
    are not clamped: they can shrink bands or produce the empty set.

    Call :meth:`update` once per receipt, in issue order. Delayed feedback
    uses the issued threshold and feature snapshot. The caller schedules
    label arrival; this object has no clock or outstanding-receipt queue.
    A lost receipt requires starting a new calibrator. State is not copied
    or pickled. Concurrent calls serialize under a lock held only while
    detached from Python; callers must still deliver updates in issue order.
    A forked child must create its own calibrator rather than use inherited
    state, whose mutex could have been locked at the fork.

    This does not provide split conformal's exchangeability-based
    finite-sample guarantee. The fixed-step tracker's mathematical long-run
    average coverage result needs immediate feedback and bounded scores and
    features; it is not conditional or rolling-window coverage. No general
    delayed-feedback guarantee is claimed. Validate at deployment delays.

    Args:
        alpha: Target miscoverage, strictly between zero and one.
        learning_rate: Positive finite fixed step, in score units.
        initial_threshold: Finite starting intercept, default zero.
        lags: Non-negative number of previously observed scores to include.
        covariates: Non-negative number of caller features per prediction.
        coefficient_bound: Positive finite L2 bound on non-bias coefficients.

    Raises:
        HessboostError: A parameter is outside its domain.
        TypeError: A parameter has the wrong type.
    """

    __slots__ = ("_core",)

    def __init__(
        self,
        alpha: float,
        learning_rate: float,
        *,
        initial_threshold: float = 0.0,
        lags: int = 0,
        covariates: int = 0,
        coefficient_bound: float = 1.0,
    ) -> None:
        self._core = _hessboost.OnlineCalibrator(
            _online_number("alpha", alpha),
            _online_number("learning_rate", learning_rate),
            _online_number("initial_threshold", initial_threshold),
            (_online_count("lags", lags), _online_count("covariates", covariates)),
            _online_number("coefficient_bound", coefficient_bound),
        )

    def predict(self, covariates: ArrayLike | None = None) -> CalibrationForecast:
        """Issue a threshold and retain its immutable feature snapshot.

        ``covariates`` must be a finite, real, one-dimensional vector of
        the configured length. ``None`` supplies an empty vector. A refused
        prediction does not consume an issue index.
        """
        values = np.asarray([] if covariates is None else covariates)
        if values.ndim != 1:
            raise HessboostError("covariates must be one-dimensional")
        if values.dtype.kind not in "iuf":
            raise TypeError(f"covariates must be real numbers, got {values.dtype}")
        return CalibrationForecast._from_core(
            self._core.predict(np.ascontiguousarray(values, dtype=np.float64))
        )

    def update(self, forecast: CalibrationForecast, score: float) -> None:
        """Apply a finite score to its original forecast receipt.

        Foreign, duplicate, out-of-order receipts and invalid scores are
        refused without changing calibration state. Ties are covered.
        """
        if not isinstance(forecast, CalibrationForecast):
            raise TypeError(
                f"forecast must be a CalibrationForecast, got {type(forecast).__name__}"
            )
        self._core.update(forecast._core, _online_number("score", score))

    @property
    def alpha(self) -> float:
        """Target miscoverage."""
        return self._core.alpha

    @property
    def learning_rate(self) -> float:
        """Fixed step size in score units."""
        return self._core.learning_rate

    @property
    def n_updates(self) -> int:
        """Number of successfully applied feedback observations."""
        return self._core.n_updates

    @property
    def pending(self) -> int:
        """Number of issued forecasts still awaiting feedback."""
        return self._core.pending

    @property
    def intercept(self) -> float:
        """Current bias, before lag and covariate contributions."""
        return self._core.intercept

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Copy of non-bias coefficients, lags first then covariates."""
        return self._core.coefficients

    def __copy__(self) -> Self:
        raise TypeError("online calibration state cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        raise TypeError("online calibration state cannot be copied")
