"""Statistical inference for Boulevard boosting: confidence intervals for
the regression function ``f(x)``, prediction and reproduction intervals,
and the honest leaf refit.

Train with ``{"booster": "boulevard", ...}`` (squared error only; BRAT-D with
``boulevard_dropout``, BRAT-P with ``num_parallel_tree > 1``, which needs
``eta = 1``), then fit a :class:`BoulevardInference` on the rows the model
was trained on, or refitted on with :func:`honest_refit`::

    from hessboost.inference import BoulevardInference, honest_refit

    params = {"booster": "boulevard", "eta": 0.8, "boulevard_dropout": 0.5,
              "subsample": 0.8, "max_depth": 4, "min_child_weight": 5}
    trained = hessboost.train(params, hessboost.DMatrix(X_struct, y_struct), 200)
    model = honest_refit(trained, X_values, y_values)
    inference = BoulevardInference.fit(model, X_values, holdout=X_cal, holdout_label=y_cal)
    lower, upper = inference.confidence_intervals(X_test, alpha=0.05).T

The intervals are asymptotic and conditional on the tree structures: they
reach nominal coverage for low-dimensional smooth signals after an honest
refit and under-cover elsewhere; prediction intervals also need Gaussian
noise. The Rust crate's ``hessboost::inference`` documentation has the
algorithms (Zhou & Hooker, JMLR 2022; Fang, Tan & Hooker, NeurIPS 2025),
assumptions, and a table of validated regimes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, Self, TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _hessboost
from hessboost._booster import Booster
from hessboost._exceptions import HessboostError
from hessboost.conformal import _calibration, _check_alpha, _matrix
from hessboost.ebm import TermShape

__all__ = ["BoulevardInference", "BoulevardInfo", "EbmInference", "TermBands", "honest_refit"]


@dataclass(frozen=True)
class BoulevardInfo:
    """How a ``booster = boulevard`` model was trained
    (:attr:`hessboost.Booster.boulevard`)."""

    __module__ = "hessboost.inference"

    dropout: float
    """BRAT-D's dropout probability ``p`` (``0`` for BRAT-P)."""
    learning_rate: float
    """The learning rate ``lambda`` (``eta``; ``1`` for BRAT-P)."""
    subsample: float
    """The row subsample ratio."""
    reg_lambda: float
    """The L2 leaf penalty."""
    truncation: float | None
    """The residual truncation level, ``None`` for none."""
    seed: int
    """The training seed, which :func:`honest_refit` derives its draws from."""
    intercept_from_labels: bool
    """Whether the intercept is the training-label mean."""

    @classmethod
    def _from_core(cls, info: dict[str, Any] | None) -> BoulevardInfo | None:
        return None if info is None else cls(**info)


def _check_noise(noise_variance: float | None) -> float | None:
    if noise_variance is None:
        return None
    if isinstance(noise_variance, bool) or not isinstance(
        noise_variance, (int, float, np.floating)
    ):
        raise TypeError(f"noise_variance must be a number, got {type(noise_variance).__name__}")
    return float(noise_variance)


def _check_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise HessboostError(f"{name} must be >= 0, got {value}")
    return int(value)


class _InferenceCore(Protocol):
    """The native queries :class:`BoulevardInference` and
    :class:`EbmInference` share."""

    @property
    def noise_variance(self) -> float: ...
    def standard_errors(self, data: _hessboost.DMatrix) -> NDArray[np.float64]: ...
    def confidence_intervals(
        self, data: _hessboost.DMatrix, alpha: float
    ) -> NDArray[np.float64]: ...
    def prediction_intervals(
        self, data: _hessboost.DMatrix, alpha: float
    ) -> NDArray[np.float64]: ...


_CoreT = TypeVar("_CoreT", bound=_InferenceCore)

_NativeFit: TypeAlias = Callable[
    [
        _hessboost.Booster,
        _hessboost.DMatrix,
        _hessboost.DMatrix | None,
        float | None,
        int | None,
        int,
    ],
    _CoreT,
]
"""A native inference class's ``fit``: model, training rows, holdout rows,
noise variance, landmarks, seed."""


class _IntervalQueries(Generic[_CoreT]):
    """The fitting and the interval queries of an inference over one
    model's training rows."""

    _core: _CoreT
    _models: tuple[tuple[Booster, str], ...]

    @classmethod
    def _fit(
        cls,
        native: _NativeFit[_CoreT],
        booster: Booster,
        data: object,
        label: ArrayLike | None,
        holdout: object | None,
        holdout_label: ArrayLike | None,
        noise_variance: float | None,
        landmarks: int | None,
        seed: int,
    ) -> Self:
        self = object.__new__(cls)
        self._models = ((booster, "the model's"),)
        train = _matrix(self._models, data, label)
        held = None if holdout is None else _calibration(self._models, holdout, holdout_label)
        seed = _check_count("seed", seed) or 0
        self._core = native(
            booster._model,
            train,
            held,
            _check_noise(noise_variance),
            _check_count("landmarks", landmarks),
            seed,
        )
        return self

    def _data(self, data: object) -> _hessboost.DMatrix:
        return _matrix(self._models, data)

    @property
    def noise_variance(self) -> float:
        """The noise variance estimate."""
        return self._core.noise_variance

    def standard_errors(self, data: object) -> NDArray[np.float64]:
        """The standard error of the model's prediction at every row of
        ``data``."""
        return self._core.standard_errors(self._data(data))

    def confidence_intervals(self, data: object, *, alpha: float) -> NDArray[np.float64]:
        """``(rows, 2)`` ``[lower, upper]`` confidence intervals for the
        model's ``f(x)`` at miscoverage ``alpha``."""
        return self._core.confidence_intervals(self._data(data), _check_alpha(alpha))

    def prediction_intervals(self, data: object, *, alpha: float) -> NDArray[np.float64]:
        """``(rows, 2)`` prediction intervals for a new label (Gaussian
        noise)."""
        return self._core.prediction_intervals(self._data(data), _check_alpha(alpha))


class BoulevardInference(_IntervalQueries[_hessboost.BoulevardInference]):
    """The leaf-kernel variance machinery of one Boulevard model: build one
    with :meth:`fit`, then query any rows."""

    __module__ = "hessboost.inference"

    def __init__(self) -> None:
        raise TypeError("use BoulevardInference.fit(...)")

    @classmethod
    def fit(
        cls,
        booster: Booster,
        data: object,
        label: ArrayLike | None = None,
        *,
        holdout: object | None = None,
        holdout_label: ArrayLike | None = None,
        noise_variance: float | None = None,
        landmarks: int | None = None,
        seed: int = 0,
    ) -> Self:
        """Fits the leaf kernel of ``booster`` over ``data``, the rows it was
        trained on (or, after :func:`honest_refit`, refitted on).

        The noise variance is the mean squared residual on the labelled
        ``holdout`` rows, a known ``noise_variance``, or (neither) on the
        training rows themselves, which needs ``data``'s labels (biased
        low). ``landmarks`` switches from the exact ``n x n`` solve (at most
        8192 rows) to the Nystrom approximation from that many rows drawn
        with ``seed``.

        Raises:
            HessboostError: ``booster`` is not a Boulevard fit, ``data`` is
                not its training data, both ``holdout`` and
                ``noise_variance`` are given, or the exact solver gets too
                many rows.
        """
        return cls._fit(
            _hessboost.BoulevardInference.fit,
            booster,
            data,
            label,
            holdout,
            holdout_label,
            noise_variance,
            landmarks,
            seed,
        )

    def reproduction_intervals(self, data: object, *, alpha: float) -> NDArray[np.float64]:
        """``(rows, 2)`` intervals for the prediction of the same procedure
        retrained on an independent sample."""
        return self._core.reproduction_intervals(self._data(data), _check_alpha(alpha))

    def calibrated_prediction_intervals(self, data: object, *, alpha: float) -> NDArray[np.float64]:
        """Prediction intervals rescaled by one factor chosen on the
        ``holdout`` rows (a split-conformal quantile); needs ``holdout``."""
        return self._core.calibrated_prediction_intervals(self._data(data), _check_alpha(alpha))

    def __repr__(self) -> str:
        return f"BoulevardInference(noise_variance={self.noise_variance})"


@dataclass(frozen=True)
class TermBands:
    """Pointwise confidence bands of one term's shape function
    (:meth:`EbmInference.term_bands`), on the shape's grid."""

    __module__ = "hessboost.inference"

    shape: TermShape
    """The term's shape function."""
    standard_errors: NDArray[np.float64]
    """The standard error at every cell."""
    lower: NDArray[np.float64]
    """The shape minus ``z * standard error`` at every cell."""
    upper: NDArray[np.float64]
    """The shape plus ``z * standard error`` at every cell."""


class EbmInference(_IntervalQueries[_hessboost.EbmInference]):
    """Confidence bands on the shape functions of a Boulevard EBM
    (``{"booster": "ebm", "ebm_boulevard": True}``; Fang, Tan, Pipping &
    Hooker, AISTATS 2026). Conditional on the tree structures, so they
    under-cover across training samples (see the Rust crate's
    ``hessboost::inference::EbmInference`` validation table)."""

    __module__ = "hessboost.inference"

    def __init__(self) -> None:
        raise TypeError("use EbmInference.fit(...)")

    @classmethod
    def fit(
        cls,
        booster: Booster,
        data: object,
        label: ArrayLike | None = None,
        *,
        holdout: object | None = None,
        holdout_label: ArrayLike | None = None,
        noise_variance: float | None = None,
        landmarks: int | None = None,
        seed: int = 0,
    ) -> Self:
        """As :meth:`BoulevardInference.fit`, for a Boulevard EBM.

        Raises:
            HessboostError: ``booster`` is not a Boulevard EBM, or as
                :meth:`BoulevardInference.fit`.
        """
        return cls._fit(
            _hessboost.EbmInference.fit,
            booster,
            data,
            label,
            holdout,
            holdout_label,
            noise_variance,
            landmarks,
            seed,
        )

    @property
    def intercept_standard_error(self) -> float:
        """The standard error of the intercept."""
        return self._core.intercept_standard_error

    def term_bands(self, term: int, *, alpha: float) -> TermBands:
        """Bands on every cell of term ``term``'s shape function at
        miscoverage ``alpha``."""
        index = _check_count("term", term) or 0
        shape, se, lower, upper = self._core.term_bands(index, _check_alpha(alpha))
        return TermBands(TermShape._wrap(shape), se, lower, upper)

    def term_standard_errors(self, term: int, data: object) -> NDArray[np.float64]:
        """The standard error of term ``term``'s shape at every row of
        ``data``."""
        index = _check_count("term", term) or 0
        return self._core.term_standard_errors(index, self._data(data))

    def __repr__(self) -> str:
        return f"EbmInference(noise_variance={self.noise_variance})"


def honest_refit(booster: Booster, data: object, label: ArrayLike | None = None) -> Booster:
    """Refits every leaf of the Boulevard model ``booster`` on ``data``,
    labelled rows independent of its training data, keeping the tree
    structures; fit :class:`BoulevardInference` on ``data`` afterwards.

    Raises:
        HessboostError: ``booster`` is not a Boulevard fit, or ``data`` is
            unlabelled or weighted.
    """
    values = _calibration(((booster, "the model's"),), data, label)
    core = _hessboost.honest_refit(booster._model, values)
    return Booster._wrap(core, booster._feature_names, booster._feature_types, booster._categories)
