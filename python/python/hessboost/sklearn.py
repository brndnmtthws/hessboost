"""scikit-learn estimators: :class:`HessboostRegressor`,
:class:`HessboostClassifier`, :class:`HessboostRanker` and
:class:`HessboostDistributionRegressor`.

They follow XGBoost's scikit-learn wrapper (``XGBRegressor`` and friends):
the constructor takes XGBoost's parameter names (plus ``callbacks``, a
list of :class:`~hessboost.TrainingCallback`), ``fit`` takes sample
weights, base margins, eval sets and ``verbose`` (live per-round output,
``True`` or a period), and the fitted estimator exposes the
:class:`~hessboost.Booster`. Parameters left at ``None`` use hessboost's
(XGBoost's) defaults; ``params`` passes any other training parameter.

Needs scikit-learn (``pip install 'hessboost[scikit-learn]'``); ``import
hessboost`` itself does not import it. pandas and polars frames keep their
categorical columns as categorical features: ``eval_set`` frames are
re-coded to the training frame's categories, a frame continuing an earlier
fit (``xgb_model``) to that model's, and prediction frames to the fitted
model's (values they lack become missing).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from sklearn.utils.multiclass import (
        check_classification_targets,
    )
    from sklearn.utils.validation import (
        check_array,
        column_or_1d,
    )

    if TYPE_CHECKING:
        from hessboost._sklearn_base import ClassifierMixin, RegressorMixin
    else:
        from sklearn.base import ClassifierMixin, RegressorMixin
except ImportError as _missing:  # pragma: no cover - exercised without scikit-learn
    raise ImportError(
        "hessboost.sklearn needs scikit-learn: pip install 'hessboost[scikit-learn]'"
    ) from _missing

from hessboost._booster import Booster, Distributions
from hessboost._exceptions import HessboostError
from hessboost._sklearn_common import EvalSet, _HessboostModel

__all__ = [
    "HessboostClassifier",
    "HessboostDistributionRegressor",
    "HessboostRanker",
    "HessboostRegressor",
]


def _labels(y: ArrayLike | None, name: str) -> NDArray[Any]:
    if y is None:
        raise ValueError(f"{name} requires y to be passed, but the target y is None")
    return np.asarray(y)


class HessboostRegressor(RegressorMixin, _HessboostModel):
    """Gradient-boosted regression, as XGBoost's ``XGBRegressor``.

    The objective defaults to ``reg:squarederror``; any ``reg:*``,
    ``count:*``, ``survival:cox`` or quantile objective works. A 2-D ``y``
    trains a multi-target model predicting ``(rows, targets)``.
    """

    __module__ = "hessboost.sklearn"

    def fit(
        self,
        X: Any,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
        base_margin: ArrayLike | None = None,
        eval_set: EvalSet | None = None,
        sample_weight_eval_set: Sequence[ArrayLike | None] | None = None,
        base_margin_eval_set: Sequence[ArrayLike | None] | None = None,
        verbose: bool | int = False,
        xgb_model: Booster | _HessboostModel | str | None = None,
    ) -> Self:
        """Fits the model on ``X`` (array, sparse matrix or DataFrame) and
        ``y``. ``xgb_model`` continues from an earlier fit."""
        labels = check_array(_labels(y, type(self).__name__), ensure_2d=False, dtype=np.float64)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.reshape(-1)
        return self._fit(
            X,
            labels,
            sample_weight=sample_weight,
            base_margin=base_margin,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
            verbose=verbose,
            xgb_model=xgb_model,
            params=self._train_params(),
        )

    def predict(
        self,
        X: Any,
        *,
        output_margin: bool = False,
        iteration_range: tuple[int, int] | None = None,
        base_margin: ArrayLike | None = None,
    ) -> NDArray[np.float32]:
        """Predictions (``float32``), ``(rows,)`` or ``(rows, targets)``."""
        return self._predict(X, output_margin, iteration_range, base_margin)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags


class HessboostClassifier(ClassifierMixin, _HessboostModel):
    """Gradient-boosted classification, as XGBoost's ``XGBClassifier``.

    Any class labels work (they are encoded as ``classes_``); two classes
    train ``binary:logistic``, more ``multi:softprob``, unless ``objective``
    names another ``binary:*`` or ``multi:softprob`` objective.
    """

    __module__ = "hessboost.sklearn"

    classes_: NDArray[Any]
    n_classes_: int

    def _objective(self) -> str:
        if self.objective is not None:
            return self.objective
        return "binary:logistic" if self.n_classes_ == 2 else "multi:softprob"

    def _encode(self, y: ArrayLike) -> NDArray[np.int64]:
        labels = column_or_1d(np.asarray(y), warn=True)
        positions: NDArray[np.int64] = np.clip(
            np.searchsorted(self.classes_, labels), 0, len(self.classes_) - 1
        ).astype(np.int64)
        if not np.array_equal(self.classes_[positions], labels):
            raise HessboostError("eval_set holds labels that are not in the training classes")
        return positions

    def fit(
        self,
        X: Any,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
        base_margin: ArrayLike | None = None,
        eval_set: EvalSet | None = None,
        sample_weight_eval_set: Sequence[ArrayLike | None] | None = None,
        base_margin_eval_set: Sequence[ArrayLike | None] | None = None,
        verbose: bool | int = False,
        xgb_model: Booster | _HessboostModel | str | None = None,
    ) -> Self:
        """Fits the model on ``X`` (array, sparse matrix or DataFrame) and
        class labels ``y``."""
        labels = column_or_1d(_labels(y, type(self).__name__), warn=True)
        check_classification_targets(labels)
        classes, encoded = np.unique(labels, return_inverse=True)
        if len(classes) < 2:
            raise ValueError(
                f"{type(self).__name__} needs samples of at least 2 classes; got "
                f"{len(classes)} class"
            )
        self.classes_ = classes
        self.n_classes_ = len(classes)
        params = self._train_params()
        if params["objective"].startswith("multi:"):
            params["num_class"] = self.n_classes_
        return self._fit(
            X,
            encoded.astype(np.float32),
            sample_weight=sample_weight,
            base_margin=base_margin,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
            verbose=verbose,
            xgb_model=xgb_model,
            params=params,
            encode=self._encode,
        )

    def predict_proba(
        self,
        X: Any,
        *,
        iteration_range: tuple[int, int] | None = None,
        base_margin: ArrayLike | None = None,
    ) -> NDArray[np.float32]:
        """Class probabilities, ``(rows, n_classes)``, columns in
        ``classes_`` order.

        Raises:
            HessboostError: The objective (``binary:hinge``,
                ``multi:softmax``) predicts no probabilities.
        """
        objective = self.get_booster().objective
        if objective in ("binary:hinge", "binary:logitraw", "multi:softmax"):
            raise HessboostError(f"{objective} does not predict probabilities")
        values = self._predict(X, False, iteration_range, base_margin)
        if values.ndim == 1:
            return np.column_stack([1.0 - values, values]).astype(np.float32)
        return values

    def predict(
        self,
        X: Any,
        *,
        output_margin: bool = False,
        iteration_range: tuple[int, int] | None = None,
        base_margin: ArrayLike | None = None,
    ) -> NDArray[Any]:
        """Predicted classes (from ``classes_``), or raw margins with
        ``output_margin``."""
        values = self._predict(X, output_margin, iteration_range, base_margin)
        if output_margin:
            return values
        objective = self.get_booster().objective
        if values.ndim == 2:
            index = values.argmax(axis=1)
        elif objective == "multi:softmax":
            index = values.astype(np.int64)
        elif objective == "binary:logitraw":
            index = (values > 0).astype(np.int64)
        else:
            index = (values > 0.5).astype(np.int64)
        return self.classes_[index]


class HessboostRanker(_HessboostModel):
    """Learning to rank (LambdaMART), as XGBoost's ``XGBRanker``.

    Rows of one query must be contiguous; give the queries by ``group``
    (sizes) or ``qid`` (a sorted query id per row). ``sample_weight`` holds
    one weight per query. The objective defaults to ``rank:ndcg``.
    """

    __module__ = "hessboost.sklearn"
    _default_objective = "rank:ndcg"

    def fit(
        self,
        X: Any,
        y: ArrayLike,
        *,
        group: ArrayLike | None = None,
        qid: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        base_margin: ArrayLike | None = None,
        eval_set: EvalSet | None = None,
        eval_group: Sequence[ArrayLike] | None = None,
        eval_qid: Sequence[ArrayLike] | None = None,
        sample_weight_eval_set: Sequence[ArrayLike | None] | None = None,
        base_margin_eval_set: Sequence[ArrayLike | None] | None = None,
        verbose: bool | int = False,
        xgb_model: Booster | _HessboostModel | str | None = None,
    ) -> Self:
        """Fits the ranker on ``X``, relevance labels ``y`` and the query
        structure (``group`` or ``qid``; the same for every eval set)."""
        if group is None and qid is None:
            raise HessboostError("HessboostRanker.fit needs group or qid")
        eval_groups: list[dict[str, Any]] | None = None
        if eval_set is not None:
            if eval_group is not None:
                eval_groups = [{"group": sizes} for sizes in eval_group]
            elif eval_qid is not None:
                eval_groups = [{"qid": ids} for ids in eval_qid]
            else:
                raise HessboostError("eval_set needs eval_group or eval_qid")
        return self._fit(
            X,
            _labels(y, type(self).__name__),
            sample_weight=sample_weight,
            base_margin=base_margin,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
            verbose=verbose,
            xgb_model=xgb_model,
            params=self._train_params(),
            group={"group": group, "qid": qid},
            eval_groups=eval_groups,
        )

    def predict(
        self,
        X: Any,
        *,
        output_margin: bool = False,
        iteration_range: tuple[int, int] | None = None,
        base_margin: ArrayLike | None = None,
    ) -> NDArray[np.float32]:
        """Ranking scores, ``(rows,)``: higher ranks first within a query."""
        return self._predict(X, output_margin, iteration_range, base_margin)


class HessboostDistributionRegressor(RegressorMixin, _HessboostModel):
    """Distributional regression (NGBoost / XGBoostLSS style): predicts a
    full conditional distribution per row.

    The objective defaults to ``dist:normal``; ``dist:lognormal``,
    ``dist:gamma``, ``dist:poisson`` and ``dist:negbinomial`` are the other
    families. :meth:`predict` returns the distributions' means and
    :meth:`predict_distribution` the distributions themselves.
    """

    __module__ = "hessboost.sklearn"
    _default_objective = "dist:normal"

    def fit(
        self,
        X: Any,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
        eval_set: EvalSet | None = None,
        sample_weight_eval_set: Sequence[ArrayLike | None] | None = None,
        verbose: bool | int = False,
        xgb_model: Booster | _HessboostModel | str | None = None,
    ) -> Self:
        """Fits the model on ``X`` and ``y``."""
        labels = column_or_1d(
            check_array(_labels(y, type(self).__name__), ensure_2d=False, dtype=np.float64),
            warn=True,
        )
        if not self._objective().startswith("dist:"):
            raise HessboostError(
                f"HessboostDistributionRegressor needs a dist:* objective, got {self._objective()}"
            )
        return self._fit(
            X,
            labels,
            sample_weight=sample_weight,
            base_margin=None,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=None,
            verbose=verbose,
            xgb_model=xgb_model,
            params=self._train_params(),
        )

    def predict_distribution(
        self, X: Any, *, iteration_range: tuple[int, int] | None = None
    ) -> Distributions:
        """The predicted distribution of every row."""
        booster = self.get_booster()
        X = self._check_X(X, reset=False)
        return booster.predict_distribution(
            X, iteration_range=iteration_range, missing=self.missing
        )

    def predict(
        self, X: Any, *, iteration_range: tuple[int, int] | None = None
    ) -> NDArray[np.float64]:
        """The mean of every row's predicted distribution."""
        return self.predict_distribution(X, iteration_range=iteration_range).mean()
