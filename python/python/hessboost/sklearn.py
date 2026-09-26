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
hessboost`` itself does not import it. pandas frames keep their
``category`` columns as categorical features.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from sklearn.utils import check_random_state  # type: ignore[import-untyped]
    from sklearn.utils.multiclass import (  # type: ignore[import-untyped]
        check_classification_targets,
    )
    from sklearn.utils.validation import (  # type: ignore[import-untyped]
        check_array,
        column_or_1d,
        validate_data,
    )

    if TYPE_CHECKING:
        from hessboost._sklearn_base import BaseEstimator, ClassifierMixin, RegressorMixin
    else:
        from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
except ImportError as _missing:  # pragma: no cover - exercised without scikit-learn
    raise ImportError(
        "hessboost.sklearn needs scikit-learn: pip install 'hessboost[scikit-learn]'"
    ) from _missing

from hessboost import _data
from hessboost._core import Booster, DMatrix, Distributions, ImportanceType
from hessboost._exceptions import HessboostError
from hessboost._training import EvalsResult, TrainingCallback, train

__all__ = [
    "HessboostClassifier",
    "HessboostDistributionRegressor",
    "HessboostRanker",
    "HessboostRegressor",
]

EvalSet = Sequence[tuple[Any, ArrayLike]]
"""``(X, y)`` pairs evaluated after every round."""


class _HessboostModel(BaseEstimator):
    """The parameters and fitting shared by every estimator."""

    _default_objective = "reg:squarederror"
    _booster: Booster
    _evals_result: EvalsResult
    n_features_in_: int
    """The number of features seen in ``fit``."""
    feature_names_in_: NDArray[np.object_]
    """The feature names seen in ``fit`` (only for frames with string
    column names)."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_leaves: int | None = None,
        max_bin: int | None = None,
        grow_policy: str | None = None,
        learning_rate: float | None = None,
        objective: str | None = None,
        booster: str | None = None,
        tree_method: str | None = None,
        n_jobs: int | None = None,
        gamma: float | None = None,
        min_child_weight: float | None = None,
        max_delta_step: float | None = None,
        subsample: float | None = None,
        sampling_method: str | None = None,
        colsample_bytree: float | None = None,
        colsample_bylevel: float | None = None,
        colsample_bynode: float | None = None,
        reg_alpha: float | None = None,
        reg_lambda: float | None = None,
        scale_pos_weight: float | None = None,
        base_score: float | None = None,
        random_state: int | np.random.RandomState | None = None,
        missing: float = np.nan,
        num_parallel_tree: int | None = None,
        monotone_constraints: Any = None,
        interaction_constraints: Any = None,
        importance_type: ImportanceType | None = None,
        device: str | None = None,
        multi_strategy: str | None = None,
        eval_metric: str | Sequence[str] | None = None,
        early_stopping_rounds: int | None = None,
        callbacks: Sequence[TrainingCallback] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.grow_policy = grow_policy
        self.learning_rate = learning_rate
        self.objective = objective
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.sampling_method = sampling_method
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.device = device
        self.multi_strategy = multi_strategy
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.callbacks = callbacks
        self.params = params

    # -- parameters ---------------------------------------------------------

    _NAMED = (
        ("max_depth", "max_depth"),
        ("max_leaves", "max_leaves"),
        ("max_bin", "max_bin"),
        ("grow_policy", "grow_policy"),
        ("learning_rate", "eta"),
        ("booster", "booster"),
        ("tree_method", "tree_method"),
        ("gamma", "gamma"),
        ("min_child_weight", "min_child_weight"),
        ("max_delta_step", "max_delta_step"),
        ("subsample", "subsample"),
        ("sampling_method", "sampling_method"),
        ("colsample_bytree", "colsample_bytree"),
        ("colsample_bylevel", "colsample_bylevel"),
        ("colsample_bynode", "colsample_bynode"),
        ("reg_alpha", "alpha"),
        ("reg_lambda", "lambda"),
        ("scale_pos_weight", "scale_pos_weight"),
        ("base_score", "base_score"),
        ("num_parallel_tree", "num_parallel_tree"),
        ("monotone_constraints", "monotone_constraints"),
        ("interaction_constraints", "interaction_constraints"),
        ("device", "device"),
        ("multi_strategy", "multi_strategy"),
        ("eval_metric", "eval_metric"),
    )

    def _objective(self) -> str:
        return self.objective or self._default_objective

    def _train_params(self) -> dict[str, Any]:
        """The training parameters, XGBoost-named."""
        params: dict[str, Any] = {"objective": self._objective()}
        for attribute, name in self._NAMED:
            value = getattr(self, attribute)
            if value is not None:
                params[name] = value
        if self.n_jobs is not None:
            params["nthread"] = max(int(self.n_jobs), 0)
        if self.random_state is not None:
            if isinstance(self.random_state, numbers.Integral):
                params["seed"] = int(self.random_state)
            else:
                params["seed"] = int(check_random_state(self.random_state).randint(2**31 - 1))
        for name, value in (self.params or {}).items():
            if name in params and name != "objective":
                raise HessboostError(f"{name!r} is set both as an estimator parameter and in params")
            params[name] = value
        return params

    # -- data ---------------------------------------------------------------

    def _check_X(self, X: Any, reset: bool) -> Any:
        """``X`` validated (feature count and names) as scikit-learn does;
        pandas frames are passed on unchanged to keep their categories."""
        if _data._is_frame(X):
            validate_data(self, X, reset=reset, skip_check_array=True)
            return X
        return validate_data(
            self,
            X,
            reset=reset,
            accept_sparse=True,
            dtype=[np.float32, np.float64],
            ensure_all_finite="allow-nan",
        )

    def _matrix(
        self,
        X: Any,
        y: ArrayLike | None,
        weight: ArrayLike | None,
        base_margin: ArrayLike | None,
        extra: Mapping[str, Any] | None = None,
    ) -> DMatrix:
        return DMatrix(
            X,
            y,
            weight=weight,
            base_margin=base_margin,
            missing=self.missing,
            **(extra or {}),
        )

    def _fit(
        self,
        X: Any,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None,
        base_margin: ArrayLike | None,
        eval_set: EvalSet | None,
        sample_weight_eval_set: Sequence[ArrayLike | None] | None,
        base_margin_eval_set: Sequence[ArrayLike | None] | None,
        verbose: bool | int,
        xgb_model: Booster | _HessboostModel | str | None,
        params: dict[str, Any],
        encode: Any = None,
        group: Mapping[str, Any] | None = None,
        eval_groups: Sequence[Mapping[str, Any]] | None = None,
    ) -> Self:
        X = self._check_X(X, reset=True)
        dtrain = self._matrix(X, y, sample_weight, base_margin, group)
        evals: list[tuple[DMatrix, str]] = []
        for index, (X_eval, y_eval) in enumerate(eval_set or ()):
            X_eval = self._check_X(X_eval, reset=False)
            weight = None if sample_weight_eval_set is None else sample_weight_eval_set[index]
            margin = None if base_margin_eval_set is None else base_margin_eval_set[index]
            labels = encode(y_eval) if encode is not None else y_eval
            extra = None if eval_groups is None else eval_groups[index]
            evals.append(
                (self._matrix(X_eval, labels, weight, margin, extra), f"validation_{index}")
            )
        init = xgb_model.get_booster() if isinstance(xgb_model, _HessboostModel) else xgb_model
        self._evals_result = {}
        self._booster = train(
            params,
            dtrain,
            self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=self._evals_result,
            verbose_eval=verbose,
            xgb_model=init,
            callbacks=self.callbacks or (),
        )
        return self

    def _predict(
        self,
        X: Any,
        output_margin: bool = False,
        iteration_range: tuple[int, int] | None = None,
        base_margin: ArrayLike | None = None,
    ) -> NDArray[Any]:
        booster = self.get_booster()
        X = self._check_X(X, reset=False)
        return booster.predict(
            X,
            output_margin=output_margin,
            iteration_range=iteration_range,
            base_margin=base_margin,
            missing=self.missing,
        )

    # -- fitted state -------------------------------------------------------

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_booster")

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = True
        tags.input_tags.categorical = True
        return tags

    def get_booster(self) -> Booster:
        """The fitted booster.

        Raises:
            sklearn.exceptions.NotFittedError: The estimator is not fitted.
        """
        if not hasattr(self, "_booster"):
            from sklearn.exceptions import NotFittedError  # type: ignore[import-untyped]

            raise NotFittedError(f"this {type(self).__name__} is not fitted yet; call fit first")
        return self._booster

    def evals_result(self) -> EvalsResult:
        """The eval sets' history, ``{"validation_0": {metric: values}}``."""
        self.get_booster()
        return self._evals_result

    @property
    def best_iteration(self) -> int | None:
        """The best iteration early stopping chose, or ``None``."""
        return self.get_booster().best_iteration

    @property
    def best_score(self) -> float | None:
        """The watched metric at :attr:`best_iteration`, or ``None``."""
        return self.get_booster().best_score

    @property
    def feature_importances_(self) -> NDArray[np.float64]:
        """Importance of every feature by ``importance_type`` (default
        ``"gain"``), normalized to sum to 1 (all zeros without splits)."""
        booster = self.get_booster()
        scores = booster._model.feature_importance(self.importance_type or "gain")
        values = np.zeros(booster.num_features(), dtype=np.float64)
        for index, value in scores.items():
            values[index] = value
        total = values.sum()
        return values / total if total > 0 else values



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
        labels = check_array(
            _labels(y, type(self).__name__), ensure_2d=False, dtype=np.float64
        )
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
    full conditional distribution per row. Beyond XGBoost.

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
