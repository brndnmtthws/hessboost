"""The parameters, data handling and fitting every
:mod:`hessboost.sklearn` estimator shares (imported only by it, after its
scikit-learn check)."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data

from hessboost import _data
from hessboost._booster import Booster, ImportanceType
from hessboost._exceptions import HessboostError
from hessboost._matrix import DMatrix
from hessboost._training import EvalsResult, TrainingCallback, _init_model, train

if TYPE_CHECKING:
    from hessboost._sklearn_base import BaseEstimator
else:
    from sklearn.base import BaseEstimator

# The values the selector parameters accept (``TrainingParams::from_xgboost``
# spellings).
_GrowPolicy: TypeAlias = Literal["depthwise", "lossguide", "symmetric"]
_BoosterName: TypeAlias = Literal["gbtree", "dart", "gblinear", "boulevard", "ebm"]
_TreeMethod: TypeAlias = Literal["auto", "exact", "approx", "hist"]
_SamplingMethod: TypeAlias = Literal["uniform", "gradient_based"]
_Device: TypeAlias = Literal["cpu", "metal", "wgpu", "cuda"]
_MultiStrategy: TypeAlias = Literal["one_output_per_tree", "multi_output_tree"]

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
        grow_policy: _GrowPolicy | None = None,
        learning_rate: float | None = None,
        objective: str | None = None,
        booster: _BoosterName | None = None,
        tree_method: _TreeMethod | None = None,
        n_jobs: int | None = None,
        gamma: float | None = None,
        min_child_weight: float | None = None,
        max_delta_step: float | None = None,
        subsample: float | None = None,
        sampling_method: _SamplingMethod | None = None,
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
        device: _Device | None = None,
        multi_strategy: _MultiStrategy | None = None,
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
                raise HessboostError(
                    f"{name!r} is set both as an estimator parameter and in params"
                )
            params[name] = value
        return params

    # -- data ---------------------------------------------------------------

    def _check_X(self, X: Any, reset: bool, *, aligned: str | None = None) -> Any:
        """``X`` validated (feature count and names) as scikit-learn does;
        pandas and polars frames are passed on unchanged to keep their
        categories. A polars ``LazyFrame`` is collected once, except where an
        array ``aligned`` with its rows accompanies it (``"y"`` when fitting,
        ``"base_margin"`` when predicting): the array might not be in the
        order of the rows a collect inside hessboost yields."""
        if _data._is_polars_lazyframe(X):
            if aligned is not None:
                raise HessboostError(
                    f"X is a polars LazyFrame while {aligned} is a separate array, which may not "
                    "be in the order of the rows hessboost would collect (polars' streaming engine "
                    "keeps no row order after a join or group_by); collect it (X.collect()) and "
                    "pass the DataFrame"
                )
            X = _data.collect_frame(X)
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
        extra: Mapping[str, Any] | None,
        categories: _data.Categories,
    ) -> DMatrix:
        """``X`` and its metadata as a matrix, frame categories re-coded to
        ``categories`` (values they lack become missing)."""
        info = _data.info(label=y, weight=weight, base_margin=base_margin, **(extra or {}))
        return DMatrix._coded(X, categories, self.missing, info)

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
        X = self._check_X(X, reset=True, aligned="y")
        init = _init_model(
            xgb_model.get_booster() if isinstance(xgb_model, _HessboostModel) else xgb_model
        )
        # Continuing or refreshing reads X with the earlier model's codes, and
        # every eval set with the training ones (the earlier model's where X
        # records none, as for numpy codes).
        earlier = {} if init is None else init._categories
        dtrain = self._matrix(X, y, sample_weight, base_margin, group, earlier)
        categories = {**earlier, **dtrain._categories}
        evals: list[tuple[DMatrix, str]] = []
        for index, (raw_eval, y_eval) in enumerate(eval_set or ()):
            X_eval = self._check_X(raw_eval, reset=False, aligned="y")
            weight = None if sample_weight_eval_set is None else sample_weight_eval_set[index]
            margin = None if base_margin_eval_set is None else base_margin_eval_set[index]
            labels = encode(y_eval) if encode is not None else y_eval
            extra = None if eval_groups is None else eval_groups[index]
            evals.append(
                (
                    self._matrix(X_eval, labels, weight, margin, extra, categories),
                    f"validation_{index}",
                )
            )
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
        X = self._check_X(X, reset=False, aligned=None if base_margin is None else "base_margin")
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
            from sklearn.exceptions import NotFittedError

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
