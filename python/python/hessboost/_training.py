"""``train``, ``train_with_budget`` and ``cv``, exported from :mod:`hessboost`."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _hessboost
from hessboost._booster import Booster
from hessboost._exceptions import HessboostError
from hessboost._matrix import DMatrix, _check_schema
from hessboost.target_stats import (
    Column,
    FittedTargetEncoder,
    OrderedTargetEncoder,
    _column_indices,
    _encoded,
)

__all__ = [
    "CustomMetric",
    "CvRefit",
    "Objective",
    "TrainingCallback",
    "cv",
    "train",
    "train_with_budget",
]

Objective: TypeAlias = Callable[[NDArray[np.float32], DMatrix], tuple[ArrayLike, ArrayLike]]
"""A custom objective, as XGBoost's ``obj``: ``obj(margins, dtrain)``
returns ``(grad, hess)`` for the raw ``margins`` (``(rows,)`` or ``(rows,
outputs)``), each with one value per margin."""

CustomMetric: TypeAlias = Callable[
    [NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None], float
]
"""A custom evaluation metric: ``metric(predictions, labels, weights)``
returns one number for an eval set. ``predictions`` are post-transform
(raw margins with a custom objective); its name is the callable's
``__name__``."""

EvalsResult: TypeAlias = dict[str, dict[str, list[float]]]
"""``{eval name: {metric: per-round values}}``."""


class _Splitter(Protocol):
    """A scikit-learn cross-validation splitter (``KFold``, ``GroupKFold``,
    ``TimeSeriesSplit``, ...)."""

    def split(self, X: Any, y: Any = None, /) -> Iterable[tuple[ArrayLike, ArrayLike]]:
        """``(train_rows, test_rows)`` index pairs for the rows of ``X``."""
        ...


def _feature_index(name: object, names: list[str] | None) -> int:
    if isinstance(name, (int, np.integer)):
        return int(name)
    if names is None or name not in names:
        raise HessboostError(f"constraint names unknown feature {name!r}")
    return names.index(str(name))


def _params(params: Mapping[str, Any], dtrain: DMatrix) -> _hessboost.Params:
    """Native parameters, with XGBoost's feature-name forms of the
    constraint parameters resolved against ``dtrain``."""
    if not isinstance(params, Mapping):
        raise TypeError(f"params must be a mapping, got {type(params).__name__}")
    resolved = dict(params)
    names = dtrain._feature_names
    monotone = resolved.get("monotone_constraints")
    if isinstance(monotone, Mapping):
        directions = [0] * dtrain.num_col()
        for name, direction in monotone.items():
            directions[_feature_index(name, names)] = int(direction)
        resolved["monotone_constraints"] = directions
    interactions = resolved.get("interaction_constraints")
    if isinstance(interactions, (list, tuple)):
        resolved["interaction_constraints"] = [
            [_feature_index(name, names) for name in group] for group in interactions
        ]
    return _hessboost.Params(resolved)


def _custom_objective_params(params: Mapping[str, Any]) -> tuple[dict[str, Any], int | None]:
    """``params`` for a custom objective, without ``num_class``, and that
    count: the objective's output count (XGBoost's custom-softmax
    convention), ``None`` for one per label column (absent or ``0``)."""
    if not isinstance(params, Mapping):
        raise TypeError(f"params must be a mapping, got {type(params).__name__}")
    if "objective" in params:
        raise HessboostError(
            "obj replaces objective; params must not also set objective "
            f"(got {params['objective']!r})"
        )
    resolved = dict(params)
    outputs = resolved.pop("num_class", None)
    if outputs is None:
        return resolved, None
    if isinstance(outputs, bool) or not isinstance(outputs, (int, np.integer)) or outputs < 0:
        raise HessboostError(
            "num_class (the custom objective's output count) must be a non-negative "
            f"integer, got {outputs!r}"
        )
    return resolved, int(outputs) or None


def _init_model(xgb_model: str | os.PathLike[str] | Booster | None) -> Booster | None:
    if xgb_model is None or isinstance(xgb_model, Booster):
        return xgb_model
    return Booster(xgb_model)


def _metric_name(metric: Callable[..., object]) -> str:
    name = getattr(metric, "__name__", "")
    return name if isinstance(name, str) and name and not name.startswith("<") else "custom"


def _format_round(iteration: int, scores: Sequence[tuple[str, str, float]]) -> str:
    fields = "\t".join(f"{data}-{metric}:{value:.5f}" for data, metric, value in scores)
    return f"[{iteration}]\t{fields}"


class TrainingCallback:
    """A hook :func:`train` calls after every boosting round, on the training
    thread, in round order. Subclass it and override :meth:`after_iteration`.

    It sees each round's iteration number and the evaluation history so far,
    not the model: the model exists only once training returns. The GIL is
    held while it runs, so keep it short.
    """

    __module__ = "hessboost"

    def after_iteration(self, iteration: int, evals_log: EvalsResult) -> bool:
        """Called once the round's eval sets are scored.

        Args:
            iteration: The round's boosting iteration (after continued
                training, counted from the start of the initial model).
            evals_log: ``{eval name: {metric: values}}`` through this round
                (empty without eval sets). Do not modify it.

        Returns:
            ``True`` to stop training after this round. The booster keeps
            every completed round; with early stopping, ``best_iteration``
            is the best round so far. Default: ``False``.
        """
        return False


def train(
    params: Mapping[str, Any],
    dtrain: DMatrix,
    num_boost_round: int = 10,
    *,
    evals: Sequence[tuple[DMatrix, str]] = (),
    obj: Objective | None = None,
    custom_metric: CustomMetric | None = None,
    maximize: bool | None = None,
    early_stopping_rounds: int | None = None,
    evals_result: EvalsResult | None = None,
    verbose_eval: bool | int = True,
    xgb_model: str | os.PathLike[str] | Booster | None = None,
    callbacks: Sequence[TrainingCallback] = (),
) -> Booster:
    """Trains a booster, as XGBoost's ``xgboost.train``.

    Args:
        params: XGBoost parameters by name (``objective``, ``max_depth``,
            ``eta``/``learning_rate``, ``eval_metric``, ``nthread``,
            ``seed``, ...) plus hessboost's own options (``path_smooth``,
            ``dist_gradient``, ...). An unknown name, a value of the wrong
            type or range, and a combination hessboost does not implement
            raise :class:`HessboostError`; nothing is silently ignored.
            ``monotone_constraints`` may map feature names to directions and
            ``interaction_constraints`` list feature names.
        dtrain: The training data.
        num_boost_round: Boosting iterations (with ``process_type="update"``,
            the number of iterations of ``xgb_model`` to refresh).
        evals: ``(data, name)`` pairs evaluated after every round. Each must
            have ``dtrain``'s features: the same names and categorical
            features (where both record them) and, for frame categoricals,
            the same categories in the same order, since codes are positions
            in them. Codes without recorded categories (numpy data with
            ``feature_types``) are taken to be ``dtrain``'s.
        obj: A custom objective (see :data:`Objective`); the model then
            predicts raw margins. It replaces ``objective``, which ``params``
            must then not set; ``num_class`` in ``params`` is its output
            count, as for XGBoost's custom softmax (default: one output per
            label column of ``dtrain``).
        custom_metric: A custom metric (see :data:`CustomMetric`), reported
            after the configured (or default) metrics, as in XGBoost, and
            so the one early stopping watches.
        maximize: Whether ``custom_metric`` improves upward (default
            ``False``). Built-in metrics know their direction.
        early_stopping_rounds: Stop once the last metric on the last eval
            set has not improved for this many rounds; the booster records
            :attr:`Booster.best_iteration` and :attr:`Booster.best_score`.
        evals_result: A dict to fill with the evaluation history, updated as
            rounds complete.
        verbose_eval: Print each round's evaluation as it completes (with
            eval sets): every round for ``True``, every ``n``-th round and
            the last for an int ``n``.
        xgb_model: A booster (or model file) to continue boosting from, or
            to refresh with ``process_type="update"``. ``dtrain`` and every
            eval set must have its features, compared as for ``evals``; the
            new booster keeps its categories where ``dtrain`` records none.
        callbacks: :class:`TrainingCallback` instances run after every round;
            any returning ``True`` stops training.

    Training is deterministic: the same parameters, data and ``seed`` give
    the same model at any ``nthread``, and callbacks only observe it. The
    GIL is released while training, except while a custom objective,
    metric, or callback runs. Ctrl-C (``KeyboardInterrupt``) stops training
    at the end of the current round and raises; an exception from a
    callback, objective, or metric does the same and propagates.

    Raises:
        HessboostError: The parameters, data, or their combination are
            refused.
        KeyboardInterrupt: Training was interrupted.
    """
    if not isinstance(dtrain, DMatrix):
        raise TypeError(f"dtrain must be a DMatrix, got {type(dtrain).__name__}")
    outputs: int | None = None
    if obj is not None:
        params, outputs = _custom_objective_params(params)
    native = _params(params, dtrain)
    init = _init_model(xgb_model)
    # Every matrix meets the model being continued as well as dtrain, and a
    # side without recorded categories matches anything, so each pairing is
    # checked on its own (the checks are not transitive).
    references: list[tuple[DMatrix | Booster, str]] = [(dtrain, "dtrain's")]
    if init is not None:
        _check_schema(init, dtrain, "dtrain", "xgb_model's")
        references.append((init, "xgb_model's"))
    for data, name in evals:
        for reference, against in references:
            _check_schema(reference, data, f"eval set {str(name)!r}", against)
    request: dict[str, object] = {
        "params": native,
        "dtrain": dtrain._core,
        "num_boost_round": int(num_boost_round),
        "evals": [(data._core, str(name)) for data, name in evals],
        "early_stopping_rounds": early_stopping_rounds,
        "init_model": None if init is None else init._model,
    }
    if obj is not None:
        objective = obj

        def gradients(
            margins: NDArray[np.float32],
        ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
            grad, hess = objective(margins, dtrain)
            return (
                np.ascontiguousarray(grad, dtype=np.float32).reshape(-1),
                np.ascontiguousarray(hess, dtype=np.float32).reshape(-1),
            )

        request["obj"] = gradients
        request["outputs"] = outputs
    if custom_metric is not None:
        request["custom_metric"] = {
            "function": custom_metric,
            "name": _metric_name(custom_metric),
            "maximize": bool(maximize),
        }
    elif maximize is not None:
        raise HessboostError(
            "maximize applies to custom_metric only; built-in metrics know their direction"
        )
    log: EvalsResult = {} if evals_result is None else evals_result
    log.clear()
    period = 0 if verbose_eval is False else 1 if verbose_eval is True else int(verbose_eval)
    rounds = 0
    unprinted: str | None = None
    hooks = list(callbacks)

    def on_round(iteration: int, scores: list[tuple[str, str, float]]) -> bool:
        nonlocal rounds, unprinted
        for data, metric, value in scores:
            log.setdefault(data, {}).setdefault(metric, []).append(value)
        if period and scores:
            line = _format_round(iteration, scores)
            if rounds % period == 0:
                print(line, flush=True)
                unprinted = None
            else:
                unprinted = line
        rounds += 1
        stop = False
        for hook in hooks:
            stop = bool(hook.after_iteration(iteration, log)) or stop
        return stop

    request["on_round"] = on_round
    core, best_score = _hessboost.train(request)
    if unprinted is not None:
        print(unprinted, flush=True)
    booster = _trained(core, dtrain, init)
    booster.best_score = best_score
    return booster


def _trained(core: _hessboost.Booster, dtrain: DMatrix, init: Booster | None) -> Booster:
    """The booster of ``core``, trained on ``dtrain`` (continuing ``init``),
    with their feature schema."""
    if init is None:
        return Booster._wrap(
            core, dtrain._feature_names, dtrain._feature_types, dict(dtrain._categories)
        )
    # What dtrain does not record (numpy codes) is still the earlier model's,
    # and prediction keeps re-coding frames to it.
    return Booster._wrap(
        core,
        dtrain._feature_names if dtrain._feature_names is not None else init._feature_names,
        dtrain._feature_types if dtrain._feature_types is not None else init._feature_types,
        {**init._categories, **dtrain._categories},
    )


def train_with_budget(
    params: Mapping[str, Any],
    dtrain: DMatrix,
    budget: float = 0.5,
    *,
    iteration_limit: int | None = None,
    stopping_rounds: int | None = None,
) -> Booster:
    """Trains with one fitting ``budget`` in place of a learning rate, tree
    limits and a round count (PerpetualBooster's algorithm, reimplemented;
    see the Rust ``hessboost::training::budget`` docs).

    The budget sets the learning rate (``10 ** -budget`` for budgets up to
    1, decaying more slowly above), each tree's loss-reduction target, and
    the stopping rules; trees grow best-first while a five-fold
    generalization check accepts their splits. It is much slower than one
    fixed-round fit of the same size, since it replaces a tuning search.
    The model is an ordinary gbtree booster (predictions, SHAP and every
    model format work unchanged).

    Args:
        params: XGBoost parameters by name, as for :func:`train`. Budget
            mode reads only ``objective`` (with its parameters),
            ``base_score``, ``max_bin``, ``nthread``, and ``max_delta_step``
            for ``count:poisson``; any other parameter set to other than its
            default is refused. Objectives: ``reg:squarederror``,
            ``reg:pseudohubererror``, ``binary:logistic``,
            ``binary:logitraw``, ``reg:logistic``, ``count:poisson``,
            ``reg:gamma``, ``reg:tweedie``.
        dtrain: The training data (one label per row).
        budget: The fitting budget, in ``(0, 5)``; larger budgets train more
            trees with a smaller learning rate and fit more closely (``1.0``
            and ``1.5`` are common choices).
        iteration_limit: Lower the hard cap on boosting rounds (default:
            the budget's, 1000 to 4000).
        stopping_rounds: Weak or non-improving rounds that stop training
            (default: the budget's).

    Training is deterministic and independent of ``nthread``. The GIL is
    released while training; Ctrl-C takes effect only once it returns.

    Raises:
        HessboostError: The budget or a limit is out of range, or a
            parameter or the objective is not supported in budget mode.
    """
    if not isinstance(dtrain, DMatrix):
        raise TypeError(f"dtrain must be a DMatrix, got {type(dtrain).__name__}")
    if isinstance(budget, bool) or not isinstance(budget, (int, float, np.integer, np.floating)):
        raise TypeError(f"budget must be a number, got {type(budget).__name__}")
    core = _hessboost.train_with_budget(
        _params(params, dtrain), dtrain._core, float(budget), iteration_limit, stopping_rounds
    )
    return Booster._wrap(
        core, dtrain._feature_names, dtrain._feature_types, dict(dtrain._categories)
    )


CvHistory: TypeAlias = dict[str, NDArray[np.float64]]
"""``{"test-<metric>-mean": ..., "test-<metric>-std": ...}``: what :func:`cv`
returns, one value per round."""


@dataclass(frozen=True)
class CvRefit:
    """What ``cv(..., refit=True)`` returns: the cross-validation history
    and the booster retrained on every row at the chosen round count."""

    history: CvHistory
    """The history :func:`cv` returns without ``refit``."""
    booster: Booster
    """The booster trained on every row of ``dtrain`` for
    :attr:`num_boost_round` rounds (continuing ``xgb_model`` when given), as
    :func:`train` would train it without eval sets. With ``target_stats``,
    it was trained on ``dtrain`` encoded by :attr:`target_encoder`, so its
    encoded columns are numerical."""
    num_boost_round: int
    """The chosen round count: through the best round under early stopping,
    else every round the folds ran (the length of each history array)."""
    target_encoder: FittedTargetEncoder | None
    """With ``target_stats``, the encoder fitted on every row of ``dtrain``:
    encode new data with its ``transform`` before predicting with
    :attr:`booster`. ``None`` otherwise."""


def _history(results: list[tuple[str, list[float], list[float]]]) -> CvHistory:
    out: CvHistory = {}
    for metric, mean, std in results:
        out[f"test-{metric}-mean"] = np.asarray(mean, dtype=np.float64)
        out[f"test-{metric}-std"] = np.asarray(std, dtype=np.float64)
    return out


@overload
def cv(
    params: Mapping[str, Any],
    dtrain: DMatrix,
    num_boost_round: int = 10,
    *,
    nfold: int = 3,
    folds: _Splitter | Iterable[tuple[ArrayLike, ArrayLike]] | None = None,
    seed: int = 0,
    early_stopping_rounds: int | None = None,
    target_stats: Sequence[Column] | None = None,
    target_encoder: OrderedTargetEncoder | None = None,
    target_stats_label: ArrayLike | None = None,
    xgb_model: str | os.PathLike[str] | Booster | None = None,
    refit: Literal[False] = False,
) -> CvHistory: ...
@overload
def cv(
    params: Mapping[str, Any],
    dtrain: DMatrix,
    num_boost_round: int = 10,
    *,
    nfold: int = 3,
    folds: _Splitter | Iterable[tuple[ArrayLike, ArrayLike]] | None = None,
    seed: int = 0,
    early_stopping_rounds: int | None = None,
    target_stats: Sequence[Column] | None = None,
    target_encoder: OrderedTargetEncoder | None = None,
    target_stats_label: ArrayLike | None = None,
    xgb_model: str | os.PathLike[str] | Booster | None = None,
    refit: Literal[True],
) -> CvRefit: ...
@overload
def cv(
    params: Mapping[str, Any],
    dtrain: DMatrix,
    num_boost_round: int = 10,
    *,
    nfold: int = 3,
    folds: _Splitter | Iterable[tuple[ArrayLike, ArrayLike]] | None = None,
    seed: int = 0,
    early_stopping_rounds: int | None = None,
    target_stats: Sequence[Column] | None = None,
    target_encoder: OrderedTargetEncoder | None = None,
    target_stats_label: ArrayLike | None = None,
    xgb_model: str | os.PathLike[str] | Booster | None = None,
    refit: bool,
) -> CvHistory | CvRefit: ...
def cv(
    params: Mapping[str, Any],
    dtrain: DMatrix,
    num_boost_round: int = 10,
    *,
    nfold: int = 3,
    folds: _Splitter | Iterable[tuple[ArrayLike, ArrayLike]] | None = None,
    seed: int = 0,
    early_stopping_rounds: int | None = None,
    target_stats: Sequence[Column] | None = None,
    target_encoder: OrderedTargetEncoder | None = None,
    target_stats_label: ArrayLike | None = None,
    xgb_model: str | os.PathLike[str] | Booster | None = None,
    refit: bool = False,
) -> CvHistory | CvRefit:
    """Cross-validates ``params`` on ``dtrain``, as XGBoost's ``xgboost.cv``.

    Every fold trains ``num_boost_round`` rounds on its training rows and is
    evaluated on its test rows after each round. On ranking data (``group``
    or ``qid`` set), every fold's training and test rows must be whole query
    groups, each group's rows together and in row order (as
    ``sklearn.model_selection.GroupKFold`` over the query ids gives them);
    the shuffled default folds split groups and are refused.

    Args:
        nfold: Shuffled folds to use when ``folds`` is not given.
        folds: ``(train_rows, test_rows)`` index pairs (for example from
            :mod:`hessboost.folds`), or a scikit-learn splitter (called as
            ``folds.split(X, label)``).
        seed: The shuffle seed of the default folds.
        early_stopping_rounds: Stop on the fold-mean of the last metric;
            the results end at the best round.
        target_stats: Categorical features (indices or names) to encode
            with ordered target statistics inside each fold: the encoder is
            fitted on the fold's training rows only and encodes its test
            rows with those statistics, so no held-out label reaches an
            encoding (see :mod:`hessboost.target_stats`).
        target_encoder: The encoder ``target_stats`` uses (default:
            ``OrderedTargetEncoder()``).
        target_stats_label: The per-row target the ``target_stats`` encoder
            averages instead of ``dtrain``'s labels (which stay the training
            target): one finite value per row, split with the folds, such
            as one column of a multi-target matrix or a class's 0/1
            indicator.
        xgb_model: A booster (or model file) every fold, and the refit,
            continue, as :func:`train` continues ``xgb_model`` (with the
            same checks of ``dtrain`` against it). The history counts this
            run's rounds from 0. Not with ``target_stats``: the model was
            trained on its own encoding of those columns.
        refit: Also retrain on every row of ``dtrain`` for the chosen round
            count and return a :class:`CvRefit`.

    Returns:
        ``{"test-<metric>-mean": ..., "test-<metric>-std": ...}`` per
        metric, one value per round (pass it to ``pandas.DataFrame`` for a
        table); training-set metrics are not computed. With ``refit``, a
        :class:`CvRefit` holding that history and the retrained booster.

    Raises:
        HessboostError: The parameters, folds, or data are refused,
            ``target_stats_label`` is given without ``target_stats`` or with
            the wrong length, ``xgb_model`` is given with ``target_stats``,
            or an EBM stops its bags early (``ebm_early_stopping_rounds``:
            the folds would end their stages at different rounds).
    """
    if not isinstance(dtrain, DMatrix):
        raise TypeError(f"dtrain must be a DMatrix, got {type(dtrain).__name__}")
    native = _params(params, dtrain)
    encoding: tuple[_hessboost.OrderedTargetEncoder, list[int]] | None = None
    if target_stats is not None:
        encoder = OrderedTargetEncoder() if target_encoder is None else target_encoder
        if not isinstance(encoder, OrderedTargetEncoder):
            raise TypeError(
                f"target_encoder must be an OrderedTargetEncoder, got {type(encoder).__name__}"
            )
        encoding = (encoder._core, _column_indices(target_stats, dtrain._feature_names))
    elif target_encoder is not None:
        raise HessboostError("target_encoder needs target_stats, the columns to encode")
    labels = None
    if target_stats_label is not None:
        labels = np.ascontiguousarray(target_stats_label, dtype=np.float32)
        if labels.ndim != 1:
            raise HessboostError(
                f"target_stats_label must be one value per row, got shape {labels.shape}"
            )
    init = _init_model(xgb_model)
    if init is not None:
        _check_schema(init, dtrain, "dtrain", "xgb_model's")
    rows = dtrain.num_row()
    if folds is None:
        pairs = [
            (train.tolist(), test.tolist()) for train, test in _hessboost.k_fold(rows, nfold, seed)
        ]
    else:
        chosen: Iterable[tuple[ArrayLike, ArrayLike]]
        if hasattr(folds, "split"):
            chosen = cast(_Splitter, folds).split(np.zeros((rows, 1)), dtrain.get_label())
        else:
            chosen = folds
        pairs = [
            (np.asarray(train).reshape(-1).tolist(), np.asarray(test).reshape(-1).tolist())
            for train, test in chosen
        ]
    request: dict[str, object] = {
        "params": native,
        "dtrain": dtrain._core,
        "num_boost_round": int(num_boost_round),
        "folds": pairs,
        "early_stopping_rounds": early_stopping_rounds,
        "target_stats": encoding,
        "target_stats_label": labels,
        "init_model": None if init is None else init._model,
    }
    if not refit:
        return _history(_hessboost.cv(request))
    results, core, rounds, fitted = _hessboost.cv_refit(request)
    if encoding is None or fitted is None:
        return CvRefit(_history(results), _trained(core, dtrain, init), rounds, None)
    # The refit trained on dtrain encoded by `fitted`, its columns numerical.
    encoded = _encoded(dtrain, dtrain._core, encoding[1])
    return CvRefit(
        _history(results),
        _trained(core, encoded, None),
        rounds,
        FittedTargetEncoder._wrap(fitted, dtrain),
    )
