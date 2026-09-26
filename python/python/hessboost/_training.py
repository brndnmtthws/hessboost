"""``train`` and ``cv``, exported from :mod:`hessboost`."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _hessboost
from hessboost._core import Booster, DMatrix
from hessboost._exceptions import HessboostError

__all__ = ["CustomMetric", "Objective", "TrainingCallback", "cv", "train"]

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
        evals: ``(data, name)`` pairs evaluated after every round.
        obj: A custom objective (see :data:`Objective`); the model then
            predicts raw margins.
        custom_metric: A custom metric (see :data:`CustomMetric`), replacing
            the configured ones.
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
            to refresh with ``process_type="update"``.
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
    native = _params(params, dtrain)
    init = _init_model(xgb_model)
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
    booster = Booster._wrap(
        core, dtrain._feature_names, dtrain._feature_types, dict(dtrain._categories)
    )
    booster.best_score = best_score
    return booster


def cv(
    params: Mapping[str, Any],
    dtrain: DMatrix,
    num_boost_round: int = 10,
    *,
    nfold: int = 3,
    folds: _Splitter | Iterable[tuple[ArrayLike, ArrayLike]] | None = None,
    seed: int = 0,
    early_stopping_rounds: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Cross-validates ``params`` on ``dtrain``, as XGBoost's ``xgboost.cv``.

    Every fold trains ``num_boost_round`` rounds on its training rows and is
    evaluated on its test rows after each round.

    Args:
        nfold: Shuffled folds to use when ``folds`` is not given.
        folds: ``(train_rows, test_rows)`` index pairs (for example from
            :mod:`hessboost.folds`), or a scikit-learn splitter (called as
            ``folds.split(X, label)``).
        seed: The shuffle seed of the default folds.
        early_stopping_rounds: Stop on the fold-mean of the last metric;
            the results end at the best round.

    Returns:
        ``{"test-<metric>-mean": ..., "test-<metric>-std": ...}`` per
        metric, one value per round (pass it to ``pandas.DataFrame`` for a
        table). Training-set metrics are not computed.
    """
    if not isinstance(dtrain, DMatrix):
        raise TypeError(f"dtrain must be a DMatrix, got {type(dtrain).__name__}")
    native = _params(params, dtrain)
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
    results = _hessboost.cv(native, dtrain._core, int(num_boost_round), pairs, early_stopping_rounds)
    out: dict[str, NDArray[np.float64]] = {}
    for metric, mean, std in results:
        out[f"test-{metric}-mean"] = np.asarray(mean, dtype=np.float64)
        out[f"test-{metric}-std"] = np.asarray(std, dtype=np.float64)
    return out
