"""In-place row addition and deletion for trained models (incremental and
decremental learning, machine unlearning; beyond XGBoost).

An :class:`OnlineModel` keeps a model together with its training data and,
for approximate updates, the per-node statistics that let
:meth:`~OnlineModel.update` add and delete training rows without retraining
from scratch (after Lin et al., *Online Gradient Boosting Decision Tree:
In-Place Updates for Efficient Adding/Deleting Data*, 2025)::

    from hessboost.online import OnlineModel

    online = OnlineModel.train({"tree_method": "hist", "max_depth": 6}, dtrain, 100)
    report = online.update(hessboost.DMatrix(X_new, y_new), deletions=[3, 17])
    online.model.predict(X_test)

The update mode is :class:`Approximate` (the default) or :class:`Exact`.
``Approximate(tolerance)`` has the split robustness tolerance ``σ`` in
``(0, 1]``: a node keeps its split while it ranks within the top
``max(1, ⌊σ · candidates⌋)`` candidates of the updated statistics, else its
subtree is regrown. It touches only the changed rows and the regrown
subtrees and stays close to retraining for small changes; it forgets
deleted rows only partially. ``Exact()`` makes every update equal
:func:`hessboost.train` on :attr:`~OnlineModel.data` bit for bit, at about
the cost of retraining (the answer when unlearning must be complete).

Updates need a configuration whose retraining depends on the data alone:
``gbtree`` with ``tree_method="hist"`` (or ``"auto"``), depth-wise growth
with ``max_depth > 0``, one output, no row or column sampling, no
constraints, none of the beyond-XGBoost split options, CPU, and an
objective with per-row gradients and Newton-step leaves (not ranking,
``survival:cox``, ``reg:absoluteerror`` or ``reg:quantileerror``); the data
has one label per row and no weights, base margins, groups, label bounds or
feature weights, and the approximate mode needs numerical features. Anything
else raises :class:`~hessboost.HessboostError`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from hessboost import _data, _hessboost
from hessboost._core import Booster, DMatrix, _check_schema
from hessboost._training import _params

__all__ = ["Approximate", "Exact", "OnlineMode", "OnlineModel", "UpdateCallback", "UpdateReport"]

UpdateCallback: TypeAlias = Callable[[int], bool]
"""A per-iteration hook of :meth:`OnlineModel.update`: ``callback(iteration)``
runs after each updated boosting iteration, on the update's thread with the
GIL held; returning ``True`` abandons the update."""


@dataclass(frozen=True)
class UpdateReport:
    """What an :meth:`OnlineModel.update` did."""

    __module__ = "hessboost.online"

    nodes_kept: int
    """Nodes whose split (or leaf) was kept, over all trees."""
    subtrees_regrown: int
    """Subtrees regrown (the exact mode regrows every tree)."""
    rows_refreshed: int
    """Rows whose gradients were recomputed in at least one tree."""


@dataclass(frozen=True)
class Exact:
    """The exact mode: every update equals :func:`hessboost.train` on the
    updated data bit for bit, at about the cost of retraining."""


@dataclass(frozen=True)
class Approximate:
    """The approximate mode with the split robustness tolerance ``σ``: a node
    keeps its split while it ranks within the top ``max(1, ⌊σ · candidates⌋)``
    candidates; ``1`` regrows only splits that stopped being valid. Needs
    ``0 < tolerance <= 1`` (the default ``0.1`` is the paper's
    recommendation)."""

    tolerance: float = 0.1

    def __post_init__(self) -> None:
        tolerance = self.tolerance
        if isinstance(tolerance, bool) or not isinstance(
            tolerance, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"tolerance must be a number, got {type(tolerance).__name__}")
        object.__setattr__(self, "tolerance", float(tolerance))


OnlineMode: TypeAlias = Exact | Approximate
"""How an :class:`OnlineModel` updates: :class:`Exact` or :class:`Approximate`."""


def _online_params(mode: OnlineMode) -> _hessboost.OnlineParams:
    if isinstance(mode, Exact):
        return _hessboost.OnlineParams.exact()
    if isinstance(mode, Approximate):
        return _hessboost.OnlineParams.approximate(mode.tolerance)
    raise TypeError(f"mode must be Exact or Approximate, got {type(mode).__name__}")


def _check_matrix(data: object, name: str) -> DMatrix:
    if not isinstance(data, DMatrix):
        raise TypeError(f"{name} must be a DMatrix, got {type(data).__name__}")
    return data


def _rows(deletions: Sequence[int] | NDArray[np.integer]) -> NDArray[np.int64]:
    rows = np.asarray(deletions).reshape(-1)
    if rows.size and rows.dtype.kind not in "iu":
        raise TypeError(f"deletions must be integer row indices, got dtype {rows.dtype}")
    return np.ascontiguousarray(rows, dtype=np.int64)


class OnlineModel:
    """A trained model with its training data, updatable in place. Build one
    with :meth:`train` or :meth:`from_model`; see :mod:`hessboost.online`.

    Updates are atomic: a refused, interrupted, or stopped update leaves the
    model and data as they were. While an update runs, :attr:`model`,
    :attr:`data` and another :meth:`update` of the same online model raise
    :class:`~hessboost.HessboostError`, from other threads and from the
    update's own callback alike; :meth:`num_row` and :attr:`mode` stay
    readable.
    """

    __module__ = "hessboost.online"

    _core: _hessboost.OnlineModel
    _mode: OnlineMode
    _feature_names: list[str] | None
    _feature_types: list[str] | None
    _categories: _data.Categories

    def __init__(self) -> None:
        raise TypeError("use OnlineModel.train(...) or OnlineModel.from_model(...)")

    @classmethod
    def _wrap(
        cls,
        core: _hessboost.OnlineModel,
        mode: OnlineMode,
        feature_names: list[str] | None,
        feature_types: list[str] | None,
        categories: _data.Categories,
    ) -> Self:
        self = object.__new__(cls)
        self._core = core
        self._mode = mode
        self._feature_names = feature_names
        self._feature_types = feature_types
        self._categories = categories
        return self

    @classmethod
    def train(
        cls,
        params: Mapping[str, Any],
        dtrain: DMatrix,
        num_boost_round: int = 10,
        mode: OnlineMode = Approximate(),
    ) -> Self:
        """Trains ``num_boost_round`` iterations on ``dtrain``, as
        :func:`hessboost.train` does, and keeps what updates need.

        Args:
            params: XGBoost parameters by name, read as :func:`hessboost.train`
                reads them.
            dtrain: The training data.
            num_boost_round: Boosting iterations; every update keeps this
                many.
            mode: The update mode, :class:`Approximate` (default, with
                tolerance ``0.1``) or :class:`Exact`.

        The GIL is released while training. Ctrl-C stops training at the end
        of the current round and raises ``KeyboardInterrupt``.

        Raises:
            HessboostError: The parameters or data are refused, or cannot be
                updated in place, or the tolerance is outside ``(0, 1]``.
        """
        dtrain = _check_matrix(dtrain, "dtrain")
        core = _hessboost.OnlineModel.train(
            _params(params, dtrain), dtrain._core, int(num_boost_round), _online_params(mode)
        )
        return cls._wrap(
            core, mode, dtrain._feature_names, dtrain._feature_types, dict(dtrain._categories)
        )

    @classmethod
    def from_model(
        cls,
        booster: Booster,
        params: Mapping[str, Any],
        dtrain: DMatrix,
        mode: OnlineMode = Approximate(),
    ) -> Self:
        """Resumes from ``booster``, trained with ``params`` on ``dtrain``
        (for example one loaded from a file), rebuilding the update state by
        replaying its trees over ``dtrain``. The GIL is released meanwhile.

        Raises:
            HessboostError: The refusals of :meth:`train`, or ``booster`` is
                not a model ``params`` could have trained on ``dtrain``
                (another objective or ``max_delta_step``, several outputs,
                weighted trees, categorical trees in the :class:`Approximate` mode,
                linear leaves such as an imported LightGBM ``linear_tree``
                model's, another feature count or other features), or it was
                early-stopped (slice it to its best iterations first).
        """
        if not isinstance(booster, Booster):
            raise TypeError(f"booster must be a Booster, got {type(booster).__name__}")
        dtrain = _check_matrix(dtrain, "dtrain")
        _check_schema(booster, dtrain, "dtrain", "the model's")
        core = _hessboost.OnlineModel.from_model(
            booster._model, _params(params, dtrain), dtrain._core, _online_params(mode)
        )
        # What dtrain does not record (numpy codes) is still the model's.
        return cls._wrap(
            core,
            mode,
            dtrain._feature_names if dtrain._feature_names is not None else booster._feature_names,
            dtrain._feature_types if dtrain._feature_types is not None else booster._feature_types,
            {**booster._categories, **dtrain._categories},
        )

    @overload
    def update(
        self,
        additions: DMatrix | None = None,
        deletions: Sequence[int] | NDArray[np.integer] = (),
        *,
        callback: None = None,
    ) -> UpdateReport: ...
    @overload
    def update(
        self,
        additions: DMatrix | None = None,
        deletions: Sequence[int] | NDArray[np.integer] = (),
        *,
        callback: UpdateCallback,
    ) -> UpdateReport | None: ...
    def update(
        self,
        additions: DMatrix | None = None,
        deletions: Sequence[int] | NDArray[np.integer] = (),
        *,
        callback: UpdateCallback | None = None,
    ) -> UpdateReport | None:
        """Adds the rows of ``additions`` and deletes the rows ``deletions``
        (indices into :attr:`data`), then updates the model. The new data is
        the kept rows in their order followed by the added ones.

        Args:
            additions: Rows to add, with labels, the training data's
                features, and no other metadata.
            deletions: Distinct row indices of :attr:`data` to delete.
            callback: Called after every updated iteration (see
                :data:`UpdateCallback`); returning ``True`` abandons the
                update.

        Returns:
            What the update did, or ``None`` when ``callback`` abandoned it.

        The GIL is released while updating, except while ``callback`` runs.
        Ctrl-C abandons the update at the end of the current iteration and
        raises ``KeyboardInterrupt``; an exception from ``callback`` does the
        same and propagates. A refused, abandoned, or interrupted update
        changes nothing. The update is applied only after a last check for
        Ctrl-C once every iteration is done; one pressed later is raised
        after ``update`` returns.

        Raises:
            HessboostError: The model is being updated, or the change is
                refused: out-of-range or repeated deletions, deleting every
                row, additions without labels or with metadata or other
                features, (in the :class:`Approximate` mode) added values beyond the
                training data's bins, or updated data retraining refuses
                (such as labels outside the objective's domain).
            ModelFormatError: The update overflows ``float32`` (extreme
                labels or margins), as training refuses such a model.
            KeyboardInterrupt: The update was interrupted.
        """
        matrix = None
        if additions is not None:
            added = _check_matrix(additions, "additions")
            _check_schema(self, added, "additions", "the training data's")
            matrix = added._core
        on_round: Callable[[int, list[tuple[str, str, float]]], bool] | None = None
        if callback is not None:
            hook = callback

            def stop(iteration: int, _scores: list[tuple[str, str, float]]) -> bool:
                return bool(hook(iteration))

            on_round = stop

        report = self._core.update(matrix, _rows(deletions), on_round)
        return None if report is None else UpdateReport(*report)

    @property
    def model(self) -> Booster:
        """The current model (a new :class:`~hessboost.Booster` holding a
        copy of it; later updates do not change it)."""
        return Booster._wrap(
            self._core.model, self._feature_names, self._feature_types, dict(self._categories)
        )

    @property
    def data(self) -> DMatrix:
        """The current training data: the original rows minus deletions plus
        additions, in update order (a copy)."""
        matrix = DMatrix.__new__(DMatrix)
        matrix._core = self._core.data
        matrix._feature_names = self._feature_names
        matrix._feature_types = self._feature_types
        matrix._categories = self._categories
        return matrix

    def num_row(self) -> int:
        """The number of rows of :attr:`data`, without copying it."""
        return self._core.num_row

    @property
    def mode(self) -> OnlineMode:
        """The update mode."""
        return self._mode

    def __repr__(self) -> str:
        return f"OnlineModel(rows={self.num_row()}, mode={self.mode})"
