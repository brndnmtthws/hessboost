"""Explainable boosting machines (GA2M): shape functions of a model trained
with ``{"booster": "ebm", ...}``.

An EBM is a sum of per-term functions: one of each feature (main terms)
and, with ``ebm_interactions = k``, of the ``k`` feature pairs FAST ranks
highest. :func:`shape_functions` merges each term's trees into one
piecewise-constant function on the grid its splits cut (categorical
features by category), with a last cell for missing values on every axis::

    from hessboost.ebm import shape_functions

    booster = hessboost.train({"booster": "ebm", "eta": 0.04, "ebm_interactions": 1,
                               "max_leaves": 3, "grow_policy": "lossguide"}, dtrain, 300)
    shapes = shape_functions(booster)
    main = shapes.terms[0]
    main.axes[0].edges, main.values          # thresholds and per-cell values
    shapes.intercept + sum(t.value(x[t.features]) for t in shapes.terms)  # the margin

With ``ebm_boulevard`` the model also supports confidence bands on its
shape functions (:class:`hessboost.inference.EbmInference`). See the Rust
crate's ``hessboost::ebm`` documentation for the algorithms (Lou et al.,
KDD 2012/2013; InterpretML; Fang, Tan, Pipping & Hooker, AISTATS 2026).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from hessboost import _hessboost
from hessboost._core import Booster

__all__ = [
    "CategoricalAxis",
    "EbmBoulevard",
    "EbmInfo",
    "NumericAxis",
    "ShapeFunctions",
    "TermAxis",
    "TermShape",
    "shape_functions",
]


@dataclass(frozen=True)
class NumericAxis:
    """A numerical feature's cells: ``(-inf, e_0), [e_0, e_1), ...,
    [e_last, inf)``, then the missing cell."""

    __module__ = "hessboost.ebm"

    edges: NDArray[np.float32]
    """The sorted thresholds the term's trees split at."""

    @property
    def cells(self) -> int:
        """Number of cells, the missing cell included."""
        return len(self.edges) + 2


@dataclass(frozen=True)
class CategoricalAxis:
    """A categorical feature's cells: one per listed category code, one
    "other" cell for every category no split sends left, then the missing
    cell."""

    __module__ = "hessboost.ebm"

    categories: tuple[int, ...]
    """The category codes the term's trees split on, ascending."""

    @property
    def cells(self) -> int:
        """Number of cells, the missing cell included."""
        return len(self.categories) + 2


TermAxis: TypeAlias = NumericAxis | CategoricalAxis
"""The cells along one feature of a term."""


def _axis(kind: str, payload: Any) -> TermAxis:
    if kind == "numeric":
        return NumericAxis(np.asarray(payload, dtype=np.float32))
    return CategoricalAxis(tuple(int(c) for c in payload))


class TermShape:
    """One term's shape function, centered on the training rows:
    :attr:`values` has one array axis per feature, with the cells of
    :attr:`axes` (missing cell last along each)."""

    __module__ = "hessboost.ebm"

    _core: _hessboost.TermShape
    features: tuple[int, ...]
    """The term's features (one, or two ascending)."""
    axes: tuple[TermAxis, ...]
    """The cells along each feature."""
    values: NDArray[np.float64]
    """The shape's value on every cell."""

    def __init__(self) -> None:
        raise TypeError("shapes come from shape_functions(...)")

    @classmethod
    def _wrap(cls, core: _hessboost.TermShape) -> TermShape:
        self = object.__new__(cls)
        self._core = core
        self.features = tuple(core.features)
        self.axes = tuple(_axis(kind, payload) for kind, payload in core.axes)
        self.values = core.values
        return self

    def cell(self, x: Sequence[float]) -> tuple[int, ...]:
        """The grid index (one per feature) of the cell holding the feature
        values ``x`` (NaN is missing; a category code no split names is the
        "other" cell)."""
        flat = self._core.cell([float(v) for v in x])
        return tuple(int(i) for i in np.unravel_index(flat, self.values.shape))

    def value(self, x: Sequence[float]) -> float:
        """The shape's value at the feature values ``x``."""
        return self._core.value([float(v) for v in x])

    def __repr__(self) -> str:
        return f"TermShape(features={self.features}, cells={self.values.shape})"


@dataclass(frozen=True)
class ShapeFunctions:
    """Every term's shape and the intercept: ``intercept + sum of shapes``
    is the model's margin."""

    __module__ = "hessboost.ebm"

    intercept: float
    """``base_score`` plus every term's training mean."""
    terms: tuple[TermShape, ...]
    """One shape per term: the main terms (by feature), then the pairs."""


@dataclass(frozen=True)
class EbmBoulevard:
    """The Boulevard settings of an ``ebm_boulevard`` fit."""

    __module__ = "hessboost.ebm"

    learning_rate: float
    """The learning rate ``lambda`` (``eta``)."""
    subsample: float
    """The row subsample ratio."""
    reg_lambda: float
    """The L2 leaf penalty."""


@dataclass(frozen=True)
class EbmInfo:
    """How a ``booster = ebm`` model was trained
    (:attr:`hessboost.Booster.ebm`)."""

    __module__ = "hessboost.ebm"

    terms: tuple[tuple[int, ...], ...]
    """The features of each term."""
    tree_terms: tuple[int, ...]
    """The term of each tree."""
    term_means: tuple[float, ...]
    """Each term's raw training mean, moved into the intercept."""
    boulevard: EbmBoulevard | None
    """The Boulevard settings, or ``None`` for a classic EBM."""

    @classmethod
    def _from_core(cls, info: dict[str, Any] | None) -> EbmInfo | None:
        if info is None:
            return None
        boulevard = info["boulevard"]
        return cls(
            terms=tuple(tuple(t) for t in info["terms"]),
            tree_terms=tuple(info["tree_terms"]),
            term_means=tuple(info["term_means"]),
            boulevard=None if boulevard is None else EbmBoulevard(**boulevard),
        )


def shape_functions(booster: Booster) -> ShapeFunctions:
    """The intercept and every term's shape function of the EBM
    ``booster``.

    Raises:
        HessboostError: ``booster`` is not an EBM.
    """
    intercept, terms = _hessboost.shape_functions(booster._model)
    return ShapeFunctions(intercept, tuple(TermShape._wrap(t) for t in terms))
