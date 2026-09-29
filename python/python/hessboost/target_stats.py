"""CatBoost-style ordered target statistics: categorical columns encoded as
smoothed target means, without a row's own label in its training encoding.

With prior ``P`` and prior weight ``a``, a training row of category ``c`` is
encoded from the rows before it in a seeded random permutation, ``(sum of
their labels in c + a·P) / (their count in c + a)``; other data (the
statistics' ``transform``) uses every training row of ``c``. Unseen
categories encode to ``P``; missing entries stay missing. See the Rust
``hessboost::data::target_stats`` docs for the parameters' trade-offs::

    from hessboost.target_stats import OrderedTargetEncoder

    encoder = OrderedTargetEncoder(seed=7)
    dtrain_encoded, stats = encoder.fit_transform(dtrain, ["city"])
    booster = hessboost.train(params, dtrain_encoded, 100)
    booster.predict(stats.transform(test_frame))

To cross-validate a model trained on encoded columns, pass the columns to
:func:`hessboost.cv` (``target_stats=``), which fits the encoder on each
fold's training rows only: encoding the whole matrix first would put every
test row's label into its own encoding.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from hessboost import _data, _hessboost
from hessboost._exceptions import HessboostError
from hessboost._matrix import DMatrix, _matrix_for

__all__ = ["Column", "FittedTargetEncoder", "OrderedTargetEncoder", "TargetKind"]

TargetKind: TypeAlias = Literal["regression", "binary"]
"""The labels an encoder accepts: any finite values (``"regression"``) or
0/1 (``"binary"``). Both encode the smoothed target mean."""

Column: TypeAlias = int | str
"""A feature by index, or by name on a matrix with feature names."""


def _column_indices(columns: Sequence[Column], names: list[str] | None) -> list[int]:
    """``columns`` as feature indices (out-of-range indices are left for
    the native encoder to refuse)."""
    if isinstance(columns, (str, int)) or not isinstance(columns, Sequence):
        raise TypeError(
            f"columns must be a sequence of feature indices or names, got {type(columns).__name__}"
        )
    return [_column_index(column, names) for column in columns]


def _column_index(column: Column, names: list[str] | None) -> int:
    if isinstance(column, (int, np.integer)) and not isinstance(column, bool):
        if column < 0:
            raise HessboostError(f"feature index {column} is negative")
        return int(column)
    if isinstance(column, str):
        if names is None:
            raise HessboostError(f"feature {column!r} named, but the data has no feature names")
        if column not in names:
            raise HessboostError(f"unknown feature {column!r}")
        return names.index(column)
    raise TypeError(f"a column must be a feature index or name, got {type(column).__name__}")


def _encoded(source: DMatrix, core: _hessboost.DMatrix, columns: list[int]) -> DMatrix:
    """A matrix of ``core``: ``source`` with ``columns`` numerical."""
    out = source._with_core(core)
    if source._feature_types is not None:
        out._feature_types = [
            "q" if column in columns else kind for column, kind in enumerate(source._feature_types)
        ]
    out._categories = {
        column: values for column, values in source._categories.items() if column not in columns
    }
    return out


class OrderedTargetEncoder:
    """An unfitted ordered target-statistics encoder.

    Args:
        seed: Seed of the permutations.
        prior_weight: The prior's pseudo-count ``a``; must be ``> 0``.
        prior: A fixed prior ``P`` (default: the mean training label). The
            default gives every row ``1/n`` of its own label; a fixed prior
            keeps the training encoding strictly free of it.
        permutations: The number of permutations the training encoding
            averages (at least 1). A few (2--4) smooth it; many converge to
            a leave-one-out mean, which trees can exploit on the training
            rows.
        target: Which labels are accepted (see :data:`TargetKind`);
            multiclass labels are not supported: fit one encoder per class
            on 0/1 indicators (``label=``).

    Raises:
        HessboostError: A setting is out of range.
    """

    __module__ = "hessboost.target_stats"

    _core: _hessboost.OrderedTargetEncoder

    def __init__(
        self,
        *,
        seed: int = 0,
        prior_weight: float = 1.0,
        prior: float | None = None,
        permutations: int = 1,
        target: TargetKind = "regression",
    ) -> None:
        self._core = _hessboost.OrderedTargetEncoder(
            prior_weight=prior_weight,
            prior=prior,
            permutations=permutations,
            seed=seed,
            target=target,
        )
        self._repr = (
            f"OrderedTargetEncoder(seed={seed}, prior_weight={prior_weight}, prior={prior}, "
            f"permutations={permutations}, target={target!r})"
        )

    def fit_transform(
        self, dtrain: DMatrix, columns: Sequence[Column], label: ArrayLike | None = None
    ) -> tuple[DMatrix, FittedTargetEncoder]:
        """Fits statistics on ``dtrain`` and encodes its categorical
        ``columns``.

        Args:
            dtrain: The training matrix.
            columns: The categorical features to encode, by index or name.
            label: The per-row target the statistics average (``(rows,)``,
                finite). Default: ``dtrain``'s labels, which must then be
                one per row; pass one column of a multi-target matrix, or a
                class's 0/1 indicator, here. ``dtrain``'s own labels are kept
                either way.

        Returns:
            ``dtrain`` with ``columns`` replaced by their ordered encodings
            (now numerical; everything else kept) for training, and the
            fitted statistics for evaluation and test data.

        Raises:
            HessboostError: A column is unknown, repeated, or not
                categorical, or the labels are missing, of the wrong length,
                or outside ``target``.
        """
        if not isinstance(dtrain, DMatrix):
            raise TypeError(f"dtrain must be a DMatrix, got {type(dtrain).__name__}")
        indices = _column_indices(columns, dtrain._feature_names)
        labels = None
        if label is not None:
            labels = np.asarray(label, dtype=np.float32)
            if labels.ndim != 1:
                raise HessboostError(f"label must be one value per row, got shape {labels.shape}")
            labels = np.ascontiguousarray(labels)
        core, fitted = self._core.fit_transform(dtrain._core, indices, labels)
        encoded = _encoded(dtrain, core, indices)
        return encoded, FittedTargetEncoder._wrap(fitted, dtrain)

    def __repr__(self) -> str:
        return self._repr


class FittedTargetEncoder:
    """Ordered target statistics fitted on a training matrix: each encoded
    column's smoothed target mean per category over every training row.
    Build with :meth:`OrderedTargetEncoder.fit_transform`."""

    __module__ = "hessboost.target_stats"

    _core: _hessboost.FittedTargetEncoder
    _feature_names: list[str] | None
    _feature_types: list[str] | None
    _categories: _data.Categories

    def __init__(self) -> None:
        raise TypeError("use OrderedTargetEncoder.fit_transform(...)")

    @classmethod
    def _wrap(cls, core: _hessboost.FittedTargetEncoder, dtrain: DMatrix) -> FittedTargetEncoder:
        self = object.__new__(cls)
        self._core = core
        self._feature_names = dtrain._feature_names
        self._feature_types = dtrain._feature_types
        self._categories = dict(dtrain._categories)
        return self

    def transform(self, data: object, *, missing: float = np.nan) -> DMatrix:
        """``data`` (a :class:`~hessboost.DMatrix`, or anything it accepts,
        with ``missing`` marking missing dense entries) with the encoded
        columns replaced by the fitted statistics. It must have the training
        matrix's features; a DataFrame is re-coded to its categories (a
        category training never saw encodes to the prior, a null stays
        missing), and data without feature types (a numpy array) takes the
        training matrix's.

        Raises:
            HessboostError: ``data``'s features differ from the training
                matrix's.
        """
        matrix = _matrix_for(
            data,
            [(self, "the encoder's training data's")],
            missing=missing,
            unseen=frozenset(self.columns),
        )
        if matrix._feature_types is None and self._feature_types is not None:
            categorical = [column for column, kind in enumerate(self._feature_types) if kind == "c"]
            matrix = matrix._with_core(matrix._core.with_info({"categorical": categorical}))
            matrix._feature_types = self._feature_types
        return _encoded(matrix, self._core.transform(matrix._core), self.columns)

    def encode(self, column: Column, code: int) -> float:
        """The encoding of category ``code`` (its integer code, a frame
        category's position in its categories) in ``column``: the prior for
        a category unseen in training.

        Raises:
            HessboostError: ``column`` is not encoded.
        """
        index = _column_index(column, self._feature_names)
        value = self._core.encode(index, int(code))
        if value is None:
            raise HessboostError(f"feature {column!r} is not target-encoded")
        return value

    @property
    def prior(self) -> float:
        """The prior ``P``, which unseen categories encode to."""
        return self._core.prior

    @property
    def columns(self) -> list[int]:
        """The encoded feature indices, in the order they were given."""
        return self._core.columns

    def __repr__(self) -> str:
        return f"FittedTargetEncoder(columns={self.columns}, prior={self.prior})"
