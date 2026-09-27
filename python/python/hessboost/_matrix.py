"""``DMatrix`` (exported from :mod:`hessboost`) and the feature boundary
every pairing of data with a model or a training matrix goes through."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _data, _hessboost
from hessboost._exceptions import HessboostError

__all__ = ["DMatrix"]


class DMatrix:
    """A dataset: features plus labels, weights, base margins, ranking
    groups and feature metadata, as XGBoost's ``xgboost.DMatrix``.

    ``data`` may be a 2-D numpy array of any numeric dtype and memory
    layout (converted to C-contiguous ``float32``, without a copy when it
    already is), a pandas ``DataFrame`` (column names become
    ``feature_names``; ``category`` columns become categorical features
    coded by position in their categories, which are recorded), a scipy
    sparse matrix (absent entries are missing), or any array-like numpy
    accepts. Values equal to ``missing`` (every NaN by default) are missing;
    infinities are refused.

    A matrix is coded when it is built, so a model or matrix it is used with
    (``predict``, ``train``'s ``evals`` and ``xgb_model``) must have its
    features: the same names and categorical features (where both record
    them) and the same categories in the same order; a mismatch raises
    :class:`HessboostError`. Codes without recorded categories (numpy data
    with ``feature_types``) are taken to be the other side's. Frames passed
    to prediction directly are re-coded to the model's categories instead.

    Args:
        data: The ``(rows, features)`` feature matrix.
        label: ``(rows,)`` labels, or ``(rows, targets)`` for multi-target
            models. NaN labels are refused.
        weight: ``(rows,)`` non-negative instance weights; for ranking data,
            one weight per query group (as XGBoost).
        base_margin: ``(rows,)`` or ``(rows, outputs)`` starting margins,
            replacing the model's intercept for these rows in training,
            evaluation and prediction.
        missing: The value marking a missing dense entry.
        feature_names: One unique name per feature (default: a frame's
            column names, else none).
        feature_types: ``"q"`` (numerical) or ``"c"`` (categorical; values
            are non-negative integer codes) per feature.
        group: Ranking query-group sizes, in row order.
        qid: Alternatively, a sorted query id per row.
        label_lower_bound: ``survival:aft`` interval lower bounds.
        label_upper_bound: ``survival:aft`` interval upper bounds
            (``inf`` for right-censored rows).
        feature_weights: Per-feature column-sampling weights.
        enable_categorical: Accept pandas ``category`` columns (as XGBoost
            requires spelling out).

    Raises:
        HessboostError: The inputs do not fit together (lengths, shapes,
            non-finite values, invalid category codes).
        TypeError: An input has an unsupported type or dtype.
    """

    __module__ = "hessboost"

    _core: _hessboost.DMatrix
    _feature_names: list[str] | None
    _feature_types: list[str] | None
    _categories: _data.Categories

    def __init__(
        self,
        data: object,
        label: ArrayLike | None = None,
        *,
        weight: ArrayLike | None = None,
        base_margin: ArrayLike | None = None,
        missing: float = np.nan,
        feature_names: Sequence[str] | None = None,
        feature_types: Sequence[str] | None = None,
        group: ArrayLike | None = None,
        qid: ArrayLike | None = None,
        label_lower_bound: ArrayLike | None = None,
        label_upper_bound: ArrayLike | None = None,
        feature_weights: ArrayLike | None = None,
        enable_categorical: bool = True,
    ) -> None:
        info = _data.info(
            label=label,
            weight=weight,
            base_margin=base_margin,
            group=group,
            qid=qid,
            label_lower_bound=label_lower_bound,
            label_upper_bound=label_upper_bound,
            feature_weights=feature_weights,
        )
        self._set(
            _data.features(
                data,
                missing=missing,
                feature_names=feature_names,
                feature_types=feature_types,
                enable_categorical=enable_categorical,
                reference=None,
                info=info,
            )
        )

    def _set(self, features: _data.Features) -> None:
        self._core = features.core
        self._feature_names = features.feature_names
        self._feature_types = features.feature_types
        self._categories = features.categories

    @classmethod
    def _coded(
        cls, data: object, categories: _data.Categories, missing: float, info: dict[str, object]
    ) -> DMatrix:
        """``data`` converted with its pandas categories re-coded to
        ``categories`` (a model's or a training matrix's; values they lack
        become missing)."""
        matrix = cls.__new__(cls)
        matrix._set(
            _data.features(
                data,
                missing=missing,
                feature_names=None,
                feature_types=None,
                enable_categorical=True,
                reference=categories or None,
                info=info,
            )
        )
        return matrix

    def num_row(self) -> int:
        """The number of rows."""
        return self._core.num_row

    def num_col(self) -> int:
        """The number of features."""
        return self._core.num_col

    @property
    def feature_names(self) -> list[str] | None:
        """The feature names, or ``None``. Assigning checks the count and
        uniqueness."""
        return None if self._feature_names is None else list(self._feature_names)

    @feature_names.setter
    def feature_names(self, names: Sequence[str] | None) -> None:
        self._feature_names = None if names is None else _data._check_names(names, self.num_col())

    @property
    def feature_types(self) -> list[str] | None:
        """``"q"``/``"c"`` per feature, or ``None`` when every feature is
        numerical by default."""
        return None if self._feature_types is None else list(self._feature_types)

    def get_label(self) -> NDArray[np.float32]:
        """The labels, ``(rows,)`` or ``(rows, targets)``; empty when unset."""
        return _or_empty(self._core.label)

    def get_weight(self) -> NDArray[np.float32]:
        """The instance weights; empty when unset."""
        return _or_empty(self._core.weight)

    def get_base_margin(self) -> NDArray[np.float32]:
        """The base margins, ``(rows,)`` or ``(rows, outputs)``; empty when
        unset."""
        return _or_empty(self._core.base_margin)

    def get_group(self) -> NDArray[np.int64]:
        """The query-group sizes; empty when unset."""
        group = self._core.group
        return np.asarray([] if group is None else group, dtype=np.int64)

    def get_label_bounds(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """The ``survival:aft`` ``(lower, upper)`` label bounds; empty when
        unset."""
        return _or_empty(self._core.label_lower_bound), _or_empty(self._core.label_upper_bound)

    def get_feature_weights(self) -> NDArray[np.float32]:
        """The column-sampling feature weights; empty when unset."""
        return _or_empty(self._core.feature_weights)

    def set_info(
        self,
        *,
        label: ArrayLike | None = None,
        weight: ArrayLike | None = None,
        base_margin: ArrayLike | None = None,
        group: ArrayLike | None = None,
        qid: ArrayLike | None = None,
        label_lower_bound: ArrayLike | None = None,
        label_upper_bound: ArrayLike | None = None,
        feature_weights: ArrayLike | None = None,
    ) -> None:
        """Replaces the given metadata (``None`` keeps the current value),
        validated as the constructor validates it. The features are copied
        once per call."""
        info = _data.info(
            label=label,
            weight=weight,
            base_margin=base_margin,
            group=group,
            qid=qid,
            label_lower_bound=label_lower_bound,
            label_upper_bound=label_upper_bound,
            feature_weights=feature_weights,
        )
        if info:
            self._core = self._core.with_info({**info, "categorical": None})

    def slice(self, rindex: ArrayLike) -> DMatrix:
        """A new matrix of the rows ``rindex`` (in that order), with their
        labels, weights, margins and bounds; query groups are dropped."""
        rows = np.ascontiguousarray(np.asarray(rindex).reshape(-1), dtype=np.int64)
        return self._with_core(self._core.select_rows(rows))

    def _with_core(self, core: _hessboost.DMatrix) -> DMatrix:
        """A matrix of ``core`` with this one's features."""
        out = DMatrix.__new__(DMatrix)
        out._core = core
        out._feature_names = self._feature_names
        out._feature_types = self._feature_types
        out._categories = self._categories
        return out

    def __repr__(self) -> str:
        return f"DMatrix(rows={self.num_row()}, features={self.num_col()})"


_RECODE_HINT = "; pass the DataFrame itself, which is re-coded to the model's categories"
"""Ends a categories error where a frame would have been re-coded."""


def _or_empty(values: NDArray[np.float32] | None) -> NDArray[np.float32]:
    return np.empty(0, dtype=np.float32) if values is None else values


def _feature_label(names: list[str] | None, column: int) -> str:
    return repr(names[column]) if names is not None and column < len(names) else str(column)


class _Features(Protocol):
    """The recorded features :func:`_check_schema` compares: a matrix's, a
    model's, or an online model's training data's."""

    _feature_names: list[str] | None
    _feature_types: list[str] | None
    _categories: _data.Categories


def _check_schema(
    reference: _Features,
    data: _Features,
    subject: str,
    against: str,
    *,
    names: bool = True,
    hint: str = "",
) -> None:
    """Refuses ``data`` whose features ``reference`` would read differently:
    other feature names (checked with ``names``), other categorical
    features, or a categorical feature whose recorded categories (their
    values and order, which define its codes) differ. Only what both sides
    record is compared: category codes without recorded categories (numpy
    data with ``feature_types``, a model loaded from a file) are taken to be
    the reference's codes. ``subject`` and ``against`` name the two sides in
    the error; ``hint`` ends a categories error."""
    ref_names, data_names = reference._feature_names, data._feature_names
    if names and ref_names is not None and data_names is not None and ref_names != data_names:
        raise HessboostError(
            f"{subject}: feature names differ from {against}: {ref_names} vs {data_names}"
        )
    label_names = ref_names if ref_names is not None else data_names
    ref_types, data_types = reference._feature_types, data._feature_types
    if ref_types is not None and data_types is not None and len(ref_types) == len(data_types):
        ref_columns = [column for column, kind in enumerate(ref_types) if kind == "c"]
        data_columns = [column for column, kind in enumerate(data_types) if kind == "c"]
        if ref_columns != data_columns:
            raise HessboostError(
                f"{subject}: categorical features differ from {against}: "
                f"[{', '.join(_feature_label(label_names, c) for c in ref_columns)}] vs "
                f"[{', '.join(_feature_label(label_names, c) for c in data_columns)}]"
            )
    for column, categories in data._categories.items():
        expected = reference._categories.get(column)
        if expected is not None and expected != categories:
            raise HessboostError(
                f"{subject}: the categories of feature {_feature_label(label_names, column)} "
                f"differ from {against}{hint}"
            )


def _matrix_for(
    data: object,
    references: Sequence[tuple[_Features, str]],
    *,
    label: ArrayLike | None = None,
    base_margin: ArrayLike | None = None,
    require_label: bool = False,
    missing: float = np.nan,
    validate_names: bool = True,
) -> DMatrix:
    """``data`` (a :class:`DMatrix` or anything it accepts) as a matrix
    every one of ``references`` reads, each given with its name in errors
    (``"the model's"``), and labelled by ``label`` when given.

    A :class:`DMatrix` is taken as it is (``base_margin`` belongs on it).
    Other input is converted with ``missing``, ``label`` and
    ``base_margin``, its pandas categories re-coded to the references' (the
    first reference's where two record a feature's; values they lack become
    missing); ``require_label`` refuses it unlabelled. Either way it is
    checked against every reference with :func:`_check_schema` (feature
    names only with ``validate_names``), since a reference without recorded
    categories matches anything. With no references this is the training
    matrix of ``data`` and ``label``."""
    if isinstance(data, DMatrix):
        if base_margin is not None:
            raise HessboostError("set base_margin on the DMatrix, not in predict()")
        matrix = data
    else:
        if require_label and label is None:
            raise HessboostError("calibration needs labels: pass label= or a labelled DMatrix")
        categories: _data.Categories = {}
        for reference, _ in reversed(references):
            categories.update(reference._categories)
        info = _data.info(label=label, base_margin=base_margin)
        matrix = DMatrix._coded(data, categories, missing, info)
        label = None
    for reference, against in references:
        _check_schema(
            reference, matrix, "the data", against, names=validate_names, hint=_RECODE_HINT
        )
    if label is None:
        return matrix
    return matrix._with_core(
        matrix._core.with_info({**_data.info(label=label), "categorical": None})
    )
