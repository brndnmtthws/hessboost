"""``DMatrix`` and ``Booster``, exported from :mod:`hessboost`."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _data, _hessboost
from hessboost._exceptions import HessboostError

if TYPE_CHECKING:
    from hessboost.inference import BoulevardInfo

__all__ = [
    "Booster",
    "DMatrix",
    "Distributions",
    "ImportanceType",
    "ModelFormat",
    "Uncertainty",
]

Distributions = _hessboost.Distributions


@dataclass(frozen=True)
class Uncertainty:
    """A virtual ensemble's uncertainty decomposition
    (:meth:`Booster.predict_uncertainty`), after CatBoost.

    ``knowledge``, ``data`` and ``total`` are ``(rows,)``, or ``(rows, K)``
    for multi-output regression (one per output) and multi-label
    classification (one per label column). ``mean`` has its own width: a
    multiclass model's is ``(rows, classes)`` while its uncertainties are
    ``(rows,)``.
    """

    mean: NDArray[np.float64]
    """The members' mean prediction: predictions for regression, predicted
    means for ``dist:*``, probabilities for classification (one per class
    for ``multi:*``, so ``(rows, classes)``)."""
    knowledge: NDArray[np.float64]
    """Knowledge (epistemic) uncertainty: the variance of the members'
    predictions (means for ``dist:*``) for regression, the mutual
    information (total minus data) for classification."""
    data: NDArray[np.float64] | None
    """Data (aleatoric) uncertainty: the mean predicted variance for
    ``dist:*``, the mean member entropy for classification; ``None`` for
    plain regression."""
    total: NDArray[np.float64] | None
    """``data + knowledge`` for ``dist:*``; the entropy of the mean
    probabilities for classification; ``None`` for plain regression."""


ModelFormat: TypeAlias = Literal["binary", "json", "xgboost-json", "xgboost-ubjson"]
"""A model file format:

* ``"binary"``: hessboost's native binary format (compressed, checksummed,
  lossless; readable by every later release).
* ``"json"``: hessboost's native JSON (the same model as readable text).
* ``"xgboost-json"`` / ``"xgboost-ubjson"``: XGBoost 3.x's JSON and UBJSON
  model documents, loadable by XGBoost itself. Export refuses what XGBoost
  cannot load (``gblinear``, ``linear_tree`` leaves, ``dist:*`` objectives,
  custom objectives).
"""

ImportanceType: TypeAlias = Literal["weight", "gain", "total_gain", "cover", "total_cover"]
"""XGBoost's ``importance_type``: split counts, average or total gain,
average or total cover (Hessian)."""

PathLike: TypeAlias = str | os.PathLike[str]


def _format_for(path: PathLike, format: ModelFormat | None) -> ModelFormat:
    """``format``, or the one ``path``'s extension names: ``.json`` native
    JSON, ``.ubj`` XGBoost UBJSON, anything else native binary."""
    if format is not None:
        return format
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    if suffix == ".json":
        return "json"
    if suffix == ".ubj":
        return "xgboost-ubjson"
    return "binary"


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
        out = DMatrix.__new__(DMatrix)
        out._core = self._core.select_rows(rows)
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


class Booster:
    """A trained gradient-boosting model, as XGBoost's ``xgboost.Booster``.

    Get one from :func:`hessboost.train`, or load a saved model with
    ``Booster(model_file)`` or :meth:`load_model`. The model itself is
    immutable: :meth:`load_model` and continued training replace it, and
    slicing returns a new booster, so a booster may be shared between
    threads.

    Args:
        model_file: A path or the bytes of a saved model in any format
            (detected from the content), including LightGBM text models, or ``None`` for an empty booster
            to :meth:`load_model` into.
    """

    __module__ = "hessboost"

    _core: _hessboost.Booster | None
    _feature_names: list[str] | None
    _feature_types: list[str] | None
    _categories: _data.Categories
    best_score: float | None
    """With early stopping, the watched metric at :attr:`best_iteration`."""

    def __init__(self, model_file: PathLike | bytes | bytearray | memoryview | None = None) -> None:
        self._core = None
        self._feature_names = None
        self._feature_types = None
        self._categories = {}
        self.best_score = None
        if model_file is not None:
            self.load_model(model_file)

    @classmethod
    def _wrap(
        cls,
        core: _hessboost.Booster,
        feature_names: list[str] | None,
        feature_types: list[str] | None,
        categories: _data.Categories,
    ) -> Booster:
        booster = cls()
        booster._core = core
        booster._feature_names = feature_names
        booster._feature_types = feature_types
        booster._categories = categories
        return booster

    @property
    def _model(self) -> _hessboost.Booster:
        if self._core is None:
            raise HessboostError("the booster holds no model: train one or call load_model")
        return self._core

    @property
    def feature_names(self) -> list[str] | None:
        """The training features' names, if known. Not stored in model
        files (pickling keeps them); assign them after loading if needed."""
        return None if self._feature_names is None else list(self._feature_names)

    @feature_names.setter
    def feature_names(self, names: Sequence[str] | None) -> None:
        self._feature_names = (
            None if names is None else _data._check_names(names, self.num_features())
        )

    @property
    def feature_types(self) -> list[str] | None:
        """``"q"``/``"c"`` per training feature, if known."""
        return None if self._feature_types is None else list(self._feature_types)

    @property
    def objective(self) -> str:
        """The objective the model was trained with, e.g. ``"binary:logistic"``."""
        return self._model.objective

    @property
    def boulevard(self) -> BoulevardInfo | None:
        """How a ``booster = boulevard`` model was trained, or ``None`` (see
        :mod:`hessboost.inference`)."""
        from hessboost.inference import BoulevardInfo

        return BoulevardInfo._from_core(self._model.boulevard)

    @property
    def best_iteration(self) -> int | None:
        """The best iteration early stopping chose, or ``None``. Prediction
        uses iterations ``0..best_iteration`` by default."""
        return self._model.best_iteration

    def num_boosted_rounds(self) -> int:
        """The number of boosting iterations."""
        return self._model.num_boosted_rounds

    def num_features(self) -> int:
        """The number of features the model takes."""
        return self._model.num_features

    def _matrix(
        self, data: object, base_margin: ArrayLike | None, missing: float, validate: bool
    ) -> DMatrix:
        """``data`` as a matrix for this model: other input converted with its
        pandas categories re-coded to the model's, and either checked with
        :func:`_check_schema` (feature names only with ``validate``)."""
        if isinstance(data, DMatrix):
            if base_margin is not None:
                raise HessboostError("set base_margin on the DMatrix, not in predict()")
            matrix = data
        else:
            info = _data.info(base_margin=base_margin)
            matrix = DMatrix._coded(data, self._categories, missing, info)
        _check_schema(self, matrix, "the data", "the model's", names=validate, hint=_RECODE_HINT)
        return matrix

    @staticmethod
    def _range(iteration_range: tuple[int, int] | None) -> tuple[int, int] | None:
        if iteration_range is None:
            return None
        begin, end = iteration_range
        if begin < 0 or end < 0:
            raise HessboostError(f"iteration_range {iteration_range} must be non-negative")
        return int(begin), int(end)

    def predict(
        self,
        data: object,
        *,
        output_margin: bool = False,
        pred_leaf: bool = False,
        pred_contribs: bool = False,
        pred_interactions: bool = False,
        iteration_range: tuple[int, int] | None = None,
        validate_features: bool = True,
        base_margin: ArrayLike | None = None,
        missing: float = np.nan,
    ) -> NDArray[Any]:
        """Predicts every row of ``data`` (a :class:`DMatrix` or anything
        its constructor accepts).

        Returns, for ``rows`` rows, ``F`` features and ``K`` outputs (shapes
        drop the ``K`` axis for single-output models):

        * default: the objective's predictions (probabilities for
          ``binary:logistic`` and ``multi:softprob``, class indices for
          ``multi:softmax``), ``(rows,)`` or ``(rows, K)``;
        * ``output_margin``: raw margins, ``(rows,)`` or ``(rows, K)``;
        * ``pred_contribs``: SHAP values, ``(rows, [K,] F + 1)``, bias last;
          each row sums to its margin;
        * ``pred_interactions``: SHAP interaction values,
          ``(rows, [K,] F + 1, F + 1)``;
        * ``pred_leaf``: the leaf index reached in every tree,
          ``(rows, trees)`` ``int32``.

        Args:
            iteration_range: ``(begin, end)`` boosting iterations, ``end=0``
                meaning the last; ``None`` uses iterations through
                :attr:`best_iteration` (all without early stopping).
                Contribution, interaction and leaf ranges start at 0.
            validate_features: Refuse data whose feature names differ from
                the model's.
            base_margin: Starting margins for array input (a
                :class:`DMatrix` carries its own).
            missing: The missing-value marker for array input.
        """
        flags = sum(map(bool, (output_margin, pred_leaf, pred_contribs, pred_interactions)))
        if flags > 1:
            raise HessboostError(
                "output_margin, pred_leaf, pred_contribs and pred_interactions are exclusive"
            )
        model = self._model
        matrix = self._matrix(data, base_margin, missing, validate_features)._core
        iterations = self._range(iteration_range)
        if pred_leaf:
            return model.predict_leaf(matrix, iterations)
        kind = (
            "margin"
            if output_margin
            else "contribs"
            if pred_contribs
            else "interactions"
            if pred_interactions
            else "value"
        )
        return model.predict(matrix, kind, iterations)

    def predict_distribution(
        self,
        data: object,
        *,
        iteration_range: tuple[int, int] | None = None,
        base_margin: ArrayLike | None = None,
        missing: float = np.nan,
    ) -> Distributions:
        """The conditional distribution a ``dist:*`` model predicts for
        every row of ``data``: means, variances, quantiles, intervals, CDF,
        log density and CRPS, vectorized over rows.

        Raises:
            HessboostError: The model's objective is not ``dist:*``.
        """
        model = self._model
        matrix = self._matrix(data, base_margin, missing, True)._core
        return model.predict_distribution(matrix, self._range(iteration_range))

    def predict_virtual_ensembles(
        self,
        data: object,
        count: int = 10,
        *,
        output_margin: bool = False,
        base_margin: ArrayLike | None = None,
        missing: float = np.nan,
    ) -> tuple[NDArray[np.float32], list[int]]:
        """Predicts ``data`` with a virtual ensemble of ``count`` members
        (CatBoost's ``virtual_ensembles_predict``): the models after several
        iterations of the second half of this model's iterations, rebuilt
        exactly from its model shrinkage. Meant for models trained with
        ``posterior_sampling``; any tree model is accepted.

        Returns:
            The members' predictions (``output_margin``: raw margins),
            member-major: ``(count, rows)``, or ``(count, rows, K)`` with
            ``K`` values per row as :meth:`predict` returns them; and each
            member's iteration count, ascending (the last is the whole
            model).

        Raises:
            HessboostError: ``count`` is 0, the model has too few iterations
                for ``count`` members (about ``2 * count``), or it is a
                ``gblinear`` model.
        """
        if count < 1:
            raise HessboostError(f"count must be at least 1, got {count}")
        matrix = self._matrix(data, base_margin, missing, True)._core
        return self._model.predict_virtual_ensembles(matrix, count, output_margin)

    def predict_uncertainty(
        self,
        data: object,
        count: int = 10,
        *,
        base_margin: ArrayLike | None = None,
        missing: float = np.nan,
    ) -> Uncertainty:
        """The knowledge, data and total uncertainty of every row of
        ``data`` under a virtual ensemble of ``count`` members
        (:meth:`predict_virtual_ensembles`), decomposed as CatBoost does:
        variances for regression and ``dist:*``, entropies for
        classification (see :class:`Uncertainty`).

        Raises:
            HessboostError: The objective has no decomposition (ranking,
                ``binary:hinge``, custom objectives), or as
                :meth:`predict_virtual_ensembles`.
        """
        if count < 1:
            raise HessboostError(f"count must be at least 1, got {count}")
        matrix = self._matrix(data, base_margin, missing, True)._core
        mean, knowledge, aleatoric, total = self._model.predict_uncertainty(matrix, count)
        return Uncertainty(mean=mean, knowledge=knowledge, data=aleatoric, total=total)

    def get_score(self, importance_type: ImportanceType = "weight") -> dict[str, float]:
        """Feature importance by feature name (``f0``, ``f1``, ... without
        names), for every feature used in a split."""
        scores = self._model.feature_importance(importance_type)
        names = self._feature_names
        return {
            (names[index] if names is not None else f"f{index}"): value
            for index, value in scores.items()
        }

    def save_raw(self, format: ModelFormat = "binary") -> bytes:
        """The model encoded as ``format``."""
        return self._model.save(format)

    def save_model(self, fname: PathLike, *, format: ModelFormat | None = None) -> None:
        """Writes the model to ``fname`` as ``format``, by default the one
        the extension names: ``.json`` native JSON, ``.ubj`` XGBoost UBJSON,
        anything else native binary. Pass ``format="xgboost-json"`` for a
        JSON file XGBoost can load.

        Feature names, categories and :attr:`best_score` are not part of
        the model formats; pickle the booster to keep them.
        """
        data = self.save_raw(_format_for(fname, format))
        with open(fname, "wb") as file:
            file.write(data)

    def load_model(
        self,
        fname: PathLike | bytes | bytearray | memoryview,
        *,
        format: ModelFormat | Literal["lightgbm"] | None = None,
    ) -> None:
        """Replaces the model with one read from a path or bytes, in
        ``format`` or (by default) the format its content has: native
        binary or JSON, an XGBoost JSON or UBJSON document, or a LightGBM
        4.x text model (``lightgbm.Booster.save_model``; import only). An
        imported LightGBM model predicts LightGBM's values for inputs with
        missing values as ``NaN`` and categorical features as non-negative
        codes; LightGBM models with no exact equivalent (``zero_as_missing``
        splits a threshold cannot express, ``sigmoid`` other than 1, random
        forests, ...) raise :class:`ModelFormatError`. See the Rust crate's
        "LightGBM import" docs for the mapping.

        Raises:
            ModelFormatError: The content is not a valid model.
            OSError: The file cannot be read.
        """
        if isinstance(fname, (bytes, bytearray, memoryview)):
            data = bytes(fname)
        else:
            with open(fname, "rb") as file:
                data = file.read()
        self._core = _hessboost.Booster.load(data, format or "auto")
        self._feature_names = None
        self._feature_types = None
        self._categories = {}
        self.best_score = None

    @overload
    def __getitem__(self, key: int) -> Booster: ...
    @overload
    def __getitem__(self, key: slice) -> Booster: ...
    def __getitem__(self, key: int | slice) -> Booster:
        """``booster[i]`` holds iteration ``i``; ``booster[begin:end:step]``
        every ``step``-th iteration of the range (Python slice semantics,
        negative indices included). The slice has no ``best_iteration``."""
        rounds = self.num_boosted_rounds()
        if isinstance(key, slice):
            begin, end, step = key.indices(rounds)
            if step < 1:
                raise HessboostError("booster slices need a positive step")
        elif isinstance(key, (int, np.integer)):
            index = int(key) + rounds if key < 0 else int(key)
            if not 0 <= index < rounds:
                raise IndexError(f"iteration {key} out of range for {rounds} iterations")
            begin, end, step = index, index + 1, 1
        else:
            raise TypeError(f"booster indices must be int or slice, not {type(key).__name__}")
        core = self._model.slice(begin, end, step)
        return Booster._wrap(core, self._feature_names, self._feature_types, self._categories)

    def copy(self) -> Booster:
        """An independent booster holding the same model."""
        booster = Booster._wrap(
            self._model, self._feature_names, self._feature_types, dict(self._categories)
        )
        booster.best_score = self.best_score
        return booster

    def __copy__(self) -> Booster:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, object]) -> Booster:
        return self.copy()

    def __getstate__(self) -> dict[str, object]:
        return {
            "model": None if self._core is None else self._core.save("binary"),
            "feature_names": self._feature_names,
            "feature_types": self._feature_types,
            "categories": self._categories,
            "best_score": self.best_score,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        model = state["model"]
        self._core = None if model is None else _hessboost.Booster.load(model, "binary")
        self._feature_names = state["feature_names"]
        self._feature_types = state["feature_types"]
        self._categories = state["categories"]
        self.best_score = state["best_score"]

    def __repr__(self) -> str:
        if self._core is None:
            return "Booster(empty)"
        core = self._core
        best = "" if core.best_iteration is None else f", best_iteration={core.best_iteration}"
        return (
            f"Booster(objective={core.objective!r}, rounds={core.num_boosted_rounds}, "
            f"features={core.num_features}, outputs={core.num_outputs}{best})"
        )
