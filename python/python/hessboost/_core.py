"""``DMatrix`` and ``Booster``, exported from :mod:`hessboost`."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _data, _hessboost
from hessboost._exceptions import HessboostError

__all__ = ["Booster", "DMatrix", "Distributions", "ImportanceType", "ModelFormat"]

Distributions = _hessboost.Distributions

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
    ``feature_names``; ``category`` columns become categorical features,
    their categories recorded and later re-coded at prediction), a scipy
    sparse matrix (absent entries are missing), or any array-like numpy
    accepts. Values equal to ``missing`` (every NaN by default) are missing;
    infinities are refused.

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
    def _for_model(
        cls, data: object, booster: Booster, missing: float, info: dict[str, object]
    ) -> DMatrix:
        """``data`` converted for ``booster``: pandas categories re-coded to
        the ones it was trained on."""
        matrix = cls.__new__(cls)
        matrix._set(
            _data.features(
                data,
                missing=missing,
                feature_names=None,
                feature_types=None,
                enable_categorical=True,
                reference=booster._categories or None,
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


def _or_empty(values: NDArray[np.float32] | None) -> NDArray[np.float32]:
    return np.empty(0, dtype=np.float32) if values is None else values


class Booster:
    """A trained gradient-boosting model, as XGBoost's ``xgboost.Booster``.

    Get one from :func:`hessboost.train`, or load a saved model with
    ``Booster(model_file)`` or :meth:`load_model`. The model itself is
    immutable: :meth:`load_model` and continued training replace it, and
    slicing returns a new booster, so a booster may be shared between
    threads.

    Args:
        model_file: A path or the bytes of a saved model in any format
            (detected from the content), or ``None`` for an empty booster
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

    def _matrix(self, data: object, base_margin: ArrayLike | None, missing: float, validate: bool) -> DMatrix:
        if isinstance(data, DMatrix):
            if base_margin is not None:
                raise HessboostError("set base_margin on the DMatrix, not in predict()")
            matrix = data
            if matrix._categories and self._categories:
                for column, categories in matrix._categories.items():
                    if self._categories.get(column, categories) != categories:
                        raise HessboostError(
                            f"the categories of feature {column} differ from training; pass the "
                            "DataFrame to predict() directly, which re-codes them"
                        )
        else:
            info = _data.info(base_margin=base_margin)
            matrix = DMatrix._for_model(data, self, missing, info)
        if validate and self._feature_names is not None and matrix._feature_names is not None:
            if matrix._feature_names != self._feature_names:
                raise HessboostError(
                    f"feature names differ: the model has {self._feature_names}, the data "
                    f"{matrix._feature_names}"
                )
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
        format: ModelFormat | None = None,
    ) -> None:
        """Replaces the model with one read from a path or bytes, in
        ``format`` or (by default) the format its content has: native
        binary or JSON, or an XGBoost JSON or UBJSON document.

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
