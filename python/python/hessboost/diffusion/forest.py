"""ForestFlow and ForestDiffusion: synthetic tabular rows and missing-value
imputation with boosted trees (Jolicoeur-Martineau, Fatras and Kilian,
AISTATS 2024).

A :class:`ForestModel` learns the joint distribution of a table's columns,
optionally per class of a label, with one GBDT per noise level (and per
class). It then generates new rows, and a diffusion model also imputes the
missing (NaN) entries of rows, any number of times, keeping the observed
ones::

    from hessboost.diffusion.forest import ForestModel, ForestParams

    params = ForestParams(column_kinds=["continuous", "integer", "categorical"])
    model = ForestModel.fit(params, X)
    values, labels = model.generate(1000, seed=0)  # (1000, columns) float32, None

    model = ForestModel.fit(ForestParams.diffusion(), X_with_nans)
    filled = model.impute(X_with_nans, n_imputations=5, seed=0)  # (5, rows, columns)

Columns are encoded by their :data:`ColumnKind` (categorical columns as
dummies against their smallest value), independently of a
:class:`~hessboost.DMatrix`'s feature types: generated values are in the
data's own coding (a pandas category column's codes, for example). Decoding
rounds integer columns and clips every column to its training range.

The configuration is a frozen dataclass whose ``training`` mapping holds
XGBoost parameters, read as :func:`hessboost.train` reads them; the
presets' mappings hold every setting (the reference's XGBoost settings:
``hist``, depth 7, ``eta = 0.3``, ``lambda = 0``). Generation and
imputation are deterministic for a given model, input and seed at any
thread count. :meth:`ForestModel.fit` releases the GIL but cannot be
interrupted: Ctrl-C takes effect once it returns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _data, _hessboost
from hessboost._core import _RECODE_HINT, DMatrix, _check_schema
from hessboost.diffusion import (
    PathLike,
    _choice,
    _count,
    _number,
    _store_count,
    _store_training,
)

__all__ = [
    "ColumnKind",
    "Diffusion",
    "ForestMethod",
    "ForestModel",
    "ForestParams",
    "Repaint",
]


@dataclass(frozen=True)
class Diffusion:
    """ForestDiffusion: the VP SDE with ``β(t) = β_min + (β_max - β_min) t``
    (``0 < beta_min < beta_max``); the GBDTs predict the noise. Generates
    and imputes."""

    beta_min: float = 0.1
    beta_max: float = 8.0

    def __post_init__(self) -> None:
        _number(self, "beta_min")
        _number(self, "beta_max")


ForestMethod: TypeAlias = Literal["flow"] | Diffusion
"""``"flow"`` (ForestFlow: flow matching on the straight path from noise to
data, generation only) or :class:`Diffusion`."""

ColumnKind: TypeAlias = Literal["continuous", "integer", "categorical"]
"""How a column is encoded and decoded: a real value; an ordinal or binary
value (rounded); or categories (dummy-coded against the smallest, decoded
to the most likely one)."""

_COLUMN_KINDS = ("continuous", "integer", "categorical")


@dataclass(frozen=True)
class Repaint:
    """RePaint resampling (Lugmayr et al., 2022) for
    :meth:`ForestModel.impute`: every ``ceil(jump · n_t)`` steps the
    missing entries are pushed back up the forward SDE and re-denoised,
    ``resample - 1`` times per segment.

    Args:
        resample: Passes over each segment (``>= 1``; 1 disables it).
        jump: Segment length as a fraction of ``n_t``, in ``(0, 1]``.
    """

    resample: int = 5
    jump: float = 0.1

    def __post_init__(self) -> None:
        _store_count(self, "resample")
        _number(self, "jump")


def _default_training() -> Mapping[str, Any]:
    training: dict[str, Any] = _hessboost.ForestParams.preset("default")["training"]
    return training


def _method_json(method: ForestMethod) -> str:
    _choice("method", method, ("flow",), (Diffusion,))
    if isinstance(method, Diffusion):
        return json.dumps({"diffusion": {"beta_min": method.beta_min, "beta_max": method.beta_max}})
    return json.dumps(method)


def _method(text: str) -> ForestMethod:
    value = json.loads(text)
    if value == "flow":
        return "flow"
    return Diffusion(**value["diffusion"])


@dataclass(frozen=True)
class ForestParams:
    """The configuration of :meth:`ForestModel.fit`, validated when built.
    The defaults are :meth:`default`'s.

    Args:
        method: ``"flow"`` or :class:`Diffusion`.
        n_t: Noise levels, and GBDTs per class (``>= 2``).
        duplicate_k: Noisy copies of each row (``> 0``).
        column_kinds: One :data:`ColumnKind` per column, or ``None`` for
            all continuous.
        training: Every GBDT's XGBoost parameters (objective
            ``reg:squarederror``, ``scale_pos_weight`` 1).
        num_boost_round: Boosting rounds of each GBDT (``> 0``).
        seed: Seed of the training noise.

    Raises:
        HessboostError: ``n_t < 2``, a zero count, invalid ``β``, an unknown
            column kind, or a refused ``training`` mapping.
        TypeError: A field has the wrong type.
    """

    method: ForestMethod = "flow"
    n_t: int = 50
    duplicate_k: int = 100
    column_kinds: Sequence[ColumnKind] | None = None
    training: Mapping[str, Any] = field(default_factory=_default_training)
    num_boost_round: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("n_t", "duplicate_k", "num_boost_round", "seed"):
            _store_count(self, name)
        kinds = self.column_kinds
        if kinds is not None:
            if isinstance(kinds, str) or not isinstance(kinds, Sequence):
                raise TypeError(
                    "column_kinds must be a sequence of column kinds or None, got "
                    f"{type(kinds).__name__}"
                )
            for kind in kinds:
                _choice("column kind", kind, _COLUMN_KINDS, ())
            object.__setattr__(self, "column_kinds", tuple(kinds))
        _store_training(self, "training")
        self._build()

    def _build(self) -> _hessboost.ForestParams:
        """The validated native configuration (built again by every
        :meth:`ForestModel.fit`, so it sees the fields as they are)."""
        return _hessboost.ForestParams(
            {
                "method": _method_json(self.method),
                "n_t": self.n_t,
                "duplicate_k": self.duplicate_k,
                "column_kinds": None if self.column_kinds is None else list(self.column_kinds),
                "training": _hessboost.Params(self.training),
                "num_boost_round": self.num_boost_round,
                "seed": self.seed,
            }
        )

    @classmethod
    def _preset(cls, name: Literal["default", "diffusion"]) -> Self:
        description = _hessboost.ForestParams.preset(name)
        return cls(
            method=_method(description["method"]),
            n_t=description["n_t"],
            duplicate_k=description["duplicate_k"],
            column_kinds=description["column_kinds"],
            training=description["training"],
            num_boost_round=description["num_boost_round"],
            seed=description["seed"],
        )

    @classmethod
    def default(cls) -> Self:
        """The reference's ForestFlow configuration: 50 noise levels, 100
        copies of each row, 100 rounds of depth-7 ``hist`` trees with
        ``eta = 0.3`` and ``lambda = 0``."""
        return cls._preset("default")

    @classmethod
    def diffusion(cls) -> Self:
        """The reference's ForestDiffusion configuration: :meth:`default`
        with :class:`Diffusion` (``β`` from 0.1 to 8)."""
        return cls._preset("diffusion")


class ForestModel:
    """A fitted ForestFlow / ForestDiffusion model. Build one with
    :meth:`fit` or a loader; it is immutable and may be shared between
    threads."""

    _core: _hessboost.ForestModel
    _feature_names: list[str] | None
    _feature_types: list[str] | None
    _categories: _data.Categories

    def __init__(self) -> None:
        raise TypeError("use ForestModel.fit(...) or a loader such as ForestModel.from_bytes")

    @classmethod
    def _wrap(
        cls,
        core: _hessboost.ForestModel,
        feature_names: list[str] | None = None,
        feature_types: list[str] | None = None,
        categories: _data.Categories | None = None,
    ) -> Self:
        self = object.__new__(cls)
        self._core = core
        self._feature_names = feature_names
        self._feature_types = feature_types
        self._categories = {} if categories is None else categories
        return self

    @staticmethod
    def _labelled(matrix: DMatrix, label: ArrayLike | None) -> _hessboost.DMatrix:
        if label is None:
            return matrix._core
        return matrix._core.with_info({**_data.info(label=label), "categorical": None})

    @classmethod
    def fit(cls, params: ForestParams, data: object, label: ArrayLike | None = None) -> Self:
        """Fits a model of the rows of ``data`` (a :class:`~hessboost.DMatrix`
        or anything it accepts; NaN entries are missing). Labels (``label``,
        else a matrix's own), if any, are class labels: one set of GBDTs is
        trained per class, and generation reproduces their proportions.
        Rows missing every feature are dropped.

        Runs with the GIL released and cannot be interrupted: Ctrl-C takes
        effect once it returns.

        Raises:
            HessboostError: The data has weights, base margins, groups,
                label bounds, feature weights or a label matrix; the column
                kinds do not fit the columns; a column has no observed
                value; or training fails.
            TypeError: ``params`` is not a :class:`ForestParams`.
        """
        if not isinstance(params, ForestParams):
            raise TypeError(f"params must be ForestParams, got {type(params).__name__}")
        if isinstance(data, DMatrix):
            matrix = data
            core = cls._labelled(matrix, label)
        else:
            matrix = DMatrix(data, label)
            core = matrix._core
        return cls._wrap(
            _hessboost.ForestModel.fit(params._build(), core),
            matrix._feature_names,
            matrix._feature_types,
            matrix._categories,
        )

    def generate(
        self, n_rows: int, *, seed: int = 0
    ) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
        """``n_rows`` synthetic rows: ``(values, labels)`` with ``values``
        ``(n_rows, n_columns)`` ``float32`` and, for a class-conditional
        model, each row's label drawn from the training proportions,
        ``(n_rows,)`` (else ``None``).

        Raises:
            HessboostError: ``n_rows`` is 0 or too large, or the sampler
                diverges.
        """
        return self._core.generate(_count("n_rows", n_rows), _count("seed", seed))

    def generate_for_labels(self, labels: ArrayLike, *, seed: int = 0) -> NDArray[np.float32]:
        """One synthetic row per entry of ``labels``, from that class's
        model, ``(len(labels), n_columns)`` ``float32``.

        Raises:
            HessboostError: The model is unconditional, ``labels`` is empty,
                or holds a label the model was not trained on.
        """
        classes = np.ascontiguousarray(np.asarray(labels, dtype=np.float32).reshape(-1))
        values, _ = self._core.generate_for_labels(classes, _count("seed", seed))
        return values

    def impute(
        self,
        data: object,
        label: ArrayLike | None = None,
        *,
        n_imputations: int = 1,
        repaint: Repaint | None = None,
        seed: int = 0,
    ) -> NDArray[np.float32]:
        """Imputes the missing entries of every row of ``data`` (a
        :class:`~hessboost.DMatrix` or anything it accepts) ``n_imputations``
        times, ``(n_imputations, rows, n_columns)`` ``float32``; observed
        entries are kept (rounded and clipped like generated values). A
        class-conditional model reads each row's class from ``label`` (else
        the matrix's labels). Frames are re-coded to the training
        categories.

        Raises:
            HessboostError: The model is a flow model (imputation needs
                :class:`Diffusion`), ``n_imputations`` is 0, the repaint
                settings are out of range, the data has metadata ``fit``
                refuses (weights, base margins, groups, label bounds,
                feature weights or a label matrix), labels are missing or
                unknown, a categorical value was unseen in training, or the
                column count differs.
        """
        if repaint is not None and not isinstance(repaint, Repaint):
            raise TypeError(f"repaint must be Repaint or None, got {type(repaint).__name__}")
        if isinstance(data, DMatrix):
            matrix = data
            core = self._labelled(matrix, label)
        else:
            info = {} if label is None else _data.info(label=label)
            matrix = DMatrix._coded(data, self._categories, np.nan, info)
            core = matrix._core
        _check_schema(self, matrix, "the data", "the model's", hint=_RECODE_HINT)
        return self._core.impute(
            core,
            _count("n_imputations", n_imputations),
            None if repaint is None else (repaint.resample, repaint.jump),
            _count("seed", seed),
        )

    @property
    def method(self) -> ForestMethod:
        """``"flow"`` or the :class:`Diffusion` settings."""
        return _method(self._core.method)

    @property
    def n_t(self) -> int:
        """The number of noise levels."""
        return self._core.n_t

    @property
    def n_columns(self) -> int:
        """The number of columns a row has."""
        return self._core.n_columns

    @property
    def classes(self) -> NDArray[np.float64]:
        """The sorted class labels of a class-conditional model (empty for
        an unconditional one)."""
        return self._core.classes

    @property
    def feature_names(self) -> list[str] | None:
        """The training columns' names, if known (not stored in model files;
        pickling keeps them)."""
        return None if self._feature_names is None else list(self._feature_names)

    def to_bytes(self) -> bytes:
        """The model in its binary format (zstd-compressed, magic
        ``HBFF``)."""
        return self._core.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> Self:
        """Reads a model written by :meth:`to_bytes`.

        Raises:
            ModelFormatError: The bytes are not a valid forest model.
        """
        return cls._wrap(_hessboost.ForestModel.from_bytes(bytes(data)))

    def save_binary(self, path: PathLike) -> None:
        """Writes :meth:`to_bytes` to ``path``."""
        data = self.to_bytes()
        with open(path, "wb") as file:
            file.write(data)

    @classmethod
    def load_binary(cls, path: PathLike) -> Self:
        """Reads a file written by :meth:`save_binary`.

        Raises:
            ModelFormatError: The file is not a valid forest model.
            OSError: The file cannot be read.
        """
        with open(path, "rb") as file:
            return cls.from_bytes(file.read())

    def to_json(self) -> str:
        """The model as JSON, each GBDT in hessboost's native JSON."""
        return self._core.to_json()

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Reads a model written by :meth:`to_json`.

        Raises:
            ModelFormatError: The text is not a valid forest model.
        """
        return cls._wrap(_hessboost.ForestModel.from_json(text))

    def save_json(self, path: PathLike) -> None:
        """Writes :meth:`to_json` to ``path``."""
        text = self.to_json()
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)

    @classmethod
    def load_json(cls, path: PathLike) -> Self:
        """Reads a file written by :meth:`save_json`.

        Raises:
            ModelFormatError: The file is not a valid forest model.
            OSError: The file cannot be read.
        """
        with open(path, encoding="utf-8") as file:
            return cls.from_json(file.read())

    def __getstate__(self) -> dict[str, object]:
        return {
            "model": self._core.to_bytes(),
            "feature_names": self._feature_names,
            "feature_types": self._feature_types,
            "categories": self._categories,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._core = _hessboost.ForestModel.from_bytes(state["model"])
        self._feature_names = state["feature_names"]
        self._feature_types = state["feature_types"]
        self._categories = state["categories"]

    def __repr__(self) -> str:
        method = "flow" if self.method == "flow" else "diffusion"
        return (
            f"ForestModel(method={method}, n_t={self.n_t}, columns={self.n_columns}, "
            f"classes={len(self.classes)})"
        )
