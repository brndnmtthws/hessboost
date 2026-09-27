"""ForestFlow and ForestDiffusion: synthetic tabular rows and missing-value
imputation with boosted trees (Jolicoeur-Martineau, Fatras and Kilian,
AISTATS 2024).

A :class:`ForestModel` learns the joint distribution of a table's columns,
optionally per class of a label, with one GBDT per noise level (and per
class). It then samples new rows, and a diffusion model also imputes the
missing (NaN) entries of rows, any number of times, keeping the observed
ones::

    from hessboost.diffusion.forest import ForestModel, ForestParams

    params = ForestParams(column_kinds=["continuous", "integer", "categorical"])
    model = ForestModel.fit(params, X)
    synthetic = model.sample(1000, seed=0)  # ForestSamples: (1000, columns) float32, None

    model = ForestModel.fit(ForestParams.forest_diffusion(), X_with_nans)
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
from hessboost._matrix import _matrix_for
from hessboost._model_io import PathLike, _SchemaState, read_bytes, write_bytes
from hessboost.diffusion import (
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
    "ForestSamples",
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
    training: dict[str, Any] = _hessboost.ForestParams.preset("forest_flow")["training"]
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
    def _preset(cls, name: Literal["forest_flow", "forest_diffusion"]) -> Self:
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
    def forest_flow(cls) -> Self:
        """The reference's ForestFlow configuration (the defaults): 50 noise
        levels, 100 copies of each row, 100 rounds of depth-7 ``hist`` trees
        with ``eta = 0.3`` and ``lambda = 0``."""
        return cls._preset("forest_flow")

    @classmethod
    def forest_diffusion(cls) -> Self:
        """The reference's ForestDiffusion configuration: :meth:`forest_flow`
        with :class:`Diffusion` (``β`` from 0.1 to 8)."""
        return cls._preset("forest_diffusion")


@dataclass(frozen=True)
class ForestSamples:
    """Synthetic rows from :meth:`ForestModel.sample` or
    :meth:`ForestModel.sample_for_labels`.

    Attributes:
        values: ``(rows, n_columns)`` ``float32``.
        labels: Each row's class label, ``(rows,)`` ``float32``, for a
            class-conditional model (else ``None``).
    """

    values: NDArray[np.float32]
    labels: NDArray[np.float32] | None


class ForestModel(_SchemaState):
    """A fitted ForestFlow / ForestDiffusion model. Build one with
    :meth:`fit` or a loader; it is immutable and may be shared between
    threads."""

    _core: _hessboost.ForestModel

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
        self._set_schema(feature_names, feature_types, {} if categories is None else categories)
        return self

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
        matrix = _matrix_for(data, (), label=label)
        return cls._wrap(
            _hessboost.ForestModel.fit(params._build(), matrix._core),
            matrix._feature_names,
            matrix._feature_types,
            matrix._categories,
        )

    def sample(self, n_rows: int, *, seed: int = 0) -> ForestSamples:
        """``n_rows`` synthetic rows, ``(n_rows, n_columns)``, with each
        row's label drawn from the training proportions for a
        class-conditional model (else ``labels`` is ``None``).

        Raises:
            HessboostError: ``n_rows`` is 0 or too large, or the sampler
                diverges.
        """
        values, labels = self._core.sample(_count("n_rows", n_rows), _count("seed", seed))
        return ForestSamples(values, labels)

    def sample_for_labels(self, labels: ArrayLike, *, seed: int = 0) -> ForestSamples:
        """One synthetic row per entry of ``labels``, from that class's
        model, ``(len(labels), n_columns)``, labelled with them.

        Raises:
            HessboostError: The model is unconditional, ``labels`` is empty,
                or holds a label the model was not trained on.
        """
        classes = np.ascontiguousarray(np.asarray(labels, dtype=np.float32).reshape(-1))
        values, drawn = self._core.sample_for_labels(classes, _count("seed", seed))
        return ForestSamples(values, drawn)

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
        matrix = _matrix_for(data, ((self, "the model's"),), label=label)
        return self._core.impute(
            matrix._core,
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

    def to_bytes(self, format: Literal["binary", "json"] = "binary") -> bytes:
        """The model encoded as ``format``: ``"binary"``, its binary format
        (zstd-compressed, magic ``HBFF``), or ``"json"``, UTF-8 JSON with
        each GBDT in hessboost's native JSON.

        Feature names and categories are not part of the formats; pickle
        the model to keep them."""
        return self._core.encode(format)

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        format: Literal["auto", "binary", "json"] = "auto",
    ) -> Self:
        """Reads a model written by :meth:`to_bytes` as ``format``, by
        default the one the bytes look like.

        Raises:
            ModelFormatError: The bytes are in neither format, or not a
                valid forest model.
        """
        return cls._wrap(_hessboost.ForestModel.decode(bytes(data), format))

    def save(self, path: PathLike, format: Literal["binary", "json"] = "binary") -> None:
        """Writes :meth:`to_bytes` of ``format`` to ``path``.

        Raises:
            OSError: The file cannot be written.
        """
        write_bytes(path, self.to_bytes(format))

    @classmethod
    def load(cls, path: PathLike, format: Literal["auto", "binary", "json"] = "auto") -> Self:
        """Reads a file written by :meth:`save`, as :meth:`from_bytes` reads
        bytes.

        Raises:
            ModelFormatError: The file is in neither format, or not a valid
                forest model.
            OSError: The file cannot be read.
        """
        return cls.from_bytes(read_bytes(path), format)

    def _model_state(self) -> bytes:
        return self._core.encode("binary")

    def _restore_model(self, model: bytes) -> None:
        self._core = _hessboost.ForestModel.decode(model, "binary")

    def __repr__(self) -> str:
        method = "flow" if self.method == "flow" else "diffusion"
        return (
            f"ForestModel(method={method}, n_t={self.n_t}, columns={self.n_columns}, "
            f"classes={len(self.classes)})"
        )
