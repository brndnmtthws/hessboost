"""``Booster``, exported from :mod:`hessboost`, with its prediction and
model-format types, and the compact and GPU layouts a booster converts to."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _data, _hessboost
from hessboost._core import Uncertainty
from hessboost._exceptions import HessboostError
from hessboost._matrix import _matrix_for
from hessboost._model_io import PathLike, _SchemaState, read_bytes, write_bytes

if TYPE_CHECKING:
    from hessboost.ebm import EbmInfo
    from hessboost.inference import BoulevardInfo

__all__ = [
    "Booster",
    "CompactModel",
    "Distributions",
    "GpuModel",
    "ImportanceType",
    "ModelFormat",
    "ModelSizeReport",
]

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

_GpuDevice: TypeAlias = Literal["metal", "wgpu"]


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


@dataclass(frozen=True)
class ModelSizeReport:
    """A booster's size in the native binary and compact formats, with the
    dictionary statistics of the *Trees on a Diet* paper, from
    :meth:`Booster.size_report`."""

    native_bytes: int
    """The native binary model's size (every tree, covers and gains included)."""
    compact_bytes: int
    """The compact model's size."""
    trees: int
    """The trees the compact model stores (those prediction uses)."""
    splits: int
    """Split nodes across those trees."""
    leaves: int
    """Leaves across those trees."""
    used_features: int
    """Features the trees split on."""
    thresholds: int
    """Distinct thresholds and categorical sets, over every used feature."""
    leaf_values: int
    """Distinct leaf values."""

    @property
    def compression_ratio(self) -> float:
        """``native_bytes / compact_bytes``."""
        return self.native_bytes / self.compact_bytes

    @property
    def reuse_factor(self) -> float:
        """Nodes per distinct dictionary entry, ``(splits + leaves) /
        (thresholds + leaf_values)``: how often each threshold or leaf value
        is shared."""
        return (self.splits + self.leaves) / max(self.thresholds + self.leaf_values, 1)


class Booster(_SchemaState):
    """A trained gradient-boosting model, as XGBoost's ``xgboost.Booster``.

    Get one from :func:`hessboost.train`, or load a saved model with
    ``Booster(model_file)`` or :meth:`load_model`. The model itself is
    immutable: :meth:`load_model` and continued training replace it, and
    slicing returns a new booster, so a booster may be shared between
    threads.

    Args:
        model_file: A path or the bytes of a saved model in any format
            (detected from the content), including LightGBM text models, or
            ``None`` for an empty booster to :meth:`load_model` into.
    """

    __module__ = "hessboost"

    _core: _hessboost.Booster | None
    best_score: float | None
    """With early stopping, the watched metric at :attr:`best_iteration`."""

    def __init__(self, model_file: PathLike | bytes | bytearray | memoryview | None = None) -> None:
        self._core = None
        self._set_schema(None, None, {})
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
        booster._set_schema(feature_names, feature_types, categories)
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
    def ebm(self) -> EbmInfo | None:
        """How a ``booster = ebm`` model was trained, or ``None`` (see
        :mod:`hessboost.ebm`)."""
        from hessboost.ebm import EbmInfo

        return EbmInfo._from_core(self._model.ebm)

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

    def num_trees(self) -> int:
        """The number of trees (``num_boosted_rounds() * trees per
        iteration``; the column count of ``pred_leaf``)."""
        return self._model.num_trees

    @property
    def num_outputs(self) -> int:
        """Raw outputs (margins) per row: ``num_class`` for multiclass, one
        per target or quantile otherwise."""
        return self._model.num_outputs

    @property
    def num_targets(self) -> int:
        """Label columns the model was trained on."""
        return self._model.num_targets

    @property
    def num_parallel_tree(self) -> int:
        """Trees per output per iteration (a boosted forest's size)."""
        return self._model.num_parallel_tree

    @property
    def base_margins(self) -> list[float]:
        """The per-output intercepts, in margin space."""
        return self._model.base_margins

    @property
    def vector_leaves(self) -> bool:
        """Whether each tree predicts every output (``multi_strategy=
        "multi_output_tree"``), rather than one output per tree."""
        return self._model.vector_leaves

    def _matrix(
        self, data: object, base_margin: ArrayLike | None, missing: float, validate: bool = True
    ) -> _hessboost.DMatrix:
        """``data`` as this model reads it (:func:`_matrix_for`; feature
        names checked only with ``validate``)."""
        return _matrix_for(
            data,
            ((self, "the model's"),),
            base_margin=base_margin,
            missing=missing,
            validate_names=validate,
        )._core

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
        matrix = self._matrix(data, base_margin, missing, validate_features)
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

    def to_gpu(self, device: _GpuDevice | None = None) -> GpuModel:
        """Lays this model out for GPU batch prediction: the forest, category
        pools, and per-tree weights are uploaded once, and each prediction
        call uploads its rows. Predictions are bit-identical to the CPU's.

        Args:
            device: ``"metal"`` (macOS; faster than the CPU from roughly a
                few thousand row-trees upward) or ``"wgpu"`` (Vulkan, Metal,
                or DirectX 12; unmeasured on real GPUs so far, and slower
                than the CPU on a software adapter such as Mesa's lavapipe,
                which it picks only when there is no other). ``None``: Metal
                on macOS, wgpu elsewhere.

        Raises:
            HessboostError: ``device`` is unknown or cannot predict here
                (:meth:`GpuModel.available` is ``False``; the message says
                why: no usable GPU, or a wgpu adapter that reassociates float
                additions, which would change the predictions), or the model
                is a ``gblinear`` or ``linear_tree`` model (which do not
                predict through the forest).
        """
        return GpuModel._wrap(self._model.to_gpu(device), self)

    def to_compact(self) -> CompactModel:
        """This model in the bit-packed compact layout (*Boosted Trees on a
        Diet*), for memory-constrained inference: it predicts bit-identical
        values and margins to :meth:`predict` with the default iterations
        (through :attr:`best_iteration`), and stores only those trees,
        without covers and gains. Training with ``toad_penalty_feature`` and
        ``toad_penalty_threshold`` shrinks it further. The compact model
        keeps this booster's feature names and categories.

        Raises:
            HessboostError: The model is ``gblinear``, has ``linear_tree`` or
                vector leaves, or splits one feature both numerically and
                categorically.
        """
        return CompactModel._wrap(self._model.to_compact(), self)

    def size_report(self) -> ModelSizeReport:
        """This model's native binary versus compact size, with the compact
        layout's dictionary statistics.

        Raises:
            HessboostError: As :meth:`to_compact`.
        """
        return ModelSizeReport(**self._model.size_report())

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
        matrix = self._matrix(data, base_margin, missing)
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
        matrix = self._matrix(data, base_margin, missing)
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
        matrix = self._matrix(data, base_margin, missing)
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
        write_bytes(fname, self.save_raw(_format_for(fname, format)))

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
            ModelFormatError: The content is in none of these formats (when
                detecting it), or is not a valid model.
            OSError: The file cannot be read.
        """
        data = (
            bytes(fname) if isinstance(fname, (bytes, bytearray, memoryview)) else read_bytes(fname)
        )
        self._core = _hessboost.Booster.load(data, format or "auto")
        self._set_schema(None, None, {})
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

    def _model_state(self) -> bytes | None:
        return None if self._core is None else self._core.save("binary")

    def _restore_model(self, model: bytes | None) -> None:
        self._core = None if model is None else _hessboost.Booster.load(model, "binary")

    def __getstate__(self) -> dict[str, object]:
        return {**super().__getstate__(), "best_score": self.best_score}

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
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


class GpuModel:
    """A model laid out for GPU batch prediction, from
    :meth:`Booster.to_gpu`: on Metal (macOS) or through wgpu (Vulkan, Metal,
    DirectX 12). Wraps ``hessboost._hessboost.GpuModel`` with the same
    feature-name checks as :meth:`Booster.predict`; unlike a booster it
    holds no file state and cannot be pickled.
    """

    __module__ = "hessboost"

    _core: _hessboost.GpuModel
    _model: Booster

    def __init__(self) -> None:
        raise TypeError("use Booster.to_gpu()")

    @classmethod
    def _wrap(cls, core: _hessboost.GpuModel, model: Booster) -> GpuModel:
        gpu = cls.__new__(cls)
        gpu._core = core
        gpu._model = model
        return gpu

    @property
    def booster(self) -> Booster:
        """The model this GPU predictor was built from."""
        return self._model

    @property
    def device(self) -> _GpuDevice:
        """The backend this model predicts on: ``"metal"`` or ``"wgpu"``."""
        return self._core.device

    @staticmethod
    def available(device: _GpuDevice | None = None) -> bool:
        """Whether ``device`` (``None``: Metal on macOS, wgpu elsewhere) can
        predict here, so that :meth:`Booster.to_gpu` lays forest models out
        on it: for Metal, a device with working compute pipelines (``False``
        off macOS); for wgpu, an adapter with 64-bit shader integers whose
        float additions passed the backend's addition-order check. Training
        with ``device`` needs the same GPU, except that wgpu also trains on
        an adapter that fails the check. The first call per device sets its
        backend up (picks the adapter, compiles the kernels, runs the check).

        Raises:
            HessboostError: ``device`` is not ``"metal"`` or ``"wgpu"``.
        """
        return _hessboost.GpuModel.available(device)

    @staticmethod
    def device_name(device: _GpuDevice | None = None) -> str | None:
        """The name of the GPU ``device`` (``None``: Metal on macOS, wgpu
        elsewhere) picked, if it found one (for diagnostics and benchmarks),
        including a wgpu adapter that trains but fails the addition-order
        check.

        Raises:
            HessboostError: ``device`` is not ``"metal"`` or ``"wgpu"``.
        """
        return _hessboost.GpuModel.device_name(device)

    def predict(
        self,
        data: object,
        *,
        output_margin: bool = False,
        iteration_range: tuple[int, int] | None = None,
        validate_features: bool = True,
        base_margin: ArrayLike | None = None,
        missing: float = np.nan,
    ) -> NDArray[Any]:
        """Predicts every row of ``data`` on the GPU, bit-identical to
        :meth:`Booster.predict` (values or, with ``output_margin``, raw
        margins): ``(rows,)`` or ``(rows, K)`` for ``K`` outputs.

        Args:
            iteration_range: ``(begin, end)`` boosting iterations, ``end=0``
                meaning the last; ``None`` uses iterations through the
                model's best iteration (all without early stopping).
            validate_features: Refuse data whose feature names differ from
                the model's.
            base_margin: Starting margins for array input (a
                :class:`DMatrix` carries its own).
            missing: The missing-value marker for array input.
        """
        model = self._model
        matrix = model._matrix(data, base_margin, missing, validate_features)
        kind = "margin" if output_margin else "value"
        return self._core.predict(matrix, kind, Booster._range(iteration_range))


class CompactModel(_SchemaState):
    """A tree ensemble in hessboost's bit-packed compact format (``HBTD``,
    *Boosted Trees on a Diet*), predicting bit-identical values and margins
    to the :class:`Booster` it came from while taking a fraction of its
    size. It predicts straight from the packed trees; it has no SHAP, leaf
    or iteration-range predictions (only the trees default prediction uses
    are stored). XGBoost cannot read the format.

    Get one from :meth:`Booster.to_compact`, or load a saved one with
    ``CompactModel(model_file)``.

    Args:
        model_file: A path or the bytes of a compact model
            (:meth:`save_model`/:meth:`save_raw`).

    Raises:
        ModelFormatError: The content is not a valid compact model.
        OSError: The file cannot be read.
    """

    __module__ = "hessboost"

    _core: _hessboost.CompactModel

    def __init__(self, model_file: PathLike | bytes | bytearray | memoryview) -> None:
        data = (
            bytes(model_file)
            if isinstance(model_file, (bytes, bytearray, memoryview))
            else read_bytes(model_file)
        )
        self._core = _hessboost.CompactModel.load(data)
        self._set_schema(None, None, {})

    @classmethod
    def _wrap(cls, core: _hessboost.CompactModel, booster: Booster) -> CompactModel:
        model = cls.__new__(cls)
        model._core = core
        model._set_schema(booster._feature_names, booster._feature_types, dict(booster._categories))
        return model

    @property
    def feature_names(self) -> list[str] | None:
        """The training features' names, if known (not stored in the file;
        pickling keeps them). Assignable, as :attr:`Booster.feature_names`."""
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
        """The objective that drives :meth:`predict`."""
        return self._core.objective

    def num_features(self) -> int:
        """The number of features the model takes."""
        return self._core.num_features

    def num_trees(self) -> int:
        """The number of stored trees."""
        return self._core.num_trees

    @property
    def num_outputs(self) -> int:
        """Raw outputs (margins) per row."""
        return self._core.num_outputs

    @property
    def size_bytes(self) -> int:
        """The serialized size in bytes."""
        return self._core.size_bytes

    @property
    def used_features(self) -> list[int]:
        """The indices of the features the trees split on, ascending."""
        return self._core.used_features

    @property
    def num_thresholds(self) -> int:
        """Distinct thresholds and categorical sets, over every used feature."""
        return self._core.num_thresholds

    @property
    def num_leaf_values(self) -> int:
        """Distinct leaf values."""
        return self._core.num_leaf_values

    def predict(
        self,
        data: object,
        *,
        output_margin: bool = False,
        validate_features: bool = True,
        base_margin: ArrayLike | None = None,
        missing: float = np.nan,
    ) -> NDArray[np.float32]:
        """Predicts every row of ``data`` (a :class:`DMatrix` or anything
        its constructor accepts; frames are re-coded to the model's
        categories): the objective's predictions or, with ``output_margin``,
        raw margins, ``(rows,)`` or ``(rows, K)`` for ``K`` outputs,
        bit-identical to :meth:`Booster.predict`.

        Args:
            validate_features: Refuse data whose feature names differ from
                the model's.
            base_margin: Starting margins for array input (a
                :class:`DMatrix` carries its own).
            missing: The missing-value marker for array input.
        """
        matrix = _matrix_for(
            data,
            ((self, "the model's"),),
            base_margin=base_margin,
            missing=missing,
            validate_names=validate_features,
        )
        return self._core.predict(matrix._core, output_margin)

    def save_raw(self) -> bytes:
        """The serialized model."""
        return self._core.save()

    def save_model(self, fname: PathLike) -> None:
        """Writes the model to ``fname``. Feature names and categories are
        not part of the format; pickle the model to keep them."""
        write_bytes(fname, self.save_raw())

    def _model_state(self) -> bytes:
        return self._core.save()

    def _restore_model(self, model: bytes) -> None:
        self._core = _hessboost.CompactModel.load(model)

    def __repr__(self) -> str:
        return (
            f"CompactModel(objective={self.objective!r}, trees={self.num_trees()}, "
            f"features={self.num_features()}, bytes={self.size_bytes})"
        )
