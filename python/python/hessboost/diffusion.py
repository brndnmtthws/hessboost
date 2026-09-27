"""Nonparametric probabilistic regression: conditional diffusion and flow
matching with boosted trees as the score or velocity model.

A :class:`DiffusionModel` learns the whole conditional distribution
``p(y | x)`` of a scalar or vector label, with no parametric family:
multimodal, skewed, heavy-tailed, heteroscedastic and correlated
multivariate labels are all in reach. Instead of a density it returns
draws; :func:`mean`, :func:`quantiles` and :func:`crps` summarize them::

    from hessboost.diffusion import DiffusionModel, DiffusionParams, quantiles

    model = DiffusionModel.fit(DiffusionParams.flow_matching(), X_train, y_train)
    draws = model.sample(X_test, 200, seed=0)  # (rows, 200, outputs) float32
    bands = quantiles(draws, [0.05, 0.5, 0.95])  # (rows, 3, outputs)

Training repeats every labelled row :attr:`~DiffusionParams.n_repeats`
times, noises its (standardized) label to a random time ``t``, and boosts
one squared-error GBDT on ``(y_t, x, t)`` to reconstruct the score of a
Gaussian SDE (:class:`Score`) or the velocity of a Gaussian path
(:class:`FlowMatching`). Sampling integrates the reverse-time SDE or ODE in
:attr:`~DiffusionParams.n_steps` steps.

A configuration is a :class:`DiffusionParams`: frozen dataclasses mirroring
``hessboost::diffusion``, validated when built. The presets are
:meth:`DiffusionParams.default` (DiffGBM's score-side recipe, also what
``DiffusionParams()`` gives), :meth:`DiffusionParams.treeffuser` and
:meth:`DiffusionParams.flow_matching`; change a field with
:func:`dataclasses.replace`::

    import dataclasses

    params = dataclasses.replace(DiffusionParams.treeffuser(), n_repeats=10)

The GBDTs' ``training`` parameters are an XGBoost ``params`` mapping, read
as :func:`hessboost.train` reads one (from XGBoost's defaults, unknown keys
refused). The presets' mappings hold every setting (LightGBM's defaults,
as Treeffuser and DiffGBM train); override keys with
``{**params.training, "max_leaves": 15}``.

Draws are deterministic for a given model, input, ``n_samples`` and seed at
any thread count, and the first ``k`` draws of a row are the same for every
``n_samples >= k``. :meth:`DiffusionModel.fit` releases the GIL but cannot
be interrupted: Ctrl-C takes effect once it returns.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _data, _hessboost
from hessboost._core import _RECODE_HINT, DMatrix, _check_schema
from hessboost._exceptions import HessboostError

__all__ = [
    "DiffusionModel",
    "DiffusionParams",
    "EarlyStopping",
    "Edm",
    "FlowMatching",
    "FlowPath",
    "LogNoiseNormal",
    "Method",
    "OdeSolver",
    "Parameterization",
    "Residualizer",
    "Score",
    "Sde",
    "SubVariancePreserving",
    "TimeSampling",
    "VarianceExploding",
    "VariancePreserving",
    "crps",
    "mean",
    "quantiles",
]

PathLike: TypeAlias = str | os.PathLike[str]


def _number(owner: object, name: str) -> None:
    """Refuses a non-numeric field ``name`` of the frozen ``owner`` and stores
    it as a ``float``."""
    value = getattr(owner, name)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    object.__setattr__(owner, name, float(value))


def _count(name: str, value: object) -> int:
    """``value`` as a non-negative ``int``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise HessboostError(f"{name} must be non-negative, got {value}")
    return int(value)


def _store_count(owner: object, name: str) -> None:
    object.__setattr__(owner, name, _count(name, getattr(owner, name)))


def _store_training(owner: object, name: str) -> None:
    """Refuses a non-mapping field ``name`` of the frozen ``owner`` and stores
    a ``dict`` copy."""
    value = getattr(owner, name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping of parameters, got {type(value).__name__}")
    object.__setattr__(owner, name, dict(value))


def _choice(name: str, value: object, choices: tuple[str, ...], classes: tuple[type, ...]) -> None:
    """Refuses ``value`` unless it is one of ``choices`` or an instance of
    ``classes``."""
    if isinstance(value, classes):
        return
    if isinstance(value, str):
        if value not in choices:
            raise HessboostError(f"unknown {name} {value!r}; expected one of {list(choices)}")
        return
    expected = [repr(choice) for choice in choices] + [cls.__name__ for cls in classes]
    raise TypeError(f"{name} must be {' or '.join(expected)}, got {type(value).__name__}")


@dataclass(frozen=True)
class VarianceExploding:
    """The variance-exploding SDE, ``σ(t) = σ_min (σ_max/σ_min)^t``, with
    prior ``N(0, σ_max²)``. Needs ``0 < sigma_min < sigma_max``."""

    sigma_min: float = 0.01
    sigma_max: float = 20.0

    def __post_init__(self) -> None:
        _number(self, "sigma_min")
        _number(self, "sigma_max")


@dataclass(frozen=True)
class VariancePreserving:
    """The variance-preserving process with the linear schedule ``β(t) =
    β_min + (β_max - β_min) t`` and prior ``N(0, 1)``: an SDE of
    :class:`Score` or DDPM's path of :class:`FlowMatching`. Needs ``0 <
    beta_min < beta_max``."""

    beta_min: float = 0.1
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        _number(self, "beta_min")
        _number(self, "beta_max")


@dataclass(frozen=True)
class SubVariancePreserving:
    """The sub-variance-preserving SDE (kernel variance ``(1 - e^{-B})²``)
    with the linear ``β`` schedule. Needs ``0 < beta_min < beta_max``."""

    beta_min: float = 0.1
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        _number(self, "beta_min")
        _number(self, "beta_max")


@dataclass(frozen=True)
class Edm:
    """EDM preconditioning (Karras et al., 2022) of the score target, with
    data scale ``sigma_data > 0`` (``1`` suits the standardized labels)."""

    sigma_data: float = 1.0

    def __post_init__(self) -> None:
        _number(self, "sigma_data")


@dataclass(frozen=True)
class LogNoiseNormal:
    """Training times whose log noise scale is normal with ``mean`` and
    ``std > 0`` (EDM's log-normal noise levels)."""

    mean: float = -1.2
    std: float = 1.2

    def __post_init__(self) -> None:
        _number(self, "mean")
        _number(self, "std")


Sde: TypeAlias = VarianceExploding | VariancePreserving | SubVariancePreserving
"""The noising SDE of :class:`Score`."""

Parameterization: TypeAlias = Literal["noise"] | Edm
"""The score GBDT's target: ``"noise"`` (Treeffuser's negated noise) or
:class:`Edm` preconditioning."""

TimeSampling: TypeAlias = Literal["uniform"] | LogNoiseNormal
"""Distribution of the training times: ``"uniform"`` on ``[1e-5, 1]``
(Treeffuser) or :class:`LogNoiseNormal`."""

FlowPath: TypeAlias = Literal["linear", "trigonometric"] | VariancePreserving
"""The Gaussian path ``y_t = a(t) y_0 + b(t) z`` of :class:`FlowMatching`:
``"linear"`` (rectified flow), ``"trigonometric"``, or
:class:`VariancePreserving`."""

OdeSolver: TypeAlias = Literal["euler", "heun"]
"""The reverse-ODE integrator of :class:`FlowMatching`."""

_SDES = (VarianceExploding, VariancePreserving, SubVariancePreserving)


@dataclass(frozen=True)
class Score:
    """Score-based diffusion: the GBDT reconstructs the score of an SDE's
    marginals; sampling runs Euler-Maruyama on the reverse SDE. The
    defaults are DiffGBM's score-side recipe.

    Args:
        sde: The noising SDE.
        parameterization: The GBDT's regression target.
        noise_level_feature: Add ``ln σ(t)`` as a feature after ``t``.
        time_sampling: Distribution of the training times.
    """

    sde: Sde = VarianceExploding()
    parameterization: Parameterization = Edm()
    noise_level_feature: bool = True
    time_sampling: TimeSampling = LogNoiseNormal()

    def __post_init__(self) -> None:
        if not isinstance(self.sde, _SDES):
            raise TypeError(
                "sde must be VarianceExploding, VariancePreserving or SubVariancePreserving, "
                f"got {type(self.sde).__name__}"
            )
        _choice("parameterization", self.parameterization, ("noise",), (Edm,))
        if not isinstance(self.noise_level_feature, bool):
            raise TypeError(
                f"noise_level_feature must be a bool, got {type(self.noise_level_feature).__name__}"
            )
        _choice("time_sampling", self.time_sampling, ("uniform",), (LogNoiseNormal,))


@dataclass(frozen=True)
class FlowMatching:
    """Conditional flow matching: the GBDT regresses a Gaussian path's
    velocity; sampling integrates the reverse ODE. The defaults are
    DiffGBM's flow-matching configuration.

    Args:
        path: The probability path from data (``t = 0``) to ``N(0, I)``.
        time_sampling: Distribution of the training times (``"uniform"``
            also anchors 5% of the rows at ``t = 1``).
        solver: Euler (one GBDT evaluation per step) or Heun (two).
    """

    path: FlowPath = VariancePreserving()
    time_sampling: TimeSampling = LogNoiseNormal()
    solver: OdeSolver = "heun"

    def __post_init__(self) -> None:
        _choice("path", self.path, ("linear", "trigonometric"), (VariancePreserving,))
        _choice("time_sampling", self.time_sampling, ("uniform",), (LogNoiseNormal,))
        _choice("solver", self.solver, ("euler", "heun"), ())


Method: TypeAlias = Score | FlowMatching
"""Score diffusion or flow matching, with its settings."""

_TAGS: dict[type, str] = {
    VarianceExploding: "variance_exploding",
    VariancePreserving: "variance_preserving",
    SubVariancePreserving: "sub_variance_preserving",
    Edm: "edm",
    LogNoiseNormal: "log_noise_normal",
}
"""The crate's (serde) name of each variant class."""

_VARIANTS: dict[str, type] = {tag: cls for cls, tag in _TAGS.items()}


_Variant: TypeAlias = (
    VarianceExploding | VariancePreserving | SubVariancePreserving | Edm | LogNoiseNormal
)


def _encode(value: _Variant | str) -> object:
    """A variant in the crate's JSON form: a unit variant's name, else
    ``{name: fields}``."""
    if isinstance(value, str):
        return value
    return {_TAGS[type(value)]: dataclasses.asdict(value)}


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        return value
    ((tag, fields),) = value.items()
    return _VARIANTS[tag](**fields)


def _method_json(method: Method) -> str:
    if isinstance(method, Score):
        body: dict[str, object] = {
            "score": {
                "sde": _encode(method.sde),
                "parameterization": _encode(method.parameterization),
                "noise_level_feature": method.noise_level_feature,
                "time_sampling": _encode(method.time_sampling),
            }
        }
    elif isinstance(method, FlowMatching):
        body = {
            "flow_matching": {
                "path": _encode(method.path),
                "time_sampling": _encode(method.time_sampling),
                "solver": method.solver,
            }
        }
    else:
        raise TypeError(f"method must be Score or FlowMatching, got {type(method).__name__}")
    return json.dumps(body)


def _method(text: str) -> Method:
    ((tag, body),) = json.loads(text).items()
    if tag == "score":
        return Score(
            sde=_decode(body["sde"]),
            parameterization=_decode(body["parameterization"]),
            noise_level_feature=body["noise_level_feature"],
            time_sampling=_decode(body["time_sampling"]),
        )
    return FlowMatching(
        path=_decode(body["path"]),
        time_sampling=_decode(body["time_sampling"]),
        solver=body["solver"],
    )


def _preset_training(preset: str, residualizer: bool) -> dict[str, Any]:
    description = _hessboost.DiffusionParams.preset(preset)
    training: dict[str, Any] = (
        description["residualizer"][1] if residualizer else description["training"]
    )
    return training


def _default_training() -> Mapping[str, Any]:
    return _preset_training("default", residualizer=False)


def _default_residualizer_training() -> Mapping[str, Any]:
    return _preset_training("default", residualizer=True)


@dataclass(frozen=True)
class EarlyStopping:
    """Early stopping of the score/velocity GBDT on a validation split of
    the original rows (taken before repetition).

    Args:
        rounds: Rounds without improvement of the validation RMSE before
            stopping (``> 0``).
        eval_fraction: Fraction of the rows held out, in ``(0, 1)``.
    """

    rounds: int = 50
    eval_fraction: float = 0.1

    def __post_init__(self) -> None:
        _store_count(self, "rounds")
        _number(self, "eval_fraction")


@dataclass(frozen=True)
class Residualizer:
    """Cross-fitted conditional-mean residualization (DiffGBM's): ``folds``
    GBDTs estimate ``E[y | x]`` out of fold, the diffusion learns the
    distribution of the residuals, and sampling adds the fold models'
    averaged mean back. Needs at least 80 training rows.

    Args:
        folds: Cross-fitting folds (``>= 2``; at most one per 40 rows are
            used).
        training: The fold models' XGBoost parameters (objective
            ``reg:squarederror``, ``scale_pos_weight`` 1); default: learning rate 0.05, depth 6,
            31 leaves, 20 rows per leaf.
        num_boost_round: Boosting rounds of each fold model (``> 0``).
    """

    folds: int = 5
    training: Mapping[str, Any] = field(default_factory=_default_residualizer_training)
    num_boost_round: int = 100

    def __post_init__(self) -> None:
        _store_count(self, "folds")
        _store_training(self, "training")
        _store_count(self, "num_boost_round")


@dataclass(frozen=True)
class DiffusionParams:
    """The configuration of :meth:`DiffusionModel.fit`, validated when
    built. The defaults are :meth:`default`'s.

    Args:
        method: :class:`Score` diffusion or :class:`FlowMatching`.
        n_repeats: Noisy copies of each training row (``> 0``).
        n_steps: Sampler integration steps (``> 0``), stored with the model.
        training: The score/velocity GBDT's XGBoost parameters (objective
            ``reg:squarederror``, ``scale_pos_weight`` 1); default: LightGBM's defaults (leaf-wise,
            31 leaves, learning rate 0.1, 20 rows per leaf, no L2 penalty,
            255 bins).
        num_boost_round: Maximum boosting rounds of that GBDT (``> 0``).
        early_stopping: Stop on a validation split, or ``None`` to train
            every round on all rows.
        residualizer: Diffuse conditional-mean residuals, or ``None`` to
            diffuse the standardized labels.
        seed: Seed of the validation split, the residualizer's folds, and
            the training noise and times (the GBDTs' own sampling uses
            their ``seed`` parameter).

    Raises:
        HessboostError: A count is zero, a process parameter or fraction is
            out of range, or a ``training`` mapping is refused (unknown
            keys, an objective other than ``reg:squarederror``, or
            ``scale_pos_weight`` other than 1).
        TypeError: A field has the wrong type.
    """

    method: Method = Score()
    n_repeats: int = 30
    n_steps: int = 50
    training: Mapping[str, Any] = field(default_factory=_default_training)
    num_boost_round: int = 3000
    early_stopping: EarlyStopping | None = EarlyStopping()
    residualizer: Residualizer | None = field(default_factory=Residualizer)
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("n_repeats", "n_steps", "num_boost_round", "seed"):
            _store_count(self, name)
        _store_training(self, "training")
        stop, residualizer = self.early_stopping, self.residualizer
        if stop is not None and not isinstance(stop, EarlyStopping):
            raise TypeError(
                f"early_stopping must be EarlyStopping or None, got {type(stop).__name__}"
            )
        if residualizer is not None and not isinstance(residualizer, Residualizer):
            raise TypeError(
                f"residualizer must be Residualizer or None, got {type(residualizer).__name__}"
            )
        self._build()

    def _build(self) -> _hessboost.DiffusionParams:
        """The validated native configuration (built again by every
        :meth:`DiffusionModel.fit`, so it sees the fields as they are)."""
        stop, residualizer = self.early_stopping, self.residualizer
        request = {
            "method": _method_json(self.method),
            "n_repeats": self.n_repeats,
            "n_steps": self.n_steps,
            "training": _hessboost.Params(self.training),
            "num_boost_round": self.num_boost_round,
            "early_stopping": None if stop is None else (stop.rounds, stop.eval_fraction),
            "residualizer": None
            if residualizer is None
            else (
                residualizer.folds,
                _hessboost.Params(residualizer.training),
                residualizer.num_boost_round,
            ),
            "seed": self.seed,
        }
        return _hessboost.DiffusionParams(request)

    @classmethod
    def _preset(cls, name: str) -> Self:
        description = _hessboost.DiffusionParams.preset(name)
        stop = description["early_stopping"]
        residualizer = description["residualizer"]
        return cls(
            method=_method(description["method"]),
            n_repeats=description["n_repeats"],
            n_steps=description["n_steps"],
            training=description["training"],
            num_boost_round=description["num_boost_round"],
            early_stopping=None if stop is None else EarlyStopping(*stop),
            residualizer=None if residualizer is None else Residualizer(*residualizer),
            seed=description["seed"],
        )

    @classmethod
    def default(cls) -> Self:
        """DiffGBM's score-side recipe: VE SDE, EDM preconditioning, a
        log-noise feature, log-normal noise levels, conditional-mean
        residualization, 30 repeats, 50 Euler-Maruyama steps, up to 3000
        rounds stopped after 50 without improvement on 10% of the rows."""
        return cls._preset("default")

    @classmethod
    def treeffuser(cls) -> Self:
        """Treeffuser's published recipe: :meth:`default` with the noise
        parameterization, no noise-level feature, uniform training times,
        and no residualization."""
        return cls._preset("treeffuser")

    @classmethod
    def flow_matching(cls) -> Self:
        """DiffGBM's flow-matching configuration: :meth:`default` with
        :class:`FlowMatching`'s defaults (VP path, log-normal noise
        levels, Heun) and 5 steps."""
        return cls._preset("flow_matching")


def _draws(samples: ArrayLike) -> NDArray[np.float32]:
    draws = np.ascontiguousarray(samples, dtype=np.float32)
    if draws.ndim != 3:
        raise HessboostError(
            f"samples must be a (rows, samples, outputs) array, got shape {draws.shape}"
        )
    return draws


def mean(samples: ArrayLike) -> NDArray[np.float64]:
    """The Monte Carlo mean of each row's draws, ``(rows, outputs)``, from
    ``(rows, samples, outputs)`` draws (as :meth:`DiffusionModel.sample`
    returns)."""
    return _hessboost.samples_mean(_draws(samples))


def quantiles(samples: ArrayLike, levels: ArrayLike) -> NDArray[np.float64]:
    """Empirical quantiles of each row's draws at ``levels`` (each in
    ``[0, 1]``; linear interpolation between order statistics, numpy's
    default), ``(rows, len(levels), outputs)``.

    Raises:
        HessboostError: A level is outside ``[0, 1]``.
    """
    levels = np.asarray(levels, dtype=np.float64).reshape(-1)
    return _hessboost.samples_quantiles(_draws(samples), levels.tolist())


def crps(samples: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """The continuous ranked probability score of each label under its
    row's draws, ``(rows, outputs)``: ``mean |X - y| - mean |X - X'| / 2``
    over the draws (lower is better). ``y`` is ``(rows, outputs)``, or
    ``(rows,)`` for a single output.

    Raises:
        HessboostError: ``y`` does not hold one finite label per row and
            output.
    """
    draws = _draws(samples)
    labels = np.ascontiguousarray(y, dtype=np.float32)
    rows, _, outputs = draws.shape
    if labels.shape != (rows, outputs) and not (outputs == 1 and labels.shape == (rows,)):
        raise HessboostError(
            f"y must be ({rows}, {outputs})"
            + (f" or ({rows},)" if outputs == 1 else "")
            + f" for these samples, got shape {labels.shape}"
        )
    return _hessboost.samples_crps(draws, labels)


class DiffusionModel:
    """A fitted conditional diffusion or flow-matching model of ``p(y |
    x)``. Build one with :meth:`fit` or a loader. The model is immutable
    (setting :attr:`n_steps` swaps in a copy), so it may be shared between
    threads."""

    _core: _hessboost.DiffusionModel
    _feature_names: list[str] | None
    _feature_types: list[str] | None
    _categories: _data.Categories

    def __init__(self) -> None:
        raise TypeError("use DiffusionModel.fit(...) or a loader such as DiffusionModel.from_bytes")

    @classmethod
    def _wrap(
        cls,
        core: _hessboost.DiffusionModel,
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

    @classmethod
    def fit(cls, params: DiffusionParams, data: object, label: ArrayLike | None = None) -> Self:
        """Fits a model of the labels' distribution given the features:
        ``data`` is a :class:`~hessboost.DMatrix` (whose labels, or label
        matrix for a vector label, are used unless ``label`` is given) or
        anything it accepts, with ``label`` ``(rows,)`` or ``(rows,
        outputs)``. Deterministic for fixed ``params`` and data at any
        thread count.

        Runs with the GIL released and cannot be interrupted: Ctrl-C takes
        effect once it returns.

        Raises:
            HessboostError: The data has no labels, or has weights, base
                margins, groups, label bounds or feature weights; too few
                rows for the validation split (2) or the residualizer (80);
                or training fails.
            TypeError: ``params`` is not a :class:`DiffusionParams`.
        """
        if not isinstance(params, DiffusionParams):
            raise TypeError(f"params must be DiffusionParams, got {type(params).__name__}")
        if isinstance(data, DMatrix):
            matrix = data
            core = (
                matrix._core
                if label is None
                else matrix._core.with_info({**_data.info(label=label), "categorical": None})
            )
        else:
            matrix = DMatrix(data, label)
            core = matrix._core
        return cls._wrap(
            _hessboost.DiffusionModel.fit(params._build(), core),
            matrix._feature_names,
            matrix._feature_types,
            matrix._categories,
        )

    def sample(self, data: object, n_samples: int, *, seed: int = 0) -> NDArray[np.float32]:
        """``n_samples`` draws from ``p(y | x)`` for every row of ``data``
        (a :class:`~hessboost.DMatrix` or anything it accepts; labels are
        ignored), a ``float32`` array of shape ``(rows, n_samples,
        outputs)`` (the crate's ``[row][sample][output]`` layout). Frames
        are re-coded to the training categories.

        Raises:
            HessboostError: ``n_samples`` is 0, the features differ from the
                training data's, ``data`` has base margins, or the sampler
                diverges (use more :attr:`n_steps`).
        """
        matrix = (
            data
            if isinstance(data, DMatrix)
            else DMatrix._coded(data, self._categories, np.nan, {})
        )
        _check_schema(self, matrix, "the data", "the model's", hint=_RECODE_HINT)
        return self._core.sample(matrix._core, _count("n_samples", n_samples), _count("seed", seed))

    @property
    def method(self) -> Method:
        """The method and its settings."""
        return _method(self._core.method)

    @property
    def n_steps(self) -> int:
        """Sampler integration steps. Assigning (``> 0``) swaps in a copy of
        the model that samples with that many: more steps follow the
        learned dynamics more closely at a proportional cost."""
        return self._core.n_steps

    @n_steps.setter
    def n_steps(self, n_steps: int) -> None:
        self._core = self._core.with_n_steps(_count("n_steps", n_steps))

    @property
    def n_features(self) -> int:
        """The number of features the model conditions on."""
        return self._core.n_features

    @property
    def n_outputs(self) -> int:
        """The number of label columns the model samples."""
        return self._core.n_outputs

    @property
    def is_residualized(self) -> bool:
        """Whether the model was fitted with a :class:`Residualizer`."""
        return self._core.is_residualized

    @property
    def feature_names(self) -> list[str] | None:
        """The training features' names, if known (not stored in model
        files; pickling keeps them)."""
        return None if self._feature_names is None else list(self._feature_names)

    def to_bytes(self) -> bytes:
        """The model in the native binary format (zstd-compressed, magic
        ``HBDM``)."""
        return self._core.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> Self:
        """Reads a model written by :meth:`to_bytes`.

        Raises:
            ModelFormatError: The bytes are not a valid diffusion model.
        """
        return cls._wrap(_hessboost.DiffusionModel.from_bytes(bytes(data)))

    def save_binary(self, path: PathLike) -> None:
        """Writes :meth:`to_bytes` to ``path``."""
        data = self.to_bytes()
        with open(path, "wb") as file:
            file.write(data)

    @classmethod
    def load_binary(cls, path: PathLike) -> Self:
        """Reads a file written by :meth:`save_binary`.

        Raises:
            ModelFormatError: The file is not a valid diffusion model.
            OSError: The file cannot be read.
        """
        with open(path, "rb") as file:
            return cls.from_bytes(file.read())

    def to_json(self) -> str:
        """The model as JSON: the method, the label standardization, the
        residualizer, and each GBDT in hessboost's native JSON."""
        return self._core.to_json()

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Reads a model written by :meth:`to_json`.

        Raises:
            ModelFormatError: The text is not a valid diffusion model.
        """
        return cls._wrap(_hessboost.DiffusionModel.from_json(text))

    def save_json(self, path: PathLike) -> None:
        """Writes :meth:`to_json` to ``path``."""
        text = self.to_json()
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)

    @classmethod
    def load_json(cls, path: PathLike) -> Self:
        """Reads a file written by :meth:`save_json`.

        Raises:
            ModelFormatError: The file is not a valid diffusion model.
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
        self._core = _hessboost.DiffusionModel.from_bytes(state["model"])
        self._feature_names = state["feature_names"]
        self._feature_types = state["feature_types"]
        self._categories = state["categories"]

    def __repr__(self) -> str:
        method = type(self.method).__name__
        return (
            f"DiffusionModel(method={method}, features={self.n_features}, "
            f"outputs={self.n_outputs}, n_steps={self.n_steps}, "
            f"residualized={self.is_residualized})"
        )
