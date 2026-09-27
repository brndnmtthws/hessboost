"""hessboost: fast, deterministic gradient boosting in Rust.

Build a :class:`DMatrix`, pass a dict of parameters (XGBoost's names) to
:func:`train`, and predict with the returned :class:`Booster`::

    import hessboost

    dtrain = hessboost.DMatrix(X, label=y)
    booster = hessboost.train({"objective": "binary:logistic", "max_depth": 4}, dtrain, 100)
    probabilities = booster.predict(X_test)

Submodules:

* :mod:`hessboost.sklearn` -- scikit-learn estimators (needs scikit-learn)
* :mod:`hessboost.conformal` -- conformal prediction intervals
* :mod:`hessboost.ebm` -- explainable boosting machines' shape functions
* :mod:`hessboost.inference` -- Boulevard confidence intervals for ``f(x)`` and EBM bands
* :mod:`hessboost.diffusion` -- nonparametric ``p(y | x)`` by tree-based
  diffusion and flow matching
* :mod:`hessboost.folds` -- k-fold and forward-chaining (purged) folds
* :mod:`hessboost.online` -- adding and deleting training rows in place
"""

from importlib.metadata import version as _version

from hessboost import conformal, diffusion, ebm, folds, inference, online
from hessboost._booster import (
    Booster,
    Distributions,
    ImportanceType,
    ModelFormat,
)
from hessboost._core import Uncertainty
from hessboost._exceptions import (
    HessboostError,
    IncompatibleModelError,
    InvalidDataError,
    ModelFormatError,
)
from hessboost._matrix import DMatrix
from hessboost._training import (
    CustomMetric,
    EvalsResult,
    Objective,
    TrainingCallback,
    cv,
    train,
)

__version__: str = _version("hessboost")
"""The installed distribution's version (PEP 440, e.g. ``0.3.0rc1``)."""

__all__ = [
    "Booster",
    "CustomMetric",
    "DMatrix",
    "Distributions",
    "EvalsResult",
    "HessboostError",
    "ImportanceType",
    "IncompatibleModelError",
    "InvalidDataError",
    "ModelFormat",
    "ModelFormatError",
    "Objective",
    "TrainingCallback",
    "Uncertainty",
    "__version__",
    "conformal",
    "cv",
    "diffusion",
    "ebm",
    "folds",
    "inference",
    "online",
    "train",
]
