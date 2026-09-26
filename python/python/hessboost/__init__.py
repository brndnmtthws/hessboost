"""hessboost: XGBoost's gradient boosting, reimplemented in Rust.

The API follows XGBoost's Python package: build a :class:`DMatrix`, pass a
dict of XGBoost parameters to :func:`train`, predict with the returned
:class:`Booster`::

    import hessboost

    dtrain = hessboost.DMatrix(X, label=y)
    booster = hessboost.train({"objective": "binary:logistic", "max_depth": 4}, dtrain, 100)
    probabilities = booster.predict(X_test)

Submodules:

* :mod:`hessboost.sklearn` -- scikit-learn estimators (needs scikit-learn)
* :mod:`hessboost.conformal` -- conformal prediction intervals
* :mod:`hessboost.folds` -- k-fold and forward-chaining (purged) folds
"""

from importlib.metadata import version as _version

from hessboost import conformal, folds
from hessboost._core import Booster, DMatrix, Distributions, ImportanceType, ModelFormat
from hessboost._exceptions import HessboostError, ModelFormatError
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
    "ModelFormat",
    "ModelFormatError",
    "Objective",
    "TrainingCallback",
    "__version__",
    "conformal",
    "cv",
    "folds",
    "train",
]
