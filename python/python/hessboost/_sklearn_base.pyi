"""Type declarations of the scikit-learn base classes :mod:`hessboost.sklearn`
builds on (scikit-learn ships no type information). Type checking only:
at runtime the estimators inherit scikit-learn's own classes."""

from typing import Any, Self

from numpy.typing import ArrayLike

class BaseEstimator:
    """``sklearn.base.BaseEstimator``."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """The estimator's parameters, by name."""
    def set_params(self, **params: Any) -> Self:
        """Sets parameters by name; returns the estimator."""
    def __sklearn_tags__(self) -> Any: ...

class RegressorMixin:
    """``sklearn.base.RegressorMixin``."""

    def score(self, X: Any, y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
        """The coefficient of determination R² of the predictions."""
    def __sklearn_tags__(self) -> Any: ...

class ClassifierMixin:
    """``sklearn.base.ClassifierMixin``."""

    def score(self, X: Any, y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
        """The mean accuracy of the predictions."""
    def __sklearn_tags__(self) -> Any: ...
