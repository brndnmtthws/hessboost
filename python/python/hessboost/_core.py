"""``Uncertainty``, exported from :mod:`hessboost`.

It lives in this module, which once also held ``DMatrix`` and ``Booster``,
because its ``__module__`` (``hessboost._core``) is what pickles of it
record: moving it would break loading them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["Uncertainty"]


@dataclass(frozen=True)
class Uncertainty:
    """A virtual ensemble's uncertainty decomposition
    (:meth:`hessboost.Booster.predict_uncertainty`), after CatBoost.

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
