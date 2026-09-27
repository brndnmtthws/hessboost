"""Exceptions raised by hessboost (exported from :mod:`hessboost`)."""

__all__ = ["HessboostError", "IncompatibleModelError", "InvalidDataError", "ModelFormatError"]


class HessboostError(ValueError):
    """hessboost refused an input: invalid or unsupported parameters, data
    that does not fit (shapes, non-finite values, bad categories), or a
    configuration hessboost does not implement.

    A :class:`ValueError`, so code catching ``ValueError`` keeps working.
    Wrong argument *types* raise :class:`TypeError` instead, and file I/O
    failures raise :class:`OSError`.
    """

    __module__ = "hessboost"


class ModelFormatError(HessboostError):
    """A model could not be encoded or decoded: corrupt or truncated bytes,
    a file from an unsupported version, or a model the requested format
    cannot represent (for example a ``gblinear`` model saved as XGBoost
    JSON)."""

    __module__ = "hessboost"


class InvalidDataError(HessboostError):
    """The data was refused for its content: labels outside the objective's
    domain (for example a negative count for ``count:poisson``), negative
    weights, query groups that do not partition the rows, inconsistent
    label bounds, or metadata a method does not support. The message names
    the input and, for an evaluation set, the dataset."""

    __module__ = "hessboost"


class IncompatibleModelError(HessboostError):
    """A model passed in cannot serve the request: continuing training or
    refreshing from a model the parameters or data do not match (another
    objective, feature count or output count), slicing past its
    iterations, or asking it for something it does not hold."""

    __module__ = "hessboost"
