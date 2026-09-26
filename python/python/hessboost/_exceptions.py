"""Exceptions raised by hessboost (exported from :mod:`hessboost`)."""

__all__ = ["HessboostError", "ModelFormatError"]


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
