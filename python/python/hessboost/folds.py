"""Cross-validation folds as ``(train_rows, test_rows)`` index arrays.

Pass them to :func:`hessboost.cv` (``folds=``), or use them to hold out
rows for early stopping or conformal calibration. :func:`forward_chaining`
purges a fixed number of rows before each test block;
:func:`purged_forward` purges by each row's own label window::

    from hessboost import folds

    for train_rows, test_rows in folds.forward_chaining(len(y), 4, gap=24):
        dtrain = hessboost.DMatrix(X[train_rows], y[train_rows])
        ...
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _hessboost

__all__ = ["Split", "forward_chaining", "k_fold", "purged_forward"]

Split = tuple[NDArray[np.int64], NDArray[np.int64]]
"""One fold: its ``(train_rows, test_rows)`` row indices."""


def k_fold(n_rows: int, nfold: int, *, seed: int = 0) -> list[Split]:
    """Shuffled k-fold splits of ``n_rows`` rows (what :func:`hessboost.cv`
    uses by default): the rows are shuffled with ``seed`` and dealt to the
    ``nfold`` test sets round-robin; each fold trains on the other folds'
    rows. Only for exchangeable rows; use :func:`forward_chaining` for time
    series.

    Raises:
        HessboostError: ``nfold < 2`` or more folds than rows.
    """
    return _hessboost.k_fold(n_rows, nfold, seed)


def forward_chaining(n_rows: int, n_splits: int, *, gap: int = 0) -> list[Split]:
    """Forward-chaining (expanding-window) splits of ``n_rows`` time-ordered
    rows, like scikit-learn's ``TimeSeriesSplit``: the last ``n_splits``
    blocks of ``n_rows // (n_splits + 1)`` rows are the test blocks, and
    each fold trains on every earlier row except the ``gap`` rows right
    before its block. A ``gap`` of at least the label horizon purges
    training rows whose labels overlap the test block; no training row
    follows a test row, so no embargo is needed.

    Raises:
        HessboostError: ``n_splits`` is 0, the rows do not fill
            ``n_splits + 1`` blocks, or ``gap`` leaves the first fold no
            training rows.
    """
    return _hessboost.forward_chaining(n_rows, n_splits, gap)


def _times(values: ArrayLike, name: str) -> NDArray[np.int64]:
    array = np.asarray(values).reshape(-1)
    if np.issubdtype(array.dtype, np.datetime64):
        array = array.astype("datetime64[ns]").view(np.int64)
    elif not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must hold integer times or datetime64 values, got {array.dtype}")
    return np.ascontiguousarray(array, dtype=np.int64)


def purged_forward(
    decision_at: ArrayLike,
    label_end: ArrayLike,
    *,
    validation_fraction: float,
    blocks: int = 1,
    min_train: int = 1,
) -> list[Split]:
    """Forward folds over timestamped rows, purged by each row's label
    window: for labels that span time (forecast horizons, overlapping
    returns), where a fixed row ``gap`` cannot purge irregular schedules.

    ``decision_at[i]`` is row ``i``'s decision (feature) time and
    ``label_end[i]`` the end of its label window plus any embargo: integers
    on one time scale (e.g. epoch seconds) or ``datetime64`` values. Rows
    may come in any order and share times. The last
    ``validation_fraction`` of the distinct decision times is cut into
    ``blocks`` contiguous test blocks. Fold ``j`` tests on every row decided
    inside block ``j`` and trains on every row decided before it whose
    ``label_end`` is at or before the block's first decision time: a label
    ending exactly at the block start is kept, one ending a tick later is
    purged. A decision time is never split between training and test rows.

    When the first block's purge would leave fewer than ``min_train``
    training rows, its start moves later one decision time at a time until
    it leaves them.

    Raises:
        HessboostError: The inputs differ in length or are empty, a label
            ends before its decision, ``validation_fraction`` is not in
            ``(0, 1)``, or no split leaves ``min_train`` training rows and a
            decision time per block.
        TypeError: The times are neither integers nor ``datetime64``.
    """
    return _hessboost.purged_forward(
        _times(decision_at, "decision_at"),
        _times(label_end, "label_end"),
        float(validation_fraction),
        blocks,
        min_train,
    )
