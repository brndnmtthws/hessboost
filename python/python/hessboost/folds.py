"""Cross-validation folds as ``(train_rows, test_rows)`` index arrays.

Pass them to :func:`hessboost.cv` (``folds=``), or use them to hold out
rows for early stopping or conformal calibration. :func:`forward_chaining`
purges a fixed number of rows before each test block;
:func:`purged_forward` purges by each row's own label window,
:func:`purged_forward_at` does the same at caller-given block starts
(walk-forward validation by calendar period), both optionally over a
subset of the rows (``rows=``, such as those with a finite label), and
:class:`PurgedTail` is the purged early-stopping split of a fold's own
training rows (:func:`hessboost.cross_validate`'s ``per_fold_stopping``),
cut by time span or row count::

    from hessboost import folds

    for train_rows, test_rows in folds.forward_chaining(len(y), 4, gap=24):
        dtrain = hessboost.DMatrix(X[train_rows], y[train_rows])
        ...
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _hessboost
from hessboost._exceptions import HessboostError

__all__ = [
    "PurgedTail",
    "Split",
    "forward_chaining",
    "k_fold",
    "purged_forward",
    "purged_forward_at",
]

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


def _selected(rows: ArrayLike | None, n_rows: int) -> NDArray[np.int64] | None:
    if rows is None:
        return None
    array = np.asarray(rows).reshape(-1)
    if array.dtype == np.bool_:
        if array.size != n_rows:
            raise HessboostError(
                f"a rows mask needs one entry per row ({n_rows}), got {array.size}"
            )
        array = np.flatnonzero(array)
    elif array.size and not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"rows must be integer indices or a boolean mask, got {array.dtype}")
    return np.ascontiguousarray(array, dtype=np.int64)


def purged_forward(
    decision_at: ArrayLike,
    label_end: ArrayLike,
    *,
    validation_fraction: float,
    blocks: int = 1,
    min_train: int = 1,
    rows: ArrayLike | None = None,
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

    With ``rows`` (strictly increasing row indices, or a boolean mask with
    one entry per row), only those rows take part, for example the rows
    with a finite label: the folds are those of the selected rows' times
    alone, as indices into all the rows. The other rows are neither trained
    nor tested on, and their times are not checked (placeholders are fine).

    Raises:
        HessboostError: The inputs differ in length or are empty, a label
            ends before its decision, ``validation_fraction`` is not in
            ``(0, 1)``, no split leaves ``min_train`` training rows and a
            decision time per block, or ``rows`` selects no row, is not
            strictly increasing, is out of bounds, or is a mask of the
            wrong length.
        TypeError: The times are neither integers nor ``datetime64``, or
            ``rows`` is neither integers nor booleans.
    """
    times = _times(decision_at, "decision_at")
    return _hessboost.purged_forward(
        times,
        _times(label_end, "label_end"),
        float(validation_fraction),
        blocks,
        min_train,
        _selected(rows, len(times)),
    )


def purged_forward_at(
    decision_at: ArrayLike,
    label_end: ArrayLike,
    block_starts: ArrayLike,
    *,
    min_train: int = 1,
    rows: ArrayLike | None = None,
) -> list[Split]:
    """Forward folds over timestamped rows with test blocks starting at
    caller-given times, purged by each row's label window: walk-forward
    validation by calendar period (one fold per day, week or month).

    ``decision_at[i]`` is row ``i``'s decision (feature) time and
    ``label_end[i]`` the end of its label window plus any embargo;
    ``block_starts`` are the test blocks' start times, strictly increasing.
    All are integers on one time scale (e.g. epoch seconds) or
    ``datetime64`` values. Rows may come in any order and share times.

    Fold ``j`` tests on every row decided in
    ``[block_starts[j], block_starts[j + 1])`` (the last block: every row
    decided at or after its start) and trains on every row decided before
    ``block_starts[j]`` whose ``label_end`` is at or before it: a label
    ending exactly at the block start is kept, one ending a tick later is
    purged. Rows decided before ``block_starts[0]`` are never tested.

    Args:
        decision_at: Each row's decision time.
        label_end: Each row's label-window end (plus embargo), on the same
            time scale as ``decision_at``.
        block_starts: The test blocks' start times, strictly increasing.
        min_train: The fewest purged training rows any fold may have
            (at least 1). Defaults to 1.
        rows: The rows that take part (strictly increasing row indices, or
            a boolean mask with one entry per row), for example those with
            a finite label; the others are neither trained nor tested on,
            and their times are not checked. Defaults to every row.

    Returns:
        One ``(train_rows, test_rows)`` split per block start, in order.

    Raises:
        HessboostError: The inputs differ in length or are empty, a label
            ends before its decision, ``block_starts`` is empty or not
            strictly increasing, a block has no test rows, a block's
            purged training set has fewer than ``max(min_train, 1)`` rows,
            or ``rows`` selects no row, is not strictly increasing, is out
            of bounds, or is a mask of the wrong length.
        TypeError: The times are neither integers nor ``datetime64``, or
            ``rows`` is neither integers nor booleans.

    Example:
        Daily blocks over the last three days of hourly rows whose labels
        end an hour after their decision::

            day = 86_400
            at = np.arange(10 * 24) * 3_600
            splits = folds.purged_forward_at(at, at + 3_600, [7 * day, 8 * day, 9 * day])
            result = hessboost.cv(params, dtrain, 50, folds=splits)
    """
    times = _times(decision_at, "decision_at")
    return _hessboost.purged_forward_at(
        times,
        _times(label_end, "label_end"),
        _times(block_starts, "block_starts"),
        min_train,
        _selected(rows, len(times)),
    )


class PurgedTail:
    """The purged chronological tail of a fold's training rows, which
    per-fold early stopping (:func:`hessboost.cross_validate`'s
    ``per_fold_stopping``) sizes each fold's rounds on.

    ``decision_at`` and ``label_end`` hold every row's decision time and the
    end of its label window plus any embargo, as for
    :func:`purged_forward`. :meth:`split` starts the tail at one of the
    split rows' decision times, which ``cut`` places:

    - ``"span"`` (the default) cuts by time span (unlike
      :func:`purged_forward`, which cuts by the count of distinct decision
      times): for rows decided from ``first`` to ``last``, the tail starts
      at the first of their decision times at or after ``first + (1 -
      fraction) * (last - first)``. That offset from ``first`` is computed
      in ``float64`` and rounded up to a whole time unit, then compared
      exactly with each time's offset (so a ``fraction`` of ``0.3`` cuts a
      span of 10 units at 7, as the decimal says; past 2**53 units the span
      itself rounds).
    - ``"rows"`` cuts by row count: with the ``n`` split rows sorted by
      decision time, the tail starts at the decision time of the row at
      position ``min(n - 1, floor((1 - fraction) * n))`` (in ``float64``),
      so it holds the last ``fraction`` of the rows, plus any rows decided
      with that one (a decision time is never split).

    The two differ when the rows' density varies over time, as across the
    periods of an expanding fold: where later periods hold more rows per
    time unit, the span cut's tail holds more than ``fraction`` of the rows
    (fewer where they hold fewer), the row cut's about ``fraction``.

    The tail is every row decided at or after its start, and the
    early-stopping fit trains on the rows decided before it whose labels end
    at or before it (a label ending exactly at the split is kept, one ending
    a tick later is purged). The start never moves: a split leaving a side
    empty is refused.

    Raises:
        HessboostError: The inputs differ in length or are empty, a label
            ends before its decision, ``fraction`` is not in ``(0, 1)``, or
            ``cut`` is neither ``"span"`` nor ``"rows"``.
        TypeError: The times are neither integers nor ``datetime64``.
    """

    __slots__ = ("_cut", "_decision_at", "_fraction", "_label_end")

    def __init__(
        self,
        decision_at: ArrayLike,
        label_end: ArrayLike,
        fraction: float,
        *,
        cut: Literal["span", "rows"] = "span",
    ) -> None:
        self._decision_at = _times(decision_at, "decision_at")
        self._label_end = _times(label_end, "label_end")
        self._fraction = float(fraction)
        self._cut: Literal["span", "rows"] = cut
        _hessboost.check_purged_tail(self._decision_at, self._label_end, self._fraction, cut)

    @property
    def decision_at(self) -> NDArray[np.int64]:
        """Every row's decision time (a copy)."""
        return self._decision_at.copy()

    @property
    def label_end(self) -> NDArray[np.int64]:
        """Every row's label end (a copy)."""
        return self._label_end.copy()

    @property
    def fraction(self) -> float:
        """The tail's share of the rows' time span (``cut="span"``) or of
        the rows (``cut="rows"``)."""
        return self._fraction

    @property
    def cut(self) -> Literal["span", "rows"]:
        """How the tail is cut: ``"span"`` (by time span) or ``"rows"`` (by
        row count)."""
        return self._cut

    def split(self, rows: ArrayLike) -> tuple[int, Split]:
        """The split time of ``rows`` (a decision time, in the integer units
        of ``decision_at``: nanoseconds since the epoch for ``datetime64``
        times) and their split: ``(fit_rows, tail_rows)``, each in ``rows``
        order.

        Raises:
            HessboostError: A row is out of bounds, or a side of the split
                is empty.
            TypeError: ``rows`` are not integers.
        """
        array = np.asarray(rows).reshape(-1)
        if array.size and not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"rows must be integer indices, got {array.dtype}")
        split_at, [split] = _hessboost.purged_tail_split(
            self._decision_at,
            self._label_end,
            self._fraction,
            self._cut,
            np.ascontiguousarray(array, dtype=np.int64),
        )
        return split_at, split
