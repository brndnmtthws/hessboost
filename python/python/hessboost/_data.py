"""Conversion of user inputs (numpy arrays of any dtype and layout, pandas
and polars frames (polars 1.x and 2.x; a ``LazyFrame`` is collected once),
scipy sparse matrices, array-likes) into the row-major ``float32`` arrays
the native core borrows, and of per-row metadata a frame supplies by column
name (:func:`take_metadata`)."""

from __future__ import annotations

import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hessboost import _hessboost
from hessboost._exceptions import HessboostError

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

FeatureTypes: TypeAlias = list[str]
"""Per-feature types: ``"q"`` (numerical) or ``"c"`` (categorical)."""

Categories: TypeAlias = dict[int, list[Any]]
"""The categories of each categorical frame column, by column index."""

_NUMERICAL_TYPES = frozenset({"q", "float", "int", "i"})


@dataclass(frozen=True)
class Features:
    """A converted feature matrix with its column metadata."""

    core: _hessboost.DMatrix
    feature_names: list[str] | None
    feature_types: FeatureTypes | None
    categories: Categories


def _pandas() -> Any:
    """The pandas module if it is already imported (a frame cannot exist
    otherwise), else ``None``."""
    return sys.modules.get("pandas")


def _polars() -> Any:
    """The polars module if it is already imported, else ``None``."""
    return sys.modules.get("polars")


def _is_pandas_frame(data: object) -> TypeGuard[pd.DataFrame]:
    pd = _pandas()
    return pd is not None and isinstance(data, pd.DataFrame)


def _is_polars_frame(data: object) -> TypeGuard[pl.DataFrame]:
    pl = _polars()
    return pl is not None and isinstance(data, pl.DataFrame)


def _is_polars_lazyframe(data: object) -> TypeGuard[pl.LazyFrame]:
    pl = _polars()
    return pl is not None and isinstance(data, pl.LazyFrame)


def _is_frame(data: object) -> bool:
    """Whether ``data`` is a pandas or polars frame, which keeps its column
    names and categorical columns."""
    return _is_pandas_frame(data) or _is_polars_frame(data)


def collect_frame(data: object) -> object:
    """A polars ``LazyFrame`` collected, once, by polars' default engine
    (the streaming engine since polars 2.0, which spills to disk when the
    frame outgrows memory); anything else as it is."""
    return data.collect() if _is_polars_lazyframe(data) else data


def _refuse_categorical(name: object) -> HessboostError:
    return HessboostError(
        f"column {name!r} is categorical; pass enable_categorical=True to train on its categories"
    )


def _sparse_csr(data: Any) -> Any:
    """``data`` as a canonical scipy CSR matrix, or ``None`` if it is not a
    scipy sparse matrix or array."""
    sparse = sys.modules.get("scipy.sparse")
    if sparse is None or not sparse.issparse(data):
        return None
    csr = data.tocsr()
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
    return csr


def as_float32(values: object, name: str) -> NDArray[np.float32]:
    """``values`` as a C-contiguous ``float32`` array (no copy when it
    already is one)."""
    if values is None:
        raise TypeError(f"{name} must be array-like, got None")
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name}: complex data not supported")
    return np.ascontiguousarray(array, dtype=np.float32)


def as_vector(values: ArrayLike, name: str) -> NDArray[np.float32]:
    """``values`` as a 1-D ``float32`` array."""
    array = as_float32(values, name)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise HessboostError(f"{name} must be 1-D, got shape {array.shape}")
    return array


def _check_names(names: Sequence[str], n_cols: int) -> list[str]:
    names = [str(name) for name in names]
    if len(names) != n_cols:
        raise HessboostError(f"{len(names)} feature names for {n_cols} features")
    if len(set(names)) != len(names):
        raise HessboostError("feature names must be unique")
    return names


def _categorical_columns(feature_types: Sequence[str], n_cols: int) -> list[int]:
    if len(feature_types) != n_cols:
        raise HessboostError(f"{len(feature_types)} feature types for {n_cols} features")
    columns = []
    for column, kind in enumerate(feature_types):
        if kind == "c":
            columns.append(column)
        elif kind not in _NUMERICAL_TYPES:
            raise HessboostError(
                f"feature type {kind!r} of feature {column} is not 'q' (numerical) or 'c' "
                "(categorical)"
            )
    return columns


def _pandas_frame_values(
    frame: pd.DataFrame,
    enable_categorical: bool,
    reference: Categories | None,
    unseen: Collection[int] = (),
) -> tuple[NDArray[np.float32], list[str], FeatureTypes, Categories]:
    """A pandas frame's values as ``float32``: numeric and boolean columns as is
    (``NA`` missing), category columns as their codes (``NaN`` missing),
    re-coded to the ``reference`` categories of a trained model. A value the
    reference lacks is missing, except in the ``unseen`` columns, where it
    gets the code one past the reference's categories."""
    pd = _pandas()
    rows, cols = frame.shape
    values = np.empty((rows, cols), dtype=np.float32)
    types: FeatureTypes = []
    categories: Categories = {}
    for column, (name, series) in enumerate(frame.items()):
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            if not enable_categorical:
                raise _refuse_categorical(name)
            known = list(dtype.categories)
            wanted = None if reference is None else reference.get(column)
            codes = series.cat.codes.to_numpy()
            if wanted is not None and wanted != known:
                # Old code -> position in the training categories (-1: unseen).
                recode = pd.Index(wanted).get_indexer(dtype.categories)
                if column in unseen:
                    recode = np.where(recode >= 0, recode, len(wanted))
                # Only present codes index `recode`, which is empty when
                # the frame's categories are.
                present = codes >= 0
                recoded = np.full(codes.shape, -1, dtype=np.int64)
                recoded[present] = recode[codes[present]]
                codes = recoded
                known = wanted
            column_values = codes.astype(np.float32)
            column_values[codes < 0] = np.nan
            types.append("c")
            categories[column] = known
        elif pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
            column_values = series.to_numpy(dtype=np.float32, na_value=np.nan)
            types.append("q")
        else:
            raise TypeError(
                f"column {name!r} has dtype {dtype}; hessboost takes numeric, boolean and "
                "category columns (convert strings with .astype('category'))"
            )
        values[:, column] = column_values
    names = [str(name) for name in frame.columns]
    return values, names, types, categories


def _polars_numeric(pl: Any, dtype: Any) -> bool:
    """Whether a polars ``dtype`` reads as a number: numeric (``Decimal``
    included), ``Boolean``, or ``Null`` (a column of nothing but missing)."""
    return bool(dtype.is_numeric()) or dtype == pl.Boolean or dtype == pl.Null


def _refuse_polars_dtype(pl: Any, what: str, dtype: Any, *, categorical: bool) -> TypeError:
    """The refusal of a polars column of ``dtype``, with the conversion that
    would make it acceptable where there is one. ``categorical`` columns
    are accepted for features, not for metadata."""
    extension = getattr(pl, "Extension", None)
    if extension is not None and isinstance(dtype, extension):
        hint = "; take its storage values with .ext.storage()"
    elif dtype.is_temporal():
        hint = "; convert it to a number, e.g. with .dt.epoch() or .to_physical()"
    elif dtype == pl.String and categorical:
        hint = "; convert strings with .cast(pl.Categorical)"
    else:
        hint = ""
    accepted = (
        "numeric, Boolean, Enum and Categorical columns" if categorical else "numeric or Boolean"
    )
    return TypeError(f"{what} has dtype {dtype}; hessboost takes {accepted}{hint}")


_CONVERSION_BLOCK_BYTES = 64 << 20
"""The float32 output a polars frame is converted per row block (bounding
the intermediate ``Float32`` frame)."""


def _polars_frame_values(
    frame: pl.DataFrame,
    enable_categorical: bool,
    reference: Categories | None,
    unseen: Collection[int] = (),
) -> tuple[NDArray[np.float32], list[str], FeatureTypes, Categories]:
    """A polars frame's values as ``float32``: numeric and boolean columns as
    is (null missing), ``Enum`` and ``Categorical`` columns as positions in
    their categories (null missing), re-coded to the ``reference``
    categories of a trained model (a value they lack is missing, except in
    the ``unseen`` columns, where it gets the code one past them). An
    ``Enum``'s categories are its dtype's; a ``Categorical``'s are its
    values, sorted (as pandas infers them), since its physical codes index
    a pool other columns share.

    The conversion is a polars ``select`` of ``Float32`` expressions, which
    polars evaluates in parallel, written row-major by
    ``to_numpy(order="c")``, over row blocks of ``_CONVERSION_BLOCK_BYTES``
    of output so the intermediate frame stays small. Columns are addressed
    by position (``pl.nth``): ``pl.col`` parses a name that looks like a
    regex as one."""
    pl = _polars()
    exprs: list[Any] = []
    types: FeatureTypes = []
    categories: Categories = {}
    for column, (name, dtype) in enumerate(frame.schema.items()):
        if isinstance(dtype, (pl.Categorical, pl.Enum)):
            if not enable_categorical:
                raise _refuse_categorical(name)
            if isinstance(dtype, pl.Enum):
                known: list[Any] = dtype.categories.to_list()
            else:
                known = sorted(frame[:, column].unique().drop_nulls().cast(pl.String).to_list())
            wanted = None if reference is None else reference.get(column)
            if wanted is not None:
                known = wanted
            # Polars categories are strings; a reference's other values
            # (a pandas model's integer categories) match nothing. Null maps
            # to null explicitly: `default` would replace it too.
            positions = [
                (value, code) for code, value in enumerate(known) if isinstance(value, str)
            ]
            other = float(len(known)) if wanted is not None and column in unseen else None
            exprs.append(
                pl.nth(column)
                .cast(pl.String)
                .replace_strict(
                    [*(value for value, _ in positions), None],
                    [*(code for _, code in positions), None],
                    default=other,
                    return_dtype=pl.Float32,
                )
            )
            types.append("c")
            categories[column] = known
        elif _polars_numeric(pl, dtype):
            exprs.append(pl.nth(column).cast(pl.Float32))
            types.append("q")
        else:
            raise _refuse_polars_dtype(pl, f"column {name!r}", dtype, categorical=True)
    values = np.empty((frame.height, len(exprs)), dtype=np.float32)
    if exprs:
        # In row blocks: a block's Float32 frame is the only intermediate,
        # and every expression is Float32, so `to_numpy` gives a float32
        # block (nulls NaN) that is copied into place.
        block = max(1, _CONVERSION_BLOCK_BYTES // (4 * len(exprs)))
        for start in range(0, frame.height, block):
            part = frame.slice(start, block).select(exprs).to_numpy(order="c")
            target = values[start : start + block]
            if part.shape != target.shape:
                # polars 1.0 drops a selected column whose name reads as a
                # regex (`^...$`); never let numpy broadcast over that.
                raise HessboostError(
                    f"polars converted {part.shape[1]} of {len(exprs)} columns; a column name "
                    "that reads as a regex (^...$) needs a newer polars"
                )
            target[...] = part
    return values, list(frame.columns), types, categories


def features(
    data: object,
    *,
    missing: float,
    feature_names: Sequence[str] | None,
    feature_types: Sequence[str] | None,
    enable_categorical: bool,
    reference: Categories | None,
    info: Mapping[str, object],
    unseen: Collection[int] = (),
) -> Features:
    """Converts ``data`` and builds the native matrix with ``info``
    attached. ``reference`` holds a trained model's categories, to which
    frame categorical columns are re-coded; values it lacks are missing,
    except in the ``unseen`` columns, where they are coded one past its
    categories (a target encoder's unseen category). A polars ``LazyFrame``
    is collected (:func:`take_metadata` has already checked that nothing
    needs aligning with its rows)."""
    data = collect_frame(data)
    names: list[str] | None = None
    types: FeatureTypes | None = None
    categories: Categories = {}
    csr = _sparse_csr(data)
    if csr is not None:
        n_cols = int(csr.shape[1])
        values = None
    elif _is_pandas_frame(data):
        values, names, types, categories = _pandas_frame_values(
            data, enable_categorical, reference, unseen
        )
        n_cols = values.shape[1]
    elif _is_polars_frame(data):
        values, names, types, categories = _polars_frame_values(
            data, enable_categorical, reference, unseen
        )
        n_cols = values.shape[1]
    else:
        values = as_float32(data, "data")
        if values.ndim != 2:
            raise ValueError(
                f"data must be 2-D (rows, features), got shape {values.shape}; reshape a "
                "single feature with x.reshape(-1, 1) or a single row with x.reshape(1, -1)"
            )
        n_cols = values.shape[1]
    if feature_names is not None:
        names = _check_names(feature_names, n_cols)
    if feature_types is not None:
        types = [str(kind) for kind in feature_types]
        if categories and any(types[column] != "c" for column in categories):
            raise HessboostError("feature_types marks a categorical column as numerical")
    categorical = [] if types is None else _categorical_columns(types, n_cols)
    full_info = {**info, "categorical": categorical}
    if csr is not None:
        core = _hessboost.DMatrix.csr(
            (
                np.ascontiguousarray(csr.indptr, dtype=np.int64),
                np.ascontiguousarray(csr.indices, dtype=np.int64),
                np.ascontiguousarray(csr.data, dtype=np.float32),
            ),
            n_cols,
            full_info,
        )
    else:
        assert values is not None
        core = _hessboost.DMatrix.dense(values, float(missing), full_info)
    return Features(core, names, types, categories)


def group_sizes(group: ArrayLike | None, qid: ArrayLike | None) -> list[int] | None:
    """Query-group sizes from XGBoost's ``group`` (sizes) or ``qid`` (one
    sorted query id per row)."""
    if group is not None and qid is not None:
        raise HessboostError("pass group or qid, not both")
    if group is not None:
        sizes = np.asarray(group)
        if sizes.ndim != 1 or not np.issubdtype(sizes.dtype, np.integer):
            raise HessboostError("group must be a 1-D array of integer group sizes")
        return [int(size) for size in sizes]
    if qid is None:
        return None
    ids = np.asarray(qid).reshape(-1)
    if ids.size == 0:
        raise HessboostError("qid is empty")
    if np.any(ids[1:] < ids[:-1]):
        raise HessboostError("qid must be sorted (rows of a query are contiguous)")
    # Compared, not subtracted: query ids may be strings.
    starts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    bounds = np.concatenate([[0], starts, [ids.size]])
    return [int(size) for size in np.diff(bounds)]


def info(
    *,
    label: ArrayLike | None = None,
    weight: ArrayLike | None = None,
    base_margin: ArrayLike | None = None,
    group: ArrayLike | None = None,
    qid: ArrayLike | None = None,
    label_lower_bound: ArrayLike | None = None,
    label_upper_bound: ArrayLike | None = None,
    feature_weights: ArrayLike | None = None,
) -> dict[str, object]:
    """The native ``info`` mapping for the given metadata (``None`` fields
    are left out)."""
    out: dict[str, object] = {}
    if label is not None:
        out["label"] = as_float32(label, "label")
    if weight is not None:
        out["weight"] = as_vector(weight, "weight")
    if base_margin is not None:
        out["base_margin"] = as_float32(base_margin, "base_margin")
    sizes = group_sizes(group, qid)
    if sizes is not None:
        out["group"] = sizes
    if (label_lower_bound is None) != (label_upper_bound is None):
        raise HessboostError("pass label_lower_bound and label_upper_bound together")
    if label_lower_bound is not None and label_upper_bound is not None:
        out["label_bounds"] = (
            as_vector(label_lower_bound, "label_lower_bound"),
            as_vector(label_upper_bound, "label_upper_bound"),
        )
    if feature_weights is not None:
        out["feature_weights"] = as_vector(feature_weights, "feature_weights")
    return out


_PER_ROW_FIELDS = (
    "label",
    "weight",
    "base_margin",
    "qid",
    "label_lower_bound",
    "label_upper_bound",
)
"""The metadata with one value per row, which a frame can supply by column
name (``label`` and ``base_margin`` several, for a label matrix or
per-output margins)."""

_MATRIX_FIELDS = frozenset({"label", "base_margin"})


def _column_names(value: object) -> list[str] | None:
    """``value`` read as column names: a ``str``, or a non-empty list or
    tuple of them; ``None`` for anything else (an array)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def _lazy_alignment_error(field: str) -> HessboostError:
    by_name = "qid='...' for the query ids" if field == "group" else f"{field}='...'"
    return HessboostError(
        f"{field} is an array, but data is a polars LazyFrame, which hessboost collects itself: "
        f"its rows may not be in the order of a separately collected frame (the streaming engine "
        f"keeps no row order after a join or group_by). Name a column of the frame instead "
        f"({by_name}), or collect it (data.collect()) and pass the DataFrame along with the array"
    )


def _pandas_columns(frame: pd.DataFrame, field: str, names: list[str]) -> NDArray[Any]:
    """The ``names`` columns of a pandas frame as ``field``'s values:
    ``float32`` with ``NA`` missing (``qid`` keeps its values), ``(rows,)``
    for one name and ``(rows, k)`` for ``k``."""
    pd = _pandas()
    if field == "qid":
        return frame[names[0]].to_numpy()
    for name in names:
        dtype = frame[name].dtype
        if not (pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype)):
            raise TypeError(
                f"{field} column {name!r} has dtype {dtype}; it must be numeric or boolean"
            )
    if len(names) == 1:
        return frame[names[0]].to_numpy(dtype=np.float32, na_value=np.nan)
    return frame[names].to_numpy(dtype=np.float32, na_value=np.nan)


def _polars_columns(frame: pl.DataFrame, field: str, names: list[str]) -> NDArray[Any]:
    """The ``names`` columns of a polars frame as ``field``'s values:
    ``float32`` with null missing (``qid`` keeps its values), ``(rows,)``
    for one name and ``(rows, k)`` for ``k``. Columns are looked up by
    exact name (``get_column``), not through ``pl.col``'s regex parsing."""
    pl = _polars()
    if field == "qid":
        return frame.get_column(names[0]).to_numpy()
    for name in names:
        dtype = frame.schema[name]
        if not _polars_numeric(pl, dtype):
            raise _refuse_polars_dtype(pl, f"{field} column {name!r}", dtype, categorical=False)
    columns = [frame.get_column(name).cast(pl.Float32).to_numpy() for name in names]
    return columns[0] if len(columns) == 1 else np.column_stack(columns)


def take_metadata(data: object, **fields: object) -> tuple[object, dict[str, object]]:
    """``data`` and its metadata ``fields`` (:func:`info`'s arguments) with
    the per-row ones a frame supplies by column name resolved: a ``str``
    (``label="y"``; ``label`` and ``base_margin`` also a list of names, for
    a label matrix or per-output margins) names a column of a pandas or
    polars ``data``, which becomes that field's values and leaves the
    features. The returned data is the frame without those columns.

    A polars ``LazyFrame`` is collected here, once, and may only be given
    per-row metadata (``group`` sizes included) by column name: a frame the
    caller collected separately need not come out in the same row order
    under polars' streaming engine, so an array alongside it is refused.

    Raises:
        HessboostError: A name is not a column of ``data``, several names
            are given for a one-column field, ``group`` (sizes, not a
            column) is given by name, or an array accompanies a
            ``LazyFrame``.
        TypeError: A name is given for data that is not a frame, or names
            a column that is not numeric or boolean.
    """
    named = {
        field: names
        for field, value in fields.items()
        if (names := _column_names(value)) is not None
    }
    if _is_polars_lazyframe(data):
        for field, value in fields.items():
            if value is not None and field not in named:
                raise _lazy_alignment_error(field)
    frame = collect_frame(data)
    if not named:
        return frame, fields
    if not (_is_pandas_frame(frame) or _is_polars_frame(frame)):
        field = next(iter(named))
        raise TypeError(
            f"{field}={fields[field]!r} names a column, but data is a "
            f"{type(frame).__name__}, not a DataFrame"
        )
    for field, names in named.items():
        if field not in _PER_ROW_FIELDS:
            hint = "; query groups come from qid= as a column name" if field == "group" else ""
            raise HessboostError(
                f"{field} is not one value per row, so it cannot be a column name{hint}"
            )
        if field not in _MATRIX_FIELDS and len(names) != 1:
            raise HessboostError(f"{field} takes one column name, got {names}")
        if len(set(names)) != len(names):
            raise HessboostError(f"{field} names a column twice: {names}")
        for name in names:
            if name not in frame.columns:
                raise HessboostError(f"{field} column {name!r} is not a column of data")
    taken = list(dict.fromkeys(name for names in named.values() for name in names))
    resolved = dict(fields)
    if _is_pandas_frame(frame):
        for field, names in named.items():
            resolved[field] = _pandas_columns(frame, field, names)
        return frame.drop(columns=taken), resolved
    assert _is_polars_frame(frame)
    for field, names in named.items():
        resolved[field] = _polars_columns(frame, field, names)
    kept = [column for column, name in enumerate(frame.columns) if name not in taken]
    return frame[:, kept], resolved
