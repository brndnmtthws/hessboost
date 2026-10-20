//! `DMatrix`: hessboost's dataset container, built from borrowed row-major
//! `float32` arrays (the Python layer converts everything else), and
//! `BinnedMatrix`, a matrix's histogram bins without its values.

use crate::borrowed::BorrowedArray;
use crate::errors::{DetachExt, refuse};
use hessboost::conformal::Interval;
use hessboost::data::{
    BinnedMatrix as RustBinned, DMatrix as RustMatrix, FeatureType, GroupInfo, ViewCuts,
};
use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{
    IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::Arc;

/// The elements of `array` in row-major order, borrowed. `as_slice` also
/// accepts a Fortran-ordered buffer (in its column-major memory order),
/// which would silently transpose a matrix, so only C order is accepted; the
/// Python layer converts everything else first.
pub(crate) fn row_major<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    array: &'a numpy::PyReadonlyArray<'_, T, D>,
    name: &str,
) -> PyResult<&'a [T]> {
    if !array.is_c_contiguous() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a C-contiguous array"
        )));
    }
    Ok(array.as_slice()?)
}

/// [`row_major`] of a `(rows,)` or `(rows, columns)` array, with its
/// column count (`1` for a vector); any other rank is refused.
fn row_major_vector_or_matrix<'a, T: numpy::Element>(
    array: &'a PyReadonlyArrayDyn<'_, T>,
    name: &str,
) -> PyResult<(&'a [T], usize)> {
    let columns = match array.shape() {
        [_] => 1,
        [_, columns] => *columns,
        shape => {
            return Err(refuse(format!(
                "{name} must be 1-D or 2-D, got shape {shape:?}"
            )));
        }
    };
    Ok((row_major(array, name)?, columns))
}

/// `(rows, 2)` `[lower, upper]` bounds.
pub(crate) fn intervals_to_numpy<T: numpy::Element>(
    py: Python<'_>,
    bounds: Vec<Interval<T>>,
) -> PyResult<Bound<'_, PyArrayDyn<T>>> {
    let rows = bounds.len();
    let values = bounds
        .into_iter()
        .flat_map(|iv| [iv.lower, iv.upper])
        .collect();
    to_numpy(py, values, &[rows, 2])
}

/// A row-major matrix or vector as a numpy array of `shape`.
pub(crate) fn to_numpy<'py, T: numpy::Element>(
    py: Python<'py>,
    values: Vec<T>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    let array = ArrayD::from_shape_vec(IxDyn(shape), values)
        .map_err(|error| refuse(format!("unexpected result shape: {error}")))?;
    Ok(array.into_pyarray(py))
}

/// Per-row and per-feature metadata, passed from Python as a `dict` whose
/// absent keys leave the matrix's current value.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub struct Info<'py> {
    /// `(rows,)` or `(rows, targets)` labels.
    #[pyo3(default)]
    label: Option<PyReadonlyArrayDyn<'py, f32>>,
    /// `(rows,)` instance weights, or one per query group of ranking data.
    #[pyo3(default)]
    weight: Option<PyReadonlyArray1<'py, f32>>,
    /// `(rows,)` or `(rows, outputs)` starting margins.
    #[pyo3(default)]
    base_margin: Option<PyReadonlyArrayDyn<'py, f32>>,
    /// Contiguous query-group sizes.
    #[pyo3(default)]
    group: Option<Vec<usize>>,
    /// `(lower, upper)` survival label bounds, `(rows,)` each.
    #[pyo3(default)]
    label_bounds: Option<(PyReadonlyArray1<'py, f32>, PyReadonlyArray1<'py, f32>)>,
    /// `(features,)` column-sampling weights.
    #[pyo3(default)]
    feature_weights: Option<PyReadonlyArray1<'py, f32>>,
    /// Indices of the categorical columns (every other column is
    /// numerical); `None` keeps the current types.
    categorical: Option<Vec<usize>>,
}

/// [`Info`]'s arrays as borrowed slices, which can cross into a detached
/// closure.
struct InfoSlices<'a> {
    label: Option<(&'a [f32], usize)>,
    weight: Option<&'a [f32]>,
    base_margin: Option<&'a [f32]>,
    group: Option<&'a [usize]>,
    label_bounds: Option<(&'a [f32], &'a [f32])>,
    feature_weights: Option<&'a [f32]>,
    categorical: Option<&'a [usize]>,
}

impl Info<'_> {
    fn slices(&self) -> PyResult<InfoSlices<'_>> {
        let label = match &self.label {
            None => None,
            Some(label) => Some(row_major_vector_or_matrix(label, "label")?),
        };
        let base_margin = match &self.base_margin {
            None => None,
            Some(margin) => Some(row_major_vector_or_matrix(margin, "base_margin")?.0),
        };
        let label_bounds = match &self.label_bounds {
            None => None,
            Some((lower, upper)) => Some((
                row_major(lower, "label_lower_bound")?,
                row_major(upper, "label_upper_bound")?,
            )),
        };
        Ok(InfoSlices {
            label,
            weight: self
                .weight
                .as_ref()
                .map(|weight| row_major(weight, "weight"))
                .transpose()?,
            base_margin,
            group: self.group.as_deref(),
            label_bounds,
            feature_weights: self
                .feature_weights
                .as_ref()
                .map(|weights| row_major(weights, "feature_weights"))
                .transpose()?,
            categorical: self.categorical.as_deref(),
        })
    }
}

impl InfoSlices<'_> {
    /// `matrix` with every given field attached (each setter validates and
    /// copies).
    fn apply(&self, mut matrix: RustMatrix) -> hessboost::error::Result<RustMatrix> {
        if let Some(columns) = self.categorical {
            let mut types = vec![FeatureType::Numerical; matrix.n_cols()];
            for &column in columns {
                let slot = types.get_mut(column).ok_or_else(|| {
                    hessboost::error::HessboostError::invalid_data(
                        "feature_types",
                        format!(
                            "categorical column {column} is out of range for {} columns",
                            matrix.n_cols()
                        ),
                    )
                })?;
                *slot = FeatureType::Categorical;
            }
            matrix = matrix.with_feature_types(&types)?;
        }
        if let Some((label, targets)) = self.label {
            matrix = matrix.with_label_matrix(label, targets)?;
        }
        if let Some(group) = self.group {
            matrix = matrix.with_group_sizes(group)?;
        }
        if let Some(weight) = self.weight {
            // As in XGBoost, ranking data takes one weight per query group.
            let groups = matrix.group().map(GroupInfo::num_groups);
            matrix = match groups {
                Some(groups) if weight.len() == groups && groups != matrix.n_rows() => {
                    matrix.with_group_weights(weight)?
                }
                _ => matrix.with_weights(weight)?,
            };
        }
        if let Some(margin) = self.base_margin {
            matrix = matrix.with_base_margin(margin)?;
        }
        if let Some((lower, upper)) = self.label_bounds {
            matrix = matrix.with_label_bounds(lower, upper)?;
        }
        if let Some(weights) = self.feature_weights {
            matrix = matrix.with_feature_weights(weights)?;
        }
        Ok(matrix)
    }

    /// `binned` with every given field attached; weights and feature types,
    /// which shaped its cuts, are refused.
    fn apply_binned(&self, mut binned: RustBinned) -> hessboost::error::Result<RustBinned> {
        let fixed = |input, reason| {
            Err(hessboost::error::HessboostError::invalid_data(
                input, reason,
            ))
        };
        if self.weight.is_some() {
            return fixed(
                "weight",
                "a binned matrix keeps the weights its cuts were sketched with",
            );
        }
        if self.categorical.is_some() {
            return fixed(
                "feature_types",
                "a binned matrix keeps the feature types its cuts were computed for",
            );
        }
        if let Some((label, targets)) = self.label {
            binned = binned.with_label_matrix(label, targets)?;
        }
        if let Some(group) = self.group {
            binned = binned.with_group_sizes(group)?;
        }
        if let Some(margin) = self.base_margin {
            binned = binned.with_base_margin(margin)?;
        }
        if let Some((lower, upper)) = self.label_bounds {
            binned = binned.with_label_bounds(lower, upper)?;
        }
        if let Some(weights) = self.feature_weights {
            binned = binned.with_feature_weights(weights)?;
        }
        Ok(binned)
    }
}

/// `(rows,)` values, or `(rows, width)` when each row holds `width`.
fn rows_array<'py>(
    py: Python<'py>,
    values: Option<&[f32]>,
    rows: usize,
) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
    values
        .map(|values| {
            let width = values.len() / rows.max(1);
            let shape: &[usize] = if width == 1 { &[rows] } else { &[rows, width] };
            to_numpy(py, values.to_vec(), shape)
        })
        .transpose()
}

/// `(rows,)` labels, or `(rows, targets)` for a label matrix.
fn labels_array<'py>(
    py: Python<'py>,
    labels: Option<&[f32]>,
    rows: usize,
    targets: usize,
) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
    labels
        .map(|labels| {
            let shape: &[usize] = if targets == 1 {
                &[rows]
            } else {
                &[rows, targets]
            };
            to_numpy(py, labels.to_vec(), shape)
        })
        .transpose()
}

/// Query-group sizes, if any.
fn group_sizes(group: Option<&GroupInfo>) -> Option<Vec<usize>> {
    group.map(|group| {
        group
            .iter_ranges()
            .map(|(start, end)| end - start)
            .collect()
    })
}

/// Indices of the categorical columns.
fn categorical_columns(types: &[FeatureType]) -> Vec<usize> {
    types
        .iter()
        .enumerate()
        .filter(|(_, kind)| **kind == FeatureType::Categorical)
        .map(|(column, _)| column)
        .collect()
}

/// An immutable hessboost dataset: features plus labels, weights, margins,
/// groups and feature types. Build a changed copy with `with_info`.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct DMatrix {
    pub(crate) inner: RustMatrix,
}

#[pymethods]
impl DMatrix {
    /// A dense matrix from a C-contiguous `(rows, features)` array, where
    /// values equal to `missing` (any NaN when `missing` is NaN) are absent.
    #[staticmethod]
    fn dense(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f32>,
        missing: f32,
        info: Info<'_>,
    ) -> PyResult<Self> {
        let [rows, columns] = [data.shape()[0], data.shape()[1]];
        let values = row_major(&data, "data")?;
        let info = info.slices()?;
        let inner = py.detached(|| {
            let matrix = RustMatrix::from_dense_with_missing(values, rows, columns, missing)?;
            info.apply(matrix)
        })?;
        Ok(Self { inner })
    }

    /// A dense matrix over `data`, a read-only, C-contiguous, aligned
    /// `(rows, features)` array it keeps alive and reads in place instead of
    /// copying; values equal to `missing` are absent.
    #[staticmethod]
    fn borrowed(
        py: Python<'_>,
        data: Bound<'_, PyArray2<f32>>,
        missing: f32,
        info: Info<'_>,
    ) -> PyResult<Self> {
        let [rows, columns] = [data.shape()[0], data.shape()[1]];
        let values = Arc::new(BorrowedArray::new(&data)?);
        let info = info.slices()?;
        let inner = py.detached(|| {
            let matrix = RustMatrix::from_dense_shared(values, rows, columns, missing)?;
            info.apply(matrix)
        })?;
        Ok(Self { inner })
    }

    /// A sparse matrix from CSR arrays; absent entries are missing.
    #[staticmethod]
    fn csr(
        py: Python<'_>,
        csr: (
            PyReadonlyArray1<'_, i64>,
            PyReadonlyArray1<'_, i64>,
            PyReadonlyArray1<'_, f32>,
        ),
        n_cols: usize,
        info: Info<'_>,
    ) -> PyResult<Self> {
        let (indptr, indices, values) = csr;
        let (indptr, indices, values) = (
            row_major(&indptr, "indptr")?,
            row_major(&indices, "indices")?,
            row_major(&values, "values")?,
        );
        let info = info.slices()?;
        let inner = py.detached(|| {
            let convert = |name: &'static str, value: i64| {
                usize::try_from(value).map_err(|_| {
                    hessboost::error::HessboostError::invalid_data(
                        name,
                        format!("negative entry {value}"),
                    )
                })
            };
            let indptr = indptr
                .iter()
                .map(|&offset| convert("csr indptr", offset))
                .collect::<hessboost::error::Result<Vec<_>>>()?;
            let indices = indices
                .iter()
                .map(|&index| {
                    convert("csr indices", index).and_then(|index| {
                        u32::try_from(index).map_err(|_| {
                            hessboost::error::HessboostError::invalid_data(
                                "csr indices",
                                format!("column {index} does not fit in 32 bits"),
                            )
                        })
                    })
                })
                .collect::<hessboost::error::Result<Vec<_>>>()?;
            let matrix = RustMatrix::from_csr(indptr, indices, values.to_vec(), n_cols)?;
            info.apply(matrix)
        })?;
        Ok(Self { inner })
    }

    /// A copy with the fields present in `info` replaced, sharing this
    /// matrix's feature values (so only the metadata is copied).
    fn with_info(&self, py: Python<'_>, info: Info<'_>) -> PyResult<Self> {
        let info = info.slices()?;
        let inner = py.detached(|| info.apply(self.inner.clone()))?;
        Ok(Self { inner })
    }

    /// The rows `rows`, in that order, with their metadata (query groups
    /// only when `rows` lists whole groups, each in row order): a view
    /// sharing this matrix's feature values, cut by its own rows (`"own"`)
    /// or binned with this matrix's cuts (`"source"`).
    fn select_rows(
        &self,
        py: Python<'_>,
        rows: PyReadonlyArray1<'_, i64>,
        cuts: &str,
    ) -> PyResult<Self> {
        let cuts = match cuts {
            "own" => ViewCuts::Own,
            "source" => ViewCuts::Source,
            other => {
                return Err(refuse(format!(
                    "cuts must be \"own\" or \"source\", got {other:?}"
                )));
            }
        };
        let rows = row_major(&rows, "rows")?;
        let n = self.inner.n_rows();
        let rows = rows
            .iter()
            .map(|&row| {
                usize::try_from(row)
                    .ok()
                    .filter(|&row| row < n)
                    .ok_or_else(|| refuse(format!("row index {row} is out of range for {n} rows")))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let inner = py.detached(|| self.inner.select_rows_with_cuts(&rows, cuts))?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_row(&self) -> usize {
        self.inner.n_rows()
    }

    #[getter]
    fn num_col(&self) -> usize {
        self.inner.n_cols()
    }

    #[getter]
    fn label<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        let m = &self.inner;
        labels_array(py, m.labels(), m.n_rows(), m.n_targets())
    }

    #[getter]
    fn weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        rows_array(py, self.inner.weights(), self.inner.n_rows())
    }

    #[getter]
    fn base_margin<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        rows_array(py, self.inner.base_margin(), self.inner.n_rows())
    }

    #[getter]
    fn label_lower_bound<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        rows_array(py, self.inner.label_lower_bound(), self.inner.n_rows())
    }

    #[getter]
    fn label_upper_bound<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        rows_array(py, self.inner.label_upper_bound(), self.inner.n_rows())
    }

    #[getter]
    fn feature_weights<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        rows_array(py, self.inner.feature_weights(), self.inner.n_cols())
    }

    /// Query-group sizes, if any.
    #[getter]
    fn group(&self) -> Option<Vec<usize>> {
        group_sizes(self.inner.group())
    }

    /// Indices of the categorical columns.
    #[getter]
    fn categorical(&self) -> Vec<usize> {
        categorical_columns(self.inner.feature_types())
    }
}

/// A training matrix of a `DMatrix`'s histogram bins, without its feature
/// values; only training takes it. Build a changed copy with `with_info`.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct BinnedMatrix {
    pub(crate) inner: RustBinned,
}

#[pymethods]
impl BinnedMatrix {
    /// `data` binned with at most `max_bin` bins per numerical feature,
    /// keeping its metadata.
    #[new]
    fn new(py: Python<'_>, data: &DMatrix, max_bin: usize) -> PyResult<Self> {
        let inner = py.detached(|| RustBinned::new(&data.inner, max_bin))?;
        Ok(Self { inner })
    }

    /// A copy with the fields present in `info` replaced; weights and
    /// feature types are refused.
    fn with_info(&self, py: Python<'_>, info: Info<'_>) -> PyResult<Self> {
        let info = info.slices()?;
        let inner = py.detached(|| info.apply_binned(self.inner.clone()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_row(&self) -> usize {
        self.inner.n_rows()
    }

    #[getter]
    fn num_col(&self) -> usize {
        self.inner.n_cols()
    }

    #[getter]
    fn max_bin(&self) -> usize {
        self.inner.max_bin()
    }

    #[getter]
    fn label<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        let m = &self.inner;
        labels_array(py, m.labels(), m.n_rows(), m.n_targets())
    }

    #[getter]
    fn weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        rows_array(py, self.inner.weights(), self.inner.n_rows())
    }

    #[getter]
    fn base_margin<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        rows_array(py, self.inner.base_margin(), self.inner.n_rows())
    }

    #[getter]
    fn label_lower_bound<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        rows_array(py, self.inner.label_lower_bound(), self.inner.n_rows())
    }

    #[getter]
    fn label_upper_bound<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        rows_array(py, self.inner.label_upper_bound(), self.inner.n_rows())
    }

    #[getter]
    fn feature_weights<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        rows_array(py, self.inner.feature_weights(), self.inner.n_cols())
    }

    /// Query-group sizes, if any.
    #[getter]
    fn group(&self) -> Option<Vec<usize>> {
        group_sizes(self.inner.group())
    }

    /// Indices of the categorical columns.
    #[getter]
    fn categorical(&self) -> Vec<usize> {
        categorical_columns(self.inner.feature_types())
    }
}
