//! `DMatrix`: hessboost's dataset container, built from borrowed row-major
//! `float32` arrays (the Python layer converts everything else).

use crate::errors::{OrRaise, refuse};
use hessboost::data::{DMatrix as RustMatrix, FeatureType, GroupInfo};
use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

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
            Some(label) => {
                let targets = match label.shape() {
                    [_] => 1,
                    [_, targets] => *targets,
                    shape => {
                        return Err(refuse(format!(
                            "label must be 1-D or 2-D, got shape {shape:?}"
                        )));
                    }
                };
                Some((row_major(label, "label")?, targets))
            }
        };
        let base_margin = match &self.base_margin {
            None => None,
            Some(margin) if margin.ndim() <= 2 => Some(row_major(margin, "base_margin")?),
            Some(margin) => {
                return Err(refuse(format!(
                    "base_margin must be 1-D or 2-D, got shape {:?}",
                    margin.shape()
                )));
            }
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
                    hessboost::error::HessboostError::invalid_param(
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
}

/// An immutable hessboost dataset: features plus labels, weights, margins,
/// groups and feature types. Build a changed copy with `with_info`.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct DMatrix {
    pub(crate) inner: RustMatrix,
}

impl DMatrix {
    fn vector<'py>(
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
        let inner = py
            .detach(|| {
                let matrix = RustMatrix::from_dense_with_missing(values, rows, columns, missing)?;
                info.apply(matrix)
            })
            .or_raise()?;
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
        let inner = py
            .detach(|| {
                let convert = |name: &'static str, value: i64| {
                    usize::try_from(value).map_err(|_| {
                        hessboost::error::HessboostError::invalid_param(
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
                                hessboost::error::HessboostError::invalid_param(
                                    "csr indices",
                                    format!("column {index} does not fit in 32 bits"),
                                )
                            })
                        })
                    })
                    .collect::<hessboost::error::Result<Vec<_>>>()?;
                let matrix = RustMatrix::from_csr(indptr, indices, values.to_vec(), n_cols)?;
                info.apply(matrix)
            })
            .or_raise()?;
        Ok(Self { inner })
    }

    /// A copy with the fields present in `info` replaced.
    fn with_info(&self, py: Python<'_>, info: Info<'_>) -> PyResult<Self> {
        let info = info.slices()?;
        let inner = py.detach(|| info.apply(self.inner.clone())).or_raise()?;
        Ok(Self { inner })
    }

    /// The rows `rows`, in that order, with their metadata (groups are not
    /// carried over).
    fn select_rows(&self, py: Python<'_>, rows: PyReadonlyArray1<'_, i64>) -> PyResult<Self> {
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
        let inner = py.detach(|| self.inner.select_rows(&rows)).or_raise()?;
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
        let rows = self.inner.n_rows();
        self.inner
            .labels()
            .map(|labels| {
                let targets = self.inner.n_targets();
                if targets == 1 {
                    to_numpy(py, labels.to_vec(), &[rows])
                } else {
                    to_numpy(py, labels.to_vec(), &[rows, targets])
                }
            })
            .transpose()
    }

    #[getter]
    fn weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        Self::vector(py, self.inner.weights(), self.inner.n_rows())
    }

    #[getter]
    fn base_margin<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        Self::vector(py, self.inner.base_margin(), self.inner.n_rows())
    }

    #[getter]
    fn label_lower_bound<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        Self::vector(py, self.inner.label_lower_bound(), self.inner.n_rows())
    }

    #[getter]
    fn label_upper_bound<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        Self::vector(py, self.inner.label_upper_bound(), self.inner.n_rows())
    }

    #[getter]
    fn feature_weights<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        Self::vector(py, self.inner.feature_weights(), self.inner.n_cols())
    }

    /// Query-group sizes, if any.
    #[getter]
    fn group(&self) -> Option<Vec<usize>> {
        self.inner.group().map(|group| {
            group
                .iter_ranges()
                .map(|(start, end)| end - start)
                .collect()
        })
    }

    /// Indices of the categorical columns.
    #[getter]
    fn categorical(&self) -> Vec<usize> {
        self.inner
            .feature_types()
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == FeatureType::Categorical)
            .map(|(column, _)| column)
            .collect()
    }
}
