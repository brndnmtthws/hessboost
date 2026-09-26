//! `Booster`: a trained or loaded model, its prediction variants, feature
//! importance, slicing, and every model format.

use crate::data::{DMatrix, to_numpy};
use crate::dist::Distributions;
use crate::errors::OrRaise;
use hessboost::model::{BoostedModel, Contributions, ImportanceType, Interactions, Predictions};
use numpy::PyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::ops::Range;
use std::sync::Arc;

/// Model formats, by the names the Python layer uses.
#[derive(Clone, Copy)]
enum Format {
    Binary,
    Json,
    XgboostJson,
    XgboostUbjson,
    /// LightGBM 4.x text models (import only).
    Lightgbm,
}

impl Format {
    fn parse(name: &str) -> PyResult<Self> {
        Ok(match name {
            "binary" => Self::Binary,
            "json" => Self::Json,
            "xgboost-json" => Self::XgboostJson,
            "xgboost-ubjson" => Self::XgboostUbjson,
            "lightgbm" => Self::Lightgbm,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown model format {other:?}; expected \"binary\", \"json\", \
                     \"xgboost-json\", \"xgboost-ubjson\" or (to load) \"lightgbm\""
                )));
            }
        })
    }

    /// The format of `bytes`: a JSON document (`{` then `"` or `}`) is
    /// XGBoost's when it has a `learner` key, else native; any other `{` is
    /// UBJSON (whose keys start with a length marker); a first line `tree`
    /// is a LightGBM text model; everything else is native binary.
    fn detect(bytes: &[u8]) -> Self {
        let text = bytes.trim_ascii_start();
        if let Some(rest) = text.strip_prefix(b"tree")
            && matches!(rest.first(), Some(b'\n' | b'\r'))
        {
            return Self::Lightgbm;
        }
        let mut rest = text.iter().copied();
        if rest.next() != Some(b'{') {
            return Self::Binary;
        }
        match rest.find(|byte| !byte.is_ascii_whitespace()) {
            Some(b'"' | b'}') => {
                if bytes.windows(9).any(|window| window == b"\"learner\"") {
                    Self::XgboostJson
                } else {
                    Self::Json
                }
            }
            _ => Self::XgboostUbjson,
        }
    }
}

fn utf8(bytes: &[u8]) -> hessboost::error::Result<&str> {
    std::str::from_utf8(bytes).map_err(|error| {
        hessboost::error::HessboostError::model_format(format!("text model is not UTF-8: {error}"))
    })
}

/// What a prediction call returns.
#[derive(Clone, Copy)]
enum Kind {
    Value,
    Margin,
    Contribs,
    Interactions,
}

/// `(rows,)` for one value per row, else `(rows, width)`.
fn dense(predictions: Predictions) -> (Vec<f32>, Vec<usize>) {
    let (rows, width) = (predictions.n_rows(), predictions.width());
    let shape = if width == 1 {
        vec![rows]
    } else {
        vec![rows, width]
    };
    (predictions.into_vec(), shape)
}

/// `(rows, features + 1)`, with an output axis for multi-output models.
fn contributions(contribs: Contributions) -> (Vec<f32>, Vec<usize>) {
    let (rows, outputs, width) = (
        contribs.n_rows(),
        contribs.n_outputs(),
        contribs.n_features() + 1,
    );
    let shape = if outputs == 1 {
        vec![rows, width]
    } else {
        vec![rows, outputs, width]
    };
    (contribs.into_vec(), shape)
}

/// `(rows, features + 1, features + 1)`, with an output axis for
/// multi-output models.
fn interactions(interactions: Interactions) -> (Vec<f32>, Vec<usize>) {
    let (rows, outputs, width) = (
        interactions.n_rows(),
        interactions.n_outputs(),
        interactions.n_features() + 1,
    );
    let shape = if outputs == 1 {
        vec![rows, width, width]
    } else {
        vec![rows, outputs, width, width]
    };
    (interactions.into_vec(), shape)
}

/// A trained model, shared read-only.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct Booster {
    pub(crate) model: Arc<BoostedModel>,
}

impl Booster {
    pub(crate) fn new(model: BoostedModel) -> Self {
        Self {
            model: Arc::new(model),
        }
    }

    /// XGBoost's `iteration_range` as iterations: `(begin, 0)` runs through
    /// the last iteration. `None` means the model's default (through
    /// `best_iteration` after early stopping), which callers get from the
    /// crate's range-less methods.
    fn range(&self, (begin, end): (usize, usize)) -> Range<usize> {
        begin..if end == 0 {
            self.model.num_boost_rounds()
        } else {
            end
        }
    }
}

#[pymethods]
impl Booster {
    /// Decodes a model in `format` (`"auto"` detects it from the bytes).
    #[staticmethod]
    fn load(py: Python<'_>, data: &[u8], format: &str) -> PyResult<Self> {
        let format = if format == "auto" {
            Format::detect(data)
        } else {
            Format::parse(format)?
        };
        let model = py
            .detach(|| match format {
                Format::Binary => BoostedModel::from_bytes(data),
                Format::Json => BoostedModel::from_json(utf8(data)?),
                Format::XgboostJson => BoostedModel::from_xgboost_json(utf8(data)?),
                Format::XgboostUbjson => BoostedModel::from_xgboost_ubjson(data),
                Format::Lightgbm => BoostedModel::from_lightgbm_text(utf8(data)?),
            })
            .or_raise()?;
        Ok(Self::new(model))
    }

    /// The model encoded in `format`.
    fn save<'py>(&self, py: Python<'py>, format: &str) -> PyResult<Bound<'py, PyBytes>> {
        let format = Format::parse(format)?;
        let bytes = py
            .detach(|| match format {
                Format::Binary => self.model.to_bytes(),
                Format::Json => self.model.to_json().map(String::into_bytes),
                Format::XgboostJson => self.model.to_xgboost_json().map(String::into_bytes),
                Format::XgboostUbjson => self.model.to_xgboost_ubjson(),
                Format::Lightgbm => Err(hessboost::error::HessboostError::invalid_param(
                    "format",
                    "\"lightgbm\" is an import-only format: LightGBM models load, but do not save",
                )),
            })
            .or_raise()?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Predictions of `kind` (`value`, `margin`, `contribs`,
    /// `interactions`) shaped as XGBoost's Python package shapes them.
    #[pyo3(signature = (data, kind, iteration_range=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        kind: &str,
        iteration_range: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let kind = match kind {
            "value" => Kind::Value,
            "margin" => Kind::Margin,
            "contribs" => Kind::Contribs,
            "interactions" => Kind::Interactions,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown prediction kind {other:?}"
                )));
            }
        };
        let model = &*self.model;
        let matrix = &data.inner;
        let range = iteration_range.map(|range| self.range(range));
        let (values, shape) = py
            .detach(|| -> hessboost::error::Result<_> {
                Ok(match (kind, range) {
                    (Kind::Value, None) => dense(model.predict(matrix)?),
                    (Kind::Value, Some(range)) => dense(model.predict_range(matrix, range)?),
                    (Kind::Margin, None) => dense(model.predict_margin(matrix)?),
                    (Kind::Margin, Some(range)) => {
                        dense(model.predict_margin_range(matrix, range)?)
                    }
                    (Kind::Contribs, None) => contributions(model.predict_contribs(matrix)?),
                    (Kind::Contribs, Some(range)) => {
                        contributions(model.predict_contribs_range(matrix, range)?)
                    }
                    (Kind::Interactions, None) => interactions(model.predict_interactions(matrix)?),
                    (Kind::Interactions, Some(range)) => {
                        interactions(model.predict_interactions_range(matrix, range)?)
                    }
                })
            })
            .or_raise()?;
        to_numpy(py, values, &shape)
    }

    /// The leaf each row reaches in every tree, `(rows, trees)`; ranges
    /// start at iteration 0 and default to every iteration.
    #[pyo3(signature = (data, iteration_range=None))]
    fn predict_leaf<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        iteration_range: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyArrayDyn<i32>>> {
        let model = &*self.model;
        let range = iteration_range.map(|range| self.range(range));
        let leaves = py
            .detach(|| match range {
                None => model.predict_leaf(&data.inner),
                Some(range) => model.predict_leaf_range(&data.inner, range),
            })
            .or_raise()?;
        let (rows, trees) = (leaves.n_rows(), leaves.width());
        // Leaf ids index a tree's nodes, far below `i32::MAX`.
        let leaves = leaves
            .into_vec()
            .into_iter()
            .map(|leaf| i32::try_from(leaf).unwrap_or(i32::MAX))
            .collect();
        to_numpy(py, leaves, &[rows, trees])
    }

    /// The distribution a `dist:*` model predicts for every row.
    #[pyo3(signature = (data, iteration_range=None))]
    fn predict_distribution(
        &self,
        py: Python<'_>,
        data: &DMatrix,
        iteration_range: Option<(usize, usize)>,
    ) -> PyResult<Distributions> {
        let range = iteration_range.map(|range| self.range(range));
        let dists = py
            .detach(|| match range {
                None => self.model.predict_distribution(&data.inner),
                Some(range) => self.model.predict_distribution_range(&data.inner, range),
            })
            .or_raise()?;
        Distributions::new(dists)
    }

    /// The predictions (`output_margin`: raw margins) of a virtual ensemble
    /// of `count` members, `(count, rows)` or `(count, rows, width)`, and
    /// each member's iteration count.
    fn predict_virtual_ensembles<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        count: usize,
        output_margin: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Vec<usize>)> {
        let ensembles = py
            .detach(|| self.model.predict_virtual_ensembles(&data.inner, count))
            .or_raise()?;
        let (members, rows) = (ensembles.n_members(), ensembles.n_rows());
        let member = |m| {
            if output_margin {
                ensembles.member_margins(m)
            } else {
                ensembles.member_predictions(m)
            }
        };
        let width = member(0).map_or(1, hessboost::model::Predictions::width);
        let mut values = Vec::with_capacity(members * rows * width);
        for m in 0..members {
            values.extend_from_slice(member(m).map_or(&[], |p| p.as_slice()));
        }
        let shape = if width == 1 {
            vec![members, rows]
        } else {
            vec![members, rows, width]
        };
        Ok((
            to_numpy(py, values, &shape)?,
            ensembles.iterations().to_vec(),
        ))
    }

    /// A virtual ensemble's `(mean, knowledge, data, total)` uncertainty:
    /// `mean` `(rows,)` or `(rows, width)`; the others `(rows,)` or
    /// `(rows, columns)`, `data` and `total` `None` for plain regression.
    #[allow(
        clippy::type_complexity,
        reason = "the tuple is the extension's return shape; the public class wraps it"
    )]
    fn predict_uncertainty<'py>(
        &self,
        py: Python<'py>,
        data: &DMatrix,
        count: usize,
    ) -> PyResult<(
        Bound<'py, PyArrayDyn<f64>>,
        Bound<'py, PyArrayDyn<f64>>,
        Option<Bound<'py, PyArrayDyn<f64>>>,
        Option<Bound<'py, PyArrayDyn<f64>>>,
    )> {
        let uncertainty = py
            .detach(|| self.model.predict_uncertainty(&data.inner, count))
            .or_raise()?;
        // Every matrix has at least one row.
        let rows = data.inner.n_rows();
        let shape = |values: &[f64]| match values.len() / rows {
            1 => vec![rows],
            width => vec![rows, width],
        };
        let array = |values: Vec<f64>| {
            let shape = shape(&values);
            to_numpy(py, values, &shape)
        };
        Ok((
            array(uncertainty.mean)?,
            array(uncertainty.knowledge)?,
            uncertainty.data.map(array).transpose()?,
            uncertainty.total.map(array).transpose()?,
        ))
    }

    /// `{feature index: score}` for every feature used in a split.
    fn feature_importance<'py>(
        &self,
        py: Python<'py>,
        importance_type: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let kind = match importance_type {
            "weight" => ImportanceType::Weight,
            "gain" => ImportanceType::Gain,
            "total_gain" => ImportanceType::TotalGain,
            "cover" => ImportanceType::Cover,
            "total_cover" => ImportanceType::TotalCover,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown importance_type {other:?}; expected \"weight\", \"gain\", \
                     \"total_gain\", \"cover\" or \"total_cover\""
                )));
            }
        };
        let mut scores = self
            .model
            .feature_importance(kind)
            .into_iter()
            .collect::<Vec<_>>();
        scores.sort_unstable_by_key(|(feature, _)| *feature);
        let dict = PyDict::new(py);
        for (feature, score) in scores {
            dict.set_item(feature, score)?;
        }
        Ok(dict)
    }

    /// Every `step`-th iteration of `begin..end`.
    fn slice(&self, py: Python<'_>, begin: usize, end: usize, step: usize) -> PyResult<Self> {
        let model = py
            .detach(|| self.model.slice(begin..end, step))
            .or_raise()?;
        Ok(Self::new(model))
    }

    #[getter]
    fn objective(&self) -> &str {
        self.model.objective().name()
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.model.n_features()
    }

    #[getter]
    fn num_outputs(&self) -> usize {
        self.model.n_outputs()
    }

    #[getter]
    fn num_targets(&self) -> usize {
        self.model.n_targets()
    }

    #[getter]
    fn num_boosted_rounds(&self) -> usize {
        self.model.num_boost_rounds()
    }

    #[getter]
    fn num_trees(&self) -> usize {
        self.model.num_trees()
    }

    #[getter]
    fn num_parallel_tree(&self) -> usize {
        self.model.num_parallel_tree()
    }

    #[getter]
    fn best_iteration(&self) -> Option<usize> {
        self.model.best_iteration()
    }

    /// Per-output intercepts in margin space.
    #[getter]
    fn base_margins(&self) -> Vec<f32> {
        self.model.base_scores().to_vec()
    }

    #[getter]
    fn vector_leaves(&self) -> bool {
        self.model.has_vector_leaves()
    }

    /// The Boulevard record of a `booster = boulevard` model, else `None`.
    #[getter]
    fn boulevard<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        crate::inference::boulevard_info(py, &self.model)
    }
}
