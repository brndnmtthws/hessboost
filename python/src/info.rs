//! `Booster.model_info`: a model's layout, linear and shrinkage records, and
//! every tree's node arrays, read from the crate's public model API.

use crate::data::to_numpy;
use crate::errors::refuse;
use hessboost::model::BoostedModel;
use hessboost::tree::RegTree;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// An index the arrays store as `int32`, refused past its range rather
/// than wrapped.
fn int32(value: impl TryInto<i32>, what: &str) -> Result<i32, String> {
    value
        .try_into()
        .map_err(|_| format!("{what} exceeds the int32 range of model_info's arrays"))
}

/// A linear-leaf tree's leaf models, node-indexed, terms as CSR.
struct LinearArrays {
    intercept: Vec<f64>,
    offsets: Vec<i64>,
    features: Vec<i32>,
    coefficients: Vec<f64>,
}

/// One tree's node-indexed arrays.
struct TreeArrays {
    left: Vec<i32>,
    right: Vec<i32>,
    feature: Vec<i32>,
    threshold: Vec<f32>,
    default_left: Vec<bool>,
    categorical: Vec<bool>,
    categories: Vec<Vec<i32>>,
    /// `(nodes,)` values, or `(nodes, width)` for a vector-leaf tree.
    value: Vec<f32>,
    width: Option<usize>,
    cover: Vec<f32>,
    gain: Vec<f32>,
    linear: Option<LinearArrays>,
}

impl TreeArrays {
    fn of(tree: &RegTree) -> Result<Self, String> {
        let nodes = tree.nodes();
        let width = tree.is_vector_leaf().then(|| tree.size_leaf_vector());
        let mut arrays = TreeArrays {
            left: Vec::with_capacity(nodes.len()),
            right: Vec::with_capacity(nodes.len()),
            feature: Vec::with_capacity(nodes.len()),
            threshold: Vec::with_capacity(nodes.len()),
            default_left: Vec::with_capacity(nodes.len()),
            categorical: Vec::with_capacity(nodes.len()),
            categories: Vec::with_capacity(nodes.len()),
            value: Vec::with_capacity(nodes.len() * width.unwrap_or(1)),
            width,
            cover: Vec::with_capacity(nodes.len()),
            gain: Vec::with_capacity(nodes.len()),
            linear: None,
        };
        for (nid, node) in nodes.iter().enumerate() {
            let leaf = node.is_leaf();
            arrays.left.push(node.left);
            arrays.right.push(node.right);
            arrays.feature.push(if leaf {
                -1
            } else {
                int32(node.split_feature, "a split feature")?
            });
            arrays.threshold.push(if leaf || node.is_categorical {
                f32::NAN
            } else {
                node.split_cond
            });
            arrays.default_left.push(node.default_left);
            arrays.categorical.push(node.is_categorical);
            arrays.categories.push(
                tree.split_categories(nid)
                    .iter()
                    .map(|&category| int32(category, "a split category"))
                    .collect::<Result<_, _>>()?,
            );
            match width {
                Some(_) if leaf => arrays.value.extend_from_slice(tree.leaf_vector(nid)),
                Some(k) => arrays.value.extend(std::iter::repeat_n(f32::NAN, k)),
                None => arrays
                    .value
                    .push(if leaf { node.leaf_value } else { f32::NAN }),
            }
            arrays.cover.push(node.sum_hess);
            arrays.gain.push(node.split_gain);
        }
        if let Some(linear) = tree.linear_leaves() {
            let mut out = LinearArrays {
                intercept: Vec::with_capacity(nodes.len()),
                offsets: Vec::with_capacity(nodes.len() + 1),
                features: Vec::new(),
                coefficients: Vec::new(),
            };
            out.offsets.push(0);
            for nid in 0..nodes.len() {
                out.intercept.push(linear.intercept(nid));
                let (features, coefficients) = linear.terms(nid);
                for &feature in features {
                    out.features.push(int32(feature, "a linear-leaf feature")?);
                }
                out.coefficients.extend_from_slice(coefficients);
                out.offsets.push(out.features.len() as i64);
            }
            arrays.linear = Some(out);
        }
        Ok(arrays)
    }

    fn into_dict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let nodes = self.left.len();
        let dict = PyDict::new(py);
        dict.set_item("left", to_numpy(py, self.left, &[nodes])?)?;
        dict.set_item("right", to_numpy(py, self.right, &[nodes])?)?;
        dict.set_item("feature", to_numpy(py, self.feature, &[nodes])?)?;
        dict.set_item("threshold", to_numpy(py, self.threshold, &[nodes])?)?;
        dict.set_item("default_left", to_numpy(py, self.default_left, &[nodes])?)?;
        dict.set_item("categorical", to_numpy(py, self.categorical, &[nodes])?)?;
        let categories = self
            .categories
            .into_iter()
            .map(|set| {
                let n = set.len();
                to_numpy(py, set, &[n])
            })
            .collect::<PyResult<Vec<_>>>()?;
        dict.set_item("categories", categories)?;
        let shape = match self.width {
            Some(k) => vec![nodes, k],
            None => vec![nodes],
        };
        dict.set_item("value", to_numpy(py, self.value, &shape)?)?;
        dict.set_item("cover", to_numpy(py, self.cover, &[nodes])?)?;
        dict.set_item("gain", to_numpy(py, self.gain, &[nodes])?)?;
        let linear = match self.linear {
            Some(linear) => {
                let terms = linear.features.len();
                let d = PyDict::new(py);
                d.set_item("intercept", to_numpy(py, linear.intercept, &[nodes])?)?;
                d.set_item("offsets", to_numpy(py, linear.offsets, &[nodes + 1])?)?;
                d.set_item("features", to_numpy(py, linear.features, &[terms])?)?;
                d.set_item("coefficients", to_numpy(py, linear.coefficients, &[terms])?)?;
                Some(d)
            }
            None => None,
        };
        dict.set_item("linear", linear)?;
        Ok(dict)
    }
}

/// The per-model arrays, built without the GIL.
struct ModelArrays {
    tree_weights: Vec<f32>,
    tree_outputs: Vec<i32>,
    trees: Vec<TreeArrays>,
}

impl ModelArrays {
    fn of(model: &BoostedModel) -> Result<Self, String> {
        let vector = model.has_vector_leaves();
        let (parallel, outputs) = (model.num_parallel_tree(), model.n_outputs());
        let tree_outputs = (0..model.trees().len())
            .map(|t| {
                if vector {
                    Ok(0)
                } else {
                    // A scalar tree feeds output `(t / num_parallel_tree) % n_outputs`.
                    int32((t / parallel) % outputs, "an output index")
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(ModelArrays {
            tree_weights: model.tree_weights().collect(),
            tree_outputs,
            trees: model
                .trees()
                .iter()
                .map(TreeArrays::of)
                .collect::<Result<_, _>>()?,
        })
    }
}

/// The record `hessboost.ModelInfo` wraps: the layout, `gblinear` and
/// `shrinkage` sub-records (or `None`), and one dict of node arrays per
/// tree. The arrays are built without the GIL (no rayon).
pub(crate) fn model_info<'py>(
    py: Python<'py>,
    model: &BoostedModel,
) -> PyResult<Bound<'py, PyDict>> {
    let arrays = py.detach(|| ModelArrays::of(model)).map_err(refuse)?;
    let dict = PyDict::new(py);
    dict.set_item("objective", model.objective().name())?;
    dict.set_item("num_features", model.n_features())?;
    dict.set_item("num_outputs", model.n_outputs())?;
    dict.set_item("num_targets", model.n_targets())?;
    dict.set_item("num_class", model.num_class())?;
    dict.set_item("num_parallel_tree", model.num_parallel_tree())?;
    dict.set_item("trees_per_iteration", model.trees_per_iteration())?;
    dict.set_item("num_boosted_rounds", model.num_boost_rounds())?;
    dict.set_item("best_iteration", model.best_iteration())?;
    let base = model.base_scores();
    dict.set_item("base_margins", to_numpy(py, base.to_vec(), &[base.len()])?)?;
    dict.set_item("vector_leaves", model.has_vector_leaves())?;
    dict.set_item(
        "linear_leaves",
        arrays.trees.iter().any(|tree| tree.linear.is_some()),
    )?;
    let trees = arrays.tree_weights.len();
    dict.set_item("tree_weights", to_numpy(py, arrays.tree_weights, &[trees])?)?;
    dict.set_item("tree_outputs", to_numpy(py, arrays.tree_outputs, &[trees])?)?;
    let gblinear = match model.linear() {
        Some(linear) => {
            let d = PyDict::new(py);
            let outputs = linear.bias().len();
            d.set_item(
                "weights",
                to_numpy(
                    py,
                    linear.weights().to_vec(),
                    &[linear.weights().len() / outputs, outputs],
                )?,
            )?;
            d.set_item("bias", to_numpy(py, linear.bias().to_vec(), &[outputs])?)?;
            Some(d)
        }
        None => None,
    };
    dict.set_item("gblinear", gblinear)?;
    let shrinkage = match model.shrinkage() {
        Some(shrinkage) => {
            let d = PyDict::new(py);
            let (factors, base) = (shrinkage.factors(), shrinkage.base_scores());
            d.set_item("factors", to_numpy(py, factors.to_vec(), &[factors.len()])?)?;
            d.set_item("base_margins", to_numpy(py, base.to_vec(), &[base.len()])?)?;
            Some(d)
        }
        None => None,
    };
    dict.set_item("shrinkage", shrinkage)?;
    let list = PyList::empty(py);
    for tree in arrays.trees {
        list.append(tree.into_dict(py)?)?;
    }
    dict.set_item("trees", list)?;
    Ok(dict)
}
