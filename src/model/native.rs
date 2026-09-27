//! The native binary model format: a [section table](super::sections) behind
//! a magic and a container version, compressed as one zstd frame.
//!
//! ```text
//! b"HBM\0"   magic
//! u8         container version (CONTAINER_VERSION)
//! sections   the model (see below), nothing after the last payload
//! u64        XXH64 (seed 0) of every byte before it
//! ```
//!
//! The model's scalar fields are `model.*` sections, its objective
//! parameters `objective.*`, and the trees are stored column-wise: one array
//! per node field across all trees (`node.*`), split per tree by
//! `tree.node_count`, plus per-tree category pools, leaf vectors and leaf
//! linear models. `model.writer` (optional, not `REQUIRED`, never read back)
//! names the release that wrote the file, e.g. `hessboost 0.2.0`. A model
//! trained with model shrinkage adds `shrinkage.factors` (`f64`, one
//! coefficient per iteration) and `shrinkage.base_score` (`f32`, the
//! intercepts before shrinkage), both `REQUIRED`: a reader unaware of them
//! would read iteration ranges as tree prefixes. Their absence means no
//! shrinkage. A Boulevard fit adds the `boulevard.*` sections of its
//! [`BoulevardInfo`] (not `REQUIRED`: predictions do not read them), an
//! EBM the `ebm.*` sections of its [`EbmInfo`] (likewise optional).
//!
//! The reader is strict about everything it knows: `node.flags` bits it
//! does not define, `tree.has_linear` bytes other than 0 and 1, and bytes
//! between the last payload and the checksum are refused, so a later
//! writer can give any of them a meaning without this version misreading
//! the file.
//!
//! # Compatibility
//!
//! - A new model field is a new section; a reader of a file written before
//!   it existed gives it the value that reproduces the old behavior.
//!   Neither needs a new container version, and older readers skip it.
//! - A section older readers must not skip (one that changes predictions)
//!   keeps the `REQUIRED` flag, so they refuse the file instead of
//!   mispredicting.
//! - Only a change to the container layout itself bumps
//!   [`CONTAINER_VERSION`], with an upgrade path for the previous one.
//!
//! Files are written as a single zstd frame, so `zstd -d` recovers the
//! container; uncompressed containers are read as well. A container that
//! compresses beyond what the reader accepts from a frame (see
//! [`super::container`]) is written uncompressed instead, so every written file
//! loads. The trailing checksum catches corruption either way.

use std::sync::OnceLock;

use super::container::{ContainerSpec, WRITER};
use super::objective::{ModelObjective, StoredObjectiveParams};
use super::sections::{Sections, Writer, format_error, unknown_value, wrong_length};
use super::{BoostedModel, LinearModel, Shrinkage, TreeWeights};
use crate::ebm::{EbmBoulevard, EbmInfo};
use crate::error::Result;
use crate::inference::BoulevardInfo;
use crate::objective::AftDistribution;
use crate::objective::distributional::DistFamily;
use crate::objective::distributional::{DistGradient, DistSplitDirection};
use crate::tree::linear::LinearLeaves;
use crate::tree::{Node, RegTree};

const MAGIC: &[u8; 4] = b"HBM\0";
/// The magic of the pre-0.2.0 native format (hessboost 0.1.x), which this
/// version refuses, naming the upgrade path.
const LEGACY_MAGIC: &[u8; 4] = b"SQB\0";
/// Layout version of the container (not of the model it holds).
const CONTAINER_VERSION: u8 = 3;
/// The native model container.
const NATIVE: ContainerSpec = ContainerSpec {
    magic: *MAGIC,
    version: CONTAINER_VERSION,
    what: "native model",
    known: |name| KNOWN.contains(&name) || OBJECTIVE_SECTIONS.contains(&name),
    legacy: Some((
        *LEGACY_MAGIC,
        "a native model from hessboost 0.1.x, which 0.2.0 and later cannot read \
         (export it with 0.1.1's `save_xgboost_json` and load that)",
    )),
};

/// Whether `bytes` start like a native model file: a zstd frame, or an
/// uncompressed container of this format or of 0.1.x's.
pub(crate) fn is_native_container(bytes: &[u8]) -> bool {
    super::container::is_zstd_frame(bytes)
        || bytes.starts_with(MAGIC)
        || bytes.starts_with(LEGACY_MAGIC)
}

/// `default_left` in `node.flags`.
const DEFAULT_LEFT: u8 = 1;
/// `is_categorical` in `node.flags`.
const CATEGORICAL: u8 = 2;
/// Every bit `node.flags` defines; the reader refuses the others.
const NODE_FLAGS: u8 = DEFAULT_LEFT | CATEGORICAL;

/// Every section this version reads.
const KNOWN: &[&str] = &[
    "model.objective",
    "model.writer",
    "model.base_score",
    "model.num_class",
    "model.n_outputs",
    "model.n_targets",
    "model.n_features",
    "model.best_iteration",
    "model.tree_weights",
    "model.num_parallel_tree",
    SHRINKAGE_SECTIONS[0],
    SHRINKAGE_SECTIONS[1],
    "gblinear.weights",
    "gblinear.bias",
    "tree.node_count",
    "node.split_feature",
    "node.split_cond",
    "node.left",
    "node.right",
    "node.leaf_value",
    "node.sum_hess",
    "node.split_gain",
    "node.cat_begin",
    "node.cat_end",
    "node.flags",
    "tree.category_count",
    "tree.categories",
    "tree.size_leaf_vector",
    "tree.leaf_vectors",
    "tree.has_linear",
    "leaf_linear.offsets",
    "leaf_linear.intercepts",
    "leaf_linear.features",
    "leaf_linear.coeffs",
    "boulevard.dropout",
    "boulevard.learning_rate",
    "boulevard.subsample",
    "boulevard.reg_lambda",
    "boulevard.truncation",
    "boulevard.seed",
    "boulevard.intercept_from_labels",
    "ebm.term_sizes",
    "ebm.term_features",
    "ebm.tree_terms",
    "ebm.term_means",
    "ebm.boulevard",
];

/// The `boulevard.*` sections of a Boulevard model ([`BoulevardInfo`]),
/// all written together. Not `REQUIRED`: predictions do not read them, so a
/// reader that predates them loads the model as a plain `gbtree` ensemble.
fn write_boulevard(w: &mut Writer, info: &BoulevardInfo) {
    let f64s = [
        ("boulevard.dropout", info.dropout),
        ("boulevard.learning_rate", info.learning_rate),
        ("boulevard.subsample", info.subsample),
        ("boulevard.reg_lambda", info.reg_lambda),
        // `0` is no truncation, as in files written before it was optional.
        ("boulevard.truncation", info.truncation.unwrap_or(0.0)),
    ];
    for (name, value) in f64s {
        w.raw(name, 0, &value.to_le_bytes());
    }
    w.raw("boulevard.seed", 0, &info.seed.to_le_bytes());
    w.raw(
        "boulevard.intercept_from_labels",
        0,
        &u64::from(info.intercept_from_labels).to_le_bytes(),
    );
}

/// The [`BoulevardInfo`] of the `boulevard.*` sections: `None` when the file
/// has none of them (every model that is not a Boulevard fit, and files
/// written before they existed), an error when only some are present.
fn read_boulevard(s: &Sections) -> Result<Option<BoulevardInfo>> {
    if !s.has("boulevard.dropout") {
        if let Some(name) = KNOWN
            .iter()
            .find(|name| name.starts_with("boulevard.") && s.has(name))
        {
            return Err(format_error(format!(
                "`{name}` without `boulevard.dropout`"
            )));
        }
        return Ok(None);
    }
    let intercept_from_labels = match s.u64("boulevard.intercept_from_labels")? {
        0 => false,
        1 => true,
        other => {
            return Err(format_error(format!(
                "`boulevard.intercept_from_labels` must be 0 or 1, got {other}"
            )));
        }
    };
    Ok(Some(BoulevardInfo {
        dropout: s.f64("boulevard.dropout")?,
        learning_rate: s.f64("boulevard.learning_rate")?,
        subsample: s.f64("boulevard.subsample")?,
        reg_lambda: s.f64("boulevard.reg_lambda")?,
        truncation: Some(s.f64("boulevard.truncation")?).filter(|&t| t != 0.0),
        seed: s.u64("boulevard.seed")?,
        intercept_from_labels,
    }))
}

/// The `ebm.*` sections of an EBM ([`EbmInfo`]): the terms' feature counts
/// and features (concatenated), the term of each tree, the terms' training
/// means, and for a Boulevard EBM `ebm.boulevard` (learning rate,
/// subsample, `lambda`). Not `REQUIRED`: predictions do not read them, so a
/// reader that predates them loads the model as a plain `gbtree` ensemble.
fn write_ebm(w: &mut Writer, info: &EbmInfo) {
    fn optional<const N: usize, T: Copy>(
        w: &mut Writer,
        name: &'static str,
        values: impl IntoIterator<Item = T>,
        to_le: fn(T) -> [u8; N],
    ) {
        let payload: Vec<u8> = values.into_iter().flat_map(to_le).collect();
        w.raw(name, 0, &payload);
    }
    optional(
        w,
        "ebm.term_sizes",
        info.terms.iter().map(|t| t.len() as u32),
        u32::to_le_bytes,
    );
    optional(
        w,
        "ebm.term_features",
        info.terms.iter().flatten().copied(),
        u32::to_le_bytes,
    );
    optional(
        w,
        "ebm.tree_terms",
        info.tree_terms.iter().copied(),
        u32::to_le_bytes,
    );
    optional(
        w,
        "ebm.term_means",
        info.term_means.iter().copied(),
        f64::to_le_bytes,
    );
    if let Some(b) = &info.boulevard {
        optional(
            w,
            "ebm.boulevard",
            [b.learning_rate, b.subsample, b.reg_lambda],
            f64::to_le_bytes,
        );
    }
}

/// The [`EbmInfo`] of the `ebm.*` sections: `None` when the file has none
/// of them (every model that is not an EBM, and files written before they
/// existed), an error when only some are present or they disagree in
/// length. [`EbmInfo::validate`] checks the rest with the model.
fn read_ebm(s: &Sections) -> Result<Option<EbmInfo>> {
    if !s.has("ebm.term_sizes") {
        if let Some(name) = KNOWN
            .iter()
            .find(|name| name.starts_with("ebm.") && s.has(name))
        {
            return Err(format_error(format!("`{name}` without `ebm.term_sizes`")));
        }
        return Ok(None);
    }
    let sizes = s.array("ebm.term_sizes", u32::from_le_bytes)?;
    let features = s.array("ebm.term_features", u32::from_le_bytes)?;
    if sizes.iter().map(|&k| k as usize).sum::<usize>() != features.len() {
        return Err(wrong_length("ebm.term_features"));
    }
    let mut rest = features.as_slice();
    let terms = sizes
        .iter()
        .map(|&k| {
            let (term, tail) = rest.split_at(k as usize);
            rest = tail;
            term.to_vec()
        })
        .collect();
    let boulevard = if s.has("ebm.boulevard") {
        let v = s.array_exact("ebm.boulevard", 3, f64::from_le_bytes)?;
        Some(EbmBoulevard {
            learning_rate: v[0],
            subsample: v[1],
            reg_lambda: v[2],
        })
    } else {
        None
    };
    Ok(Some(EbmInfo {
        terms,
        tree_terms: s.array("ebm.tree_terms", u32::from_le_bytes)?,
        term_means: s.array("ebm.term_means", f64::from_le_bytes)?,
        boulevard,
    }))
}

/// The `objective.*` sections [`write_objective_params`] writes, shared by
/// the native and compact formats.
pub(super) const OBJECTIVE_SECTIONS: &[&str] = &[
    "objective.scale_pos_weight",
    "objective.max_delta_step",
    "objective.tweedie_variance_power",
    "objective.huber_slope",
    "objective.lambdarank_num_pair_per_sample",
    "objective.quantile_alpha",
    "objective.expectile_alpha",
    "objective.aft_loss_distribution",
    "objective.aft_loss_distribution_scale",
    "objective.dist_gradient",
    "objective.dist_split_direction",
    "objective.distribution",
];

/// The `shrinkage.*` sections [`write_shrinkage`] writes, shared by the
/// native and compact formats.
pub(super) const SHRINKAGE_SECTIONS: [&str; 2] = ["shrinkage.factors", "shrinkage.base_score"];

/// Write the model shrinkage record `shrinkage` (both sections `REQUIRED`).
pub(super) fn write_shrinkage(w: &mut Writer, shrinkage: &Shrinkage) {
    w.array(
        SHRINKAGE_SECTIONS[0],
        shrinkage.factors().iter().copied(),
        f64::to_le_bytes,
    );
    w.array(
        SHRINKAGE_SECTIONS[1],
        shrinkage.base_score().iter().copied(),
        f32::to_le_bytes,
    );
}

/// The model shrinkage record [`write_shrinkage`] wrote, `None` when `s`
/// has neither section (no shrinkage). The caller validates it against its
/// model.
pub(super) fn read_shrinkage(s: &Sections) -> Result<Option<Shrinkage>> {
    if !SHRINKAGE_SECTIONS.iter().any(|name| s.has(name)) {
        return Ok(None);
    }
    Ok(Some(Shrinkage::new(
        s.array(SHRINKAGE_SECTIONS[0], f64::from_le_bytes)?,
        s.array(SHRINKAGE_SECTIONS[1], f32::from_le_bytes)?,
    )))
}

/// Encode `model` as a container: zstd-compressed unless the frame would
/// expand further than [`read`] accepts (see [`ContainerSpec::seal`]).
pub(super) fn write(model: &BoostedModel) -> Result<Vec<u8>> {
    NATIVE.seal(sections(model)?)
}

/// Encode `model` as an uncompressed container, the form other containers
/// embed (a diffusion model's regressors) and [`read`] accepts as is.
pub(crate) fn write_container(model: &BoostedModel) -> Result<Vec<u8>> {
    Ok(NATIVE.frame(sections(model)?))
}

/// Append [`write_container`]'s bytes for `model` to `out`.
pub(super) fn write_container_into(model: &BoostedModel, out: &mut Vec<u8>) -> Result<()> {
    NATIVE.frame_into(sections(model)?, out);
    Ok(())
}

/// The section table of `model`.
fn sections(model: &BoostedModel) -> Result<Writer> {
    let trees = &model.trees;
    // Per-tree counts and array offsets are stored as `u32`.
    let total_nodes: usize = trees.iter().map(RegTree::num_nodes).sum();
    let total_categories: usize = trees.iter().map(|t| t.categories().len()).sum();
    if [trees.len(), total_nodes, total_categories]
        .iter()
        .any(|&n| u32::try_from(n).is_err())
    {
        return Err(format_error("model is too large for the native format"));
    }
    let mut w = Writer::default();
    write_model_sections(&mut w, model);
    write_tree_sections(&mut w, trees);
    Ok(w)
}

/// The `model.*`, `gblinear.*`, and `objective.*` sections.
fn write_model_sections(w: &mut Writer, m: &BoostedModel) {
    w.str("model.objective", m.objective.name());
    w.raw("model.writer", 0, WRITER.as_bytes());
    w.array(
        "model.base_score",
        m.base_score.iter().copied(),
        f32::to_le_bytes,
    );
    w.u64("model.num_class", m.num_class as u64);
    w.u64("model.n_outputs", m.n_outputs as u64);
    w.u64("model.n_targets", m.n_targets as u64);
    w.u64("model.n_features", m.n_features as u64);
    if let Some(best) = m.best_iteration {
        w.u64("model.best_iteration", best as u64);
    }
    w.array(
        "model.tree_weights",
        m.tree_weights.iter().copied(),
        f32::to_le_bytes,
    );
    w.u64("model.num_parallel_tree", m.num_parallel_tree as u64);
    if let Some(linear) = &m.linear {
        w.array(
            "gblinear.weights",
            linear.weights().iter().copied(),
            f32::to_le_bytes,
        );
        w.array(
            "gblinear.bias",
            linear.bias().iter().copied(),
            f32::to_le_bytes,
        );
    }
    if let Some(shrinkage) = &m.shrinkage {
        write_shrinkage(w, shrinkage);
    }
    write_objective_params(
        w,
        &StoredObjectiveParams::of(&m.objective, m.max_delta_step),
    );
    if let Some(info) = &m.boulevard {
        write_boulevard(w, info);
    }
    if let Some(info) = &m.ebm {
        write_ebm(w, info);
    }
}

/// The trees, column-wise: `tree.*` per-tree arrays, `node.*` per-node
/// arrays across all trees, and the `leaf_linear.*` pools.
fn write_tree_sections(w: &mut Writer, trees: &[RegTree]) {
    let nodes = || trees.iter().flat_map(RegTree::nodes);
    let per_tree = |count: fn(&RegTree) -> usize| trees.iter().map(move |t| count(t) as u32);
    w.array(
        "tree.node_count",
        per_tree(RegTree::num_nodes),
        u32::to_le_bytes,
    );
    w.array(
        "node.split_feature",
        nodes().map(|n| n.split_feature),
        u32::to_le_bytes,
    );
    w.array(
        "node.split_cond",
        nodes().map(|n| n.split_cond),
        f32::to_le_bytes,
    );
    w.array("node.left", nodes().map(|n| n.left), i32::to_le_bytes);
    w.array("node.right", nodes().map(|n| n.right), i32::to_le_bytes);
    w.array(
        "node.leaf_value",
        nodes().map(|n| n.leaf_value),
        f32::to_le_bytes,
    );
    w.array(
        "node.sum_hess",
        nodes().map(|n| n.sum_hess),
        f32::to_le_bytes,
    );
    w.array(
        "node.split_gain",
        nodes().map(|n| n.split_gain),
        f32::to_le_bytes,
    );
    w.array(
        "node.cat_begin",
        nodes().map(|n| n.cat_begin),
        u32::to_le_bytes,
    );
    w.array("node.cat_end", nodes().map(|n| n.cat_end), u32::to_le_bytes);
    w.array(
        "node.flags",
        nodes().map(|n| {
            (u8::from(n.default_left) * DEFAULT_LEFT) | (u8::from(n.is_categorical) * CATEGORICAL)
        }),
        |b: u8| [b],
    );
    w.array(
        "tree.category_count",
        per_tree(|t| t.categories().len()),
        u32::to_le_bytes,
    );
    w.array(
        "tree.categories",
        trees.iter().flat_map(|t| t.categories().iter().copied()),
        u32::to_le_bytes,
    );
    w.array(
        "tree.size_leaf_vector",
        per_tree(|t| t.leaf_vector_parts().0),
        u32::to_le_bytes,
    );
    w.array(
        "tree.leaf_vectors",
        trees
            .iter()
            .flat_map(|t| t.leaf_vector_parts().1.iter().copied()),
        f32::to_le_bytes,
    );
    let linear = || {
        trees
            .iter()
            .filter_map(|t| t.linear_leaves().map(LinearLeaves::parts))
    };
    w.array(
        "tree.has_linear",
        trees.iter().map(|t| u8::from(t.linear_leaves().is_some())),
        |b: u8| [b],
    );
    w.array(
        "leaf_linear.offsets",
        linear().flat_map(|p| p.0.iter().copied()),
        u32::to_le_bytes,
    );
    w.array(
        "leaf_linear.intercepts",
        linear().flat_map(|p| p.1.iter().copied()),
        f64::to_le_bytes,
    );
    w.array(
        "leaf_linear.features",
        linear().flat_map(|p| p.2.iter().copied()),
        u32::to_le_bytes,
    );
    w.array(
        "leaf_linear.coeffs",
        linear().flat_map(|p| p.3.iter().copied()),
        f64::to_le_bytes,
    );
}

/// Decode a container (zstd-compressed or not). The caller validates the
/// model it forms ([`BoostedModel::validate_structure`]).
pub(super) fn read(bytes: &[u8]) -> Result<BoostedModel> {
    NATIVE.read(bytes, read_model)
}

fn read_model(s: &Sections) -> Result<BoostedModel> {
    let name = s.str("model.objective")?;
    let stored = read_objective_params(s, StoredObjectiveParams::defaults_for(name))?;
    let num_class = s.usize("model.num_class")?;
    let objective = ModelObjective::from_stored(name, &stored, num_class)?;
    let linear = if s.has("gblinear.weights") || s.has("gblinear.bias") {
        Some(LinearModel::new(
            s.array("gblinear.weights", f32::from_le_bytes)?,
            s.array("gblinear.bias", f32::from_le_bytes)?,
        ))
    } else {
        None
    };
    Ok(BoostedModel {
        trees: read_trees(s)?,
        base_score: s.array("model.base_score", f32::from_le_bytes)?,
        objective,
        max_delta_step: stored.max_delta_step,
        num_class,
        n_outputs: s.usize("model.n_outputs")?,
        n_targets: s.usize("model.n_targets")?,
        n_features: s.usize("model.n_features")?,
        best_iteration: if s.has("model.best_iteration") {
            Some(s.usize("model.best_iteration")?)
        } else {
            None
        },
        tree_weights: TreeWeights::from_vec(s.array("model.tree_weights", f32::from_le_bytes)?),
        num_parallel_tree: s.usize("model.num_parallel_tree")?,
        linear,
        boulevard: read_boulevard(s)?,
        ebm: read_ebm(s)?,
        shrinkage: read_shrinkage(s)?,
        compact: OnceLock::new(),
    })
}

fn read_trees(s: &Sections) -> Result<Vec<RegTree>> {
    let node_count = s.array("tree.node_count", u32::from_le_bytes)?;
    let n_trees = node_count.len();
    let n_nodes = checked_sum(&node_count)?;
    // Every column's length is checked (in this order) before any value:
    // `n_nodes` sums untrusted per-tree counts, so its products may overflow.
    let u32s = |name: &str, count: usize| s.array_exact(name, count, u32::from_le_bytes);
    let f32s = |name: &str| s.array_exact(name, n_nodes, f32::from_le_bytes);
    let split_feature = u32s("node.split_feature", n_nodes)?;
    let split_cond = f32s("node.split_cond")?;
    let left = s.array_exact("node.left", n_nodes, i32::from_le_bytes)?;
    let right = s.array_exact("node.right", n_nodes, i32::from_le_bytes)?;
    let leaf_value = f32s("node.leaf_value")?;
    let sum_hess = f32s("node.sum_hess")?;
    let split_gain = f32s("node.split_gain")?;
    let cat_begin = u32s("node.cat_begin", n_nodes)?;
    let cat_end = u32s("node.cat_end", n_nodes)?;
    let flags = s.bytes_exact("node.flags", Some(n_nodes))?;
    let category_count = u32s("tree.category_count", n_trees)?;
    let size_leaf_vector = u32s("tree.size_leaf_vector", n_trees)?;
    let has_linear = s.bytes_exact("tree.has_linear", Some(n_trees))?;
    if let Some(&bad) = flags.iter().find(|&&f| f & !NODE_FLAGS != 0) {
        return Err(format_error(format!(
            "`node.flags` holds undefined bits {:#04x}",
            bad & !NODE_FLAGS
        )));
    }
    let mut categories = Cursor::new(s.array("tree.categories", u32::from_le_bytes)?);
    let mut leaf_vectors = Cursor::new(s.array("tree.leaf_vectors", f32::from_le_bytes)?);
    if let Some(&bad) = has_linear.iter().find(|&&b| b > 1) {
        return Err(format_error(format!(
            "`tree.has_linear` holds {bad}, not 0 or 1"
        )));
    }
    let mut offsets = Cursor::new(s.array("leaf_linear.offsets", u32::from_le_bytes)?);
    let mut intercepts = Cursor::new(s.array("leaf_linear.intercepts", f64::from_le_bytes)?);
    let mut features = Cursor::new(s.array("leaf_linear.features", u32::from_le_bytes)?);
    let mut coeffs = Cursor::new(s.array("leaf_linear.coeffs", f64::from_le_bytes)?);

    let mut trees = Vec::with_capacity(n_trees);
    let mut first = 0usize;
    for t in 0..n_trees {
        let n = node_count[t] as usize;
        let nodes = (first..first + n)
            .map(|i| Node {
                split_feature: split_feature[i],
                split_cond: split_cond[i],
                default_left: flags[i] & DEFAULT_LEFT != 0,
                left: left[i],
                right: right[i],
                leaf_value: leaf_value[i],
                sum_hess: sum_hess[i],
                split_gain: split_gain[i],
                is_categorical: flags[i] & CATEGORICAL != 0,
                cat_begin: cat_begin[i],
                cat_end: cat_end[i],
            })
            .collect();
        first += n;
        let width = size_leaf_vector[t] as usize;
        let n_weights = n
            .checked_mul(width)
            .ok_or_else(|| format_error("leaf vectors overflow"))?;
        let linear = if has_linear[t] == 1 {
            let offsets = offsets.take(n + 1, "leaf_linear.offsets")?;
            let n_terms = offsets.last().map_or(0, |&end| end as usize);
            Some(LinearLeaves::from_parts(
                offsets,
                intercepts.take(n, "leaf_linear.intercepts")?,
                features.take(n_terms, "leaf_linear.features")?,
                coeffs.take(n_terms, "leaf_linear.coeffs")?,
            ))
        } else {
            None
        };
        trees.push(RegTree::from_parts(
            nodes,
            categories.take(category_count[t] as usize, "tree.categories")?,
            width,
            leaf_vectors.take(n_weights, "tree.leaf_vectors")?,
            linear,
        ));
    }
    for (rest, name) in [
        (categories.rest(), "tree.categories"),
        (leaf_vectors.rest(), "tree.leaf_vectors"),
        (offsets.rest(), "leaf_linear.offsets"),
        (intercepts.rest(), "leaf_linear.intercepts"),
        (features.rest(), "leaf_linear.features"),
        (coeffs.rest(), "leaf_linear.coeffs"),
    ] {
        if rest != 0 {
            return Err(wrong_length(name));
        }
    }
    Ok(trees)
}

/// Write every objective parameter as its own `objective.*` section.
pub(super) fn write_objective_params(w: &mut Writer, p: &StoredObjectiveParams) {
    w.f64("objective.scale_pos_weight", p.scale_pos_weight);
    w.f64("objective.max_delta_step", p.max_delta_step);
    w.f64("objective.tweedie_variance_power", p.tweedie_variance_power);
    w.f64("objective.huber_slope", p.huber_slope);
    w.u64(
        "objective.lambdarank_num_pair_per_sample",
        p.lambdarank_num_pair_per_sample as u64,
    );
    w.array(
        "objective.quantile_alpha",
        p.quantile_alpha.iter().copied(),
        f64::to_le_bytes,
    );
    w.array(
        "objective.expectile_alpha",
        p.expectile_alpha.iter().copied(),
        f64::to_le_bytes,
    );
    w.str(
        "objective.aft_loss_distribution",
        p.aft_loss_distribution.name(),
    );
    w.f64(
        "objective.aft_loss_distribution_scale",
        p.aft_loss_distribution_scale,
    );
    w.str("objective.dist_gradient", p.dist_gradient.name());
    w.str(
        "objective.dist_split_direction",
        p.dist_split_direction.name(),
    );
    if let Some(family) = p.distribution {
        w.str("objective.distribution", family.objective_name());
    }
}

/// Read the `objective.*` sections; a section absent from the data keeps
/// its value from `base` (the objective's defaults).
pub(super) fn read_objective_params(
    s: &Sections,
    base: StoredObjectiveParams,
) -> Result<StoredObjectiveParams> {
    let f64s = |s: &Sections, name: &str| s.array(name, f64::from_le_bytes);
    Ok(StoredObjectiveParams {
        scale_pos_weight: or(
            s,
            "objective.scale_pos_weight",
            base.scale_pos_weight,
            Sections::f64,
        )?,
        max_delta_step: or(
            s,
            "objective.max_delta_step",
            base.max_delta_step,
            Sections::f64,
        )?,
        tweedie_variance_power: or(
            s,
            "objective.tweedie_variance_power",
            base.tweedie_variance_power,
            Sections::f64,
        )?,
        huber_slope: or(s, "objective.huber_slope", base.huber_slope, Sections::f64)?,
        lambdarank_num_pair_per_sample: or(
            s,
            "objective.lambdarank_num_pair_per_sample",
            base.lambdarank_num_pair_per_sample,
            Sections::usize,
        )?,
        quantile_alpha: or(s, "objective.quantile_alpha", base.quantile_alpha, f64s)?,
        expectile_alpha: or(s, "objective.expectile_alpha", base.expectile_alpha, f64s)?,
        aft_loss_distribution: named(
            s,
            "objective.aft_loss_distribution",
            AftDistribution::from_name,
        )?
        .unwrap_or(base.aft_loss_distribution),
        aft_loss_distribution_scale: or(
            s,
            "objective.aft_loss_distribution_scale",
            base.aft_loss_distribution_scale,
            Sections::f64,
        )?,
        dist_gradient: named(s, "objective.dist_gradient", DistGradient::from_name)?
            .unwrap_or(base.dist_gradient),
        dist_split_direction: named(
            s,
            "objective.dist_split_direction",
            DistSplitDirection::from_name,
        )?
        .unwrap_or(base.dist_split_direction),
        distribution: named(s, "objective.distribution", DistFamily::from_objective)?
            .or(base.distribution),
    })
}

/// Section `name` read with `read`, or `default` when the data lacks it.
fn or<'a, T>(
    s: &Sections<'a>,
    name: &str,
    default: T,
    read: impl FnOnce(&Sections<'a>, &str) -> Result<T>,
) -> Result<T> {
    if s.has(name) {
        read(s, name)
    } else {
        Ok(default)
    }
}

/// The enum variant stored by name in section `name`, if present.
fn named<T>(s: &Sections, name: &str, from: fn(&str) -> Option<T>) -> Result<Option<T>> {
    if !s.has(name) {
        return Ok(None);
    }
    let value = s.str(name)?;
    from(value)
        .map(Some)
        .ok_or_else(|| unknown_value(name, value))
}

/// Consumes a decoded array front to back, one tree's share at a time.
struct Cursor<T> {
    values: Vec<T>,
    at: usize,
}

impl<T: Clone> Cursor<T> {
    fn new(values: Vec<T>) -> Self {
        Cursor { values, at: 0 }
    }

    fn take(&mut self, n: usize, name: &str) -> Result<Vec<T>> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.values.len())
            .ok_or_else(|| wrong_length(name))?;
        let out = self.values[self.at..end].to_vec();
        self.at = end;
        Ok(out)
    }

    fn rest(&self) -> usize {
        self.values.len() - self.at
    }
}

fn checked_sum(counts: &[u32]) -> Result<usize> {
    counts
        .iter()
        .try_fold(0usize, |sum, &c| sum.checked_add(c as usize))
        .ok_or_else(|| format_error("section lengths overflow"))
}

#[cfg(test)]
mod tests {
    use super::super::container::{MAX_EXPANSION, ZSTD_MAGIC, decompress, xxh64};
    use super::super::sections::REQUIRED;
    use super::*;
    use crate::config::TrainingParams;
    use crate::model::Iterations;
    use crate::model::ModelFormat;
    use crate::objective::{Objective, PseudoHuber, RegLoss};
    use crate::test_support::labeled_dense;
    use crate::{model::BoostedModel, training::train};

    /// One section of a container, as `rewrite` edits them.
    struct Entry {
        name: String,
        flags: u8,
        payload: Vec<u8>,
    }

    /// Re-encode the container in `bytes` after `edit` changes its
    /// sections (left uncompressed, which readers accept).
    fn rewrite(bytes: &[u8], edit: impl FnOnce(&mut Vec<Entry>)) -> Vec<u8> {
        let container = decompress(bytes).unwrap();
        let body = &container[MAGIC.len() + 1..container.len() - 8];
        let count = u32::from_le_bytes(body[..4].try_into().unwrap()) as usize;
        let mut at = 4;
        let mut table = Vec::new();
        for _ in 0..count {
            let len = body[at] as usize;
            let name = std::str::from_utf8(&body[at + 1..at + 1 + len]).unwrap();
            let flags = body[at + 1 + len];
            let size = u64::from_le_bytes(body[at + 2 + len..at + 10 + len].try_into().unwrap());
            table.push((name.to_string(), flags, size as usize));
            at += 10 + len;
        }
        let mut entries: Vec<Entry> = table
            .into_iter()
            .map(|(name, flags, size)| {
                let payload = body[at..at + size].to_vec();
                at += size;
                Entry {
                    name,
                    flags,
                    payload,
                }
            })
            .collect();
        edit(&mut entries);
        let mut out = container[..=MAGIC.len()].to_vec();
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for e in &entries {
            out.push(e.name.len() as u8);
            out.extend_from_slice(e.name.as_bytes());
            out.push(e.flags);
            out.extend_from_slice(&(e.payload.len() as u64).to_le_bytes());
        }
        for e in entries {
            out.extend_from_slice(&e.payload);
        }
        let checksum = xxh64(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    fn model() -> (BoostedModel, crate::data::DMatrix) {
        let x: Vec<f32> = (0..40).map(|i| (i % 7) as f32).collect();
        let y: Vec<f32> = x.iter().map(|v| v * 0.5).collect();
        let data = labeled_dense(&x, 40, 1, &y);
        let params = TrainingParams::builder()
            .objective(Objective::PseudoHuber(PseudoHuber::new(3.0).unwrap()))
            .max_depth(2)
            .build()
            .unwrap();
        (train(&params, &data, 3).unwrap(), data)
    }

    /// A section this version does not know is skipped unless it is
    /// required, and then the file is refused instead of misread.
    #[test]
    fn unknown_sections_are_skipped_unless_required() {
        let (model, data) = model();
        let bytes = model.encode(ModelFormat::Binary).unwrap();
        for flags in [0, REQUIRED] {
            let edited = rewrite(&bytes, |entries| {
                entries.push(Entry {
                    name: "future.feature".into(),
                    flags,
                    payload: vec![1, 2, 3],
                });
            });
            let loaded = BoostedModel::decode(&edited, ModelFormat::Binary);
            if flags == 0 {
                assert_eq!(
                    loaded.unwrap().predict(&data, Iterations::Best).unwrap(),
                    model.predict(&data, Iterations::Best).unwrap()
                );
            } else {
                let err = loaded.unwrap_err().to_string();
                assert!(err.contains("future.feature"), "{err}");
            }
        }
    }

    /// A reader that predates the `boulevard.*` sections (here: they are
    /// renamed to names this reader does not know) still loads a Boulevard
    /// model and predicts the same, because the sections are optional.
    #[test]
    fn boulevard_sections_are_optional_for_older_readers() {
        let (_, data) = model();
        let params = TrainingParams::builder()
            .booster(crate::config::BoosterKind::Boulevard(
                crate::config::Boulevard::default(),
            ))
            .max_depth(2)
            .build()
            .unwrap();
        let model = train(&params, &data, 3).unwrap();
        let bytes = model.encode(ModelFormat::Binary).unwrap();
        let edited = rewrite(&bytes, |entries| {
            for e in entries.iter_mut() {
                if let Some(rest) = e.name.strip_prefix("boulevard.") {
                    e.name = format!("future.{rest}");
                }
            }
        });
        let loaded = BoostedModel::decode(&edited, ModelFormat::Binary).unwrap();
        assert!(loaded.boulevard().is_none());
        assert_eq!(
            loaded.predict(&data, Iterations::Best).unwrap(),
            model.predict(&data, Iterations::Best).unwrap()
        );
    }

    /// Every file names its writer in an optional section, so a reader that
    /// predates the section (or drops it) loads the file the same.
    #[test]
    fn files_name_their_writer_in_an_optional_section() {
        let (model, data) = model();
        let bytes = model.encode(ModelFormat::Binary).unwrap();
        let mut writer = None;
        let without = rewrite(&bytes, |entries| {
            let at = entries
                .iter()
                .position(|e| e.name == "model.writer")
                .unwrap();
            let entry = entries.remove(at);
            writer = Some((entry.flags, String::from_utf8(entry.payload).unwrap()));
        });
        let (flags, name) = writer.unwrap();
        assert_eq!(flags, 0);
        assert_eq!(name, concat!("hessboost ", env!("CARGO_PKG_VERSION")));
        assert_eq!(
            BoostedModel::decode(&without, ModelFormat::Binary)
                .unwrap()
                .predict(&data, Iterations::Best)
                .unwrap(),
            model.predict(&data, Iterations::Best).unwrap()
        );
    }

    /// Values a later writer could give a meaning are refused rather than
    /// ignored: undefined `node.flags` bits, `tree.has_linear` bytes other
    /// than 0 and 1, and bytes after the last section.
    #[test]
    fn undefined_encodings_are_refused() {
        let (model, _) = model();
        let bytes = model.encode(ModelFormat::Binary).unwrap();
        let set = |name: &'static str, value: u8| {
            rewrite(&bytes, |entries| {
                let entry = entries.iter_mut().find(|e| e.name == name).unwrap();
                entry.payload[0] |= value;
            })
        };
        for (edited, needle) in [
            (set("node.flags", 4), "node.flags"),
            (set("node.flags", 0x80), "node.flags"),
            (set("tree.has_linear", 2), "tree.has_linear"),
        ] {
            let err = BoostedModel::decode(&edited, ModelFormat::Binary)
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{err}");
        }

        let container = rewrite(&bytes, |_| {});
        let mut trailing = container[..container.len() - 8].to_vec();
        trailing.push(0);
        let checksum = xxh64(&trailing);
        trailing.extend_from_slice(&checksum.to_le_bytes());
        let err = BoostedModel::decode(&trailing, ModelFormat::Binary)
            .unwrap_err()
            .to_string();
        assert!(err.contains("after the last section"), "{err}");
        // The unedited re-encoding still loads: the refusals come from the
        // edits alone.
        assert!(BoostedModel::decode(&container, ModelFormat::Binary).is_ok());
    }

    /// Files of the pre-0.2.0 format are recognized by their magic and
    /// refused, naming the upgrade path.
    #[test]
    fn pre_0_2_files_are_named_in_the_refusal() {
        let (model, _) = model();
        let mut legacy = rewrite(&model.encode(ModelFormat::Binary).unwrap(), |_| {});
        legacy[..4].copy_from_slice(LEGACY_MAGIC);
        let err = BoostedModel::decode(&legacy, ModelFormat::Binary)
            .unwrap_err()
            .to_string();
        assert!(err.contains("0.1.x"), "{err}");
    }

    /// An objective parameter a file does not store takes the objective's
    /// default, as for files written before the parameter existed.
    #[test]
    fn absent_objective_sections_take_their_defaults() {
        let (model, _) = model();
        let huber = |slope| Some(Objective::PseudoHuber(PseudoHuber::new(slope).unwrap()));
        assert_eq!(model.objective().built_in().cloned(), huber(3.0));
        let edited = rewrite(&model.encode(ModelFormat::Binary).unwrap(), |entries| {
            entries.retain(|e| e.name != "objective.huber_slope");
        });
        let loaded = BoostedModel::decode(&edited, ModelFormat::Binary).unwrap();
        assert_eq!(loaded.objective().built_in().cloned(), huber(1.0));
    }

    /// Sections the model needs must be present and consistent.
    #[test]
    fn missing_or_inconsistent_sections_are_refused() {
        let (model, _) = model();
        let bytes = model.encode(ModelFormat::Binary).unwrap();
        let without = rewrite(&bytes, |entries| {
            entries.retain(|e| e.name != "node.leaf_value");
        });
        let shortened = rewrite(&bytes, |entries| {
            for e in entries.iter_mut().filter(|e| e.name == "node.split_cond") {
                e.payload.truncate(e.payload.len() - 4);
            }
        });
        for edited in [without, shortened] {
            assert!(matches!(
                BoostedModel::decode(&edited, ModelFormat::Binary),
                Err(crate::error::HessboostError::ModelFormat(_))
            ));
        }
    }

    /// Any flipped bit of a stored value is refused, not loaded as a
    /// different model.
    #[test]
    fn corrupted_values_fail_the_checksum() {
        let (model, _) = model();
        let container = rewrite(&model.encode(ModelFormat::Binary).unwrap(), |_| {});
        assert!(BoostedModel::decode(&container, ModelFormat::Binary).is_ok());
        let mut corrupt = container.clone();
        // Inside the last section's payload, just before the checksum.
        let at = corrupt.len() - 9;
        corrupt[at] ^= 1;
        let err = BoostedModel::decode(&corrupt, ModelFormat::Binary)
            .unwrap_err()
            .to_string();
        assert!(err.contains("checksum"), "{err}");
    }

    /// The container bytes of a gblinear model with `n_features` zero
    /// weights: the most compressible model there is.
    fn zero_linear_model(n_features: usize) -> Vec<u8> {
        write(&BoostedModel {
            trees: Vec::new(),
            base_score: vec![0.5],
            objective: ModelObjective::trained_with(&Objective::SquaredError(RegLoss::default())),
            max_delta_step: 0.0,
            num_class: 0,
            n_outputs: 1,
            n_targets: 1,
            n_features,
            best_iteration: None,
            tree_weights: TreeWeights::Unit,
            num_parallel_tree: 1,
            linear: Some(LinearModel::new(vec![0.0; n_features], vec![0.0])),
            shrinkage: None,
            boulevard: None,
            ebm: None,
            compact: OnceLock::new(),
        })
        .unwrap()
    }

    /// A frame expanding far beyond [`MAX_EXPANSION`] below the size cap
    /// stays compressed and loads (a 16 MiB container of zero weights
    /// compresses to a few KiB).
    #[test]
    fn highly_compressible_models_load_after_saving() {
        let n = 1 << 22;
        let bytes = zero_linear_model(n);
        assert!(bytes.starts_with(&ZSTD_MAGIC));
        assert!((bytes.len() as u64).saturating_mul(MAX_EXPANSION) < 4 * n as u64);
        let stored = read(&bytes).unwrap();
        let linear = stored.linear.unwrap();
        assert_eq!(linear.weights().len(), n);
        assert!(linear.weights().iter().all(|&w| w == 0.0));
    }

    /// Past the size cap, a container whose frame the reader would refuse is
    /// written uncompressed and loads.
    #[test]
    #[ignore = "allocates about 1 GiB"]
    fn models_past_the_frame_policy_are_written_uncompressed() {
        let n = 67_108_864;
        let bytes = zero_linear_model(n);
        assert!(bytes.starts_with(MAGIC));
        assert_eq!(read(&bytes).unwrap().linear.unwrap().weights().len(), n);
    }
}
