//! The native binary model format: a [section table](super::sections) behind
//! a magic and a container version, compressed as one zstd frame.
//!
//! ```text
//! b"SQB\0"   magic
//! u8         container version (CONTAINER_VERSION)
//! sections   the model (see below)
//! u64        XXH64 (seed 0) of every byte before it
//! ```
//!
//! The model's scalar fields are `model.*` sections, its objective
//! parameters `objective.*`, and the trees are stored column-wise: one array
//! per node field across all trees (`node.*`), split per tree by
//! `tree.node_count`, plus per-tree category pools, leaf vectors and leaf
//! linear models.
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
//! container; uncompressed containers are read as well. The trailing
//! checksum catches corruption either way.

use std::io::Read;

use super::model::LinearModel;
use super::sections::{Sections, Writer, format_error};
use crate::config::{AftDistribution, DistGradient, DistSplitDirection, ObjectiveParams};
use crate::error::Result;
use crate::objective::DistFamily;
use crate::tree::linear::LinearLeaves;
use crate::tree::{Node, RegTree};

const MAGIC: &[u8; 4] = b"SQB\0";
/// Layout version of the container (not of the model it holds).
const CONTAINER_VERSION: u8 = 3;
/// The zstd frame magic number, little-endian.
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
/// Decompressed containers up to this size always load; larger ones must
/// stay within [`MAX_EXPANSION`] of their compressed size. Together they
/// bound what a small malicious file can make the reader allocate.
const ALWAYS_ALLOWED: u64 = 256 << 20;
/// Largest accepted ratio of decompressed to compressed size above
/// [`ALWAYS_ALLOWED`].
const MAX_EXPANSION: u64 = 1 << 12;

/// `default_left` in `node.flags`.
const DEFAULT_LEFT: u8 = 1;
/// `is_categorical` in `node.flags`.
const CATEGORICAL: u8 = 2;

/// Every section this version reads.
const KNOWN: &[&str] = &[
    "model.objective",
    "model.base_score",
    "model.num_class",
    "model.n_outputs",
    "model.n_targets",
    "model.n_features",
    "model.best_iteration",
    "model.tree_weights",
    "model.num_parallel_tree",
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
];

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

/// A model's stored fields, as [`read`] returns them.
pub(super) struct Stored {
    pub(super) trees: Vec<RegTree>,
    pub(super) base_score: Vec<f32>,
    pub(super) objective: String,
    pub(super) objective_params: ObjectiveParams,
    pub(super) num_class: usize,
    pub(super) n_outputs: usize,
    pub(super) n_targets: usize,
    pub(super) n_features: usize,
    pub(super) best_iteration: Option<usize>,
    pub(super) tree_weights: Vec<f32>,
    pub(super) num_parallel_tree: usize,
    pub(super) linear: Option<LinearModel>,
}

/// The fields [`write`] reads, by reference.
pub(super) struct StoredRef<'a> {
    pub(super) trees: &'a [RegTree],
    pub(super) base_score: &'a [f32],
    pub(super) objective: &'a str,
    pub(super) objective_params: &'a ObjectiveParams,
    pub(super) num_class: usize,
    pub(super) n_outputs: usize,
    pub(super) n_targets: usize,
    pub(super) n_features: usize,
    pub(super) best_iteration: Option<usize>,
    pub(super) tree_weights: &'a [f32],
    pub(super) num_parallel_tree: usize,
    pub(super) linear: Option<&'a LinearModel>,
}

/// Encode a model as a zstd-compressed container.
pub(super) fn write(m: &StoredRef) -> Result<Vec<u8>> {
    let trees = m.trees;
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
    w.str("model.objective", m.objective);
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
    if let Some(linear) = m.linear {
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
    write_objective_params(&mut w, m.objective_params);

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

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.push(CONTAINER_VERSION);
    w.finish(&mut out);
    let checksum = xxh64(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    Ok(zstd::bulk::compress(&out, zstd::DEFAULT_COMPRESSION_LEVEL)?)
}

/// Decode a container (zstd-compressed or not). The caller validates the
/// model it forms.
pub(super) fn read(bytes: &[u8]) -> Result<Stored> {
    let decompressed;
    let container = if bytes.starts_with(&ZSTD_MAGIC) {
        decompressed = decompress(bytes)?;
        decompressed.as_slice()
    } else {
        bytes
    };
    let Some(body) = container.strip_prefix(MAGIC) else {
        return Err(format_error("invalid native model header"));
    };
    let Some(&version) = body.first() else {
        return Err(format_error("truncated native model"));
    };
    if version != CONTAINER_VERSION {
        return Err(format_error(format!(
            "unsupported native model version {version}"
        )));
    }
    let Some(split) = container
        .len()
        .checked_sub(8)
        .filter(|&at| at > MAGIC.len())
    else {
        return Err(format_error("truncated native model"));
    };
    let (checked, checksum) = container.split_at(split);
    if xxh64(checked).to_le_bytes() != checksum {
        return Err(format_error("native model checksum mismatch"));
    }
    let (s, _) = Sections::parse(&checked[MAGIC.len() + 1..], |name| {
        KNOWN.contains(&name) || OBJECTIVE_SECTIONS.contains(&name)
    })?;

    let objective = s.str("model.objective")?.to_string();
    let objective_params = read_objective_params(&s, ObjectiveParams::defaults_for(&objective))?;
    let linear = if s.has("gblinear.weights") || s.has("gblinear.bias") {
        Some(LinearModel::new(
            s.array("gblinear.weights", f32::from_le_bytes)?,
            s.array("gblinear.bias", f32::from_le_bytes)?,
        ))
    } else {
        None
    };
    Ok(Stored {
        trees: read_trees(&s)?,
        base_score: s.array("model.base_score", f32::from_le_bytes)?,
        objective,
        objective_params,
        num_class: s.usize("model.num_class")?,
        n_outputs: s.usize("model.n_outputs")?,
        n_targets: s.usize("model.n_targets")?,
        n_features: s.usize("model.n_features")?,
        best_iteration: if s.has("model.best_iteration") {
            Some(s.usize("model.best_iteration")?)
        } else {
            None
        },
        tree_weights: s.array("model.tree_weights", f32::from_le_bytes)?,
        num_parallel_tree: s.usize("model.num_parallel_tree")?,
        linear,
    })
}

fn read_trees(s: &Sections) -> Result<Vec<RegTree>> {
    let node_count = s.array("tree.node_count", u32::from_le_bytes)?;
    let n_trees = node_count.len();
    let n_nodes = checked_sum(&node_count)?;
    let column = |name: &str, width: usize| -> Result<()> {
        if s.bytes(name)?.len() == n_nodes * width {
            Ok(())
        } else {
            Err(wrong_length(name))
        }
    };
    let per_tree = |name: &str, width: usize| -> Result<()> {
        if s.bytes(name)?.len() == n_trees * width {
            Ok(())
        } else {
            Err(wrong_length(name))
        }
    };
    for name in [
        "node.split_feature",
        "node.split_cond",
        "node.left",
        "node.right",
        "node.leaf_value",
        "node.sum_hess",
        "node.split_gain",
        "node.cat_begin",
        "node.cat_end",
    ] {
        column(name, 4)?;
    }
    column("node.flags", 1)?;
    per_tree("tree.category_count", 4)?;
    per_tree("tree.size_leaf_vector", 4)?;
    per_tree("tree.has_linear", 1)?;

    let split_feature = s.array("node.split_feature", u32::from_le_bytes)?;
    let split_cond = s.array("node.split_cond", f32::from_le_bytes)?;
    let left = s.array("node.left", i32::from_le_bytes)?;
    let right = s.array("node.right", i32::from_le_bytes)?;
    let leaf_value = s.array("node.leaf_value", f32::from_le_bytes)?;
    let sum_hess = s.array("node.sum_hess", f32::from_le_bytes)?;
    let split_gain = s.array("node.split_gain", f32::from_le_bytes)?;
    let cat_begin = s.array("node.cat_begin", u32::from_le_bytes)?;
    let cat_end = s.array("node.cat_end", u32::from_le_bytes)?;
    let flags = s.bytes("node.flags")?;
    let category_count = s.array("tree.category_count", u32::from_le_bytes)?;
    let mut categories = Cursor::new(s.array("tree.categories", u32::from_le_bytes)?);
    let size_leaf_vector = s.array("tree.size_leaf_vector", u32::from_le_bytes)?;
    let mut leaf_vectors = Cursor::new(s.array("tree.leaf_vectors", f32::from_le_bytes)?);
    let has_linear = s.bytes("tree.has_linear")?;
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
        let linear = if has_linear[t] != 0 {
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
pub(super) fn write_objective_params(w: &mut Writer, p: &ObjectiveParams) {
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
    base: ObjectiveParams,
) -> Result<ObjectiveParams> {
    let f64s = |s: &Sections, name: &str| s.array(name, f64::from_le_bytes);
    Ok(ObjectiveParams {
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
        .ok_or_else(|| format_error(format!("unknown `{name}` value `{value}`")))
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

fn decompress(bytes: &[u8]) -> Result<Vec<u8>> {
    let limit = (bytes.len() as u64)
        .saturating_mul(MAX_EXPANSION)
        .max(ALWAYS_ALLOWED);
    let decoder = zstd::stream::read::Decoder::with_buffer(bytes)
        .map_err(|e| format_error(format!("zstd: {e}")))?;
    let mut out = Vec::new();
    decoder
        .take(limit.saturating_add(1))
        .read_to_end(&mut out)
        .map_err(|e| format_error(format!("zstd: {e}")))?;
    if out.len() as u64 > limit {
        return Err(format_error("zstd frame expands too far"));
    }
    Ok(out)
}

/// XXH64 of `data` with seed 0 (Collet's xxHash, 64-bit variant).
fn xxh64(data: &[u8]) -> u64 {
    const P1: u64 = 0x9E37_79B1_85EB_CA87;
    const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const P3: u64 = 0x1656_67B1_9E37_79F9;
    const P4: u64 = 0x85EB_CA77_C2B2_AE63;
    const P5: u64 = 0x27D4_EB2F_1656_67C5;
    let round = |acc: u64, lane: u64| {
        acc.wrapping_add(lane.wrapping_mul(P2))
            .rotate_left(31)
            .wrapping_mul(P1)
    };
    let merge = |acc: u64, v: u64| (acc ^ round(0, v)).wrapping_mul(P1).wrapping_add(P4);

    let (stripes, tail) = data.as_chunks::<32>();
    let mut h = if stripes.is_empty() {
        P5
    } else {
        let mut v = [P1.wrapping_add(P2), P2, 0, P1.wrapping_neg()];
        for stripe in stripes {
            for (lane, acc) in stripe.as_chunks::<8>().0.iter().zip(&mut v) {
                *acc = round(*acc, u64::from_le_bytes(*lane));
            }
        }
        let h = v[0]
            .rotate_left(1)
            .wrapping_add(v[1].rotate_left(7))
            .wrapping_add(v[2].rotate_left(12))
            .wrapping_add(v[3].rotate_left(18));
        v.into_iter().fold(h, merge)
    };
    h = h.wrapping_add(data.len() as u64);

    let (words, mut rest) = tail.as_chunks::<8>();
    for word in words {
        h = (h ^ round(0, u64::from_le_bytes(*word)))
            .rotate_left(27)
            .wrapping_mul(P1)
            .wrapping_add(P4);
    }
    if let Some((half, after)) = rest.split_first_chunk::<4>() {
        let half = u32::from_le_bytes(*half);
        h = (h ^ u64::from(half).wrapping_mul(P1))
            .rotate_left(23)
            .wrapping_mul(P2)
            .wrapping_add(P3);
        rest = after;
    }
    for &byte in rest {
        h = (h ^ u64::from(byte).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

fn checked_sum(counts: &[u32]) -> Result<usize> {
    counts
        .iter()
        .try_fold(0usize, |sum, &c| sum.checked_add(c as usize))
        .ok_or_else(|| format_error("section lengths overflow"))
}

fn wrong_length(name: &str) -> crate::error::HessboostError {
    format_error(format!("section `{name}` has the wrong length"))
}

#[cfg(test)]
mod tests {
    use super::super::sections::REQUIRED;
    use super::*;
    use crate::config::TrainingParams;
    use crate::learner::{BoostedModel, train};
    use crate::test_support::labeled_dense;

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
            .objective("reg:pseudohubererror")
            .huber_slope(3.0)
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
        let bytes = model.to_bytes().unwrap();
        for flags in [0, REQUIRED] {
            let edited = rewrite(&bytes, |entries| {
                entries.push(Entry {
                    name: "future.feature".into(),
                    flags,
                    payload: vec![1, 2, 3],
                });
            });
            let loaded = BoostedModel::from_bytes(&edited);
            if flags == 0 {
                assert_eq!(
                    loaded.unwrap().predict(&data).unwrap(),
                    model.predict(&data).unwrap()
                );
            } else {
                let err = loaded.unwrap_err().to_string();
                assert!(err.contains("future.feature"), "{err}");
            }
        }
    }

    /// An objective parameter a file does not store takes the objective's
    /// default, as for files written before the parameter existed.
    #[test]
    fn absent_objective_sections_take_their_defaults() {
        let (model, _) = model();
        assert_eq!(model.objective_params().huber_slope, 3.0);
        let edited = rewrite(&model.to_bytes().unwrap(), |entries| {
            entries.retain(|e| e.name != "objective.huber_slope");
        });
        let loaded = BoostedModel::from_bytes(&edited).unwrap();
        assert_eq!(loaded.objective_params().huber_slope, 1.0);
    }

    /// Sections the model needs must be present and consistent.
    #[test]
    fn missing_or_inconsistent_sections_are_refused() {
        let (model, _) = model();
        let bytes = model.to_bytes().unwrap();
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
                BoostedModel::from_bytes(&edited),
                Err(crate::error::HessboostError::ModelFormat(_))
            ));
        }
    }

    /// Reference digests from the `xxhash` C library, covering the short
    /// path, every tail length class, and the 32-byte stripes.
    #[test]
    fn xxh64_matches_the_reference() {
        let bytes: Vec<u8> = (0..100).collect();
        for (data, expected) in [
            (&b""[..], 0xef46_db37_51d8_e999),
            (b"a", 0xd24e_c4f1_a98c_6e5b),
            (b"abc", 0x44bc_2cf5_ad77_0999),
            (b"0123456789abcdef", 0x5c5b_90c3_4e37_6d0b),
            (&bytes[..31], 0xc346_d2b5_9b4d_8ee1),
            (&bytes[..32], 0xcbf5_9c51_16ff_32b4),
            (&bytes[..], 0x6ac1_e580_3216_6597),
        ] {
            assert_eq!(xxh64(data), expected, "{} bytes", data.len());
        }
    }

    /// Any flipped bit of a stored value is refused, not loaded as a
    /// different model.
    #[test]
    fn corrupted_values_fail_the_checksum() {
        let (model, _) = model();
        let container = rewrite(&model.to_bytes().unwrap(), |_| {});
        assert!(BoostedModel::from_bytes(&container).is_ok());
        let mut corrupt = container.clone();
        // Inside the last section's payload, just before the checksum.
        let at = corrupt.len() - 9;
        corrupt[at] ^= 1;
        let err = BoostedModel::from_bytes(&corrupt).unwrap_err().to_string();
        assert!(err.contains("checksum"), "{err}");
    }
}
