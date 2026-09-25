//! Compact, bit-packed tree-ensemble format after *Boosted Trees on a Diet*
//! (Herrmann et al., ICLR 2026, arXiv:2510.26557, §3.2), an opt-in extension
//! beyond XGBoost for memory-constrained inference.
//!
//! [`BoostedModel::to_compact`] turns a tree ensemble into a [`CompactModel`]
//! that predicts **bit-identical** margins (and transformed predictions) to
//! [`BoostedModel::predict_margin`] / [`BoostedModel::predict`], straight
//! from the packed layout. Training with the reuse penalties
//! [`toad_penalty_feature`](crate::config::TrainingParams::toad_penalty_feature)
//! and [`toad_penalty_threshold`](crate::config::TrainingParams::toad_penalty_threshold)
//! shrinks the shared dictionaries the layout references, so the two work
//! together; any tree model can be compacted, though.
//!
//! The format keeps exactly what prediction needs: node covers and split
//! gains (TreeSHAP, cover/gain importance) are dropped, and only the trees
//! [`BoostedModel::predict_margin`] uses are stored (the prefix up to
//! `best_iteration` when early stopping chose one). gblinear models,
//! linear-leaf trees (`linear_tree`) and vector-leaf trees
//! (`multi_output_tree`) are rejected. It is a hessboost format; XGBoost
//! cannot read it.
//!
//! # Layout (version 1)
//!
//! ```text
//! bytes 0..4   magic b"HBTD"
//! byte  4      format version (1)
//! bytes 5..9   u32 little-endian M, the metadata length
//! next M bytes metadata: a section table (`model::sections`, the
//!              native format's building block) holding the objective name,
//!              num_class, n_targets, num_parallel_tree, and the objective
//!              parameters when they differ from the objective's defaults;
//!              readers give sections a file lacks their default, so later
//!              additions keep older files loading
//! rest         one bit stream
//! ```
//!
//! The bit stream is least-significant-bit first: field bit `j` of a field
//! starting at stream position `p` is bit `(p + j) % 8` of byte
//! `(p + j) / 8`. Its final byte is zero-padded. Width `bits(n)` below is the
//! number of bits needed for the values `0..=n` (`bits(0) = 0`, so fields
//! with a single possible value take no space).
//!
//! 1. **Metadata:** `n_features` (32 bits), `n_outputs` (32), one `f32` base
//!    score per output (32 each), the tree count `K` (32), a weights flag (1)
//!    followed by `K` `f32` tree weights when set (DART), the default
//!    direction mode (2: `0` every split sends missing values left, `1`
//!    right, `2` a bit per split), the number of used features `|F_U|` (32),
//!    the largest per-feature dictionary size `T_max` (32), the number of
//!    distinct leaf values `L` (32), the width of heap-tree depths (6) and the
//!    width of preorder-tree node counts (6).
//! 2. **Feature & threshold map**, one entry per used feature in ascending
//!    input order: input feature index (`bits(n_features - 1)`), numeric
//!    type (2: `0` unsigned integer, `1` two's-complement integer, `2` IEEE
//!    float, `3` categorical set; the paper's 1-bit float/fixed flag plus
//!    the two kinds hessboost adds), width code `c` (3; value width `2^c`
//!    bits, `2^0..2^5`), dictionary size minus one (`bits(T_max - 1)`), and
//!    for categorical features the set-length width (6).
//! 3. **Global thresholds:** each used feature's dictionary, in map order.
//!    Numeric entries are ascending values at the feature's width: integers
//!    exactly representable in the chosen type, IEEE binary16 (width 16) or
//!    binary32 (width 32) floats; the narrowest width that reproduces every
//!    threshold bit for bit is chosen. Categorical entries are sorted
//!    left-category sets: the set length, then its categories.
//! 4. **Global leaf values:** `L` distinct `f32` leaf values (32 bits each).
//! 5. **Trees**, `K` in ensemble order (tree `t` feeds output
//!    `(t / num_parallel_tree) % n_outputs`, XGBoost's iteration-major
//!    layout). A layout bit selects:
//!    - *heap* (`0`, the paper's pointer-free layout): the depth `d` and a
//!      completeness bit, then `2^d − 1` internal slots and `2^d` leaf slots.
//!      The root is slot `0` and slot `i` has children `2i + 1` and
//!      `2i + 2`. A leaf slot holds a leaf-value reference
//!      (`bits(L − 1)`). An internal slot holds a feature reference
//!      (`bits(|F_U| − 1)`, into the map), a threshold reference
//!      (`bits(T_max − 1)`, into that feature's dictionary) and, in mode `2`,
//!      the default-left bit. Trees with leaves above depth `d` prefix every
//!      internal slot with a leaf flag and store those leaves' references in
//!      internal slots; slots below such leaves are zero.
//!    - *preorder* (`1`, for deep unbalanced trees where the heap would
//!      waste space): the node count minus one, then the nodes in preorder,
//!      each a leaf flag followed by a leaf reference or by the split fields
//!      above plus the distance to the right child (`bits(n − 1)`); the left
//!      child is the next node.
//!
//!    The encoder picks the smaller layout per tree (heap only up to depth
//!    24). Every slot of a tree has the same width, so the walk is index
//!    arithmetic.
//!
//! Routing is [`RegTree`]'s: a missing value (`NaN`
//! or the matrix's sentinel) follows the split's default direction, a
//! numeric split sends `x < threshold` left, and a categorical split sends
//! the categories of its set left. Margins accumulate `weight × leaf` per
//! output in tree order in `f32`, exactly as the native predictor does.
//!
//! # Example
//!
//! ```
//! use hessboost::model::compact::CompactModel;
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! let x: Vec<f32> = (0..400).map(|i| ((i * 37) % 101) as f32 / 10.0).collect();
//! let y: Vec<f32> = x.chunks(4).map(|r| r[0] + 2.0 * r[1]).collect();
//! let dtrain = DMatrix::from_dense(&x, 100, 4)?.with_labels(&y)?;
//! let params = TrainingParams::builder()
//!     .max_depth(3)
//!     .toad_penalty_feature(1.0)
//!     .toad_penalty_threshold(0.5)
//!     .build()?;
//! let model = train(&params, &dtrain, 20)?;
//!
//! let bytes = model.to_compact_bytes()?;
//! let compact = CompactModel::from_bytes(&bytes)?;
//! assert_eq!(compact.predict_margin(&dtrain)?, model.predict_margin(&dtrain)?);
//!
//! let report = model.size_report()?;
//! assert!(report.compact_bytes < report.native_bytes);
//! # Ok(())
//! # }
//! ```

use super::native::{OBJECTIVE_SECTIONS, read_objective_params, write_objective_params};
use super::sections::{Sections, Writer};
use crate::config::ObjectiveParams;
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::{
    BoostedModel, RowBlock, check_objective_width, initial_margins, transform_model_margins,
    validate_prediction_data,
};
use crate::tree::{Node, RegTree, scalar_tree_output};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};

const MAGIC: &[u8; 4] = b"HBTD";
const VERSION: u8 = 1;
/// Bytes before the metadata block: magic, version, metadata length.
const PREFIX_BYTES: usize = MAGIC.len() + 1 + 4;
/// Deepest tree the encoder stores in the heap layout; deeper trees always
/// use the preorder layout (a heap needs `2^(d+1) − 1` slots).
const MAX_HEAP_DEPTH: u32 = 24;
/// Zero bytes appended to the in-memory bit stream so a field read may
/// always load eight bytes.
const STREAM_PAD: usize = 8;

const KIND_UINT: u32 = 0;
const KIND_SINT: u32 = 1;
const KIND_FLOAT: u32 = 2;
const KIND_CATEGORICAL: u32 = 3;

const DEFAULT_ALL_LEFT: u32 = 0;
const DEFAULT_ALL_RIGHT: u32 = 1;
const DEFAULT_PER_NODE: u32 = 2;

/// Bits needed to store every value in `0..=n`.
fn bits(n: u64) -> u32 {
    u64::BITS - n.leading_zeros()
}

fn format_error(msg: impl Into<String>) -> HessboostError {
    HessboostError::model_format(format!("compact model: {}", msg.into()))
}

/// Metadata the transform and dimension checks need.
#[derive(Debug, Clone)]
struct Meta {
    objective: String,
    /// `None` when equal to [`ObjectiveParams::defaults_for`] the objective.
    objective_params: Option<ObjectiveParams>,
    num_class: usize,
    n_targets: usize,
    /// Trees per output in each boosting iteration: tree `t` feeds output
    /// `(t / num_parallel_tree) % n_outputs`.
    num_parallel_tree: usize,
}

/// Every metadata section besides [`OBJECTIVE_SECTIONS`].
const META_SECTIONS: &[&str] = &["objective", "num_class", "n_targets", "num_parallel_tree"];

impl Meta {
    fn objective_params(&self) -> ObjectiveParams {
        self.objective_params
            .clone()
            .unwrap_or_else(|| ObjectiveParams::defaults_for(&self.objective))
    }

    /// The metadata as a section table, the objective parameters only when
    /// they differ from the objective's defaults.
    fn section_table(&self) -> Writer {
        let mut w = Writer::default();
        w.str("objective", &self.objective);
        w.u64("num_class", self.num_class as u64);
        w.u64("n_targets", self.n_targets as u64);
        w.u64("num_parallel_tree", self.num_parallel_tree as u64);
        if let Some(params) = &self.objective_params {
            write_objective_params(&mut w, params);
        }
        w
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let (s, rest) = Sections::parse(bytes, |name| {
            META_SECTIONS.contains(&name) || OBJECTIVE_SECTIONS.contains(&name)
        })
        .map_err(|e| format_error(format!("metadata: {e}")))?;
        if !rest.is_empty() {
            return Err(format_error("metadata has trailing bytes"));
        }
        let objective = s.str("objective")?.to_string();
        let defaults = ObjectiveParams::defaults_for(&objective);
        let params = read_objective_params(&s, defaults.clone())?;
        Ok(Meta {
            objective_params: (params != defaults).then_some(params),
            objective,
            num_class: s.usize("num_class")?,
            n_targets: s.usize("n_targets")?,
            num_parallel_tree: s.usize("num_parallel_tree")?,
        })
    }
}

/// The serialized model: magic, version and length prefix, then the
/// metadata section table and the bit `stream`.
fn frame(meta: &Meta, stream: &[u8]) -> Vec<u8> {
    let table = meta.section_table();
    let mut bytes = Vec::with_capacity(PREFIX_BYTES + table.encoded_len() + stream.len());
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION);
    bytes.extend_from_slice(&[0; 4]);
    table.finish(&mut bytes);
    let meta_len = (bytes.len() - PREFIX_BYTES) as u32;
    bytes[MAGIC.len() + 1..PREFIX_BYTES].copy_from_slice(&meta_len.to_le_bytes());
    bytes.extend_from_slice(stream);
    bytes
}

/// The `(metadata, bit stream)` parts of framed `bytes`, after checking the
/// magic, the version and the metadata length.
fn split_frame(bytes: &[u8]) -> Result<(&[u8], &[u8])> {
    if bytes.len() < PREFIX_BYTES || &bytes[..MAGIC.len()] != MAGIC {
        return Err(format_error("invalid header"));
    }
    if bytes[MAGIC.len()] != VERSION {
        return Err(format_error(format!(
            "unsupported version {}",
            bytes[MAGIC.len()]
        )));
    }
    let meta_len = u32::from_le_bytes(
        bytes[MAGIC.len() + 1..PREFIX_BYTES]
            .try_into()
            .expect("four length bytes"),
    ) as usize;
    let meta_end = PREFIX_BYTES
        .checked_add(meta_len)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| format_error("truncated metadata"))?;
    Ok((&bytes[PREFIX_BYTES..meta_end], &bytes[meta_end..]))
}

// ---------------------------------------------------------------------------
// Bit I/O
// ---------------------------------------------------------------------------

/// Least-significant-bit-first bit stream writer.
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    len: usize,
}

impl BitWriter {
    /// Append the low `width` bits of `value` (the rest must be zero; a
    /// zero `value` may be any width, as padding).
    fn write(&mut self, value: u64, width: u32) {
        debug_assert!(
            width >= 64 || value >> width == 0,
            "value exceeds its field"
        );
        let mut value = value;
        let mut remaining = width;
        // Fill the partial last byte, then append whole bytes.
        let used = (self.len % 8) as u32;
        if used != 0
            && remaining > 0
            && let Some(last) = self.bytes.last_mut()
        {
            let take = (8 - used).min(remaining);
            *last |= ((value & ((1 << take) - 1)) as u8) << used;
            value >>= take;
            remaining -= take;
        }
        while remaining > 0 {
            let take = remaining.min(8);
            self.bytes.push((value & ((1 << take) - 1)) as u8);
            value >>= take;
            remaining -= take;
        }
        self.len += width as usize;
    }

    fn write_bool(&mut self, value: bool) {
        self.write(u64::from(value), 1);
    }

    fn write_f32(&mut self, value: f32) {
        self.write(u64::from(value.to_bits()), 32);
    }
}

/// Read a `width <= 32` bit field at bit `pos` of a padded stream. Callers
/// guarantee `pos + width` lies inside the unpadded stream.
#[inline]
fn read_bits(stream: &[u8], pos: usize, width: u32) -> u32 {
    debug_assert!(width <= 32);
    let byte = pos / 8;
    let word = u64::from_le_bytes(
        stream[byte..byte + 8]
            .try_into()
            .expect("the stream is padded to eight bytes"),
    );
    let mask = (1u64 << width) - 1;
    ((word >> (pos % 8)) & mask) as u32
}

/// Bounds-checked sequential reader used while parsing untrusted bytes.
struct BitReader<'a> {
    stream: &'a [u8],
    /// Bits in the unpadded stream.
    len: usize,
    pos: usize,
}

impl BitReader<'_> {
    fn remaining(&self) -> usize {
        self.len - self.pos
    }

    fn read(&mut self, width: u32) -> Result<u32> {
        if width > 32 {
            return Err(format_error(format!("field width {width} exceeds 32 bits")));
        }
        if self.remaining() < width as usize {
            return Err(format_error("truncated bit stream"));
        }
        let v = read_bits(self.stream, self.pos, width);
        self.pos += width as usize;
        Ok(v)
    }

    fn read_usize(&mut self, width: u32) -> Result<usize> {
        Ok(self.read(width)? as usize)
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read(1)? == 1)
    }

    /// `count` `f32` fields, refusing non-finite ones (`what` names one).
    fn read_finite_f32s(&mut self, count: usize, what: &str) -> Result<Vec<f32>> {
        self.ensure_fits(count, 32, what)?;
        let values = (0..count)
            .map(|_| Ok(f32::from_bits(self.read(32)?)))
            .collect::<Result<Vec<_>>>()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format_error(format!("{what}s must be finite")));
        }
        Ok(values)
    }

    /// Fail unless `count` items of at least `min_bits` each still fit, so a
    /// corrupt count cannot trigger a huge allocation.
    fn ensure_fits(&self, count: usize, min_bits: usize, what: &str) -> Result<()> {
        match count.checked_mul(min_bits) {
            Some(total) if total <= self.remaining() => Ok(()),
            _ => Err(format_error(format!(
                "{what} count {count} exceeds the data"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Threshold value encodings
// ---------------------------------------------------------------------------

/// The IEEE binary16 encoding of `v` when it represents `v` exactly.
fn f16_exact(v: f32) -> Option<u16> {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    let h = if exp == 0 {
        // Zero (f32 subnormals are far below the binary16 range).
        if man != 0 {
            return None;
        }
        sign
    } else {
        let e = exp - 127;
        if (-14..=15).contains(&e) {
            if man & 0x1fff != 0 {
                return None;
            }
            sign | (((e + 15) as u16) << 10) | (man >> 13) as u16
        } else if (-24..-14).contains(&e) {
            // binary16 subnormal: value = m · 2^-24 with m = significand >> (−e − 1).
            let significand = man | 0x80_0000;
            let shift = (-e - 1) as u32;
            if significand & ((1 << shift) - 1) != 0 {
                return None;
            }
            sign | (significand >> shift) as u16
        } else {
            return None;
        }
    };
    (f16_to_f32(h).to_bits() == bits).then_some(h)
}

/// Decode an IEEE binary16 value (exact in `f32`).
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let magnitude = match exp {
        0 => (man as f32 * f32::from_bits(0x3380_0000)).to_bits(), // m · 2^-24
        0x1f => 0x7f80_0000 | (man << 13),
        _ => ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// The integer `v` represents exactly (same bits back), if any.
fn exact_integer(v: f32) -> Option<i64> {
    let i = v as i64;
    ((i as f32).to_bits() == v.to_bits()).then_some(i)
}

/// Narrowest `(kind, width code)` that reproduces every value of `values`.
fn numeric_encoding(values: &[f32]) -> (u32, u32) {
    let ints: Option<Vec<i64>> = values.iter().map(|&v| exact_integer(v)).collect();
    for code in 0..=5u32 {
        let w = 1u32 << code;
        if let Some(ints) = &ints {
            if ints.iter().all(|&i| i >= 0 && i < (1i64 << w)) {
                return (KIND_UINT, code);
            }
            let half = 1i64 << (w - 1);
            if ints.iter().all(|&i| (-half..half).contains(&i)) {
                return (KIND_SINT, code);
            }
        }
        if w == 16 && values.iter().all(|&v| f16_exact(v).is_some()) {
            return (KIND_FLOAT, code);
        }
    }
    (KIND_FLOAT, 5)
}

fn encode_threshold(v: f32, kind: u32, width: u32) -> u64 {
    match (kind, width) {
        (KIND_UINT, _) => v as u64,
        (KIND_SINT, _) => (v as i64 as u64) & ((1u64 << width) - 1),
        (_, 16) => u64::from(f16_exact(v).expect("width chosen for exact binary16")),
        _ => u64::from(v.to_bits()),
    }
}

fn decode_threshold(raw: u32, kind: u32, width: u32) -> Result<f32> {
    let v = match (kind, width) {
        (KIND_UINT, _) => raw as f32,
        (KIND_SINT, _) => {
            let shift = 32 - width;
            ((raw << shift) as i32 >> shift) as f32
        }
        (KIND_FLOAT, 16) => f16_to_f32(raw as u16),
        (KIND_FLOAT, 32) => f32::from_bits(raw),
        _ => return Err(format_error(format!("invalid {width}-bit float threshold"))),
    };
    if !v.is_finite() {
        return Err(format_error("thresholds must be finite"));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// One used feature's dictionary.
#[derive(Debug, Clone)]
enum Dictionary {
    /// Ascending numeric thresholds (`x < t` goes left).
    Numeric(Vec<f32>),
    /// Sorted left-category sets.
    Categorical(Vec<Vec<u32>>),
}

impl Dictionary {
    fn len(&self) -> usize {
        match self {
            Dictionary::Numeric(t) => t.len(),
            Dictionary::Categorical(s) => s.len(),
        }
    }
}

/// A Feature & threshold map entry with its decoded dictionary.
#[derive(Debug, Clone)]
struct FeatureEntry {
    input: usize,
    dict: Dictionary,
}

/// Per-model field widths of the tree slots.
#[derive(Debug, Clone, Copy)]
struct Widths {
    feature_ref: u32,
    threshold_ref: u32,
    leaf_ref: u32,
    /// `1` when splits carry their own default-left bit.
    default_bit: u32,
}

impl Widths {
    /// Widths for `n_used` used features, dictionaries of at most
    /// `max_thresholds` entries, `n_leaves` leaf values and `default_mode`.
    fn new(n_used: usize, max_thresholds: usize, n_leaves: usize, default_mode: u32) -> Self {
        Widths {
            feature_ref: bits(n_used.saturating_sub(1) as u64),
            threshold_ref: bits(max_thresholds.saturating_sub(1) as u64),
            leaf_ref: bits(n_leaves.saturating_sub(1) as u64),
            default_bit: u32::from(default_mode == DEFAULT_PER_NODE),
        }
    }

    /// Width of a split's feature, threshold and default fields.
    fn split(self) -> u32 {
        self.feature_ref + self.threshold_ref + self.default_bit
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Heap { depth: u32, complete: bool },
    Preorder { nodes: u32 },
}

impl Layout {
    /// Width of one internal (heap) or node (preorder) slot.
    fn slot_width(self, w: Widths) -> u32 {
        match self {
            Layout::Heap { complete: true, .. } => w.split(),
            Layout::Heap { .. } => 1 + w.split().max(w.leaf_ref),
            Layout::Preorder { nodes } => {
                1 + (w.split() + bits(u64::from(nodes) - 1)).max(w.leaf_ref)
            }
        }
    }

    /// Total bits of the tree's slots.
    fn total_bits(self, w: Widths) -> u128 {
        let slot = u128::from(self.slot_width(w));
        match self {
            Layout::Heap { depth, .. } => {
                let leaves = 1u128 << depth;
                (leaves - 1) * slot + leaves * u128::from(w.leaf_ref)
            }
            Layout::Preorder { nodes } => u128::from(nodes) * slot,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PackedTree {
    layout: Layout,
    /// Bit offset of the tree's first slot in the padded serialized bytes.
    offset: usize,
}

/// A decoded slot.
enum Slot {
    Leaf(u32),
    Split {
        feature: u32,
        threshold: u32,
        default_left: bool,
        /// Preorder only: distance to the right child.
        right: u32,
    },
}

/// A tree ensemble in the bit-packed *Trees on a Diet* layout (see the
/// [module docs](self) for the byte layout), predicting bit-identical
/// margins to the [`BoostedModel`] it was built from.
///
/// Build one with [`BoostedModel::to_compact`] or parse
/// [`BoostedModel::to_compact_bytes`] output with [`CompactModel::from_bytes`].
/// Prediction walks the packed trees directly; the in-memory footprint is the
/// serialized bytes plus the decoded threshold and leaf dictionaries.
#[derive(Debug, Clone)]
pub struct CompactModel {
    /// The serialized form, as [`CompactModel::to_bytes`] returns it,
    /// followed by [`STREAM_PAD`] zero bytes so a field read may always load
    /// eight bytes; tree offsets index its bits.
    bytes: Vec<u8>,
    meta: Meta,
    n_features: usize,
    base_score: Vec<f32>,
    tree_weights: Option<Vec<f32>>,
    default_mode: u32,
    features: Vec<FeatureEntry>,
    leaf_values: Vec<f32>,
    widths: Widths,
    trees: Vec<PackedTree>,
}

/// The bit stream's leading metadata fields (layout item 1).
struct Header {
    n_features: usize,
    base_score: Vec<f32>,
    n_trees: usize,
    tree_weights: Option<Vec<f32>>,
    default_mode: u32,
    n_used: usize,
    max_thresholds: u32,
    n_leaves: usize,
    heap_depth_bits: u32,
    preorder_nodes_bits: u32,
}

/// One feature map entry (layout item 2) before its dictionary is read.
struct MapSpec {
    input: usize,
    kind: u32,
    /// Value width in bits.
    width: u32,
    /// Dictionary size.
    count: usize,
    /// Width of a categorical set's length (`0` for numeric features).
    len_bits: u32,
}

/// Read the [`Header`] and check it against the metadata `meta`.
fn read_header(r: &mut BitReader, meta: &Meta) -> Result<Header> {
    let n_features = r.read_usize(32)?;
    let n_outputs = r.read_usize(32)?;
    if n_features == 0 || n_outputs == 0 {
        return Err(format_error("feature and output counts must be positive"));
    }
    if meta.num_class >= 2 && n_outputs != meta.num_class {
        return Err(format_error("output count does not match num_class"));
    }
    check_objective_width(
        &meta.objective,
        &meta.objective_params(),
        meta.num_class,
        meta.n_targets,
        n_outputs,
    )?;
    let base_score = r.read_finite_f32s(n_outputs, "base score")?;
    let n_trees = r.read_usize(32)?;
    let per_iteration = n_outputs
        .checked_mul(meta.num_parallel_tree)
        .ok_or_else(|| format_error("trees per iteration overflow"))?;
    if !n_trees.is_multiple_of(per_iteration) {
        return Err(format_error(
            "tree count is not a multiple of the trees per iteration",
        ));
    }
    r.ensure_fits(n_trees, 1, "tree")?;
    let tree_weights = if r.read_bool()? {
        Some(r.read_finite_f32s(n_trees, "tree weight")?)
    } else {
        None
    };
    let default_mode = r.read(2)?;
    if default_mode > DEFAULT_PER_NODE {
        return Err(format_error("invalid default-direction mode"));
    }
    let header = Header {
        n_features,
        base_score,
        n_trees,
        tree_weights,
        default_mode,
        n_used: r.read_usize(32)?,
        max_thresholds: r.read(32)?,
        n_leaves: r.read_usize(32)?,
        heap_depth_bits: r.read(6)?,
        preorder_nodes_bits: r.read(6)?,
    };
    if header.n_used > n_features || (header.n_used == 0) != (header.max_thresholds == 0) {
        return Err(format_error("inconsistent feature map counts"));
    }
    Ok(header)
}

/// Read the feature & threshold map (layout item 2).
fn read_feature_map(r: &mut BitReader, header: &Header, widths: Widths) -> Result<Vec<MapSpec>> {
    let input_bits = bits(header.n_features as u64 - 1);
    r.ensure_fits(header.n_used, 5, "used feature")?;
    let mut map: Vec<MapSpec> = Vec::with_capacity(header.n_used);
    for _ in 0..header.n_used {
        let input = r.read_usize(input_bits)?;
        let kind = r.read(2)?;
        let code = r.read(3)?;
        if code > 5 {
            return Err(format_error("threshold width code exceeds 5"));
        }
        let count = r
            .read_usize(widths.threshold_ref)?
            .checked_add(1)
            .ok_or_else(|| format_error("dictionary size overflow"))?;
        if count > header.max_thresholds as usize {
            return Err(format_error("dictionary larger than its declared maximum"));
        }
        let len_bits = if kind == KIND_CATEGORICAL {
            r.read(6)?
        } else {
            0
        };
        if kind == KIND_CATEGORICAL && !(1..=32).contains(&len_bits) {
            return Err(format_error("category set lengths need 1 to 32 bits"));
        }
        if input >= header.n_features || map.last().is_some_and(|prev| prev.input >= input) {
            return Err(format_error("feature map entries must ascend"));
        }
        map.push(MapSpec {
            input,
            kind,
            width: 1u32 << code,
            count,
            len_bits,
        });
    }
    Ok(map)
}

/// Read each map entry's dictionary (layout item 3, the global thresholds).
fn read_dictionaries(r: &mut BitReader, map: &[MapSpec]) -> Result<Vec<FeatureEntry>> {
    let mut features = Vec::with_capacity(map.len());
    for spec in map {
        let &MapSpec {
            input,
            kind,
            width,
            count,
            len_bits,
        } = spec;
        let dict = if kind == KIND_CATEGORICAL {
            r.ensure_fits(count, len_bits as usize, "category set")?;
            let mut sets = Vec::with_capacity(count);
            for _ in 0..count {
                let len = r.read_usize(len_bits)?;
                r.ensure_fits(len, width as usize, "category")?;
                let set = (0..len)
                    .map(|_| r.read(width))
                    .collect::<Result<Vec<_>>>()?;
                if !set.is_sorted() {
                    return Err(format_error("category sets must be sorted"));
                }
                sets.push(set);
            }
            Dictionary::Categorical(sets)
        } else {
            r.ensure_fits(count, width as usize, "threshold")?;
            let values = (0..count)
                .map(|_| decode_threshold(r.read(width)?, kind, width))
                .collect::<Result<Vec<_>>>()?;
            Dictionary::Numeric(values)
        };
        features.push(FeatureEntry { input, dict });
    }
    Ok(features)
}

/// Read each tree's layout (layout item 5) and record its slot offset,
/// skipping its slots; nothing but zero padding may follow the last tree.
fn read_tree_layouts(
    r: &mut BitReader,
    header: &Header,
    widths: Widths,
) -> Result<Vec<PackedTree>> {
    let mut trees = Vec::with_capacity(header.n_trees);
    for _ in 0..header.n_trees {
        let layout = if r.read_bool()? {
            let nodes = r.read(header.preorder_nodes_bits)?.checked_add(1);
            Layout::Preorder {
                nodes: nodes.ok_or_else(|| format_error("preorder node count overflow"))?,
            }
        } else {
            let depth = r.read(header.heap_depth_bits)?;
            if depth > MAX_HEAP_DEPTH {
                return Err(format_error(format!(
                    "heap tree deeper than {MAX_HEAP_DEPTH}"
                )));
            }
            Layout::Heap {
                depth,
                complete: r.read_bool()?,
            }
        };
        let total = layout.total_bits(widths);
        if total > r.remaining() as u128 {
            return Err(format_error("truncated tree"));
        }
        trees.push(PackedTree {
            layout,
            offset: r.pos,
        });
        r.pos += total as usize;
    }
    if r.remaining() >= 8 || read_bits(r.stream, r.pos, r.remaining() as u32) != 0 {
        return Err(format_error("trailing data after the last tree"));
    }
    Ok(trees)
}

impl CompactModel {
    /// Parse bytes written by [`CompactModel::to_bytes`] /
    /// [`BoostedModel::to_compact_bytes`]. Every reference is validated, so
    /// prediction on a parsed model cannot index out of bounds.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (meta, stream) = split_frame(bytes)?;
        let meta = Meta::decode(meta)?;
        if meta.n_targets == 0 {
            return Err(format_error("n_targets must be positive"));
        }
        if meta.num_parallel_tree == 0 {
            return Err(format_error("num_parallel_tree must be positive"));
        }

        let mut padded = Vec::with_capacity(bytes.len() + STREAM_PAD);
        padded.extend_from_slice(bytes);
        padded.resize(bytes.len() + STREAM_PAD, 0);
        let too_large = || format_error("model is too large");
        let mut r = BitReader {
            stream: &padded,
            len: bytes.len().checked_mul(8).ok_or_else(too_large)?,
            pos: (bytes.len() - stream.len())
                .checked_mul(8)
                .ok_or_else(too_large)?,
        };
        let header = read_header(&mut r, &meta)?;
        let widths = Widths::new(
            header.n_used,
            header.max_thresholds as usize,
            header.n_leaves,
            header.default_mode,
        );
        let map = read_feature_map(&mut r, &header, widths)?;
        let features = read_dictionaries(&mut r, &map)?;
        let leaf_values = r.read_finite_f32s(header.n_leaves, "leaf value")?;
        let trees = read_tree_layouts(&mut r, &header, widths)?;
        let model = CompactModel {
            bytes: padded,
            meta,
            n_features: header.n_features,
            base_score: header.base_score,
            tree_weights: header.tree_weights,
            default_mode: header.default_mode,
            features,
            leaf_values,
            widths,
            trees,
        };
        for t in 0..model.trees.len() {
            model.validate_tree(t)?;
        }
        Ok(model)
    }

    /// The serialized bytes, without the padding.
    fn serialized(&self) -> &[u8] {
        &self.bytes[..self.bytes.len() - STREAM_PAD]
    }

    /// Check every reachable slot of tree `t`: references inside the
    /// dictionaries, preorder children inside the tree and after their
    /// parent (so every walk terminates).
    fn validate_tree(&self, t: usize) -> Result<()> {
        let tree = self.trees[t];
        let check = |slot: Slot, index: u32, nodes: u32| -> Result<()> {
            match slot {
                Slot::Leaf(leaf) => {
                    if leaf as usize >= self.leaf_values.len() {
                        return Err(format_error(format!(
                            "tree {t}: leaf reference out of range"
                        )));
                    }
                }
                Slot::Split {
                    feature,
                    threshold,
                    right,
                    ..
                } => {
                    let entry = self.features.get(feature as usize).ok_or_else(|| {
                        format_error(format!("tree {t}: feature reference out of range"))
                    })?;
                    if threshold as usize >= entry.dict.len() {
                        return Err(format_error(format!(
                            "tree {t}: threshold reference out of range"
                        )));
                    }
                    if let Layout::Preorder { .. } = tree.layout
                        && (index + 1 >= nodes
                            || right < 2
                            || u64::from(index) + u64::from(right) >= u64::from(nodes))
                    {
                        return Err(format_error(format!("tree {t}: child outside the tree")));
                    }
                }
            }
            Ok(())
        };
        match tree.layout {
            Layout::Preorder { nodes } => {
                for i in 0..nodes {
                    check(self.slot(tree, i, false), i, nodes)?;
                }
            }
            Layout::Heap {
                depth,
                complete: true,
            } => {
                // Every internal slot is a split and every bottom slot a leaf,
                // so all are reachable. A zero-width row decodes identically
                // in every slot and costs no input bits, so one check covers
                // it; this keeps validation work bounded by the input size.
                let internal = (1u64 << depth) - 1;
                let splits = if self.widths.split() == 0 {
                    internal.min(1)
                } else {
                    internal
                };
                for i in 0..splits {
                    check(self.slot(tree, i as u32, false), i as u32, 0)?;
                }
                let leaves = if self.widths.leaf_ref == 0 {
                    1
                } else {
                    1u64 << depth
                };
                for j in 0..leaves {
                    let i = (internal + j) as u32;
                    check(self.slot(tree, i, true), i, 0)?;
                }
            }
            Layout::Heap { depth, .. } => {
                // Flagged slots take at least one bit each, and every visited
                // bottom slot is a child of a visited split, so this walk is
                // bounded by the tree's encoded size.
                let first_leaf = (1u64 << depth) - 1;
                let mut stack = vec![0u64];
                while let Some(i) = stack.pop() {
                    let slot = self.slot(tree, i as u32, i >= first_leaf);
                    if let Slot::Split { .. } = slot {
                        stack.extend([2 * i + 1, 2 * i + 2]);
                    }
                    check(slot, i as u32, 0)?;
                }
            }
        }
        Ok(())
    }

    /// Decode slot `i` of `tree`; `leaf_level` marks the heap's bottom row.
    #[inline]
    fn slot(&self, tree: PackedTree, i: u32, leaf_level: bool) -> Slot {
        let w = self.widths;
        let s = &self.bytes;
        let (mut pos, flagged) = match tree.layout {
            Layout::Heap { depth, complete } => {
                let internal = (1usize << depth) - 1;
                let slot = tree.layout.slot_width(w) as usize;
                if leaf_level {
                    let j = i as usize - internal;
                    let pos = tree.offset + internal * slot + j * w.leaf_ref as usize;
                    return Slot::Leaf(read_bits(s, pos, w.leaf_ref));
                }
                (tree.offset + i as usize * slot, !complete)
            }
            Layout::Preorder { .. } => {
                let slot = tree.layout.slot_width(w) as usize;
                (tree.offset + i as usize * slot, true)
            }
        };
        if flagged {
            let is_leaf = read_bits(s, pos, 1) == 1;
            pos += 1;
            if is_leaf {
                return Slot::Leaf(read_bits(s, pos, w.leaf_ref));
            }
        }
        let feature = read_bits(s, pos, w.feature_ref);
        pos += w.feature_ref as usize;
        let threshold = read_bits(s, pos, w.threshold_ref);
        pos += w.threshold_ref as usize;
        let default_left = match self.default_mode {
            DEFAULT_ALL_LEFT => true,
            DEFAULT_ALL_RIGHT => false,
            _ => {
                let bit = read_bits(s, pos, 1) == 1;
                pos += 1;
                bit
            }
        };
        let right = match tree.layout {
            Layout::Preorder { nodes } => read_bits(s, pos, bits(u64::from(nodes) - 1)),
            Layout::Heap { .. } => 0,
        };
        Slot::Split {
            feature,
            threshold,
            default_left,
            right,
        }
    }

    /// Whether `row` (dense, `NaN` = missing) goes left at a split.
    #[inline]
    fn goes_left(&self, feature: u32, threshold: u32, default_left: bool, row: &[f32]) -> bool {
        let entry = &self.features[feature as usize];
        let v = row[entry.input];
        if v.is_nan() {
            return default_left;
        }
        match &entry.dict {
            Dictionary::Numeric(t) => v < t[threshold as usize],
            Dictionary::Categorical(sets) => {
                sets[threshold as usize].binary_search(&(v as u32)).is_ok()
            }
        }
    }

    /// The leaf value tree `t` assigns to `row`.
    fn tree_leaf(&self, t: usize, row: &[f32]) -> f32 {
        let tree = self.trees[t];
        // Heap slots from `first_leaf` on form the bottom (leaf) row; preorder
        // slots ignore the row flag.
        let first_leaf = match tree.layout {
            Layout::Heap { depth, .. } => (1u32 << depth) - 1,
            Layout::Preorder { .. } => u32::MAX,
        };
        let mut i = 0u32;
        let leaf = loop {
            match self.slot(tree, i, i >= first_leaf) {
                Slot::Leaf(leaf) => break leaf,
                Slot::Split {
                    feature,
                    threshold,
                    default_left,
                    right,
                } => {
                    let left = self.goes_left(feature, threshold, default_left, row);
                    i = match tree.layout {
                        Layout::Heap { .. } => 2 * i + if left { 1 } else { 2 },
                        Layout::Preorder { .. } => i + if left { 1 } else { right },
                    };
                }
            }
        };
        self.leaf_values[leaf as usize]
    }

    /// Raw margins `[row][output]` (length `n_rows × n_outputs`),
    /// bit-identical to [`BoostedModel::predict_margin`] of the source model.
    /// A dataset's per-instance `base_margin` overrides the intercepts, as
    /// for [`BoostedModel`].
    pub fn predict_margin(&self, data: &DMatrix) -> Result<Vec<f32>> {
        let k = self.n_outputs();
        validate_prediction_data(self.n_features, k, data)?;
        let mut out = initial_margins(&self.base_score, data);
        let weight = |t: usize| self.tree_weights.as_ref().map_or(1.0, |w| w[t]);
        let parallel = self.meta.num_parallel_tree;
        out.par_chunks_mut(k)
            .enumerate()
            .with_min_len(256)
            .for_each_init(
                || RowBlock::single_rows(data),
                |block, (r, margins)| {
                    block.load(r, 1);
                    let row = block.row(0).expect("single-row blocks are dense");
                    for t in 0..self.trees.len() {
                        margins[scalar_tree_output(t, parallel, k)] +=
                            weight(t) * self.tree_leaf(t, row);
                    }
                },
            );
        Ok(out)
    }

    /// Predictions in the objective's reported space, identical to
    /// [`BoostedModel::predict`] of the source model (probabilities for
    /// logistic objectives, class indices for `multi:softmax`, ...).
    pub fn predict(&self, data: &DMatrix) -> Result<Vec<f32>> {
        let margin = self.predict_margin(data)?;
        Ok(transform_model_margins(
            &self.meta.objective,
            &self.meta.objective_params(),
            self.meta.num_class,
            self.meta.n_targets,
            self.n_outputs(),
            margin,
        ))
    }

    /// The serialized model (the exact bytes it was parsed from).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.serialized().to_vec()
    }

    /// Serialized size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.serialized().len()
    }

    /// Save the serialized model to a file.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.serialized())?;
        Ok(())
    }

    /// Load a model saved with [`CompactModel::save`].
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// The objective name that drives [`CompactModel::predict`].
    pub fn objective(&self) -> &str {
        &self.meta.objective
    }

    /// Number of stored trees.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of input features the model expects.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Raw outputs per row.
    pub fn n_outputs(&self) -> usize {
        self.base_score.len()
    }

    /// Input indices of the features the trees split on (`F_U`), ascending.
    pub fn used_features(&self) -> Vec<usize> {
        self.features.iter().map(|e| e.input).collect()
    }

    /// Distinct thresholds (and categorical sets) across all features,
    /// `Σ_f |T^f|`.
    pub fn num_thresholds(&self) -> usize {
        self.features.iter().map(|e| e.dict.len()).sum()
    }

    /// Distinct leaf values in the global leaf table.
    pub fn num_leaf_values(&self) -> usize {
        self.leaf_values.len()
    }
}

/// Sizes of a model in the native and compact formats plus the dictionary
/// statistics the *Trees on a Diet* paper reports. From
/// [`BoostedModel::size_report`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelSizeReport {
    /// [`BoostedModel::to_bytes`] length (all trees, with covers and gains).
    pub native_bytes: usize,
    /// [`BoostedModel::to_compact_bytes`] length.
    pub compact_bytes: usize,
    /// Trees in the compact model (those prediction uses).
    pub trees: usize,
    /// Internal (split) nodes across those trees.
    pub splits: usize,
    /// Leaves across those trees.
    pub leaves: usize,
    /// Features the trees split on (`|F_U|`).
    pub used_features: usize,
    /// Distinct thresholds and categorical sets (`Σ_f |T^f|`).
    pub thresholds: usize,
    /// Distinct leaf values.
    pub leaf_values: usize,
}

impl ModelSizeReport {
    /// `native_bytes / compact_bytes`.
    pub fn compression_ratio(&self) -> f64 {
        self.native_bytes as f64 / self.compact_bytes as f64
    }

    /// The paper's reuse factor `ReF`: nodes and leaves per stored global
    /// value (`(splits + leaves) / (thresholds + leaf_values)`); `1` means no
    /// value is shared.
    pub fn reuse_factor(&self) -> f64 {
        (self.splits + self.leaves) as f64 / (self.thresholds + self.leaf_values).max(1) as f64
    }
}

impl BoostedModel {
    /// This model in the bit-packed *Trees on a Diet* layout (see
    /// [`crate::model::compact`]), predicting bit-identical margins
    /// from the trees [`BoostedModel::predict_margin`] uses. Fails for
    /// gblinear, linear-leaf and vector-leaf models and for trees the format
    /// cannot express (a feature split both numerically and categorically).
    pub fn to_compact(&self) -> Result<CompactModel> {
        CompactModel::from_bytes(&encode(self)?)
    }

    /// Serialize this model in the compact layout; parse the bytes with
    /// [`CompactModel::from_bytes`].
    pub fn to_compact_bytes(&self) -> Result<Vec<u8>> {
        encode(self)
    }

    /// Native versus compact size of this model plus its dictionary
    /// statistics.
    pub fn size_report(&self) -> Result<ModelSizeReport> {
        let compact = self.to_compact()?;
        let trees = &self.trees()[..compact.num_trees()];
        let splits: usize = trees
            .iter()
            .map(|t| t.nodes().iter().filter(|n| !n.is_leaf()).count())
            .sum();
        let nodes: usize = trees.iter().map(RegTree::num_nodes).sum();
        Ok(ModelSizeReport {
            native_bytes: self.to_bytes()?.len(),
            compact_bytes: compact.size_bytes(),
            trees: trees.len(),
            splits,
            leaves: nodes - splits,
            used_features: compact.features.len(),
            thresholds: compact.num_thresholds(),
            leaf_values: compact.num_leaf_values(),
        })
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Dictionary under construction for one used feature.
enum Collected {
    Numeric(BTreeSet<u32>),
    Categorical(BTreeSet<Vec<u32>>),
}

/// A feature map entry under construction: encoding and sorted dictionary.
struct MapEntry {
    input: u32,
    kind: u32,
    code: u32,
    len_bits: u32,
    dict: Dictionary,
}

/// `cats` sorted and deduplicated into `scratch`: the set identity under
/// which the dictionaries store categorical splits
/// ([`canonical_categories`](crate::tree::reuse::canonical_categories)).
fn canonical<'s>(cats: &[u32], scratch: &'s mut Vec<u32>) -> &'s [u32] {
    scratch.clear();
    scratch.extend_from_slice(cats);
    scratch.sort_unstable();
    scratch.dedup();
    scratch
}

/// Depth of the deepest leaf (root = 0) and whether every leaf sits there;
/// `stack` is scratch.
fn depth_and_completeness(tree: &RegTree, stack: &mut Vec<(usize, u32)>) -> (u32, bool) {
    stack.clear();
    stack.push((0, 0));
    let (mut shallowest, mut deepest) = (u32::MAX, 0);
    while let Some((id, d)) = stack.pop() {
        let node = tree.node(id);
        if node.is_leaf() {
            shallowest = shallowest.min(d);
            deepest = deepest.max(d);
        } else {
            stack.push((node.left as usize, d + 1));
            stack.push((node.right as usize, d + 1));
        }
    }
    (deepest, shallowest >= deepest)
}

/// Node ids in preorder (node, left subtree, right subtree) into `order`;
/// `stack` is scratch.
fn preorder(tree: &RegTree, order: &mut Vec<usize>, stack: &mut Vec<usize>) {
    order.clear();
    stack.clear();
    stack.push(0);
    while let Some(id) = stack.pop() {
        order.push(id);
        let node = tree.node(id);
        if !node.is_leaf() {
            stack.push(node.right as usize);
            stack.push(node.left as usize);
        }
    }
}

/// Refuse the models the compact layout cannot express.
fn check_encodable(model: &BoostedModel) -> Result<()> {
    if model.linear().is_some() {
        return Err(format_error(
            "gblinear models have no trees to store; use the native format",
        ));
    }
    if model
        .trees()
        .iter()
        .any(|tree| tree.linear_leaves().is_some())
    {
        return Err(format_error(
            "linear-leaf trees (`linear_tree`) have no compact encoding; use the native format",
        ));
    }
    if model.has_vector_leaves() {
        return Err(format_error(
            "vector-leaf trees (`multi_output_tree`) have no compact encoding; use the native format",
        ));
    }
    Ok(())
}

/// The dictionaries of `trees`: each used feature's thresholds or sets, the
/// distinct leaf values in first-seen order with their index, and the
/// default-direction mode.
struct Collection {
    features: BTreeMap<u32, Collected>,
    leaf_index: HashMap<u32, u32>,
    leaf_values: Vec<f32>,
    default_mode: u32,
}

fn collect(trees: &[RegTree]) -> Result<Collection> {
    let mut features: BTreeMap<u32, Collected> = BTreeMap::new();
    let mut leaf_index: HashMap<u32, u32> = HashMap::new();
    let mut leaf_values: Vec<f32> = Vec::new();
    let (mut any_left, mut any_right) = (false, false);
    let mut cats = Vec::new();
    for tree in trees {
        for node in tree.nodes() {
            if node.is_leaf() {
                leaf_index
                    .entry(node.leaf_value.to_bits())
                    .or_insert_with(|| {
                        leaf_values.push(node.leaf_value);
                        (leaf_values.len() - 1) as u32
                    });
                continue;
            }
            any_left |= node.default_left;
            any_right |= !node.default_left;
            let entry = features.entry(node.split_feature).or_insert_with(|| {
                if node.is_categorical {
                    Collected::Categorical(BTreeSet::new())
                } else {
                    Collected::Numeric(BTreeSet::new())
                }
            });
            match (entry, node.is_categorical) {
                (Collected::Numeric(set), false) => {
                    set.insert(node.split_cond.to_bits());
                }
                (Collected::Categorical(sets), true) => {
                    let set = canonical(tree.node_categories(node), &mut cats);
                    if !sets.contains(set) {
                        sets.insert(set.to_vec());
                    }
                }
                _ => {
                    return Err(format_error(format!(
                        "feature {} has both numeric and categorical splits",
                        node.split_feature
                    )));
                }
            }
        }
    }
    let default_mode = match (any_left, any_right) {
        (true, true) => DEFAULT_PER_NODE,
        (false, true) => DEFAULT_ALL_RIGHT,
        _ => DEFAULT_ALL_LEFT,
    };
    Ok(Collection {
        features,
        leaf_index,
        leaf_values,
        default_mode,
    })
}

/// Feature map entries in ascending input order, with sorted dictionaries
/// and their narrowest encodings.
fn feature_map(features: BTreeMap<u32, Collected>) -> Vec<MapEntry> {
    features
        .into_iter()
        .map(|(input, dict)| match dict {
            Collected::Numeric(set) => {
                let mut values: Vec<f32> = set.iter().map(|&b| f32::from_bits(b)).collect();
                values.sort_by(f32::total_cmp);
                let (kind, code) = numeric_encoding(&values);
                MapEntry {
                    input,
                    kind,
                    code,
                    len_bits: 0,
                    dict: Dictionary::Numeric(values),
                }
            }
            Collected::Categorical(sets) => {
                let sets: Vec<Vec<u32>> = sets.into_iter().collect();
                let max_cat = sets.iter().flatten().copied().max().unwrap_or(0);
                let code = (0..=5u32)
                    .find(|&c| bits(u64::from(max_cat)) <= 1 << c)
                    .unwrap_or(5);
                let max_len = sets.iter().map(Vec::len).max().unwrap_or(0);
                MapEntry {
                    input,
                    kind: KIND_CATEGORICAL,
                    code,
                    len_bits: bits(max_len as u64).max(1),
                    dict: Dictionary::Categorical(sets),
                }
            }
        })
        .collect()
}

/// The smaller layout of each tree (heap only up to [`MAX_HEAP_DEPTH`]),
/// and the widths of the heap depths and preorder node counts.
fn choose_layouts(trees: &[RegTree], widths: Widths) -> (Vec<Layout>, u32, u32) {
    let mut stack = Vec::new();
    let layouts: Vec<Layout> = trees
        .iter()
        .map(|tree| {
            let preorder = Layout::Preorder {
                nodes: tree.num_nodes() as u32,
            };
            let (depth, complete) = depth_and_completeness(tree, &mut stack);
            let heap = Layout::Heap { depth, complete };
            if depth <= MAX_HEAP_DEPTH && heap.total_bits(widths) <= preorder.total_bits(widths) {
                heap
            } else {
                preorder
            }
        })
        .collect();
    let (mut heap_depth_bits, mut preorder_nodes_bits) = (0, 0);
    for layout in &layouts {
        match *layout {
            Layout::Heap { depth, .. } => {
                heap_depth_bits = heap_depth_bits.max(bits(u64::from(depth)));
            }
            Layout::Preorder { nodes } => {
                preorder_nodes_bits = preorder_nodes_bits.max(bits(u64::from(nodes) - 1));
            }
        }
    }
    (layouts, heap_depth_bits, preorder_nodes_bits)
}

/// Everything [`encode`] derives from the model before writing the bit
/// stream: the stored trees, their dictionaries and field widths, and each
/// tree's layout.
struct Encoding<'a> {
    model: &'a BoostedModel,
    trees: &'a [RegTree],
    map: Vec<MapEntry>,
    leaf_index: HashMap<u32, u32>,
    leaf_values: Vec<f32>,
    default_mode: u32,
    max_thresholds: usize,
    widths: Widths,
    layouts: Vec<Layout>,
    heap_depth_bits: u32,
    preorder_nodes_bits: u32,
}

/// Per-tree buffers the tree writers reuse.
#[derive(Default)]
struct TreeScratch {
    heap: Vec<Option<usize>>,
    order: Vec<usize>,
    stack: Vec<usize>,
    position: Vec<u32>,
    cats: Vec<u32>,
}

impl<'a> Encoding<'a> {
    fn plan(model: &'a BoostedModel) -> Result<Self> {
        check_encodable(model)?;
        let trees = &model.trees()[..model.effective_num_trees()];
        let Collection {
            features,
            leaf_index,
            leaf_values,
            default_mode,
        } = collect(trees)?;
        let map = feature_map(features);
        let max_thresholds = map.iter().map(|e| e.dict.len()).max().unwrap_or(0);
        let widths = Widths::new(map.len(), max_thresholds, leaf_values.len(), default_mode);
        if u32::try_from(max_thresholds).is_err()
            || u32::try_from(leaf_values.len()).is_err()
            || u32::try_from(trees.len()).is_err()
            || u32::try_from(model.n_features()).is_err()
        {
            return Err(format_error("model too large for 32-bit counts"));
        }
        let (layouts, heap_depth_bits, preorder_nodes_bits) = choose_layouts(trees, widths);
        Ok(Encoding {
            model,
            trees,
            map,
            leaf_index,
            leaf_values,
            default_mode,
            max_thresholds,
            widths,
            layouts,
            heap_depth_bits,
            preorder_nodes_bits,
        })
    }

    /// Layout items 1 to 4: the metadata fields, the feature map, the
    /// dictionaries, and the leaf values.
    fn write_tables(&self, w: &mut BitWriter) {
        let model = self.model;
        let n_features = model.n_features();
        w.write(n_features as u64, 32);
        w.write(model.n_outputs() as u64, 32);
        for &b in model.base_scores() {
            w.write_f32(b);
        }
        let n_trees = self.trees.len();
        w.write(n_trees as u64, 32);
        let weighted = (0..n_trees).any(|t| model.tree_weight(t).to_bits() != 1.0f32.to_bits());
        w.write_bool(weighted);
        if weighted {
            for t in 0..n_trees {
                w.write_f32(model.tree_weight(t));
            }
        }
        w.write(u64::from(self.default_mode), 2);
        w.write(self.map.len() as u64, 32);
        w.write(self.max_thresholds as u64, 32);
        w.write(self.leaf_values.len() as u64, 32);
        w.write(u64::from(self.heap_depth_bits), 6);
        w.write(u64::from(self.preorder_nodes_bits), 6);

        let input_bits = bits(n_features as u64 - 1);
        for e in &self.map {
            w.write(u64::from(e.input), input_bits);
            w.write(u64::from(e.kind), 2);
            w.write(u64::from(e.code), 3);
            w.write(e.dict.len() as u64 - 1, self.widths.threshold_ref);
            if e.kind == KIND_CATEGORICAL {
                w.write(u64::from(e.len_bits), 6);
            }
        }
        for e in &self.map {
            let width = 1u32 << e.code;
            match &e.dict {
                Dictionary::Numeric(values) => {
                    for &v in values {
                        w.write(encode_threshold(v, e.kind, width), width);
                    }
                }
                Dictionary::Categorical(sets) => {
                    for set in sets {
                        w.write(set.len() as u64, e.len_bits);
                        for &c in set {
                            w.write(u64::from(c), width);
                        }
                    }
                }
            }
        }
        for &v in &self.leaf_values {
            w.write_f32(v);
        }
    }

    /// Layout item 5: every tree in its chosen layout.
    fn write_trees(&self, w: &mut BitWriter) {
        let mut scratch = TreeScratch::default();
        for (tree, &layout) in self.trees.iter().zip(&self.layouts) {
            match layout {
                Layout::Heap { depth, complete } => {
                    self.write_heap_tree(w, tree, depth, complete, &mut scratch);
                }
                Layout::Preorder { nodes } => {
                    self.write_preorder_tree(w, tree, nodes, &mut scratch);
                }
            }
        }
    }

    /// A leaf's reference, or an internal node's split fields.
    fn write_node(&self, w: &mut BitWriter, tree: &RegTree, node: &Node, cats: &mut Vec<u32>) {
        let widths = self.widths;
        if node.is_leaf() {
            w.write(
                u64::from(self.leaf_index[&node.leaf_value.to_bits()]),
                widths.leaf_ref,
            );
            return;
        }
        let feature = self
            .map
            .binary_search_by_key(&node.split_feature, |e| e.input)
            .expect("every split feature has a map entry");
        w.write(feature as u64, widths.feature_ref);
        let threshold = match &self.map[feature].dict {
            Dictionary::Numeric(values) => {
                values.binary_search_by(|v| v.total_cmp(&node.split_cond))
            }
            Dictionary::Categorical(sets) => {
                let set = canonical(tree.node_categories(node), cats);
                sets.binary_search_by(|s| s.as_slice().cmp(set))
            }
        }
        .expect("every split threshold is in its feature's dictionary");
        w.write(threshold as u64, widths.threshold_ref);
        if widths.default_bit == 1 {
            w.write_bool(node.default_left);
        }
    }

    /// A tree in the heap layout of `depth`: slot `i` holds node `heap[i]`,
    /// every slot padded to its fixed width.
    fn write_heap_tree(
        &self,
        w: &mut BitWriter,
        tree: &RegTree,
        depth: u32,
        complete: bool,
        scratch: &mut TreeScratch,
    ) {
        let slot_width = Layout::Heap { depth, complete }.slot_width(self.widths);
        w.write_bool(false);
        w.write(u64::from(depth), self.heap_depth_bits);
        w.write_bool(complete);
        let internal = (1usize << depth) - 1;
        let heap = &mut scratch.heap;
        heap.clear();
        heap.resize(2 * internal + 1, None);
        heap[0] = Some(0);
        for i in 0..internal {
            if let Some(id) = heap[i] {
                let node = tree.node(id);
                if !node.is_leaf() {
                    heap[2 * i + 1] = Some(node.left as usize);
                    heap[2 * i + 2] = Some(node.right as usize);
                }
            }
        }
        for (i, slot) in heap.iter().enumerate() {
            let start = w.len;
            if let Some(id) = *slot {
                let node = tree.node(id);
                if i < internal && !complete {
                    w.write_bool(node.is_leaf());
                }
                self.write_node(w, tree, node, &mut scratch.cats);
            }
            let width = if i < internal {
                slot_width
            } else {
                self.widths.leaf_ref
            };
            w.write(0, width - (w.len - start) as u32);
        }
    }

    /// A tree in the preorder layout of `nodes` nodes: each split stores the
    /// distance to its right child.
    fn write_preorder_tree(
        &self,
        w: &mut BitWriter,
        tree: &RegTree,
        nodes: u32,
        scratch: &mut TreeScratch,
    ) {
        let slot_width = Layout::Preorder { nodes }.slot_width(self.widths);
        w.write_bool(true);
        w.write(u64::from(nodes) - 1, self.preorder_nodes_bits);
        preorder(tree, &mut scratch.order, &mut scratch.stack);
        let position = &mut scratch.position;
        position.clear();
        position.resize(tree.num_nodes(), 0);
        for (p, &id) in scratch.order.iter().enumerate() {
            position[id] = p as u32;
        }
        let offset_bits = bits(u64::from(nodes) - 1);
        for (p, &id) in scratch.order.iter().enumerate() {
            let start = w.len;
            let node = tree.node(id);
            w.write_bool(node.is_leaf());
            self.write_node(w, tree, node, &mut scratch.cats);
            if !node.is_leaf() {
                let right = position[node.right as usize] - p as u32;
                w.write(u64::from(right), offset_bits);
            }
            w.write(0, slot_width - (w.len - start) as u32);
        }
    }

    /// The compact metadata section table.
    fn meta(&self) -> Meta {
        let model = self.model;
        Meta {
            objective: model.objective().to_string(),
            objective_params: (*model.objective_params()
                != ObjectiveParams::defaults_for(model.objective()))
            .then(|| model.objective_params().clone()),
            num_class: model.num_class(),
            n_targets: model.n_targets(),
            num_parallel_tree: model.num_parallel_tree(),
        }
    }
}

/// Serialize `model` in the compact layout.
fn encode(model: &BoostedModel) -> Result<Vec<u8>> {
    let encoding = Encoding::plan(model)?;
    let mut w = BitWriter::default();
    encoding.write_tables(&mut w);
    encoding.write_trees(&mut w);
    Ok(frame(&encoding.meta(), &w.bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BoosterKind, GrowPolicy, TrainingParams, TreeMethod};
    use crate::data::FeatureType;
    use crate::test_support::labeled_dense;
    use crate::training::{Trainer, train};

    /// Deterministic pseudo-random value in `[0, 1)`.
    fn noise(i: usize) -> f32 {
        let mut z = (i as u64).wrapping_mul(crate::rng::GOLDEN);
        z ^= z >> 31;
        z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// `n × f` features with ~10% missing values and a nonlinear target.
    fn dataset(n: usize, f: usize, missing: bool) -> (Vec<f32>, Vec<f32>) {
        let mut x: Vec<f32> = (0..n * f).map(|i| noise(i) * 10.0 - 3.0).collect();
        if missing {
            for (i, v) in x.iter_mut().enumerate() {
                if noise(i + 7_777_777) < 0.1 {
                    *v = f32::NAN;
                }
            }
        }
        let y = x
            .chunks(f)
            .map(|r| {
                let a = if r[0].is_nan() { 1.0 } else { r[0] };
                let b = if r[1].is_nan() { -1.0 } else { r[1] };
                (a * b).sin() + 0.3 * a
            })
            .collect();
        (x, y)
    }

    /// [`dataset`] as a labeled dense matrix.
    fn labeled(n: usize, f: usize, missing: bool) -> DMatrix {
        let (x, y) = dataset(n, f, missing);
        labeled_dense(&x, n, f, &y)
    }

    fn assert_bit_identical(model: &BoostedModel, data: &DMatrix) -> CompactModel {
        let compact = CompactModel::from_bytes(&model.to_compact_bytes().unwrap()).unwrap();
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(
            bits(compact.predict_margin(data).unwrap()),
            bits(model.predict_margin(data).unwrap())
        );
        assert_eq!(
            bits(compact.predict(data).unwrap()),
            bits(model.predict(data).unwrap())
        );
        compact
    }

    #[test]
    fn round_trip_is_bit_identical_across_model_kinds() {
        let (x, y) = dataset(600, 5, true);
        let data = labeled_dense(&x, 600, 5, &y);
        let labels01: Vec<f32> = y.iter().map(|&v| f32::from(v > 0.5)).collect();
        let labels3: Vec<f32> = y
            .iter()
            .map(|&v| ((v + 1.5).clamp(0.0, 2.9)) as u32 as f32)
            .collect();
        let binary = data.clone().with_labels(&labels01).unwrap();
        let multi = data.clone().with_labels(&labels3).unwrap();
        let matrix: Vec<f32> = y.iter().flat_map(|&v| [v, 1.0 - 2.0 * v]).collect();
        let targets2 = data.clone().with_label_matrix(&matrix, 2).unwrap();

        let base = || TrainingParams::builder().max_depth(4).eta(0.2);
        let cases: Vec<(TrainingParams, &DMatrix)> = vec![
            (base().build().unwrap(), &data),
            (
                base()
                    .tree_method(TreeMethod::Exact)
                    .objective("binary:logistic")
                    .build()
                    .unwrap(),
                &binary,
            ),
            (
                base()
                    .tree_method(TreeMethod::Approx)
                    .objective("multi:softprob")
                    .num_class(3)
                    .build()
                    .unwrap(),
                &multi,
            ),
            (
                base()
                    .objective("multi:softmax")
                    .num_class(3)
                    .build()
                    .unwrap(),
                &multi,
            ),
            (
                base()
                    .booster(BoosterKind::Dart)
                    .rate_drop(0.3)
                    .build()
                    .unwrap(),
                &data,
            ),
            (
                base()
                    .toad_penalty_feature(2.0)
                    .toad_penalty_threshold(0.5)
                    .build()
                    .unwrap(),
                &data,
            ),
            // Iteration-major forests: tree `t` feeds output
            // `(t / num_parallel_tree) % n_outputs`.
            (
                base()
                    .objective("multi:softprob")
                    .num_class(3)
                    .num_parallel_tree(2)
                    .build()
                    .unwrap(),
                &multi,
            ),
            (base().num_parallel_tree(2).build().unwrap(), &targets2),
            (
                base()
                    .objective("reg:quantileerror")
                    .quantile_alpha(vec![0.2, 0.8])
                    .build()
                    .unwrap(),
                &data,
            ),
        ];
        for (params, d) in cases {
            let model = train(&params, d, 15).unwrap();
            assert_bit_identical(&model, d);
        }
    }

    #[test]
    fn deep_unbalanced_trees_use_the_preorder_layout() {
        let data = labeled(800, 4, true);
        let params = TrainingParams::builder()
            .grow_policy(GrowPolicy::LossGuide)
            .max_depth(0)
            .max_leaves(48)
            .build()
            .unwrap();
        let model = train(&params, &data, 5).unwrap();
        let compact = assert_bit_identical(&model, &data);
        assert!(
            compact
                .trees
                .iter()
                .any(|t| matches!(t.layout, Layout::Preorder { .. })),
            "the case must exercise the preorder layout"
        );
    }

    #[test]
    fn categorical_csr_and_base_margin_inputs_match() {
        let n = 500;
        let mut x = Vec::with_capacity(n * 3);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let cat = (noise(i) * 9.0) as u32 as f32;
            let v = noise(i + 99_999) * 4.0;
            x.extend_from_slice(&[cat, v, noise(i + 5) * 2.0]);
            y.push(
                if [1.0, 4.0, 6.0].contains(&cat) {
                    2.0
                } else {
                    0.0
                } + v,
            );
        }
        let data = DMatrix::from_dense(&x, n, 3)
            .unwrap()
            .with_feature_types(&[
                FeatureType::Categorical,
                FeatureType::Numerical,
                FeatureType::Numerical,
            ])
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder().max_depth(3).build().unwrap();
        let model = train(&params, &data, 10).unwrap();
        assert!(
            model
                .trees()
                .iter()
                .any(|t| t.nodes().iter().any(|n| n.is_categorical))
        );
        assert_bit_identical(&model, &data);

        // The same rows as CSR (dropping feature 2 when it is small) and with
        // per-row base margins.
        let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
        for r in x.chunks(3) {
            for (f, &v) in r.iter().enumerate() {
                if f != 2 || v > 0.5 {
                    indices.push(f as u32);
                    values.push(v);
                }
            }
            indptr.push(indices.len());
        }
        let csr = DMatrix::from_csr(indptr, indices, values, 3)
            .unwrap()
            .with_base_margin(&y.iter().map(|v| v * 0.1).collect::<Vec<_>>())
            .unwrap();
        assert_bit_identical(&model, &csr);
    }

    #[test]
    fn only_the_early_stopped_prefix_is_stored() {
        let data = labeled(400, 3, false);
        let (xv, yv) = dataset(200, 3, false);
        let yv: Vec<f32> = yv.iter().map(|v| -v).collect();
        let valid = labeled_dense(&xv, 200, 3, &yv);
        let params = TrainingParams::builder().build().unwrap();
        let model = Trainer::new(&params, &data, 50)
            .eval(&valid, "valid")
            .early_stopping_rounds(3)
            .train()
            .unwrap()
            .model;
        let best = model.best_iteration().expect("early stopping triggers");
        let compact = assert_bit_identical(&model, &data);
        assert_eq!(compact.num_trees(), best + 1);
    }

    #[test]
    fn thresholds_take_the_narrowest_exact_encoding() {
        let cases: [(&[f32], u32, u32); 7] = [
            (&[0.0, 1.0], KIND_UINT, 0),
            (&[3.0, 200.0], KIND_UINT, 3),
            (&[-2.0, 1.0], KIND_SINT, 1),
            (&[-3.0, 1.0], KIND_SINT, 2),
            (&[0.5, -0.0, 100.25], KIND_FLOAT, 4),
            (&[0.1], KIND_FLOAT, 5),
            (&[f32::MIN, 3.0], KIND_FLOAT, 5),
        ];
        for (values, kind, code) in cases {
            assert_eq!(numeric_encoding(values), (kind, code), "{values:?}");
            let width = 1 << code;
            for &v in values {
                let raw = encode_threshold(v, kind, width) as u32;
                assert_eq!(
                    decode_threshold(raw, kind, width).unwrap().to_bits(),
                    v.to_bits()
                );
            }
        }
        // binary16 subnormals and the largest finite value are exact.
        for v in [
            f32::from_bits(0x3380_0000),
            3.0 * f32::from_bits(0x3380_0000),
            65504.0,
        ] {
            assert_eq!(f16_to_f32(f16_exact(v).unwrap()).to_bits(), v.to_bits());
        }
        assert!(f16_exact(65520.0).is_none());
        assert!(f16_exact(1.0 + f32::EPSILON).is_none());
    }

    #[test]
    fn compact_is_far_smaller_than_native() {
        let data = labeled(2000, 8, false);
        let params = TrainingParams::builder().max_depth(4).build().unwrap();
        let model = train(&params, &data, 100).unwrap();
        let report = model.size_report().unwrap();
        assert_eq!(report.trees, 100);
        assert_eq!(
            report.compact_bytes,
            model.to_compact_bytes().unwrap().len()
        );
        // The native format is zstd-compressed; the bit-packed layout still
        // beats it clearly (about 2.2x here with libzstd's default level).
        assert!(
            report.compression_ratio() > 1.5,
            "compact {} vs native {}",
            report.compact_bytes,
            report.native_bytes
        );
        assert!(report.reuse_factor() >= 1.0);
    }

    #[test]
    fn corrupt_input_is_rejected_without_panicking() {
        let data = labeled(300, 4, true);
        let params = TrainingParams::builder().max_depth(3).build().unwrap();
        let model = train(&params, &data, 5).unwrap();
        let bytes = model.to_compact_bytes().unwrap();
        for len in 0..bytes.len() {
            assert!(
                CompactModel::from_bytes(&bytes[..len]).is_err(),
                "prefix {len}"
            );
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(CompactModel::from_bytes(&longer).is_err());
        // Single bit flips either fail to parse or give a model that predicts.
        for i in 0..bytes.len() * 8 {
            let mut flipped = bytes.clone();
            flipped[i / 8] ^= 1 << (i % 8);
            if let Ok(m) = CompactModel::from_bytes(&flipped) {
                let _ = m.predict_margin(&data);
            }
        }
    }

    /// `bytes` with its metadata replaced by `edit` applied to the decoded
    /// metadata.
    fn with_meta(bytes: &[u8], edit: impl FnOnce(&mut Meta)) -> Vec<u8> {
        let (meta, stream) = split_frame(bytes).unwrap();
        let mut meta = Meta::decode(meta).unwrap();
        edit(&mut meta);
        frame(&meta, stream)
    }

    #[test]
    fn inconsistent_layout_metadata_is_rejected() {
        let (x, y) = dataset(200, 3, false);
        let data = DMatrix::from_dense(&x, 200, 3)
            .unwrap()
            .with_label_matrix(&[y.clone(), y].concat(), 2)
            .unwrap();
        let params = TrainingParams::builder().max_depth(2).build().unwrap();
        let bytes = train(&params, &data, 0)
            .unwrap()
            .to_compact_bytes()
            .unwrap();
        assert!(CompactModel::from_bytes(&with_meta(&bytes, |_| ())).is_ok());
        // Two outputs × 2^63 parallel trees overflows the trees per iteration.
        let overflow = with_meta(&bytes, |m| m.num_parallel_tree = 1 << 63);
        assert!(matches!(
            CompactModel::from_bytes(&overflow),
            Err(HessboostError::ModelFormat(_))
        ));
        // A three-alpha objective cannot describe a two-output layout.
        let widened = with_meta(&bytes, |m| {
            m.objective = "reg:quantileerror".to_string();
            m.n_targets = 1;
            let mut params = ObjectiveParams::defaults_for("reg:quantileerror");
            params.quantile_alpha = vec![0.1, 0.5, 0.9];
            m.objective_params = Some(params);
        });
        assert!(matches!(
            CompactModel::from_bytes(&widened),
            Err(HessboostError::ModelFormat(_))
        ));
    }

    /// `n_trees` complete depth-24 heaps whose split and leaf references are
    /// zero bits wide: each tree costs only its seven header bits, so the
    /// whole model is a few KB although it names 2^25 slots per tree.
    fn zero_width_heaps(n_trees: usize, n_leaves: u64) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write(1, 32); // n_features
        w.write(1, 32); // n_outputs
        w.write_f32(0.5); // base score
        w.write(n_trees as u64, 32);
        w.write_bool(false); // no tree weights
        w.write(u64::from(DEFAULT_ALL_LEFT), 2);
        w.write(1, 32); // used features
        w.write(1, 32); // max thresholds
        w.write(n_leaves, 32);
        w.write(5, 6); // heap depth bits
        w.write(0, 6); // preorder node-count bits
        w.write(u64::from(KIND_UINT), 2);
        w.write(0, 3); // one-bit thresholds
        w.write(1, 1); // threshold 1
        for _ in 0..n_leaves {
            w.write_f32(1.0);
        }
        for _ in 0..n_trees {
            w.write_bool(false); // heap
            w.write(24, 5);
            w.write_bool(true); // complete
        }
        let meta = Meta {
            objective: "reg:squarederror".to_string(),
            objective_params: None,
            num_class: 0,
            n_targets: 1,
            num_parallel_tree: 1,
        };
        frame(&meta, &w.bytes)
    }

    /// Validation work is bounded by the input: 4096 zero-width depth-24
    /// heaps (under 4 KB) once cost 2^25 slot checks each.
    #[test]
    fn zero_width_heaps_validate_in_bounded_time() {
        let bytes = zero_width_heaps(4096, 1);
        assert!(bytes.len() < 4096);
        let model = CompactModel::from_bytes(&bytes).unwrap();
        let data = DMatrix::from_dense(&[0.0], 1, 1).unwrap();
        assert_eq!(model.predict_margin(&data).unwrap(), [4096.5]);
        // The shared zero-width leaf reference is still range-checked.
        assert!(matches!(
            CompactModel::from_bytes(&zero_width_heaps(4096, 0)),
            Err(HessboostError::ModelFormat(_))
        ));
    }

    /// gblinear models and linear leaves have no compact encoding: refused,
    /// never flattened to their constant fallback.
    #[test]
    fn gblinear_and_linear_leaf_models_are_rejected() {
        let data = labeled(200, 3, false);
        for params in [
            TrainingParams::builder().booster(BoosterKind::GbLinear),
            TrainingParams::builder().max_depth(3).linear_tree(true),
        ] {
            let model = train(&params.build().unwrap(), &data, 3).unwrap();
            assert!(matches!(
                model.to_compact_bytes(),
                Err(HessboostError::ModelFormat(_))
            ));
        }
    }
}
