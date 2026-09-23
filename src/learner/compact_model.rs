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
//! `best_iteration` when early stopping chose one). gblinear models are
//! rejected. It is a hessboost format; XGBoost cannot read it.
//!
//! # Layout (version 1)
//!
//! ```text
//! bytes 0..4   magic b"HBTD"
//! byte  4      format version (1)
//! bytes 5..9   u32 little-endian M, the metadata length
//! next M bytes metadata (postcard): objective name, objective parameters
//!              (absent when they equal XGBoost's defaults for the objective),
//!              num_class, n_targets
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
//!    `t % n_outputs`). A layout bit selects:
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

use crate::config::ObjectiveParams;
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::learner::model::{
    BoostedModel, RowBlock, initial_margins, rebuild_objective, transform_margins,
    validate_prediction_data,
};
use crate::tree::RegTree;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
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

/// Metadata the transform and dimension checks need, stored as postcard.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Meta {
    objective: String,
    /// `None` when equal to [`ObjectiveParams::defaults_for`] the objective.
    objective_params: Option<ObjectiveParams>,
    num_class: usize,
    n_targets: usize,
}

impl Meta {
    fn objective_params(&self) -> ObjectiveParams {
        self.objective_params
            .clone()
            .unwrap_or_else(|| ObjectiveParams::defaults_for(&self.objective))
    }
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
    fn write(&mut self, value: u64, width: u32) {
        debug_assert!(
            width == 64 || value >> width == 0,
            "value exceeds its field"
        );
        for j in 0..width {
            if self.len.is_multiple_of(8) {
                self.bytes.push(0);
            }
            if (value >> j) & 1 == 1 {
                self.bytes[self.len / 8] |= 1 << (self.len % 8);
            }
            self.len += 1;
        }
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

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read(32)?))
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
    /// Stream bit offset of the tree's first slot.
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
    /// The serialized form, as [`CompactModel::to_bytes`] returns it.
    bytes: Vec<u8>,
    /// The bit stream followed by [`STREAM_PAD`] zero bytes.
    stream: Vec<u8>,
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

impl CompactModel {
    /// Encode the trees [`BoostedModel::predict_margin`] uses. Fails for
    /// gblinear models and for trees the format cannot express (a feature
    /// split both numerically and categorically).
    fn from_model(model: &BoostedModel) -> Result<Self> {
        Self::from_bytes(&encode(model)?)
    }

    /// Parse bytes written by [`CompactModel::to_bytes`] /
    /// [`BoostedModel::to_compact_bytes`]. Every reference is validated, so
    /// prediction on a parsed model cannot index out of bounds.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
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
        let meta: Meta = postcard::from_bytes(&bytes[PREFIX_BYTES..meta_end])
            .map_err(|e| format_error(format!("metadata: {e}")))?;
        if meta.n_targets == 0 {
            return Err(format_error("n_targets must be positive"));
        }

        let mut stream = bytes[meta_end..].to_vec();
        let len = stream.len() * 8;
        stream.resize(stream.len() + STREAM_PAD, 0);
        let mut r = BitReader {
            stream: &stream,
            len,
            pos: 0,
        };

        let n_features = r.read_usize(32)?;
        let n_outputs = r.read_usize(32)?;
        if n_features == 0 || n_outputs == 0 {
            return Err(format_error("feature and output counts must be positive"));
        }
        if meta.num_class >= 2 && n_outputs != meta.num_class {
            return Err(format_error("output count does not match num_class"));
        }
        r.ensure_fits(n_outputs, 32, "base score")?;
        let base_score = (0..n_outputs)
            .map(|_| r.read_f32())
            .collect::<Result<Vec<_>>>()?;
        if base_score.iter().any(|v| !v.is_finite()) {
            return Err(format_error("base scores must be finite"));
        }
        let n_trees = r.read_usize(32)?;
        if !n_trees.is_multiple_of(n_outputs) {
            return Err(format_error("tree count is not a multiple of the outputs"));
        }
        r.ensure_fits(n_trees, 1, "tree")?;
        let tree_weights = if r.read_bool()? {
            r.ensure_fits(n_trees, 32, "tree weight")?;
            let weights = (0..n_trees)
                .map(|_| r.read_f32())
                .collect::<Result<Vec<_>>>()?;
            if weights.iter().any(|w| !w.is_finite()) {
                return Err(format_error("tree weights must be finite"));
            }
            Some(weights)
        } else {
            None
        };
        let default_mode = r.read(2)?;
        if default_mode > DEFAULT_PER_NODE {
            return Err(format_error("invalid default-direction mode"));
        }
        let n_used = r.read_usize(32)?;
        let max_thresholds = r.read(32)?;
        let n_leaves = r.read_usize(32)?;
        let heap_depth_bits = r.read(6)?;
        let preorder_nodes_bits = r.read(6)?;
        if n_used > n_features || (n_used == 0) != (max_thresholds == 0) {
            return Err(format_error("inconsistent feature map counts"));
        }
        let widths = Widths {
            feature_ref: bits(n_used.saturating_sub(1) as u64),
            threshold_ref: bits(u64::from(max_thresholds.saturating_sub(1))),
            leaf_ref: bits(n_leaves.saturating_sub(1) as u64),
            default_bit: u32::from(default_mode == DEFAULT_PER_NODE),
        };

        // Feature & threshold map.
        let input_bits = bits(n_features as u64 - 1);
        r.ensure_fits(n_used, 5, "used feature")?;
        let mut map: Vec<(usize, u32, u32, usize, u32)> = Vec::with_capacity(n_used);
        for _ in 0..n_used {
            let input = r.read_usize(input_bits)?;
            let kind = r.read(2)?;
            let code = r.read(3)?;
            if code > 5 {
                return Err(format_error("threshold width code exceeds 5"));
            }
            let count = r.read_usize(widths.threshold_ref)? + 1;
            if count > max_thresholds as usize {
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
            if input >= n_features || map.last().is_some_and(|&(prev, ..)| prev >= input) {
                return Err(format_error("feature map entries must ascend"));
            }
            map.push((input, kind, 1u32 << code, count, len_bits));
        }

        // Global thresholds.
        let mut features = Vec::with_capacity(n_used);
        for &(input, kind, width, count, len_bits) in &map {
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

        // Global leaf values.
        r.ensure_fits(n_leaves, 32, "leaf value")?;
        let leaf_values = (0..n_leaves)
            .map(|_| r.read_f32())
            .collect::<Result<Vec<_>>>()?;
        if leaf_values.iter().any(|v| !v.is_finite()) {
            return Err(format_error("leaf values must be finite"));
        }

        let mut model = CompactModel {
            bytes: Vec::new(),
            stream: Vec::new(),
            meta,
            n_features,
            base_score,
            tree_weights,
            default_mode,
            features,
            leaf_values,
            widths,
            trees: Vec::with_capacity(n_trees),
        };

        // Trees: record each one's layout and slot offset, then validate it.
        for _ in 0..n_trees {
            let layout = if r.read_bool()? {
                let nodes = r.read(preorder_nodes_bits)?.checked_add(1);
                Layout::Preorder {
                    nodes: nodes.ok_or_else(|| format_error("preorder node count overflow"))?,
                }
            } else {
                let depth = r.read(heap_depth_bits)?;
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
            model.trees.push(PackedTree {
                layout,
                offset: r.pos,
            });
            r.pos += total as usize;
        }
        if r.remaining() >= 8 || read_bits(&stream, r.pos, r.remaining() as u32) != 0 {
            return Err(format_error("trailing data after the last tree"));
        }
        model.stream = stream;
        for t in 0..model.trees.len() {
            model.validate_tree(t)?;
        }
        model.bytes = bytes.to_vec();
        Ok(model)
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
            Layout::Heap { depth, .. } => {
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
        let s = &self.stream;
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
        let leaf = match tree.layout {
            Layout::Heap { depth, .. } => {
                let first_leaf = (1u32 << depth) - 1;
                let mut i = 0u32;
                loop {
                    match self.slot(tree, i, i >= first_leaf) {
                        Slot::Leaf(leaf) => break leaf,
                        Slot::Split {
                            feature,
                            threshold,
                            default_left,
                            ..
                        } => {
                            let left = self.goes_left(feature, threshold, default_left, row);
                            i = 2 * i + if left { 1 } else { 2 };
                        }
                    }
                }
            }
            Layout::Preorder { .. } => {
                let mut i = 0u32;
                loop {
                    match self.slot(tree, i, false) {
                        Slot::Leaf(leaf) => break leaf,
                        Slot::Split {
                            feature,
                            threshold,
                            default_left,
                            right,
                        } => {
                            let left = self.goes_left(feature, threshold, default_left, row);
                            i += if left { 1 } else { right };
                        }
                    }
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
        out.par_chunks_mut(k)
            .enumerate()
            .with_min_len(256)
            .for_each_init(
                || RowBlock::single_rows(data),
                |block, (r, margins)| {
                    block.load(r, 1);
                    let row = block.row(0).expect("single-row blocks are dense");
                    for t in 0..self.trees.len() {
                        margins[t % k] += weight(t) * self.tree_leaf(t, row);
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
        let objective = rebuild_objective(
            &self.meta.objective,
            &self.meta.objective_params(),
            self.meta.num_class,
            self.meta.n_targets,
        );
        Ok(transform_margins(
            &self.meta.objective,
            objective.ok().as_deref(),
            self.n_outputs(),
            margin,
        ))
    }

    /// The serialized model (the exact bytes it was parsed from).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    /// Serialized size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Save the serialized model to a file.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, &self.bytes)?;
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
    /// [`crate::learner::compact_model`]), predicting bit-identical margins.
    /// Fails for gblinear models.
    pub fn to_compact(&self) -> Result<CompactModel> {
        CompactModel::from_model(self)
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
    numeric: Vec<f32>,
    sets: Vec<Vec<u32>>,
}

fn sorted_categories(tree: &RegTree, node: &crate::tree::Node) -> Vec<u32> {
    let mut set = tree.categories()[node.cat_begin as usize..node.cat_end as usize].to_vec();
    set.sort_unstable();
    set.dedup();
    set
}

/// Depth of the deepest leaf (root = 0) and whether every leaf sits there.
fn depth_and_completeness(tree: &RegTree) -> (u32, bool) {
    let mut depths = Vec::new();
    let mut stack = vec![(0usize, 0u32)];
    while let Some((id, d)) = stack.pop() {
        let node = tree.node(id);
        if node.is_leaf() {
            depths.push(d);
        } else {
            stack.push((node.left as usize, d + 1));
            stack.push((node.right as usize, d + 1));
        }
    }
    let max = depths.iter().copied().max().unwrap_or(0);
    (max, depths.iter().all(|&d| d == max))
}

/// Node ids in preorder (node, left subtree, right subtree).
fn preorder(tree: &RegTree) -> Vec<usize> {
    let mut order = Vec::with_capacity(tree.num_nodes());
    let mut stack = vec![0usize];
    while let Some(id) = stack.pop() {
        order.push(id);
        let node = tree.node(id);
        if !node.is_leaf() {
            stack.push(node.right as usize);
            stack.push(node.left as usize);
        }
    }
    order
}

/// Serialize `model` in the compact layout.
fn encode(model: &BoostedModel) -> Result<Vec<u8>> {
    if model.linear().is_some() {
        return Err(format_error(
            "gblinear models have no trees to store; use the native format",
        ));
    }
    let trees = &model.trees()[..model.effective_ntrees()];
    let n_features = model.n_features();

    // Dictionaries: used features, their thresholds/sets, leaf values.
    let mut collected: BTreeMap<u32, Collected> = BTreeMap::new();
    let mut leaf_index: HashMap<u32, u32> = HashMap::new();
    let mut leaf_values: Vec<f32> = Vec::new();
    let (mut any_left, mut any_right) = (false, false);
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
            let entry = collected.entry(node.split_feature).or_insert_with(|| {
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
                    sets.insert(sorted_categories(tree, node));
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

    // Feature map entries with sorted dictionaries and reference lookups.
    let mut map: Vec<MapEntry> = Vec::with_capacity(collected.len());
    let mut feature_ref: HashMap<u32, u32> = HashMap::new();
    let mut numeric_ref: HashMap<(u32, u32), u32> = HashMap::new();
    let mut set_ref: HashMap<(u32, Vec<u32>), u32> = HashMap::new();
    for (&input, dict) in &collected {
        feature_ref.insert(input, map.len() as u32);
        match dict {
            Collected::Numeric(set) => {
                let mut values: Vec<f32> = set.iter().map(|&b| f32::from_bits(b)).collect();
                values.sort_by(f32::total_cmp);
                for (i, v) in values.iter().enumerate() {
                    numeric_ref.insert((input, v.to_bits()), i as u32);
                }
                let (kind, code) = numeric_encoding(&values);
                map.push(MapEntry {
                    input,
                    kind,
                    code,
                    len_bits: 0,
                    numeric: values,
                    sets: Vec::new(),
                });
            }
            Collected::Categorical(sets) => {
                let sets: Vec<Vec<u32>> = sets.iter().cloned().collect();
                let max_cat = sets.iter().flatten().copied().max().unwrap_or(0);
                let code = (0..=5u32)
                    .find(|&c| bits(u64::from(max_cat)) <= 1 << c)
                    .unwrap_or(5);
                let max_len = sets.iter().map(Vec::len).max().unwrap_or(0);
                for (i, s) in sets.iter().enumerate() {
                    set_ref.insert((input, s.clone()), i as u32);
                }
                map.push(MapEntry {
                    input,
                    kind: KIND_CATEGORICAL,
                    code,
                    len_bits: bits(max_len as u64).max(1),
                    numeric: Vec::new(),
                    sets,
                });
            }
        }
    }
    let max_thresholds = map
        .iter()
        .map(|e| e.numeric.len().max(e.sets.len()))
        .max()
        .unwrap_or(0);
    let widths = Widths {
        feature_ref: bits(map.len().saturating_sub(1) as u64),
        threshold_ref: bits(max_thresholds.saturating_sub(1) as u64),
        leaf_ref: bits(leaf_values.len().saturating_sub(1) as u64),
        default_bit: u32::from(default_mode == DEFAULT_PER_NODE),
    };
    if u32::try_from(max_thresholds).is_err()
        || u32::try_from(leaf_values.len()).is_err()
        || u32::try_from(trees.len()).is_err()
        || u32::try_from(n_features).is_err()
    {
        return Err(format_error("model too large for 32-bit counts"));
    }

    // Per-tree layout choice.
    let layouts: Vec<Layout> = trees
        .iter()
        .map(|tree| {
            let preorder = Layout::Preorder {
                nodes: tree.num_nodes() as u32,
            };
            let (depth, complete) = depth_and_completeness(tree);
            let heap = Layout::Heap { depth, complete };
            if depth <= MAX_HEAP_DEPTH && heap.total_bits(widths) <= preorder.total_bits(widths) {
                heap
            } else {
                preorder
            }
        })
        .collect();
    let heap_depth_bits = layouts
        .iter()
        .filter_map(|l| match l {
            Layout::Heap { depth, .. } => Some(bits(u64::from(*depth))),
            Layout::Preorder { .. } => None,
        })
        .max()
        .unwrap_or(0);
    let preorder_nodes_bits = layouts
        .iter()
        .filter_map(|l| match l {
            Layout::Preorder { nodes } => Some(bits(u64::from(*nodes) - 1)),
            Layout::Heap { .. } => None,
        })
        .max()
        .unwrap_or(0);

    let mut w = BitWriter::default();
    w.write(n_features as u64, 32);
    w.write(model.n_outputs() as u64, 32);
    for &b in model.base_scores() {
        w.write_f32(b);
    }
    w.write(trees.len() as u64, 32);
    let weights: Vec<f32> = (0..trees.len()).map(|t| model.tree_weight(t)).collect();
    let weighted = weights.iter().any(|&x| x.to_bits() != 1.0f32.to_bits());
    w.write_bool(weighted);
    if weighted {
        for &x in &weights {
            w.write_f32(x);
        }
    }
    w.write(u64::from(default_mode), 2);
    w.write(map.len() as u64, 32);
    w.write(max_thresholds as u64, 32);
    w.write(leaf_values.len() as u64, 32);
    w.write(u64::from(heap_depth_bits), 6);
    w.write(u64::from(preorder_nodes_bits), 6);

    let input_bits = bits(n_features as u64 - 1);
    for e in &map {
        w.write(u64::from(e.input), input_bits);
        w.write(u64::from(e.kind), 2);
        w.write(u64::from(e.code), 3);
        let count = e.numeric.len().max(e.sets.len());
        w.write(count as u64 - 1, widths.threshold_ref);
        if e.kind == KIND_CATEGORICAL {
            w.write(u64::from(e.len_bits), 6);
        }
    }
    for e in &map {
        let width = 1u32 << e.code;
        for &v in &e.numeric {
            w.write(encode_threshold(v, e.kind, width), width);
        }
        for set in &e.sets {
            w.write(set.len() as u64, e.len_bits);
            for &c in set {
                w.write(u64::from(c), width);
            }
        }
    }
    for &v in &leaf_values {
        w.write_f32(v);
    }

    // Split fields of an internal node.
    let write_split = |w: &mut BitWriter, tree: &RegTree, node: &crate::tree::Node| {
        let f = node.split_feature;
        w.write(u64::from(feature_ref[&f]), widths.feature_ref);
        let t = if node.is_categorical {
            set_ref[&(f, sorted_categories(tree, node))]
        } else {
            numeric_ref[&(f, node.split_cond.to_bits())]
        };
        w.write(u64::from(t), widths.threshold_ref);
        if widths.default_bit == 1 {
            w.write_bool(node.default_left);
        }
    };
    for (tree, &layout) in trees.iter().zip(&layouts) {
        let slot_width = layout.slot_width(widths);
        match layout {
            Layout::Heap { depth, complete } => {
                w.write_bool(false);
                w.write(u64::from(depth), heap_depth_bits);
                w.write_bool(complete);
                let internal = (1usize << depth) - 1;
                let mut heap: Vec<Option<usize>> = vec![None; 2 * internal + 1];
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
                        if node.is_leaf() {
                            w.write(
                                u64::from(leaf_index[&node.leaf_value.to_bits()]),
                                widths.leaf_ref,
                            );
                        } else {
                            write_split(&mut w, tree, node);
                        }
                    }
                    let width = if i < internal {
                        slot_width
                    } else {
                        widths.leaf_ref
                    };
                    w.write(0, width - (w.len - start) as u32);
                }
            }
            Layout::Preorder { nodes } => {
                w.write_bool(true);
                w.write(u64::from(nodes) - 1, preorder_nodes_bits);
                let order = preorder(tree);
                let mut position = vec![0u32; tree.num_nodes()];
                for (p, &id) in order.iter().enumerate() {
                    position[id] = p as u32;
                }
                let offset_bits = bits(u64::from(nodes) - 1);
                for (p, &id) in order.iter().enumerate() {
                    let start = w.len;
                    let node = tree.node(id);
                    w.write_bool(node.is_leaf());
                    if node.is_leaf() {
                        w.write(
                            u64::from(leaf_index[&node.leaf_value.to_bits()]),
                            widths.leaf_ref,
                        );
                    } else {
                        write_split(&mut w, tree, node);
                        let right = position[node.right as usize] - p as u32;
                        w.write(u64::from(right), offset_bits);
                    }
                    w.write(0, slot_width - (w.len - start) as u32);
                }
            }
        }
    }

    let meta = Meta {
        objective: model.objective().to_string(),
        objective_params: (*model.objective_params()
            != ObjectiveParams::defaults_for(model.objective()))
        .then(|| model.objective_params().clone()),
        num_class: model.num_class(),
        n_targets: model.n_targets(),
    };
    let meta = postcard::to_stdvec(&meta).map_err(|e| format_error(e.to_string()))?;
    let mut bytes = Vec::with_capacity(PREFIX_BYTES + meta.len() + w.bytes.len());
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION);
    bytes.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&meta);
    bytes.extend_from_slice(&w.bytes);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BoosterKind, GrowPolicy, TrainingParams, TreeMethod};
    use crate::data::FeatureType;
    use crate::learner::{train, train_with_eval};

    /// Deterministic pseudo-random value in `[0, 1)`.
    fn noise(i: usize) -> f32 {
        let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
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
        let data = DMatrix::from_dense(&x, 600, 5)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let labels01: Vec<f32> = y.iter().map(|&v| f32::from(v > 0.5)).collect();
        let labels3: Vec<f32> = y
            .iter()
            .map(|&v| ((v + 1.5).clamp(0.0, 2.9)) as u32 as f32)
            .collect();
        let binary = data.clone().with_labels(&labels01).unwrap();
        let multi = data.clone().with_labels(&labels3).unwrap();

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
        ];
        for (params, d) in cases {
            let model = train(&params, d, 15).unwrap();
            assert_bit_identical(&model, d);
        }
    }

    #[test]
    fn deep_unbalanced_trees_use_the_preorder_layout() {
        let (x, y) = dataset(800, 4, true);
        let data = DMatrix::from_dense(&x, 800, 4)
            .unwrap()
            .with_labels(&y)
            .unwrap();
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
        let (x, y) = dataset(400, 3, false);
        let data = DMatrix::from_dense(&x, 400, 3)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let (xv, yv) = dataset(200, 3, false);
        let yv: Vec<f32> = yv.iter().map(|v| -v).collect();
        let valid = DMatrix::from_dense(&xv, 200, 3)
            .unwrap()
            .with_labels(&yv)
            .unwrap();
        let params = TrainingParams::builder().build().unwrap();
        let model = train_with_eval(&params, &data, 50, &[(&valid, "valid")], Some(3))
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
        let (x, y) = dataset(2000, 8, false);
        let data = DMatrix::from_dense(&x, 2000, 8)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder().max_depth(4).build().unwrap();
        let model = train(&params, &data, 100).unwrap();
        let report = model.size_report().unwrap();
        assert_eq!(report.trees, 100);
        assert_eq!(
            report.compact_bytes,
            model.to_compact_bytes().unwrap().len()
        );
        assert!(
            report.compression_ratio() > 4.0,
            "compact {} vs native {}",
            report.compact_bytes,
            report.native_bytes
        );
        assert!(report.reuse_factor() >= 1.0);
    }

    #[test]
    fn corrupt_input_is_rejected_without_panicking() {
        let (x, y) = dataset(300, 4, true);
        let data = DMatrix::from_dense(&x, 300, 4)
            .unwrap()
            .with_labels(&y)
            .unwrap();
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

    #[test]
    fn gblinear_models_are_rejected() {
        let (x, y) = dataset(100, 3, false);
        let data = DMatrix::from_dense(&x, 100, 3)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .booster(BoosterKind::GbLinear)
            .build()
            .unwrap();
        let model = train(&params, &data, 3).unwrap();
        assert!(matches!(
            model.to_compact_bytes(),
            Err(HessboostError::ModelFormat(_))
        ));
    }
}
