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
//!              num_class, n_targets, num_parallel_tree, the objective
//!              parameters when they differ from the objective's defaults,
//!              and a shrunk model's shrinkage record (the native format's
//!              `REQUIRED` `shrinkage.*` sections); readers give sections a
//!              file lacks their default, so later additions keep older
//!              files loading
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
//! output in tree order in `f32`, exactly as the native predictor does. A
//! model trained with model shrinkage keeps its closed-form tree weights and
//! intercepts in the bit stream, but predicts from its shrinkage record as
//! the native predictor does (every iteration shrinks the margins, then adds
//! its trees at weight `1`; see `model::shrinkage`).
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

mod bitstream;
mod decode;
mod encode;

use super::native::{
    OBJECTIVE_SECTIONS, SHRINKAGE_SECTIONS, read_objective_params, read_shrinkage,
    write_objective_params, write_shrinkage,
};
use super::objective::{ModelObjective, StoredObjectiveParams};
use super::sections::{Sections, Writer};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::{
    BoostedModel, RowBlock, Shrinkage, initial_margins, shrink_margins, transform_model_margins,
    validate_prediction_data,
};
use crate::tree::{RegTree, scalar_tree_output};
use bitstream::read_bits;
use encode::encode;
use rayon::prelude::*;

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

/// A feature map entry's value type (the 2-bit numeric type field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum ThresholdKind {
    /// Unsigned integer thresholds.
    Unsigned = 0,
    /// Two's-complement integer thresholds.
    Signed = 1,
    /// IEEE binary16 or binary32 thresholds.
    Float = 2,
    /// Left-category sets.
    Categorical = 3,
}

impl TryFrom<u32> for ThresholdKind {
    type Error = HessboostError;

    fn try_from(raw: u32) -> Result<Self> {
        Ok(match raw {
            0 => ThresholdKind::Unsigned,
            1 => ThresholdKind::Signed,
            2 => ThresholdKind::Float,
            3 => ThresholdKind::Categorical,
            _ => return Err(format_error("invalid numeric type")),
        })
    }
}

/// Where splits send missing values (the 2-bit default direction mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum DefaultDirection {
    /// Every split sends missing values left.
    AllLeft = 0,
    /// Every split sends missing values right.
    AllRight = 1,
    /// Each split stores its own default-left bit.
    PerNode = 2,
}

impl TryFrom<u32> for DefaultDirection {
    type Error = HessboostError;

    fn try_from(raw: u32) -> Result<Self> {
        Ok(match raw {
            0 => DefaultDirection::AllLeft,
            1 => DefaultDirection::AllRight,
            2 => DefaultDirection::PerNode,
            _ => return Err(format_error("invalid default-direction mode")),
        })
    }
}

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
    objective: ModelObjective,
    /// The `max_delta_step` training used.
    max_delta_step: f64,
    num_class: usize,
    n_targets: usize,
    /// Trees per output in each boosting iteration: tree `t` feeds output
    /// `(t / num_parallel_tree) % n_outputs`.
    num_parallel_tree: usize,
    /// The model shrinkage record, when the model was trained with it.
    shrinkage: Option<Shrinkage>,
}

/// Every metadata section besides [`OBJECTIVE_SECTIONS`] and
/// [`SHRINKAGE_SECTIONS`].
const META_SECTIONS: &[&str] = &["objective", "num_class", "n_targets", "num_parallel_tree"];

impl Meta {
    /// The metadata as a section table, the objective parameters only when
    /// they differ from the objective's defaults
    /// ([`StoredObjectiveParams::defaults_for`]).
    fn section_table(&self) -> Writer {
        let mut w = Writer::default();
        let name = self.objective.name();
        w.str("objective", name);
        w.u64("num_class", self.num_class as u64);
        w.u64("n_targets", self.n_targets as u64);
        w.u64("num_parallel_tree", self.num_parallel_tree as u64);
        let params = StoredObjectiveParams::of(&self.objective, self.max_delta_step);
        if params != StoredObjectiveParams::defaults_for(name) {
            write_objective_params(&mut w, &params);
        }
        if let Some(shrinkage) = &self.shrinkage {
            write_shrinkage(&mut w, shrinkage);
        }
        w
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let (s, rest) = Sections::parse(bytes, |name| {
            META_SECTIONS.contains(&name)
                || OBJECTIVE_SECTIONS.contains(&name)
                || SHRINKAGE_SECTIONS.contains(&name)
        })
        .map_err(|e| format_error(format!("metadata: {e}")))?;
        if !rest.is_empty() {
            return Err(format_error("metadata has trailing bytes"));
        }
        let name = s.str("objective")?;
        let params = read_objective_params(&s, StoredObjectiveParams::defaults_for(name))?;
        let num_class = s.usize("num_class")?;
        Ok(Meta {
            objective: ModelObjective::from_stored(name, &params, num_class)?,
            max_delta_step: params.max_delta_step,
            num_class,
            n_targets: s.usize("n_targets")?,
            num_parallel_tree: s.usize("num_parallel_tree")?,
            shrinkage: read_shrinkage(&s)?,
        })
    }
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
    fn new(
        n_used: usize,
        max_thresholds: usize,
        n_leaves: usize,
        default_mode: DefaultDirection,
    ) -> Self {
        Widths {
            feature_ref: bits(n_used.saturating_sub(1) as u64),
            threshold_ref: bits(max_thresholds.saturating_sub(1) as u64),
            leaf_ref: bits(n_leaves.saturating_sub(1) as u64),
            default_bit: u32::from(default_mode == DefaultDirection::PerNode),
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
    default_mode: DefaultDirection,
    features: Vec<FeatureEntry>,
    leaf_values: Vec<f32>,
    widths: Widths,
    trees: Vec<PackedTree>,
}

impl CompactModel {
    /// The serialized bytes, without the padding.
    fn serialized(&self) -> &[u8] {
        &self.bytes[..self.bytes.len() - STREAM_PAD]
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
            DefaultDirection::AllLeft => true,
            DefaultDirection::AllRight => false,
            DefaultDirection::PerNode => {
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
    pub fn predict_margin(&self, data: &DMatrix) -> Result<super::Predictions> {
        let k = self.n_outputs();
        validate_prediction_data(self.n_features, k, data)?;
        let shrinkage = self.meta.shrinkage.as_ref();
        let (mut out, weights) = match shrinkage {
            // The record's own recurrence; the stored weights are its
            // closed form.
            Some(shrinkage) => (shrinkage.start_margins(data), None),
            None => (
                initial_margins(&self.base_score, data),
                self.tree_weights.as_deref(),
            ),
        };
        let weight = |t: usize| weights.map_or(1.0, |w| w[t]);
        let parallel = self.meta.num_parallel_tree;
        let per = parallel * k;
        out.par_chunks_mut(k)
            .enumerate()
            .with_min_len(256)
            .for_each_init(
                || RowBlock::single_rows(data),
                |block, (r, margins)| {
                    block.load(r, 1);
                    let row = block.row(0).expect("single-row blocks are dense");
                    for t in 0..self.trees.len() {
                        if let Some(shrinkage) = shrinkage
                            && t % per == 0
                        {
                            shrink_margins(margins, shrinkage.factors()[t / per]);
                        }
                        margins[scalar_tree_output(t, parallel, k)] +=
                            weight(t) * self.tree_leaf(t, row);
                    }
                },
            );
        if let Some(shrinkage) = shrinkage {
            shrinkage.finish_margins(data, &mut out);
        }
        Ok(super::Predictions::new(out, data.n_rows(), k))
    }

    /// Predictions in the objective's reported space, identical to
    /// [`BoostedModel::predict`] of the source model (probabilities for
    /// logistic objectives, class indices for `multi:softmax`, ...).
    pub fn predict(&self, data: &DMatrix) -> Result<super::Predictions> {
        let margin = self.predict_margin(data)?;
        Ok(transform_model_margins(
            &self.meta.objective,
            self.meta.max_delta_step,
            self.meta.n_targets,
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

    /// The objective that drives [`CompactModel::predict`].
    pub fn objective(&self) -> &ModelObjective {
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

#[cfg(test)]
mod tests;
