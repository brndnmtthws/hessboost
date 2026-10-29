//! What-if prediction for rows that differ from one base row in a few
//! features ([`RowVariants`]), and margin bounds over a box of feature
//! values ([`BoostedModel::margin_bounds_into`]).
//!
//! Both walk every tree once for many rows. Up to [`CHUNK`] variants travel
//! down a tree together as the bits of a `u64` mask, split only at nodes
//! testing a varying feature (every other node routes them all by the base
//! row); a box follows every child some row inside it reaches. Leaves add
//! with the row methods' `f32` arithmetic in their order, so a variant's
//! margins are its row's bit for bit. Like the row methods, neither
//! allocates nor enters rayon: the recursion is bounded by tree depth and
//! every buffer is on the stack.

use super::{BoostedModel, Iterations, LinearModel, Transform, shrink_margins};
use crate::data::{check_len, rejects_dense};
use crate::error::{HessboostError, Result};
use crate::tree::{RegTree, scalar_tree_output};
use std::ops::Range;

/// Variants one tree walk carries: one bit of a `u64` mask each.
const CHUNK: usize = 64;

/// `multi:softmax` classes whose margins a chunk of variants forms at once,
/// on the stack (more are formed chunk by chunk).
const CLASS_CHUNK: usize = 64;

/// Outputs of a vector-leaf tree whose leaf ranges one bounds walk collects
/// on the stack (wider leaves walk once per group).
const OUTPUT_CHUNK: usize = 64;

/// Varying features whose chunk values a walk copies lane-major onto the
/// stack (later ones are read strided from the caller's rows).
const LANE_FEATURES: usize = 8;

/// Features with a byte in a call's slot map ([`Slots`]).
const SLOT_MAP: usize = 1024;

/// [`Slots`]: the feature does not vary.
const NOT_VARYING: u8 = u8::MAX;

/// [`Slots`]: the feature varies at a position past a byte; search for it.
const SCAN: u8 = u8::MAX - 1;

/// Active variants from which a numeric split tests every lane at once.
const DENSE_SPLIT: u32 = 16;

/// Active variants from which a single-output leaf adds to every lane.
const DENSE_ADD: u32 = 8;

/// Rows that differ from one base row in a few features: variant `v` is
/// `base` with `base[features[j]]` replaced by `values[v * features.len() +
/// j]`, for every `j`. [`BoostedModel::predict_margin_variants_into`] and
/// [`BoostedModel::predict_variants_into`] predict all of them bit for bit
/// as [`BoostedModel::predict_margin_row_into`] and
/// [`BoostedModel::predict_row_into`] predict each materialized row, but walk
/// every tree once per 64 variants instead of once per variant: a search
/// scoring hundreds of candidate values of a few inputs against one fixed
/// context. `NaN` is a missing value, in `base` and `values` alike.
///
/// ```
/// use hessboost::model::RowVariants;
/// use hessboost::prelude::*;
///
/// # fn main() -> Result<()> {
/// let x: Vec<f32> = (0..120).map(|i| (i % 40) as f32).collect();
/// let y: Vec<f32> = x.chunks(3).map(|row| row[0] - row[2]).collect();
/// let dtrain = DMatrix::from_dense(&x, 40, 3)?.with_labels(&y)?;
/// let model = train(&TrainingParams::default(), &dtrain, 5)?;
///
/// // Three candidate values of feature 2 against one fixed row.
/// let base = [4.0, f32::NAN, 1.0];
/// let candidates = [0.0, 10.0, 30.0];
/// let variants = RowVariants::new(&base, &[2], &candidates)?;
/// assert_eq!(variants.len(), 3);
/// let mut margins = [0.0; 3];
/// model.predict_margin_variants_into(&variants, Iterations::Best, &mut margins)?;
/// for (&value, &margin) in candidates.iter().zip(&margins) {
///     let mut row = base;
///     row[2] = value;
///     assert_eq!(margin, model.predict_margin_row(&row, Iterations::Best)?);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RowVariants<'a> {
    base: &'a [f32],
    features: &'a [usize],
    /// Row-major, `len() x features.len()`.
    values: &'a [f32],
    /// Bit `f % 256` set for every varying feature `f`: a node splitting on
    /// another feature routes by `base` without searching `features`.
    filter: [u64; 4],
}

impl<'a> RowVariants<'a> {
    /// The variants of `base` in which the features `features` (distinct
    /// indices) take the values `values`: one row of `features.len()`
    /// values per variant, row-major. The rows are read in place, never
    /// copied; the model checks `base` and the indices when it predicts.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] (`variants`) when `features` is
    /// empty or lists a feature twice, [`HessboostError::DimensionMismatch`]
    /// when `values` is not whole rows of `features.len()` values.
    pub fn new(base: &'a [f32], features: &'a [usize], values: &'a [f32]) -> Result<Self> {
        if features.is_empty() {
            return Err(HessboostError::invalid_param(
                "variants",
                "at least one feature must vary",
            ));
        }
        let mut filter = [0u64; 4];
        for (i, &f) in features.iter().enumerate() {
            if features[..i].contains(&f) {
                return Err(HessboostError::invalid_param(
                    "variants",
                    format!("feature {f} is listed twice"),
                ));
            }
            filter[(f >> 6) & 3] |= 1 << (f & 63);
        }
        let m = features.len();
        if !values.len().is_multiple_of(m) {
            return Err(HessboostError::dimension_mismatch(
                "variant values (whole rows of the varying features)",
                values.len().next_multiple_of(m),
                values.len(),
            ));
        }
        Ok(Self {
            base,
            features,
            values,
            filter,
        })
    }

    /// The number of variants.
    pub fn len(&self) -> usize {
        self.values.len() / self.features.len()
    }

    /// Whether there are no variants.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The variants in consecutive groups of up to [`CHUNK`], each one tree
    /// walk's mask.
    fn chunks(&self) -> impl Iterator<Item = RowVariants<'a>> {
        let this = *self;
        self.values
            .chunks(CHUNK * self.features.len())
            .map(move |values| RowVariants { values, ..this })
    }

    /// The position of feature `f` in `features`, if it varies.
    #[inline]
    fn slot(&self, f: usize) -> Option<usize> {
        if (self.filter[(f >> 6) & 3] >> (f & 63)) & 1 == 0 {
            return None;
        }
        self.features.iter().position(|&g| g == f)
    }

    /// The mask of every variant of a chunk (at most [`CHUNK`] of them).
    fn mask(&self) -> u64 {
        let n = self.len();
        if n >= CHUNK {
            u64::MAX
        } else {
            (1u64 << n) - 1
        }
    }
}

/// A dense value as the trees read it: `NaN` is missing.
#[inline]
fn present(x: f32) -> Option<f32> {
    (!x.is_nan()).then_some(x)
}

/// The indices of the set bits of `mask`, lowest first.
#[inline]
fn bits(mut mask: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (mask != 0).then(|| {
            let v = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            v
        })
    })
}

/// Which varying feature (its position in `features`) each feature is: a
/// byte per feature below [`SLOT_MAP`], built once per call, so a node
/// visit finds its feature's slot in one load; later features (and slots
/// past a byte) fall back to [`RowVariants::slot`].
struct Slots<'a> {
    variants: RowVariants<'a>,
    map: [u8; SLOT_MAP],
}

impl<'a> Slots<'a> {
    fn new(variants: RowVariants<'a>) -> Self {
        let mut map = [NOT_VARYING; SLOT_MAP];
        for (j, &f) in variants.features.iter().enumerate() {
            if let Some(entry) = map.get_mut(f) {
                *entry = u8::try_from(j).ok().filter(|&j| j < SCAN).unwrap_or(SCAN);
            }
        }
        Self { variants, map }
    }

    /// The position of feature `f` in `features`, if it varies.
    #[inline]
    fn get(&self, f: usize) -> Option<usize> {
        match self.map.get(f) {
            Some(&NOT_VARYING) => None,
            Some(&SCAN) | None => self.variants.slot(f),
            Some(&j) => Some(usize::from(j)),
        }
    }
}

/// One chunk's walk: its variants, the call's [`Slots`], and the values of
/// its first [`LANE_FEATURES`] varying features copied lane-major (lane `v`
/// is variant `v`), so a split reads one contiguous column.
struct Walk<'s, 'a> {
    slots: &'s Slots<'a>,
    chunk: RowVariants<'a>,
    lanes: [[f32; CHUNK]; LANE_FEATURES],
}

impl<'s, 'a> Walk<'s, 'a> {
    /// The walk of `chunk` (at most [`CHUNK`] variants of the call's).
    fn new(slots: &'s Slots<'a>, chunk: RowVariants<'a>) -> Self {
        let m = chunk.features.len();
        // Lanes past the chunk's variants stay NaN; masks never select them.
        let mut lanes = [[f32::NAN; CHUNK]; LANE_FEATURES];
        for (v, row) in chunk.values.chunks_exact(m).enumerate() {
            for (lane, &x) in lanes.iter_mut().zip(row) {
                lane[v] = x;
            }
        }
        Self {
            slots,
            chunk,
            lanes,
        }
    }

    /// Variant `v`'s value of varying feature `j`.
    #[inline]
    fn lane(&self, v: usize, j: usize) -> f32 {
        match self.lanes.get(j) {
            Some(lane) => lane[v],
            None => self.chunk.values[v * self.chunk.features.len() + j],
        }
    }

    /// Variant `v`'s value of feature `f`.
    #[inline]
    fn value(&self, v: usize, f: usize) -> f32 {
        match self.slots.get(f) {
            Some(j) => self.lane(v, j),
            None => self.chunk.base[f],
        }
    }

    /// The variants of `mask` that a numeric split `x < threshold` on
    /// varying feature `j` sends left: [`split_goes_left`]'s test (a missing
    /// value takes `default_left`, a present one goes left below the
    /// threshold, so a tie goes right) without branches. With many active
    /// variants every lane of the contiguous column is tested at once.
    ///
    /// [`split_goes_left`]: crate::tree::split_goes_left
    #[inline]
    fn numeric_left(&self, j: usize, threshold: f32, default_left: bool, mask: u64) -> u64 {
        let goes = |x: f32| u64::from((x < threshold) | (x.is_nan() & default_left));
        if mask.count_ones() >= DENSE_SPLIT
            && let Some(lane) = self.lanes.get(j)
        {
            let mut left = 0u64;
            for (v, &x) in lane.iter().enumerate() {
                left |= goes(x) << v;
            }
            return left & mask;
        }
        let mut left = 0u64;
        for v in bits(mask) {
            left |= goes(self.lane(v, j)) << v;
        }
        left
    }
}

/// `acc[v] += c` for the variants `v` of `mask`, in one contiguous block of
/// [`CHUNK`] lanes (a single-output chunk's margins): with many active
/// variants every lane adds and inactive ones keep their value, a select
/// LLVM vectorizes; the sums are those of the per-variant additions.
#[inline]
fn add_masked(acc: &mut [f32], mask: u64, c: f32) {
    if mask.count_ones() >= DENSE_ADD
        && let Ok(block) = <&mut [f32; CHUNK]>::try_from(&mut *acc)
    {
        for (v, a) in block.iter_mut().enumerate() {
            let sum = *a + c;
            *a = if (mask >> v) & 1 == 1 { sum } else { *a };
        }
        return;
    }
    for v in bits(mask) {
        acc[v] += c;
    }
}

/// Send the variants `mask` of `walk`'s chunk from node `nid` of `tree` to
/// their leaves, calling `leaf(leaf_id, variants)` once per leaf reached with
/// the variants reaching it. A node testing a varying feature splits the
/// mask by each variant's value as the tree routes it (numeric splits
/// through [`Walk::numeric_left`], categorical ones through
/// [`RegTree::goes_left`]: categories as stored, missing values the default
/// direction); any other node sends the whole mask where the base row goes.
/// Recursion only at a split mask, so its depth is bounded by the tree's.
fn route(
    tree: &RegTree,
    walk: &Walk<'_, '_>,
    mut nid: usize,
    mut mask: u64,
    leaf: &mut impl FnMut(usize, u64),
) {
    let nodes = tree.nodes();
    loop {
        let node = &nodes[nid];
        if node.is_leaf() {
            leaf(nid, mask);
            return;
        }
        let f = node.split_feature as usize;
        let Some(j) = walk.slots.get(f) else {
            nid = if tree.goes_left(node, present(walk.chunk.base[f])) {
                node.left as usize
            } else {
                node.right as usize
            };
            continue;
        };
        let left = if node.is_categorical {
            let mut left = 0u64;
            for v in bits(mask) {
                left |= u64::from(tree.goes_left(node, present(walk.lane(v, j)))) << v;
            }
            left
        } else {
            walk.numeric_left(j, node.split_cond, node.default_left, mask)
        };
        let right = mask & !left;
        if right == 0 {
            nid = node.left as usize;
        } else if left == 0 {
            nid = node.right as usize;
        } else {
            route(tree, walk, node.left as usize, left, leaf);
            (nid, mask) = (node.right as usize, right);
        }
    }
}

/// Call `leaf(leaf_id)` for every leaf of `tree` below node `nid` that a row
/// `x` with `lower[f] <= x[f] <= upper[f]` (`NaN` bounds: missing) reaches:
/// a child is reachable when the tree sends the box's lower end there
/// (left) or its upper end (right), which for a threshold `x < c` is every
/// child some value in between reaches, and for a fixed value or a missing
/// feature exactly one. Recursion only where both children are reachable.
fn reach(
    tree: &RegTree,
    lower: &[f32],
    upper: &[f32],
    mut nid: usize,
    leaf: &mut impl FnMut(usize),
) {
    let nodes = tree.nodes();
    loop {
        let node = &nodes[nid];
        if node.is_leaf() {
            leaf(nid);
            return;
        }
        let f = node.split_feature as usize;
        let left = tree.goes_left(node, present(lower[f]));
        let right = !tree.goes_left(node, present(upper[f]));
        if left && right {
            reach(tree, lower, upper, node.left as usize, leaf);
            nid = node.right as usize;
        } else if left {
            nid = node.left as usize;
        } else {
            nid = node.right as usize;
        }
    }
}

/// A lower bound plus the least `term` a row in the box adds. Where opposite
/// infinities meet (finite terms overflowing on one side, an unbounded
/// feature on the other) the sum is `NaN`: `-inf` then bounds every row,
/// unless the box is one row (`point`), whose margin adds exactly so.
#[inline]
fn add_lower(sum: f32, term: f32, point: bool) -> f32 {
    let s = sum + term;
    if s.is_nan() && !point {
        f32::NEG_INFINITY
    } else {
        s
    }
}

/// [`add_lower`] for an upper bound: `+inf` where opposite infinities meet.
#[inline]
fn add_upper(sum: f32, term: f32, point: bool) -> f32 {
    let s = sum + term;
    if s.is_nan() && !point {
        f32::INFINITY
    } else {
        s
    }
}

impl BoostedModel {
    /// Write the margins of every variant of `variants` into `out`
    /// ([`RowVariants::len`] rows of [`Self::n_outputs`] margins,
    /// variant-major): variant `v`'s margins are bit for bit
    /// [`Self::predict_margin_row_into`] of its materialized row, for every
    /// model that method predicts (tree weights, model shrinkage, scalar and
    /// vector leaves, categorical splits, linear leaves, `gblinear`, missing
    /// values, [`Iterations::Best`] and ranges). Never materializes a row:
    /// each tree is walked once per 64 variants, splitting them only where a
    /// node tests a varying feature. Allocates nothing and never enters
    /// rayon.
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] when the base row does not hold
    /// one value per feature or `out` one margin per variant and output,
    /// [`HessboostError::InvalidParameter`] (`variants`) for a varying
    /// feature past the model's, [`HessboostError::InvalidData`] for an
    /// infinite value (`row` in the base row, `variants` in the values),
    /// plus the errors of [`Self::predict_margin_row_into`] for
    /// `iterations`.
    ///
    /// ```
    /// use hessboost::model::RowVariants;
    /// use hessboost::objective::{Multiclass, Objective};
    /// use hessboost::prelude::*;
    ///
    /// # fn main() -> Result<()> {
    /// let x: Vec<f32> = (0..60).map(|i| (i % 30) as f32).collect();
    /// let y: Vec<f32> = (0..30).map(|i| (i % 3) as f32).collect();
    /// let dtrain = DMatrix::from_dense(&x, 30, 2)?.with_labels(&y)?;
    /// let params = TrainingParams::builder()
    ///     .objective(Objective::Softprob(Multiclass::new(3)?))
    ///     .build()?;
    /// let model = train(&params, &dtrain, 5)?;
    ///
    /// // Feature 0 at 0, 5 and missing; feature 1 fixed at 4.
    /// let candidates = [0.0, 5.0, f32::NAN];
    /// let variants = RowVariants::new(&[0.0, 4.0], &[0], &candidates)?;
    /// let mut margins = [0.0; 9];
    /// model.predict_margin_variants_into(&variants, Iterations::Best, &mut margins)?;
    /// let mut row = [0.0; 3];
    /// model.predict_margin_row_into(&[5.0, 4.0], Iterations::Best, &mut row)?;
    /// assert_eq!(&margins[3..6], &row);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_margin_variants_into(
        &self,
        variants: &RowVariants<'_>,
        iterations: impl Into<Iterations>,
        out: &mut [f32],
    ) -> Result<()> {
        let iterations = self.variants_request(variants, iterations.into())?;
        let k = self.n_outputs;
        check_len(
            "variant margins",
            out.len(),
            variants.len().saturating_mul(k),
        )?;
        let slots = Slots::new(*variants);
        for (chunk, out) in variants.chunks().zip(out.chunks_mut(CHUNK * k)) {
            self.chunk_margins(&Walk::new(&slots, chunk), iterations.clone(), 0..k, out);
        }
        Ok(())
    }

    /// Write the predictions of every variant of `variants` into `out`
    /// ([`RowVariants::len`] rows of [`Self::prediction_width`] values,
    /// variant-major): [`Self::predict_margin_variants_into`]'s margins
    /// through the objective's transform, so variant `v`'s values are bit for
    /// bit [`Self::predict_row_into`] of its materialized row
    /// (`multi:softmax`'s class index included). Allocates nothing and never
    /// enters rayon.
    ///
    /// # Errors
    ///
    /// Those of [`Self::predict_margin_variants_into`], with `out` holding
    /// one prediction row per variant.
    ///
    /// ```
    /// use hessboost::model::RowVariants;
    /// use hessboost::objective::{Objective, RegLoss};
    /// use hessboost::prelude::*;
    ///
    /// # fn main() -> Result<()> {
    /// let x: Vec<f32> = (0..80).map(|i| (i % 20) as f32).collect();
    /// let y: Vec<f32> = x.chunks(2).map(|row| f32::from(u8::from(row[0] > 9.0))).collect();
    /// let dtrain = DMatrix::from_dense(&x, 40, 2)?.with_labels(&y)?;
    /// let params = TrainingParams::builder()
    ///     .objective(Objective::BinaryLogistic(RegLoss::default()))
    ///     .build()?;
    /// let model = train(&params, &dtrain, 5)?;
    ///
    /// // Both features varied together.
    /// let values = [1.0, 2.0, 15.0, f32::NAN];
    /// let variants = RowVariants::new(&[0.0, 0.0], &[0, 1], &values)?;
    /// let mut probabilities = [0.0; 2];
    /// model.predict_variants_into(&variants, Iterations::Best, &mut probabilities)?;
    /// assert_eq!(probabilities[1], model.predict_row(&[15.0, f32::NAN], Iterations::Best)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_variants_into(
        &self,
        variants: &RowVariants<'_>,
        iterations: impl Into<Iterations>,
        out: &mut [f32],
    ) -> Result<()> {
        let iterations = self.variants_request(variants, iterations.into())?;
        let k = self.n_outputs;
        let transform = self.transform();
        check_len(
            "variant predictions",
            out.len(),
            variants.len().saturating_mul(transform.width(k)),
        )?;
        let slots = Slots::new(*variants);
        if let Transform::ClassIndex = transform {
            for (chunk, out) in variants.chunks().zip(out.chunks_mut(CHUNK)) {
                self.chunk_classes(&Walk::new(&slots, chunk), iterations.clone(), out);
            }
        } else {
            for (chunk, out) in variants.chunks().zip(out.chunks_mut(CHUNK * k)) {
                self.chunk_margins(&Walk::new(&slots, chunk), iterations.clone(), 0..k, out);
                // One row at a time, as `predict_row_into` transforms: a
                // single row never goes parallel.
                for row in out.chunks_exact_mut(k) {
                    transform.apply(row, k);
                }
            }
        }
        Ok(())
    }

    /// Write bounds on the margins of every row in a box of feature values
    /// into `out_lower` and `out_upper` (one per output,
    /// [`Self::n_outputs`]): every row `x` with `lower[f] <= x[f] <=
    /// upper[f]` for every feature has margins
    /// ([`Self::predict_margin_row_into`]) within them. A feature whose
    /// bounds are both `NaN` is missing in every such row; an infinite
    /// endpoint leaves that side unbounded. A box of one row (`lower ==
    /// upper`) bounds that row's margins exactly, both bounds equal to them.
    ///
    /// Each tree adds the least and greatest of its weighted leaf values
    /// over the leaves some row in the box reaches (per output for vector
    /// leaves), with the row margins' `f32` arithmetic in their order (and a
    /// shrunk model's shrink steps): sound because `f32` rounding is
    /// monotone. A `gblinear` model bounds each feature's term by its
    /// endpoints (a zero weight adds 0 even on an unbounded feature). The
    /// bounds are as tight as each tree's reachable leaves allow, not the
    /// tightest over the box: trees are bounded independently. Where a sum
    /// meets opposite infinities (a term overflowing `f32` against an
    /// unbounded feature), the bound widens to `-inf` or `+inf` rather than
    /// `NaN`. Allocates nothing and never enters rayon.
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] when `lower` or `upper` does not
    /// hold one value per feature or an output buffer one per output,
    /// [`HessboostError::InvalidParameter`] (`bounds`) for a feature with
    /// `NaN` in one bound only, `lower > upper`, a lower bound of `+inf` or
    /// an upper bound of `-inf` (no row has such a value), or an interval
    /// (`lower != upper`) on a categorical feature (one a tree splits by
    /// category: categorical features must be fixed),
    /// [`HessboostError::IncompatibleModel`] (`linear_tree`) for a model
    /// with linear leaves, plus the errors of
    /// [`Self::predict_margin_row_into`] for `iterations`.
    ///
    /// ```
    /// use hessboost::prelude::*;
    ///
    /// # fn main() -> Result<()> {
    /// let x: Vec<f32> = (0..120).map(|i| (i % 40) as f32).collect();
    /// let y: Vec<f32> = x.chunks(3).map(|row| row[0] - row[2]).collect();
    /// let dtrain = DMatrix::from_dense(&x, 40, 3)?.with_labels(&y)?;
    /// let model = train(&TrainingParams::default(), &dtrain, 5)?;
    ///
    /// // Feature 0 in [5, 20], feature 1 missing, feature 2 at least 0.
    /// let lower = [5.0, f32::NAN, 0.0];
    /// let upper = [20.0, f32::NAN, f32::INFINITY];
    /// let (mut low, mut high) = ([0.0], [0.0]);
    /// model.margin_bounds_into(&lower, &upper, Iterations::Best, &mut low, &mut high)?;
    /// let margin = model.predict_margin_row(&[12.5, f32::NAN, 7.0], Iterations::Best)?;
    /// assert!(low[0] <= margin && margin <= high[0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn margin_bounds_into(
        &self,
        lower: &[f32],
        upper: &[f32],
        iterations: impl Into<Iterations>,
        out_lower: &mut [f32],
        out_upper: &mut [f32],
    ) -> Result<()> {
        let (iterations, point) = self.bounds_request(lower, upper, iterations.into())?;
        let k = self.n_outputs;
        check_len("lower margin bounds", out_lower.len(), k)?;
        check_len("upper margin bounds", out_upper.len(), k)?;
        if let Some(linear) = &self.linear {
            self.linear_bounds(linear, lower, upper, point, out_lower, out_upper);
            return Ok(());
        }
        let start = match &self.shrinkage {
            Some(shrinkage) => shrinkage.base_scores(),
            None => &self.base_score[..],
        };
        out_lower.copy_from_slice(start);
        out_upper.copy_from_slice(start);
        let per = self.trees_per_iteration();
        let parallel = self.num_parallel_tree;
        let vector = self.has_vector_leaves();
        for i in iterations {
            if let Some(shrinkage) = &self.shrinkage {
                // Coefficients are positive, so shrinking keeps the order.
                let factor = shrinkage.factors()[i];
                shrink_margins(out_lower, factor);
                shrink_margins(out_upper, factor);
            }
            for t in i * per..(i + 1) * per {
                let weight = self.variant_tree_weight(t);
                let tree = &self.trees[t];
                if vector {
                    for first in (0..k).step_by(OUTPUT_CHUNK) {
                        let outputs = first..(first + OUTPUT_CHUNK).min(k);
                        let mut least = [f32::INFINITY; OUTPUT_CHUNK];
                        let mut most = [f32::NEG_INFINITY; OUTPUT_CHUNK];
                        reach(tree, lower, upper, 0, &mut |nid| {
                            let leaf = &tree.leaf_vector(nid)[outputs.clone()];
                            for ((&x, lo), hi) in leaf.iter().zip(&mut least).zip(&mut most) {
                                let c = weight * x;
                                *lo = lo.min(c);
                                *hi = hi.max(c);
                            }
                        });
                        for (o, (&lo, &hi)) in outputs.zip(least.iter().zip(&most)) {
                            out_lower[o] = add_lower(out_lower[o], lo, point);
                            out_upper[o] = add_upper(out_upper[o], hi, point);
                        }
                    }
                } else {
                    let nodes = tree.nodes();
                    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                    reach(tree, lower, upper, 0, &mut |nid| {
                        let c = weight * nodes[nid].leaf_value;
                        lo = lo.min(c);
                        hi = hi.max(c);
                    });
                    let o = scalar_tree_output(t, parallel, k);
                    out_lower[o] = add_lower(out_lower[o], lo, point);
                    out_upper[o] = add_upper(out_upper[o], hi, point);
                }
            }
        }
        Ok(())
    }

    /// Check the variants against the model as [`Self::predict_margin_row_into`]
    /// checks a row, and resolve `iterations` as it does. Every scan runs
    /// on the calling thread.
    fn variants_request(
        &self,
        variants: &RowVariants<'_>,
        iterations: Iterations,
    ) -> Result<Range<usize>> {
        check_len(
            "prediction feature count",
            variants.base.len(),
            self.n_features,
        )?;
        if let Some(&f) = variants.features.iter().find(|&&f| f >= self.n_features) {
            return Err(HessboostError::invalid_param(
                "variants",
                format!(
                    "feature {f} is out of range for a model of {} features",
                    self.n_features
                ),
            ));
        }
        if rejects_dense(variants.base, f32::NAN) {
            return Err(HessboostError::invalid_data(
                "row",
                "non-missing feature values must be finite",
            ));
        }
        if rejects_dense(variants.values, f32::NAN) {
            return Err(HessboostError::invalid_data(
                "variants",
                "non-missing feature values must be finite",
            ));
        }
        self.margin_iterations(iterations)
    }

    /// Check a box of feature values (see [`Self::margin_bounds_into`]) and
    /// resolve `iterations` as the row methods do; also whether the box is a
    /// single row (every feature fixed or missing).
    fn bounds_request(
        &self,
        lower: &[f32],
        upper: &[f32],
        iterations: Iterations,
    ) -> Result<(Range<usize>, bool)> {
        check_len("lower bounds feature count", lower.len(), self.n_features)?;
        check_len("upper bounds feature count", upper.len(), self.n_features)?;
        let mut intervals = false;
        for (f, (&lo, &hi)) in lower.iter().zip(upper).enumerate() {
            if lo.is_nan() != hi.is_nan() {
                return Err(HessboostError::invalid_param(
                    "bounds",
                    format!("feature {f} has NaN in one bound only; NaN in both marks it missing"),
                ));
            }
            if lo > hi {
                return Err(HessboostError::invalid_param(
                    "bounds",
                    format!("feature {f}: lower bound {lo} is above upper bound {hi}"),
                ));
            }
            if lo == f32::INFINITY || hi == f32::NEG_INFINITY {
                return Err(HessboostError::invalid_param(
                    "bounds",
                    format!("feature {f}: [{lo}, {hi}] holds no finite value"),
                ));
            }
            intervals |= !lo.is_nan() && lo != hi;
        }
        if self.trees.iter().any(|tree| tree.linear_leaves().is_some()) {
            return Err(HessboostError::incompatible_model(
                "linear_tree",
                "margin bounds are not defined for models with linear leaves",
            ));
        }
        if intervals {
            let categorical = self
                .trees
                .iter()
                .flat_map(RegTree::nodes)
                .filter(|node| !node.is_leaf() && node.is_categorical)
                .map(|node| node.split_feature as usize)
                .find(|&f| !lower[f].is_nan() && lower[f] != upper[f]);
            if let Some(f) = categorical {
                return Err(HessboostError::invalid_param(
                    "bounds",
                    format!(
                        "feature {f} is categorical: categorical features must be fixed \
                         (lower == upper)"
                    ),
                ));
            }
        }
        let iterations = self.margin_iterations(iterations)?;
        Ok((iterations, !intervals))
    }

    /// Tree `t`'s weight in the row margins: `1` for a shrunk model, whose
    /// shrink steps scale its trees instead.
    #[inline]
    fn variant_tree_weight(&self, t: usize) -> f32 {
        if self.shrinkage.is_some() {
            1.0
        } else {
            self.tree_weight(t)
        }
    }

    /// The margins of outputs `outputs` of the (at most [`CHUNK`]) variants
    /// of `walk`'s chunk from the resolved `iterations`, into `out`
    /// (`[variant][output]`), with the row margins' arithmetic cell by cell:
    /// the intercept, then gblinear's bias and feature terms in feature
    /// order, or every tree's weighted value in tree order per output (a
    /// shrunk model's shrink before each iteration's trees). One margin per
    /// variant accumulates in a whole stack block of [`CHUNK`] lanes, which
    /// leaves add to at once ([`add_masked`]).
    fn chunk_margins(
        &self,
        walk: &Walk<'_, '_>,
        iterations: Range<usize>,
        outputs: Range<usize>,
        out: &mut [f32],
    ) {
        let n = walk.chunk.len();
        debug_assert_eq!(out.len(), n * outputs.len());
        if let Some(linear) = &self.linear {
            self.linear_variant_margins(linear, walk, &outputs, out);
        } else if outputs.len() == 1 {
            let mut lanes = [0.0f32; CHUNK];
            self.tree_margins(walk, iterations, outputs, &mut lanes);
            out.copy_from_slice(&lanes[..n]);
        } else {
            self.tree_margins(walk, iterations, outputs, out);
        }
    }

    /// [`Self::chunk_margins`] of a tree model into `acc`: rows of
    /// `outputs.len()` margins, one per variant of `walk`'s chunk, and for a
    /// single output possibly more rows (lanes no variant is in, which keep
    /// whatever they hold).
    fn tree_margins(
        &self,
        walk: &Walk<'_, '_>,
        iterations: Range<usize>,
        outputs: Range<usize>,
        acc: &mut [f32],
    ) {
        let width = outputs.len();
        let start = match &self.shrinkage {
            Some(shrinkage) => shrinkage.base_scores(),
            None => &self.base_score[..],
        };
        for row in acc.chunks_exact_mut(width) {
            row.copy_from_slice(&start[outputs.clone()]);
        }
        let k = self.n_outputs;
        let per = self.trees_per_iteration();
        let parallel = self.num_parallel_tree;
        let vector = self.has_vector_leaves();
        let all = walk.chunk.mask();
        for i in iterations {
            if let Some(shrinkage) = &self.shrinkage {
                shrink_margins(acc, shrinkage.factors()[i]);
            }
            // A scalar tree feeds one output; within an iteration the trees
            // of the outputs `outputs` are contiguous.
            let trees = if vector {
                i * per..(i + 1) * per
            } else {
                i * per + outputs.start * parallel..i * per + outputs.end * parallel
            };
            for t in trees {
                let weight = self.variant_tree_weight(t);
                let tree = &self.trees[t];
                let nodes = tree.nodes();
                if vector {
                    route(tree, walk, 0, all, &mut |nid, mask| {
                        let leaf = &tree.leaf_vector(nid)[outputs.clone()];
                        for v in bits(mask) {
                            for (m, &x) in acc[v * width..(v + 1) * width].iter_mut().zip(leaf) {
                                *m += weight * x;
                            }
                        }
                    });
                    continue;
                }
                let slot = scalar_tree_output(t, parallel, k) - outputs.start;
                match tree.linear_leaves() {
                    None if width == 1 => route(tree, walk, 0, all, &mut |nid, mask| {
                        add_masked(acc, mask, weight * nodes[nid].leaf_value);
                    }),
                    None => route(tree, walk, 0, all, &mut |nid, mask| {
                        let c = weight * nodes[nid].leaf_value;
                        for v in bits(mask) {
                            acc[v * width + slot] += c;
                        }
                    }),
                    Some(linear) => route(tree, walk, 0, all, &mut |nid, mask| {
                        let constant = nodes[nid].leaf_value;
                        for v in bits(mask) {
                            let value = linear
                                .predict(nid, constant, |f| present(walk.value(v, f as usize)));
                            acc[v * width + slot] += weight * value;
                        }
                    }),
                }
            }
        }
    }

    /// `multi:softmax`'s class of every variant of `walk`'s chunk into
    /// `out`: the first of equal largest margins, as the row methods pick
    /// it, from margins formed [`CLASS_CHUNK`] classes at a time on the
    /// stack.
    fn chunk_classes(&self, walk: &Walk<'_, '_>, iterations: Range<usize>, out: &mut [f32]) {
        let (n, k) = (walk.chunk.len(), self.n_outputs);
        let mut buffer = [0.0f32; CHUNK * CLASS_CHUNK];
        let mut best = [(0usize, f32::NAN); CHUNK];
        for first in (0..k).step_by(CLASS_CHUNK) {
            let classes = first..(first + CLASS_CHUNK).min(k);
            let width = classes.len();
            let margins = &mut buffer[..n * width];
            self.chunk_margins(walk, iterations.clone(), classes.clone(), margins);
            for (best, row) in best.iter_mut().zip(margins.chunks_exact(width)) {
                for (class, &margin) in classes.clone().zip(row) {
                    if class == 0 || margin > best.1 {
                        *best = (class, margin);
                    }
                }
            }
        }
        for (o, &(class, _)) in out.iter_mut().zip(&best) {
            *o = class as f32;
        }
    }

    /// gblinear's margins of outputs `outputs` of the variants of `chunk`
    /// into `out`, as the row margins form them: intercept plus bias, then
    /// each present feature's `f64` product added in `f32` in feature order.
    /// The terms before the first varying feature are the base row's, formed
    /// once.
    fn linear_variant_margins(
        &self,
        linear: &LinearModel,
        walk: &Walk<'_, '_>,
        outputs: &Range<usize>,
        out: &mut [f32],
    ) {
        let width = outputs.len();
        let k = self.n_outputs;
        let add = |margins: &mut [f32], f: usize, x: f32| {
            if x.is_nan() {
                return;
            }
            let weights = &linear.weights()[f * k..(f + 1) * k][outputs.clone()];
            for (m, &w) in margins.iter_mut().zip(weights) {
                *m += (f64::from(w) * f64::from(x)) as f32;
            }
        };
        let chunk = &walk.chunk;
        let shared = chunk.features.iter().copied().min().unwrap_or(0);
        let (first, rest) = out.split_at_mut(width);
        for ((m, &b), &bias) in first
            .iter_mut()
            .zip(&self.base_score[outputs.clone()])
            .zip(&linear.bias()[outputs.clone()])
        {
            *m = b + bias;
        }
        for f in 0..shared {
            add(first, f, chunk.base[f]);
        }
        for row in rest.chunks_exact_mut(width) {
            row.copy_from_slice(first);
        }
        for (v, row) in out.chunks_exact_mut(width).enumerate() {
            for f in shared..self.n_features {
                add(row, f, walk.value(v, f));
            }
        }
    }

    /// gblinear's margin bounds over the box `lower..=upper`: each present
    /// feature's term bounded by its terms at the two endpoints, formed and
    /// added as the row margins form them (a zero weight adds 0 at an
    /// infinite endpoint, where its product is undefined), with
    /// [`add_lower`]/[`add_upper`] (`point`: the box is one row).
    fn linear_bounds(
        &self,
        linear: &LinearModel,
        lower: &[f32],
        upper: &[f32],
        point: bool,
        out_lower: &mut [f32],
        out_upper: &mut [f32],
    ) {
        let k = self.n_outputs;
        for (((lo, hi), &b), &bias) in out_lower
            .iter_mut()
            .zip(out_upper.iter_mut())
            .zip(&self.base_score)
            .zip(linear.bias())
        {
            *lo = b + bias;
            *hi = *lo;
        }
        for (f, (&a, &z)) in lower.iter().zip(upper).enumerate() {
            if a.is_nan() {
                continue;
            }
            let weights = &linear.weights()[f * k..(f + 1) * k];
            for ((lo, hi), &w) in out_lower.iter_mut().zip(out_upper.iter_mut()).zip(weights) {
                let term = |x: f32| {
                    if w == 0.0 && x.is_infinite() {
                        0.0
                    } else {
                        (f64::from(w) * f64::from(x)) as f32
                    }
                };
                let (s, e) = (term(a), term(z));
                *lo = add_lower(*lo, s.min(e), point);
                *hi = add_upper(*hi, s.max(e), point);
            }
        }
    }
}
