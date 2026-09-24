//! CatBoost-style **ordered target statistics**: an opt-in encoder that turns
//! categorical columns into numeric ones. This is an extension beyond XGBoost;
//! nothing in training uses it unless you call it.
//!
//! # Encoding
//!
//! With prior `P` and prior weight `a`, a category `c` is encoded as a smoothed
//! target mean. Following Prokhorenkova et al., *CatBoost: unbiased boosting
//! with categorical features* (`NeurIPS` 2018), training rows only see the
//! rows that precede them in a random permutation `σ`:
//!
//! ```text
//! train row i:  (Σ_{j: σ(j) < σ(i), x_j = c} y_j + a·P) / (#{j: σ(j) < σ(i), x_j = c} + a)
//! inference:    (Σ_{j: x_j = c} y_j + a·P) / (#{j: x_j = c} + a)
//! ```
//!
//! The preceding rows never include the row itself, so apart from the default
//! prior (see below) a row's own target never enters its training encoding: a
//! tree cannot learn "this encoded value means this row's label" the way it
//! can with a plain in-sample target mean. Inference ([`FittedTargetEncoder::transform`])
//! uses statistics over every training row. Categories never seen in training
//! map to `P`; a missing category stays missing.
//!
//! # Parameters
//!
//! - `prior` (`P`): defaults to the mean training label (the positive rate
//!   for 0/1 labels). Every training row contributes `1/n` to that default;
//!   set [`OrderedTargetEncoderBuilder::prior`] to a fixed value to keep the
//!   encoding strictly free of the row's own target.
//! - `prior_weight` (`a`, default 1): pseudo-count of the prior; must be `> 0`.
//! - `permutations` (default 1) and `seed` (default 0): the training encoding
//!   is the average over `permutations` independent seeded permutations, all
//!   shared by every encoded column (CatBoost also shares one permutation
//!   across features). CatBoost itself trains different trees on different
//!   permutations; a static encoding cannot switch per tree, so the choice
//!   here is averaging. Averaging lowers the variance of rows that fall early
//!   in a permutation and never uses a row's own target, but as the count grows
//!   the average converges to a leave-one-out mean: within a category it
//!   becomes a decreasing function of the row's own label, which trees can
//!   exploit on the training set (the target shift that ordered statistics
//!   exist to avoid). The default is therefore one permutation; a few (2–4)
//!   trade a little of that protection for smoother encodings.
//!
//! # Targets
//!
//! Regression and binary labels ([`TargetKind`]) use the formula above.
//! Multiclass labels are rejected: the mean of class indices imposes an
//! arbitrary order on the classes, and CatBoost's per-class encodings would
//! replace one column with `num_class` columns and renumber every later
//! feature. To get per-class statistics, fit one encoder per class on 0/1
//! indicator labels. Instance weights are ignored (unweighted counts, as in
//! CatBoost's counters).
//!
//! ```
//! use hessboost::data::{OrderedTargetEncoder, TargetKind};
//! use hessboost::prelude::*;
//!
//! # fn main() -> Result<()> {
//! // Column 0 holds category codes, column 1 is numeric.
//! let x = [0.0, 0.5, 1.0, 0.1, 0.0, 0.9, 2.0, 0.3, 1.0, 0.7, 0.0, 0.2];
//! let y = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
//! let dtrain = DMatrix::from_dense(&x, 6, 2)?
//!     .with_labels(&y)?
//!     .with_feature_types(&[FeatureType::Categorical, FeatureType::Numerical])?;
//!
//! let encoder = OrderedTargetEncoder::builder()
//!     .target(TargetKind::Binary)
//!     .seed(7)
//!     .build()?;
//! let (encoded, fitted) = encoder.fit_transform(&dtrain, &[0])?;
//! assert_eq!(encoded.feature_types()[0], FeatureType::Numerical);
//!
//! // Category 0 has labels {1, 0, 1}; the prior is the positive rate 0.5.
//! assert_eq!(fitted.encode(0, 0), Some((2.0 + 0.5) / (3.0 + 1.0)));
//! // Category 9 was never seen: it maps to the prior.
//! assert_eq!(fitted.encode(0, 9), Some(0.5));
//! # Ok(())
//! # }
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::data::dmatrix::check_len;
use crate::data::{DMatrix, FeatureType};
use crate::error::{HessboostError, Result};

/// Dense category id of a row whose category is missing. Category codes are
/// below `2^32` (enforced by [`DMatrix::with_feature_types`]) and dense ids
/// count distinct codes, so no real id reaches this value.
const NO_CATEGORY: u32 = u32::MAX;

/// Which labels an [`OrderedTargetEncoder`] accepts. Both kinds encode the
/// smoothed target mean; the kind only decides label validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TargetKind {
    /// Any finite labels; the default prior is the label mean.
    #[default]
    Regression,
    /// Labels must be 0 or 1; the default prior is the positive rate.
    /// Multiclass labels are rejected (see the [module docs](self)).
    Binary,
}

/// Unfitted ordered target-statistics encoder. Build with
/// [`OrderedTargetEncoder::builder`], then [`fit_transform`](Self::fit_transform)
/// the training matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedTargetEncoder {
    prior_weight: f64,
    prior: Option<f64>,
    permutations: usize,
    seed: u64,
    target: TargetKind,
}

/// Builder for [`OrderedTargetEncoder`]; defaults are documented on each setter.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedTargetEncoderBuilder {
    encoder: OrderedTargetEncoder,
}

impl Default for OrderedTargetEncoderBuilder {
    fn default() -> Self {
        OrderedTargetEncoderBuilder {
            encoder: OrderedTargetEncoder {
                prior_weight: 1.0,
                prior: None,
                permutations: 1,
                seed: 0,
                target: TargetKind::Regression,
            },
        }
    }
}

impl OrderedTargetEncoderBuilder {
    /// Pseudo-count `a` of the prior (default 1). Must be finite and `> 0`.
    #[must_use]
    pub fn prior_weight(mut self, weight: f64) -> Self {
        self.encoder.prior_weight = weight;
        self
    }

    /// Fixed prior `P` (default: the mean training label). Must be finite as
    /// an `f32`, since unseen categories encode to it.
    #[must_use]
    pub fn prior(mut self, prior: f64) -> Self {
        self.encoder.prior = Some(prior);
        self
    }

    /// Number of averaged permutations for the training encoding (default 1,
    /// must be `>= 1`). See the [module docs](self) for the trade-off.
    #[must_use]
    pub fn permutations(mut self, permutations: usize) -> Self {
        self.encoder.permutations = permutations;
        self
    }

    /// Seed of the permutation RNG (default 0).
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.encoder.seed = seed;
        self
    }

    /// Label kind (default [`TargetKind::Regression`]).
    #[must_use]
    pub fn target(mut self, target: TargetKind) -> Self {
        self.encoder.target = target;
        self
    }

    /// Validate the settings and produce the encoder.
    pub fn build(self) -> Result<OrderedTargetEncoder> {
        let e = &self.encoder;
        if !e.prior_weight.is_finite() || e.prior_weight <= 0.0 {
            return Err(HessboostError::invalid_param(
                "prior_weight",
                "must be finite and > 0",
            ));
        }
        if e.prior.is_some_and(|p| !(p as f32).is_finite()) {
            return Err(HessboostError::invalid_param(
                "prior",
                "must be finite and within the f32 range",
            ));
        }
        if e.permutations == 0 {
            return Err(HessboostError::invalid_param(
                "permutations",
                "must be >= 1",
            ));
        }
        Ok(self.encoder)
    }
}

/// Category codes of one encoded column, compacted to dense ids.
struct ColumnCodes {
    /// Distinct category codes, ascending.
    categories: Vec<u32>,
    /// Per-row index into `categories`, or [`NO_CATEGORY`] when missing.
    ids: Vec<u32>,
}

impl OrderedTargetEncoder {
    /// Start a builder with the defaults (`prior_weight = 1`, mean-label prior,
    /// one permutation, seed 0, regression labels).
    pub fn builder() -> OrderedTargetEncoderBuilder {
        OrderedTargetEncoderBuilder::default()
    }

    /// Fit statistics on the labelled training matrix `data` and encode its
    /// `columns`, which must be distinct and [`FeatureType::Categorical`].
    ///
    /// Returns the training matrix with those columns replaced by their
    /// ordered encodings (now [`FeatureType::Numerical`]; everything else,
    /// storage kind and metadata included, is kept) and the fitted encoder for
    /// evaluation and test data. The result uses a NaN missing sentinel;
    /// entries missing in `data` stay missing.
    pub fn fit_transform(
        &self,
        data: &DMatrix,
        columns: &[usize],
    ) -> Result<(DMatrix, FittedTargetEncoder)> {
        let n_rows = data.n_rows();
        let labels = data.labels().ok_or_else(|| {
            HessboostError::invalid_param("labels", "ordered target statistics need labels")
        })?;
        if labels.len() != n_rows {
            return Err(HessboostError::invalid_param(
                "labels",
                "ordered target statistics need exactly one label per row",
            ));
        }
        if self.target == TargetKind::Binary && labels.iter().any(|&y| y != 0.0 && y != 1.0) {
            return Err(HessboostError::invalid_param(
                "labels",
                "TargetKind::Binary needs 0/1 labels (multiclass targets are not supported)",
            ));
        }
        let slots = column_slots(data, columns)?;
        let prior = self
            .prior
            .unwrap_or_else(|| labels.iter().map(|&y| f64::from(y)).sum::<f64>() / n_rows as f64);
        let a = self.prior_weight;
        let codes = collect_codes(data, &slots, columns.len());

        let mut encodings = vec![vec![0f64; n_rows]; codes.len()];
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut order: Vec<usize> = (0..n_rows).collect();
        for _ in 0..self.permutations {
            order.shuffle(&mut rng);
            encodings
                .par_iter_mut()
                .zip(codes.par_iter())
                .for_each(|(enc, col)| ordered_pass(&order, col, labels, prior, a, enc));
        }

        let scale = 1.0 / self.permutations as f64;
        let encoded = data
            .map_values(|row, col, v| match slots[col] {
                Some(s) => (encodings[s][row] * scale) as f32,
                None => v,
            })
            .with_feature_types(&numeric_types(data, columns.iter().copied()))?;

        let columns = codes
            .into_iter()
            .zip(columns)
            .map(|(col, &column)| ColumnEncoding::fit(column, col, labels, prior, a))
            .collect();
        let fitted = FittedTargetEncoder {
            n_cols: data.n_cols(),
            prior: prior as f32,
            columns,
        };
        Ok((encoded, fitted))
    }
}

/// Map each column index to its position in `columns`, rejecting duplicates,
/// out-of-range, and non-categorical columns.
fn column_slots(data: &DMatrix, columns: &[usize]) -> Result<Vec<Option<usize>>> {
    if columns.is_empty() {
        return Err(HessboostError::invalid_param(
            "columns",
            "name at least one categorical column to encode",
        ));
    }
    let mut slots = vec![None; data.n_cols()];
    for (slot, &col) in columns.iter().enumerate() {
        if col >= data.n_cols() {
            return Err(HessboostError::FeatureOutOfBounds {
                index: col,
                num_features: data.n_cols(),
            });
        }
        if slots[col].replace(slot).is_some() {
            return Err(HessboostError::invalid_param(
                "columns",
                format!("column {col} is listed twice"),
            ));
        }
        require_categorical(data, col)?;
    }
    Ok(slots)
}

fn require_categorical(data: &DMatrix, col: usize) -> Result<()> {
    if data.feature_types()[col] == FeatureType::Categorical {
        Ok(())
    } else {
        Err(HessboostError::invalid_param(
            "columns",
            format!("column {col} is not categorical; mark it with with_feature_types"),
        ))
    }
}

/// `data`'s feature types with `columns` switched to numerical.
fn numeric_types(data: &DMatrix, columns: impl IntoIterator<Item = usize>) -> Vec<FeatureType> {
    let mut types = data.feature_types().to_vec();
    for col in columns {
        types[col] = FeatureType::Numerical;
    }
    types
}

/// Read every encoded column's category codes in one pass over the rows and
/// compact them to dense ids.
fn collect_codes(data: &DMatrix, slots: &[Option<usize>], n_encoded: usize) -> Vec<ColumnCodes> {
    let mut raw = vec![vec![NO_CATEGORY; data.n_rows()]; n_encoded];
    data.for_each_entry(|row, col, v| {
        if let Some(s) = slots[col as usize] {
            // Categorical values are validated non-negative integers < 2^32.
            raw[s][row] = v as u32;
        }
    });
    raw.into_par_iter()
        .map(|mut ids| {
            let mut categories: Vec<u32> =
                ids.iter().copied().filter(|&c| c != NO_CATEGORY).collect();
            categories.sort_unstable();
            categories.dedup();
            for id in &mut ids {
                if *id != NO_CATEGORY {
                    // Every present code is in `categories` by construction.
                    *id = categories.binary_search(id).unwrap_or_default() as u32;
                }
            }
            ColumnCodes { categories, ids }
        })
        .collect()
}

/// Add one permutation's ordered encodings of `col` to `enc`: each row sees
/// the target sum and count of the rows before it in `order`.
fn ordered_pass(
    order: &[usize],
    col: &ColumnCodes,
    labels: &[f32],
    prior: f64,
    a: f64,
    enc: &mut [f64],
) {
    let mut sums = vec![0f64; col.categories.len()];
    let mut counts = vec![0f64; col.categories.len()];
    for &row in order {
        let id = col.ids[row];
        if id == NO_CATEGORY {
            continue;
        }
        let id = id as usize;
        enc[row] += smoothed_mean(sums[id], counts[id], prior, a);
        sums[id] += f64::from(labels[row]);
        counts[id] += 1.0;
    }
}

/// `(sum + a·P) / (count + a)`, evaluated as `sum / (count + a) + P · (a / (count + a))`
/// so that no `a·P` product can overflow or underflow for extreme `a`: an
/// empty count gives exactly `P`. The result is a convex combination of the
/// label mean and `P`, so it stays within the `f32` range of both.
fn smoothed_mean(sum: f64, count: f64, prior: f64, a: f64) -> f64 {
    let denom = count + a;
    sum / denom + prior * (a / denom)
}

/// Inference encodings of one encoded column. Only `f32` values are stored:
/// they are exactly what [`FittedTargetEncoder::transform`] emits, and they
/// survive a JSON round trip bit for bit (`f64` statistics would not with
/// `serde_json`'s default float parser).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ColumnEncoding {
    /// Feature index of the encoded column.
    column: usize,
    /// Category codes seen in training, strictly ascending.
    categories: Vec<u32>,
    /// `(sum_c + a·P) / (count_c + a)` per category.
    values: Vec<f32>,
}

impl ColumnEncoding {
    /// Accumulate the totals in row order, so they do not depend on the seed.
    fn fit(column: usize, codes: ColumnCodes, labels: &[f32], prior: f64, a: f64) -> Self {
        let mut sums = vec![0f64; codes.categories.len()];
        let mut counts = vec![0f64; codes.categories.len()];
        for (&id, &y) in codes.ids.iter().zip(labels) {
            if id != NO_CATEGORY {
                sums[id as usize] += f64::from(y);
                counts[id as usize] += 1.0;
            }
        }
        let values = sums
            .iter()
            .zip(&counts)
            .map(|(&sum, &count)| smoothed_mean(sum, count, prior, a) as f32)
            .collect();
        ColumnEncoding {
            column,
            categories: codes.categories,
            values,
        }
    }

    fn encode(&self, category: u32, prior: f32) -> f32 {
        match self.categories.binary_search(&category) {
            Ok(i) => self.values[i],
            Err(_) => prior,
        }
    }
}

/// A fitted ordered target-statistics encoder: the per-category encodings over
/// the full training set, used to encode evaluation and test data.
///
/// Serializable with serde (e.g. `serde_json`) so it can be stored next to a
/// model; deserialization validates the contents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "FittedRepr")]
pub struct FittedTargetEncoder {
    n_cols: usize,
    prior: f32,
    columns: Vec<ColumnEncoding>,
}

/// Unvalidated serde mirror of [`FittedTargetEncoder`].
#[derive(Deserialize)]
struct FittedRepr {
    n_cols: usize,
    prior: f32,
    columns: Vec<ColumnEncoding>,
}

impl TryFrom<FittedRepr> for FittedTargetEncoder {
    type Error = HessboostError;

    fn try_from(r: FittedRepr) -> Result<Self> {
        let bad = |reason: &str| HessboostError::ModelFormat(format!("target encoder: {reason}"));
        if !r.prior.is_finite() {
            return Err(bad("prior must be finite"));
        }
        if r.columns.is_empty() {
            return Err(bad("no encoded columns"));
        }
        // Sized by the supplied columns, not the untrusted `n_cols` header.
        let mut seen = std::collections::HashSet::with_capacity(r.columns.len());
        for c in &r.columns {
            if c.column >= r.n_cols || !seen.insert(c.column) {
                return Err(bad("encoded columns must be distinct and < n_cols"));
            }
            if c.values.len() != c.categories.len() {
                return Err(bad("categories and values differ in length"));
            }
            if c.categories.windows(2).any(|w| w[0] >= w[1]) {
                return Err(bad("categories must be strictly ascending"));
            }
            if c.values.iter().any(|v| !v.is_finite()) {
                return Err(bad("encoded values must be finite"));
            }
        }
        Ok(FittedTargetEncoder {
            n_cols: r.n_cols,
            prior: r.prior,
            columns: r.columns,
        })
    }
}

impl FittedTargetEncoder {
    /// Encode `data` with the full-training-set statistics. `data` must have
    /// the training column count and the encoded columns must be categorical.
    /// Unseen categories map to the prior; missing entries stay missing.
    /// Output conventions match [`OrderedTargetEncoder::fit_transform`].
    pub fn transform(&self, data: &DMatrix) -> Result<DMatrix> {
        check_len("target encoder feature count", data.n_cols(), self.n_cols)?;
        let mut slots = vec![None; self.n_cols];
        for (s, c) in self.columns.iter().enumerate() {
            require_categorical(data, c.column)?;
            slots[c.column] = Some(s);
        }
        data.map_values(|_, col, v| match slots[col] {
            Some(s) => self.columns[s].encode(v as u32, self.prior),
            None => v,
        })
        .with_feature_types(&numeric_types(data, self.columns()))
    }

    /// Inference encoding of `category` in encoded column `column`
    /// (`(sum + a·P) / (count + a)`, or the prior for unseen categories), or
    /// `None` when `column` is not encoded.
    pub fn encode(&self, column: usize, category: u32) -> Option<f32> {
        self.columns
            .iter()
            .find(|c| c.column == column)
            .map(|c| c.encode(category, self.prior))
    }

    /// The prior `P`, which unseen categories encode to.
    pub fn prior(&self) -> f32 {
        self.prior
    }

    /// Feature count of the training matrix.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Encoded column indices, in the order given to `fit_transform`.
    pub fn columns(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.columns.iter().map(|c| c.column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAT_NUM: [FeatureType; 2] = [FeatureType::Categorical, FeatureType::Numerical];

    /// Two columns (category code, numeric) with labels; `cats` may hold NaN.
    fn matrix(cats: &[f32], labels: &[f32]) -> DMatrix {
        let x: Vec<f32> = cats
            .iter()
            .enumerate()
            .flat_map(|(i, &c)| [c, i as f32])
            .collect();
        crate::test_support::labeled_dense(&x, cats.len(), 2, labels)
            .with_feature_types(&CAT_NUM)
            .unwrap()
    }

    fn column(m: &DMatrix, col: usize) -> Vec<Option<f32>> {
        (0..m.n_rows()).map(|r| m.get(r, col)).collect()
    }

    fn encoder(permutations: usize, seed: u64) -> OrderedTargetEncoder {
        OrderedTargetEncoder::builder()
            .prior(0.5)
            .permutations(permutations)
            .seed(seed)
            .build()
            .unwrap()
    }

    fn crafted() -> (Vec<f32>, Vec<f32>) {
        let cats: Vec<f32> = (0..40).map(|i| (i % 3) as f32).collect();
        let labels: Vec<f32> = (0..40).map(|i| ((i * 7 + 3) % 5) as f32 * 0.5).collect();
        (cats, labels)
    }

    #[test]
    fn training_encoding_never_uses_own_target() {
        let (cats, labels) = crafted();
        for permutations in [1, 3] {
            let enc = encoder(permutations, 11);
            let (base, _) = enc.fit_transform(&matrix(&cats, &labels), &[0]).unwrap();
            let base = column(&base, 0);
            let mut others_moved = false;
            for i in 0..labels.len() {
                let mut perturbed = labels.clone();
                perturbed[i] += 7.0;
                let (out, _) = enc.fit_transform(&matrix(&cats, &perturbed), &[0]).unwrap();
                let out = column(&out, 0);
                assert_eq!(
                    out[i].map(f32::to_bits),
                    base[i].map(f32::to_bits),
                    "row {i}"
                );
                others_moved |= out != base;
            }
            // The targets are really used: perturbing a row moves other rows.
            assert!(others_moved);
        }
    }

    #[test]
    fn single_permutation_follows_the_ordered_formula() {
        // One category with constant label 2: the row at position m of the
        // permutation is encoded (2m + a·P) / (m + a), whatever the order.
        let (a, p) = (1.5, 0.25);
        let n = 12;
        let enc = OrderedTargetEncoder::builder()
            .prior(p)
            .prior_weight(a)
            .seed(3)
            .build()
            .unwrap();
        let (out, _) = enc
            .fit_transform(&matrix(&vec![4.0; n], &vec![2.0; n]), &[0])
            .unwrap();
        let mut got: Vec<f32> = column(&out, 0).into_iter().map(Option::unwrap).collect();
        got.sort_by(f32::total_cmp);
        let want: Vec<f32> = (0..n)
            .map(|m| ((2.0 * m as f64 + a * p) / (m as f64 + a)) as f32)
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn inference_uses_all_training_rows_and_prior_for_unseen() {
        let nan = f32::NAN;
        let cats = [0.0, 1.0, 0.0, nan, 1.0, 0.0];
        let labels = [1.0, 4.0, 2.0, 9.0, 6.0, 0.5];
        let enc = OrderedTargetEncoder::builder()
            .prior_weight(2.0)
            .build()
            .unwrap();
        let (train, fitted) = enc.fit_transform(&matrix(&cats, &labels), &[0]).unwrap();
        let prior = labels.iter().map(|&y| f64::from(y)).sum::<f64>() / 6.0;
        assert_eq!(fitted.prior(), prior as f32);
        assert_eq!(train.get(3, 0), None);

        let test = matrix(&[1.0, 0.0, 5.0, nan], &[0.0; 4]);
        let out = fitted.transform(&test).unwrap();
        let c0 = ((1.0 + 2.0 + 0.5) + 2.0 * prior) / (3.0 + 2.0);
        let c1 = ((4.0 + 6.0) + 2.0 * prior) / (2.0 + 2.0);
        assert_eq!(
            column(&out, 0),
            vec![Some(c1 as f32), Some(c0 as f32), Some(prior as f32), None]
        );
        assert_eq!(column(&out, 1), column(&test, 1));
        assert_eq!(out.feature_types(), &[FeatureType::Numerical; 2]);
        assert_eq!(fitted.encode(0, 5), Some(prior as f32));
        assert_eq!(fitted.encode(1, 0), None);
    }

    #[test]
    fn seed_determines_the_training_encoding() {
        let (cats, labels) = crafted();
        let data = matrix(&cats, &labels);
        let run = |seed| column(&encoder(2, seed).fit_transform(&data, &[0]).unwrap().0, 0);
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn many_permutations_converge_to_leave_one_out() {
        // One category, alternating 0/1 labels. Averaging many permutations
        // approaches the leave-one-out mean (S - y_i + aP) / (n - 1 + a) up to
        // a monotone map, which separates the labels perfectly; one permutation
        // does not. This is why the default is a single permutation.
        let n = 20;
        let cats = vec![0.0; n];
        let labels: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
        let separated = |permutations| {
            let (out, _) = encoder(permutations, 1)
                .fit_transform(&matrix(&cats, &labels), &[0])
                .unwrap();
            let enc = column(&out, 0);
            let max_pos = (0..n)
                .filter(|&i| labels[i] == 1.0)
                .map(|i| enc[i].unwrap());
            let min_neg = (0..n)
                .filter(|&i| labels[i] == 0.0)
                .map(|i| enc[i].unwrap());
            max_pos.fold(f32::MIN, f32::max) < min_neg.fold(f32::MAX, f32::min)
        };
        assert!(separated(20_000));
        assert!(!separated(1));
    }

    #[test]
    fn csr_and_dense_inputs_encode_identically() {
        let nan = f32::NAN;
        let cats = [2.0, nan, 2.0, 7.0, nan, 7.0, 2.0];
        let labels = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let dense = matrix(&cats, &labels);
        let mut indptr = vec![0];
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for (i, &c) in cats.iter().enumerate() {
            if !c.is_nan() {
                indices.push(0);
                values.push(c);
            }
            indices.push(1);
            values.push(i as f32);
            indptr.push(values.len());
        }
        let csr = DMatrix::from_csr(indptr, indices, values, 2)
            .unwrap()
            .with_labels(&labels)
            .unwrap()
            .with_feature_types(&CAT_NUM)
            .unwrap();
        let enc = encoder(3, 9);
        let (d, fd) = enc.fit_transform(&dense, &[0]).unwrap();
        let (s, fs) = enc.fit_transform(&csr, &[0]).unwrap();
        assert_eq!(fd, fs);
        for col in 0..2 {
            assert_eq!(column(&d, col), column(&s, col));
        }
        assert_eq!(column(&s, 0)[1], None);
        assert!(s.csr_parts().is_some());
    }

    #[test]
    fn non_nan_missing_sentinel_is_preserved_and_cannot_collide() {
        // Sentinel 0: an all-zero-target category encodes to exactly 0.0,
        // which must not turn into a missing value.
        let x = [3.0, 0.0, 3.0, 5.0, 0.0, 1.0];
        let data = DMatrix::from_dense_with_missing(&x, 3, 2, 0.0)
            .unwrap()
            .with_labels(&[0.0, 0.0, 1.0])
            .unwrap()
            .with_feature_types(&CAT_NUM)
            .unwrap();
        let enc = OrderedTargetEncoder::builder().prior(0.0).build().unwrap();
        let (out, fitted) = enc.fit_transform(&data, &[0]).unwrap();
        assert_eq!(column(&out, 1), vec![None, Some(5.0), Some(1.0)]);
        assert_eq!(out.get(0, 0), Some(0.0));
        assert_eq!(fitted.transform(&data).unwrap().get(0, 0), Some(0.0));
    }

    #[test]
    fn serde_round_trip_preserves_transform_and_rejects_corruption() {
        let (cats, labels) = crafted();
        let data = matrix(&cats, &labels);
        let (_, fitted) = encoder(1, 0).fit_transform(&data, &[0]).unwrap();
        let json = serde_json::to_string(&fitted).unwrap();
        let back: FittedTargetEncoder = serde_json::from_str(&json).unwrap();
        assert_eq!(back, fitted);
        assert_eq!(
            column(&back.transform(&data).unwrap(), 0),
            column(&fitted.transform(&data).unwrap(), 0)
        );
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let rejects = |corrupt: fn(&mut serde_json::Value)| {
            let mut v = value.clone();
            corrupt(&mut v["columns"][0]);
            serde_json::from_value::<FittedTargetEncoder>(v).is_err()
        };
        assert!(rejects(|c| c["categories"][0] = 2.into()), "unsorted");
        assert!(
            rejects(|c| c["values"] = serde_json::json!([1.0])),
            "lengths"
        );
        assert!(rejects(|c| c["column"] = 2.into()), "column out of range");
    }

    #[test]
    fn deserialization_does_not_allocate_from_the_declared_width() {
        // A tiny document declaring `usize::MAX` columns: validation must not
        // allocate per declared column (no capacity-overflow panic), while
        // duplicate columns are still refused.
        let doc = |columns: &str| {
            format!(
                r#"{{"n_cols": {}, "prior": 0.5, "columns": [{columns}]}}"#,
                usize::MAX
            )
        };
        let column = r#"{"column": 7, "categories": [0, 1], "values": [0.25, 0.75]}"#;
        let wide: FittedTargetEncoder = serde_json::from_str(&doc(column)).unwrap();
        assert_eq!(wide.n_cols, usize::MAX);
        assert!(
            serde_json::from_str::<FittedTargetEncoder>(&doc(&format!("{column}, {column}")))
                .is_err()
        );
    }

    #[test]
    fn extreme_prior_weights_keep_the_smoothed_mean_finite() {
        // With `a = 1e308` the product `a·P` overflows to +inf, and with the
        // smallest positive `a` it underflows to 0. The encoding must still
        // stay the prior for an empty prefix, and the encoder must round-trip.
        let data = matrix(&[0.0], &[2.0]);
        for (a, prior, first, fitted_value) in [
            (1e308, None, 2.0, 2.0),
            (f64::from_bits(1), Some(0.25), 0.25, 2.0),
        ] {
            let mut builder = OrderedTargetEncoder::builder().prior_weight(a);
            if let Some(p) = prior {
                builder = builder.prior(p);
            }
            let (out, fitted) = builder.build().unwrap().fit_transform(&data, &[0]).unwrap();
            assert_eq!(out.get(0, 0), Some(first), "a = {a:e}");
            assert_eq!(fitted.encode(0, 0), Some(fitted_value), "a = {a:e}");
            let json = serde_json::to_string(&fitted).unwrap();
            assert_eq!(
                serde_json::from_str::<FittedTargetEncoder>(&json).unwrap(),
                fitted
            );
        }
        // A prior outside the f32 range could not be stored in the encoder.
        assert!(
            OrderedTargetEncoder::builder()
                .prior(1e300)
                .build()
                .is_err()
        );
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let (cats, labels) = crafted();
        let data = matrix(&cats, &labels);
        let enc = encoder(1, 0);
        let unlabeled = DMatrix::from_dense(&[0.0, 1.0], 1, 2)
            .unwrap()
            .with_feature_types(&CAT_NUM)
            .unwrap();
        assert!(enc.fit_transform(&unlabeled, &[0]).is_err());
        assert!(enc.fit_transform(&data, &[]).is_err());
        assert!(enc.fit_transform(&data, &[1]).is_err(), "numeric column");
        assert!(enc.fit_transform(&data, &[0, 0]).is_err());
        assert!(enc.fit_transform(&data, &[2]).is_err());
        let binary = OrderedTargetEncoder::builder()
            .target(TargetKind::Binary)
            .build()
            .unwrap();
        let multiclass = matrix(&[0.0, 1.0, 1.0], &[0.0, 2.0, 1.0]);
        assert!(binary.fit_transform(&multiclass, &[0]).is_err());
        assert!(
            OrderedTargetEncoder::builder()
                .prior_weight(0.0)
                .build()
                .is_err()
        );
        assert!(
            OrderedTargetEncoder::builder()
                .permutations(0)
                .build()
                .is_err()
        );
        assert!(
            OrderedTargetEncoder::builder()
                .prior(f64::NAN)
                .build()
                .is_err()
        );

        let (_, fitted) = enc.fit_transform(&data, &[0]).unwrap();
        let wide = DMatrix::from_dense(&[0.0; 3], 1, 3).unwrap();
        assert!(fitted.transform(&wide).is_err());
        let numeric = DMatrix::from_dense(&[0.0, 1.0], 1, 2).unwrap();
        assert!(fitted.transform(&numeric).is_err());
    }
}
