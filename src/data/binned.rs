//! [`BinnedMatrix`]: a training matrix holding histogram bins instead of
//! feature values.

use crate::data::{DMatrix, FeatureType, GroupInfo, MetaInfo, SketchedRows};
use crate::error::{HessboostError, Result};

/// A training matrix that holds its rows' histogram bins instead of their
/// feature values, as XGBoost's `QuantileDMatrix` does: `tree_method =
/// hist` growth reads nothing else, so a matrix binned once trains without
/// its values in memory.
///
/// [`BinnedMatrix::new`] bins a [`DMatrix`] with `max_bin`'s cuts from its
/// values, sample weights, feature types, and missing value, the
/// index that matrix's own `hist` training uses (and shares with it), so
/// training on the binned matrix is training on the `DMatrix`, bit for bit.
/// The weights and feature types are therefore fixed; the labels and the
/// other metadata training reads besides can change
/// ([`with_labels`](Self::with_labels) and the other setters).
///
/// Only [`Trainer::new`](crate::training::Trainer::new) and
/// [`train`](crate::training::train) take one. Training refuses what would
/// read the values: another `max_bin`, a `tree_method` other than `hist`, a
/// booster other than `gbtree`, continuation (`init_model`), `device =
/// cuda`, and every option outside the `hist` growth it can run (linear
/// leaves, refresh, model shrinkage and Langevin noise, reuse penalties; the
/// error names them). Evaluate and predict on a `DMatrix`.
///
/// ```
/// use hessboost::data::BinnedMatrix;
/// use hessboost::prelude::*;
///
/// # fn main() -> Result<()> {
/// let x: Vec<f32> = (0..400).map(|i| (i % 37) as f32).collect();
/// let y: Vec<f32> = x.iter().map(|v| (v * 0.3).sin()).collect();
/// let data = DMatrix::from_dense(&x, 400, 1)?.with_labels(&y)?;
/// let params = TrainingParams::builder().max_bin(64).build()?;
/// let binned = BinnedMatrix::new(&data, 64)?;
/// let from_bins = train(&params, &binned, 10)?;
/// assert_eq!(
///     from_bins.predict(&data, Iterations::Best)?,
///     train(&params, &data, 10)?.predict(&data, Iterations::Best)?
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BinnedMatrix {
    /// The source's metadata over its index ([`DMatrix::without_values`]).
    matrix: DMatrix,
    max_bin: usize,
}

impl BinnedMatrix {
    /// Bin `data` with at most `max_bin` bins per numerical feature (a
    /// categorical one keeps a bin per category), keeping its metadata. The
    /// index is `data`'s own: built here unless an earlier training or
    /// binning of `data` (or of a matrix sharing its values) built it, and
    /// freed once nothing holds it.
    ///
    /// Every row must be one the cuts are sketched from, so that its bins
    /// route it through a tree as its values would: a row with a zero
    /// sample weight is refused (for a [`ViewCuts::Source`] view, a row its
    /// source weighs zero); select the weighted rows instead.
    ///
    /// [`ViewCuts::Source`]: crate::data::ViewCuts::Source
    pub fn new(data: &DMatrix, max_bin: usize) -> Result<Self> {
        crate::check::ensure(
            "max_bin",
            max_bin >= 2,
            format!("must be >= 2, got {max_bin}"),
        )?;
        if data.sketched_rows() != SketchedRows::All {
            return Err(HessboostError::invalid_data(
                "weights",
                "a binned matrix needs the cuts sketched from every row, so no row may have \
                 a zero sample weight (its source's, in a view with its source's cuts)",
            ));
        }
        let index = data.hist_index(max_bin);
        Ok(BinnedMatrix {
            matrix: data.without_values(index, max_bin),
            max_bin,
        })
    }

    /// The matrix training reads.
    pub(crate) fn matrix(&self) -> &DMatrix {
        &self.matrix
    }

    /// The `max_bin` the matrix was binned with, which training must use.
    pub fn max_bin(&self) -> usize {
        self.max_bin
    }

    /// Number of rows (instances).
    pub fn n_rows(&self) -> usize {
        self.matrix.n_rows()
    }

    /// Number of columns (features).
    pub fn n_cols(&self) -> usize {
        self.matrix.n_cols()
    }

    /// Labels, row-major `[row][target]` ([`DMatrix::labels`]).
    pub fn labels(&self) -> Option<&[f32]> {
        self.matrix.labels()
    }

    /// Label columns per row ([`DMatrix::n_targets`]).
    pub fn n_targets(&self) -> usize {
        self.matrix.n_targets()
    }

    /// Survival lower bounds ([`DMatrix::label_lower_bound`]).
    pub fn label_lower_bound(&self) -> Option<&[f32]> {
        self.matrix.label_lower_bound()
    }

    /// Survival upper bounds ([`DMatrix::label_upper_bound`]).
    pub fn label_upper_bound(&self) -> Option<&[f32]> {
        self.matrix.label_upper_bound()
    }

    /// The sample weights training reads, `data`'s, fixed at binning (a
    /// [`ViewCuts::Source`](crate::data::ViewCuts::Source) view's own,
    /// although its source's sketched its cuts).
    pub fn weights(&self) -> Option<&[f32]> {
        self.matrix.weights()
    }

    /// Per-row base margins ([`DMatrix::base_margin`]).
    pub fn base_margin(&self) -> Option<&[f32]> {
        self.matrix.base_margin()
    }

    /// Ranking query groups ([`DMatrix::group`]).
    pub fn group(&self) -> Option<&GroupInfo> {
        self.matrix.group()
    }

    /// Split groups ([`DMatrix::split_groups`]).
    pub fn split_groups(&self) -> Option<&[u64]> {
        self.matrix.split_groups()
    }

    /// The feature types the cuts were computed for.
    pub fn feature_types(&self) -> &[FeatureType] {
        self.matrix.feature_types()
    }

    /// Column-sampling weights ([`DMatrix::feature_weights`]).
    pub fn feature_weights(&self) -> Option<&[f32]> {
        self.matrix.feature_weights()
    }

    /// The metadata objectives and metrics read ([`DMatrix::info`]).
    pub fn info(&self) -> MetaInfo<'_> {
        self.matrix.info()
    }

    /// [`DMatrix::with_labels`].
    pub fn with_labels(self, labels: &[f32]) -> Result<Self> {
        self.map(|m| m.with_labels(labels))
    }

    /// [`DMatrix::with_label_matrix`].
    pub fn with_label_matrix(self, labels: &[f32], n_targets: usize) -> Result<Self> {
        self.map(|m| m.with_label_matrix(labels, n_targets))
    }

    /// [`DMatrix::with_label_bounds`].
    pub fn with_label_bounds(self, lower: &[f32], upper: &[f32]) -> Result<Self> {
        self.map(|m| m.with_label_bounds(lower, upper))
    }

    /// [`DMatrix::with_base_margin`].
    pub fn with_base_margin(self, base_margin: &[f32]) -> Result<Self> {
        self.map(|m| m.with_base_margin(base_margin))
    }

    /// [`DMatrix::with_group_sizes`].
    pub fn with_group_sizes(self, sizes: &[usize]) -> Result<Self> {
        self.map(|m| m.with_group_sizes(sizes))
    }

    /// [`DMatrix::with_split_groups`].
    pub fn with_split_groups(self, groups: &[u64]) -> Result<Self> {
        self.map(|m| m.with_split_groups(groups))
    }

    /// [`DMatrix::with_feature_weights`].
    pub fn with_feature_weights(self, weights: &[f32]) -> Result<Self> {
        self.map(|m| m.with_feature_weights(weights))
    }

    /// This matrix with `set` applied to its metadata.
    fn map(self, set: impl FnOnce(DMatrix) -> Result<DMatrix>) -> Result<Self> {
        Ok(BinnedMatrix {
            matrix: set(self.matrix)?,
            max_bin: self.max_bin,
        })
    }
}
