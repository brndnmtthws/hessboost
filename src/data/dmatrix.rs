//! The core dataset container: a feature matrix plus training metadata.

use crate::data::bin_cache::{BinCache, BinKey};
use crate::data::ghist::{GHistIndex, PARALLEL_BINNING_CELLS};
use crate::data::meta::{FeatureType, GroupInfo, LabelBounds, Labels, MetaInfo};
use crate::data::quantile::HistCuts;
use crate::error::{HessboostError, Result};
use rayon::prelude::*;
use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Dense inputs with at least this many values are validated and copied in
/// parallel: below it the serial scan and copy are faster than rayon's
/// dispatch (measured on a 192-core aarch64 host).
const PARALLEL_COPY_VALUES: usize = 1 << 22;

/// Values per parallel task of the dense validation scan and copy.
const PARALLEL_BLOCK: usize = 1 << 18;

/// Values per branch-free block of the dense validation scan: each block is
/// OR-reduced without early exit (which vectorizes), and the scan stops at
/// the first block holding a refused value.
const SCAN_BLOCK: usize = 256;

/// Returns `true` if `v` should be treated as missing given the sentinel
/// `missing`. NaN sentinels match any NaN. Otherwise an exact bit-compatible
/// equality is used (mirroring XGBoost's semantics).
#[inline]
pub(crate) fn is_missing(v: f32, missing: f32) -> bool {
    if missing.is_nan() {
        v.is_nan()
    } else {
        v == missing
    }
}

/// Length check shared by every constructor and metadata setter: `got` must
/// equal `expected`, else [`HessboostError::DimensionMismatch`].
#[inline]
pub(crate) fn check_len(what: &'static str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(HessboostError::dimension_mismatch(what, expected, got));
    }
    Ok(())
}

/// Validate CSR `indptr` against `nnz` stored entries: first offset 0,
/// monotonic offsets within bounds, terminal offset `== nnz`.
/// Caller must ensure `indptr` is non-empty (`from_csr` rejects that first).
fn check_csr(indptr: &[usize], nnz: usize) -> Result<()> {
    if indptr[0] != 0 {
        return Err(HessboostError::invalid_data(
            "csr indptr",
            "the first offset must be 0",
        ));
    }
    for pair in indptr.windows(2) {
        if pair[0] > pair[1] || pair[1] > nnz {
            return Err(HessboostError::invalid_data(
                "csr indptr",
                "offsets must be monotonic and within the values array",
            ));
        }
    }
    check_len("csr indptr terminal", indptr[indptr.len() - 1], nnz)
}

/// Reject any non-finite value of `values` as invalid input `name`.
fn check_finite(name: &'static str, values: &[f32], reason: &'static str) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(HessboostError::invalid_data(name, reason));
    }
    Ok(())
}

/// Reject weights that are negative or non-finite (`invalid`), or none of
/// which is positive (`none_positive`), as invalid input `name`.
fn check_weights(
    name: &'static str,
    weights: &[f32],
    invalid: &'static str,
    none_positive: &'static str,
) -> Result<()> {
    if weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(HessboostError::invalid_data(name, invalid));
    }
    if !weights.iter().any(|v| *v > 0.0) {
        return Err(HessboostError::invalid_data(name, none_positive));
    }
    Ok(())
}

/// Validate a row-major dense matrix: a non-empty shape matching
/// `data.len()`, every non-missing value finite. Returns whether `data` is
/// large enough to be copied in parallel.
fn check_dense(data: &[f32], n_rows: usize, n_cols: usize, missing: f32) -> Result<bool> {
    if n_rows == 0 || n_cols == 0 {
        return Err(HessboostError::EmptyDataset(
            "from_dense: zero rows or columns",
        ));
    }
    let expected = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| HessboostError::invalid_data("data", "n_rows * n_cols overflows usize"))?;
    check_len("dense data length", data.len(), expected)?;
    if rejects_dense_values(data, missing) {
        return Err(HessboostError::invalid_data(
            "data",
            "non-missing feature values must be finite",
        ));
    }
    Ok(parallel_values(data.len()))
}

/// Whether `len` dense values are scanned and copied in parallel.
fn parallel_values(len: usize) -> bool {
    len >= PARALLEL_COPY_VALUES && rayon::current_num_threads() > 1
}

/// Whether dense `values` hold what a dense matrix with the `missing`
/// sentinel refuses (a non-missing value that is not finite), in parallel
/// blocks for large inputs. Shared with the predictions of borrowed rows.
pub(crate) fn rejects_dense_values(values: &[f32], missing: f32) -> bool {
    if parallel_values(values.len()) {
        values
            .par_chunks(PARALLEL_BLOCK)
            .any(|chunk| rejects_dense(chunk, missing))
    } else {
        rejects_dense(values, missing)
    }
}

/// [`rejects_dense_values`] on the calling thread, whatever the length: what
/// a single prediction row is checked with, since the row methods never
/// enter rayon. With the NaN sentinel the only refused values are
/// infinities; other sentinels need the general test. Each [`SCAN_BLOCK`] is
/// reduced branch-free.
pub(crate) fn rejects_dense(values: &[f32], missing: f32) -> bool {
    if missing.is_nan() {
        values
            .chunks(SCAN_BLOCK)
            .any(|block| block.iter().fold(false, |bad, v| bad | v.is_infinite()))
    } else {
        values.chunks(SCAN_BLOCK).any(|block| {
            block
                .iter()
                .fold(false, |bad, &v| bad | ((v != missing) & !v.is_finite()))
        })
    }
}

/// Copy of `data`, in parallel [`PARALLEL_BLOCK`]s when `parallel`.
fn copy_dense(data: &[f32], parallel: bool) -> Vec<f32> {
    if !parallel {
        return data.to_vec();
    }
    let mut values = vec![0.0; data.len()];
    values
        .par_chunks_mut(PARALLEL_BLOCK)
        .zip(data.par_chunks(PARALLEL_BLOCK))
        .for_each(|(dst, src)| dst.copy_from_slice(src));
    values
}

/// Backing storage for the feature matrix, shared between clones: cloning
/// a [`DMatrix`] copies only its metadata, and a mutation of shared values
/// copies them first ([`Arc::make_mut`]), so a clone never changes its
/// source. Each array is its own `Arc<Vec<_>>`, so wrapping an owned `Vec`
/// copies nothing and a value-only rewrite keeps sharing the CSR structure.
/// A row view ([`DMatrix::select_rows`]) shares its source's storage and
/// reads it through its row map.
#[derive(Debug, Clone)]
enum Storage {
    /// Row-major dense matrix of `n_cols` values per storage row.
    Dense(Arc<Vec<f32>>),
    /// Compressed sparse row: `indptr` has one entry per storage row, plus one.
    Csr {
        indptr: Arc<Vec<usize>>,
        indices: Arc<Vec<u32>>,
        values: Arc<Vec<f32>>,
    },
}

/// Where a row view's histogram index takes its cuts
/// ([`DMatrix::select_rows_with_cuts`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ViewCuts {
    /// The view's own rows, as a copy of them would be cut: training on the
    /// view is training on such a copy, bit for bit.
    #[default]
    Own,
    /// The cuts the source matrix bins with (its sample weights, feature
    /// types and missing value): the view's rows are its source's bins,
    /// gathered, with no cuts computed and no value binned again, and every
    /// view of one source shares its splits' thresholds.
    Source,
}

/// Which rows of a matrix the cuts of its `hist` index were sketched from
/// ([`DMatrix::sketched_rows`]). A sketched row's values lie below their
/// feature's last cut and each of its categories has a bin, so its bins
/// route it through a tree split on those cuts as its values do. Binning
/// clamps an unsketched value beyond the last cut into the last bin, and an
/// unsketched category into bin 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SketchedRows {
    /// Every row.
    All,
    /// Every row with a nonzero sample weight: the others add nothing to
    /// the gradient sums, whatever leaf their margins come from.
    Weighted,
    /// Not every row with a nonzero weight: a [`ViewCuts::Source`] view of
    /// rows its source weighs zero.
    Partial,
}

/// The source a [`ViewCuts::Source`] view takes its cuts from: the source
/// matrix (without the metadata its binning does not read) and the view's
/// rows of it.
#[derive(Debug)]
struct CutSource {
    matrix: DMatrix,
    rows: Vec<usize>,
}

/// A dataset: features in dense or sparse form, plus labels (one or more
/// targets per row), label bounds, weights, base margins, ranking groups, and
/// per-feature metadata.
///
/// Missing values are first-class: in dense storage any entry equal to the
/// [`DMatrix::missing`] sentinel (NaN by default) is treated as absent, and in
/// sparse storage absent columns are missing. Split finding learns a default
/// direction for absent values, matching
/// XGBoost's sparsity-aware algorithm.
///
/// Clones share the feature values: `clone` copies only the metadata
/// (labels, weights, and the rest), and no operation on a clone changes the
/// matrix it was cloned from. They also share the histogram indexes binned
/// from those values (`hist_index`), so a binning is computed once for
/// every copy and freed with the values. A row selection
/// ([`select_rows`](Self::select_rows)) is a view of its source's values,
/// copying only the metadata.
#[derive(Debug, Clone)]
pub struct DMatrix {
    n_rows: usize,
    n_cols: usize,
    storage: Storage,
    /// The storage row of each of this matrix's rows, in order (`None`:
    /// storage row `r` is row `r`, for all `n_rows`).
    rows: Option<Arc<[usize]>>,
    /// A [`ViewCuts::Source`] view's source.
    cut_source: Option<Arc<CutSource>>,
    /// The histogram indexes binned from these rows of `storage`, shared
    /// exactly as they are: new values (`map_values`, `dense_values_mut`)
    /// and new row selections come with a new cache.
    bins: Arc<BinCache>,
    missing: f32,
    /// Row-major `[row][target]`, length `n_rows * n_targets`.
    labels: Option<Vec<f32>>,
    n_targets: NonZeroUsize,
    /// Lower and upper label bounds, `n_rows` each, attached together.
    label_bounds: Option<(Vec<f32>, Vec<f32>)>,
    weights: Option<Vec<f32>>,
    base_margin: Option<Vec<f32>>,
    group: Option<GroupInfo>,
    feature_types: Vec<FeatureType>,
    feature_weights: Option<Vec<f32>>,
}

/// A dense matrix's rows: its storage, read through its row map.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseRows<'a> {
    values: &'a [f32],
    n_cols: usize,
    rows: Option<&'a [usize]>,
}

impl<'a> DenseRows<'a> {
    /// Row `r`'s values (missing entries hold the matrix's sentinel).
    #[inline(always)]
    pub(crate) fn row(self, r: usize) -> &'a [f32] {
        let at = self.rows.map_or(r, |rows| rows[r]);
        &self.values[at * self.n_cols..][..self.n_cols]
    }

    /// Every row back to back, row-major, when the matrix holds its
    /// storage's rows in order.
    #[inline]
    pub(crate) fn contiguous(self) -> Option<&'a [f32]> {
        self.rows.is_none().then_some(self.values)
    }
}

/// A CSR matrix's rows: its storage, read through its row map.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CsrRows<'a> {
    indptr: &'a [usize],
    indices: &'a [u32],
    values: &'a [f32],
    rows: Option<&'a [usize]>,
}

impl<'a> CsrRows<'a> {
    /// Row `r`'s stored `(indices, values)` (entries equal to the missing
    /// sentinel are stored but absent).
    #[inline(always)]
    pub(crate) fn row(self, r: usize) -> (&'a [u32], &'a [f32]) {
        let at = self.rows.map_or(r, |rows| rows[r]);
        let span = self.indptr[at]..self.indptr[at + 1];
        (&self.indices[span.clone()], &self.values[span])
    }
}

impl DMatrix {
    /// Shared constructor: validated storage plus default (empty) metadata.
    fn new(n_rows: usize, n_cols: usize, storage: Storage, missing: f32) -> Self {
        DMatrix {
            n_rows,
            n_cols,
            storage,
            rows: None,
            cut_source: None,
            bins: Arc::default(),
            missing,
            labels: None,
            n_targets: NonZeroUsize::MIN,
            label_bounds: None,
            weights: None,
            base_margin: None,
            group: None,
            feature_types: vec![FeatureType::Numerical; n_cols],
            feature_weights: None,
        }
    }

    /// The storage row holding row `r`.
    #[inline(always)]
    fn storage_row(&self, r: usize) -> usize {
        self.rows.as_ref().map_or(r, |rows| rows[r])
    }

    /// Build a dense matrix from a row-major slice of length `n_rows * n_cols`,
    /// copying it. The missing sentinel defaults to NaN.
    /// [`DMatrix::from_dense_vec`] takes an owned `Vec` without copying.
    pub fn from_dense(data: &[f32], n_rows: usize, n_cols: usize) -> Result<Self> {
        Self::from_dense_with_missing(data, n_rows, n_cols, f32::NAN)
    }

    /// Builds the matrix [`DMatrix::from_dense`] builds, taking ownership of
    /// `data` instead of copying it, after the same checks (a non-empty shape
    /// matching `data.len()`, every value finite or NaN). NaN marks a missing
    /// value.
    pub fn from_dense_vec(data: Vec<f32>, n_rows: usize, n_cols: usize) -> Result<Self> {
        check_dense(&data, n_rows, n_cols, f32::NAN)?;
        Ok(Self::new(
            n_rows,
            n_cols,
            Storage::Dense(Arc::new(data)),
            f32::NAN,
        ))
    }

    /// Build a dense matrix with an explicit missing-value sentinel.
    pub fn from_dense_with_missing(
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
        missing: f32,
    ) -> Result<Self> {
        let parallel = check_dense(data, n_rows, n_cols, missing)?;
        let values = copy_dense(data, parallel);
        Ok(Self::new(
            n_rows,
            n_cols,
            Storage::Dense(Arc::new(values)),
            missing,
        ))
    }

    /// Build a matrix from compressed-sparse-row arrays.
    ///
    /// `indptr` must have `n_rows + 1` entries. Row `i` spans
    /// `indices[indptr[i]..indptr[i + 1]]`. Absent columns are treated as
    /// missing (sparsity-aware), so the sentinel is set to NaN.
    pub fn from_csr(
        indptr: Vec<usize>,
        indices: Vec<u32>,
        values: Vec<f32>,
        n_cols: usize,
    ) -> Result<Self> {
        if indptr.is_empty() {
            return Err(HessboostError::EmptyDataset("from_csr: empty indptr"));
        }
        let n_rows = indptr.len() - 1;
        if n_rows == 0 || n_cols == 0 {
            return Err(HessboostError::EmptyDataset(
                "from_csr: zero rows or columns",
            ));
        }
        check_len("csr indices/values length", values.len(), indices.len())?;
        check_csr(&indptr, values.len())?;
        if let Some(&m) = indices.iter().max()
            && (m as usize) >= n_cols
        {
            return Err(HessboostError::FeatureOutOfBounds {
                index: m as usize,
                num_features: n_cols,
            });
        }
        check_finite(
            "csr values",
            &values,
            "stored feature values must be finite",
        )?;
        let mut seen = std::collections::HashSet::new();
        for row in 0..n_rows {
            seen.clear();
            for &col in &indices[indptr[row]..indptr[row + 1]] {
                if !seen.insert(col) {
                    return Err(HessboostError::invalid_data(
                        "csr indices",
                        format!("duplicate column {col} in row {row}"),
                    ));
                }
            }
        }
        Ok(Self::new(
            n_rows,
            n_cols,
            Storage::Csr {
                indptr: Arc::new(indptr),
                indices: Arc::new(indices),
                values: Arc::new(values),
            },
            f32::NAN,
        ))
    }

    /// Attach regression/classification labels (`len == n_rows`), one target
    /// per row.
    pub fn with_labels(self, labels: &[f32]) -> Result<Self> {
        self.with_label_matrix(labels, 1)
    }

    /// Attach a label matrix with `n_targets` targets per row, laid out
    /// row-major `[row][target]` (`len == n_rows * n_targets`).
    pub fn with_label_matrix(mut self, labels: &[f32], n_targets: usize) -> Result<Self> {
        let Some(n_targets) = NonZeroUsize::new(n_targets) else {
            return Err(HessboostError::invalid_data(
                "labels",
                "n_targets must be at least 1",
            ));
        };
        let expected = self.n_rows.checked_mul(n_targets.get()).ok_or_else(|| {
            HessboostError::invalid_data("labels", "n_rows * n_targets overflows usize")
        })?;
        check_len("labels", labels.len(), expected)?;
        check_finite("labels", labels, "all labels must be finite")?;
        self.labels = Some(labels.to_vec());
        self.n_targets = n_targets;
        Ok(self)
    }

    /// Attach interval-censored label bounds (`len == n_rows` each), as used
    /// by `survival:aft`.
    ///
    /// NaN is rejected in both. `upper` may be `+inf` (right censoring) and
    /// `lower` may be `<= 0` (left censoring); as in XGBoost the bounds are
    /// not checked against each other.
    pub fn with_label_bounds(mut self, lower: &[f32], upper: &[f32]) -> Result<Self> {
        check_len("label_lower_bound", lower.len(), self.n_rows)?;
        check_len("label_upper_bound", upper.len(), self.n_rows)?;
        for (name, bound) in [("label_lower_bound", lower), ("label_upper_bound", upper)] {
            if bound.iter().any(|v| v.is_nan()) {
                return Err(HessboostError::invalid_data(
                    name,
                    "label bounds must not be NaN",
                ));
            }
        }
        self.label_bounds = Some((lower.to_vec(), upper.to_vec()));
        Ok(self)
    }

    /// Attach per-feature column-sampling weights (`len == n_cols`), XGBoost's
    /// `DMatrix` `feature_weights`. When set on the training matrix, every
    /// `colsample_bytree`/`bylevel`/`bynode` stage draws features without
    /// replacement with probability proportional to their weight (see
    /// [`ColumnSampler`](crate::tree::sampler::ColumnSampler)); with every
    /// ratio at `1.0` they have no effect. Weights below `1e-6`, zero
    /// included, are floored at `1e-6` as in XGBoost, so a zero weight is an
    /// epsilon weight, not an exclusion: such a feature is rarely drawn
    /// against much larger weights but can still be selected, and it is as
    /// likely as any other feature weighted below `1e-6`. Not stored in the
    /// model. Training refuses them with `booster = gblinear`, with
    /// `process_type = update`, and in budget mode, none of which samples
    /// columns.
    pub fn with_feature_weights(mut self, weights: &[f32]) -> Result<Self> {
        check_len("feature_weights", weights.len(), self.n_cols)?;
        check_weights(
            "feature_weights",
            weights,
            "feature weights must be finite and non-negative",
            "at least one feature weight must be positive",
        )?;
        self.feature_weights = Some(weights.to_vec());
        Ok(self)
    }

    /// Attach per-instance weights (`len == n_rows`).
    pub fn with_weights(mut self, weights: &[f32]) -> Result<Self> {
        check_len("weights", weights.len(), self.n_rows)?;
        check_weights(
            "weights",
            weights,
            "weights must be finite and non-negative",
            "at least one weight must be positive",
        )?;
        self.weights = Some(weights.to_vec());
        Ok(self)
    }

    /// Attach a per-instance base margin (raw prediction offset, `len == n_rows`
    /// for single-output objectives).
    ///
    /// The margin replaces the model intercept for this matrix's rows, in
    /// training, evaluation, and prediction (XGBoost `base_margin`), e.g. to
    /// boost from a pretrained model's logits. It is not saved with the
    /// model: predicting on a matrix without one uses the intercept.
    pub fn with_base_margin(mut self, base_margin: &[f32]) -> Result<Self> {
        if base_margin.is_empty() || !base_margin.len().is_multiple_of(self.n_rows) {
            return Err(HessboostError::invalid_data(
                "base_margin",
                "length must be a non-zero multiple of n_rows",
            ));
        }
        check_finite("base_margin", base_margin, "all margins must be finite")?;
        self.base_margin = Some(base_margin.to_vec());
        Ok(self)
    }

    /// Attach ranking group information (sizes sum to `n_rows`).
    pub fn with_group_sizes(mut self, sizes: &[usize]) -> Result<Self> {
        if sizes.is_empty() || sizes.contains(&0) {
            return Err(HessboostError::invalid_data(
                "group_sizes",
                "groups must be non-empty and every group must contain a row",
            ));
        }
        let total = sizes.iter().try_fold(0usize, |acc, &s| acc.checked_add(s));
        let Some(total) = total else {
            return Err(HessboostError::invalid_data(
                "group_sizes",
                "group-size sum overflows usize",
            ));
        };
        check_len("group sizes sum", total, self.n_rows)?;
        self.group = Some(GroupInfo::from_sizes(sizes));
        Ok(self)
    }

    /// Attach one weight per ranking group.
    ///
    /// Group sizes must be attached first. Internally each group weight is
    /// expanded across that group's rows so objectives and metrics receive the
    /// XGBoost-compatible per-query weighting semantics.
    pub fn with_group_weights(mut self, weights: &[f32]) -> Result<Self> {
        let group = self.group.as_ref().ok_or_else(|| {
            HessboostError::invalid_data("group_weights", "attach group sizes first")
        })?;
        check_len("group_weights length", weights.len(), group.num_groups())?;
        let invalid = "weights must be finite and non-negative with at least one positive value";
        check_weights("group_weights", weights, invalid, invalid)?;
        let mut expanded = Vec::with_capacity(self.n_rows);
        for ((start, end), &weight) in group.iter_ranges().zip(weights) {
            expanded.extend(std::iter::repeat_n(weight, end - start));
        }
        self.weights = Some(expanded);
        Ok(self)
    }

    /// Set the feature types (`len == n_cols`).
    pub fn with_feature_types(mut self, types: &[FeatureType]) -> Result<Self> {
        check_len("feature_types length", types.len(), self.n_cols)?;
        self.feature_types = types.to_vec();
        let invalid = |col: usize, v: f32| {
            HessboostError::invalid_data(
                "categorical feature",
                format!("feature {col} contains invalid category value {v}"),
            )
        };
        let is_invalid = |v: f32| v < 0.0 || v.fract() != 0.0 || v >= u32::MAX as f32;
        match &self.storage {
            Storage::Dense(_) => {
                for (col, ty) in types.iter().enumerate() {
                    if *ty == FeatureType::Categorical {
                        for row in 0..self.n_rows {
                            if let Some(v) = self.get(row, col)
                                && is_invalid(v)
                            {
                                return Err(invalid(col, v));
                            }
                        }
                    }
                }
            }
            // One pass over the stored entries; the first invalid entry by
            // column, then row, is the one reported, as the dense scan does.
            Storage::Csr { .. } => {
                let mut first: Option<(usize, usize, f32)> = None;
                self.for_each_entry(|row, col, v| {
                    let col = col as usize;
                    if types[col] == FeatureType::Categorical
                        && is_invalid(v)
                        && first.is_none_or(|(c, r, _)| (col, row) < (c, r))
                    {
                        first = Some((col, row, v));
                    }
                });
                if let Some((col, _, v)) = first {
                    return Err(invalid(col, v));
                }
            }
        }
        Ok(self)
    }

    /// Number of rows (instances).
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of feature columns.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The missing-value sentinel.
    #[inline]
    pub fn missing(&self) -> f32 {
        self.missing
    }

    /// Labels, if attached: row-major `[row][target]`, length
    /// `n_rows * n_targets`.
    #[inline]
    pub fn labels(&self) -> Option<&[f32]> {
        self.labels.as_deref()
    }

    /// Number of label columns (targets) per row; 1 unless set by
    /// [`DMatrix::with_label_matrix`].
    #[inline]
    pub fn n_targets(&self) -> usize {
        self.n_targets.get()
    }

    /// Lower label bounds for interval-censored labels, if attached.
    #[inline]
    pub fn label_lower_bound(&self) -> Option<&[f32]> {
        self.label_bounds
            .as_ref()
            .map(|(lower, _)| lower.as_slice())
    }

    /// Upper label bounds for interval-censored labels, if attached.
    #[inline]
    pub fn label_upper_bound(&self) -> Option<&[f32]> {
        self.label_bounds
            .as_ref()
            .map(|(_, upper)| upper.as_slice())
    }

    /// Weights, if attached.
    #[inline]
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// Base margin, if attached.
    #[inline]
    pub fn base_margin(&self) -> Option<&[f32]> {
        self.base_margin.as_deref()
    }

    /// Ranking group info, if attached.
    #[inline]
    pub fn group(&self) -> Option<&GroupInfo> {
        self.group.as_ref()
    }

    /// Feature types. This is always populated and defaults to all numerical.
    #[inline]
    pub fn feature_types(&self) -> &[FeatureType] {
        &self.feature_types
    }

    /// Per-feature sampling weights, if attached.
    #[inline]
    pub fn feature_weights(&self) -> Option<&[f32]> {
        self.feature_weights.as_deref()
    }

    /// Borrowed view of the per-row training metadata that objectives and
    /// metrics consume.
    pub fn info(&self) -> MetaInfo<'_> {
        MetaInfo {
            n_rows: self.n_rows,
            labels: self
                .labels
                .as_deref()
                .map(|values| Labels::new(values, self.n_targets)),
            weights: self.weights.as_deref(),
            group: self.group.as_ref(),
            bounds: self
                .label_bounds
                .as_ref()
                .map(|(lower, upper)| LabelBounds::new(lower, upper)),
        }
    }

    /// Fetch a single value, returning `None` when the entry is missing.
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row >= self.n_rows || col >= self.n_cols {
            return None;
        }
        let v = if let Some(rows) = self.dense_rows() {
            rows.row(row)[col]
        } else {
            let (indices, values) = self.csr_rows()?.row(row);
            // Rows are not assumed sorted by column; linear scan of the row.
            let k = indices.iter().position(|&index| index as usize == col)?;
            values[k]
        };
        (!is_missing(v, self.missing)).then_some(v)
    }

    /// A dense matrix's rows (missing entries hold the sentinel), or `None`
    /// for sparse storage.
    #[inline]
    pub(crate) fn dense_rows(&self) -> Option<DenseRows<'_>> {
        match &self.storage {
            Storage::Dense(values) => Some(DenseRows {
                values,
                n_cols: self.n_cols,
                rows: self.rows.as_deref(),
            }),
            Storage::Csr { .. } => None,
        }
    }

    /// A CSR matrix's rows, or `None` for dense storage. Entries equal to
    /// the missing sentinel may be present and must be treated as absent by
    /// callers.
    #[inline]
    pub(crate) fn csr_rows(&self) -> Option<CsrRows<'_>> {
        match &self.storage {
            Storage::Dense(_) => None,
            Storage::Csr {
                indptr,
                indices,
                values,
            } => Some(CsrRows {
                indptr,
                indices,
                values,
                rows: self.rows.as_deref(),
            }),
        }
    }

    /// Mutable row-major values of a dense matrix, or `None` for sparse
    /// storage. Values shared with a clone (or a row view's source) are
    /// copied first, a view's rows only, so the others keep their own;
    /// either way the matrix's binned indexes no longer apply and are
    /// dropped, and a [`ViewCuts::Source`] view cuts its own rows from then
    /// on. Callers keep the constructors' invariant: every value is finite
    /// or the missing sentinel.
    #[inline]
    pub(crate) fn dense_values_mut(&mut self) -> Option<&mut [f32]> {
        let own = match (&self.storage, &self.rows) {
            (Storage::Csr { .. }, _) => return None,
            (Storage::Dense(values), Some(rows)) => {
                let n_cols = self.n_cols;
                let mut own = Vec::with_capacity(rows.len() * n_cols);
                for &r in rows.iter() {
                    own.extend_from_slice(&values[r * n_cols..(r + 1) * n_cols]);
                }
                Some(own)
            }
            (Storage::Dense(_), None) => None,
        };
        if let Some(own) = own {
            self.storage = Storage::Dense(Arc::new(own));
            self.rows = None;
        }
        self.bins = Arc::default();
        self.cut_source = None;
        match &mut self.storage {
            Storage::Dense(data) => Some(Arc::make_mut(data).as_mut_slice()),
            Storage::Csr { .. } => None,
        }
    }

    /// The histogram index of the feature values binned with `max_bin`
    /// cuts from this matrix's sample weights, feature types and missing
    /// value ([`HistCuts::from_dmatrix`]), or for a [`ViewCuts::Source`]
    /// view its source's index, gathered: built on first use, then shared
    /// by every matrix sharing the values (clones and metadata copies,
    /// across threads), and freed with them. See
    /// [`bin_cache`](crate::data::bin_cache).
    pub(crate) fn hist_index(&self, max_bin: usize) -> Arc<GHistIndex> {
        if let Some(source) = &self.cut_source {
            // The gather may wait on its source's own (parallel) build, so
            // with threads to spare it always builds on a pool of its own.
            let parallel = rayon::current_num_threads() > 1;
            let key = source.matrix.bin_key(max_bin);
            return self.bins.get_or_build(&key, parallel, || {
                source.matrix.hist_index(max_bin).select_rows(&source.rows)
            });
        }
        let parallel = rayon::current_num_threads() > 1
            && self.n_rows.saturating_mul(self.n_cols) >= PARALLEL_BINNING_CELLS;
        self.bins
            .get_or_build(&self.bin_key(max_bin), parallel, || {
                GHistIndex::from_dmatrix(self, HistCuts::from_dmatrix(self, max_bin))
            })
    }

    /// Everything besides the values this matrix bins with under `max_bin`.
    fn bin_key(&self, max_bin: usize) -> BinKey {
        BinKey::new(
            max_bin,
            self.missing,
            &self.feature_types,
            self.weights.as_deref(),
        )
    }

    /// Which of this matrix's rows the cuts of its `hist` index were
    /// sketched from: those with a nonzero sample weight in the matrix the
    /// cuts are computed on (this one, or a [`ViewCuts::Source`] view's
    /// source).
    pub(crate) fn sketched_rows(&self) -> SketchedRows {
        let Some(source) = &self.cut_source else {
            return match self.weights.as_deref() {
                Some(w) if w.contains(&0.0) => SketchedRows::Weighted,
                _ => SketchedRows::All,
            };
        };
        // The view's rows in the matrix its cuts are computed on.
        let mut rows = Cow::Borrowed(source.rows.as_slice());
        let mut cut = &source.matrix;
        while let Some(next) = &cut.cut_source {
            rows = Cow::Owned(rows.iter().map(|&r| next.rows[r]).collect());
            cut = &next.matrix;
        }
        let Some(cut_weights) = cut.weights.as_deref() else {
            return SketchedRows::All;
        };
        let mut sketched = SketchedRows::All;
        for (r, &row) in rows.iter().enumerate() {
            if cut_weights[row] == 0.0 {
                if self.weights.as_deref().is_none_or(|w| w[r] != 0.0) {
                    return SketchedRows::Partial;
                }
                sketched = SketchedRows::Weighted;
            }
        }
        sketched
    }

    /// Copy of this matrix, metadata included, with every non-missing stored
    /// value replaced by `f(row, col, value)`. Missing dense entries become NaN
    /// and the result's sentinel is NaN, so a mapped value can never collide
    /// with a non-NaN sentinel. Feature types are copied unchanged. The values
    /// are written to new storage holding exactly this matrix's rows (a CSR
    /// result of a whole matrix shares its `indptr` and `indices`); this
    /// matrix is unchanged.
    pub(crate) fn map_values(&self, mut f: impl FnMut(usize, usize, f32) -> f32) -> Self {
        let mut map = |row: usize, col: usize, v: f32, missing: f32| {
            if is_missing(v, self.missing) {
                missing
            } else {
                f(row, col, v)
            }
        };
        let storage = match (&self.storage, self.rows.is_some()) {
            (Storage::Dense(_), _) => {
                let rows = self.dense_rows().expect("dense storage has dense rows");
                let mut mapped = Vec::with_capacity(self.n_rows * self.n_cols);
                for row in 0..self.n_rows {
                    let values = rows.row(row);
                    mapped.extend(
                        values
                            .iter()
                            .enumerate()
                            .map(|(col, &v)| map(row, col, v, f32::NAN)),
                    );
                }
                Storage::Dense(Arc::new(mapped))
            }
            (
                Storage::Csr {
                    indptr,
                    indices,
                    values,
                },
                false,
            ) => {
                let mut mapped = Vec::with_capacity(values.len());
                for row in 0..self.n_rows {
                    let span = indptr[row]..indptr[row + 1];
                    mapped
                        .extend(span.map(|k| map(row, indices[k] as usize, values[k], values[k])));
                }
                Storage::Csr {
                    indptr: Arc::clone(indptr),
                    indices: Arc::clone(indices),
                    values: Arc::new(mapped),
                }
            }
            (Storage::Csr { .. }, true) => {
                let rows = self.csr_rows().expect("sparse storage has CSR rows");
                let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
                for row in 0..self.n_rows {
                    let (row_indices, row_values) = rows.row(row);
                    for (&index, &v) in row_indices.iter().zip(row_values) {
                        indices.push(index);
                        values.push(map(row, index as usize, v, v));
                    }
                    indptr.push(values.len());
                }
                Storage::Csr {
                    indptr: Arc::new(indptr),
                    indices: Arc::new(indices),
                    values: Arc::new(values),
                }
            }
        };
        let mut out = self.clone();
        out.storage = storage;
        out.rows = None;
        out.cut_source = None;
        out.bins = Arc::default();
        out.missing = f32::NAN;
        out
    }

    /// Visit every non-missing `(index, value)` entry of `row`, in storage
    /// order (ascending columns for dense storage). `row` must be in bounds.
    #[inline]
    pub(crate) fn for_row_entry(&self, row: usize, mut f: impl FnMut(u32, f32)) {
        if let Some(rows) = self.dense_rows() {
            for (c, &v) in rows.row(row).iter().enumerate() {
                if !is_missing(v, self.missing) {
                    f(c as u32, v);
                }
            }
        } else if let Some(rows) = self.csr_rows() {
            let (indices, values) = rows.row(row);
            for (&index, &v) in indices.iter().zip(values) {
                if !is_missing(v, self.missing) {
                    f(index, v);
                }
            }
        }
    }

    /// Build a compressed-sparse-**column** view for column-oriented split
    /// finding (used by the exact tree method). Each column lists its
    /// non-missing `(row, value)` pairs.
    pub(crate) fn to_csc(&self) -> CscView {
        let mut col_counts = vec![0usize; self.n_cols];
        self.for_each_entry(|_row, col, _v| col_counts[col as usize] += 1);

        let mut col_ptr = vec![0usize; self.n_cols + 1];
        for (c, count) in col_counts.iter_mut().enumerate() {
            col_ptr[c + 1] = col_ptr[c] + *count;
            // From here on the count is column `c`'s fill cursor.
            *count = col_ptr[c];
        }
        let nnz = col_ptr[self.n_cols];
        let mut rows = vec![0u32; nnz];
        let mut vals = vec![0f32; nnz];
        let mut cursor = col_counts;
        self.for_each_entry(|row, col, v| {
            let c = col as usize;
            let pos = cursor[c];
            rows[pos] = row as u32;
            vals[pos] = v;
            cursor[c] = pos + 1;
        });
        CscView {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            col_ptr,
            rows,
            vals,
        }
    }

    /// The matrix of only `rows` (in the given order, repeats allowed), a
    /// view sharing this matrix's feature values (nothing is copied but the
    /// metadata: labels, every target of each row, label bounds, weights,
    /// base margin, ranking groups, and feature metadata). Training on it
    /// is training on a copy of those rows, bit for bit: the view cuts its
    /// own rows ([`ViewCuts::Own`]; see
    /// [`select_rows_with_cuts`](Self::select_rows_with_cuts)).
    ///
    /// On a matrix with query groups, `rows` must list whole groups back to
    /// back, each group's rows together and in row order (a group may repeat
    /// or be left out); the result has one group per listed group. A
    /// selection that splits a group, reorders its rows, or interleaves them
    /// with another group's is refused, as is an empty one.
    pub fn select_rows(&self, rows: &[usize]) -> Result<Self> {
        self.select_rows_with_cuts(rows, ViewCuts::Own)
    }

    /// [`select_rows`](Self::select_rows) whose histogram index takes its
    /// cuts as `cuts` says: [`ViewCuts::Source`] gathers this matrix's own
    /// bins for the rows (computing no cuts and binning nothing again, and
    /// sharing this matrix's index, built on first use, with its other
    /// trainings and views).
    pub fn select_rows_with_cuts(&self, rows: &[usize], cuts: ViewCuts) -> Result<Self> {
        if let Some(&row) = rows.iter().find(|&&row| row >= self.n_rows) {
            return Err(HessboostError::invalid_param(
                "rows",
                format!("row index {row} is out of bounds for {} rows", self.n_rows),
            ));
        }
        if rows.is_empty() {
            return Err(HessboostError::EmptyDataset(
                "select_rows: no rows selected",
            ));
        }
        let group_sizes = self
            .selected_group_sizes(rows)
            .map_err(|reason| HessboostError::invalid_param("rows", format!("rows {reason}")))?;
        // Each row's `stride` consecutive values of `v`, in `rows` order.
        let gather = |v: &[f32], stride: usize| {
            let mut selected = Vec::with_capacity(rows.len() * stride);
            for &r in rows {
                selected.extend_from_slice(&v[r * stride..(r + 1) * stride]);
            }
            selected
        };
        let cut_source = match cuts {
            ViewCuts::Own => None,
            ViewCuts::Source => Some(Arc::new(CutSource {
                matrix: self.binning_source(),
                rows: rows.to_vec(),
            })),
        };
        Ok(DMatrix {
            n_rows: rows.len(),
            n_cols: self.n_cols,
            storage: self.storage.clone(),
            rows: Some(rows.iter().map(|&r| self.storage_row(r)).collect()),
            cut_source,
            bins: Arc::default(),
            missing: self.missing,
            labels: self
                .labels
                .as_deref()
                .map(|l| gather(l, self.n_targets.get())),
            n_targets: self.n_targets,
            label_bounds: self
                .label_bounds
                .as_ref()
                .map(|(lo, hi)| (gather(lo, 1), gather(hi, 1))),
            weights: self.weights.as_deref().map(|w| gather(w, 1)),
            base_margin: self
                .base_margin
                .as_deref()
                .map(|bm| gather(bm, bm.len() / self.n_rows)),
            group: group_sizes
                .filter(|sizes| !sizes.is_empty())
                .map(|sizes| GroupInfo::from_sizes(&sizes)),
            feature_types: self.feature_types.clone(),
            feature_weights: self.feature_weights.clone(),
        })
    }

    /// This matrix without the metadata binning does not read (labels,
    /// bounds, base margins, groups, feature weights), sharing its values
    /// and bin cache: what a [`ViewCuts::Source`] view keeps of its source.
    fn binning_source(&self) -> DMatrix {
        DMatrix {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            storage: self.storage.clone(),
            rows: self.rows.clone(),
            cut_source: self.cut_source.clone(),
            bins: Arc::clone(&self.bins),
            missing: self.missing,
            labels: None,
            n_targets: NonZeroUsize::MIN,
            label_bounds: None,
            weights: self.weights.clone(),
            base_margin: None,
            group: None,
            feature_types: self.feature_types.clone(),
            feature_weights: None,
        }
    }

    /// The query-group sizes of in-bounds `rows` selected from this matrix
    /// ([`select_rows`](Self::select_rows)): `None` without groups, else one
    /// size per whole group `rows` lists. A selection that is not whole
    /// groups back to back, each in row order, is refused with the reason,
    /// which completes "rows …".
    pub(crate) fn selected_group_sizes(
        &self,
        rows: &[usize],
    ) -> std::result::Result<Option<Vec<usize>>, String> {
        let Some(group) = &self.group else {
            return Ok(None);
        };
        let ptr = &group.group_ptr;
        let mut sizes = Vec::new();
        let mut at = 0;
        while let Some(&row) = rows.get(at) {
            // `ptr` starts at 0 and ends at `n_rows`, past every row, so the
            // group holding `row` is the last one starting at or before it.
            let g = ptr.partition_point(|&start| start <= row) - 1;
            let (start, end) = (ptr[g], ptr[g + 1]);
            let run = rows.get(at..at + (end - start));
            if !run.is_some_and(|run| run.iter().copied().eq(start..end)) {
                return Err(format!(
                    "split query group {g} (rows {start}..{end}), reorder it, or interleave it \
                     with other rows; select whole groups, each group's rows together and in \
                     row order"
                ));
            }
            sizes.push(end - start);
            at += end - start;
        }
        Ok(Some(sizes))
    }

    /// Visit every non-missing entry as `(row, col, value)`, in row order.
    pub(crate) fn for_each_entry(&self, mut f: impl FnMut(usize, u32, f32)) {
        for r in 0..self.n_rows {
            self.for_row_entry(r, |c, v| f(r, c, v));
        }
    }
}

/// A compressed-sparse-column view of a [`DMatrix`], built by
/// [`DMatrix::to_csc`]. Within each column the `(row, value)` pairs are stored
/// in row order. Callers that need value-sorted order (e.g. the exact split
/// finder) sort per-column slices themselves.
#[derive(Debug, Clone)]
pub(crate) struct CscView {
    n_rows: usize,
    n_cols: usize,
    col_ptr: Vec<usize>,
    rows: Vec<u32>,
    vals: Vec<f32>,
}

impl CscView {
    /// Number of rows in the originating matrix.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The `(rows, values)` slices of non-missing entries for a column.
    #[inline]
    pub fn column(&self, col: usize) -> (&[u32], &[f32]) {
        let (s, e) = (self.col_ptr[col], self.col_ptr[col + 1]);
        (&self.rows[s..e], &self.vals[s..e])
    }

    /// Number of non-missing entries in a column.
    #[inline]
    pub fn col_len(&self, col: usize) -> usize {
        self.col_ptr[col + 1] - self.col_ptr[col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dense() -> DMatrix {
        // 3 rows x 2 cols, with a NaN missing in row 1 col 0.
        let data = vec![1.0, 2.0, f32::NAN, 5.0, 3.0, 6.0];
        DMatrix::from_dense(&data, 3, 2).unwrap()
    }

    /// A matrix and its metadata copies bin their shared values once per
    /// `max_bin` and sample weighting; new values bin afresh (the copy
    /// sees its own values) while the source keeps its index.
    #[test]
    fn copies_share_binned_indexes_until_their_values_change() {
        let (n, cols) = (200, 2);
        let values: Vec<f32> = (0..n * cols).map(|i| (i % 37) as f32).collect();
        let data = DMatrix::from_dense(&values, n, cols).unwrap();
        let index = data.hist_index(16);
        let labels = vec![1.0; n];
        let labeled = data.clone().with_labels(&labels).unwrap();
        assert!(Arc::ptr_eq(&index, &labeled.hist_index(16)));
        assert!(!Arc::ptr_eq(&index, &data.hist_index(8)));
        let weights: Vec<f32> = (0..n).map(|r| 1.0 + (r % 3) as f32).collect();
        let weighted = data.clone().with_weights(&weights).unwrap();
        let weighted_index = weighted.hist_index(16);
        assert!(!Arc::ptr_eq(&index, &weighted_index));
        let reference = HistCuts::from_dmatrix(&weighted, 16);
        assert_eq!(
            serde_json::to_value(weighted_index.cuts()).unwrap(),
            serde_json::to_value(&reference).unwrap(),
            "the weighted copy bins with its own weights"
        );
        let mut changed = data.clone();
        changed.dense_values_mut().unwrap()[0] = 1e6;
        let changed_index = changed.hist_index(16);
        let (_, last) = changed_index.cuts().feature_bins(0);
        assert_eq!(changed_index.feature_bin(0, 0), Some(last as u32 - 1));
        assert!(Arc::ptr_eq(&index, &data.hist_index(16)));
        assert_ne!(index.feature_bin(0, 0), changed_index.feature_bin(0, 0));
    }

    /// A [`ViewCuts::Source`] view trains exactly as a copy of its rows
    /// binned with the source's cuts, and reads the source's index (built
    /// once, shared with the source's own trainings).
    #[test]
    fn source_cut_views_train_as_copies_with_the_sources_cuts() {
        use crate::config::TrainingParams;
        use crate::model::ModelFormat;
        let (n, cols) = (4_000, 3);
        let x: Vec<f32> = (0..n * cols)
            .map(|i| {
                if i % 11 == 0 {
                    f32::NAN
                } else {
                    ((i * 7919) % 1013) as f32 / 9.0
                }
            })
            .collect();
        let y: Vec<f32> = (0..n).map(|r| ((r * 31) % 17) as f32).collect();
        let data = DMatrix::from_dense(&x, n, cols)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let rows: Vec<usize> = (0..n).filter(|r| r % 3 != 1).collect();
        let view = data.select_rows_with_cuts(&rows, ViewCuts::Source).unwrap();
        let copy = DMatrix::from_dense_vec(
            rows.iter()
                .flat_map(|&r| x[r * cols..(r + 1) * cols].iter().copied())
                .collect(),
            rows.len(),
            cols,
        )
        .unwrap()
        .with_labels(&rows.iter().map(|&r| y[r]).collect::<Vec<_>>())
        .unwrap();
        let params = TrainingParams::builder()
            .max_depth(4)
            .subsample(0.8)
            .build()
            .unwrap();
        let source_index = data.hist_index(params.max_bin);
        // Seed the copy's cache with its rows binned by the source's cuts.
        let seeded = copy
            .bins
            .get_or_build(&copy.bin_key(params.max_bin), false, || {
                GHistIndex::from_dmatrix(&copy, source_index.cuts().clone())
            });
        let fit = |data: &DMatrix| {
            crate::training::train(&params, data, 8)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(fit(&view), fit(&copy));
        assert!(Arc::ptr_eq(&seeded, &copy.hist_index(params.max_bin)));
        assert!(Arc::ptr_eq(&source_index, &data.hist_index(params.max_bin)));
    }

    /// Writing a view's values copies only its rows: its source keeps its
    /// values, and the view bins its new ones.
    #[test]
    fn writing_a_view_leaves_its_source_unchanged() {
        let data = DMatrix::from_dense(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        let mut view = data.select_rows(&[2, 0]).unwrap();
        assert_eq!(view.dense_values_mut().unwrap(), &[5.0, 6.0, 1.0, 2.0]);
        view.dense_values_mut().unwrap()[0] = 50.0;
        assert_eq!(view.get(0, 0), Some(50.0));
        assert_eq!(data.get(2, 0), Some(5.0));
        assert_eq!(view.select_rows(&[1, 0]).unwrap().get(1, 0), Some(50.0));
    }

    #[test]
    fn dense_get_and_missing() {
        let d = sample_dense();
        assert_eq!(d.get(0, 0), Some(1.0));
        assert_eq!(d.get(1, 0), None); // missing
        assert_eq!(d.get(1, 1), Some(5.0));
    }

    #[test]
    fn row_entries_skip_missing() {
        let d = sample_dense();
        let mut entries = Vec::new();
        d.for_row_entry(1, |index, value| entries.push((index, value)));
        assert_eq!(entries, [(1, 5.0)]);
    }

    #[test]
    fn csc_matches_dense() {
        let d = sample_dense();
        let csc = d.to_csc();
        // Column 0 has rows {0, 2} (row 1 is missing).
        let (rows, vals) = csc.column(0);
        assert_eq!(rows, &[0, 2]);
        assert_eq!(vals, &[1.0, 3.0]);
        // Column 1 has all three rows.
        assert_eq!(csc.col_len(1), 3);
    }

    #[test]
    fn csr_roundtrip() {
        // Same logical matrix as sample_dense but sparse (row 1 col 0 absent).
        let indptr = vec![0, 2, 3, 5];
        let indices = vec![0, 1, 1, 0, 1];
        let values = vec![1.0, 2.0, 5.0, 3.0, 6.0];
        let d = DMatrix::from_csr(indptr, indices, values, 2).unwrap();
        assert_eq!(d.get(0, 0), Some(1.0));
        assert_eq!(d.get(1, 0), None);
        assert_eq!(d.get(2, 1), Some(6.0));
        let csc = d.to_csc();
        let (rows, vals) = csc.column(0);
        assert_eq!(rows, &[0, 2]);
        assert_eq!(vals, &[1.0, 3.0]);
    }

    #[test]
    fn label_length_checked() {
        let d = sample_dense();
        assert!(d.clone().with_labels(&[1.0, 2.0]).is_err());
        assert!(d.with_labels(&[1.0, 2.0, 3.0]).is_ok());
    }

    #[test]
    fn malformed_csr_is_rejected() {
        assert!(DMatrix::from_csr(vec![1, 1], vec![], vec![], 2).is_err());
        assert!(DMatrix::from_csr(vec![0, 2, 1], vec![0], vec![1.0], 2).is_err());
        assert!(DMatrix::from_csr(vec![0, 2], vec![0, 0], vec![1.0, 2.0], 2).is_err());
        assert!(DMatrix::from_csr(vec![0, 1], vec![0], vec![f32::INFINITY], 2).is_err());
    }

    #[test]
    fn metadata_values_are_validated() {
        let d = sample_dense();
        assert!(d.clone().with_weights(&[1.0, -1.0, 1.0]).is_err());
        assert!(d.clone().with_weights(&[0.0, 0.0, 0.0]).is_err());
        assert!(d.clone().with_base_margin(&[0.0, 1.0]).is_err());
        assert!(d.clone().with_group_sizes(&[1, 0, 2]).is_err());
        assert!(d.select_rows(&[3]).is_err());
    }

    #[test]
    fn categorical_values_must_be_non_negative_integers() {
        let d = DMatrix::from_dense(&[0.0, 1.5], 2, 1).unwrap();
        assert!(d.with_feature_types(&[FeatureType::Categorical]).is_err());
    }

    #[test]
    fn label_matrix_is_row_major_and_validated() {
        let d = sample_dense();
        assert!(d.clone().with_label_matrix(&[1.0; 5], 2).is_err());
        assert!(d.clone().with_label_matrix(&[], 0).is_err());
        assert!(
            d.clone()
                .with_label_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, f32::NAN], 2)
                .is_err()
        );
        let m = d
            .with_label_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2)
            .unwrap();
        assert_eq!(m.n_targets(), 2);
        let info = m.info();
        assert_eq!((info.n_rows, info.n_targets()), (3, 2));
        assert_eq!(info.label_values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Re-attaching single-target labels resets the target count.
        assert_eq!(m.with_labels(&[0.0, 1.0, 2.0]).unwrap().n_targets(), 1);
    }

    #[test]
    fn label_bounds_allow_censoring_but_not_nan() {
        let d = sample_dense();
        let inf = f32::INFINITY;
        let censored = d
            .clone()
            .with_label_bounds(&[0.0, -1.0, 2.0], &[1.0, inf, 1.5])
            .unwrap();
        assert_eq!(censored.label_lower_bound().unwrap(), &[0.0, -1.0, 2.0]);
        assert_eq!(censored.label_upper_bound().unwrap(), &[1.0, inf, 1.5]);
        assert!(censored.labels().is_none());
        assert!(censored.info().labels.is_none());
        assert!(
            d.clone()
                .with_label_bounds(&[f32::NAN, 0.0, 0.0], &[1.0; 3])
                .is_err()
        );
        assert!(
            d.clone()
                .with_label_bounds(&[0.0; 3], &[1.0, f32::NAN, 1.0])
                .is_err()
        );
        assert!(d.with_label_bounds(&[0.0; 2], &[1.0; 3]).is_err());
    }

    #[test]
    fn feature_weights_are_validated() {
        let d = sample_dense();
        assert!(d.clone().with_feature_weights(&[1.0]).is_err());
        assert!(d.clone().with_feature_weights(&[1.0, -0.5]).is_err());
        assert!(d.clone().with_feature_weights(&[0.0, 0.0]).is_err());
        assert!(
            d.clone()
                .with_feature_weights(&[1.0, f32::INFINITY])
                .is_err()
        );
        let w = d.with_feature_weights(&[0.0, 2.0]).unwrap();
        assert_eq!(w.feature_weights().unwrap(), &[0.0, 2.0]);
    }

    /// Cross-validation folds must keep every target of a selected row
    /// together, and carry bounds and feature weights along.
    #[test]
    fn select_rows_carries_multi_target_labels_and_metadata() {
        let d = sample_dense()
            .with_label_matrix(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], 2)
            .unwrap()
            .with_label_bounds(&[1.0, 2.0, 3.0], &[1.5, f32::INFINITY, 3.5])
            .unwrap()
            .with_feature_weights(&[0.25, 0.75])
            .unwrap();
        let s = d.select_rows(&[2, 0]).unwrap();
        assert_eq!(s.n_targets(), 2);
        assert_eq!(s.labels().unwrap(), &[30.0, 31.0, 10.0, 11.0]);
        assert_eq!(s.label_lower_bound().unwrap(), &[3.0, 1.0]);
        assert_eq!(s.label_upper_bound().unwrap(), &[3.5, 1.5]);
        assert_eq!(s.feature_weights().unwrap(), &[0.25, 0.75]);
    }

    /// A dense selection reads its source's dense values, missing entries
    /// (under the source's non-NaN sentinel) staying missing, and an empty
    /// one is refused.
    #[test]
    fn select_rows_keeps_dense_storage_and_missing_entries() {
        let d = DMatrix::from_dense_with_missing(&[1.0, -1.0, -1.0, 4.0, 5.0, 6.0], 3, 2, -1.0)
            .unwrap();
        let s = d.select_rows(&[1, 0, 1]).unwrap();
        assert!(s.dense_rows().is_some());
        let expected = [None, Some(4.0), Some(1.0), None, None, Some(4.0)];
        for (cell, &want) in expected.iter().enumerate() {
            assert_eq!(s.get(cell / 2, cell % 2), want, "cell {cell}");
        }
        assert!(d.select_rows(&[]).is_err());
    }

    fn is_rows_refusal(result: &Result<DMatrix>) -> bool {
        matches!(
            result,
            Err(HessboostError::InvalidParameter { name: "rows", .. })
        )
    }

    /// Whole query groups keep their groups (and each row's expanded group
    /// weight), in the order listed; any other selection of grouped rows is
    /// refused rather than silently ungrouped.
    #[test]
    fn select_rows_keeps_whole_query_groups() {
        let x: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let d = DMatrix::from_dense(&x, 6, 1)
            .unwrap()
            .with_group_sizes(&[2, 3, 1])
            .unwrap()
            .with_group_weights(&[1.0, 2.0, 3.0])
            .unwrap();
        let s = d.select_rows(&[5, 0, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(s.group().unwrap().group_ptr, [0, 1, 3, 6, 7]);
        assert_eq!(s.weights().unwrap(), &[3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0]);
        assert_eq!(s.get(0, 0), Some(5.0));
        // Part of group 1, a group out of row order, and two interleaved
        // groups.
        assert!(is_rows_refusal(&d.select_rows(&[0, 1, 2, 3])));
        assert!(is_rows_refusal(&d.select_rows(&[3, 2, 4])));
        assert!(is_rows_refusal(&d.select_rows(&[1, 0])));
        assert!(is_rows_refusal(&d.select_rows(&[0, 5, 1])));
        // Without groups any selection goes.
        assert!(
            sample_dense()
                .select_rows(&[2, 0])
                .unwrap()
                .group()
                .is_none()
        );
    }

    /// Categorical validation on sparse storage reports the first invalid
    /// entry by column, then row, like the dense scan.
    #[test]
    fn csr_categorical_validation_reports_the_first_invalid_column() {
        let d = DMatrix::from_csr(
            vec![0, 2, 4],
            vec![1, 0, 0, 1],
            vec![2.5, 1.0, -3.0, 7.5],
            2,
        )
        .unwrap();
        let types = [FeatureType::Categorical; 2];
        let err = d
            .clone()
            .with_feature_types(&types)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("feature 0 contains invalid category value -3"),
            "{err}"
        );
        let numeric_first = [FeatureType::Numerical, FeatureType::Categorical];
        let err = d
            .clone()
            .with_feature_types(&numeric_first)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("feature 1 contains invalid category value 2.5"),
            "{err}"
        );
        let d = DMatrix::from_csr(vec![0, 1, 2], vec![1, 0], vec![3.0, 1.5], 2).unwrap();
        assert!(d.with_feature_types(&numeric_first).is_ok());
    }

    #[test]
    fn group_weights_expand_across_query_rows() {
        let d = sample_dense()
            .with_group_sizes(&[2, 1])
            .unwrap()
            .with_group_weights(&[0.5, 2.0])
            .unwrap();
        assert_eq!(d.weights().unwrap(), &[0.5, 0.5, 2.0]);
    }
}
