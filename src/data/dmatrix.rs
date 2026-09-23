//! The core dataset container: a feature matrix plus training metadata.

use crate::data::meta::{FeatureType, GroupInfo, MetaInfo};
use crate::error::{HessboostError, Result};

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
/// equal `expected`. Preserves the [`HessboostError::DimensionMismatch`] shape
/// each call site already returned.
#[inline]
pub(crate) fn check_len(what: &'static str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(HessboostError::DimensionMismatch {
            what,
            expected,
            got,
        });
    }
    Ok(())
}

/// Validate CSR `indptr` against `nnz` stored entries: first offset 0,
/// monotonic offsets within bounds, terminal offset `== nnz`.
/// Caller must ensure `indptr` is non-empty (`from_csr` rejects that first).
pub(crate) fn check_csr(indptr: &[usize], nnz: usize) -> Result<()> {
    if indptr[0] != 0 {
        return Err(HessboostError::invalid_param(
            "csr indptr",
            "the first offset must be 0",
        ));
    }
    for pair in indptr.windows(2) {
        if pair[0] > pair[1] || pair[1] > nnz {
            return Err(HessboostError::invalid_param(
                "csr indptr",
                "offsets must be monotonic and within the values array",
            ));
        }
    }
    check_len("csr indptr terminal", indptr[indptr.len() - 1], nnz)
}

/// Reject non-finite labels (shared by the single- and multi-target setters).
fn check_finite_labels(labels: &[f32]) -> Result<()> {
    if labels.iter().any(|v| !v.is_finite()) {
        return Err(HessboostError::invalid_param(
            "labels",
            "all labels must be finite",
        ));
    }
    Ok(())
}

/// Backing storage for the feature matrix.
#[derive(Debug, Clone)]
enum Storage {
    /// Row-major dense matrix of length `n_rows * n_cols`.
    Dense(Vec<f32>),
    /// Compressed sparse row: `indptr` has `n_rows + 1` entries.
    Csr {
        indptr: Vec<usize>,
        indices: Vec<u32>,
        values: Vec<f32>,
    },
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
#[derive(Debug, Clone)]
pub struct DMatrix {
    n_rows: usize,
    n_cols: usize,
    storage: Storage,
    missing: f32,
    /// Row-major `[row][target]`, length `n_rows * n_targets`.
    labels: Option<Vec<f32>>,
    n_targets: usize,
    label_lower_bound: Option<Vec<f32>>,
    label_upper_bound: Option<Vec<f32>>,
    weights: Option<Vec<f32>>,
    base_margin: Option<Vec<f32>>,
    group: Option<GroupInfo>,
    feature_types: Vec<FeatureType>,
    feature_weights: Option<Vec<f32>>,
}

/// A single materialized `(feature_index, value)` entry from a row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry {
    /// Column index of the feature.
    pub index: u32,
    /// Feature value (guaranteed non-missing when yielded by row iterators).
    pub value: f32,
}

impl DMatrix {
    /// Shared constructor: validated storage plus default (empty) metadata.
    fn new(n_rows: usize, n_cols: usize, storage: Storage, missing: f32) -> Self {
        DMatrix {
            n_rows,
            n_cols,
            storage,
            missing,
            labels: None,
            n_targets: 1,
            label_lower_bound: None,
            label_upper_bound: None,
            weights: None,
            base_margin: None,
            group: None,
            feature_types: vec![FeatureType::Numerical; n_cols],
            feature_weights: None,
        }
    }

    /// Build a dense matrix from a row-major slice of length `n_rows * n_cols`.
    /// The missing sentinel defaults to NaN.
    pub fn from_dense(data: &[f32], n_rows: usize, n_cols: usize) -> Result<Self> {
        Self::from_dense_with_missing(data, n_rows, n_cols, f32::NAN)
    }

    /// Build a dense matrix with an explicit missing-value sentinel.
    pub fn from_dense_with_missing(
        data: &[f32],
        n_rows: usize,
        n_cols: usize,
        missing: f32,
    ) -> Result<Self> {
        if n_rows == 0 || n_cols == 0 {
            return Err(HessboostError::EmptyDataset(
                "from_dense: zero rows or columns",
            ));
        }
        let expected = n_rows.checked_mul(n_cols).ok_or_else(|| {
            HessboostError::invalid_param("matrix shape", "n_rows * n_cols overflows usize")
        })?;
        check_len("dense data length", data.len(), expected)?;
        // With the NaN sentinel the only rejected values are infinities, a
        // branch-free check the compiler vectorizes; other sentinels need the
        // general test.
        let invalid = if missing.is_nan() {
            data.iter().any(|v| v.is_infinite())
        } else {
            data.iter().any(|&v| v != missing && !v.is_finite())
        };
        if invalid {
            return Err(HessboostError::invalid_param(
                "dense data",
                "non-missing feature values must be finite",
            ));
        }
        Ok(Self::new(
            n_rows,
            n_cols,
            Storage::Dense(data.to_vec()),
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
        if values.iter().any(|v| !v.is_finite()) {
            return Err(HessboostError::invalid_param(
                "csr values",
                "stored feature values must be finite",
            ));
        }
        for row in 0..n_rows {
            let mut seen = std::collections::HashSet::new();
            for &col in &indices[indptr[row]..indptr[row + 1]] {
                if !seen.insert(col) {
                    return Err(HessboostError::invalid_param(
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
                indptr,
                indices,
                values,
            },
            f32::NAN,
        ))
    }

    /// Attach regression/classification labels (`len == n_rows`), one target
    /// per row.
    pub fn with_labels(mut self, labels: &[f32]) -> Result<Self> {
        check_len("labels", labels.len(), self.n_rows)?;
        check_finite_labels(labels)?;
        self.labels = Some(labels.to_vec());
        self.n_targets = 1;
        Ok(self)
    }

    /// Attach a label matrix with `n_targets` targets per row, laid out
    /// row-major `[row][target]` (`len == n_rows * n_targets`).
    pub fn with_label_matrix(mut self, labels: &[f32], n_targets: usize) -> Result<Self> {
        if n_targets == 0 {
            return Err(HessboostError::invalid_param(
                "labels",
                "n_targets must be at least 1",
            ));
        }
        let expected = self.n_rows.checked_mul(n_targets).ok_or_else(|| {
            HessboostError::invalid_param("labels", "n_rows * n_targets overflows usize")
        })?;
        check_len("labels", labels.len(), expected)?;
        check_finite_labels(labels)?;
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
        if lower.iter().any(|v| v.is_nan()) {
            return Err(HessboostError::invalid_param(
                "label_lower_bound",
                "label bounds must not be NaN",
            ));
        }
        if upper.iter().any(|v| v.is_nan()) {
            return Err(HessboostError::invalid_param(
                "label_upper_bound",
                "label bounds must not be NaN",
            ));
        }
        self.label_lower_bound = Some(lower.to_vec());
        self.label_upper_bound = Some(upper.to_vec());
        Ok(self)
    }

    /// Attach per-feature sampling weights (`len == n_cols`), used by the
    /// column sampler.
    pub fn with_feature_weights(mut self, weights: &[f32]) -> Result<Self> {
        check_len("feature_weights", weights.len(), self.n_cols)?;
        if weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(HessboostError::invalid_param(
                "feature_weights",
                "feature weights must be finite and non-negative",
            ));
        }
        if !weights.iter().any(|v| *v > 0.0) {
            return Err(HessboostError::invalid_param(
                "feature_weights",
                "at least one feature weight must be positive",
            ));
        }
        self.feature_weights = Some(weights.to_vec());
        Ok(self)
    }

    /// Attach per-instance weights (`len == n_rows`).
    pub fn with_weights(mut self, weights: &[f32]) -> Result<Self> {
        check_len("weights", weights.len(), self.n_rows)?;
        if weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(HessboostError::invalid_param(
                "weights",
                "weights must be finite and non-negative",
            ));
        }
        if !weights.iter().any(|v| *v > 0.0) {
            return Err(HessboostError::invalid_param(
                "weights",
                "at least one weight must be positive",
            ));
        }
        self.weights = Some(weights.to_vec());
        Ok(self)
    }

    /// Attach a per-instance base margin (raw prediction offset, `len == n_rows`
    /// for single-output objectives).
    pub fn with_base_margin(mut self, base_margin: &[f32]) -> Result<Self> {
        if base_margin.is_empty() || !base_margin.len().is_multiple_of(self.n_rows) {
            return Err(HessboostError::invalid_param(
                "base_margin",
                "length must be a non-zero multiple of n_rows",
            ));
        }
        if base_margin.iter().any(|v| !v.is_finite()) {
            return Err(HessboostError::invalid_param(
                "base_margin",
                "all margins must be finite",
            ));
        }
        self.base_margin = Some(base_margin.to_vec());
        Ok(self)
    }

    /// Attach ranking group information (sizes sum to `n_rows`).
    pub fn with_group_sizes(mut self, sizes: &[usize]) -> Result<Self> {
        if sizes.is_empty() || sizes.contains(&0) {
            return Err(HessboostError::invalid_param(
                "group_sizes",
                "groups must be non-empty and every group must contain a row",
            ));
        }
        let total = sizes.iter().try_fold(0usize, |acc, &s| acc.checked_add(s));
        let Some(total) = total else {
            return Err(HessboostError::invalid_param(
                "group_sizes",
                "group-size sum overflows usize",
            ));
        };
        let g = GroupInfo::from_sizes(sizes);
        check_len("group sizes sum", total, self.n_rows)?;
        self.group = Some(g);
        Ok(self)
    }

    /// Attach one weight per ranking group.
    ///
    /// Group sizes must be attached first. Internally each group weight is
    /// expanded across that group's rows so objectives and metrics receive the
    /// XGBoost-compatible per-query weighting semantics.
    pub fn with_group_weights(mut self, weights: &[f32]) -> Result<Self> {
        let group = self.group.as_ref().ok_or_else(|| {
            HessboostError::invalid_param("group_weights", "attach group sizes first")
        })?;
        check_len("group_weights length", weights.len(), group.num_groups())?;
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
            || !weights.iter().any(|weight| *weight > 0.0)
        {
            return Err(HessboostError::invalid_param(
                "group_weights",
                "weights must be finite and non-negative with at least one positive value",
            ));
        }
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
        for (col, ty) in types.iter().enumerate() {
            if *ty == FeatureType::Categorical {
                for row in 0..self.n_rows {
                    if let Some(v) = self.get(row, col)
                        && (v < 0.0 || v.fract() != 0.0 || v >= u32::MAX as f32)
                    {
                        return Err(HessboostError::invalid_param(
                            "categorical feature",
                            format!("feature {col} contains invalid category value {v}"),
                        ));
                    }
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
        self.n_targets
    }

    /// Lower label bounds for interval-censored labels, if attached.
    #[inline]
    pub fn label_lower_bound(&self) -> Option<&[f32]> {
        self.label_lower_bound.as_deref()
    }

    /// Upper label bounds for interval-censored labels, if attached.
    #[inline]
    pub fn label_upper_bound(&self) -> Option<&[f32]> {
        self.label_upper_bound.as_deref()
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
            labels: self.labels.as_deref().unwrap_or(&[]),
            n_targets: self.n_targets,
            weights: self.weights.as_deref(),
            group: self.group.as_ref(),
            label_lower_bound: self.label_lower_bound.as_deref(),
            label_upper_bound: self.label_upper_bound.as_deref(),
        }
    }

    /// Fetch a single value, returning `None` when the entry is missing.
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row >= self.n_rows || col >= self.n_cols {
            return None;
        }
        match &self.storage {
            Storage::Dense(data) => {
                let v = data[row * self.n_cols + col];
                if is_missing(v, self.missing) {
                    None
                } else {
                    Some(v)
                }
            }
            Storage::Csr {
                indptr,
                indices,
                values,
            } => {
                let (s, e) = (indptr[row], indptr[row + 1]);
                // Rows are not assumed sorted by column; linear scan of the row.
                for k in s..e {
                    if indices[k] as usize == col {
                        let v = values[k];
                        return if is_missing(v, self.missing) {
                            None
                        } else {
                            Some(v)
                        };
                    }
                }
                None
            }
        }
    }

    /// Raw row-major storage of a dense matrix (missing entries hold the
    /// sentinel), or `None` for sparse storage.
    #[inline]
    pub(crate) fn dense_values(&self) -> Option<&[f32]> {
        match &self.storage {
            Storage::Dense(data) => Some(data),
            Storage::Csr { .. } => None,
        }
    }

    /// Raw `(indptr, indices, values)` of a CSR matrix, or `None` for dense storage.
    /// Entries equal to the missing sentinel may be present and must be treated
    /// as absent by callers.
    #[inline]
    pub(crate) fn csr_parts(&self) -> Option<(&[usize], &[u32], &[f32])> {
        match &self.storage {
            Storage::Dense(_) => None,
            Storage::Csr {
                indptr,
                indices,
                values,
            } => Some((indptr, indices, values)),
        }
    }

    /// Copy of this matrix, metadata included, with every non-missing stored
    /// value replaced by `f(row, col, value)`. Missing dense entries become NaN
    /// and the result's sentinel is NaN, so a mapped value can never collide
    /// with a non-NaN sentinel. Feature types are copied unchanged.
    pub(crate) fn map_values(&self, mut f: impl FnMut(usize, usize, f32) -> f32) -> Self {
        let mut out = self.clone();
        match &mut out.storage {
            Storage::Dense(data) => {
                for (row, values) in data.chunks_exact_mut(self.n_cols).enumerate() {
                    for (col, v) in values.iter_mut().enumerate() {
                        *v = if is_missing(*v, self.missing) {
                            f32::NAN
                        } else {
                            f(row, col, *v)
                        };
                    }
                }
            }
            Storage::Csr {
                indptr,
                indices,
                values,
            } => {
                for row in 0..self.n_rows {
                    for k in indptr[row]..indptr[row + 1] {
                        if !is_missing(values[k], self.missing) {
                            values[k] = f(row, indices[k] as usize, values[k]);
                        }
                    }
                }
            }
        }
        out.missing = f32::NAN;
        out
    }

    /// Visit every non-missing `(index, value)` entry of `row`, in storage
    /// order. `row` must be in bounds.
    fn for_row_entry(&self, row: usize, mut f: impl FnMut(u32, f32)) {
        match &self.storage {
            Storage::Dense(data) => {
                let base = row * self.n_cols;
                for c in 0..self.n_cols {
                    let v = data[base + c];
                    if !is_missing(v, self.missing) {
                        f(c as u32, v);
                    }
                }
            }
            Storage::Csr {
                indptr,
                indices,
                values,
            } => {
                let (s, e) = (indptr[row], indptr[row + 1]);
                for k in s..e {
                    let v = values[k];
                    if !is_missing(v, self.missing) {
                        f(indices[k], v);
                    }
                }
            }
        }
    }

    /// Materialize a single row's non-missing `(index, value)` entries into
    /// `out`. Reuses the buffer to avoid per-row allocation in hot loops.
    pub fn row_into(&self, row: usize, out: &mut Vec<Entry>) {
        out.clear();
        if row >= self.n_rows {
            return;
        }
        self.for_row_entry(row, |index, value| out.push(Entry { index, value }));
    }

    /// Build a compressed-sparse-**column** view for column-oriented split
    /// finding (used by the exact tree method). Each column lists its
    /// non-missing `(row, value)` pairs.
    pub fn to_csc(&self) -> CscView {
        let mut col_counts = vec![0usize; self.n_cols];
        self.for_each_entry(|_row, col, _v| col_counts[col as usize] += 1);

        let mut col_ptr = vec![0usize; self.n_cols + 1];
        #[allow(clippy::needless_range_loop)]
        for c in 0..self.n_cols {
            col_ptr[c + 1] = col_ptr[c] + col_counts[c];
        }
        let nnz = col_ptr[self.n_cols];
        let mut rows = vec![0u32; nnz];
        let mut vals = vec![0f32; nnz];
        let mut cursor = col_ptr.clone();
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

    /// Build a new matrix containing only `rows` (in the given order), carrying
    /// over labels (every target of each row), label bounds, weights, base
    /// margin, and feature metadata. Used for cross-validation folds. Ranking
    /// group info is not carried over.
    pub fn select_rows(&self, rows: &[usize]) -> Result<Self> {
        if let Some(&row) = rows.iter().find(|&&row| row >= self.n_rows) {
            return Err(HessboostError::invalid_param(
                "rows",
                format!("row index {row} is out of bounds for {} rows", self.n_rows),
            ));
        }
        let mut indptr = Vec::with_capacity(rows.len() + 1);
        indptr.push(0usize);
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        let mut buf: Vec<Entry> = Vec::new();
        for &r in rows {
            self.row_into(r, &mut buf);
            for e in &buf {
                indices.push(e.index);
                values.push(e.value);
            }
            indptr.push(values.len());
        }
        let mut out = DMatrix::from_csr(indptr, indices, values, self.n_cols)?;
        out.feature_types.clone_from(&self.feature_types);
        out.feature_weights.clone_from(&self.feature_weights);
        out.n_targets = self.n_targets;
        if let Some(l) = &self.labels {
            let k = self.n_targets;
            let mut selected = Vec::with_capacity(rows.len() * k);
            for &r in rows {
                selected.extend_from_slice(&l[r * k..(r + 1) * k]);
            }
            out.labels = Some(selected);
        }
        if let Some(lo) = &self.label_lower_bound {
            out.label_lower_bound = Some(rows.iter().map(|&r| lo[r]).collect());
        }
        if let Some(hi) = &self.label_upper_bound {
            out.label_upper_bound = Some(rows.iter().map(|&r| hi[r]).collect());
        }
        if let Some(w) = &self.weights {
            out.weights = Some(rows.iter().map(|&r| w[r]).collect());
        }
        if let Some(bm) = &self.base_margin {
            let nout = bm.len() / self.n_rows;
            let mut selected = Vec::with_capacity(rows.len() * nout);
            for &r in rows {
                selected.extend_from_slice(&bm[r * nout..(r + 1) * nout]);
            }
            out.base_margin = Some(selected);
        }
        Ok(out)
    }

    /// Visit every non-missing entry as `(row, col, value)`.
    fn for_each_entry(&self, mut f: impl FnMut(usize, u32, f32)) {
        match &self.storage {
            Storage::Dense(data) => {
                for r in 0..self.n_rows {
                    let base = r * self.n_cols;
                    for c in 0..self.n_cols {
                        let v = data[base + c];
                        if !is_missing(v, self.missing) {
                            f(r, c as u32, v);
                        }
                    }
                }
            }
            Storage::Csr {
                indptr,
                indices,
                values,
            } => {
                for r in 0..self.n_rows {
                    for k in indptr[r]..indptr[r + 1] {
                        let v = values[k];
                        if !is_missing(v, self.missing) {
                            f(r, indices[k], v);
                        }
                    }
                }
            }
        }
    }
}

/// A compressed-sparse-column view of a [`DMatrix`], built by
/// [`DMatrix::to_csc`]. Within each column the `(row, value)` pairs are stored
/// in row order. Callers that need value-sorted order (e.g. the exact split
/// finder) sort per-column slices themselves.
#[derive(Debug, Clone)]
pub struct CscView {
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

    #[test]
    fn dense_get_and_missing() {
        let d = sample_dense();
        assert_eq!(d.get(0, 0), Some(1.0));
        assert_eq!(d.get(1, 0), None); // missing
        assert_eq!(d.get(1, 1), Some(5.0));
    }

    #[test]
    fn row_into_skips_missing() {
        let d = sample_dense();
        let mut buf = Vec::new();
        d.row_into(1, &mut buf);
        assert_eq!(
            buf,
            vec![Entry {
                index: 1,
                value: 5.0
            }]
        );
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
        assert_eq!((info.n_rows, info.n_targets), (3, 2));
        assert_eq!(info.labels, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        assert!(censored.info().labels.is_empty());
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
