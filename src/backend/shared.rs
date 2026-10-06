//! The host-side plumbing both GPU backends share: which nodes and
//! prediction calls may run on the GPU, and the data movement around their
//! kernels. The kernels themselves stay backend-specific.

use std::ops::Range;

use rayon::prelude::*;

use crate::backend::exact_sum::SumDomain;
use crate::data::DMatrix;
use crate::data::ghist::GHistIndex;
use crate::error::{HessboostError, Result};
use crate::model::{BoostedModel, Iterations, Predictions, initial_margins};
use crate::objective::GradPair;
use crate::tree::gain::GradStats;

/// Upper bound on GPU buffer sizes (entries), keeping index math in `u32`.
pub(crate) const MAX_BUFFER_ENTRIES: usize = 1 << 30;

/// The shape of the binned index a GPU histogram backend sized its buffers
/// for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IndexShape {
    pub(crate) n_rows: usize,
    pub(crate) n_cols: usize,
    pub(crate) total_bins: usize,
}

impl IndexShape {
    /// The shape of `index`.
    pub(crate) fn of(index: &GHistIndex) -> Self {
        Self {
            n_rows: index.n_rows(),
            n_cols: index.n_cols(),
            total_bins: index.total_bins(),
        }
    }

    /// Whether a build's inputs fit buffers sized for this shape. The
    /// kernels do not bounds-check, so inputs the buffers were not sized for
    /// (another index shape, more rows than the index holds, a row past its
    /// end, another histogram length) must never reach a dispatch; they take
    /// the CPU path, which checks them.
    pub(crate) fn fits(&self, ghist: &GHistIndex, rows: &[u32], out: &[GradStats]) -> bool {
        Self::of(ghist) == *self
            && out.len() == self.total_bins
            && rows.len() <= self.n_rows
            && rows.iter().all(|&r| (r as usize) < self.n_rows)
    }
}

/// The identity and exactness statistics of the gradient slice a GPU
/// histogram backend has staged: a node goes to the GPU only when the slice
/// it is built from is the staged one and its sums are exact on both paths
/// (see `backend::exact_sum`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct StagedSlice {
    /// (address, length) identity of the staged slice; length 0 when
    /// nothing is staged (a staged slice always has `n_rows > 0` entries).
    addr: usize,
    len: usize,
    grad: SumDomain,
    hess: SumDomain,
}

impl StagedSlice {
    /// Nothing staged.
    pub(crate) const NONE: Self = Self {
        addr: 0,
        len: 0,
        grad: SumDomain::EMPTY,
        hess: SumDomain::EMPTY,
    };

    /// The identity and statistics of `gpair`, about to be staged.
    pub(crate) fn of(gpair: &[GradPair]) -> Self {
        Self {
            addr: gpair.as_ptr().addr(),
            len: gpair.len(),
            grad: SumDomain::of_slice(gpair, |p| p.grad),
            hess: SumDomain::of_slice(gpair, |p| p.hess),
        }
    }

    /// Whether `gpair` is the staged slice.
    pub(crate) fn holds(&self, gpair: &[GradPair]) -> bool {
        self.len != 0 && self.len == gpair.len() && self.addr == gpair.as_ptr().addr()
    }

    /// Whether every sum of at most `n` staged gradient pairs is exact on
    /// both paths.
    pub(crate) fn sums_exact(&self, n: usize) -> bool {
        self.grad.sums_exact(n) && self.hess.sums_exact(n)
    }

    /// `p` in grains, the integers the kernels sum: integer multiples of
    /// each component's grain, exact whenever a node's sums can be.
    pub(crate) fn units(&self, p: GradPair) -> [i64; 2] {
        [self.grad.units(p.grad), self.hess.units(p.hess)]
    }

    /// A bin's GPU sums in grains back in value space, exactly (sums below
    /// `2^53` grains, scaled by a power of two).
    pub(crate) fn bin(&self, [grad, hess]: [i64; 2]) -> GradStats {
        GradStats {
            grad: self.grad.value(grad),
            hess: self.hess.value(hess),
        }
    }

    /// Rows one workgroup may scan with a scatter kernel's shared 32-bit
    /// accumulators: a grain count `k` is split as `k = hi * 2^16 + lo`, so
    /// one workgroup's `hi` sum must stay inside an `i32` (`lo` is 16-bit
    /// each and sums inside a `u32`). Past it the node runs on the CPU
    /// backend (or, on Metal, the register kernels), whose sums are exact by
    /// the same argument.
    pub(crate) fn scatter_row_bound(&self) -> usize {
        let bound = |domain: &SumDomain| -> u64 {
            let max = domain.max_units();
            if max == 0 {
                return u64::from(u32::MAX);
            }
            let hi = max.div_ceil(1 << 16);
            // Both accumulators stay exact: `hi` in an `i32`, `lo` in a `u32`
            // (the largest 16-bit sum, 65535 per row).
            (((1u64 << 31) - 1) / hi).min((u64::from(u32::MAX) - 1) / 65_535)
        };
        usize::try_from(bound(&self.grad).min(bound(&self.hess))).unwrap_or(usize::MAX)
    }
}

/// Refuse a model whose predictions do not come from its compact forest,
/// the only part of a model the GPU predictors hold: `gblinear` models
/// (linear weights) and `linear_tree` models (per-leaf linear models).
pub(crate) fn ensure_forest_model(model: &BoostedModel) -> Result<()> {
    if model.linear().is_some() {
        return Err(HessboostError::incompatible_model(
            "model",
            "gblinear models predict from their linear weights, not the tree forest",
        ));
    }
    if model
        .trees()
        .iter()
        .any(|tree| tree.linear_leaves().is_some())
    {
        return Err(HessboostError::incompatible_model(
            "model",
            "`linear_tree` models predict through per-leaf linear models, \
             which the GPU forest does not hold",
        ));
    }
    Ok(())
}

/// What a GPU margin prediction leaves to the backend's forest walk.
pub(crate) enum MarginPlan {
    /// The margins, computed without the GPU: on the CPU for a model with
    /// model shrinkage (whose per-iteration shrink-then-add arithmetic
    /// repeats training's), or just the base margins for an empty batch or
    /// tree range.
    Done(Predictions),
    /// Walk trees `trees` over every row, adding each tree's leaf value onto
    /// `margins` (the base margins, `n_rows × n_outputs`, row-major).
    Walk {
        trees: Range<usize>,
        margins: Vec<f32>,
    },
}

/// Everything a GPU margin prediction of `data` from `iterations` does
/// before the forest walk: the CPU path for shrunk models, the data and
/// iteration checks, the base margins, and the bound on the dense row copy
/// the backends upload ([`materialize_rows`]).
pub(crate) fn plan_margins(
    model: &BoostedModel,
    data: &DMatrix,
    iterations: Iterations,
) -> Result<MarginPlan> {
    if model.shrinkage().is_some() {
        return model.predict_margin(data, iterations).map(MarginPlan::Done);
    }
    model.validate_prediction_data(data)?;
    let trees = model.iteration_trees(model.resolve_iterations(iterations, "iterations")?);
    let n = data.n_rows();
    let n_cols = data.n_cols();
    let margins = initial_margins(model.base_scores(), data);
    if trees.is_empty() || n == 0 {
        return Ok(MarginPlan::Done(Predictions::new(
            margins,
            n,
            model.n_outputs(),
        )));
    }
    if n > MAX_BUFFER_ENTRIES || n.checked_mul(n_cols).is_none_or(|e| e > MAX_BUFFER_ENTRIES) {
        return Err(HessboostError::invalid_data(
            "data",
            format!(
                "GPU prediction needs a dense row copy ({n} rows x {n_cols} features) \
                 that exceeds 4 GiB"
            ),
        ));
    }
    Ok(MarginPlan::Walk { trees, margins })
}

/// Write `data`'s rows starting at row `begin` into `rows` as a dense
/// `NaN`-for-missing matrix, the same materialization the CPU's row blocks
/// use: dense NaN-sentinel matrices copy in place, a dense matrix with
/// another sentinel maps sentinel values to `NaN`, and CSR rows materialize
/// per entry. `rows` holds a whole number of rows; it is one prediction
/// block of the batch.
pub(crate) fn materialize_rows(data: &DMatrix, begin: usize, rows: &mut [f32]) {
    let n_cols = data.n_cols();
    if let Some(dense) = data.dense_values()
        && data.missing().is_nan()
        && dense.len() == data.n_rows() * n_cols
    {
        let start = begin * n_cols;
        rows.copy_from_slice(&dense[start..start + rows.len()]);
        return;
    }
    let missing = data.missing();
    rows.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(i, row)| {
            let r = begin + i;
            for (f, slot) in row.iter_mut().enumerate() {
                *slot = match data.get(r, f) {
                    Some(v) if v != missing || missing.is_nan() => v,
                    _ => f32::NAN,
                };
            }
        });
}

/// Helpers both backends' unit tests share.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::data::DMatrix;
    use crate::data::ghist::GHistIndex;
    use crate::data::quantile::HistCuts;
    use crate::objective::GradPair;
    use crate::tree::gain::GradStats;
    use crate::tree::hist::{CpuBackend, HistogramBackend};

    /// The single-threaded CPU histogram of `rows`: what a GPU build must
    /// reproduce bit for bit.
    pub(crate) fn cpu_hist(index: &GHistIndex, rows: &[u32], gpair: &[GradPair]) -> Vec<GradStats> {
        let mut out = vec![GradStats::default(); index.total_bins()];
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| CpuBackend.build(index, rows, gpair, &mut out));
        out
    }

    /// A one-feature dataset whose rows cycle through `values` distinct
    /// feature values (one bin each).
    pub(crate) fn one_feature(n: usize, values: usize) -> GHistIndex {
        let x: Vec<f32> = (0..n).map(|i| (i % values) as f32).collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256))
    }
}
