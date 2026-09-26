//! Stochastic Gradient Langevin Boosting (SGLB) and model shrinkage in the
//! training loop (beyond XGBoost; CatBoost's `langevin`,
//! `diffusion_temperature`, `model_shrink_rate`, `model_shrink_mode`, and
//! `posterior_sampling`).
//!
//! SGLB (Ustimenko and Prokhorenkova, ICML 2021, Algorithm 2) changes three
//! things in every boosting iteration, all mirrored from CatBoost:
//!
//! 1. **Model shrinkage.** Before the iteration's gradients, the model built
//!    so far (every margin, the intercept included) is multiplied by
//!    `s_i = 1 - rate * eta` (`constant`) or `1 - rate / i` (`decreasing`)
//!    for `i >= 1` (`TrainOneIteration` in
//!    `catboost/private/libs/algo/train.cpp`; `s_0 = 1`). This is the exact
//!    step of the L2 prior's gradient, the `(1 - γε)` factor of the paper.
//!    The coefficients are recorded ([`crate::model::Shrinkage`]) and the
//!    trees stored unscaled.
//! 2. **Noisy structure search.** The tree structure is searched on the
//!    gradients plus Gaussian noise of standard deviation
//!    `sigma = sqrt(2 / (eta * T))` (`CalcLangevinNoiseRate` in
//!    `catboost/private/libs/algo_helpers/langevin_utils.cpp`, applied to
//!    the per-row derivatives by `AddLangevinNoiseToDerivatives`): every
//!    row's gradient gets `sigma * ξ`, the same for every row whatever its
//!    Hessian or weight, exactly as CatBoost adds it to every weighted
//!    derivative. This is also the paper's prescription: Algorithm 2 (and
//!    eq. 7) perturbs the gradient vector with isotropic noise
//!    `N(0, (2N / (εβ)) I_N)` for the structure search, with no
//!    per-row (Hessian) covariance.
//! 3. **Independent leaf noise.** With the structure fixed, every leaf is
//!    re-estimated from the noise-free gradients of its rows with fresh noise
//!    on its gradient sum: `G + sigma * sqrt(|H| + λ) * ξ`
//!    (`AddLangevinNoiseToLeafNewtonSum`, CatBoost's Newton leaves), then
//!    XGBoost's leaf weight `-Tα(G) / (H + λ)` (clamped to
//!    `max_delta_step`). The paper draws the structure and the leaf noise
//!    independently so that the tree distribution does not depend on the
//!    leaf noise.
//!
//! Every draw is a keyed standard normal ([`crate::rng::keyed_normal`]):
//! structure noise by `(seed, iteration)` and `row * n_outputs + output`,
//! leaf noise by `(seed, iteration, tree)` and `node * width + output`, so
//! results do not depend on the thread count, and a run stopped after `k`
//! rounds grows exactly the first `k` iterations of a longer one.

use rayon::prelude::*;

use crate::config::{ModelShrinkMode, TrainingParams};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::objective::GradPair;
use crate::rng::{keyed_normal, stream_key};
use crate::tree::RegTree;
use crate::tree::builder::{LeafRows, xgb_calc_weight};
use crate::tree::gain::{GradStats, RegParams};

use super::train::TreeOutput;

/// Stream salt of the structure-search noise.
const STRUCTURE_STREAM: u64 = 0x5347_4C42_5F73_7472;
/// Stream salt of the leaf noise.
const LEAF_STREAM: u64 = 0x5347_4C42_5F6C_6566;
/// Rows per parallel noise task.
const NOISE_CHUNK: usize = 8192;

/// A training run's SGLB settings, resolved against the training rows.
pub(super) struct Sglb {
    /// The Langevin noise, when `langevin` (or posterior sampling) is on.
    pub(super) langevin: Option<Langevin>,
    /// The per-iteration model shrinkage, when its rate is positive.
    pub(super) shrink: Option<Shrink>,
}

impl Sglb {
    /// Resolve `params` for `n_rows` training rows: posterior sampling's
    /// derived temperature and rate (CatBoost's
    /// `AdjustPosteriorSamplingDeafultValues`), whose coefficient must stay
    /// positive.
    pub(super) fn resolve(params: &TrainingParams, n_rows: usize) -> Result<Sglb> {
        if params.posterior_sampling && n_rows == 0 {
            return Err(HessboostError::invalid_param(
                "posterior_sampling",
                "needs training rows to derive its temperature and shrink rate from",
            ));
        }
        let rate = params.effective_model_shrink_rate(n_rows);
        if params.posterior_sampling && rate * params.eta >= 1.0 {
            return Err(HessboostError::invalid_param(
                "posterior_sampling",
                format!(
                    "the shrink coefficient 1 - eta / (2 * rows) must stay positive, got eta {} \
                     for {n_rows} rows",
                    params.eta
                ),
            ));
        }
        let langevin = params.langevin_on().then(|| {
            let temperature = params.effective_diffusion_temperature(n_rows);
            let reg = RegParams::from_params(params);
            Langevin {
                sigma: (2.0 / (params.eta * temperature)).sqrt(),
                reg,
                seed: params.seed,
            }
        });
        let shrink = (rate > 0.0).then_some(Shrink {
            mode: params.model_shrink_mode,
            rate,
            eta: params.eta,
        });
        Ok(Sglb { langevin, shrink })
    }
}

/// The model shrinkage schedule.
pub(super) struct Shrink {
    mode: ModelShrinkMode,
    rate: f64,
    eta: f64,
}

impl Shrink {
    /// The coefficient the model is multiplied by at the start of
    /// `iteration`: `1` for the first, then CatBoost's
    /// `1 - rate * eta` (`constant`) or `1 - rate / iteration`
    /// (`decreasing`).
    pub(super) fn factor(&self, iteration: usize) -> f64 {
        if iteration == 0 {
            return 1.0;
        }
        match self.mode {
            ModelShrinkMode::Constant => 1.0 - self.rate * self.eta,
            ModelShrinkMode::Decreasing => 1.0 - self.rate / iteration as f64,
        }
    }
}

/// The Langevin noise of a run: its scale `sigma = sqrt(2 / (eta * T))`,
/// the leaf regularization, and the seed keying its streams.
pub(super) struct Langevin {
    sigma: f64,
    reg: RegParams,
    seed: u64,
}

/// Where a tree's leaves are re-estimated: the noise-free gradients of
/// every output (`[row][n_out]`), the rows the tree was grown on, and the
/// builder's final row partitions when it kept them (else the rows are
/// routed through the tree).
pub(super) struct LeafRenewal<'a> {
    pub(super) data: &'a DMatrix,
    pub(super) gpair: &'a [GradPair],
    pub(super) n_out: usize,
    pub(super) rows: &'a [u32],
    pub(super) leaf_rows: &'a [LeafRows],
    pub(super) iteration: usize,
    /// The tree's index within its iteration (keys its leaf noise).
    pub(super) tree: usize,
}

impl Langevin {
    /// Fill `noisy` with `gpair` (`[row][n_out]`) plus iteration
    /// `iteration`'s structure noise, `g + sigma * ξ` per cell (CatBoost's
    /// unit per-row noise; Hessians unchanged), and return it.
    pub(super) fn structure_gradients<'a>(
        &self,
        gpair: &[GradPair],
        iteration: usize,
        noisy: &'a mut Vec<GradPair>,
    ) -> &'a [GradPair] {
        let key = stream_key(&[self.seed, STRUCTURE_STREAM, iteration as u64]);
        let sigma = self.sigma;
        noisy.clear();
        noisy.extend_from_slice(gpair);
        noisy
            .par_chunks_mut(NOISE_CHUNK)
            .enumerate()
            .for_each(|(chunk, cells)| {
                let first = chunk * NOISE_CHUNK;
                for (i, cell) in cells.iter_mut().enumerate() {
                    let z = keyed_normal(key, (first + i) as u64);
                    cell.grad = (f64::from(cell.grad) + sigma * z) as f32;
                }
            });
        noisy
    }

    /// Re-estimate every leaf of `tree` (feeding `output`) from the
    /// noise-free gradients of its rows plus independent leaf noise, before
    /// the learning rate is applied.
    pub(super) fn renew_leaves(&self, tree: &mut RegTree, output: TreeOutput, at: &LeafRenewal) {
        let (first, width) = match output {
            TreeOutput::Scalar(k) => (k, 1),
            TreeOutput::Vector => (0, at.n_out),
        };
        let mut stats = vec![GradStats::default(); tree.num_nodes() * width];
        let mut add = |node: usize, row: u32| {
            let cells = &at.gpair[row as usize * at.n_out + first..][..width];
            for (stat, &cell) in stats[node * width..][..width].iter_mut().zip(cells) {
                stat.add(GradStats::from_pair(cell));
            }
        };
        if at.leaf_rows.is_empty() {
            let leaves: Vec<usize> = at
                .rows
                .par_iter()
                .with_min_len(1024)
                .map(|&row| tree.leaf_id_with(|f| at.data.get(row as usize, f as usize)))
                .collect();
            for (&row, &leaf) in at.rows.iter().zip(&leaves) {
                add(leaf, row);
            }
        } else {
            for leaf in at.leaf_rows {
                for &row in &leaf.rows {
                    add(leaf.node, row);
                }
            }
        }
        let key = stream_key(&[self.seed, LEAF_STREAM, at.iteration as u64, at.tree as u64]);
        let mut values = vec![0.0f32; width];
        for node in 0..tree.num_nodes() {
            if !tree.node(node).is_leaf() {
                continue;
            }
            for (out, value) in values.iter_mut().enumerate() {
                let GradStats { grad, hess } = stats[node * width + out];
                let scale = self.sigma * (hess.abs() + self.reg.lambda).sqrt();
                let z = keyed_normal(key, (node * width + out) as u64);
                *value = xgb_calc_weight(GradStats::new(grad + scale * z, hess), &self.reg) as f32;
            }
            match output {
                TreeOutput::Scalar(_) => tree.set_leaf_value(node, values[0]),
                TreeOutput::Vector => tree.set_leaf_vector(node, &values),
            }
        }
    }
}
