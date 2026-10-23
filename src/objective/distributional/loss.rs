//! [`DistLoss`]: the `dist:*` training loss.

use super::{DistFamily, DistGradient, DistSplitDirection, MAX_PARAMS};
use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};
use crate::objective::{GradPair, Loss, MIN_HESS, SplitGradient, check_label_domain};
use crate::rng::splitmix64;

/// Floor of the second-order statistic: the objectives' [`MIN_HESS`], widened.
pub(super) const MIN_CURVATURE: f64 = MIN_HESS as f64;

/// A `dist:*` objective: the negative log-likelihood of a [`DistFamily`],
/// with the second-order statistic chosen by [`DistGradient`] (see the
/// [module docs](self)).
#[derive(Debug, Clone, Copy)]
pub(crate) struct DistLoss {
    family: DistFamily,
    gradient: DistGradient,
    /// Parallel-gradient-boosting direction and seed for shared trees, set
    /// only for `multi_strategy = multi_output_tree`.
    shared: Option<(DistSplitDirection, u64)>,
}

impl DistLoss {
    /// The objective for `family` with gradient mode `gradient`, growing one
    /// tree per parameter (no reduced split gradients).
    pub(crate) fn new(family: DistFamily, gradient: DistGradient) -> Self {
        DistLoss {
            family,
            gradient,
            shared: None,
        }
    }

    /// Grow shared vector-leaf trees (`multi_strategy = multi_output_tree`)
    /// with the given split direction: [`Loss::split_gradient`] then
    /// returns the gradients of the parameter the direction selects for the
    /// round (`seed` drives [`DistSplitDirection::Random`]), or `None` for
    /// [`DistSplitDirection::All`] and one-parameter families.
    #[must_use]
    pub(crate) fn with_split_direction(mut self, direction: DistSplitDirection, seed: u64) -> Self {
        self.shared = Some((direction, seed));
        self
    }

    /// The parameter whose gradients drive the structure of round
    /// `iteration`'s shared tree, if any.
    pub(crate) fn split_parameter(&self, iteration: usize) -> Option<usize> {
        let k = self.family.n_params();
        match self.shared? {
            _ if k < 2 => None,
            (DistSplitDirection::All, _) => None,
            (DistSplitDirection::Cyclic, _) => Some(iteration % k),
            (DistSplitDirection::Random, seed) => {
                let draw = splitmix64(seed ^ splitmix64(iteration as u64));
                Some((draw % k as u64) as usize)
            }
        }
    }

    /// One row's `(gradient, curvature)` per parameter, before row weights.
    /// The Fisher and natural modes take
    /// [`DistFamily::fisher_curvature`] (the Fisher diagonal, with the
    /// Student-t `κ` entry floored at the magnitude of its observed curvature).
    fn row_pairs(self, eta: &[f64], y: f64) -> ([f64; MAX_PARAMS], [f64; MAX_PARAMS]) {
        let g = self.family.gradient(eta, y);
        match self.gradient {
            DistGradient::Fisher => (
                g,
                self.family
                    .fisher_curvature(eta, y)
                    .map(|v| v.max(MIN_CURVATURE)),
            ),
            DistGradient::Hessian => {
                let h = self.family.hessian(eta, y);
                (g, std::array::from_fn(|j| h[j][j].max(MIN_CURVATURE)))
            }
            DistGradient::Natural => {
                let i = self
                    .family
                    .fisher_curvature(eta, y)
                    .map(|v| v.max(MIN_CURVATURE));
                (std::array::from_fn(|j| g[j] / i[j]), [1.0; MAX_PARAMS])
            }
        }
    }
}

impl Loss for DistLoss {
    fn name(&self) -> &str {
        self.family.objective_name()
    }

    fn n_outputs(&self) -> usize {
        self.family.n_params()
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        let k = self.family.n_params();
        let n = labels.len();
        crate::objective::rowwise_gradient(
            n,
            k,
            preds,
            labels,
            weights,
            out,
            |preds, labels, weights, out| {
                for (i, (row, out_row)) in preds
                    .chunks_exact(k)
                    .zip(out.chunks_exact_mut(k))
                    .enumerate()
                {
                    let w = weights.map_or(1.0, |ws| f64::from(ws[i]));
                    if w == 0.0 {
                        out_row.fill(GradPair::default());
                        continue;
                    }
                    let mut eta = [0.0; MAX_PARAMS];
                    for (e, &m) in eta.iter_mut().zip(row) {
                        *e = f64::from(m);
                    }
                    let (g, h) = self.row_pairs(&eta[..k], f64::from(labels[i]));
                    for (j, o) in out_row.iter_mut().enumerate() {
                        *o = GradPair::new((w * g[j]) as f32, (w * h[j]) as f32);
                    }
                }
            },
        );
    }

    /// Margins to natural parameters (the log links, clamped; the
    /// Student-t scale from its orthogonal margin).
    fn pred_transform(&self, preds: &mut [f32]) {
        let k = self.family.n_params();
        let mut eta = [0.0; MAX_PARAMS];
        for row in preds.chunks_exact_mut(k) {
            for (e, &m) in eta.iter_mut().zip(row.iter()) {
                *e = f64::from(m);
            }
            let natural = self.family.natural_params(&eta[..k]);
            for (v, p) in row.iter_mut().zip(natural) {
                *v = p as f32;
            }
        }
    }

    /// Natural parameters to margins (inverse links; a parameter outside
    /// the link's range maps to `NaN`, which training rejects). The
    /// Student-t margins are `ln s = ln σ + ln(1 + 1/ν)` and
    /// `κ = ln(ν - 1)` (`NaN` for `ν <= 1`).
    fn probs_to_margins(&self, scores: &mut [f32]) {
        let k = self.family.n_params();
        for row in scores.chunks_exact_mut(k) {
            if self.family == DistFamily::StudentT {
                let (sigma, df) = (f64::from(row[1]), f64::from(row[2]));
                row[1] = if sigma > 0.0 && df > 1.0 {
                    (sigma.ln() + df.recip().ln_1p()) as f32
                } else {
                    f32::NAN
                };
                row[2] = if df > 1.0 {
                    (df - 1.0).ln() as f32
                } else {
                    f32::NAN
                };
                continue;
            }
            for (j, v) in row.iter_mut().enumerate() {
                if self.family.log_link(j) {
                    *v = if *v > 0.0 { v.ln() } else { f32::NAN };
                }
            }
        }
    }

    /// A scalar cannot set several distribution parameters; a one-parameter
    /// family (`dist:poisson`) takes a value in its support.
    fn validate_base_score(&self, base_score: f64) -> Result<()> {
        if self.family.n_params() > 1 {
            return Err(HessboostError::invalid_param(
                "base_score",
                "a scalar cannot set the several parameters of a `dist:*` objective; \
                 supply per-row `base_margin` instead",
            ));
        }
        if self.family.log_link(0) {
            crate::objective::check_base_score_domain(
                base_score,
                crate::objective::OutputDomain::Positive,
            )?;
        }
        Ok(())
    }

    /// Maximum-likelihood fit of the marginal label distribution.
    fn base_margins_info(&self, info: &MetaInfo) -> Vec<f32> {
        self.family
            .mle_margins(info.label_values(), info.weights)
            .into_iter()
            .map(|m| m as f32)
            .collect()
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        check_label_domain(info, |y| self.family.below_support(f64::from(y)))
    }

    fn default_metric(&self) -> crate::metric::EvalMetric {
        crate::metric::EvalMetric::Nll(self.family)
    }

    /// Parallel gradient boosting (Chapelle et al., 2026): the chosen
    /// parameter's gradient column, `⟨∇L_i, e_m⟩` per row.
    fn split_gradient(&self, iteration: usize, gpair: &[GradPair]) -> Option<SplitGradient> {
        let m = self.split_parameter(iteration)?;
        let k = self.family.n_params();
        Some(SplitGradient {
            gpair: gpair.iter().skip(m).step_by(k).copied().collect(),
            n_targets: 1,
        })
    }
}
