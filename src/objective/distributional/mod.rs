//! Distributional (probabilistic) boosting: objectives that predict a full
//! conditional distribution `p(y | x)` instead of a point, in the style of
//! NGBoost (Duan et al., 2020, arXiv:1910.03225) and XGBoostLSS (März, 2019,
//! arXiv:1907.03178). Opt-in and beyond XGBoost: the objective names live
//! outside XGBoost's namespace, and XGBoost-format export refuses them.
//!
//! # Families and parameterizations
//!
//! Objective `dist:<family>` gives the model one output per distribution
//! parameter (`n_outputs = n_params`, one tree per parameter and round on the
//! ordinary `one_output_per_tree` path, or one shared tree per round with
//! `multi_output_tree`, see below). Each output is an *unconstrained*
//! margin `η`; positive parameters use a log link. Log-link margins are
//! clamped to `[-30, 30]` ([`LOG_LINK_BOUND`]) before the link, in the
//! gradients as in prediction, so no parameter over- or underflows.
//!
//! | objective | [`DistFamily`] | margins | natural parameters |
//! |---|---|---|---|
//! | `dist:normal` | `Normal` | `(μ, ln σ)` | mean `μ`, standard deviation `σ` |
//! | `dist:lognormal` | `LogNormal` | `(μ, ln σ)` | `ln y ~ N(μ, σ²)` |
//! | `dist:gamma` | `Gamma` | `(ln m, ln a)` | mean `m`, shape `a` (rate `a / m`) |
//! | `dist:poisson` | `Poisson` | `ln λ` | rate `λ` |
//! | `dist:negbinomial` | `NegativeBinomial` | `(ln m, ln r)` | mean `m`, size `r` (variance `m + m²/r`) |
//!
//! The parameterizations are chosen *orthogonal*: the Fisher information of
//! each family is diagonal in its margins, so the diagonal Fisher that fits
//! XGBoost's per-output second-order trees is the full Fisher matrix, and
//! the per-row natural gradient `I(η)⁻¹ ∇η` is elementwise.
//!
//! # Gradients and the "Hessian" (`dist_gradient`)
//!
//! The loss is the negative log-likelihood `-ln p(y | η)` (the log scoring
//! rule). Every mode uses its exact gradient `g = ∇η NLL`; the
//! [`DistGradient`] parameter selects what the trees see as the second-order
//! statistic:
//!
//! - [`DistGradient::Fisher`] (default): `(g, diag I(η))`, the expected
//!   Hessian (Fisher scoring). Always positive, and independent of the label.
//!   Because the Fisher matrix is diagonal here, a leaf's Newton step
//!   `-Σg / (ΣI + λ)` is a natural-gradient step.
//! - [`DistGradient::Hessian`]: `(g, diag ∇²η NLL)`, the diagonal of the
//!   exact (observed) Hessian as XGBoostLSS uses it. The negative-binomial
//!   size entry turns negative for counts far above the mean and the
//!   Normal / LogNormal `ln σ` entry `2z²` vanishes at `y = μ`; values are
//!   floored at `1e-16`, like XGBoost's own objectives.
//! - [`DistGradient::Natural`]: NGBoost's natural gradient, `(I(η)⁻¹ g, 1)`:
//!   trees regress the per-row natural gradient by (weighted) least
//!   squares, and `eta` is the step size (no line search).
//!
//! Row weights multiply both statistics. The per-family formulas (`t = y/m`,
//! `z = (y - μ)/σ`, `ψ` digamma, `ψ'` trigamma):
//!
//! - Normal: `g = (-z/σ, 1 - z²)`, `I = (1/σ², 2)`, exact diagonal
//!   `(1/σ², 2z²)`. LogNormal is the same on `ln y`.
//! - Gamma: `g = (a(1 - t), a(ψ(a) - ln a + t - 1 - ln t))`,
//!   `I = (a, a²(ψ'(a) - 1/a))`, exact diagonal `(a t, g₂ + a²(ψ'(a) - 1/a))`.
//! - Poisson: `g = λ - y`, `I = λ` (exactly `count:poisson` without its
//!   `max_delta_step` Hessian inflation).
//! - Negative binomial: `g₁ = r(m - y)/(r + m)`, `g₂ = -r D` with
//!   `D = ψ(y + r) - ψ(r) + ln(r/(r + m)) + (m - y)/(r + m)`;
//!   `I₁ = m r/(m + r)` and `I₂ = r² (E[ψ'(r) - ψ'(Y + r)] - m/(r(r + m)))`.
//!   The expectation has no closed form: it is the series
//!   `Σ_k P(Y > k)/(r + k)²`, summed from the probability mass function
//!   until the upper tail falls below `1e-12` (at most 100 000 terms, then
//!   closed with a geometric-tail estimate; the terms below
//!   `m - 12·sd` use `P(Y > k) = 1` and the trigamma difference). When 24
//!   standard deviations span more than 50 000 values, the equivalent
//!   `ψ'(r) - 1/r - E[ψ'(r + Y) - 1/(r + Y) + (Y - m)²/((r + m)²(r + Y))]`
//!   is summed instead over blocks of values, the same blocks as the count
//!   CRPS.
//!
//! # Intercepts
//!
//! The intercept is the maximum-likelihood fit of the *marginal* (weighted)
//! label distribution: sample mean and (biased) standard deviation for
//! Normal / LogNormal (on `ln y`), the mean and the shape solving
//! `ln a - ψ(a) = ln ȳ - mean(ln y)` for Gamma, the mean for Poisson, and
//! the mean and the size solving the profile score equation (by bisection
//! on `ln r`) for the negative binomial. A scalar `base_score` is only
//! accepted for the one-parameter `dist:poisson` (as its rate).
//!
//! # Shared trees: parallel gradient boosting
//!
//! With `multi_strategy = multi_output_tree` every round grows *one*
//! vector-leaf tree for all parameters instead of one tree per parameter.
//! [`DistSplitDirection`] (`dist_split_direction`) selects its structure:
//!
//! - [`DistSplitDirection::Random`] (default) and
//!   [`DistSplitDirection::Cyclic`] implement parallel gradient boosting
//!   (Chapelle, Vayatis, Falissard & Sedki, 2026, arXiv:2607.13550,
//!   Algorithm 1). The common descent direction of a round is a canonical
//!   basis vector `e_m`: parameter `m` is drawn uniformly at random from
//!   `seed` and the iteration (the paper's choice), or swept as
//!   `iteration mod n_params`; either visits every parameter infinitely
//!   often, the paper's convergence condition. The tree structure is grown
//!   from that parameter's gradient pairs alone
//!   ([`Loss::split_gradient`](crate::objective::Loss::split_gradient), the projected pseudo-residuals
//!   `⟨∇L_i, e_m⟩`), and every leaf then takes the per-parameter Newton step
//!   `-G_k / (H_k + λ)` over its rows. That is the second-order form of the
//!   paper's leaf-wise multidimensional line search `argmin_γ Σ L(g + h γ)`:
//!   with the diagonal curvature of the `dist_gradient` mode the line search
//!   separates across parameters, which the paper's convergence argument
//!   also relies on. With [`DistGradient::Natural`] (unit Hessians) the
//!   structure fit is the paper's least-squares fit of the projected
//!   pseudo-residuals, on the natural gradient.
//! - [`DistSplitDirection::All`]: plain vector-leaf trees, whose split gain
//!   sums over every parameter's gradients.
//!
//! One-parameter families (`dist:poisson`) keep ordinary trees. Shared trees
//! need one structure search per round instead of one per parameter, and
//! all parameters move together each round.
//!
//! # Predictions
//!
//! [`BoostedModel::predict`](crate::model::BoostedModel::predict) returns
//! the natural parameters `[row][parameter]` (the table's last column);
//! [`BoostedModel::predict_distribution`](crate::model::BoostedModel::predict_distribution)
//! returns one [`Dist`] per row with its mean, variance, CDF, quantiles,
//! log density, CRPS, intervals, and inverse-CDF sampling. Metrics `nll`
//! (the default) and `crps` score them.

mod count;
mod dist;
mod family;
mod loss;
pub(crate) mod special;

use serde::{Deserialize, Serialize};

pub use family::{Dist, DistFamily};
pub(crate) use loss::DistLoss;

/// The second-order statistic the `dist:*` distributional objectives give
/// the trees (beyond XGBoost; see [`crate::objective::distributional`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum DistGradient {
    /// Gradient of the negative log-likelihood with the diagonal Fisher
    /// information as Hessian (Fisher scoring; a natural-gradient Newton
    /// step for the orthogonal parameterizations used).
    #[default]
    Fisher,
    /// Gradient with the diagonal of the exact (observed) Hessian, floored
    /// at `1e-16` (XGBoostLSS-style).
    Hessian,
    /// NGBoost's natural gradient `I⁻¹ ∇` with unit Hessian: trees regress
    /// the natural gradient by least squares.
    Natural,
}

/// How the shared tree of a `dist:*` objective chooses its structure under
/// `multi_strategy = multi_output_tree` (beyond XGBoost; see
/// [`crate::objective::distributional`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum DistSplitDirection {
    /// Parallel gradient boosting (Chapelle et al., 2026, Algorithm 1): each
    /// round grows the structure from the gradients of one distribution
    /// parameter drawn uniformly at random (seeded by `seed` and the
    /// iteration), a canonical descent direction `e_m`.
    #[default]
    Random,
    /// Parallel gradient boosting with a deterministic sweep: parameter
    /// `iteration mod n_params` drives round `iteration`.
    Cyclic,
    /// Plain vector-leaf trees: the split gain sums over every parameter.
    All,
}

stored_names! {
    DistGradient { Fisher => "fisher", Hessian => "hessian", Natural => "natural" }
    DistSplitDirection { Random => "random", Cyclic => "cyclic", All => "all" }
}

/// A `dist:*` objective: the distribution family whose parameters the model
/// predicts, the second-order statistic its trees see, and, for shared
/// (vector-leaf) trees, how their structure is chosen.
///
/// ```
/// use hessboost::objective::distributional::{
///     DistFamily, DistGradient, DistSplitDirection, Distributional,
/// };
///
/// let normal = Distributional::new(DistFamily::Normal)
///     .with_gradient(DistGradient::Natural)
///     .with_split_direction(DistSplitDirection::Cyclic);
/// assert_eq!(normal.split_direction(), Some(DistSplitDirection::Cyclic));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Distributional {
    family: DistFamily,
    gradient: DistGradient,
    split_direction: Option<DistSplitDirection>,
}

impl Distributional {
    /// The objective `dist:<family>` with Fisher scoring and, for shared
    /// trees, the default random split direction.
    pub fn new(family: DistFamily) -> Self {
        Distributional {
            family,
            gradient: DistGradient::Fisher,
            split_direction: None,
        }
    }

    /// Give the trees `gradient`'s second-order statistic (default
    /// [`DistGradient::Fisher`]).
    #[must_use]
    pub fn with_gradient(mut self, gradient: DistGradient) -> Self {
        self.gradient = gradient;
        self
    }

    /// Choose the structure of shared (vector-leaf) trees by `direction`
    /// (default [`DistSplitDirection::Random`]). Only shared trees have one:
    /// training refuses it without `multi_strategy = multi_output_tree`.
    #[must_use]
    pub fn with_split_direction(mut self, direction: DistSplitDirection) -> Self {
        self.split_direction = Some(direction);
        self
    }

    /// The distribution family.
    pub fn family(&self) -> DistFamily {
        self.family
    }

    /// The trees' second-order statistic.
    pub fn gradient(&self) -> DistGradient {
        self.gradient
    }

    /// The shared-tree split direction, `None` for the default (random).
    pub fn split_direction(&self) -> Option<DistSplitDirection> {
        self.split_direction
    }
}

/// Bound on log-link margins: `ln` of a positive parameter is clamped to
/// `[-LOG_LINK_BOUND, LOG_LINK_BOUND]` before the link is applied.
pub const LOG_LINK_BOUND: f64 = 30.0;

#[cfg(test)]
mod tests;
