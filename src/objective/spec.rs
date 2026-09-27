//! [`Objective`], the typed learning objective of a configuration, and the
//! loss each built-in one trains with.

use super::distributional::{
    DistFamily, DistGradient, DistLoss, DistSplitDirection, Distributional,
};
use super::{
    AbsoluteError, Aft, AftDistribution, AftLoss, Cox, Expectile, Expectiles, Gamma, Hinge,
    LambdaMart, LambdaRank, LogisticLoss, Loss, Multiclass, Poisson, PseudoHuber, PseudoHuberLoss,
    Quantile, Quantiles, RegLoss, Softmax, SquaredError, SquaredLogError, Tweedie, TweedieLoss,
    Xendcg, multi_target::MultiTarget,
};
use crate::error::{HessboostError, Result};
use std::fmt;
use std::sync::Arc;

/// The learning objective: a built-in XGBoost (or `dist:*`) objective with
/// its parameters, or a custom [`Loss`]. The variants' XGBoost names are
/// given by [`Objective::name`]; parameterized variants wrap a parameter
/// struct that validates on construction. The objectives XGBoost trains
/// through `RegLossObj` (`reg:squarederror`, `reg:gamma`, and the logistic
/// ones) carry its `scale_pos_weight` ([`RegLoss`]); the default is
/// `reg:squarederror` at weight `1`.
///
/// ```
/// use hessboost::objective::{Multiclass, Objective, Quantiles, RegLoss, Tweedie};
///
/// # fn main() -> hessboost::error::Result<()> {
/// let tweedie = Objective::Tweedie(Tweedie::new(1.3)?);
/// assert_eq!(tweedie.name(), "reg:tweedie");
/// let classes = Objective::Softprob(Multiclass::new(3)?);
/// assert_eq!(classes.num_class(), Some(3));
/// let bands = Objective::Quantile(Quantiles::new([0.1, 0.5, 0.9])?);
/// let reweighted = Objective::SquaredError(RegLoss::new(2.0)?);
/// # let _ = (bands, reweighted);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Objective {
    /// `reg:squarederror` (XGBoost's alias `reg:linear`): squared error.
    SquaredError(RegLoss),
    /// `reg:squaredlogerror`: squared log error, labels `> -1`.
    SquaredLogError,
    /// `reg:pseudohubererror`: the pseudo-Huber loss.
    PseudoHuber(PseudoHuber),
    /// `reg:absoluteerror`: absolute error, leaves re-estimated after
    /// growth.
    AbsoluteError,
    /// `reg:quantileerror`: one output per quantile, leaves re-estimated
    /// after growth.
    Quantile(Quantiles),
    /// `reg:expectileerror`: one output per expectile.
    Expectile(Expectiles),
    /// `reg:logistic`: logistic regression on probabilities, evaluated with
    /// `rmse`.
    RegLogistic(RegLoss),
    /// `binary:logistic`: binary classification, predicting probabilities.
    BinaryLogistic(RegLoss),
    /// `binary:logitraw`: binary classification, predicting margins.
    BinaryLogitRaw(RegLoss),
    /// `binary:hinge`: the hinge loss, predicting 0 or 1.
    BinaryHinge,
    /// `multi:softmax`: multiclass, predicting the class index.
    Softmax(Multiclass),
    /// `multi:softprob`: multiclass, predicting class probabilities.
    Softprob(Multiclass),
    /// `count:poisson`: Poisson regression. Its default `max_delta_step`
    /// is 0.7 (see
    /// [`TrainingParams::max_delta_step`](crate::config::TrainingParams::max_delta_step)).
    Poisson,
    /// `reg:gamma`: gamma regression with a log link.
    Gamma(RegLoss),
    /// `reg:tweedie`: Tweedie regression with a log link.
    Tweedie(Tweedie),
    /// `rank:pairwise`: LambdaMART on pairwise loss.
    RankPairwise(LambdaRank),
    /// `rank:ndcg`: LambdaMART on NDCG.
    RankNdcg(LambdaRank),
    /// `rank:map`: LambdaMART on MAP.
    RankMap(LambdaRank),
    /// `rank:xendcg`: LightGBM's XE-NDCG listwise ranking loss
    /// (`rank_xendcg`; beyond XGBoost, so its models are saved in the
    /// native formats only). Its per-round random targets are keyed by
    /// [`seed`](crate::config::TrainingParams::seed), iteration, query, and
    /// document, so the trees differ from LightGBM's.
    RankXendcg,
    /// `survival:cox`: Cox proportional hazards (non-positive labels are
    /// right-censored).
    Cox,
    /// `survival:aft`: accelerated failure time on label bounds.
    Aft(Aft),
    /// `dist:<family>`: distributional boosting (beyond XGBoost; see
    /// [`crate::objective::distributional`]).
    Dist(Distributional),
    /// A custom loss ([`CustomLoss`](crate::objective::CustomLoss) or any
    /// [`Loss`]): the gradients, transform, intercept, default metric, and
    /// `base_score` domain all come from it. A model trained with it
    /// records the loss's name ([`ModelObjective::name`], with no
    /// [`built_in`](crate::model::ModelObjective::built_in) objective) and
    /// predicts untransformed margins once saved; the name must not be a
    /// built-in objective's. Its default `max_delta_step` is 0 (unbounded).
    ///
    /// [`ModelObjective::name`]: crate::model::ModelObjective::name
    Custom(Arc<dyn Loss>),
}

impl Default for Objective {
    /// `reg:squarederror`, XGBoost's default objective, at its default
    /// `scale_pos_weight` of `1`.
    fn default() -> Self {
        Objective::SquaredError(RegLoss::default())
    }
}

impl fmt::Debug for dyn Loss + '_ {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Loss").field("name", &self.name()).finish()
    }
}

impl PartialEq for Objective {
    /// Built-in objectives compare by parameters, custom losses by
    /// identity (the same `Arc`).
    fn eq(&self, other: &Self) -> bool {
        use Objective as O;
        match (self, other) {
            (O::SquaredLogError, O::SquaredLogError)
            | (O::AbsoluteError, O::AbsoluteError)
            | (O::BinaryHinge, O::BinaryHinge)
            | (O::Poisson, O::Poisson)
            | (O::RankXendcg, O::RankXendcg)
            | (O::Cox, O::Cox) => true,
            (O::PseudoHuber(a), O::PseudoHuber(b)) => a == b,
            (O::Quantile(a), O::Quantile(b)) => a == b,
            (O::Expectile(a), O::Expectile(b)) => a == b,
            (O::SquaredError(a), O::SquaredError(b))
            | (O::RegLogistic(a), O::RegLogistic(b))
            | (O::BinaryLogistic(a), O::BinaryLogistic(b))
            | (O::BinaryLogitRaw(a), O::BinaryLogitRaw(b))
            | (O::Gamma(a), O::Gamma(b)) => a == b,
            (O::Softmax(a), O::Softmax(b)) | (O::Softprob(a), O::Softprob(b)) => a == b,
            (O::Tweedie(a), O::Tweedie(b)) => a == b,
            (O::RankPairwise(a), O::RankPairwise(b))
            | (O::RankNdcg(a), O::RankNdcg(b))
            | (O::RankMap(a), O::RankMap(b)) => a == b,
            (O::Aft(a), O::Aft(b)) => a == b,
            (O::Dist(a), O::Dist(b)) => a == b,
            (O::Custom(a), O::Custom(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// How an objective trains on a label matrix (several targets per row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LabelMatrix {
    /// One output per label column, each the single-target loss
    /// ([`MultiTarget`]): XGBoost's elementwise objectives.
    PerColumn,
    /// The loss fits every column itself.
    Native,
    /// One target per row only.
    Refused,
}

/// What building a loss needs besides the objective: the label columns of
/// the data, the resolved `max_delta_step`, and, for vector-leaf trees, the
/// seed of a random `dist:*` split direction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LossContext {
    /// Label columns per row.
    pub(crate) n_targets: usize,
    /// The `max_delta_step` in effect (Poisson's Hessian stabilizer).
    pub(crate) max_delta_step: f64,
    /// `Some(seed)` when the trees are shared vector-leaf trees
    /// (`multi_strategy = multi_output_tree`), which a `dist:*` objective
    /// grows along one parameter per round.
    pub(crate) shared_tree_seed: Option<u64>,
    /// The configuration's [`seed`](crate::config::TrainingParams::seed),
    /// which keys XE-NDCG's random targets.
    pub(crate) seed: u64,
}

/// The parameters of the built-in objectives by their XGBoost names, at
/// XGBoost's defaults: the flat form ([`TrainingParams::from_xgboost`]) and
/// the model formats store them this way.
///
/// [`TrainingParams::from_xgboost`]: crate::config::TrainingParams::from_xgboost
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ObjectiveParts {
    pub(crate) num_class: usize,
    pub(crate) scale_pos_weight: f64,
    pub(crate) tweedie_variance_power: f64,
    pub(crate) huber_slope: f64,
    pub(crate) lambdarank_num_pair_per_sample: usize,
    pub(crate) quantile_alpha: Vec<f64>,
    pub(crate) expectile_alpha: Vec<f64>,
    pub(crate) aft_loss_distribution: AftDistribution,
    pub(crate) aft_loss_distribution_scale: f64,
    pub(crate) dist_gradient: DistGradient,
    /// `None` is the default (random) direction.
    pub(crate) dist_split_direction: Option<DistSplitDirection>,
}

impl Default for ObjectiveParts {
    fn default() -> Self {
        ObjectiveParts {
            num_class: 0,
            scale_pos_weight: RegLoss::default().scale_pos_weight(),
            tweedie_variance_power: Tweedie::default().variance_power(),
            huber_slope: PseudoHuber::default().slope(),
            lambdarank_num_pair_per_sample: LambdaRank::default().num_pair_per_sample(),
            quantile_alpha: Vec::new(),
            expectile_alpha: Vec::new(),
            aft_loss_distribution: AftDistribution::Normal,
            aft_loss_distribution_scale: Aft::default().scale(),
            dist_gradient: DistGradient::Fisher,
            dist_split_direction: None,
        }
    }
}

impl Objective {
    /// A custom loss as the objective.
    pub fn custom(loss: impl Loss + 'static) -> Self {
        Objective::Custom(Arc::new(loss))
    }

    /// The objective's XGBoost name (`reg:squarederror`, `dist:normal`,
    /// ...), a custom loss's [`Loss::name`].
    pub fn name(&self) -> &str {
        match self {
            Objective::SquaredError(_) => "reg:squarederror",
            Objective::SquaredLogError => "reg:squaredlogerror",
            Objective::PseudoHuber(_) => "reg:pseudohubererror",
            Objective::AbsoluteError => "reg:absoluteerror",
            Objective::Quantile(_) => "reg:quantileerror",
            Objective::Expectile(_) => "reg:expectileerror",
            Objective::RegLogistic(_) => "reg:logistic",
            Objective::BinaryLogistic(_) => "binary:logistic",
            Objective::BinaryLogitRaw(_) => "binary:logitraw",
            Objective::BinaryHinge => "binary:hinge",
            Objective::Softmax(_) => "multi:softmax",
            Objective::Softprob(_) => "multi:softprob",
            Objective::Poisson => "count:poisson",
            Objective::Gamma(_) => "reg:gamma",
            Objective::Tweedie(_) => "reg:tweedie",
            Objective::RankPairwise(_) => "rank:pairwise",
            Objective::RankNdcg(_) => "rank:ndcg",
            Objective::RankMap(_) => "rank:map",
            Objective::RankXendcg => "rank:xendcg",
            Objective::Cox => "survival:cox",
            Objective::Aft(_) => "survival:aft",
            Objective::Dist(dist) => dist.family().objective_name(),
            Objective::Custom(loss) => loss.name(),
        }
    }

    /// The class count of a multiclass objective.
    pub fn num_class(&self) -> Option<usize> {
        match self {
            Objective::Softmax(classes) | Objective::Softprob(classes) => Some(classes.num_class()),
            _ => None,
        }
    }

    /// The distribution family of a `dist:*` objective.
    pub fn dist_family(&self) -> Option<DistFamily> {
        match self {
            Objective::Dist(dist) => Some(dist.family()),
            _ => None,
        }
    }

    /// XGBoost's `max_delta_step` for this objective when none is
    /// configured: `0.7` for `count:poisson`, which XGBoost's learner
    /// injects before configuring the objective and tree updater; `0`,
    /// unconstrained, otherwise.
    pub(crate) fn default_max_delta_step(&self) -> f64 {
        if matches!(self, Objective::Poisson) {
            0.7
        } else {
            0.0
        }
    }

    /// Whether the objective re-estimates its leaves after growth
    /// (XGBoost's adaptive leaves: `reg:absoluteerror`,
    /// `reg:quantileerror`).
    pub(crate) fn has_adaptive_leaves(&self) -> bool {
        matches!(self, Objective::AbsoluteError | Objective::Quantile(_))
    }

    /// Whether the objective is `reg:squarederror` without reweighting
    /// (`scale_pos_weight` at `1`): the plain least-squares fit Boulevard
    /// and its inference assume.
    pub(crate) fn is_unweighted_squared_error(&self) -> bool {
        matches!(self, Objective::SquaredError(r) if *r == RegLoss::default())
    }

    /// Whether the objective ranks documents within query groups (the
    /// `rank:*` objectives).
    pub(crate) fn is_ranking(&self) -> bool {
        matches!(
            self,
            Objective::RankPairwise(_)
                | Objective::RankNdcg(_)
                | Objective::RankMap(_)
                | Objective::RankXendcg
        )
    }
    /// Whether the objective classifies `0`/`1` labels (`binary:logistic`,
    /// `binary:logitraw`, `binary:hinge`).
    pub(crate) fn is_binary_classifier(&self) -> bool {
        matches!(
            self,
            Objective::BinaryLogistic(_) | Objective::BinaryLogitRaw(_) | Objective::BinaryHinge
        )
    }

    /// Whether predictions are class indices (`multi:softmax`).
    pub(crate) fn predicts_class_index(&self) -> bool {
        matches!(self, Objective::Softmax(_))
    }

    /// How the objective trains on a label matrix.
    pub(crate) fn label_matrix(&self) -> LabelMatrix {
        match self {
            Objective::SquaredError(_)
            | Objective::PseudoHuber(_)
            | Objective::RegLogistic(_)
            | Objective::BinaryLogistic(_) => LabelMatrix::PerColumn,
            Objective::AbsoluteError | Objective::Custom(_) => LabelMatrix::Native,
            Objective::SquaredLogError
            | Objective::Quantile(_)
            | Objective::Expectile(_)
            | Objective::BinaryLogitRaw(_)
            | Objective::BinaryHinge
            | Objective::Softmax(_)
            | Objective::Softprob(_)
            | Objective::Poisson
            | Objective::Gamma(_)
            | Objective::Tweedie(_)
            | Objective::RankPairwise(_)
            | Objective::RankNdcg(_)
            | Objective::RankMap(_)
            | Objective::RankXendcg
            | Objective::Cox
            | Objective::Aft(_)
            | Objective::Dist(_) => LabelMatrix::Refused,
        }
    }

    /// The flat XGBoost keys whose values this objective reads.
    pub(crate) fn parameter_keys(&self) -> &'static [&'static str] {
        match self {
            Objective::PseudoHuber(_) => &["huber_slope"],
            Objective::Quantile(_) => &["quantile_alpha"],
            Objective::Expectile(_) => &["expectile_alpha"],
            Objective::SquaredError(_)
            | Objective::RegLogistic(_)
            | Objective::BinaryLogistic(_)
            | Objective::BinaryLogitRaw(_)
            | Objective::Gamma(_) => &["scale_pos_weight"],
            Objective::Softmax(_) | Objective::Softprob(_) => &["num_class"],
            Objective::Tweedie(_) => &["tweedie_variance_power"],
            Objective::RankPairwise(_) | Objective::RankNdcg(_) | Objective::RankMap(_) => {
                &["lambdarank_num_pair_per_sample"]
            }
            Objective::Aft(_) => &["aft_loss_distribution", "aft_loss_distribution_scale"],
            Objective::Dist(_) => &["dist_gradient", "dist_split_direction"],
            Objective::SquaredLogError
            | Objective::AbsoluteError
            | Objective::BinaryHinge
            | Objective::Poisson
            | Objective::RankXendcg
            | Objective::Cox
            | Objective::Custom(_) => &[],
        }
    }

    /// Whether `name` is a built-in objective's (XGBoost's `reg:linear`
    /// alias and the `dist:*` names included), whatever its parameters.
    pub(crate) fn is_built_in_name(name: &str) -> bool {
        Objective::from_parts(name, &ObjectiveParts::default()).is_some()
    }

    /// The built-in objective XGBoost names `name` (`reg:linear` is
    /// `reg:squarederror`), with its parameters from `parts`: `None` for a
    /// name that is not a built-in objective, an error when its parameters
    /// are invalid.
    pub(crate) fn from_parts(name: &str, parts: &ObjectiveParts) -> Option<Result<Objective>> {
        let reg_loss = || RegLoss::new(parts.scale_pos_weight);
        let classes = || Multiclass::new(parts.num_class);
        let rank = || LambdaRank::new(parts.lambdarank_num_pair_per_sample);
        let objective = match name {
            "reg:squarederror" | "reg:linear" => reg_loss().map(Objective::SquaredError),
            "reg:squaredlogerror" => Ok(Objective::SquaredLogError),
            "reg:pseudohubererror" => {
                PseudoHuber::new(parts.huber_slope).map(Objective::PseudoHuber)
            }
            "reg:absoluteerror" => Ok(Objective::AbsoluteError),
            "reg:quantileerror" => {
                Quantiles::new(parts.quantile_alpha.iter().copied()).map(Objective::Quantile)
            }
            "reg:expectileerror" => {
                Expectiles::new(parts.expectile_alpha.iter().copied()).map(Objective::Expectile)
            }
            "reg:logistic" => reg_loss().map(Objective::RegLogistic),
            "binary:logistic" => reg_loss().map(Objective::BinaryLogistic),
            "binary:logitraw" => reg_loss().map(Objective::BinaryLogitRaw),
            "binary:hinge" => Ok(Objective::BinaryHinge),
            "multi:softmax" => classes().map(Objective::Softmax),
            "multi:softprob" => classes().map(Objective::Softprob),
            "count:poisson" => Ok(Objective::Poisson),
            "reg:gamma" => reg_loss().map(Objective::Gamma),
            "reg:tweedie" => Tweedie::new(parts.tweedie_variance_power).map(Objective::Tweedie),
            "rank:pairwise" => rank().map(Objective::RankPairwise),
            "rank:ndcg" => rank().map(Objective::RankNdcg),
            "rank:map" => rank().map(Objective::RankMap),
            "rank:xendcg" => Ok(Objective::RankXendcg),
            "survival:cox" => Ok(Objective::Cox),
            "survival:aft" => Aft::new(
                parts.aft_loss_distribution,
                parts.aft_loss_distribution_scale,
            )
            .map(Objective::Aft),
            other => {
                let family = DistFamily::from_objective(other)?;
                let dist = Distributional::new(family).with_gradient(parts.dist_gradient);
                Ok(Objective::Dist(match parts.dist_split_direction {
                    Some(direction) => dist.with_split_direction(direction),
                    None => dist,
                }))
            }
        };
        Some(objective)
    }

    /// The objective's parameters by their XGBoost names, every other one
    /// at its default (all of them for a custom loss).
    pub(crate) fn parts(&self) -> ObjectiveParts {
        let d = ObjectiveParts::default();
        match self {
            Objective::PseudoHuber(huber) => ObjectiveParts {
                huber_slope: huber.slope(),
                ..d
            },
            Objective::Quantile(q) => ObjectiveParts {
                quantile_alpha: q.alpha().to_vec(),
                ..d
            },
            Objective::Expectile(e) => ObjectiveParts {
                expectile_alpha: e.alpha().to_vec(),
                ..d
            },
            Objective::SquaredError(r)
            | Objective::RegLogistic(r)
            | Objective::BinaryLogistic(r)
            | Objective::BinaryLogitRaw(r)
            | Objective::Gamma(r) => ObjectiveParts {
                scale_pos_weight: r.scale_pos_weight(),
                ..d
            },
            Objective::Softmax(c) | Objective::Softprob(c) => ObjectiveParts {
                num_class: c.num_class(),
                ..d
            },
            Objective::Tweedie(t) => ObjectiveParts {
                tweedie_variance_power: t.variance_power(),
                ..d
            },
            Objective::RankPairwise(r) | Objective::RankNdcg(r) | Objective::RankMap(r) => {
                ObjectiveParts {
                    lambdarank_num_pair_per_sample: r.num_pair_per_sample(),
                    ..d
                }
            }
            Objective::Aft(aft) => ObjectiveParts {
                aft_loss_distribution: aft.distribution(),
                aft_loss_distribution_scale: aft.scale(),
                ..d
            },
            Objective::Dist(dist) => ObjectiveParts {
                dist_gradient: dist.gradient(),
                dist_split_direction: dist.split_direction(),
                ..d
            },
            Objective::SquaredLogError
            | Objective::AbsoluteError
            | Objective::BinaryHinge
            | Objective::Poisson
            | Objective::RankXendcg
            | Objective::Cox
            | Objective::Custom(_) => d,
        }
    }

    /// The loss this objective trains with in `context`. A custom loss is
    /// itself; the built-in ones fit a label matrix per column where
    /// XGBoost does ([`LabelMatrix`]) and refuse one otherwise.
    ///
    /// # Errors
    ///
    /// Several label columns for an objective that fits one per row.
    pub(crate) fn build_loss(&self, context: &LossContext) -> Result<Arc<dyn Loss>> {
        let n_targets = context.n_targets;
        let single: Box<dyn Loss> = match self {
            Objective::Custom(loss) => return Ok(Arc::clone(loss)),
            Objective::AbsoluteError => return Ok(Arc::new(AbsoluteError::new(n_targets))),
            Objective::SquaredError(r) => Box::new(SquaredError::new(r.scale_pos_weight() as f32)),
            Objective::SquaredLogError => Box::new(SquaredLogError),
            Objective::PseudoHuber(huber) => Box::new(PseudoHuberLoss::new(*huber)),
            Objective::Quantile(q) => Box::new(Quantile::from_levels(q.clone())),
            Objective::Expectile(e) => Box::new(Expectile::from_levels(e.clone())),
            Objective::RegLogistic(r) => {
                Box::new(LogisticLoss::regression(r.scale_pos_weight() as f32))
            }
            Objective::BinaryLogistic(r) => {
                Box::new(LogisticLoss::new(r.scale_pos_weight() as f32))
            }
            Objective::BinaryLogitRaw(r) => {
                Box::new(LogisticLoss::raw(r.scale_pos_weight() as f32))
            }
            Objective::BinaryHinge => Box::new(Hinge),
            Objective::Softmax(c) => Box::new(Softmax::new(c.num_class(), false)),
            Objective::Softprob(c) => Box::new(Softmax::new(c.num_class(), true)),
            Objective::Poisson => Box::new(Poisson::new(context.max_delta_step as f32)),
            Objective::Gamma(r) => Box::new(Gamma::new(r.scale_pos_weight() as f32)),
            Objective::Tweedie(t) => Box::new(TweedieLoss::new(*t)),
            Objective::RankPairwise(r) => Box::new(LambdaMart::pairwise(r.num_pair_per_sample())),
            Objective::RankNdcg(r) => Box::new(LambdaMart::ndcg(r.num_pair_per_sample())),
            Objective::RankMap(r) => Box::new(LambdaMart::map(r.num_pair_per_sample())),
            Objective::RankXendcg => Box::new(Xendcg::new(context.seed)),
            Objective::Cox => Box::new(Cox),
            Objective::Aft(aft) => Box::new(AftLoss::new(aft.distribution(), aft.scale() as f32)),
            Objective::Dist(dist) => {
                let loss = DistLoss::new(dist.family(), dist.gradient());
                Box::new(match context.shared_tree_seed {
                    Some(seed) => {
                        loss.with_split_direction(dist.split_direction().unwrap_or_default(), seed)
                    }
                    None => loss,
                })
            }
        };
        if n_targets <= 1 {
            return Ok(Arc::from(single));
        }
        match self.label_matrix() {
            LabelMatrix::PerColumn => Ok(Arc::new(MultiTarget::new(single, n_targets))),
            LabelMatrix::Native | LabelMatrix::Refused => Err(HessboostError::invalid_param(
                "labels",
                format!(
                    "objective `{}` supports one target per row, got {n_targets}",
                    self.name()
                ),
            )),
        }
    }
}
