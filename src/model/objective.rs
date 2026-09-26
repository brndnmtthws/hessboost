//! The objective a model records ([`ModelObjective`]) and its stored form:
//! the objective's name plus the flat parameter record the native, native
//! JSON, and compact formats keep (unchanged since 0.2.0).

use crate::error::{HessboostError, Result};
use crate::objective::distributional::{DistFamily, DistGradient, DistSplitDirection};
use crate::objective::{AftDistribution, Objective, ObjectiveParts};
use serde::{Deserialize, Serialize};

/// The objective a model was trained with (or imported with): a built-in
/// objective with its parameters, or the name of one this crate does not
/// implement, whose predictions are untransformed margins.
///
/// ```
/// use hessboost::model::ModelObjective;
/// use hessboost::objective::Objective;
///
/// let known = ModelObjective::BuiltIn(Objective::Poisson);
/// assert_eq!(known.name(), "count:poisson");
/// assert_eq!(known.built_in(), Some(&Objective::Poisson));
/// let custom = ModelObjective::Other("my:loss".to_owned());
/// assert_eq!(custom.built_in(), None);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ModelObjective {
    /// A built-in objective (never [`Objective::Custom`]), which gives the
    /// prediction transform and XGBoost export.
    BuiltIn(Objective),
    /// A custom loss's [`name`](crate::objective::Loss::name), or an
    /// XGBoost objective hessboost does not implement: the model predicts
    /// margins, and XGBoost export refuses it.
    Other(String),
}

impl ModelObjective {
    /// The objective's name, as the model formats store it.
    pub fn name(&self) -> &str {
        match self {
            ModelObjective::BuiltIn(objective) => objective.name(),
            ModelObjective::Other(name) => name,
        }
    }

    /// The built-in objective, if it is one.
    pub fn built_in(&self) -> Option<&Objective> {
        match self {
            ModelObjective::BuiltIn(objective) => Some(objective),
            ModelObjective::Other(_) => None,
        }
    }

    /// What a model trained with `objective` records: a custom loss by its
    /// name, a built-in objective as itself.
    pub(crate) fn trained_with(objective: &Objective) -> Self {
        match objective {
            Objective::Custom(loss) => ModelObjective::Other(loss.name().to_owned()),
            built_in => ModelObjective::BuiltIn(built_in.clone()),
        }
    }

    /// The objective the formats store as `name` with `stored` parameters
    /// and class count `num_class`: the built-in objective of that name
    /// (only the parameters it reads matter), else [`ModelObjective::Other`].
    ///
    /// # Errors
    ///
    /// A [`HessboostError::ModelFormat`] for a built-in objective whose
    /// stored parameters are invalid, or a stored distribution family that
    /// is not the objective's.
    pub(crate) fn from_stored(
        name: &str,
        stored: &StoredObjectiveParams,
        num_class: usize,
    ) -> Result<Self> {
        // Derived from the objective name, not configured: the formats keep
        // it so a reader can check it.
        if stored.distribution != DistFamily::from_objective(name) {
            return Err(HessboostError::model_format(format!(
                "objective parameters name distribution {:?} for objective `{name}`",
                stored.distribution
            )));
        }
        let parts = ObjectiveParts {
            num_class,
            scale_pos_weight: stored.scale_pos_weight,
            tweedie_variance_power: stored.tweedie_variance_power,
            huber_slope: stored.huber_slope,
            lambdarank_num_pair_per_sample: stored.lambdarank_num_pair_per_sample,
            quantile_alpha: stored.quantile_alpha.clone(),
            expectile_alpha: stored.expectile_alpha.clone(),
            aft_loss_distribution: stored.aft_loss_distribution,
            aft_loss_distribution_scale: stored.aft_loss_distribution_scale,
            dist_gradient: stored.dist_gradient,
            dist_split_direction: match stored.dist_split_direction {
                DistSplitDirection::Random => None,
                direction => Some(direction),
            },
        };
        match Objective::from_parts(name, &parts) {
            None => Ok(ModelObjective::Other(name.to_owned())),
            Some(Ok(objective)) => Ok(ModelObjective::BuiltIn(objective)),
            Some(Err(e)) => Err(HessboostError::model_format(format!(
                "invalid objective parameters: {e}"
            ))),
        }
    }
}

/// The objective parameters the model formats store, in their stored
/// layout: every built-in objective's parameters at the values the model's
/// objective gives them (the defaults for the ones it does not read), plus
/// the `max_delta_step` training used and the `dist:*` family. Native JSON
/// names the members as the fields; the native binary and compact formats
/// store `objective.<field>` sections.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct StoredObjectiveParams {
    pub(crate) scale_pos_weight: f64,
    pub(crate) max_delta_step: f64,
    pub(crate) tweedie_variance_power: f64,
    pub(crate) huber_slope: f64,
    pub(crate) lambdarank_num_pair_per_sample: usize,
    pub(crate) quantile_alpha: Vec<f64>,
    pub(crate) expectile_alpha: Vec<f64>,
    pub(crate) aft_loss_distribution: AftDistribution,
    pub(crate) aft_loss_distribution_scale: f64,
    pub(crate) dist_gradient: DistGradient,
    pub(crate) dist_split_direction: DistSplitDirection,
    pub(crate) distribution: Option<DistFamily>,
}

impl StoredObjectiveParams {
    /// The record of `objective` trained with `max_delta_step`.
    pub(crate) fn of(objective: &ModelObjective, max_delta_step: f64) -> Self {
        let parts = objective
            .built_in()
            .map_or_else(ObjectiveParts::default, Objective::parts);
        StoredObjectiveParams {
            scale_pos_weight: parts.scale_pos_weight,
            max_delta_step,
            tweedie_variance_power: parts.tweedie_variance_power,
            huber_slope: parts.huber_slope,
            lambdarank_num_pair_per_sample: parts.lambdarank_num_pair_per_sample,
            quantile_alpha: parts.quantile_alpha,
            expectile_alpha: parts.expectile_alpha,
            aft_loss_distribution: parts.aft_loss_distribution,
            aft_loss_distribution_scale: parts.aft_loss_distribution_scale,
            dist_gradient: parts.dist_gradient,
            dist_split_direction: parts.dist_split_direction.unwrap_or_default(),
            distribution: DistFamily::from_objective(objective.name()),
        }
    }

    /// The record an objective named `objective` has when no parameter is
    /// stored: every default, with XGBoost's `max_delta_step = 0.7` for
    /// `count:poisson`.
    pub(crate) fn defaults_for(objective: &str) -> Self {
        let max_delta_step = if objective == "count:poisson" {
            0.7
        } else {
            0.0
        };
        let mut defaults =
            StoredObjectiveParams::of(&ModelObjective::Other(String::new()), max_delta_step);
        defaults.distribution = DistFamily::from_objective(objective);
        defaults
    }
}

/// [`StoredObjectiveParams`] as native JSON stores them, with every member
/// optional: [`PartialStoredObjectiveParams::fill`] takes each missing one
/// from the recorded objective's defaults
/// ([`StoredObjectiveParams::defaults_for`]), which a per-field serde default
/// could not (they depend on the objective). A stored value is read as
/// strictly as the full record (`null` only for `distribution`).
#[derive(Deserialize, Default)]
pub(crate) struct PartialStoredObjectiveParams {
    #[serde(default)]
    scale_pos_weight: Stored<f64>,
    #[serde(default)]
    max_delta_step: Stored<f64>,
    #[serde(default)]
    tweedie_variance_power: Stored<f64>,
    #[serde(default)]
    huber_slope: Stored<f64>,
    #[serde(default)]
    lambdarank_num_pair_per_sample: Stored<usize>,
    #[serde(default)]
    quantile_alpha: Stored<Vec<f64>>,
    #[serde(default)]
    expectile_alpha: Stored<Vec<f64>>,
    #[serde(default)]
    aft_loss_distribution: Stored<AftDistribution>,
    #[serde(default)]
    aft_loss_distribution_scale: Stored<f64>,
    #[serde(default)]
    dist_gradient: Stored<DistGradient>,
    #[serde(default)]
    dist_split_direction: Stored<DistSplitDirection>,
    #[serde(default)]
    distribution: Stored<Option<DistFamily>>,
}

impl PartialStoredObjectiveParams {
    /// The stored parameters, each missing one taken from `objective`'s
    /// defaults.
    pub(crate) fn fill(self, objective: &str) -> StoredObjectiveParams {
        let d = StoredObjectiveParams::defaults_for(objective);
        let PartialStoredObjectiveParams {
            scale_pos_weight,
            max_delta_step,
            tweedie_variance_power,
            huber_slope,
            lambdarank_num_pair_per_sample,
            quantile_alpha,
            expectile_alpha,
            aft_loss_distribution,
            aft_loss_distribution_scale,
            dist_gradient,
            dist_split_direction,
            distribution,
        } = self;
        StoredObjectiveParams {
            scale_pos_weight: scale_pos_weight.unwrap_or(d.scale_pos_weight),
            max_delta_step: max_delta_step.unwrap_or(d.max_delta_step),
            tweedie_variance_power: tweedie_variance_power.unwrap_or(d.tweedie_variance_power),
            huber_slope: huber_slope.unwrap_or(d.huber_slope),
            lambdarank_num_pair_per_sample: lambdarank_num_pair_per_sample
                .unwrap_or(d.lambdarank_num_pair_per_sample),
            quantile_alpha: quantile_alpha.unwrap_or(d.quantile_alpha),
            expectile_alpha: expectile_alpha.unwrap_or(d.expectile_alpha),
            aft_loss_distribution: aft_loss_distribution.unwrap_or(d.aft_loss_distribution),
            aft_loss_distribution_scale: aft_loss_distribution_scale
                .unwrap_or(d.aft_loss_distribution_scale),
            dist_gradient: dist_gradient.unwrap_or(d.dist_gradient),
            dist_split_direction: dist_split_direction.unwrap_or(d.dist_split_direction),
            distribution: distribution.unwrap_or(d.distribution),
        }
    }
}

/// A member that is absent from the document, or present with a value
/// (which may itself be `None`: `Option<Option<_>>` would read `null` as
/// absent).
#[derive(Default)]
enum Stored<T> {
    #[default]
    Absent,
    Present(T),
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Stored<T> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        T::deserialize(deserializer).map(Stored::Present)
    }
}

impl<T> Stored<T> {
    fn unwrap_or(self, default: T) -> T {
        match self {
            Stored::Absent => default,
            Stored::Present(value) => value,
        }
    }
}
