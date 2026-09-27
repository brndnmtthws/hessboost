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
/// The two never overlap: a built-in objective is never a custom loss, and
/// a recorded name is never a built-in objective's (a model file naming
/// one loads as that objective).
///
/// ```
/// use hessboost::model::ModelObjective;
/// use hessboost::objective::{CustomLoss, GradPair, Objective};
///
/// # fn main() -> hessboost::error::Result<()> {
/// let known = ModelObjective::new(Objective::Poisson)?;
/// assert_eq!(known.name(), "count:poisson");
/// assert_eq!(known.built_in(), Some(&Objective::Poisson));
/// let loss = CustomLoss::new("my:loss", 1, |p, y, _, out| {
///     for ((g, p), y) in out.iter_mut().zip(p).zip(y) {
///         *g = GradPair::new(p - y, 1.0);
///     }
/// });
/// let custom = ModelObjective::new(Objective::custom(loss))?;
/// assert_eq!((custom.name(), custom.built_in()), ("my:loss", None));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ModelObjective(Recorded);

/// [`ModelObjective`]'s representation, private so that its invariant
/// holds.
#[derive(Debug, Clone, PartialEq)]
enum Recorded {
    /// A built-in objective (never [`Objective::Custom`]), which gives the
    /// prediction transform and XGBoost export.
    BuiltIn(Objective),
    /// A custom loss's [`name`](crate::objective::Loss::name), or an
    /// XGBoost objective hessboost does not implement (never a built-in
    /// objective's name): the model predicts margins, and XGBoost export
    /// refuses it.
    Other(String),
}

impl ModelObjective {
    /// What a model trained with `objective` records: a built-in objective
    /// as itself, a custom loss by its name.
    ///
    /// # Errors
    ///
    /// A custom loss named like a built-in objective (`invalid parameter
    /// "objective"`): a saved model would reload as that objective, so
    /// training refuses it too.
    pub fn new(objective: Objective) -> Result<Self> {
        Ok(ModelObjective(match objective {
            Objective::Custom(loss) if Objective::is_built_in_name(loss.name()) => {
                return Err(HessboostError::invalid_param(
                    "objective",
                    format!(
                        "the custom loss is named `{}`, a built-in objective's name, as \
                         which a saved model would reload; rename the loss",
                        loss.name()
                    ),
                ));
            }
            Objective::Custom(loss) => Recorded::Other(loss.name().to_owned()),
            built_in => Recorded::BuiltIn(built_in),
        }))
    }

    /// The objective's name, as the model formats store it.
    pub fn name(&self) -> &str {
        match &self.0 {
            Recorded::BuiltIn(objective) => objective.name(),
            Recorded::Other(name) => name,
        }
    }

    /// The built-in objective, if it is one.
    pub fn built_in(&self) -> Option<&Objective> {
        match &self.0 {
            Recorded::BuiltIn(objective) => Some(objective),
            Recorded::Other(_) => None,
        }
    }

    /// [`ModelObjective::new`] for a configuration that validated (whose
    /// custom loss, if any, has a name no built-in objective has).
    pub(crate) fn trained_with(objective: &Objective) -> Self {
        ModelObjective(match objective {
            Objective::Custom(loss) => Recorded::Other(loss.name().to_owned()),
            built_in => Recorded::BuiltIn(built_in.clone()),
        })
    }

    /// The objective the formats store as `name` with `stored` parameters
    /// and class count `num_class`: the built-in objective of that name
    /// (only the parameters it reads matter), else the name.
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
            None => Ok(ModelObjective(Recorded::Other(name.to_owned()))),
            Some(Ok(objective)) => Ok(ModelObjective(Recorded::BuiltIn(objective))),
            Some(Err(e)) => Err(HessboostError::model_format(format!(
                "invalid objective parameters: {e}"
            ))),
        }
    }
}

/// Declares [`StoredObjectiveParams`] (serialized with its fields in
/// declaration order) and its all-optional mirror
/// [`PartialStoredObjectiveParams`] with `fill`, from one member list. A new
/// stored objective parameter is one line here plus its value in
/// [`StoredObjectiveParams::of`].
macro_rules! stored_objective_params {
    ($(#[doc = $sdoc:literal])* ; $(#[doc = $pdoc:literal])* ; $($name:ident: $ty:ty),* $(,)?) => {
        $(#[doc = $sdoc])*
        #[derive(Debug, Clone, PartialEq, Serialize)]
        pub(crate) struct StoredObjectiveParams {
            $(pub(crate) $name: $ty,)*
        }

        $(#[doc = $pdoc])*
        #[derive(Deserialize, Default)]
        pub(crate) struct PartialStoredObjectiveParams {
            $(#[serde(default)] $name: Stored<$ty>,)*
        }

        impl PartialStoredObjectiveParams {
            /// The stored parameters, each missing one taken from `objective`'s
            /// defaults.
            pub(crate) fn fill(self, objective: &str) -> StoredObjectiveParams {
                let d = StoredObjectiveParams::defaults_for(objective);
                StoredObjectiveParams {
                    $($name: self.$name.unwrap_or(d.$name),)*
                }
            }
        }
    };
}

stored_objective_params! {
    /// The objective parameters the model formats store, in their stored
    /// layout: every built-in objective's parameters at the values the model's
    /// objective gives them (the defaults for the ones it does not read), plus
    /// the `max_delta_step` training used and the `dist:*` family. Native JSON
    /// names the members as the fields; the native binary and compact formats
    /// store `objective.<field>` sections.
    ;
    /// [`StoredObjectiveParams`] as native JSON stores them, with every member
    /// optional: [`PartialStoredObjectiveParams::fill`] takes each missing one
    /// from the recorded objective's defaults
    /// ([`StoredObjectiveParams::defaults_for`]), which a per-field serde default
    /// could not (they depend on the objective). A stored value is read as
    /// strictly as the full record (`null` only for `distribution`).
    ;
    scale_pos_weight: f64,
    max_delta_step: f64,
    tweedie_variance_power: f64,
    huber_slope: f64,
    lambdarank_num_pair_per_sample: usize,
    quantile_alpha: Vec<f64>,
    expectile_alpha: Vec<f64>,
    aft_loss_distribution: AftDistribution,
    aft_loss_distribution_scale: f64,
    dist_gradient: DistGradient,
    dist_split_direction: DistSplitDirection,
    distribution: Option<DistFamily>,
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
        let mut defaults = StoredObjectiveParams::of(
            &ModelObjective(Recorded::Other(String::new())),
            max_delta_step,
        );
        defaults.distribution = DistFamily::from_objective(objective);
        defaults
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::{CustomLoss, GradPair, RegLoss};

    /// A custom loss named like a built-in objective cannot be recorded
    /// (it would reload as that objective), and a stored built-in name
    /// always loads as the objective, so the two states never overlap.
    #[test]
    fn recorded_names_are_never_built_in() {
        let loss = |name: &'static str| {
            Objective::custom(CustomLoss::new(name, 1, |_, _, _, out: &mut [GradPair]| {
                out.fill(GradPair::new(0.0, 1.0));
            }))
        };
        for name in ["reg:squarederror", "reg:linear", "dist:normal"] {
            assert!(ModelObjective::new(loss(name)).is_err(), "{name}");
        }
        let custom = ModelObjective::new(loss("custom:mine")).unwrap();
        assert_eq!((custom.name(), custom.built_in()), ("custom:mine", None));
        let defaults = StoredObjectiveParams::defaults_for("reg:linear");
        let stored = ModelObjective::from_stored("reg:linear", &defaults, 0).unwrap();
        assert_eq!(
            stored.built_in(),
            Some(&Objective::SquaredError(RegLoss::default()))
        );
        let unknown = ModelObjective::from_stored("rank:foo", &defaults, 0).unwrap();
        assert_eq!((unknown.name(), unknown.built_in()), ("rank:foo", None));
    }
}
