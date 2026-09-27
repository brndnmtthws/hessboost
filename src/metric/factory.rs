//! [`EvalMetric`], the typed metric selection: its names, its built
//! metrics, and the XGBoost metric names `from_xgboost` reads.

use super::curve::{Auc, AucPr};
use super::distributional::{DistCrps, DistNll};
use super::elementwise::{Mape, PseudoHuberError, Rmsle};
use super::quantile::{ExpectileError, QuantileError};
use super::ranking::{MeanAveragePrecision, Ndcg, Precision};
use super::survival::{AftNLogLik, CoxNLogLik, IntervalRegressionAccuracy};
use super::{
    ErrorRate, GammaNLogLik, LogLoss, MError, MLogLoss, Mae, Metric, PoissonNLogLik, Rmse,
    TweedieNLogLik, tweedie_name,
};
use crate::error::{HessboostError, Result};
use crate::objective::distributional::DistFamily;
use crate::objective::{Aft, AftDistribution, Expectiles, PseudoHuber, Quantiles, Tweedie};
use std::borrow::Cow;
use std::num::NonZeroUsize;

/// A rank cutoff of the ranking metrics (`ndcg`, `map`, `pre`): the top
/// `k` documents of every query group, or the whole list. XGBoost's `@k`
/// metric suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cutoff {
    top_k: Option<NonZeroUsize>,
}

impl Cutoff {
    /// No cutoff: `ndcg` and `map` score the whole list; `pre` without a
    /// cutoff scores the top 32, as XGBoost does, and is still named `pre`.
    pub fn all() -> Self {
        Cutoff { top_k: None }
    }

    /// The top `k` documents (`@k`).
    ///
    /// # Errors
    ///
    /// `k` is 0 (XGBoost's lower bound on the top-k it reads from the
    /// suffix is 1).
    pub fn top(k: usize) -> Result<Self> {
        NonZeroUsize::new(k).map(Cutoff::from).ok_or_else(|| {
            HessboostError::invalid_param("eval_metric", "the `@k` cutoff must be at least 1")
        })
    }

    /// The cutoff `k`, `None` for the whole list.
    pub fn top_k(&self) -> Option<NonZeroUsize> {
        self.top_k
    }

    /// The cutoff as the ranking metrics compute with it.
    fn k(self) -> Option<usize> {
        self.top_k.map(NonZeroUsize::get)
    }
}

impl From<NonZeroUsize> for Cutoff {
    fn from(k: NonZeroUsize) -> Self {
        Cutoff { top_k: Some(k) }
    }
}

/// A built-in evaluation metric with its parameters: the typed form of
/// XGBoost's `eval_metric` names, which training builds with
/// [`EvalMetric::build`]. Every metric carries the parameters it evaluates
/// with (unlike XGBoost, where `mphe`, `quantile`, `expectile`, and
/// `aft-nloglik` read the objective's parameters);
/// [`TrainingParams::from_xgboost`](crate::config::TrainingParams::from_xgboost)
/// reads XGBoost's names and fills those parameters the way XGBoost does.
///
/// [`name`](EvalMetric::name) is XGBoost's `evals_result` key, suffix
/// included (`ndcg@5`, `tweedie-nloglik@1.5`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EvalMetric {
    /// Root-mean-square error (`rmse`).
    Rmse,
    /// Root-mean-square log error (`rmsle`).
    Rmsle,
    /// Mean absolute error (`mae`).
    Mae,
    /// Mean absolute percentage error (`mape`).
    Mape,
    /// Mean pseudo-Huber error (`mphe`) with this slope.
    Mphe(PseudoHuber),
    /// Binary log loss (`logloss`); raw margins for `binary:logitraw`.
    LogLoss,
    /// Binary error rate at threshold 0.5 (`error`).
    Error,
    /// ROC AUC (`auc`); the mean per-target AUC for a label matrix.
    Auc,
    /// Area under the precision-recall curve (`aucpr`).
    AucPr,
    /// Multiclass log loss (`mlogloss`), one probability per class.
    MLogLoss,
    /// Multiclass error rate (`merror`).
    MError,
    /// Poisson negative log-likelihood (`poisson-nloglik`).
    PoissonNLogLik,
    /// Gamma negative log-likelihood (`gamma-nloglik`).
    GammaNLogLik,
    /// Tweedie negative log-likelihood (`tweedie-nloglik@rho`) at this
    /// variance power.
    TweedieNLogLik(Tweedie),
    /// Normalized discounted cumulative gain (`ndcg`, `ndcg@k`).
    Ndcg(Cutoff),
    /// Mean average precision (`map`, `map@k`).
    Map(Cutoff),
    /// Precision (`pre`, `pre@k`).
    Precision(Cutoff),
    /// Pinball loss at these quantiles (`quantile`), one prediction per
    /// level.
    Quantile(Quantiles),
    /// Expectile loss at these levels (`expectile`).
    Expectile(Expectiles),
    /// Cox proportional-hazards negative partial log-likelihood
    /// (`cox-nloglik`).
    CoxNLogLik,
    /// Accelerated-failure-time negative log-likelihood (`aft-nloglik`)
    /// under this noise model.
    AftNLogLik(Aft),
    /// Fraction of predictions inside their label interval
    /// (`interval-regression-accuracy`).
    IntervalRegressionAccuracy,
    /// Negative log-likelihood of predicted distributions of this family
    /// (`nll`, beyond XGBoost; see [`crate::objective::distributional`]).
    Nll(DistFamily),
    /// Continuous ranked probability score of predicted distributions of
    /// this family (`crps`, beyond XGBoost).
    Crps(DistFamily),
}

/// XGBoost's name of a ranking metric: `base@k` with a cutoff, else `base`.
pub(super) fn cutoff_name(base: &str, k: Option<usize>) -> String {
    k.map_or_else(|| base.to_string(), |k| format!("{base}@{k}"))
}

impl EvalMetric {
    /// The metric's spelling in XGBoost's flat `eval_metric`, which
    /// [`TrainingParams::from_xgboost`](crate::config::TrainingParams::from_xgboost)
    /// reads back to the same metric: [`name`](Self::name), except that a
    /// Tweedie power is written in full rather than rounded to the six
    /// digits of its `evals_result` key.
    pub(crate) fn flat_name(&self) -> Cow<'static, str> {
        match self {
            EvalMetric::TweedieNLogLik(tweedie) => {
                Cow::Owned(format!("tweedie-nloglik@{}", tweedie.variance_power()))
            }
            _ => self.name(),
        }
    }

    /// XGBoost's `evals_result` key: the metric's name with its suffix
    /// (`ndcg@5`, `tweedie-nloglik@1.5`), as
    /// [`Metric::name`] of the built metric reports it.
    pub fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            EvalMetric::Rmse => "rmse",
            EvalMetric::Rmsle => "rmsle",
            EvalMetric::Mae => "mae",
            EvalMetric::Mape => "mape",
            EvalMetric::Mphe(_) => "mphe",
            EvalMetric::LogLoss => "logloss",
            EvalMetric::Error => "error",
            EvalMetric::Auc => "auc",
            EvalMetric::AucPr => "aucpr",
            EvalMetric::MLogLoss => "mlogloss",
            EvalMetric::MError => "merror",
            EvalMetric::PoissonNLogLik => "poisson-nloglik",
            EvalMetric::GammaNLogLik => "gamma-nloglik",
            EvalMetric::TweedieNLogLik(tweedie) => {
                return Cow::Owned(tweedie_name(tweedie.variance_power()));
            }
            EvalMetric::Ndcg(cutoff) => return Cow::Owned(cutoff_name("ndcg", cutoff.k())),
            EvalMetric::Map(cutoff) => return Cow::Owned(cutoff_name("map", cutoff.k())),
            EvalMetric::Precision(cutoff) => return Cow::Owned(cutoff_name("pre", cutoff.k())),
            EvalMetric::Quantile(_) => "quantile",
            EvalMetric::Expectile(_) => "expectile",
            EvalMetric::CoxNLogLik => "cox-nloglik",
            EvalMetric::AftNLogLik(_) => "aft-nloglik",
            EvalMetric::IntervalRegressionAccuracy => "interval-regression-accuracy",
            EvalMetric::Nll(_) => "nll",
            EvalMetric::Crps(_) => "crps",
        })
    }

    /// The metric, ready to evaluate the predictions of a model with
    /// `n_outputs` outputs per row (`mlogloss` and `merror` read one
    /// probability per class).
    ///
    /// # Errors
    ///
    /// `mlogloss` or `merror` for fewer than two outputs.
    ///
    /// ```
    /// use hessboost::metric::{EvalMetric, Metric};
    ///
    /// # fn main() -> hessboost::error::Result<()> {
    /// let rmse = EvalMetric::Rmse.build(1)?;
    /// assert_eq!(rmse.eval(&[1.0, 3.0], &[1.0, 1.0], None), 2f64.sqrt());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(&self, n_outputs: usize) -> Result<Box<dyn Metric>> {
        let classes = || {
            if n_outputs >= 2 {
                Ok(n_outputs)
            } else {
                Err(HessboostError::invalid_param(
                    "eval_metric",
                    format!(
                        "`{}` scores one probability per class and needs a multiclass model, \
                         got {n_outputs} output(s)",
                        self.name()
                    ),
                ))
            }
        };
        Ok(match self {
            EvalMetric::Rmse => Box::new(Rmse),
            EvalMetric::Rmsle => Box::new(Rmsle),
            EvalMetric::Mae => Box::new(Mae),
            EvalMetric::Mape => Box::new(Mape),
            EvalMetric::Mphe(huber) => Box::new(PseudoHuberError::new(huber.slope() as f32)),
            EvalMetric::LogLoss => Box::new(LogLoss),
            EvalMetric::Error => Box::new(ErrorRate),
            EvalMetric::Auc => Box::new(Auc),
            EvalMetric::AucPr => Box::new(AucPr),
            EvalMetric::MLogLoss => Box::new(MLogLoss {
                num_class: classes()?,
            }),
            EvalMetric::MError => Box::new(MError {
                num_class: classes()?,
            }),
            EvalMetric::PoissonNLogLik => Box::new(PoissonNLogLik),
            EvalMetric::GammaNLogLik => Box::new(GammaNLogLik),
            EvalMetric::TweedieNLogLik(tweedie) => {
                Box::new(TweedieNLogLik::new(tweedie.variance_power()))
            }
            EvalMetric::Ndcg(cutoff) => Box::new(Ndcg::new(cutoff.k())),
            EvalMetric::Map(cutoff) => Box::new(MeanAveragePrecision::new(cutoff.k())),
            EvalMetric::Precision(cutoff) => Box::new(Precision::new(cutoff.k())),
            EvalMetric::Quantile(quantiles) => Box::new(QuantileError::new(quantiles.alpha_f32())),
            EvalMetric::Expectile(expectiles) => {
                Box::new(ExpectileError::new(expectiles.alpha_f32()))
            }
            EvalMetric::CoxNLogLik => Box::new(CoxNLogLik),
            EvalMetric::AftNLogLik(aft) => {
                Box::new(AftNLogLik::new(aft.distribution(), aft.scale() as f32))
            }
            EvalMetric::IntervalRegressionAccuracy => Box::new(IntervalRegressionAccuracy),
            EvalMetric::Nll(family) => Box::new(DistNll::new(*family)),
            EvalMetric::Crps(family) => Box::new(DistCrps::new(*family)),
        })
    }

    /// The flat objective-parameter keys the metric XGBoost names `name`
    /// reads (XGBoost's metrics read them from the objective's parameters,
    /// whatever the objective).
    pub(crate) fn borrowed_keys(name: &str) -> &'static [&'static str] {
        match name {
            "mphe" => &["huber_slope"],
            "quantile" => &["quantile_alpha"],
            "expectile" => &["expectile_alpha"],
            "aft-nloglik" => &["aft_loss_distribution", "aft_loss_distribution_scale"],
            _ => &[],
        }
    }

    /// The metric XGBoost names `name`, with the parameters XGBoost would
    /// give it: the `@k` cutoff of `ndcg`/`map`/`pre` and the `@rho` power of
    /// `tweedie-nloglik` (1.5 without one) from the suffix, the rest from
    /// `source` (the flat parameters `mphe`, `quantile`, `expectile`, and
    /// `aft-nloglik` read, and the `dist:*` family `nll` and `crps` score).
    ///
    /// Only those four names take an `@` suffix; any other suffix, including
    /// XGBoost's `error@t` threshold and the `-` variants (`ndcg@3-`), is
    /// refused, as are unknown names.
    pub(crate) fn from_xgboost(name: &str, source: &XgboostMetricSource<'_>) -> Result<Self> {
        let (base, suffix) = match name.split_once('@') {
            Some((b, s)) => (b, Some(s)),
            None => (name, None),
        };
        if suffix.is_some() && !matches!(base, "tweedie-nloglik" | "ndcg" | "map" | "pre") {
            return Err(invalid_metric(
                name,
                &format!("`{base}` takes no `@` suffix"),
            ));
        }
        let cutoff = || rank_cutoff(name, suffix);
        Ok(match base {
            "rmse" => EvalMetric::Rmse,
            "rmsle" => EvalMetric::Rmsle,
            "mae" => EvalMetric::Mae,
            "mape" => EvalMetric::Mape,
            "mphe" => EvalMetric::Mphe(source.huber()?),
            "logloss" => EvalMetric::LogLoss,
            "error" => EvalMetric::Error,
            "auc" => EvalMetric::Auc,
            "aucpr" => EvalMetric::AucPr,
            "mlogloss" => EvalMetric::MLogLoss,
            "merror" => EvalMetric::MError,
            "poisson-nloglik" => EvalMetric::PoissonNLogLik,
            "gamma-nloglik" => EvalMetric::GammaNLogLik,
            "tweedie-nloglik" => EvalMetric::TweedieNLogLik(tweedie_power(name, suffix)?),
            "ndcg" => EvalMetric::Ndcg(cutoff()?),
            "map" => EvalMetric::Map(cutoff()?),
            "pre" => EvalMetric::Precision(cutoff()?),
            "quantile" => EvalMetric::Quantile(source.quantiles()?),
            "expectile" => EvalMetric::Expectile(source.expectiles()?),
            "cox-nloglik" => EvalMetric::CoxNLogLik,
            "aft-nloglik" => EvalMetric::AftNLogLik(source.aft()?),
            "interval-regression-accuracy" => EvalMetric::IntervalRegressionAccuracy,
            "nll" | "crps" => {
                let family = source.distribution.ok_or_else(|| {
                    HessboostError::invalid_param(
                        "eval_metric",
                        format!(
                            "`{name}` scores predicted distributions and needs a `dist:*` objective"
                        ),
                    )
                })?;
                if base == "nll" {
                    EvalMetric::Nll(family)
                } else {
                    EvalMetric::Crps(family)
                }
            }
            other => return Err(HessboostError::unknown("metric", other)),
        })
    }
}

/// The flat XGBoost parameters the metrics named in `eval_metric` read
/// ([`EvalMetric::from_xgboost`]), as XGBoost gives them whatever the
/// objective.
pub(crate) struct XgboostMetricSource<'a> {
    /// `huber_slope` (`mphe`).
    pub(crate) huber_slope: f64,
    /// `quantile_alpha` (`quantile`).
    pub(crate) quantile_alpha: &'a [f64],
    /// `expectile_alpha` (`expectile`).
    pub(crate) expectile_alpha: &'a [f64],
    /// `aft_loss_distribution` (`aft-nloglik`).
    pub(crate) aft_loss_distribution: AftDistribution,
    /// `aft_loss_distribution_scale` (`aft-nloglik`).
    pub(crate) aft_loss_distribution_scale: f64,
    /// The family of a `dist:*` objective (`nll`, `crps`).
    pub(crate) distribution: Option<DistFamily>,
}

impl XgboostMetricSource<'_> {
    fn huber(&self) -> Result<PseudoHuber> {
        PseudoHuber::new(self.huber_slope)
    }

    fn quantiles(&self) -> Result<Quantiles> {
        Quantiles::new(self.quantile_alpha.iter().copied())
    }

    fn expectiles(&self) -> Result<Expectiles> {
        Expectiles::new(self.expectile_alpha.iter().copied())
    }

    fn aft(&self) -> Result<Aft> {
        Aft::new(self.aft_loss_distribution, self.aft_loss_distribution_scale)
    }
}

/// The parameter error of metric `name`.
fn invalid_metric(name: &str, reason: &str) -> HessboostError {
    HessboostError::invalid_param("eval_metric", format!("`{name}`: {reason}"))
}

/// The rank cutoff `@k` of ranking metric `name`: decimal digits only
/// (`usize::from_str` also takes a leading `+`), so `2.9`, `abc`, `1@2`, or
/// an empty suffix are refused rather than truncated or dropped.
fn rank_cutoff(name: &str, suffix: Option<&str>) -> Result<Cutoff> {
    match suffix {
        None => Ok(Cutoff::all()),
        Some(s) if s.ends_with('-') => Err(invalid_metric(
            name,
            "the `-` variants of the ranking metrics are not implemented",
        )),
        Some(s) => match s.parse::<usize>().ok().and_then(NonZeroUsize::new) {
            Some(k) if s.bytes().all(|b| b.is_ascii_digit()) => Ok(Cutoff::from(k)),
            _ => Err(invalid_metric(
                name,
                "the `@k` cutoff must be a positive integer",
            )),
        },
    }
}

/// The variance power `@rho` of `tweedie-nloglik` (`1.5` without a
/// suffix), in the range of the objective's `tweedie_variance_power`,
/// whose default metric this is.
fn tweedie_power(name: &str, suffix: Option<&str>) -> Result<Tweedie> {
    match suffix {
        None => Ok(Tweedie::default()),
        Some(s) => s
            .parse::<f64>()
            .ok()
            .and_then(|rho| Tweedie::new(rho).ok())
            .ok_or_else(|| invalid_metric(name, "the variance power `@rho` must be in [1, 2)")),
    }
}

/// The metric XGBoost names `name`, parameterized from `source` and built
/// for `n_outputs` outputs (tests).
#[cfg(test)]
pub(crate) fn named(
    name: &str,
    n_outputs: usize,
    source: &XgboostMetricSource<'_>,
) -> Result<Box<dyn Metric>> {
    EvalMetric::from_xgboost(name, source)?.build(n_outputs)
}

/// XGBoost's default flat parameters of the metrics (tests): slope 1, no
/// alphas, normal AFT noise at scale 1, no `dist:*` family.
#[cfg(test)]
pub(crate) const DEFAULT_SOURCE: XgboostMetricSource<'static> = XgboostMetricSource {
    huber_slope: 1.0,
    quantile_alpha: &[],
    expectile_alpha: &[],
    aft_loss_distribution: AftDistribution::Normal,
    aft_loss_distribution_scale: 1.0,
    distribution: None,
};
