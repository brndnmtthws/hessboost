//! The objective and `base_score` mapping, incl. XGBoost's parameter blocks.

use super::parse::{scalar_count, scalar_f64};
use crate::error::{HessboostError, Result};
use crate::model::objective::{ModelObjective, StoredObjectiveParams};
use crate::objective::{AftDistribution, Loss, Objective};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// base_score link handling
// ---------------------------------------------------------------------------

/// The loss of the imported objective for `n_targets` label columns, if it
/// is one we implement. A built-in objective that rejects that layout (too
/// many targets) is a format error rather than a silently untransformed
/// model.
pub(super) fn build_objective(
    objective: &ModelObjective,
    n_targets: usize,
    max_delta_step: f64,
) -> Result<Option<Arc<dyn Loss>>> {
    match crate::model::rebuild_objective(objective, max_delta_step, n_targets) {
        None => Ok(None),
        Some(Ok(loss)) => Ok(Some(loss)),
        Some(Err(error)) if n_targets > 1 => Err(HessboostError::model_format(format!(
            "`num_target` {n_targets}: {error}"
        ))),
        Some(Err(error)) => Err(HessboostError::model_format(format!(
            "objective `{}`: {error}",
            objective.name()
        ))),
    }
}

/// Render the per-output margin intercepts as XGBoost 3.x's `base_score`
/// vector string, `"[v0,v1,...]"`, in the space XGBoost stores it in: the
/// whole row mapped through [`Loss::margins_to_probs`], the inverse of
/// the objective's `ProbToMargin`. Multiclass (softmax) values pass through
/// unchanged, since XGBoost's softmax `ProbToMargin` is the identity while
/// its transform normalizes across classes.
pub(super) fn format_base_score(margins: &[f32], objective: &dyn Loss) -> String {
    let mut stored = margins.to_vec();
    if !matches!(objective.name(), "multi:softmax" | "multi:softprob") {
        objective.margins_to_probs(&mut stored);
    }
    format_float_vector(stored)
}

/// Most outputs a single `base_score` entry is broadcast to. A document's
/// declared output counts (`num_target`, `num_class`) are bounded only by
/// its trees, and a tree-less document has none; past this, the intercepts
/// must be listed one per output (as XGBoost 3.x writes them), so the
/// imported model stays proportional to the document.
pub(super) const MAX_BROADCAST_OUTPUTS: usize = 1 << 16;

/// Parse XGBoost 3.x's `base_score` vector string (`"[5E-1]"`,
/// `"[a,b,c]"`) into per-output margin intercepts. One entry applies to every
/// output (XGBoost `HandleOldFormat`) of a model with at most
/// [`MAX_BROADCAST_OUTPUTS`] outputs; otherwise the length must equal
/// `n_outputs`. The vector is mapped through the objective's inverse link
/// (values pass through unchanged for an objective we cannot reconstruct).
pub(super) fn parse_base_score(
    stored: &str,
    objective: Option<&dyn Loss>,
    n_outputs: usize,
) -> Result<Vec<f32>> {
    let invalid = || HessboostError::model_format(format!("invalid `base_score` `{stored}`"));
    let inner = stored
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(invalid)?;
    let values = inner
        .split(',')
        .map(|v| v.trim().parse::<f32>().ok())
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(invalid)?;
    let mut values = match values.len() {
        1 if n_outputs <= MAX_BROADCAST_OUTPUTS => vec![values[0]; n_outputs],
        len if len == n_outputs => values,
        1 => {
            return Err(HessboostError::model_format(format!(
                "`base_score` has one entry for {n_outputs} outputs; above \
                 {MAX_BROADCAST_OUTPUTS} outputs it must list one entry per output"
            )));
        }
        len => {
            return Err(HessboostError::model_format(format!(
                "`base_score` has {len} entries for {n_outputs} outputs"
            )));
        }
    };
    if let Some(obj) = objective {
        obj.probs_to_margins(&mut values);
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// Objective parameter blocks
// ---------------------------------------------------------------------------

/// `(block, key)` under `learner.objective` where XGBoost 3.4.1 keeps each
/// retained parameter (`SaveConfig` of the objective owning it).
pub(super) const SCALE_POS_WEIGHT: (&str, &str) = ("reg_loss_param", "scale_pos_weight");
pub(super) const MAX_DELTA_STEP: (&str, &str) = ("poisson_regression_param", "max_delta_step");
pub(super) const TWEEDIE_VARIANCE_POWER: (&str, &str) =
    ("tweedie_regression_param", "tweedie_variance_power");
pub(super) const HUBER_SLOPE: (&str, &str) = ("pseudo_huber_param", "huber_slope");
pub(super) const LAMBDARANK_NUM_PAIR: (&str, &str) =
    ("lambdarank_param", "lambdarank_num_pair_per_sample");
pub(super) const SOFTMAX_NUM_CLASS: (&str, &str) = ("softmax_multiclass_param", "num_class");
pub(super) const QUANTILE_ALPHA: (&str, &str) = ("quantile_loss_param", "quantile_alpha");
pub(super) const EXPECTILE_ALPHA: (&str, &str) = ("expectile_loss_param", "expectile_alpha");
/// The `survival:aft` block, holding `aft_loss_distribution` and
/// `aft_loss_distribution_scale`.
pub(super) const AFT_LOSS_PARAM: &str = "aft_loss_param";

/// Encode `values` as XGBoost's float vector string (`"[0.1,0.5,0.9]"`): the
/// form of `base_score` and of `ParamArray<float>` alpha lists.
pub(super) fn format_float_vector(values: impl IntoIterator<Item = f32>) -> String {
    use std::fmt::Write;
    let mut out = String::from("[");
    for (i, v) in values.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        // Writing to a `String` cannot fail.
        let _ = write!(out, "{v}");
    }
    out.push(']');
    out
}

/// Decode XGBoost's `ParamArray<float>` string: a JSON array or a single
/// number, with `(..)` accepted for `[..]` like XGBoost's reader. Entries are
/// rounded to `f32` as XGBoost stores them. `None` for anything else.
pub(super) fn parse_param_array(text: &str) -> Option<Vec<f64>> {
    let text = text.trim();
    let text = match text.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
        Some(inner) => format!("[{inner}]"),
        None => text.to_string(),
    };
    let as_f32 = |v: &Value| v.as_f64().map(|v| f64::from(v as f32));
    match serde_json::from_str::<Value>(&text).ok()? {
        Value::Array(entries) => entries.iter().map(as_f32).collect(),
        number @ Value::Number(_) => as_f32(&number).map(|v| vec![v]),
        _ => None,
    }
}

/// Build the `objective` sub-document with the parameter block XGBoost 3.4.1
/// writes for each objective (its `SaveConfig`), so upstream XGBoost accepts
/// the file. The objective's parameters (and Poisson's `max_delta_step`) are
/// written as XGBoost's stringified numbers; LambdaRank parameters hessboost
/// does not have are written at XGBoost's defaults, and the alpha lists as
/// XGBoost's array strings. Every `RegLossObj` objective (`reg:squarederror`,
/// `reg:gamma`, and the logistic ones) writes its `scale_pos_weight` in
/// `reg_loss_param`. Objectives without parameters (`reg:squaredlogerror`,
/// `binary:hinge`, `reg:absoluteerror`, `survival:cox`) write their name
/// only.
pub(super) fn objective_to_json(objective: &Objective, max_delta_step: f64) -> Value {
    let mut out = Map::with_capacity(2);
    out.insert(
        "name".to_string(),
        Value::String(objective.name().to_string()),
    );
    let ((block, key), value) = match objective {
        Objective::Aft(aft) => {
            // `AftDistribution` serializes as XGBoost's lowercase names.
            let fields = json!({
                "aft_loss_distribution": aft.distribution(),
                "aft_loss_distribution_scale": aft.scale().to_string(),
            });
            out.insert(AFT_LOSS_PARAM.to_string(), fields);
            return Value::Object(out);
        }
        Objective::Cox
        | Objective::SquaredLogError
        | Objective::BinaryHinge
        | Objective::AbsoluteError
        | Objective::Dist(_)
        | Objective::RankXendcg
        | Objective::Custom(_) => {
            return Value::Object(out);
        }
        Objective::Quantile(q) => (
            QUANTILE_ALPHA,
            format_float_vector(q.alpha().iter().map(|&v| v as f32)),
        ),
        Objective::Expectile(e) => (
            EXPECTILE_ALPHA,
            format_float_vector(e.alpha().iter().map(|&v| v as f32)),
        ),
        Objective::Softmax(c) | Objective::Softprob(c) => {
            (SOFTMAX_NUM_CLASS, c.num_class().to_string())
        }
        Objective::Poisson => (MAX_DELTA_STEP, max_delta_step.to_string()),
        Objective::Tweedie(t) => (TWEEDIE_VARIANCE_POWER, t.variance_power().to_string()),
        Objective::PseudoHuber(h) => (HUBER_SLOPE, h.slope().to_string()),
        Objective::RankPairwise(r) | Objective::RankNdcg(r) | Objective::RankMap(r) => {
            (LAMBDARANK_NUM_PAIR, r.num_pair_per_sample().to_string())
        }
        Objective::SquaredError(r)
        | Objective::RegLogistic(r)
        | Objective::BinaryLogistic(r)
        | Objective::BinaryLogitRaw(r)
        | Objective::Gamma(r) => (SCALE_POS_WEIGHT, r.scale_pos_weight().to_string()),
    };
    let mut fields = Map::new();
    if block == LAMBDARANK_NUM_PAIR.0 {
        // The LambdaRank settings the model does not retain, at XGBoost's
        // defaults (the pairing that hessboost implements is `topk`).
        for (k, v) in [
            ("lambdarank_bias_norm", "1"),
            ("lambdarank_normalization", "1"),
            ("lambdarank_pair_method", "topk"),
            ("lambdarank_score_normalization", "1"),
            ("lambdarank_unbiased", "0"),
            ("ndcg_exp_gain", "1"),
        ] {
            fields.insert(k.to_string(), Value::String(v.to_string()));
        }
    }
    fields.insert(key.to_string(), Value::String(value));
    out.insert(block.to_string(), Value::Object(fields));
    Value::Object(out)
}

/// Read the objective's parameter block (the inverse of [`objective_to_json`])
/// back into the stored parameter record. Missing blocks or fields keep XGBoost's
/// defaults for `objective` (e.g. `max_delta_step = 0.7` for `count:poisson`);
/// a present value that does not parse (or a block that is not an object) is
/// a format error, never a silent default.
pub(super) fn objective_params_from_json(
    objective: &str,
    obj: Option<&Value>,
) -> Result<StoredObjectiveParams> {
    let mut params = StoredObjectiveParams::defaults_for(objective);
    let Some(obj) = obj else {
        return Ok(params);
    };
    let invalid =
        |key: &str, value: &Value| HessboostError::model_format(format!("invalid `{key}` {value}"));
    for (param, value) in [
        (SCALE_POS_WEIGHT, &mut params.scale_pos_weight),
        (MAX_DELTA_STEP, &mut params.max_delta_step),
        (TWEEDIE_VARIANCE_POWER, &mut params.tweedie_variance_power),
        (HUBER_SLOPE, &mut params.huber_slope),
        (
            (AFT_LOSS_PARAM, "aft_loss_distribution_scale"),
            &mut params.aft_loss_distribution_scale,
        ),
    ] {
        if let Some(v) = objective_param(obj, param)? {
            *value = scalar_f64(v).ok_or_else(|| invalid(param.1, v))?;
        }
    }
    // XGBoost writes `u32::MAX` (`LambdaRankParam::NotSet`) when unset; the
    // pair count then follows `lambdarank_pair_method`, whose `topk` default
    // is what the defaults already hold. Other counts,
    // including XGBoost's out-of-range `0`, are checked with the parameters.
    if let Some(v) = objective_param(obj, LAMBDARANK_NUM_PAIR)? {
        let count = scalar_count(v).ok_or_else(|| invalid(LAMBDARANK_NUM_PAIR.1, v))?;
        if count != u32::MAX as usize {
            params.lambdarank_num_pair_per_sample = count;
        }
    }
    for (param, alpha) in [
        (QUANTILE_ALPHA, &mut params.quantile_alpha),
        (EXPECTILE_ALPHA, &mut params.expectile_alpha),
    ] {
        if let Some(v) = objective_param(obj, param)? {
            *alpha = v
                .as_str()
                .and_then(parse_param_array)
                .ok_or_else(|| invalid(param.1, v))?;
        }
    }
    let distribution = (AFT_LOSS_PARAM, "aft_loss_distribution");
    if let Some(v) = objective_param(obj, distribution)? {
        params.aft_loss_distribution =
            AftDistribution::deserialize(v).map_err(|_| invalid(distribution.1, v))?;
    }
    Ok(params)
}

/// The value of `key` in the parameter block `block` of the objective
/// document `obj`, `None` when the block or the key is absent. A present
/// block that is not an object is malformed.
pub(super) fn objective_param<'a>(
    obj: &'a Value,
    (block, key): (&str, &str),
) -> Result<Option<&'a Value>> {
    match obj.get(block) {
        None => Ok(None),
        Some(Value::Object(fields)) => Ok(fields.get(key)),
        Some(other) => Err(HessboostError::model_format(format!(
            "objective parameter block `{block}` is not an object: {other}"
        ))),
    }
}
