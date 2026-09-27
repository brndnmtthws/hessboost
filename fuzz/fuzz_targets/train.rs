#![no_main]
//! End-to-end training on small fuzzed datasets and configurations. Any
//! configuration `TrainingParams::from_xgboost` accepts must train or return an
//! error, never panic; a trained model must be deterministic across thread
//! counts and survive every prediction and serialization API.
use hessboost::config::BoosterKind;
use hessboost::data::FeatureType;
use hessboost::prelude::*;
use libfuzzer_sys::arbitrary::{Arbitrary, Error as ArbError, Result as ArbResult, Unstructured};
use libfuzzer_sys::fuzz_target;
use serde_json::{Map, Value, json};
use std::num::NonZeroUsize;

#[path = "common.rs"]
mod common;

const MAX_ROWS: usize = 24;
const MAX_COLS: usize = 4;
const MAX_TARGETS: usize = 3;
const MAX_ROUNDS: usize = 4;

const OBJECTIVES: &[&str] = &[
    "reg:squarederror",
    "reg:squaredlogerror",
    "reg:pseudohubererror",
    "reg:absoluteerror",
    "reg:quantileerror",
    "reg:expectileerror",
    "reg:logistic",
    "reg:gamma",
    "reg:tweedie",
    "count:poisson",
    "binary:logistic",
    "binary:logitraw",
    "binary:hinge",
    "multi:softmax",
    "multi:softprob",
    "rank:pairwise",
    "rank:ndcg",
    "rank:map",
    "survival:cox",
    "survival:aft",
    "dist:normal",
    "dist:lognormal",
    "dist:gamma",
    "dist:poisson",
    "dist:negbinomial",
];

const METRICS: &[&str] = &[
    "rmse",
    "rmsle",
    "mae",
    "mape",
    "mphe",
    "logloss",
    "error",
    "error@0.7",
    "auc",
    "aucpr",
    "mlogloss",
    "merror",
    "ndcg",
    "ndcg@2",
    "map",
    "map@2-",
    "pre@2",
    "poisson-nloglik",
    "gamma-nloglik",
    "tweedie-nloglik",
    "cox-nloglik",
    "aft-nloglik",
    "interval-regression-accuracy",
    "quantile",
    "expectile",
    "nll",
    "crps",
];

/// A value from `interesting`, or occasionally any `f64`, so validation sees
/// both typical and out-of-range settings.
fn param(u: &mut Unstructured, interesting: &[f64]) -> ArbResult<f64> {
    if u.ratio(1, 8)? {
        u.arbitrary()
    } else {
        u.choose(interesting).copied()
    }
}

/// Small integers (so splits and ties are common), missing, and
/// occasionally arbitrary bit patterns.
fn value(u: &mut Unstructured) -> ArbResult<f32> {
    Ok(match u.int_in_range(0u8..=11)? {
        0 => f32::NAN,
        1 => u.arbitrary()?,
        n => f32::from(n) - 5.0,
    })
}

fn values(u: &mut Unstructured, n: usize) -> ArbResult<Vec<f32>> {
    (0..n).map(|_| value(u)).collect()
}

/// `cargo fuzz fmt train <input>` prints a crashing case.
#[derive(Debug)]
struct Case {
    params: TrainingParams,
    dtrain: DMatrix,
    rounds: usize,
    early_stopping: Option<usize>,
}

/// Applies a fallible `DMatrix` builder step; `None` skips the input.
fn step(d: DMatrix, f: impl FnOnce(DMatrix) -> Result<DMatrix>) -> Option<DMatrix> {
    f(d).ok()
}

fn dataset(u: &mut Unstructured, objective: &str, num_class: usize) -> ArbResult<Option<DMatrix>> {
    let n_rows = u.int_in_range(1..=MAX_ROWS)?;
    let n_cols = u.int_in_range(1..=MAX_COLS)?;
    let x = values(u, n_rows * n_cols)?;
    let Ok(mut d) = DMatrix::from_dense(&x, n_rows, n_cols) else {
        return Ok(None);
    };

    // Labels: class indices for classifiers, small non-negative values
    // elsewhere, occasionally anything; sometimes a label matrix.
    let n_targets = if u.ratio(1, 6)? {
        u.int_in_range(2..=MAX_TARGETS)?
    } else {
        1
    };
    let mut labels = Vec::with_capacity(n_rows * n_targets);
    for _ in 0..n_rows * n_targets {
        labels.push(match u.int_in_range(0u8..=9)? {
            0 => u.arbitrary()?,
            1 => -1.0,
            n if objective.starts_with("multi:") => f32::from(n % num_class.max(2) as u8),
            n if objective.starts_with("binary:") => f32::from(n % 2),
            n => f32::from(n) * 0.5,
        });
    }
    let Some(next) = step(d, |d| d.with_label_matrix(&labels, n_targets)) else {
        return Ok(None);
    };
    d = next;

    if objective == "survival:aft" || u.ratio(1, 16)? {
        let lower = values(u, n_rows)?
            .iter()
            .map(|v| v.abs())
            .collect::<Vec<_>>();
        let upper = lower
            .iter()
            .map(|&lo| {
                Ok(match u.int_in_range(0u8..=3)? {
                    0 => f32::INFINITY,
                    1 => lo,
                    n => lo + f32::from(n),
                })
            })
            .collect::<ArbResult<Vec<_>>>()?;
        let Some(next) = step(d, |d| d.with_label_bounds(&lower, &upper)) else {
            return Ok(None);
        };
        d = next;
    }
    if u.ratio(1, 4)? {
        let weights = (0..n_rows)
            .map(|_| Ok(f32::from(u.int_in_range(0u8..=4)?) * 0.5))
            .collect::<ArbResult<Vec<_>>>()?;
        let Some(next) = step(d, |d| d.with_weights(&weights)) else {
            return Ok(None);
        };
        d = next;
    }
    if u.ratio(1, 4)? {
        let types = (0..n_cols)
            .map(|_| {
                Ok(if u.arbitrary()? {
                    FeatureType::Categorical
                } else {
                    FeatureType::Numerical
                })
            })
            .collect::<ArbResult<Vec<_>>>()?;
        let Some(next) = step(d, |d| d.with_feature_types(&types)) else {
            return Ok(None);
        };
        d = next;
    }
    if u.ratio(1, 8)? {
        let weights = values(u, n_cols)?;
        let Some(next) = step(d, |d| d.with_feature_weights(&weights)) else {
            return Ok(None);
        };
        d = next;
    }
    if u.ratio(1, 8)? {
        let len = n_rows * *u.choose(&[1, num_class.max(1), n_targets])?;
        let margin = values(u, len)?;
        let Some(next) = step(d, |d| d.with_base_margin(&margin)) else {
            return Ok(None);
        };
        d = next;
    }
    if objective.starts_with("rank:") || u.ratio(1, 8)? {
        let mut sizes = Vec::new();
        let mut left = n_rows;
        while left > 0 {
            let size = u.int_in_range(1..=left)?;
            sizes.push(size);
            left -= size;
        }
        let Some(next) = step(d, |d| d.with_group_sizes(&sizes)) else {
            return Ok(None);
        };
        d = next;
        if u.ratio(1, 4)? {
            let weights = values(u, sizes.len())?;
            let Some(next) = step(d, |d| d.with_group_weights(&weights)) else {
                return Ok(None);
            };
            d = next;
        }
    }
    Ok(Some(d))
}

fn alphas(u: &mut Unstructured) -> ArbResult<Vec<f64>> {
    (0..u.int_in_range(0..=3)?)
        .map(|_| param(u, &[0.5, 0.1, 0.9, 0.0, 1.0]))
        .collect()
}

/// Whether `objective` reads the objective-parameter key `key` (with
/// `shared_trees` for `multi_strategy = multi_output_tree`, the only
/// setting a `dist:*` split direction applies to).
fn objective_reads(objective: &str, key: &str, shared_trees: bool) -> bool {
    match key {
        "num_class" => objective.starts_with("multi:"),
        "scale_pos_weight" => matches!(
            objective,
            "binary:logistic" | "binary:logitraw" | "reg:logistic"
        ),
        "tweedie_variance_power" => objective == "reg:tweedie",
        "huber_slope" => objective == "reg:pseudohubererror",
        "lambdarank_num_pair_per_sample" => objective.starts_with("rank:"),
        "quantile_alpha" => objective == "reg:quantileerror",
        "expectile_alpha" => objective == "reg:expectileerror",
        "aft_loss_distribution" | "aft_loss_distribution_scale" => objective == "survival:aft",
        "dist_gradient" => objective.starts_with("dist:"),
        "dist_split_direction" => objective.starts_with("dist:") && shared_trees,
        _ => false,
    }
}

/// Whether the metric `metric` borrows the objective-parameter key `key`.
fn metric_borrows(metric: &str, key: &str) -> bool {
    match metric {
        "mphe" => key == "huber_slope",
        "quantile" => key == "quantile_alpha",
        "expectile" => key == "expectile_alpha",
        "aft-nloglik" => key.starts_with("aft_loss_distribution"),
        _ => false,
    }
}

fn case(u: &mut Unstructured) -> ArbResult<Option<Case>> {
    let objective = *u.choose(OBJECTIVES)?;
    let num_class = u.int_in_range(0..=4)?;
    let Some(dtrain) = dataset(u, objective, num_class)? else {
        return Ok(None);
    };
    let n_cols = dtrain.n_cols();

    // XGBoost's flat form, read in a fixed order so a corpus input keeps
    // its meaning; `from_xgboost` refuses what `validate` would. The
    // objective parameters are drawn in place but set only when the
    // objective or the metric (drawn last) reads them, since `from_xgboost`
    // refuses a key nothing reads; likewise every option-group key is drawn
    // but set only when its switch is on.
    let mut flat = Map::new();
    let mut objective_params: Vec<(&str, Value)> = Vec::new();
    let mut set = |key: &str, value: Value| {
        flat.insert(key.to_owned(), value);
    };
    let booster = *u.choose(&["gbtree", "dart", "gblinear"])?;
    set("booster", json!(booster));
    set("seed", json!(u.arbitrary::<u64>()?));
    set("objective", json!(objective));
    objective_params.push(("num_class", json!(num_class)));
    if u.arbitrary()? {
        set("base_score", json!(param(u, &[0.0, 0.5, 1.0, -1.0, 2.0])?));
    }
    objective_params.push(("tweedie_variance_power", json!(param(u, &[1.5, 1.0, 2.0])?)));
    objective_params.push(("huber_slope", json!(param(u, &[1.0, 0.1, 10.0])?)));
    objective_params.push((
        "lambdarank_num_pair_per_sample",
        json!(u.int_in_range(0..=4)?),
    ));
    objective_params.push(("quantile_alpha", json!(alphas(u)?)));
    objective_params.push(("expectile_alpha", json!(alphas(u)?)));
    objective_params.push((
        "aft_loss_distribution",
        json!(*u.choose(&["normal", "logistic", "extreme"])?),
    ));
    objective_params.push((
        "aft_loss_distribution_scale",
        json!(param(u, &[1.0, 0.5, 2.0])?),
    ));
    objective_params.push(("dist_gradient", json!(*u.choose(&["fisher", "hessian"])?)));
    objective_params.push((
        "dist_split_direction",
        json!(*u.choose(&["random", "cyclic"])?),
    ));
    set("eta", json!(param(u, &[0.3, 0.1, 1.0, 1e-3, 10.0])?));
    set("gamma", json!(param(u, &[0.0, 0.5, 10.0])?));
    set("max_depth", json!(u.int_in_range(0..=6)?));
    set("max_leaves", json!(u.int_in_range(0..=8)?));
    set("min_child_weight", json!(param(u, &[1.0, 0.0, 0.1, 5.0])?));
    if u.ratio(1, 4)? {
        set("max_delta_step", json!(param(u, &[0.0, 0.7, 1.0])?));
    }
    set("subsample", json!(param(u, &[1.0, 0.5, 0.1])?));
    set("colsample_bytree", json!(param(u, &[1.0, 0.5, 0.0])?));
    set("colsample_bylevel", json!(param(u, &[1.0, 0.5, 0.0])?));
    set("colsample_bynode", json!(param(u, &[1.0, 0.5, 0.0])?));
    set("lambda", json!(param(u, &[1.0, 0.0, 10.0])?));
    set("alpha", json!(param(u, &[0.0, 1.0])?));
    objective_params.push(("scale_pos_weight", json!(param(u, &[1.0, 0.5, 4.0])?)));
    set(
        "tree_method",
        json!(*u.choose(&["auto", "exact", "approx", "hist"])?),
    );
    set(
        "grow_policy",
        json!(*u.choose(&["depthwise", "lossguide", "symmetric"])?),
    );
    set("max_bin", json!(u.int_in_range(0..=32)?));
    set("num_parallel_tree", json!(u.int_in_range(0..=3)?));
    set(
        "sampling_method",
        json!(*u.choose(&["uniform", "gradient_based"])?),
    );
    let multi_strategy = *u.choose(&["one_output_per_tree", "multi_output_tree"])?;
    set("multi_strategy", json!(multi_strategy));
    let extra_trees = u.ratio(1, 6)?;
    set("extra_trees", json!(extra_trees));
    let extra_seed = u.arbitrary::<u64>()?;
    if extra_trees {
        set("extra_seed", json!(extra_seed));
    }
    set("path_smooth", json!(param(u, &[0.0, 0.0, 1.0])?));
    let linear_tree = u.ratio(1, 6)?;
    set("linear_tree", json!(linear_tree));
    let linear_lambda = param(u, &[0.0, 1.0])?;
    if linear_tree {
        set("linear_lambda", json!(linear_lambda));
    }
    let quantized = u.ratio(1, 6)?;
    set("use_quantized_grad", json!(quantized));
    let quantization = [
        ("num_grad_quant_bins", json!(u.int_in_range(0..=8)?)),
        ("stochastic_rounding", json!(u.arbitrary::<bool>()?)),
        ("quant_train_renew_leaf", json!(u.arbitrary::<bool>()?)),
    ];
    if quantized {
        for (key, value) in quantization {
            set(key, value);
        }
    }
    let dropout = [
        ("rate_drop", json!(param(u, &[0.0, 0.5, 1.0])?)),
        ("skip_drop", json!(param(u, &[0.0, 0.5, 1.0])?)),
    ];
    if booster == "dart" {
        for (key, value) in dropout {
            set(key, value);
        }
    }
    set("toad_penalty_feature", json!(param(u, &[0.0, 0.0, 1.0])?));
    set("toad_penalty_threshold", json!(param(u, &[0.0, 0.0, 1.0])?));
    if u.ratio(1, 4)? {
        let monotone = (0..n_cols)
            .map(|_| u.choose(&[0, 1, -1]).copied())
            .collect::<ArbResult<Vec<i8>>>()?;
        set("monotone_constraints", json!(monotone));
    }
    if u.ratio(1, 4)? {
        let groups = (0..u.int_in_range(1..=3)?)
            .map(|_| {
                (0..u.int_in_range(1..=n_cols)?)
                    .map(|_| u.int_in_range(0..=n_cols as u32 - 1))
                    .collect::<ArbResult<Vec<_>>>()
            })
            .collect::<ArbResult<Vec<_>>>()?;
        set("interaction_constraints", json!(groups));
    }
    let metric = if u.ratio(1, 4)? {
        Some(*u.choose(METRICS)?)
    } else {
        None
    };
    if let Some(metric) = metric {
        set("eval_metric", json!([metric]));
    }
    let shared_trees = multi_strategy == "multi_output_tree";
    for (key, value) in objective_params {
        if objective_reads(objective, key, shared_trees)
            || metric.is_some_and(|metric| metric_borrows(metric, key))
        {
            set(key, value);
        }
    }
    let Ok(params) = TrainingParams::from_xgboost(flat) else {
        return Ok(None);
    };

    Ok(Some(Case {
        params,
        dtrain,
        rounds: u.int_in_range(1..=MAX_ROUNDS)?,
        early_stopping: if u.arbitrary()? {
            Some(u.int_in_range(1..=2)?)
        } else {
            None
        },
    }))
}

impl<'a> Arbitrary<'a> for Case {
    fn arbitrary(u: &mut Unstructured<'a>) -> ArbResult<Self> {
        case(u)?.ok_or(ArbError::IncorrectFormat)
    }
}

fn fit(case: &Case, nthread: usize) -> Option<BoostedModel> {
    let mut params = case.params.clone();
    params.nthread = NonZeroUsize::new(nthread);
    let mut trainer = Trainer::new(&params, &case.dtrain, case.rounds);
    // gblinear refuses evaluation sets and early stopping; attaching them
    // would reject every linear case before it trains.
    if params.booster != BoosterKind::GbLinear {
        trainer = trainer.eval(&case.dtrain, "train");
        if let Some(rounds) = case.early_stopping {
            trainer = trainer.early_stopping_rounds(rounds);
        }
    }
    trainer.train().ok().map(|result| result.model)
}

fuzz_target!(|case: Case| {
    let Some(model) = fit(&case, 1) else {
        return;
    };
    assert_eq!(model.n_features(), case.dtrain.n_cols());
    common::exercise(&model);

    let margin = model
        .predict_margin(&case.dtrain)
        .expect("a model predicts its training data");
    assert_eq!(margin.len(), case.dtrain.n_rows() * model.n_outputs());
    let parallel = fit(&case, 3).expect("training succeeds whatever the thread count");
    let parallel = parallel
        .predict_margin(&case.dtrain)
        .expect("a model predicts its training data");
    assert!(
        common::same_bits(&margin, &parallel),
        "training depends on the thread count"
    );
});
