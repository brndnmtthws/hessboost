#![no_main]
//! End-to-end training on small fuzzed datasets and configurations. Any
//! configuration `TrainingParams::validate` accepts must train or return an
//! error, never panic; a trained model must be deterministic across thread
//! counts and survive every prediction and serialization API.
use hessboost::config::{
    AftDistribution, BoosterKind, DistGradient, DistSplitDirection, GrowPolicy, Monotone,
    MultiStrategy, SamplingMethod, TreeMethod,
};
use hessboost::data::FeatureType;
use hessboost::prelude::*;
use libfuzzer_sys::arbitrary::{Arbitrary, Error as ArbError, Result as ArbResult, Unstructured};
use libfuzzer_sys::fuzz_target;

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

fn case(u: &mut Unstructured) -> ArbResult<Option<Case>> {
    let objective = *u.choose(OBJECTIVES)?;
    let num_class = u.int_in_range(0..=4)?;
    let Some(dtrain) = dataset(u, objective, num_class)? else {
        return Ok(None);
    };
    let n_cols = dtrain.n_cols();

    let mut params = TrainingParams {
        booster: *u.choose(&[
            BoosterKind::GbTree,
            BoosterKind::Dart,
            BoosterKind::GbLinear,
        ])?,
        seed: u.arbitrary()?,
        objective: objective.to_string(),
        num_class,
        base_score: if u.arbitrary()? {
            Some(param(u, &[0.0, 0.5, 1.0, -1.0, 2.0])?)
        } else {
            None
        },
        tweedie_variance_power: param(u, &[1.5, 1.0, 2.0])?,
        huber_slope: param(u, &[1.0, 0.1, 10.0])?,
        lambdarank_num_pair_per_sample: u.int_in_range(0..=4)?,
        quantile_alpha: alphas(u)?,
        expectile_alpha: alphas(u)?,
        aft_loss_distribution: *u.choose(&[
            AftDistribution::Normal,
            AftDistribution::Logistic,
            AftDistribution::Extreme,
        ])?,
        aft_loss_distribution_scale: param(u, &[1.0, 0.5, 2.0])?,
        dist_gradient: *u.choose(&[DistGradient::Fisher, DistGradient::Hessian])?,
        dist_split_direction: *u
            .choose(&[DistSplitDirection::Random, DistSplitDirection::Cyclic])?,
        eta: param(u, &[0.3, 0.1, 1.0, 1e-3, 10.0])?,
        gamma: param(u, &[0.0, 0.5, 10.0])?,
        max_depth: u.int_in_range(0..=6)?,
        max_leaves: u.int_in_range(0..=8)?,
        min_child_weight: param(u, &[1.0, 0.0, 0.1, 5.0])?,
        max_delta_step: if u.ratio(1, 4)? {
            Some(param(u, &[0.0, 0.7, 1.0])?)
        } else {
            None
        },
        subsample: param(u, &[1.0, 0.5, 0.1])?,
        colsample_bytree: param(u, &[1.0, 0.5, 0.0])?,
        colsample_bylevel: param(u, &[1.0, 0.5, 0.0])?,
        colsample_bynode: param(u, &[1.0, 0.5, 0.0])?,
        lambda: param(u, &[1.0, 0.0, 10.0])?,
        alpha: param(u, &[0.0, 1.0])?,
        scale_pos_weight: param(u, &[1.0, 0.5, 4.0])?,
        tree_method: *u.choose(&[
            TreeMethod::Auto,
            TreeMethod::Exact,
            TreeMethod::Approx,
            TreeMethod::Hist,
        ])?,
        grow_policy: *u.choose(&[
            GrowPolicy::DepthWise,
            GrowPolicy::LossGuide,
            GrowPolicy::Symmetric,
        ])?,
        max_bin: u.int_in_range(0..=32)?,
        num_parallel_tree: u.int_in_range(0..=3)?,
        sampling_method: *u.choose(&[SamplingMethod::Uniform, SamplingMethod::GradientBased])?,
        multi_strategy: *u.choose(&[
            MultiStrategy::OneOutputPerTree,
            MultiStrategy::MultiOutputTree,
        ])?,
        extra_trees: u.ratio(1, 6)?,
        extra_seed: u.arbitrary()?,
        path_smooth: param(u, &[0.0, 0.0, 1.0])?,
        linear_tree: u.ratio(1, 6)?,
        linear_lambda: param(u, &[0.0, 1.0])?,
        use_quantized_grad: u.ratio(1, 6)?,
        num_grad_quant_bins: u.int_in_range(0..=8)?,
        stochastic_rounding: u.arbitrary()?,
        quant_train_renew_leaf: u.arbitrary()?,
        rate_drop: param(u, &[0.0, 0.5, 1.0])?,
        skip_drop: param(u, &[0.0, 0.5, 1.0])?,
        toad_penalty_feature: param(u, &[0.0, 0.0, 1.0])?,
        toad_penalty_threshold: param(u, &[0.0, 0.0, 1.0])?,
        ..TrainingParams::default()
    };
    if u.ratio(1, 4)? {
        for _ in 0..n_cols {
            params.monotone_constraints.push(*u.choose(&[
                Monotone::None,
                Monotone::Increasing,
                Monotone::Decreasing,
            ])?);
        }
    }
    if u.ratio(1, 4)? {
        for _ in 0..u.int_in_range(1..=3)? {
            let group = (0..u.int_in_range(1..=n_cols)?)
                .map(|_| u.int_in_range(0..=n_cols as u32 - 1))
                .collect::<ArbResult<Vec<_>>>()?;
            params.interaction_constraints.push(group);
        }
    }
    if u.ratio(1, 4)? {
        params.eval_metric.push((*u.choose(METRICS)?).to_string());
    }
    if params.validate().is_err() {
        return Ok(None);
    }

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
    let params = TrainingParams {
        nthread,
        ..case.params.clone()
    };
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
