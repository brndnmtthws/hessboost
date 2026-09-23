//! The gradient-boosting training loop.

use crate::config::{
    BoosterKind, GrowPolicy, MultiStrategy, ObjectiveParams, ProcessType, SamplingMethod,
    TrainingParams, TreeMethod,
};
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::data::{DMatrix, MetaInfo};
use crate::error::{HessboostError, Result};
use crate::learner::model::{BoostedModel, ModelSpec};
use crate::learner::sampling::gradient_based_sample;
use crate::metric::create_metrics;
use crate::objective::{GradPair, create_objective};
use crate::tree::RegTree;
use crate::tree::builder::{
    ExactTreeBuilder, HistTreeBuilder, SortedColumns, all_rows, check_symmetric_input,
};
use crate::tree::sampler::ColumnSampler;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Prepared, reusable per-round builder state, chosen by `tree_method`.
enum Prepared {
    Exact(SortedColumns),
    Hist(GHistIndex),
    /// `tree_method=approx`: Hessian-weighted cuts. XGBoost regenerates them
    /// every round from a sorted-column summary unless the objective has a
    /// constant Hessian, in which case the round-0 streaming sketch is built
    /// once and reused (`BatchParam::regen = !const_hess`).
    Approx {
        const_hess: bool,
        cached: std::sync::OnceLock<GHistIndex>,
    },
}

impl Prepared {
    /// Grow one tree for this round's gradients and samples.
    fn build_tree(
        &self,
        params: &TrainingParams,
        dtrain: &DMatrix,
        gpair: &[GradPair],
        rows: &[u32],
        sampler: &mut ColumnSampler,
    ) -> RegTree {
        match self {
            Prepared::Exact(cols) => {
                ExactTreeBuilder::new(params).build(cols, dtrain, gpair, rows, sampler)
            }
            Prepared::Hist(ghist) => {
                HistTreeBuilder::new(params).build(ghist, gpair, rows, sampler)
            }
            Prepared::Approx { const_hess, cached } => {
                let bin = || {
                    let hessians: Vec<f32> = gpair.iter().map(|g| g.hess).collect();
                    let cuts = HistCuts::from_dmatrix_weighted(
                        dtrain,
                        params.max_bin,
                        &hessians,
                        !const_hess,
                    );
                    GHistIndex::from_dmatrix(dtrain, cuts)
                };
                if *const_hess {
                    HistTreeBuilder::new(params).build(
                        cached.get_or_init(bin),
                        gpair,
                        rows,
                        sampler,
                    )
                } else {
                    HistTreeBuilder::new(params).build(&bin(), gpair, rows, sampler)
                }
            }
        }
    }
}

/// Resolve `tree_method` (handling `Auto`) and prepare the matching builder
/// state once, up front. `const_hess` is the objective's
/// [`Objective::const_hess`](crate::objective::Objective::const_hess), which
/// decides whether `approx` regenerates its cuts every round.
fn prepare_builder(
    params: &TrainingParams,
    dtrain: &DMatrix,
    const_hess: bool,
) -> Result<Prepared> {
    let method = match params.tree_method {
        // Auto favors the histogram method, as modern XGBoost does.
        TreeMethod::Auto | TreeMethod::Hist => TreeMethod::Hist,
        TreeMethod::Exact => TreeMethod::Exact,
        TreeMethod::Approx => TreeMethod::Approx,
    };
    if method == TreeMethod::Exact
        && params.sampling_method == SamplingMethod::GradientBased
        && params.subsample < 1.0
    {
        return Err(HessboostError::invalid_param(
            "sampling_method",
            "`gradient_based` sampling requires `tree_method=hist` or `approx`; \
             `exact` supports only `uniform`",
        ));
    }
    if method == TreeMethod::Exact && params.grow_policy == GrowPolicy::LossGuide {
        return Err(HessboostError::invalid_param(
            "grow_policy",
            "`lossguide` growth requires `tree_method=hist`",
        ));
    }
    if params.grow_policy == GrowPolicy::Symmetric {
        check_symmetric_input(method, dtrain)?;
    }
    Ok(match method {
        TreeMethod::Hist => {
            let cuts = HistCuts::from_dmatrix(dtrain, params.max_bin);
            Prepared::Hist(GHistIndex::from_dmatrix(dtrain, cuts))
        }
        TreeMethod::Approx => Prepared::Approx {
            const_hess,
            cached: std::sync::OnceLock::new(),
        },
        _ => Prepared::Exact(SortedColumns::from_dmatrix(dtrain)),
    })
}

/// A named evaluation dataset watched during training.
pub type EvalSet<'a> = (&'a DMatrix, &'a str);

/// One row of the evaluation history: the metric values computed at the end of
/// a boosting round.
#[derive(Debug, Clone)]
pub struct RoundEval {
    /// The 0-based boosting iteration.
    pub iteration: usize,
    /// `(dataset_name, metric_name, value)` triples.
    pub scores: Vec<(String, String, f64)>,
}

/// The result of training: the model plus the per-round evaluation history.
#[derive(Debug)]
pub struct TrainResult {
    /// The trained model.
    pub model: BoostedModel,
    /// Evaluation history (empty when no eval sets were supplied).
    pub history: Vec<RoundEval>,
}

/// Train a model with default settings (no eval sets, no early stopping).
pub fn train(
    params: &TrainingParams,
    dtrain: &DMatrix,
    num_boost_round: usize,
) -> Result<BoostedModel> {
    Ok(train_with_eval(params, dtrain, num_boost_round, &[], None)?.model)
}

/// Train a model, watching `evals` and optionally stopping early.
///
/// Early stopping monitors the **last** metric of the **last** eval set (as in
/// XGBoost): training halts when it fails to improve for
/// `early_stopping_rounds` consecutive rounds, and the model's
/// `best_iteration` is set accordingly.
pub fn train_with_eval(
    params: &TrainingParams,
    dtrain: &DMatrix,
    num_boost_round: usize,
    evals: &[EvalSet],
    early_stopping_rounds: Option<usize>,
) -> Result<TrainResult> {
    let objective = create_objective(params, dtrain.n_targets())?;
    train_impl(
        params,
        dtrain,
        num_boost_round,
        evals,
        early_stopping_rounds,
        objective.as_ref(),
        None,
    )
}

/// Train with a user-supplied [`Metric`](crate::metric::Metric) (the
/// custom-metric hook).
///
/// The supplied `metric` replaces the metrics that would otherwise be built
/// from `eval_metric`/the objective default: it is the sole metric reported for
/// each eval set and the one driving early stopping (per its `maximize`).
pub fn train_with_custom_metric(
    params: &TrainingParams,
    dtrain: &DMatrix,
    num_boost_round: usize,
    evals: &[EvalSet],
    early_stopping_rounds: Option<usize>,
    metric: Box<dyn crate::metric::Metric>,
) -> Result<TrainResult> {
    let objective = create_objective(params, dtrain.n_targets())?;
    train_impl(
        params,
        dtrain,
        num_boost_round,
        evals,
        early_stopping_rounds,
        objective.as_ref(),
        Some(metric),
    )
}

/// Train with a user-supplied [`Objective`](crate::objective::Objective) (the
/// custom-objective hook).
pub fn train_with_objective(
    params: &TrainingParams,
    dtrain: &DMatrix,
    num_boost_round: usize,
    objective: &dyn crate::objective::Objective,
) -> Result<BoostedModel> {
    Ok(train_impl(params, dtrain, num_boost_round, &[], None, objective, None)?.model)
}

/// The core boosting loop, generic over single- and multi-output objectives.
///
/// Margins and gradients are laid out `[instance][output]`. Each round computes
/// all gradients, then grows one tree per output from that output's gradient
/// slice. This is the multi-output generalization of gradient boosting used by
/// multiclass.
fn train_impl(
    params: &TrainingParams,
    dtrain: &DMatrix,
    num_boost_round: usize,
    evals: &[EvalSet],
    early_stopping_rounds: Option<usize>,
    objective: &dyn crate::objective::Objective,
    metric_override: Option<Box<dyn crate::metric::Metric>>,
) -> Result<TrainResult> {
    if params.nthread > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(params.nthread)
            .build()
            .map_err(|error| HessboostError::invalid_param("nthread", error.to_string()))?;
        return pool.install(|| {
            train_impl_inner(
                params,
                dtrain,
                num_boost_round,
                evals,
                early_stopping_rounds,
                objective,
                metric_override,
            )
        });
    }
    train_impl_inner(
        params,
        dtrain,
        num_boost_round,
        evals,
        early_stopping_rounds,
        objective,
        metric_override,
    )
}

fn train_impl_inner(
    params: &TrainingParams,
    dtrain: &DMatrix,
    num_boost_round: usize,
    evals: &[EvalSet],
    early_stopping_rounds: Option<usize>,
    objective: &dyn crate::objective::Objective,
    metric_override: Option<Box<dyn crate::metric::Metric>>,
) -> Result<TrainResult> {
    params.validate()?;
    reject_unimplemented(params)?;

    if !params.missing.is_nan() {
        return Err(HessboostError::invalid_param(
            "missing",
            "set the sentinel when constructing DMatrix with from_dense_with_missing",
        ));
    }
    if early_stopping_rounds == Some(0) {
        return Err(HessboostError::invalid_param(
            "early_stopping_rounds",
            "must be greater than zero",
        ));
    }
    if early_stopping_rounds.is_some() && evals.is_empty() {
        return Err(HessboostError::invalid_param(
            "early_stopping_rounds",
            "requires at least one evaluation dataset",
        ));
    }

    if objective.requires_labels() && dtrain.labels().is_none() {
        return Err(HessboostError::EmptyDataset("train: dtrain has no labels"));
    }
    let info = dtrain.info();
    let n = dtrain.n_rows();
    let n_features = dtrain.n_cols();
    let n_out = objective.n_outputs();
    if let Some(base_score) = params.base_score {
        let invalid = match params.objective.as_str() {
            "binary:logistic" | "reg:logistic" => !(0.0 < base_score && base_score < 1.0),
            "count:poisson" | "reg:gamma" | "reg:tweedie" | "survival:cox" | "survival:aft" => {
                base_score <= 0.0
            }
            _ => false,
        };
        if invalid {
            return Err(HessboostError::invalid_param(
                "base_score",
                "is outside the objective's valid output domain",
            ));
        }
    }
    validate_dataset(
        objective,
        dtrain,
        dtrain.n_targets(),
        n_features,
        n_out,
        "dtrain",
    )?;
    for (data, name) in evals {
        validate_dataset(objective, data, dtrain.n_targets(), n_features, n_out, name)?;
    }
    if params.monotone_constraints.len() > n_features {
        return Err(HessboostError::invalid_param(
            "monotone_constraints",
            "contains more entries than the training matrix has features",
        ));
    }
    for group in &params.interaction_constraints {
        if group.is_empty() {
            return Err(HessboostError::invalid_param(
                "interaction_constraints",
                "constraint groups cannot be empty",
            ));
        }
        if let Some(&feature) = group.iter().find(|&&f| f as usize >= n_features) {
            return Err(HessboostError::FeatureOutOfBounds {
                index: feature as usize,
                num_features: n_features,
            });
        }
    }

    // Per-output intercepts in margin space. A user-supplied `base_score` is
    // given in prediction space and broadcast to every output through the
    // objective's link (XGBoost `ProbToMargin`; for multiclass this is a
    // uniform nonzero margin, as in XGBoost); otherwise the objective estimates
    // them from the labels (XGBoost `InitEstimation`).
    let base_margins = match params.base_score {
        Some(bs) => {
            let mut scores = vec![bs as f32; n_out];
            objective.probs_to_margins(&mut scores);
            scores
        }
        None => objective.base_margins_info(&info),
    };
    if base_margins.len() != n_out {
        return Err(HessboostError::DimensionMismatch {
            what: "objective base_margins length",
            expected: n_out,
            got: base_margins.len(),
        });
    }
    if base_margins.iter().any(|m| !m.is_finite()) {
        return Err(HessboostError::invalid_param(
            "base_score",
            format!("estimated intercept is not finite ({base_margins:?}); check the labels"),
        ));
    }

    // The model records the objective's own name and output count (not the
    // configured string / `num_class`): a `reg:linear` alias is saved as
    // `reg:squarederror` like XGBoost does, and a custom objective's outputs
    // determine the tree layout even though `num_class` is 0.
    let mut model = BoostedModel::new(
        base_margins,
        ModelSpec {
            objective: objective.name().to_string(),
            objective_params: ObjectiveParams::from_params(params),
            num_class: params.num_class,
            n_outputs: n_out,
            n_targets: dtrain.n_targets(),
            n_features,
        },
    );

    // The linear (`gblinear`) booster fits a coordinate-descent linear model
    // instead of growing trees; it skips the tree/dart path entirely. Eval sets
    // and early stopping are rejected for it (the history stays empty).
    if params.booster == BoosterKind::GbLinear {
        if !evals.is_empty() || early_stopping_rounds.is_some() {
            return Err(HessboostError::invalid_param(
                "booster",
                "gblinear does not yet support evaluation sets or early stopping",
            ));
        }
        let linear = crate::booster::gblinear::train_gblinear(
            params,
            dtrain,
            num_boost_round,
            &model.initial_margins(dtrain),
            n_out,
            objective,
        );
        model.set_linear(linear);
        return Ok(TrainResult {
            model,
            history: Vec::new(),
        });
    }

    let prepared = prepare_builder(params, dtrain, objective.const_hess())?;

    // Incremental margin caches (length rows × n_out). A dataset's per-instance
    // `base_margin`, when present, overrides the per-output intercepts.
    let mut train_margin = model.initial_margins(dtrain);
    let mut eval_margins: Vec<Vec<f32>> = evals
        .iter()
        .map(|(d, _)| model.initial_margins(d))
        .collect();

    // A caller-supplied metric replaces the configured/default metric list.
    let metrics = match metric_override {
        Some(m) => vec![m],
        None => create_metrics(
            &params.eval_metric,
            &objective.default_metric(),
            params.num_class,
            &ObjectiveParams::from_params(params),
        )?,
    };
    if dtrain.n_targets() > 1
        && let Some(metric) = metrics.iter().find(|m| !m.supports_label_matrix())
    {
        return Err(HessboostError::invalid_param(
            "eval_metric",
            format!(
                "metric `{}` does not support multi-target labels",
                metric.name()
            ),
        ));
    }

    let mut gpair = vec![GradPair::default(); n * n_out];
    // Per-output gradient buffer reused across classes (single-output aliases it).
    let mut gpair_k = vec![GradPair::default(); n];
    let mut history: Vec<RoundEval> = Vec::new();

    // Early-stopping bookkeeping.
    let maximize = metrics.last().is_some_and(|m| m.maximize());
    let mut best_score = if maximize {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let mut best_iter = 0usize;
    let mut rounds_since_improve = 0usize;

    let is_dart = params.booster == BoosterKind::Dart;

    for round in 0..num_boost_round {
        if is_dart {
            dart_round(
                &mut model,
                params,
                dtrain,
                &prepared,
                objective,
                &info,
                n,
                n_out,
                n_features,
                round,
                &mut gpair,
                &mut gpair_k,
            );
            // DART rescales earlier trees' weights each round, so the cached
            // Eval margins are no longer additive. Recompute them from the
            // (weighted) ensemble.
            for (ei, (d, _)) in evals.iter().enumerate() {
                eval_margins[ei] = model.predict_margin_limited_unchecked(d, 0);
            }
        } else {
            // 1. Gradients from the current margins (all outputs at once).
            objective.gradient_info(&train_margin, &info, &mut gpair);

            // 2. Row subsampling is shared across the round's per-output trees.
            let mut rng = round_rng(params, round, 0);
            let row_subset = sample_rows(n, params, &mut rng);

            // 3. One tree per output.
            for k in 0..n_out {
                // Retaining the final row partitions replaces a per-row tree
                // traversal of the raw feature matrix with one sequential
                // pass per leaf (constant leaves only).
                let (tree, leaf_rows) = match &prepared {
                    Prepared::Hist(ghist)
                        if params.grow_policy != GrowPolicy::LossGuide
                            && row_subset.len() == n
                            && !gradient_sampling(params)
                            && !params.linear_tree =>
                    {
                        let gk: &[GradPair] = gather_output(&gpair, &mut gpair_k, n_out, k);
                        let mut sampler = make_column_sampler(
                            n_features,
                            dtrain.feature_weights(),
                            params,
                            &mut rng,
                        );
                        let (mut tree, leaf_rows) = HistTreeBuilder::new(params)
                            .build_with_leaf_rows(ghist, gk, &row_subset, &mut sampler);
                        tree.scale_leaves(params.eta as f32);
                        (tree, leaf_rows)
                    }
                    _ => (
                        fit_output_tree(
                            params,
                            &prepared,
                            dtrain,
                            &gpair,
                            &mut gpair_k,
                            &mut rng,
                            n_out,
                            k,
                            round,
                            &row_subset,
                            n_features,
                        ),
                        Vec::new(),
                    ),
                };

                // Row partitions already identify training leaves when every
                // row participated in depthwise histogram construction.
                if leaf_rows.is_empty() {
                    update_tree_margins(&tree, dtrain, &mut train_margin, n_out, k);
                } else {
                    apply_leaf_rows(&tree, &leaf_rows, &mut train_margin, n_out, k);
                }
                for (ei, (d, _)) in evals.iter().enumerate() {
                    update_tree_margins(&tree, d, &mut eval_margins[ei], n_out, k);
                }

                model.push_tree_weighted(tree, 1.0);
            }
        }

        // 4. Evaluate metrics on each eval set.
        if !evals.is_empty() {
            let mut scores = Vec::new();
            let mut last_metric_value = 0.0;
            for (ei, (d, name)) in evals.iter().enumerate() {
                let mut preds = eval_margins[ei].clone();
                objective.eval_transform(&mut preds);
                let d_info = d.info();
                for m in &metrics {
                    let v = m.eval_info(&preds, &d_info);
                    scores.push((name.to_string(), m.name().to_string(), v));
                    last_metric_value = v;
                }
            }
            history.push(RoundEval {
                iteration: round,
                scores,
            });

            // 5. Early stopping on the last metric of the last eval set.
            if let Some(patience) = early_stopping_rounds {
                let improved = if maximize {
                    last_metric_value > best_score
                } else {
                    last_metric_value < best_score
                };
                if improved {
                    best_score = last_metric_value;
                    best_iter = round;
                    rounds_since_improve = 0;
                } else {
                    rounds_since_improve += 1;
                    if rounds_since_improve >= patience {
                        model.set_best_iteration(Some(best_iter));
                        break;
                    }
                }
            }
        }
    }

    Ok(TrainResult { model, history })
}

/// Refuse configurations the training loop does not act on yet, so a
/// setting is never accepted and silently ignored. Each clause goes away
/// with the feature that implements it.
fn reject_unimplemented(params: &TrainingParams) -> Result<()> {
    let unsupported = |name: &'static str, reason: &str| {
        Err(HessboostError::invalid_param(
            name,
            format!("{reason} is not implemented"),
        ))
    };
    if params.num_parallel_tree != 1 {
        return unsupported("num_parallel_tree", "a value other than 1");
    }
    if params.multi_strategy != MultiStrategy::OneOutputPerTree {
        return unsupported("multi_strategy", "`multi_output_tree`");
    }
    if params.process_type != ProcessType::Default {
        return unsupported("process_type", "`update`");
    }
    Ok(())
}

/// Add one tree's predictions to one output column. Rows are independent, so
/// parallel traversal preserves each row's floating-point addition order.
fn update_tree_margins(
    tree: &RegTree,
    data: &DMatrix,
    margins: &mut [f32],
    n_out: usize,
    output: usize,
) {
    let update = |(row, margin): (usize, &mut [f32])| {
        margin[output] += tree.predict_row(data, row);
    };
    if data.n_rows() >= 4096 && rayon::current_num_threads() > 1 {
        margins
            .par_chunks_mut(n_out)
            .with_min_len(1024)
            .enumerate()
            .for_each(update);
    } else {
        margins.chunks_mut(n_out).enumerate().for_each(update);
    }
}

/// Add each leaf's value to the margins of the rows that reached it. Leaf row
/// lists are ascending, so each parallel row chunk locates its slice of every
/// leaf by binary search. The per-row addition order is unchanged.
fn apply_leaf_rows(
    tree: &RegTree,
    leaf_rows: &[crate::tree::builder::LeafRows],
    margins: &mut [f32],
    n_out: usize,
    output: usize,
) {
    const CHUNK_ROWS: usize = 8192;
    let n = margins.len() / n_out;
    if n < 2 * CHUNK_ROWS || rayon::current_num_threads() <= 1 {
        for leaf in leaf_rows {
            let value = tree.node(leaf.node).leaf_value;
            for &row in &leaf.rows {
                margins[row as usize * n_out + output] += value;
            }
        }
        return;
    }
    margins
        .par_chunks_mut(CHUNK_ROWS * n_out)
        .enumerate()
        .for_each(|(chunk, margins)| {
            let first = (chunk * CHUNK_ROWS) as u32;
            let last = first + (margins.len() / n_out) as u32;
            for leaf in leaf_rows {
                let value = tree.node(leaf.node).leaf_value;
                let start = leaf.rows.partition_point(|&row| row < first);
                let end = start + leaf.rows[start..].partition_point(|&row| row < last);
                for &row in &leaf.rows[start..end] {
                    margins[(row - first) as usize * n_out + output] += value;
                }
            }
        });
}

/// Perform one DART (Dropout Additive Regression Trees) boosting round.
///
/// With probability `1 - skip_drop` a dropout set `D` is selected from the trees
/// built so far (each dropped independently with probability `rate_drop`, at
/// least one when any exist). The round's gradients are computed from the
/// ensemble **excluding** `D`. The new per-output trees are then fit on those
/// gradients. Using XGBoost's `tree` normalization, if `k = |D|` the new trees
/// get weight `1/(k+eta)` and each dropped tree is rescaled by `k/(k+eta)`.
#[allow(clippy::too_many_arguments)]
fn dart_round(
    model: &mut BoostedModel,
    params: &TrainingParams,
    dtrain: &DMatrix,
    prepared: &Prepared,
    objective: &dyn crate::objective::Objective,
    info: &MetaInfo,
    n: usize,
    n_out: usize,
    n_features: usize,
    round: usize,
    gpair: &mut [GradPair],
    gpair_k: &mut [GradPair],
) {
    let mut rng = round_rng(params, round, 0x0DA27);

    // 1. Select the dropout set over the trees built so far.
    let existing = model.num_trees();
    let mut dropped = vec![false; existing];
    let mut drop_indices: Vec<usize> = Vec::new();
    let skip = rng.random::<f64>() < params.skip_drop;
    if !skip && existing > 0 {
        for (i, d) in dropped.iter_mut().enumerate() {
            if rng.random::<f64>() < params.rate_drop {
                *d = true;
                drop_indices.push(i);
            }
        }
        if drop_indices.is_empty() {
            // Guarantee at least one dropped tree, as XGBoost does.
            let i = rng.random_range(0..existing);
            dropped[i] = true;
            drop_indices.push(i);
        }
    }
    let k = drop_indices.len();

    // 2. Gradients from the ensemble minus the dropout set.
    let margin_excl = model.predict_margin_dropout(dtrain, &dropped);
    objective.gradient_info(&margin_excl, info, gpair);

    // 3. Fit one new tree per output on those gradients.
    let row_subset = sample_rows(n, params, &mut rng);
    let eta = params.eta as f32;
    let new_weight = if k == 0 { 1.0 } else { 1.0 / (k as f32 + eta) };
    for kk in 0..n_out {
        let tree = fit_output_tree(
            params,
            prepared,
            dtrain,
            gpair,
            gpair_k,
            &mut rng,
            n_out,
            kk,
            round,
            &row_subset,
            n_features,
        );
        model.push_tree_weighted(tree, new_weight);
    }

    // 4. Rescale the dropped trees so the ensemble stays balanced.
    let factor = if k == 0 {
        1.0
    } else {
        k as f32 / (k as f32 + eta)
    };
    for &i in &drop_indices {
        model.scale_tree_weight(i, factor);
    }
}

/// Borrow the gradient slice for output `k`: the whole buffer for
/// single-output objectives, otherwise gather output `k`'s pairs into
/// `scratch` (length `n`) and borrow that. Shared by the boosting loop and
/// the DART rounds.
fn gather_output<'a>(
    gpair: &'a [GradPair],
    scratch: &'a mut [GradPair],
    n_out: usize,
    k: usize,
) -> &'a [GradPair] {
    if n_out == 1 {
        gpair
    } else {
        for (r, dst) in scratch.iter_mut().enumerate() {
            *dst = gpair[r * n_out + k];
        }
        scratch
    }
}

/// The RNG for one boosting round: `seed ^ round * 0x9E37_79B9`, plus a
/// booster-specific `salt` (`0` for gbtree, `0x0DA27` for DART) so the two
/// boosters draw from different streams.
fn round_rng(params: &TrainingParams, round: usize, salt: u64) -> StdRng {
    StdRng::seed_from_u64(params.seed ^ (round as u64).wrapping_mul(0x9E37_79B9) ^ salt)
}

/// Fit one output's tree for a boosting round: gather that output's gradient
/// slice, apply gradient-based row sampling when configured (per tree, as
/// XGBoost does), derive its column sampler, build the tree, and shrink its
/// leaves by `eta`. The caller owns the round RNG (already seeded and salted),
/// the round's uniform row subset, and what happens to the tree (margin
/// updates, contribution weight).
#[allow(clippy::too_many_arguments)]
fn fit_output_tree(
    params: &TrainingParams,
    prepared: &Prepared,
    dtrain: &DMatrix,
    gpair: &[GradPair],
    gpair_k: &mut [GradPair],
    rng: &mut StdRng,
    n_out: usize,
    k: usize,
    round: usize,
    row_subset: &[u32],
    n_features: usize,
) -> RegTree {
    let gk: &[GradPair] = gather_output(gpair, gpair_k, n_out, k);
    let sampled = if gradient_sampling(params) {
        gradient_based_sample(gk, 1, params.subsample, rng)
    } else {
        None
    };
    let (gk, rows) = match &sampled {
        Some(s) => (s.gpair.as_slice(), s.rows.as_slice()),
        None => (gk, row_subset),
    };
    let mut sampler = make_column_sampler(n_features, dtrain.feature_weights(), params, rng);
    let mut tree = prepared.build_tree(params, dtrain, gk, rows, &mut sampler);
    // LightGBM keeps the first round's trees constant.
    if params.linear_tree && round > 0 {
        crate::tree::linear::fit_linear_leaves(&mut tree, dtrain, gk, rows, params.linear_lambda);
    }
    tree.scale_leaves(params.eta as f32);
    tree
}

/// Bernoulli row subsampling (each row kept with probability `subsample`),
/// matching XGBoost's default sampling method. Guarantees at least one row.
/// Gradient-based sampling keeps every row here; it samples each tree's
/// gradients in [`fit_output_tree`] instead.
fn sample_rows(n: usize, params: &TrainingParams, rng: &mut StdRng) -> Vec<u32> {
    let subsample = params.subsample;
    if subsample >= 1.0 || params.sampling_method == SamplingMethod::GradientBased {
        return all_rows(n);
    }
    let mut rows: Vec<u32> = (0..n as u32)
        .filter(|_| rng.random::<f64>() < subsample)
        .collect();
    if rows.is_empty() {
        rows.push(rng.random_range(0..n as u32));
    }
    rows
}

/// Whether trees are grown on gradient-based (MVS) row samples.
fn gradient_sampling(params: &TrainingParams) -> bool {
    params.sampling_method == SamplingMethod::GradientBased && params.subsample < 1.0
}

/// Shape and metadata checks for the training matrix and every eval set,
/// followed by the objective's own [`validate_info`] label-domain checks.
///
/// [`validate_info`]: crate::objective::Objective::validate_info
fn validate_dataset(
    objective: &dyn crate::objective::Objective,
    data: &DMatrix,
    n_targets: usize,
    n_features: usize,
    n_out: usize,
    name: &str,
) -> Result<()> {
    match data.labels() {
        None if objective.requires_labels() => {
            return Err(HessboostError::invalid_param(
                "evals",
                format!("dataset `{name}` has no labels"),
            ));
        }
        None => {}
        Some(labels) => {
            let expected = data.n_rows().checked_mul(n_targets).ok_or_else(|| {
                HessboostError::invalid_param("labels", "expected length overflows usize")
            })?;
            if labels.len() != expected {
                return Err(HessboostError::DimensionMismatch {
                    what: "labels length (n_rows * training n_targets)",
                    expected,
                    got: labels.len(),
                });
            }
        }
    }
    if data.n_cols() != n_features {
        return Err(HessboostError::DimensionMismatch {
            what: "dataset feature count",
            expected: n_features,
            got: data.n_cols(),
        });
    }
    if let Some(margin) = data.base_margin() {
        let expected = data.n_rows().checked_mul(n_out).ok_or_else(|| {
            HessboostError::invalid_param("base_margin", "expected length overflows usize")
        })?;
        if margin.len() != data.n_rows() && margin.len() != expected {
            return Err(HessboostError::DimensionMismatch {
                what: "base_margin length",
                expected,
                got: margin.len(),
            });
        }
    }
    objective
        .validate_info(&data.info())
        .map_err(|error| name_dataset(error, name))
}

/// Name the offending dataset in a [`validate_info`] error: the first
/// "dataset" in the reason becomes ``dataset `name` `` (reasons without that
/// word get a ``dataset `name`: `` prefix).
///
/// [`validate_info`]: crate::objective::Objective::validate_info
fn name_dataset(error: HessboostError, dataset: &str) -> HessboostError {
    match error {
        HessboostError::InvalidParameter { name, reason } => {
            let named = format!("dataset `{dataset}`");
            let reason = if reason.contains("dataset") {
                reason.replacen("dataset", &named, 1)
            } else {
                format!("{named}: {reason}")
            };
            HessboostError::InvalidParameter { name, reason }
        }
        other => other,
    }
}

/// Build one tree's column sampler, seeded from `rng`: the `colsample_bytree`
/// pool, then the `bylevel`/`bynode` draws, weighted by the training matrix's
/// feature weights when it has them.
fn make_column_sampler(
    n_features: usize,
    feature_weights: Option<&[f32]>,
    params: &TrainingParams,
    rng: &mut StdRng,
) -> ColumnSampler {
    ColumnSampler::new(
        n_features,
        feature_weights,
        params.colsample_bytree,
        params.colsample_bylevel,
        params.colsample_bynode,
        rng.random::<u64>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Metric, Rmse};

    /// A learnable 1-D step function: y = 0 for x<0.5, y = 1 for x>=0.5.
    fn step_dataset(n: usize) -> DMatrix {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let xi = i as f32 / n as f32;
            x.push(xi);
            y.push(if xi >= 0.5 { 1.0 } else { 0.0 });
        }
        DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap()
    }

    #[test]
    fn parallel_margin_updates_preserve_output_columns() {
        let n = 4103;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                if i % 11 == 0 {
                    f32::NAN
                } else {
                    (i % 7) as f32
                }
            })
            .collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let mut tree = RegTree::with_root(n as f32);
        tree.expand(0, 0, 3.0, true, -0.25, 1.0, 0.75, 1.0);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for outputs in [1, 3] {
            let mut actual: Vec<f32> = (0..n * outputs).map(|i| i as f32 / 100.0).collect();
            let mut expected = actual.clone();
            for output in 0..outputs {
                for row in 0..n {
                    expected[row * outputs + output] += tree.predict_row(&data, row);
                }
                pool.install(|| update_tree_margins(&tree, &data, &mut actual, outputs, output));
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn regression_reduces_training_error() {
        let d = step_dataset(100);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 50).unwrap();
        assert_eq!(model.num_trees(), 50);

        let preds = model.predict(&d).unwrap();
        let rmse = Rmse.eval(&preds, d.labels().unwrap(), None);
        // The step is exactly representable; boosting should nearly fit it.
        assert!(rmse < 0.05, "rmse too high: {rmse}");
    }

    #[test]
    fn binary_logistic_separates_classes() {
        let d = step_dataset(100);
        let params = TrainingParams::builder()
            .objective("binary:logistic")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 50).unwrap();
        let preds = model.predict(&d).unwrap(); // probabilities
        // Low-x rows -> ~0, high-x rows -> ~1.
        assert!(preds[0] < 0.1, "expected ~0, got {}", preds[0]);
        assert!(preds[99] > 0.9, "expected ~1, got {}", preds[99]);
    }

    #[test]
    fn base_score_only_model_predicts_mean() {
        // Zero rounds -> prediction is just the base score (label mean).
        let d = step_dataset(10);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .build()
            .unwrap();
        let model = train(&params, &d, 0).unwrap();
        let preds = model.predict(&d).unwrap();
        let mean = d.labels().unwrap().iter().sum::<f32>() / 10.0;
        for p in preds {
            assert!((p - mean).abs() < 1e-6);
        }
    }

    #[test]
    fn hist_and_exact_reach_similar_accuracy() {
        let d = step_dataset(120);
        let mk = |method: crate::config::TreeMethod| {
            let params = TrainingParams::builder()
                .objective("reg:squarederror")
                .tree_method(method)
                .max_depth(3)
                .eta(0.3)
                .build()
                .unwrap();
            let model = train(&params, &d, 60).unwrap();
            let preds = model.predict(&d).unwrap();
            Rmse.eval(&preds, d.labels().unwrap(), None)
        };
        let rmse_hist = mk(crate::config::TreeMethod::Hist);
        let rmse_exact = mk(crate::config::TreeMethod::Exact);
        assert!(rmse_hist < 0.05, "hist rmse {rmse_hist}");
        assert!(rmse_exact < 0.05, "exact rmse {rmse_exact}");
        // The two methods should land very close on this cleanly-binnable problem.
        assert!((rmse_hist - rmse_exact).abs() < 0.02);
    }

    #[test]
    fn approx_reduces_training_error() {
        let d = step_dataset(120);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(crate::config::TreeMethod::Approx)
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 50).unwrap();
        assert_eq!(model.num_trees(), 50);
        let preds = model.predict(&d).unwrap();
        let rmse = Rmse.eval(&preds, d.labels().unwrap(), None);
        assert!(rmse < 0.05, "approx rmse too high: {rmse}");
    }

    #[test]
    fn approx_and_hist_reach_similar_accuracy() {
        let d = step_dataset(120);
        let mk = |method: crate::config::TreeMethod| {
            let params = TrainingParams::builder()
                .objective("reg:squarederror")
                .tree_method(method)
                .max_depth(3)
                .eta(0.3)
                .build()
                .unwrap();
            let model = train(&params, &d, 60).unwrap();
            let preds = model.predict(&d).unwrap();
            Rmse.eval(&preds, d.labels().unwrap(), None)
        };
        let rmse_approx = mk(crate::config::TreeMethod::Approx);
        let rmse_hist = mk(crate::config::TreeMethod::Hist);
        assert!(rmse_approx < 0.05, "approx rmse {rmse_approx}");
        assert!(rmse_hist < 0.05, "hist rmse {rmse_hist}");
        // Both land close on this cleanly-binnable problem.
        assert!((rmse_approx - rmse_hist).abs() < 0.02);
    }

    #[test]
    fn lossguide_trains_end_to_end() {
        let d = step_dataset(120);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(crate::config::TreeMethod::Hist)
            .grow_policy(GrowPolicy::LossGuide)
            .max_leaves(16)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 60).unwrap();
        let preds = model.predict(&d).unwrap();
        let rmse = Rmse.eval(&preds, d.labels().unwrap(), None);
        assert!(rmse < 0.06, "lossguide rmse {rmse}");
    }

    #[test]
    fn exact_rejects_lossguide() {
        let d = step_dataset(20);
        let params = TrainingParams::builder()
            .tree_method(crate::config::TreeMethod::Exact)
            .grow_policy(GrowPolicy::LossGuide)
            .max_leaves(8)
            .build()
            .unwrap();
        assert!(train(&params, &d, 5).is_err());
    }

    #[test]
    fn multiclass_softprob_learns_three_classes() {
        // 1-D feature partitioned into 3 regions -> 3 classes.
        let n = 150;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let xi = i as f32 / n as f32; // 0..1
            x.push(xi);
            y.push(if xi < 0.33 {
                0.0
            } else if xi < 0.66 {
                1.0
            } else {
                2.0
            });
        }
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("multi:softprob")
            .num_class(3)
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 60).unwrap();
        assert_eq!(model.n_outputs(), 3);
        assert_eq!(model.num_trees(), 180); // 60 rounds * 3 classes
        assert_eq!(model.num_boost_rounds(), 60);

        // Probabilities: shape n*3, each row sums to 1.
        let probs = model.predict(&d).unwrap();
        assert_eq!(probs.len(), n * 3);
        for i in 0..n {
            let s: f32 = probs[i * 3..i * 3 + 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-4);
        }

        // Predicted classes match the region labels on almost all rows.
        let classes = model.predict_class(&d).unwrap();
        let correct = classes
            .iter()
            .zip(&y)
            .filter(|(c, l)| **c == **l as u32)
            .count();
        assert!(correct as f32 / n as f32 > 0.95, "accuracy {correct}/{n}");
    }

    #[test]
    fn poisson_trains_and_predicts_positive_rates() {
        let n = 200;
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut s: u64 = 7;
        let mut rng = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as f32) / (1u32 << 31) as f32
        };
        for _ in 0..n {
            let xi = rng();
            x.push(xi);
            // rate increases with x; sample a rough count.
            let rate = 1.0 + 5.0 * xi;
            y.push(rate.round());
        }
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("count:poisson")
            .max_depth(3)
            .eta(0.2)
            .build()
            .unwrap();
        let model = train(&params, &d, 60).unwrap();
        let preds = model.predict(&d).unwrap(); // rates (exp transform)
        assert!(preds.iter().all(|&p| p > 0.0), "rates must be positive");
        // Higher x should predict a higher rate: compare mean predicted rate for
        // low-x vs high-x rows (the feature is randomized, so bucket by value).
        let (mut lo_sum, mut lo_n, mut hi_sum, mut hi_n) = (0.0f32, 0, 0.0f32, 0);
        for i in 0..n {
            if x[i] < 0.5 {
                lo_sum += preds[i];
                lo_n += 1;
            } else {
                hi_sum += preds[i];
                hi_n += 1;
            }
        }
        let lo_mean = lo_sum / lo_n as f32;
        let hi_mean = hi_sum / hi_n as f32;
        assert!(
            lo_mean < hi_mean,
            "rate should rise with x: {lo_mean} vs {hi_mean}"
        );
    }

    #[test]
    fn custom_objective_matches_builtin_squared_error() {
        use crate::objective::{CustomObjective, GradPair};
        let d = step_dataset(80);

        let builtin = {
            let p = TrainingParams::builder()
                .objective("reg:squarederror")
                .max_depth(3)
                .eta(0.3)
                .base_score(0.0)
                .build()
                .unwrap();
            train(&p, &d, 30).unwrap().predict(&d).unwrap()
        };

        let custom = {
            let p = TrainingParams::builder()
                .objective("custom")
                .max_depth(3)
                .eta(0.3)
                .build()
                .unwrap();
            let obj = CustomObjective::new("custom", 1, 0.0, "rmse", |preds, labels, w, out| {
                for i in 0..preds.len() {
                    let wi = w.map_or(1.0, |ws| ws[i]);
                    out[i] = GradPair::new((preds[i] - labels[i]) * wi, wi);
                }
            });
            train_with_objective(&p, &d, 30, &obj)
                .unwrap()
                .predict(&d)
                .unwrap()
        };

        for (a, b) in builtin.iter().zip(&custom) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn custom_multi_output_objective_trains_with_stride_and_round_trips() {
        use crate::objective::{CustomObjective, GradPair};
        let d = step_dataset(80);
        let n = d.n_rows();
        let rounds = 5usize;

        // Two outputs, `[row][output]` layout: output 0 fits the label, output 1
        // fits its negation. Same data with mirrored targets, so the learned
        // outputs must mirror each other.
        let obj = CustomObjective::new("custom:two", 2, 0.0, "rmse", |preds, labels, w, out| {
            for i in 0..labels.len() {
                let wi = w.map_or(1.0, |ws| ws[i]);
                out[2 * i] = GradPair::new((preds[2 * i] - labels[i]) * wi, wi);
                out[2 * i + 1] = GradPair::new((preds[2 * i + 1] + labels[i]) * wi, wi);
            }
        });
        let p = TrainingParams::builder().max_depth(2).build().unwrap();
        let model = train_with_objective(&p, &d, rounds, &obj).unwrap();

        assert_eq!(model.n_outputs(), 2);
        assert_eq!(model.base_scores().len(), 2);
        assert_eq!(model.num_trees(), 2 * rounds);

        let margin = model.predict_margin(&d).unwrap();
        assert_eq!(margin.len(), 2 * n);
        // Unknown objective name: `predict` falls back to raw margins.
        assert_eq!(model.predict(&d).unwrap(), margin);

        let labels = d.labels().unwrap();
        let (mut err0, mut err1, mut err_init) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let (o0, o1) = (margin[2 * i], margin[2 * i + 1]);
            assert!((o0 + o1).abs() < 1e-5, "row {i}: {o0} vs {o1} not mirrored");
            err0 += (o0 - labels[i]).abs();
            err1 += (o1 + labels[i]).abs();
            err_init += labels[i].abs(); // initial margin is 0.0
        }
        assert!(
            err0 < 0.5 * err_init,
            "output 0 did not learn: {err0} vs {err_init}"
        );
        assert!(
            err1 < 0.5 * err_init,
            "output 1 did not learn: {err1} vs {err_init}"
        );

        let via_json = BoostedModel::from_json(&model.to_json().unwrap()).unwrap();
        assert_eq!(via_json.n_outputs(), 2);
        assert_eq!(via_json.predict_margin(&d).unwrap(), margin);
        let via_bytes = BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
        assert_eq!(via_bytes.n_outputs(), 2);
        assert_eq!(via_bytes.predict_margin(&d).unwrap(), margin);
    }

    #[test]
    fn ranking_ndcg_improves_over_rounds() {
        use crate::metric::Ndcg;
        // Query groups whose single feature is correlated with relevance, so a
        // ranker can learn to order documents. Docs are laid out in ascending
        // relevance (the worst initial order given zero starting margins).
        let n_groups = 30usize;
        let per = 6usize;
        let n = n_groups * per;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let sizes = vec![per; n_groups];
        let mut s: u64 = 42;
        let mut rng = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as f32) / (1u32 << 31) as f32
        };
        for _ in 0..n_groups {
            for d in 0..per {
                let rel = d as f32; // relevance grade 0..per-1
                let noise = (rng() - 0.5) * 0.8;
                x.push(rel + noise);
                y.push(rel);
            }
        }
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap()
            .with_group_sizes(&sizes)
            .unwrap();

        let params = TrainingParams::builder()
            .objective("rank:ndcg")
            .max_depth(3)
            .eta(0.2)
            .build()
            .unwrap();
        let res = train_with_eval(&params, &d, 40, &[(&d, "train")], None).unwrap();
        assert!(!res.history.is_empty());

        let ndcg_of = |r: &RoundEval| r.scores.iter().find(|(_, m, _)| m == "ndcg").unwrap().2;
        let first = ndcg_of(&res.history[0]);
        let last = ndcg_of(res.history.last().unwrap());

        // Baseline NDCG of the untrained (all-equal-score) ranking.
        let base = Ndcg::new(None).eval_grouped(&vec![0.0; n], &y, None, d.group());
        assert!(last >= first - 1e-9, "ndcg regressed: {first} -> {last}");
        assert!(
            last > base + 1e-3,
            "training should beat the untrained baseline: {base} -> {last}"
        );
        assert!(last > 0.9, "final ndcg should be high, got {last}");
    }

    #[test]
    fn dart_trains_reduces_error_and_roundtrips() {
        use crate::config::BoosterKind;
        let d = step_dataset(120);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .booster(BoosterKind::Dart)
            .rate_drop(0.1)
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 60).unwrap();
        assert_eq!(model.num_trees(), 60);

        // It should learn the step: RMSE well below a constant predictor.
        let preds = model.predict(&d).unwrap();
        let rmse = Rmse.eval(&preds, d.labels().unwrap(), None);
        assert!(rmse < 0.1, "dart rmse too high: {rmse}");

        // Native serde round-trip preserves predictions (weights included).
        let bytes = model.to_bytes().unwrap();
        let restored = BoostedModel::from_bytes(&bytes).unwrap();
        let after = restored.predict(&d).unwrap();
        for (a, b) in preds.iter().zip(&after) {
            assert!((a - b).abs() < 1e-6, "roundtrip mismatch {a} vs {b}");
        }

        // JSON round-trip too.
        let json = model.to_json().unwrap();
        let rj = BoostedModel::from_json(&json).unwrap();
        let aj = rj.predict(&d).unwrap();
        for (a, b) in preds.iter().zip(&aj) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn gbtree_unchanged_by_weight_field() {
        // A default gbtree model carries all-1.0 weights, so predictions must be
        // bit-for-bit the unweighted tree sum.
        let d = step_dataset(100);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 40).unwrap();
        // Compare weighted prediction against a manual unit-weight tree sum.
        let preds = model.predict_margin(&d).unwrap();
        let n = d.n_rows();
        let mut manual = vec![model.base_score(); n];
        for tree in model.trees() {
            for (row, m) in manual.iter_mut().enumerate() {
                *m += tree.predict_row(&d, row);
            }
        }
        for (a, b) in preds.iter().zip(&manual) {
            assert_eq!(*a, *b, "gbtree weighting changed the sum");
        }
    }

    #[test]
    fn categorical_split_beats_numeric_on_non_ordinal_pattern() {
        use crate::data::FeatureType;
        // One feature, 4 categories. Label is a NON-ordinal function of the
        // category: {0,2} -> 0, {1,3} -> 1. A single numeric `x < t` threshold
        // cannot separate {0,2} from {1,3}; a categorical set-split can.
        let cats = [0.0f32, 1.0, 2.0, 3.0];
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..40 {
            for &c in &cats {
                x.push(c);
                y.push(if (c as u32) % 2 == 1 { 1.0 } else { 0.0 });
            }
        }
        let n = x.len();
        let numeric = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let categorical = numeric
            .clone()
            .with_feature_types(&[FeatureType::Categorical])
            .unwrap();

        // Depth-1 stumps: the numeric model can only threshold, the categorical
        // model can partition the category set in a single node.
        let mk = |d: &DMatrix| {
            let p = TrainingParams::builder()
                .objective("reg:squarederror")
                .max_depth(1)
                .eta(0.3)
                .build()
                .unwrap();
            let m = train(&p, d, 40).unwrap();
            Rmse.eval(&m.predict(d).unwrap(), d.labels().unwrap(), None)
        };
        let rmse_num = mk(&numeric);
        let rmse_cat = mk(&categorical);

        // Categorical nearly fits the pattern; numeric is left far behind.
        assert!(rmse_cat < 0.02, "categorical rmse too high: {rmse_cat}");
        assert!(rmse_num > 0.05, "numeric unexpectedly fit it: {rmse_num}");
        assert!(
            rmse_cat < rmse_num,
            "categorical ({rmse_cat}) should beat numeric ({rmse_num})"
        );
    }

    #[test]
    fn exact_categorical_beats_numeric_on_non_ordinal_pattern() {
        use crate::data::FeatureType;
        // Same non-ordinal pattern as the hist test, but forcing tree_method=exact.
        let cats = [0.0f32, 1.0, 2.0, 3.0];
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..40 {
            for &c in &cats {
                x.push(c);
                y.push(if (c as u32) % 2 == 1 { 1.0 } else { 0.0 });
            }
        }
        let n = x.len();
        let numeric = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let categorical = numeric
            .clone()
            .with_feature_types(&[FeatureType::Categorical])
            .unwrap();

        let mk = |d: &DMatrix| {
            let p = TrainingParams::builder()
                .objective("reg:squarederror")
                .tree_method(crate::config::TreeMethod::Exact)
                .max_depth(1)
                .eta(0.3)
                .build()
                .unwrap();
            let m = train(&p, d, 40).unwrap();
            Rmse.eval(&m.predict(d).unwrap(), d.labels().unwrap(), None)
        };
        let rmse_num = mk(&numeric);
        let rmse_cat = mk(&categorical);
        assert!(
            rmse_cat < 0.02,
            "exact categorical rmse too high: {rmse_cat}"
        );
        assert!(rmse_num > 0.05, "numeric unexpectedly fit it: {rmse_num}");
        assert!(
            rmse_cat < rmse_num,
            "exact categorical ({rmse_cat}) should beat numeric ({rmse_num})"
        );
    }

    #[test]
    fn exact_monotone_increasing_predictions_nondecreasing() {
        use crate::config::Monotone;
        // A V-shaped target: the unconstrained fit dips then rises. Under an
        // increasing constraint with tree_method=exact, predictions must be
        // non-decreasing in the feature.
        let n = 80;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let xi = i as f32 / n as f32;
            x.push(xi);
            y.push((xi - 0.5).abs());
        }
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(crate::config::TreeMethod::Exact)
            .max_depth(4)
            .eta(0.3)
            .monotone_constraints(vec![Monotone::Increasing])
            .build()
            .unwrap();
        let model = train(&params, &d, 50).unwrap();
        let preds = model.predict(&d).unwrap();
        let mut prev = f32::NEG_INFINITY;
        for (i, p) in preds.iter().enumerate() {
            assert!(
                *p >= prev - 1e-4,
                "monotonicity violated at row {i}: {p} < {prev}"
            );
            prev = *p;
        }
    }

    #[test]
    fn base_margin_is_the_initial_margin() {
        // With 0 rounds and a per-instance base margin, the raw margin
        // prediction must equal that base margin exactly.
        let d = step_dataset(50);
        let bm: Vec<f32> = (0..50).map(|i| i as f32 * 0.01 - 0.25).collect();
        let d_bm = d.with_base_margin(&bm).unwrap();
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .build()
            .unwrap();
        let model = train(&params, &d_bm, 0).unwrap();
        let margin = model.predict_margin(&d_bm).unwrap();
        for (m, b) in margin.iter().zip(&bm) {
            assert!((m - b).abs() < 1e-6, "{m} vs {b}");
        }
    }

    #[test]
    fn base_margin_affects_training() {
        let d = step_dataset(60);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let plain = train(&params, &d, 10).unwrap().predict_margin(&d).unwrap();

        let bm = vec![2.0f32; 60];
        let d_bm = d.with_base_margin(&bm).unwrap();
        let shifted = train(&params, &d_bm, 10)
            .unwrap()
            .predict_margin(&d_bm)
            .unwrap();

        // A nonzero starting margin changes the fitted margins.
        assert!(
            plain
                .iter()
                .zip(&shifted)
                .any(|(a, b)| (a - b).abs() > 1e-4)
        );
    }

    #[test]
    fn colsample_bynode_changes_the_model() {
        // Multi-feature dataset so column sampling has features to drop.
        let (n, f) = (400usize, 8usize);
        let mut x = vec![0f32; n * f];
        let mut y = vec![0f32; n];
        let mut s: u64 = 3;
        let mut rng = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as f32) / (1u32 << 31) as f32
        };
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..f {
                let v = rng();
                x[i * f + j] = v;
                acc += v * (j as f32 + 1.0);
            }
            y[i] = acc;
        }
        let d = DMatrix::from_dense(&x, n, f)
            .unwrap()
            .with_labels(&y)
            .unwrap();

        let train_with = |bynode: f64| {
            let p = TrainingParams::builder()
                .objective("reg:squarederror")
                .max_depth(4)
                .eta(0.3)
                .colsample_bynode(bynode)
                .seed(1)
                .build()
                .unwrap();
            train(&p, &d, 20).unwrap().predict(&d).unwrap()
        };
        let full = train_with(1.0);
        let sampled = train_with(0.5);
        // With per-node sampling active, the fitted model must differ.
        let differs = full.iter().zip(&sampled).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "colsample_bynode had no effect on the model");
    }

    /// Build a 3-feature dataset for the linear target y = 2*x0 - 3*x1 + 0.5*x2
    /// with a little deterministic noise.
    fn linear_dataset(n: usize) -> DMatrix {
        let f = 3usize;
        let mut x = vec![0f32; n * f];
        let mut y = vec![0f32; n];
        let mut s: u64 = 11;
        let mut rng = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as f32) / (1u32 << 31) as f32
        };
        for i in 0..n {
            let x0 = rng();
            let x1 = rng();
            let x2 = rng();
            x[i * f] = x0;
            x[i * f + 1] = x1;
            x[i * f + 2] = x2;
            let noise = (rng() - 0.5) * 0.02;
            y[i] = 2.0 * x0 - 3.0 * x1 + 0.5 * x2 + noise;
        }
        DMatrix::from_dense(&x, n, f)
            .unwrap()
            .with_labels(&y)
            .unwrap()
    }

    #[test]
    fn gblinear_fits_linear_target() {
        use crate::config::BoosterKind;
        let d = linear_dataset(400);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .booster(BoosterKind::GbLinear)
            .eta(0.5)
            .base_score(0.0)
            .build()
            .unwrap();
        let model = train(&params, &d, 200).unwrap();
        assert_eq!(model.num_trees(), 0);

        let preds = model.predict(&d).unwrap();
        let y = d.labels().unwrap();
        let rmse = Rmse.eval(&preds, y, None);

        // Baseline: predicting the label mean.
        let mean = y.iter().sum::<f32>() / y.len() as f32;
        let mean_preds = vec![mean; y.len()];
        let rmse_mean = Rmse.eval(&mean_preds, y, None);

        assert!(rmse < 0.1, "gblinear rmse too high: {rmse}");
        assert!(
            rmse < rmse_mean * 0.25,
            "gblinear ({rmse}) should be far below the mean predictor ({rmse_mean})"
        );
    }

    #[test]
    fn gblinear_roundtrips() {
        use crate::config::BoosterKind;
        let d = linear_dataset(200);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .booster(BoosterKind::GbLinear)
            .eta(0.5)
            .build()
            .unwrap();
        let model = train(&params, &d, 100).unwrap();
        let before = model.predict(&d).unwrap();

        // Binary serde round-trip.
        let bytes = model.to_bytes().unwrap();
        let restored = BoostedModel::from_bytes(&bytes).unwrap();
        let after = restored.predict(&d).unwrap();
        for (a, b) in before.iter().zip(&after) {
            assert!((a - b).abs() < 1e-6, "binary roundtrip mismatch {a} vs {b}");
        }

        // JSON round-trip.
        let json = model.to_json().unwrap();
        let rj = BoostedModel::from_json(&json).unwrap();
        let aj = rj.predict(&d).unwrap();
        for (a, b) in before.iter().zip(&aj) {
            assert!((a - b).abs() < 1e-6, "json roundtrip mismatch {a} vs {b}");
        }
    }

    #[test]
    fn custom_metric_matches_builtin_rmse_early_stopping() {
        use crate::metric::CustomMetric;
        let d = step_dataset(80);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();

        // Builtin path: default `rmse` metric drives early stopping.
        let builtin = train_with_eval(&params, &d, 200, &[(&d, "train")], Some(5)).unwrap();

        // Custom path: a CustomMetric reimplementing RMSE (minimize).
        let rmse_metric = CustomMetric::new("rmse", false, |preds, labels, weights| {
            let mut sq = 0.0f64;
            let mut wsum = 0.0f64;
            for i in 0..preds.len() {
                let w = weights.map_or(1.0, |ws: &[f32]| f64::from(ws[i]));
                let diff = f64::from(preds[i]) - f64::from(labels[i]);
                sq += w * diff * diff;
                wsum += w;
            }
            if wsum > 0.0 { (sq / wsum).sqrt() } else { 0.0 }
        });
        let custom = train_with_custom_metric(
            &params,
            &d,
            200,
            &[(&d, "train")],
            Some(5),
            Box::new(rmse_metric),
        )
        .unwrap();

        // Early stopping fired identically, so the two models match tree-for-tree.
        assert_eq!(
            builtin.model.num_trees(),
            custom.model.num_trees(),
            "custom-metric early stopping diverged from builtin rmse"
        );
        assert_eq!(
            builtin.model.best_iteration(),
            custom.model.best_iteration()
        );
        let pb = builtin.model.predict(&d).unwrap();
        let pc = custom.model.predict(&d).unwrap();
        for (a, b) in pb.iter().zip(&pc) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }

        // The custom metric was actually recorded in the history.
        let names: Vec<&str> = custom.history[0]
            .scores
            .iter()
            .map(|(_, m, _)| m.as_str())
            .collect();
        assert_eq!(names, vec!["rmse"]);
    }

    #[test]
    fn early_stopping_sets_best_iteration() {
        let d = step_dataset(80);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        // Use the training set as its own eval just to exercise the mechanism.
        let res = train_with_eval(&params, &d, 200, &[(&d, "train")], Some(5)).unwrap();
        // Should stop well before 200 rounds once RMSE plateaus.
        assert!(res.model.num_trees() < 200);
        assert!(res.model.best_iteration().is_some());
        assert!(!res.history.is_empty());
    }

    #[test]
    fn multiclass_softmax_returns_one_label_per_row() {
        let x = [0.0, 0.1, 0.5, 0.6, 0.9, 1.0];
        let y = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let d = DMatrix::from_dense(&x, 6, 1)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("multi:softmax")
            .num_class(3)
            .max_depth(2)
            .build()
            .unwrap();
        let model = train(&params, &d, 20).unwrap();
        let predictions = model.predict(&d).unwrap();
        assert_eq!(predictions.len(), d.n_rows());
        assert!(
            predictions
                .iter()
                .all(|value| value.fract() == 0.0 && *value < 3.0)
        );
        assert_eq!(
            model.predict_class(&d).unwrap(),
            predictions
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>()
        );
    }

    /// XGBoost's `LogisticRegression::CheckLabel` accepts any probability in
    /// `[0, 1]` for both logistic objectives, and rejects anything outside.
    #[test]
    fn logistic_objectives_accept_probability_labels() {
        let x: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
        let soft = [0.25f32, 0.75, 0.0, 1.0, 0.5, 0.9, 0.1, 0.6];
        for objective in ["reg:logistic", "binary:logistic"] {
            let params = TrainingParams::builder()
                .objective(objective)
                .max_depth(2)
                .build()
                .unwrap();
            let d = DMatrix::from_dense(&x, 8, 1)
                .unwrap()
                .with_labels(&soft)
                .unwrap();
            let model = train(&params, &d, 3).unwrap();
            assert_eq!(model.objective(), objective);
            assert!(
                model
                    .predict(&d)
                    .unwrap()
                    .iter()
                    .all(|p| (0.0..=1.0).contains(p))
            );
            for bad in [1.5f32, -0.1] {
                let mut labels = soft;
                labels[0] = bad;
                let d = DMatrix::from_dense(&x, 8, 1)
                    .unwrap()
                    .with_labels(&labels)
                    .unwrap();
                assert!(
                    matches!(
                        train(&params, &d, 3),
                        Err(HessboostError::InvalidParameter { .. })
                    ),
                    "{objective} should reject label {bad}"
                );
            }
        }
    }

    #[test]
    fn count_base_score_is_in_reported_space() {
        let d = DMatrix::from_dense(&[0.0, 1.0], 2, 1)
            .unwrap()
            .with_labels(&[1.0, 2.0])
            .unwrap();
        let params = TrainingParams::builder()
            .objective("count:poisson")
            .base_score(0.5)
            .build()
            .unwrap();
        let model = train(&params, &d, 0).unwrap();
        assert!(
            model
                .predict(&d)
                .unwrap()
                .iter()
                .all(|prediction| (*prediction - 0.5).abs() < 1e-6)
        );
    }

    #[test]
    fn invalid_training_and_evaluation_inputs_return_errors() {
        let d = step_dataset(20);
        let params = TrainingParams::builder()
            .objective("binary:logistic")
            .build()
            .unwrap();
        assert!(train_with_eval(&params, &d, 2, &[], Some(1)).is_err());
        assert!(train_with_eval(&params, &d, 2, &[(&d, "eval")], Some(0)).is_err());

        let unlabeled = DMatrix::from_dense(&[0.0, 1.0], 2, 1).unwrap();
        assert!(train_with_eval(&params, &d, 2, &[(&unlabeled, "eval")], None).is_err());
        let wrong_features = DMatrix::from_dense(&[0.0, 0.0], 1, 2)
            .unwrap()
            .with_labels(&[0.0])
            .unwrap();
        assert!(train_with_eval(&params, &d, 2, &[(&wrong_features, "eval")], None).is_err());
        let model = train(&params, &d, 2).unwrap();
        assert!(model.predict(&wrong_features).is_err());
        assert!(model.predict_margin(&wrong_features).is_err());
        assert!(model.predict_leaf(&wrong_features).is_err());
    }

    /// Label-domain checks run through `Objective::validate_info` for every
    /// eval set, and the error names the offending dataset.
    #[test]
    fn eval_set_label_domain_errors_name_the_dataset() {
        let d = step_dataset(20);
        let params = TrainingParams::builder()
            .objective("binary:logistic")
            .build()
            .unwrap();
        let holdout = DMatrix::from_dense(&[0.0, 1.0], 2, 1)
            .unwrap()
            .with_labels(&[0.0, 2.0])
            .unwrap();
        match train_with_eval(&params, &d, 2, &[(&holdout, "holdout")], None) {
            Err(HessboostError::InvalidParameter { name, reason }) => {
                assert_eq!(name, "labels");
                assert_eq!(
                    reason,
                    "dataset `holdout` has labels outside the objective's valid domain"
                );
            }
            other => panic!("expected a label-domain error, got {other:?}"),
        }
    }

    /// Settings the training loop does not act on are refused rather than
    /// silently ignored.
    #[test]
    fn unimplemented_settings_are_rejected() {
        let d = step_dataset(8);
        let base = || TrainingParams::builder();
        let cases = [
            (
                "num_parallel_tree",
                base().num_parallel_tree(2).build_unchecked(),
            ),
            (
                "multi_strategy",
                base()
                    .multi_strategy(MultiStrategy::MultiOutputTree)
                    .build_unchecked(),
            ),
            (
                "process_type",
                base().process_type(ProcessType::Update).build_unchecked(),
            ),
        ];
        for (param, params) in cases {
            assert!(
                matches!(
                    train(&params, &d, 1),
                    Err(HessboostError::InvalidParameter { name, .. }) if name == param
                ),
                "{param}"
            );
        }
    }

    /// Label matrices reach only the objectives and metrics that model them,
    /// and every eval set must carry as many label columns as the training
    /// matrix.
    #[test]
    fn target_count_mismatches_are_rejected() {
        let d = step_dataset(4);
        let params = TrainingParams::default();
        let x: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let two_targets = DMatrix::from_dense(&x, 4, 1)
            .unwrap()
            .with_label_matrix(&[1.0; 8], 2)
            .unwrap();
        let poisson = TrainingParams::builder()
            .objective("count:poisson")
            .build()
            .unwrap();
        assert!(matches!(
            train(&poisson, &two_targets, 1),
            Err(HessboostError::InvalidParameter { name, .. }) if name == "labels"
        ));
        assert!(matches!(
            train_with_eval(&params, &d, 1, &[(&two_targets, "eval")], None),
            Err(HessboostError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            train_with_eval(&params, &two_targets, 1, &[(&d, "eval")], None),
            Err(HessboostError::DimensionMismatch { .. })
        ));
        let ndcg = TrainingParams::builder()
            .eval_metric("ndcg")
            .build()
            .unwrap();
        assert!(matches!(
            train(&ndcg, &two_targets, 1),
            Err(HessboostError::InvalidParameter { name, .. }) if name == "eval_metric"
        ));
        // Three margins per row fit neither one per row nor one per target.
        let bad_margin = two_targets.clone().with_base_margin(&[0.0; 12]).unwrap();
        assert!(matches!(
            train(&params, &bad_margin, 1),
            Err(HessboostError::DimensionMismatch { .. })
        ));
    }

    /// 128 rows over two integer features, weighted, with a label matrix whose
    /// two columns are unrelated functions of the features (probabilities
    /// for the logistic objectives).
    fn two_target_dataset(logistic: bool) -> (DMatrix, [Vec<f32>; 2], Vec<f32>) {
        let n = 128;
        let x: Vec<f32> = (0..n)
            .flat_map(|i| [(i % 32) as f32, ((i * 7) % 11) as f32])
            .collect();
        let (a, b): (Vec<f32>, Vec<f32>) = (0..n)
            .map(|i| {
                let (x0, x1) = (x[2 * i], x[2 * i + 1]);
                if logistic {
                    (
                        f32::from(u8::from(x0 > 12.0)),
                        f32::from(u8::from(x1 < 4.0)),
                    )
                } else {
                    (x0 * 0.5 - 3.0, (x1 - 5.0).powi(2))
                }
            })
            .unzip();
        let matrix: Vec<f32> = a.iter().zip(&b).flat_map(|(&p, &q)| [p, q]).collect();
        let weights: Vec<f32> = (0..n).map(|i| 0.5 + (i % 4) as f32 * 0.5).collect();
        let d = DMatrix::from_dense(&x, n, 2)
            .unwrap()
            .with_label_matrix(&matrix, 2)
            .unwrap()
            .with_weights(&weights)
            .unwrap();
        (d, [a, b], weights)
    }

    /// With `one_output_per_tree`, output `j` of a multi-target model is
    /// bit for bit the single-target model trained on label column `j`
    /// (same trees, same per-target intercept), for every tree method and
    /// multi-target objective.
    #[test]
    fn multi_target_outputs_equal_per_column_models() {
        for (objective, logistic) in [
            ("reg:squarederror", false),
            ("reg:pseudohubererror", false),
            ("binary:logistic", true),
            ("reg:logistic", true),
        ] {
            for method in [TreeMethod::Hist, TreeMethod::Exact, TreeMethod::Approx] {
                let (d, cols, weights) = two_target_dataset(logistic);
                let params = TrainingParams::builder()
                    .objective(objective)
                    .tree_method(method)
                    .max_depth(3)
                    .build()
                    .unwrap();
                let model = train(&params, &d, 4).unwrap();
                assert_eq!((model.n_outputs(), model.n_targets()), (2, 2));
                let preds = model.predict(&d).unwrap();
                assert_eq!(preds.len(), 2 * d.n_rows());
                for (j, col) in cols.iter().enumerate() {
                    let single = d
                        .clone()
                        .with_labels(col)
                        .unwrap()
                        .with_weights(&weights)
                        .unwrap();
                    let reference = train(&params, &single, 4).unwrap();
                    assert_eq!(
                        model.base_scores()[j].to_bits(),
                        reference.base_score().to_bits(),
                        "{objective} {method:?} intercept {j}"
                    );
                    let expected = reference.predict(&single).unwrap();
                    for (row, e) in expected.iter().enumerate() {
                        assert_eq!(
                            preds[row * 2 + j].to_bits(),
                            e.to_bits(),
                            "{objective} {method:?} ({row},{j})"
                        );
                    }
                }
            }
        }
    }

    /// A multi-target model round-trips through the native and XGBoost
    /// formats with its target count, and `predict_class` thresholds each
    /// label independently.
    #[test]
    fn multi_label_model_round_trips_and_classifies_per_label() {
        let (d, cols, _) = two_target_dataset(true);
        let params = TrainingParams::builder()
            .objective("binary:logistic")
            .eta(0.5)
            .build()
            .unwrap();
        let model = train(&params, &d, 10).unwrap();
        let preds = model.predict(&d).unwrap();
        let classes = model.predict_class(&d).unwrap();
        assert_eq!(classes.len(), 2 * d.n_rows());
        for (i, (&c, &p)) in classes.iter().zip(&preds).enumerate() {
            assert_eq!(c, u32::from(p > 0.5), "cell {i}");
            assert_eq!(c as f32, cols[i % 2][i / 2], "separable cell {i}");
        }
        for restored in [
            BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
            BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
            BoostedModel::from_xgboost_json(&model.to_xgboost_json().unwrap()).unwrap(),
            BoostedModel::from_xgboost_ubjson(&model.to_xgboost_ubjson().unwrap()).unwrap(),
        ] {
            assert_eq!(restored.n_targets(), 2);
            assert_eq!(restored.predict(&d).unwrap(), preds);
        }
    }

    /// An objective that learns from label bounds only: gradients and the
    /// intercept come from `MetaInfo`, and no ordinary labels are required.
    struct BoundsMidpoint;

    impl BoundsMidpoint {
        fn target(info: &MetaInfo, row: usize) -> f32 {
            let lo = info.label_lower_bound.expect("validated")[row];
            let hi = info.label_upper_bound.expect("validated")[row];
            f32::midpoint(lo, hi)
        }
    }

    impl crate::objective::Objective for BoundsMidpoint {
        fn name(&self) -> &'static str {
            "test:bounds_midpoint"
        }

        fn gradient(&self, _: &[f32], _: &[f32], _: Option<&[f32]>, _: &mut [GradPair]) {
            unreachable!("training must call gradient_info");
        }

        fn gradient_info(&self, preds: &[f32], info: &MetaInfo, out: &mut [GradPair]) {
            for (row, (p, g)) in preds.iter().zip(out.iter_mut()).enumerate() {
                *g = GradPair::new(p - Self::target(info, row), 1.0);
            }
        }

        fn base_margins(
            &self,
            _: &[f32],
            _: Option<&[f32]>,
            _: Option<&crate::data::GroupInfo>,
        ) -> Vec<f32> {
            unreachable!("training must call base_margins_info");
        }

        fn base_margins_info(&self, info: &MetaInfo) -> Vec<f32> {
            let sum: f32 = (0..info.n_rows).map(|row| Self::target(info, row)).sum();
            vec![sum / info.n_rows as f32]
        }

        fn validate_info(&self, info: &MetaInfo) -> Result<()> {
            if info.label_lower_bound.is_none() || info.label_upper_bound.is_none() {
                return Err(HessboostError::invalid_param(
                    "label_lower_bound",
                    "dataset has no label bounds",
                ));
            }
            Ok(())
        }

        fn requires_labels(&self) -> bool {
            false
        }

        fn default_metric(&self) -> String {
            "rmse".to_string()
        }
    }

    #[test]
    fn training_routes_through_metadata_hooks() {
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let lower: Vec<f32> = (0..32).map(|i| if i < 16 { 0.0 } else { 4.0 }).collect();
        let upper: Vec<f32> = lower.iter().map(|lo| lo + 2.0).collect();
        let d = DMatrix::from_dense(&x, 32, 1)
            .unwrap()
            .with_label_bounds(&lower, &upper)
            .unwrap();
        let params = TrainingParams::builder()
            .eta(1.0)
            .lambda(0.0)
            .build()
            .unwrap();
        let model = train_with_objective(&params, &d, 3, &BoundsMidpoint).unwrap();
        // Base margin is the mean midpoint (1 and 5 → 3); one full-step tree
        // then lands every row on its own midpoint.
        assert_eq!(model.base_score(), 3.0);
        let preds = model.predict_margin(&d).unwrap();
        for (row, p) in preds.iter().enumerate() {
            let expected = if row < 16 { 1.0 } else { 5.0 };
            assert!((p - expected).abs() < 1e-3, "row {row}: {p}");
        }

        let unbounded = DMatrix::from_dense(&x, 32, 1).unwrap();
        match train_with_objective(&params, &unbounded, 1, &BoundsMidpoint) {
            Err(HessboostError::InvalidParameter { name, reason }) => {
                assert_eq!(name, "label_lower_bound");
                assert_eq!(reason, "dataset `dtrain` has no label bounds");
            }
            other => panic!("expected validate_info to reject, got {other:?}"),
        }
    }

    #[test]
    fn exact_interaction_constraints_confine_each_path() {
        fn visit(tree: &RegTree, node: usize, path: &mut Vec<u32>) {
            let current = tree.node(node);
            if current.is_leaf() {
                assert!(path.iter().all(|feature| *feature == path[0]));
                return;
            }
            path.push(current.split_feature);
            visit(tree, current.left as usize, path);
            visit(tree, current.right as usize, path);
            path.pop();
        }

        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..128 {
            let a = (i & 1) as f32;
            let b = ((i >> 1) & 1) as f32;
            x.extend_from_slice(&[a, b]);
            y.push(f32::from(a != b));
        }
        let d = DMatrix::from_dense(&x, 128, 2)
            .unwrap()
            .with_labels(&y)
            .unwrap();
        let params = TrainingParams::builder()
            .tree_method(TreeMethod::Exact)
            .max_depth(3)
            .interaction_constraints(vec![vec![0], vec![1]])
            .build()
            .unwrap();
        let model = train(&params, &d, 3).unwrap();
        for tree in model.trees() {
            visit(tree, 0, &mut Vec::new());
        }
    }
}
