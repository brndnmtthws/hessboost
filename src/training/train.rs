//! The gradient-boosting training loop.

use crate::config::{
    BoosterKind, Device, GrowPolicy, ObjectiveParams, ProcessType, SamplingMethod, TrainingParams,
    TreeMethod,
};
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::data::{DMatrix, MetaInfo};
use crate::error::{HessboostError, Result};
use crate::metric::{Metric, create_metrics};
use crate::model::{BoostedModel, ModelSpec, check_objective_width};
use crate::objective::{GradPair, Objective, create_objective};
use crate::rng::Rng;
use crate::training::continuation::{require_model_for_update, resume_model};
use crate::training::multi_output;
use crate::training::refresh::refresh_tree;
use crate::training::sampling::{GradientSample, gradient_based_sample};
use crate::tree::RegTree;
use crate::tree::builder::{
    ExactTreeBuilder, HistTreeBuilder, LeafRows, SortedColumns, all_rows, check_symmetric_input,
};
use crate::tree::reuse::ReuseSet;
use crate::tree::sampler::ColumnSampler;
use rayon::prelude::*;
use std::ops::ControlFlow;
use std::sync::OnceLock;

/// What every boosting round of one training run reads: the parameters, the
/// training matrix with its metadata, and the objective.
#[derive(Clone, Copy)]
pub(super) struct TrainContext<'a> {
    pub(super) params: &'a TrainingParams,
    pub(super) dtrain: &'a DMatrix,
    pub(super) info: &'a MetaInfo<'a>,
    pub(super) objective: &'a dyn Objective,
}

/// The gradients one tree grows on and the rows that take part: an output's
/// gradients and its uniform row subset, or their gradient-based sample.
#[derive(Clone, Copy)]
struct TreeSample<'a> {
    gpair: &'a [GradPair],
    rows: &'a [u32],
    /// Under `approx` with per-round cuts, the gradient index that every
    /// tree of this output's forest builds from these same gradients
    /// (`None` elsewhere): built before the parallel trees start, or by the
    /// first tree that needs it on the serial path.
    forest_index: Option<&'a OnceLock<GHistIndex>>,
}
use crate::tree::hist::{CpuBackend, HistogramBackend};

/// Prepared, reusable per-round builder state, chosen by `tree_method`.
enum Prepared {
    Exact(SortedColumns),
    /// Histogram method: the binned dataset plus the backend its histograms
    /// are built on (the CPU's, or the Metal GPU's when `device = metal`).
    Hist {
        index: GHistIndex,
        backend: Box<dyn HistogramBackend>,
        /// Every training value was sketched (no row has a zero sample
        /// weight), so the builder's row partitions equal routing each
        /// training row through the finished tree by raw value: numeric
        /// values lie below their feature's last cut, and categorical cuts
        /// hold every category. A zero-weight row's value can lie beyond the
        /// last cut, where binning clamps it into the last bin while the tree
        /// routes it by its threshold, so linear leaves then route instead of
        /// reading the builder's rows.
        rows_route_like_trees: bool,
    },
    /// `tree_method=approx`: Hessian-weighted cuts. XGBoost regenerates them
    /// every round from a sorted-column summary unless the objective has a
    /// constant Hessian, in which case the first tree's streaming sketch (of
    /// its sampled Hessians) is built once and reused (`BatchParam::regen =
    /// !const_hess`); continued training replays it
    /// ([`Prepared::resume_approx_cache`]).
    Approx {
        const_hess: bool,
        cached: OnceLock<GHistIndex>,
    },
}

impl Prepared {
    /// Grow one tree on `sample`, with the rows that reached each leaf when
    /// `capture_rows` (histogram and exact methods; empty otherwise). With reuse
    /// penalties (`reuse` is `Some`) the split search is penalized by the
    /// ensemble's dictionary, which the new tree's splits then extend.
    /// `rounding_seed` keys the stochastic rounding of quantized training.
    fn build_tree(
        &self,
        run: &TrainContext,
        sample: TreeSample,
        sampler: &mut ColumnSampler,
        reuse: Option<&mut ReuseSet>,
        rounding_seed: u64,
        capture_rows: bool,
    ) -> (RegTree, Vec<LeafRows>) {
        let TrainContext { params, dtrain, .. } = *run;
        let TreeSample {
            gpair,
            rows,
            forest_index,
        } = sample;
        let hist = |ghist: &GHistIndex,
                    backend: &dyn HistogramBackend,
                    reuse: Option<&ReuseSet>,
                    sampler: &mut ColumnSampler| {
            let builder = HistTreeBuilder::new(params)
                .with_rounding_seed(rounding_seed)
                .with_reuse(reuse, ghist.cuts())
                .with_backend(backend);
            if capture_rows {
                builder.build_with_leaf_rows(ghist, gpair, rows, sampler)
            } else {
                (builder.build(ghist, gpair, rows, sampler), Vec::new())
            }
        };
        let (tree, leaf_rows) = match self {
            Prepared::Exact(cols) => {
                let builder = ExactTreeBuilder::new(params).with_reuse(reuse.as_deref());
                if capture_rows {
                    builder.build_with_leaf_rows(cols, dtrain, gpair, rows, sampler)
                } else {
                    (
                        builder.build(cols, dtrain, gpair, rows, sampler),
                        Vec::new(),
                    )
                }
            }
            Prepared::Hist { index, backend, .. } => {
                hist(index, backend.as_ref(), reuse.as_deref(), sampler)
            }
            Prepared::Approx { const_hess, cached } => {
                let bin = || approx_index(params, dtrain, gpair, *const_hess);
                // `approx` never runs with a GPU device: `validate` refuses
                // the combination, so its histograms always use the CPU.
                let cpu = CpuBackend;
                if *const_hess {
                    hist(cached.get_or_init(bin), &cpu, reuse.as_deref(), sampler)
                } else if let Some(shared) = forest_index {
                    hist(shared.get_or_init(bin), &cpu, reuse.as_deref(), sampler)
                } else {
                    hist(&bin(), &cpu, reuse.as_deref(), sampler)
                }
            }
        };
        if let Some(reuse) = reuse {
            reuse.record_tree(&tree);
        }
        (tree, leaf_rows)
    }

    /// The per-round gradient indices of `approx` with a non-constant
    /// Hessian when a forest has several trees: one per output, built before
    /// the parallel trees start (or by the first tree that needs it on the
    /// serial path), since every tree of an output's forest
    /// weights its cuts by the same gradients (one row sample and, under
    /// gradient-based sampling, one gradient sample serve the whole forest,
    /// [`Self::samples_per_forest`]). Empty otherwise.
    fn forest_indices(&self, n_out: usize, num_parallel_tree: usize) -> Vec<OnceLock<GHistIndex>> {
        match self {
            Prepared::Approx {
                const_hess: false, ..
            } if num_parallel_tree > 1 => (0..n_out).map(|_| OnceLock::new()).collect(),
            _ => Vec::new(),
        }
    }

    /// Whether one row sample serves every parallel tree of an output: XGBoost
    /// 3.4.2's `GlobalApproxUpdater::Update` samples once before its tree loop
    /// and grows the whole forest from those gradients and sketch Hessians,
    /// whereas the hist and exact updaters sample each tree.
    fn samples_per_forest(&self) -> bool {
        matches!(self, Prepared::Approx { .. })
    }

    /// Continued training: seed the constant-Hessian `approx` cache with the
    /// cuts an uninterrupted run holds. That run cached the cuts of its first
    /// tree (iteration 0, output 0), whose Hessians gradient-based sampling
    /// zeroes or rescales, so they depend on that tree's sample; XGBoost
    /// keeps them in the training matrix's gradient-index cache, which a
    /// continuation on the same matrix reuses. Replays iteration 0 from
    /// `margin0` (the intercept margins): its gradients and its RNG draws up
    /// to the first sample. Without gradient sampling every round's constant
    /// Hessians agree, and the cache fills lazily as usual.
    fn resume_approx_cache(
        &self,
        run: &TrainContext,
        margin0: &[f32],
        gpair: &mut [GradPair],
        gpair_k: &mut [GradPair],
        n_out: usize,
    ) -> Result<()> {
        let TrainContext {
            params,
            info,
            objective,
            ..
        } = *run;
        let const_hess_approx = matches!(
            self,
            Prepared::Approx {
                const_hess: true,
                ..
            }
        );
        if !const_hess_approx || !gradient_sampling(params) {
            return Ok(());
        }
        // Iteration 0's draws before its first sample: a DART round first
        // draws its skip variate (`select_dropout` over an empty ensemble
        // draws nothing more), and `sample_rows` draws nothing under
        // gradient sampling.
        let mut rng = if params.booster == BoosterKind::Dart {
            let mut rng = round_rng(params, 0, DART_SALT);
            let _skip = rng.f64();
            rng
        } else {
            round_rng(params, 0, 0)
        };
        objective.gradient_info(margin0, info, gpair);
        let g0 = gather_output(gpair, gpair_k, n_out, 0);
        let sampled = gradient_based_sample(g0, 1, params.subsample, &mut rng)?;
        let g0 = sampled.as_ref().map_or(g0, |s| s.gpair.as_slice());
        self.fill_approx_cache(run, g0);
        Ok(())
    }

    /// Build the constant-Hessian `approx` cache from `gpair`, the gradients
    /// its first tree reads, unless it is already built (a no-op for every
    /// other builder). The first tree of a run is output 0's, so the parallel
    /// slot loop calls this with output 0's gradients before growing trees
    /// for several outputs at once: whichever tree ran first would otherwise
    /// pick the cuts, and a custom objective's constant Hessians may differ
    /// by output. XGBoost 3.4.2 likewise keeps the first gradient index its
    /// training matrix builds (`BatchParam::regen` is false), whichever
    /// output group later reads it.
    fn fill_approx_cache(&self, run: &TrainContext, gpair: &[GradPair]) {
        if let Prepared::Approx {
            const_hess: true,
            cached,
        } = self
        {
            cached.get_or_init(|| approx_index(run.params, run.dtrain, gpair, true));
        }
    }
}

/// The `approx` gradient index of one tree: cuts weighted by `gpair`'s
/// Hessians (a sorted-column summary unless the Hessian is constant).
fn approx_index(
    params: &TrainingParams,
    dtrain: &DMatrix,
    gpair: &[GradPair],
    const_hess: bool,
) -> GHistIndex {
    let cuts =
        HistCuts::from_dmatrix_hessians(dtrain, params.max_bin, |row| gpair[row].hess, !const_hess);
    GHistIndex::from_dmatrix(dtrain, cuts)
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
    if method == TreeMethod::Exact && gradient_sampling(params) {
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
            let index = GHistIndex::from_dmatrix(dtrain, cuts);
            let backend = hist_backend(params, &index)?;
            let rows_route_like_trees = dtrain.weights().is_none_or(|w| !w.contains(&0.0));
            Prepared::Hist {
                index,
                backend,
                rows_route_like_trees,
            }
        }
        TreeMethod::Approx => Prepared::Approx {
            const_hess,
            cached: OnceLock::new(),
        },
        _ => Prepared::Exact(SortedColumns::from_dmatrix(dtrain)),
    })
}

/// The histogram backend a training run builds on: the Metal GPU's when
/// `device = metal` (the parameter validation has already checked the
/// platform and feature), else the CPU's.
fn hist_backend(params: &TrainingParams, index: &GHistIndex) -> Result<Box<dyn HistogramBackend>> {
    match params.device {
        Device::Cpu => {
            let backend: Box<dyn HistogramBackend> = Box::new(CpuBackend);
            Ok(backend)
        }
        Device::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                let backend: Box<dyn HistogramBackend> =
                    Box::new(crate::backend::metal::MetalHistBackend::new(index)?);
                Ok(backend)
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                // Unreachable in practice: `TrainingParams::validate` refuses
                // `device = metal` without the feature, and training always
                // validates first.
                let _ = index;
                Err(HessboostError::invalid_param(
                    "device",
                    "`metal` requires building with the `metal` feature on macOS",
                ))
            }
        }
    }
}

/// A named evaluation dataset watched during training.
pub(crate) type EvalSet<'a> = (&'a DMatrix, &'a str);

/// One row of the evaluation history: the metric values computed at the end of
/// a boosting round.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RoundEval {
    /// The 0-based boosting iteration of the model (after continued training,
    /// counted from the start of the initial model).
    pub iteration: usize,
    /// `(dataset_name, metric_name, value)` triples.
    pub scores: Vec<(String, String, f64)>,
}

/// The result of [`Trainer::train`]: the model plus the per-round evaluation
/// history.
#[derive(Debug)]
#[non_exhaustive]
pub struct TrainResult {
    /// The trained model.
    pub model: BoostedModel,
    /// Evaluation history (empty when no eval sets were supplied). Each
    /// entry's `iteration` is the model's absolute iteration index, which
    /// after continued training starts at the initial model's
    /// [`num_boost_rounds`](BoostedModel::num_boost_rounds).
    pub history: Vec<RoundEval>,
    /// With [`early_stopping_rounds`](Trainer::early_stopping_rounds), the
    /// watched metric's value at the model's
    /// [`best_iteration`](BoostedModel::best_iteration) (XGBoost's
    /// `best_score`), whether or not training stopped early; `None` without
    /// early stopping or when no round ran.
    pub best_score: Option<f64>,
}

/// Train a model for `num_boost_round` iterations with no eval sets, early
/// stopping, or custom hooks. Shorthand for
/// `Trainer::new(params, dtrain, num_boost_round).train()?.model`; use
/// [`Trainer`] for everything else.
pub fn train(
    params: &TrainingParams,
    dtrain: &DMatrix,
    num_boost_round: usize,
) -> Result<BoostedModel> {
    Ok(Trainer::new(params, dtrain, num_boost_round).train()?.model)
}

/// Configures one training run: XGBoost's `xgb.train` with its optional
/// arguments as builder methods.
///
/// ```
/// use hessboost::prelude::*;
///
/// # fn main() -> Result<()> {
/// let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let dtrain = DMatrix::from_dense(&x[..6], 6, 1)?.with_labels(&x[..6])?;
/// let dvalid = DMatrix::from_dense(&x[6..], 2, 1)?.with_labels(&x[6..])?;
/// let params = TrainingParams::builder().max_depth(2).build()?;
///
/// let result = Trainer::new(&params, &dtrain, 100)
///     .eval(&dvalid, "valid")
///     .early_stopping_rounds(5)
///     .train()?;
/// assert!(!result.history.is_empty());
///
/// // Continue boosting from the trained model.
/// let more = Trainer::new(&params, &dtrain, 10)
///     .init_model(&result.model)
///     .train()?
///     .model;
/// assert_eq!(more.num_boost_rounds(), result.model.num_boost_rounds() + 10);
/// # Ok(())
/// # }
/// ```
pub struct Trainer<'a> {
    params: &'a TrainingParams,
    dtrain: &'a DMatrix,
    num_boost_round: usize,
    evals: Vec<EvalSet<'a>>,
    early_stopping_rounds: Option<usize>,
    objective: Option<&'a dyn Objective>,
    metric: Option<Box<dyn Metric>>,
    init_model: Option<&'a BoostedModel>,
    on_round: Option<RoundHook<'a>>,
}

/// The per-round hook of [`Trainer::on_round`].
type RoundHook<'a> = Box<dyn FnMut(&RoundEval) -> ControlFlow<()> + Send + 'a>;

impl<'a> Trainer<'a> {
    /// Train on `dtrain` for (at most) `num_boost_round` iterations with
    /// `params`, which name the objective and metrics.
    pub fn new(params: &'a TrainingParams, dtrain: &'a DMatrix, num_boost_round: usize) -> Self {
        Trainer {
            params,
            dtrain,
            num_boost_round,
            evals: Vec::new(),
            early_stopping_rounds: None,
            objective: None,
            metric: None,
            init_model: None,
            on_round: None,
        }
    }

    /// Evaluate the metrics on `data` after every round, reporting them
    /// under `name` in [`TrainResult::history`]. Call once per eval set; the
    /// order is kept.
    #[must_use]
    pub fn eval(mut self, data: &'a DMatrix, name: &'a str) -> Self {
        self.evals.push((data, name));
        self
    }

    /// Stop when the watched metric fails to improve for `rounds`
    /// consecutive rounds. As in XGBoost, the watched metric is the **last**
    /// metric of the **last** eval set. Needs at least one
    /// [`eval`](Self::eval) set and `rounds > 0`.
    ///
    /// The model's [`best_iteration`](BoostedModel::best_iteration) and
    /// [`TrainResult::best_score`] record the best round whether training
    /// stopped early or ran all `rounds` (as XGBoost's `best_iteration`
    /// does), so plain prediction uses the iterations up to the best one
    /// either way. When the metric never improves (it is NaN), the best
    /// round is this run's first.
    ///
    /// After [`init_model`](Self::init_model) the early-stopping state starts
    /// fresh; `best_iteration` and the history's iterations are absolute
    /// iteration indices of the continued model (XGBoost's `starting_round`
    /// offset).
    #[must_use]
    pub fn early_stopping_rounds(mut self, rounds: usize) -> Self {
        self.early_stopping_rounds = Some(rounds);
        self
    }

    /// Boost `objective` (the custom-objective hook, e.g. a
    /// [`CustomObjective`](crate::objective::CustomObjective)) instead of
    /// the one `params.objective` names. The model records the objective's
    /// [`name`](Objective::name); a built-in objective must match `params`'
    /// objective settings so the saved model can be loaded again.
    #[must_use]
    pub fn objective(mut self, objective: &'a dyn Objective) -> Self {
        self.objective = Some(objective);
        self
    }

    /// Report `metric` (the custom-metric hook, e.g. a
    /// [`CustomMetric`](crate::metric::CustomMetric)) instead of the metrics
    /// `eval_metric` or the objective's default would build: it is the sole
    /// metric reported for each eval set and the one driving early stopping
    /// (per its [`maximize`](Metric::maximize)).
    #[must_use]
    pub fn custom_metric(mut self, metric: Box<dyn Metric>) -> Self {
        self.metric = Some(metric);
        self
    }

    /// Continue training `model` for `num_boost_round` more iterations
    /// (XGBoost's `xgb.train(..., xgb_model=model)`).
    ///
    /// The new iterations start from `model`'s full current margins (every
    /// tree, whatever its `best_iteration`) and are appended to a copy of it.
    /// The copy keeps the model's intercepts unless `params.base_score` is
    /// set, which replaces them (as XGBoost's `set_param` does); the
    /// intercept is never re-estimated. `params` must use the model's
    /// objective, `num_class`, `num_parallel_tree`, booster family (tree or
    /// `gblinear`), feature count and label width; they otherwise drive the
    /// new iterations, including the objective's hyper-parameters, which the
    /// result records. The per-round RNG continues from the model's iteration
    /// count, so training `a` rounds and continuing for `b` grows the same
    /// trees as training `a + b` rounds with the same parameters. DART tree
    /// weights carry over and are rescaled by later dropouts. The copy's
    /// `best_iteration` is cleared.
    ///
    /// With `process_type=update` the model's trees are not extended but
    /// refreshed on `dtrain` (XGBoost's `updater=refresh`): round `i`
    /// recomputes the statistics, and with `refresh_leaf` the leaf values, of
    /// iteration `i`'s trees from the gradients of the already refreshed
    /// iterations. The result holds exactly the `num_boost_round` refreshed
    /// iterations (at most the model's count), as in XGBoost. Update mode
    /// needs a gbtree model without DART weights or linear leaves, no
    /// monotone constraints, and no feature weights on `dtrain`; settings
    /// refresh does not read (row and column sampling, symmetric growth,
    /// DART dropout, the beyond-XGBoost tree options) must keep their
    /// defaults, while XGBoost's tree-shape settings (`tree_method`,
    /// `max_depth`, `min_child_weight`, ...) are accepted.
    ///
    /// The model must be structurally valid, as every loaded model is.
    #[must_use]
    pub fn init_model(mut self, model: &'a BoostedModel) -> Self {
        self.init_model = Some(model);
        self
    }

    /// Call `hook` after every boosting round, once the round's eval sets
    /// are scored: progress reporting, custom stopping rules, or
    /// cancellation. It runs on the training thread, in round order, and
    /// sees the round as [`TrainResult::history`] records it (with empty
    /// `scores` when there are no eval sets). Returning
    /// [`ControlFlow::Break`] ends training after that round; the result
    /// keeps every completed round and is otherwise what training for that
    /// many rounds would have produced.
    ///
    /// With [`early_stopping_rounds`](Self::early_stopping_rounds), the
    /// hook also sees the round on which patience runs out, and a `Break`
    /// still records the best round so far as
    /// [`best_iteration`](BoostedModel::best_iteration) (with its
    /// [`TrainResult::best_score`]). With `process_type=update` the model
    /// holds the iterations refreshed so far. For `gblinear`, which stores
    /// no boosting iterations, [`RoundEval::iteration`] counts this run's
    /// rounds from 0.
    ///
    /// Observing never changes the model: training with a hook that always
    /// continues gives the same result as training without one.
    ///
    /// ```
    /// use hessboost::prelude::*;
    /// use std::ops::ControlFlow;
    ///
    /// # fn main() -> Result<()> {
    /// let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    /// let dtrain = DMatrix::from_dense(&x, 6, 1)?.with_labels(&x)?;
    /// let params = TrainingParams::builder().max_depth(2).build()?;
    /// let mut seen = Vec::new();
    /// let result = Trainer::new(&params, &dtrain, 100)
    ///     .on_round(|round| {
    ///         seen.push(round.iteration);
    ///         if round.iteration == 4 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    ///     })
    ///     .train()?;
    /// assert_eq!(result.model.num_boost_rounds(), 5);
    /// assert_eq!(seen, [0, 1, 2, 3, 4]);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn on_round(mut self, hook: impl FnMut(&RoundEval) -> ControlFlow<()> + Send + 'a) -> Self {
        self.on_round = Some(Box::new(hook));
        self
    }

    /// Run the configured training.
    pub fn train(self) -> Result<TrainResult> {
        let built;
        let objective = if let Some(objective) = self.objective {
            objective
        } else {
            built = create_objective(self.params, self.dtrain.n_targets())?;
            built.as_ref()
        };
        let params = self.params;
        let result = with_thread_pool(params, || train_impl(self, objective))?;
        validate_trained_model(&result.model)?;
        Ok(result)
    }
}

/// Check that a freshly trained model would load again. Arithmetic that
/// overflows `f32` (from extreme labels, weights, or margins) leaves
/// non-finite values the model formats refuse; report it when training
/// returns, not at load time.
pub(crate) fn validate_trained_model(model: &BoostedModel) -> Result<()> {
    model.validate_structure().map_err(|e| {
        let reason = match e {
            HessboostError::ModelFormat(reason) => reason,
            other => other.to_string(),
        };
        HessboostError::model_format(format!("training produced an invalid model: {reason}"))
    })
}

/// Refuse a `num_class >= 2` that differs from `objective`'s output count:
/// saved models require the two to agree, so a `num_class` the objective
/// does not use would train an unloadable model.
pub(crate) fn check_num_class(params: &TrainingParams, objective: &dyn Objective) -> Result<()> {
    let n_out = objective.n_outputs();
    if params.num_class >= 2 && params.num_class != n_out {
        return Err(HessboostError::invalid_param(
            "num_class",
            format!(
                "objective `{}` has {n_out} outputs, so num_class {} does not apply to it",
                objective.name(),
                params.num_class
            ),
        ));
    }
    Ok(())
}

/// Run `train` on a dedicated pool of `params.nthread` threads, or on the
/// global rayon pool when `nthread` is `0`.
pub(crate) fn with_thread_pool<T: Send>(
    params: &TrainingParams,
    train: impl FnOnce() -> Result<T> + Send,
) -> Result<T> {
    if params.nthread == 0 {
        return train();
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(params.nthread)
        .build()
        .map_err(|error| HessboostError::invalid_param("nthread", error.to_string()))?;
    pool.install(train)
}

/// Refuse a `missing` parameter other than the default: the sentinel belongs
/// to the training matrix.
pub(crate) fn reject_missing_param(params: &TrainingParams) -> Result<()> {
    if !params.missing.is_nan() {
        return Err(HessboostError::invalid_param(
            "missing",
            "set the sentinel when constructing DMatrix with from_dense_with_missing",
        ));
    }
    Ok(())
}

/// Refuse `dtrain`'s feature weights on a training path that samples no
/// columns (`reason` names it): they only steer the tree builders' column
/// sampling.
pub(crate) fn reject_feature_weights(dtrain: &DMatrix, reason: &'static str) -> Result<()> {
    if dtrain.feature_weights().is_some() {
        return Err(HessboostError::invalid_param("feature_weights", reason));
    }
    Ok(())
}

/// The core boosting loop, generic over single- and multi-output objectives.
///
/// Margins and gradients are laid out `[instance][output]`. Each round computes
/// all gradients, then grows `num_parallel_tree` trees per output from that
/// output's gradient slice. This is the multi-output generalization of
/// gradient boosting used by multiclass. `init_model` continues training
/// from an existing model ([`Trainer::init_model`]). `objective` is the
/// trainer's own or the one `params` name.
fn train_impl(trainer: Trainer<'_>, objective: &dyn Objective) -> Result<TrainResult> {
    let Trainer {
        params,
        dtrain,
        num_boost_round,
        evals,
        early_stopping_rounds,
        objective: _,
        metric: metric_override,
        init_model,
        mut on_round,
    } = trainer;
    let evals: &[EvalSet] = &evals;
    validate_request(
        &TrainRequest {
            params,
            dtrain,
            evals,
            early_stopping_rounds,
        },
        objective,
    )?;
    let info = dtrain.info();
    let n = dtrain.n_rows();
    let n_out = objective.n_outputs();
    let intercepts = || initial_intercepts(params, objective, &info, n_out);
    let mut model = if let Some(init) = init_model {
        resume_model(init, params, objective, dtrain, num_boost_round, intercepts)?
    } else {
        require_model_for_update(params)?;
        new_model(params, objective, dtrain, intercepts()?)
    };

    if params.booster == BoosterKind::GbLinear {
        return train_linear(
            &TrainRequest {
                params,
                dtrain,
                evals,
                early_stopping_rounds,
            },
            num_boost_round,
            objective,
            model,
            &mut |iteration| {
                on_round.as_mut().map_or(ControlFlow::Continue(()), |hook| {
                    hook(&RoundEval {
                        iteration,
                        scores: Vec::new(),
                    })
                })
            },
        );
    }

    // `process_type=update` refreshes the model's own trees (re-appended one
    // iteration per round) instead of growing new ones, so it needs no
    // builder state.
    let mut plan = if params.process_type == ProcessType::Update {
        RoundPlan::Refresh(model.take_trees())
    } else {
        RoundPlan::Grow(prepare_builder(params, dtrain, objective.const_hess())?)
    };
    // Continued training numbers its rounds after the model's iterations, so
    // the per-round RNG streams continue where the earlier run stopped.
    let start_iteration = model.num_boost_rounds();
    // Opt-in reuse penalties: the features and thresholds the ensemble already
    // uses, extended by every tree the loop grows. `None` on the default path.
    let reuse = ReuseSet::from_params(params, dtrain.n_cols(), model.trees());
    let margins = MarginCaches::new(&model, dtrain, evals);
    let mut eval_plan = EvalPlan::new(
        params,
        objective,
        metric_override,
        evals,
        dtrain.n_targets(),
    )?;

    let run = TrainContext {
        params,
        dtrain,
        info: &info,
        objective,
    };
    let mut state = RoundState {
        model,
        margins,
        gpair: vec![GradPair::default(); n * n_out],
        // Per-output gradient scratch; single-output objectives read `gpair`
        // itself ([`gather_output`]).
        gpair_k: if n_out > 1 {
            vec![GradPair::default(); n]
        } else {
            Vec::new()
        },
        reuse,
    };
    if start_iteration > 0
        && let RoundPlan::Grow(prepared) = &plan
    {
        prepared.resume_approx_cache(
            &run,
            &state.model.margin_from_trees(dtrain, 0..0),
            &mut state.gpair,
            &mut state.gpair_k,
            n_out,
        )?;
    }
    let mut history: Vec<RoundEval> = Vec::new();
    let mut stopping = early_stopping_rounds
        .map(|patience| EarlyStopping::new(patience, eval_plan.maximize(), start_iteration));
    // `multi_strategy = multi_output_tree` grows vector-leaf trees when there
    // is more than one output (a single output keeps scalar trees, as
    // XGBoost's `LeafLength` does).
    let vector_leaf = multi_output::vector_leaf(params, n_out);

    for round in 0..num_boost_round {
        let iteration = start_iteration + round;
        match &mut plan {
            RoundPlan::Grow(Prepared::Hist { index: ghist, .. }) if vector_leaf => {
                multi_output::boost_round(
                    &multi_output::VectorRound { run, ghist },
                    &mut state.model,
                    iteration,
                    &mut state.margins,
                    &mut state.gpair,
                )?;
            }
            RoundPlan::Refresh(queue) => refresh_round(&run, queue, iteration, &mut state)?,
            RoundPlan::Grow(prepared) => grow_round(&run, prepared, iteration, &mut state)?,
        }
        let mut stop = false;
        if !evals.is_empty() {
            let score = eval_plan.record(objective, iteration, &state.margins, &mut history);
            if let Some(stopping) = &mut stopping {
                stop = stopping.observe(iteration, score);
            }
        }
        if let Some(hook) = &mut on_round {
            let flow = match history.last() {
                Some(round) if round.iteration == iteration => hook(round),
                _ => hook(&RoundEval {
                    iteration,
                    scores: Vec::new(),
                }),
            };
            stop |= flow.is_break();
        }
        if stop {
            break;
        }
    }

    let mut model = state.model;
    // XGBoost records the best iteration whenever early stopping is on, not
    // only when patience runs out.
    let mut best_round_score = None;
    if let Some(stopping) = &stopping
        && let Some(first) = history.first()
    {
        let best_iter = stopping.best_round();
        let round = &history[best_iter - first.iteration];
        best_round_score = round.scores.last().map(|&(_, _, v)| v);
        model.set_best_iteration(Some(best_iter));
    }

    Ok(TrainResult {
        model,
        history,
        best_score: best_round_score,
    })
}

/// The linear (`gblinear`) booster: fit `model`'s coordinate-descent linear
/// model instead of growing trees, continuing from its weights and margins,
/// with `after_round` after each round. Eval sets and early stopping are
/// refused (the history stays empty).
fn train_linear(
    request: &TrainRequest,
    num_boost_round: usize,
    objective: &dyn Objective,
    mut model: BoostedModel,
    after_round: &mut dyn FnMut(usize) -> ControlFlow<()>,
) -> Result<TrainResult> {
    let &TrainRequest {
        params,
        dtrain,
        evals,
        early_stopping_rounds,
    } = request;
    if !evals.is_empty() || early_stopping_rounds.is_some() {
        return Err(HessboostError::invalid_param(
            "booster",
            "gblinear does not yet support evaluation sets or early stopping",
        ));
    }
    let linear = crate::training::gblinear::train_gblinear(
        params,
        dtrain,
        num_boost_round,
        model.margin_from_trees(dtrain, 0..0),
        objective,
        model.linear(),
        after_round,
    )?;
    model.set_linear(linear);
    Ok(TrainResult {
        model,
        history: Vec::new(),
        best_score: None,
    })
}

/// What [`validate_request`] checks: the training call's configuration and
/// data, before anything is built.
struct TrainRequest<'a> {
    params: &'a TrainingParams,
    dtrain: &'a DMatrix,
    evals: &'a [EvalSet<'a>],
    early_stopping_rounds: Option<usize>,
}

/// Refuse a training call that cannot run: invalid parameters, early
/// stopping without eval sets, unlabeled or mismatched data, constraints
/// naming missing features, feature weights on a path that samples no
/// columns, and objective settings a saved model could not rebuild.
fn validate_request(request: &TrainRequest, objective: &dyn Objective) -> Result<()> {
    let &TrainRequest {
        params,
        dtrain,
        evals,
        early_stopping_rounds,
    } = request;
    params.validate()?;
    multi_output::validate(params, objective.n_outputs())?;
    reject_missing_param(params)?;
    EarlyStopping::check_patience(early_stopping_rounds)?;
    if early_stopping_rounds.is_some() && evals.is_empty() {
        return Err(HessboostError::invalid_param(
            "early_stopping_rounds",
            "requires at least one evaluation dataset",
        ));
    }

    if objective.requires_labels() && dtrain.labels().is_none() {
        return Err(HessboostError::EmptyDataset("train: dtrain has no labels"));
    }
    let n_features = dtrain.n_cols();
    let n_out = objective.n_outputs();
    check_num_class(params, objective)?;
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
    if params.booster == BoosterKind::GbLinear {
        reject_feature_weights(dtrain, "gblinear does not sample columns")?;
    }
    if params.process_type == ProcessType::Update {
        reject_feature_weights(dtrain, "`process_type=update` does not sample columns")?;
    }

    BoostedModel::check_iteration_size(n_out, params.num_parallel_tree)?;
    // A built-in objective passed to `Trainer::objective` is recorded by
    // name with `params`' objective settings; refuse settings that would not
    // rebuild it, since the saved model could not be loaded again.
    let objective_params = ObjectiveParams::for_objective(params, objective.name());
    check_objective_width(
        objective.name(),
        &objective_params,
        params.num_class,
        dtrain.n_targets(),
        n_out,
    )
    .map_err(|e| {
        HessboostError::invalid_param(
            "objective",
            format!("the training parameters do not describe the given objective: {e}"),
        )
    })
}

/// What the tree-growing and refresh rounds update: the ensemble, its
/// margin caches, the gradient buffers, and the reuse dictionary.
struct RoundState<'a> {
    model: BoostedModel,
    margins: MarginCaches<'a>,
    /// Every output's gradients, `[row][n_out]`.
    gpair: Vec<GradPair>,
    /// One output's gradients gathered from `gpair` (empty for
    /// single-output objectives, which read `gpair` directly).
    gpair_k: Vec<GradPair>,
    reuse: Option<ReuseSet>,
}

/// `process_type=update`: refresh iteration `iteration`'s trees of `queue`
/// output by output, from the gradients of the already refreshed ones, and
/// re-append them.
fn refresh_round(
    run: &TrainContext,
    queue: &mut [RegTree],
    iteration: usize,
    state: &mut RoundState,
) -> Result<()> {
    let TrainContext {
        params,
        dtrain,
        info,
        objective,
    } = *run;
    let n_out = objective.n_outputs();
    let parallel = params.num_parallel_tree;
    // Gradients from the already refreshed iterations; iteration `i`'s trees
    // are then refreshed in place, output by output.
    objective.gradient_info(&state.margins.train, info, &mut state.gpair);
    multi_output::reject_split_gradient(objective, iteration, &state.gpair)?;
    let per_iteration = n_out * parallel;
    for slot in 0..per_iteration {
        let k = slot / parallel;
        let gk = gather_output(&state.gpair, &mut state.gpair_k, n_out, k);
        let mut tree = std::mem::replace(
            &mut queue[iteration * per_iteration + slot],
            RegTree::with_root(0.0),
        );
        refresh_tree(&mut tree, dtrain, gk, params, tree_eta(params));
        state.margins.add_tree(&tree, TreeOutput::Scalar(k), None);
        state.model.push_tree_weighted(tree, 1.0);
    }
    Ok(())
}

/// Grow one iteration's scalar-leaf trees (gbtree or DART) with `prepared`
/// and append them.
fn grow_round(
    run: &TrainContext,
    prepared: &Prepared,
    iteration: usize,
    state: &mut RoundState,
) -> Result<()> {
    let TrainContext {
        params,
        dtrain,
        objective,
        ..
    } = *run;
    let n = dtrain.n_rows();
    let n_out = objective.n_outputs();
    let parallel = params.num_parallel_tree;
    // 1. Gradients from the current margins (all outputs at once), DART's
    //    from the ensemble minus this round's dropout set.
    let (mut rng, dropped) = round_gradients(
        run,
        &state.model,
        iteration,
        &state.margins.train,
        &mut state.gpair,
    );
    multi_output::reject_split_gradient(objective, iteration, &state.gpair)?;
    let weight = dart_new_tree_weight(dropped.as_deref().unwrap_or_default(), params);

    // 2. Uniform row subsets, drawn before the trees and shared across the
    //    per-output fits.
    let row_subsets = iteration_row_subsets(n, params, prepared.samples_per_forest(), &mut rng);
    // An output's gradient-based sample, when its whole forest shares one.
    let mut forest_sample = None;
    let forest_indices = prepared.forest_indices(n_out, parallel);
    let grow = GrowRound {
        run,
        prepared,
        gpair: &state.gpair,
        n_out,
        iteration,
        forest_indices: &forest_indices,
    };

    // 3. `num_parallel_tree` trees per output from the same gradients,
    //    output-major like XGBoost's layout.
    let slots: Vec<TreeSlot> = (0..n_out * parallel)
        .map(|slot| {
            let row_subset = &row_subsets[(slot % parallel) % row_subsets.len()];
            // Retaining the final row partitions replaces per-row tree
            // traversals of the raw feature matrix with one sequential pass
            // per leaf: the training margin update's, when every row took
            // part, and the linear-leaf fit's. Linear leaves use them only
            // when they equal raw routing (see `rows_route_like_trees`).
            let (routed, linear_rows) = match prepared {
                Prepared::Hist {
                    rows_route_like_trees,
                    ..
                } => (true, params.linear_tree && *rows_route_like_trees),
                // The exact builder routes rows by their raw values, as
                // prediction does.
                Prepared::Exact(_) => (true, false),
                Prepared::Approx { .. } => (false, false),
            };
            let margin_rows = routed
                && dropped.is_none()
                && row_subset.len() == n
                && !gradient_sampling(params)
                && (!params.linear_tree || linear_rows);
            TreeSlot {
                output: slot / parallel,
                parallel: slot % parallel,
                rows: row_subset,
                capture_rows: margin_rows || linear_rows,
                margin_rows,
            }
        })
        .collect();
    // The trees of an iteration share the round's gradients and do not read
    // each other. Without gradient-based sampling or a reuse dictionary, each
    // tree's RNG draws are its column sampler and rounding seed, drawn here
    // in slot order as the sequential path draws them; the trees are then
    // grown in parallel. A GPU backend stages one tree's gradients at a
    // time, so it keeps the sequential path.
    let trees: Vec<(RegTree, Vec<LeafRows>)> = if slots.len() > 1
        && state.reuse.is_none()
        && !gradient_sampling(params)
        && params.device == Device::Cpu
        && rayon::current_num_threads() > 1
    {
        // The first tree's cuts, before any tree reads them.
        prepared.fill_approx_cache(
            run,
            gather_output(&state.gpair, &mut state.gpair_k, n_out, 0),
        );
        let draws: Vec<(ColumnSampler, u64)> = slots
            .iter()
            .map(|_| {
                let sampler = make_column_sampler(dtrain, params, &mut rng);
                (sampler, quantization_seed(params, &mut rng))
            })
            .collect();
        // Every output's gradients gathered once, output-major, for all of
        // its parallel trees (single-output objectives read `gpair`).
        let gathered = gather_outputs(&state.gpair, n_out);
        let output_gpair = |k: usize| {
            if n_out == 1 {
                &state.gpair[..]
            } else {
                &gathered[k * n..(k + 1) * n]
            }
        };
        // Build the forests' shared `approx` indices here, before the
        // parallel trees read them: an index built inside a tree task would
        // run its own parallel loops while other tasks wait on it, and a
        // worker waiting there can steal a task that waits on the same
        // index again.
        for (k, index) in forest_indices.iter().enumerate() {
            index.get_or_init(|| approx_index(params, dtrain, output_gpair(k), false));
        }
        slots
            .par_iter()
            .zip(draws)
            .map(|(slot, (mut sampler, rounding_seed))| {
                let gk = output_gpair(slot.output);
                let sample = TreeSample {
                    gpair: gk,
                    rows: slot.rows,
                    forest_index: grow.forest_index(slot.output),
                };
                grow_sampled_tree(&grow, slot, sample, &mut sampler, rounding_seed, None)
            })
            .collect()
    } else {
        slots
            .iter()
            .map(|slot| {
                fit_output_tree(
                    &grow,
                    slot,
                    &mut state.gpair_k,
                    &mut rng,
                    &mut forest_sample,
                    state.reuse.as_mut(),
                )
            })
            .collect::<Result<_>>()?
    };

    for (slot, (tree, leaf_rows)) in slots.iter().zip(trees) {
        // DART's gradients come from the ensemble, not the margin caches
        // (`finish_dart` recomputes the eval ones).
        if dropped.is_none() {
            // The builder's final row partitions already identify the training
            // leaves when every row took part in growing the tree.
            let captured = slot.margin_rows.then_some(leaf_rows.as_slice());
            state
                .margins
                .add_tree(&tree, TreeOutput::Scalar(slot.output), captured);
        }
        state.model.push_tree_weighted(tree, weight);
    }
    if let Some(dropped) = &dropped {
        finish_dart(&mut state.model, params, dropped, &mut state.margins);
    }
    Ok(())
}

/// The eval sets' metrics and the buffer their predictions are transformed
/// in, reused every round.
struct EvalPlan<'a> {
    evals: &'a [EvalSet<'a>],
    infos: Vec<MetaInfo<'a>>,
    metrics: Vec<Box<dyn Metric>>,
    preds: Vec<f32>,
}

impl<'a> EvalPlan<'a> {
    /// The metrics every eval set reports (`metric_override`, else the
    /// configured or default ones), refusing a metric that cannot read the
    /// label layout or the model's prediction width.
    fn new(
        params: &TrainingParams,
        objective: &dyn Objective,
        metric_override: Option<Box<dyn Metric>>,
        evals: &'a [EvalSet<'a>],
        n_targets: usize,
    ) -> Result<Self> {
        let n_out = objective.n_outputs();
        let metrics = match metric_override {
            Some(m) => vec![m],
            None => configured_metrics(params, objective)?,
        };
        if n_targets > 1
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
        let infos: Vec<MetaInfo> = evals.iter().map(|(data, _)| data.info()).collect();
        for (info, (_, name)) in infos.iter().zip(evals) {
            for metric in &metrics {
                metric
                    .validate_info(info)
                    .and_then(|()| check_prediction_width(metric.as_ref(), info, n_out))
                    .map_err(|error| name_dataset(error, name))?;
            }
        }
        Ok(EvalPlan {
            evals,
            infos,
            metrics,
            preds: Vec::new(),
        })
    }

    /// Whether the early-stopping metric (the last one) is maximized.
    fn maximize(&self) -> bool {
        self.metrics.last().is_some_and(|m| m.maximize())
    }

    /// Evaluate every metric on every eval set's `margins`, append the
    /// scores to `history`, and return the last one (the early-stopping
    /// metric of the last eval set).
    fn record(
        &mut self,
        objective: &dyn Objective,
        iteration: usize,
        margins: &MarginCaches,
        history: &mut Vec<RoundEval>,
    ) -> f64 {
        let mut scores = Vec::with_capacity(self.evals.len() * self.metrics.len());
        let mut last_metric_value = 0.0;
        for (ei, (_, name)) in self.evals.iter().enumerate() {
            self.preds.clear();
            self.preds.extend_from_slice(&margins.evals[ei]);
            objective.eval_transform(&mut self.preds);
            for m in &self.metrics {
                let v = m.eval_info(&self.preds, &self.infos[ei]);
                scores.push((name.to_string(), m.name().to_string(), v));
                last_metric_value = v;
            }
        }
        history.push(RoundEval { iteration, scores });
        last_metric_value
    }
}

/// XGBoost's early-stopping rule: a round improves on the best score so
/// far only strictly (so a NaN score never does), and training stops after
/// `patience` rounds without improvement. Shared by [`Trainer`] and
/// [`CrossValidation`](crate::training::CrossValidation).
pub(crate) struct EarlyStopping {
    patience: usize,
    maximize: bool,
    best_score: f64,
    best_round: usize,
    since_improved: usize,
}

impl EarlyStopping {
    /// Refuse early stopping with a patience of zero rounds.
    pub(crate) fn check_patience(rounds: Option<usize>) -> Result<()> {
        if rounds == Some(0) {
            return Err(HessboostError::invalid_param(
                "early_stopping_rounds",
                "must be greater than zero",
            ));
        }
        Ok(())
    }

    /// Tracking that starts at round `first_round`, which stays the best
    /// one when no score ever improves.
    pub(crate) fn new(patience: usize, maximize: bool, first_round: usize) -> Self {
        EarlyStopping {
            patience,
            maximize,
            best_score: if maximize {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            },
            best_round: first_round,
            since_improved: 0,
        }
    }

    /// Record `round`'s `score`; `true` once patience has run out.
    pub(crate) fn observe(&mut self, round: usize, score: f64) -> bool {
        let improved = if self.maximize {
            score > self.best_score
        } else {
            score < self.best_score
        };
        if improved {
            self.best_score = score;
            self.best_round = round;
            self.since_improved = 0;
            false
        } else {
            self.since_improved += 1;
            self.since_improved >= self.patience
        }
    }

    /// The best round so far.
    pub(crate) fn best_round(&self) -> usize {
        self.best_round
    }
}

/// The metrics training evaluates without a custom metric:
/// `params.eval_metric`, or `objective`'s default. The last one is the
/// early-stopping metric.
pub(crate) fn configured_metrics(
    params: &TrainingParams,
    objective: &dyn Objective,
) -> Result<Vec<Box<dyn Metric>>> {
    create_metrics(
        &params.eval_metric,
        &objective.default_metric(),
        params.num_class,
        &ObjectiveParams::for_objective(params, objective.name()),
    )
}

/// An empty model that `objective` trains on `dtrain` with `params`, starting
/// from the margin-space intercepts `base_margins`. The model records the
/// objective's own name and output count (not the configured string /
/// `num_class`): a `reg:linear` alias is saved as `reg:squarederror` like
/// XGBoost does, and a custom objective's outputs determine the tree layout
/// even though `num_class` is 0.
pub(crate) fn new_model(
    params: &TrainingParams,
    objective: &dyn crate::objective::Objective,
    dtrain: &DMatrix,
    base_margins: Vec<f32>,
) -> BoostedModel {
    let mut model = BoostedModel::new(
        base_margins,
        ModelSpec {
            objective: objective.name().to_string(),
            objective_params: ObjectiveParams::for_objective(params, objective.name()),
            num_class: params.num_class,
            n_outputs: objective.n_outputs(),
            n_targets: dtrain.n_targets(),
            n_features: dtrain.n_cols(),
        },
    );
    model.set_num_parallel_tree(params.num_parallel_tree);
    model
}

/// Per-output intercepts in margin space. A user-supplied `base_score` is
/// given in prediction space and broadcast to every output through the
/// objective's link (XGBoost `ProbToMargin`; for multiclass this is a
/// uniform nonzero margin, as in XGBoost); otherwise the objective estimates
/// them from the labels (XGBoost `InitEstimation`).
pub(crate) fn initial_intercepts(
    params: &TrainingParams,
    objective: &dyn crate::objective::Objective,
    info: &MetaInfo,
    n_out: usize,
) -> Result<Vec<f32>> {
    if let Some(base_score) = params.base_score {
        if n_out > 1
            && crate::objective::distributional::DistFamily::from_objective(&params.objective)
                .is_some()
        {
            return Err(HessboostError::invalid_param(
                "base_score",
                "a scalar cannot set the several parameters of a `dist:*` objective; \
                 supply per-row `base_margin` instead",
            ));
        }
        let invalid = match params.objective.as_str() {
            "binary:logistic" | "reg:logistic" => !(0.0 < base_score && base_score < 1.0),
            "count:poisson" | "reg:gamma" | "reg:tweedie" | "survival:cox" | "survival:aft"
            | "dist:poisson" => base_score <= 0.0,
            _ => false,
        };
        if invalid {
            return Err(HessboostError::invalid_param(
                "base_score",
                "is outside the objective's valid output domain",
            ));
        }
    }
    let base_margins = match params.base_score {
        Some(bs) => {
            let mut scores = vec![bs as f32; n_out];
            objective.probs_to_margins(&mut scores);
            scores
        }
        None => objective.base_margins_info(info),
    };
    if base_margins.len() != n_out {
        return Err(HessboostError::dimension_mismatch(
            "objective base_margins length",
            n_out,
            base_margins.len(),
        ));
    }
    if base_margins.iter().any(|m| !m.is_finite()) {
        return Err(HessboostError::invalid_param(
            "base_score",
            format!("estimated intercept is not finite ({base_margins:?}); check the labels"),
        ));
    }
    Ok(base_margins)
}

/// What the boosting rounds do to the ensemble.
enum RoundPlan {
    /// Grow new trees with the prepared builder state.
    Grow(Prepared),
    /// `process_type=update`: refresh the queued trees of the initial model,
    /// one iteration per round.
    Refresh(Vec<RegTree>),
}

/// The learning rate applied to each new tree: `eta / num_parallel_tree`
/// (XGBoost divides the rate across a forest so a whole iteration moves by
/// `eta`), in `f32` as XGBoost's `learning_rate` is.
pub(super) fn tree_eta(params: &TrainingParams) -> f32 {
    params.eta as f32 / params.num_parallel_tree as f32
}

/// Apply `update` to every `(row, row margins)` pair of `margins`
/// (`[row][n_out]`), in parallel for large inputs. Rows are independent, so
/// parallel traversal preserves each row's floating-point addition order.
pub(super) fn for_each_row_margins(
    margins: &mut [f32],
    n_out: usize,
    update: impl Fn((usize, &mut [f32])) + Sync + Send,
) {
    if margins.len() / n_out >= 4096 && rayon::current_num_threads() > 1 {
        margins
            .par_chunks_mut(n_out)
            .with_min_len(1024)
            .enumerate()
            .for_each(update);
    } else {
        margins.chunks_mut(n_out).enumerate().for_each(update);
    }
}

/// Which margins of a row a tree adds to.
#[derive(Clone, Copy)]
pub(super) enum TreeOutput {
    /// A scalar-leaf tree feeding one output.
    Scalar(usize),
    /// A vector-leaf tree feeding every output.
    Vector,
}

/// The margins training keeps current, `[row][n_out]`: the training
/// matrix's and each eval set's, starting from the model's full current
/// predictions (a dataset's per-instance `base_margin`, when present,
/// overrides the per-output intercepts). Each new tree adds to every cell
/// once, so every cell sums its trees in ensemble order.
pub(super) struct MarginCaches<'a> {
    dtrain: &'a DMatrix,
    eval_sets: &'a [EvalSet<'a>],
    n_out: usize,
    /// The training matrix's margins.
    pub(super) train: Vec<f32>,
    /// Each eval set's margins, in eval-set order.
    pub(super) evals: Vec<Vec<f32>>,
}

impl<'a> MarginCaches<'a> {
    pub(super) fn new(
        model: &BoostedModel,
        dtrain: &'a DMatrix,
        eval_sets: &'a [EvalSet<'a>],
    ) -> Self {
        let trees = 0..model.num_trees();
        MarginCaches {
            dtrain,
            eval_sets,
            n_out: model.n_outputs(),
            train: model.margin_from_trees(dtrain, trees.clone()),
            evals: eval_sets
                .iter()
                .map(|(d, _)| model.margin_from_trees(d, trees.clone()))
                .collect(),
        }
    }

    /// Add `tree`'s predictions to every cache. `leaf_rows`, when given,
    /// lists the leaf of every training row and replaces the training
    /// matrix's traversal.
    pub(super) fn add_tree(
        &mut self,
        tree: &RegTree,
        output: TreeOutput,
        leaf_rows: Option<&[LeafRows]>,
    ) {
        match leaf_rows {
            Some(leaf_rows) => {
                let train = &mut self.train;
                apply_leaf_rows(tree, self.dtrain, leaf_rows, train, self.n_out, output);
            }
            None => add_tree_margins(tree, self.dtrain, &mut self.train, self.n_out, output),
        }
        for (margins, (d, _)) in self.evals.iter_mut().zip(self.eval_sets) {
            add_tree_margins(tree, d, margins, self.n_out, output);
        }
    }

    /// Recompute the eval caches from `model` (after a DART rescaling, which
    /// makes them non-additive).
    pub(super) fn recompute_evals(&mut self, model: &BoostedModel) {
        for (margins, (d, _)) in self.evals.iter_mut().zip(self.eval_sets) {
            *margins = model.margin_from_trees(d, 0..model.num_trees());
        }
    }
}

/// Add `tree`'s prediction of every row of `data` to `margins`.
fn add_tree_margins(
    tree: &RegTree,
    data: &DMatrix,
    margins: &mut [f32],
    n_out: usize,
    output: TreeOutput,
) {
    match output {
        TreeOutput::Scalar(k) => for_each_row_margins(margins, n_out, |(row, margin)| {
            margin[k] += tree.predict_row(data, row);
        }),
        TreeOutput::Vector => for_each_row_margins(margins, n_out, |(row, margin)| {
            let leaf = tree.leaf_id_with(|f| data.get(row, f as usize));
            for (m, &v) in margin.iter_mut().zip(tree.leaf_vector(leaf)) {
                *m += v;
            }
        }),
    }
}

/// Add each leaf's value (or vector), or its linear model's output, to the
/// margins of the training rows of `data` that reached it.
fn apply_leaf_rows(
    tree: &RegTree,
    data: &DMatrix,
    leaf_rows: &[LeafRows],
    margins: &mut [f32],
    n_out: usize,
    output: TreeOutput,
) {
    match (output, tree.linear_leaves()) {
        (TreeOutput::Scalar(k), None) => apply_leaf_values(
            leaf_rows,
            margins,
            n_out,
            |node| tree.node(node).leaf_value,
            |margins, base, _, value| margins[base + k] += value,
        ),
        // `RegTree::predict_row`'s linear-leaf arithmetic, without routing.
        (TreeOutput::Scalar(k), Some(linear)) => apply_leaf_values(
            leaf_rows,
            margins,
            n_out,
            |node| node,
            |margins, base, row, node| {
                let get = |f: u32| data.get(row as usize, f as usize);
                margins[base + k] += linear.predict(node, tree.node(node).leaf_value, get);
            },
        ),
        (TreeOutput::Vector, _) => apply_leaf_values(
            leaf_rows,
            margins,
            n_out,
            |node| tree.leaf_vector(node),
            |margins, base, _, value: &[f32]| {
                for (m, &v) in margins[base..base + n_out].iter_mut().zip(value) {
                    *m += v;
                }
            },
        ),
    }
}

/// `add(margins, row * n_out, row, value(leaf))` for every row of every leaf of
/// `leaf_rows`, in parallel row chunks for large inputs. Leaf row lists are
/// ascending, so each chunk locates its slice of every leaf by binary
/// search; each row still receives one addition per tree.
fn apply_leaf_values<V: Copy + Send + Sync>(
    leaf_rows: &[LeafRows],
    margins: &mut [f32],
    n_out: usize,
    value: impl Fn(usize) -> V + Sync,
    add: impl Fn(&mut [f32], usize, u32, V) + Sync,
) {
    const CHUNK_ROWS: usize = 8192;
    let n = margins.len() / n_out;
    if n < 2 * CHUNK_ROWS || rayon::current_num_threads() <= 1 {
        for leaf in leaf_rows {
            let v = value(leaf.node);
            for &row in &leaf.rows {
                add(margins, row as usize * n_out, row, v);
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
                let v = value(leaf.node);
                let start = leaf.rows.partition_point(|&row| row < first);
                let end = start + leaf.rows[start..].partition_point(|&row| row < last);
                for &row in &leaf.rows[start..end] {
                    add(margins, (row - first) as usize * n_out, row, v);
                }
            }
        });
}

/// The RNG and gradients of one tree-growing round, filled into `gpair`.
/// gbtree takes the gradients at the cached `margin`. DART (Dropout Additive
/// Regression Trees) first draws a dropout set `D` over the trees built so
/// far ([`select_dropout`]) and takes the gradients of the ensemble
/// **excluding** `D`, whose tree ids it returns. Using XGBoost's `tree`
/// normalization, if `k = |D|` the round's new trees then get weight
/// `1/(k+eta)` ([`dart_new_tree_weight`]) and [`finish_dart`] rescales each
/// dropped tree by `k/(k+eta)`.
pub(super) fn round_gradients(
    run: &TrainContext,
    model: &BoostedModel,
    iteration: usize,
    margin: &[f32],
    gpair: &mut [GradPair],
) -> (Rng, Option<Vec<usize>>) {
    let TrainContext {
        params,
        dtrain,
        info,
        objective,
    } = *run;
    if params.booster != BoosterKind::Dart {
        objective.gradient_info(margin, info, gpair);
        return (round_rng(params, iteration, 0), None);
    }
    let mut rng = round_rng(params, iteration, DART_SALT);
    let (dropped, drop_indices) = select_dropout(model, params, &mut rng);
    let margin_excl = model.predict_margin_dropout(dtrain, &dropped);
    objective.gradient_info(&margin_excl, info, gpair);
    (rng, Some(drop_indices))
}

/// The DART round RNG's booster salt.
const DART_SALT: u64 = 0x0DA27;

/// Draw a DART round's dropout set over the trees built so far: skipped with
/// probability `skip_drop`, otherwise each tree independently with
/// probability `rate_drop`, and at least one tree when any exist (as
/// XGBoost). Returns the per-tree mask and the dropped indices.
fn select_dropout(
    model: &BoostedModel,
    params: &TrainingParams,
    rng: &mut Rng,
) -> (Vec<bool>, Vec<usize>) {
    let existing = model.num_trees();
    let mut dropped = vec![false; existing];
    let mut drop_indices: Vec<usize> = Vec::new();
    let skip = rng.f64() < params.skip_drop;
    if !skip && existing > 0 {
        for (i, d) in dropped.iter_mut().enumerate() {
            if rng.f64() < params.rate_drop {
                *d = true;
                drop_indices.push(i);
            }
        }
        if drop_indices.is_empty() {
            // Guarantee at least one dropped tree, as XGBoost does.
            let i = rng.range(0..existing);
            dropped[i] = true;
            drop_indices.push(i);
        }
    }
    (dropped, drop_indices)
}

/// XGBoost's `tree` normalization weight of a DART round's new trees:
/// `1 / (k + eta)` for `k` dropped trees, `1` when none were dropped.
pub(super) fn dart_new_tree_weight(drop_indices: &[usize], params: &TrainingParams) -> f32 {
    let k = drop_indices.len();
    if k == 0 {
        1.0
    } else {
        1.0 / (k as f32 + params.eta as f32)
    }
}

/// Finish a DART round: rescale its dropped trees by `k / (k + eta)` so the
/// ensemble stays balanced, then recompute the eval margin caches, which the
/// rescaling makes non-additive.
pub(super) fn finish_dart(
    model: &mut BoostedModel,
    params: &TrainingParams,
    drop_indices: &[usize],
    margins: &mut MarginCaches,
) {
    let k = drop_indices.len() as f32;
    let factor = k / (k + params.eta as f32);
    for &i in drop_indices {
        model.scale_tree_weight(i, factor);
    }
    margins.recompute_evals(model);
}

/// Borrow the gradient slice for output `k`: the whole buffer for
/// single-output objectives, otherwise gather output `k`'s pairs into
/// `scratch` (length `n`) and borrow that.
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

/// Every output's gradients of `gpair` (`[row][n_out]`) gathered
/// output-major (`[output][row]`, as [`gather_output`] gathers one); empty
/// for a single output.
fn gather_outputs(gpair: &[GradPair], n_out: usize) -> Vec<GradPair> {
    if n_out == 1 {
        return Vec::new();
    }
    let n = gpair.len() / n_out;
    let mut out = vec![GradPair::default(); gpair.len()];
    out.par_chunks_exact_mut(n)
        .enumerate()
        .for_each(|(k, column)| {
            for (r, dst) in column.iter_mut().enumerate() {
                *dst = gpair[r * n_out + k];
            }
        });
    out
}

/// The RNG for one boosting round: `seed ^ round * 0x9E37_79B9`, plus a
/// booster-specific `salt` (`0` for gbtree, `0x0DA27` for DART) so the two
/// boosters draw from different streams.
fn round_rng(params: &TrainingParams, round: usize, salt: u64) -> Rng {
    Rng::new(params.seed ^ (round as u64).wrapping_mul(0x9E37_79B9) ^ salt)
}

/// One tree-growing boosting iteration: what each of its trees reads.
struct GrowRound<'a> {
    run: &'a TrainContext<'a>,
    prepared: &'a Prepared,
    /// Every output's gradients for this iteration, `[row][n_out]`.
    gpair: &'a [GradPair],
    n_out: usize,
    /// The model's absolute iteration index.
    iteration: usize,
    /// One gradient index per output, shared by that output's forest
    /// ([`Prepared::forest_indices`]); empty when trees build their own.
    forest_indices: &'a [OnceLock<GHistIndex>],
}

impl GrowRound<'_> {
    /// The gradient index output `output`'s forest shares, if any.
    fn forest_index(&self, output: usize) -> Option<&OnceLock<GHistIndex>> {
        self.forest_indices.get(output)
    }
}

/// Which tree of an iteration to grow: parallel tree `parallel` of output
/// `output`, on the uniform row subset `rows`, keeping its leaves' rows when
/// `capture_rows` (see [`Prepared::build_tree`]) and adding it to the
/// training margins from them when `margin_rows`.
struct TreeSlot<'a> {
    output: usize,
    parallel: usize,
    rows: &'a [u32],
    capture_rows: bool,
    margin_rows: bool,
}

/// Fit the tree `slot` of the iteration `grow`: gather that output's
/// gradient slice (into `scratch` for multi-output objectives), apply
/// gradient-based row sampling when configured (per tree, as XGBoost's hist
/// updater does, or once per output forest under `approx`, kept in
/// `forest_sample` by the forest's first tree for the rest), derive its
/// column sampler, build the tree, fit linear leaves when configured, and
/// shrink its leaves by `eta / num_parallel_tree`. The caller owns the round
/// RNG (already seeded and salted), the reuse dictionary, and what happens
/// to the tree (margin updates, contribution weight).
fn fit_output_tree(
    grow: &GrowRound,
    slot: &TreeSlot,
    scratch: &mut [GradPair],
    rng: &mut Rng,
    forest_sample: &mut Option<GradientSample>,
    reuse: Option<&mut ReuseSet>,
) -> Result<(RegTree, Vec<LeafRows>)> {
    let TrainContext { params, dtrain, .. } = *grow.run;
    let (prepared, n_out) = (grow.prepared, grow.n_out);
    let gk: &[GradPair] = gather_output(grow.gpair, scratch, n_out, slot.output);
    let own;
    let sampled = if !gradient_sampling(params) {
        None
    } else if prepared.samples_per_forest() {
        if slot.parallel == 0 {
            *forest_sample = gradient_based_sample(gk, 1, params.subsample, rng)?;
        }
        forest_sample.as_ref()
    } else {
        own = gradient_based_sample(gk, 1, params.subsample, rng)?;
        own.as_ref()
    };
    let (gk, rows) = match sampled {
        Some(s) => (s.gpair.as_slice(), s.rows.as_slice()),
        None => (gk, slot.rows),
    };
    let mut sampler = make_column_sampler(dtrain, params, rng);
    let rounding_seed = quantization_seed(params, rng);
    let sample = TreeSample {
        gpair: gk,
        rows,
        forest_index: grow.forest_index(slot.output),
    };
    Ok(grow_sampled_tree(
        grow,
        slot,
        sample,
        &mut sampler,
        rounding_seed,
        reuse,
    ))
}

/// The part of [`fit_output_tree`] after its RNG draws: build the tree on
/// `sample`, fit linear leaves when configured, and shrink its leaves.
fn grow_sampled_tree(
    grow: &GrowRound,
    slot: &TreeSlot,
    sample: TreeSample,
    sampler: &mut ColumnSampler,
    rounding_seed: u64,
    reuse: Option<&mut ReuseSet>,
) -> (RegTree, Vec<LeafRows>) {
    let TrainContext { params, dtrain, .. } = *grow.run;
    let TreeSample {
        gpair: gk, rows, ..
    } = sample;
    let (mut tree, leaf_rows) = grow.prepared.build_tree(
        grow.run,
        sample,
        sampler,
        reuse,
        rounding_seed,
        slot.capture_rows,
    );
    // LightGBM keeps the first iteration's trees constant.
    if params.linear_tree && grow.iteration > 0 {
        let lambda = params.linear_lambda;
        if leaf_rows.is_empty() {
            crate::tree::linear::fit_linear_leaves(&mut tree, dtrain, gk, rows, lambda);
        } else {
            crate::tree::linear::fit_captured_linear_leaves(
                &mut tree, dtrain, gk, &leaf_rows, lambda,
            );
        }
    }
    tree.scale_leaves(tree_eta(params));
    (tree, leaf_rows)
}

/// The stochastic-rounding seed of one quantized tree, drawn from the
/// iteration's RNG after the tree's column sampler, so every tree of an
/// iteration (outputs and parallel trees alike) rounds independently and
/// continued training resumes the same streams. Draws nothing unless
/// `use_quantized_grad` is on, leaving the default RNG streams untouched.
fn quantization_seed(params: &TrainingParams, rng: &mut Rng) -> u64 {
    if params.use_quantized_grad {
        rng.next_u64()
    } else {
        0
    }
}

/// Bernoulli row subsampling (each row kept with probability `subsample`),
/// matching XGBoost's default sampling method. Guarantees at least one row.
/// Gradient-based sampling keeps every row here; it samples the gradients in
/// [`fit_output_tree`] instead.
pub(super) fn sample_rows(n: usize, params: &TrainingParams, rng: &mut Rng) -> Vec<u32> {
    let subsample = params.subsample;
    if subsample >= 1.0 || params.sampling_method == SamplingMethod::GradientBased {
        return all_rows(n);
    }
    // Sized for the expected sample plus a few standard deviations.
    let expected = n as f64 * subsample;
    let mut rows: Vec<u32> = Vec::with_capacity((expected + 4.0 * expected.sqrt() + 16.0) as usize);
    rows.extend((0..n as u32).filter(|_| rng.f64() < subsample));
    if rows.is_empty() {
        rows.push(rng.range(0..n) as u32);
    }
    rows
}

/// One iteration's uniform row subsets, drawn before its trees: one per
/// parallel tree, or a single subset for the whole forest when
/// `per_forest` (`approx`, [`Prepared::samples_per_forest`]) or when there is
/// no uniform sampling (every tree then reads all rows, and [`sample_rows`]
/// draws nothing). Parallel tree `p` uses entry `p % len`, shared across its
/// per-output fits.
pub(super) fn iteration_row_subsets(
    n: usize,
    params: &TrainingParams,
    per_forest: bool,
    rng: &mut Rng,
) -> Vec<Vec<u32>> {
    let uniform = params.subsample < 1.0 && params.sampling_method == SamplingMethod::Uniform;
    let draws = if per_forest || !uniform {
        1
    } else {
        params.num_parallel_tree
    };
    (0..draws).map(|_| sample_rows(n, params, rng)).collect()
}

/// Whether trees are grown on gradient-based (MVS) row samples.
pub(super) fn gradient_sampling(params: &TrainingParams) -> bool {
    params.sampling_method == SamplingMethod::GradientBased && params.subsample < 1.0
}

/// Shape and metadata checks for the training matrix and every eval set,
/// followed by the objective's own [`validate_info`] label-domain checks.
///
/// [`validate_info`]: crate::objective::Objective::validate_info
pub(crate) fn validate_dataset(
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
                return Err(HessboostError::dimension_mismatch(
                    "labels length (n_rows * training n_targets)",
                    expected,
                    labels.len(),
                ));
            }
        }
    }
    if data.n_cols() != n_features {
        return Err(HessboostError::dimension_mismatch(
            "dataset feature count",
            n_features,
            data.n_cols(),
        ));
    }
    if let Some(margin) = data.base_margin() {
        let expected = data.n_rows().checked_mul(n_out).ok_or_else(|| {
            HessboostError::invalid_param("base_margin", "expected length overflows usize")
        })?;
        if margin.len() != data.n_rows() && margin.len() != expected {
            return Err(HessboostError::dimension_mismatch(
                "base_margin length",
                expected,
                margin.len(),
            ));
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

/// Refuse a metric that reads a different number of predictions per row
/// than the model's `n_out` outputs (XGBoost's "label and prediction size
/// not match"): an elementwise metric on an alpha-list, multiclass, or
/// distributional model, a multiclass metric on a single-output model, and
/// so on. A metric of any width ([`Metric::prediction_width`] `None`, the
/// custom-metric hook) needs a whole number of outputs per label column.
///
/// [`Metric::prediction_width`]: crate::metric::Metric::prediction_width
fn check_prediction_width(
    metric: &dyn crate::metric::Metric,
    info: &MetaInfo,
    n_out: usize,
) -> Result<()> {
    let reason = match metric.prediction_width(info) {
        Some(width) if width != n_out => format!(
            "metric `{}` reads {width} prediction(s) per row of dataset, but the model has \
             {n_out} outputs",
            metric.name()
        ),
        None if !n_out.is_multiple_of(info.n_targets.max(1)) => format!(
            "metric `{}` needs a whole number of the model's {n_out} outputs per label \
             column of dataset ({} columns)",
            metric.name(),
            info.n_targets
        ),
        _ => return Ok(()),
    };
    Err(HessboostError::invalid_param("eval_metric", reason))
}

/// Build one tree's column sampler over `dtrain`'s features, seeded from
/// `rng`: the `colsample_bytree` pool, then the `bylevel`/`bynode` draws,
/// weighted by the training matrix's feature weights when it has them.
pub(super) fn make_column_sampler(
    dtrain: &DMatrix,
    params: &TrainingParams,
    rng: &mut Rng,
) -> ColumnSampler {
    ColumnSampler::new(
        dtrain.n_cols(),
        dtrain.feature_weights(),
        params.colsample_bytree,
        params.colsample_bylevel,
        params.colsample_bynode,
        rng.next_u64(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Metric, Rmse};
    use crate::test_support::labeled_dense;
    use crate::tree::{ChildLeaf, SplitRule};

    /// A deterministic uniform `[0, 1)` stream (a 64-bit LCG's top 31 bits).
    fn lcg(mut s: u64) -> impl FnMut() -> f32 {
        move || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as f32) / (1u32 << 31) as f32
        }
    }

    /// A learnable 1-D step function: y = 0 for x<0.5, y = 1 for x>=0.5.
    fn step_dataset(n: usize) -> DMatrix {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let xi = i as f32 / n as f32;
            x.push(xi);
            y.push(if xi >= 0.5 { 1.0 } else { 0.0 });
        }
        labeled_dense(&x, n, 1, &y)
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
        tree.expand(
            0,
            SplitRule::numeric(0, 3.0, true),
            ChildLeaf::new(-0.25, 1.0),
            ChildLeaf::new(0.75, 1.0),
        );
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
                pool.install(|| {
                    add_tree_margins(
                        &tree,
                        &data,
                        &mut actual,
                        outputs,
                        TreeOutput::Scalar(output),
                    );
                });
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn chunked_leaf_rows_match_the_serial_traversal() {
        // Enough rows for the chunked parallel path (two 8192-row chunks
        // and a partial third); leaves interleave across chunk boundaries.
        let n = 20_000;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                if i % 13 == 0 {
                    f32::NAN
                } else {
                    (i % 7) as f32
                }
            })
            .collect();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let leaf_rows_of = |tree: &RegTree| {
            let mut leaves: Vec<LeafRows> = Vec::new();
            for row in 0..n {
                let node = tree.leaf_id_with(|f| data.get(row, f as usize));
                match leaves.iter_mut().find(|l| l.node == node) {
                    Some(leaf) => leaf.rows.push(row as u32),
                    None => leaves.push(LeafRows {
                        node,
                        rows: vec![row as u32],
                    }),
                }
            }
            leaves
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let start = |outputs: usize| -> Vec<f32> {
            (0..n * outputs).map(|i| (i % 97) as f32 / 8.0).collect()
        };

        let mut scalar = RegTree::with_root(n as f32);
        scalar.expand(
            0,
            SplitRule::numeric(0, 3.0, true),
            ChildLeaf::new(-0.25, 1.0),
            ChildLeaf::new(0.75, 1.0),
        );
        let leaves = leaf_rows_of(&scalar);
        let mut linear = scalar.clone();
        let gpair: Vec<GradPair> = x
            .iter()
            .map(|&v| GradPair::new(-(2.0 * v.max(0.0) + 1.0), 1.0))
            .collect();
        let rows: Vec<u32> = (0..n as u32).collect();
        crate::tree::linear::fit_linear_leaves(&mut linear, &data, &gpair, &rows, 0.5);
        assert!(linear.linear_leaves().is_some());
        for tree in [&scalar, &linear] {
            let output = TreeOutput::Scalar(1);
            let mut expected = start(3);
            add_tree_margins(tree, &data, &mut expected, 3, output);
            let mut actual = start(3);
            pool.install(|| apply_leaf_rows(tree, &data, &leaves, &mut actual, 3, output));
            assert_eq!(actual, expected);
        }

        let mut vector = RegTree::with_vector_root(3, n as f32);
        let (left, right) = vector.expand(
            0,
            SplitRule::numeric(0, 3.0, false),
            ChildLeaf::new(0.0, 1.0),
            ChildLeaf::new(0.0, 1.0),
        );
        let (ll, lr) = vector.expand(
            left,
            SplitRule::numeric(0, 1.0, true),
            ChildLeaf::new(0.0, 1.0),
            ChildLeaf::new(0.0, 1.0),
        );
        vector.set_leaf_vector(ll, &[0.5, -1.0, 0.125]);
        vector.set_leaf_vector(lr, &[-0.75, 0.25, 2.0]);
        vector.set_leaf_vector(right, &[1.5, 0.0625, -0.5]);
        let leaves = leaf_rows_of(&vector);
        let mut expected = start(3);
        add_tree_margins(&vector, &data, &mut expected, 3, TreeOutput::Vector);
        let mut actual = start(3);
        pool.install(|| {
            apply_leaf_rows(&vector, &data, &leaves, &mut actual, 3, TreeOutput::Vector);
        });
        assert_eq!(actual, expected);
    }

    /// Squared error around `y` with fixed, per-output Hessians: output 0
    /// weights row 0 heavily, output 1 row 3, so their Hessian-weighted
    /// `approx` cuts differ.
    struct PerOutputHessians;

    impl Objective for PerOutputHessians {
        fn name(&self) -> &'static str {
            "custom:per_output_hessians"
        }
        fn n_outputs(&self) -> usize {
            2
        }
        fn gradient(&self, preds: &[f32], labels: &[f32], _: Option<&[f32]>, out: &mut [GradPair]) {
            const HESS: [[f32; 4]; 2] = [[100.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 100.0]];
            for (i, (g, p)) in out.iter_mut().zip(preds).enumerate() {
                let (row, output) = (i / 2, i % 2);
                let h = HESS[output][row];
                *g = GradPair::new(h * (p - labels[row]), h);
            }
        }
        fn const_hess(&self) -> bool {
            true
        }
        fn default_metric(&self) -> String {
            "rmse".into()
        }
    }

    #[test]
    fn approx_constant_hessian_cuts_do_not_depend_on_thread_count() {
        // The parallel slot loop grows both outputs' trees at once; the
        // cached cuts must still come from output 0, as the serial loop's.
        let d = labeled_dense(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let fit = |nthread: usize| {
            let params = TrainingParams::builder()
                .tree_method(TreeMethod::Approx)
                .max_bin(2)
                .max_depth(1)
                .min_child_weight(0.0)
                .nthread(nthread)
                .build()
                .unwrap();
            let model = Trainer::new(&params, &d, 3)
                .objective(&PerOutputHessians)
                .train()
                .unwrap()
                .model;
            model.predict_margin(&d).unwrap()
        };
        let serial = fit(1);
        for _ in 0..32 {
            assert_eq!(fit(4), serial);
        }
    }

    /// Zero-weight rows are not sketched, so one whose value lies beyond the
    /// last cut is binned into the last bin but routed right of a split on
    /// it by the finished tree. Linear leaves must then be fitted from raw
    /// routing, as without captured rows: fitting the second tree from the
    /// builder's rows gave leaf 3 intercept 1.0 instead of 0.5, and row 0 a
    /// prediction of 2.0 instead of 1.5.
    #[test]
    fn linear_leaves_route_zero_weight_rows_like_the_tree() {
        let nan = f32::NAN;
        let x = [0.0, 1.0, 0.0, nan, 1.0, 0.0, 0.0, 100.0, 0.0, 101.0];
        let data = DMatrix::from_dense(&x, 5, 2)
            .unwrap()
            .with_labels(&[2.0, 0.0, 10.0, 0.0, 0.0])
            .unwrap()
            .with_weights(&[1.0, 1.0, 1.0, 0.0, 0.0])
            .unwrap();
        let params = TrainingParams::builder()
            .tree_method(TreeMethod::Hist)
            .linear_tree(true)
            .base_score(0.0)
            .eta(1.0)
            .lambda(1.0)
            .linear_lambda(1.0)
            .max_depth(2)
            .build()
            .unwrap();
        let model = Trainer::new(&params, &data, 2).train().unwrap().model;
        assert_eq!(model.predict(&data).unwrap(), [1.5, 0.0, 7.5, 0.0, 0.0]);
        let linear = model.trees()[1].linear_leaves().unwrap();
        let intercepts: Vec<u64> = (0..5).map(|id| linear.intercept(id).to_bits()).collect();
        let routed: Vec<u64> = [0.0f64, 0.0, 2.5, 0.5, -0.0].map(f64::to_bits).to_vec();
        assert_eq!(intercepts, routed);
    }

    #[test]
    fn approx_forests_share_one_index_per_output_across_threads() {
        // Per-round `approx` cuts (non-constant Hessians) are shared by an
        // output's parallel trees; the parallel slot loop builds them before
        // growing the trees, and every thread count grows the same forest.
        // 20,000 rows × 4 features reach the parallel cut construction
        // (65,536 cells), whose nested rayon loops deadlocked when tree
        // tasks built the shared index themselves.
        let n = 20_000;
        let x: Vec<f32> = (0..n * 4)
            .map(|i| ((i * 7919) % 1009) as f32 / 1009.0)
            .collect();
        let y: Vec<f32> = x
            .chunks(4)
            .map(|r| f32::from(r[0] + 0.5 * r[1] > 0.7))
            .collect();
        let d = labeled_dense(&x, n, 4, &y);
        let fit = |nthread: usize| {
            let params = TrainingParams::builder()
                .objective("multi:softprob")
                .num_class(2)
                .tree_method(TreeMethod::Approx)
                .num_parallel_tree(4)
                .max_depth(3)
                .nthread(nthread)
                .build()
                .unwrap();
            train(&params, &d, 3).unwrap().predict_margin(&d).unwrap()
        };
        let serial = fit(1);
        for _ in 0..4 {
            assert_eq!(fit(8), serial);
        }
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
    fn tree_methods_reach_similar_accuracy() {
        let d = step_dataset(120);
        let rmse = |method: TreeMethod| {
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
        let rmse_hist = rmse(TreeMethod::Hist);
        assert!(rmse_hist < 0.05, "hist rmse {rmse_hist}");
        for method in [TreeMethod::Exact, TreeMethod::Approx] {
            let other = rmse(method);
            assert!(other < 0.05, "{method:?} rmse {other}");
            // The methods land very close on this cleanly-binnable problem.
            assert!((other - rmse_hist).abs() < 0.02, "{method:?} rmse {other}");
        }
    }

    #[test]
    fn lossguide_trains_end_to_end() {
        let d = step_dataset(120);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(TreeMethod::Hist)
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
            .tree_method(TreeMethod::Exact)
            .grow_policy(GrowPolicy::LossGuide)
            .max_leaves(8)
            .build()
            .unwrap();
        assert!(matches!(
            train(&params, &d, 5),
            Err(HessboostError::InvalidParameter { name, .. }) if name == "grow_policy"
        ));
    }

    #[test]
    fn objective_override_records_its_own_distribution() {
        // `params` name a `dist:*` objective, but `Trainer::objective`
        // boosts squared error: the model's objective metadata must follow
        // the objective it records, or loading refuses it.
        let x: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..24).map(|i| (i % 5) as f32).collect();
        let d = labeled_dense(&x, 24, 1, &y);
        let params = TrainingParams::builder()
            .objective("dist:normal")
            .build()
            .unwrap();
        let squared = crate::objective::SquaredError;
        let model = Trainer::new(&params, &d, 3)
            .objective(&squared)
            .train()
            .unwrap()
            .model;
        assert_eq!(model.objective(), "reg:squarederror");
        assert_eq!(model.objective_params().distribution, None);
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
        // Continued training records the initial model's objective too.
        let more = Trainer::new(&params, &d, 2)
            .objective(&squared)
            .init_model(&model)
            .train()
            .unwrap()
            .model;
        assert_eq!(more.objective_params().distribution, None);
        assert_eq!(more.num_boost_rounds(), 5);
    }

    #[test]
    fn num_class_must_match_the_objective_outputs() {
        let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..12).map(|i| (i % 3) as f32).collect();
        let d = labeled_dense(&x, 12, 1, &y);
        let params = |objective: &str, num_class| {
            TrainingParams::builder()
                .objective(objective)
                .num_class(num_class)
                .build()
                .unwrap()
        };
        // A stray num_class used to train a single-output model that
        // `from_bytes`/`from_json` then refused.
        assert!(matches!(
            train(&params("reg:squarederror", 3), &d, 2),
            Err(HessboostError::InvalidParameter { name, .. }) if name == "num_class"
        ));
        for num_class in [0, 1] {
            let model = train(&params("reg:squarederror", num_class), &d, 2).unwrap();
            BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
        }
        let model = train(&params("multi:softprob", 3), &d, 2).unwrap();
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
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
        let d = labeled_dense(&x, n, 1, &y);
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
        let mut rng = lcg(7);
        let x: Vec<f32> = (0..n).map(|_| rng()).collect();
        // The rate increases with x; the label is a rough count.
        let y: Vec<f32> = x.iter().map(|xi| (1.0 + 5.0 * xi).round()).collect();
        let d = labeled_dense(&x, n, 1, &y);
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
        let mean = |high: bool| {
            let bucket: Vec<f32> = (0..n)
                .filter(|&i| (x[i] >= 0.5) == high)
                .map(|i| preds[i])
                .collect();
            bucket.iter().sum::<f32>() / bucket.len() as f32
        };
        let (lo_mean, hi_mean) = (mean(false), mean(true));
        assert!(
            lo_mean < hi_mean,
            "rate should rise with x: {lo_mean} vs {hi_mean}"
        );
    }

    #[test]
    fn custom_objective_matches_builtin_squared_error() {
        use crate::objective::CustomObjective;
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
            Trainer::new(&p, &d, 30)
                .objective(&obj)
                .train()
                .unwrap()
                .model
                .predict(&d)
                .unwrap()
        };

        for (a, b) in builtin.iter().zip(&custom) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn custom_multi_output_objective_trains_with_stride_and_round_trips() {
        use crate::objective::CustomObjective;
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
        let model = Trainer::new(&p, &d, rounds)
            .objective(&obj)
            .train()
            .unwrap()
            .model;

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
        let mut rng = lcg(42);
        for _ in 0..n_groups {
            for d in 0..per {
                let rel = d as f32; // relevance grade 0..per-1
                let noise = (rng() - 0.5) * 0.8;
                x.push(rel + noise);
                y.push(rel);
            }
        }
        let d = labeled_dense(&x, n, 1, &y)
            .with_group_sizes(&sizes)
            .unwrap();

        let params = TrainingParams::builder()
            .objective("rank:ndcg")
            .max_depth(3)
            .eta(0.2)
            .build()
            .unwrap();
        let res = Trainer::new(&params, &d, 40)
            .eval(&d, "train")
            .train()
            .unwrap();
        assert!(!res.history.is_empty());

        // rank:ndcg's default metric: XGBoost's `ndcg@32`.
        let ndcg_of = |r: &RoundEval| r.scores.iter().find(|(_, m, _)| m == "ndcg@32").unwrap().2;
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

        // Native and JSON round-trips preserve predictions (weights included).
        for restored in [
            BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
            BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
        ] {
            assert_eq!(restored.predict(&d).unwrap(), preds);
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
        let numeric = labeled_dense(&x, x.len(), 1, &y);
        let categorical = numeric
            .clone()
            .with_feature_types(&[FeatureType::Categorical])
            .unwrap();

        // Depth-1 stumps: the numeric model can only threshold, the categorical
        // model can partition the category set in a single node.
        for method in [TreeMethod::Auto, TreeMethod::Exact] {
            let mk = |d: &DMatrix| {
                let p = TrainingParams::builder()
                    .objective("reg:squarederror")
                    .tree_method(method)
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
            assert!(
                rmse_cat < 0.02,
                "{method:?} categorical rmse too high: {rmse_cat}"
            );
            assert!(
                rmse_num > 0.05,
                "{method:?} numeric unexpectedly fit it: {rmse_num}"
            );
        }
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
        let d = labeled_dense(&x, n, 1, &y);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(TreeMethod::Exact)
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

    /// Training starts from a per-instance base margin: with 0 rounds the
    /// margin is exactly the base margin, and trees then fit the labels'
    /// residuals from it (not from the intercept).
    #[test]
    fn training_starts_from_the_base_margin() {
        let d = step_dataset(60);
        let bm: Vec<f32> = (0..60).map(|i| i as f32 * 0.01 + 1.5).collect();
        let d_bm = d.with_base_margin(&bm).unwrap();
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let initial = train(&params, &d_bm, 0).unwrap();
        assert_eq!(initial.predict_margin(&d_bm).unwrap(), bm);

        let fitted = train(&params, &d_bm, 10).unwrap();
        let margin = fitted.predict_margin(&d_bm).unwrap();
        let rmse = Rmse.eval(&margin, d_bm.labels().unwrap(), None);
        assert!(
            rmse < 0.2,
            "the trees did not fit the base-margin residuals: {rmse}"
        );
    }

    #[test]
    fn colsample_bynode_changes_the_model() {
        // Multi-feature dataset so column sampling has features to drop.
        let (n, f) = (400usize, 8usize);
        let mut x = vec![0f32; n * f];
        let mut y = vec![0f32; n];
        let mut rng = lcg(3);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..f {
                let v = rng();
                x[i * f + j] = v;
                acc += v * (j as f32 + 1.0);
            }
            y[i] = acc;
        }
        let d = labeled_dense(&x, n, f, &y);

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
        let mut rng = lcg(11);
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
        labeled_dense(&x, n, f, &y)
    }

    #[test]
    fn gblinear_fits_linear_target() {
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
        let d = linear_dataset(200);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .booster(BoosterKind::GbLinear)
            .eta(0.5)
            .build()
            .unwrap();
        let model = train(&params, &d, 100).unwrap();
        let before = model.predict(&d).unwrap();

        for restored in [
            BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
            BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
        ] {
            assert_eq!(restored.predict(&d).unwrap(), before);
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
        let builtin = Trainer::new(&params, &d, 200)
            .eval(&d, "train")
            .early_stopping_rounds(5)
            .train()
            .unwrap();

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
        let custom = Trainer::new(&params, &d, 200)
            .eval(&d, "train")
            .early_stopping_rounds(5)
            .custom_metric(Box::new(rmse_metric))
            .train()
            .unwrap();

        // Early stopping fired `patience` rounds after the best iteration,
        // identically for both metrics, so the models match tree for tree.
        let best = builtin.model.best_iteration().expect("stops early");
        assert_eq!(builtin.history.len(), best + 6);
        assert_eq!(builtin.model.num_trees(), best + 6);
        assert_eq!(custom.model.best_iteration(), Some(best));
        assert_eq!(
            custom.model.predict(&d).unwrap(),
            builtin.model.predict(&d).unwrap()
        );

        // The custom metric was actually recorded in the history.
        let names: Vec<&str> = custom.history[0]
            .scores
            .iter()
            .map(|(_, m, _)| m.as_str())
            .collect();
        assert_eq!(names, vec!["rmse"]);
    }

    #[test]
    fn multiclass_softmax_returns_one_label_per_row() {
        let x = [0.0, 0.1, 0.5, 0.6, 0.9, 1.0];
        let y = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let d = labeled_dense(&x, 6, 1, &y);
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
            let d = labeled_dense(&x, 8, 1, &soft);
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
                let d = labeled_dense(&x, 8, 1, &labels);
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
        let d = labeled_dense(&[0.0, 1.0], 2, 1, &[1.0, 2.0]);
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
        assert!(
            Trainer::new(&params, &d, 2)
                .early_stopping_rounds(1)
                .train()
                .is_err()
        );
        assert!(
            Trainer::new(&params, &d, 2)
                .eval(&d, "eval")
                .early_stopping_rounds(0)
                .train()
                .is_err()
        );

        let unlabeled = DMatrix::from_dense(&[0.0, 1.0], 2, 1).unwrap();
        assert!(
            Trainer::new(&params, &d, 2)
                .eval(&unlabeled, "eval")
                .train()
                .is_err()
        );
        let wrong_features = labeled_dense(&[0.0, 0.0], 1, 2, &[0.0]);
        assert!(
            Trainer::new(&params, &d, 2)
                .eval(&wrong_features, "eval")
                .train()
                .is_err()
        );
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
        let holdout = labeled_dense(&[0.0, 1.0], 2, 1, &[0.0, 2.0]);
        match Trainer::new(&params, &d, 2)
            .eval(&holdout, "holdout")
            .train()
        {
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
            Trainer::new(&params, &d, 1)
                .eval(&two_targets, "eval")
                .train(),
            Err(HessboostError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            Trainer::new(&params, &two_targets, 1)
                .eval(&d, "eval")
                .train(),
            Err(HessboostError::DimensionMismatch { .. })
        ));
        let ndcg = TrainingParams::builder()
            .eval_metric("ndcg")
            .build()
            .unwrap();
        eval_metric_rejection(train(&ndcg, &two_targets, 1), "ndcg");
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
        let model = Trainer::new(&params, &d, 3)
            .objective(&BoundsMidpoint)
            .train()
            .unwrap()
            .model;
        // Base margin is the mean midpoint (1 and 5 → 3); one full-step tree
        // then lands every row on its own midpoint.
        assert_eq!(model.base_score(), 3.0);
        let preds = model.predict_margin(&d).unwrap();
        for (row, p) in preds.iter().enumerate() {
            let expected = if row < 16 { 1.0 } else { 5.0 };
            assert!((p - expected).abs() < 1e-3, "row {row}: {p}");
        }

        let unbounded = DMatrix::from_dense(&x, 32, 1).unwrap();
        assert!(matches!(
            Trainer::new(&params, &unbounded, 1).objective(&BoundsMidpoint).train(),
            Err(HessboostError::InvalidParameter { name, .. }) if name == "label_lower_bound"
        ));
    }

    /// The reason of an `eval_metric` rejection of the `context` run,
    /// panicking on any other outcome.
    fn eval_metric_rejection<T: std::fmt::Debug>(result: Result<T>, context: &str) -> String {
        match result {
            Err(HessboostError::InvalidParameter {
                name: "eval_metric",
                reason,
            }) => reason,
            other => panic!("{context}: expected an `eval_metric` rejection, got {other:?}"),
        }
    }

    #[test]
    fn label_metrics_are_refused_on_bound_only_eval_sets() {
        // `survival:aft` trains on label bounds alone, so the label slice is
        // empty: metrics reading ordinary labels must be refused, not index it.
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let lower: Vec<f32> = (0..32).map(|i| 1.0 + (i % 4) as f32).collect();
        let upper: Vec<f32> = lower.iter().map(|lo| lo + 1.0).collect();
        let d = DMatrix::from_dense(&x, 32, 1)
            .unwrap()
            .with_label_bounds(&lower, &upper)
            .unwrap();
        let params = |metric: &str| {
            TrainingParams::builder()
                .objective("survival:aft")
                .eval_metric(metric)
                .build()
                .unwrap()
        };
        for metric in ["aft-nloglik", "interval-regression-accuracy"] {
            let run = Trainer::new(&params(metric), &d, 2)
                .eval(&d, "eval")
                .train()
                .unwrap();
            assert!(run.history[1].scores[0].2.is_finite(), "{metric}");
        }
        for metric in ["rmse", "mae", "cox-nloglik"] {
            let run = Trainer::new(&params(metric), &d, 2)
                .eval(&d, "eval")
                .train();
            let reason = eval_metric_rejection(run, metric);
            assert!(reason.contains("`eval`"), "{reason}");
        }
    }

    #[test]
    fn per_row_metrics_refuse_label_matrices() {
        // The survival metrics read one interval and weight per row, and
        // `pre@k` ranks one label per row within query groups; on a label
        // matrix they must be refused rather than index the row weights per
        // cell or rank every target's cells together.
        let n = 24;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..2 * n).map(|i| (i % 5) as f32).collect();
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_label_matrix(&y, 2)
            .unwrap()
            .with_weights(&vec![1.0; n])
            .unwrap()
            .with_group_sizes(&[12, 12])
            .unwrap();
        for metric in ["aft-nloglik", "interval-regression-accuracy", "pre@3"] {
            let params = TrainingParams::builder()
                .eval_metric(metric)
                .build()
                .unwrap();
            eval_metric_rejection(
                Trainer::new(&params, &d, 1).eval(&d, "eval").train(),
                metric,
            );
        }
    }

    #[test]
    fn metrics_must_match_the_prediction_width() {
        // Elementwise metrics read one prediction per label; a model with
        // more outputs than label columns must be refused before the first
        // evaluation instead of indexing past the labels.
        let n = 30;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| 1.0 + (i % 3) as f32).collect();
        let d = labeled_dense(&x, n, 1, &y);
        let quantile = || {
            TrainingParams::builder()
                .objective("reg:quantileerror")
                .quantile_alpha(vec![0.2, 0.8])
        };
        let expectile = || {
            TrainingParams::builder()
                .objective("reg:expectileerror")
                .expectile_alpha(vec![0.3, 0.5, 0.7])
        };
        let normal = || TrainingParams::builder().objective("dist:normal");
        let softprob = || {
            TrainingParams::builder()
                .objective("multi:softprob")
                .num_class(4)
        };
        let run = |params: TrainingParams| {
            Trainer::new(&params, &d, 2)
                .eval(&d, "eval")
                .train()
                .map(|r| r.history)
        };
        for (params, metric) in [
            (quantile().eval_metric("rmse"), "rmse"),
            (expectile().eval_metric("mae"), "mae"),
            (normal().eval_metric("rmse"), "rmse"),
            (softprob().eval_metric("rmse"), "rmse"),
            (softprob().eval_metric("auc"), "auc"),
            (quantile().eval_metric("logloss"), "logloss"),
            (
                TrainingParams::builder().eval_metric("mlogloss"),
                "mlogloss",
            ),
        ] {
            let reason = eval_metric_rejection(run(params.build().unwrap()), metric);
            assert!(
                reason.contains(metric) && reason.contains("`eval`"),
                "{reason}"
            );
        }
        // The defaults and matching metrics still evaluate.
        for params in [
            quantile(),
            quantile().eval_metric("quantile"),
            expectile(),
            normal(),
            normal().eval_metric("crps"),
            softprob(),
            softprob().eval_metric("merror"),
        ] {
            let history = run(params.build().unwrap()).unwrap();
            assert!(
                history[1].scores.iter().all(|s| s.2.is_finite()),
                "{history:?}"
            );
        }
    }

    #[test]
    fn custom_objective_outputs_must_match_the_label_layout() {
        // A custom objective reads one label per row or one per output; a
        // label matrix of another width must be refused before it reaches the
        // gradient closure.
        let n = 20;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..2 * n).map(|i| (i % 4) as f32).collect();
        let d = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_label_matrix(&y, 2)
            .unwrap();
        let objective = |k: usize| {
            crate::objective::CustomObjective::new("custom:k", k, 0.0, "rmse", |_p, _y, _w, out| {
                for g in out.iter_mut() {
                    *g = GradPair::new(0.1, 1.0);
                }
            })
        };
        let sums = || {
            Box::new(crate::metric::CustomMetric::new(
                "sum",
                false,
                |p, _y, _w| p.iter().map(|&v| f64::from(v)).sum(),
            ))
        };
        let params = TrainingParams::builder().build().unwrap();
        let run = |k: usize| {
            Trainer::new(&params, &d, 1)
                .eval(&d, "eval")
                .objective(&objective(k))
                .custom_metric(sums())
                .train()
        };
        assert!(run(2).is_ok());
        for k in [1, 3] {
            assert!(matches!(
                run(k),
                Err(HessboostError::InvalidParameter { name, .. }) if name == "objective"
            ));
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
        let d = labeled_dense(&x, 128, 2, &y);
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
