//! `booster = ebm`: cyclic GA²M boosting (classic, with outer bags) and the
//! Boulevard-averaged EBM of Fang, Tan, Pipping & Hooker (AISTATS 2026),
//! each followed by FAST pair selection and pair-term boosting. See
//! [`crate::ebm`] for the algorithms.

use std::collections::VecDeque;
use std::ops::ControlFlow;

use rayon::prelude::*;

use super::boulevard::{Recursion, RoundRequest, Schedule, tree_rows};
use super::train::{Prepared, TrainContext, TreeSample, sample_rows};
use crate::config::{Device, TrainingParams};
use crate::data::DMatrix;
use crate::data::quantile::HistCuts;
use crate::ebm::{EbmBoulevard, EbmInfo};
use crate::error::{HessboostError, Result};
use crate::metric::Metric;
use crate::model::BoostedModel;
use crate::objective::GradPair;
use crate::rng::{GOLDEN, Rng, splitmix64};
use crate::tree::RegTree;
use crate::tree::builder::all_rows;
use crate::tree::sampler::ColumnSampler;

/// The classic EBM's round RNG salt.
const EBM_SALT: u64 = 0xEB_0C7C;
/// The Boulevard EBM's recursion salt (per stage: `^ stage`).
const EBM_BOULEVARD_SALT: u64 = 0xEBB0_07E7;
/// The outer-bag row draw's salt.
const EBM_BAG_SALT: u64 = 0xEB_BA6;

/// One term being boosted: its index in [`EbmInfo::terms`] and features.
type Term<'a> = (u32, &'a [u32]);

/// One Boulevard stage's trees (with their terms) and the fitted margins
/// `base + stage` over the training rows.
struct StageFit {
    trees: Vec<(u32, RegTree)>,
    fitted: Vec<f64>,
}

/// What a run grows: every tree with its term, and the pair terms FAST
/// picked.
struct Grown {
    trees: Vec<(u32, RegTree)>,
    pairs: Vec<Vec<u32>>,
}

/// Grow one tree of `features` on `gpair` over `rows`, raw (no learning
/// rate).
fn grow(
    run: &TrainContext,
    prepared: &Prepared,
    gpair: &[GradPair],
    rows: &[u32],
    features: &[u32],
    seed: u64,
) -> RegTree {
    let mut sampler = ColumnSampler::only(features.to_vec(), seed);
    let sample = TreeSample {
        gpair,
        rows,
        forest_index: None,
    };
    prepared
        .build_tree(run, sample, &mut sampler, None, seed, false)
        .0
}

/// Whether independent trees or bags may be grown in parallel: several
/// threads and the CPU backend (the Metal backend serves one tree at a time).
fn parallel(params: &TrainingParams) -> bool {
    params.device == Device::Cpu && rayon::current_num_threads() > 1
}

/// The per-round hook of training ([`Trainer::on_round`](super::Trainer::on_round))
/// across the stages: rounds are numbered from 0 through the main-effect
/// stage and on through the pair stage, and a `Break` ends training.
struct Hook<'h> {
    after_round: &'h mut dyn FnMut(usize) -> ControlFlow<()>,
    done: usize,
    stopped: bool,
}

impl Hook<'_> {
    /// Report a finished round; whether training goes on.
    fn next(&mut self) -> bool {
        self.stopped |= (self.after_round)(self.done).is_break();
        self.done += 1;
        !self.stopped
    }
}

/// Train `rounds` EBM rounds of main effects, then of the
/// [`Ebm::interactions`](crate::config::Ebm::interactions) FAST pairs, into
/// `model` (which holds only the intercept) and record its [`EbmInfo`],
/// calling `after_round` after every round of either stage. A `Break` stops
/// training there: the model keeps the completed rounds (a stopped
/// Boulevard stage averages those), and a stop in the main-effect stage
/// skips the pairs.
pub(super) fn boost(
    run: &TrainContext,
    prepared: &Prepared,
    model: &mut BoostedModel,
    rounds: usize,
    metric: Option<&dyn Metric>,
    after_round: &mut dyn FnMut(usize) -> ControlFlow<()>,
) -> Result<()> {
    let mut hook = Hook {
        after_round,
        done: 0,
        stopped: false,
    };
    let params = run.params;
    let p = run.dtrain.n_cols();
    let max_pairs = p * p.saturating_sub(1) / 2;
    if params.ebm_settings().interactions() > max_pairs {
        return Err(HessboostError::invalid_param(
            "ebm_interactions",
            format!(
                "{p} features have {max_pairs} pairs, got {}",
                params.ebm_settings().interactions()
            ),
        ));
    }
    let mains: Vec<Vec<u32>> = (0..p as u32).map(|f| vec![f]).collect();
    let mu = f64::from(model.base_scores()[0]);
    let (Grown { trees, pairs }, boulevard) = if params.ebm_settings().boulevard() {
        let info = EbmBoulevard {
            learning_rate: params.eta,
            subsample: params.subsample,
            reg_lambda: params.lambda,
        };
        (
            boulevard(run, prepared, &mains, mu, rounds, &mut hook)?,
            Some(info),
        )
    } else {
        (
            classic(run, prepared, &mains, mu, rounds, metric, &mut hook)?,
            None,
        )
    };
    let mut terms = mains;
    terms.extend(pairs);
    let mut tree_terms = Vec::with_capacity(trees.len());
    for (term, tree) in trees {
        tree_terms.push(term);
        model.push_tree_weighted(tree, 1.0);
    }
    let mut info = EbmInfo {
        terms,
        tree_terms,
        term_means: Vec::new(),
        boulevard,
    };
    info.term_means = info.term_means_on(model, run.dtrain);
    model.set_ebm(Some(info));
    Ok(())
}

/// Every row's gradients at `margins`.
fn gradients(run: &TrainContext, margins: &[f32]) -> Vec<GradPair> {
    let mut gpair = vec![GradPair::default(); margins.len()];
    run.objective.gradient_info(margins, run.info, &mut gpair);
    gpair
}

/// The rows of `pool` kept with probability `subsample` (at least one).
fn subsample_of(pool: &[u32], subsample: f64, rng: &mut Rng) -> Vec<u32> {
    if subsample >= 1.0 {
        return pool.to_vec();
    }
    bernoulli_rows(pool, rng, |_| subsample)
}

/// The rows of an outer bag's `pool` one classic tree trains on: each kept
/// with probability `subsample`, or under class-balanced bagging with its
/// class's fraction (a label-`1` row with `pos_fraction`, any other with
/// `neg_fraction`), at least one.
fn tree_sample(pool: &[u32], params: &TrainingParams, labels: &[f32], rng: &mut Rng) -> Vec<u32> {
    let Some(bagging) = params.balanced_bagging else {
        return subsample_of(pool, params.subsample, rng);
    };
    let (pos, neg) = (bagging.pos_fraction(), bagging.neg_fraction());
    bernoulli_rows(pool, rng, |row| {
        if labels[row as usize] == 1.0 {
            pos
        } else {
            neg
        }
    })
}

/// The rows of `pool`, each kept with probability `keep(row)`, in pool
/// order (one draw per row); a random row of `pool` if none is kept.
fn bernoulli_rows(pool: &[u32], rng: &mut Rng, keep: impl Fn(u32) -> f64) -> Vec<u32> {
    let mut rows: Vec<u32> = pool
        .iter()
        .copied()
        .filter(|&row| rng.f64() < keep(row))
        .collect();
    if rows.is_empty() {
        rows.push(pool[rng.range(0..pool.len())]);
    }
    rows
}

/// The rows an outer bag does not train on, which it early-stops on.
struct Holdout {
    rows: Vec<u32>,
    data: DMatrix,
}

/// A bag's early stopping in one stage (InterpretML's rule): after every
/// tree, the held-out score `m` (lower is better); stop once the best score
/// of the last `window` trees fails to beat the best before them by the
/// relative tolerance; keep the trees up to the best score.
struct Stopper {
    window: VecDeque<f64>,
    capacity: usize,
    tolerance: f64,
    /// Best score of every tree so far, and of the trees before the window.
    min_all: f64,
    min_before: f64,
    /// The best score (the stage's starting model included), how many of
    /// the bag's trees it had, and its margins.
    best: f64,
    best_len: usize,
    best_margins: Vec<f32>,
    done: bool,
}

impl Stopper {
    fn new(capacity: usize, tolerance: f64, start: f64, bag: &Bag) -> Self {
        Stopper {
            // Grown as trees arrive: the patience may exceed any run.
            window: VecDeque::new(),
            capacity,
            tolerance,
            min_all: f64::INFINITY,
            min_before: f64::INFINITY,
            best: start,
            best_len: bag.trees.len(),
            best_margins: bag.margins.clone(),
            done: false,
        }
    }

    /// Record the score `m` of the model with `len` trees and `margins`.
    fn observe(&mut self, m: f64, len: usize, margins: &[f32]) {
        let m = if m.is_nan() { f64::INFINITY } else { m };
        if m < self.best {
            self.best = m;
            self.best_len = len;
            self.best_margins.copy_from_slice(margins);
        }
        let mut tolerance = self.min_all.abs().min(self.min_before.abs()) * self.tolerance;
        if !tolerance.is_finite() {
            tolerance = 0.0;
        }
        self.min_all = self.min_all.min(m);
        if self.window.len() == self.capacity
            && let Some(oldest) = self.window.pop_front()
        {
            self.min_before = self.min_before.min(oldest);
        }
        self.window.push_back(m);
        let recent = self.window.iter().copied().fold(f64::INFINITY, f64::min);
        if self.window.len() == self.capacity && self.min_before - tolerance <= recent {
            self.done = true;
        }
    }
}

/// How a bag scores its held-out rows: the early-stopping metric and
/// whether it is maximized (scores are negated then, so lower is better).
struct Scorer<'m> {
    metric: &'m dyn Metric,
    sign: f64,
}

/// One outer bag of the classic EBM: its rows, its margins over every
/// training row, its trees (learning rate applied, not yet `1/B`), and with
/// early stopping its held-out rows and the current stage's stopper.
struct Bag {
    index: u64,
    rows: Vec<u32>,
    margins: Vec<f32>,
    trees: Vec<(u32, RegTree)>,
    holdout: Option<Holdout>,
    stopper: Option<Stopper>,
}

impl Bag {
    fn new(run: &TrainContext, index: usize, mu: f64) -> Result<Self> {
        let TrainContext { params, dtrain, .. } = *run;
        let n = dtrain.n_rows();
        let index = index as u64;
        let rows = if params.ebm_settings().bag_fraction() >= 1.0 {
            all_rows(n)
        } else {
            let mut rng = Rng::new(splitmix64(
                params.seed ^ EBM_BAG_SALT ^ index.wrapping_mul(GOLDEN),
            ));
            subsample_of(&all_rows(n), params.ebm_settings().bag_fraction(), &mut rng)
        };
        let holdout = if params.ebm_settings().early_stopping_rounds() > 0 {
            let mut in_bag = vec![false; n];
            for &r in &rows {
                in_bag[r as usize] = true;
            }
            let held: Vec<usize> = (0..n).filter(|&r| !in_bag[r]).collect();
            if held.is_empty() {
                return Err(HessboostError::invalid_param(
                    "ebm_early_stopping_rounds",
                    format!(
                        "outer bag {index} holds out no rows to stop on; lower `ebm_bag_fraction`"
                    ),
                ));
            }
            Some(Holdout {
                rows: held.iter().map(|&r| r as u32).collect(),
                data: dtrain.select_rows(&held)?,
            })
        } else {
            None
        };
        Ok(Bag {
            index,
            rows,
            margins: vec![mu as f32; n],
            trees: Vec::new(),
            holdout,
            stopper: None,
        })
    }

    /// The held-out score of the bag's current margins (lower is better).
    fn score(&self, run: &TrainContext, scorer: &Scorer) -> f64 {
        let Some(holdout) = &self.holdout else {
            return 0.0;
        };
        let mut preds: Vec<f32> = holdout
            .rows
            .iter()
            .map(|&r| self.margins[r as usize])
            .collect();
        run.objective.eval_transform(&mut preds);
        scorer.sign * scorer.metric.eval_info(&preds, &holdout.data.info())
    }

    /// Start a stage of `terms` terms: with early stopping, a stopper
    /// seeded with the current model.
    fn start_stage(&mut self, run: &TrainContext, scorer: Option<&Scorer>, terms: usize) {
        self.stopper = scorer.map(|scorer| {
            let capacity = run
                .params
                .ebm_settings()
                .early_stopping_rounds()
                .saturating_mul(terms);
            let start = self.score(run, scorer);
            Stopper::new(
                capacity,
                run.params.ebm_settings().early_stopping_tolerance(),
                start,
                self,
            )
        });
    }

    /// End a stage: an early-stopped bag keeps its trees up to its best
    /// held-out score.
    fn finish_stage(&mut self) {
        if let Some(stopper) = self.stopper.take() {
            self.trees.truncate(stopper.best_len);
            self.margins = stopper.best_margins;
        }
    }

    /// Whether the bag's current stage has stopped early.
    fn stopped(&self) -> bool {
        self.stopper.as_ref().is_some_and(|s| s.done)
    }

    /// Round `round` of cyclic boosting over `terms`: each tree fits the
    /// gradients of everything before it; with early stopping each tree is
    /// scored, and the round ends where the bag stops.
    fn cycle(
        &mut self,
        run: &TrainContext,
        prepared: &Prepared,
        terms: &[Term],
        round: (u64, u64),
        scorer: Option<&Scorer>,
    ) {
        let (stage, round) = round;
        let params = run.params;
        let eta = params.eta as f32;
        let key = params.seed
            ^ EBM_SALT
            ^ stage.wrapping_mul(GOLDEN)
            ^ splitmix64(self.index ^ round.wrapping_mul(GOLDEN));
        let mut rng = Rng::new(splitmix64(key));
        let labels = run.dtrain.labels().unwrap_or_default();
        for &(term, features) in terms {
            if self.stopped() {
                return;
            }
            let gpair = gradients(run, &self.margins);
            let rows = tree_sample(&self.rows, params, labels, &mut rng);
            let seed = rng.next_u64();
            prepared.fill_approx_cache(run, &gpair);
            let mut tree = grow(run, prepared, &gpair, &rows, features, seed);
            tree.scale_leaves(eta);
            for (m, p) in self.margins.iter_mut().zip(tree_rows(&tree, run.dtrain)) {
                *m += p;
            }
            self.trees.push((term, tree));
            if let Some(scorer) = scorer {
                let score = self.score(run, scorer);
                if let Some(stopper) = &mut self.stopper {
                    stopper.observe(score, self.trees.len(), &self.margins);
                }
            }
        }
    }
}

/// Run up to `rounds` rounds of `terms` in every bag (the bags of a round
/// in parallel when allowed), reporting each round to `hook`, until the
/// hook stops or every bag has stopped early; then each bag keeps its best
/// trees.
fn cycle_bags(
    run: &TrainContext,
    prepared: &Prepared,
    bags: &mut [Bag],
    terms: &[Term],
    stage: (u64, usize),
    scorer: Option<&Scorer>,
    hook: &mut Hook,
) {
    let (stage, rounds) = stage;
    for bag in bags.iter_mut() {
        bag.start_stage(run, scorer, terms.len());
    }
    for round in 0..rounds as u64 {
        let f = |bag: &mut Bag| bag.cycle(run, prepared, terms, (stage, round), scorer);
        if parallel(run.params) && bags.len() > 1 {
            bags.par_iter_mut().for_each(f);
        } else {
            bags.iter_mut().for_each(f);
        }
        if !hook.next() || bags.iter().all(Bag::stopped) {
            break;
        }
    }
    for bag in bags.iter_mut() {
        bag.finish_stage();
    }
}

/// The classic EBM: cyclic main effects per bag, FAST on the bag-averaged
/// main effects, then cyclic pairs per bag; every tree scaled by `1/B`.
fn classic(
    run: &TrainContext,
    prepared: &Prepared,
    mains: &[Vec<u32>],
    mu: f64,
    rounds: usize,
    metric: Option<&dyn Metric>,
    hook: &mut Hook,
) -> Result<Grown> {
    let params = run.params;
    let n = run.dtrain.n_rows();
    let n_bags = params.ebm_settings().outer_bags();
    let mut bags = (0..n_bags)
        .map(|b| Bag::new(run, b, mu))
        .collect::<Result<Vec<Bag>>>()?;
    let metric = metric.filter(|_| params.ebm_settings().early_stopping_rounds() > 0);
    let scorer = metric.map(|metric| Scorer {
        metric,
        sign: if metric.maximize() { -1.0 } else { 1.0 },
    });
    let main_terms: Vec<Term> = mains
        .iter()
        .enumerate()
        .map(|(t, f)| (t as u32, f.as_slice()))
        .collect();
    cycle_bags(
        run,
        prepared,
        &mut bags,
        &main_terms,
        (0, rounds),
        scorer.as_ref(),
        hook,
    );
    let mut main_trees: Vec<(u32, RegTree)> = Vec::new();
    for bag in &mut bags {
        main_trees.append(&mut bag.trees);
    }
    let mut pairs = Vec::new();
    if params.ebm_settings().interactions() > 0 && !hook.stopped {
        let inv = 1.0 / n_bags as f64;
        let averaged: Vec<f32> = (0..n)
            .map(|i| {
                let sum: f64 = bags.iter().map(|b| f64::from(b.margins[i]) - mu).sum();
                (mu + sum * inv) as f32
            })
            .collect();
        pairs = fast_pairs(
            run,
            &gradients(run, &averaged),
            params.ebm_settings().interactions(),
        );
        let pair_terms: Vec<Term> = pairs
            .iter()
            .enumerate()
            .map(|(k, f)| ((mains.len() + k) as u32, f.as_slice()))
            .collect();
        cycle_bags(
            run,
            prepared,
            &mut bags,
            &pair_terms,
            (1, rounds),
            scorer.as_ref(),
            hook,
        );
    }
    let mut trees = main_trees;
    for bag in &mut bags {
        trees.append(&mut bag.trees);
    }
    if n_bags > 1 {
        let inv = 1.0 / n_bags as f32;
        for (_, tree) in &mut trees {
            tree.scale_leaves(inv);
        }
    }
    Ok(Grown { trees, pairs })
}

/// The Boulevard EBM: a Boulevard stage of the main effects from the
/// intercept, FAST on its residual gradients, then a Boulevard stage of the
/// pairs from the first stage's fit.
fn boulevard(
    run: &TrainContext,
    prepared: &Prepared,
    mains: &[Vec<u32>],
    mu: f64,
    rounds: usize,
    hook: &mut Hook,
) -> Result<Grown> {
    let params = run.params;
    let n = run.dtrain.n_rows();
    let main_terms: Vec<Term> = mains
        .iter()
        .enumerate()
        .map(|(t, f)| (t as u32, f.as_slice()))
        .collect();
    let base = vec![mu; n];
    let StageFit { mut trees, fitted } =
        boulevard_stage(run, prepared, &main_terms, &base, (0, rounds), hook)?;
    let mut pairs = Vec::new();
    if params.ebm_settings().interactions() > 0 && !hook.stopped {
        let margins: Vec<f32> = fitted.iter().map(|&m| m as f32).collect();
        pairs = fast_pairs(
            run,
            &gradients(run, &margins),
            params.ebm_settings().interactions(),
        );
        let pair_terms: Vec<Term> = pairs
            .iter()
            .enumerate()
            .map(|(k, f)| ((mains.len() + k) as u32, f.as_slice()))
            .collect();
        let stage = boulevard_stage(run, prepared, &pair_terms, &fitted, (1, rounds), hook)?;
        trees.extend(stage.trees);
    }
    Ok(Grown { trees, pairs })
}

/// One Boulevard stage over `terms` from the per-row margins `base`: every
/// round's trees fit the same residuals of `base` plus the stage's current
/// average, each centered on the training rows, and the stage's leaves end
/// scaled by Algorithm 1's `(1 + λ)/λ · λ/B` for the `B` rounds run
/// (`stage` is the stage index and its rounds; `hook` may stop it early).
fn boulevard_stage(
    run: &TrainContext,
    prepared: &Prepared,
    terms: &[Term],
    base: &[f64],
    stage: (u64, usize),
    hook: &mut Hook,
) -> Result<StageFit> {
    let (stage, rounds) = stage;
    let TrainContext { params, dtrain, .. } = *run;
    let n = dtrain.n_rows();
    let labels = dtrain.labels().unwrap_or_default();
    let schedule = Schedule {
        dropout: 0.0,
        learning_rate: params.eta,
        truncation: 0.0,
        parallel: 1,
        seed: params.seed,
        salt: EBM_BOULEVARD_SALT ^ stage,
    };
    let mut recursion = Recursion::new(schedule, n);
    let mut trees: Vec<(u32, RegTree)> = Vec::with_capacity(rounds * terms.len());
    let mut total = vec![0.0f64; n];
    for _ in 0..rounds {
        recursion.step(|request| {
            let RoundRequest { offsets, rng, .. } = request;
            let margins: Vec<f32> = base
                .iter()
                .zip(&offsets[0])
                .map(|(&b, &o)| (b + o) as f32)
                .collect();
            let gpair = gradients(run, &margins);
            let rows: Vec<Vec<u32>> = terms
                .iter()
                .map(|_| sample_rows(n, params, labels, rng))
                .collect();
            let seeds: Vec<u64> = terms.iter().map(|_| rng.next_u64()).collect();
            prepared.fill_approx_cache(run, &gpair);
            let build = |k: usize| grow(run, prepared, &gpair, &rows[k], terms[k].1, seeds[k]);
            let grown: Vec<RegTree> = if parallel(params) && terms.len() > 1 {
                (0..terms.len()).into_par_iter().map(build).collect()
            } else {
                (0..terms.len()).map(build).collect()
            };
            let mut round_sum = vec![0.0f64; n];
            for (&(term, _), mut tree) in terms.iter().zip(grown) {
                let preds = tree_rows(&tree, dtrain);
                let mean = preds.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
                tree.shift_leaves(-mean as f32);
                for (s, v) in round_sum.iter_mut().zip(tree_rows(&tree, dtrain)) {
                    *s += f64::from(v);
                }
                trees.push((term, tree));
            }
            for (t, &s) in total.iter_mut().zip(&round_sum) {
                *t += s;
            }
            Ok(vec![round_sum.iter().map(|&s| s as f32).collect()])
        })?;
        if !hook.next() {
            break;
        }
    }
    let scale = recursion.scale();
    for (_, tree) in &mut trees {
        tree.scale_leaves(scale as f32);
    }
    let fitted = base
        .iter()
        .zip(&total)
        .map(|(&b, &t)| b + scale * t)
        .collect();
    Ok(StageFit { trees, fitted })
}

/// FAST (Lou, Caruana, Gehrke & Hooker, KDD 2013): rank every feature pair
/// by its best four-quadrant split of `gpair` on the features' histogram
/// bins, `Σ_q G_q² / (H_q + λ) − G² / (H + λ)` over the rows present in
/// both, and return the top `k` (ties broken by feature order). A
/// categorical feature has one bin per category, ordered by the category's
/// mean gradient `G / (H + λ)` (the order in which a binary partition's
/// best split is a cut, as in LightGBM's and XGBoost's categorical search).
fn fast_pairs(run: &TrainContext, gpair: &[GradPair], k: usize) -> Vec<Vec<u32>> {
    let TrainContext { params, dtrain, .. } = *run;
    let p = dtrain.n_cols();
    let cuts = HistCuts::from_dmatrix(dtrain, params.max_bin);
    let bins: Vec<Vec<u32>> = (0..p)
        .into_par_iter()
        .map(|f| {
            let start = cuts.feature_bins(f).0 as u32;
            (0..dtrain.n_rows())
                .map(|row| match dtrain.get(row, f) {
                    Some(v) if !v.is_nan() => cuts.bin_of(f, v) - start,
                    _ => u32::MAX,
                })
                .collect()
        })
        .collect();
    let lambda = params.lambda;
    let bins: Vec<Vec<u32>> = bins
        .into_iter()
        .enumerate()
        .map(|(f, b)| {
            if cuts.is_categorical(f) {
                order_categories(b, cuts.num_bins(f), gpair, lambda)
            } else {
                b
            }
        })
        .collect();
    let pairs: Vec<(usize, usize)> = (0..p)
        .flat_map(|a| (a + 1..p).map(move |b| (a, b)))
        .collect();
    let mut scored: Vec<(f64, usize)> = pairs
        .par_iter()
        .enumerate()
        .map(|(i, &(a, b))| {
            (
                pair_gain(
                    &bins[a],
                    &bins[b],
                    cuts.num_bins(a),
                    cuts.num_bins(b),
                    gpair,
                    lambda,
                ),
                i,
            )
        })
        .collect();
    scored.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    scored
        .into_iter()
        .take(k)
        .map(|(_, i)| vec![pairs[i].0 as u32, pairs[i].1 as u32])
        .collect()
}

/// Renumber a categorical feature's bins by their mean gradient
/// `G / (H + λ)` (ties by bin), so FAST's cuts over them are the binary
/// partitions worth scoring.
fn order_categories(bins: Vec<u32>, m: usize, gpair: &[GradPair], lambda: f64) -> Vec<u32> {
    let mut g = vec![0.0f64; m];
    let mut h = vec![0.0f64; m];
    for (&b, gp) in bins.iter().zip(gpair) {
        if b != u32::MAX {
            g[b as usize] += f64::from(gp.grad);
            h[b as usize] += f64::from(gp.hess);
        }
    }
    let ratio = |b: usize| {
        if h[b] + lambda > 0.0 {
            g[b] / (h[b] + lambda)
        } else {
            0.0
        }
    };
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| ratio(a).total_cmp(&ratio(b)).then(a.cmp(&b)));
    let mut rank = vec![0u32; m];
    for (r, &b) in order.iter().enumerate() {
        rank[b] = r as u32;
    }
    bins.into_iter()
        .map(|b| if b == u32::MAX { b } else { rank[b as usize] })
        .collect()
}

/// FAST's score of one pair (`bins` per row, `u32::MAX` missing).
fn pair_gain(
    bins_a: &[u32],
    bins_b: &[u32],
    ma: usize,
    mb: usize,
    gpair: &[GradPair],
    lambda: f64,
) -> f64 {
    // Prefix sums over the `ma × mb` bin histogram, `(ma + 1) × (mb + 1)`.
    let s = mb + 1;
    let mut g = vec![0.0f64; (ma + 1) * s];
    let mut h = vec![0.0f64; (ma + 1) * s];
    for ((&a, &b), gp) in bins_a.iter().zip(bins_b).zip(gpair) {
        if a == u32::MAX || b == u32::MAX {
            continue;
        }
        let cell = (a as usize + 1) * s + b as usize + 1;
        g[cell] += f64::from(gp.grad);
        h[cell] += f64::from(gp.hess);
    }
    for arr in [&mut g, &mut h] {
        for i in 1..=ma {
            for j in 1..=mb {
                arr[i * s + j] +=
                    arr[(i - 1) * s + j] + arr[i * s + j - 1] - arr[(i - 1) * s + j - 1];
            }
        }
    }
    let score = |gs: f64, hs: f64| {
        if hs + lambda > 0.0 {
            gs * gs / (hs + lambda)
        } else {
            0.0
        }
    };
    let (gt, ht) = (g[ma * s + mb], h[ma * s + mb]);
    let mut best = score(gt, ht);
    for i in 1..ma {
        for j in 1..mb {
            let (g00, h00) = (g[i * s + j], h[i * s + j]);
            let (g0, h0) = (g[i * s + mb], h[i * s + mb]);
            let (g1, h1) = (g[ma * s + j], h[ma * s + j]);
            let quadrants = score(g00, h00)
                + score(g0 - g00, h0 - h00)
                + score(g1 - g00, h1 - h00)
                + score(gt - g0 - g1 + g00, ht - h0 - h1 + h00);
            best = best.max(quadrants);
        }
    }
    best - score(gt, ht)
}

/// Refuse the training data `booster = ebm` cannot use: feature weights
/// (it samples no columns).
pub(super) fn validate_data(dtrain: &DMatrix) -> Result<()> {
    if dtrain.feature_weights().is_some() {
        return Err(HessboostError::invalid_param(
            "feature_weights",
            "`booster = ebm` does not sample columns",
        ));
    }
    Ok(())
}
