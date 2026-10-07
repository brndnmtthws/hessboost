//! The classic cyclic EBM: outer bags, round-robin term boosting, and
//! per-bag early stopping on held-out rows.

use super::fast::fast_pairs;
use super::{EBM_BAG_SALT, EBM_SALT, Grown, GrownTree, Hook, Term, gradients, grow, parallel};
use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::metric::Metric;
use crate::rng::{GOLDEN, Rng, splitmix64};
use crate::training::boulevard::tree_rows;
use crate::training::prepare::{Prepared, TrainContext};
use crate::training::row_sampling::bernoulli_rows;
use crate::tree::builder::all_rows;
use rayon::prelude::*;
use std::collections::VecDeque;

/// The rows of `pool` kept with probability `subsample` (at least one).
fn subsample_of(pool: &[u32], subsample: f64, rng: &mut Rng) -> Vec<u32> {
    if subsample >= 1.0 {
        return pool.to_vec();
    }
    let expected = pool.len() as f64 * subsample;
    bernoulli_rows(pool.len(), |i| pool[i], |_| subsample, expected, rng)
}

/// The rows of `pool` (whose queries are `row_queries`, out of `queries`)
/// in the queries kept with probability `fraction`, one draw per query in
/// query order. When no query of the pool is kept, the whole query of a
/// random pool row: a tree always trains on whole queries.
fn query_sample(
    pool: &[u32],
    row_queries: &[u32],
    queries: usize,
    fraction: f64,
    rng: &mut Rng,
) -> Vec<u32> {
    let mut kept: Vec<bool> = (0..queries).map(|_| rng.f64() < fraction).collect();
    let in_kept = |kept: &[bool]| -> Vec<u32> {
        pool.iter()
            .zip(row_queries)
            .filter(|&(_, &query)| kept[query as usize])
            .map(|(&row, _)| row)
            .collect()
    };
    let rows = in_kept(&kept);
    if !rows.is_empty() {
        return rows;
    }
    kept[row_queries[rng.range(0..pool.len())] as usize] = true;
    in_kept(&kept)
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

/// Where a round of [`Bag::cycle`] falls: its stage and its index in the
/// stage (which key the round's RNG), and its index among the reported
/// rounds ([`Hook`]).
#[derive(Clone, Copy)]
struct RoundAt {
    stage: u64,
    round: u64,
    reported: usize,
}

/// One outer bag of the classic EBM: its rows (with query bagging, each
/// row's query index and the query count), its margins over every training
/// row, its trees of both stages (final leaf values: learning rate, then
/// `1/B`), and with early stopping its held-out rows and the current
/// stage's stopper.
struct Bag {
    index: u64,
    rows: Vec<u32>,
    row_queries: Vec<u32>,
    queries: usize,
    margins: Vec<f32>,
    trees: Vec<GrownTree>,
    holdout: Option<Holdout>,
    stopper: Option<Stopper>,
    /// Trees this bag has grown over both stages: each tree's gradient
    /// iteration.
    grown: usize,
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
            let fraction = params.ebm_settings().bag_fraction();
            bernoulli_rows(n, |i| i as u32, |_| fraction, n as f64 * fraction, &mut rng)
        };
        let holdout = if params.ebm_settings().early_stopping().is_some() {
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
        // Query bagging keeps whole queries: each bag row's query index.
        let (row_queries, queries) = match (params.bagging_by_query, dtrain.group()) {
            (Some(_), Some(group)) => {
                let mut query_of = vec![0u32; n];
                let mut queries = 0;
                for (query, (start, end)) in group.iter_ranges().enumerate() {
                    query_of[start..end].fill(query as u32);
                    queries += 1;
                }
                let row_queries = rows.iter().map(|&r| query_of[r as usize]).collect();
                (row_queries, queries)
            }
            _ => (Vec::new(), 0),
        };
        Ok(Bag {
            index,
            rows,
            row_queries,
            queries,
            margins: vec![mu as f32; n],
            trees: Vec::new(),
            holdout,
            stopper: None,
            grown: 0,
        })
    }

    /// The bag rows one classic tree trains on: each kept with probability
    /// `subsample`; under class-balanced bagging with its class's fraction
    /// (a label-`1` row with `pos_fraction`, any other with
    /// `neg_fraction`); under query bagging the rows of the queries kept
    /// with probability `fraction` ([`query_sample`]). At least one row.
    fn tree_sample(&self, params: &TrainingParams, labels: &[f32], rng: &mut Rng) -> Vec<u32> {
        if let Some(bagging) = params.bagging_by_query {
            return query_sample(
                &self.rows,
                &self.row_queries,
                self.queries,
                bagging.fraction(),
                rng,
            );
        }
        let Some(bagging) = params.balanced_bagging else {
            return subsample_of(&self.rows, params.subsample, rng);
        };
        let (pos, neg) = (bagging.pos_fraction(), bagging.neg_fraction());
        let rows = &self.rows;
        let fraction = |row: u32| {
            if labels[row as usize] == 1.0 {
                pos
            } else {
                neg
            }
        };
        bernoulli_rows(
            rows.len(),
            |i| rows[i],
            fraction,
            rows.len() as f64 * pos.max(neg),
            rng,
        )
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
        let stopping = run.params.ebm_settings().early_stopping();
        self.stopper = scorer.zip(stopping).map(|(scorer, stopping)| {
            let capacity = stopping.rounds().get().saturating_mul(terms);
            let start = self.score(run, scorer);
            Stopper::new(capacity, stopping.tolerance(), start, self)
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

    /// Round `at` of cyclic boosting over `terms`: each tree fits the
    /// gradients of everything before it; with early stopping each tree is
    /// scored, and the round ends where the bag stops.
    fn cycle(
        &mut self,
        run: &TrainContext,
        prepared: &Prepared,
        terms: &[Term],
        at: RoundAt,
        scorer: Option<&Scorer>,
    ) {
        let RoundAt {
            stage,
            round,
            reported,
        } = at;
        let params = run.params;
        let eta = params.eta as f32;
        let n_bags = params.ebm_settings().outer_bags();
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
            let gpair = gradients(run, &self.margins, self.grown);
            self.grown += 1;
            let rows = self.tree_sample(params, labels, &mut rng);
            let seed = rng.next_u64();
            prepared.fill_approx_cache(run, &gpair);
            let mut tree = grow(run, prepared, &gpair, &rows, features, seed);
            tree.scale_leaves(eta);
            for (m, p) in self.margins.iter_mut().zip(tree_rows(&tree, run.dtrain)) {
                *m += p;
            }
            // The bag's own margins take the tree whole; the model its
            // `1/B` share.
            if n_bags > 1 {
                tree.scale_leaves(1.0 / n_bags as f32);
            }
            self.trees.push(GrownTree {
                round: reported,
                term,
                tree,
            });
            if let Some(scorer) = scorer {
                let score = self.score(run, scorer);
                if let Some(stopper) = &mut self.stopper {
                    stopper.observe(score, self.trees.len(), &self.margins);
                }
            }
        }
    }
}

/// Every bag's trees in model order: round by round and, within a round,
/// bag by bag in bag order, each bag's in term order (the sort is stable).
fn model_order(bags: &[Bag]) -> Vec<&GrownTree> {
    let mut trees: Vec<&GrownTree> = bags.iter().flat_map(|bag| &bag.trees).collect();
    trees.sort_by_key(|t| t.round);
    trees
}

/// Run up to `rounds` rounds of `terms` in every bag (the bags of a round
/// in parallel when allowed), adding each round's trees to the eval margins
/// in model order and reporting the round to `hook`, until the hook stops
/// or every bag has stopped early; then each bag keeps its best trees.
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
        let at = RoundAt {
            stage,
            round,
            reported: hook.done,
        };
        let f = |bag: &mut Bag| bag.cycle(run, prepared, terms, at, scorer);
        if parallel(run.params) && bags.len() > 1 {
            bags.par_iter_mut().for_each(f);
        } else {
            bags.iter_mut().for_each(f);
        }
        for bag in bags.iter() {
            let start = bag.trees.partition_point(|t| t.round < at.reported);
            for grown in &bag.trees[start..] {
                hook.add_tree(&grown.tree);
            }
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
/// main effects, then cyclic pairs per bag; every tree scaled by `1/B` and
/// the trees laid out round by round ([`model_order`]).
pub(super) fn classic(
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
    let metric = metric.filter(|_| params.ebm_settings().early_stopping().is_some());
    let scorer = metric.map(|metric| Scorer {
        metric,
        sign: if metric.maximize() { -1.0 } else { 1.0 },
    });
    // An early-stopped bag drops its trees past its best score when the
    // stage ends, so the pair stage boosts (and its eval margins restart)
    // from the kept main effects.
    let restart = (scorer.is_some() && params.ebm_settings().interactions() > 0)
        .then(|| hook.margins.evals.clone());
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
    let main_rounds = hook.done;
    let mut pairs = Vec::new();
    if params.ebm_settings().interactions() > 0 && !hook.stopped {
        if let Some(initial) = restart {
            hook.margins.evals = initial;
            for grown in model_order(&bags) {
                hook.add_tree(&grown.tree);
            }
        }
        let inv = 1.0 / n_bags as f64;
        let averaged: Vec<f32> = (0..n)
            .map(|i| {
                let sum: f64 = bags.iter().map(|b| f64::from(b.margins[i]) - mu).sum();
                (mu + sum * inv) as f32
            })
            .collect();
        pairs = fast_pairs(
            run,
            prepared,
            &gradients(run, &averaged, rounds),
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
    let mut trees: Vec<GrownTree> = bags.into_iter().flat_map(|bag| bag.trees).collect();
    // Model order ([`model_order`]); the sort is stable.
    trees.sort_by_key(|t| t.round);
    Ok(Grown {
        trees,
        pairs,
        main_rounds,
    })
}

#[cfg(test)]
mod tests {
    use super::query_sample;
    use crate::rng::Rng;

    /// With no query drawn, a tree still gets one whole query (its rows in
    /// the bag), never a lone row of a multi-row query.
    #[test]
    fn an_empty_query_draw_falls_back_to_a_whole_query() {
        // Queries 0 (rows 0..3), 1 (rows 3..10), 2 (rows 10..12); the bag
        // lacks row 4.
        let query_of = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2];
        let pool: Vec<u32> = (0..12).filter(|&r| r != 4).collect();
        let row_queries: Vec<u32> = pool.iter().map(|&r| query_of[r as usize]).collect();
        let whole = |q: u32| -> Vec<u32> {
            pool.iter()
                .copied()
                .filter(|&r| query_of[r as usize] == q)
                .collect()
        };
        let mut seen = [false; 3];
        for seed in 0..64 {
            let rows = query_sample(&pool, &row_queries, 3, 1e-12, &mut Rng::new(seed));
            let q = query_of[rows[0] as usize];
            assert_eq!(rows, whole(q), "seed {seed}");
            seen[q as usize] = true;
        }
        assert_eq!(seen, [true; 3]);
    }
}
