//! `booster = ebm`: cyclic GA²M boosting (classic, with outer bags) and the
//! Boulevard-averaged EBM of Fang, Tan, Pipping & Hooker (AISTATS 2026),
//! each followed by FAST pair selection and pair-term boosting. See
//! [`crate::ebm`] for the algorithms.

use rayon::prelude::*;

use super::boulevard::{Recursion, RoundRequest, Schedule, tree_rows};
use super::train::{Prepared, TrainContext, TreeSample, sample_rows};
use crate::config::{Device, TrainingParams};
use crate::data::DMatrix;
use crate::data::quantile::HistCuts;
use crate::ebm::{EbmBoulevard, EbmInfo};
use crate::error::{HessboostError, Result};
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

/// Train `rounds` EBM rounds of main effects, then of the
/// [`ebm_interactions`](TrainingParams::ebm_interactions) FAST pairs, into
/// `model` (which holds only the intercept) and record its [`EbmInfo`].
pub(super) fn boost(
    run: &TrainContext,
    prepared: &Prepared,
    model: &mut BoostedModel,
    rounds: usize,
) -> Result<()> {
    let params = run.params;
    let p = run.dtrain.n_cols();
    let max_pairs = p * p.saturating_sub(1) / 2;
    if params.ebm_interactions > max_pairs {
        return Err(HessboostError::invalid_param(
            "ebm_interactions",
            format!(
                "{p} features have {max_pairs} pairs, got {}",
                params.ebm_interactions
            ),
        ));
    }
    let mains: Vec<Vec<u32>> = (0..p as u32).map(|f| vec![f]).collect();
    let mu = f64::from(model.base_scores()[0]);
    let (Grown { trees, pairs }, boulevard) = if params.ebm_boulevard {
        let info = EbmBoulevard {
            learning_rate: params.eta,
            subsample: params.subsample,
            reg_lambda: params.lambda,
        };
        (boulevard(run, prepared, &mains, mu, rounds)?, Some(info))
    } else {
        (classic(run, prepared, &mains, mu, rounds), None)
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
    let mut rows: Vec<u32> = pool
        .iter()
        .copied()
        .filter(|_| rng.f64() < subsample)
        .collect();
    if rows.is_empty() {
        rows.push(pool[rng.range(0..pool.len())]);
    }
    rows
}

/// One outer bag of the classic EBM: its rows, its margins over every
/// training row, and its trees (learning rate applied, not yet `1/B`).
struct Bag {
    index: u64,
    rows: Vec<u32>,
    margins: Vec<f32>,
    trees: Vec<(u32, RegTree)>,
}

impl Bag {
    fn new(params: &TrainingParams, n: usize, index: usize, mu: f64) -> Self {
        let index = index as u64;
        let rows = if params.ebm_bag_fraction >= 1.0 {
            all_rows(n)
        } else {
            let mut rng = Rng::new(splitmix64(
                params.seed ^ EBM_BAG_SALT ^ index.wrapping_mul(GOLDEN),
            ));
            subsample_of(&all_rows(n), params.ebm_bag_fraction, &mut rng)
        };
        Bag {
            index,
            rows,
            margins: vec![mu as f32; n],
            trees: Vec::new(),
        }
    }

    /// Cyclic boosting of `terms` for `rounds` rounds: each tree fits the
    /// gradients of everything before it.
    fn cycle(
        &mut self,
        run: &TrainContext,
        prepared: &Prepared,
        terms: &[Term],
        rounds: usize,
        stage: u64,
    ) {
        let params = run.params;
        let eta = params.eta as f32;
        for round in 0..rounds as u64 {
            let key = params.seed
                ^ EBM_SALT
                ^ stage.wrapping_mul(GOLDEN)
                ^ splitmix64(self.index ^ round.wrapping_mul(GOLDEN));
            let mut rng = Rng::new(splitmix64(key));
            for &(term, features) in terms {
                let gpair = gradients(run, &self.margins);
                let rows = subsample_of(&self.rows, params.subsample, &mut rng);
                let seed = rng.next_u64();
                prepared.fill_approx_cache(run, &gpair);
                let mut tree = grow(run, prepared, &gpair, &rows, features, seed);
                tree.scale_leaves(eta);
                for (m, p) in self.margins.iter_mut().zip(tree_rows(&tree, run.dtrain)) {
                    *m += p;
                }
                self.trees.push((term, tree));
            }
        }
    }
}

/// Run `f` on every bag, in parallel when allowed, keeping bag order.
fn each_bag(params: &TrainingParams, bags: &mut [Bag], f: impl Fn(&mut Bag) + Sync + Send) {
    if parallel(params) && bags.len() > 1 {
        bags.par_iter_mut().for_each(f);
    } else {
        bags.iter_mut().for_each(f);
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
) -> Grown {
    let params = run.params;
    let n = run.dtrain.n_rows();
    let n_bags = params.ebm_outer_bags;
    let mut bags: Vec<Bag> = (0..n_bags).map(|b| Bag::new(params, n, b, mu)).collect();
    let main_terms: Vec<Term> = mains
        .iter()
        .enumerate()
        .map(|(t, f)| (t as u32, f.as_slice()))
        .collect();
    each_bag(params, &mut bags, |bag| {
        bag.cycle(run, prepared, &main_terms, rounds, 0);
    });
    let mut main_trees: Vec<(u32, RegTree)> = Vec::new();
    for bag in &mut bags {
        main_trees.append(&mut bag.trees);
    }
    let mut pairs = Vec::new();
    if params.ebm_interactions > 0 {
        let inv = 1.0 / n_bags as f64;
        let averaged: Vec<f32> = (0..n)
            .map(|i| {
                let sum: f64 = bags.iter().map(|b| f64::from(b.margins[i]) - mu).sum();
                (mu + sum * inv) as f32
            })
            .collect();
        pairs = fast_pairs(run, &gradients(run, &averaged), params.ebm_interactions);
        let pair_terms: Vec<Term> = pairs
            .iter()
            .enumerate()
            .map(|(k, f)| ((mains.len() + k) as u32, f.as_slice()))
            .collect();
        each_bag(params, &mut bags, |bag| {
            bag.cycle(run, prepared, &pair_terms, rounds, 1);
        });
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
    Grown { trees, pairs }
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
        boulevard_stage(run, prepared, &main_terms, &base, rounds, 0)?;
    let mut pairs = Vec::new();
    if params.ebm_interactions > 0 {
        let margins: Vec<f32> = fitted.iter().map(|&m| m as f32).collect();
        pairs = fast_pairs(run, &gradients(run, &margins), params.ebm_interactions);
        let pair_terms: Vec<Term> = pairs
            .iter()
            .enumerate()
            .map(|(k, f)| ((mains.len() + k) as u32, f.as_slice()))
            .collect();
        trees.extend(boulevard_stage(run, prepared, &pair_terms, &fitted, rounds, 1)?.trees);
    }
    Ok(Grown { trees, pairs })
}

/// One Boulevard stage over `terms` from the per-row margins `base`: every
/// round's trees fit the same residuals of `base` plus the stage's current
/// average, each centered on the training rows, and the stage's leaves end
/// scaled by Algorithm 1's `(1 + λ)/λ · λ/B`.
fn boulevard_stage(
    run: &TrainContext,
    prepared: &Prepared,
    terms: &[Term],
    base: &[f64],
    rounds: usize,
    stage: u64,
) -> Result<StageFit> {
    let TrainContext { params, dtrain, .. } = *run;
    let n = dtrain.n_rows();
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
            let rows: Vec<Vec<u32>> = terms.iter().map(|_| sample_rows(n, params, rng)).collect();
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
/// both, and return the top `k` (ties broken by feature order).
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

/// Refuse the training data `booster = ebm` cannot use: categorical
/// features (shape functions need thresholds) and feature weights (no
/// column sampling).
pub(super) fn validate_data(dtrain: &DMatrix) -> Result<()> {
    if dtrain
        .feature_types()
        .contains(&crate::data::FeatureType::Categorical)
    {
        return Err(HessboostError::invalid_param(
            "feature_types",
            "`booster = ebm` supports numerical features only",
        ));
    }
    if dtrain.feature_weights().is_some() {
        return Err(HessboostError::invalid_param(
            "feature_weights",
            "`booster = ebm` does not sample columns",
        ));
    }
    Ok(())
}
