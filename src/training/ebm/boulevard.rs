//! The Boulevard EBM: each stage's terms boosted on the shared
//! Boulevard `Recursion`.

use super::fast::fast_pairs;
use super::{
    EBM_BOULEVARD_SALT, Grown, GrownTree, Hook, StageFit, Term, gradients, grow, parallel,
};
use crate::error::Result;
use crate::training::boulevard::{Recursion, RoundRequest, Schedule, tree_rows};
use crate::training::prepare::{Prepared, TrainContext};
use crate::training::row_sampling::sample_rows;
use crate::tree::RegTree;
use rayon::prelude::*;

/// The Boulevard EBM: a Boulevard stage of the main effects from the
/// intercept, FAST on its residual gradients, then a Boulevard stage of the
/// pairs from the first stage's fit.
pub(super) fn boulevard(
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
    let eval_base: Vec<Vec<f64>> = hook
        .eval_data
        .iter()
        .map(|d| vec![mu; d.n_rows()])
        .collect();
    let StageFit {
        mut trees,
        fitted,
        eval_fitted,
    } = boulevard_stage(
        run,
        prepared,
        &main_terms,
        (&base, &eval_base),
        (0, rounds),
        hook,
    )?;
    let main_rounds = hook.done;
    let mut pairs = Vec::new();
    if params.ebm_settings().interactions() > 0 && !hook.stopped {
        let margins: Vec<f32> = fitted.iter().map(|&m| m as f32).collect();
        pairs = fast_pairs(
            run,
            prepared,
            &gradients(run, &margins, rounds),
            params.ebm_settings().interactions(),
        );
        let pair_terms: Vec<Term> = pairs
            .iter()
            .enumerate()
            .map(|(k, f)| ((mains.len() + k) as u32, f.as_slice()))
            .collect();
        let stage = boulevard_stage(
            run,
            prepared,
            &pair_terms,
            (&fitted, &eval_fitted),
            (1, rounds),
            hook,
        )?;
        trees.extend(stage.trees);
    }
    Ok(Grown {
        trees,
        pairs,
        main_rounds,
    })
}

/// One Boulevard stage over `terms` from the per-row margins `base` (of the
/// training rows, then of every eval set's): every round's trees fit the
/// same residuals of the training base plus the stage's current average,
/// each centered on the training rows, and the stage's leaves end scaled
/// by Algorithm 1's `(1 + λ)/λ · λ/B` for the `B` rounds run (`stage` is
/// the stage index and its rounds; `hook` may stop it early). After every
/// round the eval margins are `base + scale · raw sums` in `f64`, as
/// `booster = boulevard` computes them.
fn boulevard_stage(
    run: &TrainContext,
    prepared: &Prepared,
    terms: &[Term],
    base: (&[f64], &[Vec<f64>]),
    stage: (u64, usize),
    hook: &mut Hook,
) -> Result<StageFit> {
    let (base, eval_base) = base;
    let (stage, rounds) = stage;
    let TrainContext { params, dtrain, .. } = *run;
    let n = dtrain.n_rows();
    let schedule = Schedule {
        dropout: 0.0,
        learning_rate: params.eta,
        truncation: None,
        parallel: 1,
        seed: params.seed,
        salt: EBM_BOULEVARD_SALT ^ stage,
    };
    let mut recursion = Recursion::new(schedule, n);
    let mut trees: Vec<GrownTree> = Vec::with_capacity(rounds * terms.len());
    let mut total = vec![0.0f64; n];
    let eval_data = hook.eval_data.clone();
    let mut eval_total: Vec<Vec<f64>> = eval_base.iter().map(|b| vec![0.0; b.len()]).collect();
    for round in 0..rounds {
        // Continuous across the two stages, as the round hook counts.
        let iteration = stage as usize * rounds + round;
        let reported = hook.done;
        recursion.step(|request| {
            let RoundRequest { offsets, rng, .. } = request;
            let margins: Vec<f32> = base
                .iter()
                .zip(&offsets[0])
                .map(|(&b, &o)| (b + o) as f32)
                .collect();
            let gpair = gradients(run, &margins, iteration);
            let rows: Vec<Vec<u32>> = terms
                .iter()
                .map(|_| sample_rows(n, params, run.rows, rng))
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
                for (sums, data) in eval_total.iter_mut().zip(&eval_data) {
                    for (s, v) in sums.iter_mut().zip(tree_rows(&tree, data)) {
                        *s += f64::from(v);
                    }
                }
                trees.push(GrownTree {
                    round: reported,
                    term,
                    tree,
                });
            }
            for (t, &s) in total.iter_mut().zip(&round_sum) {
                *t += s;
            }
            Ok(vec![round_sum.iter().map(|&s| s as f32).collect()])
        })?;
        let scale = recursion.scale();
        for ((margins, base), sums) in hook
            .margins
            .evals
            .iter_mut()
            .zip(eval_base)
            .zip(&eval_total)
        {
            for ((m, &b), &s) in margins.iter_mut().zip(base).zip(sums) {
                *m = (b + scale * s) as f32;
            }
        }
        if !hook.next() {
            break;
        }
    }
    let scale = recursion.scale();
    for GrownTree { tree, .. } in &mut trees {
        tree.scale_leaves(scale as f32);
    }
    let fit = |base: &[f64], total: &[f64]| -> Vec<f64> {
        base.iter()
            .zip(total)
            .map(|(&b, &t)| b + scale * t)
            .collect()
    };
    Ok(StageFit {
        trees,
        fitted: fit(base, &total),
        eval_fitted: eval_base
            .iter()
            .zip(&eval_total)
            .map(|(b, t)| fit(b, t))
            .collect(),
    })
}
