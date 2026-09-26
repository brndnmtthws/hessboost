//! `booster = boulevard`: the Boulevard recursions of Zhou & Hooker (JMLR
//! 2022) and Fang, Tan & Hooker (NeurIPS 2025) — BRAT-D (dropout, one tree
//! per round) and BRAT-P (`num_parallel_tree` trees per round, each leaving
//! its own slot out). See [`crate::inference`] for the algorithms and the
//! statistics built on them.
//!
//! [`Recursion`] owns what the residuals depend on (the raw training-row
//! predictions of the trees so far) and draws each round's dropout set; a
//! caller supplies the trees. Training grows them ([`boost`]); the honest
//! refit ([`crate::inference::honest_refit`]) re-estimates fixed tree
//! structures' leaves on other rows. Trees are kept raw (unscaled) while the
//! recursion runs; the model's leaves get [`Recursion::scale`] at the end.

use rayon::prelude::*;

use super::train::{
    MarginCaches, Prepared, TrainContext, TreeSample, make_column_sampler, sample_rows,
};
use crate::config::{BoosterKind, TrainingParams};
use crate::data::DMatrix;
use crate::error::Result;
use crate::inference::BoulevardInfo;
use crate::model::BoostedModel;
use crate::objective::GradPair;
use crate::rng::Rng;
use crate::tree::RegTree;
use crate::tree::reuse::ReuseSet;
use crate::tree::sampler::ColumnSampler;

/// Rows per parallel chunk of the per-row residual sums.
const ROW_CHUNK: usize = 4096;

/// The Boulevard round RNG's salt (gbtree uses `0`, DART `0x0DA27`).
pub(crate) const BOULEVARD_SALT: u64 = 0xB0_07E7;

/// The Boulevard settings that shape the recursion.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Schedule {
    /// BRAT-D's dropout probability `p`.
    pub(crate) dropout: f64,
    /// The learning rate `λ`.
    pub(crate) learning_rate: f64,
    /// The residual truncation level `M` (`0` = none).
    pub(crate) truncation: f64,
    /// Trees per round (`1`: BRAT-D; more: BRAT-P).
    pub(crate) parallel: usize,
    /// Base of the per-round RNG seeds.
    pub(crate) seed: u64,
    /// Salt separating this recursion's streams from other uses of `seed`.
    pub(crate) salt: u64,
}

impl Schedule {
    /// The schedule of training with `params`.
    pub(crate) fn from_params(params: &TrainingParams) -> Self {
        Schedule {
            dropout: params.boulevard_dropout,
            learning_rate: params.eta,
            truncation: params.boulevard_truncation,
            parallel: params.num_parallel_tree,
            seed: params.seed,
            salt: BOULEVARD_SALT,
        }
    }

    /// The schedule recorded in `info` for a model of `parallel` trees per
    /// round, with the RNG salt `salt`.
    pub(crate) fn from_info(info: &BoulevardInfo, parallel: usize, salt: u64) -> Self {
        Schedule {
            dropout: info.dropout,
            learning_rate: info.learning_rate,
            truncation: info.truncation,
            parallel,
            seed: info.seed,
            salt,
        }
    }

    /// The factor that turns the raw tree sum after `rounds` rounds into
    /// the Boulevard prediction (minus the intercept): `(1 + λq) / B` for
    /// BRAT-D (Algorithm 1's final `(1 + λq)/λ` rescaling of the `λ/B`
    /// average), `1 / B` for BRAT-P.
    pub(crate) fn scale(&self, rounds: usize) -> f64 {
        let b = rounds.max(1) as f64;
        if self.parallel > 1 {
            1.0 / b
        } else {
            (1.0 + self.learning_rate * (1.0 - self.dropout)) / b
        }
    }

    fn truncate(&self, v: f64) -> f64 {
        if self.truncation > 0.0 {
            v.clamp(-self.truncation, self.truncation)
        } else {
            v
        }
    }
}

/// One call for new trees: round `index`, the trees of slots
/// `first_slot..first_slot + offsets.len()`, each fitted to the labels minus
/// the intercept minus its `offsets` entry (per row), drawing its row sample
/// and column sampler from `rng`.
pub(crate) struct RoundRequest<'a> {
    pub(crate) index: usize,
    pub(crate) first_slot: usize,
    pub(crate) offsets: &'a [Vec<f64>],
    pub(crate) rng: &'a mut Rng,
}

/// The state of a Boulevard recursion over `n` rows.
pub(crate) struct Recursion {
    schedule: Schedule,
    n: usize,
    rounds: usize,
    /// Every tree's raw prediction of every row (BRAT-D with dropout only).
    per_tree: Vec<Vec<f32>>,
    /// The raw sum of every tree (BRAT-D) or of each slot's trees (BRAT-P),
    /// per row.
    slots: Vec<Vec<f64>>,
}

impl Recursion {
    pub(crate) fn new(schedule: Schedule, n: usize) -> Self {
        Recursion {
            schedule,
            n,
            rounds: 0,
            per_tree: Vec::new(),
            slots: vec![vec![0.0; n]; schedule.parallel],
        }
    }

    /// The factor of the raw tree sum after the rounds run so far.
    pub(crate) fn scale(&self) -> f64 {
        self.schedule.scale(self.rounds)
    }

    /// Run one round: draw its RNG and dropout set, form each slot's
    /// offsets, and ask `fit` for the trees' raw row predictions (one
    /// vector per requested slot).
    pub(crate) fn step(
        &mut self,
        mut fit: impl FnMut(RoundRequest) -> Result<Vec<Vec<f32>>>,
    ) -> Result<()> {
        let s = self.schedule;
        let b = self.rounds;
        let mut rng = Rng::new(s.seed ^ (b as u64).wrapping_mul(0x9E37_79B9) ^ s.salt);
        if s.parallel == 1 {
            let offset = self.dropout_offset(&mut rng);
            let preds = fit(RoundRequest {
                index: b,
                first_slot: 0,
                offsets: std::slice::from_ref(&offset),
                rng: &mut rng,
            })?;
            let preds = preds.into_iter().next().unwrap_or_default();
            add_rows(&mut self.slots[0], &preds);
            if s.dropout > 0.0 {
                self.per_tree.push(preds);
            }
        } else if b == 0 {
            // Warm start: the first round boosts its trees in sequence.
            let mut running = vec![0.0; self.n];
            for k in 0..s.parallel {
                let offset: Vec<f64> = running.iter().map(|&v| s.truncate(v)).collect();
                let preds = fit(RoundRequest {
                    index: 0,
                    first_slot: k,
                    offsets: std::slice::from_ref(&offset),
                    rng: &mut rng,
                })?;
                let preds = preds.into_iter().next().unwrap_or_default();
                add_rows(&mut running, &preds);
                add_rows(&mut self.slots[k], &preds);
            }
        } else {
            let inv = 1.0 / b as f64;
            let slots = &self.slots;
            let offsets: Vec<Vec<f64>> = (0..s.parallel)
                .map(|k| {
                    let mut out = vec![0.0; self.n];
                    out.par_chunks_mut(ROW_CHUNK)
                        .enumerate()
                        .for_each(|(c, chunk)| {
                            let start = c * ROW_CHUNK;
                            for (i, o) in chunk.iter_mut().enumerate() {
                                let row = start + i;
                                let mut sum = 0.0;
                                for (l, slot) in slots.iter().enumerate() {
                                    if l != k {
                                        sum += slot[row];
                                    }
                                }
                                *o = s.truncate(sum * inv);
                            }
                        });
                    out
                })
                .collect();
            let preds = fit(RoundRequest {
                index: b,
                first_slot: 0,
                offsets: &offsets,
                rng: &mut rng,
            })?;
            for (slot, p) in self.slots.iter_mut().zip(&preds) {
                add_rows(slot, p);
            }
        }
        self.rounds += 1;
        Ok(())
    }

    /// BRAT-D's offset of round `b = self.rounds`: `(λ / b) Σ_{kept} t_s`,
    /// each of the `b` earlier trees kept with probability `1 − p` (one
    /// draw per tree, in tree order), truncated.
    fn dropout_offset(&self, rng: &mut Rng) -> Vec<f64> {
        let s = self.schedule;
        let b = self.rounds;
        if b == 0 {
            return vec![0.0; self.n];
        }
        let factor = s.learning_rate / b as f64;
        let total = &self.slots[0];
        if s.dropout == 0.0 {
            return total.iter().map(|&v| s.truncate(factor * v)).collect();
        }
        let kept: Vec<bool> = (0..b).map(|_| rng.f64() >= s.dropout).collect();
        let n_kept = kept.iter().filter(|&&k| k).count();
        // Sum whichever set is smaller: the kept trees, or the dropped ones
        // subtracted from the total.
        let from_kept = n_kept <= b - n_kept;
        let trees: Vec<&[f32]> = self
            .per_tree
            .iter()
            .zip(&kept)
            .filter(|&(_, &k)| k == from_kept)
            .map(|(t, _)| t.as_slice())
            .collect();
        let mut out = vec![0.0; self.n];
        out.par_chunks_mut(ROW_CHUNK)
            .enumerate()
            .for_each(|(c, chunk)| {
                let start = c * ROW_CHUNK;
                for (i, o) in chunk.iter_mut().enumerate() {
                    let row = start + i;
                    let partial: f64 = trees.iter().map(|t| f64::from(t[row])).sum();
                    let sum = if from_kept {
                        partial
                    } else {
                        total[row] - partial
                    };
                    *o = s.truncate(factor * sum);
                }
            });
        out
    }
}

/// `sum += preds`, row by row.
fn add_rows(sum: &mut [f64], preds: &[f32]) {
    for (s, &p) in sum.iter_mut().zip(preds) {
        *s += f64::from(p);
    }
}

/// Every row's raw prediction by `tree`.
pub(crate) fn tree_rows(tree: &RegTree, data: &DMatrix) -> Vec<f32> {
    (0..data.n_rows())
        .into_par_iter()
        .with_min_len(1024)
        .map(|row| tree.predict_row(data, row))
        .collect()
}

/// What [`boost`] updates: the model (raw trees while boosting), the eval
/// margin caches, and the reuse dictionary.
pub(super) struct BoostState<'m, 'a> {
    pub(super) model: &'m mut BoostedModel,
    pub(super) margins: &'m mut MarginCaches<'a>,
    pub(super) reuse: &'m mut Option<ReuseSet>,
}

/// Train `rounds` Boulevard rounds into `state.model` with the prepared
/// builder, calling `after_round(iteration, margins)` once each round's
/// eval margins are current; then scale the leaves and record the
/// [`BoulevardInfo`].
pub(super) fn boost(
    run: &TrainContext,
    prepared: &Prepared,
    state: BoostState<'_, '_>,
    rounds: usize,
    mut after_round: impl FnMut(usize, &MarginCaches),
) -> Result<()> {
    let TrainContext {
        params,
        dtrain,
        info,
        objective,
    } = *run;
    debug_assert_eq!(params.booster, BoosterKind::Boulevard);
    let BoostState {
        model,
        margins,
        reuse,
    } = state;
    let n = dtrain.n_rows();
    let mu = model.base_scores()[0];
    let mut recursion = Recursion::new(Schedule::from_params(params), n);
    let mut eval_sums: Vec<Vec<f64>> = margins.evals.iter().map(|m| vec![0.0; m.len()]).collect();
    let eval_sets: Vec<&DMatrix> = margins.eval_data().collect();
    for round in 0..rounds {
        recursion.step(|request| {
            let RoundRequest { offsets, rng, .. } = request;
            // Each slot's gradients at the margins `μ + offset`.
            let gpairs: Vec<Vec<GradPair>> = offsets
                .iter()
                .map(|offset| {
                    let margins: Vec<f32> =
                        offset.iter().map(|&o| (f64::from(mu) + o) as f32).collect();
                    let mut gpair = vec![GradPair::default(); n];
                    objective.gradient_info(&margins, info, &mut gpair);
                    gpair
                })
                .collect();
            // Row samples, then column samplers, in slot order.
            let rows: Vec<Vec<u32>> = gpairs.iter().map(|_| sample_rows(n, params, rng)).collect();
            let mut samplers: Vec<_> = gpairs
                .iter()
                .map(|_| make_column_sampler(dtrain, params, rng))
                .collect();
            prepared.fill_approx_cache(run, &gpairs[0]);
            let build = |k: usize, sampler: &mut ColumnSampler, reuse: Option<&mut ReuseSet>| {
                let sample = TreeSample {
                    gpair: &gpairs[k],
                    rows: &rows[k],
                    forest_index: None,
                };
                prepared.build_tree(run, sample, sampler, reuse, 0, false).0
            };
            let trees: Vec<RegTree> = if gpairs.len() > 1
                && reuse.is_none()
                && params.device == crate::config::Device::Cpu
                && rayon::current_num_threads() > 1
            {
                samplers
                    .par_iter_mut()
                    .enumerate()
                    .map(|(k, sampler)| build(k, sampler, None))
                    .collect()
            } else {
                samplers
                    .iter_mut()
                    .enumerate()
                    .map(|(k, sampler)| build(k, sampler, reuse.as_mut()))
                    .collect()
            };
            let mut preds = Vec::with_capacity(trees.len());
            for tree in trees {
                preds.push(tree_rows(&tree, dtrain));
                for (sum, data) in eval_sums.iter_mut().zip(&eval_sets) {
                    add_rows(sum, &tree_rows(&tree, data));
                }
                model.push_tree_weighted(tree, 1.0);
            }
            Ok(preds)
        })?;
        let scale = recursion.scale();
        for (margins, sum) in margins.evals.iter_mut().zip(&eval_sums) {
            for (m, &s) in margins.iter_mut().zip(sum) {
                *m = (f64::from(mu) + scale * s) as f32;
            }
        }
        after_round(round, &*margins);
    }
    model.scale_all_leaves(recursion.scale() as f32);
    model.set_boulevard(Some(BoulevardInfo {
        dropout: params.boulevard_dropout,
        learning_rate: params.eta,
        subsample: params.subsample,
        reg_lambda: params.lambda,
        truncation: params.boulevard_truncation,
        seed: params.seed,
        intercept_from_labels: params.base_score.is_none(),
    }));
    Ok(())
}
