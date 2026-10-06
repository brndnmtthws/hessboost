//! `booster = ebm`: cyclic GA²M boosting (classic, with outer bags) and the
//! Boulevard-averaged EBM of Fang, Tan, Pipping & Hooker (AISTATS 2026),
//! each followed by FAST pair selection and pair-term boosting. See
//! [`crate::ebm`] for the algorithms.

use std::ops::ControlFlow;

mod boulevard;
mod classic;
mod fast;

use boulevard::boulevard;
use classic::classic;

use super::prepare::{Prepared, TrainContext, TreeSample};
use crate::config::{Device, TrainingParams};
use crate::data::DMatrix;
use crate::ebm::{EbmBoulevard, EbmInfo};
use crate::error::{HessboostError, Result};
use crate::metric::Metric;
use crate::model::BoostedModel;
use crate::objective::GradPair;
use crate::tree::RegTree;
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
/// threads and the CPU backend (a GPU backend serves one tree at a time).
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

/// Every row's gradients at `margins` in boosting round `iteration` (read
/// by losses whose gradients draw per round, e.g. `rank:xendcg`).
fn gradients(run: &TrainContext, margins: &[f32], iteration: usize) -> Vec<GradPair> {
    let mut gpair = vec![GradPair::default(); margins.len()];
    run.objective
        .gradient_info_at(margins, run.info, &mut gpair, iteration);
    gpair
}

/// Refuse the training data `booster = ebm` cannot use: feature weights
/// (it samples no columns).
pub(super) fn validate_data(dtrain: &DMatrix) -> Result<()> {
    if dtrain.feature_weights().is_some() {
        return Err(HessboostError::invalid_data(
            "feature_weights",
            "`booster = ebm` does not sample columns",
        ));
    }
    Ok(())
}
