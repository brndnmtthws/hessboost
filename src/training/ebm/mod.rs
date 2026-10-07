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

use super::margins::{MarginCaches, TreeOutput, add_tree_margins};
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

/// One Boulevard stage's trees and the fitted margins `base + stage` over
/// the training rows and over every eval set's rows.
struct StageFit {
    trees: Vec<GrownTree>,
    fitted: Vec<f64>,
    eval_fitted: Vec<Vec<f64>>,
}

/// A grown tree: the reported round that grew it ([`Hook`]), its term, and
/// the tree with its final leaf values.
struct GrownTree {
    round: usize,
    term: u32,
    tree: RegTree,
}

/// What a run grows: every tree in model order (round by round), the pair
/// terms FAST picked, and how many of the reported rounds were main-effect
/// rounds.
struct Grown {
    trees: Vec<GrownTree>,
    pairs: Vec<Vec<u32>>,
    main_rounds: usize,
}

/// A finished EBM run, before it becomes the model ([`Boosted::into_model`]).
pub(super) struct Boosted {
    grown: Grown,
    /// The main-effect terms, one per feature.
    mains: Vec<Vec<u32>>,
    boulevard: Option<EbmBoulevard>,
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
/// stage and on through the pair stage, each reported with the eval sets'
/// margins of the model training holds after it, and a `Break` ends
/// training.
struct Hook<'h, 'a> {
    after_round: &'h mut dyn FnMut(usize, &MarginCaches) -> ControlFlow<()>,
    /// The eval sets' margins (the training rows' are not kept current).
    margins: &'h mut MarginCaches<'a>,
    /// The eval sets' matrices, in eval-set order.
    eval_data: Vec<&'a DMatrix>,
    done: usize,
    stopped: bool,
}

impl Hook<'_, '_> {
    /// Add `tree`'s predictions to every eval set's margins.
    fn add_tree(&mut self, tree: &RegTree) {
        for (margins, data) in self.margins.evals.iter_mut().zip(&self.eval_data) {
            add_tree_margins(tree, data, margins, 1, TreeOutput::Scalar(0));
        }
    }

    /// Report a finished round (the eval margins current); whether
    /// training goes on.
    fn next(&mut self) -> bool {
        self.stopped |= (self.after_round)(self.done, self.margins).is_break();
        self.done += 1;
        !self.stopped
    }
}

/// Train `rounds` EBM rounds of main effects, then of the
/// [`Ebm::interactions`](crate::config::Ebm::interactions) FAST pairs, from
/// `model`'s intercept, calling `after_round` after every round of either
/// stage with `margins`' eval margins of the model training holds after
/// that round. A `Break` stops training there: the run keeps the completed
/// rounds (a stopped Boulevard stage averages those), and a stop in the
/// main-effect stage skips the pairs.
pub(super) fn boost(
    run: &TrainContext,
    prepared: &Prepared,
    model: &BoostedModel,
    rounds: usize,
    metric: Option<&dyn Metric>,
    margins: &mut MarginCaches,
    after_round: &mut dyn FnMut(usize, &MarginCaches) -> ControlFlow<()>,
) -> Result<Boosted> {
    let mut hook = Hook {
        after_round,
        eval_data: margins.eval_data().collect(),
        margins,
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
    let (grown, boulevard) = if params.ebm_settings().boulevard() {
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
    Ok(Boosted {
        grown,
        mains,
        boulevard,
    })
}

impl Boosted {
    /// Write the run's trees into `model` (which holds only the intercept)
    /// and record its [`EbmInfo`], term means over `dtrain`. With
    /// `through`, only what training held after reported round `through`
    /// (a classic EBM's early-stopping best round): the trees through that
    /// round, a prefix of the round-major layout, and no pair terms when it
    /// is a main-effect round, as after a `Break` there.
    pub(super) fn into_model(
        self,
        model: &mut BoostedModel,
        dtrain: &DMatrix,
        through: Option<usize>,
    ) {
        let Boosted {
            grown:
                Grown {
                    mut trees,
                    mut pairs,
                    main_rounds,
                },
            mains,
            boulevard,
        } = self;
        if let Some(round) = through {
            debug_assert!(boulevard.is_none(), "a Boulevard EBM averages every round");
            trees.truncate(trees.partition_point(|t| t.round <= round));
            if round < main_rounds {
                pairs.clear();
            }
        }
        let mut terms = mains;
        terms.extend(pairs);
        let mut tree_terms = Vec::with_capacity(trees.len());
        for GrownTree { term, tree, .. } in trees {
            tree_terms.push(term);
            model.push_tree_weighted(tree, 1.0);
        }
        let mut info = EbmInfo {
            terms,
            tree_terms,
            term_means: Vec::new(),
            boulevard,
        };
        info.term_means = info.term_means_on(model, dtrain);
        model.set_ebm(Some(info));
    }
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
