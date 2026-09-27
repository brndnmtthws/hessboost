//! DART dropout and the per-round RNG.

use super::margins::MarginCaches;
use super::prepare::TrainContext;
use crate::config::{BoosterKind, Dart, TrainingParams};
use crate::model::BoostedModel;
use crate::objective::GradPair;
use crate::rng::Rng;

/// The RNG and gradients of one tree-growing round, filled into `gpair`.
/// gbtree takes the gradients at the cached `margin`. DART (Dropout Additive
/// Regression Trees) first draws a dropout set `D` over the trees built so
/// far ([`select_dropout`]) and takes the gradients of the ensemble
/// **excluding** `D`, whose tree ids it returns. Using XGBoost's `tree`
/// normalization, if `k = |D|` the round's new trees then get weight
/// `1/(k+eta)` ([`dart_new_tree_weight`]) and [`finish_dart`] rescales each
/// dropped tree by `k/(k+eta)`. A round that drops nothing (DART without
/// dropout, a skipped dropout, or no tree drawn) returns `None` and trains as
/// gbtree.
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
        ..
    } = *run;
    let mut rng = round_rng(params, iteration, round_salt(params));
    let dropout = match params.booster {
        BoosterKind::Dart(dart) if dart.has_dropout() => select_dropout(model, &dart, &mut rng),
        _ => None,
    };
    let Some((dropped, drop_indices)) = dropout else {
        // Nothing dropped: the round reads the ensemble's own margins and
        // its trees are not normalized, as in gbtree.
        objective.gradient_info_at(margin, info, gpair, iteration);
        return (rng, None);
    };
    let margin_excl = model.predict_margin_dropout(dtrain, &dropped);
    objective.gradient_info_at(&margin_excl, info, gpair, iteration);
    (rng, Some(drop_indices))
}

/// The DART round RNG's booster salt.
const DART_SALT: u64 = 0x0DA27;

/// The salt of a round's RNG: DART's when its dropout can drop a tree,
/// else gbtree's (a DART booster without dropout trains exactly as
/// gbtree).
pub(super) fn round_salt(params: &TrainingParams) -> u64 {
    match params.booster {
        BoosterKind::Dart(dart) if dart.has_dropout() => DART_SALT,
        _ => 0,
    }
}

/// Draw a DART round's dropout set over the trees built so far, as
/// XGBoost's `GBTree::DropTrees` (uniform sampling): nothing and no draws
/// over an empty ensemble; skipped with probability `skip_drop`; otherwise
/// each tree independently with probability `rate_drop`, plus one tree at
/// random when none was drawn and `one_drop` is set. Returns the per-tree
/// mask and the dropped indices, or `None` when nothing is dropped.
pub(super) fn select_dropout(
    model: &BoostedModel,
    dart: &Dart,
    rng: &mut Rng,
) -> Option<(Vec<bool>, Vec<usize>)> {
    let existing = model.num_trees();
    if existing == 0 {
        return None;
    }
    if dart.skip_drop() > 0.0 && rng.f64() < dart.skip_drop() {
        return None;
    }
    let mut dropped = vec![false; existing];
    let mut drop_indices: Vec<usize> = Vec::new();
    for (i, d) in dropped.iter_mut().enumerate() {
        if rng.f64() < dart.rate_drop() {
            *d = true;
            drop_indices.push(i);
        }
    }
    if drop_indices.is_empty() && dart.one_drop() {
        let i = rng.range(0..existing);
        dropped[i] = true;
        drop_indices.push(i);
    }
    (!drop_indices.is_empty()).then_some((dropped, drop_indices))
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
/// ensemble stays balanced, then recompute the margin caches, which the
/// rescaling makes non-additive (a later round that drops nothing reads the
/// training one).
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
    margins.recompute(model);
}

/// The RNG for one boosting round: `seed ^ round * 0x9E37_79B9`, plus a
/// booster-specific `salt` (`0` for gbtree, `0x0DA27` for DART) so the two
/// boosters draw from different streams.
pub(super) fn round_rng(params: &TrainingParams, round: usize, salt: u64) -> Rng {
    Rng::new(params.seed ^ (round as u64).wrapping_mul(0x9E37_79B9) ^ salt)
}
