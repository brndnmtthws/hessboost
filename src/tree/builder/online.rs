//! Split ranking for in-place data updates
//! ([`crate::training::online`]): where a node's current split stands
//! among the candidates a histogram now offers, scored exactly as the
//! histogram builder scores them.

use super::{BELOW_ALL_VALUES, SplitPos, SplitScorer, xgb_node_gain};
use crate::data::quantile::HistCuts;
use crate::tree::constraints::Bounds;
use crate::tree::gain::{GradStats, RegParams};

/// A node's current split among the candidates of its updated histogram.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SplitRank {
    /// Valid candidates (finite loss change, children above
    /// `min_child_weight`).
    pub(crate) candidates: usize,
    /// Candidates scoring strictly better than the current split; `None` if
    /// the current split is no longer a valid candidate.
    pub(crate) better: Option<usize>,
    /// The current split's loss change (`0` when invalid).
    pub(crate) loss_chg: f32,
}

/// Rank the numeric split `(feature, split_cond, default_left)` among every
/// numeric candidate of `hist` (the node's per-bin sums over `cuts`, total
/// `total`), scored with the builder's [`SplitScorer`] (no monotone bounds).
/// `dense` is the index's density, which decides whether missing-value
/// directions are enumerated.
pub(crate) fn rank_split(
    reg: &RegParams,
    cuts: &HistCuts,
    hist: &[GradStats],
    total: GradStats,
    dense: bool,
    (feature, split_cond, default_left): (u32, f32, bool),
) -> SplitRank {
    let bounds = Bounds::default();
    let scorer = SplitScorer {
        reg,
        root_gain: xgb_node_gain(total, reg, bounds),
        bounds,
        dir: 0,
    };
    let mut gains = Vec::new();
    let mut current = None;
    for f in 0..cuts.n_features() {
        let (fs, fe) = cuts.feature_bins(f);
        if cuts.is_categorical(f) || fe <= fs + 1 {
            continue;
        }
        super::for_each_numeric_split(&hist[fs..fe], fs, total, dense, |pos, children| {
            let Some(score) = scorer.loss_chg(children.left, children.right) else {
                return;
            };
            if !score.loss_chg.is_finite() {
                return;
            }
            gains.push(score.loss_chg);
            let cond = match pos {
                SplitPos::Bin(bin) => cuts.cut_value(bin),
                SplitPos::BelowBins => BELOW_ALL_VALUES,
                SplitPos::Value(v) => v,
            };
            if f as u32 == feature && cond == split_cond && children.default_left == default_left {
                current = Some(score.loss_chg);
            }
        });
    }
    SplitRank {
        candidates: gains.len(),
        better: current.map(|c| gains.iter().filter(|&&g| g > c).count()),
        loss_chg: current.unwrap_or(0.0),
    }
}
