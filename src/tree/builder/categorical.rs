//! XGBoost's categorical split search (`HistEvaluator`'s one-hot and
//! partition enumeration), shared by the histogram and exact builders.

use super::shared::xgb_calc_weight;
use super::split::SplitScorer;
use super::{BestSplit, Children, Score, need_replace};
use crate::tree::gain::GradStats;
use crate::tree::reuse::CategoricalPenalty;

/// XGBoost's default `max_cat_to_onehot`: categorical features with fewer
/// categories enumerate one-hot splits instead of partitions.
pub(super) const MAX_CAT_TO_ONEHOT: usize = 4;

/// XGBoost's default `max_cat_threshold`: the most categories one side of a
/// partition split enumerates.
pub(super) const MAX_CAT_THRESHOLD: usize = 64;

/// The best categorical split of one feature, in XGBoost's orientation: its
/// set of categories goes to the right child, `children.default_left` routes
/// missing values.
struct CatCandidate {
    children: Children,
    score: Score<f32>,
}

/// XGBoost's scalar categorical split search (`HistEvaluator`), shared by the
/// histogram and exact builders. `cats` holds every category of `feature` in
/// ascending order with its node statistics (zero when the node has none);
/// `total` is the node's statistics, including missing values.
///
/// - Fewer than [`MAX_CAT_TO_ONEHOT`] categories (`EnumerateOneHot`): each
///   category alone on one side, with missing values on either side.
/// - Otherwise (`EnumeratePart`): categories stably sorted by their weight
///   (`CalcWeightCat`), then scanned forward (a growing prefix of the order
///   on one side, missing values on the other) and backward (a growing
///   suffix on the other side, missing values with it), at most
///   [`MAX_CAT_THRESHOLD`] categories deep.
///
/// Candidates are scored with [`SplitScorer::loss_chg`] and compared with XGBoost's
/// tie rule ([`need_replace`]). XGBoost routes the chosen set to its right
/// child; the recorded split stores that set as the tree's left child, with
/// the children (and the missing direction) swapped to match. `penalty`
/// (opt-in reuse penalties) is subtracted from each candidate's loss change
/// before it competes; `None` leaves the search untouched.
pub(super) fn sweep_categorical(
    best: &mut BestSplit,
    cats: &[(u32, GradStats)],
    total: GradStats,
    scorer: &SplitScorer,
    feature: u32,
    penalty: Option<&dyn CategoricalPenalty>,
) {
    let n = cats.len();
    let reg = scorer.reg;
    let score = |children: Children, set: &dyn Fn() -> Vec<u32>| {
        let mut score = scorer.loss_chg(children.left, children.right)?;
        if let Some(penalty) = penalty {
            score.loss_chg -= penalty.categorical_penalty(feature, &set()) as f32;
        }
        Some(score)
    };
    // `SplitEntry::Update` on a per-feature entry that starts at zero.
    let offer =
        |local: &mut Option<CatCandidate>, children: Children, set: &dyn Fn() -> Vec<u32>| {
            let Some(score) = score(children, set) else {
                return false;
            };
            let incumbent = local.as_ref().map_or(0.0, |c| c.score.loss_chg);
            if !need_replace(incumbent, feature, score.loss_chg, feature) {
                return false;
            }
            *local = Some(CatCandidate { children, score });
            true
        };
    // `p_best->Update(best)`: the feature's split against the node's best.
    let merge = |best: &mut BestSplit, local: Option<CatCandidate>, mut set: Vec<u32>| {
        let Some(CatCandidate { children, score }) = local else {
            return;
        };
        if need_replace(best.loss_chg as f32, best.feature, score.loss_chg, feature) {
            set.sort_unstable();
            *best =
                BestSplit::categorical(feature, children.swapped(), score.swapped().into(), set);
            best.children_swapped = true;
        }
    };

    if n < MAX_CAT_TO_ONEHOT {
        let mut present = GradStats::default();
        for &(_, stats) in cats {
            present.add(stats);
        }
        let missing = total.sub(present);
        let mut local = None;
        let mut chosen = 0;
        for &(cat, stats) in cats {
            let single = || vec![cat];
            // Missing values with the other categories, then with this one.
            let mut right = stats;
            let missing_left = Children::new(true, total.sub(right), right);
            if offer(&mut local, missing_left, &single) {
                chosen = cat;
            }
            right.add(missing);
            let missing_right = Children::new(false, total.sub(right), right);
            if offer(&mut local, missing_right, &single) {
                chosen = cat;
            }
        }
        merge(best, local, vec![chosen]);
        return;
    }

    let weight = |s: GradStats| -> f32 {
        if s.hess < reg.min_child_weight {
            0.0
        } else {
            xgb_calc_weight(s, reg) as f32
        }
    };
    let keys: Vec<f32> = cats.iter().map(|&(_, s)| weight(s)).collect();
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&l, &r| {
        keys[l]
            .partial_cmp(&keys[r])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let set_of =
        |partition: usize| -> Vec<u32> { sorted[..partition].iter().map(|&c| cats[c].0).collect() };
    let depth = MAX_CAT_THRESHOLD.min(n);
    for forward in [true, false] {
        let mut acc = GradStats::default();
        let mut local = None;
        let mut best_partition = 0;
        for step in 0..depth - 1 {
            let j = if forward { step } else { n - 1 - step };
            acc.add(cats[sorted[j]].1);
            // The set (XGBoost's right child) is the first `partition`
            // sorted categories: the scanned prefix going forward, the
            // unscanned rest going backward.
            let partition = if forward { step + 1 } else { j };
            let (left, right) = if forward {
                (total.sub(acc), acc)
            } else {
                (acc, total.sub(acc))
            };
            if offer(&mut local, Children::new(forward, left, right), &|| {
                set_of(partition)
            }) {
                best_partition = partition;
            }
        }
        merge(best, local, set_of(best_partition));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::builder::SplitLocation;
    use crate::tree::builder::shared::xgb_node_gain;
    use crate::tree::constraints::Bounds;
    use crate::tree::gain::RegParams;

    fn unregularized() -> RegParams {
        RegParams {
            lambda: 0.0,
            alpha: 0.0,
            max_delta_step: 0.0,
            min_child_weight: 0.0,
        }
    }

    fn sweep(cats: &[(u32, GradStats)], total: GradStats) -> BestSplit {
        let reg = unregularized();
        let mut best = BestSplit::none();
        let scorer = SplitScorer {
            reg: &reg,
            root_gain: xgb_node_gain(total, &reg, Bounds::default()),
            bounds: Bounds::default(),
            dir: 0,
        };
        sweep_categorical(&mut best, cats, total, &scorer, 0, None);
        best
    }

    fn categories(best: &BestSplit) -> &[u32] {
        match &best.location {
            SplitLocation::Categories(categories) => categories,
            SplitLocation::Numeric(pos) => panic!("numeric split at {pos:?}"),
        }
    }

    /// A feature with a single category still separates it from the missing
    /// values (one-hot search, fewer than four categories).
    #[test]
    fn a_lone_category_splits_from_missing_values() {
        let best = sweep(&[(0, GradStats::new(2.0, 2.0))], GradStats::new(0.0, 4.0));
        assert_eq!(categories(&best), [0]);
        assert!(!best.default_left, "missing values go to the other child");
        assert!((best.loss_chg - 4.0).abs() < 1e-6, "{}", best.loss_chg);
    }

    /// Four categories ordered by weight (`-2, -1, 1, 2`) plus missing values
    /// whose weight (`±3`) puts them with one end of that order: the search
    /// scans both directions, so the missing values join the side that fits
    /// them, and the recorded set is `{0, 1}` either way.
    #[test]
    fn missing_values_join_the_side_that_fits_them() {
        let cats = [
            (0, GradStats::new(2.0, 1.0)),
            (1, GradStats::new(1.0, 1.0)),
            (2, GradStats::new(-1.0, 1.0)),
            (3, GradStats::new(-2.0, 1.0)),
        ];
        // G = 3 and -6 over H = 2 and 3 against the parent's 9/5.
        let expected = 4.5 + 12.0 - 1.8;
        for (missing_grad, with_set) in [(-3.0, false), (3.0, true)] {
            let best = sweep(&cats, GradStats::new(missing_grad, 5.0));
            assert_eq!(categories(&best), [0, 1], "missing gradient {missing_grad}");
            assert_eq!(
                best.default_left, with_set,
                "missing gradient {missing_grad}"
            );
            assert!(
                (best.loss_chg - expected).abs() < 1e-5,
                "missing gradient {missing_grad}: {}",
                best.loss_chg
            );
        }
    }
}
