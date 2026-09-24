//! Feature- and threshold-reuse penalties from *Boosted Trees on a Diet*
//! (Herrmann et al., ICLR 2026, arXiv:2510.26557, §3.1 and Appendix A), an
//! opt-in extension beyond XGBoost.
//!
//! The paper adds `ι·|F_U| + ξ·Σ_{f∈F_U} |T^f|` to each tree's regularizer,
//! where `F_U` is the set of features the ensemble already splits on and
//! `T^f` the set of thresholds already used for feature `f`. Greedy split
//! search then scores every candidate with the modified gain
//!
//! ```text
//! Δ_l(I, f, t) = Δ(I, f, t) − s_f·ι − s_t·ξ
//! ```
//!
//! where `s_f = 1` iff `f ∉ F_U` and `s_t = 1` iff `t ∉ T^f` (a new feature
//! always brings a new threshold). `ι` is
//! [`toad_penalty_feature`](crate::config::TrainingParams::toad_penalty_feature)
//! and `ξ` is
//! [`toad_penalty_threshold`](crate::config::TrainingParams::toad_penalty_threshold).
//! The penalty is subtracted from the candidate's loss change *before* the
//! argmax, so a reused threshold can beat a slightly better new one, and the
//! penalized value is what the node is validated with (`Δ_l ≥ gamma`) and
//! stored as its split gain. The penalties share `gamma`'s units: the XGBoost
//! loss change `G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)`.
//!
//! A threshold is identified by its `f32` bit pattern (`split_cond`), so the
//! identity survives per-round cut regeneration (`approx`) and is exactly
//! what the compact model dictionary stores. The missing-only split (the
//! builders' `BELOW_ALL_VALUES` threshold) is a threshold like any other. For a
//! categorical feature the "threshold" is the set of categories routed left
//! (compared as a sorted set).
//!
//! `F_U` and `T^f` include every split committed before a node is evaluated:
//! all earlier trees of the ensemble (across outputs) plus the current
//! tree's earlier levels (depthwise) or earlier expansions (lossguide). Nodes
//! evaluated together see the same dictionary, so the result is independent
//! of thread scheduling.

use crate::config::TrainingParams;
use crate::data::quantile::HistCuts;
use crate::tree::RegTree;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

/// The penalty for one candidate, in `f32` like the loss changes it adjusts:
/// `ι + ξ` for a new feature, `ξ` for a new threshold of a used feature, `0`
/// for a reused `(feature, threshold)` pair.
#[inline]
fn penalty(feature_used: bool, threshold_used: bool, iota: f32, xi: f32) -> f32 {
    if !feature_used {
        iota + xi
    } else if !threshold_used {
        xi
    } else {
        0.0
    }
}

/// Penalty source for the categorical sweep: the penalty of routing the
/// given (unsorted) category set left on `feature`.
pub(crate) trait CategoricalPenalty {
    /// Penalty of the candidate set `cats_left` on `feature`.
    fn categorical_penalty(&self, feature: u32, cats_left: &[u32]) -> f64;
}

/// A sorted, deduplicated copy of a category set: the identity under which
/// sets are compared here and stored in the compact model dictionary.
pub(crate) fn canonical_categories(cats: &[u32]) -> Vec<u32> {
    let mut set = cats.to_vec();
    set.sort_unstable();
    set.dedup();
    set
}

/// The ensemble's dictionary of used features and thresholds, plus the two
/// penalty weights. Built from the trees trained so far and extended with
/// every new tree ([`ReuseSet::record_tree`]).
#[derive(Debug, Clone)]
pub(crate) struct ReuseSet {
    iota: f32,
    xi: f32,
    features: Vec<bool>,
    /// Per feature: `split_cond` bit patterns of its numeric splits.
    thresholds: Vec<BTreeSet<u32>>,
    /// Per feature: sorted left-category sets of its categorical splits.
    category_sets: Vec<BTreeSet<Vec<u32>>>,
}

impl ReuseSet {
    /// The dictionary for training with `params`, seeded with the splits of
    /// `trees` (the ensemble built so far). `None` when both penalties are
    /// zero: the default training path then runs untouched.
    pub(crate) fn from_params(
        params: &TrainingParams,
        n_features: usize,
        trees: &[RegTree],
    ) -> Option<Self> {
        if params.toad_penalty_feature == 0.0 && params.toad_penalty_threshold == 0.0 {
            return None;
        }
        let mut set = ReuseSet {
            iota: params.toad_penalty_feature as f32,
            xi: params.toad_penalty_threshold as f32,
            features: vec![false; n_features],
            thresholds: vec![BTreeSet::new(); n_features],
            category_sets: vec![BTreeSet::new(); n_features],
        };
        for tree in trees {
            set.record_tree(tree);
        }
        Some(set)
    }

    /// Add every split of `tree` to the dictionary.
    pub(crate) fn record_tree(&mut self, tree: &RegTree) {
        for node in tree.nodes().iter().filter(|n| !n.is_leaf()) {
            if node.is_categorical {
                self.record_categorical(node.split_feature, tree.node_categories(node));
            } else {
                self.record_numeric(node.split_feature, node.split_cond);
            }
        }
    }

    /// Mark a numeric split on `feature` at `threshold` as used.
    pub(crate) fn record_numeric(&mut self, feature: u32, threshold: f32) {
        let f = feature as usize;
        self.features[f] = true;
        self.thresholds[f].insert(threshold.to_bits());
    }

    /// Mark a categorical split on `feature` routing `cats_left` left as used.
    pub(crate) fn record_categorical(&mut self, feature: u32, cats_left: &[u32]) {
        let f = feature as usize;
        self.features[f] = true;
        self.category_sets[f].insert(canonical_categories(cats_left));
    }

    /// Penalty of a numeric split on `feature` at `threshold`.
    #[inline]
    pub(crate) fn numeric_penalty(&self, feature: u32, threshold: f32) -> f32 {
        let f = feature as usize;
        penalty(
            self.features[f],
            self.thresholds[f].contains(&threshold.to_bits()),
            self.iota,
            self.xi,
        )
    }
}

impl CategoricalPenalty for ReuseSet {
    fn categorical_penalty(&self, feature: u32, cats_left: &[u32]) -> f64 {
        let f = feature as usize;
        let used =
            self.features[f] && self.category_sets[f].contains(&canonical_categories(cats_left));
        f64::from(penalty(self.features[f], used, self.iota, self.xi))
    }
}

/// [`ReuseSet`] projected onto one histogram index, for the hist builder's
/// split scan: numeric thresholds become per-bin flags so the scan pays one
/// load per candidate. Flags are atomics because the builder evaluates nodes
/// concurrently while it commits splits serially between evaluations
/// (commits and reads never overlap; `Relaxed` suffices because the rayon
/// fork/join points order them).
#[derive(Debug)]
pub(crate) struct HistReuse {
    iota: f32,
    xi: f32,
    feature_used: Vec<AtomicBool>,
    /// Per global bin: its cut value is a used threshold of its feature.
    bin_used: Vec<AtomicBool>,
    /// Per feature: the missing-only split (`BELOW_ALL_VALUES`) is used.
    below_used: Vec<AtomicBool>,
    category_sets: Mutex<Vec<BTreeSet<Vec<u32>>>>,
}

impl HistReuse {
    /// Number of histogram bins this projection covers.
    pub(crate) fn n_bins(&self) -> usize {
        self.bin_used.len()
    }

    /// Project `set` onto the bins of `cuts`; `below` is the threshold the
    /// hist builder writes for a missing-only split.
    pub(crate) fn new(set: &ReuseSet, cuts: &HistCuts, below: f32) -> Self {
        let n_features = cuts.n_features();
        let mut bin_used: Vec<AtomicBool> = (0..cuts.total_bins())
            .map(|_| AtomicBool::new(false))
            .collect();
        let mut below_used = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let used = &set.thresholds[f];
            let (fs, fe) = cuts.feature_bins(f);
            if !cuts.is_categorical(f) {
                for (i, flag) in bin_used[fs..fe].iter_mut().enumerate() {
                    *flag.get_mut() = used.contains(&cuts.cut_value(fs + i).to_bits());
                }
            }
            below_used.push(AtomicBool::new(used.contains(&below.to_bits())));
        }
        HistReuse {
            iota: set.iota,
            xi: set.xi,
            feature_used: set.features.iter().map(|&u| AtomicBool::new(u)).collect(),
            bin_used,
            below_used,
            category_sets: Mutex::new(set.category_sets.clone()),
        }
    }

    /// Penalty of a numeric split on `feature` at global bin boundary `bin`
    /// (`None`: the missing-only split).
    #[inline]
    pub(crate) fn bin_penalty(&self, feature: u32, bin: Option<usize>) -> f32 {
        let f = feature as usize;
        let threshold_used = match bin {
            Some(b) => self.bin_used[b].load(Ordering::Relaxed),
            None => self.below_used[f].load(Ordering::Relaxed),
        };
        penalty(
            self.feature_used[f].load(Ordering::Relaxed),
            threshold_used,
            self.iota,
            self.xi,
        )
    }

    /// Mark a committed numeric split as used.
    pub(crate) fn commit_numeric(&self, feature: u32, bin: Option<usize>) {
        self.feature_used[feature as usize].store(true, Ordering::Relaxed);
        match bin {
            Some(b) => self.bin_used[b].store(true, Ordering::Relaxed),
            None => self.below_used[feature as usize].store(true, Ordering::Relaxed),
        }
    }

    /// Mark a committed categorical split as used.
    pub(crate) fn commit_categorical(&self, feature: u32, cats_left: &[u32]) {
        self.feature_used[feature as usize].store(true, Ordering::Relaxed);
        self.category_sets
            .lock()
            .unwrap_or_else(PoisonError::into_inner)[feature as usize]
            .insert(canonical_categories(cats_left));
    }
}

impl CategoricalPenalty for HistReuse {
    fn categorical_penalty(&self, feature: u32, cats_left: &[u32]) -> f64 {
        let f = feature as usize;
        let feature_used = self.feature_used[f].load(Ordering::Relaxed);
        let used = feature_used
            && self
                .category_sets
                .lock()
                .unwrap_or_else(PoisonError::into_inner)[f]
                .contains(&canonical_categories(cats_left));
        f64::from(penalty(feature_used, used, self.iota, self.xi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DMatrix;
    use crate::data::ghist::GHistIndex;
    use crate::learner::train;
    use crate::objective::GradPair;
    use crate::tree::builder::{ExactTreeBuilder, HistTreeBuilder, SortedColumns, all_rows};
    use crate::tree::sampler::ColumnSampler;

    /// Grow one depth-1 tree with the hist and the exact builder under
    /// `params`, seeding the dictionary from `seed` trees.
    fn stumps(
        params: &TrainingParams,
        data: &DMatrix,
        gpair: &[GradPair],
        seed: &[RegTree],
    ) -> [RegTree; 2] {
        let set = ReuseSet::from_params(params, data.n_cols(), seed);
        let cuts = HistCuts::from_dmatrix(data, 256);
        let ghist = GHistIndex::from_dmatrix(data, cuts);
        let rows = all_rows(data.n_rows());
        let hist = HistTreeBuilder::new(params)
            .with_reuse(set.as_ref(), ghist.cuts())
            .build(&ghist, gpair, &rows, &mut ColumnSampler::all(data.n_cols()));
        let exact = ExactTreeBuilder::new(params)
            .with_reuse(set.as_ref())
            .build(
                &SortedColumns::from_dmatrix(data),
                data,
                gpair,
                &rows,
                &mut ColumnSampler::all(data.n_cols()),
            );
        [hist, exact]
    }

    fn stump_params(iota: f64, xi: f64) -> TrainingParams {
        TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .toad_penalty_feature(iota)
            .toad_penalty_threshold(xi)
            .build()
            .unwrap()
    }

    /// Feature 0 separates the gradients perfectly (loss change 8); feature
    /// 1 misplaces one row of each side (loss change 2).
    fn two_feature_problem() -> (DMatrix, Vec<GradPair>) {
        let x = [
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 1.0, //
            4.0, 0.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0,
        ];
        let g: Vec<GradPair> = (0..8)
            .map(|i| GradPair::new(if i < 4 { 1.0 } else { -1.0 }, 1.0))
            .collect();
        (DMatrix::from_dense(&x, 8, 2).unwrap(), g)
    }

    #[test]
    fn a_split_is_taken_iff_its_gain_exceeds_both_penalties() {
        let (data, g) = two_feature_problem();
        for (iota, xi, splits) in [
            (5.0, 2.9, true),
            (5.0, 3.1, false),
            (7.9, 0.0, true),
            (8.1, 0.0, false),
        ] {
            for tree in stumps(&stump_params(iota, xi), &data, &g, &[]) {
                assert_eq!(tree.num_nodes() == 3, splits, "iota {iota}, xi {xi}");
                if splits {
                    let root = tree.node(0);
                    assert_eq!(root.split_feature, 0);
                    // The stored gain is the penalized one, eq. 3.
                    assert!((f64::from(root.split_gain) - (8.0 - iota - xi)).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn categorical_splits_pay_for_new_sets() {
        use crate::data::FeatureType;
        let x = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let data = DMatrix::from_dense(&x, 8, 1)
            .unwrap()
            .with_feature_types(&[FeatureType::Categorical])
            .unwrap();
        let g: Vec<GradPair> = (0..8)
            .map(|i| GradPair::new(if i < 4 { 1.0 } else { -1.0 }, 1.0))
            .collect();
        let [seed, _] = stumps(&stump_params(0.0, 0.0), &data, &g, &[]);
        assert!(seed.node(0).is_categorical);
        for (iota, xi, seeded, splits) in [
            (5.0, 2.9, false, true),
            (5.0, 3.1, false, false),
            (100.0, 100.0, true, true),
        ] {
            let seeds = if seeded {
                std::slice::from_ref(&seed)
            } else {
                &[]
            };
            for tree in stumps(&stump_params(iota, xi), &data, &g, seeds) {
                assert_eq!(tree.num_nodes() == 3, splits, "iota {iota}, xi {xi}");
            }
        }
    }

    #[test]
    fn a_used_feature_and_threshold_beat_a_better_new_one() {
        let (data, g) = two_feature_problem();
        let free = stump_params(0.0, 1.0);
        let [seed, _] = stumps(&TrainingParams::default(), &data, &g, &[]);
        // Seed the dictionary with feature 1's best split from an ensemble
        // tree grown on gradients where only feature 1 matters.
        let g1: Vec<GradPair> = (0..8)
            .map(|i| {
                GradPair::new(
                    if data.get(i, 1) == Some(0.0) {
                        1.0
                    } else {
                        -1.0
                    },
                    1.0,
                )
            })
            .collect();
        let [seed1_hist, seed1_exact] = stumps(&free, &data, &g1, &[]);
        assert_eq!(seed1_hist.node(0).split_feature, 1);
        assert_eq!(seed.node(0).split_feature, 0);
        // ι = 7 makes feature 0 (gain 8) worth 1 − ξ, below feature 1's
        // reused threshold (gain 2, no penalty).
        let params = stump_params(7.0, 1.0);
        let [hist, _] = stumps(&params, &data, &g, std::slice::from_ref(&seed1_hist));
        let [_, exact] = stumps(&params, &data, &g, std::slice::from_ref(&seed1_exact));
        for (tree, seed) in [(hist, &seed1_hist), (exact, &seed1_exact)] {
            assert_eq!(tree.node(0).split_feature, 1);
            assert_eq!(tree.node(0).split_cond, seed.node(0).split_cond);
        }
        // Without the seed, feature 1 is new too and feature 0 wins again.
        let [hist, exact] = stumps(&params, &data, &g, &[]);
        assert!(hist.num_nodes() == 1 || hist.node(0).split_feature == 0);
        assert!(exact.num_nodes() == 1 || exact.node(0).split_feature == 0);
    }

    /// Ten informative features, all of which plain boosting uses.
    fn wide_problem() -> DMatrix {
        let n = 1500;
        let f = 10;
        let x: Vec<f32> = (0..n * f)
            .map(|i| {
                ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect();
        let y: Vec<f32> = x
            .chunks(f)
            .map(|r| r.iter().enumerate().map(|(j, v)| v * (j + 1) as f32).sum())
            .collect();
        crate::test_support::labeled_dense(&x, n, f, &y)
    }

    fn dictionary(model: &crate::learner::BoostedModel) -> (usize, usize) {
        let set =
            ReuseSet::from_params(&params(1.0, 1.0), model.n_features(), model.trees()).unwrap();
        let features = set.features.iter().filter(|&&u| u).count();
        let thresholds = set.thresholds.iter().map(BTreeSet::len).sum();
        (features, thresholds)
    }

    #[test]
    fn penalties_shrink_the_ensemble_dictionary() {
        let data = wide_problem();
        let fit = |iota: f64, xi: f64| {
            let p = TrainingParams::builder()
                .max_depth(3)
                .toad_penalty_feature(iota)
                .toad_penalty_threshold(xi)
                .build()
                .unwrap();
            train(&p, &data, 30).unwrap()
        };
        let (base_f, base_t) = dictionary(&fit(0.0, 0.0));
        let (feat_f, _) = dictionary(&fit(200.0, 0.0));
        let (_, thr_t) = dictionary(&fit(0.0, 5.0));
        assert_eq!(base_f, 10);
        assert!(feat_f < base_f, "{feat_f} features vs {base_f}");
        assert!(thr_t < base_t / 2, "{thr_t} thresholds vs {base_t}");
    }

    #[test]
    fn penalized_training_is_deterministic_across_thread_counts() {
        let data = wide_problem();
        let fit = |threads: usize, method: crate::config::TreeMethod| {
            let p = TrainingParams::builder()
                .max_depth(4)
                .nthread(threads)
                .tree_method(method)
                .toad_penalty_feature(10.0)
                .toad_penalty_threshold(1.0)
                .build()
                .unwrap();
            train(&p, &data, 10).unwrap().to_bytes().unwrap()
        };
        for method in [
            crate::config::TreeMethod::Hist,
            crate::config::TreeMethod::Exact,
        ] {
            assert_eq!(fit(1, method), fit(8, method));
        }
    }

    fn params(iota: f64, xi: f64) -> TrainingParams {
        TrainingParams::builder()
            .toad_penalty_feature(iota)
            .toad_penalty_threshold(xi)
            .build()
            .unwrap()
    }

    #[test]
    fn penalties_must_be_non_negative_and_need_trees() {
        let build = |b: crate::config::TrainingParamsBuilder| b.build().is_err();
        assert!(build(TrainingParams::builder().toad_penalty_feature(-1.0)));
        assert!(build(
            TrainingParams::builder().toad_penalty_threshold(f64::NAN)
        ));
        assert!(build(
            TrainingParams::builder()
                .booster(crate::config::BoosterKind::GbLinear)
                .toad_penalty_threshold(1.0)
        ));
    }

    #[test]
    fn zero_penalties_disable_the_dictionary() {
        assert!(ReuseSet::from_params(&TrainingParams::default(), 3, &[]).is_none());
        assert!(ReuseSet::from_params(&params(0.0, 1.0), 3, &[]).is_some());
    }

    #[test]
    fn penalty_distinguishes_new_feature_new_threshold_and_reuse() {
        let mut set = ReuseSet::from_params(&params(4.0, 1.0), 3, &[]).unwrap();
        assert_eq!(set.numeric_penalty(1, 0.5), 5.0);
        set.record_numeric(1, 0.5);
        assert_eq!(set.numeric_penalty(1, 0.5), 0.0);
        assert_eq!(set.numeric_penalty(1, 0.75), 1.0);
        assert_eq!(set.numeric_penalty(0, 0.5), 5.0);
        // Thresholds are bit patterns: -0.0 is a different threshold.
        set.record_numeric(2, 0.0);
        assert_eq!(set.numeric_penalty(2, -0.0), 1.0);
    }

    #[test]
    fn categorical_sets_compare_as_sets() {
        let mut set = ReuseSet::from_params(&params(4.0, 1.0), 2, &[]).unwrap();
        set.record_categorical(0, &[3, 1]);
        assert_eq!(set.categorical_penalty(0, &[1, 3]), 0.0);
        assert_eq!(set.categorical_penalty(0, &[1]), 1.0);
        assert_eq!(set.categorical_penalty(1, &[1, 3]), 5.0);
    }

    #[test]
    fn hist_projection_matches_the_dictionary() {
        let x: Vec<f32> = (0..32).map(|i| (i % 8) as f32).collect();
        let data = DMatrix::from_dense(&x, 16, 2).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let (fs, fe) = cuts.feature_bins(1);
        let used_bin = fs + (fe - fs) / 2;
        let mut set = ReuseSet::from_params(&params(4.0, 1.0), 2, &[]).unwrap();
        set.record_numeric(1, cuts.cut_value(used_bin));
        let hist = HistReuse::new(&set, &cuts, f32::MIN);
        for bin in fs..fe {
            let expected = if bin == used_bin { 0.0 } else { 1.0 };
            assert_eq!(hist.bin_penalty(1, Some(bin)), expected);
        }
        assert_eq!(hist.bin_penalty(1, None), 1.0);
        assert_eq!(hist.bin_penalty(0, Some(cuts.feature_bins(0).0)), 5.0);
        hist.commit_numeric(0, None);
        assert_eq!(hist.bin_penalty(0, None), 0.0);
        assert_eq!(hist.bin_penalty(0, Some(cuts.feature_bins(0).0)), 1.0);
    }
}
