//! LightGBM's split-search options for the histogram builder (opt-in, beyond
//! XGBoost): `extra_trees` and `path_smooth`.
//!
//! When either is enabled, [`SplitOptions::evaluate`] replaces the builder's
//! XGBoost split search for the node; with both off the builder never reaches
//! this module, so default training keeps XGBoost's arithmetic bit for bit.
//!
//! **Extra trees** (LightGBM `extra_trees`, after Geurts et al.'s extremely
//! randomized trees): each numerical feature is scored at a single boundary
//! drawn uniformly between the node's lowest and highest occupied histogram
//! bin, once with missing values right and — when the feature has missing
//! values in the node — once with them left. LightGBM draws among all of a
//! feature's bins, so its draw can leave a child empty; drawing inside the
//! occupied range keeps every threshold inside the node's value range, as in
//! the original algorithm. Categorical features score one random prefix of the
//! gradient-ordered categories (LightGBM draws one prefix length as well). The
//! draws of a node come from a stream seeded by `extra_seed`, the tree's seed,
//! and the node id, so they do not depend on evaluation order: serial and
//! parallel growth pick the same splits.
//!
//! **Path smoothing** (LightGBM `path_smooth`, `s > 0`): a child's output is
//! pulled toward its parent's,
//! `w = w_raw·(n/s)/(n/s + 1) + w_parent/(n/s + 1)`, where `w_raw` is the
//! regularized optimum after `max_delta_step` clipping, `n` the child's row
//! count, and the result is then clamped to the monotone bounds (LightGBM's
//! `CalculateSplittedLeafOutput`). The root is not smoothed and the shrinkage
//! compounds down each path. Candidates are scored at the smoothed outputs,
//! `−(2·Tα(G)·w + (H+λ)·w²)` per child, minus the same score of the node itself
//! smoothed toward its own output (LightGBM's `gain_shift`). As in LightGBM's
//! histogram search, a candidate child's row count is estimated from its share
//! of the node's Hessian (`round(n·H_child/H_node)`, exact for constant
//! Hessians); leaves keep the outputs their split recorded.

use super::{
    BestSplit, K_RT_EPS, SplitPos, candidate_gain, xgb_loss_chg, xgb_node_gain, xgb_update,
};
use crate::config::TrainingParams;
use crate::data::quantile::HistCuts;
use crate::tree::constraints::{Bounds, MonotoneConstraints, gain_at_weight, satisfies};
use crate::tree::gain::{GradStats, RegParams, threshold_l1};
use crate::tree::regtree::RegTree;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// The node a split search runs for.
#[derive(Debug, Clone, Copy)]
pub(super) struct NodeCtx {
    /// Node id in the tree being grown; seeds the node's `extra_trees` draws.
    pub(super) id: usize,
    /// Training rows in the node (`n` of path smoothing).
    pub(super) rows: usize,
    /// The node's own output, which path smoothing pulls its children toward.
    pub(super) output: f64,
    /// Seed of the tree being grown ([`crate::tree::sampler::ColumnSampler::seed`]).
    pub(super) tree_seed: u64,
}

/// The enabled LightGBM split options. Built only when at least one is on.
#[derive(Debug, Clone, Copy)]
pub(super) struct SplitOptions {
    /// `Some(extra_seed)` when `extra_trees` is enabled.
    extra_seed: Option<u64>,
    /// Path smoothing strength (`0` = off).
    path_smooth: f64,
}

impl SplitOptions {
    /// The options enabled by `params`, or `None` when the builder should run
    /// XGBoost's split search unchanged.
    pub(super) fn from_params(params: &TrainingParams) -> Option<Self> {
        (params.extra_trees || params.path_smooth > 0.0).then_some(SplitOptions {
            extra_seed: params.extra_trees.then_some(params.extra_seed),
            path_smooth: params.path_smooth,
        })
    }

    /// Whether leaf outputs are path-smoothed (and so fixed at split time).
    pub(super) fn smoothing(&self) -> bool {
        self.path_smooth > 0.0
    }

    /// Find the best split of one node from its histogram, under the enabled
    /// options. `features` is already filtered by column sampling and
    /// interaction constraints; the candidate order and tie rule follow the
    /// builder's XGBoost search.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate(
        &self,
        cuts: &HistCuts,
        dense: bool,
        hist: &[GradStats],
        total: GradStats,
        features: &[u32],
        bounds: Bounds,
        cons: &MonotoneConstraints,
        reg: &RegParams,
        node: NodeCtx,
    ) -> BestSplit {
        let scorer = if self.smoothing() {
            Scorer::Smooth(Smoothed::new(self.path_smooth, total, node, reg))
        } else {
            Scorer::Xgb {
                root_gain: xgb_node_gain(total, reg, bounds),
            }
        };
        let mut rng = self
            .extra_seed
            .map(|seed| StdRng::seed_from_u64(node_seed(seed, node.tree_seed, node.id)));
        let constrained = cons.is_active();
        let mut best = BestSplit::none();
        for &f in features {
            let (fs, fe) = cuts.feature_bins(f as usize);
            if fe <= fs + 1 {
                continue; // degenerate feature, no interior boundary
            }
            let dir = cons.dir(f as usize);
            let ctx = Candidate {
                feature: f,
                total,
                bounds,
                dir,
                reg,
                scorer: &scorer,
            };
            if cuts.is_categorical(f as usize) {
                let mut cats: Vec<(u32, GradStats)> = (fs..fe)
                    .filter(|&i| hist[i].hess > 0.0)
                    .map(|i| (cuts.cut_value(i) as u32, hist[i]))
                    .collect();
                ctx.categorical(&mut best, &mut cats, constrained, rng.as_mut());
            } else if let Some(rng) = rng.as_mut() {
                ctx.random_numeric(&mut best, &hist[fs..fe], fs, dense, rng);
            } else {
                ctx.all_numeric(&mut best, &hist[fs..fe], fs, dense);
            }
        }
        best
    }
}

/// With path smoothing every leaf already holds the smoothed output its
/// parent's split recorded; only a root that never split still needs its own
/// (unsmoothed) output.
pub(super) fn finalize_smoothed_leaves(tree: &mut RegTree, root: GradStats, reg: &RegParams) {
    if tree.node(0).is_leaf() {
        tree.set_leaf_value(0, raw_output(root, reg) as f32);
    }
}

/// The unsmoothed output of the root, which path smoothing pulls the root's
/// children toward (LightGBM does not smooth the root).
pub(super) fn root_output(root: GradStats, reg: &RegParams) -> f64 {
    raw_output(root, reg)
}

/// Regularized optimum `−Tα(G)/(H+λ)`, clipped to `max_delta_step`, in `f64`.
fn raw_output(stats: GradStats, reg: &RegParams) -> f64 {
    if stats.hess <= 0.0 {
        return 0.0;
    }
    let w = -threshold_l1(stats.grad, reg.alpha) / (stats.hess + reg.lambda);
    if reg.max_delta_step > 0.0 && w.abs() > reg.max_delta_step {
        reg.max_delta_step.copysign(w)
    } else {
        w
    }
}

/// LightGBM's path smoothing of output `w` of a node with `n` rows toward its
/// parent's output, with strength `s`.
fn smooth(w: f64, n: f64, s: f64, parent: f64) -> f64 {
    let ratio = n / s;
    w * ratio / (ratio + 1.0) + parent / (ratio + 1.0)
}

/// Seed of one node's `extra_trees` draws (`SplitMix64` finalizer chain).
fn node_seed(extra_seed: u64, tree_seed: u64, node: usize) -> u64 {
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    mix(mix(mix(extra_seed) ^ tree_seed) ^ node as u64)
}

/// Path-smoothing state of one node's split search.
struct Smoothed {
    strength: f64,
    rows: f64,
    parent: f64,
    /// Rows per unit of Hessian, estimating a candidate child's row count.
    rows_per_hess: f64,
    /// Score of the node itself smoothed toward its own output.
    shift: f64,
}

impl Smoothed {
    fn new(strength: f64, total: GradStats, node: NodeCtx, reg: &RegParams) -> Self {
        let rows = node.rows as f64;
        let own = smooth(raw_output(total, reg), rows, strength, node.output);
        Smoothed {
            strength,
            rows,
            parent: node.output,
            rows_per_hess: rows / total.hess,
            shift: gain_at_weight(total, reg, own),
        }
    }

    /// Loss change and bounded, smoothed child outputs of one candidate, or
    /// `None` when a child lacks Hessian or the monotone direction fails.
    fn score(
        &self,
        left: GradStats,
        right: GradStats,
        reg: &RegParams,
        bounds: Bounds,
        dir: i8,
    ) -> Option<(f64, f64, f64)> {
        let mcw = reg.min_child_weight;
        if !(left.hess > 0.0 && right.hess > 0.0 && left.hess >= mcw && right.hess >= mcw) {
            return None;
        }
        let left_rows = (left.hess * self.rows_per_hess)
            .round()
            .clamp(0.0, self.rows);
        let output = |stats: GradStats, rows: f64| {
            bounds.clamp(smooth(
                raw_output(stats, reg),
                rows,
                self.strength,
                self.parent,
            ))
        };
        let wl = output(left, left_rows);
        let wr = output(right, self.rows - left_rows);
        if !satisfies(dir, wl, wr) {
            return None;
        }
        let gain = gain_at_weight(left, reg, wl) + gain_at_weight(right, reg, wr) - self.shift;
        Some((gain, wl, wr))
    }
}

/// How candidates are scored: XGBoost's arithmetic, or at smoothed outputs.
enum Scorer {
    Xgb { root_gain: f32 },
    Smooth(Smoothed),
}

/// One feature's candidate evaluation context within a node.
struct Candidate<'a> {
    feature: u32,
    total: GradStats,
    bounds: Bounds,
    dir: i8,
    reg: &'a RegParams,
    scorer: &'a Scorer,
}

impl Candidate<'_> {
    /// Score a numeric candidate and keep it if it beats `best` (XGBoost's
    /// `f32` comparison and tie rule).
    fn offer_numeric(
        &self,
        best: &mut BestSplit,
        pos: SplitPos,
        default_left: bool,
        left: GradStats,
        right: GradStats,
    ) {
        let scored = match self.scorer {
            Scorer::Xgb { root_gain } => {
                xgb_loss_chg(left, right, *root_gain, self.reg, self.bounds, self.dir)
            }
            Scorer::Smooth(smoothed) => smoothed
                .score(left, right, self.reg, self.bounds, self.dir)
                .map(|(g, wl, wr)| (g as f32, wl as f32, wr as f32)),
        };
        if let Some((loss_chg, wl, wr)) = scored {
            xgb_update(
                best,
                loss_chg,
                self.feature,
                pos,
                default_left,
                left,
                right,
                wl,
                wr,
            );
        }
    }

    /// Every boundary of the feature: the forward pass (missing right) and,
    /// when the node has missing values for it, the backward pass (missing
    /// left), exactly as the builder's XGBoost search enumerates them.
    fn all_numeric(&self, best: &mut BestSplit, bins: &[GradStats], first: usize, dense: bool) {
        let mut acc = GradStats::default();
        for (offset, &bin) in bins.iter().enumerate() {
            acc.add(bin);
            let right = self.total.sub(acc);
            self.offer_numeric(best, SplitPos::Bin(first + offset), false, acc, right);
        }
        if dense || acc == self.total {
            return;
        }
        let mut suffix = GradStats::default();
        for offset in (0..bins.len()).rev() {
            suffix.add(bins[offset]);
            let left = self.total.sub(suffix);
            let pos = if offset == 0 {
                SplitPos::BelowBins
            } else {
                SplitPos::Bin(first + offset - 1)
            };
            self.offer_numeric(best, pos, true, left, suffix);
        }
    }

    /// One random boundary between the node's lowest and highest occupied
    /// bin, scored with missing values right and (when present) left.
    fn random_numeric(
        &self,
        best: &mut BestSplit,
        bins: &[GradStats],
        first: usize,
        dense: bool,
        rng: &mut StdRng,
    ) {
        let occupied = |i: &usize| bins[*i].hess > 0.0;
        let (Some(lo), Some(hi)) = (
            (0..bins.len()).find(occupied),
            (0..bins.len()).rev().find(occupied),
        ) else {
            return;
        };
        if hi <= lo {
            return; // a single occupied bin cannot be split
        }
        // Bins `<= cut` go left, so both sides hold an occupied bin.
        let cut = rng.random_range(lo..hi);
        let mut left = GradStats::default();
        for &bin in &bins[..=cut] {
            left.add(bin);
        }
        let mut suffix = GradStats::default();
        for &bin in &bins[cut + 1..] {
            suffix.add(bin);
        }
        let pos = SplitPos::Bin(first + cut);
        self.offer_numeric(best, pos, false, left, self.total.sub(left));
        let mut present = left;
        present.add(suffix);
        if !dense && present != self.total {
            self.offer_numeric(best, pos, true, self.total.sub(suffix), suffix);
        }
    }

    /// Prefix partitions of the categories ordered by gradient/Hessian ratio
    /// (the builder's sorted-partition sweep); with `rng`, one random prefix.
    fn categorical(
        &self,
        best: &mut BestSplit,
        cats: &mut [(u32, GradStats)],
        constrained: bool,
        rng: Option<&mut StdRng>,
    ) {
        if cats.len() < 2 {
            return; // no interior partition
        }
        let reg = self.reg;
        let ratio = |s: GradStats| s.grad / (s.hess + reg.lambda);
        cats.sort_by(|a, b| ratio(a.1).total_cmp(&ratio(b.1)));
        let chosen = rng.map(|rng| rng.random_range(1..cats.len()));
        let parent_gain = match self.scorer {
            Scorer::Xgb { root_gain } => f64::from(*root_gain),
            Scorer::Smooth(_) => 0.0,
        };
        let mut left = GradStats::default();
        let mut cats_left: Vec<u32> = Vec::new();
        for (i, &(cat, stats)) in cats[..cats.len() - 1].iter().enumerate() {
            left.add(stats);
            cats_left.push(cat);
            if chosen.is_some_and(|len| len != i + 1) {
                continue;
            }
            // `total` includes any missing mass, which stays on the right.
            let right = self.total.sub(left);
            let scored = match self.scorer {
                Scorer::Xgb { .. } => {
                    if left.hess < reg.min_child_weight || right.hess < reg.min_child_weight {
                        continue;
                    }
                    candidate_gain(
                        left,
                        right,
                        parent_gain,
                        self.bounds,
                        self.dir,
                        constrained,
                        reg,
                    )
                }
                Scorer::Smooth(smoothed) => smoothed.score(left, right, reg, self.bounds, self.dir),
            };
            let Some((gain, wl, wr)) = scored else {
                continue;
            };
            if gain > best.loss_chg + K_RT_EPS {
                *best = BestSplit::categorical(
                    gain,
                    self.feature,
                    left,
                    right,
                    wl,
                    wr,
                    cats_left.clone(),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::gp;
    use super::*;
    use crate::data::DMatrix;
    use crate::data::ghist::GHistIndex;
    use crate::objective::GradPair;
    use crate::tree::builder::{HistTreeBuilder, all_rows};
    use crate::tree::sampler::ColumnSampler;

    fn binned(data: &DMatrix) -> GHistIndex {
        GHistIndex::from_dmatrix(data, HistCuts::from_dmatrix(data, 64))
    }

    /// Pseudo-random rows with an additive target, some values missing.
    fn noisy(n: usize, features: usize, missing: bool) -> (DMatrix, Vec<GradPair>) {
        let mut state = 7u64;
        let mut values = Vec::with_capacity(n * features);
        let mut gpair = Vec::with_capacity(n);
        for row in 0..n {
            let mut target = 0.0;
            for col in 0..features {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let v = (state >> 40) as f32 / (1u64 << 24) as f32;
                target += v * (col + 1) as f32;
                let absent = missing && (row * 7 + col * 3) % 10 == 0;
                values.push(if absent { f32::NAN } else { v });
            }
            gpair.push(gp(features as f32 - target, 1.0));
        }
        (DMatrix::from_dense(&values, n, features).unwrap(), gpair)
    }

    fn grow(params: &TrainingParams, data: &DMatrix, gpair: &[GradPair], seed: u64) -> RegTree {
        let ghist = binned(data);
        let features = (0..data.n_cols() as u32).collect();
        HistTreeBuilder::new(params).build(
            &ghist,
            gpair,
            &all_rows(data.n_rows()),
            &mut ColumnSampler::new(features, 1.0, 1.0, seed),
        )
    }

    fn extra(depth: usize) -> TrainingParams {
        TrainingParams::builder()
            .max_depth(depth)
            .extra_trees(true)
            .build()
            .unwrap()
    }

    #[test]
    fn extra_trees_are_reproducible_and_vary_with_the_tree_seed() {
        let (data, gpair) = noisy(600, 4, true);
        let params = extra(4);
        let a = grow(&params, &data, &gpair, 11);
        assert_eq!(a, grow(&params, &data, &gpair, 11));
        let exhaustive = grow(
            &TrainingParams::builder().max_depth(4).build().unwrap(),
            &data,
            &gpair,
            11,
        );
        assert_ne!(
            a, exhaustive,
            "random thresholds must differ from the best ones"
        );
        assert_ne!(a, grow(&params, &data, &gpair, 12));
        let reseeded = TrainingParams {
            extra_seed: 7,
            ..params
        };
        assert_ne!(a, grow(&reseeded, &data, &gpair, 11));
    }

    #[test]
    fn extra_trees_thresholds_fall_inside_each_node_value_range() {
        let (data, gpair) = noisy(800, 3, true);
        let tree = grow(&extra(6), &data, &gpair, 3);
        assert!(tree.num_nodes() > 15, "tree too small to be meaningful");
        // Route every row and check each split against the present values of
        // the rows that reached its node: some fall below the threshold and
        // some at or above it, so neither child is empty.
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); tree.num_nodes()];
        for row in 0..data.n_rows() {
            let mut nid = 0usize;
            loop {
                members[nid].push(row);
                let node = tree.node(nid);
                if node.is_leaf() {
                    break;
                }
                let go_left = data
                    .get(row, node.split_feature as usize)
                    .map_or(node.default_left, |v| v < node.split_cond);
                nid = if go_left { node.left } else { node.right } as usize;
            }
        }
        for (nid, node) in tree.nodes().iter().enumerate() {
            if node.is_leaf() {
                continue;
            }
            let present: Vec<f32> = members[nid]
                .iter()
                .filter_map(|&r| data.get(r, node.split_feature as usize))
                .collect();
            let below = present.iter().filter(|&&v| v < node.split_cond).count();
            assert!(
                below > 0 && below < present.len(),
                "node {nid}: threshold {} outside the node's range",
                node.split_cond
            );
        }
    }

    #[test]
    fn extra_trees_split_categorical_features_on_a_random_prefix() {
        use crate::data::FeatureType;
        let n = 400;
        let x: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| gp(if i % 8 < 4 { 1.0 } else { -1.0 }, 1.0))
            .collect();
        let data = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_feature_types(&[FeatureType::Categorical])
            .unwrap();
        let sets: std::collections::BTreeSet<usize> = (0..16)
            .map(|seed| {
                let tree = grow(&extra(1), &data, &gpair, seed);
                let root = tree.node(0);
                assert!(root.is_categorical);
                (root.cat_end - root.cat_begin) as usize
            })
            .collect();
        assert!(sets.len() > 1, "prefix length never varied: {sets:?}");
    }

    /// Four rows per quarter of `[0, 1)` with distinct targets and unit
    /// Hessians, so every count estimate is exact.
    fn quarters() -> (DMatrix, Vec<GradPair>, [f64; 4]) {
        let targets = [-3.0, -1.0, 2.0, 6.0];
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for (q, &t) in targets.iter().enumerate() {
            for j in 0..4 {
                x.push(q as f32 * 0.25 + j as f32 * 0.05);
                gpair.push(gp(-(t as f32), 1.0));
            }
        }
        (DMatrix::from_dense(&x, 16, 1).unwrap(), gpair, targets)
    }

    #[test]
    fn path_smoothing_pulls_every_leaf_toward_its_parent() {
        let (data, gpair, targets) = quarters();
        let (lambda, s) = (1.0, 3.0);
        let params = TrainingParams::builder()
            .max_depth(2)
            .lambda(lambda)
            .min_child_weight(0.0)
            .path_smooth(s)
            .build()
            .unwrap();
        let tree = grow(&params, &data, &gpair, 0);
        assert_eq!(tree.num_leaves(), 4);
        // Four unit-Hessian rows per quarter: G = −4·Σt and H = 4·len.
        let optimum = |ts: &[f64]| 4.0 * ts.iter().sum::<f64>() / (4.0 * ts.len() as f64 + lambda);
        let blend =
            |raw: f64, n: f64, parent: f64| raw * (n / s) / (n / s + 1.0) + parent / (n / s + 1.0);
        let root = optimum(&targets);
        let halves = [
            blend(optimum(&targets[..2]), 8.0, root),
            blend(optimum(&targets[2..]), 8.0, root),
        ];
        for (q, &t) in targets.iter().enumerate() {
            let parent = halves[q / 2];
            let want = blend(optimum(&[t]), 4.0, parent);
            let got = tree.predict_row(&data, q * 4);
            assert!(
                (f64::from(got) - want).abs() < 1e-5,
                "quarter {q}: {got} vs {want}"
            );
            let unsmoothed = optimum(&[t]);
            assert!((want - parent).abs() < (unsmoothed - parent).abs());
        }
    }

    #[test]
    fn stronger_smoothing_moves_leaves_closer_to_the_root() {
        let (data, gpair) = noisy(500, 3, false);
        let spread = |s: f64| {
            let params = TrainingParams::builder()
                .max_depth(3)
                .path_smooth(s)
                .build()
                .unwrap();
            let tree = grow(&params, &data, &gpair, 0);
            let preds: Vec<f32> = (0..data.n_rows())
                .map(|r| tree.predict_row(&data, r))
                .collect();
            let mean = preds.iter().sum::<f32>() / preds.len() as f32;
            preds.iter().map(|p| (p - mean).abs()).sum::<f32>()
        };
        let (weak, strong) = (spread(1.0), spread(1000.0));
        assert!(strong < weak * 0.5, "spread {strong} vs {weak}");
    }
}
