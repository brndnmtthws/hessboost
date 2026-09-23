//! Exact greedy tree construction (XGBoost's `tree_method=exact`, `ColMaker`).
//!
//! For each node we scan every feature's value-sorted entries and evaluate every
//! candidate threshold the way XGBoost's `ColMaker` does: a backward
//! (descending) scan sends missing values left and runs for every feature; a
//! forward (ascending) scan sends them right and runs only for features that
//! actually have missing values and are not constant. Each scan closes with the
//! endpoint candidate that puts every present value on one side and the
//! missing mass on the other. Growth is level-wise (depth-wise): a whole level
//! is scanned per feature pass.
//!
//! Monotone and interaction constraints are honored during split search.

use super::{
    BELOW_ALL_VALUES, BestSplit, InteractionState, K_RT_EPS, SplitPos, build_interaction_sets,
    finalize_leaf_values, next_allowed, permits, sum_rows, sweep_categorical, xgb_loss_chg,
    xgb_node_gain, xgb_update,
};
use crate::config::TrainingParams;
use crate::data::{DMatrix, FeatureType};
use crate::objective::GradPair;
use crate::tree::constraints::{Bounds, MonotoneConstraints, child_bounds};
use crate::tree::gain::{GradStats, RegParams};
use crate::tree::regtree::RegTree;
use crate::tree::sampler::ColumnSampler;
use std::collections::HashMap;

/// Value-sorted column index over a [`DMatrix`], built once and reused across
/// boosting rounds. Within each column, `(row, value)` pairs are sorted by
/// ascending value. Missing entries are omitted (sparsity-aware).
#[derive(Debug, Clone)]
pub struct SortedColumns {
    n_rows: usize,
    n_cols: usize,
    col_ptr: Vec<usize>,
    rows: Vec<u32>,
    vals: Vec<f32>,
}

impl SortedColumns {
    /// Build the value-sorted column index from a dataset.
    pub fn from_dmatrix(data: &DMatrix) -> Self {
        let csc = data.to_csc();
        let n_cols = csc.n_cols();
        let mut col_ptr = vec![0usize; n_cols + 1];
        #[allow(clippy::needless_range_loop)]
        for c in 0..n_cols {
            col_ptr[c + 1] = col_ptr[c] + csc.col_len(c);
        }
        let nnz = col_ptr[n_cols];
        let mut rows = vec![0u32; nnz];
        let mut vals = vec![0f32; nnz];
        #[allow(clippy::needless_range_loop)]
        for c in 0..n_cols {
            let (crows, cvals) = csc.column(c);
            // Sort this column's entries by ascending value (NaN cannot appear:
            // missing entries were excluded when building the CSC).
            let mut order: Vec<usize> = (0..crows.len()).collect();
            order.sort_by(|&a, &b| cvals[a].partial_cmp(&cvals[b]).unwrap());
            let base = col_ptr[c];
            for (k, &o) in order.iter().enumerate() {
                rows[base + k] = crows[o];
                vals[base + k] = cvals[o];
            }
        }
        SortedColumns {
            n_rows: csc.n_rows(),
            n_cols,
            col_ptr,
            rows,
            vals,
        }
    }

    /// Number of rows in the source matrix.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    #[inline]
    fn column(&self, f: usize) -> (&[u32], &[f32]) {
        let (s, e) = (self.col_ptr[f], self.col_ptr[f + 1]);
        (&self.rows[s..e], &self.vals[s..e])
    }
}

/// A split that was applied at a level, used to route rows into their children.
struct Split {
    nid: usize,
    feature: u32,
    threshold: f32,
    default_left: bool,
    is_categorical: bool,
    cat_left: Vec<u32>,
    left_id: usize,
    right_id: usize,
}

/// Exact greedy tree builder.
pub struct ExactTreeBuilder<'a> {
    params: &'a TrainingParams,
    reg: RegParams,
    cons: MonotoneConstraints,
    interaction_sets: Option<Vec<Vec<u32>>>,
}

impl<'a> ExactTreeBuilder<'a> {
    /// Create a builder bound to a training configuration.
    pub fn new(params: &'a TrainingParams) -> Self {
        ExactTreeBuilder {
            params,
            reg: RegParams::from_params(params),
            cons: MonotoneConstraints::from_params(&params.monotone_constraints),
            interaction_sets: build_interaction_sets(&params.interaction_constraints),
        }
    }

    /// Grow a single tree.
    ///
    /// * `cols`: value-sorted column index over the *full* dataset.
    /// * `data`: the dataset (for routing rows after a split).
    /// * `gpair`: per-row gradient/Hessian (length = dataset rows).
    /// * `row_subset`: the sampled rows to train this tree on.
    /// * `sampler`: per-tree column sampler. The exact builder draws one subset
    ///   per level (shared across that level's nodes).
    pub fn build(
        &self,
        cols: &SortedColumns,
        data: &DMatrix,
        gpair: &[GradPair],
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
    ) -> RegTree {
        let n_rows = cols.n_rows();

        // node_of_row[r] = current node id for row r, or -1 if r is not sampled.
        let mut node_of_row = vec![-1i32; n_rows];
        for &r in row_subset {
            node_of_row[r as usize] = 0;
        }
        let root = sum_rows(gpair, row_subset);

        let mut tree = RegTree::with_root(root.hess as f32);
        let mut node_stats: Vec<GradStats> = vec![root];
        // Per-node monotone weight bounds (default `±∞` when unconstrained).
        let mut node_bounds: Vec<Bounds> = vec![Bounds::default()];
        let mut node_allowed: Vec<Option<InteractionState>> = vec![None];

        // With no monotone constraints the closed-form gain path is exact; the
        // bounded path is used otherwise.
        let constrained = self.cons.is_active();
        let ftypes = data.feature_types();

        let depth_limit = if self.params.max_depth == 0 {
            usize::MAX
        } else {
            self.params.max_depth
        };

        let mut active: Vec<usize> = vec![0];
        let mut depth = 0;

        while depth < depth_limit && !active.is_empty() {
            // One column subset for the whole level (bylevel ∘ bynode).
            let feature_subset = sampler.sample();
            let k = active.len();
            // slot_of_node maps an active node id to its dense slot index.
            let mut slot_of_node = vec![usize::MAX; tree.num_nodes()];
            for (slot, &nid) in active.iter().enumerate() {
                slot_of_node[nid] = slot;
            }

            let mut best = vec![BestSplit::none(); k];

            // XGBoost's `root_gain`: the node's own structure score as `f32`.
            let mut root_gain = vec![0f32; k];
            for (slot, &nid) in active.iter().enumerate() {
                root_gain[slot] = xgb_node_gain(node_stats[nid], &self.reg, node_bounds[nid]);
            }

            // Scratch buffers, reused per feature and scan direction.
            let mut acc = vec![GradStats::default(); k];
            let mut last_val = vec![0f32; k];

            for &f in &feature_subset {
                let (crows, cvals) = cols.column(f as usize);
                let dir = self.cons.dir(f as usize);

                // Categorical features use a set-membership split instead of a
                // numeric threshold.
                if ftypes[f as usize] == FeatureType::Categorical {
                    // Gather per-node, per-category statistics for this feature.
                    let mut cat_stats: Vec<HashMap<u32, GradStats>> = vec![HashMap::new(); k];
                    for (&rr, &val) in crows.iter().zip(cvals) {
                        let r = rr as usize;
                        let nid = node_of_row[r];
                        if nid < 0 {
                            continue;
                        }
                        let slot = slot_of_node[nid as usize];
                        if slot == usize::MAX {
                            continue;
                        }
                        if !permits(node_allowed[nid as usize].as_ref(), f) {
                            continue;
                        }
                        let gp = gpair[r];
                        cat_stats[slot]
                            .entry(val as u32)
                            .or_default()
                            .add(GradStats::from_pair(gp));
                    }
                    for (slot, &nid) in active.iter().enumerate() {
                        if !permits(node_allowed[nid].as_ref(), f) {
                            continue;
                        }
                        let mut cats: Vec<(u32, GradStats)> =
                            cat_stats[slot].iter().map(|(&c, &s)| (c, s)).collect();
                        sweep_categorical(
                            &mut best[slot],
                            &mut cats,
                            node_stats[nid],
                            f64::from(root_gain[slot]),
                            node_bounds[nid],
                            dir,
                            constrained,
                            &self.reg,
                            f,
                        );
                    }
                    continue;
                }

                // `NeedForwardSearch`: only a column with missing values that is
                // not constant scans forward (missing right); every column
                // scans backward (missing left).
                let indicator = !cvals.is_empty() && cvals[0] == cvals[cvals.len() - 1];
                let scan = |d_step: i8,
                            acc: &mut [GradStats],
                            last_val: &mut [f32],
                            best: &mut [BestSplit]| {
                    acc.fill(GradStats::default());
                    let mut visit = |r: usize, val: f32| {
                        let nid = node_of_row[r];
                        if nid < 0 {
                            return;
                        }
                        let slot = slot_of_node[nid as usize];
                        if slot == usize::MAX || !permits(node_allowed[nid as usize].as_ref(), f) {
                            return;
                        }
                        let e = &mut acc[slot];
                        // `UpdateEnumeration`: the first rows with positive Hessian
                        // only seed the running statistics.
                        if e.hess != 0.0
                            && val != last_val[slot]
                            && e.hess >= self.reg.min_child_weight
                        {
                            let c = node_stats[nid as usize].sub(*e);
                            if c.hess >= self.reg.min_child_weight {
                                let (left, right) = if d_step < 0 { (c, *e) } else { (*e, c) };
                                // ColMaker's midpoint `(fvalue + last) * 0.5f`
                                // overflows to `±inf` for two same-sign values
                                // near `±f32::MAX`. Only then fall back to the
                                // halved form, which is finite and still lies
                                // between the two values, so the partition is
                                // unchanged. Trees must stay finite.
                                let last = last_val[slot];
                                let mut mid = f32::midpoint(val, last);
                                if !mid.is_finite() {
                                    mid = val * 0.5 + last * 0.5;
                                }
                                let thr = if mid == val { last } else { mid };
                                self.try_split(
                                    &mut best[slot],
                                    left,
                                    right,
                                    root_gain[slot],
                                    node_bounds[nid as usize],
                                    dir,
                                    f,
                                    thr,
                                    d_step < 0,
                                );
                            }
                        }
                        e.add(GradStats::from_pair(gpair[r]));
                        last_val[slot] = val;
                    };
                    if d_step > 0 {
                        for (&rr, &val) in crows.iter().zip(cvals) {
                            visit(rr as usize, val);
                        }
                    } else {
                        for (&rr, &val) in crows.iter().zip(cvals).rev() {
                            visit(rr as usize, val);
                        }
                    }
                    // Endpoint: every present value on the scanned side, the
                    // missing mass on the other.
                    for (slot, &nid) in active.iter().enumerate() {
                        let e = acc[slot];
                        let c = node_stats[nid].sub(e);
                        if e.hess >= self.reg.min_child_weight
                            && c.hess >= self.reg.min_child_weight
                        {
                            let last = last_val[slot];
                            let gap = last.abs() + K_RT_EPS as f32;
                            let thr = if d_step > 0 { last + gap } else { last - gap };
                            // ColMaker's `last_fvalue ± delta` overflows to `±inf`
                            // for `|last|` near `f32::MAX`; the tree must stay
                            // finite. Backward (missing left) needs every present
                            // `v >= thr`, which `BELOW_ALL_VALUES` satisfies.
                            // Forward (missing right) needs `v < thr`: `f32::MAX`
                            // works unless `last` is itself `f32::MAX`, in which
                            // case no finite threshold represents the partition
                            // and the candidate is skipped.
                            let thr = if thr.is_finite() {
                                thr
                            } else if d_step < 0 {
                                BELOW_ALL_VALUES
                            } else if last < f32::MAX {
                                f32::MAX
                            } else {
                                continue;
                            };
                            let (left, right) = if d_step < 0 { (c, e) } else { (e, c) };
                            self.try_split(
                                &mut best[slot],
                                left,
                                right,
                                root_gain[slot],
                                node_bounds[nid],
                                dir,
                                f,
                                thr,
                                d_step < 0,
                            );
                        }
                    }
                };
                if cvals.len() < n_rows && !indicator {
                    scan(1, &mut acc, &mut last_val, &mut best);
                }
                scan(-1, &mut acc, &mut last_val, &mut best);
            }

            let mut next_active = Vec::new();
            let mut splits: Vec<Split> = Vec::new();

            for &nid in &active {
                let slot = slot_of_node[nid];
                let b = &best[slot];
                if !b.valid(self.params.gamma, self.reg.min_child_weight) {
                    continue; // stays a leaf; value finalized below
                }

                // Monotone child bounds derived from the (bounded) child weights.
                let dir = self.cons.dir(b.feature as usize);
                let (lb_bounds, rb_bounds) =
                    child_bounds(node_bounds[nid], dir, b.w_left, b.w_right);

                // Children carry XGBoost's bounded `f32` weight, so the
                // `leaf_value` field of nodes that later split records the value
                // they had as a leaf at expansion time. Leaves are overwritten
                // by the finalize pass below.
                let (lw, rw) = (b.w_left as f32, b.w_right as f32);
                let (left_id, right_id) = if b.is_categorical {
                    tree.expand_categorical(
                        nid,
                        b.feature,
                        &b.cat_left,
                        b.default_left,
                        lw,
                        b.left.hess as f32,
                        rw,
                        b.right.hess as f32,
                    )
                } else {
                    tree.expand(
                        nid,
                        b.feature,
                        b.threshold,
                        b.default_left,
                        lw,
                        b.left.hess as f32,
                        rw,
                        b.right.hess as f32,
                    )
                };
                tree.set_split_gain(nid, b.loss_chg as f32);
                debug_assert_eq!(left_id, node_stats.len());
                node_stats.push(b.left);
                node_stats.push(b.right);
                node_bounds.push(lb_bounds);
                node_bounds.push(rb_bounds);
                let allowed = next_allowed(
                    node_allowed[nid].as_ref(),
                    b.feature,
                    self.interaction_sets.as_deref(),
                );
                node_allowed.push(allowed.clone());
                node_allowed.push(allowed);
                next_active.push(left_id);
                next_active.push(right_id);
                splits.push(Split {
                    nid,
                    feature: b.feature,
                    threshold: b.threshold,
                    default_left: b.default_left,
                    is_categorical: b.is_categorical,
                    cat_left: b.cat_left.clone(),
                    left_id,
                    right_id,
                });
            }

            // Route each sampled row into its child for the nodes that split.
            if !splits.is_empty() {
                // Map nid -> split info for O(1) routing.
                let mut split_of_node = vec![usize::MAX; tree.num_nodes()];
                for (idx, s) in splits.iter().enumerate() {
                    split_of_node[s.nid] = idx;
                }
                #[allow(clippy::needless_range_loop)]
                for r in 0..n_rows {
                    let nid = node_of_row[r];
                    if nid < 0 {
                        continue;
                    }
                    let si = split_of_node[nid as usize];
                    if si == usize::MAX {
                        continue;
                    }
                    let s = &splits[si];
                    let go_left = match data.get(r, s.feature as usize) {
                        Some(v) => {
                            if s.is_categorical {
                                // Present categories in the left set go left;
                                // every other present category goes right.
                                s.cat_left.contains(&(v as u32))
                            } else {
                                v < s.threshold
                            }
                        }
                        None => s.default_left,
                    };
                    node_of_row[r] = if go_left {
                        s.left_id as i32
                    } else {
                        s.right_id as i32
                    };
                }
            }

            active = next_active;
            depth += 1;
        }

        // Finalize every leaf's weight (respecting each leaf's monotone bounds).
        finalize_leaf_values(&mut tree, &node_stats, &node_bounds, &self.reg);
        tree
    }

    /// Evaluate one candidate partition exactly as XGBoost's `ColMaker` does
    /// (`CalcSplitGain − root_gain` in `f32`, `SplitEntry::Update` tie rule)
    /// and record it in `best` when it wins.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn try_split(
        &self,
        best: &mut BestSplit,
        left: GradStats,
        right: GradStats,
        root_gain: f32,
        bounds: Bounds,
        dir: i8,
        feature: u32,
        threshold: f32,
        default_left: bool,
    ) {
        if let Some((loss_chg, wl, wr)) =
            xgb_loss_chg(left, right, root_gain, &self.reg, bounds, dir)
        {
            xgb_update(
                best,
                loss_chg,
                feature,
                SplitPos::Value(threshold),
                default_left,
                left,
                right,
                wl,
                wr,
            );
        }
    }
}

/// Utility: the full row index `0..n_rows` as `u32` (no subsampling).
pub fn all_rows(n_rows: usize) -> Vec<u32> {
    (0..n_rows as u32).collect()
}

/// Utility: the full feature index `0..n_cols` as `u32` (no column sampling).
pub fn all_features(n_cols: usize) -> Vec<u32> {
    (0..n_cols as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{gp, monotone_v_shape_data};
    use super::*;
    use crate::config::TrainingParams;

    /// A clean separable problem: feature 0 perfectly separates the sign of the
    /// gradient at threshold 0.5, so the root should split there.
    #[test]
    fn splits_on_separating_feature() {
        // 4 rows, 1 feature. values 0,0,1,1. gradients push low->+, high->-.
        let x = vec![0.0f32, 0.0, 1.0, 1.0];
        let data = DMatrix::from_dense(&x, 4, 1).unwrap();
        let cols = SortedColumns::from_dmatrix(&data);
        // squared-error-like gradients: left group wants negative weight, right positive
        let gpair = vec![gp(1.0, 1.0), gp(1.0, 1.0), gp(-1.0, 1.0), gp(-1.0, 1.0)];

        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .build()
            .unwrap();
        let b = ExactTreeBuilder::new(&params);
        let tree = b.build(
            &cols,
            &data,
            &gpair,
            &all_rows(4),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );

        assert_eq!(
            tree.num_nodes(),
            3,
            "root should have split into two leaves"
        );
        let root = tree.node(0);
        assert_eq!(root.split_feature, 0);
        assert!((root.split_cond - 0.5).abs() < 1e-6);
        // left leaf: G=2,H=2 -> w=-1 ; right leaf: G=-2,H=2 -> w=+1
        assert!((tree.predict_row(&data, 0) - (-1.0)).abs() < 1e-6);
        assert!((tree.predict_row(&data, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn no_split_when_gain_below_gamma() {
        let x = vec![0.0f32, 1.0];
        let data = DMatrix::from_dense(&x, 2, 1).unwrap();
        let cols = SortedColumns::from_dmatrix(&data);
        let gpair = vec![gp(1.0, 1.0), gp(-1.0, 1.0)];
        let params = TrainingParams::builder()
            .max_depth(3)
            .gamma(1e9) // impossibly high min split loss
            .build()
            .unwrap();
        let b = ExactTreeBuilder::new(&params);
        let tree = b.build(
            &cols,
            &data,
            &gpair,
            &all_rows(2),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );
        assert_eq!(tree.num_nodes(), 1, "no split should be taken");
    }

    #[test]
    fn missing_values_pick_a_direction() {
        // 3 rows, feature 0 missing for row 2. Non-missing rows separate cleanly.
        let x = vec![0.0f32, 1.0, f32::NAN];
        let data = DMatrix::from_dense(&x, 3, 1).unwrap();
        let cols = SortedColumns::from_dmatrix(&data);
        // row2 (missing) shares the sign of the high group.
        let gpair = vec![gp(1.0, 1.0), gp(-1.0, 1.0), gp(-1.0, 1.0)];
        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .build()
            .unwrap();
        let b = ExactTreeBuilder::new(&params);
        let tree = b.build(
            &cols,
            &data,
            &gpair,
            &all_rows(3),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );
        assert_eq!(tree.num_nodes(), 3);
        // The missing row should be routed with the negative-gradient group
        // (right, positive weight). default_left should therefore be false.
        assert!(!tree.node(0).default_left);
        assert!(tree.predict_row(&data, 2) > 0.0);
    }

    #[test]
    fn monotone_increasing_is_enforced() {
        use crate::config::Monotone;
        let (data, gpair) = monotone_v_shape_data();
        let n = data.n_rows();
        let cols = SortedColumns::from_dmatrix(&data);
        let params = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(1.0)
            .monotone_constraints(vec![Monotone::Increasing])
            .build()
            .unwrap();
        let tree = ExactTreeBuilder::new(&params).build(
            &cols,
            &data,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );

        // Predictions must be non-decreasing in x under the increasing constraint.
        let mut prev = f32::NEG_INFINITY;
        for i in 0..n {
            let p = tree.predict_row(&data, i);
            assert!(
                p >= prev - 1e-5,
                "monotonicity violated at row {i}: {p} < {prev}"
            );
            prev = p;
        }

        // Sanity: the unconstrained fit on the same data is *not* monotone, so
        // the constraint is doing real work.
        let unconstrained = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(1.0)
            .build()
            .unwrap();
        let free = ExactTreeBuilder::new(&unconstrained).build(
            &cols,
            &data,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );
        let mut any_decrease = false;
        let mut prev = f32::NEG_INFINITY;
        for i in 0..n {
            let p = free.predict_row(&data, i);
            if p < prev - 1e-5 {
                any_decrease = true;
            }
            prev = p;
        }
        assert!(any_decrease, "unconstrained fit should be non-monotone");
    }

    #[test]
    fn categorical_splits_on_non_ordinal_pattern() {
        use crate::data::FeatureType;
        // 4 categories with a NON-ordinal target: {0,2} vs {1,3}. A numeric
        // threshold cannot separate them; a set-membership split can.
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for _ in 0..10 {
            for c in 0u32..4 {
                x.push(c as f32);
                // Residual around 0.5: even cats want negative weight, odd positive.
                let g = if c % 2 == 0 { 0.5 } else { -0.5 };
                gpair.push(gp(g, 1.0));
            }
        }
        let n = x.len();
        let data = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_feature_types(&[FeatureType::Categorical])
            .unwrap();
        let cols = SortedColumns::from_dmatrix(&data);
        let params = TrainingParams::builder()
            .max_depth(1)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(1.0)
            .build()
            .unwrap();
        let tree = ExactTreeBuilder::new(&params).build(
            &cols,
            &data,
            &gpair,
            &all_rows(n),
            &mut crate::tree::sampler::ColumnSampler::all(1),
        );

        assert_eq!(tree.num_nodes(), 3, "root should split");
        assert!(tree.node(0).is_categorical, "split should be categorical");

        // Prediction for a bare category value.
        let pred = |c: f32| tree.leaf_id_with(|_| Some(c));
        // Even categories share a leaf; odd categories share the other leaf.
        assert_eq!(pred(0.0), pred(2.0));
        assert_eq!(pred(1.0), pred(3.0));
        assert_ne!(pred(0.0), pred(1.0), "the two groups must be separated");

        // Even cats (positive grad) want negative weight; odd cats positive.
        let val = |c: f32| tree.node(pred(c)).leaf_value;
        assert!(val(0.0) < 0.0 && val(2.0) < 0.0);
        assert!(val(1.0) > 0.0 && val(3.0) > 0.0);
    }

    /// Train a small exact regressor on one feature and check that every split
    /// threshold is finite, prediction works (debug builds assert finiteness
    /// in the compact forest), and the model survives a native JSON round trip.
    /// Returns the predictions.
    fn train_exact_finite(x: &[f32], y: &[f32]) -> Vec<f32> {
        use crate::config::TreeMethod;
        use crate::learner::{BoostedModel, train};
        let n = x.len();
        let data = DMatrix::from_dense(x, n, 1)
            .unwrap()
            .with_labels(y)
            .unwrap();
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(TreeMethod::Exact)
            .max_depth(2)
            .eta(0.5)
            .build()
            .unwrap();
        let model = train(&params, &data, 3).unwrap();
        let mut n_splits = 0;
        for tree in model.trees() {
            for node in tree.nodes().iter().filter(|n| !n.is_leaf()) {
                n_splits += 1;
                assert!(
                    node.split_cond.is_finite(),
                    "non-finite split_cond {}",
                    node.split_cond
                );
            }
        }
        assert!(n_splits > 0, "the model should have split");
        let pred = model.predict(&data).unwrap();
        let back = BoostedModel::from_json(&model.to_json().unwrap()).unwrap();
        assert_eq!(back.predict(&data).unwrap(), pred);
        pred
    }

    #[test]
    fn backward_endpoint_near_neg_max_stays_finite() {
        // Constant `-f32::MAX` column plus missing rows: only the backward
        // endpoint separates them, and ColMaker's `last - (|last| + eps)`
        // is `-inf` there. The fallback `f32::MIN` keeps every present value
        // on the right (`v >= thr`) with missing on the left.
        let x = [
            -f32::MAX,
            -f32::MAX,
            -f32::MAX,
            -f32::MAX,
            f32::NAN,
            f32::NAN,
        ];
        let y = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0];
        let pred = train_exact_finite(&x, &y);
        assert!(
            pred[0] < pred[4],
            "present rows must be separated from missing"
        );
        assert_eq!(pred[0], pred[3]);
        assert_eq!(pred[4], pred[5]);
    }

    #[test]
    fn forward_endpoint_near_pos_max_stays_finite() {
        // `3e38` rows, zeros and missing rows: the forward scan's endpoint
        // (`last + (|last| + eps)`) overflows to `+inf`; the fallback
        // `f32::MAX` still sends every present value left (`v < thr`).
        let x = [3e38f32, 3e38, 0.0, 0.0, f32::NAN, f32::NAN];
        let y = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0];
        let pred = train_exact_finite(&x, &y);
        assert!(
            pred[0] < pred[4],
            "present rows must be separated from missing"
        );
        assert_eq!(pred[0], pred[2]);
        assert_eq!(pred[4], pred[5]);
    }

    #[test]
    fn midpoint_near_pos_max_stays_finite() {
        // Two same-sign values near `f32::MAX`: `(2e38 + 3e38) * 0.5` is
        // `+inf`; the halved form `2.5e38` lies between them.
        let x = [2e38f32, 2e38, 3e38, 3e38];
        let y = [0.0, 0.0, 10.0, 10.0];
        let pred = train_exact_finite(&x, &y);
        assert!(pred[0] < pred[2], "the two value groups must be separated");
        assert_eq!(pred[0], pred[1]);
        assert_eq!(pred[2], pred[3]);
    }
}
