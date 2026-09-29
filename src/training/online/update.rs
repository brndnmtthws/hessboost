//! One approximate update: the trees walked in order, each from the root
//! down, keeping ranked splits and regrowing the rest (see the
//! [module docs](super#method)).

use std::num::NonZeroUsize;
use std::ops::{ControlFlow, Range};

use super::cache::{Cache, NodeCache, accumulate, clear_subtree, stats_of};
use super::{UpdateReport, interrupted};
use crate::config::TrainingParams;
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::data::{DMatrix, LabelBounds, Labels, MetaInfo};
use crate::error::Result;
use crate::model::BoostedModel;
use crate::objective::{GRADIENT_BLOCK_ROWS, GradPair, Loss};
use crate::training::api::RoundEval;
use crate::tree::builder::HistTreeBuilder;
use crate::tree::builder::online::rank_split;
use crate::tree::gain::{GradStats, RegParams, calc_weight};
use crate::tree::sampler::ColumnSampler;
use crate::tree::{ChildLeaf, RegTree, SplitRule};

/// One approximate update.
pub(super) struct Incremental<'a> {
    pub(super) params: &'a TrainingParams,
    pub(super) tolerance: f64,
    pub(super) old: &'a DMatrix,
    pub(super) new: &'a DMatrix,
    pub(super) deleted: &'a [bool],
    pub(super) model: &'a BoostedModel,
}

/// A changed row's contribution routed down a tree.
struct Delta {
    /// Row of the deleted-rows matrix (`true`) or of the new data.
    deleted: bool,
    row: usize,
    /// The change of the row's gradient pair.
    g: GradStats,
    /// The row's global bins.
    bins: Vec<u32>,
}

impl Incremental<'_> {
    pub(super) fn run(
        &self,
        cache: &mut Cache,
        on_round: &mut dyn FnMut(RoundEval<'_>) -> ControlFlow<()>,
    ) -> Result<(Vec<RegTree>, UpdateReport)> {
        let (old, new) = (self.old, self.new);
        let deleted_rows: Vec<usize> = (0..old.n_rows()).filter(|&r| self.deleted[r]).collect();
        let gone = if deleted_rows.is_empty() {
            None
        } else {
            Some(old.select_rows(&deleted_rows)?)
        };
        let n_added = new.n_rows() + deleted_rows.len() - old.n_rows();
        let new_to_old: Vec<Option<usize>> = (0..old.n_rows())
            .filter(|&r| !self.deleted[r])
            .map(Some)
            .chain(std::iter::repeat_n(None, n_added))
            .collect();
        let objective = self.params.loss(1)?;
        let info = new.info();
        let reg = RegParams::from_params(self.params);
        let eta = self.params.eta as f32;
        let base = self.model.base_scores()[0];
        // Margins under the updated trees, kept for fresh rows only.
        let mut fresh: Vec<bool> = new_to_old.iter().map(Option::is_none).collect();
        let mut margins = vec![base; new.n_rows()];
        // The fresh rows' gradients (other rows' entries are stale).
        let mut fresh_grads = vec![GradPair::default(); new.n_rows()];
        let mut ghist: Option<GHistIndex> = None;
        let mut report = UpdateReport::default();
        cache.dense &= (0..new.n_rows())
            .filter(|&i| new_to_old[i].is_none())
            .all(|i| {
                let mut present = 0;
                new.for_row_entry(i, |_, _| present += 1);
                present == new.n_cols()
            });
        let mut trees: Vec<RegTree> = Vec::with_capacity(self.model.num_trees());
        for (m, old_tree) in self.model.trees().iter().enumerate() {
            let tc = &mut cache.trees[m];
            let old_grads = std::mem::take(&mut tc.grads);
            fresh_gradients(
                objective.as_ref(),
                &info,
                &margins,
                &fresh,
                &mut fresh_grads,
            );
            let cur: Vec<GradPair> = (0..new.n_rows())
                .map(|i| match new_to_old[i] {
                    Some(r) if !fresh[i] => old_grads[r],
                    _ => fresh_grads[i],
                })
                .collect();
            let cuts = &cache.cuts;
            let mut deltas: Vec<Delta> = Vec::new();
            if let Some(gone) = &gone {
                for (k, &r) in deleted_rows.iter().enumerate() {
                    deltas.push(Delta {
                        deleted: true,
                        row: k,
                        g: negate(stats_of(old_grads[r])),
                        bins: row_bins(cuts, gone, k),
                    });
                }
            }
            for i in (0..new.n_rows()).filter(|&i| fresh[i]) {
                let g = match new_to_old[i] {
                    // Kept rows sit on the same path in the kept structure:
                    // their change is the difference.
                    Some(r) => stats_of(cur[i]).sub(stats_of(old_grads[r])),
                    None => stats_of(cur[i]),
                };
                deltas.push(Delta {
                    deleted: false,
                    row: i,
                    g,
                    bins: row_bins(cuts, new, i),
                });
            }
            let ctx = TreeUpdate {
                run: self,
                dense: cache.dense,
                reg: &reg,
                eta,
                old_tree,
                gone: gone.as_ref(),
                cur: &cur,
                cuts,
            };
            let old_nodes = std::mem::take(&mut tc.nodes);
            let (tree, nodes, regrown_rows) =
                ctx.update(old_nodes, deltas, &mut ghist, &mut report);
            tc.nodes = nodes;
            tc.grads = cur;
            for (i, v) in margins.iter_mut().enumerate() {
                if fresh[i] {
                    *v += tree.predict_row(new, i);
                }
            }
            trees.push(tree);
            for i in regrown_rows {
                if !std::mem::replace(&mut fresh[i], true) {
                    // Tree by tree from the intercept, as prediction adds
                    // them (a sum of the trees first rounds differently).
                    margins[i] = trees
                        .iter()
                        .fold(base, |margin, t| margin + t.predict_row(new, i));
                }
            }
            if on_round(RoundEval::unscored(m)).is_break() {
                return Err(interrupted());
            }
        }
        report.rows_refreshed = fresh.iter().filter(|&&f| f).count();
        Ok((trees, report))
    }
}

/// Write the gradient pair of every `fresh` row of `info` at `margins` to
/// `out`, as one [`Loss::gradient_info`] call on every row would, without
/// evaluating the loss on most other rows. The loss runs on the runs of
/// [`GRADIENT_BLOCK_ROWS`]-row blocks holding a fresh row (a short last block
/// joins the one before), where the built-in per-row losses
/// `check_supported` admits compute each row exactly as on the whole batch;
/// other rows' entries of `out` are left unspecified.
fn fresh_gradients(
    objective: &dyn Loss,
    info: &MetaInfo<'_>,
    margins: &[f32],
    fresh: &[bool],
    out: &mut [GradPair],
) {
    let n = fresh.len();
    let blocks = (n / GRADIENT_BLOCK_ROWS).max(1);
    let block = |b: usize| {
        let start = b * GRADIENT_BLOCK_ROWS;
        start..if b + 1 == blocks {
            n
        } else {
            start + GRADIENT_BLOCK_ROWS
        }
    };
    let mut b = 0;
    while b < blocks {
        let first = b;
        while b < blocks && fresh[block(b)].contains(&true) {
            b += 1;
        }
        if b > first {
            let rows = block(first).start..block(b - 1).end;
            objective.gradient_info(
                &margins[rows.clone()],
                &row_info(info, rows.clone()),
                &mut out[rows],
            );
        }
        // Block `b`, if any, holds no fresh row.
        b += 1;
    }
}

/// The metadata of rows `rows` of `info`, which carries none of the
/// metadata `check_data` refuses (several targets, groups).
fn row_info<'a>(info: &MetaInfo<'a>, rows: Range<usize>) -> MetaInfo<'a> {
    debug_assert!(info.n_targets() == 1 && info.group.is_none());
    MetaInfo {
        n_rows: rows.len(),
        labels: info
            .labels
            .map(|labels| Labels::single(&labels.values()[rows.clone()])),
        weights: info.weights.map(|w| &w[rows.clone()]),
        group: None,
        bounds: info
            .bounds
            .map(|b| LabelBounds::new(&b.lower()[rows.clone()], &b.upper()[rows])),
    }
}

fn negate(g: GradStats) -> GradStats {
    GradStats::new(-g.grad, -g.hess)
}

/// The global bins of row `row` of `data` under `cuts` (present features).
fn row_bins(cuts: &HistCuts, data: &DMatrix, row: usize) -> Vec<u32> {
    let mut bins = Vec::with_capacity(data.n_cols());
    data.for_row_entry(row, |c, v| bins.push(cuts.bin_of(c as usize, v)));
    bins
}

/// The update of one tree.
struct TreeUpdate<'a> {
    run: &'a Incremental<'a>,
    /// Whether the data (old and new) has no missing value.
    dense: bool,
    reg: &'a RegParams,
    eta: f32,
    old_tree: &'a RegTree,
    /// The deleted rows.
    gone: Option<&'a DMatrix>,
    /// The gradient every row of the new data contributes now.
    cur: &'a [GradPair],
    cuts: &'a HistCuts,
}

impl TreeUpdate<'_> {
    fn value(&self, d: &Delta, feature: usize) -> Option<f32> {
        match (d.deleted, self.gone) {
            (true, Some(m)) => m.get(d.row, feature),
            _ => self.run.new.get(d.row, feature),
        }
    }

    /// The updated tree, its node caches, and the rows that reached a
    /// regrown subtree. `ghist`, the new data's index, is built on the first
    /// regrowth.
    fn update(
        &self,
        mut old_nodes: Vec<NodeCache>,
        deltas: Vec<Delta>,
        ghist: &mut Option<GHistIndex>,
        report: &mut UpdateReport,
    ) -> (RegTree, Vec<NodeCache>, Vec<usize>) {
        let params = self.run.params;
        let new = self.run.new;
        let mut tree = RegTree::with_root(0.0);
        let mut nodes: Vec<NodeCache> = vec![NodeCache::default()];
        let mut regrown_rows = Vec::new();
        // (old node, new node, depth, deltas reaching it)
        let mut queue = std::collections::VecDeque::from([(0usize, 0usize, 0usize, deltas)]);
        while let Some((old_id, new_id, depth, deltas)) = queue.pop_front() {
            let old = *self.old_tree.node(old_id);
            let mut cache = std::mem::take(&mut old_nodes[old_id]);
            // A split no row of the cached data reached has no histogram
            // yet (a model resumed on other data, or a regrown node the
            // rows route around): its sums are all zero.
            if !old.is_leaf() && !deltas.is_empty() && cache.hist.is_empty() {
                cache.hist = vec![GradStats::default(); self.cuts.total_bins()];
            }
            for d in &deltas {
                cache.stats.add(d.g);
                if !old.is_leaf() {
                    for &bin in &d.bins {
                        cache.hist[bin as usize].add(d.g);
                    }
                }
            }
            let touched = !deltas.is_empty();
            if old.is_leaf() {
                let value = if touched {
                    (calc_weight(cache.stats, self.reg) as f32) * self.eta
                } else {
                    old.leaf_value
                };
                tree.set_leaf_value(new_id, value);
                tree.set_sum_hess(new_id, cache.stats.hess as f32);
                nodes[new_id] = cache;
                report.nodes_kept += 1;
                continue;
            }
            // An untouched node's histogram, and so its ranking, is as
            // before: its split stays.
            let (keep, gain) = if touched {
                let rank = rank_split(
                    self.reg,
                    self.cuts,
                    &cache.hist,
                    cache.stats,
                    self.dense,
                    (old.split_feature, old.split_cond, old.default_left),
                );
                let allowed = ((self.run.tolerance * rank.candidates as f64) as usize).max(1);
                let keep = rank.better.is_some_and(|b| b < allowed)
                    && f64::from(rank.loss_chg) >= params.gamma;
                (keep, rank.loss_chg)
            } else {
                (true, old.split_gain)
            };
            if keep {
                let (l, r) = tree.expand(
                    new_id,
                    SplitRule::numeric(old.split_feature, old.split_cond, old.default_left),
                    ChildLeaf::new(0.0, 0.0),
                    ChildLeaf::new(0.0, 0.0),
                );
                tree.set_sum_hess(new_id, cache.stats.hess as f32);
                tree.set_split_gain(new_id, gain);
                nodes[new_id] = cache;
                nodes.resize(tree.num_nodes(), NodeCache::default());
                let (mut left, mut right) = (Vec::new(), Vec::new());
                for d in deltas {
                    let value = self.value(&d, old.split_feature as usize);
                    if self.old_tree.child(old_id, value) == old.left as usize {
                        left.push(d);
                    } else {
                        right.push(d);
                    }
                }
                queue.push_back((old.left as usize, l, depth + 1, left));
                queue.push_back((old.right as usize, r, depth + 1, right));
                report.nodes_kept += 1;
                continue;
            }
            // Regrow: every row of the new data now reaching this node, with
            // the builder on the new data's index.
            let index = match ghist {
                Some(index) => &*index,
                None => ghist.insert(GHistIndex::from_dmatrix(new, self.cuts.clone())),
            };
            let rows: Vec<u32> = (0..new.n_rows())
                .filter(|&i| tree.leaf_id_with(|f| new.get(i, f as usize)) == new_id)
                .map(|i| i as u32)
                .collect();
            // A split node lies above `max_depth` (`check_supported` requires
            // one, `from_model` refuses deeper trees), so its subtree keeps at
            // least one level.
            let sub_params = TrainingParams {
                max_depth: params
                    .max_depth
                    .and_then(|limit| NonZeroUsize::new(limit.get().saturating_sub(depth))),
                ..params.clone()
            };
            let mut sampler = ColumnSampler::new(new.n_cols(), None, 1.0, 1.0, 1.0, params.seed);
            let sub = HistTreeBuilder::new(&sub_params).build(index, self.cur, &rows, &mut sampler);
            graft(&mut tree, new_id, &sub, 0, self.eta);
            nodes.resize(tree.num_nodes(), NodeCache::default());
            clear_subtree(&tree, new_id, &mut nodes);
            accumulate(
                &tree,
                new_id,
                new,
                index,
                rows.iter().map(|&i| (i as usize, self.cur[i as usize])),
                &mut nodes,
            );
            regrown_rows.extend(rows.iter().map(|&i| i as usize));
            report.subtrees_regrown += 1;
        }
        (tree, nodes, regrown_rows)
    }
}

/// Copy `src`'s subtree at `src_id` into `dst` at leaf `dst_id`, scaling
/// leaf weights by `eta` (the builder leaves them unshrunk).
fn graft(dst: &mut RegTree, dst_id: usize, src: &RegTree, src_id: usize, eta: f32) {
    let node = *src.node(src_id);
    dst.set_sum_hess(dst_id, node.sum_hess);
    if node.is_leaf() {
        dst.set_leaf_value(dst_id, node.leaf_value * eta);
        return;
    }
    let (l, r) = dst.expand(
        dst_id,
        SplitRule::numeric(node.split_feature, node.split_cond, node.default_left),
        ChildLeaf::new(0.0, 0.0),
        ChildLeaf::new(0.0, 0.0),
    );
    dst.set_split_gain(dst_id, node.split_gain);
    graft(dst, l, src, node.left as usize, eta);
    graft(dst, r, src, node.right as usize, eta);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::{Objective, RegLoss};
    use crate::rng::Rng;

    /// The vector kernels (logistic on NEON and AVX2, Poisson on NEON) and
    /// the parallel chunking give every fresh row its whole-batch bits, for
    /// batches below, at, and past the vector and chunk thresholds, with
    /// scattered and contiguous fresh rows and inputs that force the scalar
    /// fallback of a vector block.
    #[test]
    fn fresh_rows_get_their_whole_batch_gradients() {
        let mut rng = Rng::new(7);
        for objective in [
            Objective::BinaryLogistic(RegLoss::default()),
            Objective::Poisson,
        ] {
            let loss = TrainingParams {
                objective,
                ..TrainingParams::default()
            }
            .loss(1)
            .unwrap();
            for n in [1, 15, 16, 17, 31, 33, 100, 16_390, 40_007] {
                let margins: Vec<f32> = (0..n)
                    .map(|_| match rng.below(50) {
                        0 => 95.0,
                        1 => -120.0,
                        _ => (rng.f32() - 0.5) * 12.0,
                    })
                    .collect();
                let labels: Vec<f32> = (0..n).map(|_| f32::from(rng.below(2) == 1)).collect();
                let info = MetaInfo::new(&labels, None, None);
                let mut whole = vec![GradPair::default(); n];
                loss.gradient_info(&margins, &info, &mut whole);
                for share in [1, 20, 200] {
                    let mut fresh: Vec<bool> = (0..n).map(|_| rng.below(share) == 0).collect();
                    // Rows added at the end are fresh together.
                    fresh[n - n.min(40)..].fill(true);
                    let mut out = vec![GradPair::default(); n];
                    fresh_gradients(loss.as_ref(), &info, &margins, &fresh, &mut out);
                    for i in (0..n).filter(|&i| fresh[i]) {
                        assert_eq!(
                            (out[i].grad.to_bits(), out[i].hess.to_bits()),
                            (whole[i].grad.to_bits(), whole[i].hess.to_bits()),
                            "row {i} of {n}"
                        );
                    }
                }
            }
        }
    }
}
