//! The histogram builder's split search: every sampled feature's numeric
//! scans (a wide node's in parallel chunks) and categorical sweeps, merged in
//! feature order with XGBoost's tie rule. Under `min_child_groups` every
//! search sees only the candidates the node's split groups leave legal.

use super::{HistTreeBuilder, NodeCtx, Search};
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::tree::builder::categorical::sweep_categorical;
use crate::tree::builder::groups::{CategoryGroups, LegalCuts, NodeGroups};
use crate::tree::builder::shared::{InteractionState, permits, rayon_available, xgb_node_gain};
use crate::tree::builder::split::{
    NumericInput, NumericScan, SplitScorer, for_each_numeric_split, scan_numeric_pair,
    with_scan_scratch,
};
use crate::tree::builder::{BestSplit, need_replace, xgb_update};
use crate::tree::gain::GradStats;
use crate::tree::reuse::CategoricalPenalty;
use rayon::prelude::*;
use std::borrow::Cow;

/// Split candidates (feature bins) at which a node's numeric scans run in
/// parallel chunks of about [`SCAN_TASK_BINS`] candidates each.
const PARALLEL_SCAN_BINS: usize = 4096;
const SCAN_TASK_BINS: usize = 2048;

impl HistTreeBuilder<'_> {
    /// Find the best split for `node` from its histogram, enumerating each
    /// sampled feature's bins as XGBoost's histogram evaluator does
    /// ([`for_each_numeric_split`]). Candidates are scored and compared with
    /// XGBoost's `f32` arithmetic and tie rule, so near-equal gains resolve
    /// the same way. Monotone bounds are honored through the bounded child
    /// weights. With LightGBM split options enabled the search is delegated
    /// to [`SplitOptions::evaluate`](crate::tree::builder::lightgbm::SplitOptions::evaluate).
    /// `groups` (`min_child_groups`, the node's rows) leaves out the
    /// candidates whose children hold too few split groups.
    pub(super) fn evaluate(
        &self,
        ghist: &GHistIndex,
        hist: &[GradStats],
        feature_subset: &[u32],
        allowed: Option<&InteractionState>,
        node: NodeCtx,
        groups: Option<&NodeGroups>,
    ) -> BestSplit {
        // Every child's groups are the node's: too few, and nothing is legal.
        if groups.is_some_and(|groups| !groups.splittable()) {
            return BestSplit::none();
        }
        let feature_subset = permitted(feature_subset, allowed);
        if groups.is_some() && self.options.is_none() && self.reuse.is_none() {
            let search = Search {
                ghist,
                hist,
                node,
                groups,
                legal: &[],
            };
            if let Some(best) = self.merge_lazily(&search, &feature_subset) {
                return best;
            }
        }
        // Each plain numeric feature's legal cuts, by feature id (empty
        // without a constraint).
        let legal = groups.map_or_else(Vec::new, |groups| {
            groups.numeric_cuts(ghist, &feature_subset)
        });
        let search = Search {
            ghist,
            hist,
            node,
            groups,
            legal: &legal,
        };
        if let Some(options) = &self.options {
            return options.evaluate(
                &search,
                &feature_subset,
                &self.config.cons,
                &self.config.reg,
            );
        }
        // The scans of plain numeric features do not depend on the
        // incumbent, so they are computed up front (a wide search in
        // parallel); they are then merged in feature order. Reuse penalties
        // depend on each candidate's bin, so they scan in the merge.
        let scans = if self.reuse.is_some() {
            Vec::new()
        } else {
            self.numeric_scans(&search, &feature_subset)
        };
        self.merge(&search, &feature_subset, scans)
    }

    /// [`scan_numeric_splits`] of every plain numeric feature of
    /// `feature_subset` under `search`'s legal cuts (by position; `None` for
    /// the others), two at a time so their prefix-sum chains overlap, in
    /// parallel chunks for a wide search.
    fn numeric_scans(&self, search: &Search, feature_subset: &[u32]) -> Vec<Option<NumericScan>> {
        let cuts = search.ghist.cuts();
        let node_scorer = self.node_scorer(search.node);
        let scan_chunk = |chunk: &[u32]| -> Vec<Option<NumericScan>> {
            let input = |f: u32| numeric_input(search, f, self.scorer(node_scorer, f));
            let mut out: Vec<Option<NumericScan>> = chunk.iter().map(|_| None).collect();
            with_scan_scratch(|[sa, sb]| {
                let mut pending = None;
                for (i, &f) in chunk.iter().enumerate() {
                    if !plain_numeric(cuts, f) {
                        continue;
                    }
                    match pending.take() {
                        None => pending = Some(i),
                        Some(j) => {
                            let [x, y] = scan_numeric_pair(&input(chunk[j]), &input(f), [sa, sb]);
                            (out[j], out[i]) = (Some(x), Some(y));
                        }
                    }
                }
                if let Some(j) = pending {
                    out[j] = Some(input(chunk[j]).scan(sa));
                }
            });
            out
        };
        Self::parallel_scans(cuts, feature_subset, scan_chunk)
            .unwrap_or_else(|| scan_chunk(feature_subset))
    }

    /// The node's scorer (no monotone direction yet).
    pub(super) fn node_scorer(&self, node: NodeCtx) -> SplitScorer<'_> {
        SplitScorer {
            reg: &self.config.reg,
            root_gain: xgb_node_gain(node.stats, &self.config.reg, node.bounds),
            bounds: node.bounds,
            dir: 0,
        }
    }

    /// `node_scorer` with feature `f`'s monotone direction.
    pub(super) fn scorer<'s>(&self, node_scorer: SplitScorer<'s>, f: u32) -> SplitScorer<'s> {
        SplitScorer {
            dir: self.config.cons.dir(f as usize),
            ..node_scorer
        }
    }

    /// The best split of `feature_subset` (already restricted to the
    /// permitted features) in feature order with XGBoost's tie rule, from
    /// `scans` (each plain numeric feature's [`scan_numeric_splits`] by
    /// position; a feature without one, or past the end, scans here) and
    /// the node's histogram, which is read for categorical features, reuse
    /// penalties, NaN scans, and features without a scan.
    fn merge(
        &self,
        search: &Search,
        feature_subset: &[u32],
        mut scans: Vec<Option<NumericScan>>,
    ) -> BestSplit {
        let mut best = BestSplit::none();
        for (i, &f) in feature_subset.iter().enumerate() {
            let scanned = scans.get_mut(i).and_then(Option::take);
            self.merge_feature(search, f, scanned, search.legal(f), &mut best);
        }
        best
    }

    /// Merge feature `f`'s candidates into `best` as [`Self::merge`] does in
    /// its turn: its categorical sweeps, or its numeric scan (`scanned`, or
    /// scanned here when `None`) under `legal` (`min_child_groups`'s cuts of
    /// a numeric feature; `None`: every candidate).
    fn merge_feature(
        &self,
        search: &Search,
        f: u32,
        scanned: Option<NumericScan>,
        legal: Option<LegalCuts>,
        best: &mut BestSplit,
    ) {
        let &Search {
            ghist,
            hist,
            node,
            groups,
            ..
        } = search;
        let cuts = ghist.cuts();
        let dense = ghist.is_dense();
        let total = node.stats;
        let (fs, fe) = cuts.feature_bins(f as usize);
        let scorer = self.scorer(self.node_scorer(node), f);

        if cuts.is_categorical(f as usize) {
            let category = groups.map(|node| CategoryGroups {
                node,
                ghist,
                feature: f as usize,
            });
            sweep_categorical(
                best,
                &category_stats(cuts, hist, f),
                total,
                &scorer,
                f,
                self.reuse.as_ref().map(|r| r as &dyn CategoricalPenalty),
                category.as_ref(),
            );
            return;
        }
        if fe <= fs + 1 {
            return; // degenerate feature, no interior boundary
        }
        let allowed = |pos, default_left| {
            legal.is_none_or(|legal| legal.allows_numeric(fs, pos, default_left))
        };

        if let Some(reuse) = &self.reuse {
            for_each_numeric_split(&hist[fs..fe], fs, total, dense, |pos, children| {
                if !allowed(pos, children.default_left) {
                    return;
                }
                let Some(mut score) = scorer.loss_chg(children.left, children.right) else {
                    return;
                };
                score.loss_chg -= reuse.bin_penalty(f, pos.bin());
                xgb_update(best, f, pos, children, score);
            });
            return;
        }
        let scanned = scanned.unwrap_or_else(|| {
            let input = NumericInput {
                legal,
                ..numeric_input(search, f, scorer)
            };
            with_scan_scratch(|[s, _]| input.scan(s))
        });
        match scanned {
            NumericScan::Empty => {}
            NumericScan::Best {
                loss_chg,
                pos,
                children,
            } => {
                if need_replace(best.loss_chg as f32, best.feature, loss_chg, f)
                    && let Some(score) = scorer.loss_chg(children.left, children.right)
                {
                    xgb_update(best, f, pos, children, score);
                }
            }
            NumericScan::Nan => {
                for_each_numeric_split(&hist[fs..fe], fs, total, dense, |pos, children| {
                    if allowed(pos, children.default_left)
                        && let Some(score) = scorer.loss_chg(children.left, children.right)
                    {
                        xgb_update(best, f, pos, children, score);
                    }
                });
            }
        }
    }

    /// [`Self::merge`] under `min_child_groups`, counting a feature's
    /// groups only while its best unconstrained candidate could still win:
    /// its legal ones score no higher. Features are visited by that bound,
    /// highest first (lower features first among equals), and each merges
    /// as in its turn. The tie rule orders every finite candidate by loss
    /// change, then lower feature, the same in any visiting order, so the
    /// split is the sequential merge's. `None` when a numeric scan scores
    /// NaN, whose replay depends on the order the sequential merge keeps.
    fn merge_lazily(&self, search: &Search, feature_subset: &[u32]) -> Option<BestSplit> {
        let &Search {
            ghist,
            hist,
            node,
            groups,
            ..
        } = search;
        let groups = groups?;
        let cuts = ghist.cuts();
        let node_scorer = self.node_scorer(node);
        // `search` holds no legal cuts: these scans are unconstrained.
        let scans = self.numeric_scans(search, feature_subset);
        // Each feature's best unconstrained loss change, where it has one.
        let mut bounds: Vec<(f32, u32)> = Vec::with_capacity(feature_subset.len());
        for (i, &f) in feature_subset.iter().enumerate() {
            let bound = if cuts.is_categorical(f as usize) {
                let mut probe = BestSplit::none();
                let scorer = self.scorer(node_scorer, f);
                let stats = category_stats(cuts, hist, f);
                sweep_categorical(&mut probe, &stats, node.stats, &scorer, f, None, None);
                // Only a positive loss change replaces the empty split.
                (probe.loss_chg > 0.0).then_some(probe.loss_chg as f32)
            } else {
                match scans.get(i) {
                    Some(Some(NumericScan::Best { loss_chg, .. })) => Some(*loss_chg),
                    Some(Some(NumericScan::Nan)) => return None,
                    _ => None,
                }
            };
            bounds.extend(bound.map(|bound| (bound, f)));
        }
        bounds.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut best = BestSplit::none();
        for (bound, f) in bounds {
            let incumbent = best.loss_chg as f32;
            if bound < incumbent {
                break;
            }
            // At an equal bound only a lower feature can take the split.
            if bound == incumbent && f >= best.feature {
                continue;
            }
            let legal =
                (!cuts.is_categorical(f as usize)).then(|| groups.numeric(ghist, f as usize));
            let scanned = match legal {
                Some(legal) => {
                    let input = NumericInput {
                        legal: Some(legal),
                        ..numeric_input(search, f, self.scorer(node_scorer, f))
                    };
                    match with_scan_scratch(|[s, _]| input.scan(s)) {
                        NumericScan::Nan => return None,
                        scan => Some(scan),
                    }
                }
                None => None,
            };
            self.merge_feature(search, f, scanned, legal, &mut best);
        }
        Some(best)
    }

    /// `scan_chunk` over parallel chunks of `feature_subset`, concatenated,
    /// when the subset holds enough candidates to pay for the tasks. `None`
    /// otherwise.
    fn parallel_scans(
        cuts: &HistCuts,
        feature_subset: &[u32],
        scan_chunk: impl Fn(&[u32]) -> Vec<Option<NumericScan>> + Sync,
    ) -> Option<Vec<Option<NumericScan>>> {
        if !rayon_available() {
            return None;
        }
        let bins = |f: u32| {
            let (fs, fe) = cuts.feature_bins(f as usize);
            fe - fs
        };
        let candidates: usize = feature_subset.iter().map(|&f| bins(f)).sum();
        if candidates < PARALLEL_SCAN_BINS {
            return None;
        }
        let per_task = (SCAN_TASK_BINS * feature_subset.len()).div_ceil(candidates);
        Some(
            feature_subset
                .par_chunks(per_task.max(1))
                .flat_map_iter(&scan_chunk)
                .collect(),
        )
    }
}

/// `feature_subset` restricted to the features the interaction state
/// `allowed` permits (a sorted set; `None`: every feature).
pub(super) fn permitted<'a>(
    feature_subset: &'a [u32],
    allowed: Option<&InteractionState>,
) -> Cow<'a, [u32]> {
    match allowed {
        Some(_) => feature_subset
            .iter()
            .copied()
            .filter(|&f| permits(allowed, f))
            .collect(),
        None => Cow::Borrowed(feature_subset),
    }
}

/// Whether feature `f` is numeric with an interior boundary: the features
/// [`scan_numeric_splits`](crate::tree::builder::split::scan_numeric_splits)
/// scans.
pub(super) fn plain_numeric(cuts: &HistCuts, f: u32) -> bool {
    let (fs, fe) = cuts.feature_bins(f as usize);
    !cuts.is_categorical(f as usize) && fe > fs + 1
}

/// Feature `f`'s numeric scan input from `search`'s node, under its legal
/// cuts (`min_child_groups`).
fn numeric_input<'a>(search: &Search<'a>, f: u32, scorer: SplitScorer<'a>) -> NumericInput<'a> {
    let (fs, fe) = search.ghist.cuts().feature_bins(f as usize);
    NumericInput {
        bins: &search.hist[fs..fe],
        first: fs,
        total: search.node.stats,
        dense: search.ghist.is_dense(),
        scorer,
        legal: search.legal(f),
    }
}

/// Every category bin of categorical feature `f` with its node statistics,
/// empty ones included, as XGBoost enumerates them (a lone category can
/// still split present from missing values).
fn category_stats(cuts: &HistCuts, hist: &[GradStats], f: u32) -> Vec<(u32, GradStats)> {
    let (fs, fe) = cuts.feature_bins(f as usize);
    (fs..fe)
        .map(|i| (cuts.cut_value(i) as u32, hist[i]))
        .collect()
}
