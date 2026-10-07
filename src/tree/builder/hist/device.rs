//! Device-resident growth: a [`RowEngine`] keeps the tree's rows on its
//! device, so each depthwise level (or loss-guided expansion) partitions
//! every splitting node and builds every needed child histogram in one
//! batch there. The host keeps everything that decides the tree — node
//! order, the column sampler, bounds, interaction state, and the merge of
//! the split search — so the tree is the host builder's, bit for bit.
//!
//! Under resident growth the numeric and categorical histograms never
//! leave the device. A bounded, recycled slot pool holds the frontier (or
//! the loss-guided host heap), siblings are subtracted from their original
//! parent histograms, and the device returns only each node's winning split.
//! Feature-order merging and stable category ordering retain the CPU's bits;
//! non-total NaN comparisons explicitly request a host histogram replay.

use super::search::{permitted, plain_numeric};
use super::{Child, HistTreeBuilder, NodeCtx, NodeEntry, NodeStore, PendingSplit};
use crate::config::GrowPolicy;
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::objective::GradPair;
use crate::tree::builder::partition::{category_left, with_sibling};
use crate::tree::builder::shared::{InteractionState, LeafRows, rayon_available, sum_rows};
use crate::tree::builder::split::SplitScorer;
use crate::tree::builder::{BestSplit, Children, SplitLocation, SplitPos, limit_or_unbounded};
use crate::tree::gain::GradStats;
use crate::tree::hist::{
    HistSlot, NodeScan, Partitioned, RowEngine, RowRule, RowSplit, ScanRequest, Segment,
};
use crate::tree::regtree::RegTree;
use crate::tree::sampler::ColumnSampler;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::BinaryHeap;

/// Combined rows of a batch of splits at which their children's split
/// searches run in parallel.
const PARALLEL_FINISH_ROWS: usize = 4096;

/// What every device-growth step reads: the engine holding the rows, the
/// binned index, and the host copy of the tree's gradients (`None` when
/// only the device has them).
#[derive(Clone, Copy)]
struct Device<'a> {
    engine: &'a dyn RowEngine,
    ghist: &'a GHistIndex,
    gpair: Option<&'a [GradPair]>,
}

/// What a device-grown tree reports about its leaves' rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LeafReport {
    /// Nothing (the rows are not needed).
    Nothing,
    /// Each leaf's rows, read back to the host.
    Rows,
    /// Each leaf's device segment (the rows stay on the device).
    Segments,
}

/// A device-grown tree, its leaves' rows (when read back), and its leaves'
/// device segments (when kept on the device), each with its node id.
type DeviceTree = (RegTree, Vec<LeafRows>, Vec<(usize, Segment)>);

/// A bounded histogram pool. A split reuses its parent's slot for the
/// subtracted sibling and borrows one for its built child; final leaves
/// return their slots without rebuilding any rounded histogram.
struct Slots {
    next: HistSlot,
    reserved: usize,
    free: Vec<HistSlot>,
}

impl Slots {
    fn take(&mut self) -> Option<HistSlot> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        let slot = self.next;
        ((slot as usize) < self.reserved).then(|| {
            self.next += 1;
            slot
        })
    }

    fn release(&mut self, slot: Option<HistSlot>) {
        if let Some(slot) = slot {
            self.free.push(slot);
        }
    }
}

/// One node's resident split search: its histogram's slot, its sampled
/// features, interaction state, and context.
struct ResidentNode<'n> {
    slot: HistSlot,
    features: &'n [u32],
    allowed: Option<&'n InteractionState>,
    ctx: NodeCtx,
}

impl HistTreeBuilder<'_> {
    /// The backend's row engine, when this tree can grow on it: depthwise
    /// or loss-guided growth on `f64` histograms, without reuse penalties
    /// (whose commits a device failure's regrowth would repeat).
    pub(super) fn device_engine(&self) -> Option<&dyn RowEngine> {
        let params = self.config.params;
        let supported = params.quantized.is_none()
            && self.reuse.is_none()
            && matches!(
                params.grow_policy,
                GrowPolicy::DepthWise | GrowPolicy::LossGuide
            );
        supported.then(|| self.backend.row_engine()).flatten()
    }

    /// At most one slot per live leaf, bounded by row count and the growth
    /// limits. Loss-guided scheduling stays in the existing host heap;
    /// only its frontier histograms and split searches remain on device.
    fn resident_slots(&self, rows: usize) -> Option<usize> {
        if self.options.is_some() {
            return None;
        }
        let params = self.config.params;
        let depth = params.max_depth.map(|depth| {
            let exponent = depth.get() - usize::from(params.grow_policy == GrowPolicy::DepthWise);
            u32::try_from(exponent)
                .ok()
                .and_then(|exponent| 1usize.checked_shl(exponent))
                .unwrap_or(usize::MAX)
        });
        let leaves = if params.grow_policy == GrowPolicy::LossGuide {
            limit_or_unbounded(params.max_leaves)
        } else {
            usize::MAX
        };
        Some(depth.unwrap_or(usize::MAX).min(leaves).min(rows.max(1)))
    }

    /// Grow the tree on `engine`; `None` after a device failure (the
    /// caller regrows it on the host from the sampler's starting state, or
    /// redoes the round when `gpair` is `None`: the gradients then exist
    /// only on the device).
    pub(super) fn build_on_device(
        &self,
        engine: &dyn RowEngine,
        ghist: &GHistIndex,
        gpair: Option<&[GradPair]>,
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
        report: LeafReport,
    ) -> Option<DeviceTree> {
        let seg = engine.begin_tree(ghist, row_subset)?;
        // The root's statistics, the host's `sum_rows` bit for bit (on the
        // host only for non-finite gradients).
        let root_stats = match (engine.root_total(seg), gpair) {
            (Some(stats), _) => stats,
            (None, Some(gpair)) => sum_rows(gpair, row_subset),
            (None, None) => return None,
        };
        let on = Device {
            engine,
            ghist,
            gpair,
        };
        let mut slots = self
            .resident_slots(row_subset.len())
            .filter(|&reserved| engine.reserve_hists(ghist, reserved) == Some(true))
            .map(|reserved| Slots {
                next: 0,
                reserved,
                free: Vec::new(),
            });
        let root = if let Some(slots) = &mut slots {
            let slot = slots.take()?;
            engine.build_resident(ghist, gpair, &[(seg, slot)], &[])?;
            let (features, ctx) = self.root_context(sampler, root_stats, row_subset.len());
            let node = ResidentNode {
                slot,
                features: &features,
                allowed: None,
                ctx,
            };
            let best = self.resident_search(on, &[node])?.pop()?;
            NodeEntry {
                seg: Some(seg),
                slot: Some(slot),
                ..Self::root_node(ctx.tree_seed, best)
            }
        } else {
            let root_hist = engine
                .histograms(ghist, gpair, &[seg])?
                .into_iter()
                .next()?;
            NodeEntry {
                seg: Some(seg),
                ..self.root_entry(ghist, sampler, root_stats, root_hist, row_subset.len())
            }
        };
        let mut tree = RegTree::with_root(root_stats.hess as f32);
        let mut store = NodeStore::new(root_stats, report != LeafReport::Nothing);
        if self.config.params.grow_policy == GrowPolicy::LossGuide {
            self.grow_device_lossguide(on, &mut tree, &mut store, sampler, root, slots.as_mut())?;
        } else {
            self.grow_device_depthwise(on, &mut tree, &mut store, sampler, root, slots.as_mut())?;
        }
        self.finish_tree(&mut tree, &store, root_stats);
        let leaf_rows = if report == LeafReport::Rows {
            let segs: Vec<_> = store.device_leaves.iter().map(|&(_, seg)| seg).collect();
            let rows = engine.rows(&segs)?;
            store
                .device_leaves
                .iter()
                .zip(rows)
                .map(|(&(node, _), rows)| LeafRows { node, rows })
                .collect()
        } else {
            Vec::new()
        };
        let segments = if report == LeafReport::Segments {
            store.device_leaves
        } else {
            Vec::new()
        };
        Some((tree, leaf_rows, segments))
    }

    /// Grow a tree on the device whose gradients only the device holds
    /// ([`RowEngine::gradients`]): the tree (leaves not yet scaled by
    /// the learning rate) and each leaf's device segment. `None` when the
    /// backend has no row engine for this configuration, or after a device
    /// failure; the trainer then redoes the round on the host.
    pub(crate) fn build_device_staged(
        &self,
        ghist: &GHistIndex,
        rows: &[u32],
        sampler: &mut ColumnSampler,
    ) -> Option<(RegTree, Vec<(usize, Segment)>)> {
        let engine = self.device_engine()?;
        self.build_on_device(engine, ghist, None, rows, sampler, LeafReport::Segments)
            .map(|(tree, _, segments)| (tree, segments))
    }

    fn grow_device_depthwise(
        &self,
        on: Device<'_>,
        tree: &mut RegTree,
        store: &mut NodeStore,
        sampler: &mut ColumnSampler,
        root: NodeEntry,
        mut slots: Option<&mut Slots>,
    ) -> Option<()> {
        let ghist = on.ghist;
        let limit = limit_or_unbounded(self.config.params.max_depth);
        let mut frontier = vec![root];
        let mut depth = 0;
        while depth < limit && !frontier.is_empty() {
            let mut pending = Vec::with_capacity(frontier.len());
            for entry in frontier.drain(..) {
                let slot = entry.slot;
                if self.valid(&entry.best) {
                    if let Some(split) =
                        self.prepare_split(tree, store, ghist.cuts(), sampler, entry)
                    {
                        pending.push(split);
                    } else if let Some(slots) = slots.as_deref_mut() {
                        slots.release(slot);
                    }
                } else {
                    if let Some(slots) = slots.as_deref_mut() {
                        slots.release(slot);
                    }
                    store.record_leaf(entry);
                }
            }
            let children = match slots.as_deref_mut() {
                Some(slots) => self.resident_children(on, pending, slots)?,
                None => self.device_children(on, pending)?,
            };
            frontier = children
                .into_iter()
                .flat_map(|(left, right)| [left, right])
                .collect();
            depth += 1;
        }
        for entry in frontier {
            store.record_leaf(entry);
        }
        Some(())
    }

    fn grow_device_lossguide(
        &self,
        on: Device<'_>,
        tree: &mut RegTree,
        store: &mut NodeStore,
        sampler: &mut ColumnSampler,
        root: NodeEntry,
        mut slots: Option<&mut Slots>,
    ) -> Option<()> {
        let ghist = on.ghist;
        let limit = limit_or_unbounded(self.config.params.max_depth);
        let max_leaves = limit_or_unbounded(self.config.params.max_leaves);
        let mut heap = BinaryHeap::new();
        heap.push(root);
        let mut n_leaves = 1usize;
        while let Some(entry) = heap.pop() {
            let slot = entry.slot;
            if n_leaves >= max_leaves {
                store.record_leaf(entry);
                break;
            }
            if entry.depth >= limit || !self.valid(&entry.best) {
                if let Some(slots) = slots.as_deref_mut() {
                    slots.release(slot);
                }
                store.record_leaf(entry);
                continue;
            }
            let split = self.prepare_split(tree, store, ghist.cuts(), sampler, entry);
            n_leaves += 1;
            if let Some(split) = split {
                let mut children = match slots.as_deref_mut() {
                    Some(slots) => self.resident_children(on, vec![split], slots)?,
                    None => self.device_children(on, vec![split])?,
                };
                let (left, right) = children.pop()?;
                heap.push(left);
                heap.push(right);
            } else if let Some(slots) = slots.as_deref_mut() {
                slots.release(slot);
            }
        }
        for entry in heap {
            store.record_leaf(entry);
        }
        Some(())
    }

    /// One device partition of every split of `pending`, in order.
    fn partition_pending(on: Device<'_>, pending: &[PendingSplit]) -> Option<Vec<Partitioned>> {
        let cuts = on.ghist.cuts();
        let tables: Vec<Option<Vec<bool>>> = pending
            .iter()
            .map(|split| match &split.entry.best.location {
                SplitLocation::Categories(categories) => {
                    Some(category_left(cuts, split.entry.best.feature, categories))
                }
                SplitLocation::Numeric(_) => None,
            })
            .collect();
        let splits: Option<Vec<RowSplit<'_>>> = pending
            .iter()
            .zip(&tables)
            .map(|(split, table)| {
                let best = &split.entry.best;
                let (fs, _) = cuts.feature_bins(best.feature as usize);
                let rule = match (&best.location, table) {
                    (_, Some(table)) => RowRule::Table(table),
                    (SplitLocation::Numeric(pos), None) => {
                        // Present bins up to the split bin go left.
                        let limit = pos.bin().map_or(0, |s| (s + 1).saturating_sub(fs));
                        RowRule::Below(u32::try_from(limit).ok()?)
                    }
                    (SplitLocation::Categories(_), None) => return None,
                };
                Some(RowSplit {
                    seg: split.entry.seg?,
                    feature: best.feature,
                    rule,
                    default_left: best.default_left,
                })
            })
            .collect();
        on.engine.partition(on.ghist, &splits?)
    }

    /// The children of every split of `pending`: one device partition of
    /// all of them, one batch of the smaller children's histograms (the
    /// siblings by subtraction here), then their split searches.
    fn device_children(
        &self,
        on: Device<'_>,
        mut pending: Vec<PendingSplit>,
    ) -> Option<Vec<(NodeEntry, NodeEntry)>> {
        let Device {
            engine,
            ghist,
            gpair,
        } = on;
        if pending.is_empty() {
            return Some(Vec::new());
        }
        let parts = Self::partition_pending(on, &pending)?;

        // The smaller child of every non-terminal split is built; its
        // sibling is the parent minus it, as the host builder does.
        let built: Vec<Segment> = pending
            .iter()
            .zip(&parts)
            .filter(|(split, _)| !split.terminal)
            .map(|(_, part)| smaller(part))
            .collect();
        let mut hists = engine.histograms(ghist, gpair, &built)?.into_iter();
        let mut items = Vec::with_capacity(pending.len());
        for (split, part) in pending.iter_mut().zip(&parts) {
            let parent = std::mem::take(&mut split.entry.hist);
            let (left_hist, right_hist) = if split.terminal {
                (Vec::new(), Vec::new())
            } else {
                let left_smaller = part.left.len <= part.right.len;
                with_sibling(parent, hists.next()?, left_smaller)
            };
            items.push((part, left_hist, right_hist));
        }
        let finish = |(split, (part, left_hist, right_hist)): (PendingSplit, (&_, _, _))| {
            let part: &Partitioned = part;
            self.finish_children(
                ghist,
                split,
                device_child(part.left, left_hist),
                device_child(part.right, right_hist),
                None,
            )
        };
        let rows: usize = parts.iter().map(|p| p.left.len + p.right.len).sum();
        let pairs = pending.into_iter().zip(items);
        Some(
            if pairs.len() > 1 && rows >= PARALLEL_FINISH_ROWS && rayon_available() {
                pairs
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(finish)
                    .collect()
            } else {
                pairs.map(finish).collect()
            },
        )
    }

    /// [`Self::device_children`] under resident growth: the smaller child
    /// of each non-terminal split is built into a new slot, its sibling
    /// subtracted in the parent's, and every child searched on the device.
    fn resident_children(
        &self,
        on: Device<'_>,
        pending: Vec<PendingSplit>,
        slots: &mut Slots,
    ) -> Option<Vec<(NodeEntry, NodeEntry)>> {
        if pending.is_empty() {
            return Some(Vec::new());
        }
        let parts = Self::partition_pending(on, &pending)?;
        // Release all terminal parents before borrowing any child slot:
        // terminal splits can appear after non-terminal ones in node order.
        for split in &pending {
            if split.terminal {
                slots.release(split.entry.slot);
            }
        }
        let mut build = Vec::new();
        let mut siblings = Vec::new();
        // Each split's `(left, right)` child slots; `None` when terminal.
        let mut child_slots = Vec::with_capacity(pending.len());
        for (split, part) in pending.iter().zip(&parts) {
            let parent = split.entry.slot?;
            if split.terminal {
                child_slots.push(None);
                continue;
            }
            let built = slots.take()?;
            build.push((smaller(part), built));
            siblings.push((parent, built));
            child_slots.push(Some(if part.left.len <= part.right.len {
                (built, parent)
            } else {
                (parent, built)
            }));
        }
        on.engine
            .build_resident(on.ghist, on.gpair, &build, &siblings)?;
        let contexts: Vec<_> = pending
            .iter()
            .zip(&parts)
            .map(|(split, part)| self.child_contexts(split, part.left.len, part.right.len))
            .collect();
        let mut nodes = Vec::new();
        for ((split, (allowed, left_ctx, right_ctx)), slots) in
            pending.iter().zip(&contexts).zip(&child_slots)
        {
            if let Some((left, right)) = *slots {
                nodes.push(ResidentNode {
                    slot: left,
                    features: &split.left_features,
                    allowed: allowed.as_ref(),
                    ctx: *left_ctx,
                });
                nodes.push(ResidentNode {
                    slot: right,
                    features: &split.right_features,
                    allowed: allowed.as_ref(),
                    ctx: *right_ctx,
                });
            }
        }
        let mut bests = self.resident_search(on, &nodes)?.into_iter();
        let mut out = Vec::with_capacity(pending.len());
        for ((split, part), slots) in pending.into_iter().zip(&parts).zip(child_slots) {
            let searched = match slots {
                Some(_) => Some((bests.next()?, bests.next()?)),
                None => None,
            };
            let (left, right) = self.finish_children(
                on.ghist,
                split,
                device_child(part.left, Vec::new()),
                device_child(part.right, Vec::new()),
                searched,
            );
            out.push(match slots {
                Some((left_slot, right_slot)) => (
                    NodeEntry {
                        slot: Some(left_slot),
                        ..left
                    },
                    NodeEntry {
                        slot: Some(right_slot),
                        ..right
                    },
                ),
                None => (left, right),
            });
        }
        Some(out)
    }

    /// One winner per resident node; only non-total comparisons require
    /// its histogram. Child weights are recomputed by the existing host
    /// scorer, without repeating any bin scan or feature merge.
    fn resident_search(
        &self,
        on: Device<'_>,
        nodes: &[ResidentNode<'_>],
    ) -> Option<Vec<BestSplit>> {
        let cuts = on.ghist.cuts();
        let features: Vec<Cow<'_, [u32]>> = nodes
            .iter()
            .map(|node| permitted(node.features, node.allowed))
            .collect();
        let scanned: Vec<Vec<(u32, i8)>> = features
            .iter()
            .map(|features| {
                features
                    .iter()
                    .copied()
                    .filter(|&f| cuts.is_categorical(f as usize) || plain_numeric(cuts, f))
                    .map(|f| (f, self.config.cons.dir(f as usize)))
                    .collect()
            })
            .collect();
        let requests: Vec<ScanRequest<'_>> = nodes
            .iter()
            .zip(&scanned)
            .map(|(node, features)| ScanRequest {
                slot: node.slot,
                total: node.ctx.stats,
                root_gain: self.node_scorer(node.ctx).root_gain,
                lower: node.ctx.bounds.lower as f32,
                upper: node.ctx.bounds.upper as f32,
                features,
            })
            .collect();
        let results = on
            .engine
            .scan_resident(on.ghist, &self.config.reg, &requests)?;
        if results.len() != nodes.len() {
            return None;
        }
        nodes
            .iter()
            .zip(results)
            .map(|(node, scan)| self.resident_merge(on, node, scan))
            .collect()
    }

    /// Decode a device winner, or replay this node's search with its
    /// histogram when a NaN prevents an order-preserving reduction.
    fn resident_merge(
        &self,
        on: Device<'_>,
        node: &ResidentNode<'_>,
        scan: NodeScan,
    ) -> Option<BestSplit> {
        match scan {
            NodeScan::Replay(_) => {
                let hist = on.engine.read_hist(node.slot)?;
                Some(self.evaluate(on.ghist, &hist, node.features, node.allowed, node.ctx))
            }
            scan => resident_best(scan, on.ghist.cuts(), node.ctx.stats, |feature| {
                self.scorer(self.node_scorer(node.ctx), feature)
            }),
        }
    }
}

/// The child of a partition with fewer rows (the left one on a tie), whose
/// histogram is built.
fn smaller(part: &Partitioned) -> Segment {
    if part.left.len <= part.right.len {
        part.left
    } else {
        part.right
    }
}

/// A child whose rows are the device segment `seg`.
fn device_child(seg: Segment, hist: Vec<GradStats>) -> Child {
    Child {
        len: seg.len,
        rows: Vec::new(),
        seg: Some(seg),
        hist,
        quant: None,
    }
}

/// Convert the winning device split with the same host scorer and
/// categorical child swap used by the CPU search.
fn resident_best<'a>(
    scan: NodeScan,
    cuts: &HistCuts,
    total: GradStats,
    scorer: impl Fn(u32) -> SplitScorer<'a>,
) -> Option<BestSplit> {
    match scan {
        NodeScan::Empty => Some(BestSplit::none()),
        NodeScan::Replay(_) => None,
        NodeScan::Numeric {
            feature,
            loss_chg,
            backward,
            offset,
            acc,
        } => {
            let (first, _) = cuts.feature_bins(feature as usize);
            let offset = offset as usize;
            let (pos, children) = if backward {
                (
                    SplitPos::backward(first, offset),
                    Children::new(true, total.sub(acc), acc),
                )
            } else {
                (
                    SplitPos::Bin(first + offset),
                    Children::new(false, acc, total.sub(acc)),
                )
            };
            let score = scorer(feature).loss_chg(children.left, children.right)?;
            let mut best = BestSplit::numeric(feature, pos, children, score.into());
            best.loss_chg = f64::from(loss_chg);
            Some(best)
        }
        NodeScan::Categorical {
            feature,
            loss_chg,
            default_left,
            left,
            right,
            categories,
        } => {
            let score = scorer(feature).loss_chg(right, left)?.swapped();
            let mut best = BestSplit::categorical(
                feature,
                Children::new(default_left, left, right),
                score.into(),
                categories,
            );
            best.loss_chg = f64::from(loss_chg);
            best.children_swapped = true;
            Some(best)
        }
    }
}

#[cfg(all(test, target_os = "linux", feature = "cuda"))]
mod tests {
    use super::*;
    use crate::backend::cuda::{CudaHistBackend, available, unavailable_reason};
    use crate::config::{Monotone, TrainingParams};
    use crate::data::{DMatrix, FeatureType};
    use crate::tree::hist::{CpuBackend, HistogramBackend, ScanFallback, zeroed};

    fn has_device() -> bool {
        if let Some(reason) = unavailable_reason() {
            assert!(
                std::env::var_os("HESSBOOST_REQUIRE_CUDA").is_none(),
                "CUDA required: {reason}"
            );
            return false;
        }
        true
    }

    /// Repeated gradient groups give exactly equal f32 category weights.
    /// Duplicate categorical columns also exercise the feature-index tie
    /// rule against a numeric column and a reversed feature traversal.
    fn fixture(categories: usize) -> (GHistIndex, Vec<GradPair>) {
        let n = categories * 4 + 32;
        let mut values = Vec::with_capacity(n * 3);
        let mut gradients = Vec::with_capacity(n);
        for row in 0..n {
            let category = row % categories;
            let missing = row >= categories * 4;
            let value = if missing { f32::NAN } else { category as f32 };
            values.extend([value, (row % 13) as f32, value]);
            let grad = if missing {
                -5.0
            } else {
                (category % 7) as f32 - 3.0
            };
            gradients.push(GradPair::new(grad, 1.0));
        }
        let data = DMatrix::from_dense_with_missing(&values, n, 3, f32::NAN)
            .unwrap()
            .with_feature_types(&[
                FeatureType::Categorical,
                FeatureType::Numerical,
                FeatureType::Categorical,
            ])
            .unwrap();
        (
            GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256)),
            gradients,
        )
    }

    fn assert_split_bits(actual: &BestSplit, expected: &BestSplit) {
        assert_eq!(actual.loss_chg.to_bits(), expected.loss_chg.to_bits());
        assert_eq!(actual.feature, expected.feature);
        assert_eq!(actual.default_left, expected.default_left);
        assert_eq!(actual.children_swapped, expected.children_swapped);
        assert_eq!(actual.left.grad.to_bits(), expected.left.grad.to_bits());
        assert_eq!(actual.left.hess.to_bits(), expected.left.hess.to_bits());
        assert_eq!(actual.right.grad.to_bits(), expected.right.grad.to_bits());
        assert_eq!(actual.right.hess.to_bits(), expected.right.hess.to_bits());
        assert_eq!(actual.w_left.to_bits(), expected.w_left.to_bits());
        assert_eq!(actual.w_right.to_bits(), expected.w_right.to_bits());
        match (&actual.location, &expected.location) {
            (SplitLocation::Categories(a), SplitLocation::Categories(b)) => assert_eq!(a, b),
            (SplitLocation::Numeric(a), SplitLocation::Numeric(b)) => match (a, b) {
                (SplitPos::Bin(a), SplitPos::Bin(b)) => assert_eq!(a, b),
                (SplitPos::BelowBins, SplitPos::BelowBins) => {}
                (SplitPos::Value(a), SplitPos::Value(b)) => assert_eq!(a.to_bits(), b.to_bits()),
                _ => panic!("device numeric boundary has a different kind"),
            },
            _ => panic!("device split has a different kind"),
        }
    }

    #[test]
    fn compact_device_category_winners_match_cpu_bits_without_histogram_readback() {
        if !has_device() {
            return;
        }
        for categories in [1, 3, 4, 64, 257, 4097] {
            let (index, gradients) = fixture(categories);
            let backend = CudaHistBackend::new(&index, 0).unwrap();
            let rows: Vec<u32> = (0..index.n_rows() as u32).collect();
            backend.prepare(&index, &gradients);
            let seg = backend.begin_tree(&index, &rows).unwrap();
            assert_eq!(backend.reserve_hists(&index, 1), Some(true));
            backend
                .build_resident(&index, Some(&gradients), &[(seg, 0)], &[])
                .unwrap();
            let total = backend.root_total(seg).unwrap();
            let mut hist = zeroed(index.total_bins());
            CpuBackend.build(&index, &rows, &gradients, &mut hist);
            for direction in [Monotone::None, Monotone::Increasing, Monotone::Decreasing] {
                let params = TrainingParams::builder()
                    .min_child_weight(0.0)
                    .alpha(0.1)
                    .lambda(0.75)
                    .monotone_constraints(vec![direction, Monotone::None, direction])
                    .build()
                    .unwrap();
                let builder = HistTreeBuilder::new(&params);
                let (_, ctx) = builder.root_context(&mut ColumnSampler::all(3), total, rows.len());
                for features in [vec![0, 1, 2], vec![2, 1, 0], vec![0], vec![1]] {
                    let scanned: Vec<_> = features
                        .iter()
                        .map(|&f| (f, builder.config.cons.dir(f as usize)))
                        .collect();
                    let request = ScanRequest {
                        slot: 0,
                        total,
                        root_gain: builder.node_scorer(ctx).root_gain,
                        lower: ctx.bounds.lower as f32,
                        upper: ctx.bounds.upper as f32,
                        features: &scanned,
                    };
                    let before = backend.scan_diagnostics().winner_readback_bytes;
                    let scan = backend
                        .scan_resident(&index, &builder.config.reg, &[request])
                        .unwrap()
                        .pop()
                        .unwrap();
                    if features == [1] {
                        assert_eq!(
                            backend.scan_diagnostics().winner_readback_bytes - before,
                            40
                        );
                    }
                    assert!(
                        !matches!(scan, NodeScan::Replay(_)),
                        "{categories} categories"
                    );
                    let actual = resident_best(scan, index.cuts(), total, |f| {
                        builder.scorer(builder.node_scorer(ctx), f)
                    })
                    .unwrap();
                    let expected = builder.evaluate(&index, &hist, &features, None, ctx);
                    assert_split_bits(&actual, &expected);
                }
            }
            let counts = backend.scan_diagnostics();
            assert_eq!(counts.device_nodes, 12);
            assert!(counts.categorical_features > 0);
            assert!(counts.numeric_features > 0);
            assert!(counts.winner_readback_bytes <= 12 * 312);
            assert_eq!(counts.numeric_score_replays, 0);
            assert_eq!(counts.categorical_order_replays, 0);
            assert_eq!(counts.categorical_score_replays, 0);
            assert!(available(), "{:?}", unavailable_reason());
        }
    }

    #[test]
    fn categorical_and_numeric_lossguide_keep_resident_histograms_and_host_heap_order() {
        if !has_device() {
            return;
        }
        for categories in [1, 3, 4, 64, 257] {
            let (index, gradients) = fixture(categories);
            for policy in [GrowPolicy::DepthWise, GrowPolicy::LossGuide] {
                let params = TrainingParams::builder()
                    .grow_policy(policy)
                    .max_depth(4)
                    .max_leaves(11)
                    .min_child_weight(0.0)
                    .alpha(0.125)
                    .lambda(0.75)
                    .build()
                    .unwrap();
                let rows: Vec<u32> = (0..index.n_rows() as u32).collect();
                let mut sampler = ColumnSampler::new(3, None, 1.0, 1.0, 1.0, 71);
                let mut expected_sampler = sampler.clone();
                let expected = HistTreeBuilder::new(&params).build(
                    &index,
                    &gradients,
                    &rows,
                    &mut expected_sampler,
                );
                let backend = CudaHistBackend::new(&index, 0).unwrap();
                let actual = HistTreeBuilder::new(&params).with_backend(&backend).build(
                    &index,
                    &gradients,
                    &rows,
                    &mut sampler,
                );
                assert_eq!(
                    serde_json::to_vec(&actual).unwrap(),
                    serde_json::to_vec(&expected).unwrap(),
                    "{categories} {policy:?}"
                );
                assert_eq!(sampler.sample(2), expected_sampler.sample(2));
                let counts = backend.scan_diagnostics();
                assert!(counts.device_nodes > 1, "must scan the children on device");
                assert!(counts.categorical_features > 0);
                assert!(counts.numeric_features > 0);
                assert_eq!(counts.categorical_order_replays, 0);
                assert_eq!(counts.categorical_score_replays, 0);
                assert_eq!(backend.node_counts().cpu_nodes, 0);
                assert!(available(), "{:?}", unavailable_reason());
            }
        }
    }

    #[test]
    fn nan_category_order_and_score_request_explicit_host_replay() {
        if !has_device() {
            return;
        }
        let (index, gradients) = fixture(4);
        let backend = CudaHistBackend::new(&index, 0).unwrap();
        let rows: Vec<u32> = (0..index.n_rows() as u32).collect();
        backend.prepare(&index, &gradients);
        let seg = backend.begin_tree(&index, &rows).unwrap();
        assert_eq!(backend.reserve_hists(&index, 1), Some(true));
        backend
            .build_resident(&index, Some(&gradients), &[(seg, 0)], &[])
            .unwrap();
        let params = TrainingParams::builder()
            .min_child_weight(0.0)
            .build()
            .unwrap();
        let builder = HistTreeBuilder::new(&params);
        let total = backend.root_total(seg).unwrap();
        let (_, ctx) = builder.root_context(&mut ColumnSampler::all(3), total, rows.len());
        let node = ResidentNode {
            slot: 0,
            features: &[0],
            allowed: None,
            ctx,
        };
        let mut hist = zeroed(index.total_bins());
        CpuBackend.build(&index, &rows, &gradients, &mut hist);
        hist[0].hess = f64::NAN;
        backend.replace_resident_hist_for_test(0, &hist).unwrap();
        let request = ScanRequest {
            slot: 0,
            total,
            root_gain: builder.node_scorer(ctx).root_gain,
            lower: f32::NEG_INFINITY,
            upper: f32::INFINITY,
            features: &[(0, 0)],
        };
        assert_eq!(
            backend
                .scan_resident(&index, &builder.config.reg, &[request])
                .unwrap(),
            vec![NodeScan::Replay(ScanFallback::CategoricalOrder)]
        );
        let on = Device {
            engine: &backend,
            ghist: &index,
            gpair: Some(&gradients),
        };
        let actual = builder
            .resident_merge(on, &node, NodeScan::Replay(ScanFallback::CategoricalOrder))
            .unwrap();
        let expected = builder.evaluate(&index, &hist, &[0], None, ctx);
        assert_split_bits(&actual, &expected);
        assert_eq!(backend.scan_diagnostics().categorical_order_replays, 1);
        // Finite category keys with a NaN parent baseline cannot be
        // reduced as ordinary winners either. Test both feature kinds.
        CpuBackend.build(&index, &rows, &gradients, &mut hist);
        backend.replace_resident_hist_for_test(0, &hist).unwrap();
        for (feature, reason) in [
            (0, ScanFallback::CategoricalScore),
            (1, ScanFallback::NumericScore),
        ] {
            let nan_request = ScanRequest {
                root_gain: f32::NAN,
                features: &[(feature, 0)],
                ..request
            };
            assert_eq!(
                backend
                    .scan_resident(&index, &builder.config.reg, &[nan_request])
                    .unwrap(),
                vec![NodeScan::Replay(reason)]
            );
        }
        assert_eq!(backend.scan_diagnostics().categorical_score_replays, 1);
        assert_eq!(backend.scan_diagnostics().numeric_score_replays, 1);
        assert!(available(), "{:?}", unavailable_reason());
    }
}
