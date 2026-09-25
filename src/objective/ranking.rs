//! Learning-to-rank objectives (XGBoost 3.4.1 LambdaRank).
//!
//! Query groups are contiguous row blocks ([`crate::data::GroupInfo`]). Within
//! each group, documents are stably ranked by descending prediction. The
//! default XGBoost `topk` pair method visits `(i,j)` for every model-rank
//! `i < min(group_size, 32)` and every `j > i`; unequal-label pairs receive the
//! pairwise logistic lambda, optionally weighted by the exact NDCG or MAP
//! change. Pair values, accumulation, per-query normalization, and query
//! weighting use XGBoost's float/double conversion points.

use super::{GradPair, MIN_HESS_F64, Objective, check_label_domain};
use crate::data::{GroupInfo, MetaInfo};
use crate::error::{HessboostError, Result};
use crate::metric::{argsort_desc, group_ranges};
use rayon::prelude::*;

/// Which ranking loss the LambdaMART objective optimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RankMode {
    /// Plain pairwise logistic loss (`rank:pairwise`). Every pair is weighted 1.
    Pairwise,
    /// Pairs weighted by |ΔNDCG| (`rank:ndcg`).
    Ndcg,
    /// Pairs weighted by |ΔMAP| (`rank:map`).
    Map,
}

/// The LambdaMART pairwise ranking objective.
///
/// Supports three XGBoost-compatible modes: `rank:pairwise`, `rank:ndcg`, and
/// `rank:map`. See the module-level documentation for the algorithm.
#[derive(Debug, Clone, Copy)]
pub struct LambdaMart {
    mode: RankMode,
    top_k: usize,
}

impl LambdaMart {
    /// Plain pairwise logistic ranking (`rank:pairwise`).
    pub fn pairwise(top_k: usize) -> Self {
        LambdaMart {
            mode: RankMode::Pairwise,
            top_k,
        }
    }

    /// NDCG-weighted LambdaMART (`rank:ndcg`).
    pub fn ndcg(top_k: usize) -> Self {
        LambdaMart {
            mode: RankMode::Ndcg,
            top_k,
        }
    }

    /// MAP-weighted LambdaMART (`rank:map`).
    pub fn map(top_k: usize) -> Self {
        LambdaMart {
            mode: RankMode::Map,
            top_k,
        }
    }

    /// Accumulate XGBoost's CPU top-k LambdaRank gradient for one query:
    /// `p`, `y` and `out` are that query's rows.
    fn accumulate_group(
        &self,
        p: &[f32],
        y: &[f32],
        query_weight: f32,
        weight_norm: f32,
        out: &mut [GradPair],
    ) {
        let n = p.len();
        if n < 2 {
            return;
        }
        let order = argsort_desc(p);
        let metric = MetricCtx::build(self.mode, y, &order, self.top_k);
        let best_score = p[order[0]];
        let worst_score = p[*order.last().unwrap()];
        let mut sum_lambda = 0.0f64;
        for i in 0..n.min(self.top_k) {
            for j in i + 1..n {
                let mut rank_high = i;
                let mut rank_low = j;
                let mut idx_high = order[rank_high];
                let mut idx_low = order[rank_low];
                if y[idx_high] == y[idx_low] {
                    continue;
                }
                if y[idx_high] < y[idx_low] {
                    std::mem::swap(&mut rank_high, &mut rank_low);
                    std::mem::swap(&mut idx_high, &mut idx_low);
                }
                let score_diff = p[idx_high] - p[idx_low]; // float subtraction
                let delta_score = f64::from(score_diff.abs());
                let sigmoid = f64::from(crate::simd::sigmoid_scalar(score_diff));
                let mut delta = metric
                    .delta(y[idx_high], y[idx_low], rank_high, rank_low)
                    .abs();
                if best_score != worst_score {
                    delta /= delta_score + 0.01;
                }
                let lambda = (sigmoid - 1.0) * delta;
                let hessian = (sigmoid * (1.0 - sigmoid)).max(MIN_HESS_F64) * delta * 2.0;
                let pg = GradPair::new(lambda as f32, hessian as f32);
                out[idx_high].grad += pg.grad;
                out[idx_high].hess += pg.hess;
                out[idx_low].grad -= pg.grad;
                out[idx_low].hess += pg.hess;
                sum_lambda += -2.0 * f64::from(pg.grad);
            }
        }

        normalize_group(out, sum_lambda, query_weight, weight_norm);
    }
}

/// XGBoost `CalcLambdaForGroup`'s scaling of one query's accumulated pairs:
/// `norm` (double) scales them only when it differs from 1, then `w`
/// (float), then `w_norm` (double); each `GradientPair::operator*(float)`
/// rounds its factor to f32 and multiplies separately, so the three factors
/// are never pre-combined.
fn normalize_group(out: &mut [GradPair], sum_lambda: f64, query_weight: f32, weight_norm: f32) {
    let norm = if sum_lambda > 0.0 {
        (sum_lambda + 1.0).log2() / sum_lambda
    } else {
        1.0
    };
    if norm != 1.0 {
        let norm = norm as f32;
        for g in out.iter_mut() {
            g.grad *= norm;
            g.hess *= norm;
        }
    }
    for g in out.iter_mut() {
        g.grad *= query_weight;
        g.hess *= query_weight;
        g.grad *= weight_norm;
        g.hess *= weight_norm;
    }
}

/// Queries at least this many rows in total compute their gradients in
/// parallel (each query writes only its own rows).
const PARALLEL_RANK_ROWS: usize = 4096;

impl Objective for LambdaMart {
    fn name(&self) -> &str {
        match self.mode {
            RankMode::Pairwise => "rank:pairwise",
            RankMode::Ndcg => "rank:ndcg",
            RankMode::Map => "rank:map",
        }
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        // Without group info the whole batch is one query group.
        self.gradient_grouped(preds, labels, weights, None, out);
    }

    fn gradient_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&GroupInfo>,
        out: &mut [GradPair],
    ) {
        super::check_gradient_inputs(labels.len(), 1, preds, labels, weights, out);
        out.fill(GradPair::default());
        let ranges = group_ranges(preds.len(), group);
        // Ranking weights are per query. DMatrix expands group weights across
        // rows, so the first row of each group recovers the query weight.
        let group_weights: Vec<f32> = ranges
            .iter()
            .map(|&(start, _)| weights.map_or(1.0, |w| w[start]))
            .collect();
        let sum_w: f64 = group_weights.iter().map(|&w| f64::from(w)).sum();
        let weight_norm = if sum_w == 0.0 {
            0.0
        } else {
            (ranges.len() as f64 / sum_w) as f32
        };
        // The ranges tile the rows in order, so each query gets its own
        // disjoint slice of `out`.
        let mut queries = Vec::with_capacity(ranges.len());
        let mut rest = &mut out[..];
        let mut offset = 0;
        for (&(start, end), &weight) in ranges.iter().zip(&group_weights) {
            let (_, tail) = std::mem::take(&mut rest).split_at_mut(start - offset);
            let (rows, tail) = tail.split_at_mut(end - start);
            rest = tail;
            offset = end;
            queries.push((start, rows, weight));
        }
        let query = |(start, out, weight): (usize, &mut [GradPair], f32)| {
            let end = start + out.len();
            self.accumulate_group(
                &preds[start..end],
                &labels[start..end],
                weight,
                weight_norm,
                out,
            );
        };
        if preds.len() >= PARALLEL_RANK_ROWS
            && queries.len() > 1
            && rayon::current_num_threads() > 1
        {
            queries.into_par_iter().for_each(query);
        } else {
            queries.into_iter().for_each(query);
        }
    }

    /// One label per row in the objective's domain, and non-empty query
    /// groups that cover the rows in order, each with one constant weight
    /// (lengths are checked before any group is sliced).
    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        info.check_layout()?;
        super::check_label_width(info, 1)?;
        match self.mode {
            // NDCG gains are `2^label - 1` in a `u32`: relevance in [0, 31].
            RankMode::Ndcg => check_label_domain(info, |y| !(0.0..=31.0).contains(&y))?,
            RankMode::Pairwise | RankMode::Map => check_label_domain(info, |y| y < 0.0)?,
        }
        let Some(group) = info.group else {
            return Err(HessboostError::invalid_param(
                "group_sizes",
                "ranking dataset requires group information",
            ));
        };
        if !group.partitions(info.n_rows) || group.iter_ranges().any(|(start, end)| start == end) {
            return Err(HessboostError::invalid_param(
                "group_sizes",
                format!(
                    "dataset has {} rows, but its query groups are not non-empty consecutive \
                     row ranges covering them",
                    info.n_rows
                ),
            ));
        }
        if let Some(weights) = info.weights {
            for (start, end) in group.iter_ranges() {
                if weights[start..end]
                    .iter()
                    .any(|weight| *weight != weights[start])
                {
                    return Err(HessboostError::invalid_param(
                        "weights",
                        "ranking dataset requires one constant weight per query group",
                    ));
                }
            }
        }
        Ok(())
    }

    fn default_metric(&self) -> String {
        // XGBoost `RankEvalMetric`: `ndcg@k` for `rank:pairwise` and
        // `rank:ndcg`, `map@k` for `rank:map`, with `k` the `topk` pair count.
        let base = match self.mode {
            RankMode::Pairwise | RankMode::Ndcg => "ndcg",
            RankMode::Map => "map",
        };
        format!("{base}@{}", self.top_k)
    }
}

/// XGBoost's NDCG gain `2^label - 1` (`rank:ndcg` labels lie in `[0, 31]`).
fn ndcg_gain(label: f32) -> f64 {
    f64::from((1u32 << label as u32) - 1)
}

/// Per-query data for XGBoost's metric deltas. Ranks are model-score ranks.
enum MetricCtx {
    Uniform,
    Ndcg { discounts: Vec<f64>, inv_idcg: f64 },
    Map { n_rel: Vec<f64>, acc: Vec<f64> },
}

impl MetricCtx {
    fn build(mode: RankMode, labels: &[f32], order: &[usize], top_k: usize) -> Self {
        match mode {
            RankMode::Pairwise => MetricCtx::Uniform,
            RankMode::Ndcg => {
                let discounts: Vec<f64> = (0..labels.len())
                    .map(|i| 1.0 / ((i + 2) as f64).log2())
                    .collect();
                let mut ideal: Vec<usize> = (0..labels.len()).collect();
                ideal.sort_by(|&a, &b| labels[b].total_cmp(&labels[a])); // stable
                let idcg: f64 = ideal
                    .iter()
                    .take(labels.len().min(top_k))
                    .enumerate()
                    .map(|(rank, &idx)| discounts[rank] * ndcg_gain(labels[idx]))
                    .sum();
                MetricCtx::Ndcg {
                    discounts,
                    inv_idcg: if idcg == 0.0 { 0.0 } else { 1.0 / idcg },
                }
            }
            RankMode::Map => {
                let mut n_rel = vec![0.0; labels.len()];
                let mut acc = vec![0.0; labels.len()];
                for (rank, &idx) in order.iter().enumerate() {
                    let y = f64::from(labels[idx]);
                    n_rel[rank] = y + if rank == 0 { 0.0 } else { n_rel[rank - 1] };
                    acc[rank] = y / (rank + 1) as f64 + if rank == 0 { 0.0 } else { acc[rank - 1] };
                }
                MetricCtx::Map { n_rel, acc }
            }
        }
    }

    fn delta(&self, y_high: f32, y_low: f32, rank_high: usize, rank_low: usize) -> f64 {
        match self {
            MetricCtx::Uniform => 1.0,
            MetricCtx::Ndcg {
                discounts,
                inv_idcg,
            } => {
                let (gain_high, gain_low) = (ndcg_gain(y_high), ndcg_gain(y_low));
                let original = gain_high * discounts[rank_high] + gain_low * discounts[rank_low];
                let changed = gain_low * discounts[rank_high] + gain_high * discounts[rank_low];
                (original - changed) * inv_idcg
            }
            MetricCtx::Map { n_rel, acc } => {
                let (mut rh, mut rl, mut yh, mut yl) =
                    (rank_high, rank_low, f64::from(y_high), f64::from(y_low));
                if rh > rl {
                    std::mem::swap(&mut rh, &mut rl);
                    std::mem::swap(&mut yh, &mut yl);
                }
                let total = *n_rel.last().unwrap();
                let (m, n) = (n_rel[rl], n_rel[rh]);
                let b = acc[rl - 1] - acc[rh];
                if yh < yl {
                    (m / (rl + 1) as f64 - (n + 1.0) / (rh + 1) as f64 - b) / total
                } else {
                    (n / (rh + 1) as f64 - m / (rl + 1) as f64 + b) / total
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gradients of `obj` over query groups of `sizes`.
    fn grouped(
        obj: LambdaMart,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        sizes: &[usize],
    ) -> Vec<GradPair> {
        let mut out = vec![GradPair::default(); preds.len()];
        let group = GroupInfo::from_sizes(sizes);
        obj.gradient_grouped(preds, labels, weights, Some(&group), &mut out);
        out
    }

    /// Queries computed in parallel give the serial gradients bit for bit,
    /// with empty groups and query weights in the mix.
    #[test]
    fn parallel_queries_match_serial() {
        let mut sizes: Vec<usize> = (0..400).map(|g| 1 + (g * 37) % 60).collect();
        sizes[3] = 0;
        let n: usize = sizes.iter().sum();
        assert!(n >= PARALLEL_RANK_ROWS);
        let preds: Vec<f32> = (0..n).map(|i| ((i * 7919) % 1009) as f32 / 101.0).collect();
        let labels: Vec<f32> = (0..n).map(|i| ((i * 31) % 5) as f32).collect();
        let mut weights = Vec::with_capacity(n);
        for (g, &size) in sizes.iter().enumerate() {
            weights.extend(std::iter::repeat_n(0.5 + (g % 4) as f32 * 0.25, size));
        }
        let pool = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        };
        for obj in [
            LambdaMart::pairwise(32),
            LambdaMart::ndcg(8),
            LambdaMart::map(32),
        ] {
            for w in [None, Some(weights.as_slice())] {
                let serial = pool(1).install(|| grouped(obj, &preds, &labels, w, &sizes));
                let parallel = pool(4).install(|| grouped(obj, &preds, &labels, w, &sizes));
                assert_eq!(serial, parallel, "{}", obj.name());
            }
        }
    }

    #[test]
    fn pairwise_pushes_relevant_up() {
        // One group of 3 docs, labels 2 > 1 > 0, all scores equal at start.
        let obj = LambdaMart::ndcg(32);
        let out = grouped(obj, &[0.0, 0.0, 0.0], &[2.0, 1.0, 0.0], None, &[3]);
        // Negative gradient => leaf value positive => score goes up.
        // Most-relevant doc should get the most-negative gradient.
        assert!(out[0].grad < out[1].grad, "{out:?}");
        assert!(out[1].grad < out[2].grad, "{out:?}");
        assert!(out[0].grad < 0.0 && out[2].grad > 0.0);
        // Hessians are non-negative.
        assert!(out.iter().all(|g| g.hess >= 0.0));
    }

    #[test]
    fn no_pairs_when_all_labels_equal() {
        let obj = LambdaMart::pairwise(32);
        let out = grouped(obj, &[0.5, -0.2, 1.0], &[1.0, 1.0, 1.0], None, &[3]);
        assert!(out.iter().all(|g| g.grad == 0.0 && g.hess == 0.0));
    }

    #[test]
    fn groups_are_independent() {
        // Two groups; a cross-group pair must never be formed.
        let obj = LambdaMart::pairwise(32);
        let out = grouped(obj, &[0.0; 4], &[1.0, 0.0, 0.0, 1.0], None, &[2, 2]);
        // Within each group the relevant doc is pushed up, the other down.
        assert!(out[0].grad < 0.0 && out[1].grad > 0.0);
        assert!(out[3].grad < 0.0 && out[2].grad > 0.0);
    }

    /// Hand-computed XGBoost `LambdaGrad` for one `rank:pairwise` pair whose
    /// query has distinct best/worst scores (`delta_metric = 1 / (|Δs| + 0.01)`).
    fn pairwise_pair(s_high: f32, s_low: f32) -> (f32, f32) {
        let diff = s_high - s_low;
        let sigmoid = f64::from(1.0f32 / ((-diff).min(88.7).exp() + 1.0));
        let delta = 1.0 / (f64::from(diff.abs()) + 0.01);
        let lambda = (sigmoid - 1.0) * delta;
        let hessian = (sigmoid * (1.0 - sigmoid)).max(MIN_HESS_F64) * delta * 2.0;
        (lambda as f32, hessian as f32)
    }

    #[test]
    fn signed_zero_scores_tie_in_input_order() {
        // `-0.0` and `0.0` compare equal under XGBoost's stable
        // `std::greater<>` argsort, so doc 0 stays ranked first. With
        // `top_k = 1` the pairs are exactly (0,1) and (0,2); a total-order sort
        // would rank doc 1 first and pair (1,0),(1,2) instead.
        let obj = LambdaMart::pairwise(1);
        let preds = [-0.0f32, 0.0, -1.0];
        let labels = [2.0f32, 1.0, 0.0];
        let out = grouped(obj, &preds, &labels, None, &[3]);

        let (g01, h01) = pairwise_pair(preds[0], preds[1]);
        let (g02, h02) = pairwise_pair(preds[0], preds[2]);
        let sum_lambda = -2.0 * f64::from(g01) + -2.0 * f64::from(g02);
        let norm = ((sum_lambda + 1.0).log2() / sum_lambda) as f32;

        assert_ne!(out[0].grad, 0.0);
        assert_eq!(out[0].grad, (g01 + g02) * norm, "{out:?}");
        assert_eq!(out[0].hess, (h01 + h02) * norm, "{out:?}");
        // Doc 1 only ever sees the (0,1) pair.
        assert_eq!(out[1].grad, -g01 * norm, "{out:?}");
        assert_eq!(out[1].hess, h01 * norm, "{out:?}");
        assert_eq!(out[2].grad, -g02 * norm, "{out:?}");
        assert_eq!(out[2].hess, h02 * norm, "{out:?}");
    }

    #[test]
    fn query_weights_scale_as_sequential_f32_products() {
        // XGBoost applies `norm`, `w`, and `w_norm` as three separate f32
        // multiplications; folding them into one scale (`norm * w * w_norm`)
        // rounds differently for these weights.
        let obj = LambdaMart::ndcg(32);
        let preds = [0.3f32, -0.7, 1.1, 0.2, -0.4, 0.9, 0.05];
        let labels = [2.0f32, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0];
        let sizes = [3, 4];
        let group_w = [0.3f32, 1.7];
        let weights = [0.3f32, 0.3, 0.3, 1.7, 1.7, 1.7, 1.7];

        // Unweighted: `w = 1`, `w_norm = 1`, so this is `pairs * norm` exactly.
        let normed = grouped(obj, &preds, &labels, None, &sizes);
        let out = grouped(obj, &preds, &labels, Some(&weights), &sizes);

        let sum_w = f64::from(group_w[0]) + f64::from(group_w[1]);
        let w_norm = (2.0 / sum_w) as f32;
        for (i, (got, base)) in out.iter().zip(&normed).enumerate() {
            let w = if i < 3 { group_w[0] } else { group_w[1] };
            assert_eq!(
                got.grad.to_bits(),
                ((base.grad * w) * w_norm).to_bits(),
                "{i}"
            );
            assert_eq!(
                got.hess.to_bits(),
                ((base.hess * w) * w_norm).to_bits(),
                "{i}"
            );
        }
    }

    #[test]
    fn default_metric_follows_mode_and_top_k() {
        assert_eq!(LambdaMart::pairwise(32).default_metric(), "ndcg@32");
        assert_eq!(LambdaMart::ndcg(5).default_metric(), "ndcg@5");
        assert_eq!(LambdaMart::map(10).default_metric(), "map@10");
    }

    /// Groups or weights that do not match the rows are refused before any
    /// group is sliced (a group of 2 over 1 row used to index out of bounds).
    #[test]
    fn validate_info_refuses_inconsistent_groups_and_weights() {
        let obj = LambdaMart::pairwise(32);
        let refused = |labels: &[f32], weights: Option<&[f32]>, group: GroupInfo| {
            let err = obj.validate_info(&MetaInfo::new(labels, weights, Some(&group)));
            assert!(
                matches!(err, Err(HessboostError::InvalidParameter { .. })),
                "{group:?}: {err:?}"
            );
        };
        refused(&[1.0], Some(&[1.0]), GroupInfo::from_sizes(&[2]));
        refused(&[1.0, 0.0], None, GroupInfo::from_sizes(&[1]));
        refused(&[1.0, 0.0], None, GroupInfo::from_sizes(&[2, 0]));
        refused(&[1.0, 0.0], Some(&[1.0]), GroupInfo::from_sizes(&[2]));
        let unordered = GroupInfo {
            group_ptr: vec![0, 3, 1, 3],
        };
        refused(&[1.0, 0.0, 1.0], None, unordered);
        let fine = GroupInfo::from_sizes(&[1, 2]);
        let info = MetaInfo::new(&[1.0, 0.0, 1.0], Some(&[2.0, 1.0, 1.0]), Some(&fine));
        obj.validate_info(&info).unwrap();
    }
}
