//! Learning-to-rank objectives (XGBoost 3.4.1 LambdaRank).
//!
//! Query groups are contiguous row blocks ([`crate::data::GroupInfo`]). Within
//! each group, documents are stably ranked by descending prediction. The
//! default XGBoost `topk` pair method visits `(i,j)` for every model-rank
//! `i < min(group_size, 32)` and every `j > i`; unequal-label pairs receive the
//! pairwise logistic lambda, optionally weighted by the exact NDCG or MAP
//! change. Pair values, accumulation, per-query normalization, and query
//! weighting use XGBoost's float/double conversion points.

use super::{GradPair, Objective};
use crate::data::GroupInfo;
use crate::metric::{argsort_desc, group_ranges};

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
pub struct LambdaMartObjective {
    mode: RankMode,
    top_k: usize,
}

impl LambdaMartObjective {
    /// Plain pairwise logistic ranking (`rank:pairwise`).
    pub fn pairwise(top_k: usize) -> Self {
        LambdaMartObjective {
            mode: RankMode::Pairwise,
            top_k,
        }
    }

    /// NDCG-weighted LambdaMART (`rank:ndcg`).
    pub fn ndcg(top_k: usize) -> Self {
        LambdaMartObjective {
            mode: RankMode::Ndcg,
            top_k,
        }
    }

    /// MAP-weighted LambdaMART (`rank:map`).
    pub fn map(top_k: usize) -> Self {
        LambdaMartObjective {
            mode: RankMode::Map,
            top_k,
        }
    }

    /// Accumulate XGBoost's CPU top-k LambdaRank gradient for one query.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_group(
        &self,
        preds: &[f32],
        labels: &[f32],
        start: usize,
        end: usize,
        query_weight: f32,
        weight_norm: f32,
        out: &mut [GradPair],
    ) {
        let n = end - start;
        if n < 2 {
            return;
        }
        let p = &preds[start..end];
        let y = &labels[start..end];
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
                let delta_score = score_diff.abs() as f64;
                let sigmoid = (1.0f32 / ((-score_diff).min(88.7).exp() + 1.0)) as f64;
                let mut delta = metric
                    .delta(y[idx_high], y[idx_low], rank_high, rank_low)
                    .abs();
                if best_score != worst_score {
                    delta /= delta_score + 0.01;
                }
                let lambda = (sigmoid - 1.0) * delta;
                let hessian = (sigmoid * (1.0 - sigmoid)).max(1e-16) * delta * 2.0;
                let pg = GradPair::new(lambda as f32, hessian as f32);
                out[start + idx_high].grad += pg.grad;
                out[start + idx_high].hess += pg.hess;
                out[start + idx_low].grad -= pg.grad;
                out[start + idx_low].hess += pg.hess;
                sum_lambda += -2.0 * pg.grad as f64;
            }
        }

        // XGBoost `CalcLambdaForGroup`: `norm` (double) scales the pairs only
        // when it differs from 1, then `w` (float), then `w_norm` (double);
        // each `GradientPair::operator*(float)` rounds its factor to f32 and
        // multiplies separately, so the three factors are never pre-combined.
        let norm = if sum_lambda > 0.0 {
            (sum_lambda + 1.0).log2() / sum_lambda
        } else {
            1.0
        };
        let group = &mut out[start..end];
        if norm != 1.0 {
            let norm = norm as f32;
            for g in group.iter_mut() {
                g.grad *= norm;
                g.hess *= norm;
            }
        }
        for g in group.iter_mut() {
            g.grad *= query_weight;
            g.hess *= query_weight;
            g.grad *= weight_norm;
            g.hess *= weight_norm;
        }
    }

    /// Shared gradient computation over an optional group layout.
    fn compute(
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
        let sum_w: f64 = group_weights.iter().map(|&w| w as f64).sum();
        let weight_norm = if sum_w == 0.0 {
            0.0
        } else {
            (ranges.len() as f64 / sum_w) as f32
        };
        for ((start, end), weight) in ranges.into_iter().zip(group_weights) {
            self.accumulate_group(preds, labels, start, end, weight, weight_norm, out);
        }
    }
}

impl Objective for LambdaMartObjective {
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
        self.compute(preds, labels, weights, None, out);
    }

    fn gradient_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&GroupInfo>,
        out: &mut [GradPair],
    ) {
        self.compute(preds, labels, weights, group, out);
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
                    .map(|(rank, &idx)| {
                        let gain = ((1u32 << labels[idx] as u32) - 1) as f64;
                        discounts[rank] * gain
                    })
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
                    let y = labels[idx] as f64;
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
                let gain_high = ((1u32 << y_high as u32) - 1) as f64;
                let gain_low = ((1u32 << y_low as u32) - 1) as f64;
                let original = gain_high * discounts[rank_high] + gain_low * discounts[rank_low];
                let changed = gain_low * discounts[rank_high] + gain_high * discounts[rank_low];
                (original - changed) * inv_idcg
            }
            MetricCtx::Map { n_rel, acc } => {
                let (mut rh, mut rl, mut yh, mut yl) =
                    (rank_high, rank_low, y_high as f64, y_low as f64);
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
    use crate::data::GroupInfo;

    #[test]
    fn pairwise_pushes_relevant_up() {
        // One group of 3 docs, labels 2 > 1 > 0, all scores equal at start.
        let obj = LambdaMartObjective::ndcg(32);
        let preds = [0.0f32, 0.0, 0.0];
        let labels = [2.0f32, 1.0, 0.0];
        let g = GroupInfo::from_sizes(&[3]);
        let mut out = vec![GradPair::default(); 3];
        obj.gradient_grouped(&preds, &labels, None, Some(&g), &mut out);
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
        let obj = LambdaMartObjective::pairwise(32);
        let preds = [0.5f32, -0.2, 1.0];
        let labels = [1.0f32, 1.0, 1.0];
        let g = GroupInfo::from_sizes(&[3]);
        let mut out = vec![GradPair::default(); 3];
        obj.gradient_grouped(&preds, &labels, None, Some(&g), &mut out);
        assert!(out.iter().all(|g| g.grad == 0.0 && g.hess == 0.0));
    }

    #[test]
    fn groups_are_independent() {
        // Two groups; a cross-group pair must never be formed.
        let obj = LambdaMartObjective::pairwise(32);
        let preds = [0.0f32, 0.0, 0.0, 0.0];
        let labels = [1.0f32, 0.0, 0.0, 1.0];
        let g = GroupInfo::from_sizes(&[2, 2]);
        let mut out = vec![GradPair::default(); 4];
        obj.gradient_grouped(&preds, &labels, None, Some(&g), &mut out);
        // Within each group the relevant doc is pushed up, the other down.
        assert!(out[0].grad < 0.0 && out[1].grad > 0.0);
        assert!(out[3].grad < 0.0 && out[2].grad > 0.0);
    }

    /// Hand-computed XGBoost `LambdaGrad` for one `rank:pairwise` pair whose
    /// query has distinct best/worst scores (`delta_metric = 1 / (|Δs| + 0.01)`).
    fn pairwise_pair(s_high: f32, s_low: f32) -> (f32, f32) {
        let diff = s_high - s_low;
        let sigmoid = (1.0f32 / ((-diff).min(88.7).exp() + 1.0)) as f64;
        let delta = 1.0 / (diff.abs() as f64 + 0.01);
        let lambda = (sigmoid - 1.0) * delta;
        let hessian = (sigmoid * (1.0 - sigmoid)).max(1e-16) * delta * 2.0;
        (lambda as f32, hessian as f32)
    }

    #[test]
    fn signed_zero_scores_tie_in_input_order() {
        // `-0.0` and `0.0` compare equal under XGBoost's stable
        // `std::greater<>` argsort, so doc 0 stays ranked first. With
        // `top_k = 1` the pairs are exactly (0,1) and (0,2); a total-order sort
        // would rank doc 1 first and pair (1,0),(1,2) instead.
        let obj = LambdaMartObjective::pairwise(1);
        let preds = [-0.0f32, 0.0, -1.0];
        let labels = [2.0f32, 1.0, 0.0];
        let g = GroupInfo::from_sizes(&[3]);
        let mut out = vec![GradPair::default(); 3];
        obj.gradient_grouped(&preds, &labels, None, Some(&g), &mut out);

        let (g01, h01) = pairwise_pair(preds[0], preds[1]);
        let (g02, h02) = pairwise_pair(preds[0], preds[2]);
        let sum_lambda = -2.0 * g01 as f64 + -2.0 * g02 as f64;
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
        let obj = LambdaMartObjective::ndcg(32);
        let preds = [0.3f32, -0.7, 1.1, 0.2, -0.4, 0.9, 0.05];
        let labels = [2.0f32, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0];
        let g = GroupInfo::from_sizes(&[3, 4]);
        let group_w = [0.3f32, 1.7];
        let weights = [0.3f32, 0.3, 0.3, 1.7, 1.7, 1.7, 1.7];

        // Unweighted: `w = 1`, `w_norm = 1`, so this is `pairs * norm` exactly.
        let mut normed = vec![GradPair::default(); 7];
        obj.gradient_grouped(&preds, &labels, None, Some(&g), &mut normed);
        let mut out = vec![GradPair::default(); 7];
        obj.gradient_grouped(&preds, &labels, Some(&weights), Some(&g), &mut out);

        let sum_w = group_w[0] as f64 + group_w[1] as f64;
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
        assert_eq!(
            LambdaMartObjective::pairwise(32).default_metric(),
            "ndcg@32"
        );
        assert_eq!(LambdaMartObjective::ndcg(5).default_metric(), "ndcg@5");
        assert_eq!(LambdaMartObjective::map(10).default_metric(), "map@10");
    }
}
