//! XE-NDCG ranking objective matching LightGBM 4.6's `RankXENDCG` formula
//! (Bruch, WWW 2020, <https://arxiv.org/abs/1911.09798>).
//!
//! The randomized target `2^label - U(0, 1)`, softmax probabilities, and
//! approximate gradient and diagonal Hessian terms follow LightGBM's
//! `src/objective/rank_objective.hpp`. Draws are stateless SplitMix64 values
//! keyed by seed, boosting iteration, query, and document. LightGBM seeds a
//! mutable `Random` stream for each query with `objective_seed + query_index`;
//! the formula matches, but the RNG values and trained trees differ.
//! Unlike LightGBM's direct `1 - rho` denominator, softmax complements are
//! computed stably; for complements below `f32::EPSILON`, the higher-order
//! correction is replaced by its finite first-order gradient limit.

use super::{GradPair, Objective, check_label_domain, check_label_width};
use crate::data::{GroupInfo, MetaInfo};
use crate::error::{HessboostError, Result};
use crate::metric::group_ranges;
use crate::rng::{GOLDEN, mix64};

use rayon::prelude::*;

/// Query batches this large use the same disjoint-group parallel strategy as LambdaMART.
const PARALLEL_XENDCG_ROWS: usize = 4096;
/// XE-NDCG objective (LightGBM `rank_xendcg`, named `rank:xendcg` here).
pub struct Xendcg {
    seed: u64,
}

struct QueryGradient<'a> {
    seed: u64,
    iteration: usize,
    query: usize,
    scores: &'a [f32],
    labels: &'a [f32],
    weights: Option<&'a [f32]>,
}

#[derive(Default)]
struct QueryScratch {
    rho: Vec<f64>,
    one_minus_rho: Vec<f64>,
    target: Vec<f64>,
}

impl Xendcg {
    /// Construct XE-NDCG with the seed used to key its random draws.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn accumulate_query(
        context: &QueryGradient<'_>,
        out: &mut [GradPair],
        scratch: &mut QueryScratch,
    ) {
        let QueryGradient {
            seed,
            iteration,
            query,
            scores,
            labels,
            weights,
        } = context;
        let QueryScratch {
            rho,
            one_minus_rho,
            target,
        } = scratch;
        let n = scores.len();
        if n <= 1 {
            out.fill(GradPair::default());
            return;
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        rho.clear();
        rho.extend(
            scores
                .iter()
                .map(|&score| (f64::from(score) - f64::from(max_score)).exp()),
        );
        let denominator: f64 = rho.iter().sum();
        let max_index = scores
            .iter()
            .position(|&score| score == max_score)
            .unwrap_or(0);
        let max_count = scores.iter().filter(|&&score| score == max_score).count();
        one_minus_rho.clear();
        one_minus_rho.extend(rho.iter().enumerate().map(|(index, &value)| {
            if max_count == 1 && index == max_index {
                rho.iter()
                    .enumerate()
                    .filter(|(tail_index, _)| *tail_index != max_index)
                    .map(|(_, probability)| probability)
                    .sum::<f64>()
                    / denominator
            } else {
                (denominator - value) / denominator
            }
        }));
        for probability in rho.iter_mut() {
            *probability /= denominator;
        }

        let seed_key = mix64(*seed);
        let iter_key = mix64(seed_key ^ (*iteration as u64).wrapping_mul(GOLDEN));
        let query_key = mix64(iter_key ^ (*query as u64).wrapping_mul(GOLDEN));
        target.clear();
        target.extend(labels.iter().enumerate().map(|(doc, &label)| {
            let bits = mix64(query_key ^ (doc as u64).wrapping_mul(GOLDEN));
            let draw = (bits >> 40) as f32 * (1.0 / (1u32 << 24) as f32);
            2.0f64.powf(f64::from(label)) - f64::from(draw)
        }));
        let inv_target_sum = 1.0 / target.iter().sum::<f64>().max(1e-15);
        // The higher-order correction divides twice by 1 - rho. When the
        // softmax is saturated, use the finite first-order gradient limit.
        if one_minus_rho.iter().any(|&q| q < f64::from(f32::EPSILON)) {
            for (i, pair) in out.iter_mut().enumerate() {
                pair.grad = (rho[i] - target[i] * inv_target_sum) as f32;
                pair.hess = (rho[i] * one_minus_rho[i]) as f32;
                if let Some(weights) = weights {
                    pair.grad *= weights[i];
                    pair.hess *= weights[i];
                }
            }
            return;
        }
        let mut sum_l1 = 0.0;
        for i in 0..n {
            let term = -target[i] * inv_target_sum + rho[i];
            out[i].grad = term as f32;
            target[i] = term / one_minus_rho[i];
            sum_l1 += target[i];
        }
        let mut sum_l2 = 0.0;
        for i in 0..n {
            let term = rho[i] * (sum_l1 - target[i]);
            out[i].grad += term as f32;
            target[i] = term / one_minus_rho[i];
            sum_l2 += target[i];
        }

        for i in 0..n {
            out[i].grad += (rho[i] * (sum_l2 - target[i])) as f32;
            out[i].hess = (rho[i] * one_minus_rho[i]) as f32;
            if let Some(weights) = weights {
                out[i].grad *= weights[i];
                out[i].hess *= weights[i];
            }
        }
    }
}

impl Objective for Xendcg {
    fn name(&self) -> &'static str {
        "rank:xendcg"
    }

    fn gradient(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        out: &mut [GradPair],
    ) {
        self.gradient_grouped_at(preds, labels, weights, None, out, 0);
    }

    fn gradient_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&GroupInfo>,
        out: &mut [GradPair],
    ) {
        self.gradient_grouped_at(preds, labels, weights, group, out, 0);
    }

    fn gradient_grouped_at(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&GroupInfo>,
        out: &mut [GradPair],
        iteration: usize,
    ) {
        super::check_gradient_inputs(labels.len(), 1, preds, labels, weights, out);
        out.fill(GradPair::default());
        let ranges = group_ranges(preds.len(), group);
        let mut queries = Vec::with_capacity(ranges.len());
        let mut rest = &mut out[..];
        let mut offset = 0;
        for (start, end) in ranges {
            let (_, tail) = std::mem::take(&mut rest).split_at_mut(start - offset);
            let (query_out, tail) = tail.split_at_mut(end - start);
            rest = tail;
            offset = end;
            queries.push((start, query_out));
        }
        let process_query =
            |scratch: &mut QueryScratch,
             (query, (start, query_out)): (usize, (usize, &mut [GradPair]))| {
                let end = start + query_out.len();
                Self::accumulate_query(
                    &QueryGradient {
                        seed: self.seed,
                        iteration,
                        query,
                        scores: &preds[start..end],
                        labels: &labels[start..end],
                        weights: weights.map(|w| &w[start..end]),
                    },
                    query_out,
                    scratch,
                );
            };
        if preds.len() >= PARALLEL_XENDCG_ROWS
            && queries.len() > 1
            && rayon::current_num_threads() > 1
        {
            queries
                .into_par_iter()
                .enumerate()
                .for_each_init(QueryScratch::default, process_query);
        } else {
            let mut scratch = QueryScratch::default();
            for item in queries.into_iter().enumerate() {
                process_query(&mut scratch, item);
            }
        }
    }

    fn gradient_info_at(
        &self,
        preds: &[f32],
        info: &MetaInfo,
        out: &mut [GradPair],
        iteration: usize,
    ) {
        self.gradient_grouped_at(preds, info.labels, info.weights, info.group, out, iteration);
    }

    fn validate_info(&self, info: &MetaInfo) -> Result<()> {
        info.check_layout()?;
        check_label_width(info, 1)?;
        check_label_domain(info, |y| !(0.0..=31.0).contains(&y) || y.fract() != 0.0)?;
        let Some(group) = info.group else {
            return Err(HessboostError::invalid_param(
                "group_sizes",
                "ranking dataset requires group information",
            ));
        };
        if !group.partitions(info.n_rows) || group.iter_ranges().any(|(start, end)| start == end) {
            return Err(HessboostError::invalid_param(
                "group_sizes",
                "ranking dataset requires non-empty consecutive query groups covering all rows",
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
        "ndcg".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_gradient_matches_independent_formula() {
        let scores = [0.0, 0.0];
        let labels = [1.0, 0.0];
        let query_key = mix64(mix64(0));
        let draws: Vec<f64> = (0..2)
            .map(|doc| {
                let bits = mix64(query_key ^ (doc as u64).wrapping_mul(GOLDEN));
                f64::from((bits >> 40) as f32 * (1.0 / (1u32 << 24) as f32))
            })
            .collect();
        let target = [2.0 - draws[0], 1.0 - draws[1]];
        let z = target[0] + target[1];
        let rho = [0.5, 0.5];
        let first = [-target[0] / z + rho[0], -target[1] / z + rho[1]];
        let params1 = [first[0] / 0.5, first[1] / 0.5];
        let sum_l1 = params1[0] + params1[1];
        let second = [
            rho[0] * (sum_l1 - params1[0]),
            rho[1] * (sum_l1 - params1[1]),
        ];
        let params2 = [second[0] / 0.5, second[1] / 0.5];
        let sum_l2 = params2[0] + params2[1];
        let expected = [
            (first[0] + second[0] + rho[0] * (sum_l2 - params2[0])) as f32,
            (first[1] + second[1] + rho[1] * (sum_l2 - params2[1])) as f32,
        ];
        let objective = Xendcg::new(0);
        let mut actual = [GradPair::default(); 2];
        objective.gradient_grouped_at(&scores, &labels, None, None, &mut actual, 0);
        assert_eq!(actual[0].grad, expected[0]);
        assert_eq!(actual[1].grad, expected[1]);
        assert_eq!(actual[0].hess, 0.25);
        assert_eq!(actual[1].hess, 0.25);
        let mut again = [GradPair::default(); 2];
        objective.gradient_grouped_at(&scores, &labels, None, None, &mut again, 4);
        assert_ne!(actual, again);
    }
    #[test]
    fn extreme_score_margins_keep_gradients_finite() {
        let labels = [1.0, 0.0];
        let objective = Xendcg::new(0);
        let mut actual = [GradPair::default(); 2];
        for scores in [[100.0, 0.0], [f32::MAX, -f32::MAX]] {
            objective.gradient_grouped_at(&scores, &labels, None, None, &mut actual, 0);
            assert!(
                actual
                    .iter()
                    .all(|pair| pair.grad.is_finite() && pair.hess.is_finite())
            );
            assert!(actual.iter().all(|pair| pair.hess >= 0.0));
        }
    }

    #[test]
    fn grouped_gradients_are_bit_identical_across_thread_counts() {
        let scores = vec![0.0; 4096];
        let labels: Vec<f32> = (0..4096).map(|i| (i % 5) as f32).collect();
        let group = GroupInfo::from_sizes(&vec![8; 512]);
        let objective = Xendcg::new(123);
        let compute = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut result = vec![GradPair::default(); scores.len()];
                objective.gradient_grouped_at(
                    &scores,
                    &labels,
                    None,
                    Some(&group),
                    &mut result,
                    12,
                );
                result
            })
        };
        assert_eq!(compute(1), compute(4));
    }
}
