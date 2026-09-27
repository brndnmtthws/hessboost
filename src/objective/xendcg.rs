//! XE-NDCG ranking objective matching LightGBM 4.6's `RankXENDCG` formula
//! (Bruch, WWW 2020, <https://arxiv.org/abs/1911.09798>).
//!
//! The randomized target `2^label - U(0, 1)`, softmax probabilities, and
//! approximate gradient and diagonal Hessian terms follow LightGBM's
//! `src/objective/rank_objective.hpp`. Each document's `U` is position
//! `doc` of the counter-based stream keyed by (seed, a salt of its own,
//! boosting iteration, query) ([`crate::rng::stream_key`]). LightGBM seeds a
//! mutable `Random` stream for each query with `objective_seed + query_index`;
//! the formula matches, but the RNG values and trained trees differ.
//! Unlike LightGBM's direct `1 - rho` denominator, softmax complements are
//! computed stably; for complements below `f32::EPSILON`, the higher-order
//! correction is replaced by its finite first-order gradient limit.

use super::{GradPair, Loss, check_label_domain, check_label_width};
use crate::data::{GroupInfo, MetaInfo};
use crate::error::{HessboostError, Result};
use crate::metric::group_ranges;
use crate::rng::{keyed_unit_f32, stream_key};

use rayon::prelude::*;

/// Salt of the XE-NDCG target stream (`"xendcg"`), keeping its draws apart
/// from every other keyed stream of the same seed.
const TARGET_STREAM: u64 = 0x0000_7865_6E64_6367;
/// Query batches this large use the same disjoint-group parallel strategy as LambdaMART.
const PARALLEL_XENDCG_ROWS: usize = 4096;
/// XE-NDCG loss (LightGBM `rank_xendcg`, named `rank:xendcg` here).
pub(crate) struct Xendcg {
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
    pub(crate) fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// The gradients of boosting round `iteration`, query by query (the
    /// whole input is one query without `group`).
    fn query_gradients(
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

        let key = stream_key(&[*seed, TARGET_STREAM, *iteration as u64, *query as u64]);
        target.clear();
        target.extend(labels.iter().enumerate().map(|(doc, &label)| {
            let draw = keyed_unit_f32(key, doc as u64);
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

impl Loss for Xendcg {
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
        self.query_gradients(preds, labels, weights, None, out, 0);
    }

    fn gradient_grouped(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        group: Option<&GroupInfo>,
        out: &mut [GradPair],
    ) {
        self.query_gradients(preds, labels, weights, group, out, 0);
    }

    fn gradient_info_at(
        &self,
        preds: &[f32],
        info: &MetaInfo,
        out: &mut [GradPair],
        iteration: usize,
    ) {
        self.query_gradients(preds, info.labels, info.weights, info.group, out, iteration);
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

    fn default_metric(&self) -> crate::metric::EvalMetric {
        // LightGBM's default `ndcg` metric, over whole queries.
        crate::metric::EvalMetric::Ndcg(crate::metric::Cutoff::all())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LightGBM's `RankXENDCG::GetGradientsForOneQuery` in plain `f64`
    /// (softmax, targets `2^label - U` normalized, then the three
    /// correction terms), for the documents' draws `u`.
    fn reference(scores: &[f64], labels: &[f64], u: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f64 = exp.iter().sum();
        let rho: Vec<f64> = exp.iter().map(|e| e / total).collect();
        let phi: Vec<f64> = labels
            .iter()
            .zip(u)
            .map(|(l, u)| 2f64.powf(*l) - u)
            .collect();
        let phi_sum: f64 = phi.iter().sum();
        let n = scores.len();
        let first: Vec<f64> = (0..n).map(|i| -phi[i] / phi_sum + rho[i]).collect();
        let p1: Vec<f64> = (0..n).map(|i| first[i] / (1.0 - rho[i])).collect();
        let l1: f64 = p1.iter().sum();
        let second: Vec<f64> = (0..n).map(|i| rho[i] * (l1 - p1[i])).collect();
        let p2: Vec<f64> = (0..n).map(|i| second[i] / (1.0 - rho[i])).collect();
        let l2: f64 = p2.iter().sum();
        let grad = (0..n)
            .map(|i| first[i] + second[i] + rho[i] * (l2 - p2[i]))
            .collect();
        let hess = rho.iter().map(|r| r * (1.0 - r)).collect();
        (grad, hess)
    }

    /// Every query's gradients, at the default seed and iteration 0 (the
    /// stream positions a bare `mix64` chain from 0 made degenerate) and at
    /// a later seed and iteration, with unequal scores, match the reference
    /// formula fed the draws of that query's documented stream (seed, salt,
    /// iteration, query).
    #[test]
    fn query_gradient_matches_independent_formula() {
        let scores = [0.0, 0.0, 0.7, -0.4, 1.3];
        let labels = [2.0, 0.0, 1.0, 0.0, 3.0];
        let group = GroupInfo::from_sizes(&[2, 3]);
        let widen = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
        for (seed, iteration) in [(0, 0), (97, 3)] {
            let gradients = |iteration: usize| {
                let mut out = [GradPair::default(); 5];
                Xendcg::new(seed).query_gradients(
                    &scores,
                    &labels,
                    None,
                    Some(&group),
                    &mut out,
                    iteration,
                );
                out
            };
            let actual = gradients(iteration);
            for (query, (start, end)) in group.iter_ranges().enumerate() {
                let key = stream_key(&[seed, TARGET_STREAM, iteration as u64, query as u64]);
                let u: Vec<f64> = (0..end - start)
                    .map(|doc| f64::from(keyed_unit_f32(key, doc as u64)))
                    .collect();
                let (grad, hess) =
                    reference(&widen(&scores[start..end]), &widen(&labels[start..end]), &u);
                for (pair, (g, h)) in actual[start..end].iter().zip(grad.iter().zip(&hess)) {
                    assert!(
                        (f64::from(pair.grad) - g).abs() < 1e-6,
                        "{} vs {g}",
                        pair.grad
                    );
                    assert!(
                        (f64::from(pair.hess) - h).abs() < 1e-6,
                        "{} vs {h}",
                        pair.hess
                    );
                }
            }
            // Another iteration redraws the targets.
            assert_ne!(actual, gradients(iteration + 1));
        }
    }

    #[test]
    fn extreme_score_margins_keep_gradients_finite() {
        let labels = [1.0, 0.0];
        let objective = Xendcg::new(0);
        let mut actual = [GradPair::default(); 2];
        for scores in [[100.0, 0.0], [f32::MAX, -f32::MAX]] {
            objective.query_gradients(&scores, &labels, None, None, &mut actual, 0);
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
                objective.query_gradients(&scores, &labels, None, Some(&group), &mut result, 12);
                result
            })
        };
        assert_eq!(compute(1), compute(4));
    }
}
