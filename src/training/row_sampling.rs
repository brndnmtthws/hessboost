//! Row and column sampling: uniform, class-balanced, and query-level row
//! subsets and per-tree column samplers (gradient-based sampling is in
//! `sampling`).

use crate::config::{SamplingMethod, TrainingParams};
use crate::data::{DMatrix, GroupInfo};
use crate::rng::Rng;
use crate::tree::builder::all_rows;
use crate::tree::sampler::ColumnSampler;

/// Bernoulli row subsampling (each row kept with probability `subsample`),
/// matching XGBoost's default sampling method, or with LightGBM's
/// class-balanced bagging (a positive row, label `1`, kept with
/// probability `pos_fraction`, any other with `neg_fraction`) in its
/// place. Guarantees at least one row. Gradient-based sampling keeps every
/// row here; it samples the gradients in `round::fit_output_tree` instead.
pub(super) fn sample_rows(
    n: usize,
    params: &TrainingParams,
    meta: RowMeta<'_>,
    rng: &mut Rng,
) -> Vec<u32> {
    if params.sampling_method == SamplingMethod::GradientBased {
        return all_rows(n);
    }
    let Some(bagging) = params.balanced_bagging else {
        return if params.subsample >= 1.0 {
            all_rows(n)
        } else {
            bernoulli_sample(n, params.subsample, rng)
        };
    };
    let (pos, neg) = (bagging.pos_fraction(), bagging.neg_fraction());
    let labels = &meta.labels[..n];
    let positives = meta.positives;
    let mut rows = with_sample_capacity(positives as f64 * pos + (n - positives) as f64 * neg);
    rows.extend((0..n as u32).filter(|&row| {
        let fraction = if labels[row as usize] == 1.0 {
            pos
        } else {
            neg
        };
        rng.f64() < fraction
    }));
    if rows.is_empty() {
        rows.push(rng.range(0..n) as u32);
    }
    rows
}

/// A row buffer sized for a Bernoulli sample of `expected` rows plus a few
/// standard deviations.
fn with_sample_capacity(expected: f64) -> Vec<u32> {
    Vec::with_capacity((expected + 4.0 * expected.sqrt() + 16.0) as usize)
}

/// The indices in `0..n` kept by one `rng` draw each with probability
/// `fraction`, or one random index when none is kept.
fn bernoulli_sample(n: usize, fraction: f64, rng: &mut Rng) -> Vec<u32> {
    let mut kept = with_sample_capacity(n as f64 * fraction);
    kept.extend((0..n as u32).filter(|_| rng.f64() < fraction));
    if kept.is_empty() {
        kept.push(rng.range(0..n) as u32);
    }
    kept
}

/// The training metadata row sampling reads: the labels and their count
/// of positives (class-balanced bagging) and the query groups (query
/// bagging). Built once per training run, so no draw rescans the labels.
#[derive(Clone, Copy)]
pub(super) struct RowMeta<'a> {
    labels: &'a [f32],
    /// Rows labelled `1`, counted only under class-balanced bagging (it
    /// sizes each draw's row buffer).
    positives: usize,
    group: Option<&'a GroupInfo>,
}

impl<'a> RowMeta<'a> {
    /// `data`'s labels (empty without any), their positives when `params`
    /// bags by class, and its query groups.
    pub(super) fn of(data: &'a DMatrix, params: &TrainingParams) -> Self {
        let labels = data.labels().unwrap_or_default();
        let positives = if params.balanced_bagging.is_some() {
            labels.iter().filter(|&&label| label == 1.0).count()
        } else {
            0
        };
        RowMeta {
            labels,
            positives,
            group: data.group(),
        }
    }
}

/// One iteration's row subsets (uniform, class-balanced, or by query),
/// drawn before its trees: one per parallel tree, or a single subset for
/// the whole forest when `per_forest` (`approx`,
/// [`Prepared::samples_per_forest`](super::prepare::Prepared::samples_per_forest)) or when there is no row sampling
/// (every tree then reads all rows, and [`sample_rows`] draws nothing).
/// Parallel tree `p` uses [`RoundRows::rows`]`(p)`, shared across its
/// per-output fits. With query bagging each subset is the rows of the
/// query groups (the whole matrix without any) a draw keeps. `all` is
/// every training row, ascending, which an unsampled round borrows.
pub(super) fn iteration_row_subsets<'a>(
    params: &TrainingParams,
    per_forest: bool,
    meta: RowMeta<'_>,
    all: &'a [u32],
    rng: &mut Rng,
) -> RoundRows<'a> {
    let n = all.len();
    let samples = params.sampling_method == SamplingMethod::Uniform
        && (params.subsample < 1.0
            || params.balanced_bagging.is_some()
            || params.bagging_by_query.is_some());
    if !samples {
        // `sample_rows` would return every row without a draw (query
        // bagging always samples: it requires uniform sampling).
        return RoundRows::All(all);
    }
    let draws = if per_forest {
        1
    } else {
        params.num_parallel_tree
    };
    let Some(bagging) = params.bagging_by_query else {
        return RoundRows::Sampled(
            (0..draws)
                .map(|_| sample_rows(n, params, meta, rng))
                .collect(),
        );
    };
    let queries: Vec<(usize, usize)> = meta
        .group
        .map_or_else(|| vec![(0, n)], |group| group.iter_ranges().collect());
    RoundRows::Sampled(
        (0..draws)
            .map(|_| {
                bernoulli_sample(queries.len(), bagging.fraction(), rng)
                    .into_iter()
                    .flat_map(|query| {
                        let (start, end) = queries[query as usize];
                        start as u32..end as u32
                    })
                    .collect()
            })
            .collect(),
    )
}

/// One iteration's row subsets ([`iteration_row_subsets`]).
#[derive(Debug, PartialEq)]
pub(super) enum RoundRows<'a> {
    /// No row sampling: every tree reads every row (ascending).
    All(&'a [u32]),
    /// The drawn subsets, one per draw.
    Sampled(Vec<Vec<u32>>),
}

impl RoundRows<'_> {
    /// The rows parallel tree `parallel` grows on.
    pub(super) fn rows(&self, parallel: usize) -> &[u32] {
        match self {
            RoundRows::All(all) => all,
            RoundRows::Sampled(subsets) => &subsets[parallel % subsets.len()],
        }
    }
}

/// Whether trees are grown on gradient-based (MVS) row samples.
pub(super) fn gradient_sampling(params: &TrainingParams) -> bool {
    params.sampling_method == SamplingMethod::GradientBased && params.subsample < 1.0
}

/// Build one tree's column sampler over `dtrain`'s features, seeded from
/// `rng`: the `colsample_bytree` pool, then the `bylevel`/`bynode` draws,
/// weighted by the training matrix's feature weights when it has them.
pub(super) fn make_column_sampler(
    dtrain: &DMatrix,
    params: &TrainingParams,
    rng: &mut Rng,
) -> ColumnSampler {
    ColumnSampler::new(
        dtrain.n_cols(),
        dtrain.feature_weights(),
        params.colsample_bytree,
        params.colsample_bylevel,
        params.colsample_bynode,
        rng.next_u64(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::{LambdaRank, Objective, RegLoss};

    /// Query bagging keeps or drops each query whole, and its draw does not
    /// depend on the thread count.
    #[test]
    fn query_subsets_keep_whole_groups_at_any_thread_count() {
        let group_sizes = vec![3, 7, 2, 8, 4, 5, 6];
        let group = GroupInfo::from_sizes(&group_sizes);
        let n: usize = group_sizes.iter().sum();
        let all = all_rows(n);
        let params = TrainingParams::builder()
            .objective(Objective::RankNdcg(LambdaRank::default()))
            .bagging_by_query(crate::config::QueryBagging::new(0.5).unwrap())
            .seed(17)
            .build()
            .unwrap();
        let select = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                iteration_row_subsets(
                    &params,
                    false,
                    RowMeta {
                        labels: &[],
                        positives: 0,
                        group: Some(&group),
                    },
                    &all,
                    &mut Rng::new(41),
                )
            })
        };
        let one = select(1);
        assert_eq!(one, select(4));
        let selected = one.rows(0);
        assert!(!selected.is_empty() && selected.len() < n);
        for (query, (start, end)) in group.iter_ranges().enumerate() {
            let included = selected
                .iter()
                .filter(|&&row| (start..end).contains(&(row as usize)))
                .count();
            assert!(
                included == 0 || included == end - start,
                "query {query} split"
            );
        }
    }

    #[test]
    fn balanced_row_sampler_respects_class_fractions() {
        let labels: Vec<f32> = (0..20_000)
            .map(|row| if row < 4_000 { 1.0 } else { 0.0 })
            .collect();
        let params = TrainingParams::builder()
            .objective(Objective::BinaryLogistic(RegLoss::default()))
            .balanced_bagging(crate::config::BalancedBagging::new(0.6, 0.1).unwrap())
            .seed(53)
            .build()
            .unwrap();
        let meta = |labels| RowMeta {
            labels,
            positives: labels.iter().filter(|&&label| label == 1.0).count(),
            group: None,
        };
        let selected = sample_rows(labels.len(), &params, meta(&labels), &mut Rng::new(77));
        let positives = selected
            .iter()
            .filter(|&&row| labels[row as usize] == 1.0)
            .count();
        let negatives = selected.len() - positives;
        let pos_rate = positives as f64 / 4_000.0;
        let neg_rate = negatives as f64 / 16_000.0;
        assert!((0.57..0.63).contains(&pos_rate), "positive rate {pos_rate}");
        assert!(
            (0.092..0.108).contains(&neg_rate),
            "negative rate {neg_rate}"
        );
        // Without positives the negative fraction still applies.
        let negatives_only = vec![0.0f32; 20_000];
        let kept = sample_rows(20_000, &params, meta(&negatives_only), &mut Rng::new(77)).len();
        assert!(
            (1_840..2_160).contains(&kept),
            "kept {kept} of 20000 negatives"
        );
    }
}
