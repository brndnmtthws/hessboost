//! FAST pair ranking: the interaction terms worth boosting.

use crate::data::quantile::HistCuts;
use crate::objective::GradPair;
use crate::training::prepare::TrainContext;
use rayon::prelude::*;

/// FAST (Lou, Caruana, Gehrke & Hooker, KDD 2013): rank every feature pair
/// by its best four-quadrant split of `gpair` on the features' histogram
/// bins, `Σ_q G_q² / (H_q + λ) − G² / (H + λ)` over the rows present in
/// both, and return the top `k` (ties broken by feature order). A
/// categorical feature has one bin per category, ordered by the category's
/// mean gradient `G / (H + λ)` (the order in which a binary partition's
/// best split is a cut, as in LightGBM's and XGBoost's categorical search).
pub(super) fn fast_pairs(run: &TrainContext, gpair: &[GradPair], k: usize) -> Vec<Vec<u32>> {
    let TrainContext { params, dtrain, .. } = *run;
    let p = dtrain.n_cols();
    let cuts = HistCuts::from_dmatrix(dtrain, params.max_bin);
    let bins: Vec<Vec<u32>> = (0..p)
        .into_par_iter()
        .map(|f| {
            let start = cuts.feature_bins(f).0 as u32;
            (0..dtrain.n_rows())
                .map(|row| match dtrain.get(row, f) {
                    Some(v) if !v.is_nan() => cuts.bin_of(f, v) - start,
                    _ => u32::MAX,
                })
                .collect()
        })
        .collect();
    let lambda = params.lambda;
    let bins: Vec<Vec<u32>> = bins
        .into_iter()
        .enumerate()
        .map(|(f, b)| {
            if cuts.is_categorical(f) {
                order_categories(b, cuts.num_bins(f), gpair, lambda)
            } else {
                b
            }
        })
        .collect();
    let pairs: Vec<(usize, usize)> = (0..p)
        .flat_map(|a| (a + 1..p).map(move |b| (a, b)))
        .collect();
    // One pair of prefix grids per worker, reused across its pairs.
    let mut scored: Vec<(f64, usize)> = pairs
        .par_iter()
        .enumerate()
        .map_init(PairGrids::default, |grids, (i, &(a, b))| {
            let gain = pair_gain(
                &PairBins {
                    a: &bins[a],
                    b: &bins[b],
                    ma: cuts.num_bins(a),
                    mb: cuts.num_bins(b),
                },
                gpair,
                lambda,
                grids,
            );
            (gain, i)
        })
        .collect();
    scored.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    scored
        .into_iter()
        .take(k)
        .map(|(_, i)| vec![pairs[i].0 as u32, pairs[i].1 as u32])
        .collect()
}

/// Renumber a categorical feature's bins by their mean gradient
/// `G / (H + λ)` (ties by bin), so FAST's cuts over them are the binary
/// partitions worth scoring.
fn order_categories(bins: Vec<u32>, m: usize, gpair: &[GradPair], lambda: f64) -> Vec<u32> {
    let mut g = vec![0.0f64; m];
    let mut h = vec![0.0f64; m];
    for (&b, gp) in bins.iter().zip(gpair) {
        if b != u32::MAX {
            g[b as usize] += f64::from(gp.grad);
            h[b as usize] += f64::from(gp.hess);
        }
    }
    let ratio = |b: usize| {
        if h[b] + lambda > 0.0 {
            g[b] / (h[b] + lambda)
        } else {
            0.0
        }
    };
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| ratio(a).total_cmp(&ratio(b)).then(a.cmp(&b)));
    let mut rank = vec![0u32; m];
    for (r, &b) in order.iter().enumerate() {
        rank[b] = r as u32;
    }
    bins.into_iter()
        .map(|b| if b == u32::MAX { b } else { rank[b as usize] })
        .collect()
}

/// One pair's features: each row's bin (`u32::MAX` missing) and the bin
/// counts.
struct PairBins<'a> {
    a: &'a [u32],
    b: &'a [u32],
    ma: usize,
    mb: usize,
}

/// The gradient and Hessian prefix grids [`pair_gain`] fills, kept by a
/// worker across pairs (resized, and zeroed, per pair).
#[derive(Default)]
struct PairGrids {
    g: Vec<f64>,
    h: Vec<f64>,
}

/// FAST's score of one pair, computed in `grids`.
fn pair_gain(bins: &PairBins, gpair: &[GradPair], lambda: f64, grids: &mut PairGrids) -> f64 {
    let PairBins {
        a: bins_a,
        b: bins_b,
        ma,
        mb,
    } = *bins;
    // Prefix sums over the `ma × mb` bin histogram, `(ma + 1) × (mb + 1)`.
    let s = mb + 1;
    let PairGrids { g, h } = grids;
    for grid in [&mut *g, &mut *h] {
        grid.clear();
        grid.resize((ma + 1) * s, 0.0);
    }
    for ((&a, &b), gp) in bins_a.iter().zip(bins_b).zip(gpair) {
        if a == u32::MAX || b == u32::MAX {
            continue;
        }
        let cell = (a as usize + 1) * s + b as usize + 1;
        g[cell] += f64::from(gp.grad);
        h[cell] += f64::from(gp.hess);
    }
    for arr in [&mut *g, &mut *h] {
        for i in 1..=ma {
            for j in 1..=mb {
                arr[i * s + j] +=
                    arr[(i - 1) * s + j] + arr[i * s + j - 1] - arr[(i - 1) * s + j - 1];
            }
        }
    }
    let score = |gs: f64, hs: f64| {
        if hs + lambda > 0.0 {
            gs * gs / (hs + lambda)
        } else {
            0.0
        }
    };
    let (gt, ht) = (g[ma * s + mb], h[ma * s + mb]);
    let mut best = score(gt, ht);
    for i in 1..ma {
        for j in 1..mb {
            let (g00, h00) = (g[i * s + j], h[i * s + j]);
            let (g0, h0) = (g[i * s + mb], h[i * s + mb]);
            let (g1, h1) = (g[ma * s + j], h[ma * s + j]);
            let quadrants = score(g00, h00)
                + score(g0 - g00, h0 - h00)
                + score(g1 - g00, h1 - h00)
                + score(gt - g0 - g1 + g00, ht - h0 - h1 + h00);
            best = best.max(quadrants);
        }
    }
    best - score(gt, ht)
}
