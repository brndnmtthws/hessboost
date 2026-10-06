//! FAST pair ranking: the interaction terms worth boosting.

use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::objective::GradPair;
use crate::training::prepare::{Prepared, TrainContext};
use rayon::prelude::*;

/// FAST (Lou, Caruana, Gehrke & Hooker, KDD 2013): rank every feature pair
/// by its best four-quadrant split of `gpair` on the features' histogram
/// bins, `Σ_q G_q² / (H_q + λ) − G² / (H + λ)` over the rows present in
/// both, and return the top `k` (ties broken by feature order). A
/// categorical feature has one bin per category, ordered by the category's
/// mean gradient `G / (H + λ)` (the order in which a binary partition's
/// best split is a cut, as in LightGBM's and XGBoost's categorical search).
/// The bins are those of `tree_method = hist`'s index (`prepared`), whose
/// cuts are `max_bin`'s; other tree methods get the same index built here.
pub(super) fn fast_pairs(
    run: &TrainContext,
    prepared: &Prepared,
    gpair: &[GradPair],
    k: usize,
) -> Vec<Vec<u32>> {
    let TrainContext { params, dtrain, .. } = *run;
    let built;
    let index = if let Prepared::Hist { index, .. } = prepared {
        index
    } else {
        built = GHistIndex::from_dmatrix(dtrain, HistCuts::from_dmatrix(dtrain, params.max_bin));
        &built
    };
    let cuts = index.cuts();
    let lambda = params.lambda;
    let bins: Vec<Vec<u32>> = index
        .feature_columns()
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
    let p = bins.len();
    let pairs: Vec<(usize, usize)> = (0..p)
        .flat_map(|a| (a + 1..p).map(move |b| (a, b)))
        .collect();
    // One grid per worker, reused across its pairs.
    let mut scored: Vec<(f64, usize)> = pairs
        .par_iter()
        .enumerate()
        .map_init(Vec::new, |grid, (i, &(a, b))| {
            let gain = pair_gain(
                &PairBins {
                    a: &bins[a],
                    b: &bins[b],
                    ma: cuts.num_bins(a),
                    mb: cuts.num_bins(b),
                },
                gpair,
                lambda,
                grid,
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

/// Rows of a pair's grid whose prefix sums [`prefix_sums`] runs together.
const PREFIX_BAND: usize = 4;

/// FAST's score of one pair. `grid`, a worker's buffer reused across
/// pairs, gets `ma + 1` rows of `mb + 1` cells of `[gradient, Hessian]`
/// sums: the pair's bin histogram below a zero row and right of a zero
/// column, then its 2D prefix sums.
fn pair_gain(bins: &PairBins, gpair: &[GradPair], lambda: f64, grid: &mut Vec<[f64; 2]>) -> f64 {
    let PairBins {
        a: bins_a,
        b: bins_b,
        ma,
        mb,
    } = *bins;
    let s = mb + 1;
    grid.clear();
    grid.resize((ma + 1) * s, [0.0; 2]);
    for ((&a, &b), gp) in bins_a.iter().zip(bins_b).zip(gpair) {
        if a == u32::MAX || b == u32::MAX {
            continue;
        }
        let cell = &mut grid[(a as usize + 1) * s + b as usize + 1];
        cell[0] += f64::from(gp.grad);
        cell[1] += f64::from(gp.hess);
    }
    prefix_sums(grid, s);
    best_cut(grid, ma, mb, lambda)
}

/// Turn the histogram in `grid` (rows of `s` cells, the first row and
/// column zero) into its 2D prefix sums in place,
/// `P[i][j] = c[i][j] + ((P[i - 1][j] + P[i][j - 1]) - P[i - 1][j - 1])`.
/// Along a row that recurrence is one chain of dependent additions, so
/// [`PREFIX_BAND`] rows run at once, each a column behind the row above.
fn prefix_sums(grid: &mut [[f64; 2]], s: usize) {
    let rows = grid.len() / s;
    let mut i = 1;
    while i < rows {
        let (above, below) = grid.split_at_mut(i * s);
        let prev = &above[(i - 1) * s..];
        if i + PREFIX_BAND <= rows {
            prefix_band::<PREFIX_BAND>(prev, &mut below[..PREFIX_BAND * s], s);
            i += PREFIX_BAND;
        } else {
            prefix_band::<1>(prev, &mut below[..s], s);
            i += 1;
        }
    }
}

/// [`prefix_sums`] of the `R` rows of `band` below the finished row
/// `prev`: step `t` computes column `t - r` of band row `r`, whose
/// neighbors above were computed by earlier steps.
#[inline(always)]
fn prefix_band<const R: usize>(prev: &[[f64; 2]], band: &mut [[f64; 2]], s: usize) {
    // Each row's left neighbor, starting at its zero column.
    let mut lefts = [[0.0f64; 2]; R];
    for t in 1..s - 1 + R {
        for (r, left) in lefts.iter_mut().enumerate() {
            if t <= r || t - r >= s {
                continue;
            }
            let j = t - r;
            let (up, diag) = if r == 0 {
                (prev[j], prev[j - 1])
            } else {
                (band[(r - 1) * s + j], band[(r - 1) * s + j - 1])
            };
            let cell = &mut band[r * s + j];
            *cell = [
                cell[0] + ((up[0] + left[0]) - diag[0]),
                cell[1] + ((up[1] + left[1]) - diag[1]),
            ];
            *left = *cell;
        }
    }
}

/// The best four-quadrant score over the prefix sums in `grid`, less the
/// unsplit score: cut `(i, j)` splits the first feature's bins below `i`
/// from the rest and the second's below `j`. A feature with one bin has
/// no cut.
fn best_cut(grid: &[[f64; 2]], ma: usize, mb: usize, lambda: f64) -> f64 {
    let score = |gs: f64, hs: f64| {
        if hs + lambda > 0.0 {
            gs * gs / (hs + lambda)
        } else {
            0.0
        }
    };
    let s = mb + 1;
    let last = &grid[ma * s..];
    let [gt, ht] = last[mb];
    let root = score(gt, ht);
    let mut best = root;
    if ma > 1 && mb > 1 {
        for row in grid[s..ma * s].chunks_exact(s) {
            let [g0, h0] = row[mb];
            let (gr, hr) = (gt - g0, ht - h0);
            for (&[g00, h00], &[g1, h1]) in row[1..mb].iter().zip(&last[1..mb]) {
                let quadrants = score(g00, h00)
                    + score(g0 - g00, h0 - h00)
                    + score(g1 - g00, h1 - h00)
                    + score(gr - g1 + g00, hr - h1 + h00);
                best = best.max(quadrants);
            }
        }
    }
    best - root
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FAST's score of one pair from its definition: each cut's quadrant
    /// sums added up over the rows.
    fn reference_gain(bins: &PairBins, gpair: &[GradPair], lambda: f64) -> f64 {
        let score = |g: f64, h: f64| {
            if h + lambda > 0.0 {
                g * g / (h + lambda)
            } else {
                0.0
            }
        };
        let rows: Vec<(usize, usize, [f64; 2])> = bins
            .a
            .iter()
            .zip(bins.b)
            .zip(gpair)
            .filter(|&((&a, &b), _)| a != u32::MAX && b != u32::MAX)
            .map(|((&a, &b), gp)| {
                let stats = [f64::from(gp.grad), f64::from(gp.hess)];
                (a as usize, b as usize, stats)
            })
            .collect();
        let (g, h) = rows
            .iter()
            .fold((0.0, 0.0), |(g, h), &(_, _, [dg, dh])| (g + dg, h + dh));
        let root = score(g, h);
        let mut best = root;
        for i in 1..bins.ma {
            for j in 1..bins.mb {
                let mut quadrants = [[0.0f64; 2]; 4];
                for &(a, b, [dg, dh]) in &rows {
                    let q = &mut quadrants[2 * usize::from(a >= i) + usize::from(b >= j)];
                    q[0] += dg;
                    q[1] += dh;
                }
                best = best.max(quadrants.iter().map(|&[g, h]| score(g, h)).sum());
            }
        }
        best - root
    }

    /// Integer gradients keep every sum exact, so the prefix-sum scan and
    /// the definition agree bit for bit, over grids whose row counts leave
    /// every remainder of [`PREFIX_BAND`], with missing rows, zero
    /// Hessians, and one-bin features.
    #[test]
    fn pair_gain_matches_the_definition() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        let mut grid = Vec::new();
        for (ma, mb) in [
            (1, 1),
            (1, 4),
            (4, 1),
            (2, 2),
            (3, 5),
            (5, 3),
            (6, 9),
            (8, 2),
            (13, 7),
        ] {
            for lambda in [0.0, 1.0] {
                let n = 300;
                let mut column = |m: usize| -> Vec<u32> {
                    (0..n)
                        .map(|_| {
                            if next(5) == 0 {
                                u32::MAX
                            } else {
                                next(m as u64) as u32
                            }
                        })
                        .collect()
                };
                let (a, b) = (column(ma), column(mb));
                let gpair: Vec<GradPair> = (0..n)
                    .map(|_| GradPair::new(next(7) as f32 - 3.0, next(3) as f32))
                    .collect();
                let bins = PairBins {
                    a: &a,
                    b: &b,
                    ma,
                    mb,
                };
                let gain = pair_gain(&bins, &gpair, lambda, &mut grid);
                let want = reference_gain(&bins, &gpair, lambda);
                assert_eq!(
                    gain.to_bits(),
                    want.to_bits(),
                    "{ma} x {mb}, lambda {lambda}"
                );
            }
        }
    }
}
