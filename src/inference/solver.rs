//! Solvers of the kernel ridge systems `(c I + K) u = k(x)` behind the
//! Boulevard variance: exact (dense Cholesky of the `n × n` system) or
//! Nyström (a rank-`r` approximation `K ≈ L Lᵀ` from `s` uniformly sampled
//! landmark rows, then Woodbury's identity), following Fang, Tan & Hooker's
//! Appendix A: `O(n s²)` to prepare, `O(nnz(k) s + s²)` per point.
//!
//! A solver answers, for a block of kernel vectors `k_a`, the Gram matrix
//! `u_aᵀ u_b` of the solutions and their sums `1ᵀ u_a`, which is all the
//! variance, interval, and test computations need.

use rayon::prelude::*;

use super::kernel::LeafKernel;
use super::linalg::{backward_solve, cholesky_in_place, dot, forward_solve, pivoted_cholesky};
use crate::error::{HessboostError, Result};
use crate::rng::Rng;

/// Rows of the Nyström factor summed together into one partial Gram matrix
/// (fixed blocks, so the sum is independent of the thread count).
const GRAM_BLOCK: usize = 1024;

/// Relative diagonal tolerance below which a Nyström landmark is dropped as
/// a numerical combination of the others.
const LANDMARK_TOL: f64 = 1e-10;

/// The RNG salt of the Nyström landmark draw.
const NYSTROM_SALT: u64 = 0x4E59_5354;

/// How the kernel ridge systems are solved.
pub(super) enum RidgeSolver {
    /// `L Lᵀ = c I + K`, the full `n × n` factor.
    Exact { factor: Vec<f64>, n: usize },
    /// `K ≈ F Fᵀ` with `F` (`n × r`, row-major), and the Cholesky factor of
    /// `M = c I + Fᵀ F` (`r × r`), plus `Fᵀ 1`.
    Nystrom {
        f: Vec<f64>,
        n: usize,
        r: usize,
        m_factor: Vec<f64>,
        f_sums: Vec<f64>,
    },
}

/// The solutions of one block of right-hand sides.
pub(super) struct Solved {
    /// `u_aᵀ u_b`, `m × m` row-major.
    pub(super) gram: Vec<f64>,
    /// `1ᵀ u_a`.
    pub(super) sums: Vec<f64>,
}

/// A factorization that failed although `c I + K` is positive definite in
/// exact arithmetic (a non-finite kernel).
fn not_positive_definite() -> HessboostError {
    HessboostError::invalid_param(
        "solver",
        "the kernel ridge system is not numerically positive definite",
    )
}

impl RidgeSolver {
    /// Factor `c I + K` exactly.
    pub(super) fn exact(kernel: &LeafKernel, c: f64) -> Result<Self> {
        let n = kernel.n();
        let mut a = kernel.dense();
        for i in 0..n {
            a[i * n + i] += c;
        }
        cholesky_in_place(&mut a, n).map_err(|_| not_positive_definite())?;
        Ok(RidgeSolver::Exact { factor: a, n })
    }

    /// The Nyström approximation of `c I + K` from `landmarks` rows drawn
    /// uniformly without replacement (all rows when `landmarks >= n`) with
    /// `seed`; landmarks that are numerically combinations of the others
    /// are dropped (diagonally pivoted Cholesky of their kernel block).
    pub(super) fn nystrom(
        kernel: &LeafKernel,
        c: f64,
        landmarks: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = kernel.n();
        let mut rows: Vec<usize> = (0..n).collect();
        if landmarks < n {
            Rng::new(seed ^ NYSTROM_SALT).shuffle(&mut rows);
            rows.truncate(landmarks);
            rows.sort_unstable();
        }
        let s = rows.len();
        // W = K[S, S], one landmark's kernel row at a time.
        let mut w = vec![0.0; s * s];
        w.par_chunks_mut(s.max(1)).zip(&rows).for_each_init(
            || vec![0.0; n],
            |scratch, (out, &row)| {
                scratch.fill(0.0);
                kernel.add_row(row, scratch);
                for (o, &j) in out.iter_mut().zip(&rows) {
                    *o = scratch[j];
                }
            },
        );
        let pivoted = pivoted_cholesky(&w, s, LANDMARK_TOL);
        let r = pivoted.rank();
        let pivot_rows: Vec<usize> = pivoted.pivots.iter().map(|&p| rows[p]).collect();
        // F = K[:, P] L_Wᵀ⁻¹: fill column t with pivot t's kernel row, then
        // solve every row against L_W.
        let mut f = vec![0.0; n * r];
        let mut scratch = vec![0.0; n];
        for (t, &row) in pivot_rows.iter().enumerate() {
            scratch.fill(0.0);
            kernel.add_row(row, &mut scratch);
            for (i, &v) in scratch.iter().enumerate() {
                f[i * r + t] = v;
            }
        }
        let lw = &pivoted.factor;
        f.par_chunks_mut(r.max(1)).for_each(|row| {
            forward_solve(lw, r, row, 1);
        });
        // G = Fᵀ F over fixed row blocks, summed in block order.
        let partials: Vec<(Vec<f64>, Vec<f64>)> = f
            .par_chunks(GRAM_BLOCK * r.max(1))
            .map(|block| {
                let mut g = vec![0.0; r * r];
                let mut sums = vec![0.0; r];
                for row in block.chunks_exact(r.max(1)) {
                    for (a, &fa) in row.iter().enumerate() {
                        sums[a] += fa;
                        if fa != 0.0 {
                            super::linalg::axpy(fa, &row[..=a], &mut g[a * r..=a * r + a]);
                        }
                    }
                }
                (g, sums)
            })
            .collect();
        let mut m = vec![0.0; r * r];
        let mut f_sums = vec![0.0; r];
        for (g, sums) in partials {
            for (a, b) in m.iter_mut().zip(&g) {
                *a += b;
            }
            for (a, b) in f_sums.iter_mut().zip(&sums) {
                *a += b;
            }
        }
        for a in 0..r {
            m[a * r + a] += c;
        }
        cholesky_in_place(&mut m, r).map_err(|_| not_positive_definite())?;
        Ok(RidgeSolver::Nystrom {
            f,
            n,
            r,
            m_factor: m,
            f_sums,
        })
    }

    /// Solve `(c I + K) u_a = k_a` for the `m` kernel vectors `kvecs`
    /// (`m × n`, one per row) and return their Gram matrix and sums.
    pub(super) fn solve(&self, kvecs: &[f64], m: usize, c: f64) -> Solved {
        match self {
            RidgeSolver::Exact { factor, n } => {
                let n = *n;
                // Right-hand sides as the columns of an `n × m` matrix.
                let mut u = vec![0.0; n * m];
                for (a, k) in kvecs.chunks_exact(n).enumerate() {
                    for (i, &v) in k.iter().enumerate() {
                        u[i * m + a] = v;
                    }
                }
                forward_solve(factor, n, &mut u, m);
                backward_solve(factor, n, &mut u, m);
                let mut gram = vec![0.0; m * m];
                let mut sums = vec![0.0; m];
                for row in u.chunks_exact(m) {
                    for a in 0..m {
                        sums[a] += row[a];
                        for b in 0..=a {
                            gram[a * m + b] += row[a] * row[b];
                        }
                    }
                }
                symmetrize(&mut gram, m);
                Solved { gram, sums }
            }
            RidgeSolver::Nystrom {
                f,
                n,
                r,
                m_factor,
                f_sums,
            } => {
                let (n, r) = (*n, *r);
                // v_a = Fᵀ k_a, then z_a = M⁻¹ v_a.
                let mut v = vec![0.0; m * r];
                for (a, k) in kvecs.chunks_exact(n).enumerate() {
                    let va = &mut v[a * r..(a + 1) * r];
                    for (i, &ki) in k.iter().enumerate() {
                        if ki != 0.0 {
                            super::linalg::axpy(ki, &f[i * r..(i + 1) * r], va);
                        }
                    }
                }
                // Columns of an `r × m` matrix for the solves.
                let mut z = vec![0.0; r * m];
                for a in 0..m {
                    for t in 0..r {
                        z[t * m + a] = v[a * r + t];
                    }
                }
                forward_solve(m_factor, r, &mut z, m);
                backward_solve(m_factor, r, &mut z, m);
                // Back to one row per query.
                let mut zq = vec![0.0; m * r];
                for t in 0..r {
                    for a in 0..m {
                        zq[a * r + t] = z[t * m + a];
                    }
                }
                let mut gram = vec![0.0; m * m];
                let mut sums = vec![0.0; m];
                for a in 0..m {
                    let ka = &kvecs[a * n..(a + 1) * n];
                    let za = &zq[a * r..(a + 1) * r];
                    let k_sum: f64 = ka.iter().sum();
                    sums[a] = (k_sum - dot(za, f_sums)) / c;
                    for b in 0..=a {
                        let kb = &kvecs[b * n..(b + 1) * n];
                        let zb = &zq[b * r..(b + 1) * r];
                        let vz = dot(&v[a * r..(a + 1) * r], zb);
                        gram[a * m + b] = (dot(ka, kb) - vz - c * dot(za, zb)) / (c * c);
                    }
                }
                symmetrize(&mut gram, m);
                Solved { gram, sums }
            }
        }
    }
}

/// Copy the lower triangle of the `m × m` matrix `g` onto its upper one.
fn symmetrize(g: &mut [f64], m: usize) {
    for a in 0..m {
        for b in 0..a {
            g[b * m + a] = g[a * m + b];
        }
    }
}
