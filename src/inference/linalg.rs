//! Dense `f64` linear algebra for the kernel ridge systems of
//! [`crate::inference`]: a blocked Cholesky factorization, triangular solves
//! with many right-hand sides, and a diagonally pivoted Cholesky that finds
//! the numerical rank of a positive semidefinite matrix.
//!
//! Matrices are row-major. Every reduction sums in a fixed order (dot
//! products in four interleaved lanes combined the same way every time), and
//! the parallel loops split work by rows of the data, never by thread, so
//! results are bit-identical at any thread count.

use rayon::prelude::*;

/// Diagonal block size of [`cholesky_in_place`]: the panel of a block
/// (`n × BLOCK` values) stays in cache while the trailing rows read it.
const BLOCK: usize = 64;

/// `Σ a_i b_i` over the common length, in four interleaved lanes.
pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut lanes = [0.0f64; 4];
    let (chunks_a, rest_a) = a.as_chunks::<4>();
    let (chunks_b, rest_b) = b.as_chunks::<4>();
    for (x, y) in chunks_a.iter().zip(chunks_b) {
        for l in 0..4 {
            lanes[l] += x[l] * y[l];
        }
    }
    let mut tail = 0.0;
    for (x, y) in rest_a.iter().zip(rest_b) {
        tail += x * y;
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + tail
}

/// `y += alpha * x`.
pub(super) fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// Overwrite the lower triangle of the symmetric positive definite `n × n`
/// matrix `a` with its Cholesky factor `L` (`a = L Lᵀ`); the strict upper
/// triangle is left as it was and never read. Right-looking and blocked:
/// each diagonal block is factored, the rows below it are solved against it
/// (in parallel), and the trailing lower triangle is updated from that
/// panel (in parallel over rows).
///
/// Returns the first row whose pivot is not positive (the matrix is not
/// numerically positive definite) as the error.
pub(super) fn cholesky_in_place(a: &mut [f64], n: usize) -> Result<(), usize> {
    debug_assert_eq!(a.len(), n * n);
    let mut diag_block = vec![0.0; BLOCK * BLOCK];
    let mut panel: Vec<f64> = Vec::new();
    for kb in (0..n).step_by(BLOCK) {
        let ke = (kb + BLOCK).min(n);
        let nb = ke - kb;
        // 1. The diagonal block, unblocked.
        for j in kb..ke {
            let row_j = &mut a[j * n..(j + 1) * n];
            let s = row_j[j] - dot(&row_j[kb..j], &row_j[kb..j]);
            if !(s > 0.0 && s.is_finite()) {
                return Err(j);
            }
            let d = s.sqrt();
            row_j[j] = d;
            for i in j + 1..ke {
                let (upper, lower) = a.split_at_mut(i * n);
                let row_j = &upper[j * n..j * n + n];
                let row_i = &mut lower[..n];
                row_i[j] = (row_i[j] - dot(&row_i[kb..j], &row_j[kb..j])) / d;
            }
        }
        if ke == n {
            break;
        }
        for (r, i) in (kb..ke).enumerate() {
            diag_block[r * BLOCK..r * BLOCK + nb].copy_from_slice(&a[i * n + kb..i * n + ke]);
        }
        let (_, below) = a.split_at_mut(ke * n);
        // 2. The panel: rows below the block solved against its factor.
        below.par_chunks_mut(n).with_min_len(16).for_each(|row| {
            for (c, j) in (kb..ke).enumerate() {
                let lj = &diag_block[c * BLOCK..c * BLOCK + c];
                row[j] = (row[j] - dot(&row[kb..j], lj)) / diag_block[c * BLOCK + c];
            }
        });
        panel.clear();
        for row in below.chunks(n) {
            panel.extend_from_slice(&row[kb..ke]);
        }
        // 3. The trailing lower triangle, minus the panel's outer product.
        below
            .par_chunks_mut(n)
            .with_min_len(8)
            .enumerate()
            .for_each(|(r, row)| {
                let pi = &panel[r * nb..(r + 1) * nb];
                for (c, v) in row[ke..=ke + r].iter_mut().enumerate() {
                    *v -= dot(pi, &panel[c * nb..(c + 1) * nb]);
                }
            });
    }
    Ok(())
}

/// Solve `L Y = B` in place for the `m` columns of `b` (`n × m`,
/// row-major), `L` the lower triangle of the `n × n` factor `l`.
pub(super) fn forward_solve(l: &[f64], n: usize, b: &mut [f64], m: usize) {
    for i in 0..n {
        let (done, rest) = b.split_at_mut(i * m);
        let row = &mut rest[..m];
        let li = &l[i * n..i * n + i];
        for (k, &lik) in li.iter().enumerate() {
            if lik != 0.0 {
                axpy(-lik, &done[k * m..(k + 1) * m], row);
            }
        }
        let d = l[i * n + i];
        for v in row.iter_mut() {
            *v /= d;
        }
    }
}

/// Solve `Lᵀ X = Y` in place for the `m` columns of `y` (`n × m`,
/// row-major), reading `L` by rows.
pub(super) fn backward_solve(l: &[f64], n: usize, y: &mut [f64], m: usize) {
    for i in (0..n).rev() {
        let (top, rest) = y.split_at_mut(i * m);
        let row = &mut rest[..m];
        let d = l[i * n + i];
        for v in row.iter_mut() {
            *v /= d;
        }
        let li = &l[i * n..i * n + i];
        for (k, &lik) in li.iter().enumerate() {
            if lik != 0.0 {
                axpy(-lik, row, &mut top[k * m..(k + 1) * m]);
            }
        }
    }
}

/// A diagonally pivoted Cholesky factorization of a positive semidefinite
/// matrix `A`, stopped at its numerical rank `r`: `A[p, p] = L Lᵀ` for the
/// pivots `p` (`r` distinct indices, in pivot order) and the `r × r` lower
/// factor `L` (row-major).
pub(super) struct PivotedCholesky {
    pub(super) pivots: Vec<usize>,
    pub(super) factor: Vec<f64>,
}

impl PivotedCholesky {
    /// The numerical rank.
    pub(super) fn rank(&self) -> usize {
        self.pivots.len()
    }
}

/// Factor the symmetric positive semidefinite `n × n` matrix `a`, pivoting
/// on the largest remaining diagonal (the lowest index among ties) and
/// stopping once it falls to `rel_tol` times the largest diagonal of `a`:
/// the dropped indices are, to that tolerance, linear combinations of the
/// pivots. Non-finite or non-positive diagonals are never pivots.
pub(super) fn pivoted_cholesky(a: &[f64], n: usize, rel_tol: f64) -> PivotedCholesky {
    let mut residual: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    let largest = residual
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(0.0f64, f64::max);
    let tol = rel_tol * largest;
    let mut perm: Vec<usize> = (0..n).collect();
    // Row `j` holds the factor row of the index at pivot position `j`.
    let mut l = vec![0.0; n * n];
    let mut rank = 0;
    for k in 0..n {
        let mut best = None;
        for (j, &q) in perm.iter().enumerate().skip(k) {
            let d = residual[q];
            if d.is_finite() && d > tol && best.is_none_or(|(_, bd)| d > bd) {
                best = Some((j, d));
            }
        }
        let Some((jb, d)) = best else {
            break;
        };
        perm.swap(k, jb);
        for c in 0..k {
            l.swap(k * n + c, jb * n + c);
        }
        let p = perm[k];
        let lkk = d.sqrt();
        l[k * n + k] = lkk;
        let (head, tail) = l.split_at_mut((k + 1) * n);
        let lk = &head[k * n..k * n + k];
        let rest = &perm[k + 1..];
        tail.par_chunks_mut(n)
            .with_min_len(64)
            .zip(rest)
            .for_each(|(row, &q)| {
                row[k] = (a[q * n + p] - dot(&row[..k], lk)) / lkk;
            });
        for (row, &q) in tail.chunks(n).zip(rest) {
            residual[q] -= row[k] * row[k];
        }
        rank = k + 1;
    }
    let mut factor = vec![0.0; rank * rank];
    for i in 0..rank {
        factor[i * rank..=i * rank + i].copy_from_slice(&l[i * n..=i * n + i]);
    }
    perm.truncate(rank);
    PivotedCholesky {
        pivots: perm,
        factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A random symmetric positive definite matrix `B Bᵀ + n I`.
    fn spd(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = crate::rng::Rng::new(seed);
        let b: Vec<f64> = (0..n * n).map(|_| rng.f64() - 0.5).collect();
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = dot(&b[i * n..(i + 1) * n], &b[j * n..(j + 1) * n]);
            }
            a[i * n + i] += n as f64;
        }
        a
    }

    #[test]
    fn blocked_cholesky_solves_across_block_boundaries() {
        // Sizes below, at, and past the block size exercise every branch.
        for n in [1, 5, BLOCK, BLOCK + 3, 2 * BLOCK + 17] {
            let a = spd(n, n as u64);
            let mut l = a.clone();
            cholesky_in_place(&mut l, n).unwrap();
            let x: Vec<f64> = (0..2 * n).map(|i| (i as f64).sin()).collect();
            // b = A x for two right-hand sides.
            let mut b = vec![0.0; 2 * n];
            for i in 0..n {
                for c in 0..2 {
                    b[i * 2 + c] = (0..n).map(|j| a[i * n + j] * x[j * 2 + c]).sum();
                }
            }
            forward_solve(&l, n, &mut b, 2);
            backward_solve(&l, n, &mut b, 2);
            for (got, want) in b.iter().zip(&x) {
                assert!((got - want).abs() < 1e-9, "n={n}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn pivoted_cholesky_finds_the_rank_of_duplicated_columns() {
        // Rows 0 and 2 of the Gram matrix of vectors (v, w, v) coincide.
        let v = [1.0, 2.0, 0.5];
        let w = [0.0, 1.0, -1.0];
        let rows = [v, w, v];
        let mut a = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                a[i * 3 + j] = dot(&rows[i], &rows[j]);
            }
        }
        let p = pivoted_cholesky(&a, 3, 1e-12);
        assert_eq!(p.rank(), 2);
        assert!(p.pivots.contains(&1));
    }
}
