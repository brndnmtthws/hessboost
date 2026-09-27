use hessboost::prelude::DMatrix;

use super::labeled_dense;

fn features(i: usize) -> [f32; 4] {
    [0, 1, 2, 3].map(|j| ((i * (7 + 3 * j) + 11 * j) % 97) as f32 / 97.0)
}

/// Four-feature deterministic smooth regression data used for continuation and hook tests.
pub fn regression(n: usize, shift: f32) -> DMatrix {
    let mut x = Vec::with_capacity(n * 4);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let f = features(i);
        y.push(2.0 * f[0] - 3.0 * f[1] * f[1] + 0.5 * f[2] + shift);
        x.extend(f);
    }
    labeled_dense(&x, 4, &y)
}

/// Smooth regression labels with the continuation test's original noise values.
pub fn continuation_noisy(n: usize, salt: usize) -> DMatrix {
    let mut x = Vec::with_capacity(n * 4);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let f = features(i);
        let label = 2.0 * f[0] - 3.0 * f[1] * f[1] + 0.5 * f[2];
        y.push(label + ((i * (31 + salt)) % 23) as f32 / 23.0 - 0.5);
        x.extend(f);
    }
    labeled_dense(&x, 4, &y)
}

/// Four-feature smooth regression data with the round hook test's deterministic noise.
pub fn noisy(n: usize, salt: usize) -> DMatrix {
    let mut x = Vec::with_capacity(n * 4);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let f = features(i);
        let noise = (((i + salt) * 2_654_435_761) % 1000) as f32 / 1000.0 - 0.5;
        y.push(2.0 * f[0] - 3.0 * f[1] * f[1] + 0.5 * f[2] + noise);
        x.extend(f);
    }
    labeled_dense(&x, 4, &y)
}
