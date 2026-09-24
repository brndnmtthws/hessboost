use super::*;

fn assert_grad_pairs_close(actual: &[GradPair], expected: &[GradPair]) {
    for (actual, expected) in actual.iter().zip(expected) {
        for (name, actual, expected) in [
            ("gradient", actual.grad, expected.grad),
            ("Hessian", actual.hess, expected.hess),
        ] {
            let tolerance = 1.0e-6 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "SIMD {name} {actual} differs from scalar {expected}"
            );
        }
    }
}

/// The vector exponential matches `f32::exp` within 7e-7 relative error and
/// reproduces NaN, infinities, and underflow to zero exactly.
fn assert_exp_close(actual: &[f32], expected: &[f32]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else if expected.is_infinite() || expected == 0.0 {
            assert_eq!(actual, expected);
        } else {
            let relative = ((actual - expected) / expected).abs();
            assert!(
                relative <= 7.0e-7,
                "SIMD exp {actual} differs from scalar {expected} by {relative}"
            );
        }
    }
}

/// `(loss, weight)` sums agree within a relative `tolerance`.
fn assert_sums_close(actual: (f64, f64), expected: (f64, f64), tolerance: f64) {
    assert!((actual.0 - expected.0).abs() <= expected.0.abs() * tolerance);
    assert!((actual.1 - expected.1).abs() <= expected.1.abs() * tolerance);
}

/// `start + (i % period) · step` for `i` in `0..len`: the periodic test
/// inputs (`period == len` gives a linear ramp).
fn sawtooth(len: usize, period: usize, start: f32, step: f32) -> Vec<f32> {
    (0..len)
        .map(|i| start + (i % period) as f32 * step)
        .collect()
}

/// Scalar `softmax_scalar` over every `num_class` row of `values`.
fn scalar_softmax_rows(values: &[f32], num_class: usize) -> Vec<f32> {
    let mut out = values.to_vec();
    for row in out.chunks_mut(num_class) {
        softmax_scalar(row);
    }
    out
}

/// [`softmax_gradient_rows_scalar`] over every row of a complete matrix.
fn scalar_softmax_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> Vec<GradPair> {
    let mut out = vec![GradPair::default(); preds.len()];
    softmax_gradient_rows_scalar(
        preds,
        labels,
        weights,
        1e-16,
        &mut out,
        0..labels.len(),
        num_class,
    );
    out
}

#[test]
fn sigmoid_dispatch_is_close_to_scalar() {
    let mut actual = sawtooth(4_103, 4_103, -20.0, 40.0 / 4_102.0);
    let expected: Vec<f32> = actual.iter().map(|&value| sigmoid_scalar(value)).collect();
    sigmoid_inplace(&mut actual);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 3.0e-7,
            "SIMD sigmoid {actual} differs from scalar {expected}"
        );
    }
}

#[test]
fn exp_dispatch_is_close_to_scalar_and_preserves_special_values() {
    let mut actual = sawtooth(8_195, 8_195, -80.0, 160.0 / 8_192.0);
    actual.extend([f32::NEG_INFINITY, f32::INFINITY, f32::NAN]);
    let expected: Vec<f32> = actual.iter().map(|value| value.exp()).collect();
    exp_inplace(&mut actual);
    assert_exp_close(&actual, &expected);
}

#[test]
fn exp_dispatch_handles_special_values_inside_vector_blocks() {
    // With a length divisible by every vector width there is no scalar tail,
    // so these special values exercise the in-block fallback paths rather
    // than landing after the vector loop.
    let mut actual = sawtooth(8_192, 8_192, -80.0, 160.0 / 8_192.0);
    for (index, value) in [
        (3, f32::NEG_INFINITY),
        (67, f32::INFINITY),
        (131, f32::NAN),
        (259, f32::MIN),
        (1027, f32::MAX),
        (2051, f32::from_bits(1)),
    ] {
        actual[index] = value;
    }
    let expected: Vec<f32> = actual.iter().map(|value| value.exp()).collect();
    exp_inplace(&mut actual);
    assert_exp_close(&actual, &expected);
}

#[test]
fn logistic_gradient_dispatch_is_close_to_scalar() {
    let preds = sawtooth(4_103, 4_103, -20.0, 40.0 / 4_102.0);
    let labels: Vec<f32> = (0..preds.len()).map(|i| (i % 2) as f32).collect();
    let weights = sawtooth(preds.len(), 13, 0.5, 0.125);
    for weights in [None, Some(weights.as_slice())] {
        let mut expected = vec![GradPair::default(); preds.len()];
        let mut actual = vec![GradPair::default(); preds.len()];
        scalar::logistic_gradient(
            &preds,
            &labels,
            weights,
            1.5,
            1e-16,
            &mut expected,
            0..preds.len(),
        );
        logistic_gradient(&preds, &labels, weights, 1.5, 1e-16, &mut actual);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual.grad - expected.grad).abs() <= 5.0e-7);
            assert!((actual.hess - expected.hess).abs() <= 3.0e-7);
        }
    }
}

#[test]
fn count_gradients_are_close_to_scalar() {
    let preds = sawtooth(4_103, 4_103, -5.0, 10.0 / 4_102.0);
    let labels = sawtooth(preds.len(), 101, 0.25, 0.02);
    let weights = sawtooth(preds.len(), 13, 0.5, 0.125);

    for weights in [None, Some(weights.as_slice())] {
        let mut actual = vec![GradPair::default(); preds.len()];
        let mut expected = vec![GradPair::default(); preds.len()];
        let range = 0..preds.len();

        poisson_gradient(&preds, &labels, weights, 0.7, &mut actual);
        scalar::poisson_gradient(&preds, &labels, weights, 0.7, &mut expected, range.clone());
        assert_grad_pairs_close(&actual, &expected);

        gamma_gradient(&preds, &labels, weights, &mut actual);
        scalar::gamma_gradient(&preds, &labels, weights, &mut expected, range.clone());
        assert_grad_pairs_close(&actual, &expected);

        tweedie_gradient(&preds, &labels, weights, 1.5, &mut actual);
        scalar::tweedie_gradient(&preds, &labels, weights, 1.5, &mut expected, range);
        assert_grad_pairs_close(&actual, &expected);
    }
}

#[test]
fn wide_softmax_dispatch_is_close_to_scalar() {
    for num_class in [8, 32, 33, 128, 131] {
        let original = sawtooth(num_class, 103, -2.5, 0.05);
        let expected = scalar_softmax_rows(&original, num_class);
        let mut actual = original;
        softmax_rows_inplace(&mut actual, num_class);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 3.0e-7);
        }
    }

    // A wide dynamic range takes the scalar exceptional path so exponent
    // underflow behavior remains identical.
    let mut actual = vec![-100.0; 32];
    actual[0] = 100.0;
    let expected = scalar_softmax_rows(&actual, 32);
    softmax_rows_inplace(&mut actual, 32);
    assert_eq!(actual, expected);
}

#[test]
fn softmax_matrix_and_gradient_are_close_to_scalar() {
    for num_class in [4, 8, 17, 32, 33, 128, 131] {
        let rows = 257;
        let original = sawtooth(rows * num_class, 211, -2.5, 0.025);
        let labels: Vec<f32> = (0..rows).map(|row| (row % num_class) as f32).collect();
        let weight_values = sawtooth(rows, 11, 0.5, 0.125);

        let mut transformed = original.clone();
        let expected_transform = scalar_softmax_rows(&original, num_class);
        softmax_rows_inplace(&mut transformed, num_class);
        for (actual, expected) in transformed.iter().zip(&expected_transform) {
            assert!((actual - expected).abs() <= 3.0e-7);
        }

        for weights in [None, Some(weight_values.as_slice())] {
            let mut actual = vec![GradPair::default(); original.len()];
            let expected = scalar_softmax_gradient(&original, &labels, weights, num_class);
            softmax_gradient(&original, &labels, weights, num_class, 1e-16, &mut actual);
            assert_grad_pairs_close(&actual, &expected);
        }
    }
}

#[test]
fn short_softmax_batches_match_scalar_across_boundaries() {
    for num_class in [2, 3, 4] {
        for rows in (0..18).chain([127, 256, 257]) {
            let original: Vec<f32> = (0..rows * num_class)
                .map(|i| ((i * 71) % 211) as f32 * 0.375 - 39.0)
                .collect();
            let labels: Vec<f32> = (0..rows).map(|i| (i % num_class) as f32).collect();
            let weights: Vec<f32> = (0..rows).map(|i| (i % 7) as f32 * 0.25).collect();
            let expected = scalar_softmax_rows(&original, num_class);
            // Offset and guard both buffers to exercise unaligned stores
            // and catch writes past batch/remainder boundaries, including odd rows.
            let mut values = vec![1234.0; original.len() + 2];
            values[1..=original.len()].copy_from_slice(&original);
            softmax_rows_inplace(&mut values[1..=original.len()], num_class);
            assert_eq!(values[0], 1234.0);
            assert_eq!(values[original.len() + 1], 1234.0);
            for (actual, expected) in values[1..=original.len()].iter().zip(expected) {
                assert!((actual - expected).abs() <= 3e-7);
            }
            for weight in [None, Some(weights.as_slice())] {
                let guard = GradPair::new(1234.0, 5678.0);
                let mut actual = vec![guard; original.len() + 2];
                let expected = scalar_softmax_gradient(&original, &labels, weight, num_class);
                softmax_gradient(
                    &original,
                    &labels,
                    weight,
                    num_class,
                    1e-16,
                    &mut actual[1..=original.len()],
                );
                for index in [0, original.len() + 1] {
                    assert_eq!(actual[index].grad, guard.grad);
                    assert_eq!(actual[index].hess, guard.hess);
                }
                assert_grad_pairs_close(&actual[1..=original.len()], &expected);
            }
        }
    }
}

#[test]
fn short_softmax_exceptional_rows_and_label_casts_match_scalar() {
    let close = |actual: f32, expected: f32| {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else if expected.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            assert!((actual - expected).abs() <= 3e-7, "{actual} != {expected}");
        }
    };
    let labels = [
        f32::NAN,
        -2.0,
        f32::INFINITY,
        1.5,
        f32::MAX,
        u32::MAX as f32,
        2.0,
        3.0,
        f32::NEG_INFINITY,
    ];
    let weights = [0.0, 0.25, 1.0, f32::NAN, f32::INFINITY, -1.0, 2.0, 0.5, 1.0];
    for num_class in [2, 3, 4] {
        for position in 0..labels.len() * num_class {
            for exceptional in [
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -81.0,
                81.0,
                -80.0,
                80.0,
                f32::MAX,
            ] {
                let mut original = vec![0.0; labels.len() * num_class];
                original[position] = exceptional;
                let expected = scalar_softmax_rows(&original, num_class);
                let mut actual = original.clone();
                softmax_rows_inplace(&mut actual, num_class);
                for (&actual, &expected) in actual.iter().zip(&expected) {
                    close(actual, expected);
                }
                for weight in [None, Some(weights.as_slice())] {
                    let mut actual = vec![GradPair::default(); original.len()];
                    let expected = scalar_softmax_gradient(&original, &labels, weight, num_class);
                    softmax_gradient(&original, &labels, weight, num_class, 1e-16, &mut actual);
                    for (actual, expected) in actual.iter().zip(expected) {
                        close(actual.grad, expected.grad);
                        close(actual.hess, expected.hess);
                    }
                }
            }
        }
    }
}

/// A `num_class` matrix of 24 rows cycling through rows whose maximum is
/// below `f32::MIN_POSITIVE`: all `-200.0` (XGBoost's `MIN_POSITIVE` shift
/// underflows every exponential, giving `0/0`), all `-30.0` (finite under that
/// shift), and ordinary margins.
fn underflowing_softmax_rows(num_class: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = 24;
    let mut preds = Vec::with_capacity(rows * num_class);
    for row in 0..rows {
        for class in 0..num_class {
            preds.push(match row % 3 {
                0 => -200.0,
                1 => -30.0,
                _ => ((row * 7 + class * 5) % 11) as f32 * 0.4 - 2.0,
            });
        }
    }
    let labels = (0..rows)
        .map(|row| ((row * 5) % num_class) as f32)
        .collect();
    let weights = (0..rows).map(|row| 0.5 + (row % 4) as f32 * 0.25).collect();
    (preds, labels, weights)
}

/// Run `kernel` over [`underflowing_softmax_rows`] with and without weights
/// and compare it against `softmax_gradient_rows_scalar`: the `-200.0` rows
/// must reproduce XGBoost's `0/0` bit for bit, the others within tolerance.
fn assert_underflowing_softmax_gradient_matches_scalar(
    num_class: usize,
    kernel: impl Fn(&[f32], &[f32], Option<&[f32]>, &mut [GradPair]),
) {
    let (preds, labels, weights) = underflowing_softmax_rows(num_class);
    for weights in [None, Some(weights.as_slice())] {
        let expected = scalar_softmax_gradient(&preds, &labels, weights, num_class);
        let mut actual = vec![GradPair::default(); preds.len()];
        kernel(&preds, &labels, weights, &mut actual);
        for (row, (actual, expected)) in actual
            .chunks(num_class)
            .zip(expected.chunks(num_class))
            .enumerate()
        {
            if row % 3 == 0 {
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!(expected.grad.is_nan(), "scalar row {row} did not underflow");
                    assert_eq!(
                        actual.grad.to_bits(),
                        expected.grad.to_bits(),
                        "num_class {num_class} row {row}: SIMD gradient {} differs from scalar {}",
                        actual.grad,
                        expected.grad
                    );
                    assert_eq!(actual.hess.to_bits(), expected.hess.to_bits());
                }
            } else {
                assert_grad_pairs_close(actual, expected);
            }
        }
    }
}

#[test]
fn softmax_gradient_underflowing_rows_match_scalar() {
    for num_class in [2, 3, 4, 8] {
        assert_underflowing_softmax_gradient_matches_scalar(
            num_class,
            |preds, labels, weights, out| {
                softmax_gradient(preds, labels, weights, num_class, 1e-16, out);
            },
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_softmax_gradient_kernels_underflow_like_scalar() {
    if !neon_available() {
        return;
    }
    for num_class in [2, 3, 4] {
        assert_underflowing_softmax_gradient_matches_scalar(
            num_class,
            |preds, labels, weights, out| {
                // SAFETY: NEON was detected and the inputs form complete
                // matrices of `num_class` columns; K is 2, 3, or 4.
                unsafe {
                    match num_class {
                        2 => {
                            aarch64::short_softmax_gradient::<2>(
                                preds, labels, weights, 1e-16, out,
                            );
                        }
                        3 => {
                            aarch64::short_softmax_gradient::<3>(
                                preds, labels, weights, 1e-16, out,
                            );
                        }
                        _ => {
                            aarch64::short_softmax_gradient::<4>(
                                preds, labels, weights, 1e-16, out,
                            );
                        }
                    }
                }
            },
        );
    }
    assert_underflowing_softmax_gradient_matches_scalar(8, |preds, labels, weights, out| {
        // SAFETY: NEON was detected and the inputs form a complete matrix of
        // eight columns.
        unsafe { aarch64::softmax_gradient(preds, labels, weights, 8, 1e-16, out) }
    });
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_softmax_gradient_kernels_underflow_like_scalar() {
    if !avx2_fma_available() {
        return;
    }
    for num_class in [2, 4] {
        assert_underflowing_softmax_gradient_matches_scalar(
            num_class,
            |preds, labels, weights, out| {
                // SAFETY: AVX2/FMA were detected and the inputs form complete
                // matrices of `num_class` columns; K is 2 or 4.
                unsafe {
                    if num_class == 2 {
                        x86_64::short_softmax_gradient::<2>(preds, labels, weights, 1e-16, out);
                    } else {
                        x86_64::short_softmax_gradient::<4>(preds, labels, weights, 1e-16, out);
                    }
                }
            },
        );
    }
}

#[test]
fn count_le_16_matches_scalar_on_special_values() {
    // Exactly 16 cuts takes the vector path; NaN cuts never count, ties count,
    // and the cuts need not be sorted.
    let cuts: [f32; 16] = [
        f32::NEG_INFINITY,
        -3.0,
        -0.0,
        0.0,
        0.0,
        1.5,
        f32::NAN,
        2.0,
        2.0,
        7.25,
        f32::INFINITY,
        -1e-40,
        1e-40,
        f32::MAX,
        f32::MIN,
        4.0,
    ];
    for value in [
        f32::NEG_INFINITY,
        -3.0,
        -0.0,
        0.0,
        1e-40,
        2.0,
        4.0,
        7.25,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ] {
        let expected = cuts.iter().filter(|&&cut| cut <= value).count();
        assert_eq!(count_le(&cuts, value), expected, "value {value}");
    }
    // Other lengths use the scalar path unchanged.
    let short = &cuts[..15];
    assert_eq!(
        count_le(short, 2.0),
        short.iter().filter(|&&cut| cut <= 2.0).count()
    );
    assert_eq!(count_le(&[], 2.0), 0);
}

#[test]
fn pointwise_metric_sums_are_close_to_scalar() {
    let preds: Vec<f32> = (0..4_103).map(|i| (i % 1_001) as f32 * 0.001).collect();
    let labels: Vec<f32> = (0..preds.len()).map(|i| (i % 2) as f32).collect();
    let weights = sawtooth(preds.len(), 17, 0.5, 0.0625);
    for weights in [None, Some(weights.as_slice())] {
        let range = 0..preds.len();
        assert_sums_close(
            squared_error_sum(&preds, &labels, weights),
            scalar::distance_sum::<true>(&preds, &labels, weights, range.clone()),
            1e-12,
        );
        assert_sums_close(
            absolute_error_sum(&preds, &labels, weights),
            scalar::distance_sum::<false>(&preds, &labels, weights, range.clone()),
            1e-12,
        );
        assert_eq!(
            classification_error_sum(&preds, &labels, weights),
            scalar::classification_error_sum(&preds, &labels, weights, range)
        );
    }
}

#[test]
fn logarithmic_metric_sums_are_close_to_scalar() {
    let preds = sawtooth(4_103, 999, 0.001, 0.001);
    let binary_labels: Vec<f32> = (0..preds.len()).map(|i| (i % 2) as f32).collect();
    let positive_labels = sawtooth(preds.len(), 101, 0.25, 0.02);
    let weights = sawtooth(preds.len(), 17, 0.51, 0.061);
    // Raw margins outside [0, 1] (`binary:logitraw`) reach XGBoost's
    // unclamped 1e-16 floor of each log argument.
    let margins = sawtooth(4_103, 999, -0.5, 0.002);
    let range = 0..preds.len();
    for weights in [None, Some(weights.as_slice())] {
        for p in [&preds, &margins] {
            assert_sums_close(
                log_loss_sum(p, &binary_labels, weights),
                scalar::log_loss(p, &binary_labels, weights, range.clone()),
                2e-12,
            );
        }
        assert_sums_close(
            positive_nloglik_sum::<true>(&preds, &positive_labels, weights),
            scalar::positive_nloglik::<true>(&preds, &positive_labels, weights, range.clone()),
            2e-12,
        );
        assert_sums_close(
            positive_nloglik_sum::<false>(&preds, &positive_labels, weights),
            scalar::positive_nloglik::<false>(&preds, &positive_labels, weights, range.clone()),
            2e-12,
        );
    }
}

#[test]
fn tweedie_metric_sum_is_close_to_scalar() {
    let preds = sawtooth(4_103, 2_003, 1e-6, 0.05);
    let labels = sawtooth(preds.len(), 101, 0.25, 0.02);
    let weight_values = sawtooth(preds.len(), 17, 0.51, 0.061);
    for rho in [1.1, 1.5, 1.9] {
        for weights in [None, Some(weight_values.as_slice())] {
            let actual = tweedie_nloglik_sum(&preds, &labels, weights, rho);
            let expected = scalar::tweedie_nloglik(&preds, &labels, weights, rho, 0..preds.len());
            assert_sums_close(actual, expected, 3e-12);
        }
    }
}

#[test]
fn multiclass_metric_sums_are_close_to_scalar() {
    let rows = 4_103;
    for num_class in [3, 8, 17, 32, 131] {
        let preds = sawtooth(rows * num_class, 999, 0.001, 0.001);
        let labels: Vec<f32> = (0..rows)
            .map(|row| ((row * 7) % num_class) as f32)
            .collect();
        let weight_values = sawtooth(rows, 17, 0.51, 0.061);
        for weights in [None, Some(weight_values.as_slice())] {
            let actual = multiclass_log_loss_sum(&preds, &labels, weights, num_class);
            let expected =
                scalar::multiclass_log_loss(&preds, &labels, weights, num_class, 0..rows);
            assert_sums_close(actual, expected, 2e-12);

            assert_eq!(
                multiclass_error_sum(&preds, &labels, weights, num_class),
                multiclass_error_sum_rows(&preds, &labels, weights, num_class, argmax_scalar)
            );
        }
    }
}

#[test]
fn multiclass_metric_fallback_preserves_ties_and_nonfinite_values() {
    let num_class = 8;
    let labels = vec![0.0; 17];
    let mut preds = vec![0.25; labels.len() * num_class];
    preds[num_class] = f32::NAN;
    preds[2 * num_class] = f32::INFINITY;
    assert_eq!(
        multiclass_error_sum(&preds, &labels, None, num_class),
        multiclass_error_sum_rows(&preds, &labels, None, num_class, argmax_scalar)
    );

    let actual = multiclass_log_loss_sum(&preds, &labels, None, num_class);
    let expected = scalar::multiclass_log_loss(&preds, &labels, None, num_class, 0..labels.len());
    assert_eq!(actual.1, expected.1);
    assert_eq!(actual.0.is_nan(), expected.0.is_nan());
}
