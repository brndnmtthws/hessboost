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

#[test]
fn sigmoid_dispatch_is_close_to_scalar() {
    let mut actual: Vec<f32> = (0..4_103)
        .map(|i| i as f32 * (40.0 / 4_102.0) - 20.0)
        .collect();
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
    let mut actual: Vec<f32> = (0..8_195)
        .map(|i| i as f32 * (160.0 / 8_192.0) - 80.0)
        .collect();
    actual.extend([f32::NEG_INFINITY, f32::INFINITY, f32::NAN]);
    let expected: Vec<f32> = actual.iter().map(|value| value.exp()).collect();
    exp_inplace(&mut actual);
    for (actual, expected) in actual.iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else if expected.is_infinite() || expected == 0.0 {
            assert_eq!(*actual, expected);
        } else {
            let relative = ((actual - expected) / expected).abs();
            assert!(
                relative <= 7.0e-7,
                "SIMD exp {actual} differs from scalar {expected} by {relative}"
            );
        }
    }
}

#[test]
fn exp_dispatch_handles_special_values_inside_vector_blocks() {
    // With a length divisible by every vector width there is no scalar tail,
    // so these special values exercise the in-block fallback paths rather
    // than landing after the vector loop.
    let mut actual: Vec<f32> = (0..8_192)
        .map(|i| i as f32 * (160.0 / 8_192.0) - 80.0)
        .collect();
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
    for (actual, expected) in actual.iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else if expected.is_infinite() || expected == 0.0 {
            assert_eq!(*actual, expected);
        } else {
            let relative = ((actual - expected) / expected).abs();
            assert!(
                relative <= 7.0e-7,
                "SIMD exp {actual} differs from scalar {expected} by {relative}"
            );
        }
    }
}

#[test]
fn logistic_gradient_dispatch_is_close_to_scalar() {
    let preds: Vec<f32> = (0..4_103)
        .map(|i| i as f32 * (40.0 / 4_102.0) - 20.0)
        .collect();
    let labels: Vec<f32> = (0..preds.len()).map(|i| (i % 2) as f32).collect();
    let weights: Vec<f32> = (0..preds.len())
        .map(|i| 0.5 + (i % 13) as f32 * 0.125)
        .collect();
    for weights in [None, Some(weights.as_slice())] {
        let mut expected = vec![GradPair::default(); preds.len()];
        let mut actual = vec![GradPair::default(); preds.len()];
        logistic_gradient_scalar(&preds, &labels, weights, 1.5, 1e-16, &mut expected);
        logistic_gradient(&preds, &labels, weights, 1.5, 1e-16, &mut actual);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual.grad - expected.grad).abs() <= 5.0e-7);
            assert!((actual.hess - expected.hess).abs() <= 3.0e-7);
        }
    }
}

#[test]
fn count_gradients_are_close_to_scalar() {
    let preds: Vec<f32> = (0..4_103)
        .map(|i| i as f32 * (10.0 / 4_102.0) - 5.0)
        .collect();
    let labels: Vec<f32> = (0..preds.len())
        .map(|i| 0.25 + (i % 101) as f32 * 0.02)
        .collect();
    let weights: Vec<f32> = (0..preds.len())
        .map(|i| 0.5 + (i % 13) as f32 * 0.125)
        .collect();

    for weights in [None, Some(weights.as_slice())] {
        let mut actual = vec![GradPair::default(); preds.len()];
        let mut expected = vec![GradPair::default(); preds.len()];

        poisson_gradient(&preds, &labels, weights, 0.7, &mut actual);
        for i in 0..preds.len() {
            let weight = weights.map_or(1.0, |values| values[i]);
            expected[i] = GradPair::new(
                (preds[i].exp() - labels[i]) * weight,
                (preds[i] + 0.7).exp() * weight,
            );
        }
        assert_grad_pairs_close(&actual, &expected);

        gamma_gradient(&preds, &labels, weights, &mut actual);
        for i in 0..preds.len() {
            let weight = weights.map_or(1.0, |values| values[i]);
            let scaled = labels[i] * (-preds[i]).exp();
            expected[i] = GradPair::new((1.0 - scaled) * weight, scaled * weight);
        }
        assert_grad_pairs_close(&actual, &expected);

        let rho = 1.5;
        tweedie_gradient(&preds, &labels, weights, rho, &mut actual);
        for i in 0..preds.len() {
            let weight = weights.map_or(1.0, |values| values[i]);
            let exp_1 = ((1.0 - rho) * preds[i]).exp();
            let exp_2 = ((2.0 - rho) * preds[i]).exp();
            expected[i] = GradPair::new(
                (-labels[i] * exp_1 + exp_2) * weight,
                (-labels[i] * (1.0 - rho) * exp_1 + (2.0 - rho) * exp_2) * weight,
            );
        }
        assert_grad_pairs_close(&actual, &expected);
    }
}

#[test]
fn wide_softmax_dispatch_is_close_to_scalar() {
    for num_class in [8, 32, 33, 128, 131] {
        let original: Vec<f32> = (0..num_class)
            .map(|i| (i % 103) as f32 * 0.05 - 2.5)
            .collect();
        let mut expected = original.clone();
        softmax_scalar(&mut expected);
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
    let mut expected = actual.clone();
    softmax_scalar(&mut expected);
    softmax_rows_inplace(&mut actual, 32);
    assert_eq!(actual, expected);
}

#[test]
fn softmax_matrix_and_gradient_are_close_to_scalar() {
    for num_class in [4, 8, 17, 32, 33, 128, 131] {
        let rows = 257;
        let original: Vec<f32> = (0..rows * num_class)
            .map(|i| (i % 211) as f32 * 0.025 - 2.5)
            .collect();
        let labels: Vec<f32> = (0..rows).map(|row| (row % num_class) as f32).collect();
        let weight_values: Vec<f32> = (0..rows)
            .map(|row| 0.5 + (row % 11) as f32 * 0.125)
            .collect();

        let mut transformed = original.clone();
        let mut expected_transform = original.clone();
        for row in expected_transform.chunks_mut(num_class) {
            softmax_scalar(row);
        }
        softmax_rows_inplace(&mut transformed, num_class);
        for (actual, expected) in transformed.iter().zip(&expected_transform) {
            assert!((actual - expected).abs() <= 3.0e-7);
        }

        for weights in [None, Some(weight_values.as_slice())] {
            let mut actual = vec![GradPair::default(); original.len()];
            let mut expected = vec![GradPair::default(); original.len()];
            for row in 0..rows {
                let base = row * num_class;
                softmax_gradient_row_scalar(
                    &original[base..base + num_class],
                    labels[row] as usize,
                    weights.map_or(1.0, |values| values[row]),
                    1e-16,
                    &mut expected[base..base + num_class],
                );
            }
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
            let mut expected = original.clone();
            for row in expected.chunks_mut(num_class) {
                softmax_scalar(row);
            }
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
                let mut expected = vec![GradPair::default(); original.len()];
                for row in 0..rows {
                    let base = row * num_class;
                    softmax_gradient_row_scalar(
                        &original[base..base + num_class],
                        labels[row] as usize,
                        weight.map_or(1.0, |w| w[row]),
                        1e-16,
                        &mut expected[base..base + num_class],
                    );
                }
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
                let mut expected = original.clone();
                for row in expected.chunks_mut(num_class) {
                    softmax_scalar(row);
                }
                let mut actual = original.clone();
                softmax_rows_inplace(&mut actual, num_class);
                for (&actual, &expected) in actual.iter().zip(&expected) {
                    close(actual, expected);
                }
                for weight in [None, Some(weights.as_slice())] {
                    let mut actual = vec![GradPair::default(); original.len()];
                    let mut expected = actual.clone();
                    for row in 0..labels.len() {
                        let base = row * num_class;
                        softmax_gradient_row_scalar(
                            &original[base..base + num_class],
                            labels[row] as usize,
                            weight.map_or(1.0, |w| w[row]),
                            1e-16,
                            &mut expected[base..base + num_class],
                        );
                    }
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
        let mut expected = vec![GradPair::default(); preds.len()];
        softmax_gradient_rows_scalar(
            &preds,
            &labels,
            weights,
            1e-16,
            &mut expected,
            0..labels.len(),
            num_class,
        );
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

#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_backend_detection_is_cached() {
    let first = neon_available();
    assert_eq!(first, *NEON_AVAILABLE);
    assert_eq!(first, neon_available());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_64_backend_detection_is_cached() {
    let first = avx2_fma_available();
    assert_eq!(first, *AVX2_FMA_AVAILABLE);
    assert_eq!(first, avx2_fma_available());
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
    let weights: Vec<f32> = (0..preds.len())
        .map(|i| 0.5 + (i % 17) as f32 * 0.0625)
        .collect();
    for weights in [None, Some(weights.as_slice())] {
        for squared in [false, true] {
            let actual = if squared {
                squared_error_sum(&preds, &labels, weights)
            } else {
                absolute_error_sum(&preds, &labels, weights)
            };
            let mut expected = (0.0, 0.0);
            for i in 0..preds.len() {
                let weight = weights.map_or(1.0, |values| f64::from(values[i]));
                let difference = f64::from(preds[i]) - f64::from(labels[i]);
                expected.0 += weight
                    * if squared {
                        difference * difference
                    } else {
                        difference.abs()
                    };
                expected.1 += weight;
            }
            assert!((actual.0 - expected.0).abs() <= expected.0.abs() * 1e-12);
            assert!((actual.1 - expected.1).abs() <= expected.1.abs() * 1e-12);
        }

        let actual = classification_error_sum(&preds, &labels, weights);
        let mut expected = (0.0, 0.0);
        for i in 0..preds.len() {
            let weight = weights.map_or(1.0, |values| f64::from(values[i]));
            if (preds[i] > 0.5) != (labels[i] > 0.5) {
                expected.0 += weight;
            }
            expected.1 += weight;
        }
        assert_eq!(actual, expected);
    }
}

#[test]
fn logarithmic_metric_sums_are_close_to_scalar() {
    let preds: Vec<f32> = (0..4_103)
        .map(|i| 0.001 + (i % 999) as f32 * 0.001)
        .collect();
    let binary_labels: Vec<f32> = (0..preds.len()).map(|i| (i % 2) as f32).collect();
    let positive_labels: Vec<f32> = (0..preds.len())
        .map(|i| 0.25 + (i % 101) as f32 * 0.02)
        .collect();
    let weights: Vec<f32> = (0..preds.len())
        .map(|i| 0.51 + (i % 17) as f32 * 0.061)
        .collect();
    for weights in [None, Some(weights.as_slice())] {
        let actual = log_loss_sum(&preds, &binary_labels, weights);
        let mut expected = (0.0, 0.0);
        for i in 0..preds.len() {
            let weight = weights.map_or(1.0, |values| f64::from(values[i]));
            let probability = f64::from(preds[i]).clamp(1e-15, 1.0 - 1e-15);
            let label = f64::from(binary_labels[i]);
            expected.0 +=
                weight * -(label * probability.ln() + (1.0 - label) * (1.0 - probability).ln());
            expected.1 += weight;
        }
        assert!((actual.0 - expected.0).abs() <= expected.0.abs() * 2e-12);
        assert!((actual.1 - expected.1).abs() <= expected.1.abs() * 2e-12);

        for gamma in [false, true] {
            let actual = if gamma {
                positive_nloglik_sum::<true>(&preds, &positive_labels, weights)
            } else {
                positive_nloglik_sum::<false>(&preds, &positive_labels, weights)
            };
            let mut expected = (0.0, 0.0);
            for i in 0..preds.len() {
                let weight = weights.map_or(1.0, |values| f64::from(values[i]));
                let prediction = f64::from(preds[i]).max(1e-8);
                let label = f64::from(positive_labels[i]);
                expected.0 += weight
                    * if gamma {
                        label / prediction + prediction.ln()
                    } else {
                        prediction - label * prediction.ln()
                    };
                expected.1 += weight;
            }
            assert!((actual.0 - expected.0).abs() <= expected.0.abs() * 2e-12);
            assert!((actual.1 - expected.1).abs() <= expected.1.abs() * 2e-12);
        }
    }
}

#[test]
fn tweedie_metric_sum_is_close_to_scalar() {
    let preds: Vec<f32> = (0..4_103)
        .map(|index| 1e-6 + (index % 2_003) as f32 * 0.05)
        .collect();
    let labels: Vec<f32> = (0..preds.len())
        .map(|index| 0.25 + (index % 101) as f32 * 0.02)
        .collect();
    let weight_values: Vec<f32> = (0..preds.len())
        .map(|index| 0.51 + (index % 17) as f32 * 0.061)
        .collect();
    for rho in [1.1, 1.5, 1.9] {
        for weights in [None, Some(weight_values.as_slice())] {
            let actual = tweedie_nloglik_sum(&preds, &labels, weights, rho);
            let expected = tweedie_nloglik_sum_scalar(&preds, &labels, weights, rho);
            assert!((actual.0 - expected.0).abs() <= expected.0.abs() * 3e-12);
            assert!((actual.1 - expected.1).abs() <= expected.1.abs() * 2e-12);
        }
    }
}

#[test]
fn multiclass_metric_sums_are_close_to_scalar() {
    let rows = 4_103;
    for num_class in [3, 8, 17, 32, 131] {
        let preds: Vec<f32> = (0..rows * num_class)
            .map(|index| 0.001 + (index % 999) as f32 * 0.001)
            .collect();
        let labels: Vec<f32> = (0..rows)
            .map(|row| ((row * 7) % num_class) as f32)
            .collect();
        let weight_values: Vec<f32> = (0..rows)
            .map(|row| 0.51 + (row % 17) as f32 * 0.061)
            .collect();
        for weights in [None, Some(weight_values.as_slice())] {
            let actual = multiclass_log_loss_sum(&preds, &labels, weights, num_class);
            let expected = multiclass_log_loss_sum_scalar(&preds, &labels, weights, num_class);
            assert!((actual.0 - expected.0).abs() <= expected.0.abs() * 2e-12);
            assert!((actual.1 - expected.1).abs() <= expected.1.abs() * 2e-12);

            assert_eq!(
                multiclass_error_sum(&preds, &labels, weights, num_class),
                multiclass_error_sum_scalar(&preds, &labels, weights, num_class)
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
        multiclass_error_sum_scalar(&preds, &labels, None, num_class)
    );

    let actual = multiclass_log_loss_sum(&preds, &labels, None, num_class);
    let expected = multiclass_log_loss_sum_scalar(&preds, &labels, None, num_class);
    assert_eq!(actual.1, expected.1);
    assert_eq!(actual.0.is_nan(), expected.0.is_nan());
}
