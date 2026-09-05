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
            values[1..original.len() + 1].copy_from_slice(&original);
            softmax_rows_inplace(&mut values[1..original.len() + 1], num_class);
            assert_eq!(values[0], 1234.0);
            assert_eq!(values[original.len() + 1], 1234.0);
            for (actual, expected) in values[1..original.len() + 1].iter().zip(expected) {
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
                    &mut actual[1..original.len() + 1],
                );
                for index in [0, original.len() + 1] {
                    assert_eq!(actual[index].grad, guard.grad);
                    assert_eq!(actual[index].hess, guard.hess);
                }
                assert_grad_pairs_close(&actual[1..original.len() + 1], &expected);
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

#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_backend_detection_is_cached() {
    let first = neon_available();
    let cached = *NEON_AVAILABLE.get().expect("backend must be initialized");
    assert_eq!(first, cached);
    assert_eq!(first, neon_available());
}

#[cfg(target_arch = "aarch64")]
#[test]
fn dense_split_scan_matches_scalar_candidate_order() {
    use crate::tree::gain::calc_gain;

    for length in [16, 17, 255, 256] {
        let histogram: Vec<GradStats> = (0..length)
            .map(|index| {
                GradStats::new(
                    (index % 19) as f64 - 9.0 + index as f64 * 0.003,
                    0.5 + (index % 7) as f64 * 0.25,
                )
            })
            .collect();
        let mut total = GradStats::default();
        for &stats in &histogram {
            total.add(stats);
        }
        for (alpha, lambda, min_child_weight) in
            [(0.0, 1.0, 1.0), (0.75, 0.25, 2.0), (4.0, 2.0, 0.0)]
        {
            let reg = RegParams {
                alpha,
                lambda,
                min_child_weight,
                max_delta_step: 0.0,
            };
            let parent_gain = calc_gain(total, &reg);
            let mut accumulated = GradStats::default();
            let mut expected: Option<SplitCandidate> = None;
            let mut best_loss = 0.0;
            for (index, &stats) in histogram.iter().take(histogram.len() - 1).enumerate() {
                accumulated.add(stats);
                let right = total.sub(accumulated);
                if accumulated.hess >= min_child_weight && right.hess >= min_child_weight {
                    let loss = calc_gain(accumulated, &reg) + calc_gain(right, &reg) - parent_gain;
                    if loss > best_loss + 1e-6 {
                        best_loss = loss;
                        expected = Some(SplitCandidate {
                            loss_change: loss,
                            split_offset: index,
                            left: accumulated,
                            right,
                        });
                    }
                }
            }

            let DenseSplitScan::Scanned(actual) =
                dense_unconstrained_best_split(&histogram, total, &reg, parent_gain, 1e-6)
            else {
                panic!("NEON split scan should dispatch on this host");
            };
            match (actual, expected) {
                (Some(actual), Some(expected)) => {
                    assert_eq!(actual.split_offset, expected.split_offset);
                    assert_eq!(actual.left, expected.left);
                    assert_eq!(actual.right, expected.right);
                    assert!((actual.loss_change - expected.loss_change).abs() <= 1e-12);
                }
                (None, None) => {}
                _ => panic!("NEON and scalar scans disagreed on candidate presence"),
            }
        }
    }
}

#[test]
fn grad_stats_sum_is_close_to_scalar() {
    for length in [0, 1, 15, 16, 17, 255, 256, 257] {
        let values: Vec<GradStats> = (0..length)
            .map(|index| {
                GradStats::new(
                    (index % 19) as f64 * 0.03125 - 0.25,
                    0.5 + (index % 7) as f64 * 0.125,
                )
            })
            .collect();
        let mut expected = GradStats::default();
        for &value in &values {
            expected.add(value);
        }
        let actual = sum_grad_stats(&values);
        assert!((actual.grad - expected.grad).abs() <= 1e-12);
        assert!((actual.hess - expected.hess).abs() <= 1e-12);
    }
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
                let weight = weights.map_or(1.0, |values| values[i] as f64);
                let difference = preds[i] as f64 - labels[i] as f64;
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
            let weight = weights.map_or(1.0, |values| values[i] as f64);
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
            let weight = weights.map_or(1.0, |values| values[i] as f64);
            let probability = (preds[i] as f64).clamp(1e-15, 1.0 - 1e-15);
            let label = binary_labels[i] as f64;
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
                let weight = weights.map_or(1.0, |values| values[i] as f64);
                let prediction = (preds[i] as f64).max(1e-8);
                let label = positive_labels[i] as f64;
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
