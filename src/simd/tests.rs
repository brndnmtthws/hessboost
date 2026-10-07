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
    // Every third gamma label exactly 1, which `scale_pos_weight` reweights.
    let gamma_labels: Vec<f32> = labels
        .iter()
        .enumerate()
        .map(|(i, &y)| if i % 3 == 0 { 1.0 } else { y })
        .collect();
    let weights = sawtooth(preds.len(), 13, 0.5, 0.125);

    for weights in [None, Some(weights.as_slice())] {
        let mut actual = vec![GradPair::default(); preds.len()];
        let mut expected = vec![GradPair::default(); preds.len()];
        let range = 0..preds.len();

        poisson_gradient(&preds, &labels, weights, 0.7, &mut actual);
        scalar::poisson_gradient(&preds, &labels, weights, 0.7, &mut expected, range.clone());
        assert_grad_pairs_close(&actual, &expected);

        gamma_gradient(&preds, &gamma_labels, weights, 1.5, &mut actual);
        scalar::gamma_gradient(
            &preds,
            &gamma_labels,
            weights,
            1.5,
            &mut expected,
            range.clone(),
        );
        assert_grad_pairs_close(&actual, &expected);

        tweedie_gradient(&preds, &labels, weights, 1.5, &mut actual);
        scalar::tweedie_gradient(&preds, &labels, weights, 1.5, &mut expected, range);
        assert_grad_pairs_close(&actual, &expected);
    }
}

#[test]
fn softmax_gradient_is_close_to_scalar() {
    for num_class in [4, 8, 17, 32, 33, 128, 131] {
        let rows = 257;
        let original = sawtooth(rows * num_class, 211, -2.5, 0.025);
        let labels: Vec<f32> = (0..rows).map(|row| (row % num_class) as f32).collect();
        let weight_values = sawtooth(rows, 11, 0.5, 0.125);

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
fn shap_edge_terms_match_the_scalar_expression_bit_for_bit() {
    // SHAP values must equal XGBoost's per-lane arithmetic exactly, including
    // overflowing, underflowing, and NaN lanes, so the vector kernel is
    // compared bitwise with the scalar expression (fused multiply-add on
    // AArch64, as `shap::madd` rounds).
    let scalar = |alpha: f32, h: f32, u: f32| {
        let denominator = if cfg!(target_arch = "aarch64") {
            alpha.mul_add(u, 1.0)
        } else {
            alpha * u + 1.0
        };
        alpha * h / denominator
    };
    let u = [
        0.0,
        0.019_855_07,
        0.101_666_76,
        0.237_233_8,
        0.408_282_68,
        0.591_717_3,
        0.762_766_2,
        1.0,
    ];
    let h = [
        1.5,
        -2.25,
        1e-30,
        3.0e38,
        -0.0,
        f32::INFINITY,
        f32::NAN,
        7.0,
    ];
    for alpha in [
        -1.0,
        -0.999_999,
        -0.5,
        0.0,
        0.3,
        1.0,
        12.5,
        1e30,
        -3.4e38,
        f32::NAN,
    ] {
        let terms = shap_edge_terms(alpha, &h, &u);
        for i in 0..8 {
            let expected = scalar(alpha, h[i], u[i]);
            assert_eq!(
                terms[i].to_bits(),
                expected.to_bits(),
                "alpha {alpha} lane {i}: {} vs {expected}",
                terms[i]
            );
        }
    }
}

#[test]
fn shap_basis_kernels_match_the_scalar_expressions_bit_for_bit() {
    // The child basis `c · (1 + α·u)` and the divided-out factor
    // `c / (1 + α·u)` must equal the per-lane scalar arithmetic exactly;
    // the division declines (`None`) whenever an input lane or denominator is
    // not finite, where the caller redoes the lanes in `f64`.
    let denominator = |alpha: f32, u: f32| {
        if cfg!(target_arch = "aarch64") {
            alpha.mul_add(u, 1.0)
        } else {
            alpha * u + 1.0
        }
    };
    let u = [
        0.0,
        0.019_855_07,
        0.101_666_76,
        0.237_233_8,
        0.408_282_68,
        0.591_717_3,
        0.762_766_2,
        1.0,
    ];
    let finite = [1.5, -2.25, 1e-30, 3.0e38, -0.0, 1e-45, 7.0, 0.125];
    let mut non_finite = finite;
    non_finite[5] = f32::INFINITY;
    let mut nan = finite;
    nan[2] = f32::NAN;
    for alpha in [
        -1.0,
        -0.999_999,
        -0.5,
        0.0,
        0.3,
        1.0,
        12.5,
        1e30,
        -3.4e38,
        f32::NAN,
    ] {
        for c in [finite, non_finite, nan] {
            let scaled = shap_scaled_basis(alpha, &c, &u);
            for i in 0..8 {
                let expected = c[i] * denominator(alpha, u[i]);
                assert_eq!(
                    scaled[i].to_bits(),
                    expected.to_bits(),
                    "scaled alpha {alpha} lane {i}"
                );
            }
            let old: [f32; 8] = std::array::from_fn(|i| denominator(alpha, u[i]));
            let all_finite = c.iter().chain(&old).all(|v| v.is_finite());
            match shap_divided_basis(alpha, &c, &u) {
                Some(divided) => {
                    assert!(all_finite, "alpha {alpha}: divided a non-finite lane");
                    for i in 0..8 {
                        let expected = c[i] / old[i];
                        assert_eq!(
                            divided[i].to_bits(),
                            expected.to_bits(),
                            "divided alpha {alpha} lane {i}"
                        );
                    }
                }
                None => assert!(!all_finite, "alpha {alpha}: declined finite lanes"),
            }
        }
    }
}

#[test]
fn pointwise_metric_sums_are_close_to_scalar() {
    let preds: Vec<f32> = (0..4_103).map(|i| (i % 1_001) as f32 * 0.001).collect();
    let labels: Vec<f32> = (0..preds.len()).map(|i| (i % 2) as f32).collect();
    let weights = sawtooth(preds.len(), 17, 0.5, 0.0625);
    for weights in [None, Some(RowWeights::from(weights.as_slice()))] {
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
    for weights in [None, Some(RowWeights::from(weights.as_slice()))] {
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
        for weights in [None, Some(RowWeights::from(weight_values.as_slice()))] {
            let actual = tweedie_nloglik_sum(&preds, &labels, weights, rho);
            let expected = scalar::tweedie_nloglik(&preds, &labels, weights, rho, 0..preds.len());
            assert_sums_close(actual, expected, 3e-12);
        }
    }
}

#[test]
fn strided_row_weights_sum_like_their_repeated_cell_weights() {
    // Odd lengths and strides put rows across vector lanes and the tail.
    for (cells, stride) in [(4_103, 3), (4_101, 7), (4_100, 2)] {
        let preds = sawtooth(cells, 999, 0.001, 0.001);
        let labels: Vec<f32> = (0..cells).map(|i| (i % 2) as f32).collect();
        let rows = sawtooth(cells.div_ceil(stride), 17, 0.51, 0.061);
        let repeated: Vec<f32> = rows
            .iter()
            .flat_map(|&w| std::iter::repeat_n(w, stride))
            .take(cells)
            .collect();
        let strided = Some(RowWeights::new(&rows, stride));
        let flat = Some(RowWeights::from(repeated.as_slice()));
        let pairs = [
            (
                squared_error_sum(&preds, &labels, strided),
                squared_error_sum(&preds, &labels, flat),
            ),
            (
                absolute_error_sum(&preds, &labels, strided),
                absolute_error_sum(&preds, &labels, flat),
            ),
            (
                classification_error_sum(&preds, &labels, strided),
                classification_error_sum(&preds, &labels, flat),
            ),
            (
                log_loss_sum(&preds, &labels, strided),
                log_loss_sum(&preds, &labels, flat),
            ),
            (
                positive_nloglik_sum::<true>(&preds, &labels, strided),
                positive_nloglik_sum::<true>(&preds, &labels, flat),
            ),
            (
                positive_nloglik_sum::<false>(&preds, &labels, strided),
                positive_nloglik_sum::<false>(&preds, &labels, flat),
            ),
            (
                tweedie_nloglik_sum(&preds, &labels, strided, 1.5),
                tweedie_nloglik_sum(&preds, &labels, flat, 1.5),
            ),
        ];
        for (kernel, (strided, flat)) in pairs.into_iter().enumerate() {
            assert_eq!(
                strided.0.to_bits(),
                flat.0.to_bits(),
                "kernel {kernel} stride {stride}"
            );
            assert_eq!(
                strided.1.to_bits(),
                flat.1.to_bits(),
                "kernel {kernel} stride {stride}"
            );
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

/// The softmax gradient runs the architecture's specialized kernel for every
/// class count it covers, bit for bit, and the scalar rows otherwise.
#[test]
fn softmax_gradient_dispatch_selects_the_kernel_for_each_class_count() {
    for num_class in 2..=9usize {
        let rows = 37;
        let preds: Vec<f32> = (0..rows * num_class)
            .map(|i| ((i * 7919) % 211) as f32 / 23.0 - 4.0)
            .collect();
        let labels: Vec<f32> = (0..rows).map(|r| (r % num_class) as f32).collect();
        let weights: Vec<f32> = (0..rows).map(|r| 0.5 + (r % 3) as f32 * 0.25).collect();
        let mut dispatched = vec![GradPair::default(); preds.len()];
        softmax_gradient(
            &preds,
            &labels,
            Some(&weights),
            num_class,
            1e-16,
            &mut dispatched,
        );

        let mut expected = vec![GradPair::default(); preds.len()];
        let mut vector = false;
        #[cfg(target_arch = "aarch64")]
        if neon_available() {
            vector = true;
            // SAFETY: NEON was detected and the inputs form complete matrices
            // of `num_class` columns; each kernel matches its class count.
            unsafe {
                match num_class {
                    2 => aarch64::short_softmax_gradient::<2>(
                        &preds,
                        &labels,
                        Some(&weights),
                        1e-16,
                        &mut expected,
                    ),
                    3 => aarch64::short_softmax_gradient::<3>(
                        &preds,
                        &labels,
                        Some(&weights),
                        1e-16,
                        &mut expected,
                    ),
                    4 => aarch64::short_softmax_gradient::<4>(
                        &preds,
                        &labels,
                        Some(&weights),
                        1e-16,
                        &mut expected,
                    ),
                    8.. => aarch64::softmax_gradient(
                        &preds,
                        &labels,
                        Some(&weights),
                        num_class,
                        1e-16,
                        &mut expected,
                    ),
                    _ => vector = false,
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        if avx2_fma_available() && (num_class == 2 || num_class == 4) {
            vector = true;
            // SAFETY: AVX2/FMA were detected and the inputs form complete
            // matrices of `num_class` columns; each kernel matches its class
            // count.
            unsafe {
                if num_class == 2 {
                    x86_64::short_softmax_gradient::<2>(
                        &preds,
                        &labels,
                        Some(&weights),
                        1e-16,
                        &mut expected,
                    );
                } else {
                    x86_64::short_softmax_gradient::<4>(
                        &preds,
                        &labels,
                        Some(&weights),
                        1e-16,
                        &mut expected,
                    );
                }
            }
        }
        if !vector {
            softmax_gradient_rows_scalar(
                &preds,
                &labels,
                Some(&weights),
                1e-16,
                &mut expected,
                0..rows,
                num_class,
            );
        }
        let bits = |pairs: &[GradPair]| -> Vec<(u32, u32)> {
            pairs
                .iter()
                .map(|p| (p.grad.to_bits(), p.hess.to_bits()))
                .collect()
        };
        assert_eq!(
            bits(&dispatched),
            bits(&expected),
            "gradient, {num_class} classes"
        );
    }
}
