use super::*;

#[test]
fn split_prefilter_preserves_ties_and_sequential_epsilon() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return;
    }
    let reg = RegParams {
        alpha: 0.0,
        lambda: 1.0,
        min_child_weight: 1.0,
        max_delta_step: 0.0,
    };
    for length in [16, 17] {
        for increment in [0.0, 1e-7] {
            let mut histogram = vec![GradStats::default(); length];
            histogram[0] = GradStats::new(-1.0, 1.0);
            histogram[1] = GradStats::new(-increment, 1.0);
            histogram[2] = GradStats::new(1.0 + increment, 1.0);
            let total = GradStats::new(0.0, 3.0);
            for epsilon in [0.0, 1e-6, 1.0] {
                // SAFETY: NEON was detected; the kernel bounds its loads.
                let candidate = unsafe {
                    dense_unconstrained_best_split(&histogram, total, &reg, 0.0, epsilon)
                };
                let expected = if epsilon == 1.0 {
                    None
                } else if epsilon == 0.0 && increment > 0.0 {
                    Some(1)
                } else {
                    Some(0)
                };
                assert_eq!(candidate.map(|value| value.split_offset), expected);
            }
        }
    }
}

#[test]
fn vector_gain_matches_scalar_l1_thresholds() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return;
    }
    let gradients = [
        f64::NEG_INFINITY,
        -f64::MAX,
        -4.0,
        -2.0,
        -1.0,
        -f64::MIN_POSITIVE,
        -f64::from_bits(1),
        -0.0,
        0.0,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        1.0,
        2.0,
        4.0,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
        0.5,
    ];
    for alpha in [
        0.0,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1.0,
        2.0,
        f64::MAX,
    ] {
        for lambda in [0.0, 1.0] {
            let reg = RegParams {
                alpha,
                lambda,
                min_child_weight: 0.0,
                max_delta_step: 0.0,
            };
            for pair in gradients.chunks_exact(2) {
                let hessians = [0.5, 123.0];
                let mut actual = [0.0; 2];
                // SAFETY: NEON was detected and every array has two lanes.
                unsafe {
                    vst1q_f64(
                        actual.as_mut_ptr(),
                        unconstrained_gainq(
                            vld1q_f64(pair.as_ptr()),
                            vld1q_f64(hessians.as_ptr()),
                            &reg,
                        ),
                    );
                }
                for lane in 0..2 {
                    let expected = calc_gain(GradStats::new(pair[lane], hessians[lane]), &reg);
                    assert_eq!(actual[lane].to_bits(), expected.to_bits());
                }
            }
        }
    }
}

#[test]
fn finite_extrema_reject_exceptional_values_in_every_lane() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return;
    }
    for length in [8, 16, 17, 32, 33, 128, 131] {
        let values: Vec<f32> = (0..length).map(|index| (index % 13) as f32 - 6.0).collect();
        let expected = (
            values.iter().copied().fold(f32::INFINITY, f32::min),
            values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        );
        // SAFETY: NEON was detected, and the kernel bounds its own loads.
        assert_eq!(unsafe { finite_min_max(&values) }, Some(expected));
        for index in 0..length {
            for exceptional in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
                let mut input = values.clone();
                input[index] = exceptional;
                // SAFETY: NEON was detected, and all loads are bounded.
                assert_eq!(unsafe { finite_min_max(&input) }, None);
            }
        }
    }
}

#[test]
fn vector_log_is_accurate_across_metric_range() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return;
    }
    let mut values = Vec::new();
    for index in 0..20_000 {
        let exponent = -34.5 + index as f64 * (123.0 / 19_999.0);
        values.push(exponent.exp());
    }
    // Exercise the full reduced mantissa interval and adjacent f64 values
    // at the range-reduction boundary, including extreme metric exponents.
    for index in 0..20_000 {
        values.push(1.0 + index as f64 / 20_000.0);
    }
    for exponent in [-50, -27, -1, 0, 1, 64, 127, 128] {
        for mantissa in [1.0_f64, std::f64::consts::SQRT_2, 2.0] {
            for bits in [
                mantissa.to_bits() - 1,
                mantissa.to_bits(),
                mantissa.to_bits() + 1,
            ] {
                values.push(f64::from_bits(bits) * 2.0_f64.powi(exponent));
            }
        }
    }
    values.extend([0.5, 0.999_999_999, 1.0, 1.000_000_001, 2.0]);
    if values.len() % 2 != 0 {
        values.push(1.0);
    }

    for pair in values.chunks_exact(2) {
        let mut actual = [0.0; 2];
        // SAFETY: runtime detection proves NEON support, and both arrays
        // contain two f64 lanes.
        unsafe {
            let input = vld1q_f64(pair.as_ptr());
            vst1q_f64(actual.as_mut_ptr(), logq_f64(input));
        }
        for (&actual, &value) in actual.iter().zip(pair) {
            let expected = value.ln();
            assert!(
                (actual - expected).abs() <= 2.5e-14,
                "SIMD log({value})={actual} differs from scalar {expected}"
            );
        }
    }
}

#[test]
fn vector_exp_f64_is_accurate_across_tweedie_range() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return;
    }
    let values: Vec<f64> = (0..20_000)
        .map(|index| -90.0 + index as f64 * (180.0 / 19_999.0))
        .collect();
    for pair in values.chunks_exact(2) {
        let mut actual = [0.0; 2];
        // SAFETY: runtime detection proves NEON support, and both arrays
        // contain exactly two f64 lanes.
        unsafe {
            let input = vld1q_f64(pair.as_ptr());
            vst1q_f64(actual.as_mut_ptr(), expq_f64(input));
        }
        for (&actual, &value) in actual.iter().zip(pair) {
            let expected = value.exp();
            let relative = ((actual - expected) / expected).abs();
            assert!(
                relative <= 2e-15,
                "SIMD exp({value})={actual} differs from scalar {expected} by {relative}"
            );
        }
    }
}
