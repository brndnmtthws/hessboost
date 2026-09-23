use super::super::neon_available;
use super::*;
#[test]
fn vector_log_is_accurate_across_metric_range() {
    if !neon_available() {
        return;
    }
    let mut values = Vec::new();
    for index in 0..20_000 {
        let exponent = -34.5 + f64::from(index) * (123.0 / 19_999.0);
        values.push(exponent.exp());
    }
    // Exercise the full reduced mantissa interval and adjacent f64 values
    // at the range-reduction boundary, including extreme metric exponents.
    for index in 0..20_000 {
        values.push(1.0 + f64::from(index) / 20_000.0);
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

    for pair in values.as_chunks::<2>().0 {
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
    if !neon_available() {
        return;
    }
    let values: Vec<f64> = (0..20_000)
        .map(|index| -90.0 + f64::from(index) * (180.0 / 19_999.0))
        .collect();
    for pair in values.as_chunks::<2>().0 {
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
