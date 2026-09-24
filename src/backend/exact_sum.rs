//! The exactness domain of the Metal backend's histogram sums.
//!
//! The CPU accumulates each histogram bin as a chain of `f64` additions in
//! row order; the GPU splits the rows into slices, accumulates each slice in
//! a double-float `(hi, lo)` pair of `f32`s, and merges the slice pairs in
//! order. The two agree bit for bit exactly when neither rounds, which
//! [`SumDomain::sums_exact`] decides from two statistics of the gradient
//! slice: the largest magnitude and the coarsest power of two every value
//! is a multiple of.
//!
//! # Proof
//!
//! Let every value be a multiple of `u = 2^grain` with `|value| <= max`,
//! and let a node sum at most `n` values, `B = n * max`. Every partial sum
//! either path forms (a prefix of one bin's rows on the CPU; a prefix of a
//! slice's rows, or of the slice totals, on the GPU) sums a subset of the
//! values, so it is a multiple of `u` of magnitude at most `B`. Multiples of
//! `u` up to `2^24 u` are `f32`s and up to `2^53 u` are `f64`s. By induction
//! over the operations, while every earlier one was exact:
//!
//! - CPU: each `f64` add is exact for `B <= 2^53 u`, so a bin holds the
//!   exact sum `S`.
//! - GPU scan (`DF_ADD`: two-sum `s + e = hi + x`, `t = lo + e`, fast
//!   two-sum of `(s, t)`): the pair is normalized, `|lo| <= ulp(hi)/2 <=
//!   2^-24 |hi|`, and the two-sums are exact, so the one candidate for
//!   rounding is `t`, a multiple of `u` with `|t| <= 2^-24 (|hi| + |s|) <=
//!   2^-23 B (1 + 2^-22)`: exact for `B <= 2^47 u / (1 + 2^-22)`. The fast
//!   two-sum is exact because `|t| <= |s|` or `s = 0`: if `|s| >= ulp(hi)`,
//!   `|t| <= |s|/2 + 2^-24 |s|`; otherwise `hi + x` cancels, so (Sterbenz)
//!   `e = 0`, `t = lo`, and `s`, a nonzero multiple of `ulp(hi)/2`, is at
//!   least `|lo|`.
//! - GPU merge (`DF_ADD_DF`: two-sum of the highs, `t = (alo + blo) + e`,
//!   two-sum of `(s, t)`): both adds are bounded by `2^-24 (|ahi| + |bhi| +
//!   |s|) <= 3 * 2^-24 B (1 + 2^-22)`, exact for `B <= 2^48 u / (3 (1 +
//!   2^-22))`, about `2^46.4 u`.
//! - The final `f64(hi) + f64(lo) = S` is exact for `B <= 2^53 u`.
//! - Overflow: every `f32` intermediate (including inside the two-sums)
//!   stays below `2 B (1 + 2^-22)`, finite for `B <= 2^126`.
//! - Subnormals, which a GPU may flush to zero: for `u >= 2^-126` every
//!   nonzero multiple of `u` is a normal `f32`.
//!
//! So `B <= 2^46 u` suffices; [`DOMAIN_BITS`] keeps one bit of margin below
//! it, which also absorbs the `f64` rounding of `max * n`.

/// `log2(B / u)` up to which both paths are exact (see the module docs).
const DOMAIN_BITS: i32 = 45;
/// `log2` of the largest `B` whose `f32` intermediates cannot overflow.
const OVERFLOW_BITS: i32 = 126;
/// The finest grain whose multiples are all normal `f32`s (`f32`'s
/// smallest normal exponent).
const MIN_GRAIN: i32 = -126;

/// The magnitude statistics of one gradient component (all gradients, or
/// all Hessians, of a slice) that decide whether its histogram sums are
/// exact on both the CPU and the GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SumDomain {
    /// Largest magnitude (0 when every value is zero).
    max: f32,
    /// Exponent `k` of the largest `2^k` every nonzero value is a multiple
    /// of (`i32::MAX` when every value is zero).
    grain: i32,
    /// Whether every value is finite (no NaN, no infinity).
    finite: bool,
}

impl SumDomain {
    /// The statistics of no values: every sum is (vacuously) exact.
    pub(crate) const EMPTY: Self = Self {
        max: 0.0,
        grain: i32::MAX,
        finite: true,
    };

    /// The statistics of `values`.
    pub(crate) fn of(values: impl IntoIterator<Item = f32>) -> Self {
        values.into_iter().fold(Self::EMPTY, |domain, v| {
            if !v.is_finite() {
                Self {
                    finite: false,
                    ..domain
                }
            } else if v == 0.0 {
                domain
            } else {
                Self {
                    max: domain.max.max(v.abs()),
                    grain: domain.grain.min(grain(v)),
                    finite: domain.finite,
                }
            }
        })
    }

    /// Whether every sum of at most `n` of the values, in any order and
    /// grouping the GPU uses, is computed exactly by both the CPU's `f64`
    /// chain and the GPU's double-float accumulation, so that the two agree
    /// bit for bit. Non-finite values are never exact (NaN payloads and
    /// infinities do not carry through the double-float pair), nor are
    /// grains below `2^-126` (subnormal intermediates).
    pub(crate) fn sums_exact(&self, n: usize) -> bool {
        if !self.finite {
            return false;
        }
        if self.grain == i32::MAX {
            return true;
        }
        if self.grain < MIN_GRAIN {
            return false;
        }
        // `max * n` rounds at most 2^-53 relatively, inside the margin.
        let bound = f64::from(self.max) * n as f64;
        bound <= pow2(OVERFLOW_BITS) && bound <= pow2(DOMAIN_BITS + self.grain)
    }
}

/// The exponent of the lowest set bit of a finite nonzero `f32`: the value
/// is an odd multiple of `2^grain`.
fn grain(v: f32) -> i32 {
    let bits = v.to_bits();
    let biased = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    let (significand, exponent) = if biased == 0 {
        (mantissa, -149)
    } else {
        (mantissa | 0x80_0000, biased - 150)
    };
    exponent + significand.trailing_zeros() as i32
}

/// `2^k` as an `f64`, exactly (`k` in the normal range).
fn pow2(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model of the GPU kernels' arithmetic (`DF_ADD`, `DF_ADD_DF`, and
    /// the final `f64` sum in `backend/metal.rs`), in the same `f32`
    /// operation order: each slice of `slice_len` values accumulates from
    /// `(0, 0)`, then the slice pairs merge in order.
    fn gpu_sum(values: &[f32], slice_len: usize) -> f64 {
        fn two_sum(a: f32, b: f32) -> (f32, f32) {
            let s = a + b;
            let bb = s - a;
            (s, (a - (s - bb)) + (b - bb))
        }
        fn fast_two_sum(a: f32, b: f32) -> (f32, f32) {
            let s = a + b;
            (s, b - (s - a))
        }
        let (mut hi, mut lo) = (0.0f32, 0.0f32);
        for slice in values.chunks(slice_len) {
            let (mut shi, mut slo) = (0.0f32, 0.0f32);
            for &x in slice {
                let (s, e) = two_sum(shi, x);
                (shi, slo) = fast_two_sum(s, slo + e);
            }
            let (s, e) = two_sum(hi, shi);
            (hi, lo) = two_sum(s, (lo + slo) + e);
        }
        f64::from(hi) + f64::from(lo)
    }

    /// The CPU's sequential `f64` chain.
    fn cpu_sum(values: &[f32]) -> f64 {
        values.iter().fold(0.0f64, |s, &x| s + f64::from(x))
    }

    #[test]
    fn grain_is_the_lowest_set_bit() {
        assert_eq!(grain(1.0), 0);
        assert_eq!(grain(-3.0), 0);
        assert_eq!(grain(0.75), -2);
        assert_eq!(grain(2f32.powi(50)), 50);
        assert_eq!(grain(8_388_609.0), 0); // 2^23 + 1
        assert_eq!(grain(f32::from_bits(1)), -149); // smallest subnormal
        assert_eq!(grain(f32::from_bits(0x40_0000)), -127); // 2^-127
        assert_eq!(grain(1.0 + f32::EPSILON), -23);
    }

    /// The review's triggering input: in 128-row slices, the six values
    /// sum to 1 exactly in `f64` (and 64 slices to 64), but need 51
    /// significand bits, beyond the double-float's reach; the domain check
    /// sends it to the CPU.
    #[test]
    fn wide_dynamic_range_is_outside_the_domain() {
        let six = [
            2f32.powi(50),
            2f32.powi(26),
            2f32.powi(23) + 1.0,
            -(2f32.powi(50)),
            -(2f32.powi(26)),
            -(2f32.powi(23)),
        ];
        let mut values = Vec::new();
        for _ in 0..64 {
            values.extend(six);
            values.extend([0.0; 122]);
        }
        assert_eq!(cpu_sum(&values), 64.0);
        let domain = SumDomain::of(values.iter().copied());
        assert!(!domain.sums_exact(values.len()));
        assert!(!domain.sums_exact(6));
    }

    /// The boundary: `n * max` up to `2^45` grains is in, one more row is
    /// out; zeros, whole-number grains, and non-finite values.
    #[test]
    fn domain_boundary() {
        // Hessians of squared error: all 1.0, grain 2^0.
        let ones = SumDomain::of([1.0f32; 3]);
        assert!(ones.sums_exact(1 << 45));
        assert!(!ones.sums_exact((1 << 45) + 1));
        // An odd multiple of 2^-10 decides the grain; the max is 4.
        let mixed = SumDomain::of([4.0, -0.5, 3.0 * 2f32.powi(-10), 0.0]);
        assert!(mixed.sums_exact(1 << 33));
        assert!(!mixed.sums_exact((1 << 33) + 1));
        assert!(SumDomain::of([0.0f32, -0.0]).sums_exact(usize::MAX));
        assert!(SumDomain::EMPTY.sums_exact(1 << 40));
        assert!(!SumDomain::of([1.0, f32::NAN]).sums_exact(1));
        assert!(!SumDomain::of([f32::INFINITY]).sums_exact(1));
        // Grains below 2^-126 could produce subnormal intermediates.
        assert!(SumDomain::of([2f32.powi(-126)]).sums_exact(2));
        assert!(!SumDomain::of([f32::from_bits(1)]).sums_exact(1));
        assert!(!SumDomain::of([f32::MIN_POSITIVE + f32::from_bits(1)]).sums_exact(1));
        // Overflow: coarse grains, but `n * max` past 2^126.
        let huge = SumDomain::of([2f32.powi(120)]);
        assert!(huge.sums_exact(64));
        assert!(!huge.sums_exact(65));
    }

    /// Inside the domain the modeled GPU arithmetic equals the CPU chain
    /// bit for bit, at the domain's edge: mostly-positive large values
    /// (partial sums near `n * max`, so the pair's low word collects large
    /// rounding errors) mixed with odd small ones. The unrenormalized
    /// double-float the kernels used before loses low bits in about one
    /// check in twenty here.
    #[test]
    fn in_domain_sums_match_the_cpu_chain() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for case in 0..200 {
            let n = 4096 + (next() % 4096) as usize;
            // Largest magnitude 2^top with grain 2^0: `n * 2^top <= 2^45`.
            let top = 45 - (n as f64).log2().ceil() as i32;
            let values: Vec<f32> = (0..n)
                .map(|_| {
                    let r = next();
                    let sign = if (r >> 40) % 8 == 0 { -1.0 } else { 1.0 };
                    let v = match (r >> 1) % 4 {
                        // Large values with full 24-bit significands.
                        0 | 1 => ((r >> 8) % (1 << 24)) as f32 * 2f32.powi(top - 24),
                        // Odd small values.
                        2 => ((r >> 8) % 1000 * 2 + 1) as f32,
                        _ => 2f32.powi(top),
                    };
                    sign * v
                })
                .collect();
            let domain = SumDomain::of(values.iter().copied());
            assert!(domain.sums_exact(n), "case {case} must be in the domain");
            for slice_len in [1, 7, 128, 1024, n] {
                assert_eq!(
                    gpu_sum(&values, slice_len),
                    cpu_sum(&values),
                    "case {case}, slices of {slice_len}"
                );
            }
        }
    }
}
