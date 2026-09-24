//! The crate's random numbers: [`Rng`], the seeded sequential generator
//! behind row, column, DART, fold and permutation sampling, and the
//! SplitMix64 mixing shared by the counter-based streams (`extra_trees` node
//! seeds, quantized stochastic rounding, `dist:*` split-direction draws) and
//! the per-block row-sampling seeds.

use std::ops::Range;

/// SplitMix64's state increment (the 64-bit golden ratio).
pub(crate) const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 finalizer: a bijective 64-bit mix with full avalanche.
#[inline]
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One SplitMix64 step: advance `z` by [`GOLDEN`], then [`mix64`].
#[inline]
pub(crate) fn splitmix64(z: u64) -> u64 {
    mix64(z.wrapping_add(GOLDEN))
}

/// A seeded xoshiro256++ generator (Blackman and Vigna, 2019), its state
/// expanded from a `u64` seed by SplitMix64 as the authors recommend. Not
/// cryptographic; the same seed always gives the same stream on every
/// platform.
#[derive(Debug, Clone)]
pub(crate) struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// The generator seeded with `seed`.
    pub(crate) fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0; 4];
        for word in &mut s {
            z = z.wrapping_add(GOLDEN);
            *word = mix64(z);
        }
        // Four consecutive SplitMix64 outputs are distinct (`mix64` is a
        // bijection), so at most one is zero: the state is never all zero.
        Rng { s }
    }

    /// The next 64 random bits.
    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform on `[0, 1)`: the top 24 bits, every value exactly
    /// representable.
    #[inline]
    pub(crate) fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform on `[0, 1)`: the top 53 bits, every value exactly
    /// representable.
    #[inline]
    pub(crate) fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform on `0..n` without modulo bias (Lemire, 2019: one multiply,
    /// and a division only in the rare rejection zone). `n` must be
    /// positive.
    #[inline]
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0, "empty sampling range");
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        if (m as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
            }
        }
        (m >> 64) as u64
    }

    /// Uniform on the non-empty `range`.
    #[inline]
    pub(crate) fn range(&mut self, range: Range<usize>) -> usize {
        debug_assert!(range.start < range.end, "empty sampling range");
        range.start + self.below((range.end - range.start) as u64) as usize
    }

    /// Shuffle `items` uniformly in place (Fisher–Yates).
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i as u64 + 1) as usize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xoshiro256++ reference output (the authors' `xoshiro256plusplus.c`)
    /// from the state `[1, 2, 3, 4]`.
    #[test]
    fn matches_the_reference_stream() {
        let mut rng = Rng { s: [1, 2, 3, 4] };
        let expected = [
            41_943_041,
            58_720_359,
            3_588_806_011_781_223,
            3_591_011_842_654_386,
            9_228_616_714_210_784_205,
        ];
        for e in expected {
            assert_eq!(rng.next_u64(), e);
        }
    }

    #[test]
    fn bounded_draws_are_unbiased_and_in_range() {
        let mut rng = Rng::new(7);
        let n = 6;
        let mut counts = [0u32; 6];
        for _ in 0..60_000 {
            counts[rng.below(n) as usize] += 1;
        }
        // Each count is Binomial(60000, 1/6): sd ≈ 91, so ±500 is > 5 sd.
        assert!(
            counts.iter().all(|&c| c.abs_diff(10_000) < 500),
            "{counts:?}"
        );
        // A range just above 2^63 lands in the rejection zone half the time.
        let big = (1u64 << 63) + 1;
        assert!((0..1000).all(|_| rng.below(big) < big));
        assert!((0..1000).all(|_| (5..8).contains(&rng.range(5..8))));
    }

    #[test]
    fn shuffle_is_a_uniform_permutation() {
        let mut rng = Rng::new(3);
        let mut first = [0u32; 4];
        for _ in 0..40_000 {
            let mut v = [0, 1, 2, 3];
            rng.shuffle(&mut v);
            let mut sorted = v;
            sorted.sort_unstable();
            assert_eq!(sorted, [0, 1, 2, 3]);
            first[v[0]] += 1;
        }
        assert!(first.iter().all(|&c| c.abs_diff(10_000) < 500), "{first:?}");
    }
}
