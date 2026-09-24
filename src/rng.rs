//! SplitMix64 mixing shared by the crate's counter-based random streams
//! (`extra_trees` node seeds, quantized stochastic rounding, `dist:*`
//! split-direction draws) and the per-block row-sampling seeds.

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
