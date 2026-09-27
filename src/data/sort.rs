//! Float sort keys and the stable LSD radix sort shared by cut
//! construction (`quantile`) and the weighted quantile sketch (`sketch`).

/// Below this length the comparison sort beats the radix passes' fixed cost.
const RADIX_MIN_LEN: usize = 2048;
const RADIX_BITS: u32 = 11;
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;

/// Monotone map from `f32` to `u32` under [`f32::total_cmp`] order:
/// negative values reverse all bits, others set the sign bit.
#[inline]
pub(super) fn sort_key(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Inverse of [`sort_key`].
#[inline]
pub(super) fn unsort_key(key: u32) -> f32 {
    f32::from_bits(if key & 0x8000_0000 != 0 {
        key & !0x8000_0000
    } else {
        !key
    })
}

/// Buffers [`radix_sort`] reuses across calls.
#[derive(Default)]
pub(super) struct RadixScratch<T> {
    spare: Vec<T>,
    /// Bucket counts of the three passes, `3 × RADIX_BUCKETS`.
    counts: Vec<u32>,
}

/// Stable three-pass LSD radix sort of `items` by `key` (11, 11, and 10 bit
/// digits; a digit every key shares is skipped), ascending in the key's
/// unsigned order. Inputs of `2^32` items or more (beyond the `u32` bucket
/// counts) take a comparison sort by the same key instead.
pub(super) fn radix_sort<T: Copy + Default>(
    items: &mut Vec<T>,
    scratch: &mut RadixScratch<T>,
    key: impl Fn(&T) -> u32,
) {
    let n = items.len();
    if n < 2 {
        return;
    }
    let Ok(n32) = u32::try_from(n) else {
        items.sort_by_key(key);
        return;
    };
    let RadixScratch { spare, counts } = scratch;
    // Bucket counts for all passes in one sweep.
    counts.clear();
    counts.resize(3 * RADIX_BUCKETS, 0);
    for item in items.iter() {
        let k = key(item);
        for (pass, count) in counts
            .as_chunks_mut::<RADIX_BUCKETS>()
            .0
            .iter_mut()
            .enumerate()
        {
            count[((k >> (RADIX_BITS * pass as u32)) & (RADIX_BUCKETS as u32 - 1)) as usize] += 1;
        }
    }
    // Every slot of the spare buffer is written before it is read, so only
    // slots it gains are initialized.
    spare.resize(n, T::default());
    // Passes ping-pong between the two buffers; track which one holds the data.
    let mut in_spare = false;
    for (pass, count) in counts
        .as_chunks_mut::<RADIX_BUCKETS>()
        .0
        .iter_mut()
        .enumerate()
    {
        // A pass whose digit is constant across the input is a no-op.
        if count.contains(&n32) {
            continue;
        }
        let mut offset = 0;
        for c in count.iter_mut() {
            let start = offset;
            offset += *c;
            *c = start;
        }
        let shift = RADIX_BITS * pass as u32;
        let (src, dst) = if in_spare {
            (&*spare, &mut *items)
        } else {
            (&*items, &mut *spare)
        };
        for item in src {
            let bucket = ((key(item) >> shift) & (RADIX_BUCKETS as u32 - 1)) as usize;
            dst[count[bucket] as usize] = *item;
            count[bucket] += 1;
        }
        in_spare = !in_spare;
    }
    if in_spare {
        std::mem::swap(items, spare);
    }
}

/// Sort `values` ascending by total order. Column sorts dominate cut
/// construction, so long inputs use [`radix_sort`] on [`sort_key`]
/// (identical order to `sort_unstable_by(f32::total_cmp)`), with `scratch`
/// reused across columns.
pub(super) fn sort_values(values: &mut Vec<f32>, scratch: &mut RadixScratch<f32>) {
    if values.len() < RADIX_MIN_LEN {
        values.sort_unstable_by(f32::total_cmp);
    } else {
        radix_sort(values, scratch, |&v| sort_key(v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_sort_matches_total_order() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for n in [RADIX_MIN_LEN, RADIX_MIN_LEN + 1, 10_007, 65_536] {
            let mut values: Vec<f32> = (0..n)
                .map(|i| match i % 11 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f32::MAX,
                    3 => f32::MIN,
                    4 => f32::MIN_POSITIVE,
                    5 => -f32::MIN_POSITIVE,
                    _ => (next() as f32 / u64::MAX as f32 - 0.5) * 1e6,
                })
                .collect();
            let mut expected = values.clone();
            expected.sort_unstable_by(f32::total_cmp);
            sort_values(&mut values, &mut RadixScratch::default());
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&values), bits(&expected));
        }
    }
}
