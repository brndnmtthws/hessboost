//! Float sort keys and the stable LSD radix sort shared by cut
//! construction (`quantile`), the weighted quantile sketch (`sketch`), the
//! quantile objective's intercept ([`stable_order`]), and exact quantile
//! leaves (`training::exact_leaves`).

/// Below this length the comparison sort beats the radix passes' fixed cost.
const RADIX_MIN_LEN: usize = 2048;
const RADIX_BITS: u32 = 11;
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;

/// Monotone map from `f32` to `u32` under [`f32::total_cmp`] order:
/// negative values reverse all bits, others set the sign bit.
#[inline]
pub(crate) fn sort_key(value: f32) -> u32 {
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
pub(crate) struct RadixScratch<T> {
    spare: Vec<T>,
    /// Bucket counts of the three passes, `3 × RADIX_BUCKETS`.
    counts: Vec<u32>,
}

/// Stable three-pass LSD radix sort of `items` by `key` (11, 11, and 10 bit
/// digits; a digit every key shares is skipped), ascending in the key's
/// unsigned order. Inputs of `2^32` items or more (beyond the `u32` bucket
/// counts) take a comparison sort by the same key instead.
pub(crate) fn radix_sort<T: Copy + Default>(
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
pub(crate) fn sort_values(values: &mut Vec<f32>, scratch: &mut RadixScratch<f32>) {
    if values.len() < RADIX_MIN_LEN {
        values.sort_unstable_by(f32::total_cmp);
    } else {
        radix_sort(values, scratch, |&v| sort_key(v));
    }
}

/// Sort `items` stably, ascending by `value` in total order (ties keep their
/// order): [`radix_sort`] on [`sort_key`] for long inputs, a stable
/// comparison sort by the same key below [`RADIX_MIN_LEN`].
pub(crate) fn sort_by_value<T: Copy + Default>(
    items: &mut Vec<T>,
    scratch: &mut RadixScratch<T>,
    value: impl Fn(&T) -> f32,
) {
    if items.len() < RADIX_MIN_LEN {
        items.sort_by_key(|item| sort_key(value(item)));
    } else {
        radix_sort(items, scratch, |item| sort_key(value(item)));
    }
}

/// Whether a small, evenly spaced sample suggests a long, unordered,
/// high-cardinality column. Comparison sorting exploits ordered runs and
/// repeated keys; radix sorting instead pays for full-column scatter passes.
/// This heuristic chooses only the algorithm, never the resulting stable
/// total order. It deliberately favors comparison sorting on uncertainty.
fn sample_prefers_radix<T>(items: &[T], value: impl Fn(&T) -> f32) -> bool {
    const SAMPLE: usize = 128;
    const MAX_DIRECTION_CHANGES: usize = 8;
    if items.len() < RADIX_MIN_LEN {
        return false;
    }
    // 128 samples detect repeats more reliably than 64 for columns with
    // hundreds of distinct values, without allocating or scanning the column.
    let step = (items.len() - 1) / (SAMPLE - 1);
    let remainder = (items.len() - 1) % (SAMPLE - 1);
    let mut keys: [u32; SAMPLE] = std::array::from_fn(|i| {
        // Equivalent to i * (len - 1) / (SAMPLE - 1), without overflow.
        let at = i * step + i * remainder / (SAMPLE - 1);
        sort_key(value(&items[at]))
    });
    let mut previous = std::cmp::Ordering::Equal;
    let mut changes = 0;
    for pair in keys.windows(2) {
        let direction = pair[0].cmp(&pair[1]);
        if direction != std::cmp::Ordering::Equal {
            changes += usize::from(previous != std::cmp::Ordering::Equal && previous != direction);
            previous = direction;
        }
    }
    // Few direction changes suggest long monotone runs, including descending
    // or organ-pipe inputs, that the comparison sort handles adaptively.
    if changes <= MAX_DIRECTION_CHANGES {
        return false;
    }
    keys.sort_unstable();
    !keys.windows(2).any(|pair| pair[0] == pair[1])
}

/// Stable total-order column sort, retaining the comparison sort's adaptive
/// behavior on small, ordered, and sampled low-cardinality inputs. Unordered
/// columns whose sample has no repeated key take the stable radix path.
/// Equal keys retain their input order under either algorithm.
pub(crate) fn sort_column_by_value<T: Copy + Default>(
    items: &mut Vec<T>,
    scratch: &mut RadixScratch<T>,
    value: impl Fn(&T) -> f32,
) {
    if sample_prefers_radix(items, &value) {
        radix_sort(items, scratch, |item| sort_key(value(item)));
    } else {
        items.sort_by(|a, b| value(a).total_cmp(&value(b)));
    }
}

/// The rows of finite `values` in ascending order under `<`, rows of equal
/// values (`-0.0` and `0.0` included) in row order: what a stable
/// comparison sort by `partial_cmp` gives (XGBoost `StableSort` with
/// `operator<`). Long inputs take [`radix_sort`] on [`sort_key`] with both
/// zeros mapped to `0.0`'s key.
pub(crate) fn stable_order(values: &[f32]) -> Vec<usize> {
    let key = |value: f32| sort_key(if value == 0.0 { 0.0 } else { value });
    if values.len() < RADIX_MIN_LEN || u32::try_from(values.len()).is_err() {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by_key(|&row| key(values[row]));
        return order;
    }
    // The length fits in `u32` (checked above).
    let mut rows: Vec<(u32, u32)> = (0u32..)
        .zip(values)
        .map(|(row, &value)| (key(value), row))
        .collect();
    radix_sort(&mut rows, &mut RadixScratch::default(), |&(key, _)| key);
    rows.into_iter().map(|(_, row)| row as usize).collect()
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

    /// Sorting weighted pairs retains the comparison sort's tie order,
    /// including signed zeros and non-finite values under total order.
    #[test]
    fn sorted_pairs_preserve_stable_weight_order() {
        let bits = |pairs: &[(f32, f32)]| {
            pairs
                .iter()
                .map(|&(v, w)| [v.to_bits(), w.to_bits()])
                .collect::<Vec<_>>()
        };
        for n in [RADIX_MIN_LEN - 1, RADIX_MIN_LEN, 10_007] {
            let mut pairs: Vec<(f32, f32)> = (0..n)
                .map(|i| {
                    let value = match i % 11 {
                        0 => -0.0,
                        1 => 0.0,
                        2 => f32::NEG_INFINITY,
                        3 => f32::INFINITY,
                        4 => f32::from_bits(0x7fc0_0001),
                        5 => f32::from_bits(0xffc0_0001),
                        _ => ((i * 7_919) % 97) as f32 - 48.0,
                    };
                    (value, i as f32)
                })
                .collect();
            let mut expected = pairs.clone();
            expected.sort_by(|a, b| a.0.total_cmp(&b.0));
            sort_by_value(&mut pairs, &mut RadixScratch::default(), |&(v, _)| v);
            assert_eq!(bits(&pairs), bits(&expected), "{n} pairs");
        }
    }

    #[test]
    fn adaptive_column_sort_selects_ordered_and_low_cardinality_routes() {
        let n = 10_007;
        let value = |v: &f32| *v;
        let mut ordered: Vec<f32> = (0..n).map(|i| i as f32).collect();
        assert!(!sample_prefers_radix(&ordered, value));
        ordered.reverse();
        assert!(!sample_prefers_radix(&ordered, value));
        ordered.reverse();
        ordered.swap(n / 3, n / 3 + 1);
        ordered.swap(2 * n / 3, 2 * n / 3 + 1);
        assert!(!sample_prefers_radix(&ordered, value));
        assert!(!sample_prefers_radix(&vec![0.0; n], value));
        let two_runs: Vec<f32> = (0..n).map(|i| ((i + n / 2) % n) as f32).collect();
        assert!(!sample_prefers_radix(&two_runs, value));
        let organ_pipe: Vec<f32> = (0..n)
            .map(|i| {
                if i < n / 2 {
                    (2 * i) as f32
                } else {
                    (2 * (n - i) - 1) as f32
                }
            })
            .collect();
        assert!(!sample_prefers_radix(&organ_pipe, value));
        for cardinality in [64, 256, 1_024] {
            let values: Vec<f32> = (0..n).map(|i| ((i * 7_919) % cardinality) as f32).collect();
            assert!(!sample_prefers_radix(&values, value), "{cardinality} keys");
        }
        let mut rng = crate::rng::Rng::new(419);
        for len in [RADIX_MIN_LEN - 1, RADIX_MIN_LEN, n] {
            let mut shuffled: Vec<f32> = (0..len).map(|i| i as f32).collect();
            for i in (1..len).rev() {
                let other = rng.range(0..i + 1);
                shuffled.swap(i, other);
            }
            assert_eq!(sample_prefers_radix(&shuffled, value), len >= RADIX_MIN_LEN);
        }
    }

    #[test]
    fn adaptive_column_sort_preserves_comparison_bits_on_both_routes() {
        let bits = |pairs: &[(f32, f32)]| {
            pairs
                .iter()
                .map(|&(v, w)| [v.to_bits(), w.to_bits()])
                .collect::<Vec<_>>()
        };
        let mut rng = crate::rng::Rng::new(941);
        for n in [RADIX_MIN_LEN - 1, RADIX_MIN_LEN, 10_007] {
            let mut unique: Vec<f32> = (0..n).map(|i| i as f32 - n as f32 / 2.0).collect();
            for i in (1..n).rev() {
                let other = rng.range(0..i + 1);
                unique.swap(i, other);
            }
            for case in 0..5 {
                let mut pairs: Vec<(f32, f32)> = (0..n)
                    .map(|i| {
                        let v = match case {
                            0 => unique[i],
                            1 => i as f32,
                            2 => (n - i) as f32,
                            _ => match i % 7 {
                                0 => -0.0,
                                1 => 0.0,
                                2 if case == 4 => f32::from_bits(0x7fc0_0001),
                                3 if case == 4 => f32::NEG_INFINITY,
                                _ => ((i * 7_919) % 97) as f32 - 48.0,
                            },
                        };
                        (v, [0.0, 1e-20, 0.125, 1.0, 3.5, 1e8][i % 6])
                    })
                    .collect();
                let mut expected = pairs.clone();
                expected.sort_by(|a, b| a.0.total_cmp(&b.0));
                sort_column_by_value(&mut pairs, &mut RadixScratch::default(), |&(v, _)| v);
                assert_eq!(bits(&pairs), bits(&expected), "{n} pairs, case {case}");
            }
        }
    }

    /// Above and below the radix threshold, the order is the stable
    /// `partial_cmp` sort's, with the two zeros tied and many duplicates.
    #[test]
    fn stable_order_matches_the_stable_comparison_sort() {
        for n in [RADIX_MIN_LEN - 1, RADIX_MIN_LEN, 10_007] {
            let values: Vec<f32> = (0..n)
                .map(|i| match i % 7 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => -1.5,
                    _ => ((i * 7919) % 97) as f32 - 48.0,
                })
                .collect();
            let mut expected: Vec<usize> = (0..n).collect();
            expected.sort_by(|&l, &r| values[l].partial_cmp(&values[r]).unwrap());
            assert_eq!(stable_order(&values), expected, "{n} values");
        }
    }
}
