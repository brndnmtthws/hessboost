//! A split node's children: its rows routed by the split, and both
//! children's histograms from one built child and the parent.

use super::SplitLocation;
use super::shared::rayon_available;
use crate::data::ghist::{BinIndex, Bins, Column, Columns, GHistIndex, missing_bit};
use crate::data::quantile::HistCuts;
use crate::objective::GradPair;
use crate::tree::SplitRule;
use crate::tree::hist::{Histogram, HistogramBackend, subtract_in_place, zeroed};
use rayon::prelude::*;

/// Rows per parallel partition chunk. Large nodes near the root are routed in
/// row-order chunks whose halves are concatenated, so the output order matches
/// the sequential loop exactly.
const PARTITION_CHUNK_ROWS: usize = 16_384;

/// Split `rows` (kept in order) into the rows for which `$go_left` (an
/// expression of the row `$r: u32`) holds and the others. Every row is
/// written to both output slots and only the matching length advances,
/// keeping the loop free of data-dependent branches; the outputs are written
/// into spare capacity, so neither buffer is zero-filled first. A macro, not
/// a function taking a closure: the predicate is expanded into the loop, so
/// the column it reads stays in registers (measured: a closure argument adds
/// loads to the dense partition loop).
macro_rules! route_rows {
    ($rows:expr, |$r:ident| $go_left:expr) => {
        route_rows!($rows, [], |$r| $go_left)
    };
    ($rows:expr, [$($copy:ident),*], |$r:ident| $go_left:expr) => {{
        let route = |rows: &[u32]| {
            let n = rows.len();
            let mut left: Vec<u32> = Vec::with_capacity(n);
            let mut right: Vec<u32> = Vec::with_capacity(n);
            let (mut nl, mut nr) = (0usize, 0usize);
            {
                // Copied into locals so they stay in registers across the
                // loop's stores.
                $(let $copy = $copy;)*
                let (lp, rp) = (left.as_mut_ptr(), right.as_mut_ptr());
                for &$r in rows {
                    let go_left: bool = $go_left;
                    // SAFETY: `nl + nr` rows were routed before this one, so
                    // `nl, nr < n`, the reserved capacity of both buffers.
                    unsafe {
                        lp.add(nl).write($r);
                        rp.add(nr).write($r);
                    }
                    nl += usize::from(go_left);
                    nr += usize::from(!go_left);
                }
            }
            // SAFETY: `nl + nr == n` and each side's slot `k` was written at
            // the iteration where its length was `k`, so `left[..nl]` and
            // `right[..nr]` are initialized and within the reserved capacity.
            unsafe {
                left.set_len(nl);
                right.set_len(nr);
            }
            (left, right)
        };
        let rows: &[u32] = $rows;
        if rows.len() < 2 * PARTITION_CHUNK_ROWS || !rayon_available() {
            route(rows)
        } else {
            let chunks: Vec<(Vec<u32>, Vec<u32>)> =
                rows.par_chunks(PARTITION_CHUNK_ROWS).map(route).collect();
            let mut left = Vec::with_capacity(chunks.iter().map(|(l, _)| l.len()).sum());
            let mut right = Vec::with_capacity(chunks.iter().map(|(_, r)| r.len()).sum());
            for (l, r) in chunks {
                left.extend_from_slice(&l);
                right.extend_from_slice(&r);
            }
            (left, right)
        }
    }};
}

/// How a split routes rows: its feature, where present values go, and where
/// missing values go. A categorical location's categories go left.
#[derive(Debug, Clone, Copy)]
pub(super) struct SplitRoute<'a> {
    pub(super) feature: u32,
    /// A histogram position ([`SplitPos::Bin`](super::SplitPos::Bin) or [`BelowBins`](super::SplitPos::BelowBins))
    /// or the categories routed left.
    pub(super) location: &'a SplitLocation,
    pub(super) default_left: bool,
}

impl<'a> SplitRoute<'a> {
    /// The tree node's test of this route, histogram positions resolved
    /// through `cuts`.
    pub(super) fn rule(self, cuts: &HistCuts) -> SplitRule<'a> {
        match self.location {
            SplitLocation::Numeric(pos) => {
                SplitRule::numeric(self.feature, pos.threshold(cuts), self.default_left)
            }
            SplitLocation::Categories(categories) => {
                SplitRule::categorical(self.feature, categories, self.default_left)
            }
        }
    }
}

/// Split `rows` (kept in order) into the rows `route` sends left and right.
pub(super) fn partition_rows(
    ghist: &GHistIndex,
    rows: &[u32],
    route: SplitRoute,
) -> (Vec<u32>, Vec<u32>) {
    match route.location {
        SplitLocation::Numeric(pos) => partition_numeric(ghist, rows, route, pos.bin()),
        SplitLocation::Categories(categories) => {
            partition_categorical(ghist, rows, route, categories)
        }
    }
}

/// [`partition_rows`] of a numeric split: present bins up to `split_bin`
/// (global) go left (`None`: none do, only missing values follow
/// `default_left`).
fn partition_numeric(
    ghist: &GHistIndex,
    rows: &[u32],
    route: SplitRoute,
    split_bin: Option<usize>,
) -> (Vec<u32>, Vec<u32>) {
    let feature = route.feature as usize;
    let Some(columns) = ghist.columns() else {
        return partition_by_bin(ghist, rows, route, |bin| {
            split_bin.is_some_and(|s| bin <= s)
        });
    };
    // Present local bins below `limit` go left.
    let limit = split_bin.map_or(0, |s| s + 1 - ghist.cuts().feature_bins(feature).0);
    let default_left = route.default_left;
    let below = move |b: usize| b < limit;
    match columns {
        Columns::U8(set) => route_column(rows, set.column(feature), default_left, below),
        Columns::U16(set) => route_column(rows, set.column(feature), default_left, below),
        Columns::U32(set) => route_column(rows, set.column(feature), default_left, below),
    }
}

/// [`partition_rows`] of a categorical split: present bins go left when they
/// hold one of `categories`.
#[inline(never)]
fn partition_categorical(
    ghist: &GHistIndex,
    rows: &[u32],
    route: SplitRoute,
    categories: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let cuts = ghist.cuts();
    let (fs, _) = cuts.feature_bins(route.feature as usize);
    let category_left = cuts.categories_left(route.feature as usize, categories);
    partition_by_bin(ghist, rows, route, |bin| category_left[bin - fs])
}

/// [`partition_rows`] without a numeric column fast path: a present bin
/// (global) of the split feature goes left when `present_left(bin)`, a
/// missing value follows `route.default_left`. Uses the feature's column
/// when the index keeps columns, else each row's stored bins.
#[inline(always)]
fn partition_by_bin(
    ghist: &GHistIndex,
    rows: &[u32],
    route: SplitRoute,
    present_left: impl Fn(usize) -> bool + Sync + Copy,
) -> (Vec<u32>, Vec<u32>) {
    let feature = route.feature as usize;
    let (fs, fe) = ghist.cuts().feature_bins(feature);
    let default_left = route.default_left;
    if let Some(columns) = ghist.columns() {
        let local_left = move |b: usize| present_left(fs + b);
        return match columns {
            Columns::U8(set) => route_column(rows, set.column(feature), default_left, local_left),
            Columns::U16(set) => route_column(rows, set.column(feature), default_left, local_left),
            Columns::U32(set) => route_column(rows, set.column(feature), default_left, local_left),
        };
    }
    // Only sparse indexes less than half full reach here: every other one
    // keeps its columns.
    let row_ptr = ghist.row_ptr();
    let row = |r: u32| row_ptr[r as usize]..row_ptr[r as usize + 1];
    match ghist.bins() {
        Bins::U16(bins) => route_rows!(rows, |r| {
            row_goes_left(&bins[row(r)], (fs, fe), default_left, present_left)
        }),
        Bins::U32(bins) => route_rows!(rows, |r| {
            row_goes_left(&bins[row(r)], (fs, fe), default_left, present_left)
        }),
    }
}

/// Partition rows on a split using the split feature's column: a present
/// row goes left when `present_left` holds for its local bin, a missing one
/// (its bitmap bit set) when `default_left` does. Rows ascend, so the
/// column and its bitmap are read as monotone streams the hardware
/// prefetcher follows.
#[inline(always)]
#[allow(
    clippy::needless_bitwise_bool,
    reason = "branch-free routing predicates keep the partition loop free of data-dependent branches"
)]
fn route_column<B: BinIndex>(
    rows: &[u32],
    column: Column<'_, B>,
    default_left: bool,
    present_left: impl Fn(usize) -> bool + Sync + Copy,
) -> (Vec<u32>, Vec<u32>) {
    let bins = column.bins;
    match column.missing {
        None => route_rows!(rows, [present_left, bins], |r| present_left(
            bins[r as usize].index()
        )),
        Some(bits) => route_rows!(rows, [present_left, bins, bits, default_left], |r| {
            let missing = missing_bit(bits, r as usize);
            (present_left(bins[r as usize].index()) & !missing) | (missing & default_left)
        }),
    }
}

/// Whether a sparse row whose stored bins are `row` goes left on the feature
/// with global bin range `(fs, fe)`: `present_left` of its first bin in the
/// range, or `default_left` when the feature is missing.
#[inline(always)]
fn row_goes_left<B: BinIndex>(
    row: &[B],
    (fs, fe): (usize, usize),
    default_left: bool,
    present_left: impl Fn(usize) -> bool,
) -> bool {
    row.iter()
        .map(|&b| b.index())
        .find(|b| (fs..fe).contains(b))
        .map_or(default_left, present_left)
}

/// Both children's histograms from `built`, the histogram of the child
/// `built_left` names: the sibling is `parent - built`, subtracted in place.
/// The parent's buffer is dead once the node expands, so the sibling reuses it
/// without a new allocation. Returns `(left, right)`.
pub(super) fn with_sibling(
    mut parent: Histogram,
    built: Histogram,
    built_left: bool,
) -> (Histogram, Histogram) {
    subtract_in_place(&mut parent, &built);
    if built_left {
        (built, parent)
    } else {
        (parent, built)
    }
}

/// Histograms of both children of a split node: the child with fewer rows is
/// built directly and the sibling derived by subtraction ([`with_sibling`]).
pub(super) fn child_histograms(
    backend: &dyn HistogramBackend,
    ghist: &GHistIndex,
    gpair: &[GradPair],
    left_rows: &[u32],
    right_rows: &[u32],
    parent: Histogram,
) -> (Histogram, Histogram) {
    let left_smaller = left_rows.len() <= right_rows.len();
    let mut small = zeroed(parent.len());
    backend.build(
        ghist,
        if left_smaller { left_rows } else { right_rows },
        gpair,
        &mut small,
    );
    with_sibling(parent, small, left_smaller)
}

/// Child histograms restricted to a tree's feature pool. The unused bins
/// may stay stale throughout the tree: every descendant samples this pool.
pub(super) fn child_histograms_for_features(
    backend: &dyn HistogramBackend,
    ghist: &GHistIndex,
    gpair: &[GradPair],
    left_rows: &[u32],
    right_rows: &[u32],
    mut parent: Histogram,
    features: &[u32],
) -> (Histogram, Histogram) {
    if features.len() == ghist.n_cols() {
        return child_histograms(backend, ghist, gpair, left_rows, right_rows, parent);
    }
    let left_smaller = left_rows.len() <= right_rows.len();
    let mut small = zeroed(parent.len());
    backend.build_features(
        ghist,
        if left_smaller { left_rows } else { right_rows },
        gpair,
        &mut small,
        features,
    );
    for &f in features {
        let (start, end) = ghist.cuts().feature_bins(f as usize);
        subtract_in_place(&mut parent[start..end], &small[start..end]);
    }
    if left_smaller {
        (small, parent)
    } else {
        (parent, small)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DMatrix;
    use crate::tree::builder::SplitPos;
    use crate::tree::builder::test_support::binned;
    use crate::tree::in_category_set;

    /// Every layout's routing (dense columns, columns with a missing
    /// sentinel, per-row scans of a sparser index's rows) sends each row
    /// where its own bin decides, for numeric and categorical splits in both
    /// default directions, serially and in parallel chunks.
    #[test]
    fn partition_matches_per_row_routing_in_every_layout() {
        use crate::data::FeatureType;
        // Two thirds of the rows are routed: more than two parallel chunks.
        let n = 3 * PARTITION_CHUNK_ROWS + 1_000;
        let features = 4;
        let pool = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        };
        let (serial_pool, parallel_pool) = (pool(1), pool(4));
        for (missing, layout) in [(0.0, "dense"), (0.2, "missing columns"), (0.8, "sparse")] {
            let mut rng = crate::rng::Rng::new(7);
            let x: Vec<f32> = (0..n * features)
                .map(|i| {
                    if rng.f32() < missing {
                        f32::NAN
                    } else if i % features < 2 {
                        (rng.next_u64() % 12) as f32
                    } else {
                        rng.f32()
                    }
                })
                .collect();
            let types = [
                FeatureType::Categorical,
                FeatureType::Categorical,
                FeatureType::Numerical,
                FeatureType::Numerical,
            ];
            let data = DMatrix::from_dense(&x, n, features)
                .unwrap()
                .with_feature_types(&types)
                .unwrap();
            let ghist = binned(&data, 32);
            let has = (ghist.is_dense(), ghist.columns().is_some());
            assert_eq!(
                has,
                match layout {
                    "dense" => (true, true),
                    "missing columns" => (false, true),
                    _ => (false, false),
                },
                "{layout}"
            );
            let cuts = ghist.cuts();
            let rows: Vec<u32> = (0..n as u32).filter(|r| r % 3 != 1).collect();
            for feature in 0..features {
                let (fs, fe) = cuts.feature_bins(feature);
                for default_left in [false, true] {
                    let location = if feature < 2 {
                        // Unordered, with a repeat and an absent category.
                        SplitLocation::Categories(vec![11, 4, 1, 5, 4, 99])
                    } else {
                        SplitLocation::Numeric(SplitPos::Bin(fs + (fe - fs) / 2))
                    };
                    let route = SplitRoute {
                        feature: feature as u32,
                        location: &location,
                        default_left,
                    };
                    let goes_left =
                        |r: u32| match (ghist.feature_bin(r as usize, feature), &location) {
                            (Some(bin), SplitLocation::Categories(categories)) => {
                                in_category_set(categories, cuts.cut_value(bin as usize))
                            }
                            (Some(bin), SplitLocation::Numeric(pos)) => {
                                pos.bin().is_some_and(|s| bin as usize <= s)
                            }
                            (None, _) => default_left,
                        };
                    let expected: (Vec<u32>, Vec<u32>) = rows.iter().partition(|&&r| goes_left(r));
                    let serial = serial_pool.install(|| partition_rows(&ghist, &rows, route));
                    let parallel = parallel_pool.install(|| partition_rows(&ghist, &rows, route));
                    assert_eq!(serial, expected, "{layout}, feature {feature}");
                    assert_eq!(parallel, expected, "{layout}, feature {feature}");
                }
            }
        }
    }
}
