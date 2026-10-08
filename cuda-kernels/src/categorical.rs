//! The categorical split search: per-category sort keys, a stable merge
//! sort of the categories by key, the partition scan, and the merge of every
//! feature's winner in the host's feature order. It shares the numeric
//! search's scorer ([`ScanReg`]).
//!
//! A task is four `u64`s: `(request, feature)` (request in the low word),
//! `(slot, direction)`, its result index, and the first bin of its sorting
//! workspace.

use crate::train::{NumericResults, Regularization, ScanReg, shuffle_pair, warp_prefix};
use crate::{F64x2, FULL, SCAN_WARPS, ld, st};
use core::cmp::Ordering;
use cuda_device::atomic::{AtomicOrdering::Relaxed, DeviceAtomicU32};
use cuda_device::{SharedArray, kernel, launch_bounds, thread, warp};

/// Most categories a partition's left set holds (the scan's depth bound of
/// 64 bins, less one).
const CAT_SET_BINS: u64 = 63;

/// `u64` words of a node's packed winner before its category set.
const NODE_SCAN_WORDS: u64 = 7;

/// A task's request and feature.
///
/// # Safety
///
/// `task` holds a task's four words.
#[inline(always)]
unsafe fn request_feature(task: *const u64) -> (u64, u64) {
    // SAFETY: the caller's.
    let word = unsafe { ld(task, 0) };
    (u64::from(word as u32), word >> 32)
}

/// The categorical tasks a sort pass covers: `tasks[4 t..]` for `t <
/// n_tasks`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CategoryTasks {
    pub tasks: *const u64,
    pub n_tasks: u32,
}

/// The sort workspaces [`category_keys`] initializes: each category's key
/// and the identity order.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SortWorkspace {
    pub keys: *mut f32,
    pub order: *mut u32,
}

/// Each category's sort key: its weight (`CalcWeightCat`), or 0 for a
/// category whose Hessian is not positive or below `min_child_weight`;
/// `order` starts as the identity. A non-finite key flags the task's result
/// (`meta` status 3) for the host. Features below 4 categories search
/// one-hot and need no keys. `meta` is a direct parameter: an atomic on a
/// generic address would also need a local-memory path.
///
/// # Safety
///
/// `tasks` holds `n_tasks` tasks, `pool` every slot they name (`total_bins`
/// bins each), `keys` and `order` every workspace, and `meta` every result.
#[kernel]
pub unsafe extern "C" fn category_keys(
    pool: *const F64x2,
    feature_first: *const u32,
    total_bins: u64,
    work: CategoryTasks,
    regularization: Regularization,
    workspace: SortWorkspace,
    meta: *mut u32,
) {
    let CategoryTasks { tasks, n_tasks } = work;
    let SortWorkspace { keys, order } = workspace;
    let Regularization {
        lambda,
        alpha,
        max_delta_step,
        min_child_weight,
    } = regularization;
    // SAFETY: the caller's; one block per task, one thread per category.
    unsafe {
        let mut t = thread::blockIdx_x();
        while t < n_tasks {
            let task = tasks.add(4 * t as usize);
            let (_, f) = request_feature(task);
            let first = ld(feature_first, f);
            let len = u64::from(ld(feature_first, f + 1) - first);
            if len >= 4 {
                let slot = u64::from(ld(task, 1) as u32);
                let (workspace, result) = (ld(task, 3), ld(task, 2));
                let bins = pool.add((slot * total_bins + u64::from(first)) as usize);
                let mut i = u64::from(thread::threadIdx_x());
                while i < len {
                    let s = ld(bins, i);
                    let mut key = 0.0f32;
                    // CalcWeightCat checks min_child_weight before
                    // CalcWeight's non-positive-Hessian case, and does not
                    // apply node bounds. A NaN Hessian passes both (and
                    // flags the task below).
                    if s.y.partial_cmp(&min_child_weight) != Some(Ordering::Less)
                        && !matches!(
                            s.y.partial_cmp(&0.0),
                            Some(Ordering::Less | Ordering::Equal)
                        )
                    {
                        let threshold = if s.x > alpha {
                            s.x - alpha
                        } else if s.x < -alpha {
                            s.x + alpha
                        } else {
                            0.0
                        };
                        let mut weight = -threshold / (s.y + lambda);
                        if max_delta_step != 0.0 && weight.abs() > max_delta_step {
                            weight = max_delta_step.copysign(weight);
                        }
                        key = weight as f32;
                    }
                    st(keys, workspace + i, key);
                    st(order, workspace + i, i as u32);
                    if !key.is_finite() {
                        DeviceAtomicU32::from_ptr(meta.add((4 * result) as usize)).swap(3, Relaxed);
                    }
                    i += u64::from(thread::blockDim_x());
                }
            }
            t += thread::gridDim_x();
        }
    }
}

/// One pass of a stable parallel merge sort in global memory, with no
/// category-count cap: runs of `width` categories of `source` merge into
/// `dest`. A left-run item counts strictly smaller right keys, a right-run
/// item smaller-or-equal left keys, so equal weights (+0 and -0 included)
/// keep their ascending bin order at every width.
///
/// # Safety
///
/// As [`category_keys`], with `source` and `dest` holding every workspace.
#[kernel]
pub unsafe extern "C" fn category_merge(
    feature_first: *const u32,
    work: CategoryTasks,
    meta: *const u32,
    width: u64,
    keys: *const f32,
    source: *const u32,
    dest: *mut u32,
) {
    let CategoryTasks { tasks, n_tasks } = work;
    // SAFETY: the caller's; one block per task, one thread per category,
    // each writing its own `dest` slot.
    unsafe {
        let mut t = thread::blockIdx_x();
        while t < n_tasks {
            let task = tasks.add(4 * t as usize);
            let (_, f) = request_feature(task);
            let len = u64::from(ld(feature_first, f + 1)) - u64::from(ld(feature_first, f));
            let at = ld(task, 3);
            // One-hot search uses bin order directly and never compares keys
            // (so the non-total NaN keys are never merged for it).
            if ld(meta, 4 * ld(task, 2)) != 3 && len >= 4 {
                let mut i = u64::from(thread::threadIdx_x());
                while i < len {
                    let start = (i / (2 * width)) * (2 * width);
                    let middle = (start + width).min(len);
                    let end = (start + 2 * width).min(len);
                    let left = i < middle;
                    let own = if left { start } else { middle };
                    let (mut lo, mut hi) = if left { (middle, end) } else { (start, middle) };
                    let opposite = lo;
                    let index = ld(source, at + i);
                    let key = ld(keys, at + u64::from(index));
                    while lo < hi {
                        let m = lo + (hi - lo) / 2;
                        let other = ld(keys, at + u64::from(ld(source, at + m)));
                        if if left { other < key } else { other <= key } {
                            lo = m + 1;
                        } else {
                            hi = m;
                        }
                    }
                    st(dest, at + start + (i - own) + (lo - opposite), index);
                    i += u64::from(thread::blockDim_x());
                }
            }
            t += thread::gridDim_x();
        }
    }
}

/// A categorical scan's tasks: `tasks[4 t..]` for `t < n_tasks`, each
/// request's `totals` and `params` (root gain, lower, upper), and whether
/// the tree's histograms are certified exact.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CategoricalTasks {
    pub tasks: *const u64,
    pub n_tasks: u64,
    pub totals: *const F64x2,
    pub params: *const f32,
    pub exact: u32,
}

/// A categorical scan's results, per result index `r`: `meta[4 r..]`,
/// `children[2 r..]` and `sets[63 r..]` ([`scan_categorical`] writes them,
/// [`merge_scans`] reads them).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CategoricalResults {
    pub meta: *mut u32,
    pub children: *mut F64x2,
    pub sets: *mut u32,
}

/// Warp `w`'s two chains of [`CAT_SET_BINS`] staged statistics in the
/// block's shared memory, for [`scan_categorical`].
#[inline(always)]
fn chain_row(w: u32) -> *mut F64x2 {
    const ROW: usize = 2 * CAT_SET_BINS as usize;
    static mut CHAIN: SharedArray<F64x2, { SCAN_WARPS * ROW }> = SharedArray::UNINIT;
    // SAFETY: an address within the array (`w < SCAN_WARPS`).
    unsafe { SharedArray::as_raw_mut_ptr(&raw mut CHAIN).add(w as usize * ROW) }
}

/// One warp per categorical feature forms the prefix and suffix statistics
/// over the categories in key order (or, one-hot below 4 categories, lane 0
/// the missing statistics), then all lanes score candidates with the
/// numeric search's scorer. Without `exact`, lane 0 chains them in the
/// CPU's order; with `exact` (the tree's histograms certified exact, as for
/// [`scan_splits`](crate::train::scan_splits)), the warp scans each 32
/// categories at once ([`warp_prefix`]). The reduction keeps the largest
/// finite loss, then the earliest candidate. `meta[4 r..]` gets (status: 0
/// none, 1 found, 4 NaN; categories selected; loss bits; default-left |
/// backward << 1), `children[2 r..]` the right and left statistics (the
/// tree's children are XGBoost's swapped), and `sets[63 r..]` the selected
/// categories.
///
/// # Safety
///
/// As [`category_keys`], with `totals` and `params` (root gain, lower,
/// upper) every request, `order` the sorted workspaces, and `children` and
/// `sets` every result.
#[kernel]
#[launch_bounds(32)]
pub unsafe extern "C" fn scan_categorical(
    pool: *const F64x2,
    feature_first: *const u32,
    total_bins: u64,
    work: CategoricalTasks,
    regularization: Regularization,
    order: *const u32,
    out: CategoricalResults,
) {
    let CategoricalTasks {
        tasks,
        n_tasks,
        totals,
        params,
        exact,
    } = work;
    let CategoricalResults {
        meta,
        children,
        sets,
    } = out;
    // SAFETY: the caller's; each warp owns its chain row, staged (or
    // scanned) by its lanes, chained by lane 0 and read by the lanes, each
    // step between warp barriers every lane reaches; every lane reaches the
    // shuffles.
    unsafe {
        let lane = thread::threadIdx_x() & 31;
        let warp_index = thread::threadIdx_x() >> 5;
        let chain = chain_row(warp_index);
        let warps = u64::from(thread::gridDim_x()) * SCAN_WARPS as u64;
        let mut t = u64::from(thread::blockIdx_x()) * SCAN_WARPS as u64 + u64::from(warp_index);
        while t < n_tasks {
            let task = tasks.add((4 * t) as usize);
            let (result, workspace) = (ld(task, 2), ld(task, 3));
            if ld(meta, 4 * result) != 3 {
                let (request, f) = request_feature(task);
                let first = ld(feature_first, f);
                let len = ld(feature_first, f + 1) - first;
                let slot = u64::from(ld(task, 1) as u32);
                let bins = pool.add((slot * total_bins + u64::from(first)) as usize);
                let total = ld(totals, request);
                let reg = ScanReg::new(
                    regularization,
                    params,
                    request,
                    (ld(task, 1) >> 32) as u32 as i32,
                );
                let onehot = len < 4;
                let depth = len.min(64);
                let steps = if onehot { 0 } else { depth - 1 };
                let (mut mg, mut mh) = (0.0f64, 0.0f64);
                if !onehot && exact != 0 {
                    // Each chain (prefix, then suffix) a window of 32 steps
                    // at a time: every lane loads its step's `order` entry
                    // and bin, the warp scans them onto the previous
                    // window's total, and each lane stores its prefix.
                    let mut pass = 0;
                    while pass < 2 {
                        let mut carry = F64x2 { x: 0.0, y: 0.0 };
                        let mut base = 0;
                        while base < steps {
                            let step = base + lane;
                            let b = if step < steps {
                                let k = if pass == 0 { step } else { len - 1 - step };
                                ld(bins, u64::from(ld(order, workspace + u64::from(k))))
                            } else {
                                F64x2 { x: 0.0, y: 0.0 }
                            };
                            let s = warp_prefix(lane, b);
                            let a = F64x2 {
                                x: carry.x + s.x,
                                y: carry.y + s.y,
                            };
                            if step < steps {
                                st(chain, u64::from(pass * steps + step), a);
                            }
                            carry = shuffle_pair(a, (steps - base).min(32) - 1);
                            base += 32;
                        }
                        pass += 1;
                    }
                } else if !onehot {
                    // The chains' bins in key order (prefix, then suffix),
                    // staged by every lane, each loading its `order` entry
                    // and bin in parallel; lane 0 then chains them in place,
                    // its serial adds waiting on no global load.
                    let mut i = lane;
                    while i < 2 * steps {
                        let k = if i < steps { i } else { len - 1 - (i - steps) };
                        let index = ld(order, workspace + u64::from(k));
                        st(chain, u64::from(i), ld(bins, u64::from(index)));
                        i += 32;
                    }
                    warp::sync_mask(FULL);
                }
                if lane == 0 {
                    if onehot {
                        let (mut g, mut h) = (0.0f64, 0.0f64);
                        let mut i = 0;
                        while i < len {
                            let b = ld(bins, u64::from(i));
                            g += b.x;
                            h += b.y;
                            i += 1;
                        }
                        mg = total.x - g;
                        mh = total.y - h;
                    } else if exact == 0 {
                        let mut pass = 0;
                        while pass < 2 {
                            let (mut g, mut h) = (0.0f64, 0.0f64);
                            let mut step = 0;
                            while step < steps {
                                let at = u64::from(pass * steps + step);
                                let bin = ld(chain, at);
                                g += bin.x;
                                h += bin.y;
                                st(chain, at, F64x2 { x: g, y: h });
                                step += 1;
                            }
                            pass += 1;
                        }
                    }
                }
                let mg = warp::shuffle_f64_sync(FULL, mg, 0);
                let mh = warp::shuffle_f64_sync(FULL, mh, 0);
                warp::sync_mask(FULL);
                // A candidate's (left, right) statistics, from the chains
                // (or bins) that stay in place until the task ends.
                let split_of = |candidate: u32| {
                    if onehot {
                        let mut right = ld(bins, u64::from(candidate / 2));
                        if candidate & 1 != 0 {
                            right.x += mg;
                            right.y += mh;
                        }
                        let left = F64x2 {
                            x: total.x - right.x,
                            y: total.y - right.y,
                        };
                        (left, right)
                    } else {
                        let acc = ld(chain, u64::from(candidate));
                        let rest = F64x2 {
                            x: total.x - acc.x,
                            y: total.y - acc.y,
                        };
                        if candidate < steps {
                            (rest, acc)
                        } else {
                            (acc, rest)
                        }
                    }
                };
                let mut best = 0.0f32;
                let mut position = u32::MAX;
                let mut nan = false;
                let count = if onehot { 2 * len } else { 2 * steps };
                let mut candidate = lane;
                while candidate < count {
                    let (left, right) = split_of(candidate);
                    let loss = reg.score(left.x, left.y, right.x, right.y);
                    if loss.is_nan() {
                        nan = true;
                    } else if loss.is_finite() && loss > best {
                        best = loss;
                        position = candidate;
                    }
                    candidate += 32;
                }
                // Reduce the loss and position only: lane 0 then recomputes
                // the winner's children, the same operations on the same
                // statistics.
                let mut delta = 16;
                while delta > 0 {
                    let other_best = warp::shuffle_down_f32_sync(FULL, best, delta);
                    let other_position = warp::shuffle_down_sync(FULL, position, delta);
                    if other_best > best || (other_best == best && other_position < position) {
                        best = other_best;
                        position = other_position;
                    }
                    delta >>= 1;
                }
                let nan = warp::any_sync(FULL, nan);
                let position = warp::shuffle_sync(FULL, position, 0);
                let found = position != u32::MAX;
                let backward = !onehot && found && position >= steps;
                let selected = match (found, onehot) {
                    (false, _) => 0,
                    (true, true) => 1,
                    (true, false) => position % steps + 1,
                };
                if lane == 0 {
                    let status = if nan { 4 } else { u32::from(found) };
                    let default_left = if onehot { position & 1 != 0 } else { backward };
                    st(meta, 4 * result, status);
                    st(meta, 4 * result + 1, selected);
                    st(meta, 4 * result + 2, best.to_bits());
                    st(
                        meta,
                        4 * result + 3,
                        u32::from(default_left) | u32::from(backward) << 1,
                    );
                    let zero = F64x2 { x: 0.0, y: 0.0 };
                    let (left, right) = if found {
                        split_of(position)
                    } else {
                        (zero, zero)
                    };
                    st(children, 2 * result, right);
                    st(children, 2 * result + 1, left);
                }
                if found && !nan {
                    let mut i = lane;
                    while i < selected {
                        let category = if onehot {
                            position / 2
                        } else {
                            let k = if backward { len - selected + i } else { i };
                            ld(order, workspace + u64::from(k))
                        };
                        st(sets, CAT_SET_BINS * result + u64::from(i), category);
                        i += 32;
                    }
                }
                warp::sync_mask(FULL);
            }
            t += warps;
        }
    }
}

/// Every node's winner, merged in the host's feature order: node `k`'s
/// candidates are `refs[3 i..]` = (feature, kind: 0 numeric, 1 categorical,
/// result index) for `i` in `[first[k], first[k + 1])`. A finite
/// per-feature winner may be reduced; any NaN requires the host's
/// sequential replay of the whole node (header word 0, from the first such
/// candidate: 3 numeric NaN, 4 non-finite category weight, 5 categorical
/// NaN). Otherwise `out[out_at[k] ..]` gets the packed winner (`feature <<
/// 32 | kind + 1`, `default_left << 32 | bin` or the set size, loss bits,
/// the left statistics' bits, and for a categorical winner the right ones
/// and its category set), or 0 for none.
///
/// One warp per node (blocks of [`SCAN_WARPS`] warps, any grid): each lane
/// folds every 32nd candidate in order with the host's `SplitEntry::update`
/// rule from a zero loss, then the warp reduces the lanes' winners. A found
/// result's loss is finite, so that rule is the order "greater loss, then
/// lower feature, then earlier candidate", which any reduction tree keeps.
///
/// # Safety
///
/// `first` holds `n_nodes + 1` entries and `out_at` `n_nodes`, `refs` every
/// candidate they name, the metadata, statistics and sets every result, and
/// `out` room for each node's header and set.
#[kernel]
#[launch_bounds(32)]
pub unsafe extern "C" fn merge_scans(
    refs: *const u32,
    first: *const u32,
    n_nodes: u64,
    numeric: NumericResults,
    categorical: CategoricalResults,
    out_at: *const u64,
    out: *mut u64,
) {
    let NumericResults {
        meta: numeric_meta,
        acc: numeric_acc,
    } = numeric;
    let CategoricalResults {
        meta: categorical_meta,
        children: categorical_children,
        sets: categorical_sets,
    } = categorical;
    // SAFETY: the caller's; one warp per node writes its own output (lane 0
    // the header, the lanes distinct set entries after it), and every lane
    // reaches the shuffles.
    unsafe {
        let meta_of = |kind: u32, index: u32| {
            let base = if kind != 0 {
                categorical_meta
            } else {
                numeric_meta
            };
            base.add(4 * index as usize).cast_const()
        };
        let lane = thread::threadIdx_x() & 31;
        let warps = u64::from(thread::gridDim_x()) * SCAN_WARPS as u64;
        let mut node = u64::from(thread::blockIdx_x()) * SCAN_WARPS as u64
            + u64::from(thread::threadIdx_x() >> 5);
        while node < n_nodes {
            let mut best = 0.0f32;
            let mut best_feature = 0u32;
            let mut chosen = u32::MAX;
            // The lane's first candidate requiring the host's replay.
            let mut replay = u32::MAX;
            // Counted in `u64`: `first + lane` and the stride wrap a `u32`
            // near the `u32::MAX` candidates the host admits. A candidate
            // index is below `end`, so it never takes the sentinel's value.
            let end = u64::from(ld(first, node + 1));
            let mut i = u64::from(ld(first, node)) + u64::from(lane);
            while i < end {
                let candidate = refs.add(3 * i as usize);
                let (feature, kind, index) = (ld(candidate, 0), ld(candidate, 1), ld(candidate, 2));
                let meta = meta_of(kind, index);
                let status = ld(meta, 0);
                if (kind == 0 && status == 2) || (kind != 0 && status >= 3) {
                    replay = i as u32;
                    break;
                }
                if status == 1 {
                    let loss = f32::from_bits(ld(meta, 2));
                    let replace = !loss.is_infinite()
                        && if best_feature <= feature {
                            loss > best
                        } else {
                            best.partial_cmp(&loss) != Some(Ordering::Greater)
                        };
                    if replace {
                        best = loss;
                        best_feature = feature;
                        chosen = i as u32;
                    }
                }
                i += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                let other_best = warp::shuffle_down_f32_sync(FULL, best, delta);
                let other_feature = warp::shuffle_down_sync(FULL, best_feature, delta);
                let other_chosen = warp::shuffle_down_sync(FULL, chosen, delta);
                replay = replay.min(warp::shuffle_down_sync(FULL, replay, delta));
                let better = other_best > best
                    || (other_best == best
                        && (other_feature < best_feature
                            || (other_feature == best_feature && other_chosen < chosen)));
                if other_chosen != u32::MAX && (chosen == u32::MAX || better) {
                    best = other_best;
                    best_feature = other_feature;
                    chosen = other_chosen;
                }
                delta >>= 1;
            }
            let replay = warp::shuffle_sync(FULL, replay, 0);
            let chosen = warp::shuffle_sync(FULL, chosen, 0);
            let header = out.add(ld(out_at, node) as usize);
            if replay != u32::MAX || chosen == u32::MAX {
                if lane == 0 {
                    let fallback = if replay == u32::MAX {
                        0
                    } else {
                        let candidate = refs.add(3 * replay as usize);
                        let kind = ld(candidate, 1);
                        match (kind, ld(meta_of(kind, ld(candidate, 2)), 0)) {
                            (0, _) => 3,
                            (_, 3) => 4,
                            _ => 5,
                        }
                    };
                    st(header, 0, fallback);
                }
            } else {
                let candidate = refs.add(3 * chosen as usize);
                let (kind, index) = (ld(candidate, 1), ld(candidate, 2));
                let meta = meta_of(kind, index);
                if lane == 0 {
                    st(
                        header,
                        0,
                        u64::from(best_feature) << 32 | if kind != 0 { 2 } else { 1 },
                    );
                    st(
                        header,
                        1,
                        u64::from(ld(meta, 3)) << 32 | u64::from(ld(meta, 1)),
                    );
                    st(header, 2, u64::from(ld(meta, 2)));
                    let a = if kind != 0 {
                        ld(categorical_children, 2 * u64::from(index))
                    } else {
                        ld(numeric_acc, u64::from(index))
                    };
                    st(header, 3, a.x.to_bits());
                    st(header, 4, a.y.to_bits());
                    if kind != 0 {
                        let b = ld(categorical_children, 2 * u64::from(index) + 1);
                        st(header, 5, b.x.to_bits());
                        st(header, 6, b.y.to_bits());
                    }
                }
                if kind != 0 {
                    let selected = header.add(NODE_SCAN_WORDS as usize).cast::<u32>();
                    let mut i = lane;
                    while i < ld(meta, 1) {
                        let category = ld(
                            categorical_sets,
                            CAT_SET_BINS * u64::from(index) + u64::from(i),
                        );
                        st(selected, u64::from(i), category);
                        i += 32;
                    }
                }
            }
            node += warps;
        }
    }
}
