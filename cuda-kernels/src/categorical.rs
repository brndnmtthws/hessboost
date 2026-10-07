//! The categorical split search: per-category sort keys, a stable merge
//! sort of the categories by key, the partition scan, and the merge of every
//! feature's winner in the host's feature order. It shares the numeric
//! search's scorer ([`ScanReg`]).
//!
//! A task is four `u64`s: `(request, feature)` (request in the low word),
//! `(slot, direction)`, its result index, and the first bin of its sorting
//! workspace.

use crate::train::ScanReg;
use crate::{F64x2, FULL, SCAN_WARPS, grid_index, grid_threads, ld, st};
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

/// Each category's sort key: its weight (`CalcWeightCat`), or 0 for a
/// category whose Hessian is not positive or below `min_child_weight`;
/// `order` starts as the identity. A non-finite key flags the task's result
/// (`meta` status 3) for the host. Features below 4 categories search
/// one-hot and need no keys.
///
/// # Safety
///
/// `tasks` holds `n_tasks` tasks, `pool` every slot they name (`total_bins`
/// bins each), `keys` and `order` every workspace, and `meta` every result.
#[kernel]
pub unsafe fn category_keys(
    pool: *const F64x2,
    feature_first: *const u32,
    total_bins: u64,
    tasks: *const u64,
    n_tasks: u32,
    lambda: f64,
    alpha: f64,
    max_delta_step: f64,
    min_child_weight: f64,
    keys: *mut f32,
    order: *mut u32,
    meta: *mut u32,
) {
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
                    // apply node bounds.
                    if !(s.y < min_child_weight) && !(s.y <= 0.0) {
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
pub unsafe fn category_merge(
    feature_first: *const u32,
    tasks: *const u64,
    n_tasks: u32,
    width: u64,
    keys: *const f32,
    source: *const u32,
    dest: *mut u32,
    meta: *const u32,
) {
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

/// One warp per categorical feature: lane 0 forms the CPU's prefix and
/// suffix chains over the categories in key order (or, one-hot below 4
/// categories, the missing statistics), then all lanes score candidates
/// with the numeric search's scorer. The reduction keeps the largest finite
/// loss, then the earliest candidate. `meta[4 r..]` gets (status: 0 none, 1
/// found, 4 NaN; categories selected; loss bits; default-left | backward
/// << 1), `children[2 r..]` the right and left statistics (the tree's
/// children are XGBoost's swapped), and `sets[63 r..]` the selected
/// categories.
///
/// # Safety
///
/// As [`category_keys`], with `totals` and `params` (root gain, lower,
/// upper) every request, `order` the sorted workspaces, and `children` and
/// `sets` every result.
#[kernel]
#[launch_bounds(128)]
pub unsafe fn scan_categorical(
    pool: *const F64x2,
    feature_first: *const u32,
    total_bins: u64,
    tasks: *const u64,
    n_tasks: u64,
    totals: *const F64x2,
    params: *const f32,
    lambda: f64,
    alpha: f64,
    max_delta_step: f64,
    min_child_weight: f64,
    order: *const u32,
    meta: *mut u32,
    children: *mut F64x2,
    sets: *mut u32,
) {
    static mut CHAIN: SharedArray<F64x2, { SCAN_WARPS * 2 * CAT_SET_BINS as usize }> =
        SharedArray::UNINIT;
    // SAFETY: the caller's; each warp owns its chain row, written by lane 0
    // and read by the lanes between warp barriers every lane reaches.
    unsafe {
        let lane = thread::threadIdx_x() & 31;
        let warp_index = thread::threadIdx_x() >> 5;
        let chain = SharedArray::as_raw_mut_ptr(&raw mut CHAIN)
            .add(warp_index as usize * 2 * CAT_SET_BINS as usize);
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
                let reg = ScanReg {
                    lambda,
                    alpha,
                    max_delta_step,
                    min_child_weight,
                    root_gain: ld(params, 3 * request),
                    lower: ld(params, 3 * request + 1),
                    upper: ld(params, 3 * request + 2),
                    dir: (ld(task, 1) >> 32) as u32 as i32,
                };
                let onehot = len < 4;
                let depth = len.min(64);
                let steps = if onehot { 0 } else { depth - 1 };
                let (mut mg, mut mh) = (0.0f64, 0.0f64);
                if lane == 0 {
                    if onehot {
                        let (mut g, mut h) = (0.0f64, 0.0f64);
                        let mut i = 0;
                        while i < len {
                            let b = ld(bins, u64::from(i));
                            g = g + b.x;
                            h = h + b.y;
                            i += 1;
                        }
                        mg = total.x - g;
                        mh = total.y - h;
                    } else {
                        let mut pass = 0;
                        while pass < 2 {
                            let (mut g, mut h) = (0.0f64, 0.0f64);
                            let mut step = 0;
                            while step < steps {
                                let k = if pass == 1 { len - 1 - step } else { step };
                                let index = ld(order, workspace + u64::from(k));
                                let bin = ld(bins, u64::from(index));
                                g = g + bin.x;
                                h = h + bin.y;
                                st(chain, u64::from(pass * steps + step), F64x2 { x: g, y: h });
                                step += 1;
                            }
                            pass += 1;
                        }
                    }
                }
                let mg = warp::shuffle_f64_sync(FULL, mg, 0);
                let mh = warp::shuffle_f64_sync(FULL, mh, 0);
                warp::sync_mask(FULL);
                let mut best = 0.0f32;
                let mut position = u32::MAX;
                let (mut lg, mut lh, mut rg, mut rh) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                let mut nan = false;
                let count = if onehot { 2 * len } else { 2 * steps };
                let mut candidate = lane;
                while candidate < count {
                    let (left, right) = if onehot {
                        let mut right = ld(bins, u64::from(candidate / 2));
                        if candidate & 1 != 0 {
                            right.x = right.x + mg;
                            right.y = right.y + mh;
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
                    };
                    let loss = reg.score(left.x, left.y, right.x, right.y);
                    if loss.is_nan() {
                        nan = true;
                    } else if loss.is_finite() && loss > best {
                        best = loss;
                        position = candidate;
                        (lg, lh, rg, rh) = (left.x, left.y, right.x, right.y);
                    }
                    candidate += 32;
                }
                let mut delta = 16;
                while delta > 0 {
                    let other_best = warp::shuffle_down_f32_sync(FULL, best, delta);
                    let other_position = warp::shuffle_down_sync(FULL, position, delta);
                    let other_lg = warp::shuffle_down_f64_sync(FULL, lg, delta);
                    let other_lh = warp::shuffle_down_f64_sync(FULL, lh, delta);
                    let other_rg = warp::shuffle_down_f64_sync(FULL, rg, delta);
                    let other_rh = warp::shuffle_down_f64_sync(FULL, rh, delta);
                    if other_best > best || (other_best == best && other_position < position) {
                        best = other_best;
                        position = other_position;
                        (lg, lh, rg, rh) = (other_lg, other_lh, other_rg, other_rh);
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
                    st(children, 2 * result, F64x2 { x: rg, y: rh });
                    st(children, 2 * result + 1, F64x2 { x: lg, y: lh });
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
/// sequential replay of the whole node (header word 0: 3 numeric NaN, 4
/// non-finite category weight, 5 categorical NaN). Otherwise `out[out_at[k]
/// ..]` gets the packed winner (`feature << 32 | kind + 1`, `default_left
/// << 32 | bin` or the set size, loss bits, the left statistics' bits, and
/// for a categorical winner the right ones and its category set), or 0 for
/// none.
///
/// # Safety
///
/// `first` holds `n_nodes + 1` entries and `out_at` `n_nodes`, `refs` every
/// candidate they name, the metadata, statistics and sets every result, and
/// `out` room for each node's header and set.
#[kernel]
pub unsafe fn merge_scans(
    refs: *const u32,
    first: *const u32,
    n_nodes: u64,
    numeric_meta: *const u32,
    numeric_acc: *const F64x2,
    categorical_meta: *const u32,
    categorical_children: *const F64x2,
    categorical_sets: *const u32,
    out_at: *const u64,
    out: *mut u64,
) {
    // SAFETY: the caller's; one thread per node writes its own output.
    unsafe {
        let meta_of = |kind: u32, index: u32| {
            let base = if kind != 0 {
                categorical_meta
            } else {
                numeric_meta
            };
            base.add(4 * index as usize)
        };
        let mut node = grid_index();
        while node < n_nodes {
            let mut best = 0.0f32;
            let mut best_feature = 0u32;
            let mut chosen = u32::MAX;
            let mut fallback = 0u32;
            let mut i = ld(first, node);
            while i < ld(first, node + 1) {
                let candidate = refs.add(3 * i as usize);
                let (feature, kind, index) = (ld(candidate, 0), ld(candidate, 1), ld(candidate, 2));
                let meta = meta_of(kind, index);
                let status = ld(meta, 0);
                if (kind == 0 && status == 2) || (kind != 0 && status >= 3) {
                    fallback = match (kind, status) {
                        (0, _) => 3,
                        (_, 3) => 4,
                        _ => 5,
                    };
                    break;
                }
                if status == 1 {
                    let loss = f32::from_bits(ld(meta, 2));
                    let replace = !loss.is_infinite()
                        && if best_feature <= feature {
                            loss > best
                        } else {
                            !(best > loss)
                        };
                    if replace {
                        best = loss;
                        best_feature = feature;
                        chosen = i;
                    }
                }
                i += 1;
            }
            let header = out.add(ld(out_at, node) as usize);
            if fallback != 0 || chosen == u32::MAX {
                st(header, 0, u64::from(fallback));
            } else {
                let candidate = refs.add(3 * chosen as usize);
                let (kind, index) = (ld(candidate, 1), ld(candidate, 2));
                let meta = meta_of(kind, index);
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
                    let selected = header.add(NODE_SCAN_WORDS as usize).cast::<u32>();
                    let mut i = 0;
                    while i < ld(meta, 1) {
                        let category = ld(
                            categorical_sets,
                            CAT_SET_BINS * u64::from(index) + u64::from(i),
                        );
                        st(selected, u64::from(i), category);
                        i += 1;
                    }
                }
            }
            node += grid_threads();
        }
    }
}
