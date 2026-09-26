//! XGBoost's weighted quantile sketch, reproduced operation for operation.
//!
//! `tree_method=hist` and `tree_method=approx` derive their histogram cuts
//! from XGBoost's `WQuantileSketch` (a Greenwald–Khanna style merge/prune
//! summary). The cut *values* determine every split threshold a tree can
//! take, so producing the same trees as XGBoost requires the same sketch:
//! same summary budget, same level-wise merge schedule, same prune and
//! query arithmetic (including `f32` rank bookkeeping). This module follows
//! `src/common/quantile.h` / `quantile.cc` of XGBoost 3.4.1; comments name
//! the corresponding upstream routines.
//!
//! Two ingestion paths exist upstream and are both reproduced:
//!
//! - **Streaming** ([`WQSketch::push`], upstream `PushRowPage`): values arrive
//!   in row order with their sample weights, are buffered in a queue and
//!   summarised in level-wise merges. Used by `hist`.
//! - **Sorted** ([`WQSketch::push_sorted`], upstream `PushColPage` →
//!   `SetPruneSorted`): a whole sorted column is pruned straight to
//!   `max_bin` entries. Used by `approx`, whose weights are the current
//!   Hessians.

use super::quantile::{RadixScratch, radix_sort, sort_key, unsort_key};

/// One summary entry (`WQSummary::Entry`): a value with its rank interval
/// `[rmin, rmax]` and the weight `wmin` of the value itself.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry {
    rmin: f32,
    rmax: f32,
    wmin: f32,
    value: f32,
}

impl Entry {
    #[inline]
    fn new(rmin: f32, rmax: f32, wmin: f32, value: f32) -> Self {
        Entry {
            rmin,
            rmax,
            wmin,
            value,
        }
    }

    /// Smallest rank any value strictly greater than `value` can have.
    #[inline]
    fn rmin_next(self) -> f32 {
        self.rmin + self.wmin
    }

    /// Largest rank any value strictly smaller than `value` can have.
    #[inline]
    fn rmax_prev(self) -> f32 {
        self.rmax - self.wmin
    }
}

/// Upstream `SketchEpsilon`: the rank resolution for `n_samples` values.
fn sketch_epsilon(max_bin: usize, n_samples: usize) -> f64 {
    let n = n_samples.max(1);
    1.0 / max_bin.min(n) as f64
}

/// Upstream `WQuantileSketch::LimitSizeLevel`: the per-level summary size for
/// `maxn` values at rank resolution `eps`.
fn limit_size_level(maxn: usize, eps: f64) -> usize {
    if maxn == 0 {
        return 1;
    }
    // Upstream oversamples the internal summaries by `kFactor = 2`.
    let internal_eps = eps / 2.0;
    let mut nlevel = 1u32;
    loop {
        let limit = ((f64::from(nlevel) / internal_eps).ceil() as usize + 1).min(maxn);
        if (1usize << nlevel) * limit >= maxn {
            return limit;
        }
        nlevel += 1;
    }
}

/// Upstream `SketchSummaryBudget`: summary size for a sketch of `n` values.
fn summary_budget(max_bin: usize, n: usize) -> usize {
    limit_size_level(n, sketch_epsilon(max_bin, n))
}

/// `WQSummary::SetFromSorted`: build a summary from `(value, weight)` pairs
/// sorted by value, merging equal values.
fn set_from_sorted(queue: &[(f32, f32)], out: &mut Vec<Entry>) {
    out.clear();
    let mut wsum = 0f32;
    let mut i = 0;
    while i < queue.len() {
        let value = queue[i].0;
        let mut w = queue[i].1;
        let mut j = i + 1;
        while j < queue.len() && queue[j].0 == value {
            w += queue[j].1;
            j += 1;
        }
        out.push(Entry::new(wsum, wsum + w, w, value));
        wsum += w;
        i = j;
    }
}

/// [`set_from_sorted`] of values that each carry weight `1`, given as
/// ascending [`sort_key`]s.
fn set_from_sorted_keys(keys: &[u32], out: &mut Vec<Entry>) {
    out.clear();
    let mut wsum = 0f32;
    let mut i = 0;
    while i < keys.len() {
        let value = unsort_key(keys[i]);
        let mut w = 1f32;
        let mut j = i + 1;
        while j < keys.len() && unsort_key(keys[j]) == value {
            w += 1.0;
            j += 1;
        }
        out.push(Entry::new(wsum, wsum + w, w, value));
        wsum += w;
        i = j;
    }
}

/// `WQSummary::SetPrune`: shrink `data` in place to at most `maxsize`
/// entries by answering evenly spaced rank queries.
fn set_prune(data: &mut Vec<Entry>, maxsize: usize) {
    if maxsize == 0 {
        data.clear();
        return;
    }
    let src_size = data.len();
    if src_size <= maxsize {
        return;
    }
    if maxsize == 1 {
        data.truncate(1);
        return;
    }
    // `src_size > maxsize >= 2`, so indices 1 and 2 exist.
    let d = data.as_mut_slice();
    let begin = d[0].rmax;
    let range = d[src_size - 1].rmin - d[0].rmax;
    let n = maxsize - 1;
    let tail = d[src_size - 1];
    let mut left = d[1];
    let mut right = d[2];
    let mut len = 1usize;
    let (mut i, mut lastidx) = (1usize, 0usize);
    for k in 1..n {
        // Upstream evaluates this in `float`; keep the same rounding.
        let dx2 = 2.0f32 * ((k as f32 * range) / n as f32 + begin);
        // Find the first `i` with `dx2 < rmax[i+1] + rmin[i+1]`. Writes only
        // ever land at `len <= k <= i`, so reading ahead at `i + 1` is safe.
        while i < src_size - 1 && dx2 >= right.rmax + right.rmin {
            i += 1;
            left = right;
            if i < src_size - 1 {
                right = d[i + 1];
            }
        }
        if i == src_size - 1 {
            break;
        }
        if dx2 < left.rmin_next() + right.rmax_prev() {
            if i != lastidx {
                d[len] = left;
                len += 1;
                lastidx = i;
            }
        } else if i + 1 != lastidx {
            d[len] = right;
            len += 1;
            lastidx = i + 1;
        }
    }
    if lastidx != src_size - 1 {
        d[len] = tail;
        len += 1;
    }
    data.truncate(len);
}

/// `WQSummary::FixError`: re-establish rank monotonicity after a merge.
fn fix_error(data: &mut [Entry]) {
    let (mut prev_rmin, mut prev_rmax) = (0f32, 0f32);
    for e in data.iter_mut() {
        if e.rmin < prev_rmin {
            e.rmin = prev_rmin;
        } else {
            prev_rmin = e.rmin;
        }
        if e.rmax < prev_rmax {
            e.rmax = prev_rmax;
        }
        let rmin_next = e.rmin_next();
        if e.rmax < rmin_next {
            e.rmax = rmin_next;
        }
        prev_rmax = e.rmax;
    }
}

/// When either input is shorter than this, the merge runs in one pass.
const SPLIT_COMBINE_LEN: usize = 256;

/// `WQSummary::SetCombine`: merge `other` into `this`. `workspace` is scratch
/// reused across calls.
///
/// Every merged entry depends only on its position in the merge: on the
/// entries at the two cursors and the last entry taken from each input
/// (their `rmin_next`). Both summaries hold strictly increasing values, so
/// long inputs are split at a value `v` into the entries below `v` and the
/// rest, and the two merges run interleaved, their dependency chains
/// overlapping, each written to its own part of `workspace`. A value never
/// straddles the split, so the entries are exactly those of one merge. With
/// a `NaN` value (which orders against nothing) the merge runs in one pass.
fn set_combine(this: &mut Vec<Entry>, other: &[Entry], workspace: &mut Vec<Entry>) {
    if other.is_empty() {
        return;
    }
    if this.is_empty() {
        this.extend_from_slice(other);
        return;
    }
    let (a, b) = (this.as_slice(), other);
    let (na, nb) = (a.len(), b.len());
    workspace.clear();
    workspace.resize(na + nb, Entry::new(0.0, 0.0, 0.0, 0.0));
    let out = workspace.as_mut_slice();
    let whole = MergeCursor {
        ia: 0,
        ib: 0,
        aprev: 0.0,
        bprev: 0.0,
        out: 0,
    };
    let len = if na.min(nb) < SPLIT_COMBINE_LEN || a.iter().chain(b).any(|e| e.value.is_nan()) {
        let mut c = whole;
        c.finish(a, b, (na, nb), out);
        c.out
    } else {
        let ia_m = na / 2;
        let v = a[ia_m].value;
        let ib_m = b.partition_point(|e| e.value < v);
        let (mut lo, mut hi) = (
            whole,
            MergeCursor {
                ia: ia_m,
                ib: ib_m,
                aprev: ia_m.checked_sub(1).map_or(0.0, |i| a[i].rmin_next()),
                bprev: ib_m.checked_sub(1).map_or(0.0, |i| b[i].rmin_next()),
                out: ia_m + ib_m,
            },
        );
        while lo.ia < ia_m && lo.ib < ib_m && hi.ia < na && hi.ib < nb {
            lo.step(a, b, out);
            hi.step(a, b, out);
        }
        lo.finish(a, b, (ia_m, ib_m), out);
        hi.finish(a, b, (na, nb), out);
        out.copy_within(ia_m + ib_m..hi.out, lo.out);
        lo.out + (hi.out - (ia_m + ib_m))
    };
    workspace.truncate(len);
    fix_error(workspace);
    // The merged summary becomes `this`; the old one is the next scratch.
    std::mem::swap(this, workspace);
}

/// One merge of [`set_combine`] in progress: the cursors into both inputs,
/// the `rmin_next` of the last entry taken from each (`0` before the
/// first), and the next output slot.
#[derive(Clone, Copy)]
struct MergeCursor {
    ia: usize,
    ib: usize,
    aprev: f32,
    bprev: f32,
    out: usize,
}

impl MergeCursor {
    /// Merge the entry (or equal pair) at the cursors, both of which point
    /// into their inputs; branch-free, since which input comes next is
    /// data-dependent.
    #[inline(always)]
    fn step(&mut self, a: &[Entry], b: &[Entry], out: &mut [Entry]) {
        let (ea, eb) = (a[self.ia], b[self.ib]);
        let eq = ea.value == eb.value;
        let lt = ea.value < eb.value;
        // Equal values take both; otherwise the smaller (`b` also when the
        // comparison fails, as upstream's `else`).
        let take_a = eq | lt;
        let take_b = !lt;
        let pick = |c: bool, x: f32, y: f32| std::hint::select_unpredictable(c, x, y);
        // Upstream's three cases, operand for operand: a lone `a` entry adds
        // the last `b` entry's `rmin_next` and the next one's `rmax_prev`,
        // and symmetrically (`+` commutes exactly).
        let rmin = pick(take_a, ea.rmin, self.aprev) + pick(take_b, eb.rmin, self.bprev);
        let rmax = pick(take_a, ea.rmax, ea.rmax_prev()) + pick(take_b, eb.rmax, eb.rmax_prev());
        let wmin = if eq {
            ea.wmin + eb.wmin
        } else {
            pick(lt, ea.wmin, eb.wmin)
        };
        out[self.out] = Entry::new(rmin, rmax, wmin, pick(take_a, ea.value, eb.value));
        self.out += 1;
        self.aprev = pick(take_a, ea.rmin_next(), self.aprev);
        self.bprev = pick(take_b, eb.rmin_next(), self.bprev);
        self.ia += usize::from(take_a);
        self.ib += usize::from(take_b);
    }

    /// Merge up to `a[..a_end]` and `b[..b_end]`. Once one of them is used
    /// up, the other's entries take their bounds from the whole input's next
    /// entry, or its last `rmax` past the end (upstream's tail loops).
    fn finish(
        &mut self,
        a: &[Entry],
        b: &[Entry],
        (a_end, b_end): (usize, usize),
        out: &mut [Entry],
    ) {
        while self.ia < a_end && self.ib < b_end {
            self.step(a, b, out);
        }
        for ea in &a[self.ia..a_end] {
            let next = b
                .get(self.ib)
                .map_or(b[b.len() - 1].rmax, |eb| eb.rmax_prev());
            out[self.out] = Entry::new(ea.rmin + self.bprev, ea.rmax + next, ea.wmin, ea.value);
            self.out += 1;
            self.aprev = ea.rmin_next();
        }
        self.ia = self.ia.max(a_end);
        for eb in &b[self.ib..b_end] {
            let next = a
                .get(self.ia)
                .map_or(a[a.len() - 1].rmax, |ea| ea.rmax_prev());
            out[self.out] = Entry::new(eb.rmin + self.aprev, eb.rmax + next, eb.wmin, eb.value);
            self.out += 1;
            self.bprev = eb.rmin_next();
        }
        self.ib = self.ib.max(b_end);
    }
}

/// `WQSummary::SetPruneSorted`: summarise a whole column of `(value, weight)`
/// pairs sorted by value directly into at most `max_size` entries.
fn set_prune_sorted(sorted: &[(f32, f32)], max_size: usize, out: &mut Vec<Entry>) {
    out.clear();
    let Some((&(first_value, first_weight), rest)) = sorted.split_first() else {
        return;
    };
    let mut sum_total = 0f64;
    let mut unique_values = 0usize;
    for (i, &(v, w)) in sorted.iter().enumerate() {
        if i == 0 || sorted[i - 1].0 != v {
            unique_values += 1;
        }
        sum_total += f64::from(w);
    }

    // The entry of `value` with rank interval `[rmin, rmin + wmin]`.
    let entry = |rmin: f64, wmin: f64, value: f32| {
        Entry::new(rmin as f32, (rmin + wmin) as f32, wmin as f32, value)
    };
    let (mut rmin, mut wmin, mut last_value) = (0f64, f64::from(first_weight), first_value);
    if unique_values <= max_size {
        // Enough budget to keep every distinct value: exact weighted summary.
        for &(v, w) in rest {
            if last_value == v {
                wmin += f64::from(w);
                continue;
            }
            out.push(entry(rmin, wmin, last_value));
            rmin += wmin;
            last_value = v;
            wmin = f64::from(w);
        }
        out.push(entry(rmin, wmin, last_value));
        return;
    }

    // Upstream's `-1` sentinel (re)starts the scan; kept verbatim, since a
    // negative weight sum can bring `next_goal` back to `-1`.
    let mut next_goal = -1f64;
    for &(v, w) in sorted {
        if next_goal == -1.0 {
            next_goal = 0.0;
            last_value = v;
            wmin = f64::from(w);
            continue;
        }
        if last_value == v {
            wmin += f64::from(w);
            continue;
        }
        let rmax = rmin + wmin;
        let mut size = out.len();
        if rmax >= next_goal && size != max_size {
            if size == 0 || last_value > out[size - 1].value {
                out.push(entry(rmin, wmin, last_value));
                size += 1;
            }
            next_goal = if size == max_size {
                sum_total * 2.0 + f64::from(1e-5f32)
            } else {
                f64::from((size as f64 * sum_total / max_size as f64) as f32)
            };
        }
        rmin = rmax;
        wmin = f64::from(w);
        last_value = v;
    }
    if out.last().is_none_or(|e| last_value > e.value) {
        out.push(entry(rmin, wmin, last_value));
    }
}

/// `WQSummary::QueryCutValues`: materialise cut values from a summary. The
/// minimum value never becomes a cut; a sentinel above the maximum closes
/// the last bin.
fn query_cut_values(data: &[Entry], max_bin: usize, out: &mut Vec<f32>) {
    if data.is_empty() {
        out.push(1e-5);
        return;
    }
    let n = data.len();
    let advance = |mut cursor: usize, value: f32| {
        while cursor < n && data[cursor].value <= value {
            cursor += 1;
        }
        cursor
    };
    let mut last_cut = data[0].value;
    let mut next_value = advance(1, last_cut);
    if n <= max_bin {
        while next_value < n {
            let cpt = data[next_value].value;
            out.push(cpt);
            last_cut = cpt;
            next_value = advance(next_value + 1, last_cut);
        }
    } else {
        let total = f64::from(data[n - 1].rmax);
        let mut q = 0usize;
        for i in 1..max_bin {
            let rank2 = 2.0 * (i as f64 * total / max_bin as f64);
            while q < n - 2 && rank2 >= f64::from(data[q + 1].rmin + data[q + 1].rmax) {
                q += 1;
            }
            let queried = if rank2 < f64::from(data[q].rmin_next() + data[q + 1].rmax_prev()) {
                data[q]
            } else {
                data[q + 1]
            };
            let mut cpt = queried.value;
            if cpt <= last_cut {
                next_value = advance(next_value, last_cut);
                if next_value == n {
                    break;
                }
                cpt = data[next_value].value;
            } else if next_value < n && data[next_value].value <= cpt {
                next_value = advance(next_value + 1, cpt);
            }
            out.push(cpt);
            last_cut = cpt;
        }
    }
    let cpt = data[n - 1].value;
    out.push(cpt + (cpt.abs() + 1e-5));
}

/// A per-feature weighted quantile sketch (`WQuantileSketch`).
pub(crate) struct WQSketch {
    max_bin: usize,
    /// Per-level summary budget (`limit_size_`).
    limit_size: usize,
    /// Buffered `(value, weight)` pairs awaiting summarisation (`inqueue_`).
    queue: Vec<(f32, f32)>,
    /// Level summaries; `levels[l].len() <= limit_size`.
    levels: Vec<Vec<Entry>>,
    temp: Vec<Entry>,
    workspace: Vec<Entry>,
    /// Number of pushed values with non-zero weight.
    num_elements: usize,
    /// Every pushed weight is `1` ([`Self::with_unit_weights`]).
    unit_weights: bool,
    /// Scratch buffers of the radix sorts.
    sort_scratch: SketchScratch,
}

/// Radix sort buffers a [`WQSketch`] reuses, lent across sketches.
#[derive(Default)]
pub(crate) struct SketchScratch {
    pairs: RadixScratch<(f32, f32)>,
    /// The queue's values as [`sort_key`]s, when every weight is `1`.
    keys: Vec<u32>,
    key_sort: RadixScratch<u32>,
}

impl WQSketch {
    /// A sketch for a feature with `n_values` non-missing values and at most
    /// `max_bin` bins (upstream `HostSketchContainer` constructor).
    pub(crate) fn new(n_values: usize, max_bin: usize) -> Self {
        let limit_size = summary_budget(max_bin, n_values);
        WQSketch {
            max_bin,
            limit_size,
            queue: Vec::new(),
            levels: Vec::new(),
            temp: Vec::new(),
            workspace: Vec::new(),
            num_elements: 0,
            unit_weights: false,
            sort_scratch: SketchScratch::default(),
        }
    }

    /// Sort with `scratch`'s buffers (reused across sketches), returned by
    /// [`Self::into_sort_scratch`].
    pub(crate) fn with_sort_scratch(mut self, scratch: SketchScratch) -> Self {
        self.sort_scratch = scratch;
        self
    }

    /// The radix sort's buffers, for the next sketch.
    pub(crate) fn into_sort_scratch(self) -> SketchScratch {
        self.sort_scratch
    }

    /// `WQuantileSketch::Push`: add one value in row order.
    #[inline]
    pub(crate) fn push(&mut self, value: f32, weight: f32) {
        if weight == 0.0 {
            return;
        }
        self.num_elements += 1;
        if !self.queue_push(value, weight) {
            self.flush_queue();
            self.queue_push(value, weight);
        }
    }

    /// `Queue::Push`: merge into the last entry when the value repeats,
    /// otherwise append; `false` when the queue is full.
    #[inline]
    fn queue_push(&mut self, value: f32, weight: f32) -> bool {
        if let Some(last) = self.queue.last_mut()
            && last.0 == value
        {
            last.1 += weight;
            return true;
        }
        if self.queue.len() == 2 * self.limit_size {
            return false;
        }
        self.queue.push((value, weight));
        true
    }

    /// Mark every pushed weight as a small whole number (unweighted data,
    /// each weight `1`), so equal values sum to the same weight in any
    /// order and [`Self::flush_queue`] may sort with a radix sort.
    pub(crate) fn with_unit_weights(mut self) -> Self {
        self.unit_weights = true;
        self
    }

    /// `Queue::PopSummary` followed by `PushSummary`.
    fn flush_queue(&mut self) {
        // With unit weights every queued weight is a whole number below
        // `2^24` (a count of at most `num_elements` values), exact in `f32`,
        // so `set_from_sorted` sums equal values to the same weight however
        // the sort orders them: any value-ordered permutation gives the
        // summary the comparison sort gives.
        if self.unit_weights && self.num_elements < 1 << 24 {
            if self.queue.iter().all(|&(_, w)| w == 1.0) {
                // Every weight is `1` (no value repeated in a row): sort the
                // values alone, half the bytes the pairs would move.
                let SketchScratch { keys, key_sort, .. } = &mut self.sort_scratch;
                keys.clear();
                keys.extend(self.queue.iter().map(|&(v, _)| sort_key(v)));
                radix_sort(keys, key_sort, |&k| k);
                set_from_sorted_keys(keys, &mut self.temp);
                self.queue.clear();
                self.push_summary();
                return;
            }
            radix_sort(&mut self.queue, &mut self.sort_scratch.pairs, |&(v, _)| {
                sort_key(v)
            });
        } else {
            self.queue.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        }
        set_from_sorted(&self.queue, &mut self.temp);
        self.queue.clear();
        self.push_summary();
    }

    /// `WQuantileSketch::PushSummary`: level-wise merge/prune of `temp` with
    /// carry propagation.
    fn push_summary(&mut self) {
        let mut l = 0usize;
        loop {
            if self.levels.len() <= l {
                self.levels.push(Vec::new());
            }
            set_prune(&mut self.temp, self.limit_size);
            set_combine(&mut self.temp, &self.levels[l], &mut self.workspace);
            self.levels[l].clear();
            if self.temp.len() <= self.limit_size {
                break;
            }
            l += 1;
        }
        // `levels[l]` is empty here; `temp` is refilled before its next use.
        std::mem::swap(&mut self.levels[l], &mut self.temp);
    }

    /// `WQuantileSketch::PushSorted`: ingest a whole column of `(value,
    /// weight)` pairs sorted by value, pruned straight to `max_bin` entries.
    pub(crate) fn push_sorted(&mut self, sorted: &[(f32, f32)]) {
        self.num_elements += sorted.iter().filter(|&&(_, w)| w != 0.0).count();
        set_prune_sorted(sorted, self.max_bin, &mut self.temp);
        if !sorted.is_empty() {
            self.push_summary();
        }
    }

    /// `WQuantileSketch::GetSummary`: flush and merge every level into one
    /// summary of at most `max_size` entries.
    fn summary(&mut self, max_size: usize) -> Vec<Entry> {
        self.flush_queue();
        let prune_size = max_size.max(self.limit_size);
        let mut out = Vec::new();
        for level in &self.levels {
            set_combine(&mut out, level, &mut self.workspace);
            set_prune(&mut out, prune_size);
        }
        set_prune(&mut out, max_size);
        out
    }

    /// Append this feature's cut values (upstream `AllReduce` + `MakeCuts` for
    /// one worker): the final summary is pruned to the budget for the values
    /// actually seen, then queried for at most `max_bin` cuts plus the
    /// sentinel.
    pub(crate) fn cut_values(&mut self, out: &mut Vec<f32>) {
        let max_bin = self.max_bin;
        let budget = summary_budget(max_bin, self.num_elements);
        let summary = self.summary(budget);
        query_cut_values(&summary, summary.len().min(max_bin), out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuts_of(values: &[f32], max_bin: usize) -> Vec<f32> {
        let mut sketch = WQSketch::new(values.len(), max_bin);
        for &v in values {
            sketch.push(v, 1.0);
        }
        let mut out = Vec::new();
        sketch.cut_values(&mut out);
        out
    }

    /// The unit-weight sketch (radix-sorted queue) cuts exactly where the
    /// comparison-sorted one does, on streams with heavy repeats, signed
    /// zeros, negatives, and infinities, long enough to flush the queue, and
    /// on streams without consecutive repeats (queues of weight-1 values,
    /// sorted as bare keys).
    #[test]
    fn unit_weight_sketch_matches_comparison_sort() {
        let mut rng = crate::rng::Rng::new(3);
        for case in 0..9 {
            let n = 20_000 + case * 7_000;
            let values: Vec<f32> = (0..n)
                .map(|i| match if case < 6 { rng.range(0..10) } else { 9 } {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f32::NEG_INFINITY,
                    3 => ((i % 17) as f32) - 8.0,
                    _ => (rng.f32() - 0.5) * 10f32.powi(case as i32 % 6 - 2),
                })
                .collect();
            let mut radix = WQSketch::new(n, 64).with_unit_weights();
            for &v in &values {
                radix.push(v, 1.0);
            }
            let mut out = Vec::new();
            radix.cut_values(&mut out);
            let expected = cuts_of(&values, 64);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&out), bits(&expected), "case {case}");
        }
    }

    /// Upstream's `SetCombine` as one sequential merge: the reference the
    /// split, interleaved [`set_combine`] must reproduce bit for bit.
    fn set_combine_one_pass(this: &mut Vec<Entry>, other: &[Entry], workspace: &mut Vec<Entry>) {
        if other.is_empty() {
            return;
        }
        if this.is_empty() {
            this.extend_from_slice(other);
            return;
        }
        workspace.clear();
        let (a, b) = (this.as_slice(), other);
        let (mut ia, mut ib) = (0usize, 0usize);
        let (mut aprev_rmin, mut bprev_rmin) = (0f32, 0f32);
        while ia < a.len() && ib < b.len() {
            let (ea, eb) = (a[ia], b[ib]);
            if ea.value == eb.value {
                workspace.push(Entry::new(
                    ea.rmin + eb.rmin,
                    ea.rmax + eb.rmax,
                    ea.wmin + eb.wmin,
                    ea.value,
                ));
                aprev_rmin = ea.rmin_next();
                bprev_rmin = eb.rmin_next();
                ia += 1;
                ib += 1;
            } else if ea.value < eb.value {
                workspace.push(Entry::new(
                    ea.rmin + bprev_rmin,
                    ea.rmax + eb.rmax_prev(),
                    ea.wmin,
                    ea.value,
                ));
                aprev_rmin = ea.rmin_next();
                ia += 1;
            } else {
                workspace.push(Entry::new(
                    eb.rmin + aprev_rmin,
                    eb.rmax + ea.rmax_prev(),
                    eb.wmin,
                    eb.value,
                ));
                bprev_rmin = eb.rmin_next();
                ib += 1;
            }
        }
        if ia < a.len() {
            let brmax = b[b.len() - 1].rmax;
            for ea in &a[ia..] {
                workspace.push(Entry::new(
                    ea.rmin + bprev_rmin,
                    ea.rmax + brmax,
                    ea.wmin,
                    ea.value,
                ));
            }
        }
        if ib < b.len() {
            let armax = a[a.len() - 1].rmax;
            for eb in &b[ib..] {
                workspace.push(Entry::new(
                    eb.rmin + aprev_rmin,
                    eb.rmax + armax,
                    eb.wmin,
                    eb.value,
                ));
            }
        }
        fix_error(workspace);
        // The merged summary becomes `this`; the old one is the next scratch.
        std::mem::swap(this, workspace);
    }

    /// A summary of `n` distinct values drawn from `pool` (sorted, weights
    /// in `1..4`), pruned like the sketch's levels.
    fn random_summary(rng: &mut crate::rng::Rng, pool: &[f32], n: usize) -> Vec<Entry> {
        let mut values: Vec<f32> = (0..n).map(|_| pool[rng.range(0..pool.len())]).collect();
        values.sort_by(f32::total_cmp);
        let pairs: Vec<(f32, f32)> = values
            .iter()
            .map(|&v| (v, rng.range(1..4) as f32))
            .collect();
        let mut out = Vec::new();
        set_from_sorted(&pairs, &mut out);
        set_prune(&mut out, n / 2 + 1);
        out
    }

    #[test]
    fn split_combine_matches_the_one_pass_merge_bit_for_bit() {
        let mut rng = crate::rng::Rng::new(9);
        let bits = |v: &[Entry]| {
            v.iter()
                .map(|e| [e.rmin, e.rmax, e.wmin, e.value].map(f32::to_bits))
                .collect::<Vec<_>>()
        };
        for case in 0..40 {
            // Shared values across the two summaries (a small pool), signed
            // zeros, infinities, and in some cases a NaN (one-pass path).
            let mut pool: Vec<f32> = (0..[50, 4000, 100_000][case % 3])
                .map(|_| (rng.f32() - 0.5) * 100.0)
                .collect();
            pool.extend([-0.0, 0.0, f32::INFINITY, f32::NEG_INFINITY]);
            if case % 7 == 6 {
                pool.push(f32::NAN);
            }
            let n = rng.range(1..6000);
            let a = random_summary(&mut rng, &pool, n);
            let n = rng.range(1..6000);
            let b = random_summary(&mut rng, &pool, n);
            let (mut expected, mut workspace) = (a.clone(), Vec::new());
            set_combine_one_pass(&mut expected, &b, &mut workspace);
            let mut merged = a.clone();
            set_combine(&mut merged, &b, &mut workspace);
            assert_eq!(bits(&merged), bits(&expected), "case {case}");
        }
    }

    #[test]
    fn few_distinct_values_skip_the_minimum_and_add_sentinel() {
        // XGBoost never emits the minimum as a cut: bin 0 is (-inf, 1].
        assert_eq!(
            cuts_of(&[2.0, 0.0, 1.0, 1.0], 256),
            vec![1.0, 2.0, 2.0 + 2.0 + 1e-5]
        );
    }

    #[test]
    fn empty_feature_gets_the_upstream_placeholder() {
        assert_eq!(cuts_of(&[], 256), vec![1e-5]);
    }

    #[test]
    fn quantile_cuts_are_strictly_increasing_and_bounded() {
        let values: Vec<f32> = (0..5000)
            .map(|i| ((i * 7919) % 5000) as f32 * 0.001)
            .collect();
        let cuts = cuts_of(&values, 64);
        assert!(cuts.len() <= 64);
        assert!(cuts.windows(2).all(|w| w[0] < w[1]));
        assert!(*cuts.last().unwrap() > 4.999);
    }

    #[test]
    fn zero_weight_values_do_not_contribute() {
        let mut sketch = WQSketch::new(4, 256);
        for (v, w) in [(0.0, 1.0), (5.0, 0.0), (1.0, 1.0), (7.0, 0.0)] {
            sketch.push(v, w);
        }
        let mut out = Vec::new();
        sketch.cut_values(&mut out);
        assert_eq!(out, vec![1.0, 1.0 + 1.0 + 1e-5]);
    }

    #[test]
    fn sorted_path_matches_streaming_path_when_exact() {
        // With every distinct value retained both ingestion paths produce the
        // same exact summary, hence the same cuts.
        let values: Vec<f32> = (0..100).map(|i| (i % 37) as f32).collect();
        let streaming = cuts_of(&values, 256);
        let mut sorted: Vec<(f32, f32)> = values.iter().map(|&v| (v, 1.0)).collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut sketch = WQSketch::new(values.len(), 256);
        sketch.push_sorted(&sorted);
        let mut out = Vec::new();
        sketch.cut_values(&mut out);
        assert_eq!(out, streaming);
    }
}
