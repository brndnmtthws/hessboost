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
        let limit = ((nlevel as f64 / internal_eps).ceil() as usize + 1).min(maxn);
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

/// `WQSummary::SetCombine`: merge `other` into `this`. `workspace` is scratch
/// reused across calls.
fn set_combine(this: &mut Vec<Entry>, other: &[Entry], workspace: &mut Vec<Entry>) {
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
    this.clear();
    this.extend_from_slice(workspace);
}

/// `WQSummary::SetPruneSorted`: summarise a whole column of `(value, weight)`
/// pairs sorted by value directly into at most `max_size` entries.
fn set_prune_sorted(sorted: &[(f32, f32)], max_size: usize, out: &mut Vec<Entry>) {
    out.clear();
    if sorted.is_empty() {
        return;
    }
    let mut sum_total = 0f64;
    let mut unique_values = 0usize;
    for (i, &(v, w)) in sorted.iter().enumerate() {
        if i == 0 || sorted[i - 1].0 != v {
            unique_values += 1;
        }
        sum_total += w as f64;
    }

    let (mut rmin, mut wmin) = (0f64, 0f64);
    let mut last_value = 0f32;
    if unique_values <= max_size {
        // Enough budget to keep every distinct value: exact weighted summary.
        for (i, &(v, w)) in sorted.iter().enumerate() {
            if i == 0 {
                last_value = v;
                wmin = w as f64;
                continue;
            }
            if last_value == v {
                wmin += w as f64;
                continue;
            }
            let rmax = rmin + wmin;
            out.push(Entry::new(
                rmin as f32,
                rmax as f32,
                wmin as f32,
                last_value,
            ));
            rmin = rmax;
            last_value = v;
            wmin = w as f64;
        }
        let rmax = rmin + wmin;
        out.push(Entry::new(
            rmin as f32,
            rmax as f32,
            wmin as f32,
            last_value,
        ));
        return;
    }

    let mut next_goal = -1f64;
    for &(v, w) in sorted {
        if next_goal == -1.0 {
            next_goal = 0.0;
            last_value = v;
            wmin = w as f64;
            continue;
        }
        if last_value == v {
            wmin += w as f64;
        } else {
            let rmax = rmin + wmin;
            let mut size = out.len();
            if rmax >= next_goal && size != max_size {
                if size == 0 || last_value > out[size - 1].value {
                    out.push(Entry::new(
                        rmin as f32,
                        rmax as f32,
                        wmin as f32,
                        last_value,
                    ));
                    size += 1;
                }
                next_goal = if size == max_size {
                    sum_total * 2.0 + 1e-5f32 as f64
                } else {
                    (size as f64 * sum_total / max_size as f64) as f32 as f64
                };
            }
            rmin = rmax;
            wmin = w as f64;
            last_value = v;
        }
    }
    let size = out.len();
    if size == 0 || last_value > out[size - 1].value {
        let rmax = rmin + wmin;
        out.push(Entry::new(
            rmin as f32,
            rmax as f32,
            wmin as f32,
            last_value,
        ));
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
        let total = data[n - 1].rmax as f64;
        let mut q = 0usize;
        for i in 1..max_bin {
            let rank2 = 2.0 * (i as f64 * total / max_bin as f64);
            while q < n - 2 && rank2 >= (data[q + 1].rmin + data[q + 1].rmax) as f64 {
                q += 1;
            }
            let queried = if rank2 < (data[q].rmin_next() + data[q + 1].rmax_prev()) as f64 {
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
}

impl WQSketch {
    /// A sketch for a feature with `n_values` non-missing values and at most
    /// `max_bin` bins (upstream `HostSketchContainer` constructor).
    pub(crate) fn new(n_values: usize, max_bin: usize) -> Self {
        let limit_size = limit_size_level(n_values, sketch_epsilon(max_bin, n_values));
        WQSketch {
            max_bin,
            limit_size,
            queue: Vec::new(),
            levels: Vec::new(),
            temp: Vec::new(),
            workspace: Vec::new(),
            num_elements: 0,
        }
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
        if let Some(last) = self.queue.last_mut() {
            if last.0 == value {
                last.1 += weight;
                return true;
            }
        }
        if self.queue.len() == 2 * self.limit_size {
            return false;
        }
        self.queue.push((value, weight));
        true
    }

    /// `Queue::PopSummary` followed by `PushSummary`.
    fn flush_queue(&mut self) {
        self.queue.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
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
        self.levels[l].extend_from_slice(&self.temp);
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
    fn summary(mut self, max_size: usize) -> Vec<Entry> {
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
    pub(crate) fn cut_values(self, out: &mut Vec<f32>) {
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
