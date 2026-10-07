//! Resident split search: stable, unbounded-category device sorting and
//! one packed winner readback per node. Sorting workspaces are reused in
//! waves; histograms and unchosen feature results never cross the device.

use super::{
    CudaHistBackend, CudaSlice, CudaStream, DriverError, LaunchConfig, Pinned, SCAN_WARPS,
    download_pinned, fit, pinned, upload,
};
use crate::data::ghist::GHistIndex;
use crate::tree::gain::{GradStats, RegParams};
use crate::tree::hist::{NodeScan, ScanFallback, ScanRequest};
use cudarc::driver::PushKernelArg;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum ordinary sorting workspace (keys plus two index buffers).
/// A single wider feature needs its own full workspace, never a category
/// cap or host sort. The workspace is reused across the batch's waves.
const SORT_BYTES: usize = 64 << 20;
const SET_BINS: usize = 63;
const HEADER_WORDS: usize = 7;

/// Resident device work and searches that required the CPU's non-total
/// comparisons. Semantic fallbacks are not CUDA failures or category caps.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ScanDiagnostics {
    /// Nodes whose feature results were merged on the device.
    pub device_nodes: u64,
    /// Numeric features scanned on the device.
    pub numeric_features: u64,
    /// Categorical features searched on the device.
    pub categorical_features: u64,
    /// Bytes read back for packed per-node winners, including their chosen
    /// category bins but excluding explicit NaN histogram replays.
    pub winner_readback_bytes: u64,
    /// Nodes replayed because a numeric candidate scored NaN.
    pub numeric_score_replays: u64,
    /// Nodes replayed because a categorical sort key was non-finite.
    pub categorical_order_replays: u64,
    /// Nodes replayed because a categorical candidate scored NaN.
    pub categorical_score_replays: u64,
}

/// Reused descriptors, sorting scratch, device results and pinned output.
pub(super) struct ScanState {
    numeric_tasks: CudaSlice<u32>,
    totals: CudaSlice<f64>,
    params: CudaSlice<f32>,
    numeric_meta: CudaSlice<u32>,
    numeric_acc: CudaSlice<f64>,
    categorical_tasks: CudaSlice<u64>,
    keys: CudaSlice<f32>,
    order: [CudaSlice<u32>; 2],
    categorical_meta: CudaSlice<u32>,
    categorical_children: CudaSlice<f64>,
    categorical_sets: CudaSlice<u32>,
    refs: CudaSlice<u32>,
    first: CudaSlice<u32>,
    out_at: CudaSlice<u64>,
    out: CudaSlice<u64>,
    pin: Option<Pinned<u64>>,
    replays: [AtomicU64; 3],
    work: [AtomicU64; 4],
}

impl ScanState {
    pub(super) fn new(stream: &Arc<CudaStream>) -> std::result::Result<Self, DriverError> {
        Ok(Self {
            numeric_tasks: stream.alloc_zeros(4)?,
            totals: stream.alloc_zeros(2)?,
            params: stream.alloc_zeros(3)?,
            numeric_meta: stream.alloc_zeros(4)?,
            numeric_acc: stream.alloc_zeros(2)?,
            categorical_tasks: stream.alloc_zeros(4)?,
            keys: stream.alloc_zeros(1)?,
            order: [stream.alloc_zeros(1)?, stream.alloc_zeros(1)?],
            categorical_meta: stream.alloc_zeros(4)?,
            categorical_children: stream.alloc_zeros(4)?,
            categorical_sets: stream.alloc_zeros(SET_BINS)?,
            refs: stream.alloc_zeros(3)?,
            first: stream.alloc_zeros(2)?,
            out_at: stream.alloc_zeros(2)?,
            out: stream.alloc_zeros(HEADER_WORDS)?,
            pin: None,
            replays: std::array::from_fn(|_| AtomicU64::new(0)),
            work: std::array::from_fn(|_| AtomicU64::new(0)),
        })
    }

    fn snapshot(&self) -> ScanDiagnostics {
        ScanDiagnostics {
            device_nodes: self.work[0].load(Ordering::Relaxed),
            numeric_features: self.work[1].load(Ordering::Relaxed),
            categorical_features: self.work[2].load(Ordering::Relaxed),
            winner_readback_bytes: self.work[3].load(Ordering::Relaxed),
            numeric_score_replays: self.replays[0].load(Ordering::Relaxed),
            categorical_order_replays: self.replays[1].load(Ordering::Relaxed),
            categorical_score_replays: self.replays[2].load(Ordering::Relaxed),
        }
    }
}

/// A category task before it is assigned its wave's scratch span.
struct CategoryTask {
    request: u32,
    feature: u32,
    slot: u32,
    direction: i8,
    bins: usize,
}

fn scan_config(tasks: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (tasks.div_ceil(SCAN_WARPS).max(1) as u32, 1, 1),
        block_dim: ((32 * SCAN_WARPS) as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl CudaHistBackend {
    /// Counters for resident split searches replayed on the host to retain
    /// the CPU's NaN comparisons. Ordinary numeric and categorical searches
    /// of either growth policy run entirely on the device.
    #[must_use]
    pub fn scan_diagnostics(&self) -> ScanDiagnostics {
        self.state.lock().scan.snapshot()
    }

    #[cfg(test)]
    pub(crate) fn replace_resident_hist_for_test(
        &self,
        slot: u32,
        hist: &[GradStats],
    ) -> Option<()> {
        let mut state = self.lock()?;
        if slot as usize >= state.pool_slots || hist.len() != self.total_bins {
            return None;
        }
        let flat: Vec<_> = hist.iter().flat_map(|s| [s.grad, s.hess]).collect();
        let offset = slot as usize * self.total_bins * 2;
        let copied = super::copy_host(
            &self.device.stream,
            &flat,
            &mut state.pool.slice_mut(offset..offset + flat.len()),
        );
        self.ok(copied)
    }

    pub(super) fn scan_on(
        &self,
        ghist: &GHistIndex,
        reg: &RegParams,
        requests: &[ScanRequest<'_>],
    ) -> Option<Vec<NodeScan>> {
        if requests.is_empty() {
            return Some(Vec::new());
        }
        let mut state = self.lock()?;
        if !self.fits(ghist)
            || requests.len() > u32::MAX as usize
            || requests.iter().any(|request| {
                request.slot as usize >= state.pool_slots
                    || request
                        .features
                        .iter()
                        .any(|&(f, _)| f as usize >= self.n_cols)
            })
        {
            return None;
        }
        let cuts = ghist.cuts();
        let mut numeric = Vec::new();
        let mut categorical = Vec::new();
        let mut totals = Vec::with_capacity(requests.len() * 2);
        let mut params = Vec::with_capacity(requests.len() * 3);
        let mut refs = Vec::new();
        let mut first = Vec::with_capacity(requests.len() + 1);
        let mut out_at = Vec::with_capacity(requests.len() + 1);
        let mut output_words = 0;
        for (i, request) in requests.iter().enumerate() {
            first.push(u32::try_from(refs.len() / 3).ok()?);
            out_at.push(output_words as u64);
            totals.extend([request.total.grad, request.total.hess]);
            params.extend([request.root_gain, request.lower, request.upper]);
            let mut selected_bins = 0;
            for &(feature, direction) in request.features {
                let (start, end) = cuts.feature_bins(feature as usize);
                if cuts.is_categorical(feature as usize) {
                    let bins = end - start;
                    if bins == 0 {
                        continue;
                    }
                    let result = u32::try_from(categorical.len()).ok()?;
                    refs.extend([feature, 1, result]);
                    categorical.push(CategoryTask {
                        request: i as u32,
                        feature,
                        slot: request.slot,
                        direction,
                        bins,
                    });
                    selected_bins =
                        selected_bins.max(if bins < 4 { 1 } else { SET_BINS.min(bins - 1) });
                } else if end > start + 1 {
                    refs.extend([feature, 0, u32::try_from(numeric.len() / 4).ok()?]);
                    numeric.extend([i as u32, feature, request.slot, i32::from(direction) as u32]);
                }
            }
            // Numeric-only batches need no second child or category padding.
            output_words += if selected_bins == 0 {
                5
            } else {
                HEADER_WORDS + selected_bins.div_ceil(2)
            };
        }
        first.push(u32::try_from(refs.len() / 3).ok()?);
        out_at.push(output_words as u64);
        let device = &*self.device;
        let stream = &device.stream;
        let state = &mut *state;
        let scanned = (|| {
            let scan = &mut state.scan;
            upload(stream, &mut scan.totals, &totals)?;
            upload(stream, &mut scan.params, &params)?;
            upload(stream, &mut scan.refs, &refs)?;
            upload(stream, &mut scan.first, &first)?;
            upload(stream, &mut scan.out_at, &out_at)?;
            fit(stream, &mut scan.out, output_words)?;
            let total_bins = self.total_bins as u64;
            let n_numeric = numeric.len() / 4;
            if n_numeric > 0 {
                upload(stream, &mut scan.numeric_tasks, &numeric)?;
                fit(stream, &mut scan.numeric_meta, 4 * n_numeric)?;
                fit(stream, &mut scan.numeric_acc, 2 * n_numeric)?;
                let count = n_numeric as u64;
                let dense = i32::from(self.dense);
                let mut launch = stream.launch_builder(&device.kernels.scan_splits);
                launch
                    .arg(&state.pool)
                    .arg(&state.feature_first)
                    .arg(&total_bins)
                    .arg(&scan.numeric_tasks)
                    .arg(&count)
                    .arg(&scan.totals)
                    .arg(&scan.params)
                    .arg(&reg.lambda)
                    .arg(&reg.alpha)
                    .arg(&reg.max_delta_step)
                    .arg(&reg.min_child_weight)
                    .arg(&dense)
                    .arg(&mut scan.numeric_meta)
                    .arg(&mut scan.numeric_acc);
                // SAFETY: validated histogram slots and feature ids, four
                // descriptor words per task, one whole warp per task.
                unsafe { launch.launch(scan_config(n_numeric)) }?;
            }
            if !categorical.is_empty() {
                fit(stream, &mut scan.categorical_meta, 4 * categorical.len())?;
                stream.memset_zeros(&mut scan.categorical_meta)?;
                fit(
                    stream,
                    &mut scan.categorical_children,
                    4 * categorical.len(),
                )?;
                fit(
                    stream,
                    &mut scan.categorical_sets,
                    SET_BINS * categorical.len(),
                )?;
                let capacity = SORT_BYTES / 12;
                let mut next = 0;
                while next < categorical.len() {
                    let begin = next;
                    let mut workspace = 0usize;
                    let mut max_bins = 0;
                    let mut tasks = Vec::new();
                    while next < categorical.len() {
                        let task = &categorical[next];
                        if next > begin && workspace.saturating_add(task.bins) > capacity {
                            break;
                        }
                        tasks.extend([
                            u64::from(task.request) | (u64::from(task.feature) << 32),
                            u64::from(task.slot)
                                | (u64::from(i32::from(task.direction) as u32) << 32),
                            next as u64,
                            workspace as u64,
                        ]);
                        workspace += task.bins;
                        max_bins = max_bins.max(task.bins);
                        next += 1;
                    }
                    upload(stream, &mut scan.categorical_tasks, &tasks)?;
                    fit(stream, &mut scan.keys, workspace)?;
                    for order in &mut scan.order {
                        fit(stream, order, workspace)?;
                    }
                    let n_tasks = (next - begin) as u32;
                    let sort_config = LaunchConfig {
                        grid_dim: (n_tasks.min(device.sm_count.saturating_mul(8)).max(1), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut keys = stream.launch_builder(&device.kernels.category_keys);
                    keys.arg(&state.pool)
                        .arg(&state.feature_first)
                        .arg(&total_bins)
                        .arg(&scan.categorical_tasks)
                        .arg(&n_tasks)
                        .arg(&reg.lambda)
                        .arg(&reg.alpha)
                        .arg(&reg.max_delta_step)
                        .arg(&reg.min_child_weight)
                        .arg(&mut scan.keys)
                        .arg(&mut scan.order[0])
                        .arg(&mut scan.categorical_meta);
                    // SAFETY: each task's workspace range is disjoint and
                    // sized to its complete feature bin range.
                    unsafe { keys.launch(sort_config) }?;
                    let mut source = 0;
                    let mut width = 1u64;
                    while width < max_bins as u64 {
                        let [left, right] = &mut scan.order;
                        let (input, output) = if source == 0 {
                            (left, right)
                        } else {
                            (right, left)
                        };
                        let mut merge = stream.launch_builder(&device.kernels.category_merge);
                        merge
                            .arg(&state.feature_first)
                            .arg(&scan.categorical_tasks)
                            .arg(&n_tasks)
                            .arg(&width)
                            .arg(&scan.keys)
                            .arg(&*input)
                            .arg(output)
                            .arg(&scan.categorical_meta);
                        // SAFETY: stable merge ranks are a permutation of
                        // each feature's workspace; source/dest do not alias.
                        unsafe { merge.launch(sort_config) }?;
                        source = 1 - source;
                        width *= 2;
                    }
                    let count = u64::from(n_tasks);
                    let mut search = stream.launch_builder(&device.kernels.scan_categorical);
                    search
                        .arg(&state.pool)
                        .arg(&state.feature_first)
                        .arg(&total_bins)
                        .arg(&scan.categorical_tasks)
                        .arg(&count)
                        .arg(&scan.totals)
                        .arg(&scan.params)
                        .arg(&reg.lambda)
                        .arg(&reg.alpha)
                        .arg(&reg.max_delta_step)
                        .arg(&reg.min_child_weight)
                        .arg(&scan.order[source])
                        .arg(&mut scan.categorical_meta)
                        .arg(&mut scan.categorical_children)
                        .arg(&mut scan.categorical_sets);
                    // SAFETY: sorted indices cover each feature; one warp
                    // per task scores CPU-ordered prefix/suffix chains.
                    unsafe { search.launch(scan_config(n_tasks as usize)) }?;
                }
            }
            let count = requests.len() as u64;
            let mut merge = stream.launch_builder(&device.kernels.merge_scans);
            merge
                .arg(&scan.refs)
                .arg(&scan.first)
                .arg(&count)
                .arg(&scan.numeric_meta)
                .arg(&scan.numeric_acc)
                .arg(&scan.categorical_meta)
                .arg(&scan.categorical_children)
                .arg(&scan.categorical_sets)
                .arg(&scan.out_at)
                .arg(&mut scan.out);
            // SAFETY: refs name completed results and node headers have
            // disjoint spans large enough for their maximum chosen set.
            unsafe { merge.launch(device.grid(requests.len())) }?;
            let staging = pinned(stream, &mut scan.pin, output_words, false)?;
            let packed = download_pinned(stream, staging, &scan.out, output_words)?;
            scan.work[0].fetch_add(requests.len() as u64, Ordering::Relaxed);
            scan.work[1].fetch_add(n_numeric as u64, Ordering::Relaxed);
            scan.work[2].fetch_add(categorical.len() as u64, Ordering::Relaxed);
            scan.work[3].fetch_add((output_words * 8) as u64, Ordering::Relaxed);
            let mut results = Vec::with_capacity(requests.len());
            for &offset in &out_at[..requests.len()] {
                let words = &packed[offset as usize..];
                let status = words[0] as u32;
                let feature = (words[0] >> 32) as u32;
                let result = match status {
                    0 => NodeScan::Empty,
                    1 => NodeScan::Numeric {
                        feature,
                        loss_chg: f32::from_bits(words[2] as u32),
                        backward: words[1] >> 32 != 0,
                        offset: words[1] as u32,
                        acc: GradStats::new(f64::from_bits(words[3]), f64::from_bits(words[4])),
                    },
                    2 => {
                        let count = (words[1] as u32) as usize;
                        let flags = (words[1] >> 32) as u32;
                        let indices = (0..count).map(|i| {
                            let word = words[HEADER_WORDS + i / 2];
                            if i % 2 == 0 {
                                word as u32
                            } else {
                                (word >> 32) as u32
                            }
                        });
                        let (start, end) = cuts.feature_bins(feature as usize);
                        let categories = category_set(cuts, start, end, indices, flags & 2 != 0);
                        NodeScan::Categorical {
                            feature,
                            loss_chg: f32::from_bits(words[2] as u32),
                            default_left: flags & 1 != 0,
                            left: GradStats::new(
                                f64::from_bits(words[3]),
                                f64::from_bits(words[4]),
                            ),
                            right: GradStats::new(
                                f64::from_bits(words[5]),
                                f64::from_bits(words[6]),
                            ),
                            categories,
                        }
                    }
                    fallback => {
                        let index = (fallback - 3) as usize;
                        scan.replays[index].fetch_add(1, Ordering::Relaxed);
                        NodeScan::Replay(match index {
                            0 => ScanFallback::NumericScore,
                            1 => ScanFallback::CategoricalOrder,
                            _ => ScanFallback::CategoricalScore,
                        })
                    }
                };
                results.push(result);
            }
            Ok(results)
        })();
        self.ok(scanned)
    }
}

/// Backward categorical partitions select the complement of at most 63
/// suffix bins. Reconstruct that set from immutable cuts, not a device
/// histogram or the full sorted category order.
fn category_set(
    cuts: &crate::data::quantile::HistCuts,
    start: usize,
    end: usize,
    indices: impl Iterator<Item = u32>,
    complement: bool,
) -> Vec<u32> {
    let mut indices: Vec<u32> = indices.collect();
    indices.sort_unstable();
    if complement {
        let mut excluded = indices.into_iter().peekable();
        (start..end)
            .filter(|&bin| {
                if excluded.peek().copied() == Some((bin - start) as u32) {
                    excluded.next();
                    false
                } else {
                    true
                }
            })
            .map(|bin| cuts.cut_value(bin) as u32)
            .collect()
    } else {
        indices
            .into_iter()
            .map(|bin| cuts.cut_value(start + bin as usize) as u32)
            .collect()
    }
}
