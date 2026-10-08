//! Resident split search: stable, unbounded-category device sorting and
//! one packed winner readback per node. Sorting workspaces are reused in
//! waves; histograms and unchosen feature results never cross the device.

use super::{
    CudaHistBackend, CudaSlice, CudaStream, DriverError, LaunchConfig, Pinned, SCAN_WARPS, abi,
    bytes_of, download_pinned, fit, pinned,
};
use crate::backend::cuda::diagnostics::ScanDiagnostics;
use crate::data::ghist::GHistIndex;
use crate::tree::gain::{GradStats, RegParams};
use crate::tree::hist::{NodeScan, ScanFallback, ScanRequest};
use cudarc::driver::PushKernelArg;
use std::sync::Arc;

/// Entries of the ordinary sorting workspace: keys plus two index buffers,
/// 12 bytes an entry, 48 MiB in all. A power of two, as `fit` rounds each
/// buffer up to one, so a wave of at most this many entries allocates no
/// more. A single wider feature needs its own full workspace, never a
/// category cap or host sort. The workspace is reused across the batch's
/// waves.
const SORT_ENTRIES: usize = 1 << 22;
const SET_BINS: usize = 63;
const HEADER_WORDS: usize = 7;

/// Reused sorting scratch, device results and pinned output (descriptors
/// travel in the backend's packed descriptor buffer).
pub(super) struct ScanState {
    numeric_meta: CudaSlice<u32>,
    numeric_acc: CudaSlice<f64>,
    categorical_tasks: CudaSlice<u64>,
    keys: CudaSlice<f32>,
    order: [CudaSlice<u32>; 2],
    categorical_meta: CudaSlice<u32>,
    categorical_children: CudaSlice<f64>,
    categorical_sets: CudaSlice<u32>,
    out: CudaSlice<u64>,
    pin: Option<Pinned<u64>>,
    diagnostics: ScanDiagnostics,
}

impl ScanState {
    pub(super) fn new(stream: &Arc<CudaStream>) -> std::result::Result<Self, DriverError> {
        Ok(Self {
            numeric_meta: stream.alloc_zeros(4)?,
            numeric_acc: stream.alloc_zeros(2)?,
            categorical_tasks: stream.alloc_zeros(4)?,
            keys: stream.alloc_zeros(1)?,
            order: [stream.alloc_zeros(1)?, stream.alloc_zeros(1)?],
            categorical_meta: stream.alloc_zeros(4)?,
            categorical_children: stream.alloc_zeros(4)?,
            categorical_sets: stream.alloc_zeros(SET_BINS)?,
            out: stream.alloc_zeros(HEADER_WORDS)?,
            pin: None,
            diagnostics: ScanDiagnostics::default(),
        })
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
    /// What the resident split searches did (`hessboost::internals`, for
    /// tests and benchmarks).
    #[doc(hidden)]
    #[must_use]
    pub fn scan_diagnostics(&self) -> ScanDiagnostics {
        self.state.lock().scan.diagnostics
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
        // The tree's gradients sum exactly over all of its rows, so every
        // histogram and total of the tree is exact and the scans may form
        // their prefixes in any association (the CPU's chain bits).
        let exact = u32::from(state.staged.sums_exact(state.tree_len));
        let regularization = abi::Regularization {
            lambda: reg.lambda,
            alpha: reg.alpha,
            max_delta_step: reg.max_delta_step,
            min_child_weight: reg.min_child_weight,
        };
        let scanned = (|| {
            let scan = &mut state.scan;
            let staging = &mut state.staging;
            let pool = &device.pinned;
            // One queued copy of the batch's requests and numeric tasks.
            let [d_totals, d_params, d_refs, d_first, d_out_at, d_numeric] = staging.upload_parts(
                pool,
                stream,
                &mut state.desc,
                [
                    bytes_of(&totals),
                    bytes_of(&params),
                    bytes_of(&refs),
                    bytes_of(&first),
                    bytes_of(&out_at),
                    bytes_of(&numeric),
                ],
            )?;
            fit(stream, &mut scan.out, output_words)?;
            let total_bins = self.total_bins as u64;
            let n_numeric = numeric.len() / 4;
            if n_numeric > 0 {
                fit(stream, &mut scan.numeric_meta, 4 * n_numeric)?;
                fit(stream, &mut scan.numeric_acc, 2 * n_numeric)?;
            }
            let numeric_out = abi::NumericResults {
                meta: abi::ptr_mut(&mut scan.numeric_meta, stream),
                acc: abi::ptr_mut(&mut scan.numeric_acc, stream),
            };
            if n_numeric > 0 {
                let work = abi::NumericTasks {
                    tasks: d_numeric,
                    n_tasks: n_numeric as u64,
                    totals: d_totals,
                    params: d_params,
                    dense: i32::from(self.dense),
                    exact,
                };
                let mut launch = stream.launch_builder(&device.kernels.scan_splits);
                launch
                    .arg(&state.pool)
                    .arg(&state.feature_first)
                    .arg(&total_bins)
                    .arg(&work)
                    .arg(&regularization)
                    .arg(&numeric_out);
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
                let capacity = SORT_ENTRIES;
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
                    staging.upload(pool, stream, &mut scan.categorical_tasks, &tasks)?;
                    fit(stream, &mut scan.keys, workspace)?;
                    for order in &mut scan.order {
                        fit(stream, order, workspace)?;
                    }
                    let n_tasks = (next - begin) as u32;
                    let wave = abi::CategoryTasks {
                        tasks: abi::ptr(&scan.categorical_tasks, stream),
                        n_tasks,
                    };
                    let sort_config = LaunchConfig {
                        grid_dim: (n_tasks.min(device.sm_count.saturating_mul(8)).max(1), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let sort = abi::SortWorkspace {
                        keys: abi::ptr_mut(&mut scan.keys, stream),
                        order: abi::ptr_mut(&mut scan.order[0], stream),
                    };
                    let mut keys = stream.launch_builder(&device.kernels.category_keys);
                    keys.arg(&state.pool)
                        .arg(&state.feature_first)
                        .arg(&total_bins)
                        .arg(&wave)
                        .arg(&regularization)
                        .arg(&sort)
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
                            .arg(&wave)
                            .arg(&scan.categorical_meta)
                            .arg(&width)
                            .arg(&scan.keys)
                            .arg(&*input)
                            .arg(output);
                        // SAFETY: stable merge ranks are a permutation of
                        // each feature's workspace; source/dest do not alias.
                        unsafe { merge.launch(sort_config) }?;
                        source = 1 - source;
                        width *= 2;
                    }
                    let work = abi::CategoricalTasks {
                        tasks: abi::ptr(&scan.categorical_tasks, stream),
                        n_tasks: u64::from(n_tasks),
                        totals: d_totals,
                        params: d_params,
                        exact,
                    };
                    let out = abi::CategoricalResults {
                        meta: abi::ptr_mut(&mut scan.categorical_meta, stream),
                        children: abi::ptr_mut(&mut scan.categorical_children, stream),
                        sets: abi::ptr_mut(&mut scan.categorical_sets, stream),
                    };
                    let mut search = stream.launch_builder(&device.kernels.scan_categorical);
                    search
                        .arg(&state.pool)
                        .arg(&state.feature_first)
                        .arg(&total_bins)
                        .arg(&work)
                        .arg(&regularization)
                        .arg(&scan.order[source])
                        .arg(&out);
                    // SAFETY: sorted indices cover each feature; one warp
                    // per task scores CPU-ordered prefix/suffix chains.
                    unsafe { search.launch(scan_config(n_tasks as usize)) }?;
                }
            }
            let categorical_out = abi::CategoricalResults {
                meta: abi::ptr_mut(&mut scan.categorical_meta, stream),
                children: abi::ptr_mut(&mut scan.categorical_children, stream),
                sets: abi::ptr_mut(&mut scan.categorical_sets, stream),
            };
            let count = requests.len() as u64;
            let mut merge = stream.launch_builder(&device.kernels.merge_scans);
            merge
                .arg(&d_refs)
                .arg(&d_first)
                .arg(&count)
                .arg(&numeric_out)
                .arg(&categorical_out)
                .arg(&d_out_at)
                .arg(&mut scan.out);
            // SAFETY: refs name completed results and node headers have
            // disjoint spans large enough for their maximum chosen set; one
            // warp per node (the kernel strides over nodes).
            unsafe { merge.launch(scan_config(requests.len())) }?;
            let readback = pinned(pool, stream, &mut scan.pin, output_words, false)?;
            let packed = download_pinned(stream, readback, &scan.out, output_words)?;
            staging.synced();
            let counts = &mut scan.diagnostics;
            counts.device_nodes += requests.len() as u64;
            counts.numeric_features += n_numeric as u64;
            counts.categorical_features += categorical.len() as u64;
            counts.winner_readback_bytes += (output_words * 8) as u64;
            if exact != 0 {
                counts.exact_nodes += requests.len() as u64;
            }
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
                        let counts = &mut scan.diagnostics;
                        NodeScan::Replay(match fallback {
                            3 => {
                                counts.numeric_score_replays += 1;
                                ScanFallback::NumericScore
                            }
                            4 => {
                                counts.categorical_order_replays += 1;
                                ScanFallback::CategoricalOrder
                            }
                            _ => {
                                counts.categorical_score_replays += 1;
                                ScanFallback::CategoricalScore
                            }
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
