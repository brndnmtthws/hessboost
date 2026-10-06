//! NVIDIA CUDA acceleration for Linux (opt-in `cuda` feature).
//!
//! **Training** (`device = cuda`, see
//! [`TrainingParams::device`](crate::config::TrainingParams::device)): the
//! histogram construction of `tree_method = hist` runs on the GPU, and the
//! trained model is bit-identical to single-threaded CPU training (and so to
//! CPU training at any thread count). This first backend is a correctness
//! path, not yet a speedup: each node's row list is uploaded and its
//! histogram read back, one node at a time, through the same per-node seam
//! the CPU builder calls. Device-resident tree growth (level-batched
//! histograms, partition, and split search on the GPU) is the next step.
//!
//! # Requirements
//!
//! - Linux with an NVIDIA GPU and a driver supporting CUDA 12.8 or later.
//! - NVRTC (`libnvrtc`), loadable as `libnvrtc.so` or `libnvrtc.so.12`:
//!   the CUDA toolkit's `lib64` directory on the loader path (CUDA 13
//!   toolkits ship `libnvrtc.so` there), or `LD_LIBRARY_PATH` pointing at it.
//!   The kernels are compiled once per process, for the device's own
//!   architecture, straight to machine code (CUBIN), so no PTX JIT runs and
//!   a newer NVRTC than the driver is not a problem.
//! - Nothing at build time: both libraries are opened at run time, and
//!   their absence makes the backend unavailable rather than failing to
//!   load the crate.
//!
//! # Exactness
//!
//! The CPU adds each histogram bin in `f64` in a fixed order: one chain in
//! row order, or fixed chunks of rows each chained from zero and then added
//! in chunk order (`SumOrder`, chosen from the index and the row list
//! alone). Every node uses the first strategy that applies:
//!
//! 1. **Exact integers.** When every sum of the node's rows is exact
//!    (`n * max <= 2^53` gradient grains for both components, the domain
//!    the Metal backend also uses; proof in the private `exact_sum` module),
//!    the GPU sums 64-bit grain counts with atomics, in any order, and
//!    scales them back exactly.
//! 2. **Exact chunks.** For a chunked node whose *chunks* are exact, each
//!    chunk's integer sum is that chunk's `f64` chain, and the GPU then adds
//!    the chunk partials in chunk order in `f64`, the CPU's own operations.
//! 3. **Chains.** Otherwise, for a chunked node or a node below 8,192 rows,
//!    one GPU thread per (chunk, feature) runs that feature's `f64` chain in
//!    row order, and the chunk partials are added in chunk order.
//! 4. **CPU.** A single-chain node of 8,192 rows or more outside the exact
//!    domain (a dense index's unsampled root, or any node of a dense index
//!    of at most 2^18 rows) runs the CPU backend's build: one GPU thread
//!    per feature over that many rows would be slower than the CPU.
//!
//! The kernels are compiled without FP contraction, flush-to-zero, or
//! approximate division, so every `f64` operation is the single IEEE
//! operation the CPU performs; there are no floating-point atomics. Trees
//! with a non-finite gradient or Hessian (NaN payloads differ between CPU
//! and GPU arithmetic), nodes whose inputs do not match the index the
//! backend was built from, and every node after a CUDA error (errors are
//! sticky: the context is unusable afterwards) run on the CPU backend, so
//! the result is unchanged.
//!
//! # Limitations
//!
//! - Training only; prediction stays on the CPU.
//! - Refused with `device = cuda`: `tree_method = exact`/`approx`,
//!   `use_quantized_grad`, `gblinear`, `process_type = update`,
//!   `multi_strategy = multi_output_tree`, online updates, and budget mode.
//! - The whole binned index (2 or 4 bytes per stored entry), the gradients
//!   (24 bytes per row) and one row list are resident on the device.

pub(crate) mod compile;

use crate::backend::exact_sum::SumDomain;
use crate::data::ghist::{Bins, GHistIndex};
use crate::error::{HessboostError, Result};
use crate::objective::GradPair;
use crate::tree::gain::GradStats;
use crate::tree::hist::{CpuBackend, HistogramBackend, PARALLEL_THRESHOLD, SumOrder, sum_order};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg, sys,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Threads per block of every kernel.
const THREADS: u32 = 256;
/// Grid-stride kernels launch at most this many blocks per SM.
const BLOCKS_PER_SM: u32 = 8;
/// Bytes of per-chunk partial histograms held at once. A chunked node with
/// more chunks is built in waves of chunks, each reduced into the output in
/// chunk order before the next, as the CPU's waves are.
const PARTIAL_BYTES: usize = 256 << 20;
/// The oldest driver the backend runs on (`cuDriverGetVersion` encoding):
/// the API version the bindings are built against, CUDA 12.8.
const MIN_DRIVER: i32 = 12_080;

/// Whether CUDA device 0 is available: the driver and NVRTC load, the
/// device exists, and the kernels compile for it.
#[must_use]
pub fn available() -> bool {
    device(0).is_ok()
}

/// Why CUDA device 0 is unavailable (no driver, no NVRTC, no device, or a
/// kernel compile failure), for diagnostics; `None` when it is available.
#[must_use]
pub fn unavailable_reason() -> Option<String> {
    device(0).err()
}

/// The name of CUDA device 0, if it is available (for diagnostics and
/// benchmarks).
#[must_use]
pub fn device_name() -> Option<String> {
    device(0).ok().map(|device| device.name.clone())
}

/// The kernels of one device's module.
struct Kernels {
    stage_units: CudaFunction,
    units_dense: [CudaFunction; 2],
    units_csr: [CudaFunction; 2],
    chain_dense: [CudaFunction; 2],
    chain_csr: [CudaFunction; 2],
    reduce_units: CudaFunction,
    reduce_chains: CudaFunction,
}

/// One opened CUDA device: its primary context, the stream every backend
/// on it uses, and its compiled kernels.
struct Device {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    name: String,
    sm_count: u32,
    /// Set by the first CUDA error: the context is unusable afterwards, so
    /// every later build on this device runs on the CPU.
    failed: AtomicBool,
}

impl Device {
    fn open(ordinal: usize) -> std::result::Result<Self, String> {
        libraries()?;
        let count = match CudaContext::device_count() {
            Ok(count) => count,
            Err(e) if e.0 == sys::CUresult::CUDA_ERROR_NO_DEVICE => 0,
            Err(e) => return Err(format!("CUDA init failed: {e}")),
        };
        let count = usize::try_from(count).unwrap_or(0);
        if ordinal >= count {
            return Err(format!("no CUDA device {ordinal} ({count} found)"));
        }
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("CUDA context: {e}"))?;
        // SAFETY: called before this context allocates any slice, and every
        // slice of it is used on the one stream below only, so no slice
        // needs cross-stream event tracking.
        unsafe { ctx.disable_event_tracking() };
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| format!("CUDA compute capability: {e}"))?;
        let arch = format!("sm_{major}{minor}");
        let module = compile::load(&ctx, &arch)?;
        let function = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| format!("CUDA kernel `{name}`: {e}"))
        };
        let kernels = Kernels {
            stage_units: function("stage_units")?,
            units_dense: [
                function("hist_units_dense_u16")?,
                function("hist_units_dense_u32")?,
            ],
            units_csr: [
                function("hist_units_csr_u16")?,
                function("hist_units_csr_u32")?,
            ],
            chain_dense: [
                function("hist_chain_dense_u16")?,
                function("hist_chain_dense_u32")?,
            ],
            chain_csr: [
                function("hist_chain_csr_u16")?,
                function("hist_chain_csr_u32")?,
            ],
            reduce_units: function("reduce_units")?,
            reduce_chains: function("reduce_chains")?,
        };
        let stream = ctx.new_stream().map_err(|e| format!("CUDA stream: {e}"))?;
        let name = ctx.name().map_err(|e| format!("CUDA device name: {e}"))?;
        let sm_count = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| format!("CUDA SM count: {e}"))?;
        Ok(Device {
            stream,
            kernels,
            name,
            sm_count: u32::try_from(sm_count).unwrap_or(1).max(1),
            failed: AtomicBool::new(false),
        })
    }

    /// A launch shape for a grid-stride kernel over `work` items.
    fn grid(&self, work: usize) -> LaunchConfig {
        let blocks = work.div_ceil(THREADS as usize).max(1);
        let cap = (self.sm_count * BLOCKS_PER_SM) as usize;
        LaunchConfig {
            grid_dim: (blocks.min(cap) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// A launch shape with one thread per item (no grid stride).
    fn one_per(work: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (work.div_ceil(THREADS as usize).max(1) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// The driver and NVRTC both load, and the driver is new enough. Checked
/// before any other `cudarc` call: its lazy loaders panic when a library
/// is missing, and the release profile aborts on panic.
fn libraries() -> std::result::Result<(), String> {
    // SAFETY: only tries to open the shared libraries by name.
    if !unsafe { sys::is_culib_present() } {
        return Err("libcuda not found (no NVIDIA driver is installed)".into());
    }
    // SAFETY: as above.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(
            "libnvrtc not found: install the CUDA toolkit's NVRTC and put the \
                    directory holding `libnvrtc.so` on the loader path"
                .into(),
        );
    }
    let mut version = 0;
    // SAFETY: the driver library loads (checked above), and the call only
    // writes the version through the pointer.
    unsafe { sys::cuDriverGetVersion(&raw mut version) }
        .result()
        .map_err(|e| format!("CUDA driver version: {e}"))?;
    if version < MIN_DRIVER {
        return Err(format!(
            "the NVIDIA driver supports CUDA {}.{}; the backend needs 12.8 or later",
            version / 1000,
            version % 1000 / 10
        ));
    }
    Ok(())
}

/// The opened devices, by ordinal (each opened once per process; a failure
/// is remembered too).
static DEVICES: Mutex<Vec<(usize, Opened)>> = Mutex::new(Vec::new());

/// A device opened once, or why it could not be.
type Opened = std::result::Result<Arc<Device>, String>;

/// Open (once) CUDA device `ordinal`.
fn device(ordinal: usize) -> std::result::Result<Arc<Device>, String> {
    let mut devices = DEVICES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, opened)) = devices.iter().find(|(o, _)| *o == ordinal) {
        return opened.clone();
    }
    let opened = Device::open(ordinal).map(Arc::new);
    devices.push((ordinal, opened.clone()));
    opened
}

/// How many nodes (and rows) a [`CudaHistBackend`] built with each
/// strategy of the [module docs](self), for benchmarks and diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct NodeCounts {
    /// Nodes summed as exact integers in one piece (strategy 1).
    pub exact_nodes: u64,
    /// Nodes summed as exact integer chunks reduced in `f64` (strategy 2).
    pub exact_chunk_nodes: u64,
    /// Nodes summed as `f64` chains on the GPU (strategy 3).
    pub chain_nodes: u64,
    /// Nodes built by the CPU backend (strategy 4, input mismatches, and
    /// every node after a CUDA error).
    pub cpu_nodes: u64,
    /// Rows of the nodes counted in `exact_nodes`.
    pub exact_rows: u64,
    /// Rows of the nodes counted in `exact_chunk_nodes`.
    pub exact_chunk_rows: u64,
    /// Rows of the nodes counted in `chain_nodes`.
    pub chain_rows: u64,
    /// Rows of the nodes counted in `cpu_nodes`.
    pub cpu_rows: u64,
}

/// Atomic [`NodeCounts`].
#[derive(Default)]
struct Counters([AtomicU64; 8]);

impl Counters {
    fn count(&self, strategy: Strategy, rows: usize) {
        let k = strategy as usize;
        self.0[k].fetch_add(1, Ordering::Relaxed);
        self.0[k + 4].fetch_add(rows as u64, Ordering::Relaxed);
    }

    fn snapshot(&self) -> NodeCounts {
        let v = |i: usize| self.0[i].load(Ordering::Relaxed);
        NodeCounts {
            exact_nodes: v(0),
            exact_chunk_nodes: v(1),
            chain_nodes: v(2),
            cpu_nodes: v(3),
            exact_rows: v(4),
            exact_chunk_rows: v(5),
            chain_rows: v(6),
            cpu_rows: v(7),
        }
    }
}

/// A node's strategy (the [module docs](self)' numbering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    Exact = 0,
    ExactChunks = 1,
    Chains = 2,
    Cpu = 3,
}

/// The device copy of the index's row-major bin store.
enum DeviceBins {
    U16(CudaSlice<u16>),
    U32(CudaSlice<u32>),
}

/// The tree's gradient slice as staged on the device.
struct Staged {
    /// Address and length of the host slice staged (`len == 0`: none).
    addr: usize,
    len: usize,
    grad: SumDomain,
    hess: SumDomain,
}

impl Staged {
    fn holds(&self, gpair: &[GradPair]) -> bool {
        self.len != 0 && self.len == gpair.len() && self.addr == gpair.as_ptr().addr()
    }

    fn sums_exact(&self, n: usize) -> bool {
        self.grad.sums_exact(n) && self.hess.sums_exact(n)
    }
}

/// Device buffers, used by one build at a time.
struct State {
    bins: DeviceBins,
    /// CSR row offsets (`n_rows + 1`); `None` for a dense index.
    row_ptr: Option<CudaSlice<u64>>,
    /// Each feature's first global bin, then the total (`n_cols + 1`).
    feature_bins: CudaSlice<u64>,
    /// The staged `GradPair`s, two `f32`s per row.
    gpair: CudaSlice<f32>,
    /// The staged pairs in grains, two `i64`s per row.
    units: CudaSlice<i64>,
    /// The node's row list.
    rows: CudaSlice<u32>,
    /// Per-chunk partials: two 64-bit words per bin per chunk, read as
    /// integers or as `f64`s.
    partials: CudaSlice<u64>,
    /// Chunks `partials` holds.
    wave_chunks: usize,
    /// The node's histogram, two `f64`s per bin.
    out: CudaSlice<f64>,
    staged: Staged,
}

/// The CUDA histogram backend: implements [`HistogramBackend`] on an NVIDIA
/// GPU. Constructed once per training run (the index upload is
/// per-dataset); the gradient slice is uploaded by
/// [`HistogramBackend::prepare`] once per tree.
///
/// Training selects it automatically through
/// [`device = cuda`](crate::config::TrainingParams::device); constructing it
/// directly serves custom training loops against a [`GHistIndex`]. Its
/// histograms equal the CPU backend's bit for bit (see the
/// [module docs](self)). `build` must receive the index the backend was
/// built from, and the gradient slice must not change between `prepare`
/// and the tree's last `build`; inputs that do not fit the backend's
/// buffers (another index shape, a gradient slice of another length, row
/// indices past the index) never reach the GPU and take the CPU path,
/// which checks them.
pub struct CudaHistBackend {
    device: Arc<Device>,
    n_rows: usize,
    n_cols: usize,
    total_bins: usize,
    /// Whether the index stores `u32` bins (else `u16`).
    wide: bool,
    /// Whether the index is dense (rows at `r * n_cols`), else CSR.
    dense: bool,
    state: Mutex<State>,
    counters: Counters,
}

impl std::fmt::Debug for CudaHistBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaHistBackend")
            .field("device", &self.device.name)
            .field("n_rows", &self.n_rows)
            .field("n_cols", &self.n_cols)
            .field("total_bins", &self.total_bins)
            .finish_non_exhaustive()
    }
}

impl CudaHistBackend {
    /// Build the backend for `index` on CUDA device `ordinal`: upload its
    /// bin store and allocate the per-tree and per-node buffers.
    pub fn new(index: &GHistIndex, ordinal: usize) -> Result<Self> {
        let device = device(ordinal).map_err(HessboostError::gpu)?;
        let n_rows = index.n_rows();
        let n_cols = index.n_cols();
        let total_bins = index.total_bins();
        if total_bins == 0 || n_rows == 0 {
            return Err(HessboostError::invalid_data(
                "data",
                "the CUDA backend needs a non-empty binned dataset",
            ));
        }
        if u32::try_from(n_rows).is_err() {
            return Err(HessboostError::invalid_data(
                "data",
                format!("the CUDA backend indexes rows in 32 bits ({n_rows} rows)"),
            ));
        }
        let state = Self::upload(&device, index).map_err(gpu_error)?;
        Ok(CudaHistBackend {
            n_rows,
            n_cols,
            total_bins,
            wide: matches!(index.bins(), Bins::U32(_)),
            dense: index.dense_stride().is_some(),
            device,
            state: Mutex::new(state),
            counters: Counters::default(),
        })
    }

    /// The name of the device the backend runs on.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device.name
    }

    /// The nodes (and rows) built so far with each strategy.
    #[must_use]
    pub fn node_counts(&self) -> NodeCounts {
        self.counters.snapshot()
    }

    fn upload(device: &Device, index: &GHistIndex) -> std::result::Result<State, DriverError> {
        let stream = &device.stream;
        let n_rows = index.n_rows();
        let total_bins = index.total_bins();
        let bins = match index.bins() {
            Bins::U16(bins) => DeviceBins::U16(stream.clone_htod(bins)?),
            Bins::U32(bins) => DeviceBins::U32(stream.clone_htod(bins)?),
        };
        let row_ptr = if index.dense_stride().is_some() {
            None
        } else {
            let offsets: Vec<u64> = index.row_ptr().iter().map(|&o| o as u64).collect();
            Some(stream.clone_htod(&offsets)?)
        };
        let cuts = index.cuts();
        let mut offsets: Vec<u64> = (0..index.n_cols())
            .map(|f| cuts.feature_bins(f).0 as u64)
            .collect();
        offsets.push(total_bins as u64);
        let feature_bins = stream.clone_htod(&offsets)?;
        // Chunks of at least `ROWS_PER_TASK` rows, so `n_rows / 4096` bounds
        // a node's chunk count.
        let max_chunks = (n_rows / 4096).max(1);
        let wave_chunks = (PARTIAL_BYTES / (total_bins * 16)).clamp(1, max_chunks);
        Ok(State {
            bins,
            row_ptr,
            feature_bins,
            gpair: stream.alloc_zeros(n_rows * 2)?,
            units: stream.alloc_zeros(n_rows * 2)?,
            rows: stream.alloc_zeros(n_rows)?,
            partials: stream.alloc_zeros(wave_chunks * total_bins * 2)?,
            wave_chunks,
            out: stream.alloc_zeros(total_bins * 2)?,
            staged: Staged {
                addr: 0,
                len: 0,
                grad: SumDomain::EMPTY,
                hess: SumDomain::EMPTY,
            },
        })
    }

    /// Stage `gpair` on the device with its exactness statistics. A slice
    /// of any length other than `n_rows` is not staged: the builds it
    /// serves run on the CPU.
    fn stage(&self, state: &mut State, gpair: &[GradPair]) -> std::result::Result<(), DriverError> {
        // Unstage first, so a slice that fails to upload is never mistaken
        // for the previous one.
        state.staged.len = 0;
        if gpair.len() != self.n_rows {
            return Ok(());
        }
        let grad = SumDomain::of_slice(gpair, |p| p.grad);
        let hess = SumDomain::of_slice(gpair, |p| p.hess);
        // SAFETY: `GradPair` is `repr(C)` of two `f32`s, so the slice is
        // `2 * len` contiguous `f32`s.
        let flat =
            unsafe { std::slice::from_raw_parts(gpair.as_ptr().cast::<f32>(), gpair.len() * 2) };
        let stream = &self.device.stream;
        stream.memcpy_htod(flat, &mut state.gpair)?;
        let n = gpair.len() as u64;
        let (to_grad, to_hess) = (grad.unit_scale(), hess.unit_scale());
        let mut launch = stream.launch_builder(&self.device.kernels.stage_units);
        launch
            .arg(&state.gpair)
            .arg(&mut state.units)
            .arg(&n)
            .arg(&to_grad)
            .arg(&to_hess);
        // SAFETY: the kernel reads `n` `float2`s of `gpair` and writes `n`
        // `longlong2`s of `units`, both sized `2 * n_rows` words, and takes
        // `(u64, f64, f64)` scalars as passed.
        unsafe { launch.launch(self.device.grid(gpair.len())) }?;
        state.staged = Staged {
            addr: gpair.as_ptr().addr(),
            len: gpair.len(),
            grad,
            hess,
        };
        Ok(())
    }

    /// Build the histogram of `rows` into `out` on the GPU when the inputs
    /// allow it, returning the strategy used (`Strategy::Cpu`: not built,
    /// `out` unspecified).
    fn try_gpu(
        &self,
        ghist: &GHistIndex,
        rows: &[u32],
        gpair: &[GradPair],
        out: &mut [GradStats],
    ) -> Strategy {
        // The kernels do not bounds-check: inputs the device buffers were
        // not sized for must never reach a launch.
        let fits = out.len() == self.total_bins
            && ghist.n_rows() == self.n_rows
            && ghist.n_cols() == self.n_cols
            && ghist.total_bins() == self.total_bins
            && matches!(ghist.bins(), Bins::U32(_)) == self.wide
            && ghist.dense_stride().is_some() == self.dense
            && rows.len() <= self.n_rows
            && rows.iter().all(|&r| (r as usize) < self.n_rows);
        if !fits || self.device.failed.load(Ordering::Acquire) {
            return Strategy::Cpu;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.build_on(&mut state, ghist, rows, gpair, out)
            .unwrap_or_else(|_| {
                // CUDA errors are sticky: every later node runs on the CPU.
                self.device.failed.store(true, Ordering::Release);
                Strategy::Cpu
            })
    }

    /// [`try_gpu`](Self::try_gpu) once the inputs are known to fit: stage
    /// `gpair` if needed, pick the node's strategy, and run it.
    fn build_on(
        &self,
        state: &mut State,
        ghist: &GHistIndex,
        rows: &[u32],
        gpair: &[GradPair],
        out: &mut [GradStats],
    ) -> std::result::Result<Strategy, DriverError> {
        if !state.staged.holds(gpair) {
            self.stage(state, gpair)?;
            if !state.staged.holds(gpair) {
                return Ok(Strategy::Cpu);
            }
        }
        let strategy = plan(&state.staged, sum_order(ghist, rows), rows.len());
        if strategy == Strategy::Cpu {
            return Ok(strategy);
        }
        if rows.is_empty() {
            out.fill(GradStats::default());
        } else {
            self.run(state, ghist, rows, strategy, out)?;
        }
        Ok(strategy)
    }

    /// Launch `strategy`'s kernels for `rows` and read the histogram back.
    fn run(
        &self,
        state: &mut State,
        ghist: &GHistIndex,
        rows: &[u32],
        strategy: Strategy,
        out: &mut [GradStats],
    ) -> std::result::Result<(), DriverError> {
        let device = &*self.device;
        let stream = &device.stream;
        let State {
            bins,
            row_ptr,
            feature_bins,
            gpair,
            units,
            rows: dev_rows,
            partials,
            wave_chunks,
            out: dev_out,
            staged,
        } = state;
        stream.memcpy_htod(rows, &mut dev_rows.slice_mut(..rows.len()))?;
        let seg_rows = match (strategy, sum_order(ghist, rows)) {
            (Strategy::ExactChunks | Strategy::Chains, SumOrder::Blocked { grain }) => grain,
            _ => rows.len(),
        };
        let chunks = rows.len().div_ceil(seg_rows);
        let wide = usize::from(self.wide);
        let total_bins = self.total_bins as u64;
        let n_cols = self.n_cols as u64;
        let seg = seg_rows as u64;
        for (w, first) in (0..chunks).step_by(*wave_chunks).enumerate() {
            let wave = (*wave_chunks).min(chunks - first);
            let begin = first * seg_rows;
            let end = (begin + wave * seg_rows).min(rows.len());
            let wave_rows = dev_rows.slice(begin..end);
            let n = (end - begin) as u64;
            let words = wave * self.total_bins * 2;
            stream.memset_zeros(&mut partials.slice_mut(..words))?;
            let init = i32::from(w == 0);
            if strategy == Strategy::Chains {
                let segs = wave as u64;
                let threads = wave * self.n_cols;
                let mut launch;
                match (row_ptr.as_ref(), &*bins) {
                    (None, DeviceBins::U16(b)) => {
                        launch = stream.launch_builder(&device.kernels.chain_dense[wide]);
                        launch.arg(b);
                    }
                    (None, DeviceBins::U32(b)) => {
                        launch = stream.launch_builder(&device.kernels.chain_dense[wide]);
                        launch.arg(b);
                    }
                    (Some(ptr), DeviceBins::U16(b)) => {
                        launch = stream.launch_builder(&device.kernels.chain_csr[wide]);
                        launch.arg(b).arg(ptr).arg(&*feature_bins);
                    }
                    (Some(ptr), DeviceBins::U32(b)) => {
                        launch = stream.launch_builder(&device.kernels.chain_csr[wide]);
                        launch.arg(b).arg(ptr).arg(&*feature_bins);
                    }
                }
                launch
                    .arg(&n_cols)
                    .arg(&wave_rows)
                    .arg(&n)
                    .arg(&seg)
                    .arg(&segs)
                    .arg(&*gpair)
                    .arg(&mut *partials)
                    .arg(&total_bins);
                // SAFETY: one thread per (chunk, feature) of the wave; every
                // listed row is below `n_rows` (checked by `try_gpu`), so
                // bin and pair reads are in bounds, and each chunk's partial
                // (`wave <= wave_chunks` of them) fits `partials`. Arguments
                // match the kernel's parameter list in order and type.
                unsafe { launch.launch(Device::one_per(threads)) }?;
                let mut reduce = stream.launch_builder(&device.kernels.reduce_chains);
                reduce
                    .arg(&*partials)
                    .arg(&segs)
                    .arg(&total_bins)
                    .arg(&init)
                    .arg(&mut *dev_out);
                // SAFETY: reads `wave` partials and writes `total_bins`
                // pairs of `out`, both within their buffers.
                unsafe { reduce.launch(device.grid(self.total_bins)) }?;
            } else {
                let mut launch;
                let work = match (row_ptr.as_ref(), &*bins) {
                    (None, DeviceBins::U16(b)) => {
                        launch = stream.launch_builder(&device.kernels.units_dense[wide]);
                        launch.arg(b).arg(&n_cols);
                        (end - begin) * self.n_cols
                    }
                    (None, DeviceBins::U32(b)) => {
                        launch = stream.launch_builder(&device.kernels.units_dense[wide]);
                        launch.arg(b).arg(&n_cols);
                        (end - begin) * self.n_cols
                    }
                    (Some(ptr), DeviceBins::U16(b)) => {
                        launch = stream.launch_builder(&device.kernels.units_csr[wide]);
                        launch.arg(b).arg(ptr);
                        end - begin
                    }
                    (Some(ptr), DeviceBins::U32(b)) => {
                        launch = stream.launch_builder(&device.kernels.units_csr[wide]);
                        launch.arg(b).arg(ptr);
                        end - begin
                    }
                };
                launch
                    .arg(&wave_rows)
                    .arg(&n)
                    .arg(&seg)
                    .arg(&*units)
                    .arg(&mut *partials)
                    .arg(&total_bins);
                // SAFETY: as for the chains; every stored bin is below
                // `total_bins` (the index invariant).
                unsafe { launch.launch(device.grid(work)) }?;
                let segs = wave as u64;
                let (from_grad, from_hess) = (staged.grad.value_scale(), staged.hess.value_scale());
                let mut reduce = stream.launch_builder(&device.kernels.reduce_units);
                reduce
                    .arg(&*partials)
                    .arg(&segs)
                    .arg(&total_bins)
                    .arg(&from_grad)
                    .arg(&from_hess)
                    .arg(&init)
                    .arg(&mut *dev_out);
                // SAFETY: as for `reduce_chains`.
                unsafe { reduce.launch(device.grid(self.total_bins)) }?;
            }
        }
        // SAFETY: `GradStats` is `repr(C)` of two `f64`s, so `out` is
        // `2 * total_bins` contiguous `f64`s.
        let flat = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len() * 2)
        };
        stream.memcpy_dtoh(&*dev_out, flat)?;
        // Kernel faults surface at a synchronization point.
        stream.synchronize()
    }
}

/// The strategy for a node of `n` rows summed in `order` (the
/// [module docs](self)' numbering).
fn plan(staged: &Staged, order: SumOrder, n: usize) -> Strategy {
    // NaN payloads are not portable between the CPU's and the GPU's
    // arithmetic, so a non-finite slice keeps the CPU's bits by running there.
    if !(staged.grad.is_finite() && staged.hess.is_finite()) {
        return Strategy::Cpu;
    }
    if staged.sums_exact(n) {
        return Strategy::Exact;
    }
    match order {
        SumOrder::Blocked { grain } if staged.sums_exact(grain) => Strategy::ExactChunks,
        SumOrder::Blocked { .. } => Strategy::Chains,
        SumOrder::Chain if n < PARALLEL_THRESHOLD => Strategy::Chains,
        SumOrder::Chain => Strategy::Cpu,
    }
}

fn gpu_error(error: DriverError) -> HessboostError {
    HessboostError::gpu(format!("CUDA: {error}"))
}

impl HistogramBackend for CudaHistBackend {
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        let strategy = self.try_gpu(ghist, rows, gpair, out);
        if strategy == Strategy::Cpu {
            CpuBackend.build(ghist, rows, gpair, out);
        }
        self.counters.count(strategy, rows.len());
    }

    fn prepare(&self, _ghist: &GHistIndex, gpair: &[GradPair]) {
        if self.device.failed.load(Ordering::Acquire) {
            return;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // The trainer refills its gradient buffer in place every round, so a
        // new tree always restages.
        if self.stage(&mut state, gpair).is_err() {
            state.staged.len = 0;
            self.device.failed.store(true, Ordering::Release);
        }
    }
}
