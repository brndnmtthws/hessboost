//! NVIDIA CUDA acceleration for Linux (opt-in `cuda` feature).
//!
//! **Training** (`device = cuda`, see
//! [`TrainingParams::device`](crate::config::TrainingParams::device)): the
//! trained model is bit-identical to CPU training (at any thread count),
//! and the tree's rows live on the GPU. Depthwise growth handles a whole
//! level per round trip: the GPU partitions every splitting node (a stable
//! partition, so each child keeps its rows in ascending order) and builds
//! the smaller child of every split. Numeric and categorical histograms then
//! stay on the GPU: it derives each sibling by subtraction, scans candidates
//! in CPU order and arithmetic, and returns one compact winner per node.
//! Unsupported NaN comparison/sort semantics replay the node on the host.
//! Loss-guided growth keeps its heap on the host while histograms and split
//! search stay resident. Symmetric (`grow_policy = symmetric`) trees build
//! their histograms here node by node, with the rows uploaded per node.
//!
//! For `reg:squarederror` and the logistic objectives on one label column
//! in a plain `gbtree` (no row sampling, DART, linear leaves, SGLB, model
//! shrinkage, or reuse penalties), whole rounds run on the GPU: it keeps
//! the training margins, computes the gradients from them with the host's
//! operations (the logistic ones are the host's vector kernel, so they need
//! AVX2/FMA or NEON; the few trailing rows the host runs scalar are
//! computed on the host), and adds each leaf's value to its rows. A round
//! the GPU cannot reproduce (non-finite gradients, logistic margins beyond
//! the vector kernel's ±80) runs on the host. Other configurations compute
//! the gradients on the host each round, upload them once per tree, and
//! read the leaf rows back once per tree to update the margins.
//!
//! # Kernels
//!
//! The kernels are Rust, compiled to PTX by
//! [cuda-oxide](https://nvidia.github.io/cuda-rust/cuda-oxide/) from the
//! repository's `cuda-kernels/` crate and embedded here; the driver
//! JIT-compiles the PTX for the device's own architecture the first time a
//! process loads it (and caches the machine code). Building this crate
//! needs neither cuda-oxide nor a CUDA toolkit.
//!
//! # Requirements
//!
//! - Linux with an NVIDIA GPU of compute capability 7.5 (Turing) or newer
//!   and a driver supporting CUDA 12.8 or later. The driver (`libcuda`) is
//!   opened at run time: its absence makes the backend unavailable rather
//!   than failing to load the crate. No CUDA toolkit is needed.
//!
//! # Exactness
//!
//! The CPU adds each histogram bin in `f64` in a fixed order
//! (`tree::hist::sum_order`): one chain in row order for a node below
//! 8,192 rows, for a contiguous row range of a dense index with a
//! column-major copy, and for any row subset of such an index of at most
//! 2^18 rows; otherwise fixed blocks of about 4,096 rows, each chained from
//! zero and then added in block order. Every node uses the first strategy
//! that applies:
//!
//! 1. **Exact integers.** When every sum of the node's rows is exact
//!    (`n * max <= 2^53` gradient grains for both components, the domain
//!    the Metal backend also uses; proof in the private `exact_sum` module),
//!    the GPU sums 64-bit grain counts in any order (shared-memory
//!    histograms per row tile and feature group, flushed with 64-bit
//!    atomics) and scales them back exactly, whatever the CPU's order.
//! 2. **Exact blocks.** For a blocked node whose *blocks* are exact, each
//!    block's integer sum is that block's `f64` chain, and the GPU then adds
//!    the block partials in block order in `f64`, the CPU's own operations.
//! 3. **Chains.** Otherwise a blocked node runs one GPU thread per (block,
//!    feature) on dense storage and one per block on CSR (visiting stored
//!    entries once), each bin following the CPU's row-order `f64` chain and
//!    block reduction; a chain node below 8,192 rows is one such chain per
//!    feature.
//!
//! A non-exact node the CPU sums as one chain of 8,192 or more rows is
//! built on the host, from the host's gradients or the device's (read back
//! once per tree), while the GPU builds the rest of its level: swept by
//! feature on scoped threads, as many as training's thread count (`nthread`)
//! allows and at most one per 8,192 rows. The root's statistics (`sum_rows`,
//! one chain in row order) are summed on the GPU in integers when exact,
//! else on the host.
//! The kernels are compiled without floating-point contraction
//! (cuda-oxide's `--no-fmad`), flush-to-zero, or approximate division, so
//! every `f64` operation is the single IEEE operation the CPU performs;
//! there are no floating-point atomics. Trees with a non-finite gradient
//! or Hessian (NaN payloads differ between CPU and GPU arithmetic) build
//! every node on the CPU. A CUDA error is sticky (the context is unusable
//! afterwards): the tree in progress is regrown on the host, and every
//! later tree too, so the result is unchanged.
//!
//! Split search scans each feature's bins in the CPU's order with the
//! CPU's `f64` operations. When a tree's gradients sum exactly over all of
//! its rows, every histogram and total of the tree is exact in grains, so
//! any association of the prefix additions gives the CPU's bits: those
//! trees form the prefixes with warp scans instead of one lane's chain.
//!
//! # Transfers
//!
//! Descriptors (tiles, rules, scan requests) are staged in a pinned arena
//! and copied without waiting, so the host waits only where it reads the
//! device's results: partition counts and split winners (once per level,
//! or per loss-guided expansion) and the root's totals. Large copies (bins,
//! gradients, row lists) move through pooled 1 MiB pinned pieces, and each
//! device keeps released page-locked blocks (up to 64 MiB) for the next
//! training run.
//!
//! CUDA prediction is explicit through [`BoostedModel::to_cuda`](crate::model::BoostedModel::to_cuda).
//! Tree traversal and ordered margin summation run on a dedicated tracked
//! stream; objective transforms use the model's CPU implementation.
//!
//! # Limitations
//!
//! - Refused with `device = cuda`: `tree_method = exact`/`approx`,
//!   `use_quantized_grad`, `gblinear`, `process_type = update`,
//!   `multi_strategy = multi_output_tree`, online updates, and budget mode.
//!   Trees with reuse penalties (`toad_penalty_*`) grow on the host, with
//!   per-node GPU histograms.
//! - Dense bins are reencoded and transposed on device from CPU-authoritative
//!   global bins (1, 2 or 4 bytes per feature-local cell in each layout).
//!   Sparse bins stay CSR: 2 or 4 bytes per present global bin plus an 8-byte
//!   row offset; no row-by-feature buffers are allocated. The gradients use
//!   24 bytes per row, three row buffers 12 bytes per row, and device-side
//!   margins, labels and weights another 12 bytes per row. Resident growth
//!   retains open-node histograms when they fit in half the free memory.

mod abi;
mod categorical;
pub(crate) mod diagnostics;
mod kernels;
mod predict;
pub use predict::{
    GpuModel, prediction_available, prediction_device_name, prediction_unavailable_reason,
};

use crate::backend::exact_sum::SumDomain;
use crate::data::ghist::{Bins, GHistIndex};
use crate::error::{HessboostError, Result};
use crate::objective::GradPair;
use crate::tree::gain::{GradStats, RegParams};
use crate::tree::hist::{
    CpuBackend, DeviceLoss, HistSlot, Histogram, HistogramBackend, NodeScan, Partitioned,
    RowEngine, RowRule, RowSplit, ScanRequest, Segment, SumOrder, sum_order, zeroed,
};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DeviceRepr, DriverError, LaunchArgs, LaunchConfig, PushKernelArg, ValidAsZeroBits, sys,
};
use diagnostics::NodeCounts;
use parking_lot::{Mutex, MutexGuard};
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Threads per block of the elementwise kernels.
const THREADS: u32 = 256;
/// Grid-stride kernels launch at most this many blocks per SM.
const BLOCKS_PER_SM: u32 = 8;
/// Threads per histogram block (the histogram entries' `launch_bounds` in
/// `cuda-kernels/src/train.rs`). A starting point for occupancy tuning;
/// actual residency also depends on registers and shared memory and must be
/// measured on the target GPU.
const HIST_THREADS: u32 = 512;
/// Rows per partition tile (`PART_TILE` in `cuda-kernels/src/train.rs`).
const PART_TILE: usize = 4096;
/// Threads per partition block.
const PART_THREADS: u32 = 512;
/// Warps per split-scan block (`SCAN_WARPS` in `cuda-kernels/src/lib.rs`).
const SCAN_WARPS: usize = 1;
/// Most rows per histogram tile of an exact node.
const HIST_TILE: usize = 4096;
/// Fewest rows per exact tile: a small batch's tiles shrink to this so its
/// blocks still cover the device.
const MIN_HIST_TILE: usize = 512;
/// Largest shared histogram per block; XGBoost's single-target limit.
const MAX_SHARED_BYTES: usize = 96 << 10;
/// Shared memory the driver reserves per resident block.
const RESERVED_SHARED_BYTES: usize = 1 << 10;
/// Bytes of per-chunk partial histograms held at once. A level with more
/// chunks is built in waves of chunks, each reduced into the output in
/// chunk order before the next, as the CPU's waves are.
const PARTIAL_BYTES: usize = 512 << 20;
/// The oldest driver the backend runs on (`cuDriverGetVersion` encoding):
/// the API version the bindings are built against, CUDA 12.8.
const MIN_DRIVER: i32 = 12_080;
/// Most nodes one reduction launch covers (a grid's `y` limit).
const MAX_GRID_Y: usize = 65_535;

/// Why a CUDA device cannot run the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Unavailable {
    /// No NVIDIA driver (`libcuda`) is installed.
    NoDriver,
    /// The driver supports CUDA `major.minor`, older than the 12.8 the
    /// backend needs.
    OldDriver {
        /// The CUDA major version the driver supports.
        major: i32,
        /// The CUDA minor version the driver supports.
        minor: i32,
    },
    /// There is no device `ordinal`.
    NoDevice {
        /// The requested device.
        ordinal: usize,
        /// How many devices the driver reports.
        count: usize,
    },
    /// Device `ordinal` has compute capability `major.minor`, older than
    /// the kernels' 7.5 (Turing).
    OldDevice {
        /// The requested device.
        ordinal: usize,
        /// The device's compute capability major version.
        major: i32,
        /// The device's compute capability minor version.
        minor: i32,
    },
    /// Device `ordinal` had a runtime error earlier in the process; it runs
    /// nothing more.
    Disabled {
        /// The device.
        ordinal: usize,
    },
    /// Initializing the device failed (a driver error, such as a kernel
    /// module that does not load).
    Failed(String),
}

impl Unavailable {
    /// Whether the machine lacks what the backend needs (no driver, an old
    /// driver, no such device, an old device), as opposed to the backend
    /// failing on a machine that has it.
    #[must_use]
    pub fn is_environment(&self) -> bool {
        matches!(
            self,
            Self::NoDriver
                | Self::OldDriver { .. }
                | Self::NoDevice { .. }
                | Self::OldDevice { .. }
        )
    }
}

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDriver => f.write_str("libcuda not found (no NVIDIA driver is installed)"),
            Self::OldDriver { major, minor } => write!(
                f,
                "the NVIDIA driver supports CUDA {major}.{minor}; the backend needs {}.{} or later",
                MIN_DRIVER / 1000,
                MIN_DRIVER % 1000 / 10
            ),
            Self::NoDevice { ordinal, count } => {
                write!(f, "no CUDA device {ordinal} ({count} found)")
            }
            Self::OldDevice {
                ordinal,
                major,
                minor,
            } => {
                let (min_major, min_minor) = kernels::MIN_COMPUTE_CAPABILITY;
                write!(
                    f,
                    "CUDA device {ordinal} has compute capability {major}.{minor}; the backend \
                     needs {min_major}.{min_minor} or newer"
                )
            }
            Self::Disabled { ordinal } => {
                write!(f, "CUDA device {ordinal} disabled after a runtime error")
            }
            Self::Failed(reason) => f.write_str(reason),
        }
    }
}

/// Whether CUDA device `ordinal` can train, and has not had a runtime
/// error.
#[must_use]
pub fn available(ordinal: usize) -> bool {
    unavailable_reason(ordinal).is_none()
}

/// Why CUDA device `ordinal` cannot train, if it cannot.
#[must_use]
pub fn unavailable_reason(ordinal: usize) -> Option<Unavailable> {
    match device(ordinal) {
        Ok(opened) if opened.failed.load(Ordering::Acquire) => {
            Some(Unavailable::Disabled { ordinal })
        }
        Ok(_) => None,
        Err(reason) => Some(reason),
    }
}

/// The name of CUDA device `ordinal`, if it can train.
#[must_use]
pub fn device_name(ordinal: usize) -> Option<String> {
    device(ordinal)
        .ok()
        .filter(|device| !device.failed.load(Ordering::Acquire))
        .map(|device| device.name.clone())
}

/// The kernels of one device's module; per-width kernels are indexed by
/// [`DeviceBins::width`] (`u8`, `u16`, `u32`).
#[derive(Clone)]
struct Kernels {
    stage_units: CudaFunction,
    iota_rows: CudaFunction,
    chunk_totals: CudaFunction,
    hist_shared: [CudaFunction; 3],
    hist_global: [CudaFunction; 3],
    hist_chain: [CudaFunction; 3],
    hist_sparse: [CudaFunction; 3],
    hist_sparse_chain: [CudaFunction; 3],
    encode_u8: [CudaFunction; 3],
    encode_u16: [CudaFunction; 3],
    encode_u32: [CudaFunction; 3],
    route_sparse: [CudaFunction; 3],
    route_count: [CudaFunction; 3],
    route_scatter: CudaFunction,
    route_scan: CudaFunction,
    route_copy: CudaFunction,
    route_runs: CudaFunction,
    finalize_exact: CudaFunction,
    finalize_exact_sub: CudaFunction,
    reduce_chunks: CudaFunction,
    reduce_chains: CudaFunction,
    squared_error: CudaFunction,
    logistic: CudaFunction,
    grad_domain: CudaFunction,
    add_leaves: CudaFunction,
    subtract_hists: CudaFunction,
    scan_splits: CudaFunction,
    category_keys: CudaFunction,
    category_merge: CudaFunction,
    scan_categorical: CudaFunction,
    merge_scans: CudaFunction,
}

/// Cached CUDA resources plus a stream. Each training backend forks its
/// own stream before allocating buffers; kernels and failure state are shared.
struct Device {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    name: String,
    sm_count: u32,
    /// Dynamic shared memory one histogram block may use.
    shared_bytes: usize,
    /// Set by the first CUDA error: the context is unusable afterwards, so
    /// every later build on this device runs on the CPU.
    failed: Arc<AtomicBool>,
    /// Page-locked blocks the context's backends reuse.
    pinned: Arc<PinnedPool>,
}

impl Device {
    fn open(ordinal: usize) -> std::result::Result<Self, Unavailable> {
        find_device(ordinal)?;
        let ctx = CudaContext::new(ordinal).map_err(|e| failed("context", e))?;
        // SAFETY: each backend allocates, uses and frees every slice on its
        // own stream. Only immutable kernel/module handles cross backend
        // boundaries.
        unsafe { ctx.disable_event_tracking() };
        let module = kernels::load(&ctx, kernels::Module::Training)?;
        let function = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| failed(&format!("kernel `{name}`"), e))
        };
        let widths = |prefix: &str| -> std::result::Result<[CudaFunction; 3], Unavailable> {
            Ok([
                function(&format!("{prefix}_u8"))?,
                function(&format!("{prefix}_u16"))?,
                function(&format!("{prefix}_u32"))?,
            ])
        };
        let kernels = Kernels {
            stage_units: function("stage_units")?,
            iota_rows: function("iota_rows")?,
            chunk_totals: function("chunk_totals")?,
            hist_shared: widths("hist_shared")?,
            hist_global: widths("hist_global")?,
            hist_chain: widths("hist_chain")?,
            hist_sparse: widths("hist_sparse")?,
            hist_sparse_chain: widths("hist_sparse_chain")?,
            encode_u8: widths("encode_u8")?,
            encode_u16: widths("encode_u16")?,
            encode_u32: widths("encode_u32")?,
            route_sparse: widths("route_sparse")?,
            route_count: widths("route_count")?,
            route_scatter: function("route_scatter")?,
            route_scan: function("route_scan")?,
            route_copy: function("route_copy")?,
            route_runs: function("route_runs")?,
            finalize_exact: function("finalize_exact")?,
            finalize_exact_sub: function("finalize_exact_sub")?,
            reduce_chunks: function("reduce_chunks")?,
            reduce_chains: function("reduce_chains")?,
            squared_error: function("squared_error")?,
            logistic: function("logistic")?,
            grad_domain: function("grad_domain")?,
            add_leaves: function("add_leaves")?,
            subtract_hists: function("subtract_hists")?,
            scan_splits: function("scan_splits")?,
            category_keys: function("category_keys")?,
            category_merge: function("category_merge")?,
            scan_categorical: function("scan_categorical")?,
            merge_scans: function("merge_scans")?,
        };
        let attribute =
            |attribute, what: &str| ctx.attribute(attribute).map_err(|e| failed(what, e));
        let sm_count = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            "SM count",
        )?;
        let optin = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            "shared memory limit",
        )?;
        let sm_shared = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            "SM shared memory",
        )?;
        let sm_threads = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            "SM thread limit",
        )?;
        // As many histogram blocks as the SM's threads allow share its
        // shared memory (Ada: three of 32 KiB).
        let resident = usize::try_from(sm_threads).unwrap_or(0) / HIST_THREADS as usize;
        let per_block = usize::try_from(sm_shared).unwrap_or(0) / resident.max(1);
        let shared_bytes = per_block
            .saturating_sub(RESERVED_SHARED_BYTES)
            .min(usize::try_from(optin).unwrap_or(0))
            .clamp(16 << 10, MAX_SHARED_BYTES);
        for kernel in &kernels.hist_shared {
            // SAFETY: a valid function of the loaded module; the attribute
            // only raises the dynamic shared-memory cap to the device's
            // opt-in limit (or less).
            unsafe {
                sys::cuFuncSetAttribute(
                    kernel.cu_function(),
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_bytes as i32,
                )
            }
            .result()
            .map_err(|e| failed("shared memory attribute", e))?;
        }
        let stream = ctx.new_stream().map_err(|e| failed("stream", e))?;
        let name = ctx.name().map_err(|e| failed("device name", e))?;
        Ok(Device {
            stream,
            kernels,
            name,
            sm_count: u32::try_from(sm_count).unwrap_or(1).max(1),
            shared_bytes,
            failed: Arc::new(AtomicBool::new(false)),
            pinned: Arc::default(),
        })
    }

    fn for_backend(&self) -> std::result::Result<Arc<Self>, DriverError> {
        let stream = self.stream.context().new_stream()?;
        Ok(Arc::new(Self {
            stream,
            kernels: self.kernels.clone(),
            name: self.name.clone(),
            sm_count: self.sm_count,
            shared_bytes: self.shared_bytes,
            failed: self.failed.clone(),
            pinned: self.pinned.clone(),
        }))
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

    /// A grid-stride launch over `total_bins` bins for each of `nodes`
    /// nodes (the grid's `y`).
    fn per_node_bins(total_bins: usize, nodes: usize) -> LaunchConfig {
        let blocks = total_bins.div_ceil(THREADS as usize).clamp(1, 64);
        LaunchConfig {
            grid_dim: (blocks as u32, nodes as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Check that the driver loads, is new enough, and has device `ordinal`.
/// Checked before any other `cudarc` call: its lazy loaders panic when the
/// library is missing, and the release profile aborts on panic.
fn find_device(ordinal: usize) -> std::result::Result<(), Unavailable> {
    // SAFETY: only tries to open the shared library by name.
    if !unsafe { sys::is_culib_present() } {
        return Err(Unavailable::NoDriver);
    }
    let mut version = 0;
    // SAFETY: the driver library loads (checked above), and the call only
    // writes the version through the pointer.
    unsafe { sys::cuDriverGetVersion(&raw mut version) }
        .result()
        .map_err(|e| failed("driver version", e))?;
    if version < MIN_DRIVER {
        return Err(Unavailable::OldDriver {
            major: version / 1000,
            minor: version % 1000 / 10,
        });
    }
    let count = match CudaContext::device_count() {
        Ok(count) => usize::try_from(count).unwrap_or(0),
        Err(e) if e.0 == sys::CUresult::CUDA_ERROR_NO_DEVICE => 0,
        Err(e) => return Err(failed("init", e)),
    };
    if ordinal >= count {
        return Err(Unavailable::NoDevice { ordinal, count });
    }
    Ok(())
}

/// A driver error while opening a device.
fn failed(what: &str, error: impl std::fmt::Display) -> Unavailable {
    Unavailable::Failed(format!("CUDA {what}: {error}"))
}

/// The opened devices, by ordinal (each opened once per process; a failure
/// is remembered too).
static DEVICES: Mutex<Vec<(usize, Opened)>> = Mutex::new(Vec::new());

/// A device opened once, or why it could not be.
type Opened = std::result::Result<Arc<Device>, Unavailable>;

/// Open (once) CUDA device `ordinal`.
fn device(ordinal: usize) -> Opened {
    let mut devices = DEVICES.lock();
    if let Some((_, opened)) = devices.iter().find(|(o, _)| *o == ordinal) {
        return opened.clone();
    }
    let opened = Device::open(ordinal).map(Arc::new);
    devices.push((ordinal, opened.clone()));
    opened
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

/// A node's strategy: the [module docs](self)' numbering, plus the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    Exact = 0,
    ExactChunks = 1,
    Chains = 2,
    Cpu = 3,
}

/// Device bins: feature-local dense bins, or global CSR bins.
enum DeviceBins {
    U8(CudaSlice<u8>),
    U16(CudaSlice<u16>),
    U32(CudaSlice<u32>),
}

impl DeviceBins {
    /// The kernel index of this width.
    fn width(&self) -> usize {
        match self {
            DeviceBins::U8(_) => 0,
            DeviceBins::U16(_) => 1,
            DeviceBins::U32(_) => 2,
        }
    }

    fn push<'a>(&'a self, launch: &mut LaunchArgs<'a>) {
        match self {
            DeviceBins::U8(b) => launch.arg(b),
            DeviceBins::U16(b) => launch.arg(b),
            DeviceBins::U32(b) => launch.arg(b),
        };
    }
}

/// The tree's gradient slice as staged on the device.
struct Staged {
    /// Address and length of the host slice staged (`len == 0`: none;
    /// `addr == 0`: gradients computed on the device, with no host copy).
    addr: usize,
    len: usize,
    grad: SumDomain,
    hess: SumDomain,
    /// The device-computed gradients (`addr == 0`), downloaded the first
    /// time a node summed on the CPU needs them; `None` until then.
    host: Option<Vec<GradPair>>,
}

impl Staged {
    /// Whether the staged gradients are `gpair` (`None`: the device's own).
    fn holds(&self, gpair: Option<&[GradPair]>) -> bool {
        match gpair {
            None => self.len != 0 && self.addr == 0,
            Some(gpair) => {
                self.len != 0 && self.len == gpair.len() && self.addr == gpair.as_ptr().addr()
            }
        }
    }

    fn finite(&self) -> bool {
        self.grad.is_finite() && self.hess.is_finite()
    }

    fn sums_exact(&self, n: usize) -> bool {
        self.grad.sums_exact(n) && self.hess.sums_exact(n)
    }
}

/// Which device row list a histogram batch reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowSource<'a> {
    /// The tree's partitioned rows (device-resident growth).
    Tree,
    /// Rows uploaded for one [`HistogramBackend::build`] call, with their
    /// host copy (CPU-built nodes read it).
    Upload(&'a [u32]),
}

/// A resident build's destinations: node `k`'s histogram goes to pool slot
/// `targets[k]`, and each `(parent, built)` sibling becomes `parent - built`
/// in the parent's slot.
#[derive(Clone, Copy)]
struct Resident<'a> {
    targets: &'a [HistSlot],
    siblings: &'a [(HistSlot, HistSlot)],
}

/// Whether one resident build's kernels never touch a pool slot from two
/// places at once: every write (a node's target, a sibling's parent) goes
/// to a distinct slot, no sibling reads a slot another one writes, and no
/// built slot is subtracted from two parents (a child has one sibling).
fn race_free(nodes: &[(Segment, HistSlot)], siblings: &[(HistSlot, HistSlot)]) -> bool {
    let distinct = |slots: &[HistSlot]| slots.windows(2).all(|pair| pair[0] != pair[1]);
    let mut parents: Vec<HistSlot> = siblings.iter().map(|&(parent, _)| parent).collect();
    parents.sort_unstable();
    let mut built: Vec<HistSlot> = siblings.iter().map(|&(_, built)| built).collect();
    built.sort_unstable();
    let mut written: Vec<HistSlot> = nodes.iter().map(|&(_, slot)| slot).collect();
    written.extend_from_slice(&parents);
    written.sort_unstable();
    distinct(&written)
        && distinct(&built)
        && built
            .iter()
            .all(|slot| parents.binary_search(slot).is_err())
}

/// Whether no two segments share a row position: no two splits partition
/// one position, and (the tree's rows being distinct, [`check_rows`]) no
/// two leaves add to one margin.
fn disjoint(segs: impl Iterator<Item = Segment>) -> bool {
    let mut spans: Vec<(usize, usize)> = segs
        .filter(|s| s.len > 0)
        .map(|s| (s.offset, s.offset + s.len))
        .collect();
    spans.sort_unstable();
    spans.windows(2).all(|pair| pair[0].1 <= pair[1].0)
}

/// Rows per task of [`check_rows`].
const CHECK_CHUNK: usize = 1 << 16;

/// What [`check_rows`] finds in a row list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowCheck {
    /// Every row is inside the index (the kernels do not bounds-check).
    inside: bool,
    /// The rows strictly ascend, so none repeats.
    ascending: bool,
    /// The rows are one run of consecutive rows (then generated on the
    /// device instead of uploaded).
    run: bool,
}

impl RowCheck {
    const ALL: Self = Self {
        inside: true,
        ascending: true,
        run: true,
    };

    fn and(self, other: Self) -> Self {
        Self {
            inside: self.inside && other.inside,
            ascending: self.ascending && other.ascending,
            run: self.run && other.run,
        }
    }
}

/// One parallel pass over `rows`, for an index of `n_rows` rows (run before
/// taking the state lock).
fn check_rows(rows: &[u32], n_rows: usize) -> RowCheck {
    let first = rows.first().map_or(0, |&r| r as usize);
    rows.par_chunks(CHECK_CHUNK)
        .enumerate()
        .map(|(c, chunk)| {
            let base = first + c * CHECK_CHUNK;
            // A chunk also compares its first row with the previous one's last.
            let joined = c == 0 || rows[c * CHECK_CHUNK - 1] < chunk[0];
            RowCheck {
                inside: chunk.iter().all(|&r| (r as usize) < n_rows),
                ascending: joined && chunk.windows(2).all(|pair| pair[0] < pair[1]),
                run: chunk
                    .iter()
                    .enumerate()
                    .all(|(i, &r)| r as usize == base + i),
            }
        })
        .reduce(|| RowCheck::ALL, RowCheck::and)
}

/// Host partition descriptors retain capacity across levels and trees.
#[derive(Default)]
struct PartitionHost {
    segs: Vec<u64>,
    rules: Vec<u32>,
    table: Vec<u8>,
    tiles: Vec<u64>,
    split_tiles: Vec<u32>,
}

/// Device buffers, used by one call at a time.
struct State {
    bins: DeviceBins,
    /// Dense feature-major bins for routing; absent for CSR storage.
    cols: Option<DeviceBins>,
    /// CSR row offsets; absent for dense storage. Sparse bins stay global.
    row_ptr: Option<CudaSlice<u64>>,
    partition_host: PartitionHost,
    pin_counts: Option<Pinned<u32>>,
    /// Entries per dense row (zero for CSR).
    stride: u32,
    /// The partition's per-row directions (1 = left), by row position.
    flags: CudaSlice<u8>,
    /// The missing-value marker of `bins` (`u32::MAX` for a dense index,
    /// which no stored bin equals).
    sentinel: u32,
    /// Each feature's first global bin, then the total (`n_cols + 1`).
    feature_first: CudaSlice<u32>,
    /// Feature groups whose bins fit a block's shared memory (four `u32`s
    /// each), and the wider ones (built with global atomics).
    groups_shared: CudaSlice<u32>,
    n_shared: usize,
    groups_global: CudaSlice<u32>,
    n_global: usize,
    /// Bins of the widest shared group.
    group_bins: usize,
    /// Features of the widest group, which bounds a tile's rows.
    group_features: usize,
    /// The staged `GradPair`s, two `f32`s per row.
    gpair: CudaSlice<f32>,
    /// The staged pairs in grains, two `i64`s per row.
    units: CudaSlice<i64>,
    /// The tree's rows, partitioned in place, and the partition's scratch.
    tree_rows: CudaSlice<u32>,
    scratch: CudaSlice<u32>,
    /// Rows of the tree (`tree_rows[..tree_len]` is valid).
    tree_len: usize,
    /// Rows uploaded by a per-node build.
    upload: CudaSlice<u32>,
    /// Exact accumulators, two 64-bit words per bin per node.
    acc: CudaSlice<u64>,
    /// Per-chunk partials (integer or `f64`), two words per bin per chunk.
    partials: CudaSlice<u64>,
    /// Chunks `partials` holds.
    wave_slots: usize,
    /// The batch's histograms, two `f64`s per bin per node.
    out: CudaSlice<f64>,
    /// One operation's descriptors (tiles, rules, scan requests, ...),
    /// uploaded together by [`Staging::upload_parts`].
    desc: CudaSlice<u8>,
    /// The root's per-chunk totals.
    totals: CudaSlice<i64>,
    tile_left: CudaSlice<u32>,
    /// A partition's left child lengths, then each split's child run bits
    /// (`route_runs`): two words per split.
    left_len: CudaSlice<u32>,
    /// The gradient statistics the device folds.
    domain: CudaSlice<u32>,
    /// Device-side rounds: the training margins, labels and weights (the
    /// latter two keyed by the host slices they were copied from).
    margins: Option<CudaSlice<f32>>,
    labels: Option<(usize, CudaSlice<f32>)>,
    weights: Option<(usize, CudaSlice<f32>)>,
    staged: Staged,
    /// Page-locked staging: small uploads queued without waiting, and the
    /// bounded pieces of large copies both ways.
    staging: Staging,
    ring: Ring,
    /// Resident growth: the histogram slots (two `f64`s per bin per slot)
    /// and how many it holds.
    pool: CudaSlice<f64>,
    pool_slots: usize,
    /// Reused resident split descriptors, sort workspaces and pinned results.
    scan: categorical::ScanState,
}

/// Plain numeric values: every bit pattern is a valid value, so page-locked
/// bytes reused from an earlier owner are initialized values of any of them.
trait Plain: Copy + DeviceRepr + ValidAsZeroBits {}
impl Plain for u8 {}
impl Plain for u16 {}
impl Plain for u32 {}
impl Plain for u64 {}
impl Plain for i64 {}
impl Plain for f32 {}
impl Plain for f64 {}

/// Most bytes of released page-locked blocks a device keeps for reuse.
const POOL_BYTES: usize = 64 << 20;
/// Readback buffers start at this size, so their growth through a tree's
/// first levels (partition counts, scan winners) stays rare.
const MIN_PINNED_BYTES: usize = 64 << 10;
/// Bytes the descriptor arena of a backend starts with.
const STAGING_BYTES: usize = 256 << 10;
/// Bytes per piece of a large transfer, and the pieces in flight: small
/// enough that the host's copy of one piece overlaps the DMA of the last.
const PIECE_BYTES: usize = 1 << 20;
const PIECES: usize = 4;

/// Page-locked blocks of one context, released by finished owners and
/// reused by later ones: every training run builds a backend, and
/// `cuMemHostAlloc` / `cuMemFreeHost` take up to milliseconds a call.
#[derive(Default)]
struct PinnedPool(Mutex<Vec<Block>>);

/// One page-locked allocation.
struct Block {
    ptr: std::ptr::NonNull<u8>,
    bytes: usize,
    write_combined: bool,
}

// SAFETY: plain host memory with a single owner (a `Pinned` or the pool).
unsafe impl Send for Block {}

impl PinnedPool {
    /// The smallest kept block of at least `bytes` in mode `write_combined`.
    fn take(&self, bytes: usize, write_combined: bool) -> Option<Block> {
        let mut blocks = self.0.lock();
        let at = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.write_combined == write_combined && b.bytes >= bytes)
            .min_by_key(|(_, b)| b.bytes)
            .map(|(at, _)| at)?;
        Some(blocks.swap_remove(at))
    }

    /// Keep `block`, which no copy accesses any more, unless the pool would
    /// exceed [`POOL_BYTES`]; then free it.
    fn give(&self, block: Block) {
        let mut blocks = self.0.lock();
        let held: usize = blocks.iter().map(|b| b.bytes).sum();
        if held + block.bytes <= POOL_BYTES {
            blocks.push(block);
            return;
        }
        drop(blocks);
        // SAFETY: allocated by `malloc_host`; its last owner released it
        // after all DMA completed.
        let _ = unsafe { cudarc::driver::result::free_host(block.ptr.as_ptr().cast()) };
    }
}

/// Page-locked host memory: copies from and to it run at full PCIe speed
/// and asynchronously (pageable copies go through a driver bounce buffer,
/// at a fraction of the bandwidth).
struct Pinned<T> {
    ptr: std::ptr::NonNull<T>,
    len: usize,
    /// The allocation's size (at least `len` elements).
    bytes: usize,
    stream: Arc<CudaStream>,
    completion: CudaEvent,
    pending: bool,
    write_combined: bool,
    /// Where the block returns on drop; `None` frees it.
    pool: Option<Arc<PinnedPool>>,
}

// SAFETY: plain host memory owned by this value, accessed through `&self`
// / `&mut self` borrows only.
unsafe impl<T: Send> Send for Pinned<T> {}
// SAFETY: as above.
unsafe impl<T: Sync> Sync for Pinned<T> {}

impl<T: Plain> Pinned<T> {
    /// At least `len` elements, freed on drop. Write-combined memory is for
    /// uploads only; readbacks use cacheable memory.
    fn new(
        stream: &Arc<CudaStream>,
        len: usize,
        write_combined: bool,
    ) -> std::result::Result<Self, DriverError> {
        Self::alloc(stream, len, write_combined, None)
    }

    /// At least `len` elements taken from `pool` (or allocated), returned
    /// to it on drop.
    fn pooled(
        pool: &Arc<PinnedPool>,
        stream: &Arc<CudaStream>,
        len: usize,
        write_combined: bool,
    ) -> std::result::Result<Self, DriverError> {
        Self::alloc(stream, len, write_combined, Some(pool))
    }

    fn alloc(
        stream: &Arc<CudaStream>,
        len: usize,
        write_combined: bool,
        pool: Option<&Arc<PinnedPool>>,
    ) -> std::result::Result<Self, DriverError> {
        let ctx = stream.context();
        ctx.bind_to_thread()?;
        let completion = ctx.new_event(None)?;
        let bytes = len
            .max(1)
            .checked_mul(std::mem::size_of::<T>())
            .filter(|&bytes| isize::try_from(bytes).is_ok())
            .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
        let block = if let Some(block) = pool.and_then(|pool| pool.take(bytes, write_combined)) {
            block
        } else {
            let flags = if write_combined {
                sys::CU_MEMHOSTALLOC_WRITECOMBINED
            } else {
                0
            };
            // SAFETY: allocates `bytes` bytes of page-locked host memory,
            // owned by the returned value.
            let raw = unsafe { cudarc::driver::result::malloc_host(bytes, flags) }?;
            let ptr = std::ptr::NonNull::new(raw.cast::<u8>())
                .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
            // SAFETY: the allocation holds `bytes` writable bytes; zeroing
            // initializes them.
            unsafe { ptr.as_ptr().write_bytes(0, bytes) };
            Block {
                ptr,
                bytes,
                write_combined,
            }
        };
        // Page-aligned, so suitably aligned for any `T`.
        Ok(Pinned {
            ptr: block.ptr.cast(),
            len: block.bytes / std::mem::size_of::<T>(),
            bytes: block.bytes,
            stream: stream.clone(),
            completion,
            pending: false,
            write_combined,
            pool: pool.cloned(),
        })
    }

    /// Wait only for this buffer's last copy before host reuse.
    fn wait(&mut self) -> std::result::Result<(), DriverError> {
        if self.pending {
            self.completion.synchronize()?;
            self.pending = false;
        }
        Ok(())
    }

    fn as_slice(&self) -> &[T] {
        // SAFETY: `len` initialized elements (zeroed or written as `Plain`
        // values) owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `len` initialized elements owned exclusively by `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for Pinned<T> {
    fn drop(&mut self) {
        // Drain even on an error exit before an upload's event was recorded.
        // If CUDA cannot establish completion, leak rather than free or
        // reuse memory that the DMA engine may still access.
        if self.stream.synchronize().is_err() {
            return;
        }
        let block = Block {
            ptr: self.ptr.cast(),
            bytes: self.bytes,
            write_combined: self.write_combined,
        };
        match &self.pool {
            Some(pool) => pool.give(block),
            None => {
                // SAFETY: allocated by `malloc_host`, freed once after all DMA.
                let _ = unsafe { cudarc::driver::result::free_host(block.ptr.as_ptr().cast()) };
            }
        }
    }
}

/// A pooled pinned buffer of at least `len` elements in `slot`.
fn pinned<'a, T: Plain>(
    pool: &Arc<PinnedPool>,
    stream: &Arc<CudaStream>,
    slot: &'a mut Option<Pinned<T>>,
    len: usize,
    write_combined: bool,
) -> std::result::Result<&'a mut Pinned<T>, DriverError> {
    if slot
        .as_ref()
        .is_none_or(|p| p.len < len || p.write_combined != write_combined)
    {
        *slot = None;
        let len = len
            .max(MIN_PINNED_BYTES / std::mem::size_of::<T>())
            .checked_next_power_of_two()
            .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
        *slot = Some(Pinned::pooled(pool, stream, len, write_combined)?);
    }
    slot.as_mut()
        .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))
}

/// `values` as their bytes.
fn bytes_of<T: Plain>(values: &[T]) -> &[u8] {
    // SAFETY: `Plain` values have no padding or invalid bit patterns.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// `values` as their bytes, writable: any bytes are valid `Plain` values.
fn bytes_of_mut<T: Plain>(values: &mut [T]) -> &mut [u8] {
    let len = std::mem::size_of_val(values);
    // SAFETY: as for `bytes_of`; every bit pattern is a valid value.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len) }
}

/// Small uploads (descriptors, leaf values, gradient-domain seeds) staged in
/// one write-combined pinned arena and copied without waiting. A pageable
/// copy synchronizes the stream (cudarc cannot track its source), and a
/// node issued about a dozen of them, each stalling the host behind every
/// queued kernel. A queued copy's bytes are not rewritten until the stream
/// has synchronized after it: at the next readback ([`Self::synced`]), or
/// here when the arena is full.
#[derive(Default)]
struct Staging {
    arena: Option<Pinned<u8>>,
    used: usize,
}

impl Staging {
    /// `bytes` writable arena bytes (16-byte aligned) for a queued copy,
    /// waiting for the stream first when the arena is full.
    fn reserve(
        &mut self,
        pool: &Arc<PinnedPool>,
        stream: &Arc<CudaStream>,
        bytes: usize,
    ) -> std::result::Result<&mut [u8], DriverError> {
        let capacity = self.arena.as_ref().map_or(0, |arena| arena.len);
        let mut start = self.used.next_multiple_of(16);
        if start + bytes > capacity {
            // Every queued copy from the arena completes before any of it
            // is rewritten.
            stream.synchronize()?;
            start = 0;
            if bytes > capacity {
                self.arena = None;
                let size = bytes
                    .max(STAGING_BYTES)
                    .checked_next_power_of_two()
                    .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
                self.arena = Some(Pinned::pooled(pool, stream, size, true)?);
            }
        }
        self.used = start + bytes;
        let arena = self
            .arena
            .as_mut()
            .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
        Ok(&mut arena.as_mut_slice()[start..start + bytes])
    }

    /// Copy `data` to the front of `buf`, growing it as needed.
    fn upload<T: Plain>(
        &mut self,
        pool: &Arc<PinnedPool>,
        stream: &Arc<CudaStream>,
        buf: &mut CudaSlice<T>,
        data: &[T],
    ) -> std::result::Result<(), DriverError> {
        if data.is_empty() {
            return Ok(());
        }
        fit(stream, buf, data.len())?;
        let src = bytes_of(data);
        let host = self.reserve(pool, stream, src.len())?;
        host.copy_from_slice(src);
        // Other host work can change the thread's current CUDA context.
        stream.context().bind_to_thread()?;
        let (dst, _record) = buf.device_ptr_mut(stream);
        // SAFETY: the pinned source is not rewritten before the stream has
        // synchronized after this copy (see the type docs), and `buf` holds
        // at least `data.len()` elements.
        unsafe { cudarc::driver::result::memcpy_htod_async(dst, &*host, stream.cu_stream()) }?;
        Ok(())
    }

    /// Copy `parts` to consecutive 16-byte-aligned ranges at the front of
    /// `buf` (growing it as needed) with one queued copy: each part's
    /// device address. The descriptors of one operation travel together.
    fn upload_parts<const N: usize>(
        &mut self,
        pool: &Arc<PinnedPool>,
        stream: &Arc<CudaStream>,
        buf: &mut CudaSlice<u8>,
        parts: [&[u8]; N],
    ) -> std::result::Result<[sys::CUdeviceptr; N], DriverError> {
        let mut offsets = [0usize; N];
        let mut total = 0usize;
        for (offset, part) in offsets.iter_mut().zip(&parts) {
            *offset = total;
            total = (total + part.len()).next_multiple_of(16);
        }
        fit(stream, buf, total.max(16))?;
        let host = self.reserve(pool, stream, total)?;
        for (&offset, part) in offsets.iter().zip(&parts) {
            host[offset..offset + part.len()].copy_from_slice(part);
        }
        stream.context().bind_to_thread()?;
        let (dst, _record) = buf.device_ptr_mut(stream);
        if total > 0 {
            // SAFETY: as for `upload`; `buf` holds at least `total` bytes.
            unsafe { cudarc::driver::result::memcpy_htod_async(dst, &*host, stream.cu_stream()) }?;
        }
        Ok(offsets.map(|offset| dst + offset as u64))
    }

    /// The stream has synchronized: no queued copy reads the arena.
    fn synced(&mut self) {
        self.used = 0;
    }
}

/// Bounded pinned staging for large copies (bins, gradients, row lists,
/// histograms, leaf rows): [`PIECES`] pooled buffers of [`PIECE_BYTES`] per
/// direction, each rewritten only after its previous piece's copy has
/// completed. The host fills or drains one piece while DMA moves another,
/// and no transfer page-locks memory of its own size.
#[derive(Default)]
struct Ring {
    up: Vec<Pinned<u8>>,
    down: Vec<Pinned<u8>>,
}

impl Ring {
    /// Piece buffer `k` of `pieces`, allocated on first use.
    fn piece<'a>(
        pieces: &'a mut Vec<Pinned<u8>>,
        k: usize,
        pool: &Arc<PinnedPool>,
        stream: &Arc<CudaStream>,
        write_combined: bool,
    ) -> std::result::Result<&'a mut Pinned<u8>, DriverError> {
        while pieces.len() <= k {
            pieces.push(Pinned::pooled(pool, stream, PIECE_BYTES, write_combined)?);
        }
        Ok(&mut pieces[k])
    }

    /// Copy `src` into `dst`'s front; returns once the last piece is
    /// queued (its buffer's reuse waits for it).
    fn upload<T: Plain>(
        &mut self,
        pool: &Arc<PinnedPool>,
        stream: &Arc<CudaStream>,
        src: &[T],
        dst: &mut CudaSlice<T>,
    ) -> std::result::Result<(), DriverError> {
        if src.len() > dst.len() {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        stream.context().bind_to_thread()?;
        let (base, _record) = dst.device_ptr_mut(stream);
        let copied = (|| {
            for (k, chunk) in bytes_of(src).chunks(PIECE_BYTES).enumerate() {
                let piece = Self::piece(&mut self.up, k % PIECES, pool, stream, true)?;
                piece.wait()?;
                // This also runs under the backend mutex: no Rayon here,
                // which could steal another build that blocks on it.
                let host = &mut piece.as_mut_slice()[..chunk.len()];
                host.copy_from_slice(chunk);
                stream.context().bind_to_thread()?;
                let offset = (k * PIECE_BYTES) as u64;
                // SAFETY: the piece stays untouched until its completion
                // event (recorded next) passes; the destination range lies
                // within `dst`.
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        base + offset,
                        &*host,
                        stream.cu_stream(),
                    )
                }?;
                piece.completion.record(stream)?;
                piece.pending = true;
            }
            Ok(())
        })();
        if copied.is_err() {
            // Earlier pieces may still be in flight.
            let _ = stream.synchronize();
        }
        copied
    }

    /// Copy `out.len()` elements from `src`'s front (a buffer or a view of
    /// one) into `out`, waiting for them.
    fn download<T: Plain>(
        &mut self,
        pool: &Arc<PinnedPool>,
        stream: &Arc<CudaStream>,
        src: &impl DevicePtr<T>,
        out: &mut [T],
    ) -> std::result::Result<(), DriverError> {
        if out.len() > src.len() {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        let out = bytes_of_mut(out);
        let total = out.len();
        let n = total.div_ceil(PIECE_BYTES);
        let span = |k: usize| k * PIECE_BYTES..((k + 1) * PIECE_BYTES).min(total);
        stream.context().bind_to_thread()?;
        let (base, _record) = src.device_ptr(stream);
        let copied = (|| {
            for k in 0..n + PIECES {
                // Drain piece `k - PIECES` before its buffer takes piece `k`.
                if let Some(done) = k.checked_sub(PIECES).filter(|&done| done < n) {
                    let piece = &mut self.down[done % PIECES];
                    piece.wait()?;
                    let range = span(done);
                    let len = range.len();
                    out[range].copy_from_slice(&piece.as_slice()[..len]);
                }
                if k < n {
                    let range = span(k);
                    let piece = Self::piece(&mut self.down, k % PIECES, pool, stream, false)?;
                    stream.context().bind_to_thread()?;
                    let host = &mut piece.as_mut_slice()[..range.len()];
                    // SAFETY: the piece is read only after its completion
                    // event (recorded next) passes; the source range lies
                    // within `src`.
                    unsafe {
                        cudarc::driver::result::memcpy_dtoh_async(
                            host,
                            base + range.start as u64,
                            stream.cu_stream(),
                        )
                    }?;
                    piece.completion.record(stream)?;
                    piece.pending = true;
                }
            }
            Ok(())
        })();
        if copied.is_err() {
            let _ = stream.synchronize();
        }
        copied
    }
}

/// Copy `src` into `dst`'s front through the pinned buffer `staging`, in
/// pieces: the host fills piece `k + 1` while DMA moves piece `k`. Record
/// completion on the owner before returning; host reuse waits on that event.
fn upload_pinned<T: Plain>(
    stream: &Arc<CudaStream>,
    staging: &mut Pinned<T>,
    src: &[T],
    dst: &mut CudaSlice<T>,
) -> std::result::Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    staging.wait()?;
    if src.len() > staging.len || src.len() > dst.len() {
        return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
    }
    let piece = (PIECE_BYTES / std::mem::size_of::<T>()).max(1);
    let (base, _record) = dst.device_ptr_mut(stream);
    let copied = (|| {
        let host = &mut staging.as_mut_slice()[..src.len()];
        for (k, (h, s)) in host.chunks_mut(piece).zip(src.chunks(piece)).enumerate() {
            // This helper also runs under the backend mutex. Rayon here
            // could steal another build and block recursively on that mutex.
            h.copy_from_slice(s);
            let offset = (k * piece * std::mem::size_of::<T>()) as u64;
            // Other host work can change the thread's current CUDA context.
            stream.context().bind_to_thread()?;
            // SAFETY: disjoint pinned pieces remain owned and untouched until
            // completion; the destination range lies within `dst`.
            unsafe {
                cudarc::driver::result::memcpy_htod_async(base + offset, h, stream.cu_stream())
            }?;
        }
        staging.completion.record(stream)
    })();
    if copied.is_ok() {
        staging.pending = true;
    } else {
        // Earlier pieces may still be in flight even if a later enqueue fails.
        let _ = stream.synchronize();
    }
    copied
}

/// Copy `len` elements from `src`'s front into `staging` and wait for them.
fn download_pinned<'a, T: Plain>(
    stream: &Arc<CudaStream>,
    staging: &'a mut Pinned<T>,
    src: &CudaSlice<T>,
    len: usize,
) -> std::result::Result<&'a [T], DriverError> {
    stream.context().bind_to_thread()?;
    staging.wait()?;
    if len > staging.len || len > src.len() {
        return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
    }
    if len > 0 {
        let (ptr, _record) = src.device_ptr(stream);
        let host = &mut staging.as_mut_slice()[..len];
        // SAFETY: pinned destination of `len` elements, read only after
        // the synchronization below.
        unsafe { cudarc::driver::result::memcpy_dtoh_async(host, ptr, stream.cu_stream()) }?;
    }
    stream.synchronize()?;
    Ok(&staging.as_slice()[..len])
}

/// The CUDA histogram backend: implements [`HistogramBackend`] on an NVIDIA
/// GPU, and grows hist trees with their rows on the GPU (see the
/// [module docs](self)). Constructed once per training run (the index
/// upload is per-dataset); the gradient slice is uploaded by
/// [`HistogramBackend::prepare`] once per tree.
///
/// Training selects it automatically through
/// [`device = cuda`](crate::config::TrainingParams::device); constructing it
/// directly serves custom training loops against a [`GHistIndex`]. Its
/// histograms equal the CPU backend's bit for bit. `build` must receive the
/// index the backend was built from (or a clone), and the gradient slice
/// must not change between `prepare` and the tree's last `build`. A different
/// index, a gradient slice of another length, or row indices past the index
/// never reach the GPU and take the CPU path, which checks them.
pub struct CudaHistBackend {
    device: Arc<Device>,
    index_identity: u64,
    n_rows: usize,
    n_cols: usize,
    total_bins: usize,
    /// Whether the index is dense (no missing values).
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

/// Device buffers grown by [`fit`] start at this many elements, so the
/// descriptor buffers' growth through a run's first levels stays rare.
const MIN_DEVICE_LEN: usize = 1 << 10;

/// A device buffer of at least `len` elements: `buf` itself, or a fresh
/// (zeroed) one replacing it, sized up to a power of two so repeated growth
/// stays rare.
fn fit<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<T>,
    len: usize,
) -> std::result::Result<(), DriverError> {
    if buf.len() < len {
        let capacity = len
            .max(MIN_DEVICE_LEN)
            .checked_next_power_of_two()
            .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
        *buf = stream.alloc_zeros(sized::<T>(capacity)?)?;
    }
    Ok(())
}

/// Pageable source slices are not event-tracked by cudarc. Complete their
/// copy before returning to a caller that may immediately drop/reuse them.
fn copy_host<T: DeviceRepr, D: DevicePtrMut<T>>(
    stream: &Arc<CudaStream>,
    data: &[T],
    destination: &mut D,
) -> std::result::Result<(), DriverError> {
    let copied = stream.memcpy_htod(data, destination);
    let completed = stream.synchronize();
    copied.and(completed)
}

fn clone_host<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    data: &[T],
) -> std::result::Result<CudaSlice<T>, DriverError> {
    let mut result = stream.alloc_zeros(data.len().max(1))?;
    copy_host(stream, data, &mut result.slice_mut(..data.len()))?;
    Ok(result)
}

/// Feature groups over `bins_of` (bins per feature): contiguous features
/// whose bins fit `cap`, balanced, and the features wider than `cap` alone.
/// Each group is `[first feature, end feature, first global bin, bins]`.
fn feature_groups(bins_of: &[usize], cap: usize) -> (Vec<u32>, Vec<u32>) {
    let total: usize = bins_of.iter().filter(|&&b| b <= cap).sum();
    let target = total.div_ceil(total.div_ceil(cap.max(1)).max(1)).max(1);
    let (mut shared, mut global) = (Vec::new(), Vec::new());
    let mut first_bin = 0usize;
    let mut start: Option<(usize, usize)> = None;
    let mut open_bins = 0usize;
    let close = |out: &mut Vec<u32>, (f0, b0): (usize, usize), f1: usize, bins: usize| {
        out.extend([f0 as u32, f1 as u32, b0 as u32, bins as u32]);
    };
    for (f, &bins) in bins_of.iter().enumerate() {
        if bins > cap {
            if let Some(open) = start.take() {
                close(&mut shared, open, f, open_bins);
            }
            close(&mut global, (f, first_bin), f + 1, bins);
        } else {
            if let Some(open) = start
                && (open_bins + bins > cap || open_bins >= target)
            {
                close(&mut shared, open, f, open_bins);
                start = None;
            }
            if start.is_none() {
                start = Some((f, first_bin));
                open_bins = 0;
            }
            open_bins += bins;
        }
        first_bin += bins;
    }
    if let Some(open) = start {
        close(&mut shared, open, bins_of.len(), open_bins);
    }
    (shared, global)
}

/// Row stride (entries) of `n_cols` bins of `width` bytes: rows start on
/// 32-byte boundaries, so a row's bins span the fewest memory sectors.
fn row_stride(n_cols: usize, width: usize) -> usize {
    (n_cols * width).div_ceil(32) * 32 / width
}

/// Checked element count bounded by Rust's allocation and CUDA offsets.
fn sized<T>(len: usize) -> std::result::Result<usize, DriverError> {
    len.checked_mul(std::mem::size_of::<T>())
        .filter(|&bytes| isize::try_from(bytes).is_ok())
        .map(|_| len)
        .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))
}

/// Upload the CPU-authoritative global bins without a second host bin vector.
fn global_bins(
    device: &Device,
    ring: &mut Ring,
    index: &GHistIndex,
) -> std::result::Result<DeviceBins, DriverError> {
    fn copy<T: Plain>(
        device: &Device,
        ring: &mut Ring,
        src: &[T],
    ) -> std::result::Result<CudaSlice<T>, DriverError> {
        let stream = &device.stream;
        let mut dst = stream.alloc_zeros(sized::<T>(src.len().max(1))?)?;
        ring.upload(&device.pinned, stream, src, &mut dst)?;
        Ok(dst)
    }
    match index.bins() {
        Bins::U16(src) => copy(device, ring, src).map(DeviceBins::U16),
        Bins::U32(src) => copy(device, ring, src).map(DeviceBins::U32),
    }
}

/// Encode and transpose a complete dense index in one device pass.
fn encode_dense<T: DeviceRepr + ValidAsZeroBits>(
    device: &Device,
    index: &GHistIndex,
    global: &DeviceBins,
    first: &CudaSlice<u32>,
    kernels: &[CudaFunction; 3],
) -> std::result::Result<(CudaSlice<T>, CudaSlice<T>, u32), DriverError> {
    let width = std::mem::size_of::<T>();
    let stride = row_stride(index.n_cols(), width);
    let stride32 =
        u32::try_from(stride).map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
    let row_len = index
        .n_rows()
        .checked_mul(stride)
        .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
    let col_len = index
        .n_rows()
        .checked_mul(index.n_cols())
        .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
    let mut rows = device.stream.alloc_zeros::<T>(sized::<T>(row_len)?)?;
    let mut cols = device.stream.alloc_zeros::<T>(sized::<T>(col_len)?)?;
    let n_rows = index.n_rows() as u64;
    let n_cols = index.n_cols() as u32;
    let mut launch = device.stream.launch_builder(&kernels[global.width()]);
    global.push(&mut launch);
    launch
        .arg(first)
        .arg(&n_rows)
        .arg(&n_cols)
        .arg(&stride32)
        .arg(&mut rows)
        .arg(&mut cols);
    let tiles = index.n_rows().div_ceil(32) * index.n_cols().div_ceil(32);
    let config = LaunchConfig {
        grid_dim: (
            tiles.min((device.sm_count * BLOCKS_PER_SM) as usize).max(1) as u32,
            1,
            1,
        ),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: complete 32x8 blocks cooperate on 32x32 tiles, with every
    // boundary checked before loading; each dense local bin fits T.
    unsafe { launch.launch(config) }?;
    Ok((rows, cols, stride32))
}

impl CudaHistBackend {
    /// Build the backend for `index` on CUDA device `ordinal`: upload its
    /// bins and allocate the per-tree buffers.
    pub fn new(index: &GHistIndex, ordinal: usize) -> Result<Self> {
        let n_rows = index.n_rows();
        let n_cols = index.n_cols();
        let total_bins = index.total_bins();
        if total_bins == 0 || n_rows == 0 || n_cols == 0 {
            return Err(HessboostError::invalid_data(
                "data",
                "the CUDA backend needs a non-empty binned dataset",
            ));
        }
        if u32::try_from(n_rows).is_err()
            || u32::try_from(n_cols).is_err()
            || u32::try_from(total_bins).is_err()
        {
            return Err(HessboostError::invalid_data(
                "data",
                format!(
                    "the CUDA backend indexes rows and bins in 32 bits \
                     ({n_rows} rows, {total_bins} bins)"
                ),
            ));
        }
        let device = device(ordinal)
            .map_err(|reason| HessboostError::gpu(reason.to_string()))?
            .for_backend()
            .map_err(gpu_error)?;
        let state = Self::upload(&device, index)
            .inspect_err(|_| device.failed.store(true, Ordering::Release))
            .map_err(gpu_error)?;
        Ok(CudaHistBackend {
            index_identity: index.identity(),
            n_rows,
            n_cols,
            total_bins,
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

    /// The nodes (and rows) built so far with each strategy
    /// (`hessboost::internals`, for tests and benchmarks).
    #[doc(hidden)]
    #[must_use]
    pub fn node_counts(&self) -> NodeCounts {
        self.counters.snapshot()
    }

    fn upload(device: &Device, index: &GHistIndex) -> std::result::Result<State, DriverError> {
        let stream = &device.stream;
        let n_rows = index.n_rows();
        let n_cols = index.n_cols();
        let total_bins = index.total_bins();
        let cuts = index.cuts();
        let bins_of: Vec<usize> = (0..n_cols)
            .map(|f| {
                let (fs, fe) = cuts.feature_bins(f);
                fe - fs
            })
            .collect();
        let widest = bins_of.iter().copied().max().unwrap_or(0);
        let dense = index.dense_stride().is_some();
        let mut first: Vec<u32> = (0..n_cols).map(|f| cuts.feature_bins(f).0 as u32).collect();
        first.push(total_bins as u32);
        let feature_first = clone_host(stream, &first)?;
        let mut ring = Ring::default();
        let global_bins = global_bins(device, &mut ring, index)?;
        let (bins, cols, row_ptr, stride) = if dense {
            let (bins, cols, stride) = if widest <= u8::MAX as usize + 1 {
                let (r, c, s) = encode_dense::<u8>(
                    device,
                    index,
                    &global_bins,
                    &feature_first,
                    &device.kernels.encode_u8,
                )?;
                (DeviceBins::U8(r), DeviceBins::U8(c), s)
            } else if widest <= u16::MAX as usize + 1 {
                let (r, c, s) = encode_dense::<u16>(
                    device,
                    index,
                    &global_bins,
                    &feature_first,
                    &device.kernels.encode_u16,
                )?;
                (DeviceBins::U16(r), DeviceBins::U16(c), s)
            } else {
                let (r, c, s) = encode_dense::<u32>(
                    device,
                    index,
                    &global_bins,
                    &feature_first,
                    &device.kernels.encode_u32,
                )?;
                (DeviceBins::U32(r), DeviceBins::U32(c), s)
            };
            (bins, Some(cols), None, stride)
        } else {
            let offsets: Vec<u64> = index.row_ptr().iter().map(|&off| off as u64).collect();
            (global_bins, None, Some(clone_host(stream, &offsets)?), 0)
        };
        let sentinel = u32::MAX;
        let row_words = sized::<i64>(
            n_rows
                .checked_mul(2)
                .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?,
        )?;
        let hist_words = sized::<f64>(
            total_bins
                .checked_mul(2)
                .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?,
        )?;
        let cap = device.shared_bytes / 16;
        let (shared, global) = feature_groups(&bins_of, cap);
        let group_features = shared
            .chunks(4)
            .chain(global.chunks(4))
            .map(|g| (g[1] - g[0]) as usize)
            .max()
            .unwrap_or(1);
        let group_bins = shared.chunks(4).map(|g| g[3] as usize).max().unwrap_or(0);
        let alloc_u32 = |len: usize| stream.alloc_zeros::<u32>(len.max(1));
        let alloc_u64 = |len: usize| stream.alloc_zeros::<u64>(len.max(1));
        let groups_shared = clone_host(stream, &shared)?;
        let groups_global = clone_host(stream, &global)?;
        Ok(State {
            bins,
            cols,
            row_ptr,
            stride,
            flags: stream.alloc_zeros(n_rows)?,
            sentinel,
            feature_first,
            n_shared: shared.len() / 4,
            groups_shared,
            n_global: global.len() / 4,
            groups_global,
            group_bins,
            group_features,
            gpair: stream.alloc_zeros(row_words)?,
            units: stream.alloc_zeros(row_words)?,
            tree_rows: stream.alloc_zeros(n_rows)?,
            scratch: stream.alloc_zeros(n_rows)?,
            tree_len: 0,
            upload: stream.alloc_zeros(n_rows)?,
            acc: alloc_u64(hist_words)?,
            partials: alloc_u64(1)?,
            wave_slots: (PARTIAL_BYTES / (hist_words * 8)).max(1),
            out: stream.alloc_zeros(hist_words)?,
            desc: stream.alloc_zeros(1)?,
            totals: stream.alloc_zeros(2)?,
            tile_left: alloc_u32(1)?,
            left_len: alloc_u32(1)?,
            partition_host: PartitionHost::default(),
            pin_counts: None,
            domain: alloc_u32(6)?,
            margins: None,
            labels: None,
            weights: None,
            staged: Staged {
                addr: 0,
                len: 0,
                grad: SumDomain::EMPTY,
                hess: SumDomain::EMPTY,
                host: None,
            },
            staging: Staging::default(),
            ring,
            pool: stream.alloc_zeros(2)?,
            pool_slots: 0,
            scan: categorical::ScanState::new(stream)?,
        })
    }

    /// The uploaded index's immutable identity (also valid after moves/clones).
    fn fits(&self, ghist: &GHistIndex) -> bool {
        ghist.identity() == self.index_identity
    }

    /// Lock the device state, unless the device has failed.
    fn lock(&self) -> Option<MutexGuard<'_, State>> {
        if self.device.failed.load(Ordering::Acquire) {
            return None;
        }
        Some(self.state.lock())
    }

    /// `result`'s value, or `None` after marking the device failed (CUDA
    /// errors are sticky).
    fn ok<T>(&self, result: std::result::Result<T, DriverError>) -> Option<T> {
        result
            .inspect_err(|_| self.device.failed.store(true, Ordering::Release))
            .ok()
    }

    /// Put `rows` (all below `n_rows`) at the front of `dst`: generated on
    /// the device when they are one ascending `run`, else uploaded.
    fn place_rows(
        &self,
        ring: &mut Ring,
        dst: &mut CudaSlice<u32>,
        rows: &[u32],
        run: bool,
    ) -> std::result::Result<(), DriverError> {
        let stream = &self.device.stream;
        match rows.first() {
            Some(&first) if run => {
                let n = rows.len() as u64;
                let mut launch = stream.launch_builder(&self.device.kernels.iota_rows);
                launch.arg(dst).arg(&n).arg(&first);
                // SAFETY: writes `rows.len() <= n_rows` entries.
                unsafe { launch.launch(self.device.grid(rows.len())) }.map(|_| ())
            }
            _ => ring.upload(&self.device.pinned, stream, rows, dst),
        }
    }

    /// Stage `gpair` on the device with its exactness statistics. A slice
    /// of any length other than `n_rows` is not staged.
    fn stage(&self, state: &mut State, gpair: &[GradPair]) -> std::result::Result<(), DriverError> {
        // Unstage first, so a slice that fails to upload is never mistaken
        // for the previous one.
        state.staged.len = 0;
        if gpair.len() != self.n_rows {
            return Ok(());
        }
        // Do not enter Rayon while holding state: a waiting worker can steal
        // another histogram task that needs this same mutex.
        let grad = SumDomain::of(gpair.iter().map(|p| p.grad));
        let hess = SumDomain::of(gpair.iter().map(|p| p.hess));
        // SAFETY: `GradPair` is `repr(C)` of two `f32`s, so the slice is
        // `2 * len` contiguous `f32`s.
        let flat =
            unsafe { std::slice::from_raw_parts(gpair.as_ptr().cast::<f32>(), gpair.len() * 2) };
        let stream = &self.device.stream;
        state
            .ring
            .upload(&self.device.pinned, stream, flat, &mut state.gpair)?;
        self.stage_units(state, grad, hess)?;
        state.staged.addr = gpair.as_ptr().addr();
        Ok(())
    }

    /// Convert the device's gradient pairs into grains of `grad` and
    /// `hess`, the statistics of them, and record them as staged (from the
    /// device, until the caller sets a host address).
    fn stage_units(
        &self,
        state: &mut State,
        grad: SumDomain,
        hess: SumDomain,
    ) -> std::result::Result<(), DriverError> {
        if grad.sums_exact(1) && hess.sums_exact(1) {
            let stream = &self.device.stream;
            let n = self.n_rows as u64;
            let (to_grad, to_hess) = (grad.unit_scale(), hess.unit_scale());
            let mut launch = stream.launch_builder(&self.device.kernels.stage_units);
            launch
                .arg(&state.gpair)
                .arg(&mut state.units)
                .arg(&n)
                .arg(&to_grad)
                .arg(&to_hess);
            // SAFETY: `n_rows` pairs in and units out; every scaled value
            // is finite, an exact integer, and at most 2^53 in magnitude
            // (`sums_exact(1)`), so the kernel's `as i64` keeps it exactly.
            unsafe { launch.launch(self.device.grid(self.n_rows)) }?;
        }
        state.staged = Staged {
            addr: 0,
            len: self.n_rows,
            grad,
            hess,
            host: None,
        };
        Ok(())
    }

    /// The statistics of the device's gradient pairs, folded on the device.
    fn device_domains(
        &self,
        state: &mut State,
    ) -> std::result::Result<(SumDomain, SumDomain), DriverError> {
        let device = &*self.device;
        let stream = &device.stream;
        state.staging.upload(
            &device.pinned,
            stream,
            &mut state.domain,
            &[0, u32::MAX, 1, 0, u32::MAX, 1],
        )?;
        let n = self.n_rows as u64;
        let mut launch = stream.launch_builder(&device.kernels.grad_domain);
        launch.arg(&state.gpair).arg(&n).arg(&mut state.domain);
        // SAFETY: reads `n_rows` pairs and folds into six words.
        unsafe { launch.launch(device.grid(self.n_rows)) }?;
        let mut host = [0u32; 6];
        stream.memcpy_dtoh(&state.domain.slice(..6), &mut host[..])?;
        stream.synchronize()?;
        state.staging.synced();
        Ok((
            SumDomain::from_device(host[0], host[1], host[2] != 0),
            SumDomain::from_device(host[3], host[4], host[5] != 0),
        ))
    }

    /// Launch the integer histogram kernels over the `n_tiles` tiles at
    /// device address `tiles`, reading `source`.
    fn launch_tiles(
        &self,
        state: &mut State,
        source: RowSource<'_>,
        tiles: sys::CUdeviceptr,
        n_tiles: usize,
    ) -> std::result::Result<(), DriverError> {
        let device = &*self.device;
        let stream = &device.stream;
        let State {
            bins,
            row_ptr,
            stride,
            sentinel,
            feature_first,
            groups_shared,
            n_shared,
            groups_global,
            n_global,
            group_bins,
            units,
            tree_rows,
            upload,
            acc,
            partials,
            ..
        } = state;
        let rows = match source {
            RowSource::Tree => &*tree_rows,
            RowSource::Upload(_) => &*upload,
        };
        let stride = *stride;
        let total_bins = self.total_bins as u64;
        let w = bins.width();
        if let Some(row_ptr) = &*row_ptr {
            let work = abi::SparseTiles { tiles, total_bins };
            let mut launch = stream.launch_builder(&device.kernels.hist_sparse[w]);
            bins.push(&mut launch);
            launch
                .arg(row_ptr)
                .arg(rows)
                .arg(&*units)
                .arg(&mut *acc)
                .arg(&mut *partials)
                .arg(&work);
            let blocks = u32::try_from(n_tiles)
                .ok()
                .filter(|&n| i32::try_from(n).is_ok())
                .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
            // SAFETY: CSR offsets and global bins match the index; tile
            // slots are sized and cleared by the caller, sums are exact.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (HIST_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            return Ok(());
        }
        for (kernel, groups, n_groups, shared) in [
            (
                &device.kernels.hist_shared[w],
                &*groups_shared,
                *n_shared,
                *group_bins * 16,
            ),
            (
                &device.kernels.hist_global[w],
                &*groups_global,
                *n_global,
                0,
            ),
        ] {
            if n_groups == 0 {
                continue;
            }
            let work = abi::TileWork {
                tiles,
                groups: abi::ptr(groups, stream),
                total_bins,
                stride,
                sentinel: *sentinel,
                n_groups: n_groups as u32,
            };
            let mut launch = stream.launch_builder(kernel);
            bins.push(&mut launch);
            launch
                .arg(&*feature_first)
                .arg(rows)
                .arg(&*units)
                .arg(&mut *acc)
                .arg(&mut *partials)
                .arg(&work);
            let blocks = n_tiles
                .checked_mul(n_groups)
                .and_then(|n| u32::try_from(n).ok())
                .filter(|&n| i32::try_from(n).is_ok())
                .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
            let config = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (HIST_THREADS, 1, 1),
                shared_mem_bytes: shared as u32,
            };
            // SAFETY: every tile lists rows of `rows` below `n_rows` (the
            // callers' segments lie in the valid prefix), each group's
            // shared histogram fits the dynamic allocation, tile targets
            // index slots `acc` and `partials` were sized for, and every
            // stored local bin plus its feature's first bin is below
            // `total_bins`. Arguments match the kernel's parameters.
            unsafe { launch.launch(config) }?;
        }
        Ok(())
    }

    /// Reserve only the partials this batch needs, capped by a quarter of
    /// currently free memory. Smaller waves keep the same reduction order.
    fn partial_wave(
        &self,
        state: &mut State,
        chunks: usize,
    ) -> std::result::Result<usize, DriverError> {
        let words = self.total_bins * 2;
        let requested = chunks.min(state.wave_slots).max(1);
        if state.partials.len() / words >= requested {
            return Ok(requested);
        }
        let stream = &self.device.stream;
        stream.context().bind_to_thread()?;
        let (free, _) = cudarc::driver::result::mem_get_info()?;
        let held = state.partials.len() / words;
        let budget = (free / 4 / (words * 8)).max(held).max(1);
        let wave = requested.min(budget);
        if held < wave {
            // Exact capacity: power-of-two growth could exceed the budget.
            state.partials = stream.alloc_zeros(wave * words)?;
        }
        Ok(wave)
    }

    /// The histograms of `nodes` (segments of `source`'s rows; an upload's
    /// host copy feeds CPU-built nodes), put where `dest` says: read back
    /// (one per node), read back into the single node's buffer, or written
    /// to resident pool slots (both returning an empty list).
    /// CPU-built nodes use `gpair`, or the device's gradients downloaded
    /// once per staging when only the device has them.
    fn histograms_on(
        &self,
        state: &mut State,
        source: RowSource<'_>,
        ghist: &GHistIndex,
        gpair: Option<&[GradPair]>,
        nodes: &[Segment],
        mut dest: Dest<'_>,
    ) -> std::result::Result<Vec<Histogram>, DriverError> {
        let bins = self.total_bins;
        let device = &*self.device;
        let stream = &device.stream;
        let pool = &device.pinned;
        let resident = match dest {
            Dest::Resident(resident) => Some(resident),
            Dest::Read | Dest::Into(_) => None,
        };
        if let Dest::Into(out) = &dest
            && (nodes.len() != 1 || out.len() != bins)
        {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        let targets = resident.map(|r| r.targets);
        let mut results: Vec<Option<Histogram>> = (0..nodes.len()).map(|_| None).collect();
        let mut exact = Vec::new();
        let mut chunked = Vec::new();
        let mut chains = Vec::new();
        let mut cpu = Vec::new();
        let mut slots = Vec::new();
        // The resident siblings by built slot (one each: `race_free`), and
        // which of them the exact finalization subtracts.
        let siblings = resident.map_or(&[][..], |r| r.siblings);
        let mut by_built: Vec<(usize, usize)> = siblings
            .iter()
            .enumerate()
            .map(|(i, &(_, built))| (built as usize, i))
            .collect();
        by_built.sort_unstable();
        let mut fused = vec![false; siblings.len()];
        // Output slot of node `k`, the next free one without `targets`.
        let slot_of = |k: usize, next: usize| targets.map_or(next, |t| t[k] as usize);
        let max_tile = max_tile(state);
        for (k, &seg) in nodes.iter().enumerate() {
            if seg.len == 0 {
                match (targets, &mut dest) {
                    (Some(t), _) => {
                        let s = t[k] as usize;
                        stream.memset_zeros(
                            &mut state.out.slice_mut(s * bins * 2..(s + 1) * bins * 2),
                        )?;
                    }
                    (None, Dest::Into(out)) => out.fill(GradStats::default()),
                    (None, _) => results[k] = Some(zeroed(bins)),
                }
                self.counters.count(Strategy::Exact, 0);
                continue;
            }
            let (strategy, order) = plan(&state.staged, ghist, seg, max_tile);
            self.counters.count(strategy, seg.len);
            let slot = slot_of(k, slots.len());
            match (strategy, order) {
                (Strategy::Exact, _) => exact.push((k, seg, slot)),
                (Strategy::ExactChunks, SumOrder::Blocked { grain }) => {
                    chunked.push((k, seg, grain, slot));
                }
                (Strategy::Chains, SumOrder::Blocked { grain }) => {
                    chains.push((k, seg, grain, slot));
                }
                (Strategy::Chains, SumOrder::Chain) => chains.push((k, seg, seg.len, slot)),
                _ => {
                    cpu.push(k);
                    continue;
                }
            }
            slots.push(k);
        }
        if !slots.is_empty() && targets.is_none() {
            fit(stream, &mut state.out, slots.len() * bins * 2)?;
        }
        let (grad_value, hess_value) = (
            state.staged.grad.value_scale(),
            state.staged.hess.value_scale(),
        );
        // Strategy 4's inputs (the gradients and rows the CPU reads), read
        // back before this call's kernels are queued, so that the CPU builds
        // below overlap them.
        let downloaded = if gpair.is_none() && !cpu.is_empty() {
            Some(self.device_gradients(state)?)
        } else {
            None
        };
        let cpu_rows: Vec<Cow<'_, [u32]>> = match source {
            RowSource::Upload(rows) => cpu
                .iter()
                .map(|&k| Cow::Borrowed(&rows[nodes[k].offset..nodes[k].offset + nodes[k].len]))
                .collect(),
            RowSource::Tree => {
                let mut all = Vec::with_capacity(cpu.len());
                for &k in &cpu {
                    let seg = nodes[k];
                    let mut rows = vec![0u32; seg.len];
                    // Through the ring, `rows` is filled only from pieces
                    // already copied, so no copy into it outlives an error.
                    let src = state.tree_rows.slice(seg.offset..seg.offset + seg.len);
                    state.ring.download(pool, stream, &src, &mut rows)?;
                    all.push(Cow::Owned(rows));
                }
                if !all.is_empty() {
                    // Each (non-empty) download waited for the stream's
                    // work through it.
                    state.staging.synced();
                }
                all
            }
        };

        // Strategy 1: exact integers, any tiling, into per-node accumulators.
        for batch in exact.chunks(MAX_GRID_Y) {
            fit(stream, &mut state.acc, batch.len() * bins * 2)?;
            stream.memset_zeros(&mut state.acc.slice_mut(..batch.len() * bins * 2))?;
            // Rows per tile: a small batch takes smaller tiles, so its (tile,
            // feature group) blocks still cover the SMs; integer sums do not
            // depend on the tiling.
            let rows: usize = batch.iter().map(|&(_, seg, _)| seg.len).sum();
            let groups = if state.row_ptr.is_some() {
                1
            } else {
                (state.n_shared + state.n_global).max(1)
            };
            let tiles_wanted = (device.sm_count as usize * 4).div_ceil(groups);
            let tile_rows = rows
                .div_ceil(tiles_wanted)
                .clamp(MIN_HIST_TILE, HIST_TILE)
                .min(max_tile);
            let mut tiles = Vec::new();
            for (j, &(_, seg, _)) in batch.iter().enumerate() {
                for start in (0..seg.len).step_by(tile_rows) {
                    let count = tile_rows.min(seg.len - start);
                    tiles.extend([(seg.offset + start) as u64, count as u64 | (j as u64) << 32]);
                }
            }
            let total_bins = bins as u64;
            // Each node's accumulator and output slot (the resident form adds
            // the parent slot whose sibling the finalization subtracts: the
            // same `f64` subtraction as `subtract_hists`, fused per bin).
            let mut work = Vec::with_capacity(batch.len() * 3);
            for (j, &(_, _, slot)) in batch.iter().enumerate() {
                work.extend([j as u32, slot as u32]);
                if resident.is_some() {
                    let parent = by_built
                        .binary_search_by_key(&slot, |&(built, _)| built)
                        .map_or(u32::MAX, |at| {
                            let i = by_built[at].1;
                            fused[i] = true;
                            siblings[i].0
                        });
                    work.push(parent);
                }
            }
            let [d_tiles, d_work] = state.staging.upload_parts(
                pool,
                stream,
                &mut state.desc,
                [bytes_of(&tiles), bytes_of(&work)],
            )?;
            self.launch_tiles(state, source, d_tiles, tiles.len() / 2)?;
            let finalize = if resident.is_some() {
                &device.kernels.finalize_exact_sub
            } else {
                &device.kernels.finalize_exact
            };
            let mut launch = stream.launch_builder(finalize);
            launch
                .arg(&state.acc)
                .arg(&d_work)
                .arg(&total_bins)
                .arg(&grad_value)
                .arg(&hess_value)
                .arg(&mut state.out);
            // SAFETY: reads `batch.len()` accumulators and writes their
            // output slots (and distinct parent slots, validated by
            // `build_resident`), all within the sized buffers.
            unsafe { launch.launch(Device::per_node_bins(bins, batch.len())) }?;
        }

        // Strategy 2: exact chunks, the CPU's chunks, reduced in order.
        let chunk_list: Vec<(usize, usize, usize, usize)> = chunked
            .iter()
            .flat_map(|&(_, seg, grain, slot)| {
                (0..seg.len.div_ceil(grain)).map(move |c| {
                    let start = c * grain;
                    (seg.offset + start, grain.min(seg.len - start), slot, c)
                })
            })
            .collect();
        let wave_slots = if chunk_list.is_empty() {
            1
        } else {
            self.partial_wave(state, chunk_list.len())?
        };
        for wave in chunk_list.chunks(wave_slots) {
            let tiles: Vec<u64> = wave
                .iter()
                .enumerate()
                .flat_map(|(s, &(begin, count, _, _))| {
                    [begin as u64, count as u64 | ((s as u64) | 1 << 31) << 32]
                })
                .collect();
            if state.row_ptr.is_some() {
                stream.memset_zeros(&mut state.partials.slice_mut(..wave.len() * bins * 2))?;
            }
            // Consecutive chunks of one node reduce together.
            let mut ranges: Vec<u32> = Vec::new();
            let mut s = 0;
            while s < wave.len() {
                let (_, _, slot, c) = wave[s];
                let mut e = s + 1;
                while e < wave.len() && wave[e].2 == slot {
                    e += 1;
                }
                ranges.extend([s as u32, (e - s) as u32, slot as u32, u32::from(c == 0)]);
                s = e;
            }
            let [d_tiles, d_ranges] = state.staging.upload_parts(
                pool,
                stream,
                &mut state.desc,
                [bytes_of(&tiles), bytes_of(&ranges)],
            )?;
            self.launch_tiles(state, source, d_tiles, wave.len())?;
            let total_bins = bins as u64;
            for batch in 0..(ranges.len() / 4).div_ceil(MAX_GRID_Y) {
                let first = batch * MAX_GRID_Y;
                let count = (ranges.len() / 4 - first).min(MAX_GRID_Y);
                // This batch's ranges: four `u32`s each.
                let view = d_ranges + (first * 16) as u64;
                let mut launch = stream.launch_builder(&device.kernels.reduce_chunks);
                launch
                    .arg(&state.partials)
                    .arg(&view)
                    .arg(&total_bins)
                    .arg(&grad_value)
                    .arg(&hess_value)
                    .arg(&mut state.out);
                // SAFETY: each range reads partial slots of this wave and
                // writes its node's output slot.
                unsafe { launch.launch(Device::per_node_bins(bins, count)) }?;
            }
        }

        // Strategy 3: `f64` chains, one thread per (chunk, feature).
        for &(_, seg, seg_rows, slot) in &chains {
            let segs = seg.len.div_ceil(seg_rows);
            let wave_chunks = self.partial_wave(state, segs)?;
            for (w, first) in (0..segs).step_by(wave_chunks).enumerate() {
                let wave = wave_chunks.min(segs - first);
                let begin = seg.offset + first * seg_rows;
                let end = (seg.offset + (first + wave) * seg_rows).min(seg.offset + seg.len);
                stream.memset_zeros(&mut state.partials.slice_mut(..wave * bins * 2))?;
                let State {
                    bins: dev_bins,
                    row_ptr,
                    stride,
                    sentinel,
                    feature_first,
                    gpair: dev_gpair,
                    tree_rows,
                    upload: uploaded,
                    partials,
                    out,
                    ..
                } = &mut *state;
                let rows = match source {
                    RowSource::Tree => tree_rows.slice(begin..end),
                    RowSource::Upload(_) => uploaded.slice(begin..end),
                };
                let chunks = abi::Chunks {
                    n: (end - begin) as u64,
                    seg_rows: seg_rows as u64,
                    segs: wave as u64,
                    total_bins: bins as u64,
                };
                let chain = abi::ChainWork {
                    chunks,
                    stride: *stride,
                    n_cols: self.n_cols as u32,
                    sentinel: *sentinel,
                };
                let kernel = if row_ptr.is_some() {
                    &device.kernels.hist_sparse_chain[dev_bins.width()]
                } else {
                    &device.kernels.hist_chain[dev_bins.width()]
                };
                let mut launch = stream.launch_builder(kernel);
                dev_bins.push(&mut launch);
                match &*row_ptr {
                    Some(row_ptr) => launch.arg(row_ptr),
                    None => launch.arg(&*feature_first),
                };
                launch.arg(&rows).arg(&*dev_gpair).arg(&mut *partials);
                if row_ptr.is_some() {
                    launch.arg(&chunks);
                } else {
                    launch.arg(&chain);
                }
                let segs64 = chunks.segs;
                let total_bins = chunks.total_bins;
                let config = if row_ptr.is_some() {
                    device.grid(wave)
                } else {
                    Device::one_per(wave * self.n_cols)
                };
                // SAFETY: dense chains own (chunk,feature), CSR chains own
                // each whole chunk; all listed rows and partials fit.
                unsafe { launch.launch(config) }?;
                let init = i32::from(w == 0);
                let mut target = out.slice_mut(slot * bins * 2..(slot + 1) * bins * 2);
                let mut reduce = stream.launch_builder(&device.kernels.reduce_chains);
                reduce
                    .arg(&*partials)
                    .arg(&segs64)
                    .arg(&total_bins)
                    .arg(&init)
                    .arg(&mut target);
                // SAFETY: reads `wave` partials and writes one output slot.
                unsafe { reduce.launch(device.grid(bins)) }?;
            }
        }

        // Strategy 4 on the CPU, while the GPU works.
        let host_gpair = gpair.or(downloaded.as_deref());
        let threads = rayon::current_num_threads();
        for (&k, rows) in cpu.iter().zip(&cpu_rows) {
            let Some(gpair) = host_gpair else {
                return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
            };
            if let Dest::Into(out) = &mut dest {
                CpuBackend::build_locked(ghist, rows, gpair, out, threads);
                continue;
            }
            let mut hist = zeroed(bins);
            CpuBackend::build_locked(ghist, rows, gpair, &mut hist, threads);
            match targets {
                Some(t) => {
                    let s = t[k] as usize;
                    let flat: Vec<f64> = hist.iter().flat_map(|b| [b.grad, b.hess]).collect();
                    copy_host(
                        stream,
                        &flat,
                        &mut state.out.slice_mut(s * bins * 2..(s + 1) * bins * 2),
                    )?;
                }
                None => results[k] = Some(hist),
            }
        }
        if downloaded.is_some() {
            state.staged.host = downloaded;
        }
        if resident.is_some() {
            // Siblings of children built by the other strategies.
            let pairs: Vec<u32> = siblings
                .iter()
                .zip(&fused)
                .filter(|&(_, &done)| !done)
                .flat_map(|(&(parent, built), _)| [parent, built])
                .collect();
            if !pairs.is_empty() {
                let [d_pairs] = state.staging.upload_parts(
                    pool,
                    stream,
                    &mut state.desc,
                    [bytes_of(&pairs)],
                )?;
                let (n_pairs, total_bins) = ((pairs.len() / 2) as u64, bins as u64);
                let mut launch = stream.launch_builder(&device.kernels.subtract_hists);
                launch
                    .arg(&mut state.out)
                    .arg(&d_pairs)
                    .arg(&n_pairs)
                    .arg(&total_bins);
                // SAFETY: every pair names two distinct slots of the pool,
                // and each element writes only its own parent bin.
                unsafe { launch.launch(device.grid(pairs.len() / 2 * bins)) }?;
            }
            return Ok(Vec::new());
        }

        if !slots.is_empty() {
            if let Dest::Into(out) = dest {
                // SAFETY: `GradStats` is `repr(C)` of two `f64`s, so `out`
                // is `2 * len` contiguous `f64`s.
                let flat = unsafe {
                    std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len() * 2)
                };
                // Kernel faults surface at this synchronization point.
                state.ring.download(pool, stream, &state.out, flat)?;
                state.staging.synced();
                return Ok(Vec::new());
            }
            let mut all = vec![GradStats::default(); slots.len() * bins];
            // SAFETY: `GradStats` is `repr(C)` of two `f64`s, so `all` is
            // `2 * len` contiguous `f64`s.
            let flat = unsafe {
                std::slice::from_raw_parts_mut(all.as_mut_ptr().cast::<f64>(), all.len() * 2)
            };
            // Kernel faults surface at this synchronization point.
            state.ring.download(pool, stream, &state.out, flat)?;
            state.staging.synced();
            // Copy while holding state without Rayon work stealing.
            for (hist, &k) in all.chunks(bins).zip(&slots) {
                results[k] = Some(hist.to_vec());
            }
        }
        if matches!(dest, Dest::Into(_)) {
            return Ok(Vec::new());
        }
        Ok(results.into_iter().map(Option::unwrap_or_default).collect())
    }

    /// The device-computed gradients on the host: the staging's cached
    /// copy, taken (the caller puts it back), or downloaded.
    fn device_gradients(
        &self,
        state: &mut State,
    ) -> std::result::Result<Vec<GradPair>, DriverError> {
        if let Some(host) = state.staged.host.take() {
            return Ok(host);
        }
        if !state.staged.holds(None) {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        let mut host = vec![GradPair::default(); self.n_rows];
        // SAFETY: `GradPair` is `repr(C)` of two `f32`s, so `host` is
        // `2 * len` contiguous `f32`s, as the device's pairs are.
        let flat = unsafe {
            std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<f32>(), host.len() * 2)
        };
        let device = &*self.device;
        state
            .ring
            .download(&device.pinned, &device.stream, &state.gpair, flat)?;
        state.staging.synced();
        Ok(host)
    }

    fn partition_on(
        &self,
        state: &mut State,
        splits: &[RowSplit<'_>],
    ) -> std::result::Result<Vec<Partitioned>, DriverError> {
        let device = &*self.device;
        let stream = &device.stream;
        let PartitionHost {
            segs,
            rules,
            table,
            tiles: ptiles,
            split_tiles,
        } = &mut state.partition_host;
        segs.clear();
        rules.clear();
        table.clear();
        ptiles.clear();
        split_tiles.clear();
        for (s, split) in splits.iter().enumerate() {
            segs.extend([split.seg.offset as u64, split.seg.len as u64]);
            let (limit, table_at, kind) = match split.rule {
                RowRule::Below(limit) => (limit, 0, 0),
                RowRule::Table(left) => {
                    let at = u32::try_from(table.len())
                        .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
                    let end = table
                        .len()
                        .checked_add(left.len())
                        .filter(|&n| u32::try_from(n).is_ok())
                        .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
                    sized::<u8>(end)?;
                    table.extend(left.iter().map(|&l| u8::from(l)));
                    (0, at, 2)
                }
            };
            rules.extend([
                split.feature,
                limit,
                table_at,
                kind | u32::from(split.default_left),
            ]);
            let n = split.seg.len.div_ceil(PART_TILE);
            let tile_end = ptiles
                .len()
                .checked_add(n)
                .filter(|&n| u32::try_from(n).is_ok())
                .ok_or(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
            sized::<u64>(tile_end)?;
            split_tiles.extend([ptiles.len() as u32, n as u32]);
            ptiles.extend((0..n).map(|k| (s as u64) << 32 | k as u64));
        }
        let n_splits = splits.len() as u32;
        let n_tiles = ptiles.len();
        let pool = &device.pinned;
        // One queued copy of the partition's descriptors.
        let [d_segs, d_rules, d_table, d_split_tiles, d_ptiles] = state.staging.upload_parts(
            pool,
            stream,
            &mut state.desc,
            [
                bytes_of(segs.as_slice()),
                bytes_of(rules.as_slice()),
                bytes_of(table.as_slice()),
                bytes_of(split_tiles.as_slice()),
                bytes_of(ptiles.as_slice()),
            ],
        )?;
        // Each split's left length, then its children's run bits.
        fit(stream, &mut state.left_len, splits.len() * 2)?;
        let State {
            cols,
            bins,
            row_ptr,
            feature_first,
            flags,
            sentinel,
            tree_rows,
            scratch,
            tile_left,
            left_len,
            pin_counts,
            staging,
            ..
        } = state;
        let n_rows = self.n_rows as u64;
        if n_tiles > 0 {
            fit(stream, tile_left, n_tiles)?;
            let tiles_config = LaunchConfig {
                grid_dim: (n_tiles as u32, 1, 1),
                block_dim: (PART_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let route_bins = cols.as_ref().unwrap_or(&*bins);
            let kernel = if row_ptr.is_some() {
                &device.kernels.route_sparse[route_bins.width()]
            } else {
                &device.kernels.route_count[route_bins.width()]
            };
            let tiles = abi::PartTiles {
                segs: d_segs,
                ptiles: d_ptiles,
                rules: d_rules,
                tile_left: abi::ptr_mut(tile_left, stream),
            };
            let mut count = stream.launch_builder(kernel);
            route_bins.push(&mut count);
            if let Some(row_ptr) = &*row_ptr {
                count.arg(row_ptr).arg(&*feature_first);
            } else {
                count.arg(&n_rows).arg(&*sentinel);
            }
            count
                .arg(&d_table)
                .arg(&*tree_rows)
                .arg(&mut *flags)
                .arg(&tiles);
            // SAFETY: one block per tile; every segment lies in the tree's
            // valid rows, every row id is below `n_rows`, and each rule's
            // table range covers its feature's bins.
            unsafe { count.launch(tiles_config) }?;
        }
        let mut scan = stream.launch_builder(&device.kernels.route_scan);
        scan.arg(&d_split_tiles)
            .arg(&n_splits)
            .arg(&mut *tile_left)
            .arg(&mut *left_len);
        // SAFETY: one thread per split over its own tiles' counts.
        unsafe { scan.launch(Device::one_per(splits.len())) }?;
        if n_tiles > 0 {
            let tiles_config = LaunchConfig {
                grid_dim: (n_tiles as u32, 1, 1),
                block_dim: (PART_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut scatter = stream.launch_builder(&device.kernels.route_scatter);
            scatter
                .arg(&d_segs)
                .arg(&d_ptiles)
                .arg(&*tree_rows)
                .arg(&*flags)
                .arg(&*tile_left)
                .arg(&*left_len)
                .arg(&mut *scratch);
            // SAFETY: as for the count; each row lands inside its own
            // segment of `scratch` (`n_rows` long).
            unsafe { scatter.launch(tiles_config) }?;
            let mut copy = stream.launch_builder(&device.kernels.route_copy);
            copy.arg(&d_segs)
                .arg(&d_ptiles)
                .arg(&*scratch)
                .arg(&mut *tree_rows);
            // SAFETY: copies each tile's span within its segment.
            unsafe { copy.launch(tiles_config) }?;
        }
        let mut runs = stream.launch_builder(&device.kernels.route_runs);
        runs.arg(&d_segs)
            .arg(&*tree_rows)
            .arg(&n_splits)
            .arg(&mut *left_len);
        // SAFETY: one thread per split reads its own (partitioned) segment
        // and length and writes its own run word.
        unsafe { runs.launch(Device::one_per(splits.len())) }?;
        let counts = pinned(pool, stream, pin_counts, splits.len() * 2, false)?;
        let host = download_pinned(stream, counts, left_len, splits.len() * 2)?;
        staging.synced();
        let (lens, runs) = host.split_at(splits.len());
        Ok(splits
            .iter()
            .zip(lens.iter().zip(runs))
            .map(|(split, (&left_len, &runs))| {
                let left_len = left_len as usize;
                Partitioned {
                    left: Segment {
                        offset: split.seg.offset,
                        len: left_len,
                        contiguous: runs & 1 != 0,
                    },
                    right: Segment {
                        offset: split.seg.offset + left_len,
                        len: split.seg.len - left_len,
                        contiguous: runs & 2 != 0,
                    },
                }
            })
            .collect())
    }
}

/// Where [`CudaHistBackend::histograms_on`] puts the histograms.
enum Dest<'a> {
    /// Read back, one per node.
    Read,
    /// Read back into the single node's histogram.
    Into(&'a mut [GradStats]),
    /// Written to resident pool slots.
    Resident(Resident<'a>),
}

/// The longest row-order chain the GPU sums (one thread per feature walks
/// the node's rows): every node below the CPU's 8,192-row blocking
/// threshold. Longer chains, which only the CPU's feature sweeps of a
/// column-major index form, are summed on the CPU, as Metal and wgpu leave
/// the nodes they cannot reproduce there.
const MAX_CHAIN_ROWS: usize = 8191;

/// Rows per block of an exact root total (any grouping of exact sums gives
/// the chain's).
const ROOT_CHUNK: usize = 1 << 14;

/// The most rows one histogram tile may hold: `rows * features` of a
/// group must fit a `u32` (halved: the kernel's unrolled indices run up to
/// four block widths past the tile's last element).
fn max_tile(state: &State) -> usize {
    if state.row_ptr.is_some() {
        u32::MAX as usize
    } else {
        (u32::MAX as usize / 2 / state.group_features.max(1)).max(1)
    }
}

/// The strategy for the nonempty node `seg` of `ghist` (the
/// [module docs](self)' numbering) and the CPU's order for it
/// ([`sum_order`]), with tiles of at most `max_tile` rows.
fn plan(
    staged: &Staged,
    ghist: &GHistIndex,
    seg: Segment,
    max_tile: usize,
) -> (Strategy, SumOrder) {
    let order = sum_order(
        ghist.column_bins().is_some(),
        ghist.n_rows(),
        seg.len,
        seg.contiguous,
    );
    // NaN payloads are not portable between the CPU's and the GPU's
    // arithmetic, so a non-finite slice keeps the CPU's bits by running there.
    let strategy = if !staged.finite() {
        Strategy::Cpu
    } else if staged.sums_exact(seg.len) {
        Strategy::Exact
    } else {
        match order {
            SumOrder::Blocked { grain } if grain > max_tile => Strategy::Cpu,
            SumOrder::Blocked { grain } if staged.sums_exact(grain) => Strategy::ExactChunks,
            SumOrder::Blocked { .. } => Strategy::Chains,
            SumOrder::Chain if seg.len <= MAX_CHAIN_ROWS => Strategy::Chains,
            SumOrder::Chain => Strategy::Cpu,
        }
    };
    (strategy, order)
}

fn gpu_error(error: DriverError) -> HessboostError {
    HessboostError::gpu(format!("CUDA: {error}"))
}

impl HistogramBackend for CudaHistBackend {
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        let check = check_rows(rows, self.n_rows);
        let fits = self.fits(ghist)
            && out.len() == self.total_bins
            && rows.len() <= self.n_rows
            && check.inside;
        let node = Segment {
            offset: 0,
            len: rows.len(),
            contiguous: check.run,
        };
        let built = fits.then(|| self.lock()).flatten().and_then(|mut state| {
            let state = &mut *state;
            if !state.staged.holds(Some(gpair)) {
                self.ok(self.stage(state, gpair))?;
                if !state.staged.holds(Some(gpair)) {
                    return None;
                }
            }
            // A node the CPU sums is built below, outside the lock, on
            // every thread.
            if node.len > 0 && plan(&state.staged, ghist, node, max_tile(state)).0 == Strategy::Cpu
            {
                return None;
            }
            let placed = self.place_rows(&mut state.ring, &mut state.upload, rows, check.run);
            self.ok(placed)?;
            self.ok(self.histograms_on(
                state,
                RowSource::Upload(rows),
                ghist,
                Some(gpair),
                &[node],
                Dest::Into(&mut *out),
            ))
        });
        if built.is_none() {
            self.counters.count(Strategy::Cpu, rows.len());
            CpuBackend.build(ghist, rows, gpair, out);
        }
    }

    fn prepare(&self, _ghist: &GHistIndex, gpair: &[GradPair]) {
        if let Some(mut state) = self.lock() {
            // The trainer refills its gradient buffer in place every round,
            // so a new tree always restages.
            let staged = self.stage(&mut state, gpair);
            if self.ok(staged).is_none() {
                state.staged.len = 0;
            }
        }
    }

    fn row_engine(&self) -> Option<&dyn RowEngine> {
        (!self.device.failed.load(Ordering::Acquire)).then_some(self as &dyn RowEngine)
    }
}

impl RowEngine for CudaHistBackend {
    fn begin_tree(&self, ghist: &GHistIndex, rows: &[u32]) -> Option<Segment> {
        if !self.fits(ghist) || rows.len() > self.n_rows {
            return None;
        }
        // Repeated rows would give `add_leaf_values` two writes of one margin.
        let check = check_rows(rows, self.n_rows);
        if !check.inside || !check.ascending {
            return None;
        }
        let mut state = self.lock()?;
        if state.staged.len == 0 {
            return None;
        }
        let state = &mut *state;
        let placed = self.place_rows(&mut state.ring, &mut state.tree_rows, rows, check.run);
        self.ok(placed)?;
        state.tree_len = rows.len();
        Some(Segment {
            offset: 0,
            len: rows.len(),
            contiguous: check.run,
        })
    }

    fn root_total(
        &self,
        gpair: Option<&[GradPair]>,
        rows: &[u32],
        seg: Segment,
    ) -> Option<GradStats> {
        let mut state = self.lock()?;
        let state = &mut *state;
        if rows.len() != seg.len
            || seg.offset + seg.len > state.tree_len
            || !state.staged.holds(gpair)
        {
            return None;
        }
        if !state.staged.finite() || !state.staged.sums_exact(seg.len) {
            // `sum_rows`'s row-order chain, on the host.
            let downloaded = match gpair {
                Some(_) => None,
                None => Some(self.ok(self.device_gradients(state))?),
            };
            let pairs = gpair.or(downloaded.as_deref())?;
            let mut total = GradStats::default();
            for &r in rows {
                total.add(GradStats::from_pair(*pairs.get(r as usize)?));
            }
            if downloaded.is_some() {
                state.staged.host = downloaded;
            }
            return Some(total);
        }
        // Every partial sum is exact, so the chain's sum is the integer
        // total in any grouping: per-block totals on the device, added here.
        let device = &*self.device;
        let stream = &device.stream;
        let chunks = seg.len.div_ceil(ROOT_CHUNK).max(1);
        let units = (|| {
            fit(stream, &mut state.totals, chunks * 2)?;
            stream.memset_zeros(&mut state.totals.slice_mut(..chunks * 2))?;
            if seg.len > 0 {
                let rows = state.tree_rows.slice(seg.offset..seg.offset + seg.len);
                let (n, grain) = (seg.len as u64, ROOT_CHUNK as u64);
                let mut launch = stream.launch_builder(&device.kernels.chunk_totals);
                launch
                    .arg(&rows)
                    .arg(&n)
                    .arg(&grain)
                    .arg(&state.units)
                    .arg(&mut state.totals);
                let config = LaunchConfig {
                    grid_dim: (chunks as u32, 1, 1),
                    block_dim: (PART_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: one block per chunk reads its rows (below
                // `n_rows`) and writes its two totals.
                unsafe { launch.launch(config) }?;
            }
            let mut units = vec![0i64; chunks * 2];
            stream.memcpy_dtoh(&state.totals.slice(..chunks * 2), &mut units)?;
            stream.synchronize()?;
            state.staging.synced();
            Ok(units)
        })();
        let units = self.ok(units)?;
        // At most `2^53` grains in magnitude, every partial sum included.
        let (grad, hess) = units
            .chunks(2)
            .fold((0i64, 0i64), |(g, h), k| (g + k[0], h + k[1]));
        Some(GradStats::new(
            state.staged.grad.value(grad),
            state.staged.hess.value(hess),
        ))
    }

    fn partition(&self, ghist: &GHistIndex, splits: &[RowSplit<'_>]) -> Option<Vec<Partitioned>> {
        let mut state = self.lock()?;
        let cuts = ghist.cuts();
        let valid = self.fits(ghist)
            && splits.iter().all(|s| {
                if s.feature as usize >= self.n_cols {
                    return false;
                }
                let (fs, fe) = cuts.feature_bins(s.feature as usize);
                s.seg
                    .offset
                    .checked_add(s.seg.len)
                    .is_some_and(|end| end <= state.tree_len)
                    && match s.rule {
                        RowRule::Below(_) => true,
                        RowRule::Table(table) => table.len() == fe - fs,
                    }
            })
            // Overlapping splits would race on their rows' positions and
            // could repeat a row in the tree's list.
            && disjoint(splits.iter().map(|s| s.seg));
        if !valid || u32::try_from(splits.len()).is_err() {
            return None;
        }
        if splits.is_empty() {
            return Some(Vec::new());
        }
        let parts = self.partition_on(&mut state, splits);
        self.ok(parts)
    }

    fn histograms(
        &self,
        ghist: &GHistIndex,
        gpair: Option<&[GradPair]>,
        nodes: &[Segment],
    ) -> Option<Vec<Histogram>> {
        let mut state = self.lock()?;
        if !self.fits(ghist)
            || !state.staged.holds(gpair)
            || nodes.iter().any(|s| s.offset + s.len > state.tree_len)
        {
            return None;
        }
        if nodes.is_empty() {
            return Some(Vec::new());
        }
        let hists =
            self.histograms_on(&mut state, RowSource::Tree, ghist, gpair, nodes, Dest::Read);
        self.ok(hists)
    }

    fn rows(&self, segs: &[Segment]) -> Option<Vec<Vec<u32>>> {
        let mut state = self.lock()?;
        let end = segs.iter().map(|s| s.offset + s.len).max().unwrap_or(0);
        if end > state.tree_len {
            return None;
        }
        if end == 0 {
            // Every segment is empty: no download, so no wait, and the
            // arena's queued copies stay pending.
            return Some(vec![Vec::new(); segs.len()]);
        }
        let stream = &self.device.stream;
        let state = &mut *state;
        let mut all = vec![0u32; end];
        let read = state
            .ring
            .download(&self.device.pinned, stream, &state.tree_rows, &mut all);
        self.ok(read)?;
        // The download waited for the stream's work through it.
        state.staging.synced();
        Some(
            segs.iter()
                .map(|s| all[s.offset..s.offset + s.len].to_vec())
                .collect(),
        )
    }

    fn load_margins(&self, margins: &[f32]) -> Option<()> {
        let mut state = self.lock()?;
        if margins.len() != self.n_rows {
            return None;
        }
        let stream = &self.device.stream;
        let loaded = if let Some(buffer) = &mut state.margins {
            copy_host(stream, margins, buffer)
        } else {
            clone_host(stream, margins).map(|buffer| state.margins = Some(buffer))
        };
        self.ok(loaded)?;
        // A fresh margin run may reuse the same host allocation with new
        // labels or weights; addresses are only stable within a run.
        state.labels = None;
        state.weights = None;
        Some(())
    }

    fn gradients(&self, loss: DeviceLoss, labels: &[f32], weights: Option<&[f32]>) -> Option<bool> {
        let mut state = self.lock()?;
        let n = self.n_rows;
        if labels.len() != n || weights.is_some_and(|w| w.len() != n) || state.margins.is_none() {
            return None;
        }
        if let DeviceLoss::Logistic { split, .. } = loss
            && (split.lanes == 0 || split.rows > n || !split.rows.is_multiple_of(split.lanes))
        {
            return None;
        }
        let device = &*self.device;
        let stream = &device.stream;
        let state = &mut *state;
        let staged = (|| {
            state.staged.len = 0;
            // Labels and weights are copied once per run (keyed by the
            // host slice they came from).
            let key = |s: &[f32]| s.as_ptr().addr();
            if state.labels.as_ref().is_none_or(|(k, _)| *k != key(labels)) {
                state.labels = Some((key(labels), clone_host(stream, labels)?));
            }
            if let Some(weights) = weights
                && state
                    .weights
                    .as_ref()
                    .is_none_or(|(k, _)| *k != key(weights))
            {
                state.weights = Some((key(weights), clone_host(stream, weights)?));
            }
            let (Some(margins), Some((_, dev_labels))) = (&state.margins, &state.labels) else {
                return Ok(None);
            };
            let dev_weights = match (&state.weights, weights) {
                (Some((_, w)), Some(_)) => w,
                // Unread without weights; any valid pointer.
                _ => dev_labels,
            };
            let weighted = i32::from(weights.is_some());
            match loss {
                DeviceLoss::SquaredError { scale_pos_weight } => {
                    let rows = n as u64;
                    let mut launch = stream.launch_builder(&device.kernels.squared_error);
                    launch
                        .arg(margins)
                        .arg(dev_labels)
                        .arg(dev_weights)
                        .arg(&weighted)
                        .arg(&scale_pos_weight)
                        .arg(&rows)
                        .arg(&mut state.gpair);
                    // SAFETY: reads `n_rows` margins, labels and weights,
                    // and writes `n_rows` pairs.
                    unsafe { launch.launch(device.grid(n)) }?;
                }
                DeviceLoss::Logistic {
                    scale_pos_weight,
                    min_hess,
                    split,
                } => {
                    // The rows after the vectors (fewer than `lanes`, or a
                    // batch too short for vectors) run the host's scalar
                    // path, here on the host.
                    if split.rows < n {
                        let mut tail = vec![0f32; n - split.rows];
                        stream.memcpy_dtoh(&margins.slice(split.rows..n), &mut tail)?;
                        stream.synchronize()?;
                        state.staging.synced();
                        let mut pairs = vec![GradPair::default(); tail.len()];
                        crate::simd::logistic_gradient(
                            &tail,
                            &labels[split.rows..],
                            weights.map(|w| &w[split.rows..]),
                            scale_pos_weight,
                            min_hess,
                            &mut pairs,
                        );
                        let flat: Vec<f32> = pairs.iter().flat_map(|p| [p.grad, p.hess]).collect();
                        copy_host(
                            stream,
                            &flat,
                            &mut state.gpair.slice_mut(2 * split.rows..2 * n),
                        )?;
                        state.staging.synced();
                    }
                    if split.rows > 0 {
                        let params = abi::LogisticParams {
                            weighted,
                            scale_pos_weight,
                            min_hess,
                            max_input: split.max_input,
                            lanes: split.lanes as u32,
                            n: split.rows as u64,
                        };
                        let mut launch = stream.launch_builder(&device.kernels.logistic);
                        launch
                            .arg(margins)
                            .arg(dev_labels)
                            .arg(dev_weights)
                            .arg(&params)
                            .arg(&mut state.gpair);
                        // SAFETY: `split.rows <= n_rows` is a whole number
                        // of vectors, so every vector's margins are read
                        // inside the buffer; writes the first `split.rows`
                        // pairs.
                        unsafe { launch.launch(device.grid(split.rows)) }?;
                    }
                }
            }
            let (grad, hess) = self.device_domains(state)?;
            let finite = grad.is_finite() && hess.is_finite();
            self.stage_units(state, grad, hess)?;
            Ok(Some(finite))
        })();
        self.ok(staged)?
    }

    fn add_leaf_values(&self, leaves: &[(Segment, f32)]) -> Option<()> {
        let mut state = self.lock()?;
        if leaves
            .iter()
            .any(|(s, _)| s.offset + s.len > state.tree_len)
            || !disjoint(leaves.iter().map(|&(seg, _)| seg))
            || state.margins.is_none()
        {
            return None;
        }
        let device = &*self.device;
        let stream = &device.stream;
        let state = &mut *state;
        let mut segs = Vec::with_capacity(leaves.len() * 2);
        let mut values = Vec::with_capacity(leaves.len());
        let mut ptiles = Vec::new();
        for (k, (seg, value)) in leaves.iter().enumerate() {
            segs.extend([seg.offset as u64, seg.len as u64]);
            values.push(*value);
            ptiles.extend((0..seg.len.div_ceil(PART_TILE)).map(|t| (k as u64) << 32 | t as u64));
        }
        if ptiles.is_empty() {
            return Some(());
        }
        let added = (|| {
            let [d_segs, d_values, d_ptiles] = state.staging.upload_parts(
                &device.pinned,
                stream,
                &mut state.desc,
                [bytes_of(&segs), bytes_of(&values), bytes_of(&ptiles)],
            )?;
            let Some(margins) = state.margins.as_mut() else {
                return Ok(());
            };
            let mut launch = stream.launch_builder(&device.kernels.add_leaves);
            launch
                .arg(&d_segs)
                .arg(&d_values)
                .arg(&d_ptiles)
                .arg(&state.tree_rows)
                .arg(margins);
            let config = LaunchConfig {
                grid_dim: (ptiles.len() as u32, 1, 1),
                block_dim: (PART_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: one block per tile of a leaf segment inside the tree's
            // rows; each row (below `n_rows`) is in exactly one leaf: the
            // tree's rows are distinct (`begin_tree`), partitions permute
            // them within segments, and the leaves are `disjoint`.
            unsafe { launch.launch(config) }.map(|_| ())
        })();
        self.ok(added)
    }

    fn read_margins(&self, out: &mut [f32]) -> Option<()> {
        let state = self.lock()?;
        let margins = state.margins.as_ref()?;
        if out.len() != self.n_rows {
            return None;
        }
        let stream = &self.device.stream;
        let read = stream
            .memcpy_dtoh(margins, out)
            .and_then(|()| stream.synchronize());
        self.ok(read)
    }

    fn reserve_hists(&self, ghist: &GHistIndex, slots: usize) -> Option<bool> {
        let mut state = self.lock()?;
        if !self.fits(ghist) {
            return None;
        }
        if state.pool_slots >= slots {
            return Some(true);
        }
        let words = slots.checked_mul(self.total_bins.checked_mul(2)?)?;
        let stream = &self.device.stream;
        let state = &mut *state;
        let reserved = (|| {
            stream.context().bind_to_thread()?;
            let (free, _) = cudarc::driver::result::mem_get_info()?;
            // Half of what is free (plus the pool being replaced), so the
            // per-level buffers keep room to grow.
            let held = state.pool.len() * 8;
            if words.saturating_mul(8) > free / 2 + held {
                return Ok(false);
            }
            state.pool_slots = 0;
            state.pool = stream.alloc_zeros(2)?;
            state.pool = stream.alloc_zeros(words)?;
            state.pool_slots = slots;
            Ok(true)
        })();
        self.ok(reserved)
    }

    fn build_resident(
        &self,
        ghist: &GHistIndex,
        gpair: Option<&[GradPair]>,
        nodes: &[(Segment, HistSlot)],
        siblings: &[(HistSlot, HistSlot)],
    ) -> Option<()> {
        let mut state = self.lock()?;
        let slots = state.pool_slots;
        let in_pool = |slot: HistSlot| (slot as usize) < slots;
        if !self.fits(ghist)
            || !state.staged.holds(gpair)
            || nodes
                .iter()
                .any(|&(s, slot)| s.offset + s.len > state.tree_len || !in_pool(slot))
            || siblings
                .iter()
                .any(|&(parent, built)| !in_pool(parent) || !in_pool(built))
            || !race_free(nodes, siblings)
        {
            return None;
        }
        let segs: Vec<Segment> = nodes.iter().map(|&(seg, _)| seg).collect();
        let targets: Vec<HistSlot> = nodes.iter().map(|&(_, slot)| slot).collect();
        let state = &mut *state;
        // The build writes `state.out`: make it the pool for this call.
        std::mem::swap(&mut state.out, &mut state.pool);
        let resident = Resident {
            targets: &targets,
            siblings,
        };
        let built = self.histograms_on(
            state,
            RowSource::Tree,
            ghist,
            gpair,
            &segs,
            Dest::Resident(resident),
        );
        std::mem::swap(&mut state.out, &mut state.pool);
        self.ok(built).map(|_| ())
    }

    fn scan_resident(
        &self,
        ghist: &GHistIndex,
        reg: &RegParams,
        requests: &[ScanRequest<'_>],
    ) -> Option<Vec<NodeScan>> {
        self.scan_on(ghist, reg, requests)
    }

    fn read_hist(&self, slot: HistSlot) -> Option<Histogram> {
        let state = self.lock()?;
        let s = slot as usize;
        if s >= state.pool_slots {
            return None;
        }
        let bins = self.total_bins;
        let mut flat = vec![0f64; bins * 2];
        let stream = &self.device.stream;
        let read = stream
            .memcpy_dtoh(
                &state.pool.slice(s * bins * 2..(s + 1) * bins * 2),
                &mut flat,
            )
            .and_then(|()| stream.synchronize());
        self.ok(read)?;
        Some(
            flat.as_chunks::<2>()
                .0
                .iter()
                .map(|&[grad, hess]| GradStats::new(grad, hess))
                .collect(),
        )
    }
}

/// Whether CUDA device 0 trains, for device tests: a machine without one
/// skips them (unless `HESSBOOST_REQUIRE_CUDA` is set); any other reason,
/// such as a kernel module that does not load, fails the test.
#[cfg(test)]
pub(crate) fn has_device() -> bool {
    match unavailable_reason(0) {
        None => true,
        Some(reason) => {
            assert!(
                reason.is_environment() && std::env::var_os("HESSBOOST_REQUIRE_CUDA").is_none(),
                "CUDA unavailable: {reason}"
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DMatrix;
    use crate::data::quantile::HistCuts;

    fn backend() -> Option<(GHistIndex, CudaHistBackend)> {
        if !has_device() {
            return None;
        }
        let values: Vec<f32> = (0..20_000).map(|r| (r % 5) as f32).collect();
        let data = DMatrix::from_dense(&values, values.len(), 1).unwrap();
        let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256));
        let backend = CudaHistBackend::new(&index, 0).unwrap();
        Some((index, backend))
    }

    fn bits(hist: &[GradStats]) -> Vec<[u64; 2]> {
        hist.iter()
            .map(|p| [p.grad.to_bits(), p.hess.to_bits()])
            .collect()
    }

    /// A resident build, partition or leaf update that would race on the
    /// device is refused before any launch: a slot written twice, a sibling
    /// reading a slot another sibling writes, a built slot subtracted from
    /// two parents, overlapping segments, repeated tree rows.
    #[test]
    fn racing_device_writes_are_refused() {
        let seg = |offset, len| Segment {
            offset,
            len,
            contiguous: false,
        };
        let nodes = [(seg(0, 4), 0), (seg(4, 4), 1)];
        // Built children subtracted from their parents' slots.
        assert!(race_free(&nodes, &[(2, 0), (3, 1)]));
        // Two nodes into one slot.
        assert!(!race_free(&[(seg(0, 4), 0), (seg(4, 4), 0)], &[]));
        // A parent slot that is also a node's target.
        assert!(!race_free(&nodes, &[(1, 0)]));
        // Two siblings into one parent slot.
        assert!(!race_free(&nodes, &[(2, 0), (2, 1)]));
        // A sibling reading a slot another sibling writes.
        assert!(!race_free(&nodes, &[(2, 0), (3, 2)]));
        assert!(!race_free(&nodes, &[(2, 2)]));
        // One built slot subtracted from two parents.
        assert!(!race_free(&nodes, &[(2, 0), (3, 0)]));
        assert!(disjoint(
            [seg(0, 4), seg(4, 4), seg(8, 0), seg(8, 2)].into_iter()
        ));
        assert!(!disjoint([seg(0, 4), seg(3, 2)].into_iter()));
        assert!(!disjoint([seg(4, 4), seg(0, 5)].into_iter()));
        let n = 3 * CHECK_CHUNK;
        let all: Vec<u32> = (0..n as u32).collect();
        let odd: Vec<u32> = (1..n as u32).step_by(2).collect();
        let check = |inside, ascending, run| RowCheck {
            inside,
            ascending,
            run,
        };
        assert_eq!(check_rows(&all, n), check(true, true, true));
        assert_eq!(check_rows(&odd, n), check(true, true, false));
        assert_eq!(check_rows(&[0, 0], n), check(true, false, false));
        assert_eq!(check_rows(&[2, 1], n), check(true, false, false));
        assert_eq!(check_rows(&[n as u32], n), check(false, true, true));
        // A repeat across the pass's chunk boundary.
        let mut seam = all.clone();
        seam[CHECK_CHUNK] = seam[CHECK_CHUNK - 1];
        assert_eq!(check_rows(&seam, n), check(true, false, false));
    }

    #[test]
    fn repeated_prepare_and_cross_thread_use_keep_latest_gradients() {
        let Some((index, backend)) = backend() else {
            return;
        };
        let first = vec![GradPair::new(1.0, 1.0); index.n_rows()];
        let second: Vec<_> = (0..index.n_rows())
            .map(|r| GradPair::new((r % 7) as f32 - 3.0, 2.0))
            .collect();
        backend.prepare(&index, &first);
        backend.prepare(&index, &second);
        let rows: Vec<u32> = (0..index.n_rows() as u32).collect();
        let mut expected = zeroed(index.total_bins());
        CpuBackend.build(&index, &rows, &second, &mut expected);
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    backend.prepare(&index, &first);
                    backend.prepare(&index, &second);
                    let mut actual = zeroed(index.total_bins());
                    backend.build(&index, &rows, &second, &mut actual);
                    assert_eq!(bits(&actual), bits(&expected));
                    assert_eq!(backend.node_counts().cpu_nodes, 0);
                })
                .join()
                .unwrap();
        });
        // Drop with the last pinned upload still in flight.
        backend.prepare(&index, &first);
        drop(backend);
        assert!(available(0), "{:?}", unavailable_reason(0));
    }

    #[test]
    fn nonrepresentable_domains_skip_integer_conversion_and_keep_chain_bits() {
        let Some((index, backend)) = backend() else {
            return;
        };
        let stream = &backend.device.stream;
        let rows: Vec<u32> = (1..index.n_rows() as u32).collect();
        for value in [f32::NAN, f32::INFINITY, 2.0f32.powi(-100)] {
            let mut pairs = vec![GradPair::new(1.0, 1.0); index.n_rows()];
            pairs[0].grad = value;
            {
                let mut state = backend.lock().unwrap();
                stream.memset_zeros(&mut state.units).unwrap();
                stream.synchronize().unwrap();
            }
            backend.prepare(&index, &pairs);
            {
                let state = backend.lock().unwrap();
                assert!(!state.staged.grad.sums_exact(1));
                let units = stream.clone_dtoh(&state.units).unwrap();
                stream.synchronize().unwrap();
                assert!(units.iter().all(|&v| v == 0), "unsafe integer staging ran");
            }
            let mut actual = zeroed(index.total_bins());
            let mut expected = zeroed(index.total_bins());
            backend.build(&index, &rows, &pairs, &mut actual);
            CpuBackend.build(&index, &rows, &pairs, &mut expected);
            assert_eq!(bits(&actual), bits(&expected));
        }
        assert!(available(0), "{:?}", unavailable_reason(0));
    }

    #[test]
    fn same_rayon_pool_concurrent_backend_calls_do_not_reenter_state_lock() {
        let Some((index, backend)) = backend() else {
            return;
        };
        let rows: Vec<u32> = (1..index.n_rows() as u32).collect();
        let mut pairs = vec![GradPair::new(1.0, 1.0); index.n_rows()];
        pairs[0].grad = f32::NAN; // excluded from rows, but forces CPU fallback
        backend.prepare(&index, &pairs);
        let mut expected = zeroed(index.total_bins());
        CpuBackend.build(&index, &rows, &pairs, &mut expected);
        let expected = bits(&expected);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            (0..16).into_par_iter().for_each(|_| {
                backend.prepare(&index, &pairs);
                let mut actual = zeroed(index.total_bins());
                backend.build(&index, &rows, &pairs, &mut actual);
                assert_eq!(bits(&actual), expected);
            });
        });
        assert_eq!(backend.node_counts().cpu_nodes, 16);
        assert!(available(0), "{:?}", unavailable_reason(0));
    }

    #[test]
    fn resident_nonfinite_fallback_waits_for_downloaded_rows() {
        let Some((index, backend)) = backend() else {
            return;
        };
        let mut pairs: Vec<_> = (0..index.n_rows())
            .map(|r| GradPair::new((r % 7) as f32 - 3.0, 1.0))
            .collect();
        pairs[0].grad = f32::NAN;
        let rows: Vec<u32> = (1..index.n_rows() as u32).step_by(2).collect();
        backend.prepare(&index, &pairs);
        let root = backend.begin_tree(&index, &rows).unwrap();
        let split = RowSplit {
            seg: root,
            feature: 0,
            rule: RowRule::Below(2),
            default_left: false,
        };
        let children = backend.partition(&index, &[split]).unwrap()[0];
        let segments = [root, children.left, children.right];
        let hists = backend.histograms(&index, Some(&pairs), &segments).unwrap();
        let actual_rows = backend.rows(&segments).unwrap();
        let (fs, fe) = index.cuts().feature_bins(0);
        let wanted_left: Vec<u32> = rows
            .iter()
            .copied()
            .filter(|&r| index.feature_bin(r as usize, fs, fe).unwrap() < 2)
            .collect();
        let wanted_right: Vec<u32> = rows
            .iter()
            .copied()
            .filter(|&r| index.feature_bin(r as usize, fs, fe).unwrap() >= 2)
            .collect();
        assert_eq!(actual_rows[1], wanted_left);
        assert_eq!(actual_rows[2], wanted_right);
        for (hist, rows) in hists.iter().zip(&actual_rows) {
            let mut expected = zeroed(index.total_bins());
            CpuBackend.build(&index, rows, &pairs, &mut expected);
            assert_eq!(bits(hist), bits(&expected));
        }
        assert!(available(0), "{:?}", unavailable_reason(0));
    }

    #[test]
    fn fresh_margin_run_refreshes_labels_and_weights_in_place() {
        let Some((index, backend)) = backend() else {
            return;
        };
        let margins = vec![0.0; index.n_rows()];
        let mut labels = vec![1.0; index.n_rows()];
        let mut weights = vec![1.0; index.n_rows()];
        let loss = DeviceLoss::SquaredError {
            scale_pos_weight: 1.0,
        };
        backend.load_margins(&margins).unwrap();
        assert_eq!(backend.gradients(loss, &labels, Some(&weights)), Some(true));
        labels.fill(3.0);
        weights.fill(2.0);
        backend.load_margins(&margins).unwrap();
        assert_eq!(backend.gradients(loss, &labels, Some(&weights)), Some(true));
        let rows: Vec<u32> = (0..index.n_rows() as u32).collect();
        let root = backend.begin_tree(&index, &rows).unwrap();
        let total = backend.root_total(None, &rows, root).unwrap();
        assert_eq!(
            total,
            GradStats::new(-6.0 * index.n_rows() as f64, 2.0 * index.n_rows() as f64)
        );
        assert!(available(0), "{:?}", unavailable_reason(0));
    }

    #[test]
    fn csr_storage_and_all_histogram_orders_match_cpu() {
        if !has_device() {
            return;
        }
        let n = 40_003;
        let mut offsets = vec![0];
        let mut features = Vec::new();
        let mut values = Vec::new();
        for r in 0..n {
            if r % 11 != 0 {
                // Deliberately unsorted columns; zero is a present value.
                features.extend([5, 1]);
                values.extend([(r % 3) as f32, (r % 7) as f32]);
            }
            offsets.push(features.len());
        }
        let data = DMatrix::from_csr(offsets, features, values, 8).unwrap();
        let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256));
        let backend = CudaHistBackend::new(&index, 0).unwrap();
        {
            let state = backend.lock().unwrap();
            assert!(state.cols.is_none());
            assert_eq!(state.row_ptr.as_ref().unwrap().len(), n + 1);
            assert_eq!(state.stride, 0);
            let nnz = index.row_ptr()[n];
            assert!(matches!(&state.bins, DeviceBins::U16(bins) if bins.len() == nnz));
        }
        let all: Vec<u32> = (0..n as u32).collect();
        let subsets = [&all[..], &all[3..103]];
        for mode in 0..3 {
            let mut pairs: Vec<_> = (0..n)
                .map(|r| GradPair::new((r % 3) as f32 - 1.0, 1.0))
                .collect();
            if mode == 1 {
                pairs[17].grad = 2f32.powi(-38);
            }
            if mode == 2 {
                pairs[17].grad = 2f32.powi(-100);
            }
            backend.prepare(&index, &pairs);
            for rows in subsets {
                let mut expected = zeroed(index.total_bins());
                let mut actual = zeroed(index.total_bins());
                CpuBackend.build(&index, rows, &pairs, &mut expected);
                backend.build(&index, rows, &pairs, &mut actual);
                assert_eq!(bits(&actual), bits(&expected));
            }
        }
        assert_eq!(backend.node_counts().cpu_nodes, 0);
        assert!(backend.node_counts().exact_nodes > 0);
        assert!(backend.node_counts().exact_chunk_nodes > 0);
        assert!(backend.node_counts().chain_nodes > 0);
        let root = backend.begin_tree(&index, &all).unwrap();
        for table_rule in [false, true] {
            let (fs, fe) = index.cuts().feature_bins(1);
            let table: Vec<bool> = (0..fe - fs).map(|b| b % 2 == 0).collect();
            for default_left in [false, true] {
                backend.begin_tree(&index, &all).unwrap();
                let rule = if table_rule {
                    RowRule::Table(&table)
                } else {
                    RowRule::Below(2)
                };
                let split = RowSplit {
                    seg: root,
                    feature: 1,
                    rule,
                    default_left,
                };
                let children = backend.partition(&index, &[split]).unwrap()[0];
                let wanted = |left: bool| {
                    all.iter()
                        .copied()
                        .filter(|&r| {
                            let go_left =
                                index
                                    .feature_bin(r as usize, fs, fe)
                                    .map_or(default_left, |b| {
                                        let local = (b as usize) - fs;
                                        if table_rule { table[local] } else { local < 2 }
                                    });
                            go_left == left
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    backend.rows(&[children.left, children.right]).unwrap(),
                    vec![wanted(true), wanted(false)]
                );
            }
        }
        assert!(available(0), "{:?}", unavailable_reason(0));
    }

    #[test]
    fn wide_global_csr_bins_keep_sparse_device_storage() {
        if !has_device() {
            return;
        }
        // 70,000 feature bins force u32 globally; rows contain one entry
        // each. Dense expansion would exceed a gigabyte for this tiny CSR.
        let n = 70_003;
        let offsets: Vec<usize> = (0..=n).collect();
        let features: Vec<u32> = (0..n).map(|r| (r % 70_000) as u32).collect();
        let data = DMatrix::from_csr(offsets, features, vec![0.0; n], 70_000).unwrap();
        let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 2));
        let backend = CudaHistBackend::new(&index, 0).unwrap();
        {
            let state = backend.lock().unwrap();
            assert!(state.cols.is_none());
            assert_eq!(state.row_ptr.as_ref().unwrap().len(), n + 1);
            assert!(matches!(&state.bins, DeviceBins::U32(bins) if bins.len() == n));
        }
        let rows: Vec<u32> = (0..n as u32).collect();
        let pairs = vec![GradPair::new(1.0, 1.0); n];
        backend.prepare(&index, &pairs);
        let mut expected = zeroed(index.total_bins());
        let mut actual = zeroed(index.total_bins());
        CpuBackend.build(&index, &rows, &pairs, &mut expected);
        backend.build(&index, &rows, &pairs, &mut actual);
        assert_eq!(bits(&actual), bits(&expected));
        assert_eq!(backend.node_counts().cpu_nodes, 0);
        assert!(available(0), "{:?}", unavailable_reason(0));
    }

    /// `bins` widened to `u32`. `clone_dtoh` into a `Vec` is a pageable
    /// asynchronous copy: its bytes are only there once the stream has
    /// synchronized.
    fn download<T: DeviceRepr + Into<u32>>(
        stream: &Arc<CudaStream>,
        bins: &CudaSlice<T>,
    ) -> Vec<u32> {
        let host = stream.clone_dtoh(bins).unwrap();
        stream.synchronize().unwrap();
        host.into_iter().map(Into::into).collect()
    }

    /// Dense device bins take the widest feature's width (`u8`, `u16`,
    /// `u32`), encoded row-major and feature-major as the CPU bins them,
    /// and every width's histogram kernels give the CPU's histogram: exact
    /// integer sums over all rows, `f64` chains over a small subset.
    #[test]
    fn dense_bins_match_cpu_at_every_width() {
        if !has_device() {
            return;
        }
        for (n, expected_width) in [(255, 0), (513, 1), (65_537, 2)] {
            let values: Vec<f32> = (0..n * 3).map(|i| (i / 3) as f32).collect();
            let data = DMatrix::from_dense(&values, n, 3)
                .unwrap()
                .with_feature_types(&[crate::data::FeatureType::Categorical; 3])
                .unwrap();
            let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256));
            let backend = CudaHistBackend::new(&index, 0).unwrap();
            {
                let state = backend.lock().unwrap();
                assert_eq!(state.bins.width(), expected_width);
                assert!(state.row_ptr.is_none());
                assert!(state.cols.is_some());
                let stream = &backend.device.stream;
                let read = |bins: &DeviceBins| match bins {
                    DeviceBins::U8(bins) => download(stream, bins),
                    DeviceBins::U16(bins) => download(stream, bins),
                    DeviceBins::U32(bins) => download(stream, bins),
                };
                let rows = read(&state.bins);
                let cols = read(state.cols.as_ref().unwrap());
                for r in 0..n {
                    for f in 0..3 {
                        let (fs, fe) = index.cuts().feature_bins(f);
                        let local = index.feature_bin(r, fs, fe).unwrap() - fs as u32;
                        assert_eq!(rows[r * state.stride as usize + f], local);
                        assert_eq!(cols[f * n + r], local);
                    }
                }
            }
            let all: Vec<u32> = (0..n as u32).collect();
            let some: Vec<u32> = (0..n as u32).step_by(n.div_ceil(4_000)).collect();
            let exact: Vec<GradPair> = (0..n)
                .map(|r| GradPair::new((r % 7) as f32 - 3.0, 1.0))
                .collect();
            // Magnitudes from 1e-30 to 1e30, far outside the exact domain.
            let wide: Vec<GradPair> = (0..n)
                .map(|r| GradPair::new(10f32.powi((r % 61) as i32 - 30), 1.0))
                .collect();
            for (rows, gpair) in [(&all, &exact), (&some, &wide)] {
                let mut expected = zeroed(index.total_bins());
                let mut actual = zeroed(index.total_bins());
                CpuBackend.build(&index, rows, gpair, &mut expected);
                backend.prepare(&index, gpair);
                backend.build(&index, rows, gpair, &mut actual);
                assert_eq!(bits(&actual), bits(&expected), "{n} rows");
            }
            let counts = backend.node_counts();
            assert_eq!(counts.cpu_nodes, 0, "{counts:?}");
            assert!(
                counts.exact_nodes > 0 && counts.chain_nodes > 0,
                "{counts:?}"
            );
        }
        assert!(available(0), "{:?}", unavailable_reason(0));
    }

    #[test]
    fn empty_csr_entries_keep_only_offsets_and_zero_histograms() {
        if !has_device() {
            return;
        }
        let n = 257;
        let data = DMatrix::from_csr(vec![0; n + 1], vec![], vec![], 5_000).unwrap();
        let index = GHistIndex::from_dmatrix(&data, HistCuts::from_dmatrix(&data, 256));
        let backend = CudaHistBackend::new(&index, 0).unwrap();
        {
            let state = backend.lock().unwrap();
            assert!(state.cols.is_none());
            assert_eq!(state.row_ptr.as_ref().unwrap().len(), n + 1);
            assert!(matches!(&state.bins, DeviceBins::U16(bins) if bins.len() == 1));
        }
        let rows: Vec<u32> = (0..n as u32).collect();
        let pairs = vec![GradPair::new(1.0, 1.0); n];
        backend.prepare(&index, &pairs);
        let mut actual = zeroed(index.total_bins());
        backend.build(&index, &rows, &pairs, &mut actual);
        assert_eq!(bits(&actual), bits(&zeroed(index.total_bins())));
        let root = backend.begin_tree(&index, &rows).unwrap();
        for default_left in [false, true] {
            backend.begin_tree(&index, &rows).unwrap();
            let parts = backend
                .partition(
                    &index,
                    &[RowSplit {
                        seg: root,
                        feature: 4_999,
                        rule: RowRule::Below(0),
                        default_left,
                    }],
                )
                .unwrap()[0];
            assert_eq!(parts.left.len, if default_left { n } else { 0 });
        }
        assert_eq!(backend.node_counts().cpu_nodes, 0);
        assert!(available(0), "{:?}", unavailable_reason(0));
    }
}
