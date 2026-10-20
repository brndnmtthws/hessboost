//! Resident compact forests and a bounded, pipelined prediction.
//!
//! Prediction loads its own module into the device's primary context (the
//! one training uses) and runs on streams of its own. Each concurrent call
//! owns its streams and staging buffers; only the immutable forest is
//! shared. Numeric scalar forests use the shared 8-byte arena when it fits,
//! otherwise the 16-byte arena handles categorical splits, multiclass and
//! vector leaves. Leaf weighting happens once on the CPU in f32; the GPU
//! adds in tree order without FMA or FTZ.

use super::driver::{self, LaunchConfig, Memory, StreamExt};
use super::{
    Pinned, Plain, Unavailable, abi, failed, find_device, kernels, memcpy_dtoh_async,
    memcpy_htod_async, upload_pinned,
};
use crate::backend::shared::{
    MarginPlan, ensure_forest_model, materialize_rows, plan_margins, weighted_leaves,
};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::{BoostedModel, Iterations, Predictions};
use cuda_core::{CudaContext, CudaEvent, CudaFunction, CudaStream, DeviceBuffer};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

const BLOCK_ROWS: usize = 16_384;
/// Each slot's feature rows plus margins fit in 32 MiB on the device.
const SLOT_ENTRIES: usize = (32 << 20) / 4;
/// Retain at most two completed calls, never an unbounded buffer cache.
const POOL_CALLS: usize = 2;

struct Context {
    cuda: Arc<CudaContext>,
    predict8: CudaFunction,
    predict16: CudaFunction,
    name: String,
    failed: AtomicBool,
}

type Opened = std::result::Result<Arc<Context>, Unavailable>;
static CONTEXTS: Mutex<Vec<(usize, Opened)>> = Mutex::new(Vec::new());

impl Context {
    fn open(ordinal: usize) -> std::result::Result<Self, Unavailable> {
        find_device(ordinal)?;
        let cuda = CudaContext::new(ordinal).map_err(|e| failed("prediction context", e))?;
        let module = kernels::load(&cuda, kernels::Module::Prediction)?;
        let predict8 = module
            .load_function("predict8")
            .map_err(|e| failed("predict8", e))?;
        let predict16 = module
            .load_function("predict16")
            .map_err(|e| failed("predict16", e))?;
        let name = cuda.device_name().map_err(|e| failed("device name", e))?;
        Ok(Self {
            cuda,
            predict8,
            predict16,
            name,
            failed: AtomicBool::new(false),
        })
    }

    fn get(ordinal: usize) -> Opened {
        let mut contexts = CONTEXTS.lock();
        if let Some((_, opened)) = contexts.iter().find(|(o, _)| *o == ordinal) {
            return opened.clone();
        }
        let opened = Self::open(ordinal).map(Arc::new);
        contexts.push((ordinal, opened.clone()));
        opened
    }

    fn check(&self) -> Result<()> {
        if self.failed.load(Ordering::Acquire) {
            return Err(HessboostError::gpu(
                "CUDA prediction disabled after a runtime error",
            ));
        }
        Ok(())
    }

    /// `error` as the call's error; one that leaves the context unusable
    /// also disables prediction for every later call.
    fn error(&self, error: driver::Error) -> HessboostError {
        if error.poisons() {
            self.failed.store(true, Ordering::Release);
        }
        HessboostError::gpu(format!("CUDA prediction: {error}"))
    }
}

/// Whether CUDA device `ordinal` can predict, and has not had a runtime
/// error. Does not initialize the training context.
#[must_use]
pub fn prediction_available(ordinal: usize) -> bool {
    prediction_unavailable_reason(ordinal).is_none()
}

/// Why CUDA device `ordinal` cannot predict, if it cannot. Does not
/// initialize the training context.
#[must_use]
pub fn prediction_unavailable_reason(ordinal: usize) -> Option<Unavailable> {
    match Context::get(ordinal) {
        Ok(ctx) if ctx.failed.load(Ordering::Acquire) => Some(Unavailable::Disabled { ordinal }),
        Ok(_) => None,
        Err(reason) => Some(reason),
    }
}

/// The name of CUDA device `ordinal`, if it can predict.
#[must_use]
pub fn prediction_device_name(ordinal: usize) -> Option<String> {
    Context::get(ordinal)
        .ok()
        .filter(|ctx| !ctx.failed.load(Ordering::Acquire))
        .map(|ctx| ctx.name.clone())
}

enum Forest {
    Narrow {
        nodes: DeviceBuffer<u32>,
        roots: DeviceBuffer<u32>,
    },
    Wide {
        nodes: DeviceBuffer<u32>,
        categories: DeviceBuffer<u32>,
        vectors: DeviceBuffer<f32>,
        trees: DeviceBuffer<u32>,
    },
}

#[derive(Clone, Copy)]
struct Shape {
    rows: usize,
    cols: usize,
    outputs: usize,
}

impl Shape {
    fn serves(self, other: Self) -> bool {
        self.cols == other.cols && self.outputs == other.outputs && self.rows >= other.rows
    }
}

/// The entry of a full pool a finished call of shape `call` replaces: the
/// smallest pooled call that cannot serve it, else none (every pooled call
/// serves it, and it is dropped). Judged by the call's own shape, not its
/// request's, so concurrent misses of one shape all end up pooled and a
/// larger call back from a smaller request is kept.
fn replaced(pooled: impl Iterator<Item = Shape>, call: Shape) -> Option<usize> {
    pooled
        .enumerate()
        .filter(|&(_, shape)| !shape.serves(call))
        .min_by_key(|&(_, shape)| shape.rows)
        .map(|(i, _)| i)
}

struct Slot {
    rows: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    host_rows: Pinned<f32>,
    host_base: Pinned<f32>,
    host_out: Pinned<f32>,
    uploaded: CudaEvent,
    computed: CudaEvent,
    downloaded: CudaEvent,
    /// Row range whose D2H is in flight; never overwrite a live slot.
    pending: Option<(usize, usize)>,
}

/// Row blocks in flight per call: one being materialized, one uploading,
/// one computing and one downloading, so a block's host copy waits only
/// for the slot freed four blocks earlier.
const SLOTS: usize = 4;

struct Call {
    shape: Shape,
    upload: Arc<CudaStream>,
    compute: Arc<CudaStream>,
    download: Arc<CudaStream>,
    slots: Vec<Slot>,
}

impl Call {
    fn new(ctx: &Context, shape: Shape) -> driver::Result<Self> {
        let upload = ctx.cuda.new_stream()?;
        let compute = ctx.cuda.new_stream()?;
        let download = ctx.cuda.new_stream()?;
        let make_slot = || -> driver::Result<Slot> {
            Ok(Slot {
                rows: upload.alloc_zeros(shape.rows * shape.cols)?,
                out: upload.alloc_zeros(shape.rows * shape.outputs)?,
                host_rows: Pinned::new(&upload, shape.rows * shape.cols)?,
                host_base: Pinned::new(&upload, shape.rows * shape.outputs)?,
                host_out: Pinned::new(&download, shape.rows * shape.outputs)?,
                uploaded: ctx.cuda.new_event(None)?,
                computed: ctx.cuda.new_event(None)?,
                downloaded: ctx.cuda.new_event(None)?,
                pending: None,
            })
        };
        let slots = (0..SLOTS)
            .map(|_| make_slot())
            .collect::<driver::Result<_>>()?;
        Ok(Self {
            shape,
            upload,
            compute,
            download,
            slots,
        })
    }

    fn finish(slot: &mut Slot, margins: &mut [f32], outputs: usize) -> driver::Result<()> {
        if let Some((begin, rows)) = slot.pending {
            slot.downloaded.synchronize()?;
            margins[begin * outputs..(begin + rows) * outputs]
                .copy_from_slice(&slot.host_out.as_slice()[..rows * outputs]);
            slot.pending = None;
        }
        Ok(())
    }

    fn run(
        &mut self,
        gpu: &GpuModel,
        data: &DMatrix,
        trees: std::ops::Range<usize>,
        margins: &mut [f32],
    ) -> driver::Result<()> {
        let cols = self.shape.cols;
        let outputs = self.shape.outputs;
        let mut begin = 0;
        let mut batch = 0;
        while begin < data.n_rows() {
            let slot = &mut self.slots[batch % SLOTS];
            Self::finish(slot, margins, outputs)?;
            let rows = (data.n_rows() - begin).min(self.shape.rows);
            materialize_rows(data, begin, slot.host_rows.range_mut(0..rows * cols));
            slot.host_base
                .range_mut(0..rows * outputs)
                .copy_from_slice(&margins[begin * outputs..(begin + rows) * outputs]);
            // Pinned owners drain their stream on drop, including failures
            // before an event was recorded.
            self.upload.context().bind_to_thread()?;
            let features = &slot.host_rows.as_slice()[..rows * cols];
            // SAFETY: initialized pinned rows, live through the download
            // event; destination fits this block. Host reuse waits above.
            unsafe {
                memcpy_htod_async(
                    slot.rows.addr(),
                    features.as_ptr(),
                    size_of_val(features),
                    self.upload.cu_stream(),
                )
            }?;
            let base = &slot.host_base.as_slice()[..rows * outputs];
            // SAFETY: same lifetime and bounds as the feature upload.
            unsafe {
                memcpy_htod_async(
                    slot.out.addr(),
                    base.as_ptr(),
                    size_of_val(base),
                    self.upload.cu_stream(),
                )
            }?;
            slot.uploaded.record(&self.upload)?;
            self.compute.wait(&slot.uploaded)?;
            let (n_rows, n_cols) = (rows as u32, cols as u32);
            let (tree_begin, tree_end) = (trees.start as u32, trees.end as u32);
            let grid = LaunchConfig {
                grid_dim: (n_rows.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            match &gpu.forest {
                Forest::Narrow { nodes, roots } => {
                    let batch = abi::Batch8 {
                        n_rows,
                        n_cols,
                        tree_begin,
                        tree_end,
                    };
                    let mut launch = self.compute.launch_builder(&gpu.ctx.predict8);
                    launch
                        .arg(nodes)
                        .arg(roots)
                        .arg(&slot.rows)
                        .arg(&mut slot.out)
                        .arg(&batch);
                    // SAFETY: validated model/data and block-local bounds;
                    // forest nodes and all slices live through completion.
                    unsafe { launch.launch(grid) }?;
                }
                Forest::Wide {
                    nodes,
                    categories,
                    vectors,
                    trees,
                } => {
                    let batch = abi::Batch16 {
                        n_rows,
                        n_cols,
                        outputs: outputs as u32,
                        tree_begin,
                        tree_end,
                    };
                    let mut launch = self.compute.launch_builder(&gpu.ctx.predict16);
                    launch
                        .arg(nodes)
                        .arg(categories)
                        .arg(vectors)
                        .arg(trees)
                        .arg(&slot.rows)
                        .arg(&mut slot.out)
                        .arg(&batch);
                    // SAFETY: same as the narrow walk, with validated category
                    // and vector pools and per-tree output indices.
                    unsafe { launch.launch(grid) }?;
                }
            }
            slot.computed.record(&self.compute)?;
            self.download.wait(&slot.computed)?;
            let out = slot.host_out.range_mut(0..rows * outputs);
            // SAFETY: pinned output fits and is not read or reused until
            // `downloaded` completes. Drop drains the download stream.
            unsafe {
                memcpy_dtoh_async(
                    out.as_mut_ptr(),
                    slot.out.addr(),
                    size_of_val(out),
                    self.download.cu_stream(),
                )
            }?;
            slot.downloaded.record(&self.download)?;
            slot.pending = Some((begin, rows));
            begin += rows;
            batch += 1;
        }
        for slot in &mut self.slots {
            Self::finish(slot, margins, outputs)?;
        }
        Ok(())
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        // Also drain an aborted launch sequence: pinned host memory must
        // outlive DMA even if its explicit completion event was not recorded.
        let _ = self.upload.synchronize();
        let _ = self.compute.synchronize();
        let _ = self.download.synchronize();
    }
}

/// A model resident on one NVIDIA CUDA device, built with
/// [`BoostedModel::to_cuda`]. Forest traversal and tree-order f32 accumulation
/// run on the GPU; the objective transform runs on the CPU for bit parity.
/// Inputs, including CSR, materialize only bounded row blocks, not the full
/// batch. Calls may run concurrently; each owns a pinned H2D/compute/D2H
/// pipeline of four row-block slots. Runtime failures return an error, never
/// silently predict on the CPU. As with the existing GPU predictors, model
/// shrinkage uses the CPU's per-iteration shrink-then-add path explicitly.
pub struct GpuModel {
    model: Arc<BoostedModel>,
    ctx: Arc<Context>,
    forest: Forest,
    ordinal: usize,
    pool: Mutex<Vec<Call>>,
}

impl std::fmt::Debug for GpuModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuModel")
            .field("model", &self.model)
            .field("device", &self.ctx.name)
            .field("ordinal", &self.ordinal)
            .finish_non_exhaustive()
    }
}

impl GpuModel {
    /// The source model, including its CPU objective transform.
    #[must_use]
    pub fn model(&self) -> &BoostedModel {
        &self.model
    }

    /// The CUDA device ordinal this model resides on.
    #[must_use]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Raw margins, bit-identical to [`BoostedModel::predict_margin`].
    /// Model shrinkage explicitly uses CPU prediction; all other nonempty
    /// forest walks run on CUDA. No total-batch dense-copy limit applies.
    pub fn predict_margin(
        &self,
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions> {
        self.ctx.check()?;
        let (trees, mut margins) = match plan_margins(&self.model, data, iterations.into())? {
            MarginPlan::Done(margins) => return Ok(margins),
            MarginPlan::Walk { trees, margins } => (trees, margins),
        };
        let outputs = self.model.n_outputs();
        let width = data.n_cols().checked_add(outputs).ok_or_else(|| {
            HessboostError::invalid_data("data", "CUDA prediction row width overflows usize")
        })?;
        let rows = data
            .n_rows()
            .min(BLOCK_ROWS)
            .min(SLOT_ENTRIES / width.max(1));
        if rows == 0 || u32::try_from(data.n_cols()).is_err() || u32::try_from(outputs).is_err() {
            return Err(HessboostError::invalid_data(
                "data",
                "a feature/output row exceeds CUDA prediction's 32 MiB staging limit",
            ));
        }
        let shape = Shape {
            rows,
            cols: data.n_cols(),
            outputs,
        };
        let mut pool = self.pool.lock();
        let cached = pool
            .iter()
            .position(|call| call.shape.serves(shape))
            .map(|i| pool.swap_remove(i));
        drop(pool);
        let mut call = match cached {
            Some(call) => call,
            None => Call::new(&self.ctx, shape).map_err(|e| self.ctx.error(e))?,
        };
        call.run(self, data, trees, &mut margins)
            .map_err(|e| self.ctx.error(e))?;
        // Keep the call when the pool has room, or in place of a pooled
        // call that cannot serve it, so later calls of its shape reuse
        // its buffers instead of reallocating.
        let evicted = {
            let mut pool = self.pool.lock();
            if pool.len() < POOL_CALLS {
                pool.push(call);
                None
            } else if let Some(i) = replaced(pool.iter().map(|pooled| pooled.shape), call.shape) {
                Some(std::mem::replace(&mut pool[i], call))
            } else {
                Some(call)
            }
        };
        drop(evicted);
        Ok(Predictions::new(margins, data.n_rows(), outputs))
    }

    /// Predictions with the CPU's exact objective transform.
    pub fn predict(
        &self,
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions> {
        Ok(self
            .model
            .transform_margins(self.predict_margin(data, iterations)?))
    }

    /// Predicted classes, as for [`BoostedModel::predict_class`].
    pub fn predict_class(
        &self,
        data: &DMatrix,
        iterations: impl Into<Iterations>,
    ) -> Result<Predictions<u32>> {
        Ok(self.model.classes(&self.predict(data, iterations)?))
    }
}

/// Upload through an owned pinned buffer and complete before source locals
/// leave scope. The pinned owner also drains error paths before freeing DMA.
fn upload_forest<T: Plain>(
    stream: &Arc<CudaStream>,
    values: &[T],
) -> driver::Result<DeviceBuffer<T>> {
    let mut device = stream.alloc_zeros(values.len().max(1))?;
    let mut staging = Pinned::new(stream, values.len())?;
    upload_pinned(stream, &mut staging, values, &mut device)?;
    stream.synchronize()?;
    Ok(device)
}

impl BoostedModel {
    /// Upload the compact forest once to NVIDIA CUDA device `ordinal`.
    /// Requires Linux, the `cuda` feature, a CUDA 12.8+ driver and a device of
    /// compute capability 7.5 or newer.
    /// Refuses `gblinear` and per-leaf linear models. Predictions preserve CPU
    /// bits (including categorical splits, DART weights and vector leaves);
    /// objective transforms and model-shrinkage prediction run on the CPU.
    pub fn to_cuda(&self, ordinal: usize) -> Result<GpuModel> {
        ensure_forest_model(self)?;
        let nodes = self
            .trees()
            .iter()
            .try_fold(0usize, |count, tree| count.checked_add(tree.num_nodes()));
        if u32::try_from(self.num_trees()).is_err()
            || nodes.is_none_or(|count| u32::try_from(count).is_err())
            || u32::try_from(self.n_outputs()).is_err()
        {
            return Err(HessboostError::invalid_data(
                "model",
                "CUDA forest indexing exceeds 32 bits",
            ));
        }
        let ctx =
            Context::get(ordinal).map_err(|reason| HessboostError::gpu(reason.to_string()))?;
        ctx.check()?;
        let compact = self.compact_forest();
        let stream = ctx.cuda.new_stream().map_err(|e| ctx.error(e.into()))?;
        let narrow = (self.n_outputs() == 1)
            .then(|| compact.gpu_arena8(|t| self.tree_is_vector_leaf(t)))
            .flatten();
        let uploaded = (|| -> driver::Result<Forest> {
            let forest = if let Some(mut arena) = narrow {
                for (t, &root) in arena.roots.iter().enumerate() {
                    let end = arena
                        .roots
                        .get(t + 1)
                        .map_or(arena.words.len() / 2, |&r| r as usize);
                    let weight = self.tree_weight(t);
                    for node in arena.words[root as usize * 2..end * 2]
                        .as_chunks_mut::<2>()
                        .0
                    {
                        if node[1] & (1 << 31) != 0 {
                            node[0] = (weight * f32::from_bits(node[0])).to_bits();
                        }
                    }
                }
                Forest::Narrow {
                    nodes: upload_forest(&stream, &arena.words)?,
                    roots: upload_forest(&stream, &arena.roots)?,
                }
            } else {
                let parts = compact.gpu_parts();
                let (nodes, vectors) = weighted_leaves(self, &parts);
                let trees: Vec<u32> = parts
                    .roots
                    .iter()
                    .enumerate()
                    .flat_map(|(t, &root)| {
                        [
                            root,
                            self.tree_output(t) as u32,
                            u32::from(self.tree_is_vector_leaf(t)),
                            0,
                        ]
                    })
                    .collect();
                Forest::Wide {
                    nodes: upload_forest(&stream, nodes.as_flattened())?,
                    categories: upload_forest(&stream, parts.categories)?,
                    vectors: upload_forest(&stream, &vectors)?,
                    trees: upload_forest(&stream, &trees)?,
                }
            };
            stream.synchronize()?;
            Ok(forest)
        })();
        let forest = uploaded.map_err(|e| ctx.error(e))?;
        Ok(GpuModel {
            model: Arc::new(self.clone()),
            ctx,
            forest,
            ordinal,
            pool: Mutex::new(Vec::new()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelObjective, ModelSpec};
    use crate::objective::{Objective, RegLoss};
    use crate::tree::{ChildLeaf, RegTree, SplitRule};

    /// Whether device 0 predicts; a missing device skips, printing why
    /// (unless `HESSBOOST_REQUIRE_CUDA` is set), and any other reason fails.
    fn available() -> bool {
        match prediction_unavailable_reason(0) {
            None => true,
            Some(reason) => {
                assert!(
                    reason.is_environment() && std::env::var_os("HESSBOOST_REQUIRE_CUDA").is_none(),
                    "{reason}"
                );
                eprintln!("skipping CUDA prediction test: {reason}");
                false
            }
        }
    }

    fn model(trees: Vec<RegTree>, weights: Vec<f32>) -> BoostedModel {
        BoostedModel::from_parts(
            trees,
            weights,
            vec![0.0],
            ModelSpec {
                objective: ModelObjective::new(Objective::SquaredError(RegLoss::default()))
                    .unwrap(),
                max_delta_step: 0.0,
                num_class: 0,
                n_outputs: 1,
                n_targets: 1,
                n_features: 1,
            },
        )
    }

    /// Builds `model` on device 0 and checks its margins against the CPU's
    /// bit for bit over several iteration ranges, including `0..1` alone:
    /// a range reaching a large-magnitude tree can round away tree 0's
    /// small leaves and the base margins, hiding a misrouted row.
    fn parity(model: &BoostedModel, data: &DMatrix) -> GpuModel {
        let gpu = model.to_cuda(0).unwrap();
        for iterations in [
            Iterations::Best,
            Iterations::from(..),
            Iterations::from(0..1),
            Iterations::from(0..0),
        ] {
            let cpu = model.predict_margin(data, iterations).unwrap();
            let device = gpu.predict_margin(data, iterations).unwrap();
            assert_eq!(
                cpu.as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                device
                    .as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        gpu
    }

    #[test]
    fn compact_width_subnormal_comparisons_and_weighted_order() {
        if !available() {
            return;
        }
        let subnormal = f32::from_bits(3);
        let data = DMatrix::from_dense(&[-0.0, 0.0, subnormal, -subnormal, f32::NAN], 5, 1)
            .unwrap()
            .with_base_margin(&[-0.0, subnormal, -subnormal, 0.0, 0.0])
            .unwrap();
        let mut split = RegTree::with_root(1.0);
        split.expand(
            0,
            SplitRule::numeric(0, f32::from_bits(2), false),
            ChildLeaf::new(subnormal, 1.0),
            ChildLeaf::new(-subnormal, 1.0),
        );
        let mut trees = vec![split];
        for value in [16_777_216.0, 1.0, -16_777_216.0, -0.0, subnormal] {
            let mut tree = RegTree::with_root(1.0);
            tree.set_leaf_value(0, value);
            trees.push(tree);
        }
        let mut model = model(trees, vec![0.5, 1.0, 0.3, 1.0, -0.0, 0.5]);
        model.set_best_iteration(Some(2));
        assert!(matches!(
            parity(&model, &data).forest,
            Forest::Narrow { .. }
        ));
    }

    #[test]
    fn oversized_tree_uses_wide_arena_without_changing_predictions() {
        if !available() {
            return;
        }
        let mut tree = RegTree::with_root(1.0);
        // Breadth-first expansion: 32,769 valid nodes, just beyond arena8.
        for node in 0..16_384 {
            tree.expand(
                node,
                SplitRule::numeric(0, 0.5, node % 2 == 0),
                ChildLeaf::new(-0.25, 1.0),
                ChildLeaf::new(0.75, 1.0),
            );
        }
        let model = model(vec![tree], Vec::new());
        let data = DMatrix::from_dense(&[0.0, 0.5, 1.0, f32::NAN], 4, 1).unwrap();
        assert!(matches!(parity(&model, &data).forest, Forest::Wide { .. }));
    }

    #[test]
    fn categorical_casts_match_rust_saturation_and_missing_direction() {
        if !available() {
            return;
        }
        for default_left in [false, true] {
            let mut tree = RegTree::with_root(1.0);
            tree.expand(
                0,
                SplitRule::categorical(0, &[0, 2, u32::MAX], default_left),
                ChildLeaf::new(0.25, 1.0),
                ChildLeaf::new(-0.75, 1.0),
            );
            let model = model(vec![tree], Vec::new());
            let data = DMatrix::from_dense(
                &[-2.5, -0.0, 0.8, 1.9, 2.9, 4_294_967_296.0, f32::NAN],
                7,
                1,
            )
            .unwrap();
            assert!(matches!(parity(&model, &data).forest, Forest::Wide { .. }));
        }
    }

    /// Calls of 1, 2, then 3 rows fill the pool with calls too small for
    /// the third shape: the call built for it must replace one of them, so
    /// a repeat of that shape reuses its buffers.
    #[test]
    fn pool_keeps_a_call_for_a_shape_it_missed() {
        if !available() {
            return;
        }
        let mut tree = RegTree::with_root(1.0);
        tree.set_leaf_value(0, 0.5);
        let gpu = model(vec![tree], Vec::new()).to_cuda(0).unwrap();
        let rows = |n: usize| DMatrix::from_dense(&vec![0.0; n], n, 1).unwrap();
        for n in [1, 2, 3] {
            gpu.predict_margin(&rows(n), ..).unwrap();
        }
        let wanted = Shape {
            rows: 3,
            cols: 1,
            outputs: 1,
        };
        let pool = gpu.pool.lock();
        assert_eq!(pool.len(), POOL_CALLS);
        assert!(pool.iter().any(|call| call.shape.serves(wanted)));
    }

    /// A finished call replaces the smallest pooled call that cannot serve
    /// its own shape: overlapping misses of one shape both end up pooled, a
    /// larger call back from a smaller request is kept, and a call every
    /// pooled one serves is dropped.
    #[test]
    fn returning_calls_replace_pooled_calls_that_cannot_serve_them() {
        let shape = |rows| Shape {
            rows,
            cols: 1,
            outputs: 1,
        };
        // Two overlapping 3-row misses over pooled 2- and 1-row calls: the
        // first replaces the 1-row call, the second the 2-row one.
        let mut pool = [shape(2), shape(1)];
        for expect in [1, 0] {
            assert_eq!(replaced(pool.iter().copied(), shape(3)), Some(expect));
            pool[expect] = shape(3);
        }
        // A 3-row call back from a smaller request, over two 2-row calls.
        assert_eq!(
            replaced([shape(2), shape(2)].into_iter(), shape(3)),
            Some(0)
        );
        assert_eq!(replaced([shape(3), shape(2)].into_iter(), shape(2)), None);
    }
}
