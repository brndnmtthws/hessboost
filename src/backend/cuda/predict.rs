//! Resident compact forests and a bounded, double-buffered prediction pipeline.
//!
//! Prediction owns an event-tracked context, separate from training's
//! single-stream context. Each concurrent call owns its streams and staging
//! buffers; only the immutable forest is shared. Numeric scalar forests use
//! the shared 8-byte arena when it fits, otherwise the 16-byte arena handles
//! categorical splits, multiclass and vector leaves. Leaf weighting happens
//! once on the CPU in f32; the GPU adds in tree order without FMA or FTZ.

use super::{Pinned, compile, libraries, upload_pinned};
use crate::backend::shared::{ensure_forest_model, materialize_rows};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::{BoostedModel, Iterations, Predictions};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DriverError, LaunchConfig, PushKernelArg, result, sys,
};
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

type Opened = std::result::Result<Arc<Context>, String>;
static CONTEXTS: Mutex<Vec<(usize, Opened)>> = Mutex::new(Vec::new());

impl Context {
    fn open(ordinal: usize) -> std::result::Result<Self, String> {
        libraries()?;
        let count = match CudaContext::device_count() {
            Ok(count) => count,
            Err(e) if e.0 == sys::CUresult::CUDA_ERROR_NO_DEVICE => 0,
            Err(e) => return Err(format!("CUDA init: {e}")),
        };
        if ordinal >= usize::try_from(count).unwrap_or(0) {
            return Err(format!("no CUDA device {ordinal} ({count} found)"));
        }
        // An independent context: training disables tracking on its own
        // context, which must never govern this multi-stream pipeline.
        let cuda = CudaContext::new_non_primary(ordinal, 0)
            .map_err(|e| format!("CUDA prediction context: {e}"))?;
        let (major, minor) = cuda
            .compute_capability()
            .map_err(|e| format!("CUDA compute capability: {e}"))?;
        let module = compile::load_source(
            &cuda,
            &format!("sm_{major}{minor}"),
            include_str!("predict.cu"),
            c"hessboost_predict.cu",
        )?;
        let predict8 = module
            .load_function("predict8")
            .map_err(|e| format!("CUDA predict8: {e}"))?;
        let predict16 = module
            .load_function("predict16")
            .map_err(|e| format!("CUDA predict16: {e}"))?;
        let name = cuda.name().map_err(|e| format!("CUDA device name: {e}"))?;
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

    fn error(&self, error: DriverError) -> HessboostError {
        self.failed.store(true, Ordering::Release);
        HessboostError::gpu(format!("CUDA prediction: {error}"))
    }
}

/// Whether CUDA device `ordinal` can predict, including its kernel compiler
/// and any previous runtime failure. Does not initialize the training context.
#[must_use]
pub fn prediction_available(ordinal: usize) -> bool {
    Context::get(ordinal).is_ok_and(|ctx| !ctx.failed.load(Ordering::Acquire))
}

/// The name of CUDA device `ordinal` when it can predict.
#[must_use]
pub fn prediction_device_name(ordinal: usize) -> Option<String> {
    Context::get(ordinal)
        .ok()
        .filter(|ctx| !ctx.failed.load(Ordering::Acquire))
        .map(|ctx| ctx.name.clone())
}

enum Forest {
    Narrow {
        nodes: CudaSlice<u32>,
        roots: CudaSlice<u32>,
    },
    Wide {
        nodes: CudaSlice<u8>,
        categories: CudaSlice<u32>,
        vectors: CudaSlice<f32>,
        trees: CudaSlice<u32>,
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

struct Slot {
    rows: CudaSlice<f32>,
    out: CudaSlice<f32>,
    host_rows: Pinned<f32>,
    host_base: Pinned<f32>,
    host_out: Pinned<f32>,
    uploaded: CudaEvent,
    computed: CudaEvent,
    downloaded: CudaEvent,
    /// Row range whose D2H is in flight; never overwrite a live slot.
    pending: Option<(usize, usize)>,
}

struct Call {
    shape: Shape,
    upload: Arc<CudaStream>,
    compute: Arc<CudaStream>,
    download: Arc<CudaStream>,
    slots: [Slot; 2],
}

impl Call {
    fn new(ctx: &Context, shape: Shape) -> std::result::Result<Self, DriverError> {
        let upload = ctx.cuda.new_stream()?;
        let compute = ctx.cuda.new_stream()?;
        let download = ctx.cuda.new_stream()?;
        let make_slot = || -> std::result::Result<Slot, DriverError> {
            Ok(Slot {
                rows: upload.alloc_zeros(shape.rows * shape.cols)?,
                out: upload.alloc_zeros(shape.rows * shape.outputs)?,
                host_rows: Pinned::new(&upload, shape.rows * shape.cols, true)?,
                host_base: Pinned::new(&upload, shape.rows * shape.outputs, true)?,
                host_out: Pinned::new(&download, shape.rows * shape.outputs, false)?,
                uploaded: ctx.cuda.new_event(None)?,
                computed: ctx.cuda.new_event(None)?,
                downloaded: ctx.cuda.new_event(None)?,
                pending: None,
            })
        };
        let slots = [make_slot()?, make_slot()?];
        Ok(Self {
            shape,
            upload,
            compute,
            download,
            slots,
        })
    }

    fn finish(
        slot: &mut Slot,
        margins: &mut [f32],
        outputs: usize,
    ) -> std::result::Result<(), DriverError> {
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
    ) -> std::result::Result<(), DriverError> {
        let cols = self.shape.cols;
        let outputs = self.shape.outputs;
        let mut begin = 0;
        let mut batch = 0;
        while begin < data.n_rows() {
            let slot = &mut self.slots[batch % 2];
            Self::finish(slot, margins, outputs)?;
            let rows = (data.n_rows() - begin).min(self.shape.rows);
            materialize_rows(
                data,
                begin,
                &mut slot.host_rows.as_mut_slice()[..rows * cols],
            );
            slot.host_base.as_mut_slice()[..rows * outputs]
                .copy_from_slice(&margins[begin * outputs..(begin + rows) * outputs]);
            // Keep cudarc's per-slice tracking enabled. Raw copies use the
            // tracked pointer guards, and pinned owners drain their stream
            // on drop, including failures before event recording.
            self.upload.context().bind_to_thread()?;
            {
                let (dst, _record) = slot.rows.device_ptr_mut(&self.upload);
                // SAFETY: initialized pinned rows, live through the download
                // event; destination fits this block. Host reuse waits above.
                unsafe {
                    result::memcpy_htod_async(
                        dst,
                        &slot.host_rows.as_slice()[..rows * cols],
                        self.upload.cu_stream(),
                    )
                }?;
            }
            {
                let (dst, _record) = slot.out.device_ptr_mut(&self.upload);
                // SAFETY: same lifetime and bounds as the feature upload.
                unsafe {
                    result::memcpy_htod_async(
                        dst,
                        &slot.host_base.as_slice()[..rows * outputs],
                        self.upload.cu_stream(),
                    )
                }?;
            }
            slot.uploaded.record(&self.upload)?;
            self.compute.wait(&slot.uploaded)?;
            let n_rows = rows as u32;
            let n_cols = cols as u32;
            let k = outputs as u32;
            let first = trees.start as u32;
            let end = trees.end as u32;
            let grid = LaunchConfig {
                grid_dim: (n_rows.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            match &gpu.forest {
                Forest::Narrow { nodes, roots } => {
                    let mut launch = self.compute.launch_builder(&gpu.ctx.predict8);
                    launch
                        .arg(nodes)
                        .arg(roots)
                        .arg(&slot.rows)
                        .arg(&mut slot.out)
                        .arg(&n_rows)
                        .arg(&n_cols)
                        .arg(&first)
                        .arg(&end);
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
                    let mut launch = self.compute.launch_builder(&gpu.ctx.predict16);
                    launch
                        .arg(nodes)
                        .arg(categories)
                        .arg(vectors)
                        .arg(trees)
                        .arg(&slot.rows)
                        .arg(&mut slot.out)
                        .arg(&n_rows)
                        .arg(&n_cols)
                        .arg(&k)
                        .arg(&first)
                        .arg(&end);
                    // SAFETY: same as the narrow walk, with validated category
                    // and vector pools and per-tree output indices.
                    unsafe { launch.launch(grid) }?;
                }
            }
            slot.computed.record(&self.compute)?;
            self.download.wait(&slot.computed)?;
            {
                let (src, _record) = slot.out.device_ptr(&self.download);
                // SAFETY: pinned output fits and is not read or reused until
                // `downloaded` completes. Drop drains the download stream.
                unsafe {
                    result::memcpy_dtoh_async(
                        &mut slot.host_out.as_mut_slice()[..rows * outputs],
                        src,
                        self.download.cu_stream(),
                    )
                }?;
            }
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
/// batch. Calls may run concurrently; each owns a double-buffered pinned
/// H2D/compute/D2H pipeline. Runtime failures return an error, never silently
/// predict on the CPU. As with the existing GPU predictors, model shrinkage
/// uses the CPU's per-iteration shrink-then-add path explicitly.
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
        let iterations = iterations.into();
        if self.model.shrinkage().is_some() {
            return self.model.predict_margin(data, iterations);
        }
        self.model.validate_prediction_data(data)?;
        let trees = self
            .model
            .iteration_trees(self.model.resolve_iterations(iterations, "iterations")?);
        let mut margins = self.model.initial_margins(data);
        let outputs = self.model.n_outputs();
        if !trees.is_empty() && data.n_rows() != 0 {
            let width = data.n_cols().checked_add(outputs).ok_or_else(|| {
                HessboostError::invalid_data("data", "CUDA prediction row width overflows usize")
            })?;
            let rows = data
                .n_rows()
                .min(BLOCK_ROWS)
                .min(SLOT_ENTRIES / width.max(1));
            if rows == 0 || u32::try_from(data.n_cols()).is_err() || u32::try_from(outputs).is_err()
            {
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
            let mut pool = self.pool.lock();
            if pool.len() < POOL_CALLS {
                pool.push(call);
            }
        }
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
fn upload_forest<
    T: Copy + Send + Sync + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
>(
    stream: &Arc<CudaStream>,
    values: &[T],
) -> std::result::Result<CudaSlice<T>, DriverError> {
    let mut device = stream.alloc_zeros(values.len().max(1))?;
    let mut staging = Pinned::new(stream, values.len(), true)?;
    upload_pinned(stream, &mut staging, values, &mut device)?;
    stream.synchronize()?;
    Ok(device)
}

impl BoostedModel {
    /// Upload the compact forest once to NVIDIA CUDA device `ordinal`.
    /// Requires Linux, the `cuda` feature, a CUDA 12.8+ driver and NVRTC.
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
        let ctx = Context::get(ordinal).map_err(HessboostError::gpu)?;
        ctx.check()?;
        let compact = self.compact_forest();
        let stream = ctx.cuda.new_stream().map_err(|e| ctx.error(e))?;
        let narrow = (self.n_outputs() == 1)
            .then(|| compact.gpu_arena8(|t| self.tree_is_vector_leaf(t)))
            .flatten();
        let uploaded = (|| -> std::result::Result<Forest, DriverError> {
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
                let mut nodes = parts.nodes.to_vec();
                let mut vectors = parts.leaf_vectors.to_vec();
                let mut trees = Vec::with_capacity(parts.roots.len() * 4);
                for (t, &root) in parts.roots.iter().enumerate() {
                    let end = parts
                        .roots
                        .get(t + 1)
                        .map_or(nodes.len() / 16, |&r| r as usize);
                    let weight = self.tree_weight(t);
                    let vector = self.tree_is_vector_leaf(t);
                    trees.extend([root, self.tree_output(t) as u32, u32::from(vector), 0]);
                    for (offset, node) in nodes[root as usize * 16..end * 16]
                        .as_chunks_mut::<16>()
                        .0
                        .iter_mut()
                        .enumerate()
                    {
                        let left = u32::from_ne_bytes(node[8..12].try_into().expect("four bytes"));
                        if left as usize != root as usize + offset {
                            continue;
                        }
                        let aux = u32::from_ne_bytes(node[12..16].try_into().expect("four bytes"));
                        if vector {
                            for value in &mut vectors[aux as usize..aux as usize + self.n_outputs()]
                            {
                                *value *= weight;
                            }
                        } else {
                            node[12..16].copy_from_slice(
                                &(weight * f32::from_bits(aux)).to_bits().to_ne_bytes(),
                            );
                        }
                    }
                }
                Forest::Wide {
                    nodes: upload_forest(&stream, &nodes)?,
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

    fn available() -> bool {
        match Context::get(0) {
            Ok(_) => true,
            Err(reason) => {
                assert!(
                    std::env::var_os("HESSBOOST_REQUIRE_CUDA").is_none(),
                    "{reason}"
                );
                assert!(
                    ["libcuda not found", "libnvrtc not found", "no CUDA device"]
                        .iter()
                        .any(|expected| reason.contains(expected)),
                    "{reason}"
                );
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

    fn parity(model: &BoostedModel, data: &DMatrix) -> GpuModel {
        let gpu = model.to_cuda(0).unwrap();
        for iterations in [
            Iterations::Best,
            Iterations::from(..),
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
}
