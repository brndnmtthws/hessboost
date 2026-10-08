//! The kernels' by-value parameter structs, field for field the
//! `#[repr(C)]` layouts of `cuda-kernels/` (`train.rs`, `categorical.rs`,
//! `predict.rs`): each is one PTX `.param .align 8 .b8` (or `.align 4`)
//! array the driver copies from the pushed value. Device pointers are
//! `CUdeviceptr`s (`u64`), the kernels' 8-byte pointers.

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, sys};

/// `slice`'s device address (training contexts disable cudarc's event
/// tracking, so the access guard records nothing).
pub(super) fn ptr<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> sys::CUdeviceptr {
    slice.device_ptr(stream).0
}

/// `slice`'s device address, for a kernel that writes it.
pub(super) fn ptr_mut<T>(slice: &mut CudaSlice<T>, stream: &CudaStream) -> sys::CUdeviceptr {
    slice.device_ptr_mut(stream).0
}

/// `bin_dense`'s matrix shape and missing-value marker.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct DenseCells {
    pub cells: u64,
    pub n_cols: u32,
    pub missing: f32,
}

/// The `logistic` kernel's scalars.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct LogisticParams {
    pub weighted: i32,
    pub scale_pos_weight: f32,
    pub min_hess: f32,
    pub max_input: f32,
    pub lanes: u32,
    pub n: u64,
}

/// A dense integer histogram launch's tiles, feature groups, bin layout
/// and histogram width.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct TileWork {
    pub tiles: sys::CUdeviceptr,
    pub groups: sys::CUdeviceptr,
    pub total_bins: u64,
    pub stride: u32,
    pub sentinel: u32,
    pub n_groups: u32,
}

/// `segs` chunks of `seg_rows` of a chain launch's `n` rows, and the
/// partials' width.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Chunks {
    pub n: u64,
    pub seg_rows: u64,
    pub segs: u64,
    pub total_bins: u64,
}

/// A dense chain launch's chunks and bin layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ChainWork {
    pub chunks: Chunks,
    pub stride: u32,
    pub n_cols: u32,
    pub sentinel: u32,
}

/// A CSR integer histogram launch's tiles and histogram width.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct SparseTiles {
    pub tiles: sys::CUdeviceptr,
    pub total_bins: u64,
}

/// A partition's segments, tile codes, rules, and per-tile left counts.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct PartTiles {
    pub segs: sys::CUdeviceptr,
    pub ptiles: sys::CUdeviceptr,
    pub rules: sys::CUdeviceptr,
    pub tile_left: sys::CUdeviceptr,
}

/// The regularization a split search reads (the host's `RegParams`).
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Regularization {
    pub lambda: f64,
    pub alpha: f64,
    pub max_delta_step: f64,
    pub min_child_weight: f64,
}

/// `scan_splits`'s tasks with their nodes' totals and scoring parameters;
/// `exact` certifies every histogram of the tree exact.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct NumericTasks {
    pub tasks: sys::CUdeviceptr,
    pub n_tasks: u64,
    pub totals: sys::CUdeviceptr,
    pub params: sys::CUdeviceptr,
    pub dense: i32,
    pub exact: u32,
}

/// The numeric scan's per-task results.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct NumericResults {
    pub meta: sys::CUdeviceptr,
    pub acc: sys::CUdeviceptr,
}

/// A wave's categorical tasks, as the sort kernels read them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct CategoryTasks {
    pub tasks: sys::CUdeviceptr,
    pub n_tasks: u32,
}

/// The sort keys and identity order `category_keys` writes.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct SortWorkspace {
    pub keys: sys::CUdeviceptr,
    pub order: sys::CUdeviceptr,
}

/// `scan_categorical`'s tasks with their nodes' totals and scoring
/// parameters; `exact` as for [`NumericTasks`].
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct CategoricalTasks {
    pub tasks: sys::CUdeviceptr,
    pub n_tasks: u64,
    pub totals: sys::CUdeviceptr,
    pub params: sys::CUdeviceptr,
    pub exact: u32,
}

/// The categorical scan's per-result outputs.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct CategoricalResults {
    pub meta: sys::CUdeviceptr,
    pub children: sys::CUdeviceptr,
    pub sets: sys::CUdeviceptr,
}

/// A 16-byte-layout prediction batch: rows, features, outputs and trees.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Batch16 {
    pub n_rows: u32,
    pub n_cols: u32,
    pub outputs: u32,
    pub tree_begin: u32,
    pub tree_end: u32,
}

/// An 8-byte-layout prediction batch: rows, features and trees.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Batch8 {
    pub n_rows: u32,
    pub n_cols: u32,
    pub tree_begin: u32,
    pub tree_end: u32,
}

// SAFETY: each is `#[repr(C)]` of plain integers, floats and device
// addresses, with the size and alignment of the kernel parameter it is
// pushed as (the kernels' layouts above).
unsafe impl DeviceRepr for DenseCells {}
// SAFETY: as above.
unsafe impl DeviceRepr for LogisticParams {}
// SAFETY: as above.
unsafe impl DeviceRepr for TileWork {}
// SAFETY: as above.
unsafe impl DeviceRepr for Chunks {}
// SAFETY: as above.
unsafe impl DeviceRepr for ChainWork {}
// SAFETY: as above.
unsafe impl DeviceRepr for SparseTiles {}
// SAFETY: as above.
unsafe impl DeviceRepr for PartTiles {}
// SAFETY: as above.
unsafe impl DeviceRepr for Regularization {}
// SAFETY: as above.
unsafe impl DeviceRepr for NumericTasks {}
// SAFETY: as above.
unsafe impl DeviceRepr for NumericResults {}
// SAFETY: as above.
unsafe impl DeviceRepr for CategoryTasks {}
// SAFETY: as above.
unsafe impl DeviceRepr for SortWorkspace {}
// SAFETY: as above.
unsafe impl DeviceRepr for CategoricalTasks {}
// SAFETY: as above.
unsafe impl DeviceRepr for CategoricalResults {}
// SAFETY: as above.
unsafe impl DeviceRepr for Batch16 {}
// SAFETY: as above.
unsafe impl DeviceRepr for Batch8 {}

#[cfg(test)]
mod tests {
    use super::super::kernels::Module;
    use super::*;
    use std::mem::{align_of, size_of};

    /// The `(alignment, bytes)` of by-value parameter `index` of `entry` in
    /// the embedded PTX.
    fn param(module: Module, entry: &str, index: usize) -> (usize, usize) {
        let ptx = module.ptx();
        let start = ptx
            .find(&format!(".visible .entry {entry}("))
            .unwrap_or_else(|| panic!("no entry {entry}"));
        let name = format!("{entry}_param_{index}[");
        let line = ptx[start..]
            .lines()
            .take_while(|line| !line.starts_with(')'))
            .find(|line| line.contains(&name))
            .unwrap_or_else(|| panic!("{entry} has no by-value parameter {index}"));
        let field = |after: &str, end: char| -> usize {
            let from = line.find(after).expect("field") + after.len();
            line[from..line[from..].find(end).expect("end") + from]
                .trim()
                .parse()
                .expect("number")
        };
        (field(".align ", ' '), field(&name, ']'))
    }

    /// Every pushed struct has the size and alignment of the compiled
    /// kernel's parameter, so host and kernel layouts cannot drift apart
    /// without a GPU noticing.
    #[test]
    fn layouts_match_the_compiled_kernels() {
        fn of<T>() -> (usize, usize) {
            (align_of::<T>(), size_of::<T>())
        }
        let training = [
            ("bin_dense", 1, of::<DenseCells>()),
            ("logistic", 3, of::<LogisticParams>()),
            ("scan_splits", 3, of::<NumericTasks>()),
            ("scan_splits", 4, of::<Regularization>()),
            ("scan_splits", 5, of::<NumericResults>()),
            ("category_keys", 3, of::<CategoryTasks>()),
            ("category_keys", 4, of::<Regularization>()),
            ("category_keys", 5, of::<SortWorkspace>()),
            ("category_merge", 1, of::<CategoryTasks>()),
            ("scan_categorical", 3, of::<CategoricalTasks>()),
            ("scan_categorical", 4, of::<Regularization>()),
            ("scan_categorical", 6, of::<CategoricalResults>()),
            ("merge_scans", 3, of::<NumericResults>()),
            ("merge_scans", 4, of::<CategoricalResults>()),
        ];
        for (entry, index, layout) in training {
            assert_eq!(param(Module::Training, entry, index), layout, "{entry}");
        }
        for width in ["u8", "u16", "u32"] {
            for (prefix, index, layout) in [
                ("hist_shared", 6, of::<TileWork>()),
                ("hist_global", 6, of::<TileWork>()),
                ("hist_chain", 5, of::<ChainWork>()),
                ("hist_sparse", 6, of::<SparseTiles>()),
                ("hist_sparse_chain", 5, of::<Chunks>()),
                ("route_count", 6, of::<PartTiles>()),
                ("route_sparse", 6, of::<PartTiles>()),
            ] {
                let entry = format!("{prefix}_{width}");
                assert_eq!(param(Module::Training, &entry, index), layout, "{entry}");
            }
        }
        assert_eq!(param(Module::Prediction, "predict16", 6), of::<Batch16>());
        assert_eq!(param(Module::Prediction, "predict8", 4), of::<Batch8>());
    }
}
