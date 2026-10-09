//! The kernels' by-value parameter structs, field for field the
//! `#[repr(C)]` layouts of `cuda-kernels/` (`train.rs`, `categorical.rs`,
//! `predict.rs`): each is one PTX `.param .align 8 .b8` (or `.align 4`)
//! array the driver copies from the pushed value. Device pointers are
//! `CUdeviceptr`s (`u64`), the kernels' 8-byte pointers.

use super::driver::{LaunchArg, Memory, MemoryMut, Params, by_value};
use cuda_core::sys;

/// `memory`'s device address, for a kernel that reads it.
pub(super) fn ptr<T>(memory: &impl Memory<T>) -> sys::CUdeviceptr {
    memory.addr()
}

/// `memory`'s device address, for a kernel that writes it.
pub(super) fn ptr_mut<T>(memory: &mut impl MemoryMut<T>) -> sys::CUdeviceptr {
    memory.addr()
}

/// The `logistic` kernel's scalars.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct LogisticParams {
    pub scale_pos_weight: f32,
    pub min_hess: f32,
    pub max_input: f32,
    pub lanes: u32,
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

// Each is `#[repr(C)]` of plain integers, floats and device addresses, with
// the size and alignment of the kernel parameter it is pushed as (the
// kernels' layouts above; `tests` checks them against the PTX).
by_value!(
    LogisticParams,
    TileWork,
    Chunks,
    ChainWork,
    SparseTiles,
    PartTiles,
    Regularization,
    NumericTasks,
    NumericResults,
    CategoryTasks,
    SortWorkspace,
    CategoricalTasks,
    CategoricalResults,
    Batch16,
    Batch8,
);

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
            ("logistic", 6, of::<LogisticParams>()),
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

    /// The PTX type of each parameter of `entry`: `u64`, `f64`, ..., or
    /// `b8` for a by-value struct.
    fn param_types(module: Module, entry: &str) -> Vec<String> {
        let ptx = module.ptx();
        let start = ptx
            .find(&format!(".visible .entry {entry}("))
            .unwrap_or_else(|| panic!("no entry {entry}"));
        ptx[start..]
            .lines()
            .skip(1)
            .take_while(|line| !line.starts_with(')'))
            .map(|line| {
                // `.param .u64 [.ptr .align 8] name,` or, for a by-value
                // struct, `.param .align 8 .b8 name[size],`.
                let ty = line
                    .trim()
                    .trim_start_matches(".param")
                    .split_whitespace()
                    .next()
                    .expect("parameter type");
                if ty == ".align" { "b8" } else { &ty[1..] }.to_owned()
            })
            .collect()
    }

    /// The kernels that take slices have the parameters their launches push
    /// (`Launch::slice`, `pairs` and `raw_slice`: a pointer and a length
    /// each, `S` here), in order.
    #[test]
    fn slice_kernels_take_the_pushed_parameters() {
        let kernels = [
            ("stage_units", "S S f64 f64"),
            ("iota_rows", "S u32"),
            ("squared_error", "S S S f32 S"),
            ("logistic", "S S S b8 S"),
            ("grad_domain", "S S"),
            ("finalize_exact", "S S u64 f64 f64 S"),
            ("finalize_exact_sub", "S S u64 f64 f64 S"),
            ("reduce_chunks", "S S u64 f64 f64 S"),
            ("reduce_chains", "S u64 u32 S"),
            ("subtract_hists", "S S u64"),
            ("route_runs", "S S S S"),
        ];
        for (entry, signature) in kernels {
            let expected: Vec<&str> = signature
                .split(' ')
                .flat_map(|p| {
                    if p == "S" {
                        vec!["u64", "u64"]
                    } else {
                        vec![p]
                    }
                })
                .collect();
            assert_eq!(param_types(Module::Training, entry), expected, "{entry}");
        }
    }
}
