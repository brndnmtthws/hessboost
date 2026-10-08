//! What a [`CudaHistBackend`](super::CudaHistBackend) did, for tests and
//! benchmarks. Public only through the hidden `hessboost::internals`.

/// How many nodes (and rows) a backend built with each strategy of the
/// [`cuda`](super) module docs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct NodeCounts {
    /// Nodes summed as exact integers in one piece (strategy 1).
    pub exact_nodes: u64,
    /// Nodes summed as exact integer chunks reduced in `f64` (strategy 2).
    pub exact_chunk_nodes: u64,
    /// Nodes summed as `f64` chains on the GPU (strategy 3).
    pub chain_nodes: u64,
    /// Nodes built by the CPU backend (non-finite gradients, input
    /// mismatches, and every node after a CUDA error).
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

/// Resident split searches, and those replayed on the host to keep the
/// CPU's NaN comparisons (semantic fallbacks, not CUDA failures).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ScanDiagnostics {
    /// Nodes whose feature results were merged on the device.
    pub device_nodes: u64,
    /// Of those, nodes whose prefixes were formed by warp scans: their
    /// tree's gradients sum exactly over all of its rows, so every
    /// association of the additions gives the CPU's chain bits.
    pub exact_nodes: u64,
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
