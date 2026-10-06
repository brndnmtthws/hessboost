// hessboost CUDA kernels: gradient-histogram construction that reproduces
// the CPU backend's `f64` sums bit for bit (`src/backend/cuda/mod.rs`).
//
// Compiled at run time by NVRTC for the device's own architecture, with FP
// contraction off (`--fmad=false`), no flush-to-zero, and IEEE division and
// square root, so every floating-point operation here is the single IEEE
// operation the CPU performs. No libdevice transcendentals, no
// floating-point atomics: integer sums are order-free, and every `f64` sum
// is one thread's chain in the CPU's order.
//
// Layouts (all row ids `u32`, every offset 64-bit):
// - `gpair`: one `float2` (gradient, Hessian) per row, as `GradPair`.
// - `units`: one `longlong2` per row, the gradient pair in grains.
// - bins: the index's row-major store of global bins, `u16` or `u32`;
//   dense rows at `r * n_cols`, otherwise CSR through `row_ptr`.
// - integer partials: `[segment][bin][2]` 64-bit words (two's complement).
// - `f64` partials and the output: `[segment][bin]` `double2`, as
//   `GradStats`.

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef unsigned short u16;

#define GRID_STRIDE(i, n) \
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += (u64)gridDim.x * blockDim.x)

// Each gradient pair as integer multiples of its component's grain:
// `(f64)x * 2^-grain`, an exact power-of-two scaling, truncated to `i64`
// (exact for the integers the exact paths read; PTX saturates the rest,
// as Rust's `as i64` does). A zero scale (an all-zero component) gives 0.
extern "C" __global__ void stage_units(const float2* __restrict__ gpair,
                                       longlong2* __restrict__ units, u64 n,
                                       double grad_scale, double hess_scale) {
    GRID_STRIDE(i, n) {
        float2 p = gpair[i];
        units[i] = make_longlong2((i64)((double)p.x * grad_scale),
                                  (i64)((double)p.y * hess_scale));
    }
}

// Integer histogram of `n` rows of a dense index, one `(row, feature)`
// element per iteration. Row `i` of the listing adds to segment
// `i / seg_rows`. Two's-complement `u64` atomics add signed grains.
template <typename B>
__device__ void hist_units_dense(const B* __restrict__ bins, u64 n_cols,
                                 const u32* __restrict__ rows, u64 n,
                                 u64 seg_rows,
                                 const longlong2* __restrict__ units,
                                 u64* __restrict__ partials, u64 total_bins) {
    GRID_STRIDE(idx, n * n_cols) {
        u64 i = idx / n_cols;
        u64 f = idx - i * n_cols;
        u32 r = rows[i];
        u64 bin = (u64)bins[(u64)r * n_cols + f];
        longlong2 u = units[r];
        u64* h = partials + ((i / seg_rows) * total_bins + bin) * 2;
        atomicAdd(h, (u64)u.x);
        atomicAdd(h + 1, (u64)u.y);
    }
}

// The CSR form: one listed row per iteration, every stored entry of it.
template <typename B>
__device__ void hist_units_csr(const B* __restrict__ bins,
                               const u64* __restrict__ row_ptr,
                               const u32* __restrict__ rows, u64 n,
                               u64 seg_rows,
                               const longlong2* __restrict__ units,
                               u64* __restrict__ partials, u64 total_bins) {
    GRID_STRIDE(i, n) {
        u32 r = rows[i];
        longlong2 u = units[r];
        u64* base = partials + (i / seg_rows) * total_bins * 2;
        for (u64 k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
            u64* h = base + (u64)bins[k] * 2;
            atomicAdd(h, (u64)u.x);
            atomicAdd(h + 1, (u64)u.y);
        }
    }
}

// One thread per (segment, feature): the segment's rows in listing order,
// each adding its widened pair to the feature's bin. A thread owns its
// feature's bins in its segment, so every bin is one `f64` chain from
// `+0.0` in row order, exactly the CPU's (partials zeroed beforehand).
template <typename B>
__device__ void hist_chain_dense(const B* __restrict__ bins, u64 n_cols,
                                 const u32* __restrict__ rows, u64 n,
                                 u64 seg_rows, u64 segs,
                                 const float2* __restrict__ gpair,
                                 double2* __restrict__ partials,
                                 u64 total_bins) {
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= segs * n_cols) return;
    u64 seg = t / n_cols;
    u64 f = t - seg * n_cols;
    u64 begin = seg * seg_rows;
    u64 end = begin + seg_rows < n ? begin + seg_rows : n;
    double2* h = partials + seg * total_bins;
    for (u64 i = begin; i < end; ++i) {
        u32 r = rows[i];
        u64 bin = (u64)bins[(u64)r * n_cols + f];
        float2 p = gpair[r];
        double2 a = h[bin];
        a.x = a.x + (double)p.x;
        a.y = a.y + (double)p.y;
        h[bin] = a;
    }
}

// The CSR form: the thread finds its feature's entry (global bins
// `[fs, fe)`, at most one per row) by scanning the row, as the CPU's
// `find_bin` does.
template <typename B>
__device__ void hist_chain_csr(const B* __restrict__ bins,
                               const u64* __restrict__ row_ptr,
                               const u64* __restrict__ feature_bins,
                               u64 n_cols, const u32* __restrict__ rows,
                               u64 n, u64 seg_rows, u64 segs,
                               const float2* __restrict__ gpair,
                               double2* __restrict__ partials,
                               u64 total_bins) {
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= segs * n_cols) return;
    u64 seg = t / n_cols;
    u64 f = t - seg * n_cols;
    u64 fs = feature_bins[f];
    u64 fe = feature_bins[f + 1];
    u64 begin = seg * seg_rows;
    u64 end = begin + seg_rows < n ? begin + seg_rows : n;
    double2* h = partials + seg * total_bins;
    for (u64 i = begin; i < end; ++i) {
        u32 r = rows[i];
        for (u64 k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
            u64 bin = (u64)bins[k];
            if (bin >= fs && bin < fe) {
                float2 p = gpair[r];
                double2 a = h[bin];
                a.x = a.x + (double)p.x;
                a.y = a.y + (double)p.y;
                h[bin] = a;
                break;
            }
        }
    }
}

#define INSTANTIATE(B, suffix)                                                \
    extern "C" __global__ void hist_units_dense_##suffix(                     \
        const B* bins, u64 n_cols, const u32* rows, u64 n, u64 seg_rows,      \
        const longlong2* units, u64* partials, u64 total_bins) {              \
        hist_units_dense<B>(bins, n_cols, rows, n, seg_rows, units, partials, \
                            total_bins);                                      \
    }                                                                         \
    extern "C" __global__ void hist_units_csr_##suffix(                       \
        const B* bins, const u64* row_ptr, const u32* rows, u64 n,            \
        u64 seg_rows, const longlong2* units, u64* partials,                  \
        u64 total_bins) {                                                     \
        hist_units_csr<B>(bins, row_ptr, rows, n, seg_rows, units, partials,  \
                          total_bins);                                        \
    }                                                                         \
    extern "C" __global__ void hist_chain_dense_##suffix(                     \
        const B* bins, u64 n_cols, const u32* rows, u64 n, u64 seg_rows,      \
        u64 segs, const float2* gpair, double2* partials, u64 total_bins) {   \
        hist_chain_dense<B>(bins, n_cols, rows, n, seg_rows, segs, gpair,     \
                            partials, total_bins);                            \
    }                                                                         \
    extern "C" __global__ void hist_chain_csr_##suffix(                       \
        const B* bins, const u64* row_ptr, const u64* feature_bins,           \
        u64 n_cols, const u32* rows, u64 n, u64 seg_rows, u64 segs,           \
        const float2* gpair, double2* partials, u64 total_bins) {             \
        hist_chain_csr<B>(bins, row_ptr, feature_bins, n_cols, rows, n,       \
                          seg_rows, segs, gpair, partials, total_bins);       \
    }

INSTANTIATE(u16, u16)
INSTANTIATE(u32, u32)

// Integer partials of `segs` segments into the `f64` histogram, per bin in
// segment order: `out = value(p[0])` (when `init`) and then
// `out = out + value(p[s])`, the CPU's copy-then-add reduction. Each value
// `(f64)K * 2^grain` is exact (|K| <= 2^53), and every add is the CPU's.
extern "C" __global__ void reduce_units(const u64* __restrict__ partials,
                                        u64 segs, u64 total_bins,
                                        double grad_value, double hess_value,
                                        int init, double2* __restrict__ out) {
    GRID_STRIDE(b, total_bins) {
        double g, h;
        u64 s = 0;
        if (init) {
            g = (double)(i64)partials[b * 2] * grad_value;
            h = (double)(i64)partials[b * 2 + 1] * hess_value;
            s = 1;
        } else {
            g = out[b].x;
            h = out[b].y;
        }
        for (; s < segs; ++s) {
            const u64* p = partials + (s * total_bins + b) * 2;
            g = g + (double)(i64)p[0] * grad_value;
            h = h + (double)(i64)p[1] * hess_value;
        }
        out[b] = make_double2(g, h);
    }
}

// `f64` partials of `segs` segments into the histogram, per bin in
// segment order, as `reduce_units` (the first copied when `init`).
extern "C" __global__ void reduce_chains(const double2* __restrict__ partials,
                                         u64 segs, u64 total_bins, int init,
                                         double2* __restrict__ out) {
    GRID_STRIDE(b, total_bins) {
        double2 a;
        u64 s = 0;
        if (init) {
            a = partials[b];
            s = 1;
        } else {
            a = out[b];
        }
        for (; s < segs; ++s) {
            double2 p = partials[s * total_bins + b];
            a.x = a.x + p.x;
            a.y = a.y + p.y;
        }
        out[b] = a;
    }
}
