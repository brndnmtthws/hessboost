// The CPU compact forest's walk and tree-order f32 additions. Compiled with
// --fmad=false and --ftz=false; transforms stay on the CPU.
using u32 = unsigned int;
using u64 = unsigned long long;

struct Node16 { u32 slot, key, left, aux; };
struct Tree { u32 root, output, vector, pad; };

__device__ __forceinline__ u32 value_key(float value) {
    if (isnan(value)) return 0;
    // Canonicalize both zeros without flushing subnormals.
    const u32 bits = value == 0.0f ? 0 : __float_as_uint(value);
    return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ u32 category(float value) {
    // Rust's `as u32`: truncate, saturating negatives and overflow.
    if (value <= 0.0f) return 0;
    if (value >= 4294967296.0f) return 0xffffffffu;
    return __float2uint_rz(value);
}

extern "C" __global__ void predict16(
    const Node16* nodes, const u32* categories, const float* vectors,
    const Tree* trees, const float* rows, float* margins,
    u32 n_rows, u32 n_cols, u32 outputs, u32 tree_begin, u32 tree_end) {
    for (u32 row = blockIdx.x * blockDim.x + threadIdx.x; row < n_rows;
         row += blockDim.x * gridDim.x) {
        const float* input = rows + u64(row) * n_cols;
        float* out = margins + u64(row) * outputs;
        for (u32 t = tree_begin; t < tree_end; ++t) {
            const Tree tree = trees[t];
            u32 id = tree.root;
            Node16 node = nodes[id];
            while (node.left != id) {
                float value = input[node.slot / 32];
                if (node.aux & 1) {
                    bool left = (node.aux & 2) != 0;
                    if (!isnan(value)) {
                        const u32 code = category(value);
                        left = false;
                        for (u32 c = node.key; c < (node.aux >> 2); ++c) {
                            if (categories[c] == code) { left = true; break; }
                        }
                    }
                    id = node.left + u32(!left);
                } else {
                    if (node.slot & 16) value = -value;
                    id = node.left + u32(value_key(value) > node.key);
                }
                node = nodes[id];
            }
            if (tree.vector) {
                for (u32 k = 0; k < outputs; ++k)
                    out[k] = __fadd_rn(out[k], vectors[u64(node.aux) + k]);
            } else {
                out[tree.output] = __fadd_rn(out[tree.output], __uint_as_float(node.aux));
            }
        }
    }
}

extern "C" __global__ void predict8(
    const uint2* nodes, const u32* roots, const float* rows, float* margins,
    u32 n_rows, u32 n_cols, u32 tree_begin, u32 tree_end) {
    for (u32 row = blockIdx.x * blockDim.x + threadIdx.x; row < n_rows;
         row += blockDim.x * gridDim.x) {
        const float* input = rows + u64(row) * n_cols;
        float out = margins[row];
        for (u32 t = tree_begin; t < tree_end; ++t) {
            const u32 root = roots[t];
            uint2 node = nodes[root];
            while (!(node.y & 0x80000000u)) {
                float value = input[(node.y >> 15) & 0x7fffu];
                if (node.y & 0x40000000u) value = -value;
                const u32 next = root + (node.y & 0x7fffu)
                    + u32(value > __uint_as_float(node.x));
                node = nodes[next];
            }
            out = __fadd_rn(out, __uint_as_float(node.x));
        }
        margins[row] = out;
    }
}
