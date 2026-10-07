// Appended to kernels.cu so both searches use exactly the same scorer.
// A task is four u64s: (request, feature), (slot, direction), result index,
// and the first bin of its reusable sorting workspace.
#define CAT_SET_BINS 63
#define NODE_SCAN_WORDS 7

__device__ __forceinline__ u32 cat_request(const u64* task) { return (u32)task[0]; }
__device__ __forceinline__ u32 cat_feature(const u64* task) { return (u32)(task[0] >> 32); }

extern "C" __global__ void category_keys(
    const double2* __restrict__ pool, const u32* __restrict__ feature_first,
    u64 total_bins, const u64* __restrict__ tasks, u32 n_tasks,
    double lambda, double alpha, double max_delta_step, double min_child_weight,
    float* __restrict__ keys, u32* __restrict__ order, u32* __restrict__ meta) {
    for (u32 t = blockIdx.x; t < n_tasks; t += gridDim.x) {
        const u64* task = tasks + 4ull * t;
        const u32 f = cat_feature(task), first = feature_first[f];
        const u32 len = feature_first[f + 1] - first;
        if (len < 4) continue;
        const double2* bins = pool + (u64)(u32)task[1] * total_bins + first;
        for (u64 i = threadIdx.x; i < len; i += blockDim.x) {
            const double2 s = bins[i];
            float key = 0.0f;
            // CalcWeightCat checks min_child_weight before CalcWeight's
            // non-positive-Hessian case, and does not apply node bounds.
            if (!(s.y < min_child_weight) && !(s.y <= 0.0)) {
                const double threshold = s.x > alpha ? s.x - alpha
                    : (s.x < -alpha ? s.x + alpha : 0.0);
                double weight = -threshold / (s.y + lambda);
                if (max_delta_step != 0.0 && fabs(weight) > max_delta_step)
                    weight = copysign(max_delta_step, weight);
                key = __double2float_rn(weight);
            }
            keys[task[3] + i] = key;
            order[task[3] + i] = (u32)i;
            // One-hot does not compare category weights at all.
            if (len >= 4 && !isfinite(key)) atomicExch(meta + 4ull * task[2], 3u);
        }
    }
}

// Stable parallel merge sort in global memory, with no category-count cap.
// A left-run item counts strictly smaller right keys; a right-run item
// counts smaller-or-equal left keys. Thus equal weights (including +/-0)
// retain the original ascending bin order at every merge width.
extern "C" __global__ void category_merge(
    const u32* __restrict__ feature_first, const u64* __restrict__ tasks,
    u32 n_tasks, u64 width, const float* __restrict__ keys,
    const u32* __restrict__ source, u32* __restrict__ dest,
    const u32* __restrict__ meta) {
    for (u32 t = blockIdx.x; t < n_tasks; t += gridDim.x) {
        const u64* task = tasks + 4ull * t;
        if (meta[4ull * task[2]] == 3u) continue;
        const u32 f = cat_feature(task);
        const u64 len = (u64)feature_first[f + 1] - feature_first[f], at = task[3];
        // One-hot search uses bin order directly and never compares keys.
        // In particular, do not merge non-total NaN keys for that path.
        if (len < 4) continue;
        for (u64 i = threadIdx.x; i < len; i += blockDim.x) {
            const u64 start = (i / (2 * width)) * (2 * width);
            const u64 middle = min(start + width, len), end = min(start + 2 * width, len);
            const bool left = i < middle;
            const u64 own = left ? start : middle;
            u64 lo = left ? middle : start, hi = left ? end : middle;
            const u64 opposite = lo;
            const u32 index = source[at + i];
            const float key = keys[at + index];
            while (lo < hi) {
                const u64 m = lo + (hi - lo) / 2;
                const float other = keys[at + source[at + m]];
                if (left ? other < key : other <= key) lo = m + 1;
                else hi = m;
            }
            dest[at + start + (i - own) + (lo - opposite)] = index;
        }
    }
}

extern "C" __global__ void __launch_bounds__(32 * SCAN_WARPS) scan_categorical(
    const double2* __restrict__ pool, const u32* __restrict__ feature_first,
    u64 total_bins, const u64* __restrict__ tasks, u64 n_tasks,
    const double2* __restrict__ totals, const float* __restrict__ params,
    double lambda, double alpha, double max_delta_step, double min_child_weight,
    const u32* __restrict__ order, u32* __restrict__ meta,
    double2* __restrict__ children, u32* __restrict__ sets) {
    __shared__ double2 chain[SCAN_WARPS][2 * CAT_SET_BINS];
    const u32 lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const u64 warps = (u64)gridDim.x * SCAN_WARPS;
    for (u64 t = (u64)blockIdx.x * SCAN_WARPS + warp; t < n_tasks; t += warps) {
        const u64* task = tasks + 4 * t;
        const u64 result = task[2], workspace = task[3];
        if (meta[4 * result] == 3u) continue;
        const u32 request = cat_request(task), f = cat_feature(task);
        const u32 first = feature_first[f], len = feature_first[f + 1] - first;
        const double2* bins = pool + (u64)(u32)task[1] * total_bins + first;
        const double2 total = totals[request];
        ScanReg reg;
        reg.lambda = lambda;
        reg.alpha = alpha;
        reg.max_delta_step = max_delta_step;
        reg.min_child_weight = min_child_weight;
        reg.root_gain = params[3 * request];
        reg.lower = params[3 * request + 1];
        reg.upper = params[3 * request + 2];
        reg.dir = (int)(u32)(task[1] >> 32);
        const bool onehot = len < 4;
        const u32 depth = min(len, 64u), steps = onehot ? 0u : depth - 1;
        double mg = 0.0, mh = 0.0;
        if (lane == 0) {
            if (onehot) {
                double g = 0.0, h = 0.0;
                for (u32 i = 0; i < len; ++i) {
                    g = g + bins[i].x;
                    h = h + bins[i].y;
                }
                mg = total.x - g;
                mh = total.y - h;
            } else {
                for (u32 backward = 0; backward < 2; ++backward) {
                    double g = 0.0, h = 0.0;
                    for (u32 step = 0; step < steps; ++step) {
                        const u32 index = order[workspace + (backward ? len - 1 - step : step)];
                        const double2 bin = bins[index];
                        g = g + bin.x;
                        h = h + bin.y;
                        chain[warp][backward * steps + step] = make_double2(g, h);
                    }
                }
            }
        }
        mg = __shfl_sync(FULL_MASK, mg, 0);
        mh = __shfl_sync(FULL_MASK, mh, 0);
        __syncwarp();
        float best = 0.0f;
        u32 position = ~0u;
        double lg = 0.0, lh = 0.0, rg = 0.0, rh = 0.0;
        bool nan = false;
        const u32 count = onehot ? 2 * len : 2 * steps;
        for (u32 candidate = lane; candidate < count; candidate += 32) {
            double2 left, right;
            if (onehot) {
                right = bins[candidate / 2];
                if (candidate & 1) {
                    right.x = right.x + mg;
                    right.y = right.y + mh;
                }
                left = make_double2(total.x - right.x, total.y - right.y);
            } else {
                const double2 acc = chain[warp][candidate];
                if (candidate < steps) {
                    right = acc;
                    left = make_double2(total.x - acc.x, total.y - acc.y);
                } else {
                    left = acc;
                    right = make_double2(total.x - acc.x, total.y - acc.y);
                }
            }
            const float loss = scan_score(left.x, left.y, right.x, right.y, reg);
            if (isnan(loss)) nan = true;
            else if (isfinite(loss) && loss > best) {
                best = loss;
                position = candidate;
                lg = left.x; lh = left.y; rg = right.x; rh = right.y;
            }
        }
        for (int shift = 16; shift > 0; shift >>= 1) {
            const float other_best = __shfl_down_sync(FULL_MASK, best, shift);
            const u32 other_position = __shfl_down_sync(FULL_MASK, position, shift);
            const double other_lg = __shfl_down_sync(FULL_MASK, lg, shift);
            const double other_lh = __shfl_down_sync(FULL_MASK, lh, shift);
            const double other_rg = __shfl_down_sync(FULL_MASK, rg, shift);
            const double other_rh = __shfl_down_sync(FULL_MASK, rh, shift);
            if (other_best > best || (other_best == best && other_position < position)) {
                best = other_best; position = other_position;
                lg = other_lg; lh = other_lh; rg = other_rg; rh = other_rh;
            }
        }
        nan = __any_sync(FULL_MASK, nan);
        position = __shfl_sync(FULL_MASK, position, 0);
        const bool found = position != ~0u;
        const bool backward = !onehot && found && position >= steps;
        const u32 selected = !found ? 0u : (onehot ? 1u : position % steps + 1);
        if (lane == 0) {
            meta[4 * result] = nan ? 4u : (found ? 1u : 0u);
            meta[4 * result + 1] = selected;
            meta[4 * result + 2] = __float_as_uint(best);
            // Tree children are XGBoost's children swapped.
            const bool default_left = onehot ? (position & 1) != 0 : backward;
            meta[4 * result + 3] = (default_left ? 1u : 0u) | (backward ? 2u : 0u);
            children[2 * result] = make_double2(rg, rh);
            children[2 * result + 1] = make_double2(lg, lh);
        }
        if (found && !nan) {
            for (u32 i = lane; i < selected; i += 32)
                sets[CAT_SET_BINS * result + i] = onehot ? position / 2
                    : order[workspace + (backward ? len - selected + i : i)];
        }
        __syncwarp();
    }
}

// Merge in the exact host feature order. A finite per-feature winner may
// be reduced; any NaN requires sequential host replay of the whole node.
// refs are (feature, kind, result), where kind 0 is numeric and 1 categorical.
// out_at bounds each node's packed header and optional 63-bin category set.
extern "C" __global__ void merge_scans(
    const u32* __restrict__ refs, const u32* __restrict__ first, u64 n_nodes,
    const u32* __restrict__ numeric_meta, const double2* __restrict__ numeric_acc,
    const u32* __restrict__ categorical_meta, const double2* __restrict__ categorical_children,
    const u32* __restrict__ categorical_sets, const u64* __restrict__ out_at,
    u64* __restrict__ out) {
    GRID_STRIDE(node, n_nodes) {
        float best = 0.0f;
        u32 best_feature = 0u, chosen = ~0u, fallback = 0u;
        for (u32 i = first[node]; i < first[node + 1]; ++i) {
            const u32 feature = refs[3ull * i], kind = refs[3ull * i + 1], index = refs[3ull * i + 2];
            const u32* meta = (kind ? categorical_meta : numeric_meta) + 4ull * index;
            if ((!kind && meta[0] == 2u) || (kind && meta[0] >= 3u)) {
                fallback = !kind ? 3u : (meta[0] == 3u ? 4u : 5u);
                break;
            }
            if (meta[0] != 1u) continue;
            const float loss = __uint_as_float(meta[2]);
            const bool replace = !isinf(loss) && (best_feature <= feature ? loss > best : !(best > loss));
            if (replace) { best = loss; best_feature = feature; chosen = i; }
        }
        u64* header = out + out_at[node];
        if (fallback || chosen == ~0u) {
            header[0] = fallback;
            continue;
        }
        const u32 kind = refs[3ull * chosen + 1], index = refs[3ull * chosen + 2];
        const u32* meta = (kind ? categorical_meta : numeric_meta) + 4ull * index;
        header[0] = (u64)best_feature << 32 | (kind ? 2u : 1u);
        header[1] = (u64)meta[3] << 32 | meta[1];
        header[2] = meta[2];
        const double2 a = kind ? categorical_children[2ull * index] : numeric_acc[index];
        header[3] = (u64)__double_as_longlong(a.x);
        header[4] = (u64)__double_as_longlong(a.y);
        if (kind) {
            const double2 b = categorical_children[2ull * index + 1];
            header[5] = (u64)__double_as_longlong(b.x);
            header[6] = (u64)__double_as_longlong(b.y);
            u32* selected = (u32*)(header + NODE_SCAN_WORDS);
            for (u32 i = 0; i < meta[1]; ++i)
                selected[i] = categorical_sets[CAT_SET_BINS * (u64)index + i];
        }
    }
}
