//! Parsing and validating untrusted compact bytes.

use super::bitstream::{BitReader, f16_to_f32, read_bits};
use super::{
    CompactModel, DefaultDirection, Dictionary, FeatureEntry, Layout, MAGIC, MAX_HEAP_DEPTH, Meta,
    PREFIX_BYTES, PackedTree, STREAM_PAD, Slot, ThresholdKind, VERSION, Widths, bits, format_error,
};
use crate::error::Result;
use crate::model::check_objective_width;

/// The `(metadata, bit stream)` parts of framed `bytes`, after checking the
/// magic, the version and the metadata length.
pub(super) fn split_frame(bytes: &[u8]) -> Result<(&[u8], &[u8])> {
    if bytes.len() < PREFIX_BYTES || &bytes[..MAGIC.len()] != MAGIC {
        return Err(format_error("invalid header"));
    }
    if bytes[MAGIC.len()] != VERSION {
        return Err(format_error(format!(
            "unsupported version {}",
            bytes[MAGIC.len()]
        )));
    }
    let meta_len = u32::from_le_bytes(
        bytes[MAGIC.len() + 1..PREFIX_BYTES]
            .try_into()
            .expect("four length bytes"),
    ) as usize;
    let meta_end = PREFIX_BYTES
        .checked_add(meta_len)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| format_error("truncated metadata"))?;
    Ok((&bytes[PREFIX_BYTES..meta_end], &bytes[meta_end..]))
}

pub(super) fn decode_threshold(raw: u32, kind: ThresholdKind, width: u32) -> Result<f32> {
    let v = match (kind, width) {
        (ThresholdKind::Unsigned, _) => raw as f32,
        (ThresholdKind::Signed, _) => {
            let shift = 32 - width;
            ((raw << shift) as i32 >> shift) as f32
        }
        (ThresholdKind::Float, 16) => f16_to_f32(raw as u16),
        (ThresholdKind::Float, 32) => f32::from_bits(raw),
        _ => return Err(format_error(format!("invalid {width}-bit float threshold"))),
    };
    if !v.is_finite() {
        return Err(format_error("thresholds must be finite"));
    }
    Ok(v)
}

/// The bit stream's leading metadata fields (layout item 1).
struct Header {
    n_features: usize,
    base_score: Vec<f32>,
    n_trees: usize,
    tree_weights: Option<Vec<f32>>,
    default_mode: DefaultDirection,
    n_used: usize,
    max_thresholds: u32,
    n_leaves: usize,
    heap_depth_bits: u32,
    preorder_nodes_bits: u32,
}

/// One feature map entry (layout item 2) before its dictionary is read.
struct MapSpec {
    input: usize,
    kind: ThresholdKind,
    /// Value width in bits.
    width: u32,
    /// Dictionary size.
    count: usize,
    /// Width of a categorical set's length (`0` for numeric features).
    len_bits: u32,
}

/// Read the [`Header`] and check it against the metadata `meta`.
fn read_header(r: &mut BitReader, meta: &Meta) -> Result<Header> {
    let n_features = r.read_usize(32)?;
    let n_outputs = r.read_usize(32)?;
    if n_features == 0 || n_outputs == 0 {
        return Err(format_error("feature and output counts must be positive"));
    }
    if meta.num_class >= 2 && n_outputs != meta.num_class {
        return Err(format_error("output count does not match num_class"));
    }
    check_objective_width(
        &meta.objective,
        meta.max_delta_step,
        meta.num_class,
        meta.n_targets,
        n_outputs,
    )?;
    let base_score = r.read_finite_f32s(n_outputs, "base score")?;
    let n_trees = r.read_usize(32)?;
    let per_iteration = n_outputs
        .checked_mul(meta.num_parallel_tree)
        .ok_or_else(|| format_error("trees per iteration overflow"))?;
    if !n_trees.is_multiple_of(per_iteration) {
        return Err(format_error(
            "tree count is not a multiple of the trees per iteration",
        ));
    }
    r.ensure_fits(n_trees, 1, "tree")?;
    let tree_weights = if r.read_bool()? {
        Some(r.read_finite_f32s(n_trees, "tree weight")?)
    } else {
        None
    };
    let default_mode = DefaultDirection::try_from(r.read(2)?)?;
    let header = Header {
        n_features,
        base_score,
        n_trees,
        tree_weights,
        default_mode,
        n_used: r.read_usize(32)?,
        max_thresholds: r.read(32)?,
        n_leaves: r.read_usize(32)?,
        heap_depth_bits: r.read(6)?,
        preorder_nodes_bits: r.read(6)?,
    };
    if header.n_used > n_features || (header.n_used == 0) != (header.max_thresholds == 0) {
        return Err(format_error("inconsistent feature map counts"));
    }
    Ok(header)
}

/// Read the feature & threshold map (layout item 2).
fn read_feature_map(r: &mut BitReader, header: &Header, widths: Widths) -> Result<Vec<MapSpec>> {
    let input_bits = bits(header.n_features as u64 - 1);
    r.ensure_fits(header.n_used, 5, "used feature")?;
    let mut map: Vec<MapSpec> = Vec::with_capacity(header.n_used);
    for _ in 0..header.n_used {
        let input = r.read_usize(input_bits)?;
        let kind = ThresholdKind::try_from(r.read(2)?)?;
        let code = r.read(3)?;
        if code > 5 {
            return Err(format_error("threshold width code exceeds 5"));
        }
        let count = r
            .read_usize(widths.threshold_ref)?
            .checked_add(1)
            .ok_or_else(|| format_error("dictionary size overflow"))?;
        if count > header.max_thresholds as usize {
            return Err(format_error("dictionary larger than its declared maximum"));
        }
        let len_bits = if kind == ThresholdKind::Categorical {
            r.read(6)?
        } else {
            0
        };
        if kind == ThresholdKind::Categorical && !(1..=32).contains(&len_bits) {
            return Err(format_error("category set lengths need 1 to 32 bits"));
        }
        if input >= header.n_features || map.last().is_some_and(|prev| prev.input >= input) {
            return Err(format_error("feature map entries must ascend"));
        }
        map.push(MapSpec {
            input,
            kind,
            width: 1u32 << code,
            count,
            len_bits,
        });
    }
    Ok(map)
}

/// Read each map entry's dictionary (layout item 3, the global thresholds).
fn read_dictionaries(r: &mut BitReader, map: &[MapSpec]) -> Result<Vec<FeatureEntry>> {
    let mut features = Vec::with_capacity(map.len());
    for spec in map {
        let &MapSpec {
            input,
            kind,
            width,
            count,
            len_bits,
        } = spec;
        let dict = if kind == ThresholdKind::Categorical {
            r.ensure_fits(count, len_bits as usize, "category set")?;
            let mut sets = Vec::with_capacity(count);
            for _ in 0..count {
                let len = r.read_usize(len_bits)?;
                r.ensure_fits(len, width as usize, "category")?;
                let set = (0..len)
                    .map(|_| r.read(width))
                    .collect::<Result<Vec<_>>>()?;
                if !set.is_sorted() {
                    return Err(format_error("category sets must be sorted"));
                }
                sets.push(set);
            }
            Dictionary::Categorical(sets)
        } else {
            r.ensure_fits(count, width as usize, "threshold")?;
            let values = (0..count)
                .map(|_| decode_threshold(r.read(width)?, kind, width))
                .collect::<Result<Vec<_>>>()?;
            Dictionary::Numeric(values)
        };
        features.push(FeatureEntry { input, dict });
    }
    Ok(features)
}

/// Read each tree's layout (layout item 5) and record its slot offset,
/// skipping its slots; nothing but zero padding may follow the last tree.
fn read_tree_layouts(
    r: &mut BitReader,
    header: &Header,
    widths: Widths,
) -> Result<Vec<PackedTree>> {
    let mut trees = Vec::with_capacity(header.n_trees);
    for _ in 0..header.n_trees {
        let layout = if r.read_bool()? {
            let nodes = r.read(header.preorder_nodes_bits)?.checked_add(1);
            Layout::Preorder {
                nodes: nodes.ok_or_else(|| format_error("preorder node count overflow"))?,
            }
        } else {
            let depth = r.read(header.heap_depth_bits)?;
            if depth > MAX_HEAP_DEPTH {
                return Err(format_error(format!(
                    "heap tree deeper than {MAX_HEAP_DEPTH}"
                )));
            }
            Layout::Heap {
                depth,
                complete: r.read_bool()?,
            }
        };
        let total = layout.total_bits(widths);
        if total > r.remaining() as u128 {
            return Err(format_error("truncated tree"));
        }
        trees.push(PackedTree {
            layout,
            offset: r.pos,
        });
        r.pos += total as usize;
    }
    if r.remaining() >= 8 || read_bits(r.stream, r.pos, r.remaining() as u32) != 0 {
        return Err(format_error("trailing data after the last tree"));
    }
    Ok(trees)
}

impl CompactModel {
    /// Parse bytes written by [`CompactModel::encode`] /
    /// [`BoostedModel::to_compact_bytes`](crate::model::BoostedModel::to_compact_bytes).
    /// Every reference is validated, so prediction on a parsed model cannot
    /// index out of bounds.
    pub fn decode(bytes: impl AsRef<[u8]>) -> Result<Self> {
        let bytes = bytes.as_ref();
        let (meta, stream) = split_frame(bytes)?;
        let meta = Meta::decode(meta)?;
        if meta.n_targets == 0 {
            return Err(format_error("n_targets must be positive"));
        }
        if meta.num_parallel_tree == 0 {
            return Err(format_error("num_parallel_tree must be positive"));
        }

        let mut padded = Vec::with_capacity(bytes.len() + STREAM_PAD);
        padded.extend_from_slice(bytes);
        padded.resize(bytes.len() + STREAM_PAD, 0);
        let too_large = || format_error("model is too large");
        let mut r = BitReader {
            stream: &padded,
            len: bytes.len().checked_mul(8).ok_or_else(too_large)?,
            pos: (bytes.len() - stream.len())
                .checked_mul(8)
                .ok_or_else(too_large)?,
        };
        let header = read_header(&mut r, &meta)?;
        if let Some(shrinkage) = &meta.shrinkage {
            let per = header.base_score.len() * meta.num_parallel_tree;
            shrinkage
                .validate(header.n_trees / per, header.base_score.len())
                .map_err(|e| format_error(e.to_string()))?;
            let weight = |t: usize| header.tree_weights.as_ref().map_or(1.0, |w| w[t]);
            if !shrinkage.matches(per, weight, &header.base_score) {
                return Err(format_error(
                    "the tree weights and intercepts do not match the shrinkage record",
                ));
            }
        }
        let widths = Widths::new(
            header.n_used,
            header.max_thresholds as usize,
            header.n_leaves,
            header.default_mode,
        );
        let map = read_feature_map(&mut r, &header, widths)?;
        let features = read_dictionaries(&mut r, &map)?;
        let leaf_values = r.read_finite_f32s(header.n_leaves, "leaf value")?;
        let trees = read_tree_layouts(&mut r, &header, widths)?;
        let model = CompactModel {
            bytes: padded,
            meta,
            n_features: header.n_features,
            base_score: header.base_score,
            tree_weights: header.tree_weights,
            default_mode: header.default_mode,
            features,
            leaf_values,
            widths,
            trees,
        };
        for t in 0..model.trees.len() {
            model.validate_tree(t)?;
        }
        Ok(model)
    }

    /// Check every reachable slot of tree `t`: references inside the
    /// dictionaries, preorder children inside the tree and after their
    /// parent (so every walk terminates).
    fn validate_tree(&self, t: usize) -> Result<()> {
        let tree = self.trees[t];
        match tree.layout {
            Layout::Preorder { nodes } => self.validate_preorder(t, tree, nodes),
            Layout::Heap {
                depth,
                complete: true,
            } => self.validate_complete_heap(t, tree, depth),
            Layout::Heap { depth, .. } => self.validate_incomplete_heap(t, tree, depth),
        }
    }

    /// Every node of preorder tree `t` (`nodes` of them): its references,
    /// and a split's children inside the tree and after it.
    fn validate_preorder(&self, t: usize, tree: PackedTree, nodes: u32) -> Result<()> {
        for i in 0..nodes {
            let slot = self.slot(tree, i, false);
            self.check_slot_refs(t, &slot)?;
            if let Slot::Split { right, .. } = slot
                && (i + 1 >= nodes
                    || right < 2
                    || u64::from(i) + u64::from(right) >= u64::from(nodes))
            {
                return Err(format_error(format!("tree {t}: child outside the tree")));
            }
        }
        Ok(())
    }

    /// Every slot of complete heap tree `t` of `depth`.
    fn validate_complete_heap(&self, t: usize, tree: PackedTree, depth: u32) -> Result<()> {
        // Every internal slot is a split and every bottom slot a leaf, so all
        // are reachable. A zero-width row decodes identically in every slot
        // and costs no input bits, so one check covers it; this keeps
        // validation work bounded by the input size.
        let internal = (1u64 << depth) - 1;
        let splits = if self.widths.split() == 0 {
            internal.min(1)
        } else {
            internal
        };
        for i in 0..splits {
            self.check_slot_refs(t, &self.slot(tree, i as u32, false))?;
        }
        let leaves = if self.widths.leaf_ref == 0 {
            1
        } else {
            1u64 << depth
        };
        for j in 0..leaves {
            self.check_slot_refs(t, &self.slot(tree, (internal + j) as u32, true))?;
        }
        Ok(())
    }

    /// Every slot reachable from the root of incomplete heap tree `t` of
    /// `depth`.
    fn validate_incomplete_heap(&self, t: usize, tree: PackedTree, depth: u32) -> Result<()> {
        // Flagged slots take at least one bit each, and every visited bottom
        // slot is a child of a visited split, so this walk is bounded by the
        // tree's encoded size.
        let first_leaf = (1u64 << depth) - 1;
        let mut stack = vec![0u64];
        while let Some(i) = stack.pop() {
            let slot = self.slot(tree, i as u32, i >= first_leaf);
            if let Slot::Split { .. } = slot {
                stack.extend([2 * i + 1, 2 * i + 2]);
            }
            self.check_slot_refs(t, &slot)?;
        }
        Ok(())
    }

    /// A slot of tree `t` references a stored leaf value, or a used feature
    /// and a threshold inside that feature's dictionary.
    fn check_slot_refs(&self, t: usize, slot: &Slot) -> Result<()> {
        match *slot {
            Slot::Leaf(leaf) => {
                if leaf as usize >= self.leaf_values.len() {
                    return Err(format_error(format!(
                        "tree {t}: leaf reference out of range"
                    )));
                }
            }
            Slot::Split {
                feature, threshold, ..
            } => {
                let entry = self.features.get(feature as usize).ok_or_else(|| {
                    format_error(format!("tree {t}: feature reference out of range"))
                })?;
                if threshold as usize >= entry.dict.len() {
                    return Err(format_error(format!(
                        "tree {t}: threshold reference out of range"
                    )));
                }
            }
        }
        Ok(())
    }
}
