//! Serializing a [`BoostedModel`] in the compact layout.

use super::bitstream::{BitWriter, f16_exact};
use super::{
    DefaultDirection, Dictionary, Layout, MAGIC, MAX_HEAP_DEPTH, Meta, PREFIX_BYTES, ThresholdKind,
    VERSION, Widths, bits, format_error,
};
use crate::error::Result;
use crate::model::BoostedModel;
use crate::tree::{Node, RegTree};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// The serialized model: magic, version and length prefix, then the
/// metadata section table and the bit `stream`.
pub(super) fn frame(meta: &Meta, stream: &[u8]) -> Vec<u8> {
    let table = meta.section_table();
    let mut bytes = Vec::with_capacity(PREFIX_BYTES + table.encoded_len() + stream.len());
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION);
    bytes.extend_from_slice(&[0; 4]);
    table.finish(&mut bytes);
    let meta_len = (bytes.len() - PREFIX_BYTES) as u32;
    bytes[MAGIC.len() + 1..PREFIX_BYTES].copy_from_slice(&meta_len.to_le_bytes());
    bytes.extend_from_slice(stream);
    bytes
}

/// The integer `v` represents exactly (same bits back), if any.
fn exact_integer(v: f32) -> Option<i64> {
    let i = v as i64;
    ((i as f32).to_bits() == v.to_bits()).then_some(i)
}

/// Narrowest `(kind, width code)` that reproduces every value of `values`.
pub(super) fn numeric_encoding(values: &[f32]) -> (ThresholdKind, u32) {
    let ints: Option<Vec<i64>> = values.iter().map(|&v| exact_integer(v)).collect();
    for code in 0..=5u32 {
        let w = 1u32 << code;
        if let Some(ints) = &ints {
            if ints.iter().all(|&i| i >= 0 && i < (1i64 << w)) {
                return (ThresholdKind::Unsigned, code);
            }
            let half = 1i64 << (w - 1);
            if ints.iter().all(|&i| (-half..half).contains(&i)) {
                return (ThresholdKind::Signed, code);
            }
        }
        if w == 16 && values.iter().all(|&v| f16_exact(v).is_some()) {
            return (ThresholdKind::Float, code);
        }
    }
    (ThresholdKind::Float, 5)
}

pub(super) fn encode_threshold(v: f32, kind: ThresholdKind, width: u32) -> u64 {
    match (kind, width) {
        (ThresholdKind::Unsigned, _) => v as u64,
        (ThresholdKind::Signed, _) => (v as i64 as u64) & ((1u64 << width) - 1),
        (_, 16) => u64::from(f16_exact(v).expect("width chosen for exact binary16")),
        _ => u64::from(v.to_bits()),
    }
}

/// Dictionary under construction for one used feature.
enum Collected {
    Numeric(BTreeSet<u32>),
    Categorical(BTreeSet<Vec<u32>>),
}

/// A feature map entry under construction: encoding and sorted dictionary.
struct MapEntry {
    input: u32,
    kind: ThresholdKind,
    code: u32,
    len_bits: u32,
    dict: Dictionary,
}

/// `cats` sorted and deduplicated into `scratch`: the set identity under
/// which the dictionaries store categorical splits
/// ([`canonical_categories`](crate::tree::reuse::canonical_categories)).
fn canonical<'s>(cats: &[u32], scratch: &'s mut Vec<u32>) -> &'s [u32] {
    scratch.clear();
    scratch.extend_from_slice(cats);
    scratch.sort_unstable();
    scratch.dedup();
    scratch
}

/// Depth of the deepest leaf (root = 0) and whether every leaf sits there;
/// `stack` is scratch.
fn depth_and_completeness(tree: &RegTree, stack: &mut Vec<(usize, u32)>) -> (u32, bool) {
    stack.clear();
    stack.push((0, 0));
    let (mut shallowest, mut deepest) = (u32::MAX, 0);
    while let Some((id, d)) = stack.pop() {
        let node = tree.node(id);
        if node.is_leaf() {
            shallowest = shallowest.min(d);
            deepest = deepest.max(d);
        } else {
            stack.push((node.left as usize, d + 1));
            stack.push((node.right as usize, d + 1));
        }
    }
    (deepest, shallowest >= deepest)
}

/// Node ids in preorder (node, left subtree, right subtree) into `order`;
/// `stack` is scratch.
fn preorder(tree: &RegTree, order: &mut Vec<usize>, stack: &mut Vec<usize>) {
    order.clear();
    stack.clear();
    stack.push(0);
    while let Some(id) = stack.pop() {
        order.push(id);
        let node = tree.node(id);
        if !node.is_leaf() {
            stack.push(node.right as usize);
            stack.push(node.left as usize);
        }
    }
}

/// Refuse the models the compact layout cannot express.
fn check_encodable(model: &BoostedModel) -> Result<()> {
    if model.linear().is_some() {
        return Err(format_error(
            "gblinear models have no trees to store; use the native format",
        ));
    }
    if model
        .trees()
        .iter()
        .any(|tree| tree.linear_leaves().is_some())
    {
        return Err(format_error(
            "linear-leaf trees (`linear_tree`) have no compact encoding; use the native format",
        ));
    }
    if model.has_vector_leaves() {
        return Err(format_error(
            "vector-leaf trees (`multi_output_tree`) have no compact encoding; use the native format",
        ));
    }
    Ok(())
}

/// The dictionaries of `trees`: each used feature's thresholds or sets, the
/// distinct leaf values in first-seen order with their index, and the
/// default-direction mode.
struct Collection {
    features: BTreeMap<u32, Collected>,
    leaf_index: HashMap<u32, u32>,
    leaf_values: Vec<f32>,
    default_mode: DefaultDirection,
}

fn collect(trees: &[RegTree]) -> Result<Collection> {
    let mut features: BTreeMap<u32, Collected> = BTreeMap::new();
    let mut leaf_index: HashMap<u32, u32> = HashMap::new();
    let mut leaf_values: Vec<f32> = Vec::new();
    let (mut any_left, mut any_right) = (false, false);
    let mut cats = Vec::new();
    for tree in trees {
        for node in tree.nodes() {
            if node.is_leaf() {
                leaf_index
                    .entry(node.leaf_value.to_bits())
                    .or_insert_with(|| {
                        leaf_values.push(node.leaf_value);
                        (leaf_values.len() - 1) as u32
                    });
                continue;
            }
            any_left |= node.default_left;
            any_right |= !node.default_left;
            let entry = features.entry(node.split_feature).or_insert_with(|| {
                if node.is_categorical {
                    Collected::Categorical(BTreeSet::new())
                } else {
                    Collected::Numeric(BTreeSet::new())
                }
            });
            match (entry, node.is_categorical) {
                (Collected::Numeric(set), false) => {
                    set.insert(node.split_cond.to_bits());
                }
                (Collected::Categorical(sets), true) => {
                    let set = canonical(tree.node_categories(node), &mut cats);
                    if !sets.contains(set) {
                        sets.insert(set.to_vec());
                    }
                }
                _ => {
                    return Err(format_error(format!(
                        "feature {} has both numeric and categorical splits",
                        node.split_feature
                    )));
                }
            }
        }
    }
    let default_mode = match (any_left, any_right) {
        (true, true) => DefaultDirection::PerNode,
        (false, true) => DefaultDirection::AllRight,
        _ => DefaultDirection::AllLeft,
    };
    Ok(Collection {
        features,
        leaf_index,
        leaf_values,
        default_mode,
    })
}

/// Feature map entries in ascending input order, with sorted dictionaries
/// and their narrowest encodings.
fn feature_map(features: BTreeMap<u32, Collected>) -> Vec<MapEntry> {
    features
        .into_iter()
        .map(|(input, dict)| match dict {
            Collected::Numeric(set) => {
                let mut values: Vec<f32> = set.iter().map(|&b| f32::from_bits(b)).collect();
                values.sort_by(f32::total_cmp);
                let (kind, code) = numeric_encoding(&values);
                MapEntry {
                    input,
                    kind,
                    code,
                    len_bits: 0,
                    dict: Dictionary::Numeric(values),
                }
            }
            Collected::Categorical(sets) => {
                let sets: Vec<Vec<u32>> = sets.into_iter().collect();
                let max_cat = sets.iter().flatten().copied().max().unwrap_or(0);
                let code = (0..=5u32)
                    .find(|&c| bits(u64::from(max_cat)) <= 1 << c)
                    .unwrap_or(5);
                let max_len = sets.iter().map(Vec::len).max().unwrap_or(0);
                MapEntry {
                    input,
                    kind: ThresholdKind::Categorical,
                    code,
                    len_bits: bits(max_len as u64).max(1),
                    dict: Dictionary::Categorical(sets),
                }
            }
        })
        .collect()
}

/// The smaller layout of each tree (heap only up to [`MAX_HEAP_DEPTH`]),
/// and the widths of the heap depths and preorder node counts.
fn choose_layouts(trees: &[RegTree], widths: Widths) -> (Vec<Layout>, u32, u32) {
    let mut stack = Vec::new();
    let layouts: Vec<Layout> = trees
        .iter()
        .map(|tree| {
            let preorder = Layout::Preorder {
                nodes: tree.num_nodes() as u32,
            };
            let (depth, complete) = depth_and_completeness(tree, &mut stack);
            let heap = Layout::Heap { depth, complete };
            if depth <= MAX_HEAP_DEPTH && heap.total_bits(widths) <= preorder.total_bits(widths) {
                heap
            } else {
                preorder
            }
        })
        .collect();
    let (mut heap_depth_bits, mut preorder_nodes_bits) = (0, 0);
    for layout in &layouts {
        match *layout {
            Layout::Heap { depth, .. } => {
                heap_depth_bits = heap_depth_bits.max(bits(u64::from(depth)));
            }
            Layout::Preorder { nodes } => {
                preorder_nodes_bits = preorder_nodes_bits.max(bits(u64::from(nodes) - 1));
            }
        }
    }
    (layouts, heap_depth_bits, preorder_nodes_bits)
}

/// Everything [`encode`] derives from the model before writing the bit
/// stream: the stored trees, their dictionaries and field widths, and each
/// tree's layout.
struct Encoding<'a> {
    model: &'a BoostedModel,
    trees: &'a [RegTree],
    map: Vec<MapEntry>,
    leaf_index: HashMap<u32, u32>,
    leaf_values: Vec<f32>,
    default_mode: DefaultDirection,
    max_thresholds: usize,
    widths: Widths,
    layouts: Vec<Layout>,
    heap_depth_bits: u32,
    preorder_nodes_bits: u32,
}

/// Per-tree buffers the tree writers reuse.
#[derive(Default)]
struct TreeScratch {
    heap: Vec<Option<usize>>,
    order: Vec<usize>,
    stack: Vec<usize>,
    position: Vec<u32>,
    cats: Vec<u32>,
}

impl<'a> Encoding<'a> {
    fn plan(model: &'a BoostedModel) -> Result<Self> {
        check_encodable(model)?;
        let trees = &model.trees()[..model.effective_num_trees()];
        let Collection {
            features,
            leaf_index,
            leaf_values,
            default_mode,
        } = collect(trees)?;
        let map = feature_map(features);
        let max_thresholds = map.iter().map(|e| e.dict.len()).max().unwrap_or(0);
        let widths = Widths::new(map.len(), max_thresholds, leaf_values.len(), default_mode);
        if u32::try_from(max_thresholds).is_err()
            || u32::try_from(leaf_values.len()).is_err()
            || u32::try_from(trees.len()).is_err()
            || u32::try_from(model.n_features()).is_err()
        {
            return Err(format_error("model too large for 32-bit counts"));
        }
        let (layouts, heap_depth_bits, preorder_nodes_bits) = choose_layouts(trees, widths);
        Ok(Encoding {
            model,
            trees,
            map,
            leaf_index,
            leaf_values,
            default_mode,
            max_thresholds,
            widths,
            layouts,
            heap_depth_bits,
            preorder_nodes_bits,
        })
    }

    /// Layout items 1 to 4: the metadata fields, the feature map, the
    /// dictionaries, and the leaf values.
    fn write_tables(&self, w: &mut BitWriter) {
        let model = self.model;
        let n_features = model.n_features();
        w.write(n_features as u64, 32);
        w.write(model.n_outputs() as u64, 32);
        for &b in model.base_scores() {
            w.write_f32(b);
        }
        let n_trees = self.trees.len();
        w.write(n_trees as u64, 32);
        let weighted = (0..n_trees).any(|t| model.tree_weight(t).to_bits() != 1.0f32.to_bits());
        w.write_bool(weighted);
        if weighted {
            for t in 0..n_trees {
                w.write_f32(model.tree_weight(t));
            }
        }
        w.write(u64::from(self.default_mode as u32), 2);
        w.write(self.map.len() as u64, 32);
        w.write(self.max_thresholds as u64, 32);
        w.write(self.leaf_values.len() as u64, 32);
        w.write(u64::from(self.heap_depth_bits), 6);
        w.write(u64::from(self.preorder_nodes_bits), 6);

        let input_bits = bits(n_features as u64 - 1);
        for e in &self.map {
            w.write(u64::from(e.input), input_bits);
            w.write(u64::from(e.kind as u32), 2);
            w.write(u64::from(e.code), 3);
            w.write(e.dict.len() as u64 - 1, self.widths.threshold_ref);
            if e.kind == ThresholdKind::Categorical {
                w.write(u64::from(e.len_bits), 6);
            }
        }
        for e in &self.map {
            let width = 1u32 << e.code;
            match &e.dict {
                Dictionary::Numeric(values) => {
                    for &v in values {
                        w.write(encode_threshold(v, e.kind, width), width);
                    }
                }
                Dictionary::Categorical(sets) => {
                    for set in sets {
                        w.write(set.len() as u64, e.len_bits);
                        for &c in set {
                            w.write(u64::from(c), width);
                        }
                    }
                }
            }
        }
        for &v in &self.leaf_values {
            w.write_f32(v);
        }
    }

    /// Layout item 5: every tree in its chosen layout.
    fn write_trees(&self, w: &mut BitWriter) {
        let mut scratch = TreeScratch::default();
        for (tree, &layout) in self.trees.iter().zip(&self.layouts) {
            match layout {
                Layout::Heap { depth, complete } => {
                    self.write_heap_tree(w, tree, depth, complete, &mut scratch);
                }
                Layout::Preorder { nodes } => {
                    self.write_preorder_tree(w, tree, nodes, &mut scratch);
                }
            }
        }
    }

    /// A leaf's reference, or an internal node's split fields.
    fn write_node(&self, w: &mut BitWriter, tree: &RegTree, node: &Node, cats: &mut Vec<u32>) {
        let widths = self.widths;
        if node.is_leaf() {
            w.write(
                u64::from(self.leaf_index[&node.leaf_value.to_bits()]),
                widths.leaf_ref,
            );
            return;
        }
        let feature = self
            .map
            .binary_search_by_key(&node.split_feature, |e| e.input)
            .expect("every split feature has a map entry");
        w.write(feature as u64, widths.feature_ref);
        let threshold = match &self.map[feature].dict {
            Dictionary::Numeric(values) => {
                values.binary_search_by(|v| v.total_cmp(&node.split_cond))
            }
            Dictionary::Categorical(sets) => {
                let set = canonical(tree.node_categories(node), cats);
                sets.binary_search_by(|s| s.as_slice().cmp(set))
            }
        }
        .expect("every split threshold is in its feature's dictionary");
        w.write(threshold as u64, widths.threshold_ref);
        if widths.default_bit == 1 {
            w.write_bool(node.default_left);
        }
    }

    /// A tree in the heap layout of `depth`: slot `i` holds node `heap[i]`,
    /// every slot padded to its fixed width.
    fn write_heap_tree(
        &self,
        w: &mut BitWriter,
        tree: &RegTree,
        depth: u32,
        complete: bool,
        scratch: &mut TreeScratch,
    ) {
        let slot_width = Layout::Heap { depth, complete }.slot_width(self.widths);
        w.write_bool(false);
        w.write(u64::from(depth), self.heap_depth_bits);
        w.write_bool(complete);
        let internal = (1usize << depth) - 1;
        let heap = &mut scratch.heap;
        heap.clear();
        heap.resize(2 * internal + 1, None);
        heap[0] = Some(0);
        for i in 0..internal {
            if let Some(id) = heap[i] {
                let node = tree.node(id);
                if !node.is_leaf() {
                    heap[2 * i + 1] = Some(node.left as usize);
                    heap[2 * i + 2] = Some(node.right as usize);
                }
            }
        }
        for (i, slot) in heap.iter().enumerate() {
            let start = w.len;
            if let Some(id) = *slot {
                let node = tree.node(id);
                if i < internal && !complete {
                    w.write_bool(node.is_leaf());
                }
                self.write_node(w, tree, node, &mut scratch.cats);
            }
            let width = if i < internal {
                slot_width
            } else {
                self.widths.leaf_ref
            };
            w.write(0, width - (w.len - start) as u32);
        }
    }

    /// A tree in the preorder layout of `nodes` nodes: each split stores the
    /// distance to its right child.
    fn write_preorder_tree(
        &self,
        w: &mut BitWriter,
        tree: &RegTree,
        nodes: u32,
        scratch: &mut TreeScratch,
    ) {
        let slot_width = Layout::Preorder { nodes }.slot_width(self.widths);
        w.write_bool(true);
        w.write(u64::from(nodes) - 1, self.preorder_nodes_bits);
        preorder(tree, &mut scratch.order, &mut scratch.stack);
        let position = &mut scratch.position;
        position.clear();
        position.resize(tree.num_nodes(), 0);
        for (p, &id) in scratch.order.iter().enumerate() {
            position[id] = p as u32;
        }
        let offset_bits = bits(u64::from(nodes) - 1);
        for (p, &id) in scratch.order.iter().enumerate() {
            let start = w.len;
            let node = tree.node(id);
            w.write_bool(node.is_leaf());
            self.write_node(w, tree, node, &mut scratch.cats);
            if !node.is_leaf() {
                let right = position[node.right as usize] - p as u32;
                w.write(u64::from(right), offset_bits);
            }
            w.write(0, slot_width - (w.len - start) as u32);
        }
    }

    /// The compact metadata section table.
    fn meta(&self) -> Meta {
        let model = self.model;
        Meta {
            objective: model.objective().clone(),
            max_delta_step: model.max_delta_step(),
            num_class: model.num_class(),
            n_targets: model.n_targets(),
            num_parallel_tree: model.num_parallel_tree(),
            shrinkage: model.shrinkage().cloned(),
        }
    }
}

/// Serialize `model` in the compact layout.
pub(super) fn encode(model: &BoostedModel) -> Result<Vec<u8>> {
    let encoding = Encoding::plan(model)?;
    let mut w = BitWriter::default();
    encoding.write_tables(&mut w);
    encoding.write_trees(&mut w);
    Ok(frame(&encoding.meta(), &w.bytes))
}
