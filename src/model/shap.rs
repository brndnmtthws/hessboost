//! Per-prediction feature attributions via QuadratureTreeSHAP.
//!
//! This is a port of XGBoost 3.4's CPU `pred_contribs` / `pred_interactions`
//! engine (`src/predictor/interpretability/{quadrature.h,shap.cc}`). It
//! computes the path-dependent TreeSHAP values of Lundberg et al., where a
//! feature's absence is modeled by the tree's cover (Hessian) distribution,
//! but evaluates them as QuadratureTreeSHAP does: a Shapley value is the
//! integral over the participation probability `p ∈ [0, 1]` of a
//! weighted-Banzhaf polynomial, and that integral is taken with a fixed
//! 8-point Gauss–Legendre rule after the endpoint substitution `p = s²`.
//!
//! One recursive walk visits both children of every split. Each node carries
//! an 8-lane basis `c` (the path polynomial evaluated at the 8 quadrature
//! nodes); a leaf returns `h = c · Π w · leaf · weight` lane-wise, and every
//! return edge extracts its feature's contribution from the subtree's `h`.
//! A feature that repeats on a path overwrites (and afterwards restores) its
//! probability rather than entering as a second player. The rule is exact
//! for paths with at most 7 distinct features; longer paths are a quadrature
//! approximation, identical to XGBoost's.
//!
//! Arithmetic follows upstream: the rule is generated in `f64` and stored as
//! `f32`, the recurrence and all accumulation are `f32` in XGBoost's operation
//! order, and each tree's cover-weighted expected value is summed in `f64` and
//! rounded once (bottom-up for scalar trees, top-down over the leaf vectors
//! for vector-leaf trees, as upstream). One deliberate deviation: a feature
//! repeated down a path can push its path probability (the product of its
//! inverse cover fractions) or the basis past `f32`'s range, where XGBoost's
//! temporaries overflow into `NaN` attributions. The probability then
//! continues in `f64`, and each basis lane or edge sum whose `f32` result
//! overflows is redone in `f64`, so representable attributions stay finite.
//! Summed over the ensemble the contributions satisfy
//!
//! ```text
//! Σ_j contribs[j] + bias == margin(x)
//! ```
//!
//! up to `f32` rounding, where `bias == base_score[k] + Σ_tree E[f_tree]` for
//! output `k` and `margin(x)` is [`BoostedModel::predict_margin`].

use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::model::{BoostedModel, RowBlock};
use crate::tree::{RegTree, SplitTest, split_goes_left};
use rayon::prelude::*;
use std::ops::RangeBounds;
use std::sync::LazyLock;

/// Quadrature points (XGBoost's `kQuadratureTreeShapPoints`).
const POINTS: usize = 8;
/// Path probability of a feature that is not yet on the current path.
const UNSEEN: f64 = -999.0;
/// Floor for a child's cover fraction, so zero-cover branches stay reachable.
const MIN_BRANCH_WEIGHT: f32 = 1e-12;

/// One value per quadrature node.
type Lanes = [f32; POINTS];

/// The endpoint Gauss–Legendre rule: `∫₀¹ g(u) du ≈ Σ weights[i] · g(nodes[i])`.
struct QuadratureRule {
    nodes: Lanes,
    weights: Lanes,
}

/// Legendre polynomial `P_n(x)` by the three-term recurrence.
fn legendre(n: usize, x: f64) -> f64 {
    let mut p0 = 1.0;
    if n == 0 {
        return p0;
    }
    let mut p1 = x;
    for k in 2..=n {
        let k = k as f64;
        let pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
        p0 = p1;
        p1 = pk;
    }
    p1
}

/// `P_n'(x)` given `pn = P_n(x)`.
fn legendre_derivative(n: usize, x: f64, pn: f64) -> f64 {
    n as f64 * (x * pn - legendre(n - 1, x)) / (x * x - 1.0)
}

/// XGBoost's `MakeEndpointQuadrature`: Newton-refined Gauss–Legendre roots
/// `x` on `[-1, 1]`, mapped to `s = (x + 1) / 2` on `[0, 1]` and substituted
/// `u = s²` (so `du = 2s ds`), stored in increasing node order.
fn endpoint_quadrature() -> QuadratureRule {
    let mut rule = QuadratureRule {
        nodes: [0.0; POINTS],
        weights: [0.0; POINTS],
    };
    let n = POINTS as f64;
    for i in 0..POINTS {
        let theta = std::f64::consts::PI * (i as f64 + 0.75) / (n + 0.5);
        let mut x = theta.cos();
        for _ in 0..64 {
            let pn = legendre(POINTS, x);
            let dx = pn / legendre_derivative(POINTS, x, pn);
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let pn = legendre(POINTS, x);
        let dpn = legendre_derivative(POINTS, x, pn);
        let w = 2.0 / ((1.0 - x * x) * dpn * dpn);
        #[allow(
            clippy::manual_midpoint,
            reason = "XGBoost rounds the sum, then halves"
        )]
        let s = 0.5 * (x + 1.0);
        let ws = 0.5 * w;
        rule.nodes[POINTS - 1 - i] = (s * s) as f32;
        rule.weights[POINTS - 1 - i] = (2.0 * s * ws) as f32;
    }
    rule
}

static RULE: LazyLock<QuadratureRule> = LazyLock::new(endpoint_quadrature);

/// `a * b + c` rounded the way XGBoost 3.4.2's release builds round it. Its
/// C++ compilers contract such expressions into one fused multiply-add where
/// the target baseline has FMA: aarch64 builds (GCC on Linux, Clang on macOS)
/// fuse, while `x86_64` wheels target a baseline without FMA and do not.
#[inline(always)]
fn madd(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(target_arch = "aarch64") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// [`madd`] in `f64`.
#[inline(always)]
fn madd64(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(target_arch = "aarch64") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Probability of following a child: its cover fraction, `0.5` under a
/// coverless parent, floored at [`MIN_BRANCH_WEIGHT`].
fn branch_weight(cover: f32, parent_cover: f32) -> f32 {
    if parent_cover <= 0.0 {
        return 0.5;
    }
    let weight = cover / parent_cover;
    if weight < MIN_BRANCH_WEIGHT {
        MIN_BRANCH_WEIGHT
    } else {
        weight
    }
}

/// Contribution of one return edge whose feature entered the subtree with
/// probability `p_enter` and had `p_exit` above it (`1.0` = not on the path).
///
/// Path probabilities are kept in `f64` (see [`enter_child`]); ordinary ones
/// are `f32` values and take XGBoost's `f32` arithmetic. When that overflows
/// (a probability beyond `f32`, or `α·h` exceeding it) the edge is redone in
/// `f64`, whose quotients `α·h / (1 + α·u)` stay representable.
#[inline(always)]
fn edge_delta(rule: &QuadratureRule, h: &Lanes, p_enter: f64, p_exit: f64) -> f32 {
    // The lane terms are formed by `shap_edge_terms` (four at a time on
    // NEON), then summed in lane order.
    let mut acc = 0.0f32;
    if p_enter != 1.0 {
        for t in crate::simd::shap_edge_terms(p_enter as f32 - 1.0, h, &rule.nodes) {
            acc += t;
        }
    }
    if p_exit != 1.0 {
        for t in crate::simd::shap_edge_terms(p_exit as f32 - 1.0, h, &rule.nodes) {
            acc -= t;
        }
    }
    if acc.is_finite() {
        return acc;
    }
    edge_delta_f64(rule, h, p_enter, p_exit)
}

/// Rows whose return edges [`Formulation::on_return_group`] records together.
const RETURN_GROUP: usize = 2;

/// One row's return edge: the subtree return `h` and the feature's path
/// probabilities on entry and above the split ([`edge_delta`]).
#[derive(Clone, Copy)]
struct ReturnEdge<'a> {
    h: &'a Lanes,
    p_enter: f64,
    p_exit: f64,
}

/// [`edge_delta`] of `N` rows. When all take the same terms, each row's
/// lane sum is still formed in lane order, but the rows' sums advance
/// together, so their dependent chains of additions overlap.
#[inline(always)]
fn edge_deltas<const N: usize>(rule: &QuadratureRule, edges: [ReturnEdge<'_>; N]) -> [f32; N] {
    let (enter, exit) = (edges[0].p_enter != 1.0, edges[0].p_exit != 1.0);
    if edges
        .iter()
        .any(|e| (e.p_enter != 1.0) != enter || (e.p_exit != 1.0) != exit)
    {
        return edges.map(|e| edge_delta(rule, e.h, e.p_enter, e.p_exit));
    }
    let terms = |p: fn(&ReturnEdge<'_>) -> f64| -> [Lanes; N] {
        std::array::from_fn(|k| {
            crate::simd::shap_edge_terms(p(&edges[k]) as f32 - 1.0, edges[k].h, &rule.nodes)
        })
    };
    let mut acc = [0.0f32; N];
    if enter {
        let t = terms(|e| e.p_enter);
        for i in 0..POINTS {
            for (a, t) in acc.iter_mut().zip(&t) {
                *a += t[i];
            }
        }
    }
    if exit {
        let t = terms(|e| e.p_exit);
        for i in 0..POINTS {
            for (a, t) in acc.iter_mut().zip(&t) {
                *a -= t[i];
            }
        }
    }
    std::array::from_fn(|k| {
        let e = &edges[k];
        if acc[k].is_finite() {
            acc[k]
        } else {
            edge_delta_f64(rule, e.h, e.p_enter, e.p_exit)
        }
    })
}

/// [`edge_delta`] in `f64`, for edges whose `f32` sum overflows. Kept out of
/// line so the inlined `f32` path stays small.
#[cold]
#[inline(never)]
fn edge_delta_f64(rule: &QuadratureRule, h: &Lanes, p_enter: f64, p_exit: f64) -> f32 {
    let term = |p: f64| {
        let alpha = p - 1.0;
        h.iter()
            .zip(&rule.nodes)
            .map(|(&hi, &u)| alpha * f64::from(hi) / madd64(alpha, f64::from(u), 1.0))
            .sum::<f64>()
    };
    (term(p_enter) - term(p_exit)) as f32
}

/// `α / (1 + α·u)` for `α = p - 1`, in `f32` unless that overflows (a
/// probability beyond `f32`), then in `f64`.
#[inline(always)]
fn edge_factor(p: f64, u: f32) -> f32 {
    let alpha = p as f32 - 1.0;
    let factor = alpha / madd(alpha, u, 1.0);
    if factor.is_finite() {
        factor
    } else {
        let alpha = p - 1.0;
        (alpha / madd64(alpha, f64::from(u), 1.0)) as f32
    }
}

/// The per-lane factor `α_enter / (1 + α_enter·u) - α_exit / (1 + α_exit·u)`
/// of a return edge, shared by every [`interaction_delta`] of that edge.
fn edge_factors(rule: &QuadratureRule, p_enter: f64, p_exit: f64) -> Lanes {
    let mut edge = [0.0; POINTS];
    if p_exit == 1.0 {
        for (e, &u) in edge.iter_mut().zip(&rule.nodes) {
            *e = edge_factor(p_enter, u);
        }
    } else {
        for (e, &u) in edge.iter_mut().zip(&rule.nodes) {
            *e = edge_factor(p_enter, u) - edge_factor(p_exit, u);
        }
    }
    edge
}

/// The part of [`edge_delta`] attributable to a path partner currently
/// entered with probability `q`, given the edge's [`edge_factors`]; redone in
/// `f64` when the `f32` sum overflows, like [`edge_delta`].
fn interaction_delta(rule: &QuadratureRule, h: &Lanes, edge: &Lanes, q: f64) -> f32 {
    if q == 1.0 {
        return 0.0;
    }
    let alpha_q = q as f32 - 1.0;
    let mut acc = 0.0f32;
    for ((&hi, &e), &u) in h.iter().zip(edge).zip(&rule.nodes) {
        acc += alpha_q * hi * e / madd(alpha_q, u, 1.0);
    }
    if acc.is_finite() {
        return acc;
    }
    let alpha_q = q - 1.0;
    h.iter()
        .zip(edge)
        .zip(&rule.nodes)
        .map(|((&hi, &e), &u)| {
            alpha_q * f64::from(hi) * f64::from(e) / madd64(alpha_q, f64::from(u), 1.0)
        })
        .sum::<f64>() as f32
}

/// [`ShapNode::first`] marker for a leaf.
const NO_CHILD: u32 = u32::MAX;

/// A split's children in XGBoost's order. XGBoost routes a categorical split's
/// category set right, hessboost routes it left, so the importer swaps those
/// children. Walking them in XGBoost's order keeps every accumulation in
/// XGBoost's sequence.
fn xgboost_children(node: &crate::tree::Node) -> (usize, usize) {
    let (left, right) = (node.left as usize, node.right as usize);
    if node.is_categorical {
        (right, left)
    } else {
        (left, right)
    }
}

/// A tree node with its routing data and both child branch weights.
#[derive(Clone, Copy)]
struct ShapNode {
    feature: u32,
    cond: f32,
    default_left: bool,
    is_categorical: bool,
    cat_begin: u32,
    cat_end: u32,
    /// XGBoost's left child (see [`xgboost_children`]), or [`NO_CHILD`] for a
    /// leaf.
    first: u32,
    second: u32,
    first_weight: f32,
    second_weight: f32,
    /// Leaf value (`0.0` for internal nodes).
    value: f32,
}

/// A [`RegTree`] prepared for QuadratureTreeSHAP walks.
struct ShapTree {
    nodes: Vec<ShapNode>,
    categories: Vec<u32>,
    /// Distinct split features, the only columns a walk can write.
    features: Vec<u32>,
}

/// Cover-weighted expected output of the subtree at `nid`, in `f64`
/// (XGBoost's `FillRootMeanValue`). A coverless split averages its children.
fn root_mean_value(tree: &RegTree, nid: usize) -> f64 {
    let node = tree.node(nid);
    if node.is_leaf() {
        return f64::from(node.leaf_value);
    }
    let (l, r) = xgboost_children(node);
    let left_mean = root_mean_value(tree, l);
    let right_mean = root_mean_value(tree, r);
    if node.sum_hess == 0.0 {
        #[allow(
            clippy::manual_midpoint,
            reason = "XGBoost rounds the sum, then halves"
        )]
        return 0.5 * (left_mean + right_mean);
    }
    let right_part = right_mean * f64::from(tree.node(r).sum_hess);
    madd64(left_mean, f64::from(tree.node(l).sum_hess), right_part) / f64::from(node.sum_hess)
}

/// Adds `path_weight ×` the cover-weighted expected leaf vector of vector-leaf
/// subtree `nid` to `out`, one value per output (XGBoost's
/// `FillRootMeanValues`): top-down, each leaf's vector enters scaled by its
/// path's product of cover fractions, in depth-first leaf order. A coverless
/// split halves the path weight of both children.
fn root_mean_values(tree: &RegTree, nid: usize, path_weight: f64, out: &mut [f64]) {
    let node = tree.node(nid);
    if node.is_leaf() {
        for (o, &v) in out.iter_mut().zip(tree.leaf_vector(nid)) {
            *o = madd64(path_weight, f64::from(v), *o);
        }
        return;
    }
    let (l, r) = xgboost_children(node);
    if node.sum_hess == 0.0 {
        root_mean_values(tree, l, path_weight * 0.5, out);
        root_mean_values(tree, r, path_weight * 0.5, out);
    } else {
        let parent = f64::from(node.sum_hess);
        let left = path_weight * f64::from(tree.node(l).sum_hess) / parent;
        root_mean_values(tree, l, left, out);
        let right = path_weight * f64::from(tree.node(r).sum_hess) / parent;
        root_mean_values(tree, r, right, out);
    }
}

impl ShapTree {
    /// The SHAP walk tree of `tree` with leaf `id` holding `leaf_value(id)`
    /// (its value, or one output of a vector leaf).
    fn from_tree(tree: &RegTree, leaf_value: impl Fn(usize) -> f32) -> Result<Self> {
        let src = tree.nodes();
        let mut features = Vec::new();
        let mut nodes = Vec::with_capacity(src.len());
        for (id, n) in src.iter().enumerate() {
            let mut node = ShapNode {
                feature: n.split_feature,
                cond: n.split_cond,
                default_left: n.default_left,
                is_categorical: n.is_categorical,
                cat_begin: n.cat_begin,
                cat_end: n.cat_end,
                first: NO_CHILD,
                second: NO_CHILD,
                first_weight: 0.0,
                second_weight: 0.0,
                value: if n.is_leaf() { leaf_value(id) } else { 0.0 },
            };
            if !n.is_leaf() {
                let (l, r) = xgboost_children(n);
                let (left, right) = (&src[l], &src[r]);
                // `!(x >= 0)` also rejects NaN, as XGBoost's `CHECK_GE` does.
                if !(n.sum_hess >= 0.0 && left.sum_hess >= 0.0 && right.sum_hess >= 0.0) {
                    return Err(HessboostError::model_format(
                        "QuadratureTreeSHAP is undefined for trees with negative node cover",
                    ));
                }
                node.first = l as u32;
                node.second = r as u32;
                node.first_weight = branch_weight(left.sum_hess, n.sum_hess);
                node.second_weight = branch_weight(right.sum_hess, n.sum_hess);
                node.value = 0.0;
                features.push(n.split_feature);
            }
            nodes.push(node);
        }
        features.sort_unstable();
        features.dedup();
        Ok(ShapTree {
            nodes,
            categories: tree.categories().to_vec(),
            features,
        })
    }
}

/// How return edges are written: additive contributions or interactions.
trait Formulation {
    /// Enter the child of a split on `feature` with path probability `p`.
    fn push(&mut self, feature: u32, p: f64);
    /// Leave the child entered by the matching [`Formulation::push`].
    fn pop(&mut self);
    /// Record the return edge of `feature` given the subtree return `h`.
    fn on_return(
        &mut self,
        rule: &QuadratureRule,
        feature: u32,
        h: &Lanes,
        p_enter: f64,
        p_exit: f64,
    );

    /// [`Formulation::on_return`] of [`RETURN_GROUP`] rows walked in
    /// lockstep.
    #[inline(always)]
    fn on_return_group(
        forms: [&mut Self; RETURN_GROUP],
        rule: &QuadratureRule,
        feature: u32,
        edges: [ReturnEdge<'_>; RETURN_GROUP],
    ) {
        for (form, e) in forms.into_iter().zip(edges) {
            form.on_return(rule, feature, e.h, e.p_enter, e.p_exit);
        }
    }
}

/// Additive SHAP: one feature contribution per return edge.
struct Additive<'a> {
    phi: &'a mut [f32],
}

impl Formulation for Additive<'_> {
    #[inline(always)]
    fn push(&mut self, _feature: u32, _p: f64) {}

    #[inline(always)]
    fn pop(&mut self) {}

    #[inline(always)]
    fn on_return(
        &mut self,
        rule: &QuadratureRule,
        feature: u32,
        h: &Lanes,
        p_enter: f64,
        p_exit: f64,
    ) {
        self.phi[feature as usize] += edge_delta(rule, h, p_enter, p_exit);
    }

    #[inline(always)]
    fn on_return_group(
        forms: [&mut Self; RETURN_GROUP],
        rule: &QuadratureRule,
        feature: u32,
        edges: [ReturnEdge<'_>; RETURN_GROUP],
    ) {
        for (form, d) in forms.into_iter().zip(edge_deltas(rule, edges)) {
            form.phi[feature as usize] += d;
        }
    }
}

/// [`PathEntry::prev`] marker: no earlier occurrence of the feature.
const NO_ENTRY: u32 = u32::MAX;

/// One split on the live root-to-node path.
#[derive(Clone, Copy)]
struct PathEntry {
    feature: u32,
    p: f64,
    /// Index of this feature's previous occurrence on the path.
    prev: u32,
    /// A later occurrence of the same feature hides this one from partners.
    shadowed: bool,
}

/// SHAP interactions: every return edge also pairs its feature with each
/// other distinct feature on the live path (newest occurrence wins). The
/// directed pair effects land in `matrix[feature][partner]`; additive effects
/// accumulate into `diag`. Both are scaled by the tree weight.
struct Interaction<'a> {
    path: &'a mut Vec<PathEntry>,
    /// Per-feature index of its newest path entry, [`NO_ENTRY`] if absent.
    last: &'a mut [u32],
    diag: &'a mut [f32],
    matrix: &'a mut [f32],
    width: usize,
    scale: f32,
}

impl Formulation for Interaction<'_> {
    fn push(&mut self, feature: u32, p: f64) {
        let prev = self.last[feature as usize];
        if prev != NO_ENTRY {
            self.path[prev as usize].shadowed = true;
        }
        self.last[feature as usize] = self.path.len() as u32;
        self.path.push(PathEntry {
            feature,
            p,
            prev,
            shadowed: false,
        });
    }

    fn pop(&mut self) {
        let entry = self.path.pop().expect("pop follows a push");
        self.last[entry.feature as usize] = entry.prev;
        if entry.prev != NO_ENTRY {
            self.path[entry.prev as usize].shadowed = false;
        }
    }

    fn on_return(
        &mut self,
        rule: &QuadratureRule,
        feature: u32,
        h: &Lanes,
        p_enter: f64,
        p_exit: f64,
    ) {
        let f = feature as usize;
        self.diag[f] = madd(
            self.scale,
            edge_delta(rule, h, p_enter, p_exit),
            self.diag[f],
        );
        // The current split is the path's last entry and shadows the older
        // occurrences of its own feature, so every unshadowed entry before it
        // is a distinct partner.
        let (_, partners) = self.path.split_last().expect("return follows a push");
        if partners.is_empty() {
            return;
        }
        let edge = edge_factors(rule, p_enter, p_exit);
        let row = &mut self.matrix[f * self.width..(f + 1) * self.width];
        for partner in partners.iter().rev().filter(|e| !e.shadowed) {
            let pair = interaction_delta(rule, h, &edge, partner.p);
            let cell = &mut row[partner.feature as usize];
            *cell = madd(self.scale, pair, *cell);
        }
    }
}

/// One outgoing edge of a split, as [`Walk::child`] descends it.
#[derive(Clone, Copy)]
struct Branch<const R: usize> {
    /// The parent's split feature.
    feature: u32,
    /// The child node id.
    child: u32,
    /// The child's cover fraction of its parent.
    weight: f32,
    /// Whether each row takes this edge.
    satisfies: [bool; R],
}

/// QuadratureTreeSHAP walks of a tree for `R` dense rows (`NaN` = missing)
/// in lockstep. Every row visits every node in the same order (both children
/// of each split), so one traversal serves them all; each row keeps its own
/// path probabilities, bases, returns, and formulation, and its arithmetic
/// is exactly that of a walk on its own. Interleaving the rows overlaps
/// their dependent floating-point chains. `path_prob` holds [`UNSEEN`] for
/// every feature on entry and on exit.
struct Walk<'a, F, const R: usize> {
    tree: &'a ShapTree,
    /// Per feature, every row's value (`NaN` = missing).
    rows: &'a [[f32; R]],
    rule: &'a QuadratureRule,
    /// Per feature, every row's path probability.
    path_prob: &'a mut [[f64; R]],
    forms: [F; R],
}

impl<F: Formulation, const R: usize> Walk<'_, F, R> {
    fn run(mut self) {
        if self.tree.nodes[0].first == NO_CHILD {
            return;
        }
        self.node(0, &[[1.0; POINTS]; R], &[1.0; R]);
    }

    /// Walk the subtree at `nid` with each row's basis `c` and path cover
    /// product `w_prod`, returning its weighted returns (returned by value:
    /// every lane is written, so no buffer is zero-filled first).
    fn node(&mut self, nid: usize, c: &[Lanes; R], w_prod: &[f32; R]) -> [Lanes; R] {
        let node = self.tree.nodes[nid];
        if node.first == NO_CHILD {
            return std::array::from_fn(|r| {
                let scale = w_prod[r] * node.value;
                std::array::from_fn(|i| c[r][i] * scale * self.rule.weights[i])
            });
        }
        // Route with hessboost's orientation, then map onto XGBoost's order.
        let test = if node.is_categorical {
            SplitTest::Categories(
                &self.tree.categories[node.cat_begin as usize..node.cat_end as usize],
            )
        } else {
            SplitTest::Threshold(node.cond)
        };
        let goes_first = self.rows[node.feature as usize].map(|v| {
            let goes_left = split_goes_left((!v.is_nan()).then_some(v), node.default_left, test);
            goes_left != node.is_categorical
        });
        let branch = |child, weight, satisfies| Branch {
            feature: node.feature,
            child,
            weight,
            satisfies,
        };
        let mut out = self.child(branch(node.first, node.first_weight, goes_first), c, w_prod);
        let second = self.child(
            branch(node.second, node.second_weight, goes_first.map(|g| !g)),
            c,
            w_prod,
        );
        for r in 0..R {
            for i in 0..POINTS {
                out[r][i] += second[r][i];
            }
        }
        out
    }

    /// Descend `branch` from a node with each row's basis `c` and path cover
    /// product `w_prod`, returning the child's weighted returns.
    fn child(&mut self, branch: Branch<R>, c: &[Lanes; R], w_prod: &[f32; R]) -> [Lanes; R] {
        let Branch {
            feature,
            child,
            weight,
            satisfies,
        } = branch;
        let rule = self.rule;
        let f = feature as usize;
        let p_old: [f64; R] = self.path_prob[f];
        let mut p_enter = [0.0; R];
        let mut c_child = [[0.0; POINTS]; R];
        for r in 0..R {
            (p_enter[r], c_child[r]) = enter_child(rule, p_old[r], weight, satisfies[r], &c[r]);
            self.forms[r].push(feature, p_enter[r]);
        }
        self.path_prob[f] = p_enter;
        let out = self.node(child as usize, &c_child, &w_prod.map(|w| w * weight));
        let edge = |r: usize| ReturnEdge {
            h: &out[r],
            p_enter: p_enter[r],
            p_exit: if p_old[r] == UNSEEN { 1.0 } else { p_old[r] },
        };
        // Rows a group at a time, so their return sums overlap.
        let (groups, rest) = self.forms.as_chunks_mut::<RETURN_GROUP>();
        for (k, group) in groups.iter_mut().enumerate() {
            F::on_return_group(
                group.each_mut(),
                rule,
                feature,
                std::array::from_fn(|i| edge(RETURN_GROUP * k + i)),
            );
        }
        for (form, r) in rest.iter_mut().zip(R - R % RETURN_GROUP..) {
            let e = edge(r);
            form.on_return(rule, feature, e.h, e.p_enter, e.p_exit);
        }
        for form in &mut self.forms {
            form.pop();
        }
        self.path_prob[f] = p_old;
        out
    }
}

/// One row's entry into the child of a split on a feature whose current path
/// probability is `p_old` ([`UNSEEN`] when not on the path): the child's path
/// probability and its basis from the parent's `c`.
///
/// Path probabilities are stored in `f64` but computed in `f32` like
/// XGBoost's, so they are `f32` values unless a feature repeated down the
/// path divides its probability past `f32`'s range (e.g. child cover
/// fractions of `2^-32` four times, `2^128`); that one continues in `f64`,
/// and the recurrences reading it fall back to `f64` where their `f32`
/// results overflow.
#[inline(always)]
fn enter_child(
    rule: &QuadratureRule,
    p_old: f64,
    weight: f32,
    satisfies: bool,
    c: &Lanes,
) -> (f64, Lanes) {
    let seen = p_old != UNSEEN;
    let p_enter = match (satisfies, seen) {
        (false, _) => 0.0,
        (true, false) => f64::from(1.0 / weight),
        (true, true) => {
            let p = p_old as f32 / weight;
            if p.is_finite() {
                f64::from(p)
            } else {
                p_old / f64::from(weight)
            }
        }
    };
    let alpha = p_enter as f32 - 1.0;
    let mut c_child = crate::simd::shap_scaled_basis(alpha, c, &rule.nodes);
    if seen {
        let alpha_old = p_old as f32 - 1.0;
        if alpha_old != 0.0 {
            // Usually every lane is finite: divide them all at once.
            if let Some(divided) = crate::simd::shap_divided_basis(alpha_old, &c_child, &rule.nodes)
            {
                return (p_enter, divided);
            }
            let old: Lanes = std::array::from_fn(|i| madd(alpha_old, rule.nodes[i], 1.0));
            for (((ci, &u), &c0), &old) in c_child.iter_mut().zip(&rule.nodes).zip(c).zip(&old) {
                *ci = if ci.is_finite() && old.is_finite() {
                    *ci / old
                } else {
                    // A factor overflowed f32 before the overwritten one
                    // was divided out; the quotient may still be finite,
                    // so redo this lane in f64.
                    let u = f64::from(u);
                    let enter = madd64(p_enter - 1.0, u, 1.0);
                    let old = madd64(p_old - 1.0, u, 1.0);
                    (f64::from(c0) * enter / old) as f32
                };
            }
        }
    }
    (p_enter, c_child)
}

/// Rows [`BoostedModel::predict_contribs_range`] walks in lockstep. Fewer
/// than a lane group, so a single-row [`RowBlock`] loaded with them keeps
/// every one row-major.
const SHAP_ROWS: usize = 8;
const _: () = assert!(SHAP_ROWS < crate::tree::compact::LANES);

/// The per-call state of [`BoostedModel::predict_contribs_range`].
struct ContribContext<'a> {
    forest: &'a ShapForest,
    rule: &'a QuadratureRule,
    /// Initial margins, `[row][output]`.
    initial: &'a [f32],
    nf: usize,
    /// Values per output of an output row: `nf + 1`.
    width: usize,
    /// Outputs.
    k: usize,
}

impl ContribContext<'_> {
    /// The contributions of the `R` rows `row` (dense values `x`) into their
    /// output rows `out`, using `scratch`. Each row's arithmetic is that of
    /// a walk on its own.
    fn rows<const R: usize>(
        &self,
        row: [usize; R],
        x: [&[f32]; R],
        scratch: &mut GroupScratch<R>,
        mut out: [&mut [f32]; R],
    ) {
        let (forest, nf, width, k) = (self.forest, self.nf, self.width, self.k);
        let GroupScratch {
            phi,
            path_prob,
            values,
        } = scratch;
        let mut phi = phi.each_mut().map(Vec::as_mut_slice);
        values.clear();
        values.extend((0..nf).map(|f| x.map(|row| row[f])));
        for c in 0..k {
            for &ti in &forest.by_output[c] {
                let tree = &forest.trees[ti];
                for phi in &mut phi {
                    for &f in &tree.features {
                        phi[f as usize] = 0.0;
                    }
                }
                Walk {
                    tree,
                    rows: values,
                    rule: self.rule,
                    path_prob,
                    forms: phi.each_mut().map(|phi| Additive { phi }),
                }
                .run();
                let weight = forest.weights[ti];
                for (phi, out) in phi.iter().zip(&mut out) {
                    let acc = &mut out[c * width..(c + 1) * width];
                    for &f in &tree.features {
                        acc[f as usize] = madd(phi[f as usize], weight, acc[f as usize]);
                    }
                }
            }
            for (&row, out) in row.iter().zip(&mut out) {
                let acc = &mut out[c * width..(c + 1) * width];
                acc[nf] += forest.root_mean_sums[c];
                acc[nf] += self.initial[row * k + c];
            }
        }
    }
}

/// The per-worker buffers of [`ContribContext::rows`] for `R` rows.
struct GroupScratch<const R: usize> {
    /// Each row's contributions of one tree.
    phi: [Vec<f32>; R],
    /// Per feature, every row's path probability ([`UNSEEN`] between walks).
    path_prob: Vec<[f64; R]>,
    /// Per feature, every row's value, as the walk routes them.
    values: Vec<[f32; R]>,
}

impl<const R: usize> GroupScratch<R> {
    fn new(nf: usize) -> Self {
        GroupScratch {
            phi: [(); R].map(|()| vec![0.0; nf]),
            path_prob: vec![[UNSEEN; R]; nf],
            values: Vec::with_capacity(nf),
        }
    }
}

/// The instance-independent SHAP setup of an ensemble.
struct ShapForest {
    trees: Vec<ShapTree>,
    weights: Vec<f32>,
    /// Per output: `Σ weight · E[f_tree]`, summed in `f64`, rounded once.
    root_mean_sums: Vec<f32>,
    /// Per output: the ids (into `trees`) of the trees feeding it, ascending
    /// (`BoostedModel::tree_output`; a vector-leaf tree contributes one
    /// per-output view to every list).
    by_output: Vec<Vec<usize>>,
}

impl BoostedModel {
    fn shap_forest(&self, trees: &[RegTree], k: usize) -> Result<ShapForest> {
        let mut sums = vec![0f64; k];
        let mut by_output = vec![Vec::new(); k];
        let mut weights = Vec::with_capacity(trees.len());
        let mut shap_trees = Vec::with_capacity(trees.len());
        let mut root_means = vec![0f64; k];
        let mut push = |tree: ShapTree, c: usize, root_mean: f64, weight: f32| {
            by_output[c].push(shap_trees.len());
            shap_trees.push(tree);
            sums[c] = madd64(root_mean, f64::from(weight), sums[c]);
            weights.push(weight);
        };
        for (ti, tree) in trees.iter().enumerate() {
            let weight = self.tree_weight(ti);
            if tree.is_vector_leaf() {
                // XGBoost walks a vector-leaf tree once per output with that
                // output's leaf values over the shared covers (the Hessians
                // summed across targets). Its root means come from one
                // top-down pass over the leaf vectors, whose rounding differs
                // from the per-output bottom-up reduction.
                root_means.fill(0.0);
                root_mean_values(tree, 0, 1.0, &mut root_means);
                let (width, leaves) = tree.leaf_vector_parts();
                for (c, &root_mean) in root_means.iter().enumerate() {
                    let shap = ShapTree::from_tree(tree, |id| leaves[id * width + c])?;
                    push(shap, c, root_mean, weight);
                }
            } else {
                let shap = ShapTree::from_tree(tree, |id| tree.node(id).leaf_value)?;
                push(shap, self.tree_output(ti), root_mean_value(tree, 0), weight);
            }
        }
        Ok(ShapForest {
            trees: shap_trees,
            weights,
            root_mean_sums: sums.into_iter().map(|s| s as f32).collect(),
            by_output,
        })
    }

    /// Exact per-feature contributions for gblinear: `weight · x` per present
    /// feature and the intercept plus linear bias in the bias column.
    fn linear_contribs(&self, data: &DMatrix, row: usize, initial: &[f32], out: &mut [f32]) {
        let Some(linear) = self.linear() else {
            return;
        };
        let k = self.n_outputs();
        let width = out.len() / k;
        self.for_each_linear_contribution(data, row, |f, c, v| {
            out[c * width + f] = v as f32;
        });
        for c in 0..k {
            out[c * width + width - 1] =
                (f64::from(initial[row * k + c]) + f64::from(linear.bias()[c])) as f32;
        }
    }

    /// SHAP feature contributions, matching XGBoost `pred_contribs=True`
    /// (QuadratureTreeSHAP; see the module docs).
    ///
    /// For a single-output model the result is row-major with shape
    /// `n_rows × (n_features + 1)`: within each row, columns `0..n_features` are
    /// the per-feature contributions and the final column is the bias
    /// (that output's intercept plus each tree's expected value).
    ///
    /// For a multiclass model (`n_outputs > 1`) the layout is
    /// `n_rows × n_outputs × (n_features + 1)`, row-major: the contributions for
    /// row `r`, output `c`, feature `j` live at
    /// `((r * n_outputs + c) * (n_features + 1)) + j`, with the bias at column
    /// `n_features`. Tree `t` contributes to output
    /// `(t / num_parallel_tree) % n_outputs`; a vector-leaf tree contributes
    /// to every output with that output's leaf values and the tree's covers.
    ///
    /// For every row (and output) the `n_features + 1` values sum to the raw
    /// margin from [`BoostedModel::predict_margin`] up to `f32` rounding. Uses
    /// the effective iterations (`[0, best_iteration + 1)` after early
    /// stopping).
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] when `data` does not fit the
    /// model, [`HessboostError::ModelFormat`] when a tree has a negative cover,
    /// [`HessboostError::InvalidParameter`] for linear-leaf models.
    pub fn predict_contribs(&self, data: &DMatrix) -> Result<Vec<f32>> {
        self.predict_contribs_range(data, self.default_iteration_range())
    }

    /// [`Self::predict_contribs`] over the boosting iterations in
    /// `iterations` (see [`BoostedModel::predict_margin_range`]). As in
    /// XGBoost the range must start at iteration `0`; use
    /// [`BoostedModel::slice`] for a later start.
    ///
    /// # Errors
    ///
    /// As for [`Self::predict_contribs`], plus an out-of-range
    /// `iterations`.
    pub fn predict_contribs_range(
        &self,
        data: &DMatrix,
        iterations: impl RangeBounds<usize>,
    ) -> Result<Vec<f32>> {
        let pro = self.attribution_prologue(data, iterations, "contribution prediction")?;
        let (n, k, nf, width, trees) = (pro.n, pro.k, pro.nf, pro.width, pro.trees);
        let initial = pro.initial;
        let forest = self.shap_forest(trees, k)?;
        let rule = &*RULE;

        let row_len = k * width;
        let ctx = ContribContext {
            forest: &forest,
            rule,
            initial: &initial,
            nf,
            width,
            k,
        };
        let mut out = vec![0f32; n * row_len];
        out.par_chunks_mut(SHAP_ROWS * row_len)
            .enumerate()
            .for_each_init(
                || {
                    (
                        RowBlock::single_rows(data),
                        GroupScratch::<SHAP_ROWS>::new(nf),
                        GroupScratch::<1>::new(nf),
                    )
                },
                |(rows, group, single), (chunk, out_rows)| {
                    let first = chunk * SHAP_ROWS;
                    if self.linear().is_some() {
                        for (i, out_row) in out_rows.chunks_exact_mut(row_len).enumerate() {
                            self.linear_contribs(data, first + i, &initial, out_row);
                        }
                        return;
                    }
                    let count = out_rows.len() / row_len;
                    rows.load(first, count);
                    let x = |i: usize| rows.row(i).expect("single-row blocks are dense");
                    if count == SHAP_ROWS {
                        let mut outs = out_rows.chunks_exact_mut(row_len);
                        let outs = [(); SHAP_ROWS].map(|()| outs.next().expect("a full group"));
                        ctx.rows(
                            std::array::from_fn(|i| first + i),
                            std::array::from_fn(x),
                            group,
                            outs,
                        );
                    } else {
                        for (i, out_row) in out_rows.chunks_exact_mut(row_len).enumerate() {
                            ctx.rows([first + i], [x(i)], single, [out_row]);
                        }
                    }
                },
            );
        Ok(out)
    }

    /// SHAP interaction values, matching XGBoost `pred_interactions=True`
    /// (QuadratureTreeSHAP; see the module docs).
    ///
    /// For a single-output model the result is row-major with per-row shape
    /// `(n_features + 1) × (n_features + 1)`. Within a row's matrix `M`:
    ///
    /// * the off-diagonal entry `M[i][j]` (`i, j < n_features`) is the SHAP
    ///   interaction between features `i` and `j` (symmetric: `M[i][j] ==
    ///   M[j][i]`) and is split evenly between the two cells.
    /// * the diagonal entry `M[i][i]` is feature `i`'s *main* effect, set so that
    ///   the row sums to feature `i`'s SHAP value.
    /// * the final row/column (index `n_features`) carry the bias: `M[nf][nf]`
    ///   holds the intercept plus each tree's expected value `Σ E[f_tree]`, and
    ///   the remaining bias cells are zero.
    ///
    /// Consequently the whole matrix sums to the raw margin from
    /// [`BoostedModel::predict_margin`] (including the applicable base
    /// margin) up to `f32` rounding.
    ///
    /// For a multiclass model (`n_outputs > 1`) the layout is
    /// `n_rows × n_outputs × (n_features + 1)^2`, row-major: the matrix for row
    /// `r`, output `c` occupies the `(n_features + 1)^2` values starting at
    /// `(r * n_outputs + c) * (n_features + 1)^2`. Tree `t` contributes to output
    /// `(t / num_parallel_tree) % n_outputs`. Uses the effective iterations,
    /// like [`Self::predict_contribs`].
    ///
    /// # Errors
    ///
    /// As for [`BoostedModel::predict_contribs`].
    pub fn predict_interactions(&self, data: &DMatrix) -> Result<Vec<f32>> {
        self.predict_interactions_range(data, self.default_iteration_range())
    }

    /// [`Self::predict_interactions`] over the boosting iterations in
    /// `iterations`, which must start at iteration `0` (see
    /// [`Self::predict_contribs_range`]).
    ///
    /// # Errors
    ///
    /// As for [`Self::predict_contribs_range`].
    pub fn predict_interactions_range(
        &self,
        data: &DMatrix,
        iterations: impl RangeBounds<usize>,
    ) -> Result<Vec<f32>> {
        struct Scratch<'a> {
            rows: RowBlock<'a>,
            contribs: Vec<f32>,
            diag: Vec<f32>,
            path_prob: Vec<[f64; 1]>,
            /// The row's values as the walk reads them.
            values: Vec<[f32; 1]>,
            path: Vec<PathEntry>,
            last: Vec<u32>,
        }

        let pro = self.attribution_prologue(data, iterations, "interaction prediction")?;
        let (n, k, nf, width, trees) = (pro.n, pro.k, pro.nf, pro.width, pro.trees);
        let initial = pro.initial;
        let mwidth = width * width;
        let forest = self.shap_forest(trees, k)?;
        let rule = &*RULE;

        let mut out = vec![0f32; n * k * mwidth];
        out.par_chunks_mut(k * mwidth).enumerate().for_each_init(
            || Scratch {
                rows: RowBlock::single_rows(data),
                contribs: vec![0f32; k * width],
                diag: vec![0f32; width],
                path_prob: vec![[UNSEEN; 1]; nf],
                values: Vec::with_capacity(nf),
                path: Vec::new(),
                last: vec![NO_ENTRY; nf],
            },
            |s, (row, out_row)| {
                if self.linear().is_some() {
                    // A linear model has no interactions: its contributions
                    // sit on the diagonal.
                    s.contribs.fill(0.0);
                    self.linear_contribs(data, row, &initial, &mut s.contribs);
                    for (c, m) in out_row.chunks_exact_mut(mwidth).enumerate() {
                        for j in 0..width {
                            m[j * width + j] = s.contribs[c * width + j];
                        }
                    }
                    return;
                }
                s.rows.load(row, 1);
                let x = s.rows.row(0).expect("single-row blocks are dense");
                s.values.clear();
                s.values.extend(x.iter().map(|&v| [v]));
                for (c, m) in out_row.chunks_exact_mut(mwidth).enumerate() {
                    let diag = &mut s.diag;
                    diag.fill(0.0);
                    for &ti in &forest.by_output[c] {
                        Walk {
                            tree: &forest.trees[ti],
                            rows: &s.values,
                            rule,
                            path_prob: &mut s.path_prob,
                            forms: [Interaction {
                                path: &mut s.path,
                                last: &mut s.last,
                                diag,
                                matrix: m,
                                width,
                                scale: forest.weights[ti],
                            }],
                        }
                        .run();
                    }
                    diag[nf] += forest.root_mean_sums[c];
                    diag[nf] += initial[row * k + c];
                    finalize_interactions(m, diag, width);
                }
            },
        );
        Ok(out)
    }
}

/// Finish one output's `width × width` interaction matrix `m` from its
/// directed estimates: average the two estimates of each pair, then set
/// each diagonal cell so its row sums to the additive value in `diag`.
fn finalize_interactions(m: &mut [f32], diag: &[f32], width: usize) {
    for r in 0..width {
        for cc in r + 1..width {
            #[allow(
                clippy::manual_midpoint,
                reason = "XGBoost rounds the sum, then halves"
            )]
            let sym = 0.5 * (m[r * width + cc] + m[cc * width + r]);
            m[r * width + cc] = sym;
            m[cc * width + r] = sym;
        }
    }
    for r in 0..width {
        let mut value = diag[r];
        for cc in 0..width {
            if cc != r {
                value -= m[r * width + cc];
            }
        }
        m[r * width + r] = value;
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{BoosterKind, TrainingParams, TreeMethod};
    use crate::data::{DMatrix, FeatureType};
    use crate::test_support::labeled_dense;
    use crate::{model::BoostedModel, training::train};

    /// A `reg:squarederror` model with `eta = 0.3`.
    fn squared_error_model(d: &DMatrix, max_depth: usize, rounds: usize) -> BoostedModel {
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(max_depth)
            .eta(0.3)
            .build()
            .unwrap();
        train(&params, d, rounds).unwrap()
    }

    /// A 30-round regression model on 80 rows of 5 features; features 0 and
    /// 1 carry signal, the rest are noise.
    fn regression_fixture() -> (DMatrix, BoostedModel) {
        let (n, nf) = (80, 5);
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            for j in 0..nf {
                // Deterministic pseudo-random values.
                x[i * nf + j] = ((i * 31 + j * 17 + 7) % 97) as f32 / 97.0;
            }
            y[i] = 2.0 * x[i * nf] - 1.5 * x[i * nf + 1] + 0.3;
        }
        let d = labeled_dense(&x, n, nf, &y);
        let model = squared_error_model(&d, 4, 30);
        (d, model)
    }

    /// A 15-round `multi:softprob` model on 90 rows of 4 features, 3 classes.
    fn multiclass_fixture() -> (DMatrix, BoostedModel) {
        let (n, nf, k) = (90, 4, 3);
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            for j in 0..nf {
                x[i * nf + j] = ((i * 13 + j * 29 + 3) % 101) as f32 / 101.0;
            }
            y[i] = (i % k) as f32;
        }
        let d = labeled_dense(&x, n, nf, &y);
        let params = TrainingParams::builder()
            .objective("multi:softprob")
            .num_class(k)
            .max_depth(3)
            .eta(0.3)
            .build()
            .unwrap();
        let model = train(&params, &d, 15).unwrap();
        assert_eq!(model.n_outputs(), k);
        (d, model)
    }

    fn sum64(values: &[f32]) -> f64 {
        values.iter().map(|&v| f64::from(v)).sum()
    }

    /// Largest `|Σ contributions - margin|` over every row and output.
    fn max_additivity_error(contribs: &[f32], margin: &[f32], width: usize) -> f64 {
        contribs
            .chunks_exact(width)
            .zip(margin)
            .map(|(c, &m)| (sum64(c) - f64::from(m)).abs())
            .fold(0.0, f64::max)
    }

    /// `model`'s contributions on `d` have the documented layout and sum to
    /// the margin of every row and output.
    fn assert_additive(model: &BoostedModel, d: &DMatrix) {
        let width = d.n_cols() + 1;
        let contribs = model.predict_contribs(d).unwrap();
        assert_eq!(contribs.len(), d.n_rows() * model.n_outputs() * width);
        let margin = model.predict_margin(d).unwrap();
        let err = max_additivity_error(&contribs, &margin, width);
        assert!(err < 1e-4, "max additivity error {err} exceeded 1e-4");
    }

    /// `model`'s interaction matrices on `d` have the documented layout, each
    /// row sums to that feature's contribution (the bias row to the bias),
    /// each matrix to the margin, and every matrix is symmetric.
    fn assert_interactions_consistent(model: &BoostedModel, d: &DMatrix) {
        let width = d.n_cols() + 1;
        let mwidth = width * width;
        let inter = model.predict_interactions(d).unwrap();
        assert_eq!(inter.len(), d.n_rows() * model.n_outputs() * mwidth);
        let contribs = model.predict_contribs(d).unwrap();
        let margin = model.predict_margin(d).unwrap();

        let (mut row_err, mut eff_err, mut sym_err) = (0f64, 0f64, 0f64);
        let matrices = inter.chunks_exact(mwidth).zip(contribs.chunks_exact(width));
        for ((m, c), &target) in matrices.zip(&margin) {
            for (row, &cval) in m.chunks_exact(width).zip(c) {
                row_err = row_err.max((sum64(row) - f64::from(cval)).abs());
            }
            eff_err = eff_err.max((sum64(m) - f64::from(target)).abs());
            for i in 0..width {
                for j in 0..width {
                    let e = (f64::from(m[i * width + j]) - f64::from(m[j * width + i])).abs();
                    sym_err = sym_err.max(e);
                }
            }
        }
        assert!(row_err < 1e-4, "row-consistency error {row_err}");
        assert!(eff_err < 1e-4, "efficiency error {eff_err}");
        assert!(sym_err < 1e-5, "symmetry error {sym_err}");
    }

    #[test]
    fn additivity_single_output() {
        let (d, model) = regression_fixture();
        assert_additive(&model, &d);
    }

    /// Textbook path-dependent TreeSHAP (Lundberg et al., Algorithm 2) with
    /// cloned paths, independent of the arena implementation.
    mod textbook {
        use crate::model::BoostedModel;
        use crate::tree::{RegTree, in_category_set};

        #[derive(Clone, Copy)]
        struct El {
            d: i64,
            z: f64,
            o: f64,
            w: f64,
        }

        fn extend(m: &mut Vec<El>, pz: f64, po: f64, pi: i64) {
            let l = m.len();
            m.push(El {
                d: pi,
                z: pz,
                o: po,
                w: if l == 0 { 1.0 } else { 0.0 },
            });
            for i in (0..l).rev() {
                m[i + 1].w += po * m[i].w * (i + 1) as f64 / (l + 1) as f64;
                m[i].w = pz * m[i].w * (l - i) as f64 / (l + 1) as f64;
            }
        }

        fn unwind(m: &mut Vec<El>, i: usize) {
            let l = m.len() - 1;
            let (o, z) = (m[i].o, m[i].z);
            let mut n = m[l].w;
            for j in (0..l).rev() {
                if o == 0.0 {
                    m[j].w = m[j].w * (l + 1) as f64 / (z * (l - j) as f64);
                } else {
                    let t = m[j].w;
                    m[j].w = n * (l + 1) as f64 / ((j + 1) as f64 * o);
                    n = t - m[j].w * z * (l - j) as f64 / (l + 1) as f64;
                }
            }
            for j in i..l {
                m[j].d = m[j + 1].d;
                m[j].z = m[j + 1].z;
                m[j].o = m[j + 1].o;
            }
            m.pop();
        }

        fn unwound_sum(m: &[El], i: usize) -> f64 {
            let l = m.len() - 1;
            let (o, z) = (m[i].o, m[i].z);
            let mut n = m[l].w;
            let mut total = 0.0;
            for j in (0..l).rev() {
                if o == 0.0 {
                    total += m[j].w * (l + 1) as f64 / (z * (l - j) as f64);
                } else {
                    let t = n * (l + 1) as f64 / ((j + 1) as f64 * o);
                    total += t;
                    n = m[j].w - t * z * (l - j) as f64 / (l + 1) as f64;
                }
            }
            total
        }

        /// One tree's explanation of row `x`, accumulating into `phi`.
        struct Explain<'a> {
            tree: &'a RegTree,
            x: &'a [f32],
            phi: &'a mut [f64],
        }

        impl Explain<'_> {
            /// Extend path `m` with the split on feature `pi` (zero fraction
            /// `pz`, one fraction `po`), then descend into `node`.
            fn recurse(&mut self, node: usize, mut m: Vec<El>, pz: f64, po: f64, pi: i64) {
                let tree = self.tree;
                extend(&mut m, pz, po, pi);
                let n = tree.node(node);
                if n.is_leaf() {
                    for i in 1..m.len() {
                        let w = unwound_sum(&m, i);
                        self.phi[m[i].d as usize] +=
                            w * (m[i].o - m[i].z) * f64::from(n.leaf_value);
                    }
                    return;
                }
                let v = self.x[n.split_feature as usize];
                let go_left = if v.is_nan() {
                    n.default_left
                } else if n.is_categorical {
                    in_category_set(
                        &tree.categories()[n.cat_begin as usize..n.cat_end as usize],
                        v,
                    )
                } else {
                    v < n.split_cond
                };
                let (hot, cold) = if go_left {
                    (n.left as usize, n.right as usize)
                } else {
                    (n.right as usize, n.left as usize)
                };
                let cover = f64::from(n.sum_hess);
                let (mut iz, mut io) = (1.0, 1.0);
                if let Some(k) = m.iter().position(|e| e.d == i64::from(n.split_feature)) {
                    iz = m[k].z;
                    io = m[k].o;
                    unwind(&mut m, k);
                }
                let f = i64::from(n.split_feature);
                let hz = f64::from(tree.node(hot).sum_hess) / cover;
                let cz = f64::from(tree.node(cold).sum_hess) / cover;
                self.recurse(hot, m.clone(), hz * iz, io, f);
                self.recurse(cold, m, cz * iz, 0.0, f);
            }
        }

        /// Contributions for row `x` (bias last), in `f64`.
        pub fn contributions(model: &BoostedModel, x: &[f32]) -> Vec<f64> {
            let nf = x.len();
            let mut phi = vec![0f64; nf + 1];
            phi[nf] = f64::from(model.base_score());
            for tree in model.trees() {
                phi[nf] += super::super::root_mean_value(tree, 0);
                Explain {
                    tree,
                    x,
                    phi: &mut phi,
                }
                .recurse(0, Vec::new(), 1.0, 1.0, -1);
            }
            phi
        }
    }

    /// `model`'s contributions on `d` match the textbook reference on every
    /// row of `x`, the row-major features of `d`.
    fn assert_matches_textbook(model: &BoostedModel, d: &DMatrix, x: &[f32]) {
        let nf = d.n_cols();
        let contribs = model.predict_contribs(d).unwrap();
        let mut max_err = 0f64;
        for (c, row) in contribs.chunks_exact(nf + 1).zip(x.chunks_exact(nf)) {
            for (&got, want) in c.iter().zip(textbook::contributions(model, row)) {
                max_err = max_err.max((f64::from(got) - want).abs());
            }
        }
        assert!(max_err < 1e-4, "max textbook TreeSHAP error {max_err}");
    }

    /// Numeric, missing, and categorical routing against the f64 reference
    /// (exact here: depth 6 keeps every path within the rule's exact range).
    #[test]
    fn contributions_match_textbook_tree_shap() {
        let n = 96;
        let nf = 5;
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            for j in 0..nf {
                let v = if j == 4 {
                    ((i * 7) % 6) as f32
                } else {
                    ((i * 31 + j * 17 + 7) % 97) as f32 / 97.0
                };
                x[i * nf + j] = if (i + j) % 11 == 0 { f32::NAN } else { v };
            }
            let cat_effect = [0.8, -0.5, 0.0, 1.2, -1.0, 0.3];
            let cat = x[i * nf + 4];
            y[i] = 2.0 * x[i * nf].max(0.0) - 1.5 * x[i * nf + 1].max(0.0)
                + x[i * nf + 2].max(0.0) * x[i * nf + 3].max(0.0)
                + if cat.is_nan() {
                    0.0
                } else {
                    cat_effect[cat as usize]
                };
        }
        let types = [
            FeatureType::Numerical,
            FeatureType::Numerical,
            FeatureType::Numerical,
            FeatureType::Numerical,
            FeatureType::Categorical,
        ];
        let d = labeled_dense(&x, n, nf, &y)
            .with_feature_types(&types)
            .unwrap();
        let model = squared_error_model(&d, 6, 12);
        let categorical = model
            .trees()
            .iter()
            .flat_map(crate::tree::RegTree::nodes)
            .any(|node| node.is_categorical);
        assert!(categorical, "no categorical split was learned");
        assert_matches_textbook(&model, &d, &x);
    }

    #[test]
    fn additivity_multiclass() {
        let (d, model) = multiclass_fixture();
        assert_additive(&model, &d);
    }

    #[test]
    fn unused_feature_has_zero_contribution() {
        // Feature 3 is pure constant noise (never predictive) AND we verify no
        // split uses it; its contribution must be ~0 for every row.
        let n = 70;
        let nf = 4;
        let mut x = vec![0f32; n * nf];
        let mut y = vec![0f32; n];
        for i in 0..n {
            x[i * nf] = ((i * 7 + 1) % 53) as f32 / 53.0;
            x[i * nf + 1] = ((i * 11 + 2) % 53) as f32 / 53.0;
            x[i * nf + 2] = ((i * 5 + 3) % 53) as f32 / 53.0;
            x[i * nf + 3] = 0.5; // constant -> never a useful split
            y[i] = 3.0 * x[i * nf] - 2.0 * x[i * nf + 1];
        }
        let d = labeled_dense(&x, n, nf, &y);
        let model = squared_error_model(&d, 4, 25);

        // Sanity: feature 3 is never used in any split.
        let used = model
            .trees()
            .iter()
            .flat_map(|t| t.nodes().iter())
            .any(|nd| !nd.is_leaf() && nd.split_feature == 3);
        assert!(!used, "feature 3 unexpectedly used in a split");

        let contribs = model.predict_contribs(&d).unwrap();
        let width = nf + 1;
        let mut max_abs = 0f32;
        for row in 0..n {
            max_abs = max_abs.max(contribs[row * width + 3].abs());
        }
        assert!(
            max_abs < 1e-6,
            "unused feature contribution {max_abs} not ~0"
        );
    }

    #[test]
    fn interactions_single_output() {
        let (d, model) = regression_fixture();
        assert_interactions_consistent(&model, &d);
    }

    #[test]
    fn interactions_multiclass() {
        let (d, model) = multiclass_fixture();
        assert_interactions_consistent(&model, &d);
    }

    #[test]
    fn additivity_with_base_margins_dart_and_gblinear() {
        let n = 48;
        let x: Vec<f32> = (0..n)
            .flat_map(|row| [row as f32 / n as f32, (row % 7) as f32])
            .collect();
        let y: Vec<f32> = (0..n).map(|row| row as f32 / 10.0).collect();
        let base: Vec<f32> = (0..n).map(|row| row as f32 / 100.0).collect();
        let d = labeled_dense(&x, n, 2, &y).with_base_margin(&base).unwrap();

        for booster in [BoosterKind::Dart, BoosterKind::GbLinear] {
            let params = TrainingParams::builder()
                .booster(booster)
                .rate_drop(0.5)
                .eta(0.2)
                .max_depth(2)
                .build()
                .unwrap();
            let model = train(&params, &d, 8).unwrap();
            assert_additive(&model, &d);
            assert_interactions_consistent(&model, &d);
        }
    }

    /// The endpoint rule integrates `u^d` over `[0, 1]` exactly for `d ≤ 7`:
    /// that is what makes QuadratureTreeSHAP exact on paths with at most
    /// seven distinct features.
    #[test]
    fn quadrature_rule_integrates_low_degree_polynomials() {
        let rule = super::endpoint_quadrature();
        for d in 0..=7 {
            let got: f64 = (0..super::POINTS)
                .map(|i| f64::from(rule.weights[i]) * f64::from(rule.nodes[i]).powi(d))
                .sum();
            let want = 1.0 / f64::from(d + 1);
            assert!((got - want).abs() < 1e-6, "degree {d}: {got} vs {want}");
        }
        assert!(rule.nodes.windows(2).all(|w| w[0] < w[1]));
    }

    /// Depth-40 `exact` trees on six features: paths are long and repeat
    /// features many times. The quadrature stays exact (≤ 7 distinct
    /// features per path), so the contributions match the textbook f64
    /// recursion, sum to the margin, and the interaction rows sum to them.
    #[test]
    fn deep_exact_trees_stay_additive_and_exact() {
        let (n, nf) = (3000, 6);
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut x = vec![0f32; n * nf];
        for v in &mut x {
            *v = next();
        }
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let r = &x[i * nf..(i + 1) * nf];
                (r[0] * 40.0).sin() * 5.0 + r[1] * r[2] * 8.0 + (r[3] * 25.0).cos() + next()
            })
            .collect();
        let d = labeled_dense(&x, n, nf, &y);
        let params = TrainingParams::builder()
            .tree_method(TreeMethod::Exact)
            .max_depth(40)
            .min_child_weight(0.0)
            .lambda(0.0)
            .eta(0.5)
            .build()
            .unwrap();
        let model = train(&params, &d, 6).unwrap();
        let depth = |t: &crate::tree::RegTree| {
            let mut best = 0;
            let mut stack = vec![(0usize, 0usize)];
            while let Some((nid, dep)) = stack.pop() {
                best = best.max(dep);
                let node = t.node(nid);
                if !node.is_leaf() {
                    stack.push((node.left as usize, dep + 1));
                    stack.push((node.right as usize, dep + 1));
                }
            }
            best
        };
        let deepest = model.trees().iter().map(depth).max().unwrap();
        assert!(deepest >= 25, "trees only reach depth {deepest}");

        let rows = 200;
        let dsub = DMatrix::from_dense(&x[..rows * nf], rows, nf).unwrap();
        assert_additive(&model, &dsub);
        assert_interactions_consistent(&model, &dsub);
        assert_matches_textbook(&model, &dsub, &x[..rows * nf]);
    }

    /// Covers are validated like XGBoost's `CHECK_GE(sum_hess, 0)`.
    #[test]
    fn negative_cover_is_a_model_error() {
        let node = |feature: i32, left: i32, right: i32, value: f32, hess: f32| {
            format!(
                r#"{{"split_feature": {feature}, "split_cond": 0.5, "default_left": true, "left": {left}, "right": {right}, "leaf_value": {value}, "sum_hess": {hess}, "split_gain": 0.0, "is_categorical": false, "cat_begin": 0, "cat_end": 0}}"#
            )
        };
        let json = format!(
            r#"{{"trees": [{{"nodes": [{}, {}, {}], "categories": [], "linear": null}}], "base_score": [0.0], "objective": "reg:squarederror", "num_class": 0, "n_outputs": 1, "n_targets": 1, "n_features": 1, "best_iteration": null, "tree_weights": [], "num_parallel_tree": 1, "linear": null}}"#,
            node(0, 1, 2, 0.0, 1.0),
            node(0, -1, -1, 1.0, -1.0),
            node(0, -1, -1, 2.0, 2.0),
        );
        let model = crate::model::BoostedModel::from_json(&json).unwrap();
        let d = DMatrix::from_dense(&[0.2], 1, 1).unwrap();
        for result in [model.predict_contribs(&d), model.predict_interactions(&d)] {
            assert!(matches!(
                result,
                Err(crate::error::HessboostError::ModelFormat(_))
            ));
        }
    }
}
