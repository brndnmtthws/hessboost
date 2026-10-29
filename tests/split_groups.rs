//! `min_child_groups`: a split whose children hold rows of too few distinct
//! split groups is never taken, in every split search the histogram
//! builder runs; the constraint is off by default and refused where it
//! cannot act.

use std::collections::BTreeSet;

use hessboost::config::{
    BoosterKind, ExtraTrees, GrowPolicy, MultiStrategy, ProcessType, Refresh, SamplingMethod,
    TrainingParamsBuilder,
};
use hessboost::data::{BinnedMatrix, FeatureType, ViewCuts};
use hessboost::diffusion::{DiffusionModel, DiffusionParams};
use hessboost::model::Iterations;
use hessboost::prelude::*;
use hessboost::training::budget::{BudgetConfig, train_with_budget};
use hessboost::training::online::{OnlineModel, OnlineParams};
use hessboost::training::{CrossValidation, Fold};
use hessboost::tree::RegTree;

mod common;
use common::{invalid_data, invalid_param, labeled_dense, lcg, with_threads};

/// Features per row of [`table`].
const COLS: usize = 6;

/// `n` rows of [`COLS`] features, row-major, with their labels: features 0
/// and 1 trend with the row (so blocks of consecutive rows sit together),
/// 2 is noise on a few levels, 3 holds every value once, 4 is a category
/// in `0..cats` (numeric unless typed so), and 5 repeats feature 0 (so
/// candidates of two features tie); features 1 and 2 miss a fifth of
/// their values. The label steps with feature 0 and varies with the
/// others.
fn table(n: usize, cats: u32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut next = lcg(seed);
    let mut x = Vec::with_capacity(COLS * n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;
        let first = t + 0.05 * next();
        let row = [
            first,
            if next() < 0.2 {
                f32::NAN
            } else {
                t * t + 0.1 * next()
            },
            if next() < 0.2 {
                f32::NAN
            } else {
                (next() * 6.0).floor()
            },
            next(),
            (next() * cats as f32).floor(),
            first,
        ];
        let step = if row[0] < 0.5 { 0.0 } else { 2.0 };
        let middle = if row[1].is_nan() { 0.5 } else { row[1] };
        y.push(step + middle - 0.3 * row[4] + 0.2 * next());
        x.extend(row);
    }
    (x, y)
}

/// [`table`] as a dense matrix, feature 4 categorical when `categorical`.
fn dense(n: usize, cats: u32, categorical: bool, seed: u64) -> DMatrix {
    let (x, y) = table(n, cats, seed);
    let data = labeled_dense(&x, COLS, &y);
    if categorical {
        let mut types = vec![FeatureType::Numerical; COLS];
        types[4] = FeatureType::Categorical;
        data.with_feature_types(&types).unwrap()
    } else {
        data
    }
}

/// [`table`] as CSR holding about a third of its entries, too sparse for
/// an index with columns.
fn sparse(n: usize, seed: u64) -> DMatrix {
    let (x, y) = table(n, 4, seed);
    let mut keep = lcg(seed ^ 0x51);
    let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
    for row in x.as_chunks::<COLS>().0 {
        for (c, &v) in row.iter().enumerate() {
            if !v.is_nan() && keep() < 0.35 {
                indices.push(c as u32);
                values.push(v);
            }
        }
        indptr.push(values.len());
    }
    DMatrix::from_csr(indptr, indices, values, COLS)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

/// Every row its own group.
fn row_groups(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

/// Groups of `size` consecutive rows (ids spread over the `u64` range).
fn block_groups(n: usize, size: usize) -> Vec<u64> {
    (0..n)
        .map(|i| ((i / size) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect()
}

fn bytes(model: &BoostedModel) -> Vec<u8> {
    model.encode(ModelFormat::Binary).unwrap()
}

/// Every split search with its data: depth-wise, loss-guided and symmetric
/// growth, `approx`'s per-round cuts, row and column sampling, LightGBM's
/// `extra_trees` and `path_smooth` (with a categorical feature), reuse
/// penalties, wider features than the prefiltered scan takes, one-hot
/// categories, and an index without columns.
fn searches() -> Vec<(&'static str, TrainingParamsBuilder, DMatrix)> {
    let base = || TrainingParams::builder().max_depth(6).eta(0.3);
    let numeric = dense(600, 4, false, 1);
    vec![
        ("depthwise", base(), numeric.clone()),
        (
            "lossguide",
            base()
                .grow_policy(GrowPolicy::LossGuide)
                .max_depth(8)
                .max_leaves(20),
            numeric.clone(),
        ),
        (
            "symmetric",
            base().grow_policy(GrowPolicy::Symmetric).max_depth(4),
            numeric.clone(),
        ),
        (
            "approx",
            base().tree_method(TreeMethod::Approx),
            numeric.clone(),
        ),
        (
            "sampled",
            base().subsample(0.7).colsample_bynode(0.6).seed(3),
            numeric.clone(),
        ),
        (
            "extra trees",
            base().extra_trees(ExtraTrees::default()),
            dense(600, 9, true, 2),
        ),
        (
            "path smooth",
            base().path_smooth(2.0),
            dense(600, 9, true, 3),
        ),
        (
            "reuse penalties",
            base()
                .toad_penalty_feature(0.05)
                .toad_penalty_threshold(0.01),
            numeric,
        ),
        ("wide bins", base().max_bin(1024), dense(2000, 4, false, 4)),
        ("one-hot", base(), dense(600, 3, true, 5)),
        ("sparse", base(), sparse(800, 6)),
    ]
}

/// With every row its own group, a child holds as many groups as rows,
/// which under squared error (a Hessian of 1 per row) is its Hessian sum:
/// `min_child_groups = k` without `min_child_weight` allows exactly the
/// splits `min_child_weight = k` allows, so every search trains the same
/// model, bit for bit.
#[test]
fn row_groups_constrain_like_min_child_weight() {
    for (name, params, data) in searches() {
        let grouped = data
            .clone()
            .with_split_groups(&row_groups(data.n_rows()))
            .unwrap();
        let mut models = Vec::new();
        for k in [2, 9, 60] {
            let by_weight = params.clone().min_child_weight(k as f64).build().unwrap();
            let by_groups = params
                .clone()
                .min_child_weight(0.0)
                .min_child_groups(k)
                .build()
                .unwrap();
            let model = bytes(&train(&by_weight, &data, 6).unwrap());
            assert_eq!(
                model,
                bytes(&train(&by_groups, &grouped, 6).unwrap()),
                "{name}, k = {k}"
            );
            models.push(model);
        }
        // The counts bind: the largest gives another model than the smallest.
        assert_ne!(models[0], models[2], "{name}");
    }
}

/// A categorical partition's order depends on `min_child_weight`, so the
/// two constraints meet there only side by side: row groups at the same
/// count refuse nothing `min_child_weight` allows.
#[test]
fn row_groups_refuse_no_partition_min_child_weight_allows() {
    let data = dense(600, 9, true, 7);
    let grouped = data
        .clone()
        .with_split_groups(&row_groups(data.n_rows()))
        .unwrap();
    for k in [3, 25] {
        let base = TrainingParams::builder()
            .max_depth(6)
            .min_child_weight(k as f64);
        let by_weight = base.clone().build().unwrap();
        let both = base.min_child_groups(k).build().unwrap();
        assert_eq!(
            bytes(&train(&by_weight, &data, 6).unwrap()),
            bytes(&train(&both, &grouped, 6).unwrap()),
            "k = {k}"
        );
    }
}

/// The distinct groups of the training rows below every node of `tree`,
/// from each row's leaf (`leaves[r]`).
fn node_groups(tree: &RegTree, leaves: &[u32], groups: &[u64]) -> Vec<BTreeSet<u64>> {
    let nodes = tree.nodes();
    let mut parent = vec![None; nodes.len()];
    for (id, node) in nodes.iter().enumerate() {
        if !node.is_leaf() {
            parent[node.left as usize] = Some(id);
            parent[node.right as usize] = Some(id);
        }
    }
    let mut held = vec![BTreeSet::new(); nodes.len()];
    for (&leaf, &group) in leaves.iter().zip(groups) {
        let mut at = Some(leaf as usize);
        while let Some(id) = at {
            held[id].insert(group);
            at = parent[id];
        }
    }
    held
}

/// The group counts of both children of every split of `model`, over the
/// rows of `data` (grouped by `groups`) that reach them.
fn child_group_counts(model: &BoostedModel, data: &DMatrix, groups: &[u64]) -> Vec<[usize; 2]> {
    let leaves = model.predict_leaf(data, ..).unwrap();
    let width = leaves.width();
    let mut counts = Vec::new();
    for (t, tree) in model.trees().iter().enumerate() {
        let rows: Vec<u32> = leaves
            .as_slice()
            .iter()
            .skip(t)
            .step_by(width)
            .copied()
            .collect();
        let held = node_groups(tree, &rows, groups);
        for node in tree.nodes().iter().filter(|node| !node.is_leaf()) {
            counts.push([node.left, node.right].map(|child| held[child as usize].len()));
        }
    }
    counts
}

/// Every split of a constrained model leaves rows of at least `k` distinct
/// groups in each child: counted over every training row, a superset of
/// the rows the node sampled, so this holds under sampling too.
#[test]
fn every_split_keeps_enough_groups_on_both_sides() {
    let mut splits = 0;
    for (name, params, data) in searches() {
        let groups = block_groups(data.n_rows(), 13);
        let grouped = data.clone().with_split_groups(&groups).unwrap();
        for k in [2, 6] {
            let params = params.clone().min_child_groups(k).build().unwrap();
            let model = train(&params, &grouped, 6).unwrap();
            for counts in child_group_counts(&model, &grouped, &groups) {
                splits += 1;
                assert!(
                    counts.iter().all(|&count| count >= k),
                    "{name}, k = {k}: children hold {counts:?} groups"
                );
            }
        }
    }
    assert!(splits > 500, "{splits} splits checked");
}

/// A child's groups are its node's, so a node with fewer than
/// `min_child_groups` groups never splits, and one with exactly that many
/// splits only where both children keep them all.
#[test]
fn nodes_with_too_few_groups_stay_leaves() {
    let data = dense(600, 4, false, 8);
    let n = data.n_rows();
    // Three groups, each a third of the rows (so of feature 0's range).
    let groups: Vec<u64> = (0..n).map(|i| (3 * i / n) as u64).collect();
    let grouped = data.with_split_groups(&groups).unwrap();
    let params = |k| {
        TrainingParams::builder()
            .max_depth(5)
            .min_child_groups(k)
            .build()
            .unwrap()
    };
    let stumps = train(&params(4), &grouped, 5).unwrap();
    assert!(stumps.trees().iter().all(|tree| tree.num_nodes() == 1));
    let all_three = train(&params(3), &grouped, 5).unwrap();
    let counts = child_group_counts(&all_three, &grouped, &groups);
    assert_ne!(counts.len(), 0, "the trees split");
    assert!(counts.iter().all(|&counts| counts == [3, 3]), "{counts:?}");
    // The constraint binds: unconstrained trees split the groups apart.
    let free = train(&params(1), &grouped, 5).unwrap();
    assert!(
        child_group_counts(&free, &grouped, &groups)
            .iter()
            .any(|counts| counts.contains(&1))
    );
}

/// `0` and `1` constrain nothing: training with them, groups attached or
/// not, is default training bit for bit.
#[test]
fn off_leaves_the_default_model() {
    let data = dense(600, 4, false, 9);
    let grouped = data
        .clone()
        .with_split_groups(&block_groups(600, 7))
        .unwrap();
    let default = bytes(&train(&TrainingParams::default(), &data, 8).unwrap());
    for k in [0, 1] {
        let params = TrainingParams::builder()
            .min_child_groups(k)
            .build()
            .unwrap();
        assert_eq!(bytes(&train(&params, &data, 8).unwrap()), default);
        assert_eq!(bytes(&train(&params, &grouped, 8).unwrap()), default);
    }
}

/// The groups are counted feature by feature in parallel: any thread
/// count trains the same model.
#[test]
fn any_thread_count_trains_the_same_model() {
    let data = dense(3000, 9, true, 10)
        .with_split_groups(&block_groups(3000, 17))
        .unwrap();
    for policy in [
        GrowPolicy::DepthWise,
        GrowPolicy::LossGuide,
        GrowPolicy::Symmetric,
    ] {
        let mut params = TrainingParams::builder()
            .grow_policy(policy)
            .max_depth(5)
            .min_child_groups(5);
        if policy == GrowPolicy::LossGuide {
            params = params.max_leaves(24);
        }
        let data = if policy == GrowPolicy::Symmetric {
            // Symmetric growth takes numerical features only.
            dense(3000, 9, false, 10)
                .with_split_groups(&block_groups(3000, 17))
                .unwrap()
        } else {
            data.clone()
        };
        let params = params.build().unwrap();
        let one = with_threads(1, || bytes(&train(&params, &data, 6).unwrap()));
        let four = with_threads(4, || bytes(&train(&params, &data, 6).unwrap()));
        assert_eq!(one, four, "{policy:?}");
    }
}

/// A binned matrix carries its source's groups (and takes new ones), and
/// trains as its source does.
#[test]
fn binned_matrices_train_as_their_source() {
    let data = dense(800, 9, true, 11)
        .with_split_groups(&block_groups(800, 9))
        .unwrap();
    let params = TrainingParams::builder()
        .max_depth(5)
        .max_bin(64)
        .min_child_groups(4)
        .build()
        .unwrap();
    let binned = BinnedMatrix::new(&data, 64).unwrap();
    assert_eq!(binned.split_groups(), data.split_groups());
    assert_eq!(
        bytes(&train(&params, &binned, 6).unwrap()),
        bytes(&train(&params, &data, 6).unwrap())
    );
    let regrouped = binned.with_split_groups(&block_groups(800, 40)).unwrap();
    let source = data.with_split_groups(&block_groups(800, 40)).unwrap();
    assert_eq!(
        bytes(&train(&params, &regrouped, 6).unwrap()),
        bytes(&train(&params, &source, 6).unwrap())
    );
}

/// Cross-validation trains each fold on its own rows' groups: every
/// fold's margins are those of a model trained on a matrix of the fold's
/// rows and groups built from scratch.
#[test]
fn folds_count_their_own_rows_groups() {
    let (x, y) = table(600, 4, 12);
    let groups = block_groups(600, 11);
    let data = labeled_dense(&x, COLS, &y)
        .with_split_groups(&groups)
        .unwrap();
    let params = TrainingParams::builder()
        .max_depth(5)
        .min_child_groups(6)
        .build()
        .unwrap();
    let folds = Fold::k_fold(600, 3, 0).unwrap();
    let report = CrossValidation::new(&params, &data, 5, folds.clone())
        .run()
        .unwrap();
    let matrix = |rows: &[usize]| {
        let x: Vec<f32> = rows
            .iter()
            .flat_map(|&r| x[COLS * r..COLS * (r + 1)].iter().copied())
            .collect();
        let y: Vec<f32> = rows.iter().map(|&r| y[r]).collect();
        let g: Vec<u64> = rows.iter().map(|&r| groups[r]).collect();
        labeled_dense(&x, COLS, &y).with_split_groups(&g).unwrap()
    };
    for (fold, reported) in folds.iter().zip(&report.folds) {
        let model = train(&params, &matrix(&fold.train), 5).unwrap();
        let margins = model
            .predict_margin(&matrix(&fold.test), Iterations::Best)
            .unwrap();
        assert_eq!(margins.as_slice(), reported.margins.as_slice());
    }
}

/// Rows of zero weight never count toward a child's groups: they add
/// nothing to its fit, and the cuts are sketched without them, so binning
/// clamps a value of theirs beyond the last cut into the last bin while the
/// model routes it past the cut. Here group 1 holds the only weighted rows
/// with a value, so no split leaves weighted rows of two groups on both
/// sides, though the last bin holds a zero-weight row of group 2 at a value
/// the cuts never saw (under `approx`, its Hessian of zero keeps it out of
/// every round's cuts).
#[test]
fn zero_weight_rows_never_count() {
    let data = DMatrix::from_dense(&[0.0, 1.0, 10.0, f32::NAN, f32::NAN], 5, 1)
        .unwrap()
        .with_labels(&[0.0, 0.0, 0.0, 1.0, 1.0])
        .unwrap()
        .with_weights(&[1.0, 1.0, 0.0, 1.0, 1.0])
        .unwrap()
        .with_split_groups(&[1, 1, 2, 3, 4])
        .unwrap();
    for method in [TreeMethod::Hist, TreeMethod::Approx] {
        let nodes = |k| {
            let params = TrainingParams::builder()
                .max_depth(1)
                .tree_method(method)
                .min_child_groups(k)
                .build()
                .unwrap();
            train(&params, &data, 1).unwrap().trees()[0].num_nodes()
        };
        // Unconstrained, the present values split from the missing ones.
        assert_eq!(nodes(1), 3, "{method:?}");
        assert_eq!(nodes(2), 1, "{method:?}");
    }
}

/// The constraint needs groups on the training matrix (of the right
/// length) whose weighted rows the `hist` cuts all saw, a tree booster's
/// histogram split search with one output per tree, and a training path
/// that searches splits.
#[test]
fn unsupported_requests_are_refused() {
    let data = dense(300, 4, false, 13);
    let constrained = || TrainingParams::builder().min_child_groups(3);
    assert_eq!(
        invalid_data(train(&constrained().build().unwrap(), &data, 2)),
        ("split_groups", None)
    );
    assert!(matches!(
        data.clone().with_split_groups(&[1, 2, 3]),
        Err(HessboostError::DimensionMismatch { .. })
    ));
    // A view taking its source's cuts that weighs rows its source weighs
    // zero holds weighted rows its cuts never saw, which its bins need not
    // route as their values do.
    let weights: Vec<f32> = (0..300).map(|i| f32::from(u8::from(i % 10 != 0))).collect();
    let view = data
        .clone()
        .with_weights(&weights)
        .unwrap()
        .select_rows_with_cuts(&(0..300).collect::<Vec<_>>(), ViewCuts::Source)
        .unwrap()
        .with_weights(&[1.0; 300])
        .unwrap()
        .with_split_groups(&block_groups(300, 5))
        .unwrap();
    assert_eq!(
        invalid_data(train(&constrained().build().unwrap(), &view, 2)),
        ("weights", None)
    );
    for refused in [
        constrained().booster(BoosterKind::GbLinear),
        constrained().tree_method(TreeMethod::Exact),
        constrained().multi_strategy(MultiStrategy::MultiOutputTree),
    ] {
        assert_eq!(invalid_param(refused.build()), "min_child_groups");
    }
    let grouped = data
        .clone()
        .with_split_groups(&block_groups(300, 5))
        .unwrap();
    let params = constrained().build().unwrap();
    // `approx` sketches a constant Hessian's cuts once, from the first
    // tree's gradient sample, which later samples' rows may lie outside.
    let sampled = constrained()
        .tree_method(TreeMethod::Approx)
        .sampling_method(SamplingMethod::GradientBased)
        .subsample(0.5)
        .build()
        .unwrap();
    assert_eq!(
        invalid_param(train(&sampled, &grouped, 2)),
        "min_child_groups"
    );
    // Budget mode, refresh, and in-place updates run no split search that
    // could apply it.
    assert_eq!(
        invalid_param(train_with_budget(
            &params,
            &grouped,
            &BudgetConfig::new(0.5)
        )),
        "budget"
    );
    let first = train(&TrainingParams::default(), &grouped, 2).unwrap();
    let refresh = constrained()
        .process_type(ProcessType::Update(Refresh::default()))
        .build()
        .unwrap();
    assert_eq!(
        invalid_param(
            Trainer::new(&refresh, &grouped, 2)
                .init_model(&first)
                .train()
        ),
        "process_type"
    );
    assert_eq!(
        invalid_param(OnlineModel::train(
            &params,
            &grouped,
            2,
            OnlineParams::exact()
        )),
        "params"
    );
    // A diffusion model's GBDTs train on noisy copies, without groups.
    let mut diffusion = DiffusionParams::default();
    diffusion.training.min_child_groups = 2;
    assert_eq!(
        invalid_param(DiffusionModel::fit(&diffusion, &grouped)),
        "training"
    );
}
