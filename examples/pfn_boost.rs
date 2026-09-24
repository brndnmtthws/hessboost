//! Boosting from a pretrained prior: PFN-Boost / LLM-Boost (Jayawardhana et
//! al., 2025, <https://arxiv.org/abs/2502.02672>) through
//! [`DMatrix::with_base_margin`]. Run:
//! `cargo run --release --example pfn_boost` (synthetic demo) or
//! `cargo run --release --example pfn_boost -- <dir>` (your own prior scores).
//!
//! # Method
//!
//! A pretrained transformer (TabPFN, or an LLM prompted with the serialized
//! row) scores every row once. Those scores seed the ensemble in place of the
//! usual constant intercept, so the trees fit the residual of the prior; the
//! paper describes it as replacing the first tree with the transformer. The
//! margin after `i` trees is
//!
//! ```text
//! margin_i(x) = trees_1..i(x) + s * score(x) + C
//! ```
//!
//! `s >= 0` is the scaling parameter (tuned on validation data after the tree
//! hyperparameters; the paper searches `[1e-4, 1e4]` log-uniformly with
//! Optuna): `s = 0` is plain XGBoost and a large `s` approaches the prior
//! alone. `C` only centers the scaled scores; here it is the from-scratch
//! intercept minus the mean scaled training score, so `s = 0` reproduces the
//! from-scratch model exactly.
//!
//! # `base_margin` at prediction time
//!
//! A `DMatrix` with a `base_margin` starts every row at that margin *instead
//! of* the model's intercept (`base_score`), during training, for every eval
//! set, and in `predict`/`predict_margin`/`predict_contribs`. The prior is not
//! stored in the model: the eval and test matrices must carry the same
//! `s * score + C` margin, built with the training `s` and `C`. A test matrix
//! without one silently falls back to the intercept, which this example
//! prints as a warning case. When saving the model, save `s` and `C` with it.
//!
//! # Plugging in real TabPFN scores
//!
//! Pass a directory holding `train.csv`, `valid.csv`, `test.csv` (header row,
//! label in the first column, then the features) and `train_prior.csv`,
//! `valid_prior.csv`, `test_prior.csv` (header row, then one prior logit per
//! row, in the same order). This Python script writes them from TabPFN (tested
//! with `tabpfn==2.0.9`, whose v2 weights download without an account; newer
//! releases need a Prior Labs license token in `TABPFN_TOKEN`):
//!
//! ```python
//! # uv run --python 3.12 --with tabpfn==2.0.9 --with scikit-learn python dump_tabpfn.py
//! import os
//! import numpy as np
//! from sklearn.datasets import load_breast_cancer
//! from sklearn.model_selection import cross_val_predict, train_test_split
//! from tabpfn import TabPFNClassifier
//!
//! X, y = load_breast_cancer(return_X_y=True)
//! X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
//! X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=0)
//!
//! def logit(proba):
//!     p = np.clip(proba[:, 1], 1e-6, 1 - 1e-6)
//!     return np.log(p) - np.log1p(-p)
//!
//! pfn = TabPFNClassifier()
//! # Out-of-fold scores for the training rows: TabPFN scoring rows it holds in
//! # its own context is optimistic, which would make the trees under-correct.
//! prior_tr = logit(cross_val_predict(pfn, X_tr, y_tr, cv=5, method="predict_proba"))
//! pfn.fit(X_tr, y_tr)
//! splits = {
//!     "train": (X_tr, y_tr, prior_tr),
//!     "valid": (X_va, y_va, logit(pfn.predict_proba(X_va))),
//!     "test": (X_te, y_te, logit(pfn.predict_proba(X_te))),
//! }
//! os.makedirs("pfn_data", exist_ok=True)
//! for name, (X, y, prior) in splits.items():
//!     header = ",".join(["label"] + [f"f{i}" for i in range(X.shape[1])])
//!     np.savetxt(f"pfn_data/{name}.csv", np.column_stack([y, X]), delimiter=",", header=header, comments="")
//!     np.savetxt(f"pfn_data/{name}_prior.csv", prior, header="prior", comments="")
//! ```
//!
//! then `cargo run --release --example pfn_boost -- pfn_data`. For
//! `multi:softprob`, `with_base_margin` also takes `n_rows * num_class`
//! per-class scores laid out row-major.

use hessboost::config::BoosterKind;
use hessboost::data::{CsvOptions, load_csv};
use hessboost::metric::{Auc, LogLoss, Metric};
use hessboost::prelude::*;
use std::path::Path;

mod common;
use common::{fill_random, lcg};

/// Candidate values of the scaling parameter `s`, chosen on validation logloss.
const SCALES: [f32; 8] = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0];
const MAX_ROUNDS: usize = 2000;
const EARLY_STOPPING: usize = 30;

/// One data split: labelled features plus the prior's raw score per row.
struct Split {
    data: DMatrix,
    prior: Vec<f32>,
}

impl Split {
    fn labels(&self) -> &[f32] {
        self.data.labels().expect("splits are built with labels")
    }

    /// The split with `margin = s * prior + c` attached as its `base_margin`.
    fn with_prior(&self, s: f32, c: f32) -> Result<DMatrix> {
        let margin: Vec<f32> = self.prior.iter().map(|&p| s * p + c).collect();
        self.data.clone().with_base_margin(&margin)
    }
}

/// Held-out logloss and AUC of probabilities `p`.
#[derive(Clone, Copy)]
struct Score {
    logloss: f64,
    auc: f64,
}

fn score(p: &[f32], y: &[f32]) -> Score {
    Score {
        logloss: LogLoss.eval(p, y, None),
        auc: Auc.eval(p, y, None),
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// The three contenders on one dataset, all scored on the test split.
struct Comparison {
    prior: Score,
    scratch: Score,
    scratch_trees: usize,
    boosted: Score,
    boosted_trees: usize,
    scale: f32,
    /// PFN-Boost evaluated on a test matrix *without* the prior margin.
    forgot_prior: Score,
}

fn params() -> Result<TrainingParams> {
    TrainingParams::builder()
        .objective("binary:logistic")
        .eval_metric("logloss")
        .max_depth(3)
        .eta(0.05)
        .build()
}

/// Rounds actually used at prediction time (early stopping keeps the best).
fn used_rounds(model: &BoostedModel) -> usize {
    model
        .best_iteration()
        .map_or(model.num_trees(), |it| it + 1)
}

fn compare(train: &Split, valid: &Split, test: &Split) -> Result<Comparison> {
    let params = params()?;
    let prior = score(
        &test.prior.iter().map(|&z| sigmoid(z)).collect::<Vec<_>>(),
        test.labels(),
    );

    // Baseline: boosting from the constant intercept, early-stopped on `valid`.
    let scratch = Trainer::new(&params, &train.data, MAX_ROUNDS)
        .eval(&valid.data, "valid")
        .early_stopping_rounds(EARLY_STOPPING)
        .train()?
        .model;

    // PFN-Boost: seed train, valid, and test with the same `s * score + C`,
    // then pick `s` by validation logloss (the model predicts at its best
    // early-stopping iteration).
    let prior_mean =
        train.prior.iter().map(|&p| f64::from(p)).sum::<f64>() / train.prior.len() as f64;
    let mut best: Option<(f64, f32, f32, BoostedModel)> = None;
    for s in SCALES {
        let c = scratch.base_score() - s * prior_mean as f32;
        let dtrain = train.with_prior(s, c)?;
        let dvalid = valid.with_prior(s, c)?;
        let model = Trainer::new(&params, &dtrain, MAX_ROUNDS)
            .eval(&dvalid, "valid")
            .early_stopping_rounds(EARLY_STOPPING)
            .train()?
            .model;
        let valid_loss = LogLoss.eval(&model.predict(&dvalid)?, valid.labels(), None);
        if best.as_ref().is_none_or(|(loss, ..)| valid_loss < *loss) {
            best = Some((valid_loss, s, c, model));
        }
    }
    let (_, scale, c, boosted) = best.expect("SCALES is non-empty");

    let dtest = test.with_prior(scale, c)?;
    Ok(Comparison {
        prior,
        scratch: score(&scratch.predict(&test.data)?, test.labels()),
        scratch_trees: used_rounds(&scratch),
        boosted: score(&boosted.predict(&dtest)?, test.labels()),
        boosted_trees: used_rounds(&boosted),
        scale,
        forgot_prior: score(&boosted.predict(&test.data)?, test.labels()),
    })
}

fn print_header() {
    println!(
        "{:>7} | {:>15} | {:>21} | {:>27} | {:>11}",
        "n_train", "prior alone", "from scratch", "PFN-Boost (s chosen)", "no margin*"
    );
    println!(
        "{:>7} | {:>7} {:>7} | {:>7} {:>7} {:>5} | {:>7} {:>7} {:>5} {:>5} | {:>11}",
        "", "logloss", "AUC", "logloss", "AUC", "trees", "logloss", "AUC", "trees", "s", "logloss"
    );
}

fn print_row(n_train: usize, r: &Comparison) {
    println!(
        "{:>7} | {:>7.4} {:>7.4} | {:>7.4} {:>7.4} {:>5} | {:>7.4} {:>7.4} {:>5} {:>5} | {:>11.4}",
        n_train,
        r.prior.logloss,
        r.prior.auc,
        r.scratch.logloss,
        r.scratch.auc,
        r.scratch_trees,
        r.boosted.logloss,
        r.boosted.auc,
        r.boosted_trees,
        r.scale,
        r.forgot_prior.logloss,
    );
}

fn print_footer() {
    println!(
        "\n* the PFN-Boost model predicting a test matrix built WITHOUT its base_margin: \
         rows start from the intercept, so the prior is silently lost."
    );
}

// ---- Synthetic demo -------------------------------------------------------

const N_FEATURES: usize = 6;

/// Logit of the task we want to solve: a linear trend plus an interaction
/// the prior never saw.
fn target_logit(x: &[f32]) -> f32 {
    let bump = if x[3] > 0.5 && x[4] > 0.5 { 2.5 } else { 0.0 };
    4.0 * (x[0] - 0.5) - 3.0 * (x[1] - 0.5) + 2.0 * (x[2] - 0.5) + bump - 0.8
}

/// Logit of the related "pretraining" task: similar but not identical
/// coefficients and no interaction, so the prior is useful yet biased.
fn pretraining_logit(x: &[f32]) -> f32 {
    3.5 * (x[0] - 0.5) - 3.0 * (x[1] - 0.5) + 1.5 * (x[2] - 0.5)
}

/// `n` rows with uniform features and Bernoulli labels from `logit`.
fn sample(n: usize, seed: u64, logit: fn(&[f32]) -> f32) -> Result<DMatrix> {
    let mut rng = lcg(seed);
    let mut x = vec![0f32; n * N_FEATURES];
    let mut y = vec![0f32; n];
    for (row, label) in x.as_chunks_mut::<N_FEATURES>().0.iter_mut().zip(&mut y) {
        fill_random(&mut rng, row);
        *label = f32::from(u8::from(rng() < sigmoid(logit(row))));
    }
    DMatrix::from_dense(&x, n, N_FEATURES)?.with_labels(&y)
}

/// First `n` rows of `data`.
fn head(data: &DMatrix, n: usize) -> Result<DMatrix> {
    data.select_rows(&(0..n).collect::<Vec<_>>())
}

fn run_synthetic() -> Result<()> {
    // Stand-in for TabPFN: TabPFN has no Rust port, so the "pretrained prior"
    // here is a logistic model (`gblinear`) fit on a large sample from a
    // related task. Any model that emits per-row logits plays the same role.
    let pretraining = sample(20_000, 7, pretraining_logit)?;
    let prior_params = TrainingParams::builder()
        .objective("binary:logistic")
        .booster(BoosterKind::GbLinear)
        .eta(0.5)
        .build()?;
    let prior_model = train(&prior_params, &pretraining, 100)?;
    let as_split = |data: DMatrix| -> Result<Split> {
        let prior = prior_model.predict_margin(&data)?;
        Ok(Split { data, prior })
    };

    println!(
        "stand-in prior: logistic model fit on {} pretraining rows of a related task",
        pretraining.n_rows()
    );
    println!("test set: 20000 rows of the target task; train/valid = 80/20 of n_train\n");
    let test = as_split(sample(20_000, 11, target_logit)?)?;
    let pool = sample(8_000, 13, target_logit)?;

    print_header();
    for n_train in [50, 100, 250, 1000, 4000, 8000] {
        let data = head(&pool, n_train)?;
        let n_fit = n_train * 4 / 5;
        let train = as_split(head(&data, n_fit)?)?;
        let valid = as_split(data.select_rows(&(n_fit..n_train).collect::<Vec<_>>())?)?;
        let r = compare(&train, &valid, &test)?;
        print_row(n_train, &r);
    }
    print_footer();
    Ok(())
}

// ---- User-supplied prior scores -------------------------------------------

/// Read one prior logit per row (after a header line) from `path`.
fn read_prior(path: &Path) -> Result<Vec<f32>> {
    let text = std::fs::read_to_string(path)?;
    text.lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.trim().parse::<f32>().map_err(|e| {
                HessboostError::invalid_param(
                    "prior",
                    format!("{}: `{line}` is not a number ({e})", path.display()),
                )
            })
        })
        .collect()
}

fn load_split(dir: &Path, name: &str) -> Result<Split> {
    let data = load_csv(dir.join(format!("{name}.csv")), &CsvOptions::default())?;
    let prior = read_prior(&dir.join(format!("{name}_prior.csv")))?;
    if prior.len() != data.n_rows() {
        return Err(HessboostError::DimensionMismatch {
            what: "prior rows",
            expected: data.n_rows(),
            got: prior.len(),
        });
    }
    Ok(Split { data, prior })
}

fn run_csv(dir: &Path) -> Result<()> {
    let train = load_split(dir, "train")?;
    let valid = load_split(dir, "valid")?;
    let test = load_split(dir, "test")?;
    println!(
        "prior scores from {}: {} train / {} valid / {} test rows\n",
        dir.display(),
        train.data.n_rows(),
        valid.data.n_rows(),
        test.data.n_rows()
    );
    let r = compare(&train, &valid, &test)?;
    print_header();
    print_row(train.data.n_rows() + valid.data.n_rows(), &r);
    print_footer();
    Ok(())
}

fn main() -> Result<()> {
    match std::env::args_os().nth(1) {
        Some(dir) => run_csv(Path::new(&dir)),
        None => run_synthetic(),
    }
}
