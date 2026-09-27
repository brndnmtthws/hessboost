//! LightGBM import parity against real LightGBM predictions.
//!
//! Fixtures come from `scripts/gen_lightgbm_fixtures.py` (LightGBM 4.7.0,
//! single thread) and follow the schema documented there: each case is a
//! `Booster.save_model` text model plus LightGBM's predictions on a test
//! matrix. For every importable case the test checks pointwise raw scores,
//! predictions, SHAP contributions (`pred_contrib`), leaf indices, a sliced
//! model's raw scores, and that native binary, native JSON, and (where
//! representable) XGBoost JSON round trips predict identically; refused
//! cases must fail with a `ModelFormat` error naming the reason.
//!
//! ```sh
//! uv run --with-requirements scripts/requirements-lightgbm.txt python scripts/gen_lightgbm_fixtures.py
//! cargo nextest run --test lightgbm_parity --release --run-ignored only --no-capture
//! ```

use hessboost::prelude::{BoostedModel, DMatrix, HessboostError};
use serde::Deserialize;
mod common;
use common::bits::same_bits;
use common::fixtures::{fixtures_dir, nan_for_null};

/// Pointwise tolerance, relative to `max(1, |LightGBM|)`: hessboost stores
/// leaf values as `f32` and sums them in `f32`, LightGBM in `f64`.
const TOL: f64 = 1e-5;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    n_cols: usize,
    n_test: usize,
    #[serde(deserialize_with = "nan_for_null")]
    x_test: Vec<f32>,
    expect: String,
    error: Option<String>,
    objective: Option<String>,
    n_outputs: Option<usize>,
    raw: Option<Vec<f64>>,
    pred: Option<Vec<f64>>,
    contribs: Option<Vec<f64>>,
    leaf: Option<Vec<u32>>,
    slice_iterations: Option<usize>,
    raw_slice: Option<Vec<f64>>,
    xgboost_export: Option<bool>,
}
/// The largest `|a - b| / max(1, |b|)`, or an error naming the first entry
/// past [`TOL`].
fn compare(what: &str, got: &[f32], want: &[f64]) -> Result<f64, String> {
    if got.len() != want.len() {
        return Err(format!(
            "{what}: {} values, LightGBM has {}",
            got.len(),
            want.len()
        ));
    }
    let mut worst = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let delta = (f64::from(g) - w).abs() / w.abs().max(1.0);
        if delta.is_nan() || delta > TOL {
            return Err(format!("{what}[{i}]: hessboost {g}, LightGBM {w}"));
        }
        worst = worst.max(delta);
    }
    Ok(worst)
}
fn check_import(fx: &Fixture, text: &str) -> Result<String, String> {
    let model = BoostedModel::from_lightgbm_text(text).map_err(|e| format!("import: {e}"))?;
    let expected_objective = fx.objective.as_deref().unwrap_or_default();
    if model.objective().name() != expected_objective {
        return Err(format!(
            "objective {}, expected {expected_objective}",
            model.objective().name()
        ));
    }
    let k = fx.n_outputs.unwrap_or(1);
    if model.n_outputs() != k {
        return Err(format!("{} outputs, expected {k}", model.n_outputs()));
    }
    let data = DMatrix::from_dense(&fx.x_test, fx.n_test, fx.n_cols).map_err(|e| e.to_string())?;
    let margin = model.predict_margin(&data).map_err(|e| e.to_string())?;
    let raw = compare(
        "raw",
        margin.as_slice(),
        fx.raw.as_deref().unwrap_or_default(),
    )?;
    let predictions = model.predict(&data).map_err(|e| e.to_string())?;
    let pred = compare(
        "pred",
        predictions.as_slice(),
        fx.pred.as_deref().unwrap_or_default(),
    )?;
    let shap = match &fx.contribs {
        Some(contribs) => {
            let got = model.predict_contribs(&data).map_err(|e| e.to_string())?;
            format!("{:.1e}", compare("contribs", got.as_slice(), contribs)?)
        }
        // LightGBM refuses SHAP for linear trees; so does hessboost.
        None => match model.predict_contribs(&data) {
            Err(HessboostError::InvalidParameter { .. }) => "refused".to_string(),
            other => return Err(format!("contribs of a linear-leaf model: {other:?}")),
        },
    };

    // Leaf indices: LightGBM's leaf `j` is hessboost node `num_leaves - 1 + j`.
    let leaves = model.predict_leaf(&data).map_err(|e| e.to_string())?;
    let expected_leaves: Vec<u32> = fx
        .leaf
        .as_deref()
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(i, &j)| {
            let tree = &model.trees()[i % model.num_trees()];
            (tree.num_leaves() - 1) as u32 + j
        })
        .collect();
    if leaves.as_slice() != expected_leaves {
        return Err("leaf indices differ".to_string());
    }

    let iterations = fx.slice_iterations.unwrap_or(1);
    let sliced = model.slice(..iterations, 1).map_err(|e| e.to_string())?;
    let sliced_margin = sliced.predict_margin(&data).map_err(|e| e.to_string())?;
    compare(
        "raw_slice",
        sliced_margin.as_slice(),
        fx.raw_slice.as_deref().unwrap_or_default(),
    )?;

    let from_bytes = BoostedModel::from_bytes(&model.to_bytes().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let from_json = BoostedModel::from_json(&model.to_json().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    for (what, restored) in [("native binary", &from_bytes), ("native JSON", &from_json)] {
        if !same_bits(
            restored
                .predict(&data)
                .map_err(|e| e.to_string())?
                .as_slice(),
            predictions.as_slice(),
        ) {
            return Err(format!("{what} round trip changed predictions"));
        }
    }
    match (model.to_xgboost_json(), fx.xgboost_export.unwrap_or(false)) {
        (Ok(json), true) => {
            let restored = BoostedModel::from_xgboost_json(&json).map_err(|e| e.to_string())?;
            if !same_bits(
                restored
                    .predict(&data)
                    .map_err(|e| e.to_string())?
                    .as_slice(),
                predictions.as_slice(),
            ) {
                return Err("XGBoost JSON round trip changed predictions".to_string());
            }
        }
        (Err(HessboostError::ModelFormat(_)), false) => {}
        (result, expected) => {
            return Err(format!(
                "XGBoost export: {:?}, expected {}",
                result.map(|_| ()),
                if expected { "success" } else { "a refusal" }
            ));
        }
    }
    Ok(format!("raw {raw:.1e} pred {pred:.1e} contribs {shap}"))
}

fn check_refusal(fx: &Fixture, text: &str) -> Result<String, String> {
    let needle = fx.error.as_deref().unwrap_or_default();
    match BoostedModel::from_lightgbm_text(text) {
        Err(HessboostError::ModelFormat(message)) if message.contains(needle) => {
            Ok(format!("refused: {message}"))
        }
        Err(other) => Err(format!("refused with `{other}`, expected `{needle}`")),
        Ok(_) => Err(format!("imported, expected a refusal naming `{needle}`")),
    }
}

#[test]
#[ignore = "requires fixtures from scripts/gen_lightgbm_fixtures.py"]
fn lightgbm_parity() {
    let dir = fixtures_dir().join("lightgbm");
    let paths = common::fixtures::json_paths(&dir, "LightGBM", "scripts/gen_lightgbm_fixtures.py");
    let mut failures = Vec::new();
    for path in &paths {
        let fx: Fixture = common::fixtures::load_json(path);
        let text = std::fs::read_to_string(path.with_extension("txt")).unwrap();
        let result = match fx.expect.as_str() {
            "import" => check_import(&fx, &text),
            "refuse" => check_refusal(&fx, &text),
            other => Err(format!("unknown expectation `{other}`")),
        };
        match result {
            Ok(summary) => println!("ok   {:<34} {summary}", fx.name),
            Err(message) => {
                println!("FAIL {:<34} {message}", fx.name);
                failures.push(fx.name);
            }
        }
    }
    assert!(failures.is_empty(), "failing fixtures: {failures:?}");
}
