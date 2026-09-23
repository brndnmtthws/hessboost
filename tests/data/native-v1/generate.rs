//! Generator of the native-format version-1 fixtures in this directory,
//! run as an example of a hessboost 0.1.1 checkout (commit 1af4e21); see
//! `tests/native_format.rs` for the commands. Not compiled by this crate.
//!
//! Trains small models on a deterministic matrix (hist, exact, multiclass
//! softprob, DART, categorical, gblinear, and an early-stopped model) and
//! writes each as `<name>.bin` (`save_binary`) and `<name>.json`
//! (`save_json`), plus `predictions.json` with the `f32` bit patterns of
//! `predict` and `predict_margin` on the training matrix.
use hessboost::prelude::*;
use std::fmt::Write as _;

const ROWS: usize = 48;
const COLS: usize = 4;

fn feature(i: usize, j: usize) -> f32 {
    let h = (i as u64 * 2_654_435_761 + j as u64 * 40_503 + 17) % 1009;
    if h % 11 == 0 {
        f32::NAN
    } else if j == 0 {
        (h % 5) as f32
    } else {
        h as f32 / 1009.0
    }
}

fn features() -> Vec<f32> {
    (0..ROWS)
        .flat_map(|i| (0..COLS).map(move |j| feature(i, j)))
        .collect()
}

fn get(x: &[f32], i: usize, j: usize) -> f32 {
    let v = x[i * COLS + j];
    if v.is_nan() { 0.0 } else { v }
}

fn main() -> Result<()> {
    let out = std::path::Path::new(&std::env::args().nth(1).expect("output dir")).to_path_buf();
    std::fs::create_dir_all(&out)?;
    let x = features();
    let reg: Vec<f32> = (0..ROWS)
        .map(|i| 3.0 * get(&x, i, 1) - 2.0 * get(&x, i, 2) + 0.5 * get(&x, i, 0))
        .collect();
    let bin: Vec<f32> = (0..ROWS)
        .map(|i| f32::from(get(&x, i, 1) + get(&x, i, 3) > 1.0))
        .collect();
    let cls: Vec<f32> = (0..ROWS)
        .map(|i| ((get(&x, i, 2) * 3.0) as usize).min(2) as f32)
        .collect();
    let cat_bin: Vec<f32> = (0..ROWS)
        .map(|i| f32::from((get(&x, i, 0) as u32) % 2 == 1))
        .collect();
    let d = |y: &[f32]| DMatrix::from_dense(&x, ROWS, COLS).and_then(|m| m.with_labels(y));
    let base = || {
        TrainingParams::builder()
            .max_depth(2)
            .eta(0.3)
            .nthread(1)
            .seed(7)
    };

    let mut models: Vec<(&str, BoostedModel)> = Vec::new();
    models.push((
        "reg",
        train(&base().tree_method(TreeMethod::Hist).build()?, &d(&reg)?, 4)?,
    ));
    models.push((
        "binary",
        train(
            &base()
                .objective("binary:logistic")
                .tree_method(TreeMethod::Exact)
                .scale_pos_weight(2.0)
                .build()?,
            &d(&bin)?,
            3,
        )?,
    ));
    models.push((
        "softprob",
        train(
            &base().objective("multi:softprob").num_class(3).build()?,
            &d(&cls)?,
            2,
        )?,
    ));
    models.push((
        "dart",
        train(
            &base()
                .booster(BoosterKind::Dart)
                .rate_drop(0.5)
                .skip_drop(0.0)
                .build()?,
            &d(&reg)?,
            4,
        )?,
    ));
    let mut types = vec![FeatureType::Numerical; COLS];
    types[0] = FeatureType::Categorical;
    models.push((
        "categorical",
        train(
            &base().objective("binary:logistic").build()?,
            &d(&cat_bin)?.with_feature_types(&types)?,
            3,
        )?,
    ));
    models.push((
        "gblinear",
        train(
            &TrainingParams::builder()
                .booster(BoosterKind::GbLinear)
                .nthread(1)
                .build()?,
            &d(&reg)?,
            5,
        )?,
    ));
    // Early stopping against a validation set with inverted labels stops
    // after the first round, leaving trees past `best_iteration`.
    let inverted: Vec<f32> = reg.iter().map(|v| -v).collect();
    let dvalid = d(&inverted)?;
    let stopped = train_with_eval(
        &base().build()?,
        &d(&reg)?,
        8,
        &[(&dvalid, "valid")],
        Some(2),
    )?
    .model;
    assert!(stopped.best_iteration().is_some());
    models.push(("early_stop", stopped));

    let dx = DMatrix::from_dense(&x, ROWS, COLS)?;
    let dcat = DMatrix::from_dense(&x, ROWS, COLS)?.with_feature_types(&types)?;
    let mut preds = String::from("{\n");
    for (idx, (name, model)) in models.iter().enumerate() {
        model.save_binary(out.join(format!("{name}.bin")))?;
        model.save_json(out.join(format!("{name}.json")))?;
        let data = if *name == "categorical" { &dcat } else { &dx };
        let bits = |v: Vec<f32>| {
            v.iter()
                .map(|p| p.to_bits().to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        write!(
            preds,
            "  \"{name}\": {{\"predict\": [{}], \"margin\": [{}]}}{}\n",
            bits(model.predict(data)?),
            bits(model.predict_margin(data)?),
            if idx + 1 == models.len() { "" } else { "," }
        )
        .unwrap();
    }
    preds.push_str("}\n");
    std::fs::write(out.join("predictions.json"), preds)?;
    Ok(())
}
