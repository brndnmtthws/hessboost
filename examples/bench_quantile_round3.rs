//! Deterministic benchmark of sorted weighted cuts and approximate
//! logistic training. Arguments: cuts|train-cold rows features rounds depth
//! repeats layout unit|weighted [approx-logistic|hist-logistic|approx-square].
//! Layouts: dense, sparse, duplicates, constant, ordered, ordered-lowcard,
//! descending, nearly-ordered, two-runs, organ-pipe, cardinality-64,
//! cardinality-256, cardinality-1024.
//! Sparse is NaN-filled dense storage, not CSR. One warmup is discarded;
//! cuts always measures the sorted weighted path.
use hessboost::internals::HistCuts;
use hessboost::model::ModelFormat;
use hessboost::objective::RegLoss;
use hessboost::prelude::*;
use std::{hint::black_box, time::Instant};

fn checksum(values: impl IntoIterator<Item = u64>) -> u64 {
    values
        .into_iter()
        .fold(0xcbf2_9ce4_8422_2325, |hash, value| {
            (hash ^ value).wrapping_mul(0x100_0000_01b3)
        })
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let op = args.get(1).map_or("cuts", String::as_str);
    let get = |i: usize, default: usize| args.get(i).map_or(Ok(default), |s| s.parse::<usize>());
    let (n, f, rounds, depth, repeats) = (
        get(2, 100_000)?,
        get(3, 30)?,
        get(4, 20)?,
        get(5, 6)?,
        get(6, 5)?,
    );
    let layout = args.get(7).map_or("dense", String::as_str);
    let weighting = args.get(8).map_or("weighted", String::as_str);
    let control = args.get(9).map_or("approx-logistic", String::as_str);
    if !matches!(op, "cuts" | "train-cold")
        || !matches!(
            layout,
            "dense"
                | "sparse"
                | "duplicates"
                | "constant"
                | "ordered"
                | "ordered-lowcard"
                | "descending"
                | "nearly-ordered"
                | "two-runs"
                | "organ-pipe"
                | "cardinality-64"
                | "cardinality-256"
                | "cardinality-1024"
        )
        || !matches!(weighting, "unit" | "weighted")
        || !matches!(
            control,
            "approx-logistic" | "hist-logistic" | "approx-square"
        )
        || [n, f, rounds, depth, repeats].contains(&0)
    {
        return Err("expected cuts|train-cold positive_rows features rounds depth repeats layout unit|weighted [approx-logistic|hist-logistic|approx-square]; layout is dense, sparse, duplicates, constant, ordered, ordered-lowcard, descending, nearly-ordered, two-runs, organ-pipe or cardinality-64/-256/-1024".into());
    }
    let weighted = weighting == "weighted";
    let cardinality = match layout {
        "cardinality-64" => Some(64),
        "cardinality-256" => Some(256),
        "cardinality-1024" => Some(1_024),
        _ => None,
    };
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut rng = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32) / (1u32 << 31) as f32
    };
    let mut x = vec![0.0; n.checked_mul(f).ok_or("shape overflow")?];
    let mut y = vec![0.0; n];
    for (r, row) in x.chunks_exact_mut(f).enumerate() {
        let mut signal = 0.0;
        for (c, value) in row.iter_mut().enumerate() {
            let generated = rng();
            if c < 5 {
                signal += (generated - 0.5) * (c + 1) as f32;
            }
            *value = if let Some(cardinality) = cardinality {
                (generated * cardinality as f32).floor() - cardinality as f32 / 2.0
            } else if layout == "descending" {
                (n - r) as f32 / n as f32 + c as f32 / (f + 1) as f32
            } else if layout == "nearly-ordered" {
                // One neighboring pair is reversed per 1,024 rows. This
                // keeps long runs while ensuring the column is not sorted.
                let at = if r % 1_024 < 2 && (r ^ 1) < n {
                    r ^ 1
                } else {
                    r
                };
                at as f32 / n as f32 + c as f32 / (f + 1) as f32
            } else if layout == "two-runs" {
                ((r + n / 2) % n) as f32 / n as f32
            } else if layout == "organ-pipe" {
                let at = if r < n / 2 { 2 * r } else { 2 * (n - r) - 1 };
                at as f32 / n as f32
            } else if layout == "constant" {
                (c % 11) as f32 * 0.125 - 0.5
            } else if layout == "ordered" {
                r as f32 / n as f32 + c as f32 / (f + 1) as f32
            } else if layout == "ordered-lowcard" {
                (r * 16 / n) as f32 - 8.0
            } else if layout == "duplicates" {
                match (r * 37 + c * 17) % 13 {
                    0 => f32::NAN,
                    1 => -0.0,
                    2 => 0.0,
                    _ => (generated * 97.0).floor() - 48.0,
                }
            } else if layout == "sparse" && (r * 7 + c * 3) % 5 >= 2 {
                f32::NAN
            } else {
                generated
            };
        }
        let noise = (rng() - 0.5) * 0.3;
        y[r] = if signal + noise > 0.0 { 1.0 } else { 0.0 };
    }
    let weights: Vec<f32> = (0..n)
        .map(|r| [0.0, 0.125, 0.5, 1.0, 2.0, 3.75, 0.01][r % 7])
        .collect();
    let hessians: Vec<f32> = (0..n)
        .map(|r| 0.01 + ((r * 37) % 1_009) as f32 / 4_040.0)
        .collect();
    let make_data = || -> hessboost::error::Result<DMatrix> {
        let data = DMatrix::from_dense(&x, n, f)?.with_labels(&y)?;
        if weighted {
            data.with_weights(&weights)
        } else {
            Ok(data)
        }
    };
    let data = make_data()?;
    let params = TrainingParams::builder()
        .objective(if control == "approx-square" {
            Objective::SquaredError(RegLoss::default())
        } else {
            Objective::BinaryLogistic(RegLoss::default())
        })
        .tree_method(if control == "hist-logistic" {
            TreeMethod::Hist
        } else {
            TreeMethod::Approx
        })
        .max_depth(depth)
        .eta(0.1)
        .build()?;
    let mut seconds = Vec::with_capacity(repeats);
    let mut reference_checksum = None;
    let mut reference_model = None;
    for run in 0..=repeats {
        let start = Instant::now();
        let (elapsed, hash, model_hash) = if op == "cuts" {
            let cuts = black_box(HistCuts::from_dmatrix_weighted(
                black_box(&data),
                256,
                black_box(&hessians),
                true,
            ));
            let elapsed = start.elapsed().as_secs_f64();
            let hash = checksum(
                std::iter::once(cuts.n_features() as u64)
                    .chain((0..cuts.n_features()).flat_map(|feature| {
                        [
                            u64::from(cuts.is_categorical(feature)),
                            cuts.num_bins(feature) as u64,
                        ]
                    }))
                    .chain((0..cuts.total_bins()).map(|i| u64::from(cuts.cut_value(i).to_bits()))),
            );
            (elapsed, hash, hash)
        } else {
            let fresh = make_data()?;
            let model = black_box(train(&params, black_box(&fresh), rounds)?);
            let elapsed = start.elapsed().as_secs_f64();
            let predictions = model.predict_margin(&data, Iterations::Best)?;
            let hash = checksum(
                predictions
                    .as_slice()
                    .iter()
                    .map(|v| u64::from(v.to_bits())),
            );
            let model_hash = checksum(model.encode(ModelFormat::Json)?.into_iter().map(u64::from));
            (elapsed, hash, model_hash)
        };
        assert_eq!(*reference_checksum.get_or_insert(hash), hash);
        assert_eq!(*reference_model.get_or_insert(model_hash), model_hash);
        if run > 0 {
            seconds.push(elapsed);
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "op":op,"rows":n,"features":f,"rounds":rounds,"depth":depth,
            "layout":layout,"storage":"dense","missing_sentinel":"NaN","weighted":weighted,"control":control,"threads":rayon::current_num_threads(),
            "seconds":seconds,"checksum":format!("{:016x}",reference_checksum.unwrap()),
            "model_checksum":format!("{:016x}",reference_model.unwrap())
        })
    );
    Ok(())
}
