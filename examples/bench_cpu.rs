//! Deterministic CPU benchmark. Run with `RAYON_NUM_THREADS` fixed.
//! Arguments: operation rows features rounds depth colsample repeats
//! `[dense|sparse] [training_rows]`. Prediction trains on at least 10,000 rows
//! by default, independently of its batch size. Sparse means NaN-filled dense
//! input, not CSR. Operations: train, train-cold, predict, cuts, bins.
use hessboost::internals::{GHistIndex, HistCuts};
use hessboost::model::ModelFormat;
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
    let op = args.get(1).map_or("predict", String::as_str);
    let get = |i: usize, default: usize| args.get(i).map_or(Ok(default), |s| s.parse::<usize>());
    let (n, f, rounds, depth) = (get(2, 100_000)?, get(3, 30)?, get(4, 100)?, get(5, 6)?);
    let colsample: f64 = args.get(6).map_or(Ok(1.0), |s| s.parse())?;
    let repeats = get(7, 7)?;
    let layout = args.get(8).map_or("dense", String::as_str);
    let sparse = layout == "sparse";
    let training_rows = get(9, if op == "predict" { n.max(10_000) } else { n })?;
    if !matches!(op, "train" | "train-cold" | "predict" | "cuts" | "bins")
        || !matches!(layout, "dense" | "sparse")
    {
        return Err(
            "operation must be train, train-cold, predict, cuts or bins; layout must be dense or sparse".into(),
        );
    }
    if n == 0 || f == 0 || rounds == 0 || depth == 0 || repeats == 0 || training_rows == 0 {
        return Err(
            "rows, features, rounds, depth, repeats and training_rows must be positive".into(),
        );
    }
    let generated_rows = n.max(training_rows);
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut rng = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32) / (1u32 << 31) as f32
    };
    let mut x = vec![
        0.0;
        generated_rows
            .checked_mul(f)
            .ok_or("feature shape overflows")?
    ];
    let mut y = vec![0.0; generated_rows];
    for (i, row) in x.chunks_exact_mut(f).enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = rng();
            if j < 5 {
                y[i] += *value * (j + 1) as f32;
            }
            if sparse && (i * 7 + j * 3) % 5 >= 2 {
                *value = f32::NAN;
            }
        }
        y[i] += rng() * 0.1;
    }
    let data = DMatrix::from_dense(&x[..training_rows * f], training_rows, f)?
        .with_labels(&y[..training_rows])?;
    let prediction_data = DMatrix::from_dense(&x[..n * f], n, f)?;
    if matches!(op, "cuts" | "bins") {
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let hash_cuts = |cuts: &HistCuts| {
            checksum((0..cuts.total_bins()).map(|i| u64::from(cuts.cut_value(i).to_bits())))
        };
        let (samples, output_checksum) = if op == "cuts" {
            measure(repeats, || HistCuts::from_dmatrix(&data, 256), hash_cuts)
        } else {
            measure(
                repeats,
                || GHistIndex::from_dmatrix(&data, cuts.clone()),
                |index| {
                    checksum((0..training_rows).flat_map(|r| {
                        (0..f).map(move |c| index.feature_bin(r, c).map_or(u64::MAX, u64::from))
                    }))
                },
            )
        };
        println!(
            "{}",
            serde_json::json!({"op":op,"rows":n,"training_rows":training_rows,"features":f,"rounds":rounds,"depth":depth,"colsample":colsample,"sparse":sparse,"threads":rayon::current_num_threads(),"iterations_per_sample":1,"seconds":samples,"checksum":format!("{output_checksum:016x}"),"model_checksum":format!("{:016x}",hash_cuts(&cuts))})
        );
        return Ok(());
    }
    let params = TrainingParams::builder()
        .max_depth(depth)
        .eta(0.1)
        .colsample_bytree(colsample)
        .build()?;
    // Also populates the shared matrix's bin cache before every timed train.
    let model = train(&params, &data, rounds)?;
    let model_checksum = checksum(model.encode(ModelFormat::Json)?.into_iter().map(u64::from));
    let iterations_per_sample = if op == "predict" && n < 16 {
        1000
    } else if op == "predict" && n < 1024 {
        100
    } else {
        1
    };
    let mut samples = Vec::new();
    let mut reference_checksum = None;
    for run in 0..=repeats {
        let start = Instant::now();
        let predictions = if op == "predict" {
            let mut predictions =
                black_box(model.predict_margin(black_box(&prediction_data), Iterations::Best)?);
            for _ in 1..iterations_per_sample {
                predictions =
                    black_box(model.predict_margin(black_box(&prediction_data), Iterations::Best)?);
            }
            if run > 0 {
                samples.push(start.elapsed().as_secs_f64() / f64::from(iterations_per_sample));
            }
            predictions
        } else {
            let fresh;
            let training_data = if op == "train-cold" {
                fresh = DMatrix::from_dense(&x[..training_rows * f], training_rows, f)?
                    .with_labels(&y[..training_rows])?;
                &fresh
            } else {
                &data
            };
            let trained = black_box(train(&params, black_box(training_data), rounds)?);
            let elapsed = start.elapsed().as_secs_f64();
            if run > 0 {
                samples.push(elapsed);
            }
            assert_eq!(
                checksum(
                    trained
                        .encode(ModelFormat::Json)?
                        .into_iter()
                        .map(u64::from)
                ),
                model_checksum
            );
            trained.predict_margin(&prediction_data, Iterations::Best)?
        };
        let actual = checksum(
            predictions
                .as_slice()
                .iter()
                .map(|value| u64::from(value.to_bits())),
        );
        assert_eq!(
            *reference_checksum.get_or_insert(actual),
            actual,
            "prediction checksum changed on repeat {run}"
        );
    }
    println!(
        "{}",
        serde_json::json!({"op":op,"rows":n,"training_rows":training_rows,"features":f,"rounds":rounds,"depth":depth,"colsample":colsample,"sparse":sparse,"threads":rayon::current_num_threads(),"iterations_per_sample":iterations_per_sample,"training_bin_cache":if op == "train-cold" { "cold" } else { "warm" },"seconds":samples,"checksum":format!("{:016x}",reference_checksum.unwrap()),"model_checksum":format!("{model_checksum:016x}")})
    );
    Ok(())
}

fn measure<T>(
    repeats: usize,
    mut build: impl FnMut() -> T,
    hash: impl Fn(&T) -> u64,
) -> (Vec<f64>, u64) {
    let mut samples = Vec::new();
    let mut expected = None;
    for run in 0..=repeats {
        let start = Instant::now();
        let result = black_box(build());
        let elapsed = start.elapsed().as_secs_f64();
        if run > 0 {
            samples.push(elapsed);
        }
        let actual = hash(&result);
        assert_eq!(*expected.get_or_insert(actual), actual);
    }
    (samples, expected.unwrap())
}
