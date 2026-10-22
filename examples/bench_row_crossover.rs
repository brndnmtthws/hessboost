//! Compare medium-batch margins with repeated single-row margin calls.
//! Arguments: features rounds depth samples calls (defaults 30 100 6 5 100).
//! Both layouts are warmed. Batch calls own their result; the row loop reuses
//! its output buffer. Each JSON record covers one ABBA measurement group.
use hessboost::model::ModelFormat;
use hessboost::prelude::*;
use std::{hint::black_box, time::Instant};

fn hash(values: impl IntoIterator<Item = u64>) -> u64 {
    values.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, value| {
        (h ^ value).wrapping_mul(0x100_0000_01b3)
    })
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let get = |i, default| {
        args.get(i)
            .map_or(Ok(default), |s: &String| s.parse::<usize>())
    };
    let (features, rounds, depth, samples, calls) = (
        get(1, 30)?,
        get(2, 100)?,
        get(3, 6)?,
        get(4, 5)?,
        get(5, 100)?,
    );
    if [features, rounds, depth, samples, calls].contains(&0) {
        return Err("all arguments must be positive".into());
    }
    let training_rows = 10_000usize;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as f32 / (1u32 << 31) as f32
    };
    let mut x = vec![
        0.0;
        training_rows
            .checked_mul(features)
            .ok_or("shape overflow")?
    ];
    let mut y = vec![0.0; training_rows];
    for (r, row) in x.chunks_exact_mut(features).enumerate() {
        for (f, value) in row.iter_mut().enumerate() {
            *value = next();
            if f < 5 {
                y[r] += *value * (f + 1) as f32;
            }
        }
        y[r] += next() * 0.1;
    }
    let training = DMatrix::from_dense(&x, training_rows, features)?.with_labels(&y)?;
    let params = TrainingParams::builder()
        .max_depth(depth)
        .eta(0.1)
        .build()?;
    let model = train(&params, &training, rounds)?;
    let model_checksum = hash(model.encode(ModelFormat::Json)?.into_iter().map(u64::from));
    for rows in [16, 32, 64, 128, 256] {
        let values = &x[..rows * features];
        let data = DMatrix::from_dense(values, rows, features)?;
        let expected = model.predict_margin(&data, ..)?;
        let expected_bits: Vec<u32> = expected.as_slice().iter().map(|v| v.to_bits()).collect();
        let mut output = vec![0.0; rows];
        let predict_rows = |output: &mut [f32]| -> Result<()> {
            for (row, out) in values
                .chunks_exact(features)
                .zip(output.as_chunks_mut::<1>().0)
            {
                model.predict_margin_row_into(black_box(row), .., out)?;
            }
            Ok(())
        };
        predict_rows(&mut output)?;
        assert!(
            output
                .iter()
                .zip(&expected_bits)
                .all(|(a, b)| a.to_bits() == *b)
        );
        for sample in 0..samples {
            for engine in ["batch", "row", "row", "batch"] {
                let start = Instant::now();
                if engine == "batch" {
                    for _ in 0..calls {
                        black_box(model.predict_margin(black_box(&data), ..)?);
                    }
                } else {
                    for _ in 0..calls {
                        predict_rows(black_box(&mut output))?;
                    }
                }
                let seconds = start.elapsed().as_secs_f64() / calls as f64;
                assert!(
                    output
                        .iter()
                        .zip(&expected_bits)
                        .all(|(a, b)| a.to_bits() == *b)
                );
                println!(
                    "{}",
                    serde_json::json!({
                        "operation": "row_crossover", "engine": engine, "sample": sample,
                        "rows": rows, "features": features, "rounds": rounds, "depth": depth,
                        "training_rows": training_rows, "calls": calls,
                        "threads": rayon::current_num_threads(), "seconds_per_call": seconds,
                        "model_checksum": format!("{model_checksum:016x}"),
                        "prediction_checksum": format!("{:016x}", hash(expected_bits.iter().map(|&v| u64::from(v)))),
                        "row_output_reused": true,
                    })
                );
            }
        }
    }
    Ok(())
}
