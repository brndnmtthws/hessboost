//! Native Metal prediction on macOS (the `metal` feature).
//!
//! Trains a regression model on synthetic data, then compares CPU and GPU
//! batch prediction: the GPU model (`BoostedModel::to_gpu`) walks the
//! compact forest one thread per row and predicts bit-identical values,
//! faster as the batch and ensemble grow.
//!
//! Run with: `cargo run --release --features metal --example metal`

#[cfg(all(target_os = "macos", feature = "metal"))]
mod common;

fn main() {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Err(error) = metal_main() {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    eprintln!(
        "this example needs macOS and the `metal` feature: \
         cargo run --release --features metal --example metal"
    );
}

/// A noisy multi-feature regression problem.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dataset(n: usize, seed: u64) -> hessboost::error::Result<hessboost::data::DMatrix> {
    let mut next = common::lcg(seed);
    let mut x = Vec::with_capacity(n * 24);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f32> = (0..24).map(|_| next()).collect();
        let target = row
            .iter()
            .take(6)
            .enumerate()
            .map(|(j, &v)| v * (j as f32 + 1.0))
            .sum::<f32>()
            + next() * 0.1;
        x.extend(row);
        y.push(target);
    }
    hessboost::data::DMatrix::from_dense(&x, n, 24)?.with_labels(&y)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_main() -> hessboost::error::Result<()> {
    use hessboost::backend::metal;
    use hessboost::config::Device;
    use hessboost::prelude::*;

    let Some(()) = metal::available().then_some(()) else {
        eprintln!(
            "no Metal device available ({}); this example needs one",
            metal::unavailable_reason().unwrap_or_default()
        );
        return Ok(());
    };
    println!(
        "Metal device: {}",
        metal::device_name().as_deref().unwrap_or("?")
    );

    let train_data = dataset(100_000, 7)?;
    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .tree_method(TreeMethod::Hist)
        .max_depth(8)
        .eta(0.2)
        .device(Device::Metal)
        .build()?;
    let model = train(&params, &train_data, 200)?;
    println!(
        "trained {} rounds on {} rows (device = metal: bit-identical to CPU training)",
        model.num_boost_rounds(),
        train_data.n_rows()
    );

    let gpu = model.to_gpu()?;
    let batch = dataset(500_000, 11)?;

    // Correctness first: the GPU predictions are bit-identical.
    let cpu = model.predict(&batch)?;
    let metal = gpu.predict(&batch)?;
    assert_eq!(cpu, metal, "GPU predictions must be bit-identical");
    println!("GPU predictions match the CPU's bit for bit");

    // Then speed, best of three on each side.
    let (mut cpu_best, mut gpu_best) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let _ = model.predict(&batch)?;
        cpu_best = cpu_best.min(t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        let _ = gpu.predict(&batch)?;
        gpu_best = gpu_best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "predicted {} rows: CPU {:.1} ms, Metal {:.1} ms ({:.2}x)",
        batch.n_rows(),
        cpu_best * 1e3,
        gpu_best * 1e3,
        cpu_best / gpu_best,
    );
    Ok(())
}
