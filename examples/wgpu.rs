//! GPU prediction and training through wgpu (the `wgpu` feature).
//!
//! Trains a regression model with `device = wgpu` (its histograms built on
//! the GPU, bit-identical to CPU training), then compares CPU and GPU batch
//! prediction: the GPU model (`BoostedModel::to_wgpu`) walks the compact
//! forest one thread per row and predicts bit-identical values.
//!
//! Run with: `cargo run --release --features wgpu --example wgpu`
//!
//! Without a GPU, Mesa's lavapipe (`mesa-vulkan-drivers` on Debian and
//! Ubuntu) provides a software Vulkan adapter: everything runs and matches,
//! only slower than the CPU path.

#[cfg(feature = "wgpu")]
mod common;

fn main() {
    #[cfg(feature = "wgpu")]
    {
        if let Err(error) = wgpu_main() {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "wgpu"))]
    eprintln!(
        "this example needs the `wgpu` feature: \
         cargo run --release --features wgpu --example wgpu"
    );
}

/// A noisy multi-feature regression problem.
#[cfg(feature = "wgpu")]
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

#[cfg(feature = "wgpu")]
fn wgpu_main() -> hessboost::error::Result<()> {
    use hessboost::backend::wgpu;
    use hessboost::config::Device;
    use hessboost::prelude::*;

    if !wgpu::available() {
        eprintln!(
            "no usable wgpu adapter ({}); this example needs one",
            wgpu::unavailable_reason().unwrap_or_default()
        );
        return Ok(());
    }
    println!(
        "wgpu adapter: {}{}",
        wgpu::device_name().as_deref().unwrap_or("?"),
        if wgpu::is_software_adapter() == Some(true) {
            " (software renderer: correct, but slower than the CPU)"
        } else {
            ""
        }
    );

    let train_data = dataset(100_000, 7)?;
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .tree_method(TreeMethod::Hist)
        .max_depth(8)
        .eta(0.2)
        .device(Device::Wgpu)
        .build()?;
    let model = train(&params, &train_data, 50)?;
    let cpu_model = train(
        &TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(8)
            .eta(0.2)
            .build()?,
        &train_data,
        50,
    )?;
    assert_eq!(
        model.encode(ModelFormat::Binary)?,
        cpu_model.encode(ModelFormat::Binary)?,
        "device = wgpu must train the CPU model bit for bit"
    );
    println!(
        "trained {} rounds on {} rows with device = wgpu: the model is the CPU's bit for bit",
        model.num_boost_rounds(),
        train_data.n_rows()
    );

    let gpu = model.to_wgpu()?;
    let batch = dataset(200_000, 11)?;

    // Correctness first: the GPU predictions are bit-identical.
    let cpu = model.predict(&batch, Iterations::Best)?;
    let accelerated = gpu.predict(&batch, Iterations::Best)?;
    assert_eq!(cpu, accelerated, "GPU predictions must be bit-identical");
    println!("GPU predictions match the CPU's bit for bit");

    // Then speed, best of three on each side.
    let (mut cpu_best, mut gpu_best) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let _ = model.predict(&batch, Iterations::Best)?;
        cpu_best = cpu_best.min(t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        let _ = gpu.predict(&batch, Iterations::Best)?;
        gpu_best = gpu_best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "predicted {} rows: CPU {:.1} ms, wgpu {:.1} ms ({:.2}x)",
        batch.n_rows(),
        cpu_best * 1e3,
        gpu_best * 1e3,
        cpu_best / gpu_best,
    );
    Ok(())
}
