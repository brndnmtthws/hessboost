//! Coarse phase profiler for the histogram training pipeline.
//!
//! perf/flamegraph are unavailable in some sandboxes (`perf_event_paranoid`),
//! so this reconstructs the boosting loop from public building blocks and times
//! each phase: binning, gradient computation, tree construction, margin update,
//! and prediction. Run: `BENCH_DIR=<dir> cargo run --release --example profile`.

use hessboost::data::ghist::GHistIndex;
use hessboost::data::quantile::HistCuts;
use hessboost::objective::{create_objective, GradPair};
use hessboost::prelude::*;
use hessboost::tree::builder::HistTreeBuilder;
use hessboost::tree::sampler::ColumnSampler;
use std::path::Path;
use std::time::{Duration, Instant};

mod common;
use common::{load_meta, read_f32};

fn main() -> Result<()> {
    let dir = std::env::var("BENCH_DIR").expect("set BENCH_DIR");
    let dir = Path::new(&dir);
    let meta = load_meta(dir).expect("read meta.json");
    let n = meta.n_rows;
    let f = meta.n_cols;
    let rounds = meta.num_round;
    let max_depth = meta.max_depth;
    let max_bin = meta.max_bin;

    let x = read_f32(&dir.join("X.bin")).unwrap();
    let y = read_f32(&dir.join("y.bin")).unwrap();
    let d = DMatrix::from_dense(&x, n, f)?.with_labels(&y)?;
    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .tree_method(TreeMethod::Hist)
        .max_depth(max_depth)
        .eta(0.1)
        .max_bin(max_bin)
        .base_score(0.5)
        .build()?;

    // Phase: binning (done once per train() call).
    let t = Instant::now();
    let cuts = HistCuts::from_dmatrix(&d, max_bin);
    let ghist = GHistIndex::from_dmatrix(&d, cuts);
    let t_bin = t.elapsed();

    let obj = create_objective(&params)?;
    let mut margin = vec![0.5f32; n];
    let mut gpair = vec![GradPair::default(); n];
    let rows: Vec<u32> = (0..n as u32).collect();

    let (mut t_grad, mut t_build, mut t_update) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);

    let total = Instant::now();
    for _ in 0..rounds {
        let t = Instant::now();
        obj.gradient(&margin, &y, None, &mut gpair);
        t_grad += t.elapsed();

        let t = Instant::now();
        let mut sampler = ColumnSampler::all(f);
        let mut tree = HistTreeBuilder::new(&params).build(&ghist, &gpair, &rows, &mut sampler);
        tree.scale_leaves(params.eta as f32);
        t_build += t.elapsed();

        let t = Instant::now();
        for (row, m) in margin.iter_mut().enumerate() {
            *m += tree.predict_row(&d, row);
        }
        t_update += t.elapsed();
    }
    let t_loop = total.elapsed();

    let pct = |d: Duration| 100.0 * d.as_secs_f64() / t_loop.as_secs_f64();
    println!("dataset {n} x {f}, {rounds} rounds, depth {max_depth}\n");
    println!("binning (once)      {:>8.3?}", t_bin);
    println!("--- per-round loop total {:>8.3?} ---", t_loop);
    println!("  gradient          {:>8.3?}  {:5.1}%", t_grad, pct(t_grad));
    println!(
        "  tree build        {:>8.3?}  {:5.1}%",
        t_build,
        pct(t_build)
    );
    println!(
        "  margin update     {:>8.3?}  {:5.1}%",
        t_update,
        pct(t_update)
    );

    Ok(())
}
