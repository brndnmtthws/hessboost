//! LightGBM-style class-stratified bagging for imbalanced binary targets.
//! Run: `cargo run --release --example balanced_bagging`.

use hessboost::metric::{Auc, Metric};
use hessboost::prelude::*;

fn main() -> Result<()> {
    let (n, features, train_rows) = (20_000usize, 4usize, 16_000usize);
    let mut state = 7u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((state >> 33) as f32) / (1u32 << 31) as f32
    };
    let mut x = Vec::with_capacity(n * features);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let values = [
            next() * 2.0 - 1.0,
            next() * 2.0 - 1.0,
            next() * 2.0 - 1.0,
            next() * 2.0 - 1.0,
        ];
        x.extend(values);
        let probability = 1.0 / (1.0 + (-(-4.0 + 1.5 * values[0] - 1.2 * values[1])).exp());
        y.push(if next() < probability { 1.0 } else { 0.0 });
    }
    let split = train_rows * features;
    let dtrain =
        DMatrix::from_dense(&x[..split], train_rows, features)?.with_labels(&y[..train_rows])?;
    let dvalid = DMatrix::from_dense(&x[split..], n - train_rows, features)?
        .with_labels(&y[train_rows..])?;
    let params = TrainingParams::builder()
        .objective("binary:logistic")
        .tree_method(TreeMethod::Hist)
        .pos_bagging_fraction(0.7)
        .neg_bagging_fraction(0.2)
        .seed(17)
        .max_depth(4)
        .eta(0.1)
        .build()?;
    let model = train(&params, &dtrain, 100)?;
    let auc = Auc::default().eval(
        &model.predict(&dvalid)?,
        dvalid.labels().unwrap_or_default(),
        None,
    );
    println!(
        "positive fraction {:.4}, valid AUC {auc:.6}",
        y.iter().sum::<f32>() / n as f32
    );
    Ok(())
}
