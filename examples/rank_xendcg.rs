//! Train XE-NDCG ranking with whole-query subsampling. Run:
//! `cargo run --release --example rank_xendcg`.

use hessboost::prelude::*;

fn main() -> Result<()> {
    let (queries, per) = (100usize, 8usize);
    let n = queries * per;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for q in 0..queries {
        for doc in 0..per {
            let relevance = (doc % 5) as f32;
            x.push(relevance + (((q * 17 + doc * 13) % 11) as f32 - 5.0) * 0.16);
            y.push(relevance);
        }
    }
    let data = DMatrix::from_dense(&x, n, 1)?
        .with_labels(&y)?
        .with_group_sizes(&vec![per; queries])?;
    let params = TrainingParams::builder()
        .objective("rank:xendcg")
        .tree_method(TreeMethod::Hist)
        .bagging_by_query(true)
        .subsample(0.8)
        .max_depth(3)
        .eta(0.1)
        .seed(7)
        .build()?;
    let result = Trainer::new(&params, &data, 100)
        .eval(&data, "train")
        .train()?;
    let scores = &result
        .history
        .last()
        .ok_or_else(|| {
            hessboost::error::HessboostError::invalid_param("num_boost_round", "must be positive")
        })?
        .scores;
    println!("final ranking metrics: {scores:?}");
    Ok(())
}
