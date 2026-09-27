//! XGBoost's flat parameter form, the one boundary between XGBoost-named
//! key/value settings and [`TrainingParams`]: [`TrainingParams::from_xgboost`]
//! and [`TrainingParams::to_xgboost`]. The Python bindings, the parity tests,
//! and the training fuzz target all go through it.
//!
//! [`schema`] declares the keys, [`parse`] reads them, and [`emit`] writes
//! them.

mod emit;
mod parse;
mod schema;
#[cfg(test)]
mod tests;
