//! Cross-tool model interchange.
//!
//! This module implements import/export of gradient-boosted models in the
//! model schema used by upstream [XGBoost](https://github.com/dmlc/xgboost),
//! in both of its encodings: JSON text (`booster.save_model("m.json")`) and
//! UBJSON binary (`booster.save_model("m.ubj")`, `save_raw("ubj")`). It
//! complements the crate's own native
//! [`BoostedModel::to_json`](crate::learner::BoostedModel::to_json) format and
//! lets `hessboost` load models trained by real XGBoost and emit models that
//! XGBoost-compatible tooling can read.

mod ubjson;
pub mod xgboost_json;

pub use xgboost_json::{
    export_xgboost_json, export_xgboost_ubjson, import_xgboost_json, import_xgboost_ubjson,
};
