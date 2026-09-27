//! XGBoost JSON/UBJSON schema mapping. User docs: [XGBoost interchange](super#xgboost-interchange).

#[cfg(doc)]
use crate::error::HessboostError;
use crate::error::Result;
use crate::model::BoostedModel;
use crate::model::ubjson;
use document::{model_from_value, model_to_value, xgboost_typed_array};

/// Serialize a [`BoostedModel`] into XGBoost's JSON model schema.
///
/// The result is a pretty-printed JSON string equivalent to what
/// `xgboost.Booster.save_model("m.json")` produces for a `gbtree` model
/// (including DART tree weights as `weight_drop`), and is accepted by
/// [`import_xgboost_json`] as well as upstream XGBoost 3.4.2. See the module
/// docs (above) for the `base_score` space convention.
pub fn export_xgboost_json(model: &BoostedModel) -> Result<String> {
    Ok(serde_json::to_string_pretty(&model_to_value(model)?)?)
}

/// Serialize a [`BoostedModel`] into XGBoost's UBJSON model format.
///
/// The bytes hold the same document as [`export_xgboost_json`], encoded the
/// way `xgboost.Booster.save_model("m.ubj")` encodes it (typed tree arrays;
/// see [UBJSON encoding](self#ubjson-encoding)). They are accepted by
/// [`import_xgboost_ubjson`] and upstream XGBoost 3.4.2.
pub fn export_xgboost_ubjson(model: &BoostedModel) -> Result<Vec<u8>> {
    ubjson::encode(&model_to_value(model)?, &xgboost_typed_array)
}

/// Parse an XGBoost JSON model document into a [`BoostedModel`].
///
/// Accepts `gbtree` boosters, including XGBoost 3.4.1's DART layout
/// (`gbtree` plus `model.weight_drop`). Other booster kinds produce a
/// [`HessboostError::ModelFormat`]. See the module docs (above) for details and
/// the `base_score` space convention.
pub fn import_xgboost_json(json: &str) -> Result<BoostedModel> {
    model_from_value(&serde_json::from_str(json)?)
}

/// Parse an XGBoost UBJSON model (`save_model("m.ubj")` or
/// `save_raw("ubj")`) into a [`BoostedModel`].
///
/// Decoding accepts optimized (typed / counted) and plain UBJSON containers;
/// the decoded document then goes through the same mapping, with the same
/// support and errors, as [`import_xgboost_json`]. Malformed bytes produce a
/// [`HessboostError::ModelFormat`].
pub fn import_xgboost_ubjson(bytes: &[u8]) -> Result<BoostedModel> {
    model_from_value(&ubjson::decode(bytes)?)
}

mod document;
mod objective;
mod parse;
#[cfg(test)]
mod tests;
mod tree;
