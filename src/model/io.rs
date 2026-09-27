//! Model persistence: [`ModelFormat`], its detection, and
//! [`BoostedModel`]'s four codec verbs.

use super::serde::UncheckedBoostedModel;
use super::{BoostedModel, lightgbm, native, xgboost};
use crate::error::{HessboostError, Result};
use std::path::Path;

/// A file format [`BoostedModel`] reads with [`BoostedModel::decode`] /
/// [`BoostedModel::load`] and (all but [`ModelFormat::LightgbmText`])
/// writes with [`BoostedModel::encode`] / [`BoostedModel::save`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelFormat {
    /// hessboost's native binary format: a zstd-compressed, checksummed
    /// section container (magic `HBM\0`) holding everything the model
    /// needs. Files written by 0.2.0 and later keep loading in every later
    /// release.
    Binary,
    /// hessboost's native JSON: the model's fields by name, pretty-printed.
    Json,
    /// XGBoost's JSON model schema (`booster.save_model("m.json")`); see
    /// [XGBoost interchange](crate::model#xgboost-interchange).
    XgboostJson,
    /// XGBoost's UBJSON model format (`booster.save_model("m.ubj")`,
    /// `save_raw("ubj")`); see
    /// [UBJSON encoding](crate::model#ubjson-encoding).
    XgboostUbjson,
    /// A LightGBM 4.x text model (`booster.save_model("model.txt")`, or
    /// `model_to_string()`); import only, see
    /// [LightGBM import](crate::model#lightgbm-import).
    LightgbmText,
}

impl ModelFormat {
    /// The format `bytes` are in, judged from their start, or `None` when
    /// they look like none of them. A match is not a promise that the bytes
    /// decode: [`BoostedModel::decode`] still validates them.
    ///
    /// - [`Binary`](Self::Binary): a zstd frame, or an uncompressed native
    ///   container (magic `HBM\0`, or 0.1.x's `SQB\0`, which decoding
    ///   refuses with the upgrade path). Any zstd frame counts, a
    ///   compressed [`DiffusionModel`](crate::diffusion::DiffusionModel)
    ///   too.
    /// - [`XgboostUbjson`](Self::XgboostUbjson): `{` directly followed by a
    ///   UBJSON key-length marker (`i`, `U`, `I`, `l`, `L`), container
    ///   marker (`$`, `#`), or no-op (`N`, which UBJSON allows before a
    ///   key).
    /// - [`XgboostJson`](Self::XgboostJson) / [`Json`](Self::Json): after
    ///   leading whitespace, `{` then `"` or `}` (whitespace between
    ///   allowed); XGBoost's when the document holds a `"learner"` key
    ///   anywhere, native otherwise.
    /// - [`LightgbmText`](Self::LightgbmText): after leading whitespace, a
    ///   first line `tree`.
    pub fn detect(bytes: &[u8]) -> Option<Self> {
        if native::is_native_container(bytes) {
            return Some(Self::Binary);
        }
        let text = bytes.trim_ascii_start();
        if let Some(rest) = text.strip_prefix(b"tree")
            && matches!(rest.first(), Some(b'\n' | b'\r'))
        {
            return Some(Self::LightgbmText);
        }
        let rest = text.strip_prefix(b"{")?;
        if let Some(b'i' | b'U' | b'I' | b'l' | b'L' | b'$' | b'#' | b'N') = rest.first() {
            return Some(Self::XgboostUbjson);
        }
        match rest.trim_ascii_start().first() {
            Some(b'"' | b'}') => Some(if bytes.windows(9).any(|w| w == b"\"learner\"") {
                Self::XgboostJson
            } else {
                Self::Json
            }),
            _ => None,
        }
    }
}

/// `bytes` as the text of a text `format`.
fn utf8(bytes: &[u8], format: ModelFormat) -> Result<&str> {
    std::str::from_utf8(bytes).map_err(|error| {
        HessboostError::model_format(format!("{format:?} model is not UTF-8: {error}"))
    })
}

impl BoostedModel {
    /// The model encoded in `format`: [`ModelFormat::Json`] and
    /// [`ModelFormat::XgboostJson`] as UTF-8 text, the others as binary.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for
    /// [`ModelFormat::LightgbmText`] (an import-only format);
    /// [`HessboostError::ModelFormat`] for a model the format cannot express
    /// (see [XGBoost interchange](crate::model#xgboost-interchange), and a
    /// forest too large for the native format).
    pub fn encode(&self, format: ModelFormat) -> Result<Vec<u8>> {
        match format {
            ModelFormat::Binary => native::write(self),
            ModelFormat::Json => Ok(serde_json::to_vec_pretty(self)?),
            ModelFormat::XgboostJson => xgboost::export_xgboost_json(self).map(String::into_bytes),
            ModelFormat::XgboostUbjson => xgboost::export_xgboost_ubjson(self),
            ModelFormat::LightgbmText => Err(HessboostError::invalid_param(
                "format",
                "LightGBM text is an import-only format: LightGBM models load, but do not save",
            )),
        }
    }

    /// Decode a model from `bytes` in `format` (written by this or an
    /// earlier version, or by XGBoost / LightGBM for their formats).
    /// [`ModelFormat::detect`] guesses the format of unknown bytes.
    ///
    /// # Errors
    ///
    /// [`HessboostError::ModelFormat`] for malformed or inconsistent input,
    /// models the format's import refuses, and files needing a feature this
    /// version lacks; [`HessboostError::Json`] for malformed native JSON.
    pub fn decode(bytes: impl AsRef<[u8]>, format: ModelFormat) -> Result<Self> {
        let bytes = bytes.as_ref();
        match format {
            ModelFormat::Binary => {
                let model = native::read(bytes)?;
                model.validate_structure()?;
                Ok(model)
            }
            ModelFormat::Json => {
                Self::try_from(serde_json::from_slice::<UncheckedBoostedModel>(bytes)?)
            }
            ModelFormat::XgboostJson => xgboost::import_xgboost_json(utf8(bytes, format)?),
            ModelFormat::XgboostUbjson => xgboost::import_xgboost_ubjson(bytes),
            ModelFormat::LightgbmText => lightgbm::import_lightgbm_text(utf8(bytes, format)?),
        }
    }

    /// Write [`Self::encode`]`(format)` to the file at `path`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::encode`], and [`HessboostError::Io`] if the
    /// file cannot be written.
    pub fn save(&self, path: impl AsRef<Path>, format: ModelFormat) -> Result<()> {
        Ok(std::fs::write(path, self.encode(format)?)?)
    }

    /// [`Self::decode`] the file at `path` in `format`.
    ///
    /// # Errors
    ///
    /// [`HessboostError::Io`] if the file cannot be read, then the errors of
    /// [`Self::decode`].
    pub fn load(path: impl AsRef<Path>, format: ModelFormat) -> Result<Self> {
        Self::decode(std::fs::read(path)?, format)
    }
}
