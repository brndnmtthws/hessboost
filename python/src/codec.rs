//! Shared encoding at the boundary: model formats by the names the Python
//! layer uses (with `"auto"` detection on load), detached model encoding
//! into `bytes` and decoding, and the serde JSON through which method
//! configurations cross it.

use crate::errors::{DetachExt, ModelFormatError, refuse};
use hessboost::diffusion::DiffusionFormat;
use hessboost::error::HessboostError;
use hessboost::model::ModelFormat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// A crate format enum, named from Python.
pub(crate) trait Format: Copy + Send {
    /// What detection recognizes, for the error when it recognizes nothing.
    const DETECTED: &'static str;

    /// The format called `name` (`"auto"` is not one).
    fn named(name: &str) -> PyResult<Self>;

    /// The format `bytes` look like, if any.
    fn detected(bytes: &[u8]) -> Option<Self>;
}

impl Format for ModelFormat {
    const DETECTED: &'static str = "hessboost's binary and JSON formats, XGBoost's JSON and UBJSON, and LightGBM's text format";

    fn named(name: &str) -> PyResult<Self> {
        Ok(match name {
            "binary" => Self::Binary,
            "json" => Self::Json,
            "xgboost-json" => Self::XgboostJson,
            "xgboost-ubjson" => Self::XgboostUbjson,
            // Import only: encoding refuses it with the crate's error.
            "lightgbm" => Self::LightgbmText,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown model format {other:?}; expected \"binary\", \"json\", \
                     \"xgboost-json\", \"xgboost-ubjson\" or (to load) \"lightgbm\""
                )));
            }
        })
    }

    fn detected(bytes: &[u8]) -> Option<Self> {
        Self::detect(bytes)
    }
}

impl Format for DiffusionFormat {
    const DETECTED: &'static str = "the binary and JSON formats of diffusion and forest models";

    fn named(name: &str) -> PyResult<Self> {
        Ok(match name {
            "binary" => Self::Binary,
            "json" => Self::Json,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown model format {other:?}; expected \"binary\" or \"json\""
                )));
            }
        })
    }

    fn detected(bytes: &[u8]) -> Option<Self> {
        Self::detect(bytes)
    }
}

/// `encode`'s output for the format called `format`, produced without the
/// GIL, as Python `bytes`.
pub(crate) fn encode<'py, F: Format>(
    py: Python<'py>,
    format: &str,
    encode: impl FnOnce(F) -> Result<Vec<u8>, HessboostError> + Send,
) -> PyResult<Bound<'py, PyBytes>> {
    let format = F::named(format)?;
    let bytes = py.detached(move || encode(format))?;
    Ok(PyBytes::new(py, &bytes))
}

/// `decode`'s model from `data` in the format called `format` or, for
/// `"auto"`, the one `data` looks like, decoded without the GIL.
pub(crate) fn decode<'a, F: Format, M: Send>(
    py: Python<'_>,
    data: &'a [u8],
    format: &str,
    decode: impl FnOnce(&'a [u8], F) -> Result<M, HessboostError> + Send,
) -> PyResult<M> {
    let format = if format == "auto" {
        F::detected(data).ok_or_else(|| {
            ModelFormatError::new_err(format!(
                "unrecognized model format: the bytes match none of {}; pass an explicit \
                 format to decode them as one",
                F::DETECTED
            ))
        })?
    } else {
        F::named(format)?
    };
    py.detached(move || decode(data, format))
}

/// `json` decoded as a `T`, refused as an invalid `what`.
pub(crate) fn from_json<T: DeserializeOwned>(json: &str, what: &str) -> PyResult<T> {
    serde_json::from_str(json).map_err(|error| refuse(format!("invalid {what} {json}: {error}")))
}

/// `value` as JSON, refused as an indescribable `what`.
pub(crate) fn to_json<T: Serialize + ?Sized>(value: &T, what: &str) -> PyResult<String> {
    serde_json::to_string(value)
        .map_err(|error| refuse(format!("cannot describe the {what}: {error}")))
}
