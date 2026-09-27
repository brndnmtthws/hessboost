//! Shared encoding at the boundary: detached model encoding into `bytes`, and
//! the serde JSON through which method configurations cross it.

use crate::errors::{DetachExt, refuse};
use hessboost::error::HessboostError;
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// The bytes `encode` produces without the GIL, as Python `bytes`.
pub(crate) fn encode_bytes(
    py: Python<'_>,
    encode: impl FnOnce() -> Result<Vec<u8>, HessboostError> + Ungil,
) -> PyResult<Bound<'_, PyBytes>> {
    let bytes = py.detached(encode)?;
    Ok(PyBytes::new(py, &bytes))
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
