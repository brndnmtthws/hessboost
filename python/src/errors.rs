//! Error mapping onto the Python exception hierarchy of
//! `hessboost/_exceptions.py`.

use hessboost::error::HessboostError as RustError;
use pyo3::prelude::*;

pyo3::import_exception!(hessboost._exceptions, HessboostError);
pyo3::import_exception!(hessboost._exceptions, ModelFormatError);

/// Maps a hessboost error to `HessboostError` (a `ValueError`), its
/// `ModelFormatError` subclass for model (de)serialization, or `OSError`
/// (with the `errno` subclass Python picks) for I/O.
pub(crate) fn map_err(error: RustError) -> PyErr {
    match error {
        RustError::Io(error) => PyErr::from(error),
        error @ (RustError::ModelFormat(_) | RustError::Json(_)) => {
            ModelFormatError::new_err(error.to_string())
        }
        error => HessboostError::new_err(error.to_string()),
    }
}

/// A `HessboostError` with `message`, for refusals made at the boundary.
pub(crate) fn refuse(message: impl Into<String>) -> PyErr {
    HessboostError::new_err(message.into())
}

/// `Result` extension converting hessboost errors to Python exceptions.
pub(crate) trait OrRaise<T> {
    fn or_raise(self) -> PyResult<T>;
}

impl<T> OrRaise<T> for Result<T, RustError> {
    fn or_raise(self) -> PyResult<T> {
        self.map_err(map_err)
    }
}
