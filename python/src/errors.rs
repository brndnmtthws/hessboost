//! Error mapping onto the Python exception hierarchy of
//! `hessboost/_exceptions.py`.

use crate::pool;
use hessboost::error::HessboostError as RustError;
use pyo3::prelude::*;

pyo3::import_exception!(hessboost._exceptions, HessboostError);
pyo3::import_exception!(hessboost._exceptions, ModelFormatError);
pyo3::import_exception!(hessboost._exceptions, InvalidDataError);
pyo3::import_exception!(hessboost._exceptions, IncompatibleModelError);

/// Maps a hessboost error to `HessboostError` (a `ValueError`), its
/// subclasses `ModelFormatError` (model (de)serialization),
/// `InvalidDataError` and `IncompatibleModelError`, or `OSError` (with the
/// `errno` subclass Python picks) for I/O.
pub(crate) fn map_err(error: RustError) -> PyErr {
    match error {
        RustError::Io(error) => PyErr::from(error),
        error @ (RustError::ModelFormat(_) | RustError::Json(_)) => {
            ModelFormatError::new_err(error.to_string())
        }
        error @ RustError::InvalidData { .. } => InvalidDataError::new_err(error.to_string()),
        error @ RustError::IncompatibleModel { .. } => {
            IncompatibleModelError::new_err(error.to_string())
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

/// `Python::detach` for fallible hessboost work: runs `f` without the GIL
/// (attached thread state), inside the extension's rayon pool
/// ([`pool::install`]), and raises its error. Work that never touches
/// rayon uses plain `Python::detach`, sparing the hand-off to a pool thread.
pub(crate) trait DetachExt {
    fn detached<T: Send>(
        self,
        f: impl FnOnce() -> Result<T, RustError> + Send + pyo3::marker::Ungil,
    ) -> PyResult<T>;
}

impl DetachExt for Python<'_> {
    fn detached<T: Send>(
        self,
        f: impl FnOnce() -> Result<T, RustError> + Send + pyo3::marker::Ungil,
    ) -> PyResult<T> {
        self.detach(|| pool::install(f))?.or_raise()
    }
}
