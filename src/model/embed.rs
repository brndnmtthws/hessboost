//! [`EmbeddedModel`]: a model compiled into the binary.

use super::{BoostedModel, ModelFormat};
use crate::error::Result;
use std::fmt;
use std::sync::OnceLock;

/// A model whose file is compiled into the program, decoded on first use.
///
/// Built in a `static` from [`include_bytes!`], it ships the model inside
/// the executable, so a statically linked binary needs no model file at run
/// time. [`get`](Self::get) decodes the bytes once and hands every later
/// caller, on any thread, the same [`BoostedModel`]:
///
/// ```
/// use hessboost::model::EmbeddedModel;
/// use hessboost::prelude::*;
///
/// // Paths are relative to the including source file; `CARGO_MANIFEST_DIR`
/// // makes them relative to the package root instead.
/// static MODEL: EmbeddedModel = EmbeddedModel::new(
///     include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/saved/0.2.0/dart.bin")),
///     ModelFormat::Binary,
/// );
///
/// # fn main() -> Result<()> {
/// let model = MODEL.get()?;
/// let n_features = model.n_features();
/// let x = vec![0.5; 4 * n_features];
/// let preds = model.predict(&DMatrix::from_dense(&x, 4, n_features)?, Iterations::Best)?;
/// assert_eq!(preds.n_rows(), 4);
/// # Ok(())
/// # }
/// ```
///
/// Any [`ModelFormat`] embeds, XGBoost and LightGBM files included, but
/// [`ModelFormat::Binary`] is the smallest and the fastest to decode:
/// converting a model once ([`BoostedModel::load`] then
/// [`save`](BoostedModel::save) with [`ModelFormat::Binary`]) before
/// embedding it is worthwhile.
///
/// The bytes are checked when [`get`](Self::get) decodes them, not by the
/// compiler; a test calling [`get`](Self::get) catches a bad file before
/// the binary ships.
pub struct EmbeddedModel {
    bytes: &'static [u8],
    format: ModelFormat,
    model: OnceLock<BoostedModel>,
}

impl EmbeddedModel {
    /// A model to decode from `bytes` in `format` when first used.
    #[must_use]
    pub const fn new(bytes: &'static [u8], format: ModelFormat) -> Self {
        Self {
            bytes,
            format,
            model: OnceLock::new(),
        }
    }

    /// The embedded bytes.
    #[must_use]
    pub const fn bytes(&self) -> &'static [u8] {
        self.bytes
    }

    /// The format the bytes decode in.
    #[must_use]
    pub const fn format(&self) -> ModelFormat {
        self.format
    }

    /// The decoded model: [`BoostedModel::decode`] of the bytes on the
    /// first successful call, the same model afterwards. Threads racing on
    /// the first call may each decode, but all of them get the one model
    /// stored.
    ///
    /// # Errors
    ///
    /// The errors of [`BoostedModel::decode`]. A failed decode is not
    /// stored: every call on bytes that do not decode returns the error.
    pub fn get(&self) -> Result<&BoostedModel> {
        if let Some(model) = self.model.get() {
            return Ok(model);
        }
        let model = BoostedModel::decode(self.bytes, self.format)?;
        Ok(self.model.get_or_init(|| model))
    }
}

impl fmt::Debug for EmbeddedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbeddedModel")
            .field("bytes", &self.bytes.len())
            .field("format", &self.format)
            .field("decoded", &self.model.get().is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HessboostError;

    const DART: &[u8] = include_bytes!("../../tests/data/saved/0.2.0/dart.bin");

    #[test]
    fn decodes_once_and_matches_decode() {
        static MODEL: EmbeddedModel = EmbeddedModel::new(DART, ModelFormat::Binary);
        let first = MODEL.get().unwrap();
        assert!(std::ptr::eq(first, MODEL.get().unwrap()));
        let decoded = BoostedModel::decode(DART, ModelFormat::Binary).unwrap();
        assert_eq!(
            first.encode(ModelFormat::Binary).unwrap(),
            decoded.encode(ModelFormat::Binary).unwrap()
        );
    }

    #[test]
    fn bad_bytes_error_on_every_call() {
        static MODEL: EmbeddedModel = EmbeddedModel::new(DART, ModelFormat::Json);
        for _ in 0..2 {
            assert!(matches!(MODEL.get(), Err(HessboostError::Json(_))));
        }
    }
}
