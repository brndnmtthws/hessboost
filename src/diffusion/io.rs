//! [`DiffusionFormat`], the file formats of the diffusion family.

use crate::model::container;

/// A file format of [`DiffusionModel`](super::DiffusionModel) and
/// [`ForestModel`](super::forest::ForestModel), read with their `decode` /
/// `load` and written with their `encode` / `save`. These models only have
/// hessboost's own formats, so the XGBoost and LightGBM variants of
/// [`ModelFormat`](crate::model::ModelFormat) are not expressible here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DiffusionFormat {
    /// The binary format: a zstd-compressed, checksummed section container
    /// (magic `HBDM` for a diffusion model, `HBFF` for a forest model)
    /// embedding every GBDT's native container.
    Binary,
    /// JSON, with every GBDT in the native JSON format.
    Json,
}

impl DiffusionFormat {
    /// The format `bytes` are in, judged from their start, or `None` when
    /// they look like neither: [`Binary`](Self::Binary) for a zstd frame or
    /// an uncompressed `HBDM` / `HBFF` container, [`Json`](Self::Json) for
    /// `{` then `"` or `}` (whitespace allowed before and between). A match
    /// is not a promise that the bytes decode (a compressed
    /// [`BoostedModel`](crate::model::BoostedModel) is a zstd frame too).
    pub fn detect(bytes: &[u8]) -> Option<Self> {
        if container::is_zstd_frame(bytes)
            || bytes.starts_with(b"HBDM")
            || bytes.starts_with(b"HBFF")
        {
            return Some(Self::Binary);
        }
        let rest = bytes.trim_ascii_start().strip_prefix(b"{")?;
        matches!(rest.trim_ascii_start().first(), Some(b'"' | b'}')).then_some(Self::Json)
    }
}
