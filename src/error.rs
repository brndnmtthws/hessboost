//! Error types for `hessboost`.

use thiserror::Error;

/// The crate-wide result type.
pub type Result<T> = std::result::Result<T, HessboostError>;

/// Errors that can occur while building datasets, configuring, training, or
/// serializing models.
#[derive(Debug, Error)]
pub enum HessboostError {
    /// A dataset was constructed with inconsistent shapes (e.g. the label
    /// vector length does not match the number of rows).
    #[error("dimension mismatch: {what} (expected {expected}, got {got})")]
    DimensionMismatch {
        /// Human-readable name of the quantity that mismatched.
        what: &'static str,
        /// The value that was expected.
        expected: usize,
        /// The value that was actually provided.
        got: usize,
    },

    /// A configuration parameter was outside its valid range.
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter {
        /// The parameter name (matches the XGBoost parameter where applicable).
        name: &'static str,
        /// Why the value was rejected.
        reason: String,
    },

    /// The dataset was empty where at least one row/column was required.
    #[error("empty dataset: {0}")]
    EmptyDataset(&'static str),

    /// A feature index referenced during prediction or configuration does not
    /// exist in the dataset.
    #[error("feature index {index} out of bounds (num_features = {num_features})")]
    FeatureOutOfBounds {
        /// The offending feature index.
        index: usize,
        /// The number of features available.
        num_features: usize,
    },

    /// The requested objective/metric/booster name is not recognized.
    #[error("unknown {kind} `{name}`")]
    Unknown {
        /// What kind of item was being looked up (objective, metric, ...).
        kind: &'static str,
        /// The name that failed to resolve.
        name: String,
    },

    /// A parsing error while loading data (libsvm/CSV).
    #[error("parse error at line {line}: {reason}")]
    Parse {
        /// 1-based line number where parsing failed.
        line: usize,
        /// Description of the parse failure.
        reason: String,
    },

    /// A model-format (native or XGBoost JSON/UBJSON) (de)serialization error.
    #[error("model format error: {0}")]
    ModelFormat(String),

    /// An underlying I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// A JSON (de)serialization error, from the native JSON model format or
    /// the XGBoost JSON model reader and writer.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl HessboostError {
    /// Convenience constructor for [`HessboostError::InvalidParameter`].
    pub fn invalid_param(name: &'static str, reason: impl Into<String>) -> Self {
        HessboostError::InvalidParameter {
            name,
            reason: reason.into(),
        }
    }

    /// Convenience constructor for [`HessboostError::Unknown`].
    pub fn unknown(kind: &'static str, name: impl Into<String>) -> Self {
        HessboostError::Unknown {
            kind,
            name: name.into(),
        }
    }

    /// Convenience constructor for [`HessboostError::ModelFormat`].
    pub fn model_format(msg: impl Into<String>) -> Self {
        HessboostError::ModelFormat(msg.into())
    }

    /// A model document is missing the named field: ``missing `field` ``.
    pub fn missing_field(field: &str) -> Self {
        Self::model_format(format!("missing `{field}`"))
    }

    /// Convenience constructor for [`HessboostError::DimensionMismatch`].
    pub(crate) fn dimension_mismatch(what: &'static str, expected: usize, got: usize) -> Self {
        HessboostError::DimensionMismatch {
            what,
            expected,
            got,
        }
    }
}
