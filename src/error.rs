//! Error types for `hessboost`.

use std::fmt;

/// The crate-wide result type.
pub type Result<T, E = HessboostError> = std::result::Result<T, E>;

/// Errors that can occur while building datasets, configuring, training, or
/// serializing models.
#[derive(Debug)]
pub enum HessboostError {
    /// A dataset was constructed with inconsistent shapes (e.g. the label
    /// vector length does not match the number of rows).
    DimensionMismatch {
        /// Human-readable name of the quantity that mismatched.
        what: &'static str,
        /// The value that was expected.
        expected: usize,
        /// The value that was actually provided.
        got: usize,
    },

    /// A configuration parameter was outside its valid range.
    InvalidParameter {
        /// The parameter name (matches the XGBoost parameter where applicable).
        name: &'static str,
        /// Why the value was rejected.
        reason: String,
    },

    /// The dataset was empty where at least one row/column was required.
    EmptyDataset(&'static str),

    /// A feature index referenced during prediction or configuration does not
    /// exist in the dataset.
    FeatureOutOfBounds {
        /// The offending feature index.
        index: usize,
        /// The number of features available.
        num_features: usize,
    },

    /// The requested objective/metric/booster name is not recognized.
    Unknown {
        /// What kind of item was being looked up (objective, metric, ...).
        kind: &'static str,
        /// The name that failed to resolve.
        name: String,
    },

    /// A parsing error while loading data (libsvm/CSV).
    Parse {
        /// 1-based line number where parsing failed.
        line: usize,
        /// Description of the parse failure.
        reason: String,
    },

    /// A model-format (native or XGBoost JSON/UBJSON) (de)serialization error.
    ModelFormat(String),

    /// An underlying I/O error.
    Io(std::io::Error),

    /// A JSON (de)serialization error, from the native JSON model format or
    /// the XGBoost JSON model reader and writer.
    Json(serde_json::Error),
}

impl fmt::Display for HessboostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {got})"
            ),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::EmptyDataset(what) => write!(f, "empty dataset: {what}"),
            Self::FeatureOutOfBounds {
                index,
                num_features,
            } => write!(
                f,
                "feature index {index} out of bounds (num_features = {num_features})"
            ),
            Self::Unknown { kind, name } => write!(f, "unknown {kind} `{name}`"),
            Self::Parse { line, reason } => write!(f, "parse error at line {line}: {reason}"),
            Self::ModelFormat(msg) => write!(f, "model format error: {msg}"),
            Self::Io(e) => e.fmt(f),
            Self::Json(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for HessboostError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Transparent wrappers: the wrapped error's own source.
        match self {
            Self::Io(e) => e.source(),
            Self::Json(e) => e.source(),
            _ => None,
        }
    }
}

impl From<std::io::Error> for HessboostError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for HessboostError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
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
