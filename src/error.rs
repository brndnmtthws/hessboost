//! Error types for `hessboost`.
//!
//! Every fallible API returns [`Result`] with a [`HessboostError`] whose
//! variant says what was refused:
//!
//! - [`InvalidParameter`](HessboostError::InvalidParameter): a setting or
//!   argument, out of range or conflicting with another (`eta`, `nfold`,
//!   `n_samples`, an objective and a metric that do not go together).
//! - [`InvalidData`](HessboostError::InvalidData): an input's content, such
//!   as labels outside the objective's domain, negative weights, groups
//!   that do not partition the rows, or metadata a method does not support;
//!   it names the input and, for an evaluation set, the dataset.
//! - [`IncompatibleModel`](HessboostError::IncompatibleModel): a model
//!   passed in that the request does not fit: continued training or a
//!   refresh with another objective or shape, a slice past its iterations.
//! - [`DimensionMismatch`](HessboostError::DimensionMismatch) for lengths,
//!   [`ModelFormat`](HessboostError::ModelFormat) for model files, and the
//!   rest as documented on each variant.

use std::fmt;

/// The crate-wide result type.
pub type Result<T, E = HessboostError> = std::result::Result<T, E>;

/// Errors that can occur while building datasets, configuring, training, or
/// serializing models.
#[derive(Debug)]
#[non_exhaustive]
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

    /// A configuration parameter was outside its valid range, or conflicts
    /// with another parameter. Problems with the data or with a model
    /// passed in are [`InvalidData`](Self::InvalidData) and
    /// [`IncompatibleModel`](Self::IncompatibleModel).
    InvalidParameter {
        /// The parameter name (matches the XGBoost parameter where applicable).
        name: &'static str,
        /// Why the value was rejected.
        reason: String,
    },

    /// An input's content or shape was refused: label values outside the
    /// objective's domain, negative weights, groups that do not partition
    /// the rows, inconsistent label bounds, non-finite feature values,
    /// metadata a method does not support. Lengths that disagree with the
    /// row count are [`DimensionMismatch`](Self::DimensionMismatch).
    /// Non-exhaustive: match with `InvalidData { input, .. }`.
    #[non_exhaustive]
    InvalidData {
        /// The input: `"labels"`, `"weights"`, `"base_margin"`,
        /// `"group_sizes"`, `"label_bounds"`, `"feature_weights"`, `"data"`
        /// (the feature values or the matrix as a whole), and so on.
        input: &'static str,
        /// The named dataset the input belongs to (an evaluation set's
        /// name), when the refusal names one.
        dataset: Option<String>,
        /// Why the input was rejected.
        reason: String,
    },

    /// A model passed in cannot serve the request: continuing training from
    /// a model the parameters or data do not match, refreshing or pruning
    /// with a model of another shape, slicing past its iterations, or
    /// asking it for something it does not hold. Non-exhaustive: match with
    /// `IncompatibleModel { what, .. }`.
    #[non_exhaustive]
    IncompatibleModel {
        /// What the model conflicts with: the parameter or operation
        /// (`"init_model"`, `"process_type"`, `"slice"`, `"iterations"`, and
        /// so on).
        what: &'static str,
        /// Why the model does not fit.
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

    /// The requested objective/metric/parameter name is not recognized.
    Unknown {
        /// What kind of item was being looked up (objective, metric, ...).
        kind: &'static str,
        /// The name that failed to resolve.
        name: String,
        /// The closest known name, when `name` looks like a typo of it.
        suggestion: Option<&'static str>,
    },

    /// A parsing error while loading data (libsvm/CSV). Non-exhaustive:
    /// the position may gain a column; match with `Parse { line, .. }`.
    #[non_exhaustive]
    Parse {
        /// 1-based line number where parsing failed.
        line: usize,
        /// Description of the parse failure.
        reason: String,
    },

    /// A model-format (native or XGBoost JSON/UBJSON) (de)serialization error.
    ModelFormat(String),

    /// A GPU backend (Metal) failure: no device, a kernel compile or dispatch
    /// error, or a resource limit.
    Gpu(String),

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
            Self::InvalidData {
                input,
                dataset,
                reason,
            } => {
                write!(f, "invalid {input}")?;
                if let Some(dataset) = dataset {
                    write!(f, " in dataset `{dataset}`")?;
                }
                write!(f, ": {reason}")
            }
            Self::IncompatibleModel { what, reason } => {
                write!(f, "incompatible model for `{what}`: {reason}")
            }
            Self::EmptyDataset(what) => write!(f, "empty dataset: {what}"),
            Self::FeatureOutOfBounds {
                index,
                num_features,
            } => write!(
                f,
                "feature index {index} out of bounds (num_features = {num_features})"
            ),
            Self::Unknown {
                kind,
                name,
                suggestion,
            } => {
                write!(f, "unknown {kind} `{name}`")?;
                if let Some(suggestion) = suggestion {
                    write!(f, " (did you mean `{suggestion}`?)")?;
                }
                Ok(())
            }
            Self::Parse { line, reason } => write!(f, "parse error at line {line}: {reason}"),
            Self::ModelFormat(msg) => write!(f, "model format error: {msg}"),
            Self::Gpu(msg) => write!(f, "GPU backend error: {msg}"),
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

    /// Convenience constructor for [`HessboostError::InvalidData`], naming no
    /// dataset (see [`Self::in_dataset`]).
    pub fn invalid_data(input: &'static str, reason: impl Into<String>) -> Self {
        HessboostError::InvalidData {
            input,
            dataset: None,
            reason: reason.into(),
        }
    }

    /// Convenience constructor for [`HessboostError::IncompatibleModel`].
    pub fn incompatible_model(what: &'static str, reason: impl Into<String>) -> Self {
        HessboostError::IncompatibleModel {
            what,
            reason: reason.into(),
        }
    }

    /// This error with an [`InvalidData`](Self::InvalidData) refusal
    /// attributed to the dataset `name` (an evaluation set), unless it
    /// already names one. Other errors are returned unchanged.
    #[must_use]
    pub fn in_dataset(self, name: &str) -> Self {
        match self {
            HessboostError::InvalidData {
                input,
                dataset: None,
                reason,
            } => HessboostError::InvalidData {
                input,
                dataset: Some(name.to_owned()),
                reason,
            },
            other => other,
        }
    }

    /// Convenience constructor for [`HessboostError::Unknown`].
    pub fn unknown(kind: &'static str, name: impl Into<String>) -> Self {
        HessboostError::Unknown {
            kind,
            name: name.into(),
            suggestion: None,
        }
    }

    /// Convenience constructor for [`HessboostError::ModelFormat`].
    pub fn model_format(msg: impl Into<String>) -> Self {
        HessboostError::ModelFormat(msg.into())
    }

    /// Convenience constructor for [`HessboostError::Gpu`].
    pub fn gpu(msg: impl Into<String>) -> Self {
        HessboostError::Gpu(msg.into())
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
