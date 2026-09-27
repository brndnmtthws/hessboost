//! Scalar and array field parsers for XGBoost documents.

use crate::error::{HessboostError, Result};
use serde_json::Value;

/// Fetch a required object field, erroring with its name if absent.
pub(super) fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key).ok_or_else(|| HessboostError::missing_field(key))
}

/// Read an optional string field, `None` when absent; a present value that
/// is not a string is malformed rather than ignored.
pub(super) fn optional_str<'a>(v: &'a Value, key: &str) -> Result<Option<&'a str>> {
    v.get(key)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| HessboostError::model_format(format!("invalid `{key}` {value}")))
        })
        .transpose()
}

/// Read an optional non-negative integer count (XGBoost writes them as
/// numeric strings), `default` when absent. Fractional, negative, or
/// non-representable values are malformed rather than truncated.
pub(super) fn count_param(v: &Value, key: &str, default: usize) -> Result<usize> {
    let Some(value) = v.get(key) else {
        return Ok(default);
    };
    scalar_count(value)
        .ok_or_else(|| HessboostError::model_format(format!("invalid `{key}` {value}")))
}

/// A scalar JSON value as an exact non-negative integer: an integer number
/// or numeric string, or an integral value `f64` holds exactly (such as
/// `"4.0"`). Parsing every count through `f64` would round large ones.
pub(super) fn scalar_count(value: &Value) -> Option<usize> {
    /// `2^53`: every integer up to it is exact in `f64`.
    const EXACT: f64 = 9_007_199_254_740_992.0;
    let integer = match value {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse::<u64>().ok(),
        _ => None,
    };
    integer
        .or_else(|| {
            scalar_f64(value)
                .filter(|&n| (0.0..=EXACT).contains(&n) && n.fract() == 0.0)
                .map(|n| n as u64)
        })
        .and_then(|n| usize::try_from(n).ok())
}

/// Coerce a scalar JSON value (number, numeric string, or bool) to `f64`.
/// XGBoost writes learner/tree parameters as strings but arrays as numbers.
pub(super) fn scalar_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// A JSON array field read element by element, each entry coerced with
/// [`scalar_f64`] (`0` for non-scalar entries), without copying the array.
#[derive(Clone, Copy)]
pub(super) struct Scalars<'a>(&'a [Value]);

impl<'a> Scalars<'a> {
    /// The array field `key` of `v`, empty when absent or not an array.
    pub(super) fn optional(v: &'a Value, key: &str) -> Self {
        Scalars(
            v.get(key)
                .and_then(Value::as_array)
                .map_or(&[], Vec::as_slice),
        )
    }

    /// The array field `key` of `v`; a missing or non-array field is a
    /// missing-field error naming `key`.
    pub(super) fn required(v: &'a Value, key: &str) -> Result<Self> {
        v.get(key)
            .and_then(Value::as_array)
            .map(|a| Scalars(a))
            .ok_or_else(|| HessboostError::missing_field(key))
    }

    /// Entry `i`, `None` past the end.
    pub(super) fn get(self, i: usize) -> Option<f64> {
        self.0.get(i).map(|e| scalar_f64(e).unwrap_or(0.0))
    }

    /// Entry `i`, `0` past the end.
    pub(super) fn at(self, i: usize) -> f64 {
        self.get(i).unwrap_or(0.0)
    }

    /// Every entry as `f32`.
    pub(super) fn to_f32s(self) -> Vec<f32> {
        self.0
            .iter()
            .map(|e| scalar_f64(e).unwrap_or(0.0) as f32)
            .collect()
    }

    /// Every entry as `i32` (truncated).
    pub(super) fn to_i32s(self) -> Vec<i32> {
        self.0
            .iter()
            .map(|e| scalar_f64(e).unwrap_or(0.0) as i32)
            .collect()
    }
}

/// Read a JSON array whose entries are finite, non-negative integers.
pub(super) fn strict_nonnegative_integer_array(v: &Value, key: &str) -> Result<Vec<u64>> {
    let Some(value) = v.get(key) else {
        return Ok(Vec::new());
    };
    let entries = value
        .as_array()
        .ok_or_else(|| HessboostError::model_format(format!("`{key}` is not an array")))?;
    entries
        .iter()
        .map(|entry| {
            let value = scalar_f64(entry).ok_or_else(|| {
                HessboostError::model_format(format!("`{key}` contains a non-numeric entry"))
            })?;
            if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > u64::MAX as f64
            {
                return Err(HessboostError::model_format(format!(
                    "`{key}` contains an invalid integer {value}"
                )));
            }
            Ok(value as u64)
        })
        .collect()
}
