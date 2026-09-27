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

/// The array field `key` of `v`: a missing field is a missing-field error
/// naming `key`, a non-array one malformed. With `len`, an array of any
/// other length is malformed too.
fn column<'a>(v: &'a Value, key: &str, len: Option<usize>) -> Result<&'a [Value]> {
    let entries = v
        .get(key)
        .ok_or_else(|| HessboostError::missing_field(key))?
        .as_array()
        .ok_or_else(|| HessboostError::model_format(format!("`{key}` is not an array")))?;
    match len {
        Some(len) if entries.len() != len => Err(HessboostError::model_format(format!(
            "`{key}` has {} entries, not {len}",
            entries.len()
        ))),
        _ => Ok(entries),
    }
}

/// The length of the required array field `key` of `v`.
pub(super) fn column_len(v: &Value, key: &str) -> Result<usize> {
    column(v, key, None).map(<[Value]>::len)
}

/// The required array field `key` of `v` as `f32`s (of `len` entries, if
/// given); a non-numeric entry is malformed.
pub(super) fn float_column(v: &Value, key: &str, len: Option<usize>) -> Result<Vec<f32>> {
    column(v, key, len)?
        .iter()
        .map(|entry| {
            scalar_f64(entry).map(|x| x as f32).ok_or_else(|| {
                HessboostError::model_format(format!("`{key}` contains a non-numeric entry"))
            })
        })
        .collect()
}

/// [`float_column`] for a field XGBoost may omit: `None` when absent.
pub(super) fn optional_float_column(v: &Value, key: &str, len: usize) -> Result<Option<Vec<f32>>> {
    v.get(key)
        .map(|_| float_column(v, key, Some(len)))
        .transpose()
}

/// The required array field `key` of `v`, `len` integers each within
/// `range`; a non-numeric, fractional, or out-of-range entry is malformed.
pub(super) fn integer_column<T: TryFrom<i64>>(
    v: &Value,
    key: &str,
    len: usize,
    range: std::ops::RangeInclusive<i64>,
) -> Result<Vec<T>> {
    column(v, key, Some(len))?
        .iter()
        .map(|entry| {
            let value = scalar_f64(entry).ok_or_else(|| {
                HessboostError::model_format(format!("`{key}` contains a non-numeric entry"))
            })?;
            // `range` is within `i64`, so the cast is exact wherever it holds.
            let integer = value as i64;
            if value.fract() != 0.0 || !range.contains(&integer) || integer as f64 != value {
                return Err(HessboostError::model_format(format!(
                    "`{key}` contains an invalid entry {value}"
                )));
            }
            T::try_from(integer).map_err(|_| {
                HessboostError::model_format(format!("`{key}` contains an invalid entry {value}"))
            })
        })
        .collect()
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
