//! Range checks shared by the parameter constructors: each fails with
//! [`HessboostError::invalid_param`] under the parameter's key, with the
//! boundary in the message ("must be > 0, got 0").

use crate::error::{HessboostError, Result};

/// Fail with [`HessboostError::invalid_param`] unless `ok`.
pub(crate) fn ensure(name: &'static str, ok: bool, reason: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(HessboostError::invalid_param(name, reason))
    }
}

/// [`ensure`] that `v` is finite and in `[0, 1]`.
pub(crate) fn unit(name: &'static str, v: f64) -> Result<()> {
    ensure(
        name,
        v.is_finite() && (0.0..=1.0).contains(&v),
        format!("must be in [0, 1], got {v}"),
    )
}

/// [`ensure`] that `v` is in `(0, 1]`.
pub(crate) fn fraction(name: &'static str, v: f64) -> Result<()> {
    ensure(
        name,
        v > 0.0 && v <= 1.0,
        format!("must be in (0, 1], got {v}"),
    )
}

/// [`ensure`] that `v` is finite and `> 0`.
pub(crate) fn positive(name: &'static str, v: f64) -> Result<()> {
    ensure(
        name,
        v.is_finite() && v > 0.0,
        format!("must be > 0, got {v}"),
    )
}

/// [`ensure`] that `v` is finite and `>= 0`.
pub(crate) fn non_negative(name: &'static str, v: f64) -> Result<()> {
    ensure(
        name,
        v.is_finite() && v >= 0.0,
        format!("must be >= 0, got {v}"),
    )
}

/// [`ensure`] that `v` stays finite (and `> 0` when `positive`) once
/// narrowed to `f32`, as the split search and objectives use it (XGBoost's
/// `float` parameters), so a setting cannot pass validation and then
/// overflow or vanish.
pub(crate) fn narrows(name: &'static str, v: f64, positive: bool) -> Result<()> {
    let narrowed = v as f32;
    ensure(
        name,
        narrowed.is_finite() && (!positive || narrowed > 0.0),
        format!(
            "must stay {}finite in f32, got {v}",
            if positive { "positive and " } else { "" }
        ),
    )
}
