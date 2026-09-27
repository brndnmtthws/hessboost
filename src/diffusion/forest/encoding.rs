//! The forest model's column encoding: ranges and categories of the raw
//! columns, their one-hot expansion, and the min–max scaling of the result.

use super::{Column, ColumnKind, Scale};
use crate::error::{HessboostError, Result};

/// Ranges and categories of the `[row][p]` values.
pub(super) fn describe_columns(
    rows: &[f64],
    p: usize,
    kinds: &[ColumnKind],
) -> Result<Vec<Column>> {
    kinds
        .iter()
        .enumerate()
        .map(|(j, &kind)| {
            let mut observed: Vec<f64> = rows
                .chunks_exact(p)
                .map(|r| r[j])
                .filter(|v| !v.is_nan())
                .collect();
            if observed.is_empty() {
                return Err(HessboostError::invalid_param(
                    "data",
                    format!("column {j} has no observed value"),
                ));
            }
            observed.sort_by(f64::total_cmp);
            let (min, max) = (observed[0], observed[observed.len() - 1]);
            let categories = if kind == ColumnKind::Categorical {
                observed.dedup();
                observed
            } else {
                Vec::new()
            };
            Ok(Column {
                kind,
                min,
                max,
                categories,
            })
        })
        .collect()
}

/// Encode one row (`p` values, NaN missing) into `out` (`c` values).
pub(super) fn encode_row<T: Copy + Into<f64>>(columns: &[Column], row: &[T], out: &mut [f64]) {
    let mut at = 0;
    for (column, &v) in columns.iter().zip(row) {
        let v: f64 = v.into();
        match column.kind {
            ColumnKind::Categorical => {
                for (m, category) in column.categories.iter().skip(1).enumerate() {
                    out[at + m] = if v.is_nan() {
                        f64::NAN
                    } else {
                        f64::from(u8::from(v == *category))
                    };
                }
            }
            ColumnKind::Continuous | ColumnKind::Integer => out[at] = v,
        }
        at += column.width();
    }
}

/// The min–max scaling of each of the `c` columns of the encoded `[row][c]`
/// values, over their observed entries.
pub(super) fn fit_scales(encoded: &[f64], c: usize) -> Vec<Scale> {
    (0..c)
        .map(|j| {
            let (lo, hi) = encoded
                .chunks_exact(c)
                .map(|r| r[j])
                .filter(|v| !v.is_nan())
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                    (lo.min(v), hi.max(v))
                });
            let range = hi - lo;
            Scale {
                min: lo,
                range: if range > 0.0 { range } else { 1.0 },
            }
        })
        .collect()
}
