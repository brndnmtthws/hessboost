#![no_main]
//! The libsvm and CSV text loaders: arbitrary input either fails to parse or
//! yields a non-empty matrix whose labels cover every row and whose stored
//! values are finite. The first byte selects the CSV options; the rest is
//! the file contents.
use hessboost::data::{CsvOptions, read_csv, read_libsvm};
use hessboost::prelude::*;
use libfuzzer_sys::fuzz_target;

/// libsvm column counts follow the largest feature index, not the input
/// size, so a few bytes can legitimately describe a matrix with billions of
/// (empty) columns whose per-column metadata exhausts memory. Skip those.
const MAX_LIBSVM_INDEX: u64 = 1 << 12;
/// Cell-by-cell checks stop at this many cells.
const MAX_CHECKED_CELLS: usize = 1 << 16;

fn libsvm_index_too_large(contents: &[u8]) -> bool {
    // The loader rejects non-UTF-8 input and tokenizes like this.
    std::str::from_utf8(contents).is_ok_and(|text| {
        text.split_whitespace()
            .filter_map(|token| token.split_once(':')?.0.parse::<u64>().ok())
            .any(|idx| idx > MAX_LIBSVM_INDEX)
    })
}

fn check(matrix: &DMatrix) {
    let (n_rows, n_cols) = (matrix.n_rows(), matrix.n_cols());
    assert!(n_rows > 0 && n_cols > 0);
    if let Some(labels) = matrix.labels() {
        assert_eq!(labels.len(), n_rows);
    }
    if n_rows * n_cols <= MAX_CHECKED_CELLS {
        for row in 0..n_rows {
            for col in 0..n_cols {
                if let Some(value) = matrix.get(row, col) {
                    assert!(value.is_finite(), "stored value {value} at ({row}, {col})");
                }
            }
        }
    }
    assert_eq!(matrix.get(n_rows, 0), None);
    assert_eq!(matrix.get(0, n_cols), None);
}

fuzz_target!(|data: &[u8]| {
    let Some((&selector, contents)) = data.split_first() else {
        return;
    };

    if !libsvm_index_too_large(contents)
        && let Ok(matrix) = read_libsvm(contents)
    {
        check(&matrix);
        assert!(matrix.labels().is_some());
    }

    let opts = CsvOptions {
        has_header: selector & 1 != 0,
        delimiter: [',', ';', '\t', ' '][usize::from((selector >> 1) & 3)],
        label_column: match (selector >> 3) & 3 {
            0 => None,
            c => Some(usize::from(c - 1)),
        },
        na_value: (selector & 0x20 != 0).then(|| "NA".to_string()),
    };
    if let Ok(matrix) = read_csv(contents, &opts) {
        check(&matrix);
    }
});
