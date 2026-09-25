//! Text-format dataset loaders: libsvm and CSV.

use crate::data::dmatrix::DMatrix;
use crate::error::{HessboostError, Result};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::str::FromStr;

/// Parse error for 0-based `lineno` (reported 1-based).
fn parse_err(lineno: usize, reason: impl Into<String>) -> HessboostError {
    HessboostError::Parse {
        line: lineno + 1,
        reason: reason.into(),
    }
}

/// Parse a numeric field, mapping failure to a line-anchored
/// [`HessboostError::Parse`]. `what` names the field role in the message.
fn parse_num<T: FromStr>(field: &str, lineno: usize, what: &str) -> Result<T> {
    field
        .parse()
        .map_err(|_| parse_err(lineno, format!("invalid {what} `{field}`")))
}

/// Call `visit(lineno, line)` for every line of `reader` (0-based numbers,
/// the line without its `\n` or `\r\n`, like [`BufRead::lines`]), reading
/// into one reused buffer.
fn for_each_line<R: Read>(
    reader: R,
    mut visit: impl FnMut(usize, &str) -> Result<()>,
) -> Result<()> {
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    for lineno in 0.. {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        // Like `BufRead::lines`: `\r` goes only with a following `\n`.
        let text = match line.strip_suffix('\n') {
            Some(text) => text.strip_suffix('\r').unwrap_or(text),
            None => &line,
        };
        visit(lineno, text)?;
    }
    Ok(())
}

/// Load a libsvm / SVMLight file into a sparse [`DMatrix`].
///
/// Each line is `label idx:value idx:value ...` with **0-based** feature
/// indices, matching XGBoost's reader. The label becomes the matrix labels.
pub fn load_libsvm(path: impl AsRef<Path>) -> Result<DMatrix> {
    read_libsvm(std::fs::File::open(path)?)
}

/// Parse libsvm-formatted text from any reader.
pub fn read_libsvm<R: Read>(reader: R) -> Result<DMatrix> {
    let mut indptr = vec![0usize];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();
    let mut max_index = 0u32;

    for_each_line(reader, |lineno, line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(());
        }
        let mut it = line.split_whitespace();
        let label_tok = it
            .next()
            .ok_or_else(|| parse_err(lineno, "missing label"))?;
        labels.push(parse_num(label_tok, lineno, "label")?);

        for tok in it {
            let (idx_s, val_s) = tok
                .split_once(':')
                .ok_or_else(|| parse_err(lineno, format!("expected idx:value, got `{tok}`")))?;
            let idx: u32 = parse_num(idx_s, lineno, "index")?;
            let val: f32 = parse_num(val_s, lineno, "value")?;
            indices.push(idx);
            values.push(val);
            max_index = max_index.max(idx);
        }
        indptr.push(values.len());
        Ok(())
    })?;

    if labels.is_empty() {
        return Err(HessboostError::EmptyDataset("libsvm: no rows parsed"));
    }
    let n_cols = (max_index as usize) + 1;
    DMatrix::from_csr(indptr, indices, values, n_cols)?.with_labels(&labels)
}

/// Options controlling CSV parsing. Start from [`CsvOptions::default`] and
/// set the fields that differ.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CsvOptions {
    /// Whether the first line is a header row to skip.
    pub has_header: bool,
    /// Field delimiter.
    pub delimiter: char,
    /// Column index holding the label, if any. Removed from the feature matrix.
    pub label_column: Option<usize>,
    /// Text treated as a missing value (in addition to empty fields).
    pub na_value: Option<String>,
}

impl Default for CsvOptions {
    fn default() -> Self {
        CsvOptions {
            has_header: true,
            delimiter: ',',
            label_column: Some(0),
            na_value: None,
        }
    }
}

/// Load a numeric CSV into a dense [`DMatrix`] (NaN sentinel for missing).
pub fn load_csv(path: impl AsRef<Path>, opts: &CsvOptions) -> Result<DMatrix> {
    read_csv(std::fs::File::open(path)?, opts)
}

/// Parse CSV text from any reader.
pub fn read_csv<R: Read>(reader: R, opts: &CsvOptions) -> Result<DMatrix> {
    let mut flat: Vec<f32> = Vec::new();
    let mut n_rows = 0;
    let mut labels: Vec<f32> = Vec::new();
    let mut n_cols: Option<usize> = None;

    for_each_line(reader, |lineno, line| {
        if (opts.has_header && lineno == 0) || line.trim().is_empty() {
            return Ok(());
        }
        let start = flat.len();
        for (c, raw) in line.split(opts.delimiter).enumerate() {
            let field = raw.trim();
            if Some(c) == opts.label_column {
                labels.push(parse_num(field, lineno, "label")?);
                continue;
            }
            let is_na = field.is_empty() || opts.na_value.as_deref() == Some(field);
            flat.push(if is_na {
                f32::NAN
            } else {
                parse_num(field, lineno, "value")?
            });
        }
        let width = flat.len() - start;
        match n_cols {
            None => n_cols = Some(width),
            Some(expected) if expected != width => {
                return Err(parse_err(
                    lineno,
                    format!("expected {expected} columns, got {width}"),
                ));
            }
            _ => {}
        }
        n_rows += 1;
        Ok(())
    })?;

    let n_cols = n_cols.ok_or(HessboostError::EmptyDataset("csv: no data rows"))?;
    let d = DMatrix::from_dense(&flat, n_rows, n_cols)?;
    if labels.is_empty() {
        Ok(d)
    } else {
        d.with_labels(&labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parse_libsvm() {
        let text = "1 0:1.5 2:3.0\n0 1:2.0\n";
        let d = read_libsvm(Cursor::new(text)).unwrap();
        assert_eq!(d.n_rows(), 2);
        assert_eq!(d.n_cols(), 3);
        assert_eq!(d.labels(), Some(&[1.0f32, 0.0][..]));
        assert_eq!(d.get(0, 0), Some(1.5));
        assert_eq!(d.get(0, 1), None);
        assert_eq!(d.get(0, 2), Some(3.0));
        assert_eq!(d.get(1, 1), Some(2.0));
    }

    #[test]
    fn parse_csv_with_header_and_label() {
        let text = "y,f0,f1\n1.0,0.5,0.25\n0.0,,0.75\n";
        let d = read_csv(Cursor::new(text), &CsvOptions::default()).unwrap();
        assert_eq!(d.n_rows(), 2);
        assert_eq!(d.n_cols(), 2);
        assert_eq!(d.labels(), Some(&[1.0f32, 0.0][..]));
        assert_eq!(d.get(0, 0), Some(0.5));
        assert_eq!(d.get(1, 0), None); // empty field -> missing
        assert_eq!(d.get(1, 1), Some(0.75));
    }

    #[test]
    fn line_endings_match_buf_read_lines() {
        let opts = CsvOptions {
            has_header: false,
            label_column: None,
            ..CsvOptions::default()
        };
        // CRLF and LF line ends are stripped; a final line may lack one.
        let d = read_csv(Cursor::new("1,2\r\n3,4\n5,6"), &opts).unwrap();
        assert_eq!((d.n_rows(), d.n_cols()), (3, 2));
        assert_eq!(d.get(2, 1), Some(6.0));
        // A `\r` without a following `\n` is part of the line.
        let cr = CsvOptions {
            delimiter: '\r',
            ..opts
        };
        let d = read_csv(Cursor::new("1\r2\r"), &cr).unwrap();
        assert_eq!((d.n_rows(), d.n_cols()), (1, 3));
        assert_eq!(d.get(0, 2), None);
    }

    #[test]
    fn csv_column_mismatch_errors() {
        let text = "y,f0,f1\n1.0,0.5,0.25\n0.0,0.75\n";
        assert!(read_csv(Cursor::new(text), &CsvOptions::default()).is_err());
    }
}
