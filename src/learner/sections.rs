//! A table of named, typed sections: the building block of hessboost's
//! binary formats (the native model container and the compact model's
//! metadata).
//!
//! Layout (all integers little-endian, no alignment):
//!
//! ```text
//! u32                     section count
//! per section:            u8 name length, name (UTF-8), u8 flags, u64 byte length
//! section payloads        in table order, back to back
//! ```
//!
//! A payload is a UTF-8 string, one `u64` or `f64`, or a little-endian array
//! of `u8`, `u32`, `i32`, `f32` or `f64`; the section's name fixes which.
//! Readers look sections up by name, so writers may add sections freely:
//! a reader skips sections it does not know unless they carry the
//! [`REQUIRED`] flag, and refuses the data then instead of misreading it.

use std::collections::HashMap;

use crate::error::{HessboostError, Result};

/// Section flag: readers that do not know the section must refuse the data.
pub(super) const REQUIRED: u8 = 1;

/// Builds a section table.
#[derive(Default)]
pub(super) struct Writer {
    sections: Vec<(&'static str, u8, Vec<u8>)>,
}

impl Writer {
    /// Add a section with raw `payload` bytes and `flags`.
    pub(super) fn raw(&mut self, name: &'static str, flags: u8, payload: Vec<u8>) {
        debug_assert!(u8::try_from(name.len()).is_ok());
        self.sections.push((name, flags, payload));
    }

    pub(super) fn str(&mut self, name: &'static str, value: &str) {
        self.raw(name, REQUIRED, value.as_bytes().to_vec());
    }

    pub(super) fn u64(&mut self, name: &'static str, value: u64) {
        self.raw(name, REQUIRED, value.to_le_bytes().to_vec());
    }

    pub(super) fn f64(&mut self, name: &'static str, value: f64) {
        self.raw(name, REQUIRED, value.to_le_bytes().to_vec());
    }

    /// An array of 4- or 8-byte values.
    pub(super) fn array<const N: usize, T: Copy>(
        &mut self,
        name: &'static str,
        values: impl IntoIterator<Item = T>,
        to_le: fn(T) -> [u8; N],
    ) {
        let mut payload = Vec::new();
        for v in values {
            payload.extend_from_slice(&to_le(v));
        }
        self.raw(name, REQUIRED, payload);
    }

    /// Append the table and payloads to `out`.
    pub(super) fn finish(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.sections.len() as u32).to_le_bytes());
        for (name, flags, payload) in &self.sections {
            out.push(name.len() as u8);
            out.extend_from_slice(name.as_bytes());
            out.push(*flags);
            out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        }
        for (_, _, payload) in self.sections {
            out.extend_from_slice(&payload);
        }
    }
}

/// A parsed section table.
pub(super) struct Sections<'a>(HashMap<&'a str, &'a [u8]>);

impl<'a> Sections<'a> {
    /// Parse the table at the start of `bytes`, keeping the sections for
    /// which `known` holds. Returns the sections and the bytes after the
    /// last payload.
    pub(super) fn parse(bytes: &'a [u8], known: impl Fn(&str) -> bool) -> Result<(Self, &'a [u8])> {
        let mut r = Reader(bytes);
        let count = u32::from_le_bytes(r.array()?);
        let mut table = Vec::new();
        for _ in 0..count {
            let name_len = usize::from(r.take(1)?[0]);
            let name = std::str::from_utf8(r.take(name_len)?)
                .map_err(|_| format_error("section name is not UTF-8"))?;
            let flags = r.take(1)?[0];
            let len = usize::try_from(u64::from_le_bytes(r.array()?))
                .map_err(|_| format_error("section is too large"))?;
            table.push((name, flags, len));
        }
        let mut sections = HashMap::with_capacity(table.len());
        for (name, flags, len) in table {
            let payload = r.take(len)?;
            if !known(name) {
                if flags & REQUIRED != 0 {
                    return Err(format_error(format!(
                        "the data needs section `{name}`, which this version cannot read"
                    )));
                }
                continue;
            }
            if sections.insert(name, payload).is_some() {
                return Err(format_error(format!("section `{name}` appears twice")));
            }
        }
        Ok((Sections(sections), r.0))
    }

    pub(super) fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    pub(super) fn bytes(&self, name: &str) -> Result<&'a [u8]> {
        self.0
            .get(name)
            .copied()
            .ok_or_else(|| format_error(format!("missing section `{name}`")))
    }

    pub(super) fn str(&self, name: &str) -> Result<&'a str> {
        std::str::from_utf8(self.bytes(name)?)
            .map_err(|_| format_error(format!("section `{name}` is not UTF-8")))
    }

    pub(super) fn u64(&self, name: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.scalar(name)?))
    }

    /// A `u64` section holding a count or index.
    pub(super) fn usize(&self, name: &str) -> Result<usize> {
        usize::try_from(self.u64(name)?)
            .map_err(|_| format_error(format!("section `{name}` is out of range")))
    }

    pub(super) fn f64(&self, name: &str) -> Result<f64> {
        Ok(f64::from_le_bytes(self.scalar(name)?))
    }

    fn scalar(&self, name: &str) -> Result<[u8; 8]> {
        self.bytes(name)?
            .try_into()
            .map_err(|_| format_error(format!("section `{name}` is not one value")))
    }

    /// An array of 4- or 8-byte values.
    pub(super) fn array<const N: usize, T>(
        &self,
        name: &str,
        from_le: fn([u8; N]) -> T,
    ) -> Result<Vec<T>> {
        let (values, rest) = self.bytes(name)?.as_chunks::<N>();
        if !rest.is_empty() {
            return Err(format_error(format!(
                "section `{name}` is not a whole number of values"
            )));
        }
        Ok(values.iter().map(|&v| from_le(v)).collect())
    }
}

/// A cursor over untrusted bytes.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.0.len() {
            return Err(format_error("truncated data"));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }
}

pub(super) fn format_error(msg: impl Into<String>) -> HessboostError {
    HessboostError::model_format(msg)
}
