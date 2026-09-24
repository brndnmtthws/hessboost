//! Universal Binary JSON ([UBJSON](https://ubjson.org/)) codec over
//! [`serde_json::Value`], matching the dialect XGBoost reads and writes for
//! `booster.save_model("m.ubj")` / `save_raw("ubj")`.
//!
//! # Writing
//!
//! [`encode`] mirrors XGBoost's `UBJWriter` byte for byte:
//!
//! * objects are plain `{` .. `}` containers; each key is an `L` (int64)
//!   length followed by its UTF-8 bytes;
//! * strings are `S`, an `L` length, and the bytes;
//! * arrays carry a count and no end marker: `[#L<n>` followed by `n`
//!   values; arrays the caller marks as typed are written in the optimized
//!   form `[$<t>#L<n>` followed by `n` fixed-width payloads with no
//!   per-element marker;
//! * integers take the narrowest of `i` (int8), `I` (int16), `l` (int32) and
//!   `L` (int64) under XGBoost's *strict* range test (`-128 < v < 127` for
//!   int8, and likewise for the wider types), so the extreme values of a width
//!   move to the next one;
//! * floating-point numbers are `d` (float32), XGBoost's only number type; a
//!   value that float32 cannot hold exactly is written as `D` (float64), which
//!   XGBoost also reads, rather than rounded;
//! * every multi-byte payload is big-endian.
//!
//! # Reading
//!
//! [`decode`] accepts everything XGBoost's `UBJReader` does plus the rest of
//! the UBJSON draft-12 container forms, so files from other UBJSON writers load
//! too: plain (`]` / `}`-terminated), counted (`#`), and typed (`$` + `#`)
//! arrays and objects; lengths and counts given as any integer type; `N`
//! no-op padding; and `H` high-precision numbers. As in XGBoost, `C` decodes
//! to its integer code. Typed containers must use a fixed-width numeric
//! element type (`i U I l L d D C`), which also bounds every count by the
//! remaining input before anything is allocated.
//!
//! Decoding never panics on malformed input: truncation, bad markers,
//! negative or oversized counts, invalid UTF-8, non-finite numbers (which a
//! [`Value`] cannot hold), nesting deeper than 128 containers, and trailing
//! bytes are all [`HessboostError::ModelFormat`] errors.

use crate::error::{HessboostError, Result};
use serde_json::{Map, Number, Value};

/// Deepest container nesting [`decode`] accepts (`serde_json`'s own limit).
const MAX_DEPTH: usize = 128;

/// Element type of an optimized (typed) UBJSON array, one per XGBoost
/// `JsonTypedArray` storage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElementType {
    /// `d`: IEEE-754 binary32 (`F32Array`).
    F32,
    /// `D`: IEEE-754 binary64 (`F64Array`).
    F64,
    /// `i`: int8 (`I8Array`).
    I8,
    /// `U`: uint8 (`U8Array`).
    U8,
    /// `I`: int16 (`I16Array`).
    I16,
    /// `l`: int32 (`I32Array`).
    I32,
    /// `L`: int64 (`I64Array`).
    I64,
}

impl ElementType {
    fn marker(self) -> u8 {
        match self {
            Self::F32 => b'd',
            Self::F64 => b'D',
            Self::I8 => b'i',
            Self::U8 => b'U',
            Self::I16 => b'I',
            Self::I32 => b'l',
            Self::I64 => b'L',
        }
    }

    /// Append `number` as a big-endian payload of this type, or `None` when
    /// the value does not fit it exactly.
    fn push(self, number: &Number, out: &mut Vec<u8>) -> Option<()> {
        match self {
            Self::F32 => out.extend_from_slice(&exact_f32(number)?.to_be_bytes()),
            Self::F64 => out.extend_from_slice(&exact_f64(number)?.to_be_bytes()),
            Self::I8 => out.extend_from_slice(&i8::try_from(number.as_i64()?).ok()?.to_be_bytes()),
            Self::U8 => out.push(u8::try_from(number.as_i64()?).ok()?),
            Self::I16 => {
                out.extend_from_slice(&i16::try_from(number.as_i64()?).ok()?.to_be_bytes());
            }
            Self::I32 => {
                out.extend_from_slice(&i32::try_from(number.as_i64()?).ok()?.to_be_bytes());
            }
            Self::I64 => out.extend_from_slice(&number.as_i64()?.to_be_bytes()),
        }
        Some(())
    }
}

/// `number` as an `f32` when that is lossless.
fn exact_f32(number: &Number) -> Option<f32> {
    if let Some(i) = number.as_i64() {
        let f = i as f32;
        return (f as i128 == i128::from(i)).then_some(f);
    }
    if let Some(u) = number.as_u64() {
        let f = u as f32;
        return (f as i128 == i128::from(u)).then_some(f);
    }
    let v = number.as_f64()?;
    let f = v as f32;
    (f64::from(f) == v).then_some(f)
}

/// `number` as an `f64` when that is lossless.
fn exact_f64(number: &Number) -> Option<f64> {
    if let Some(i) = number.as_i64() {
        let f = i as f64;
        return (f as i128 == i128::from(i)).then_some(f);
    }
    if let Some(u) = number.as_u64() {
        let f = u as f64;
        return (f as i128 == i128::from(u)).then_some(f);
    }
    number.as_f64()
}

/// Picks the element type of an array-valued object member from its key and
/// the enclosing object; `None` writes a generic array. See [`encode`].
pub(crate) type TypedArrayFn = dyn Fn(&str, &Map<String, Value>) -> Option<ElementType>;

/// Encode `value` as UBJSON in XGBoost's dialect (see the module docs).
///
/// `typed_array(key, object)` picks the arrays written in optimized typed
/// form: it is asked about every array-valued member `key` of every
/// `object`, and `Some(t)` writes that array as a typed array of `t`. Every
/// element must then be a number `t` holds exactly; anything else is an
/// error rather than a silent conversion.
pub(crate) fn encode(value: &Value, typed_array: &TypedArrayFn) -> Result<Vec<u8>> {
    let mut encoder = Encoder {
        out: Vec::new(),
        typed_array,
    };
    encoder.value(value)?;
    Ok(encoder.out)
}

struct Encoder<'a> {
    out: Vec<u8>,
    typed_array: &'a TypedArrayFn,
}

impl Encoder<'_> {
    fn value(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Null => self.out.push(b'Z'),
            Value::Bool(b) => self.out.push(if *b { b'T' } else { b'F' }),
            Value::Number(n) => self.number(n)?,
            Value::String(s) => {
                self.out.push(b'S');
                self.str(s);
            }
            Value::Array(items) => {
                self.out.extend_from_slice(b"[#");
                self.length(items.len());
                for item in items {
                    self.value(item)?;
                }
            }
            Value::Object(map) => {
                self.out.push(b'{');
                for (key, member) in map {
                    self.str(key);
                    match member {
                        Value::Array(items) => match (self.typed_array)(key, map) {
                            Some(ty) => self.typed(key, ty, items)?,
                            None => self.value(member)?,
                        },
                        _ => self.value(member)?,
                    }
                }
                self.out.push(b'}');
            }
        }
        Ok(())
    }

    fn number(&mut self, n: &Number) -> Result<()> {
        if let Some(i) = n.as_i64() {
            // XGBoost's `UBJWriter::Visit(JsonInteger)` uses strict bounds, so
            // e.g. 127 and -128 are written as int16.
            if i64::from(i8::MIN) < i && i < i64::from(i8::MAX) {
                self.out.push(b'i');
                self.out.extend_from_slice(&(i as i8).to_be_bytes());
            } else if i64::from(i16::MIN) < i && i < i64::from(i16::MAX) {
                self.out.push(b'I');
                self.out.extend_from_slice(&(i as i16).to_be_bytes());
            } else if i64::from(i32::MIN) < i && i < i64::from(i32::MAX) {
                self.out.push(b'l');
                self.out.extend_from_slice(&(i as i32).to_be_bytes());
            } else {
                self.out.push(b'L');
                self.out.extend_from_slice(&i.to_be_bytes());
            }
        } else if n.is_u64() {
            return Err(HessboostError::model_format(format!(
                "UBJSON: integer {n} exceeds the int64 range"
            )));
        } else if let Some(f) = exact_f32(n) {
            self.out.push(b'd');
            self.out.extend_from_slice(&f.to_be_bytes());
        } else {
            let f = n.as_f64().ok_or_else(|| {
                HessboostError::model_format(format!("UBJSON: unrepresentable number {n}"))
            })?;
            self.out.push(b'D');
            self.out.extend_from_slice(&f.to_be_bytes());
        }
        Ok(())
    }

    fn typed(&mut self, key: &str, ty: ElementType, items: &[Value]) -> Result<()> {
        self.out.extend_from_slice(&[b'[', b'$', ty.marker(), b'#']);
        self.length(items.len());
        for (i, item) in items.iter().enumerate() {
            let fits = match item {
                Value::Number(n) => ty.push(n, &mut self.out).is_some(),
                _ => false,
            };
            if !fits {
                return Err(HessboostError::model_format(format!(
                    "UBJSON: `{key}[{i}]` = {item} does not fit a {ty:?} typed array"
                )));
            }
        }
        Ok(())
    }

    /// Container counts and string lengths, always int64 as XGBoost writes them.
    fn length(&mut self, n: usize) {
        self.out.push(b'L');
        // A `Vec`/`String` length never exceeds `isize::MAX`.
        self.out.extend_from_slice(&(n as i64).to_be_bytes());
    }

    fn str(&mut self, s: &str) {
        self.length(s.len());
        self.out.extend_from_slice(s.as_bytes());
    }
}

/// Decode one UBJSON value spanning all of `bytes` (see the module docs for
/// the accepted forms).
pub(crate) fn decode(bytes: &[u8]) -> Result<Value> {
    let mut decoder = Decoder { bytes, pos: 0 };
    let value = decoder.value(0)?;
    decoder.skip_noops();
    if decoder.pos != bytes.len() {
        return Err(decoder.error("trailing bytes after the top-level value"));
    }
    Ok(value)
}

struct Decoder<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn error(&self, msg: impl std::fmt::Display) -> HessboostError {
        HessboostError::model_format(format!("UBJSON at byte {}: {msg}", self.pos))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn byte(&mut self) -> Result<u8> {
        let b = self
            .peek()
            .ok_or_else(|| self.error("unexpected end of input"))?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(self.error(format!("needs {n} bytes, only {} remain", self.remaining())));
        }
        let start = self.pos;
        self.pos += n;
        let bytes: &'a [u8] = self.bytes;
        Ok(&bytes[start..self.pos])
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn skip_noops(&mut self) {
        while self.peek() == Some(b'N') {
            self.pos += 1;
        }
    }

    /// Next value, skipping `N` no-ops before its marker.
    fn value(&mut self, depth: usize) -> Result<Value> {
        self.skip_noops();
        let marker = self.byte()?;
        self.value_of(marker, depth)
    }

    /// Payload of a value whose marker has already been consumed.
    fn value_of(&mut self, marker: u8, depth: usize) -> Result<Value> {
        Ok(match marker {
            b'Z' => Value::Null,
            b'T' => Value::Bool(true),
            b'F' => Value::Bool(false),
            b'i' => Value::from(i8::from_be_bytes(self.array()?)),
            b'U' | b'C' => Value::from(self.byte()?),
            b'I' => Value::from(i16::from_be_bytes(self.array()?)),
            b'l' => Value::from(i32::from_be_bytes(self.array()?)),
            b'L' => Value::from(i64::from_be_bytes(self.array()?)),
            b'd' => {
                let v = f32::from_be_bytes(self.array()?);
                self.float(f64::from(v))?
            }
            b'D' => {
                let v = f64::from_be_bytes(self.array()?);
                self.float(v)?
            }
            b'H' => {
                let digits = self.string()?;
                let number: Number = digits
                    .parse()
                    .map_err(|_| self.error(format!("invalid high-precision number {digits:?}")))?;
                Value::Number(number)
            }
            b'S' => Value::String(self.string()?),
            b'[' => self.container(depth, false)?,
            b'{' => self.container(depth, true)?,
            other => return Err(self.error(format!("unknown marker 0x{other:02x}"))),
        })
    }

    fn float(&self, v: f64) -> Result<Value> {
        Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| self.error(format!("non-finite number {v} has no JSON representation")))
    }

    /// A length or count: an integer-typed value that must be non-negative.
    fn length(&mut self) -> Result<usize> {
        let marker = self.byte()?;
        let n = match marker {
            b'i' => i64::from(i8::from_be_bytes(self.array()?)),
            b'U' => i64::from(self.byte()?),
            b'I' => i64::from(i16::from_be_bytes(self.array()?)),
            b'l' => i64::from(i32::from_be_bytes(self.array()?)),
            b'L' => i64::from_be_bytes(self.array()?),
            other => {
                return Err(self.error(format!(
                    "length must be an integer, found marker 0x{other:02x}"
                )));
            }
        };
        usize::try_from(n).map_err(|_| self.error(format!("negative length {n}")))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.length()?;
        let bytes = self.take(n)?;
        match std::str::from_utf8(bytes) {
            Ok(s) => Ok(s.to_owned()),
            Err(e) => Err(self.error(format!("string is not UTF-8: {e}"))),
        }
    }

    /// An array or object after its opening marker, in any of the plain,
    /// counted (`#`), or typed (`$` + `#`) forms.
    fn container(&mut self, depth: usize, is_object: bool) -> Result<Value> {
        if depth >= MAX_DEPTH {
            return Err(self.error(format!("containers nest deeper than {MAX_DEPTH}")));
        }
        let element = if self.peek() == Some(b'$') {
            self.pos += 1;
            let ty = self.byte()?;
            let Some(width) = fixed_width(ty) else {
                return Err(self.error(format!(
                    "unsupported typed-container element marker 0x{ty:02x}"
                )));
            };
            if self.peek() != Some(b'#') {
                return Err(self.error("typed container without a `#` count"));
            }
            Some((ty, width))
        } else {
            None
        };
        let count = if self.peek() == Some(b'#') {
            self.pos += 1;
            let n = self.length()?;
            // Every entry occupies at least one byte (a marker or a key's
            // length marker), and typed entries a full payload: a larger count
            // cannot be satisfied, so reject it before allocating.
            let min_bytes = element.map_or(1, |(_, width)| width);
            if n.checked_mul(min_bytes)
                .is_none_or(|need| need > self.remaining())
            {
                return Err(self.error(format!(
                    "count {n} exceeds the {} remaining bytes",
                    self.remaining()
                )));
            }
            Some(n)
        } else {
            None
        };
        let depth = depth + 1;
        let element_marker = element.map(|(ty, _)| ty);
        if is_object {
            let mut map = Map::new();
            let mut entry = |this: &mut Self| -> Result<()> {
                // `N` no-ops may precede every key, counted or not.
                this.skip_noops();
                let key = this.string()?;
                let member = match element_marker {
                    Some(ty) => this.value_of(ty, depth)?,
                    None => this.value(depth)?,
                };
                // XGBoost's reader keeps the first occurrence of a key.
                map.entry(key).or_insert(member);
                Ok(())
            };
            match count {
                Some(n) => {
                    for _ in 0..n {
                        entry(self)?;
                    }
                }
                None => loop {
                    self.skip_noops();
                    if self.peek() == Some(b'}') {
                        self.pos += 1;
                        break;
                    }
                    entry(self)?;
                },
            }
            Ok(Value::Object(map))
        } else {
            let mut items = Vec::with_capacity(count.unwrap_or(0));
            match count {
                Some(n) => {
                    for _ in 0..n {
                        items.push(match element_marker {
                            Some(ty) => self.value_of(ty, depth)?,
                            None => self.value(depth)?,
                        });
                    }
                }
                None => loop {
                    self.skip_noops();
                    if self.peek() == Some(b']') {
                        self.pos += 1;
                        break;
                    }
                    items.push(self.value(depth)?);
                },
            }
            Ok(Value::Array(items))
        }
    }
}

/// Payload width of the fixed-width scalar markers allowed as a typed
/// container's element type.
fn fixed_width(marker: u8) -> Option<usize> {
    match marker {
        b'i' | b'U' | b'C' => Some(1),
        b'I' => Some(2),
        b'l' | b'd' => Some(4),
        b'L' | b'D' => Some(8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn plain(value: &Value) -> Vec<u8> {
        encode(value, &|_, _| None).unwrap()
    }

    fn typed(
        key: &'static str,
        ty: ElementType,
    ) -> impl Fn(&str, &Map<String, Value>) -> Option<ElementType> {
        move |k, _| (k == key).then_some(ty)
    }

    /// `L` length prefix as XGBoost writes it.
    fn len(n: i64) -> Vec<u8> {
        let mut out = vec![b'L'];
        out.extend_from_slice(&n.to_be_bytes());
        out
    }

    fn cat(parts: &[&[u8]]) -> Vec<u8> {
        parts.concat()
    }

    #[test]
    fn scalars_encode_to_xgboost_markers() {
        assert_eq!(plain(&Value::Null), b"Z");
        assert_eq!(plain(&json!(true)), b"T");
        assert_eq!(plain(&json!(false)), b"F");
        assert_eq!(plain(&json!(-5)), [b'i', 0xfb]);
        assert_eq!(plain(&json!(300)), [b'I', 0x01, 0x2c]);
        assert_eq!(plain(&json!(70_000)), [b'l', 0x00, 0x01, 0x11, 0x70]);
        assert_eq!(plain(&json!(1_i64 << 40)), [b'L', 0, 0, 1, 0, 0, 0, 0, 0]);
        // float32 is big-endian IEEE-754: 0.5 = 0x3f000000.
        assert_eq!(plain(&json!(0.5)), [b'd', 0x3f, 0, 0, 0]);
        // 0.1 has no exact float32 form, so it is kept as float64.
        let mut d = vec![b'D'];
        d.extend_from_slice(&0.1f64.to_be_bytes());
        assert_eq!(plain(&json!(0.1)), d);
        assert_eq!(plain(&json!("ab")), cat(&[b"S", &len(2), b"ab"]));
    }

    #[test]
    fn integer_widths_use_xgboost_strict_bounds() {
        // The extreme value of each width moves to the next wider type.
        let cases: [(i64, u8); 8] = [
            (126, b'i'),
            (-127, b'i'),
            (127, b'I'),
            (-128, b'I'),
            (32_767, b'l'),
            (-32_768, b'l'),
            (i64::from(i32::MAX), b'L'),
            (i64::from(i32::MIN), b'L'),
        ];
        for (v, marker) in cases {
            let bytes = plain(&json!(v));
            assert_eq!(bytes[0], marker, "{v}");
            assert_eq!(decode(&bytes).unwrap(), json!(v), "{v}");
        }
        assert!(encode(&json!(u64::MAX), &|_, _| None).is_err());
    }

    #[test]
    fn containers_encode_counted_arrays_and_plain_objects() {
        let v = json!({"a": [1, "x"], "b": {}});
        let expected = cat(&[
            b"{",
            &len(1),
            b"a[#",
            &len(2),
            &[b'i', 1],
            b"S",
            &len(1),
            b"x",
            &len(1),
            b"b{}}",
        ]);
        assert_eq!(plain(&v), expected);
    }

    #[test]
    fn every_value_type_round_trips() {
        let v = json!({
            "null": null,
            "bools": [true, false],
            "ints": [0, 1, -1, 126, 127, -128, 255, 32_767, -32_769, 2_147_483_647,
                     -2_147_483_648_i64, i64::MAX, i64::MIN],
            "f32": [0.5, -1.25, f64::from(f32::MAX), f64::from(f32::from_bits(1))],
            "f64": [0.1, 1.0e300, -2.5e-300],
            "strings": ["", "plain", "unicode \u{e9}\u{1f600}"],
            "nested": {"deeper": [[], [{}], {"k": [null]}]},
        });
        assert_eq!(decode(&plain(&v)).unwrap(), v);
    }

    #[test]
    fn typed_arrays_are_optimized_and_round_trip() {
        let cases = [
            (
                ElementType::F32,
                json!([0.5, -2.0, 1.5e9, f64::from(f32::MIN_POSITIVE)]),
                4,
            ),
            (ElementType::F64, json!([0.1, -3.0e200]), 8),
            (ElementType::I8, json!([-128, 0, 127]), 1),
            (ElementType::U8, json!([0, 1, 255]), 1),
            (ElementType::I16, json!([-32_768, 32_767]), 2),
            (
                ElementType::I32,
                json!([-1, 2_147_483_647, i64::from(i32::MIN)]),
                4,
            ),
            (ElementType::I64, json!([i64::MIN, 0, i64::MAX]), 8),
        ];
        for (ty, items, width) in cases {
            let v = json!({ "xs": items });
            let bytes = encode(&v, &typed("xs", ty)).unwrap();
            let n = items.as_array().unwrap().len();
            let header = cat(&[b"{", &len(2), b"xs[$", &[ty.marker()], b"#", &len(n as i64)]);
            assert_eq!(&bytes[..header.len()], header, "{ty:?}");
            assert_eq!(bytes.len(), header.len() + n * width + 1, "{ty:?}");
            assert_eq!(decode(&bytes).unwrap(), v, "{ty:?}");
        }
        // Payloads are big-endian with no per-element markers.
        let bytes = encode(&json!({"xs": [1, -2]}), &typed("xs", ElementType::I16)).unwrap();
        assert!(bytes.ends_with(&[0x00, 0x01, 0xff, 0xfe, b'}']));
    }

    #[test]
    fn typed_arrays_reject_values_their_type_cannot_hold() {
        for (ty, bad) in [
            (ElementType::U8, json!(256)),
            (ElementType::U8, json!(-1)),
            (ElementType::I8, json!(128)),
            (ElementType::I32, json!(1.5)),
            (ElementType::F32, json!(0.1)),
            (ElementType::F32, json!(16_777_217)),
            (ElementType::F64, json!(i64::MAX)),
            (ElementType::I64, json!("1")),
            (ElementType::F32, Value::Null),
        ] {
            let v = json!({ "xs": [bad] });
            assert!(encode(&v, &typed("xs", ty)).is_err(), "{ty:?} {bad}");
        }
    }

    #[test]
    fn typed_array_hint_applies_only_to_the_named_member() {
        let v = json!({"xs": [1], "ys": [1]});
        let bytes = encode(&v, &typed("xs", ElementType::U8)).unwrap();
        let expected = cat(&[
            b"{",
            &len(2),
            b"xs[$U#",
            &len(1),
            &[1],
            &len(2),
            b"ys[#",
            &len(1),
            &[b'i', 1],
            b"}",
        ]);
        assert_eq!(bytes, expected);
    }

    #[test]
    fn decode_accepts_plain_counted_and_typed_containers() {
        let expected = json!({"a": [1, 2], "b": "x"});
        let forms: [&[u8]; 5] = [
            // Plain containers, one-byte lengths.
            b"{U\x01a[U\x01U\x02]U\x01bSU\x01x}",
            // No-op padding around values and container ends.
            b"N{NU\x01aN[NU\x01NU\x02N]U\x01bSU\x01xN}N",
            // Counted object and array.
            b"{#U\x02U\x01a[#U\x02U\x01U\x02U\x01bSU\x01x",
            // Typed array (uint8 payloads).
            b"{U\x01a[$U#U\x02\x01\x02U\x01bSU\x01x}",
            // Typed int8 array inside a counted object.
            b"{#i\x02i\x01a[$i#I\x00\x02\x01\x02i\x01bSi\x01x",
        ];
        for form in forms {
            assert_eq!(decode(form).unwrap(), expected, "{form:?}");
        }
        // Typed object: every member shares the element type.
        assert_eq!(
            decode(b"{$d#U\x01U\x01k\x3f\x00\x00\x00").unwrap(),
            json!({"k": 0.5})
        );
        // No-op padding before the keys of a counted object.
        assert_eq!(decode(b"{#U\x01NU\x01kZ").unwrap(), json!({"k": null}));
        assert_eq!(
            decode(b"{#U\x02NU\x01aZNNU\x01bT").unwrap(),
            json!({"a": null, "b": true})
        );
        assert_eq!(
            decode(b"{$d#U\x01NU\x01k\x3f\x00\x00\x00").unwrap(),
            json!({"k": 0.5})
        );
        // `C` is an integer code, as in XGBoost's reader.
        assert_eq!(decode(b"CA").unwrap(), json!(65));
        assert_eq!(decode(b"HU\x0512345").unwrap(), json!(12345));
    }

    #[test]
    fn decode_reads_big_endian_numbers() {
        assert_eq!(decode(&[b'I', 0x12, 0x34]).unwrap(), json!(0x1234));
        assert_eq!(decode(&[b'l', 0xff, 0xff, 0xff, 0xfe]).unwrap(), json!(-2));
        assert_eq!(
            decode(&[b'L', 0x01, 0, 0, 0, 0, 0, 0, 0x02]).unwrap(),
            json!((1_i64 << 56) + 2)
        );
        assert_eq!(decode(&[b'd', 0xc0, 0x20, 0, 0]).unwrap(), json!(-2.5));
        let mut d = vec![b'D'];
        d.extend_from_slice(&1.0e-300f64.to_be_bytes());
        assert_eq!(decode(&d).unwrap(), json!(1.0e-300));
    }

    #[test]
    fn decode_rejects_malformed_input() {
        let bad: [&[u8]; 20] = [
            b"",
            b"N",
            b"x",
            b"i",
            b"I\x01",
            b"d\x00\x00\x00",
            b"SU\x05ab",
            b"S\x05ab",
            b"Si\xff",
            b"SU\x02\xff\xfe",
            b"[",
            b"[i\x01",
            b"{U\x01a",
            b"{U\x01ai\x01",
            b"[#U\x03i\x01",
            b"[$d#U\x02\x00\x00\x00\x00",
            b"[$S#U\x01U\x00",
            b"[$i]",
            b"[#L\x7f\xff\xff\xff\xff\xff\xff\xff",
            b"Zi\x01",
        ];
        for input in bad {
            assert!(
                matches!(decode(input), Err(HessboostError::ModelFormat(_))),
                "{input:?}"
            );
        }
        // Non-finite floats cannot become JSON numbers.
        for f in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bytes = vec![b'd'];
            bytes.extend_from_slice(&f.to_be_bytes());
            assert!(decode(&bytes).is_err());
        }
        // Nesting is bounded instead of overflowing the stack.
        assert!(decode(&vec![b'['; 100_000]).is_err());
        let mut ok = vec![b'['; MAX_DEPTH];
        ok.extend(std::iter::repeat_n(b']', MAX_DEPTH));
        assert!(decode(&ok).is_ok());
        let mut deep = vec![b'['; MAX_DEPTH + 1];
        deep.extend(std::iter::repeat_n(b']', MAX_DEPTH + 1));
        assert!(decode(&deep).is_err());
    }

    /// Deterministic xorshift64* stream for the fuzz tests.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[test]
    fn decode_never_panics_on_random_bytes() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        // Bias towards marker bytes so inputs reach deep into the parser.
        let alphabet = b"{}[]$#ZNTFiUIlLdDHCS\x00\x01\x02\x7f\x80\xff";
        for _ in 0..5000 {
            let n = rng.below(64);
            let bytes: Vec<u8> = (0..n)
                .map(|_| {
                    if rng.below(2) == 0 {
                        alphabet[rng.below(alphabet.len())]
                    } else {
                        rng.next() as u8
                    }
                })
                .collect();
            // Whatever the outcome, a success must re-encode and decode to the
            // same value.
            if let Ok(v) = decode(&bytes)
                && let Ok(reencoded) = encode(&v, &|_, _| None)
            {
                assert_eq!(decode(&reencoded).unwrap(), v);
            }
        }
    }

    #[test]
    fn decode_never_panics_on_corrupted_encodings() {
        let v = json!({
            "trees": [{"split_conditions": [0.5, -1.0, 2.0], "left_children": [1, -1, -1],
                       "tree_param": {"num_nodes": "3"}}],
            "weight_drop": [1.0, 0.25],
            "version": [3, 4, 2],
        });
        let hint = |k: &str, _: &Map<String, Value>| match k {
            "split_conditions" => Some(ElementType::F32),
            "left_children" => Some(ElementType::I32),
            _ => None,
        };
        let valid = encode(&v, &hint).unwrap();
        assert_eq!(decode(&valid).unwrap(), v);
        let mut rng = Rng(42);
        for _ in 0..5000 {
            let mut bytes = valid.clone();
            match rng.below(3) {
                0 => bytes.truncate(rng.below(valid.len())),
                1 => {
                    for _ in 0..=rng.below(4) {
                        let i = rng.below(bytes.len());
                        bytes[i] = rng.next() as u8;
                    }
                }
                _ => {
                    let i = rng.below(bytes.len());
                    bytes.insert(i, rng.next() as u8);
                }
            }
            let _ = decode(&bytes);
        }
        for cut in 0..valid.len() {
            assert!(
                decode(&valid[..cut]).is_err(),
                "prefix of {cut} bytes decoded"
            );
        }
    }
}
