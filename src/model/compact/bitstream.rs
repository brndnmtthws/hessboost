//! Least-significant-bit-first bit I/O and the binary16 threshold codec.

use super::format_error;
use crate::error::Result;

/// Least-significant-bit-first bit stream writer.
#[derive(Default)]
pub(super) struct BitWriter {
    pub(super) bytes: Vec<u8>,
    pub(super) len: usize,
}

impl BitWriter {
    /// Append the low `width` bits of `value` (the rest must be zero; a
    /// zero `value` may be any width, as padding).
    pub(super) fn write(&mut self, value: u64, width: u32) {
        debug_assert!(
            width >= 64 || value >> width == 0,
            "value exceeds its field"
        );
        let mut value = value;
        let mut remaining = width;
        // Fill the partial last byte, then append whole bytes.
        let used = (self.len % 8) as u32;
        if used != 0
            && remaining > 0
            && let Some(last) = self.bytes.last_mut()
        {
            let take = (8 - used).min(remaining);
            *last |= ((value & ((1 << take) - 1)) as u8) << used;
            value >>= take;
            remaining -= take;
        }
        while remaining > 0 {
            let take = remaining.min(8);
            self.bytes.push((value & ((1 << take) - 1)) as u8);
            value >>= take;
            remaining -= take;
        }
        self.len += width as usize;
    }

    pub(super) fn write_bool(&mut self, value: bool) {
        self.write(u64::from(value), 1);
    }

    pub(super) fn write_f32(&mut self, value: f32) {
        self.write(u64::from(value.to_bits()), 32);
    }
}

/// Read a `width <= 32` bit field at bit `pos` of a padded stream. Callers
/// guarantee `pos + width` lies inside the unpadded stream.
#[inline]
pub(super) fn read_bits(stream: &[u8], pos: usize, width: u32) -> u32 {
    debug_assert!(width <= 32);
    let byte = pos / 8;
    let word = u64::from_le_bytes(
        stream[byte..byte + 8]
            .try_into()
            .expect("the stream is padded to eight bytes"),
    );
    let mask = (1u64 << width) - 1;
    ((word >> (pos % 8)) & mask) as u32
}

/// Bounds-checked sequential reader used while parsing untrusted bytes.
pub(super) struct BitReader<'a> {
    pub(super) stream: &'a [u8],
    /// Bits in the unpadded stream.
    pub(super) len: usize,
    pub(super) pos: usize,
}

impl BitReader<'_> {
    pub(super) fn remaining(&self) -> usize {
        self.len - self.pos
    }

    pub(super) fn read(&mut self, width: u32) -> Result<u32> {
        if width > 32 {
            return Err(format_error(format!("field width {width} exceeds 32 bits")));
        }
        if self.remaining() < width as usize {
            return Err(format_error("truncated bit stream"));
        }
        let v = read_bits(self.stream, self.pos, width);
        self.pos += width as usize;
        Ok(v)
    }

    pub(super) fn read_usize(&mut self, width: u32) -> Result<usize> {
        Ok(self.read(width)? as usize)
    }

    pub(super) fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read(1)? == 1)
    }

    /// `count` `f32` fields, refusing non-finite ones (`what` names one).
    pub(super) fn read_finite_f32s(&mut self, count: usize, what: &str) -> Result<Vec<f32>> {
        self.ensure_fits(count, 32, what)?;
        let values = (0..count)
            .map(|_| Ok(f32::from_bits(self.read(32)?)))
            .collect::<Result<Vec<_>>>()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format_error(format!("{what}s must be finite")));
        }
        Ok(values)
    }

    /// Fail unless `count` items of at least `min_bits` each still fit, so a
    /// corrupt count cannot trigger a huge allocation.
    pub(super) fn ensure_fits(&self, count: usize, min_bits: usize, what: &str) -> Result<()> {
        match count.checked_mul(min_bits) {
            Some(total) if total <= self.remaining() => Ok(()),
            _ => Err(format_error(format!(
                "{what} count {count} exceeds the data"
            ))),
        }
    }
}

/// The IEEE binary16 encoding of `v` when it represents `v` exactly.
pub(super) fn f16_exact(v: f32) -> Option<u16> {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    let h = if exp == 0 {
        // Zero (f32 subnormals are far below the binary16 range).
        if man != 0 {
            return None;
        }
        sign
    } else {
        let e = exp - 127;
        if (-14..=15).contains(&e) {
            if man & 0x1fff != 0 {
                return None;
            }
            sign | (((e + 15) as u16) << 10) | (man >> 13) as u16
        } else if (-24..-14).contains(&e) {
            // binary16 subnormal: value = m · 2^-24 with m = significand >> (−e − 1).
            let significand = man | 0x80_0000;
            let shift = (-e - 1) as u32;
            if significand & ((1 << shift) - 1) != 0 {
                return None;
            }
            sign | (significand >> shift) as u16
        } else {
            return None;
        }
    };
    (f16_to_f32(h).to_bits() == bits).then_some(h)
}

/// Decode an IEEE binary16 value (exact in `f32`).
pub(super) fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let magnitude = match exp {
        0 => (man as f32 * f32::from_bits(0x3380_0000)).to_bits(), // m · 2^-24
        0x1f => 0x7f80_0000 | (man << 13),
        _ => ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(sign | magnitude)
}
