//! The container shared by hessboost's binary model formats (native `HBM`,
//! diffusion `HBDM`, forest `HBFF`): a [section table](super::sections)
//! behind a magic and a version byte, closed by a checksum, stored as one
//! zstd frame (or uncompressed when the frame would expand past the
//! reader's bound).
//!
//! ```text
//! [u8; 4]    magic
//! u8         version
//! sections   nothing after the last payload
//! u64        XXH64 (seed 0) of every byte before it
//! ```
//!
//! Each format declares a [`ContainerSpec`]; embedded native models are
//! stored as uncompressed native containers ([`write_models`],
//! [`read_models`]).

use crate::model::ModelFormat;
use std::borrow::Cow;
use std::io::Read;

use super::BoostedModel;
use super::native::write_container_into;
use super::sections::{Sections, Writer, format_error};
use crate::error::Result;

/// The optional `*.writer` section: the release that wrote the file.
pub(crate) const WRITER: &str = concat!("hessboost ", env!("CARGO_PKG_VERSION"));
/// The zstd frame magic number, little-endian.
pub(super) const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// Whether `bytes` start like a zstd frame (the packed form of every
/// section container).
pub(crate) fn is_zstd_frame(bytes: &[u8]) -> bool {
    bytes.starts_with(&ZSTD_MAGIC)
}
/// Decompressed containers up to this size always load; larger ones must
/// stay within [`MAX_EXPANSION`] of their compressed size. Together they
/// bound what a small malicious file can make the reader allocate.
const ALWAYS_ALLOWED: u64 = 256 << 20;
/// Largest accepted ratio of decompressed to compressed size above
/// [`ALWAYS_ALLOWED`].
pub(super) const MAX_EXPANSION: u64 = 1 << 12;

/// One format built on the container.
pub(crate) struct ContainerSpec {
    pub(crate) magic: [u8; 4],
    pub(crate) version: u8,
    /// The format's name in errors ("native model").
    pub(crate) what: &'static str,
    /// Whether this version reads section `name` (unknown `REQUIRED`
    /// sections are refused, unknown others skipped).
    pub(crate) known: fn(&str) -> bool,
    /// A retired magic and the error that refuses it by name.
    pub(crate) legacy: Option<([u8; 4], &'static str)>,
}

impl ContainerSpec {
    /// The uncompressed container of `w`: the form other containers embed.
    pub(crate) fn frame(&self, w: Writer) -> Vec<u8> {
        let mut out = Vec::new();
        self.frame_into(w, &mut out);
        out
    }

    /// Append the uncompressed container of `w` to `out`.
    pub(crate) fn frame_into(&self, w: Writer, out: &mut Vec<u8>) {
        let start = out.len();
        out.reserve(self.magic.len() + 1 + w.encoded_len() + 8);
        out.extend_from_slice(&self.magic);
        out.push(self.version);
        w.finish(out);
        let checksum = xxh64(&out[start..]);
        out.extend_from_slice(&checksum.to_le_bytes());
    }

    /// The file form of `w`: the container's zstd frame, or the container
    /// itself when [`ContainerSpec::read`] would refuse the frame as
    /// expanding too far (a large, highly repetitive model, such as a
    /// gblinear model of mostly zero weights). Every container of at most
    /// [`ALWAYS_ALLOWED`] bytes is compressed.
    pub(crate) fn seal(&self, w: Writer) -> Result<Vec<u8>> {
        let container = self.frame(w);
        let frame = zstd::bulk::compress(&container, zstd::DEFAULT_COMPRESSION_LEVEL)?;
        Ok(
            if expansion_accepted(frame.len() as u64, container.len() as u64) {
                frame
            } else {
                container
            },
        )
    }

    /// Decode `bytes` (compressed or not): check the header, version and
    /// checksum, parse the section table, refuse trailing bytes, and hand
    /// the sections to `f`.
    pub(crate) fn read<T>(
        &self,
        bytes: &[u8],
        f: impl FnOnce(&Sections) -> Result<T>,
    ) -> Result<T> {
        let container = unpack(bytes)?;
        if let Some((magic, refusal)) = self.legacy
            && container.starts_with(&magic)
        {
            return Err(format_error(refusal));
        }
        let table = section_table(&container, self.magic, self.version, self.what)?;
        let (s, rest) = Sections::parse(table, self.known)?;
        if !rest.is_empty() {
            return Err(format_error(format!(
                "{} unexpected bytes after the last section",
                rest.len()
            )));
        }
        f(&s)
    }
}

/// `models` as uncompressed native containers: their byte lengths and
/// their concatenation (the inverse of [`read_models`]).
pub(crate) fn write_models(models: &[BoostedModel]) -> Result<(Vec<u64>, Vec<u8>)> {
    let mut lengths = Vec::with_capacity(models.len());
    let mut blob = Vec::new();
    for m in models {
        let start = blob.len();
        write_container_into(m, &mut blob)?;
        lengths.push((blob.len() - start) as u64);
    }
    Ok((lengths, blob))
}

/// The models [`write_models`] stored: `blob` split by `lengths` (section
/// `lengths_name`), each piece a native container, with nothing left over
/// (section `blob_name`).
pub(crate) fn read_models(
    lengths: &[u64],
    mut blob: &[u8],
    lengths_name: &str,
    blob_name: &str,
) -> Result<Vec<BoostedModel>> {
    // `lengths` is untrusted: every model takes at least one byte.
    let mut models = Vec::with_capacity(lengths.len().min(blob.len()));
    for &len in lengths {
        let len = usize::try_from(len)
            .ok()
            .filter(|&len| len <= blob.len())
            .ok_or_else(|| format_error(format!("section `{lengths_name}` is out of range")))?;
        let (model, rest) = blob.split_at(len);
        models.push(BoostedModel::decode(model, ModelFormat::Binary)?);
        blob = rest;
    }
    if !blob.is_empty() {
        return Err(format_error(format!(
            "section `{blob_name}` has bytes past its models"
        )));
    }
    Ok(models)
}

/// The container in `bytes`: decompressed from its zstd frame (bounded by
/// [`expansion_limit`]) or borrowed when stored uncompressed.
fn unpack(bytes: &[u8]) -> Result<Cow<'_, [u8]>> {
    Ok(if bytes.starts_with(&ZSTD_MAGIC) {
        Cow::Owned(decompress(bytes)?)
    } else {
        Cow::Borrowed(bytes)
    })
}

/// The section table of `container` (see [`unpack`]) after checking its
/// `magic`, its `version`, and its checksum. `what` names the format in
/// errors ("native model").
fn section_table<'a>(
    container: &'a [u8],
    magic: [u8; 4],
    version: u8,
    what: &str,
) -> Result<&'a [u8]> {
    let Some(body) = container.strip_prefix(&magic) else {
        return Err(format_error(format!("invalid {what} header")));
    };
    let Some(&stored) = body.first() else {
        return Err(format_error(format!("truncated {what}")));
    };
    if stored != version {
        return Err(format_error(format!("unsupported {what} version {stored}")));
    }
    let Some(split) = container
        .len()
        .checked_sub(8)
        .filter(|&at| at > magic.len())
    else {
        return Err(format_error(format!("truncated {what}")));
    };
    let (checked, checksum) = container.split_at(split);
    if xxh64(checked).to_le_bytes() != checksum {
        return Err(format_error(format!("{what} checksum mismatch")));
    }
    Ok(&checked[magic.len() + 1..])
}

/// Largest container [`ContainerSpec::read`] decompresses from a zstd frame of `compressed`
/// bytes.
fn expansion_limit(compressed: u64) -> u64 {
    compressed.saturating_mul(MAX_EXPANSION).max(ALWAYS_ALLOWED)
}

/// Whether [`ContainerSpec::read`] accepts a zstd frame of `compressed` bytes holding a
/// container of `decompressed` bytes.
fn expansion_accepted(compressed: u64, decompressed: u64) -> bool {
    decompressed <= expansion_limit(compressed)
}

pub(super) fn decompress(bytes: &[u8]) -> Result<Vec<u8>> {
    let limit = expansion_limit(bytes.len() as u64);
    let decoder = zstd::stream::read::Decoder::with_buffer(bytes)
        .map_err(|e| format_error(format!("zstd: {e}")))?;
    let mut out = Vec::new();
    decoder
        .take(limit.saturating_add(1))
        .read_to_end(&mut out)
        .map_err(|e| format_error(format!("zstd: {e}")))?;
    if out.len() as u64 > limit {
        return Err(format_error("zstd frame expands too far"));
    }
    Ok(out)
}

/// XXH64 of `data` with seed 0 (Collet's xxHash, 64-bit variant).
pub(super) fn xxh64(data: &[u8]) -> u64 {
    const P1: u64 = 0x9E37_79B1_85EB_CA87;
    const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const P3: u64 = 0x1656_67B1_9E37_79F9;
    const P4: u64 = 0x85EB_CA77_C2B2_AE63;
    const P5: u64 = 0x27D4_EB2F_1656_67C5;
    let round = |acc: u64, lane: u64| {
        acc.wrapping_add(lane.wrapping_mul(P2))
            .rotate_left(31)
            .wrapping_mul(P1)
    };
    let merge = |acc: u64, v: u64| (acc ^ round(0, v)).wrapping_mul(P1).wrapping_add(P4);

    let (stripes, tail) = data.as_chunks::<32>();
    let mut h = if stripes.is_empty() {
        P5
    } else {
        let mut v = [P1.wrapping_add(P2), P2, 0, P1.wrapping_neg()];
        for stripe in stripes {
            for (lane, acc) in stripe.as_chunks::<8>().0.iter().zip(&mut v) {
                *acc = round(*acc, u64::from_le_bytes(*lane));
            }
        }
        let h = v[0]
            .rotate_left(1)
            .wrapping_add(v[1].rotate_left(7))
            .wrapping_add(v[2].rotate_left(12))
            .wrapping_add(v[3].rotate_left(18));
        v.into_iter().fold(h, merge)
    };
    h = h.wrapping_add(data.len() as u64);

    let (words, mut rest) = tail.as_chunks::<8>();
    for word in words {
        h = (h ^ round(0, u64::from_le_bytes(*word)))
            .rotate_left(27)
            .wrapping_mul(P1)
            .wrapping_add(P4);
    }
    if let Some((half, after)) = rest.split_first_chunk::<4>() {
        let half = u32::from_le_bytes(*half);
        h = (h ^ u64::from(half).wrapping_mul(P1))
            .rotate_left(23)
            .wrapping_mul(P2)
            .wrapping_add(P3);
        rest = after;
    }
    for &byte in rest {
        h = (h ^ u64::from(byte).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference digests from the `xxhash` C library, covering the short
    /// path, every tail length class, and the 32-byte stripes.
    #[test]
    fn xxh64_matches_the_reference() {
        let bytes: Vec<u8> = (0..100).collect();
        for (data, expected) in [
            (&b""[..], 0xef46_db37_51d8_e999),
            (b"a", 0xd24e_c4f1_a98c_6e5b),
            (b"abc", 0x44bc_2cf5_ad77_0999),
            (b"0123456789abcdef", 0x5c5b_90c3_4e37_6d0b),
            (&bytes[..31], 0xc346_d2b5_9b4d_8ee1),
            (&bytes[..32], 0xcbf5_9c51_16ff_32b4),
            (&bytes[..], 0x6ac1_e580_3216_6597),
        ] {
            assert_eq!(xxh64(data), expected, "{} bytes", data.len());
        }
    }

    /// The reader's frame policy: containers up to [`ALWAYS_ALLOWED`] bytes
    /// always decompress, larger ones only within [`MAX_EXPANSION`] of the
    /// frame; the writer keeps a container uncompressed exactly when this
    /// refuses its frame.
    #[test]
    fn frame_expansion_policy_boundaries() {
        assert!(expansion_accepted(1, ALWAYS_ALLOWED));
        assert!(!expansion_accepted(1, ALWAYS_ALLOWED + 1));
        let frame = ALWAYS_ALLOWED / MAX_EXPANSION + 1;
        assert!(expansion_accepted(frame, frame * MAX_EXPANSION));
        assert!(!expansion_accepted(frame, frame * MAX_EXPANSION + 1));
        assert!(expansion_accepted(u64::MAX, u64::MAX));
    }
}
