/// Bit patterns of the values, preserving signed zero and NaN payloads.
pub fn bits(values: impl AsRef<[f32]>) -> Vec<u32> {
    values
        .as_ref()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

/// Whether two slices contain the same `f32` bit patterns.
pub fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
