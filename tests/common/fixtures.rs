use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use std::path::{Path, PathBuf};

/// JSON `null` is the fixture encoding for a missing `f32` value.
pub fn nan_for_null<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let values: Vec<Option<f32>> = Vec::deserialize(deserializer)?;
    Ok(values
        .into_iter()
        .map(|value| value.unwrap_or(f32::NAN))
        .collect())
}

/// The fixtures directory under the crate manifest directory.
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Every `*.json` fixture in `dir`, in path order.
pub fn json_paths(dir: &Path, what: &str, script: &str) -> Vec<PathBuf> {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|error| {
        panic!(
            "{what} fixtures missing at {}: {error}; run {script}",
            dir.display()
        )
    });
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("json"))
        .collect();
    assert!(!paths.is_empty(), "no {what} fixtures in {}", dir.display());
    paths.sort();
    paths
}

/// Deserialize one JSON fixture at `path` with a path-aware parse error.
pub fn load_json<T: DeserializeOwned>(path: &Path) -> T {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()))
}

/// Every `*.json` fixture in `dir`, deserialized in path order.
pub fn load_all<T: DeserializeOwned>(dir: &Path, what: &str, script: &str) -> Vec<T> {
    json_paths(dir, what, script)
        .iter()
        .map(|path| load_json(path))
        .collect()
}
