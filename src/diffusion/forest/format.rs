//! The forest model's binary and JSON formats.
//!
//! Binary: the shared model container ([`crate::model::container`]) with
//! magic `HBFF` and its own version byte, zstd-compressed:
//!
//! ```text
//! b"HBFF"    magic
//! u8         VERSION
//! sections   see below, nothing after the last payload
//! u64        XXH64 (seed 0) of every byte before it
//! ```
//!
//! `forest.*` hold the method, `n_t` and the model layout; `columns.*` each
//! column's kind (`u8`: 0 continuous, 1 integer, 2 categorical), range and
//! categories (flattened, split by `columns.category_counts`); `scales.*`
//! the encoded columns' min–max scaling; `classes.*` the class labels and
//! proportions (empty without classes); `models.data` the GBDTs as
//! uncompressed native containers in `[class][level][column]` order, split
//! by `models.lengths`. Every section is `REQUIRED` except `forest.writer`.

use serde::Deserialize;

use super::{Column, ColumnKind, ForestMethod, ForestModel, Scale};
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::model::container::{ContainerSpec, WRITER, read_models, write_models};
use crate::model::sections::{REQUIRED, Sections, Writer, format_error, unknown_value as unknown};

/// The forest model container.
const HBFF: ContainerSpec = ContainerSpec {
    magic: *b"HBFF",
    version: 1,
    what: "forest model",
    known: |name| KNOWN.contains(&name),
    legacy: None,
};

const KNOWN: &[&str] = &[
    "forest.writer",
    "forest.method",
    "forest.beta_min",
    "forest.beta_max",
    "forest.n_t",
    "forest.per_output",
    "columns.kind",
    "columns.min",
    "columns.max",
    "columns.category_counts",
    "columns.categories",
    "scales.min",
    "scales.range",
    "classes.values",
    "classes.probs",
    "models.lengths",
    "models.data",
];

pub(super) fn write(model: &ForestModel) -> Result<Vec<u8>> {
    let mut w = Writer::default();
    w.raw("forest.writer", 0, WRITER.as_bytes());
    match model.method {
        ForestMethod::Flow => w.str("forest.method", "flow"),
        ForestMethod::Diffusion { beta_min, beta_max } => {
            w.str("forest.method", "diffusion");
            w.f64("forest.beta_min", beta_min);
            w.f64("forest.beta_max", beta_max);
        }
    }
    w.u64("forest.n_t", model.n_t as u64);
    w.u64("forest.per_output", u64::from(model.per_output));
    let columns = &model.columns;
    w.array(
        "columns.kind",
        columns.iter().map(|c| match c.kind {
            ColumnKind::Continuous => 0u8,
            ColumnKind::Integer => 1,
            ColumnKind::Categorical => 2,
        }),
        |b: u8| [b],
    );
    w.array(
        "columns.min",
        columns.iter().map(|c| c.min),
        f64::to_le_bytes,
    );
    w.array(
        "columns.max",
        columns.iter().map(|c| c.max),
        f64::to_le_bytes,
    );
    w.array(
        "columns.category_counts",
        columns.iter().map(|c| c.categories.len() as u64),
        u64::to_le_bytes,
    );
    w.array(
        "columns.categories",
        columns.iter().flat_map(|c| c.categories.iter().copied()),
        f64::to_le_bytes,
    );
    w.array(
        "scales.min",
        model.scales.iter().map(|s| s.min),
        f64::to_le_bytes,
    );
    w.array(
        "scales.range",
        model.scales.iter().map(|s| s.range),
        f64::to_le_bytes,
    );
    w.array(
        "classes.values",
        model.classes.iter().copied(),
        f64::to_le_bytes,
    );
    w.array(
        "classes.probs",
        model.class_probs.iter().copied(),
        f64::to_le_bytes,
    );
    let (lengths, data) = write_models(&model.models)?;
    w.array("models.lengths", lengths, u64::to_le_bytes);
    w.raw_owned("models.data", REQUIRED, data);
    HBFF.seal(w)
}

pub(super) fn read(bytes: &[u8]) -> Result<ForestModel> {
    HBFF.read(bytes, read_model)
}

fn read_model(s: &Sections) -> Result<ForestModel> {
    let method = match s.str("forest.method")? {
        "flow" => ForestMethod::Flow,
        "diffusion" => ForestMethod::Diffusion {
            beta_min: s.f64("forest.beta_min")?,
            beta_max: s.f64("forest.beta_max")?,
        },
        other => return Err(unknown("forest.method", other)),
    };
    let per_output = match s.u64("forest.per_output")? {
        0 => false,
        1 => true,
        other => return Err(unknown("forest.per_output", &other.to_string())),
    };
    let kinds = s.array("columns.kind", |[b]: [u8; 1]| b)?;
    let p = kinds.len();
    let mins = s.array_exact("columns.min", p, f64::from_le_bytes)?;
    let maxs = s.array_exact("columns.max", p, f64::from_le_bytes)?;
    let counts = s.array_exact("columns.category_counts", p, u64::from_le_bytes)?;
    let mut categories = s
        .array("columns.categories", f64::from_le_bytes)?
        .into_iter();
    let mut columns = Vec::with_capacity(p);
    for (((&kind, &min), &max), &count) in kinds.iter().zip(&mins).zip(&maxs).zip(&counts) {
        let kind = match kind {
            0 => ColumnKind::Continuous,
            1 => ColumnKind::Integer,
            2 => ColumnKind::Categorical,
            other => return Err(unknown("columns.kind", &other.to_string())),
        };
        let count = usize::try_from(count)
            .map_err(|_| format_error("section `columns.category_counts` is out of range"))?;
        let cats: Vec<f64> = categories.by_ref().take(count).collect();
        if cats.len() != count {
            return Err(format_error(
                "section `columns.categories` has the wrong length",
            ));
        }
        columns.push(Column {
            kind,
            min,
            max,
            categories: cats,
        });
    }
    if categories.next().is_some() {
        return Err(format_error(
            "section `columns.categories` has the wrong length",
        ));
    }
    let scale_min = s.array("scales.min", f64::from_le_bytes)?;
    let scale_range = s.array_exact("scales.range", scale_min.len(), f64::from_le_bytes)?;
    let classes = s.array("classes.values", f64::from_le_bytes)?;
    let class_probs = s.array_exact("classes.probs", classes.len(), f64::from_le_bytes)?;
    let models = read_models(
        &s.array("models.lengths", u64::from_le_bytes)?,
        s.bytes("models.data")?,
        "models.lengths",
        "models.data",
    )?;
    let model = ForestModel {
        method,
        n_t: s.usize("forest.n_t")?,
        columns,
        scales: scale_min
            .into_iter()
            .zip(scale_range)
            .map(|(min, range)| Scale { min, range })
            .collect(),
        classes,
        class_probs,
        per_output,
        models,
    };
    model.validate()?;
    Ok(model)
}

/// The serialized fields of a [`ForestModel`] (its JSON format), before
/// validation. Every field is required.
#[derive(Deserialize)]
pub(super) struct UncheckedForestModel {
    method: ForestMethod,
    n_t: usize,
    columns: Vec<Column>,
    scales: Vec<Scale>,
    classes: Vec<f64>,
    class_probs: Vec<f64>,
    per_output: bool,
    models: Vec<BoostedModel>,
}

impl TryFrom<UncheckedForestModel> for ForestModel {
    type Error = HessboostError;

    fn try_from(m: UncheckedForestModel) -> Result<Self> {
        let model = ForestModel {
            method: m.method,
            n_t: m.n_t,
            columns: m.columns,
            scales: m.scales,
            classes: m.classes,
            class_probs: m.class_probs,
            per_output: m.per_output,
            models: m.models,
        };
        model.validate()?;
        Ok(model)
    }
}
