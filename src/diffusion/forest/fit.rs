//! Fitting a [`ForestModel`]: the encoded, scaled table, the classes, and
//! one set of GBDTs per class and noise level.

use rayon::prelude::*;

use super::encoding::{describe_columns, encode_row, fit_scales};
use super::{
    Column, ColumnKind, ForestMethod, ForestModel, ForestParams, NOISE_STREAM, OutputLayout, Scale,
    level_time, refuse_metadata,
};
use crate::data::DMatrix;
use crate::diffusion::process::{Normal, try_filled};
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::rng::{Rng, splitmix64};
use crate::training::train;

/// See [`ForestModel::fit`].
pub(super) fn fit(params: &ForestParams, data: &DMatrix) -> Result<ForestModel> {
    params.validate()?;
    refuse_metadata(data)?;
    let table = prepare_table(params, data)?;
    let classes = prepare_classes(data, &table.keep);
    let models = fit_level_models(params, &table, &classes)?;
    let model = ForestModel {
        method: params.method,
        n_t: params.n_t,
        columns: table.columns,
        scales: table.scales,
        classes: classes.values,
        class_probs: classes.probs,
        layout: table.layout,
        models,
    };
    model.validate()?;
    Ok(model)
}

/// The training table: the kept rows, encoded and scaled.
struct Table {
    /// Indices of the rows of `data` with at least one observed value.
    keep: Vec<usize>,
    columns: Vec<Column>,
    scales: Vec<Scale>,
    /// Scaled encoded rows `x₁`, `[row][c]`, NaN missing.
    encoded: Vec<f64>,
    layout: OutputLayout,
}

impl Table {
    /// The number of encoded columns.
    fn c(&self) -> usize {
        self.scales.len()
    }
}

/// The encoded, scaled table of `data`'s rows with at least one observed
/// value, and the layout its missing values call for.
fn prepare_table(params: &ForestParams, data: &DMatrix) -> Result<Table> {
    let p = data.n_cols();
    let kinds = match &params.column_kinds {
        None => vec![ColumnKind::Continuous; p],
        Some(kinds) if kinds.len() == p => kinds.clone(),
        Some(kinds) => {
            return Err(HessboostError::dimension_mismatch(
                "column_kinds",
                p,
                kinds.len(),
            ));
        }
    };

    // Rows with at least one observed value, densely with NaN.
    let raw = crate::diffusion::fit::dense_features(data);
    let keep: Vec<usize> = (0..data.n_rows())
        .filter(|&r| raw[r * p..(r + 1) * p].iter().any(|v| !v.is_nan()))
        .collect();
    if keep.is_empty() {
        return Err(HessboostError::invalid_param(
            "data",
            "every row is entirely missing",
        ));
    }
    let rows: Vec<f64> = keep
        .iter()
        .flat_map(|&r| raw[r * p..(r + 1) * p].iter().map(|&v| f64::from(v)))
        .collect();
    let n = keep.len();

    let columns = describe_columns(&rows, p, &kinds)?;
    let c: usize = columns.iter().map(Column::width).sum();
    if c == 0 {
        return Err(HessboostError::invalid_param(
            "data",
            "every column is a single-category categorical: nothing to model",
        ));
    }
    let mut encoded = try_filled(n * c, 0.0, "data")?;
    for (row, out) in rows.chunks_exact(p).zip(encoded.chunks_exact_mut(c)) {
        encode_row(&columns, row, out);
    }
    let scales = fit_scales(&encoded, c);
    for row in encoded.chunks_exact_mut(c) {
        for (v, s) in row.iter_mut().zip(&scales) {
            *v = s.forward(*v);
        }
    }
    let layout = if encoded.iter().any(|v| v.is_nan()) {
        OutputLayout::PerColumn
    } else {
        OutputLayout::Joint
    };
    Ok(Table {
        keep,
        columns,
        scales,
        encoded,
        layout,
    })
}

/// The class labels of the kept rows.
struct Classes {
    /// Sorted distinct labels (empty for an unconditional model).
    values: Vec<f64>,
    /// Their training proportions (empty likewise).
    probs: Vec<f64>,
    /// Each kept row's index into `values` (`0` without classes).
    class_of: Vec<usize>,
}

impl Classes {
    /// The number of GBDT sets: one per class, or one without classes.
    fn count(&self) -> usize {
        self.values.len().max(1)
    }
}

/// The classes of `data`'s labels on the rows `keep`.
fn prepare_classes(data: &DMatrix, keep: &[usize]) -> Classes {
    let n = keep.len();
    let Some(labels) = data.labels() else {
        return Classes {
            values: Vec::new(),
            probs: Vec::new(),
            class_of: vec![0; n],
        };
    };
    let labels: Vec<f64> = keep.iter().map(|&r| f64::from(labels[r])).collect();
    let mut values = labels.clone();
    values.sort_by(f64::total_cmp);
    values.dedup();
    let class_of: Vec<usize> = labels
        .iter()
        .map(|v| values.partition_point(|c| c < v))
        .collect();
    let mut counts = vec![0.0; values.len()];
    for &k in &class_of {
        counts[k] += 1.0;
    }
    let probs = counts.iter().map(|&k| k / n as f64).collect();
    Classes {
        values,
        probs,
        class_of,
    }
}

/// Train the GBDTs in `[class][level]` or `[class][level][column]` order
/// (see [`OutputLayout`]), on noise drawn once for every level.
fn fit_level_models(
    params: &ForestParams,
    table: &Table,
    classes: &Classes,
) -> Result<Vec<BoostedModel>> {
    let (n, c) = (table.keep.len(), table.c());
    // Noise shared by every level, per duplicated row.
    let k = params.duplicate_k.get();
    let n_dup = n
        .checked_mul(k)
        .filter(|m| m.checked_mul(c).is_some())
        .ok_or_else(|| {
            HessboostError::invalid_param("duplicate_k", "the training set size overflows usize")
        })?;
    let mut noise = try_filled(n_dup * c, 0.0, "duplicate_k")?;
    let mut rng = Rng::new(splitmix64(params.seed ^ NOISE_STREAM));
    let mut normal = Normal::default();
    for v in &mut noise {
        *v = normal.draw(&mut rng);
    }

    let targets: Vec<Target> = match table.layout {
        OutputLayout::Joint => vec![Target::All],
        OutputLayout::PerColumn => (0..c).map(Target::Column).collect(),
    };
    let jobs: Vec<(usize, usize, Target)> = (0..classes.count())
        .flat_map(|class| {
            let targets = &targets;
            (0..params.n_t)
                .flat_map(move |level| targets.iter().map(move |&target| (class, level, target)))
        })
        .collect();
    let set = LevelSet {
        method: params.method,
        data: &table.encoded,
        noise: &noise,
        class_of: &classes.class_of,
        n,
        c,
        k,
    };
    jobs.into_par_iter()
        .map(|(class, level, target)| {
            let t = level_time(params.n_t, level);
            let dtrain = set.build(class, t, target)?;
            train(&params.training, &dtrain, params.num_boost_round.get())
        })
        .collect()
}

/// What one GBDT regresses.
#[derive(Debug, Clone, Copy)]
enum Target {
    /// Every encoded column ([`OutputLayout::Joint`]).
    All,
    /// One encoded column, on the rows observing it
    /// ([`OutputLayout::PerColumn`]).
    Column(usize),
}

/// The inputs of one noise level's training set.
struct LevelSet<'a> {
    method: ForestMethod,
    /// Scaled encoded rows `x₁`, `[row][c]`, NaN missing.
    data: &'a [f64],
    /// `x₀` of every duplicated row, `[copy][row][c]`.
    noise: &'a [f64],
    class_of: &'a [usize],
    n: usize,
    c: usize,
    k: usize,
}

impl LevelSet<'_> {
    /// The training matrix of `class` at time `t`: features `x_t`, labels
    /// the flow's velocity `x₁ - x₀` or the diffusion's noise `x₀`, of
    /// `target`.
    fn build(&self, class: usize, t: f64, target: Target) -> Result<DMatrix> {
        let c = self.c;
        let (alpha, std) = match self.method.sde() {
            None => (t, 1.0 - t),
            Some(sde) => sde.marginal(t),
        };
        let width = match target {
            Target::All => c,
            Target::Column(_) => 1,
        };
        let mut features = Vec::new();
        let mut labels = Vec::new();
        for copy in 0..self.k {
            for row in (0..self.n).filter(|&r| self.class_of[r] == class) {
                let x1 = &self.data[row * c..(row + 1) * c];
                if let Target::Column(o) = target
                    && x1[o].is_nan()
                {
                    continue;
                }
                let x0 = &self.noise[(copy * self.n + row) * c..(copy * self.n + row + 1) * c];
                for (&a, &z) in x1.iter().zip(x0) {
                    features.push((alpha * a + std * z) as f32);
                }
                let label = |j: usize| match self.method {
                    ForestMethod::Flow => x1[j] - x0[j],
                    ForestMethod::Diffusion { .. } => x0[j],
                };
                match target {
                    Target::Column(o) => labels.push(label(o) as f32),
                    Target::All => labels.extend((0..c).map(|j| label(j) as f32)),
                }
            }
        }
        let n_rows = labels.len() / width;
        if n_rows == 0 {
            return Err(HessboostError::invalid_param(
                "data",
                match target {
                    Target::All => format!("class {class} has no rows"),
                    Target::Column(o) => {
                        format!("class {class} has no observed value in encoded column {o}")
                    }
                },
            ));
        }
        DMatrix::from_dense_vec(features, n_rows, c)?.with_label_matrix(&labels, width)
    }
}
