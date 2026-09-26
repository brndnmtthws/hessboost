//! [`DiffusionModel::fit`]: label standardization, residualization, the
//! noisy training set, and the score/velocity GBDT.

use super::process::{Normal, draw_times};
use super::{
    DiffusionModel, DiffusionParams, FittedResidualizer, Method, Parameterization, Residualizer,
};
use crate::data::{DMatrix, FeatureType};
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::rng::{Rng, splitmix64};
use crate::training::{Trainer, train};

/// Stream of the validation split.
const SPLIT_STREAM: u64 = 0x5EED_0001;
/// Stream of the residualizer's fold assignment.
const FOLD_STREAM: u64 = 0x5EED_0002;
/// Stream of the training times and noise.
const NOISE_STREAM: u64 = 0x5EED_0003;

/// Fewest rows residualization accepts (two folds of 40, DiffGBM's
/// `MIN_RESIDUALIZE_ROWS`).
const MIN_RESIDUALIZE_ROWS: usize = 80;
/// Rows per fold the residualizer aims for (DiffGBM: `n // 40` folds).
const ROWS_PER_FOLD: usize = 40;

pub(super) fn fit(params: &DiffusionParams, data: &DMatrix) -> Result<DiffusionModel> {
    params.validate()?;
    let labels = check_data(data)?;
    let (n, d, p) = (data.n_rows(), data.n_targets(), data.n_cols());

    let (target_mean, target_scale) = standardization(labels, d);
    let standardized: Vec<f64> = labels
        .chunks_exact(d)
        .flat_map(|row| {
            row.iter()
                .zip(target_mean.iter().zip(&target_scale))
                .map(|(&y, (&m, &s))| (f64::from(y) - m) / s)
        })
        .collect();

    let (residualizer, targets) = match &params.residualizer {
        Some(r) => {
            let (fitted, residuals) = residualize(r, data, &standardized, params.seed)?;
            (Some(fitted), residuals)
        }
        None => (None, standardized),
    };

    let features = dense_features(data);
    let mut rows: Vec<usize> = (0..n).collect();
    let n_val = match &params.early_stopping {
        Some(stop) => {
            let n_val = (stop.eval_fraction * n as f64).ceil() as usize;
            if n_val >= n {
                return Err(HessboostError::invalid_param(
                    "early_stopping.eval_fraction",
                    format!("holds out {n_val} of {n} rows, leaving none to train on"),
                ));
            }
            Rng::new(splitmix64(params.seed ^ SPLIT_STREAM)).shuffle(&mut rows);
            n_val
        }
        None => 0,
    };
    let (val_rows, train_rows) = rows.split_at(n_val);

    let set = TrainingSet {
        method: &params.method,
        features: &features,
        targets: &targets,
        n_features: p,
        n_outputs: d,
        feature_types: data.feature_types(),
    };
    let mut rng = Rng::new(splitmix64(params.seed ^ NOISE_STREAM));
    let mut normal = Normal::default();
    let dtrain = set.build(train_rows, params.n_repeats, &mut rng, &mut normal)?;
    let regressor = match &params.early_stopping {
        Some(stop) => {
            let dval = set.build(val_rows, 1, &mut rng, &mut normal)?;
            Trainer::new(&params.training, &dtrain, params.num_boost_round)
                .eval(&dval, "valid")
                .early_stopping_rounds(stop.rounds)
                .train()?
                .model
        }
        None => train(&params.training, &dtrain, params.num_boost_round)?,
    };

    let model = DiffusionModel {
        method: params.method,
        n_steps: params.n_steps,
        n_features: p,
        n_outputs: d,
        target_mean,
        target_scale,
        residualizer,
        regressor,
    };
    model.validate()?;
    Ok(model)
}

/// The labels of `data`, after refusing the metadata a diffusion model
/// cannot honor.
fn check_data(data: &DMatrix) -> Result<&[f32]> {
    let refuse = |what: &str| {
        Err(HessboostError::invalid_param(
            "data",
            format!("diffusion models do not support {what}"),
        ))
    };
    if data.weights().is_some() {
        return refuse("instance weights");
    }
    if data.base_margin().is_some() {
        return refuse("base margins");
    }
    if data.group().is_some() {
        return refuse("ranking groups");
    }
    if data.label_lower_bound().is_some() || data.label_upper_bound().is_some() {
        return refuse("label bounds");
    }
    if data.feature_weights().is_some() {
        return refuse("feature weights");
    }
    let labels = data.labels().ok_or_else(|| {
        HessboostError::invalid_param("data", "fitting a diffusion model needs labels")
    })?;
    if data.n_rows() < 2 {
        return Err(HessboostError::invalid_param(
            "data",
            "fitting a diffusion model needs at least 2 rows",
        ));
    }
    Ok(labels)
}

/// Per-column mean and population standard deviation of the `[row][d]`
/// labels (scikit-learn's `StandardScaler`: a zero deviation becomes `1`).
fn standardization(labels: &[f32], d: usize) -> (Vec<f64>, Vec<f64>) {
    let n = (labels.len() / d) as f64;
    let mut mean = vec![0.0; d];
    for row in labels.chunks_exact(d) {
        for (m, &y) in mean.iter_mut().zip(row) {
            *m += f64::from(y);
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0.0; d];
    for row in labels.chunks_exact(d) {
        for ((v, &y), &m) in var.iter_mut().zip(row).zip(&mean) {
            let e = f64::from(y) - m;
            *v += e * e;
        }
    }
    let scale = var
        .into_iter()
        .map(|v| {
            let s = (v / n).sqrt();
            if s > 0.0 && s.is_finite() { s } else { 1.0 }
        })
        .collect();
    (mean, scale)
}

/// The features of `data` as a dense row-major `f32` matrix with NaN for
/// missing entries.
pub(super) fn dense_features(data: &DMatrix) -> Vec<f32> {
    let p = data.n_cols();
    let mut out = vec![f32::NAN; data.n_rows() * p];
    for (row, values) in out.chunks_exact_mut(p).enumerate() {
        data.for_row_entry(row, |col, v| values[col as usize] = v);
    }
    out
}

/// Fit the cross-fitted conditional-mean residualizer on the standardized
/// labels `y` (`[row][d]`) and return it with the diffusion targets: the
/// out-of-fold residuals, centered and scaled.
fn residualize(
    config: &Residualizer,
    data: &DMatrix,
    y: &[f64],
    seed: u64,
) -> Result<(FittedResidualizer, Vec<f64>)> {
    let (n, d) = (data.n_rows(), data.n_targets());
    if n < MIN_RESIDUALIZE_ROWS {
        return Err(HessboostError::invalid_param(
            "residualizer",
            format!(
                "cross-fitted residualization needs at least {MIN_RESIDUALIZE_ROWS} rows, got \
                 {n}; set `residualizer` to `None`"
            ),
        ));
    }
    let folds = config.folds.min((n / ROWS_PER_FOLD).max(2));
    // scikit-learn's shuffled `KFold`: a permutation cut into `folds`
    // contiguous parts, the first `n % folds` one row longer.
    let mut order: Vec<usize> = (0..n).collect();
    Rng::new(splitmix64(seed ^ FOLD_STREAM)).shuffle(&mut order);
    let labels: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let mut oof = vec![0.0; n * d];
    let mut models = Vec::with_capacity(folds);
    let mut held_out = vec![false; n];
    let mut start = 0;
    for fold in 0..folds {
        let len = n / folds + usize::from(fold < n % folds);
        let test = &order[start..start + len];
        start += len;
        held_out.fill(false);
        for &row in test {
            held_out[row] = true;
        }
        let train_rows: Vec<usize> = (0..n).filter(|&row| !held_out[row]).collect();
        let fold_labels: Vec<f32> = train_rows
            .iter()
            .flat_map(|&row| labels[row * d..(row + 1) * d].iter().copied())
            .collect();
        let dfold = data
            .select_rows(&train_rows)?
            .with_label_matrix(&fold_labels, d)?;
        let model = train(&config.training, &dfold, config.num_boost_round)?;
        let pred = model.predict_margin(&data.select_rows(test)?)?;
        for (&row, values) in test.iter().zip(pred.chunks_exact(d)) {
            for (o, &v) in oof[row * d..(row + 1) * d].iter_mut().zip(values) {
                *o = f64::from(v);
            }
        }
        models.push(model);
    }

    let mut residual: Vec<f64> = y.iter().zip(&oof).map(|(&y, &m)| y - m).collect();
    let mut center = vec![0.0; d];
    for row in residual.chunks_exact(d) {
        for (c, &r) in center.iter_mut().zip(row) {
            *c += r;
        }
    }
    for c in &mut center {
        *c /= n as f64;
    }
    let mut column = vec![0.0; n];
    let mut scale = vec![0.0; d];
    for j in 0..d {
        for (value, row) in column.iter_mut().zip(residual.chunks_exact(d)) {
            *value = row[j] - center[j];
        }
        scale[j] = robust_scale(&mut column);
    }
    for row in residual.chunks_exact_mut(d) {
        for ((r, &c), &s) in row.iter_mut().zip(&center).zip(&scale) {
            *r = (*r - c) / s;
        }
    }
    Ok((
        FittedResidualizer {
            models,
            center,
            scale,
        },
        residual,
    ))
}

/// DiffGBM's residual scale: the standard deviation after winsorizing at
/// the 1% and 99% quantiles; if that vanishes, `1.4826 ·` the median
/// absolute deviation; if that vanishes too, `1`. Reorders `values`.
fn robust_scale(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let (lo, hi) = (quantile_sorted(values, 0.01), quantile_sorted(values, 0.99));
    let n = values.len() as f64;
    let clipped = || values.iter().map(|v| v.clamp(lo, hi));
    let mean = clipped().sum::<f64>() / n;
    let std = (clipped().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    if std > f64::EPSILON {
        return std;
    }
    let median = quantile_sorted(values, 0.5);
    let mut deviations: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
    deviations.sort_unstable_by(f64::total_cmp);
    let mad = 1.4826 * quantile_sorted(&deviations, 0.5);
    if mad > f64::EPSILON { mad } else { 1.0 }
}

/// The `level` quantile of the non-empty ascending `sorted` by linear
/// interpolation (NumPy's default).
pub(super) fn quantile_sorted(sorted: &[f64], level: f64) -> f64 {
    let pos = level * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
}

/// The inputs of the noisy training set.
struct TrainingSet<'a> {
    method: &'a Method,
    /// Dense `[row][n_features]` features, NaN for missing.
    features: &'a [f32],
    /// Diffusion targets `y₀`, `[row][n_outputs]`.
    targets: &'a [f64],
    n_features: usize,
    n_outputs: usize,
    feature_types: &'a [FeatureType],
}

impl TrainingSet<'_> {
    /// The GBDT's training matrix for `rows`, each repeated `repeats` times
    /// with its own time and noise: features `[y_t, x, t, (ln σ(t))]`,
    /// labels the method's regression target.
    fn build(
        &self,
        rows: &[usize],
        repeats: usize,
        rng: &mut Rng,
        normal: &mut Normal,
    ) -> Result<DMatrix> {
        let (d, p) = (self.n_outputs, self.n_features);
        let cols = d + p + self.method.time_columns();
        let n_aug = rows
            .len()
            .checked_mul(repeats)
            .filter(|n| n.checked_mul(cols).is_some())
            .ok_or_else(|| {
                HessboostError::invalid_param("n_repeats", "the training set size overflows usize")
            })?;
        let times = match self.method {
            Method::Score(score) => draw_times(
                n_aug,
                score.time_sampling,
                |t| score.sde.marginal(t).1,
                false,
                rng,
                normal,
            ),
            Method::FlowMatching(flow) => draw_times(
                n_aug,
                flow.time_sampling,
                |t| flow.path.coefficients(t).b,
                true,
                rng,
                normal,
            ),
        };
        let mut features = vec![0.0f32; n_aug * cols];
        let mut labels = vec![0.0f32; n_aug * d];
        // Repetition-major, as Treeffuser tiles the data.
        let augmented = (0..repeats).flat_map(|_| rows.iter().copied());
        for (((row, &t), feature_row), label_row) in augmented
            .zip(&times)
            .zip(features.chunks_exact_mut(cols))
            .zip(labels.chunks_exact_mut(d))
        {
            let y0 = &self.targets[row * d..(row + 1) * d];
            feature_row[d..d + p].copy_from_slice(&self.features[row * p..(row + 1) * p]);
            feature_row[d + p] = t as f32;
            match self.method {
                Method::Score(score) => {
                    let (alpha, std) = score.sde.marginal(t);
                    if score.noise_level_feature {
                        feature_row[d + p + 1] = std.ln() as f32;
                    }
                    for j in 0..d {
                        let z = normal.draw(rng);
                        let y_t = alpha * y0[j] + std * z;
                        let (input, target) = match score.parameterization {
                            Parameterization::Noise => (y_t, -z),
                            Parameterization::Edm { sigma_data } => {
                                let c = Edm::at(sigma_data, std);
                                (c.input * y_t, (y0[j] - c.skip * y_t) / c.out)
                            }
                        };
                        feature_row[j] = input as f32;
                        label_row[j] = target as f32;
                    }
                }
                Method::FlowMatching(flow) => {
                    let c = flow.path.coefficients(t);
                    for j in 0..d {
                        let z = normal.draw(rng);
                        feature_row[j] = (c.a * y0[j] + c.b * z) as f32;
                        label_row[j] = (c.da * y0[j] + c.db * z) as f32;
                    }
                }
            }
        }
        let mut types = vec![FeatureType::Numerical; cols];
        types[d..d + p].copy_from_slice(self.feature_types);
        let mut matrix = DMatrix::from_dense_vec(features, n_aug, cols)?;
        if types.contains(&FeatureType::Categorical) {
            matrix = matrix.with_feature_types(&types)?;
        }
        matrix.with_label_matrix(&labels, d)
    }
}

/// EDM's preconditioning coefficients at noise scale `σ` (Karras et al.,
/// 2022, Table 1).
#[derive(Debug, Clone, Copy)]
pub(super) struct Edm {
    /// `c_skip`: the weight of `y_t` in the denoiser.
    pub(super) skip: f64,
    /// `c_out`: the weight of the regressor's output in the denoiser.
    pub(super) out: f64,
    /// `c_in`: the scale of `y_t` in the regressor's input.
    pub(super) input: f64,
}

impl Edm {
    pub(super) fn at(sigma_data: f64, sigma: f64) -> Self {
        let sd2 = sigma_data * sigma_data;
        let denom = sigma * sigma + sd2;
        Edm {
            skip: sd2 / denom,
            out: sigma * sigma_data / denom.sqrt(),
            input: 1.0 / denom.sqrt(),
        }
    }
}

/// The fold models' average prediction for every row of `data`
/// (`[row][d]`), the residualizer's conditional mean.
pub(super) fn residual_mean(models: &[BoostedModel], data: &DMatrix) -> Result<Vec<f64>> {
    let mut mean: Vec<f64> = Vec::new();
    for model in models {
        let pred = model.predict_margin(data)?;
        if mean.is_empty() {
            mean = vec![0.0; pred.len()];
        }
        for (m, &v) in mean.iter_mut().zip(&pred) {
            *m += f64::from(v);
        }
    }
    let k = models.len() as f64;
    for m in &mut mean {
        *m /= k;
    }
    Ok(mean)
}
