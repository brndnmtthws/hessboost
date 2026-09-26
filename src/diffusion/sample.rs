//! [`DiffusionModel::sample`]: reverse-time integration from the prior, and
//! the [`Samples`] it returns.

use super::fit::{Edm, dense_features, quantile_sorted, residual_mean};
use super::process::{T_EPS, keyed_normal};
use super::{DiffusionModel, Method, OdeSolver, Parameterization, ScoreConfig};
use crate::data::DMatrix;
use crate::error::{HessboostError, Result};
use crate::rng::splitmix64;

/// Stream of the sampler's noise.
const SAMPLE_STREAM: u64 = 0x5EED_0004;
/// `(row, sample)` pairs integrated per batch: bounds the working set
/// without affecting the result (every draw is keyed by its pair).
const CHUNK_PAIRS: usize = 1 << 14;

/// Draws from a [`DiffusionModel`], laid out row-major
/// `[row][sample][output]`: value `(r, s, o)` is at
/// `(r · n_samples + s) · n_outputs + o`.
#[derive(Debug, Clone, PartialEq)]
pub struct Samples {
    values: Vec<f32>,
    n_rows: usize,
    per_row: usize,
    n_outputs: usize,
}

impl Samples {
    /// Every draw, `[row][sample][output]`.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The draws as an owned vector, `[row][sample][output]`.
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }

    /// Rows sampled (the rows of the matrix passed to
    /// [`DiffusionModel::sample`]).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Draws per row.
    pub fn n_samples(&self) -> usize {
        self.per_row
    }

    /// Label columns per draw.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// The draws of `row`, `[sample][output]`, or `None` past the last row.
    pub fn row(&self, row: usize) -> Option<&[f32]> {
        let width = self.per_row * self.n_outputs;
        let start = row.checked_mul(width)?;
        self.values.get(start..start.checked_add(width)?)
    }

    /// Monte Carlo estimate of the conditional mean, `[row][output]`.
    pub fn mean(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.n_rows * self.n_outputs];
        let width = self.per_row * self.n_outputs;
        for (mean, draws) in out
            .chunks_exact_mut(self.n_outputs)
            .zip(self.values.chunks_exact(width))
        {
            for draw in draws.chunks_exact(self.n_outputs) {
                for (m, &v) in mean.iter_mut().zip(draw) {
                    *m += f64::from(v);
                }
            }
            for m in mean {
                *m /= self.per_row as f64;
            }
        }
        out
    }

    /// Empirical quantiles at `levels` (each in `[0, 1]`) by linear
    /// interpolation between order statistics (NumPy's default),
    /// `[row][level][output]`.
    ///
    /// # Errors
    ///
    /// [`HessboostError::InvalidParameter`] for a level outside `[0, 1]`.
    pub fn quantiles(&self, levels: &[f64]) -> Result<Vec<f64>> {
        if let Some(&level) = levels.iter().find(|l| !(0.0..=1.0).contains(*l)) {
            return Err(HessboostError::invalid_param(
                "levels",
                format!("quantile levels must be in [0, 1], got {level}"),
            ));
        }
        let (k, d) = (levels.len(), self.n_outputs);
        let mut out = vec![0.0; self.n_rows * k * d];
        let mut column = vec![0.0; self.per_row];
        for row in 0..self.n_rows {
            for o in 0..d {
                self.sorted_column(row, o, &mut column);
                for (l, &level) in levels.iter().enumerate() {
                    out[(row * k + l) * d + o] = quantile_sorted(&column, level);
                }
            }
        }
        Ok(out)
    }

    /// The continuous ranked probability score of each label under the
    /// empirical distribution of its row's draws, `[row][output]`:
    /// `mean |X - y| - ½ mean |X - X'|` over the draws `X`, `X'` (the
    /// ensemble CRPS; lower is better). `labels` is `[row][output]`.
    ///
    /// # Errors
    ///
    /// [`HessboostError::DimensionMismatch`] unless `labels` holds
    /// `n_rows · n_outputs` values; [`HessboostError::InvalidParameter`] for
    /// a non-finite label.
    pub fn crps(&self, labels: &[f32]) -> Result<Vec<f64>> {
        let d = self.n_outputs;
        if labels.len() != self.n_rows * d {
            return Err(HessboostError::DimensionMismatch {
                what: "crps labels",
                expected: self.n_rows * d,
                got: labels.len(),
            });
        }
        if labels.iter().any(|v| !v.is_finite()) {
            return Err(HessboostError::invalid_param(
                "labels",
                "all labels must be finite",
            ));
        }
        let s = self.per_row as f64;
        let mut column = vec![0.0; self.per_row];
        let mut out = vec![0.0; self.n_rows * d];
        for row in 0..self.n_rows {
            for o in 0..d {
                self.sorted_column(row, o, &mut column);
                let y = f64::from(labels[row * d + o]);
                let spread_to_label = column.iter().map(|x| (x - y).abs()).sum::<f64>() / s;
                // Σᵢⱼ |xᵢ - xⱼ| = 2 Σᵢ (2i - S + 1) x₍ᵢ₎ over the sorted draws.
                let pairwise: f64 = column
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (2.0 * i as f64 - s + 1.0) * x)
                    .sum::<f64>()
                    * 2.0;
                out[row * d + o] = spread_to_label - pairwise / (2.0 * s * s);
            }
        }
        Ok(out)
    }

    /// The draws of `(row, output)` into `column`, ascending.
    fn sorted_column(&self, row: usize, output: usize, column: &mut [f64]) {
        let base = row * self.per_row * self.n_outputs + output;
        for (s, c) in column.iter_mut().enumerate() {
            *c = f64::from(self.values[base + s * self.n_outputs]);
        }
        column.sort_unstable_by(f64::total_cmp);
    }
}

pub(super) fn sample(
    model: &DiffusionModel,
    data: &DMatrix,
    n_samples: usize,
    seed: u64,
) -> Result<Samples> {
    if n_samples == 0 {
        return Err(HessboostError::invalid_param(
            "n_samples",
            "must be at least 1",
        ));
    }
    if data.n_cols() != model.n_features {
        return Err(HessboostError::DimensionMismatch {
            what: "sampling feature count",
            expected: model.n_features,
            got: data.n_cols(),
        });
    }
    if data.base_margin().is_some() {
        return Err(HessboostError::invalid_param(
            "data",
            "diffusion models do not support base margins",
        ));
    }
    let (n_rows, d) = (data.n_rows(), model.n_outputs);
    let n_pairs = n_rows.checked_mul(n_samples);
    let Some(total) = n_pairs.and_then(|pairs| pairs.checked_mul(d)) else {
        return Err(HessboostError::invalid_param(
            "n_samples",
            "rows × samples × outputs overflows usize",
        ));
    };
    let mean = match &model.residualizer {
        Some(r) => Some(residual_mean(&r.models, data)?),
        None => None,
    };
    let sampler = Sampler {
        model,
        features: dense_features(data),
        mean,
        n_samples,
        key: splitmix64(seed ^ SAMPLE_STREAM),
    };
    let mut values = vec![0.0f32; total];
    let n_pairs = n_rows * n_samples;
    for (chunk, out) in values.chunks_mut(CHUNK_PAIRS * d).enumerate() {
        let start = chunk * CHUNK_PAIRS;
        sampler.chunk(start..(start + CHUNK_PAIRS).min(n_pairs), out)?;
    }
    Ok(Samples {
        values,
        n_rows,
        per_row: n_samples,
        n_outputs: d,
    })
}

/// Sampling state shared by the batches.
struct Sampler<'a> {
    model: &'a DiffusionModel,
    /// Dense `[row][n_features]` features, NaN for missing.
    features: Vec<f32>,
    /// The residualizer's conditional mean, `[row][n_outputs]`.
    mean: Option<Vec<f64>>,
    n_samples: usize,
    key: u64,
}

/// One batch of `(row, sample)` pairs: the regressor's input matrix and
/// the pairs' noise streams.
struct Batch {
    /// `[pair][y_t, x, t, (ln σ)]`; the `x` columns are filled once.
    input: DMatrix,
    streams: Vec<u64>,
    cols: usize,
}

impl Batch {
    /// The regressor's predictions at states `y` (`[pair][output]`, fed
    /// through `scale`) and time `t`, `[pair][output]`.
    fn predict(
        &mut self,
        model: &DiffusionModel,
        y: &[f64],
        scale: f64,
        time: [f32; 2],
    ) -> Result<Vec<f32>> {
        let d = model.n_outputs;
        let t_col = d + model.n_features;
        let values = self
            .input
            .dense_values_mut()
            .ok_or_else(|| HessboostError::model_format("sampling input must be dense"))?;
        for (row, state) in values.chunks_exact_mut(self.cols).zip(y.chunks_exact(d)) {
            for (v, &s) in row.iter_mut().zip(state) {
                *v = (s * scale) as f32;
            }
            row[t_col..].copy_from_slice(&time[..self.cols - t_col]);
        }
        model.regressor.predict_margin(&self.input)
    }
}

impl Sampler<'_> {
    /// Integrate the pairs `pairs` (index `row · n_samples + sample`) and
    /// write their draws to `out` (`[pair][output]`).
    fn chunk(&self, pairs: std::ops::Range<usize>, out: &mut [f32]) -> Result<()> {
        let model = self.model;
        let (d, p) = (model.n_outputs, model.n_features);
        let cols = d + p + model.method.time_columns();
        let m = pairs.len();
        let mut input = vec![0.0f32; m * cols];
        let mut streams = Vec::with_capacity(m);
        for (pair, row_values) in pairs.clone().zip(input.chunks_exact_mut(cols)) {
            let (row, sample) = (pair / self.n_samples, pair % self.n_samples);
            row_values[d..d + p].copy_from_slice(&self.features[row * p..(row + 1) * p]);
            streams.push(splitmix64(
                splitmix64(self.key ^ row as u64) ^ sample as u64,
            ));
        }
        let mut batch = Batch {
            input: DMatrix::from_dense_vec(input, m, cols)?,
            streams,
            cols,
        };

        let prior_std = match model.method {
            Method::Score(score) => score.sde.prior_std(),
            Method::FlowMatching(_) => 1.0,
        };
        let mut y = vec![0.0; m * d];
        for (state, &stream) in y.chunks_exact_mut(d).zip(&batch.streams) {
            for (j, v) in state.iter_mut().enumerate() {
                *v = prior_std * keyed_normal(stream, j as u64);
            }
        }
        match model.method {
            Method::Score(score) => self.reverse_sde(&score, &mut batch, &mut y)?,
            Method::FlowMatching(flow) => self.reverse_ode(flow.solver, &mut batch, &mut y)?,
        }

        for ((pair, state), draws) in pairs.zip(y.chunks_exact(d)).zip(out.chunks_exact_mut(d)) {
            let row = pair / self.n_samples;
            for (j, (&u, v)) in state.iter().zip(draws).enumerate() {
                let standardized = match (&model.residualizer, &self.mean) {
                    (Some(r), Some(mean)) => mean[row * d + j] + u * r.scale[j] + r.center[j],
                    _ => u,
                };
                let value = (standardized * model.target_scale[j] + model.target_mean[j]) as f32;
                if !value.is_finite() {
                    return Err(diverged());
                }
                *v = value;
            }
        }
        Ok(())
    }

    /// Euler–Maruyama on the reverse SDE
    /// `dY = [-c(t) Y + g(t)² ∇log p_t(Y)] ds + g(t) dW̄` from `t = 1` to
    /// `t = 10⁻⁵` (Treeffuser's sampler).
    fn reverse_sde(&self, score: &ScoreConfig, batch: &mut Batch, y: &mut [f64]) -> Result<()> {
        let model = self.model;
        let d = model.n_outputs;
        let steps = model.n_steps;
        let dt = (1.0 - T_EPS) / steps as f64;
        for step in 0..steps {
            let t = 1.0 - step as f64 * dt;
            let (alpha, std) = score.sde.marginal(t);
            let (c, g2) = score.sde.drift_diffusion(t);
            let edm = match score.parameterization {
                Parameterization::Noise => None,
                Parameterization::Edm { sigma_data } => Some(Edm::at(sigma_data, std)),
            };
            let input_scale = edm.map_or(1.0, |e| e.input);
            let pred = batch.predict(model, y, input_scale, [t as f32, std.ln() as f32])?;
            let noise_scale = (g2 * dt).sqrt();
            let counter = (step as u64 + 1) * d as u64;
            for ((state, out), &stream) in y
                .chunks_exact_mut(d)
                .zip(pred.chunks_exact(d))
                .zip(&batch.streams)
            {
                for (j, (v, &f)) in state.iter_mut().zip(out).enumerate() {
                    let f = f64::from(f);
                    let score = match edm {
                        None => f / std,
                        Some(e) => (alpha * (e.skip * *v + e.out * f) - *v) / (std * std),
                    };
                    let drift = -c * *v + g2 * score;
                    *v += drift * dt + noise_scale * keyed_normal(stream, counter + j as u64);
                }
            }
            check_finite(y)?;
        }
        Ok(())
    }

    /// The reverse ODE `dY/ds = -v(Y, 1 - s)` from `t = 1` to `t = 10⁻⁵`
    /// (DiffGBM's flow-matching sampler).
    fn reverse_ode(&self, solver: OdeSolver, batch: &mut Batch, y: &mut [f64]) -> Result<()> {
        let model = self.model;
        let steps = model.n_steps;
        let ds = (1.0 - T_EPS) / steps as f64;
        let time = |t: f64| [t as f32, 0.0];
        let mut predictor = match solver {
            OdeSolver::Euler => Vec::new(),
            OdeSolver::Heun => vec![0.0; y.len()],
        };
        for step in 0..steps {
            let t = 1.0 - step as f64 * ds;
            let v0 = batch.predict(model, y, 1.0, time(t))?;
            match solver {
                OdeSolver::Euler => {
                    for (s, &v) in y.iter_mut().zip(&v0) {
                        *s -= f64::from(v) * ds;
                    }
                }
                OdeSolver::Heun => {
                    for ((pred, &s), &v) in predictor.iter_mut().zip(y.iter()).zip(&v0) {
                        *pred = s - f64::from(v) * ds;
                    }
                    check_finite(&predictor)?;
                    let v1 = batch.predict(model, &predictor, 1.0, time(t - ds))?;
                    for ((s, &a), &b) in y.iter_mut().zip(&v0).zip(&v1) {
                        *s -= f64::midpoint(f64::from(a), f64::from(b)) * ds;
                    }
                }
            }
            check_finite(y)?;
        }
        Ok(())
    }
}

/// The states stay representable as `f32` features.
fn check_finite(y: &[f64]) -> Result<()> {
    if y.iter().all(|v| v.abs() <= f64::from(f32::MAX)) {
        Ok(())
    } else {
        Err(diverged())
    }
}

fn diverged() -> HessboostError {
    HessboostError::invalid_param(
        "n_steps",
        "sampling diverged to non-finite values; use more steps",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sorted-draw CRPS equals the pairwise definition, and the
    /// quantiles interpolate between order statistics.
    #[test]
    fn crps_and_quantiles_match_their_definitions() {
        let values = vec![0.5, 3.0, -1.0, 2.0, 0.0, 1.0, -2.0, 4.0];
        let samples = Samples {
            values: values.clone(),
            n_rows: 2,
            per_row: 2,
            n_outputs: 2,
        };
        let labels = [1.0, -0.5, 0.25, 2.0];
        let crps = samples.crps(&labels).unwrap();
        for row in 0..2 {
            for o in 0..2 {
                let draws: Vec<f64> = (0..2)
                    .map(|s| f64::from(values[(row * 2 + s) * 2 + o]))
                    .collect();
                let y = f64::from(labels[row * 2 + o]);
                let to_label = draws.iter().map(|x| (x - y).abs()).sum::<f64>() / 2.0;
                let pairwise = draws
                    .iter()
                    .flat_map(|a| draws.iter().map(move |b| (a - b).abs()))
                    .sum::<f64>()
                    / 8.0;
                assert!((crps[row * 2 + o] - (to_label - pairwise)).abs() < 1e-12);
            }
        }
        // Row 0, output 0 draws {0.5, -1}: the median is their midpoint.
        let q = samples.quantiles(&[0.0, 0.5, 1.0]).unwrap();
        assert_eq!(&q[..6], &[-1.0, 2.0, -0.25, 2.5, 0.5, 3.0]);
        assert_eq!(samples.mean()[..2], [-0.25, 2.5]);
        assert!(samples.quantiles(&[1.5]).is_err());
        assert!(samples.crps(&[0.0; 3]).is_err());
    }
}
