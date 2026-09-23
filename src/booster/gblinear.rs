//! The linear (`gblinear`) booster: a generalized linear model fit by
//! coordinate descent.
//!
//! Instead of growing trees, `gblinear` keeps a per-output weight vector and
//! bias and improves them one coordinate at a time. Each round recomputes the
//! objective's gradients/Hessians from the current margins, then, for every
//! output and feature, applies the closed-form coordinate step
//!
//! ```text
//! G_f = Σ_i g_i x_if,   H_f = Σ_i h_i x_if²,   Δw = -(G_f + reg) / (H_f + λ)
//! ```
//!
//! soft-thresholded by the L1 penalty `alpha` and scaled by the learning rate
//! `eta`. The per-output bias is the intercept feature (`x ≡ 1`, unregularized).
//! Gradients and running margins are updated incrementally after each coordinate
//! change, matching XGBoost's `CoordinateUpdater`.
//!
//! Multiclass is supported directly: each output `k` gets its own weight column
//! and bias, fit from that output's gradient slice.

use crate::config::TrainingParams;
use crate::data::DMatrix;
use crate::learner::LinearModel;
use crate::learner::model::for_each_present_value;
use crate::objective::{GradPair, Objective};

/// One feature's present entries, stored column-major as parallel `(row, value)`
/// vectors so a coordinate update touches only the rows where the feature is
/// present (missing entries contribute nothing).
struct Column {
    rows: Vec<u32>,
    vals: Vec<f32>,
}

/// Closed-form coordinate step for a single weight with L1 (`alpha`) and L2
/// (`lambda`) regularization. Mirrors XGBoost's `CoordinateDelta`: returns the
/// change in the weight (before the `eta` scaling), soft-thresholded so the step
/// never crosses past zero.
fn coordinate_delta(sum_grad: f64, sum_hess: f64, w: f64, alpha: f64, lambda: f64) -> f64 {
    if sum_hess < 1e-5 {
        return 0.0;
    }
    let sum_grad_l2 = sum_grad + lambda * w;
    let sum_hess_l2 = sum_hess + lambda;
    let tmp = w - sum_grad_l2 / sum_hess_l2;
    if tmp >= 0.0 {
        (-(sum_grad_l2 + alpha) / sum_hess_l2).max(-w)
    } else {
        (-(sum_grad_l2 - alpha) / sum_hess_l2).min(-w)
    }
}

/// Fit a linear booster by coordinate descent.
///
/// `initial_margin` contains the per-row starting margins, and
/// `n_out` is the number of outputs (`num_class` for multiclass, else 1). The
/// returned [`LinearModel`] holds `weights` laid out `[feature][output]` and a
/// per-output `bias`. Continued training passes the model's current linear
/// booster as `start` (its margins in `initial_margin`); coordinate descent
/// then resumes from its weights and bias instead of zeros.
pub(crate) fn train_gblinear(
    params: &TrainingParams,
    dtrain: &DMatrix,
    num_round: usize,
    initial_margin: &[f32],
    n_out: usize,
    objective: &dyn Objective,
    start: Option<&LinearModel>,
) -> LinearModel {
    let n = dtrain.n_rows();
    let n_features = dtrain.n_cols();
    // Label presence was validated by the training entry point (only
    // objectives that `requires_labels` need them).
    let info = dtrain.info();

    let eta = params.eta;
    let lambda = params.lambda;
    let alpha = params.alpha;

    // Column-major cache of present feature values.
    let mut cols: Vec<Column> = (0..n_features)
        .map(|_| Column {
            rows: Vec::new(),
            vals: Vec::new(),
        })
        .collect();
    for row in 0..n {
        for_each_present_value(dtrain, row, |f, x| {
            if x != 0.0 {
                let col = &mut cols[f];
                col.rows.push(row as u32);
                col.vals.push(x);
            }
        });
    }

    let (mut lin_weights, mut bias) = match start {
        Some(model) => (model.weights().to_vec(), model.bias().to_vec()),
        None => (vec![0.0f32; n_features * n_out], vec![0.0f32; n_out]),
    };

    // Running margins [instance][output]; gradients recomputed each round, then
    // updated incrementally as each coordinate moves.
    let mut margin = initial_margin.to_vec();
    let mut gpair = vec![GradPair::default(); n * n_out];

    for _round in 0..num_round {
        objective.gradient_info(&margin, &info, &mut gpair);

        for k in 0..n_out {
            // 1. Bias (intercept) update: G = Σ g, H = Σ h.
            let mut g = 0.0f64;
            let mut h = 0.0f64;
            for i in 0..n {
                let gp = gpair[i * n_out + k];
                g += f64::from(gp.grad);
                h += f64::from(gp.hess);
            }
            let db = eta * coordinate_delta(g, h, 0.0, 0.0, 0.0);
            if db != 0.0 {
                let db32 = db as f32;
                bias[k] += db32;
                for i in 0..n {
                    let gp = &mut gpair[i * n_out + k];
                    gp.grad += gp.hess * db32;
                    margin[i * n_out + k] += db32;
                }
            }

            // 2. Per-feature coordinate updates.
            for f in 0..n_features {
                let col = &cols[f];
                let mut g = 0.0f64;
                let mut h = 0.0f64;
                for (idx, &row) in col.rows.iter().enumerate() {
                    let x = f64::from(col.vals[idx]);
                    let gp = gpair[row as usize * n_out + k];
                    g += f64::from(gp.grad) * x;
                    h += f64::from(gp.hess) * x * x;
                }
                let w = f64::from(lin_weights[f * n_out + k]);
                let dw = eta * coordinate_delta(g, h, w, alpha, lambda);
                if dw == 0.0 {
                    continue;
                }
                let dw32 = dw as f32;
                lin_weights[f * n_out + k] += dw32;
                for (idx, &row) in col.rows.iter().enumerate() {
                    let x = col.vals[idx];
                    let gp = &mut gpair[row as usize * n_out + k];
                    gp.grad += gp.hess * x * dw32;
                    margin[row as usize * n_out + k] += x * dw32;
                }
            }
        }
    }

    LinearModel::new(lin_weights, bias)
}
