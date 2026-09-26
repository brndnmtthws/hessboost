//! The diffusion model's native binary and JSON formats.
//!
//! Binary: the native model container ([`crate::model::native`]) with magic
//! `HBDM` and its own version byte, zstd-compressed:
//!
//! ```text
//! b"HBDM"    magic
//! u8         VERSION
//! sections   see below, nothing after the last payload
//! u64        XXH64 (seed 0) of every byte before it
//! ```
//!
//! `diffusion.*` hold the method (by name, with its parameters), the
//! sampler's step count and the shape; `target.*` the label
//! standardization; `residual.*` the residualizer (all absent without one);
//! `regressor.model` and `residual.models` embed uncompressed native
//! containers of the GBDTs (`residual.model_lengths` splits the latter).
//! Every section is `REQUIRED` except `diffusion.writer`, and the reader
//! refuses unknown values inside known sections, as the native reader does.

use serde::Deserialize;

use super::{
    DiffusionModel, FittedResidualizer, FlowMatchingConfig, FlowPath, Method, OdeSolver,
    Parameterization, ScoreConfig, Sde, TimeSampling,
};
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::model::native::{frame, pack, section_table, unpack, write_container};
use crate::model::sections::{Sections, Writer, format_error};

const MAGIC: &[u8; 4] = b"HBDM";
const VERSION: u8 = 1;
const WRITER: &str = concat!("hessboost ", env!("CARGO_PKG_VERSION"));

/// Every section this version reads.
const KNOWN: &[&str] = &[
    "diffusion.writer",
    "diffusion.method",
    "diffusion.n_steps",
    "diffusion.n_features",
    "diffusion.n_outputs",
    "score.sde",
    "score.sde_min",
    "score.sde_max",
    "score.parameterization",
    "score.sigma_data",
    "score.noise_level_feature",
    "flow.path",
    "flow.beta_min",
    "flow.beta_max",
    "flow.solver",
    "time.sampling",
    "time.mean",
    "time.std",
    "target.mean",
    "target.scale",
    "residual.center",
    "residual.scale",
    "residual.model_lengths",
    "residual.models",
    "regressor.model",
];

pub(super) fn write(model: &DiffusionModel) -> Result<Vec<u8>> {
    let mut w = Writer::default();
    w.raw("diffusion.writer", 0, WRITER.as_bytes());
    let time_sampling = match &model.method {
        Method::Score(score) => {
            w.str("diffusion.method", "score");
            let (name, lo, hi) = match score.sde {
                Sde::VarianceExploding {
                    sigma_min,
                    sigma_max,
                } => ("variance_exploding", sigma_min, sigma_max),
                Sde::VariancePreserving { beta_min, beta_max } => {
                    ("variance_preserving", beta_min, beta_max)
                }
                Sde::SubVariancePreserving { beta_min, beta_max } => {
                    ("sub_variance_preserving", beta_min, beta_max)
                }
            };
            w.str("score.sde", name);
            w.f64("score.sde_min", lo);
            w.f64("score.sde_max", hi);
            match score.parameterization {
                Parameterization::Noise => w.str("score.parameterization", "noise"),
                Parameterization::Edm { sigma_data } => {
                    w.str("score.parameterization", "edm");
                    w.f64("score.sigma_data", sigma_data);
                }
            }
            w.u64(
                "score.noise_level_feature",
                u64::from(score.noise_level_feature),
            );
            score.time_sampling
        }
        Method::FlowMatching(flow) => {
            w.str("diffusion.method", "flow_matching");
            match flow.path {
                FlowPath::Linear => w.str("flow.path", "linear"),
                FlowPath::Trigonometric => w.str("flow.path", "trigonometric"),
                FlowPath::VariancePreserving { beta_min, beta_max } => {
                    w.str("flow.path", "variance_preserving");
                    w.f64("flow.beta_min", beta_min);
                    w.f64("flow.beta_max", beta_max);
                }
            }
            w.str(
                "flow.solver",
                match flow.solver {
                    OdeSolver::Euler => "euler",
                    OdeSolver::Heun => "heun",
                },
            );
            flow.time_sampling
        }
    };
    match time_sampling {
        TimeSampling::Uniform => w.str("time.sampling", "uniform"),
        TimeSampling::LogNoiseNormal { mean, std } => {
            w.str("time.sampling", "log_noise_normal");
            w.f64("time.mean", mean);
            w.f64("time.std", std);
        }
    }
    w.u64("diffusion.n_steps", model.n_steps as u64);
    w.u64("diffusion.n_features", model.n_features as u64);
    w.u64("diffusion.n_outputs", model.n_outputs as u64);
    w.array(
        "target.mean",
        model.target_mean.iter().copied(),
        f64::to_le_bytes,
    );
    w.array(
        "target.scale",
        model.target_scale.iter().copied(),
        f64::to_le_bytes,
    );
    if let Some(r) = &model.residualizer {
        w.array(
            "residual.center",
            r.center.iter().copied(),
            f64::to_le_bytes,
        );
        w.array("residual.scale", r.scale.iter().copied(), f64::to_le_bytes);
        let mut lengths = Vec::with_capacity(r.models.len());
        let mut models = Vec::new();
        for m in &r.models {
            let bytes = write_container(m)?;
            lengths.push(bytes.len() as u64);
            models.extend_from_slice(&bytes);
        }
        w.array("residual.model_lengths", lengths, u64::to_le_bytes);
        w.raw("residual.models", crate::model::sections::REQUIRED, &models);
    }
    w.raw(
        "regressor.model",
        crate::model::sections::REQUIRED,
        &write_container(&model.regressor)?,
    );
    pack(frame(*MAGIC, VERSION, w))
}

pub(super) fn read(bytes: &[u8]) -> Result<DiffusionModel> {
    let container = unpack(bytes)?;
    let table = section_table(&container, *MAGIC, VERSION, "diffusion model")?;
    let (s, rest) = Sections::parse(table, |name| KNOWN.contains(&name))?;
    if !rest.is_empty() {
        return Err(format_error(format!(
            "{} unexpected bytes after the last section",
            rest.len()
        )));
    }
    let time_sampling = match s.str("time.sampling")? {
        "uniform" => TimeSampling::Uniform,
        "log_noise_normal" => TimeSampling::LogNoiseNormal {
            mean: s.f64("time.mean")?,
            std: s.f64("time.std")?,
        },
        other => return Err(unknown("time.sampling", other)),
    };
    let method = match s.str("diffusion.method")? {
        "score" => {
            let (lo, hi) = (s.f64("score.sde_min")?, s.f64("score.sde_max")?);
            let sde = match s.str("score.sde")? {
                "variance_exploding" => Sde::VarianceExploding {
                    sigma_min: lo,
                    sigma_max: hi,
                },
                "variance_preserving" => Sde::VariancePreserving {
                    beta_min: lo,
                    beta_max: hi,
                },
                "sub_variance_preserving" => Sde::SubVariancePreserving {
                    beta_min: lo,
                    beta_max: hi,
                },
                other => return Err(unknown("score.sde", other)),
            };
            let parameterization = match s.str("score.parameterization")? {
                "noise" => Parameterization::Noise,
                "edm" => Parameterization::Edm {
                    sigma_data: s.f64("score.sigma_data")?,
                },
                other => return Err(unknown("score.parameterization", other)),
            };
            let noise_level_feature = match s.u64("score.noise_level_feature")? {
                0 => false,
                1 => true,
                other => return Err(unknown("score.noise_level_feature", &other.to_string())),
            };
            Method::Score(ScoreConfig {
                sde,
                parameterization,
                noise_level_feature,
                time_sampling,
            })
        }
        "flow_matching" => {
            let path = match s.str("flow.path")? {
                "linear" => FlowPath::Linear,
                "trigonometric" => FlowPath::Trigonometric,
                "variance_preserving" => FlowPath::VariancePreserving {
                    beta_min: s.f64("flow.beta_min")?,
                    beta_max: s.f64("flow.beta_max")?,
                },
                other => return Err(unknown("flow.path", other)),
            };
            let solver = match s.str("flow.solver")? {
                "euler" => OdeSolver::Euler,
                "heun" => OdeSolver::Heun,
                other => return Err(unknown("flow.solver", other)),
            };
            Method::FlowMatching(FlowMatchingConfig {
                path,
                time_sampling,
                solver,
            })
        }
        other => return Err(unknown("diffusion.method", other)),
    };
    let n_outputs = s.usize("diffusion.n_outputs")?;
    let residualizer = if s.has("residual.models") {
        let lengths = s.array("residual.model_lengths", u64::from_le_bytes)?;
        let mut blob = s.bytes("residual.models")?;
        let mut models = Vec::with_capacity(lengths.len());
        for len in lengths {
            let len = usize::try_from(len)
                .ok()
                .filter(|&len| len <= blob.len())
                .ok_or_else(|| format_error("section `residual.model_lengths` is out of range"))?;
            let (model, rest) = blob.split_at(len);
            models.push(BoostedModel::from_bytes(model)?);
            blob = rest;
        }
        if !blob.is_empty() {
            return Err(format_error(
                "section `residual.models` has bytes past its models",
            ));
        }
        Some(FittedResidualizer {
            models,
            center: s.array_exact("residual.center", n_outputs, f64::from_le_bytes)?,
            scale: s.array_exact("residual.scale", n_outputs, f64::from_le_bytes)?,
        })
    } else {
        None
    };
    let model = DiffusionModel {
        method,
        n_steps: s.usize("diffusion.n_steps")?,
        n_features: s.usize("diffusion.n_features")?,
        n_outputs,
        target_mean: s.array_exact("target.mean", n_outputs, f64::from_le_bytes)?,
        target_scale: s.array_exact("target.scale", n_outputs, f64::from_le_bytes)?,
        residualizer,
        regressor: BoostedModel::from_bytes(s.bytes("regressor.model")?)?,
    };
    model.validate()?;
    Ok(model)
}

fn unknown(name: &str, value: &str) -> HessboostError {
    format_error(format!("unknown `{name}` value `{value}`"))
}

/// The serialized fields of a [`DiffusionModel`] (its JSON format), before
/// validation. Every field is required; `residualizer` is `null` without
/// one.
#[derive(Deserialize)]
pub(super) struct UncheckedDiffusionModel {
    method: Method,
    n_steps: usize,
    n_features: usize,
    n_outputs: usize,
    target_mean: Vec<f64>,
    target_scale: Vec<f64>,
    #[serde(deserialize_with = "Option::deserialize")]
    residualizer: Option<UncheckedResidualizer>,
    regressor: BoostedModel,
}

#[derive(Deserialize)]
struct UncheckedResidualizer {
    models: Vec<BoostedModel>,
    center: Vec<f64>,
    scale: Vec<f64>,
}

impl TryFrom<UncheckedDiffusionModel> for DiffusionModel {
    type Error = HessboostError;

    fn try_from(m: UncheckedDiffusionModel) -> Result<Self> {
        let model = DiffusionModel {
            method: m.method,
            n_steps: m.n_steps,
            n_features: m.n_features,
            n_outputs: m.n_outputs,
            target_mean: m.target_mean,
            target_scale: m.target_scale,
            residualizer: m.residualizer.map(|r| FittedResidualizer {
                models: r.models,
                center: r.center,
                scale: r.scale,
            }),
            regressor: m.regressor,
        };
        model.validate()?;
        Ok(model)
    }
}
