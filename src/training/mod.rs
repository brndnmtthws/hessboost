//! Training: the boosting loop ([`train`], [`Trainer`]), cross-validation
//! ([`cv`]), and opt-in [`budget`] training.

pub mod budget;
mod continuation;
mod cv;
mod gblinear;
mod multi_output;
mod refresh;
mod sampling;
mod train;

pub use cv::{CvResult, cv};
pub(crate) use multi_output::reject_split_gradient;
pub use train::{RoundEval, TrainResult, Trainer, train};
