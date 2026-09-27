//! Training: the boosting loop ([`train`], [`Trainer`]), cross-validation
//! ([`cv`], or [`CrossValidation`] over caller-supplied or time-ordered
//! [`Fold`]s), and opt-in [`budget`] training.

pub(crate) mod boulevard;
pub mod budget;
mod continuation;
mod cv;
mod ebm;
mod gblinear;
mod multi_output;
pub mod online;
mod refresh;
mod sampling;
mod sglb;
mod train;

pub use cv::{CrossValidation, CvResult, Fold, cv};
pub(crate) use multi_output::reject_split_gradient;
pub use train::{RoundEval, Score, TrainResult, Trainer, train};
