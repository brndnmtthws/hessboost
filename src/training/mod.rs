//! Training: the boosting loop ([`train`], [`Trainer`]), cross-validation
//! ([`cv`], or [`CrossValidation`] over caller-supplied or time-ordered
//! [`Fold`]s), and opt-in [`budget`] training.

mod api;
pub(crate) mod boulevard;
pub mod budget;
mod continuation;
mod cv;
mod dart;
mod ebm;
mod eval;
mod gblinear;
mod margins;
mod multi_output;
pub mod online;
mod prepare;
mod refresh;
mod round;
mod row_sampling;
mod sampling;
mod sglb;
mod train;
mod validate;

pub use api::{EvalHistory, RoundEval, TrainResult, Trainer, train};
pub use cv::{CrossValidation, CvResult, CvRound, Fold, cv};
pub(crate) use multi_output::reject_split_gradient;
