//! Native extension module behind the `hessboost` Python package.
//!
//! Everything here is private to the package: the public API is the
//! pure-Python layer in `python/hessboost/`, which converts user inputs
//! (numpy arrays of any dtype and layout, pandas frames, scipy sparse
//! matrices) into the C-contiguous `float32` arrays these functions borrow,
//! and wraps the results. Every computation is hessboost's own; this crate
//! converts arguments and results and releases the GIL around the work.
//!
//! The module declares free-threading support (`gil_used = false`): it has no
//! `unsafe` code and no global mutable state, and every class is immutable
//! (`frozen`). Models and matrices are shared read-only between threads;
//! the Python layer swaps whole objects rather than mutating them.

mod booster;
mod conformal;
mod data;
mod dist;
mod errors;
mod params;
mod train;

use pyo3::prelude::*;

#[pymodule(gil_used = false)]
fn _hessboost(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<data::DMatrix>()?;
    m.add_class::<params::Params>()?;
    m.add_class::<booster::Booster>()?;
    m.add_class::<dist::Distributions>()?;
    m.add_class::<conformal::SplitConformal>()?;
    m.add_class::<conformal::ConformalizedQuantile>()?;
    m.add_function(wrap_pyfunction!(train::train, m)?)?;
    m.add_function(wrap_pyfunction!(train::cv, m)?)?;
    m.add_function(wrap_pyfunction!(train::k_fold, m)?)?;
    m.add_function(wrap_pyfunction!(train::forward_chaining, m)?)?;
    m.add_function(wrap_pyfunction!(train::purged_forward, m)?)?;
    Ok(())
}
