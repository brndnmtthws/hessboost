//! Survival objectives: Cox proportional hazards (`survival:cox`) and the
//! accelerated failure time model (`survival:aft`).
//!
//! Both model log-scale margins and report `exp(margin)`: a hazard ratio for
//! Cox, a survival time for AFT. The arithmetic follows XGBoost 3.4
//! (`objective/regression_obj.cu`, `objective/aft_obj.cu`,
//! `common/survival_util.h`, `common/probability_distribution.h`) operation
//! for operation, including its `f32`/`f64` boundaries.

mod aft;
mod cox;
#[cfg(test)]
mod tests;

pub(crate) use aft::{AftLoss, aft_nloglik};
pub(crate) use cox::Cox;

/// `exp` of every element in `f32` (XGBoost's `std::exp` on a float), used
/// by both survival objectives' prediction transform.
fn exp_transform(preds: &mut [f32]) {
    for p in preds {
        *p = p.exp();
    }
}

/// Row indices sorted by increasing `|label|`, ties in row order (XGBoost
/// `MetaInfo::LabelAbsSort`, a stable sort). Shared with the `cox-nloglik`
/// metric.
pub(crate) fn abs_label_order(labels: &[f32]) -> Vec<usize> {
    crate::metric::stable_argsort(labels.len(), |&a, &b| {
        labels[a].abs().total_cmp(&labels[b].abs())
    })
}
