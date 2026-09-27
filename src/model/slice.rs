use super::{BoostedModel, Iterations, Shrinkage, TreeWeights};
use crate::error::{HessboostError, Result};
use std::ops::{Range, RangeBounds};
use std::sync::OnceLock;

impl BoostedModel {
    /// A new model holding every `step`-th boosting iteration of
    /// `iterations` (Python's `booster[begin:end:step]`; e.g.
    /// `model.slice(2..8, 3)` keeps iterations 2 and 5, `model.slice(.., 1)`
    /// all of them). Each selected iteration keeps its whole
    /// forest (every output and parallel tree) together with its DART tree
    /// weights; the intercepts, objective and layout carry over unchanged and
    /// no refit happens. As in XGBoost the slice drops `best_iteration`, so
    /// it predicts with all of its iterations.
    ///
    /// A model trained with model shrinkage slices to prefixes only
    /// (`..k` with step 1): the result is the model after `k` iterations,
    /// bit for bit the model the same training run stopped after `k` rounds
    /// returns (see [`Self::predict_margin`]).
    ///
    /// `step` must be at least 1 and the range non-empty and within
    /// [`Self::num_boost_rounds`]. XGBoost 3.4.2 additionally trips an
    /// internal check when `end - begin` is not a multiple of `step`; here
    /// every step selects `ceil((end - begin) / step)` iterations, matching
    /// XGBoost wherever it succeeds.
    ///
    /// # Errors
    ///
    /// [`HessboostError::IncompatibleModel`] (`slice`) for a `gblinear`
    /// model, iterations past [`Self::num_boost_rounds`], or anything but a
    /// prefix of a shrunk model; [`HessboostError::InvalidParameter`]
    /// (`slice`) for `step == 0` or an empty or inverted range.
    pub fn slice(&self, iterations: impl RangeBounds<usize>, step: usize) -> Result<BoostedModel> {
        if self.linear.is_some() {
            return Err(HessboostError::incompatible_model(
                "slice",
                "gblinear models have no boosting iterations to slice",
            ));
        }
        if step == 0 {
            return Err(HessboostError::invalid_param(
                "slice",
                "step must be at least 1",
            ));
        }
        let Range { start: begin, end } =
            self.resolve_iterations(Iterations::from(iterations), "slice")?;
        if begin == end {
            return Err(HessboostError::invalid_param(
                "slice",
                format!("empty slice {begin}..{end} is not allowed"),
            ));
        }
        if let Some(shrinkage) = &self.shrinkage {
            if begin != 0 || step != 1 {
                return Err(HessboostError::incompatible_model(
                    "slice",
                    "a model trained with model shrinkage slices to prefixes only (`..k`, step 1)",
                ));
            }
            return Ok(self.shrunk_prefix(shrinkage, end));
        }
        let per = self.trees_per_iteration();
        let mut trees = Vec::with_capacity((end - begin).div_ceil(step) * per);
        let layers = || {
            (begin..end)
                .step_by(step)
                .map(|it| it * per..(it + 1) * per)
        };
        for layer in layers() {
            trees.extend_from_slice(&self.trees[layer]);
        }
        let tree_weights = self.tree_weights.select(layers());
        Ok(BoostedModel {
            trees,
            base_score: self.base_score.clone(),
            objective: self.objective.clone(),
            max_delta_step: self.max_delta_step,
            num_class: self.num_class,
            n_outputs: self.n_outputs,
            n_targets: self.n_targets,
            n_features: self.n_features,
            best_iteration: None,
            tree_weights,
            num_parallel_tree: self.num_parallel_tree,
            linear: None,
            shrinkage: None,
            // A slice of a Boulevard average is not itself one, and a slice
            // of an EBM drops some of its terms' trees.
            boulevard: None,
            ebm: None,
            compact: OnceLock::new(),
        })
    }

    /// The model after the first `k` iterations of this shrunk model:
    /// their trees, reweighted by the record's first `k` coefficients, and
    /// the intercepts shrunk as far ([`Shrinkage::scaling`]).
    pub(super) fn shrunk_prefix(&self, shrinkage: &Shrinkage, k: usize) -> BoostedModel {
        let per = self.trees_per_iteration();
        let (tree_weights, base_score) = shrinkage.scaling(k, per);
        BoostedModel {
            trees: self.trees[..k * per].to_vec(),
            base_score,
            objective: self.objective.clone(),
            max_delta_step: self.max_delta_step,
            num_class: self.num_class,
            n_outputs: self.n_outputs,
            n_targets: self.n_targets,
            n_features: self.n_features,
            best_iteration: None,
            tree_weights: TreeWeights::from_vec(tree_weights),
            num_parallel_tree: self.num_parallel_tree,
            linear: None,
            shrinkage: Some(shrinkage.truncated(k)),
            boulevard: None,
            ebm: None,
            compact: OnceLock::new(),
        }
    }
}
