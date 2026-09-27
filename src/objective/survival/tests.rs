use super::aft::*;
use super::cox::*;
use crate::data::MetaInfo;
use crate::objective::{Aft, Objective};
use crate::objective::{AftDistribution, GradPair, Loss, MIN_HESS_F64, gradient_pairs};
use approx::assert_relative_eq;

/// Cox gradients against the Breslow formulas evaluated by hand: rows
/// (sorted by |y|) with times 1 (event), 2 (censored), 2 (event),
/// 3 (event), all margins 0 so every risk is 1. Tied time 2 shares the
/// risk set {2, 2, 3}.
#[test]
fn cox_breslow_gradient_with_tie_and_censoring() {
    let labels = [2.0, 1.0, 3.0, -2.0];
    let out = gradient_pairs(&Cox, &[0.0; 4], &labels, None);
    // Risk-set sizes seen by events: time 1 -> 4, time 2 -> 3, time 3 -> 1.
    let r = [
        1.0 / 4.0,
        1.0 / 4.0 + 1.0 / 3.0,
        1.0 / 4.0 + 1.0 / 3.0 + 1.0,
    ];
    let s = [
        1.0 / 16.0,
        1.0 / 16.0 + 1.0 / 9.0,
        1.0 / 16.0 + 1.0 / 9.0 + 1.0,
    ];
    let expect = |r: f64, s: f64, event: bool| {
        GradPair::new((r - if event { 1.0 } else { 0.0 }) as f32, (r - s) as f32)
    };
    assert_eq!(out[1], expect(r[0], s[0], true)); // t=1 event
    assert_eq!(out[0], expect(r[1], s[1], true)); // t=2 event
    // The censored t=2 row follows the t=2 event in the stable |y| order
    // (it comes later in the input), so its terms include that event.
    assert_eq!(out[3], expect(r[1], s[1], false));
    assert_eq!(out[2], expect(r[2], s[2], true)); // t=3 event
}

#[test]
fn cox_weights_scale_gradients() {
    let labels = [1.0, -2.0, 3.0];
    let preds = [0.3, -0.2, 0.1];
    let plain = gradient_pairs(&Cox, &preds, &labels, None);
    let weighted = gradient_pairs(&Cox, &preds, &labels, Some(&[2.0, 0.5, 1.0]));
    for ((p, w), s) in plain.iter().zip(&weighted).zip([2.0f32, 0.5, 1.0]) {
        assert_relative_eq!(w.grad, p.grad * s, max_relative = 1e-6);
        assert_relative_eq!(w.hess, p.hess * s, max_relative = 1e-6);
    }
}

/// The analytic AFT gradient and Hessian are the derivatives of the loss
/// for every censoring type and distribution (central differences).
#[test]
fn aft_derivatives_match_loss() {
    let rows = [
        (2.0, 2.0),           // uncensored
        (1.5, f64::INFINITY), // right
        (0.0, 3.0),           // left
        (1.0, 4.0),           // interval
    ];
    for dist in [
        AftDistribution::Normal,
        AftDistribution::Logistic,
        AftDistribution::Extreme,
    ] {
        for &(lo, hi) in &rows {
            for pred in [-0.5, 0.4, 1.2] {
                let sigma = 0.8;
                let h = 1e-5;
                let loss = |m: f64| aft_nloglik(dist, lo, hi, m, sigma);
                let (grad, hess) = match dist {
                    AftDistribution::Normal => (
                        aft_grad_hess::<Normal>(lo, hi, pred, sigma).0,
                        aft_grad_hess::<Normal>(lo, hi, pred, sigma).1,
                    ),
                    AftDistribution::Logistic => (
                        aft_grad_hess::<Logistic>(lo, hi, pred, sigma).0,
                        aft_grad_hess::<Logistic>(lo, hi, pred, sigma).1,
                    ),
                    AftDistribution::Extreme => (
                        aft_grad_hess::<Extreme>(lo, hi, pred, sigma).0,
                        aft_grad_hess::<Extreme>(lo, hi, pred, sigma).1,
                    ),
                };
                let fd_grad = (loss(pred + h) - loss(pred - h)) / (2.0 * h);
                let fd_hess = (loss(pred + h) - 2.0 * loss(pred) + loss(pred - h)) / (h * h);
                assert_relative_eq!(grad, fd_grad, epsilon = 1e-6, max_relative = 1e-5);
                assert_relative_eq!(
                    hess.max(MIN_HESS_F64),
                    fd_hess.max(MIN_HESS_F64),
                    epsilon = 1e-4,
                    max_relative = 1e-3
                );
            }
        }
    }
}

/// Far-off predictions take XGBoost's limits instead of NaN.
#[test]
fn aft_extreme_predictions_use_limits() {
    // Uncensored, prediction far below the observed log-time: z >> 0.
    let g = aft_grad_hess::<Normal>(1.0, 1.0, -100.0, 1.0).0;
    assert_eq!(g, MIN_GRADIENT);
    assert_eq!(aft_grad_hess::<Normal>(1.0, 1.0, -100.0, 1.0).1, 1.0);
    // Right-censored with the prediction far above the bound: the loss
    // vanishes and so does the gradient.
    let g = aft_grad_hess::<Logistic>(1.0, f64::INFINITY, 1e3, 1.0).0;
    assert_eq!(g, 0.0);
    assert_eq!(
        aft_grad_hess::<Logistic>(1.0, f64::INFINITY, 1e3, 1.0).1,
        MIN_HESS_F64
    );
    // Interval far below the prediction.
    let g = aft_grad_hess::<Extreme>(1.0, 2.0, 50.0, 1.0).0;
    assert!(g.is_finite());
}

#[test]
fn aft_reads_bounds_and_weights() {
    let obj = AftLoss::new(AftDistribution::Normal, 1.0);
    let lower = [1.0, 2.0, 0.0];
    let upper = [1.0, f32::INFINITY, 3.0];
    let weights = [1.0, 2.0, 0.5];
    let preds = [0.1, 0.2, 0.3];
    let info = MetaInfo {
        n_rows: 3,
        label_lower_bound: Some(&lower),
        label_upper_bound: Some(&upper),
        ..MetaInfo::new(&[], Some(&weights), None)
    };
    obj.validate_info(&info).unwrap();
    let mut out = [GradPair::default(); 3];
    obj.gradient_info(&preds, &info, &mut out);
    for i in 0..3 {
        let (lo, hi, p) = (
            f64::from(lower[i]),
            f64::from(upper[i]),
            f64::from(preds[i]),
        );
        let g = aft_grad_hess::<Normal>(lo, hi, p, 1.0).0 as f32 * weights[i];
        let h = aft_grad_hess::<Normal>(lo, hi, p, 1.0).1 as f32 * weights[i];
        assert_eq!(out[i], GradPair::new(g, h));
    }
    let unbounded = MetaInfo::new(&[1.0], None, None);
    assert!(obj.validate_info(&unbounded).is_err());
    assert_eq!(obj.base_margins_info(&unbounded), vec![0.5f32.ln()]);
}

/// `survival:aft` trains from label bounds alone, starts from margin
/// `ln 0.5`, reports survival times, and refuses an evaluation set
/// without bounds, naming it.
#[test]
fn aft_trains_from_bounds_without_labels() {
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::training::{Trainer, train};

    let n = 60;
    let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let t: Vec<f32> = x.iter().map(|&v| (1.0 + 2.0 * v).exp()).collect();
    let upper: Vec<f32> = t
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 4 == 0 { f32::INFINITY } else { v })
        .collect();
    let d = DMatrix::from_dense(&x, n, 1)
        .unwrap()
        .with_label_bounds(&t, &upper)
        .unwrap();
    let params = TrainingParams::builder()
        .objective(Objective::Aft(Aft::default()))
        .max_depth(2)
        .eta(0.5)
        .build()
        .unwrap();
    let model = train(&params, &d, 20).unwrap();
    assert_eq!(model.base_scores(), &[0.5f32.ln()]);
    // One value per row.
    let pred = model.predict(&d).unwrap().into_vec();
    let margin = model.predict_margin(&d).unwrap().into_vec();
    for (p, m) in pred.iter().zip(&margin) {
        assert_eq!(*p, m.exp());
    }
    // The fit follows the (uncensored) times upward.
    assert!(pred[n - 1] > 3.0 * pred[1]);

    let unbounded = crate::test_support::labeled_dense(&x, n, 1, &t);
    let err = Trainer::new(&params, &d, 1)
        .eval(&unbounded, "valid")
        .train()
        .unwrap_err();
    assert!(err.to_string().contains("`valid`"), "{err}");
}
