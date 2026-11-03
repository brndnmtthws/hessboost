//! Synthetic existing-hook diagnostics, not production recommendations.
//! Run: `RAYON_NUM_THREADS=1 cargo run --release --example research_hooks -- small 3`
//! `large` uses 16,384 training rows; `calibration` compares score trackers.
//! Validation chooses each model's best iteration with patience, independently
//! of timing and test data. A separate pass times exactly that many rounds.
//! Capped searches never qualify as quality-matched comparisons.
//! Projected structures use seeded random coordinates and Newton leaves;
//! sampling weights freeze after a shared warmup, not an online gain EMA.
mod common;

use hessboost::config::{MultiStrategy, TrainingParams, TreeMethod};
use hessboost::conformal::online::{Forecast, OnlineCalibrator, OnlineParams};
use hessboost::data::{BinnedMatrix, DMatrix};
use hessboost::metric::EvalMetric;
use hessboost::model::{BoostedModel, ImportanceType};
use hessboost::objective::{CustomLoss, Objective, Quantiles, RegLoss, SplitGradient};
use hessboost::training::{Trainer, train};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde_json::json;
use std::{collections::VecDeque, error::Error, num::NonZeroUsize, sync::Arc, time::Instant};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

struct Fold {
    x: Vec<f32>,
    y: Vec<f32>,
    features: usize,
}

impl Fold {
    fn matrix(&self) -> Result<DMatrix> {
        Ok(DMatrix::from_dense(&self.x, self.y.len(), self.features)?.with_labels(&self.y)?)
    }
}

fn generate(rows: usize, features: usize, seed: u64, missing_rate: f32) -> Fold {
    let mut rng = common::lcg(seed);
    let mut x = Vec::with_capacity(rows * features);
    let mut y = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut signal = 0.0;
        let mut scale = 0.2;
        for column in 0..features {
            // Unequal-density mixture exercises distortion-based vs rank cuts.
            let u = rng();
            let value = if u < 0.85 {
                u * u
            } else {
                2.0 + 8.0 * (u - 0.85)
            };
            if column < 4 {
                signal += (value * (column + 1) as f32).sin();
                scale += value.abs() * 0.1;
            }
            x.push(if rng() < missing_rate {
                f32::NAN
            } else {
                value
            });
        }
        y.push(signal + scale * common::normal(&mut rng));
    }
    Fold { x, y, features }
}

fn params(objective: Objective, bins: usize, retention: f64, seed: u64) -> Result<TrainingParams> {
    Ok(TrainingParams::builder()
        .objective(objective)
        .tree_method(TreeMethod::Hist)
        .multi_strategy(MultiStrategy::MultiOutputTree)
        .max_depth(4)
        .max_bin(bins)
        .eta(0.1)
        .base_score(0.0)
        .colsample_bytree(retention)
        .seed(seed)
        .build()?)
}

fn quantile_objective(
    levels: &Quantiles,
    variant: &str,
    seed: u64,
    max_rounds: usize,
) -> Result<Objective> {
    let builtin = Objective::Quantile(levels.clone());
    if variant == "vector" {
        return Ok(builtin);
    }
    let loss = params(builtin, 256, 1.0, 0)?.loss(1)?;
    let transform = Arc::clone(&loss);
    let width = levels.alpha().len();
    let mut custom = CustomLoss::new("research:quantiles", width, move |p, y, w, out| {
        loss.gradient(p, y, w, out);
    })
    .with_base_margin(0.0)
    .with_default_metric(EvalMetric::Quantile(levels.clone()))
    .with_transform(move |p| transform.pred_transform(p));
    if variant == "projected_newton" {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x50_47_42);
        let directions: Vec<usize> = (0..max_rounds)
            .map(|_| rng.random_range(0..width))
            .collect();
        custom = custom.with_split_gradient(move |iteration, full| {
            let coordinate = directions[iteration];
            let reduced = full
                .chunks_exact(width)
                .map(|row| row[coordinate])
                .collect();
            Some(SplitGradient::new(reduced, 1))
        });
    }
    Ok(Objective::Custom(Arc::new(custom)))
}

fn score(pred: &[f32], labels: &[f32], levels: &[f64]) -> f64 {
    if levels.is_empty() {
        return (pred
            .iter()
            .zip(labels)
            .map(|(&p, &y)| (f64::from(p) - f64::from(y)).powi(2))
            .sum::<f64>()
            / labels.len() as f64)
            .sqrt();
    }
    pred.chunks_exact(levels.len())
        .zip(labels)
        .map(|(row, &y)| {
            row.iter()
                .zip(levels)
                .map(|(&p, &alpha)| {
                    let residual = f64::from(y) - f64::from(p);
                    residual * if residual >= 0.0 { alpha } else { alpha - 1.0 }
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / (labels.len() * levels.len()) as f64
}

fn hash(pred: &[f32]) -> String {
    format!(
        "{:016x}",
        pred.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, p| {
            (h ^ u64::from(p.to_bits())).wrapping_mul(0x100_0000_01b3)
        })
    )
}

// Lloyd iterations on finite training values only; sorted centers yield ordered
// numeric codes, not categorical features. Empty clusters retain their center.
fn centers(data: &Fold, k: usize) -> Vec<Vec<f32>> {
    (0..data.features)
        .map(|column| {
            let mut values: Vec<f32> = data
                .x
                .chunks_exact(data.features)
                .map(|row| row[column])
                .filter(|v| v.is_finite())
                .collect();
            values.sort_by(f32::total_cmp);
            let mut c: Vec<f32> = (0..k)
                .map(|j| values[(2 * j + 1) * values.len() / (2 * k)])
                .collect();
            for _ in 0..30 {
                let mut sums = vec![0.0f64; k];
                let mut counts = vec![0usize; k];
                for &value in &values {
                    let j = nearest(value, &c);
                    sums[j] += f64::from(value);
                    counts[j] += 1;
                }
                let mut changed = false;
                for j in 0..k {
                    if counts[j] > 0 {
                        let next = (sums[j] / counts[j] as f64) as f32;
                        changed |= next.to_bits() != c[j].to_bits();
                        c[j] = next;
                    }
                }
                c.sort_by(f32::total_cmp);
                if !changed {
                    break;
                }
            }
            c
        })
        .collect()
}

fn nearest(value: f32, centers: &[f32]) -> usize {
    // Voronoi boundaries: a tie belongs to the lower code.
    centers
        .windows(2)
        .position(|w| value <= w[0] + (w[1] - w[0]) * 0.5)
        .unwrap_or(centers.len() - 1)
}

fn encode(data: &Fold, centers: &[Vec<f32>]) -> Result<DMatrix> {
    let values = data
        .x
        .chunks_exact(data.features)
        .flat_map(|row| {
            row.iter()
                .zip(centers)
                .map(|(&v, c)| if v.is_nan() { v } else { nearest(v, c) as f32 })
        })
        .collect();
    Ok(DMatrix::from_dense_vec(values, data.y.len(), data.features)?.with_labels(&data.y)?)
}

struct Case<'a> {
    family: &'a str,
    variant: &'a str,
    seed: u64,
    params: TrainingParams,
    train: DMatrix,
    valid: &'a DMatrix,
    test: &'a DMatrix,
    levels: &'a [f64],
    max_rounds: usize,
    patience: NonZeroUsize,
    preprocessing_seconds: f64,
    warmup: Option<&'a BoostedModel>,
    prepared_cut_seconds: Option<f64>,
}

struct Point {
    rounds: usize,
    validation: f64,
    test: f64,
    fit_seconds: f64,
    margin_hash: String,
    prediction_hash: String,
    stopped: bool,
}

fn fit_rounds(case: &Case<'_>, rounds: usize) -> Result<BoostedModel> {
    if let Some(warmup) = case.warmup {
        let extra = rounds
            .checked_sub(warmup.num_boost_rounds())
            .ok_or("round count shorter than warmup")?;
        if extra == 0 {
            return Ok(warmup.clone());
        }
        Ok(Trainer::new(&case.params, &case.train, extra)
            .init_model(warmup)
            .train()?
            .model)
    } else {
        Ok(train(&case.params, &case.train, rounds)?)
    }
}

fn benchmark(case: &Case<'_>) -> Result<Point> {
    let start = Instant::now();
    let _bins = BinnedMatrix::new(&case.train, case.params.max_bin)?;
    let cut_seconds = case
        .prepared_cut_seconds
        .unwrap_or_else(|| start.elapsed().as_secs_f64());
    let warmup_rounds = case.warmup.map_or(0, BoostedModel::num_boost_rounds);
    let mut trainer = Trainer::new(&case.params, &case.train, case.max_rounds - warmup_rounds)
        .eval(case.valid, "validation")
        .early_stopping_rounds(case.patience);
    if let Some(warmup) = case.warmup {
        trainer = trainer.init_model(warmup);
    }
    let start = Instant::now();
    let tuned = trainer.train()?;
    let tuning_seconds = start.elapsed().as_secs_f64();
    let rounds = tuned
        .model
        .best_iteration()
        .ok_or("early stopping recorded no best round")?
        + 1;
    let evaluated_rounds = warmup_rounds + tuned.history.len();
    let stopped = evaluated_rounds >= rounds + case.patience.get();
    let chosen_validation_score = tuned.best_score.ok_or("missing validation score")?;
    drop(tuned);

    // Timing cannot choose rounds. Retrain the validation-selected count
    // without evaluation and report all repetitions, not the fastest one.
    let mut growth_samples = Vec::with_capacity(3);
    let mut model = None;
    for _ in 0..3 {
        let start = Instant::now();
        let fit = fit_rounds(case, rounds)?;
        growth_samples.push(start.elapsed().as_secs_f64());
        model = Some(fit);
    }
    let mut sorted_times = growth_samples.clone();
    sorted_times.sort_by(f64::total_cmp);
    let growth_seconds = sorted_times[1];
    let model = model.ok_or("timing produced no model")?;
    let mut validation_prediction = model.predict_margin(case.valid, ..)?.into_vec();
    let mut prediction = model.predict_margin(case.test, ..)?.into_vec();
    let margin_hash = hash(&prediction);
    if !case.levels.is_empty() {
        let transform = params(
            Objective::Quantile(Quantiles::new(case.levels.iter().copied())?),
            case.params.max_bin,
            1.0,
            case.seed,
        )?
        .loss(1)?;
        transform.pred_transform(&mut validation_prediction);
        transform.pred_transform(&mut prediction);
    }
    let validation = score(
        &validation_prediction,
        case.valid.labels().ok_or("missing labels")?,
        case.levels,
    );
    let test = score(
        &prediction,
        case.test.labels().ok_or("missing labels")?,
        case.levels,
    );
    let prediction_hash = hash(&prediction);
    let fit_seconds = case.preprocessing_seconds + cut_seconds + growth_seconds;
    println!(
        "{}",
        json!({
            "kind":"benchmark", "family":case.family,"variant":case.variant,"seed":case.seed,
            "rows":case.train.n_rows(),"features":case.train.n_cols(),"outputs":case.levels.len().max(1),
            "selected_rounds":rounds,"evaluated_rounds":evaluated_rounds,"max_rounds":case.max_rounds,
            "patience":case.patience.get(),"stopped":stopped,"validation_selection_score":chosen_validation_score,
            "max_bin":case.params.max_bin,"retention":case.params.colsample_bytree,
            "validation_error":validation,"test_error":test,
            "metric":if case.levels.is_empty() {"rmse"} else {"mean_pinball"},
            "validation_wis":(!case.levels.is_empty()).then_some(2.0*validation),
            "test_wis":(!case.levels.is_empty()).then_some(2.0*test),
            "tuning_seconds":tuning_seconds,"preprocessing_seconds":case.preprocessing_seconds,
            "cold_cut_seconds":cut_seconds,"warm_growth_seconds":growth_seconds,
            "growth_samples_seconds":growth_samples,"cold_fit_seconds":fit_seconds,
            "margin_hash":margin_hash,"prediction_hash":prediction_hash,"warmup_rounds":warmup_rounds,
            "missing_rate":if case.family=="sampling" {0.0} else {0.02}
        })
    );
    Ok(Point {
        rounds,
        validation,
        test,
        fit_seconds,
        margin_hash,
        prediction_hash,
        stopped,
    })
}

fn compare(family: &str, seed: u64, outputs: usize, variants: &[(&str, Point)]) {
    let baseline_name = if family == "quantile" {
        "vector"
    } else if family == "cuts" {
        "quantile256"
    } else {
        "all"
    };
    let baseline = &variants
        .iter()
        .find(|(name, _)| *name == baseline_name)
        .unwrap()
        .1;
    for (variant, point) in variants {
        let matched =
            baseline.stopped && point.stopped && point.validation <= baseline.validation * 1.01;
        println!(
            "{}",
            json!({
                "kind":"comparison","family":family,"variant":variant,"seed":seed,"outputs":outputs,
                "baseline":baseline_name,"baseline_stopped":baseline.stopped,"stopped":point.stopped,
                "quality_matched":matched,"relative_tolerance":0.01,"selected_rounds":point.rounds,
                "validation_error":point.validation,"test_error":point.test,"cold_fit_seconds":point.fit_seconds,
                "relative_test_error":point.test/baseline.test-1.0,
                "descriptive_speed_ratio":matched.then_some(baseline.fit_seconds/point.fit_seconds),
                "timing_caveat":"No host exclusivity claimed; do not promote on shared-host timings"
            })
        );
    }
}

fn score_quantile(values: &VecDeque<f64>, alpha: f64) -> f64 {
    let rank = (((values.len() + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    if rank > values.len() {
        return f64::INFINITY;
    }
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    sorted[rank - 1]
}

fn calibration_replay(repeats: usize) -> Result<()> {
    const ALPHA: f64 = 0.1;
    const WARM: usize = 1000;
    for repeat in 0..repeats {
        let seed = 513 + repeat as u64 * 1009;
        let mut rng = common::lcg(seed);
        let mut level = 1.0;
        let scores: Vec<f64> = (0..6000)
            .map(|t| {
                let regime = if t < 3500 { 0.7 } else { 1.7 };
                level = 0.95 * level + 0.05 * regime + 0.04 * f64::from(rng()) - 0.02;
                (level + 0.3 * f64::from(rng()) - 0.15).max(0.0)
            })
            .collect();
        let initial = score_quantile(&scores[..WARM].iter().copied().collect(), ALPHA);
        for delay in [0, 5] {
            for (name, window, lags) in [
                ("fixed", 0, None),
                ("scalar", 0, Some(0)),
                ("linear", 0, Some(1)),
                ("rolling50", 50, None),
                ("rolling200", 200, None),
            ] {
                let mut tracker = lags
                    .map(|lags| {
                        OnlineCalibrator::new(
                            OnlineParams::builder(ALPHA, 0.01)
                                .initial_threshold(initial)
                                .lags(lags)
                                .build()?,
                        )
                    })
                    .transpose()?;
                let mut observed: VecDeque<f64> =
                    scores[WARM - window..WARM].iter().copied().collect();
                let mut pending: VecDeque<(Option<Forecast>, f64)> = VecDeque::new();
                let (mut misses, mut width, mut loss) = (0usize, 0.0, 0.0);
                for &score in &scores[WARM..] {
                    let receipt = tracker.as_mut().map(|t| t.predict(&[])).transpose()?;
                    let q = receipt.as_ref().map_or_else(
                        || {
                            if window == 0 {
                                initial
                            } else {
                                score_quantile(&observed, ALPHA)
                            }
                        },
                        Forecast::threshold,
                    );
                    misses += usize::from(score > q);
                    width += 2.0 * q.max(0.0);
                    let residual = score - q;
                    loss += residual * if residual >= 0.0 { 1.0 - ALPHA } else { -ALPHA };
                    pending.push_back((receipt, score));
                    if pending.len() > delay {
                        let (receipt, value) = pending.pop_front().unwrap();
                        if let (Some(tracker), Some(receipt)) = (&mut tracker, receipt) {
                            tracker.update(&receipt, value)?;
                        }
                        if window > 0 {
                            observed.pop_front();
                            observed.push_back(value);
                        }
                    }
                }
                let n = (scores.len() - WARM) as f64;
                println!(
                    "{}",
                    json!({"kind":"calibration","seed":seed,"variant":name,"delay":delay,
                    "window":window,"coverage":1.0-misses as f64/n,"mean_width":width/n,"score_pinball":loss/n,
                    "protocol":"same original scores; observe only after prediction; parameters preselected, no tuning; final delayed labels not used"})
                );
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map_or("small", String::as_str);
    let repeats: usize = args.get(1).map_or(Ok(3), |s| s.parse())?;
    if !matches!(mode, "small" | "large" | "calibration") || repeats == 0 || args.len() > 2 {
        return Err(
            "usage: research_hooks [small|large|calibration] [positive_repetitions]".into(),
        );
    }
    if mode == "calibration" {
        return calibration_replay(repeats);
    }
    let n = if mode == "small" { 1024 } else { 16_384 };
    let max_rounds = if mode == "small" { 1024 } else { 4096 };
    let patience = NonZeroUsize::new(50).unwrap();
    println!(
        "{}",
        json!({"kind":"protocol","mode":mode,"repetitions":repeats,
        "threads":rayon::current_num_threads(),"train_rows":n,"validation_rows":n/2,"test_rows":n/2,
        "max_rounds":max_rounds,"patience":patience.get(),"data":"synthetic unequal-density heteroskedastic sparse-signal",
        "limitations":"No market dataset, no host exclusivity, no production promotion; raw seeded repetitions",
        "timing":"selection pass reported separately; 3 fixed-count refits after selection; cold cuts plus warm growth plus preprocessing; no concurrent checks allowed",
        "selection":"per-variant validation early stopping; timing and test data never select rounds; capped searches cannot quality-match",
        "projection":"seeded uniform coordinate from full quantile range each round, full-output Newton leaves, not exact-leaf PGB",
        "sampling":"shared 4-round all-feature warmup retained in every model; frozen normalized total-gain weights plus 0.05 floor, not EMA",
        "sampling_missing_rate":0.0,"other_missing_rate":0.02})
    );
    for repetition in 0..repeats {
        let seed = 42 + repetition as u64 * 1009;
        let training = generate(n, 32, seed, 0.02);
        let validation = generate(n / 2, 32, seed + 1, 0.02);
        let testing = generate(n / 2, 32, seed + 2, 0.02);
        let valid = validation.matrix()?;
        let test = testing.matrix()?;
        for width in [8, 32, 128] {
            let levels = Quantiles::new((0..width).map(|j| (j as f64 + 0.5) / width as f64))?;
            let mut variants = Vec::new();
            let mut order = ["vector", "custom_all", "projected_newton"];
            order.rotate_left(repetition % 3);
            for variant in order {
                let point = benchmark(&Case {
                    family: "quantile",
                    variant,
                    seed,
                    params: params(
                        quantile_objective(&levels, variant, seed, max_rounds)?,
                        256,
                        1.0,
                        seed,
                    )?,
                    train: training.matrix()?,
                    valid: &valid,
                    test: &test,
                    levels: levels.alpha(),
                    max_rounds,
                    patience,
                    preprocessing_seconds: 0.0,
                    warmup: None,
                    prepared_cut_seconds: None,
                })?;
                variants.push((variant, point));
            }
            let builtin = &variants
                .iter()
                .find(|(name, _)| *name == "vector")
                .unwrap()
                .1;
            let custom = &variants
                .iter()
                .find(|(name, _)| *name == "custom_all")
                .unwrap()
                .1;
            if builtin.rounds != custom.rounds
                || builtin.margin_hash != custom.margin_hash
                || builtin.prediction_hash != custom.prediction_hash
            {
                return Err("built-in/custom all-output quantile controls differ".into());
            }
            compare("quantile", seed, width, &variants);
        }
        let square = || Objective::SquaredError(RegLoss::default());
        let mut variants = Vec::new();
        let mut order = [("quantile256", 256), ("quantile16", 16), ("kmeans16", 16)];
        order.rotate_left(repetition % 3);
        for (variant, bins) in order {
            let start = Instant::now();
            let (data, v, t, preprocessing) = if variant == "kmeans16" {
                let c = centers(&training, bins);
                let data = encode(&training, &c)?;
                let v = encode(&validation, &c)?;
                let t = encode(&testing, &c)?;
                (data, v, t, start.elapsed().as_secs_f64())
            } else {
                (training.matrix()?, valid.clone(), test.clone(), 0.0)
            };
            variants.push((
                variant,
                benchmark(&Case {
                    family: "cuts",
                    variant,
                    seed,
                    params: params(square(), bins, 1.0, seed)?,
                    train: data,
                    valid: &v,
                    test: &t,
                    levels: &[],
                    max_rounds,
                    patience,
                    preprocessing_seconds: preprocessing,
                    warmup: None,
                    prepared_cut_seconds: None,
                })?,
            ));
        }
        compare("cuts", seed, 1, &variants);
        // Dense primary sampling scenario exercises the selected-column path.
        let features = if mode == "small" { 128 } else { 500 };
        let training = generate(n, features, seed, 0.0);
        let valid = generate(n / 2, features, seed + 1, 0.0).matrix()?;
        let test = generate(n / 2, features, seed + 2, 0.0).matrix()?;
        let pilot_data = training.matrix()?;
        let start = Instant::now();
        let _pilot_bins = BinnedMatrix::new(&pilot_data, 256)?;
        let pilot_cut_seconds = start.elapsed().as_secs_f64();
        let pilot_params = params(square(), 256, 1.0, seed)?;
        let start = Instant::now();
        let pilot = train(&pilot_params, &pilot_data, 4)?;
        let gains = pilot.feature_importance(ImportanceType::TotalGain);
        let total = gains.values().sum::<f64>().max(f64::MIN_POSITIVE);
        let weights: Vec<f32> = (0..training.features)
            .map(|j| {
                (0.05 + training.features as f64 * gains.get(&j).copied().unwrap_or(0.0) / total)
                    as f32
            })
            .collect();
        let pilot_seconds = start.elapsed().as_secs_f64();
        let mut variants = Vec::new();
        let mut order = [("all", 1.0), ("uniform", 0.5), ("fixed_gain", 0.5)];
        order.rotate_left(repetition % 3);
        for (variant, retention) in order {
            let mut data = pilot_data.clone();
            if variant == "fixed_gain" {
                data = data.with_feature_weights(&weights)?;
            }
            variants.push((
                variant,
                benchmark(&Case {
                    family: "sampling",
                    variant,
                    seed,
                    params: params(square(), 256, retention, seed)?,
                    train: data,
                    valid: &valid,
                    test: &test,
                    levels: &[],
                    max_rounds,
                    patience,
                    preprocessing_seconds: pilot_seconds,
                    warmup: Some(&pilot),
                    prepared_cut_seconds: Some(pilot_cut_seconds),
                })?,
            ));
        }
        compare("sampling", seed, 1, &variants);
    }
    Ok(())
}
