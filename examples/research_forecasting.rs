//! Real hourly forecasting evidence; no production projected-tree option.
//! Run with `RAYON_NUM_THREADS=1` and prepared JSON, quantile count, and seed.
use hessboost::config::{MultiStrategy, TrainingParams, TreeMethod};
use hessboost::conformal::online::{Forecast, OnlineCalibrator, OnlineParams};
use hessboost::data::{BinnedMatrix, DMatrix};
use hessboost::metric::EvalMetric;
use hessboost::model::{BoostedModel, ModelFormat};
use hessboost::objective::{CustomLoss, Objective, Quantiles, SplitGradient};
use hessboost::training::{Trainer, train};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::VecDeque, error::Error, hint::black_box, num::NonZeroUsize, sync::Arc,
    time::Instant,
};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;
const CAP: usize = 4096;
const PATIENCE: usize = 50;
const ALPHA: f64 = 0.1;
const VARIANTS: [&str; 4] = ["scalar", "vector", "custom_all", "projected_newton"];

#[derive(Deserialize)]
struct Split {
    x: Vec<f32>,
    y: Vec<f32>,
    decision_at: Vec<i64>,
    label_available_at: Vec<i64>,
}
impl Split {
    fn validate(&self, columns: usize, horizon: i64) -> Result<()> {
        let n = self.y.len();
        if n == 0
            || n.checked_mul(columns) != Some(self.x.len())
            || self.decision_at.len() != n
            || self.label_available_at.len() != n
            || self.x.iter().chain(&self.y).any(|v| !v.is_finite())
            || self.decision_at.windows(2).any(|w| w[0] >= w[1])
            || self
                .decision_at
                .iter()
                .zip(&self.label_available_at)
                .any(|(&t, &a)| t.checked_add(horizon * 3600) != Some(a))
        {
            return Err("malformed split shapes, timestamps, or finite values".into());
        }
        Ok(())
    }
    fn matrix(&self, columns: usize) -> Result<DMatrix> {
        Ok(DMatrix::from_dense(&self.x, self.y.len(), columns)?.with_labels(&self.y)?)
    }
    fn metadata(&self) -> Value {
        json!({"rows":self.y.len(),"first_decision":self.decision_at[0],
            "last_decision":self.decision_at.last(),"last_label_available":self.label_available_at.last()})
    }
}
#[derive(Deserialize)]
struct Splits {
    train: Split,
    validation: Split,
    calibration: Split,
    test: Split,
}
#[derive(Deserialize)]
struct Document {
    dataset: String,
    horizon_hours: i64,
    source_url: String,
    source_sha256: String,
    license: String,
    citation: String,
    feature_names: Vec<String>,
    label_scale: f64,
    splits: Splits,
}
impl Document {
    fn validate(&self) -> Result<()> {
        if !matches!(self.dataset.as_str(), "bike" | "beijing")
            || !matches!(self.horizon_hours, 1 | 6)
            || self.feature_names.is_empty()
            || !self.label_scale.is_finite()
            || self.label_scale <= 0.0
        {
            return Err("invalid dataset, horizon, features, or label scale".into());
        }
        let splits = [
            &self.splits.train,
            &self.splits.validation,
            &self.splits.calibration,
            &self.splits.test,
        ];
        for split in splits {
            split.validate(self.feature_names.len(), self.horizon_hours)?;
        }
        for pair in splits.windows(2) {
            if pair[0]
                .label_available_at
                .iter()
                .any(|&t| t >= pair[1].decision_at[0])
            {
                return Err("split boundary not strictly purged by label availability".into());
            }
        }
        Ok(())
    }
}
fn params(objective: Objective, scalar: bool, seed: u64) -> Result<TrainingParams> {
    Ok(TrainingParams::builder()
        .objective(objective)
        .tree_method(TreeMethod::Hist)
        .multi_strategy(if scalar {
            MultiStrategy::OneOutputPerTree
        } else {
            MultiStrategy::MultiOutputTree
        })
        .max_depth(4)
        .max_bin(256)
        .eta(0.1)
        .base_score(0.0)
        .seed(seed)
        .build()?)
}
fn objective(levels: &Quantiles, variant: &str, seed: u64) -> Result<Objective> {
    let builtin = Objective::Quantile(levels.clone());
    if matches!(variant, "scalar" | "vector") {
        return Ok(builtin);
    }
    let loss = params(builtin, false, seed)?.loss(1)?;
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
        let coordinates: Vec<usize> = (0..CAP).map(|_| rng.random_range(0..width)).collect();
        custom = custom.with_split_gradient(move |iteration, full| {
            Some(SplitGradient::new(
                full.chunks_exact(width)
                    .map(|row| row[coordinates[iteration]])
                    .collect(),
                1,
            ))
        });
    }
    Ok(Objective::Custom(Arc::new(custom)))
}
fn pinball(residual: f64, level: f64) -> f64 {
    residual * if residual >= 0.0 { level } else { level - 1.0 }
}
fn score(pred: &[f32], labels: &[f32], levels: &[f64]) -> f64 {
    pred.chunks_exact(levels.len())
        .zip(labels)
        .map(|(row, &y)| {
            row.iter()
                .zip(levels)
                .map(|(&p, &a)| pinball(f64::from(y) - f64::from(p), a))
                .sum::<f64>()
        })
        .sum::<f64>()
        / pred.len() as f64
}
fn hash(pred: &[f32]) -> String {
    format!(
        "{:016x}",
        pred.iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, p| (h ^ u64::from(
                p.to_bits()
            ))
            .wrapping_mul(0x100_0000_01b3))
    )
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}
struct Candidate {
    name: &'static str,
    params: TrainingParams,
    rounds: usize,
    capped: bool,
    growth: Vec<f64>,
    model: Option<BoostedModel>,
    raw: Vec<f32>,
    validation_raw: Vec<f32>,
}
struct Inference<'a> {
    model: &'a BoostedModel,
    matrix: &'a DMatrix,
    split: &'a Split,
    columns: usize,
    levels: &'a Quantiles,
    seed: u64,
}
fn inference(case: &Inference<'_>) -> Result<(Value, Vec<f32>)> {
    let transform = params(Objective::Quantile(case.levels.clone()), false, case.seed)?.loss(1)?;
    let raw = case.model.predict_margin(case.matrix, ..)?.into_vec();
    let mut prediction = raw.clone();
    transform.pred_transform(&mut prediction);
    let width = case.levels.alpha().len();
    let mut single = vec![0.0; prediction.len()];
    for (row, out) in case
        .split
        .x
        .chunks_exact(case.columns)
        .zip(single.chunks_exact_mut(width))
    {
        case.model.predict_margin_row_into(row, .., out)?;
    }
    if !same_bits(&single, &raw) {
        return Err("batch/single raw prediction bits differ".into());
    }
    transform.pred_transform(&mut single);
    if !same_bits(&single, &prediction) {
        return Err("batch/single transformed prediction bits differ".into());
    }
    let mut batch_times = Vec::new();
    let mut row_times = Vec::new();
    let mut batch_repeats = Vec::new();
    let mut row_repeats = Vec::new();
    let mut out = vec![0.0; width];
    // Both lazy prediction layouts were warmed above; each sample lasts >=100ms.
    for _ in 0..3 {
        let start = Instant::now();
        let mut repeats = 0usize;
        loop {
            let mut p = case.model.predict_margin(case.matrix, ..)?.into_vec();
            transform.pred_transform(&mut p);
            black_box(&p);
            repeats += 1;
            if start.elapsed().as_secs_f64() >= 0.1 {
                break;
            }
        }
        batch_times.push(start.elapsed().as_secs_f64() / repeats as f64);
        batch_repeats.push(repeats);
        let start = Instant::now();
        let mut repeats = 0usize;
        loop {
            for row in case.split.x.chunks_exact(case.columns) {
                case.model.predict_margin_row_into(row, .., &mut out)?;
                transform.pred_transform(&mut out);
                black_box(&out);
            }
            repeats += 1;
            if start.elapsed().as_secs_f64() >= 0.1 {
                break;
            }
        }
        row_times.push(start.elapsed().as_secs_f64() / (repeats * case.split.y.len()) as f64);
        row_repeats.push(repeats);
    }
    Ok((
        json!({"batch_us":median(&batch_times)*1e6,
        "batch_ns_per_row":median(&batch_times)*1e9/case.split.y.len() as f64,
        "single_ns_per_row":median(&row_times)*1e9,"batch_samples_seconds":batch_times,
        "single_samples_seconds_per_row":row_times,"batch_repeats":batch_repeats,
        "single_full_pass_repeats":row_repeats,"rows":case.split.y.len(),
        "margin_hash":hash(&raw),"prediction_hash":hash(&prediction),"single_prediction_hash":hash(&single),
        "batch_single_bits_equal":true,"batch_includes_output_allocation":true,
        "single_per_row_allocation":false,"transform":"common_builtin_quantile"}),
        prediction,
    ))
}

struct Event {
    decision: i64,
    available: i64,
    lower: f64,
    upper: f64,
    label: f64,
}
impl Event {
    fn score(&self) -> f64 {
        (self.lower - self.label).max(self.label - self.upper)
    }
}
fn events(split: &Split, predictions: &[f32], width: usize) -> Vec<Event> {
    predictions
        .chunks_exact(width)
        .enumerate()
        .map(|(i, row)| Event {
            decision: split.decision_at[i],
            available: split.label_available_at[i],
            lower: f64::from(row[0]),
            upper: f64::from(row[width - 1]),
            label: f64::from(split.y[i]),
        })
        .collect()
}
fn correction(scores: impl Iterator<Item = f64>) -> Result<f64> {
    let mut sorted: Vec<f64> = scores.collect();
    sorted.sort_by(f64::total_cmp);
    let rank = ((sorted.len() + 1) as f64 * (1.0 - ALPHA)).ceil() as usize;
    if rank == 0 || rank > sorted.len() {
        return Err("insufficient calibration scores for finite corrected quantile".into());
    }
    Ok(sorted[rank - 1])
}
#[derive(Default)]
struct Stats {
    n: usize,
    covered: usize,
    lower: usize,
    upper: usize,
    empty: usize,
    width: f64,
    interval: f64,
    loss: f64,
    q: f64,
}
impl Stats {
    fn observe(&mut self, e: &Event, q: f64) {
        let lo = e.lower - q;
        let hi = e.upper + q;
        let width = (hi - lo).max(0.0);
        self.n += 1;
        self.covered += usize::from(lo <= e.label && e.label <= hi);
        self.lower += usize::from(e.label < lo);
        self.upper += usize::from(e.label > hi);
        self.empty += usize::from(lo > hi);
        self.width += width;
        // For an empty interval both endpoint penalties can be positive; report
        // this explicit extension separately from calibrated-score pinball.
        self.interval += width + 2.0 / ALPHA * ((lo - e.label).max(0.0) + (e.label - hi).max(0.0));
        self.loss += pinball(e.score() - q, 1.0 - ALPHA);
        self.q += q;
    }
    fn json(&self) -> Value {
        let n = self.n as f64;
        json!({"rows":self.n,"coverage":self.covered as f64/n,"mean_width":self.width/n,
            "interval_score":self.interval/n,"lower_misses":self.lower,"upper_misses":self.upper,
            "empty_intervals":self.empty,"score_pinball":self.loss/n,"mean_q":self.q/n})
    }
}
struct Replay<'a> {
    initialization: &'a [Event],
    stream: &'a [Event],
    measure_from: usize,
    initial: f64,
    rate: Option<f64>,
    lags: usize,
    window: usize,
}
fn replay(case: &Replay<'_>) -> Result<(Stats, usize, Option<u64>)> {
    let mut tracker = case
        .rate
        .map(|rate| -> Result<OnlineCalibrator> {
            Ok(OnlineCalibrator::new(
                OnlineParams::builder(ALPHA, rate)
                    .initial_threshold(case.initial)
                    .lags(case.lags)
                    .covariates(0)
                    .coefficient_bound(1.0)
                    .build()?,
            )?)
        })
        .transpose()?;
    let mut pending: VecDeque<(usize, Option<Forecast>)> = VecDeque::new();
    let mut observed: VecDeque<f64> = case
        .initialization
        .iter()
        .filter(|e| e.available <= case.stream[0].decision)
        .map(Event::score)
        .collect();
    if case.window > 0 {
        while observed.len() > case.window {
            observed.pop_front();
        }
    }
    // Initialization labels delayed beyond the first stream decision are not
    // used for fixed initialization; rolling admits them when they arrive.
    let mut initial_pending: VecDeque<&Event> = case
        .initialization
        .iter()
        .filter(|e| e.available > case.stream[0].decision)
        .collect();
    let mut stats = Stats::default();
    for (i, event) in case.stream.iter().enumerate() {
        while initial_pending
            .front()
            .is_some_and(|e| e.available <= event.decision)
        {
            let e = initial_pending.pop_front().unwrap();
            if case.window > 0 {
                observed.push_back(e.score());
            }
        }
        while pending
            .front()
            .is_some_and(|(j, _)| case.stream[*j].available <= event.decision)
        {
            let (j, receipt) = pending.pop_front().unwrap();
            let original_score = case.stream[j].score();
            if let (Some(cal), Some(receipt)) = (&mut tracker, receipt) {
                cal.update(&receipt, original_score)?;
            }
            if case.window > 0 {
                observed.push_back(original_score);
            }
        }
        if case.window > 0 {
            while observed.len() > case.window {
                observed.pop_front();
            }
        }
        let receipt = tracker.as_mut().map(|cal| cal.predict(&[])).transpose()?;
        let q = if let Some(receipt) = &receipt {
            receipt.threshold()
        } else if case.window > 0 {
            correction(observed.iter().copied())?
        } else {
            case.initial
        };
        if i >= case.measure_from {
            stats.observe(event, q);
        }
        pending.push_back((i, receipt));
    }
    Ok((
        stats,
        pending.len() + initial_pending.len(),
        tracker.as_ref().map(OnlineCalibrator::n_updates),
    ))
}
fn calibration(variant: &str, calibration: &[Event], test: &[Event]) -> Result<()> {
    let half = calibration.len() / 2;
    if half == 0 {
        return Err("empty calibration tuning half".into());
    }
    let initial = correction(
        calibration[..half]
            .iter()
            .filter(|e| e.available <= calibration[half].decision)
            .map(Event::score),
    )?;
    let fixed = correction(
        calibration
            .iter()
            .filter(|e| e.available <= test[0].decision)
            .map(Event::score),
    )?;
    for window in [0, 50, 200] {
        let (stats, pending, _) = replay(&Replay {
            initialization: calibration,
            stream: test,
            measure_from: 0,
            initial: fixed,
            rate: None,
            lags: 0,
            window,
        })?;
        println!(
            "{}",
            json!({"kind":"calibration","variant":variant,
            "method":if window==0 {"fixed"} else {"rolling"},"window":window,"alpha":ALPHA,
            "fixed_q":fixed,"statistics":stats.json(),"outstanding_labels":pending,
            "calibration_labels_known_at_first_test":calibration.iter().filter(|e|e.available<=test[0].decision).count()})
        );
    }
    let mut stream = Vec::with_capacity(calibration.len() - half + test.len());
    // References would suffice for a generic scheduler; keeping one immutable
    // event sequence makes issue indexes unambiguous across the split boundary.
    for e in calibration[half..].iter().chain(test) {
        stream.push(Event {
            decision: e.decision,
            available: e.available,
            lower: e.lower,
            upper: e.upper,
            label: e.label,
        });
    }
    for lags in [0, 1] {
        let mut choices = Vec::new();
        for rate in [0.005, 0.02, 0.1] {
            let (stats, _, _) = replay(&Replay {
                initialization: &calibration[..half],
                stream: &calibration[half..],
                measure_from: 0,
                initial,
                rate: Some(rate),
                lags,
                window: 0,
            })?;
            choices.push((rate, stats.loss / stats.n as f64));
        }
        let rate = choices.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
        let (stats, pending, n_updates) = replay(&Replay {
            initialization: &calibration[..half],
            stream: &stream,
            measure_from: calibration.len() - half,
            initial,
            rate: Some(rate),
            lags,
            window: 0,
        })?;
        println!(
            "{}",
            json!({"kind":"calibration","variant":variant,"method":"online",
            "alpha":ALPHA,"lags":lags,"covariates":0,"coefficient_bound":1.0,"rate":rate,
            "initial_q":initial,"rate_candidates_and_calibration_pinball":choices,
            "initialization_rows_known":calibration[..half].iter().filter(|e|e.available<=calibration[half].decision).count(),
            "rate_selection_rows":calibration.len()-half,"statistics":stats.json(),"outstanding_labels":pending,"n_updates":n_updates,
            "replay":"independent state per rate; selected rate replayed calibration second half through test; original receipts; availability <= decision"})
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: research_forecasting <prepared.json> <8|32|128> <17|29|43>".into());
    }
    let width: usize = args[1].parse()?;
    let seed: u64 = args[2].parse()?;
    if !matches!(width, 8 | 32 | 128) || !matches!(seed, 17 | 29 | 43) {
        return Err("quantile count or seed outside preregistered grid".into());
    }
    if rayon::current_num_threads() != 1 {
        return Err("set RAYON_NUM_THREADS=1 before starting benchmark".into());
    }
    let document: Document =
        serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(&args[0])?))?;
    document.validate()?;
    let levels = Quantiles::new((0..width).map(|i| 0.05 + 0.9 * i as f64 / (width - 1) as f64))?;
    let splits = &document.splits;
    let columns = document.feature_names.len();
    let start = Instant::now();
    let training = splits.train.matrix(columns)?;
    let preparation = start.elapsed().as_secs_f64();
    let validation = splits.validation.matrix(columns)?;
    let testing = splits.test.matrix(columns)?;
    let calibrating = splits.calibration.matrix(columns)?;
    let start = Instant::now();
    let _bin_keeper = BinnedMatrix::new(&training, 256)?;
    let cold_cut = start.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"kind":"protocol","dataset":document.dataset,"horizon_hours":document.horizon_hours,
        "source_url":document.source_url,"source_sha256":document.source_sha256,"license":document.license,
        "citation":document.citation,"label_scale":document.label_scale,"feature_names":document.feature_names,
        "splits":{"train":splits.train.metadata(),"validation":splits.validation.metadata(),
        "calibration":splits.calibration.metadata(),"test":splits.test.metadata()},
        "seed":seed,"quantile_count":width,"quantiles":levels.alpha(),"threads":rayon::current_num_threads(),
        "max_rounds":CAP,"patience":PATIENCE,"max_bin":256,"max_depth":4,"eta":0.1,"base_score":0,
        "training_preparation_seconds":preparation,"cold_cut_seconds":cold_cut,
        "selection":"validation only; best prefix; cap disqualifies quality-match claim",
        "timing":"3 refits, repeat outer and rotated variants inner; shared cached bin index; tuning excluded",
        "prediction":"raw margins plus common builtin quantile transform; batch allocation included; row buffers reused",
        "projection":"seeded uniform coordinate across all outputs, full-output Newton leaves",
        "calibration":"scaled units; alpha .1; original band scores; labels only available <= decision; outstanding final labels not drained",
        "empty_interval_score":"width max(0,hi-lo) plus both endpoint penalties; score pinball also reported",
        "limitations":"public hourly domains, not market profit; no exchangeability or delayed coverage guarantee; no production promotion"})
    );
    let mut candidates = Vec::new();
    for name in VARIANTS {
        let p = params(objective(&levels, name, seed)?, name == "scalar", seed)?;
        let start = Instant::now();
        let tuned = Trainer::new(&p, &training, CAP)
            .eval(&validation, "validation")
            .early_stopping_rounds(NonZeroUsize::new(PATIENCE).unwrap())
            .train()?;
        let seconds = start.elapsed().as_secs_f64();
        let rounds = tuned
            .model
            .best_iteration()
            .ok_or("missing best iteration")?
            + 1;
        let evaluated = tuned.history.len();
        let capped = evaluated >= CAP;
        println!(
            "{}",
            json!({"kind":"selection","variant":name,"selected_rounds":rounds,
            "evaluated_rounds":evaluated,"cap_reached":capped,"patience_exhausted":evaluated>=rounds+PATIENCE,
            "validation_best_score":tuned.best_score,"tuning_seconds":seconds})
        );
        candidates.push(Candidate {
            name,
            params: p,
            rounds,
            capped,
            growth: Vec::new(),
            model: None,
            raw: Vec::new(),
            validation_raw: Vec::new(),
        });
    }
    if candidates[1].rounds != candidates[2].rounds {
        return Err("custom_all/vector selected iteration differs".into());
    }
    for repeat in 0..3 {
        for offset in 0..VARIANTS.len() {
            let index = (offset + repeat) % VARIANTS.len();
            let case = &mut candidates[index];
            let start = Instant::now();
            let model = train(&case.params, &training, case.rounds)?;
            let seconds = start.elapsed().as_secs_f64();
            let raw = model.predict_margin(&testing, ..)?.into_vec();
            let valid_raw = model.predict_margin(&validation, ..)?.into_vec();
            if repeat > 0
                && (!same_bits(&raw, &case.raw) || !same_bits(&valid_raw, &case.validation_raw))
            {
                return Err("timed refits changed prediction bits".into());
            }
            println!(
                "{}",
                json!({"kind":"fit_repeat","variant":case.name,"repeat":repeat,"order":offset,
                "growth_seconds":seconds,"test_margin_hash":hash(&raw),"validation_margin_hash":hash(&valid_raw)})
            );
            case.growth.push(seconds);
            case.raw = raw;
            case.validation_raw = valid_raw;
            case.model = Some(model);
        }
    }
    if !same_bits(&candidates[1].raw, &candidates[2].raw)
        || !same_bits(&candidates[1].validation_raw, &candidates[2].validation_raw)
    {
        return Err("custom_all/vector raw prediction bits differ".into());
    }
    println!(
        "{}",
        json!({"kind":"control","custom_all_matches_vector_best_iteration_and_raw_bits":true})
    );
    let transform = params(Objective::Quantile(levels.clone()), false, seed)?.loss(1)?;
    let mut quality = Vec::new();
    let mut costs = Vec::new();
    for case in &candidates {
        let model = case.model.as_ref().ok_or("missing refit model")?;
        let (latency, prediction) = inference(&Inference {
            model,
            matrix: &testing,
            split: &splits.test,
            columns,
            levels: &levels,
            seed,
        })?;
        let mut valid = case.validation_raw.clone();
        transform.pred_transform(&mut valid);
        let mut calibration_prediction = model.predict_margin(&calibrating, ..)?.into_vec();
        transform.pred_transform(&mut calibration_prediction);
        let test_score = score(&prediction, &splits.test.y, levels.alpha());
        let bytes = model.encode(ModelFormat::Binary)?.len();
        let growth = median(&case.growth);
        let cold = preparation + cold_cut + growth;
        println!(
            "{}",
            json!({"kind":"benchmark","variant":case.name,"selected_rounds":case.rounds,
            "cap_reached":case.capped,"validation_pinball":score(&valid,&splits.validation.y,levels.alpha()),
            "test_mean_pinball":test_score,"test_mean_pinball_original_units":test_score*document.label_scale,
            "warm_growth_seconds":growth,"growth_samples_seconds":case.growth,"cold_fit_seconds":cold,
            "cold_cut_seconds":cold_cut,"training_preparation_seconds":preparation,"inference":latency,
            "native_binary_bytes":bytes,"trees":model.trees().len(),"nodes":model.trees().iter().map(hessboost::tree::RegTree::num_nodes).sum::<usize>(),
            "encoded_custom_semantics":if matches!(case.name,"custom_all"|"projected_newton") {"raw margins; quantile transform external; custom closure not serialized"} else {"builtin quantile model"}})
        );
        quality.push(test_score);
        costs.push((cold, bytes, latency));
        calibration(
            case.name,
            &events(&splits.calibration, &calibration_prediction, width),
            &events(&splits.test, &prediction, width),
        )?;
    }
    for baseline in [0, 1] {
        println!(
            "{}",
            json!({"kind":"comparison","candidate":"projected_newton","baseline":VARIANTS[baseline],
            "relative_test_pinball":quality[3]/quality[baseline],"within_one_percent":quality[3]<=quality[baseline]*1.01,
            "uncapped":!candidates[3].capped&&!candidates[baseline].capped,
            "cold_training_speedup":costs[baseline].0/costs[3].0,"encoded_size_ratio":costs[3].1 as f64/costs[baseline].1 as f64,
            "candidate_inference":costs[3].2,"baseline_inference":costs[baseline].2,
            "promotion":"requires aggregate repeated-period evidence against BOTH baselines and explicit inference/size tradeoff; this row is not approval"})
        );
    }
    Ok(())
}
