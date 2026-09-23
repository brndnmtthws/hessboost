//! XGBoost 3.4.1 parity integration tests.
//!
//! Fixtures come from `scripts/gen_fixtures.py` (real XGBoost, single thread) and
//! follow the schema documented there. Two ignored tests consume them:
//!
//! * [`xgboost_parity`] runs the three-way check per case: **train** on the
//!   fixture data and compare test predictions (pointwise for the `exact` tier,
//!   a quality band for the RNG-driven `quality` tier), **import** the embedded
//!   XGBoost model and compare predictions, margins and SHAP contributions, and
//!   **export** the hessboost model to `fixtures/exports/` for
//!   `scripts/check_exports.py` to reload in XGBoost.
//! * [`quantile_cuts_match_xgboost`] compares `hist` quantile cuts bit-for-bit
//!   against `DMatrix.get_quantile_cut()` oracles in `fixtures/cuts/`.
//!
//! ```sh
//! uv run --with xgboost==3.4.1 --with numpy python scripts/gen_fixtures.py
//! cargo test -p hessboost --test parity --release -- --ignored --nocapture
//! uv run --with xgboost==3.4.1 --with numpy python scripts/check_exports.py
//! ```

use hessboost::data::HistCuts;
use hessboost::{
    BoostedModel, BoosterKind, DMatrix, FeatureType, GrowPolicy, HessboostError, Monotone,
    TrainingParams, TreeMethod, train,
};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

/// Rows of `x_test` on which the fixture carries SHAP contributions.
const CONTRIB_ROWS: usize = 50;

// ---------------------------------------------------------------------------
// Fixture schema
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Tier {
    Exact,
    Quality,
    Trainonly,
}

#[derive(Deserialize)]
struct Tol {
    train: f64,
    import: f64,
    contribs: f64,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    tier: Tier,
    params: Map<String, Value>,
    num_class: usize,
    num_round: usize,
    n_train: usize,
    n_test: usize,
    n_cols: usize,
    #[serde(deserialize_with = "nan_for_null")]
    x_train: Vec<f32>,
    y_train: Vec<f32>,
    #[serde(deserialize_with = "nan_for_null")]
    x_test: Vec<f32>,
    y_test: Vec<f32>,
    weights: Option<Vec<f32>>,
    group_sizes: Option<Vec<usize>>,
    test_group_sizes: Option<Vec<usize>>,
    feature_types: Option<Vec<String>>,
    xgb_pred: Vec<f32>,
    xgb_margin: Vec<f32>,
    xgb_contribs: Vec<f32>,
    xgb_model: Value,
    tol: Tol,
}

#[derive(Deserialize)]
struct CutFixture {
    name: String,
    tree_method: String,
    objective: String,
    max_bin: usize,
    n_rows: usize,
    n_cols: usize,
    #[serde(deserialize_with = "nan_for_null")]
    x: Vec<f32>,
    w: Option<Vec<f32>>,
    indptr: Vec<usize>,
    /// XGBoost's flattened cut values; each feature block opens with `-inf`
    /// (serialized as `null`, deserialized here as NaN and skipped).
    #[serde(deserialize_with = "nan_for_null")]
    cuts: Vec<f32>,
}

/// JSON `null` is the fixture encoding for a missing value.
fn nan_for_null<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
    let v: Vec<Option<f32>> = Vec::deserialize(d)?;
    Ok(v.into_iter().map(|x| x.unwrap_or(f32::NAN)).collect())
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

fn load_all<T: for<'de> Deserialize<'de>>(dir: &Path, what: &str) -> Vec<(String, T)> {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| {
        panic!(
            "{what} fixtures missing at {}: {e}; run scripts/gen_fixtures.py",
            dir.display()
        )
    });
    let mut out: Vec<(String, T)> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("json"))
        .map(|path| {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
            let value = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
            (path.display().to_string(), value)
        })
        .collect();
    assert!(!out.is_empty(), "no {what} fixtures in {}", dir.display());
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

// ---------------------------------------------------------------------------
// XGBoost params -> TrainingParams
// ---------------------------------------------------------------------------

fn f64_of(key: &str, v: &Value) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| format!("`{key}` must be a number, got {v}"))
}

fn usize_of(key: &str, v: &Value) -> Result<usize, String> {
    v.as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| format!("`{key}` must be a non-negative integer, got {v}"))
}

fn str_of<'a>(key: &str, v: &'a Value) -> Result<&'a str, String> {
    v.as_str()
        .ok_or_else(|| format!("`{key}` must be a string, got {v}"))
}

/// A parameter hessboost implements only one way: the fixture must carry exactly
/// that value, otherwise the case is not comparable.
fn expect_fixed(key: &str, v: &Value, want: &Value) -> Result<(), String> {
    if v == want {
        Ok(())
    } else {
        Err(format!(
            "`{key}` must be {want} (hessboost's only implementation), got {v}"
        ))
    }
}

/// `"(1,-1,0)"` -> per-feature [`Monotone`].
fn parse_monotone(s: &str) -> Result<Vec<Monotone>, String> {
    s.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .map(|t| match t.trim() {
            "1" => Ok(Monotone::Increasing),
            "-1" => Ok(Monotone::Decreasing),
            "0" => Ok(Monotone::None),
            other => Err(format!("monotone_constraints: bad entry `{other}`")),
        })
        .collect()
}

/// Map the fixture's XGBoost parameter dict onto the builder. Every key must be
/// handled: an unknown key is a failure, never a silent skip.
fn build_params(fx: &Fixture) -> Result<TrainingParams, String> {
    let mut b = TrainingParams::builder();
    for (key, v) in &fx.params {
        let k = key.as_str();
        b = match k {
            "objective" => b.objective(str_of(k, v)?),
            "num_class" => b.num_class(usize_of(k, v)?),
            "base_score" => b.base_score(f64_of(k, v)?),
            "eta" => b.eta(f64_of(k, v)?),
            "gamma" => b.gamma(f64_of(k, v)?),
            "max_depth" => b.max_depth(usize_of(k, v)?),
            "max_leaves" => b.max_leaves(usize_of(k, v)?),
            "min_child_weight" => b.min_child_weight(f64_of(k, v)?),
            "max_delta_step" => b.max_delta_step(f64_of(k, v)?),
            "reg_lambda" => b.lambda(f64_of(k, v)?),
            "reg_alpha" => b.alpha(f64_of(k, v)?),
            "scale_pos_weight" => b.scale_pos_weight(f64_of(k, v)?),
            "max_bin" => b.max_bin(usize_of(k, v)?),
            "subsample" => b.subsample(f64_of(k, v)?),
            "colsample_bytree" => b.colsample_bytree(f64_of(k, v)?),
            "colsample_bylevel" => b.colsample_bylevel(f64_of(k, v)?),
            "colsample_bynode" => b.colsample_bynode(f64_of(k, v)?),
            "seed" => b.seed(usize_of(k, v)? as u64),
            "tweedie_variance_power" => b.tweedie_variance_power(f64_of(k, v)?),
            "huber_slope" => b.huber_slope(f64_of(k, v)?),
            "rate_drop" => b.rate_drop(f64_of(k, v)?),
            "skip_drop" => b.skip_drop(f64_of(k, v)?),
            "tree_method" => b.tree_method(match str_of(k, v)? {
                "exact" => TreeMethod::Exact,
                "approx" => TreeMethod::Approx,
                "hist" => TreeMethod::Hist,
                other => return Err(format!("tree_method `{other}` not mapped")),
            }),
            "grow_policy" => b.grow_policy(match str_of(k, v)? {
                "depthwise" => GrowPolicy::DepthWise,
                "lossguide" => GrowPolicy::LossGuide,
                other => return Err(format!("grow_policy `{other}` not mapped")),
            }),
            "booster" => b.booster(match str_of(k, v)? {
                "gbtree" => BoosterKind::GbTree,
                "gblinear" => BoosterKind::GbLinear,
                "dart" => BoosterKind::Dart,
                other => return Err(format!("booster `{other}` not mapped")),
            }),
            "monotone_constraints" => b.monotone_constraints(parse_monotone(str_of(k, v)?)?),
            "interaction_constraints" => b.interaction_constraints(
                serde_json::from_str::<Vec<Vec<u32>>>(str_of(k, v)?)
                    .map_err(|e| format!("interaction_constraints: {e}"))?,
            ),
            // gblinear: hessboost implements exactly cyclic coordinate descent.
            "updater" => {
                expect_fixed(k, v, &Value::from("coord_descent"))?;
                b
            }
            "feature_selector" => {
                expect_fixed(k, v, &Value::from("cyclic"))?;
                b
            }
            // rank:*: hessboost pairs every document with every other in the
            // query; XGBoost does the same with `topk` truncated at the group
            // size, so both values are pinned to that configuration.
            "lambdarank_pair_method" => {
                expect_fixed(k, v, &Value::from("topk"))?;
                b
            }
            "lambdarank_num_pair_per_sample" => {
                let n = usize_of(k, v)?;
                let groups = fx
                    .group_sizes
                    .as_deref()
                    .ok_or("lambdarank_num_pair_per_sample without group_sizes")?;
                if groups.iter().any(|&g| g < n) {
                    return Err(format!("`{k}`={n} exceeds a train group size"));
                }
                b.lambdarank_num_pair_per_sample(n)
            }
            other => return Err(format!("unmapped XGBoost parameter `{other}`")),
        };
    }
    b.build().map_err(|e| format!("invalid params: {e}"))
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

/// Max |a - b|; a NaN on either side counts as infinite so it can never hide.
fn max_abs_diff(what: &str, a: &[f32], b: &[f32]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!(
            "{what}: length mismatch (hessboost {}, xgboost {})",
            a.len(),
            b.len()
        ));
    }
    Ok(a.iter().zip(b).fold(0.0f64, |m, (x, y)| {
        let d = (f64::from(*x) - f64::from(*y)).abs();
        m.max(if d.is_nan() { f64::INFINITY } else { d })
    }))
}

fn rmse(p: &[f32], y: &[f32]) -> f64 {
    let s: f64 = p
        .iter()
        .zip(y)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum();
    (s / y.len() as f64).sqrt()
}

/// Class decision per row for the transformed prediction layout of `objective`.
fn accuracy(objective: &str, p: &[f32], y: &[f32], k: usize) -> f64 {
    let hits = y
        .iter()
        .enumerate()
        .filter(|&(i, &yi)| {
            let class = if objective == "multi:softprob" {
                let row = &p[i * k..(i + 1) * k];
                row.iter()
                    .enumerate()
                    .fold(0usize, |best, (c, v)| if *v > row[best] { c } else { best })
            } else if objective.starts_with("binary:") {
                usize::from(p[i] > 0.5)
            } else {
                p[i].round() as usize
            };
            class == yi as usize
        })
        .count();
    hits as f64 / y.len() as f64
}

fn dcg(labels_in_rank_order: impl Iterator<Item = f32>) -> f64 {
    labels_in_rank_order
        .enumerate()
        .map(|(i, l)| (2f64.powf(f64::from(l)) - 1.0) / (i as f64 + 2.0).log2())
        .sum()
}

/// Mean NDCG over query groups, each evaluated on its full list (the fixture
/// groups are uniform, so this is `NDCG@group_size`). Ties broken by row order.
fn mean_ndcg(scores: &[f32], labels: &[f32], groups: &[usize]) -> f64 {
    let mut start = 0;
    let mut total = 0.0;
    for &g in groups {
        let s = &scores[start..start + g];
        let l = &labels[start..start + g];
        let mut order: Vec<usize> = (0..g).collect();
        order.sort_by(|&a, &b| {
            s[b].partial_cmp(&s[a])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut ideal = l.to_vec();
        ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let idcg = dcg(ideal.iter().copied());
        if idcg > 0.0 {
            total += dcg(order.iter().map(|&j| l[j])) / idcg;
        }
        start += g;
    }
    total / groups.len() as f64
}

fn fmt_delta(d: &Result<f64, String>) -> String {
    match d {
        Ok(v) => format!("{v:.2e}"),
        Err(_) => "ERR".to_string(),
    }
}

// ---------------------------------------------------------------------------
// One parity case
// ---------------------------------------------------------------------------

struct Row {
    name: String,
    tier: &'static str,
    train: String,
    import: String,
    margin: String,
    contribs: String,
    export: String,
    base_score: String,
}

struct Case<'a> {
    fx: &'a Fixture,
    failures: &'a mut Vec<String>,
}

impl Case<'_> {
    fn fail(&mut self, msg: impl std::fmt::Display) {
        self.failures.push(format!("{}: {msg}", self.fx.name));
    }

    /// Record `delta` against `tol`; returns the table cell.
    fn check(&mut self, what: &str, delta: &Result<f64, String>, tol: f64) -> String {
        match delta {
            Ok(d) if *d <= tol => {}
            Ok(d) => self.fail(format!("{what}: max|Δ|={d:.3e} > tol {tol:.0e}")),
            Err(e) => self.fail(format!("{what}: {e}")),
        }
        fmt_delta(delta)
    }

    fn dmatrix(&self, x: &[f32], n_rows: usize) -> Result<DMatrix, String> {
        let mut d =
            DMatrix::from_dense(x, n_rows, self.fx.n_cols).map_err(|e| format!("DMatrix: {e}"))?;
        if let Some(types) = &self.fx.feature_types {
            let types: Vec<FeatureType> = types
                .iter()
                .map(|t| match t.as_str() {
                    "c" => Ok(FeatureType::Categorical),
                    "q" => Ok(FeatureType::Numerical),
                    other => Err(format!("unsupported feature type `{other}`")),
                })
                .collect::<Result<_, _>>()?;
            d = d
                .with_feature_types(&types)
                .map_err(|e| format!("feature types: {e}"))?;
        }
        Ok(d)
    }

    fn train_matrix(&self) -> Result<DMatrix, String> {
        let fx = self.fx;
        let mut d = self
            .dmatrix(&fx.x_train, fx.n_train)?
            .with_labels(&fx.y_train)
            .map_err(|e| format!("labels: {e}"))?;
        if let Some(w) = &fx.weights {
            d = d.with_weights(w).map_err(|e| format!("weights: {e}"))?;
        }
        if let Some(g) = &fx.group_sizes {
            d = d
                .with_group_sizes(g)
                .map_err(|e| format!("group_sizes: {e}"))?;
        }
        Ok(d)
    }

    /// Assertion 1: train and compare `predict(x_test)` with `xgb_pred`.
    fn train_and_compare(&mut self, dtest: &DMatrix) -> Result<(BoostedModel, Vec<f32>), String> {
        let fx = self.fx;
        let params = build_params(fx)?;
        let dtrain = self.train_matrix()?;
        let model = train(&params, &dtrain, fx.num_round).map_err(|e| format!("train: {e}"))?;
        let preds = model.predict(dtest).map_err(|e| format!("predict: {e}"))?;
        if preds.len() != fx.xgb_pred.len() {
            return Err(format!(
                "train predict: length mismatch (hessboost {}, xgboost {})",
                preds.len(),
                fx.xgb_pred.len()
            ));
        }
        Ok((model, preds))
    }

    /// Quality tier: RMSE ratio for regression, accuracy / NDCG slack otherwise.
    fn quality_band(&mut self, preds: &[f32]) -> String {
        let fx = self.fx;
        let objective = fx.params["objective"].as_str().unwrap_or("");
        let band = fx.tol.train;
        let (metric, seq, xgb, ok) = if objective.starts_with("rank:") {
            let Some(groups) = fx.test_group_sizes.as_deref() else {
                self.fail("quality band for rank:* needs test_group_sizes");
                return "ERR".to_string();
            };
            let s = mean_ndcg(preds, &fx.y_test, groups);
            let x = mean_ndcg(&fx.xgb_pred, &fx.y_test, groups);
            ("ndcg", s, x, s >= x - band)
        } else if objective.starts_with("binary:") || objective.starts_with("multi:") {
            let s = accuracy(objective, preds, &fx.y_test, fx.num_class);
            let x = accuracy(objective, &fx.xgb_pred, &fx.y_test, fx.num_class);
            ("acc", s, x, s >= x - band)
        } else {
            let s = rmse(preds, &fx.y_test);
            let x = rmse(&fx.xgb_pred, &fx.y_test);
            ("rmse", s, x, s <= x * band + 1e-6)
        };
        if !ok {
            self.fail(format!(
                "quality band: {metric} hessboost={seq:.5} xgboost={xgb:.5} (band {band})"
            ));
        }
        format!("{metric} {seq:.4}/{xgb:.4}")
    }

    /// Assertion 2: import the embedded XGBoost model and compare predictions,
    /// margins, and SHAP contributions. XGBoost's multiclass contribution layout
    /// `(rows, num_class, n_cols + 1)` is identical to hessboost's
    /// `predict_contribs` layout, so both flatten to the same order.
    fn import_and_compare(&mut self, dtest: &DMatrix, dcontrib: &DMatrix) -> [String; 3] {
        let fx = self.fx;
        let imported = BoostedModel::from_xgboost_json(&fx.xgb_model.to_string());
        if booster_of(fx) == "gblinear" {
            return match imported {
                Err(HessboostError::ModelFormat(_)) => std::array::from_fn(|_| "n/a".to_string()),
                Err(e) => {
                    self.fail(format!("expected ModelFormat import error, got {e}"));
                    std::array::from_fn(|_| "ERR".to_string())
                }
                Ok(_) => {
                    self.fail("unsupported gblinear import unexpectedly succeeded");
                    std::array::from_fn(|_| "ERR".to_string())
                }
            };
        }
        let model = match imported {
            Ok(m) => m,
            Err(e) => {
                self.fail(format!("import: {e}"));
                return std::array::from_fn(|_| "ERR".to_string());
            }
        };
        let pred = model
            .predict(dtest)
            .map_err(|e| e.to_string())
            .and_then(|p| max_abs_diff("import predict", &p, &fx.xgb_pred));
        let margin = model
            .predict_margin(dtest)
            .map_err(|e| e.to_string())
            .and_then(|p| max_abs_diff("import margin", &p, &fx.xgb_margin));
        let contribs = model
            .predict_contribs(dcontrib)
            .map_err(|e| e.to_string())
            .and_then(|p| max_abs_diff("import contribs", &p, &fx.xgb_contribs));
        [
            self.check("import predict", &pred, fx.tol.import),
            self.check("import margin", &margin, fx.tol.import),
            self.check("import contribs", &contribs, fx.tol.contribs),
        ]
    }

    /// Assertion 3: export the trained model plus hessboost's predictions for
    /// `scripts/check_exports.py`.
    fn export(&mut self, model: &BoostedModel, preds: &[f32], dir: &Path) -> String {
        let fx = self.fx;
        if booster_of(fx) == "gblinear" {
            return "skipped".to_string();
        }
        let written = model
            .to_xgboost_json()
            .map_err(|e| format!("export: {e}"))
            .and_then(|json| {
                std::fs::write(dir.join(format!("{}.model.json", fx.name)), json)
                    .map_err(|e| format!("write model: {e}"))
            })
            .and_then(|()| {
                serde_json::to_string(preds)
                    .map_err(|e| format!("encode preds: {e}"))
                    .and_then(|p| {
                        std::fs::write(dir.join(format!("{}.pred.json", fx.name)), p)
                            .map_err(|e| format!("write preds: {e}"))
                    })
            });
        match written {
            Ok(()) => "written".to_string(),
            Err(e) => {
                self.fail(e);
                "ERR".to_string()
            }
        }
    }

    fn run(mut self, exports: &Path) -> Row {
        let fx = self.fx;
        let tier = match fx.tier {
            Tier::Exact => "exact",
            Tier::Quality => "quality",
            Tier::Trainonly => "train-only",
        };
        let mut row = Row {
            name: fx.name.clone(),
            tier,
            train: "ERR".to_string(),
            import: "ERR".to_string(),
            margin: "ERR".to_string(),
            contribs: "ERR".to_string(),
            export: "n/a".to_string(),
            base_score: "-".to_string(),
        };

        let dtest = match self.dmatrix(&fx.x_test, fx.n_test) {
            Ok(d) => d,
            Err(e) => {
                self.fail(e);
                return row;
            }
        };
        let dcontrib = match self.dmatrix(&fx.x_test[..CONTRIB_ROWS * fx.n_cols], CONTRIB_ROWS) {
            Ok(d) => d,
            Err(e) => {
                self.fail(e);
                return row;
            }
        };

        match self.train_and_compare(&dtest) {
            Ok((model, preds)) => {
                row.train = match fx.tier {
                    Tier::Exact | Tier::Trainonly => {
                        let d = max_abs_diff("train predict", &preds, &fx.xgb_pred);
                        self.check("train predict", &d, fx.tol.train)
                    }
                    Tier::Quality => self.quality_band(&preds),
                };
                row.base_score = format!(
                    "{:?}",
                    model
                        .base_scores()
                        .iter()
                        .map(|v| (f64::from(*v) * 1e4).round() / 1e4)
                        .collect::<Vec<_>>()
                );
                row.export = self.export(&model, &preds, exports);
            }
            Err(e) => self.fail(e),
        }

        [row.import, row.margin, row.contribs] = self.import_and_compare(&dtest, &dcontrib);
        row
    }
}

fn booster_of(fx: &Fixture) -> &str {
    fx.params
        .get("booster")
        .and_then(Value::as_str)
        .unwrap_or("gbtree")
}

fn xgb_base_score(fx: &Fixture) -> String {
    fx.xgb_model
        .pointer("/learner/learner_model_param/base_score")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string()
}

#[test]
#[ignore = "requires fixtures from scripts/gen_fixtures.py"]
fn xgboost_parity() {
    let dir = fixtures_dir();
    let exports = dir.join("exports");
    std::fs::create_dir_all(&exports).expect("create fixtures/exports");
    let fixtures: Vec<(String, Fixture)> = load_all(&dir, "parity");

    let mut failures = Vec::new();
    println!(
        "{:<26} {:<7} {:<22} {:<9} {:<9} {:<9} {:<8} base_score hessboost | xgboost",
        "case", "tier", "train", "import", "margin", "contribs", "export"
    );
    for (_, fx) in &fixtures {
        let row = Case {
            fx,
            failures: &mut failures,
        }
        .run(&exports);
        println!(
            "{:<26} {:<7} {:<22} {:<9} {:<9} {:<9} {:<8} {} | {}",
            row.name,
            row.tier,
            row.train,
            row.import,
            row.margin,
            row.contribs,
            row.export,
            row.base_score,
            xgb_base_score(fx)
        );
    }
    assert!(
        failures.is_empty(),
        "{} parity failure(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
#[ignore = "requires fixtures from scripts/gen_fixtures.py"]
fn quantile_cuts_match_xgboost() {
    let fixtures: Vec<(String, CutFixture)> = load_all(&fixtures_dir().join("cuts"), "cut");
    let mut failures: Vec<String> = Vec::new();
    for (_, fx) in &fixtures {
        let mut d = DMatrix::from_dense(&fx.x, fx.n_rows, fx.n_cols).expect("dense matrix");
        if let Some(w) = &fx.w {
            d = d.with_weights(w).expect("weights");
        }
        let cuts = match fx.tree_method.as_str() {
            "hist" => HistCuts::from_dmatrix(&d, fx.max_bin),
            "approx" => {
                // Round-0 Hessian at base_score 0.5 on all-zero labels: 1 for
                // squared error, 0.25 for logistic, times the sample weight
                // (the objective already folds the weight into the Hessian).
                let unit = match fx.objective.as_str() {
                    "reg:squarederror" => 1.0,
                    "binary:logistic" => 0.25,
                    other => panic!("{}: unsupported cut oracle objective {other}", fx.name),
                };
                let hessians: Vec<f32> = (0..fx.n_rows)
                    .map(|r| unit * fx.w.as_ref().map_or(1.0, |w| w[r]))
                    .collect();
                let const_hess = fx.objective == "reg:squarederror";
                HistCuts::from_dmatrix_weighted(&d, fx.max_bin, &hessians, !const_hess)
            }
            other => panic!("{}: unsupported cut oracle tree_method {other}", fx.name),
        };
        let mut case_ok = true;
        for f in 0..fx.n_cols {
            // Skip the leading -inf of XGBoost's block.
            let xgb = &fx.cuts[fx.indptr[f] + 1..fx.indptr[f + 1]];
            let (start, end) = cuts.feature_bins(f);
            let seq: Vec<f32> = (start..end).map(|i| cuts.cut_value(i)).collect();
            let first_diff = seq
                .iter()
                .zip(xgb)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            let problem = match first_diff {
                Some(i) => Some(format!(
                    "feature {f} cut {i}: hessboost {:e} ({:#010x}) vs xgboost {:e} ({:#010x})",
                    seq[i],
                    seq[i].to_bits(),
                    xgb[i],
                    xgb[i].to_bits()
                )),
                None if seq.len() != xgb.len() => Some(format!(
                    "feature {f}: {} cuts vs xgboost {}",
                    seq.len(),
                    xgb.len()
                )),
                None => None,
            };
            if let Some(p) = problem {
                case_ok = false;
                failures.push(format!("{}: {p}", fx.name));
            }
        }
        println!(
            "{:<32} rows={:<7} cols={} max_bin={:<4} cuts={:<5} {}",
            fx.name,
            fx.n_rows,
            fx.n_cols,
            fx.max_bin,
            cuts.total_bins(),
            if case_ok { "OK" } else { "FAIL" }
        );
    }
    assert!(
        failures.is_empty(),
        "{} cut mismatch(es):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
