//! Public-API inference grid with controlled model feature density and no training.
//!
//! Run a release binary with `RAYON_NUM_THREADS` fixed. Defaults: both matrix and
//! borrowed-row margins, two warmup samples and seven measured samples. Each
//! sample repeats the call (500 times below 16 rows, 50 below 1,024, otherwise 3)
//! and reports seconds per call. Allocation and input validation remain timed;
//! model decoding, matrix construction, hashing, and scalar verification do not.
//!
//! Flags: `--list`, `--case NAME` (repeatable), `--api matrix|rows|both`,
//! `--warmup N`, `--repeats N`, `--iterations N`, `--cold-layout`. The cold mode
//! decodes a fresh model outside timing, then times its first call (one call per
//! sample, including lazy layout construction). For example:
//! `cargo run --release --example bench_cpu_round3 -- --case w1024_p1_d4_t10_b256`.
//! Standard output is JSONL; saved/import fixtures are embedded in the binary.
//! The separate comparison script runs already-built binaries in ABBA order.
use hessboost::model::{BoostedModel, ModelFormat, Predictions};
use hessboost::prelude::DMatrix;
use serde::Serialize;
use serde_json::{Value, json};
use std::{collections::BTreeSet, hint::black_box, time::Instant};

type BenchResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SEED: u64 = 0x2545_f491_4f6c_dd1d;

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Layout {
    Numeric,
    Missing,
    CategoricalMissing,
    Fixture,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum MatrixLayout {
    Dense,
    RowView,
    FiniteSentinel,
    Csr,
}

#[derive(Clone, Copy, Serialize)]
struct Synthetic {
    features: usize,
    used_percent: usize,
    depth: usize,
    trees: usize,
}

#[derive(Clone, Copy)]
enum Source {
    Synthetic(Synthetic),
    SavedVector,
    ImportedLightgbm,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    rows: usize,
    layout: Layout,
    matrix_layout: MatrixLayout,
    source: Source,
}

impl Case {
    const fn synthetic(name: &'static str, rows: usize, layout: Layout, model: Synthetic) -> Self {
        Self {
            name,
            rows,
            layout,
            matrix_layout: MatrixLayout::Dense,
            source: Source::Synthetic(model),
        }
    }

    const fn with_matrix_layout(mut self, layout: MatrixLayout) -> Self {
        self.matrix_layout = layout;
        self
    }

    fn description(self) -> Value {
        let (source, model) = match self.source {
            Source::Synthetic(model) => ("synthetic_native_json", Some(model)),
            Source::SavedVector => ("tests/data/saved/0.4.1/vector_leaves.bin", None),
            Source::ImportedLightgbm => ("tests/data/lightgbm-4.7.0-binary.txt", None),
        };
        json!({"case": self.name, "rows": self.rows, "layout": self.layout,
            "matrix_layout": self.matrix_layout, "source": source, "requested_model": model})
    }
}

// The adjacent 15/16-row cases straddle row-forest/lane dispatch. Narrow, fully
// used, and saved/imported models are controls; 256 is one complete row block.
// Distinct split features are spread across the entire declared input width.
const CASES: &[Case] = &[
    Case::synthetic(
        "w30_p100_d4_t10_b1",
        1,
        Layout::Numeric,
        Synthetic {
            features: 30,
            used_percent: 100,
            depth: 4,
            trees: 10,
        },
    ),
    Case::synthetic(
        "w30_p8_d4_t10_b15_missing",
        15,
        Layout::Missing,
        Synthetic {
            features: 30,
            used_percent: 8,
            depth: 4,
            trees: 10,
        },
    ),
    Case::synthetic(
        "w30_p32_d4_t100_b16",
        16,
        Layout::Numeric,
        Synthetic {
            features: 30,
            used_percent: 32,
            depth: 4,
            trees: 100,
        },
    ),
    Case::synthetic(
        "w30_p100_d8_t100_b256",
        256,
        Layout::Numeric,
        Synthetic {
            features: 30,
            used_percent: 100,
            depth: 8,
            trees: 100,
        },
    ),
    Case::synthetic(
        "w30_p1_d8_t100_b10000_missing",
        10_000,
        Layout::Missing,
        Synthetic {
            features: 30,
            used_percent: 1,
            depth: 8,
            trees: 100,
        },
    ),
    Case::synthetic(
        "w256_p1_d4_t10_b1",
        1,
        Layout::Numeric,
        Synthetic {
            features: 256,
            used_percent: 1,
            depth: 4,
            trees: 10,
        },
    ),
    Case::synthetic(
        "w256_p8_d4_t10_b15_missing",
        15,
        Layout::Missing,
        Synthetic {
            features: 256,
            used_percent: 8,
            depth: 4,
            trees: 10,
        },
    ),
    Case::synthetic(
        "w256_p8_d4_t10_b16_missing",
        16,
        Layout::Missing,
        Synthetic {
            features: 256,
            used_percent: 8,
            depth: 4,
            trees: 10,
        },
    ),
    Case::synthetic(
        "w256_p32_d8_t100_b256",
        256,
        Layout::Numeric,
        Synthetic {
            features: 256,
            used_percent: 32,
            depth: 8,
            trees: 100,
        },
    ),
    Case::synthetic(
        "w256_p100_d4_t100_b10000",
        10_000,
        Layout::Numeric,
        Synthetic {
            features: 256,
            used_percent: 100,
            depth: 4,
            trees: 100,
        },
    ),
    Case::synthetic(
        "w1024_p1_d4_t10_b1",
        1,
        Layout::Numeric,
        Synthetic {
            features: 1024,
            used_percent: 1,
            depth: 4,
            trees: 10,
        },
    ),
    Case::synthetic(
        "w1024_p1_d4_t10_b15",
        15,
        Layout::Numeric,
        Synthetic {
            features: 1024,
            used_percent: 1,
            depth: 4,
            trees: 10,
        },
    ),
    Case::synthetic(
        "w1024_p1_d4_t10_b16",
        16,
        Layout::Numeric,
        Synthetic {
            features: 1024,
            used_percent: 1,
            depth: 4,
            trees: 10,
        },
    ),
    Case::synthetic(
        "w1024_p1_d4_t10_b256",
        256,
        Layout::Numeric,
        Synthetic {
            features: 1024,
            used_percent: 1,
            depth: 4,
            trees: 10,
        },
    ),
    Case::synthetic(
        "w1024_p8_d8_t100_b10000_missing",
        10_000,
        Layout::Missing,
        Synthetic {
            features: 1024,
            used_percent: 8,
            depth: 8,
            trees: 100,
        },
    ),
    Case::synthetic(
        "w1024_p32_d8_t100_b256",
        256,
        Layout::Numeric,
        Synthetic {
            features: 1024,
            used_percent: 32,
            depth: 8,
            trees: 100,
        },
    ),
    Case::synthetic(
        "w1024_p100_d4_t100_b256",
        256,
        Layout::Numeric,
        Synthetic {
            features: 1024,
            used_percent: 100,
            depth: 4,
            trees: 100,
        },
    ),
    Case::synthetic(
        "w1024_p8_d8_t100_b256_categorical_missing",
        256,
        Layout::CategoricalMissing,
        Synthetic {
            features: 1024,
            used_percent: 8,
            depth: 8,
            trees: 100,
        },
    ),
    Case::synthetic(
        "w1024_p8_d8_t100_b10000_row_view",
        10_000,
        Layout::Missing,
        Synthetic {
            features: 1024,
            used_percent: 8,
            depth: 8,
            trees: 100,
        },
    )
    .with_matrix_layout(MatrixLayout::RowView),
    Case::synthetic(
        "w1024_p8_d8_t100_b10000_finite_sentinel",
        10_000,
        Layout::Missing,
        Synthetic {
            features: 1024,
            used_percent: 8,
            depth: 8,
            trees: 100,
        },
    )
    .with_matrix_layout(MatrixLayout::FiniteSentinel),
    Case::synthetic(
        "w256_p8_d8_t100_b10000_csr",
        10_000,
        Layout::Missing,
        Synthetic {
            features: 256,
            used_percent: 8,
            depth: 8,
            trees: 100,
        },
    )
    .with_matrix_layout(MatrixLayout::Csr),
    Case {
        name: "saved_vector_b256",
        rows: 256,
        layout: Layout::Numeric,
        matrix_layout: MatrixLayout::Dense,
        source: Source::SavedVector,
    },
    Case {
        name: "imported_lightgbm_b75",
        rows: 75,
        layout: Layout::Fixture,
        matrix_layout: MatrixLayout::Dense,
        source: Source::ImportedLightgbm,
    },
];

struct Options {
    cases: Vec<String>,
    api: String,
    warmup: usize,
    repeats: usize,
    iterations: Option<u32>,
    list: bool,
    cold_layout: bool,
}

impl Options {
    fn parse() -> BenchResult<Self> {
        let mut options = Self {
            cases: Vec::new(),
            api: "both".into(),
            warmup: 2,
            repeats: 7,
            iterations: None,
            list: false,
            cold_layout: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            if flag == "--list" || flag == "--cold-layout" {
                if flag == "--list" {
                    options.list = true;
                } else {
                    options.cold_layout = true;
                }
                continue;
            }
            if !matches!(
                flag.as_str(),
                "--case" | "--api" | "--warmup" | "--repeats" | "--iterations"
            ) {
                return Err(
                    format!("unknown argument {flag}; see the example's module docs").into(),
                );
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--case" => options.cases.push(value),
                "--api" => options.api = value,
                "--warmup" => options.warmup = value.parse()?,
                "--repeats" => options.repeats = value.parse()?,
                "--iterations" => options.iterations = Some(value.parse()?),
                _ => unreachable!(),
            }
        }
        if !matches!(options.api.as_str(), "matrix" | "rows" | "both") {
            return Err("--api must be matrix, rows or both".into());
        }
        if options.repeats == 0 || options.iterations == Some(0) {
            return Err("--repeats and --iterations must be positive".into());
        }
        if options.cold_layout && options.iterations.is_some_and(|n| n != 1) {
            return Err("--cold-layout uses exactly one call per sample".into());
        }
        for name in &options.cases {
            if !CASES.iter().any(|case| case.name == name) {
                return Err(format!("unknown case {name}; use --list").into());
            }
        }
        Ok(options)
    }
}

fn hash_bytes(values: impl IntoIterator<Item = u8>) -> String {
    // FNV-1a over bytes, with f32 bit patterns fed explicitly little-endian.
    let hash = values
        .into_iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        });
    format!("{hash:016x}")
}

fn hash_floats(values: &[f32]) -> String {
    hash_bytes(
        values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes()),
    )
}

fn used_count(model: Synthetic) -> usize {
    // Nearest integer percentage, with at least one feature. Metadata reports
    // the actual count and density, particularly important for width 30.
    ((model.features * model.used_percent + 50) / 100).max(1)
}

fn synthetic_model(model: Synthetic, layout: Layout) -> BenchResult<BoostedModel> {
    let internal = (1usize << model.depth) - 1;
    let used = used_count(model);
    if used > internal * model.trees {
        return Err("not enough split nodes to reach the requested used-feature count".into());
    }
    let trees: Vec<Value> = (0..model.trees)
        .map(|tree| {
            let nodes: Vec<Value> = (0..=(2 * internal))
                .map(|node| {
                    let leaf = node >= internal;
                    let rank = (tree * internal + node) % used;
                    // Including the highest feature catches accidental prefix-only staging.
                    let feature = if used == 1 {
                        model.features - 1
                    } else {
                        rank * (model.features - 1) / (used - 1)
                    };
                    let categorical = !leaf
                        && matches!(layout, Layout::CategoricalMissing)
                        && feature.is_multiple_of(3);
                    json!({
                        "split_feature": if leaf { 0 } else { feature },
                        "split_cond": if leaf || categorical { 0.0 } else {
                            ((node * 13 + tree * 7) % 17) as f32 / 16.0 - 0.5
                        },
                        "default_left": (node + tree).is_multiple_of(2),
                        "left": if leaf { -1 } else { (2 * node + 1) as i32 },
                        "right": if leaf { -1 } else { (2 * node + 2) as i32 },
                        "leaf_value": if leaf {
                            ((node * 29 + tree * 11) % 257) as f32 / 1024.0 - 0.125
                        } else { 0.0 },
                        "sum_hess": 1.0, "split_gain": if leaf { 0.0 } else { 1.0 },
                        "is_categorical": categorical,
                        "cat_begin": 0, "cat_end": if categorical { 4 } else { 0 },
                    })
                })
                .collect();
            json!({"nodes": nodes, "categories": [0, 2, 4, 6],
            "size_leaf_vector": 0, "leaf_vectors": [], "linear": null})
        })
        .collect();
    let bytes = serde_json::to_vec(&json!({
        "trees": trees, "base_score": [0.125], "objective": "reg:squarederror",
        "num_class": 0, "n_outputs": 1, "n_targets": 1,
        "n_features": model.features, "best_iteration": null,
        "tree_weights": [], "num_parallel_tree": 1, "linear": null,
    }))?;
    Ok(BoostedModel::decode(bytes, ModelFormat::Json)?)
}

fn load_model(case: Case) -> BenchResult<BoostedModel> {
    Ok(match case.source {
        Source::Synthetic(model) => return synthetic_model(model, case.layout),
        Source::SavedVector => BoostedModel::decode(
            include_bytes!("../tests/data/saved/0.4.1/vector_leaves.bin"),
            ModelFormat::Binary,
        )?,
        Source::ImportedLightgbm => BoostedModel::decode(
            include_bytes!("../tests/data/lightgbm-4.7.0-binary.txt"),
            ModelFormat::LightgbmText,
        )?,
    })
}

fn input_rows(case: Case, model: &BoostedModel) -> BenchResult<Vec<f32>> {
    if matches!(case.source, Source::ImportedLightgbm) {
        #[derive(serde::Deserialize)]
        struct Probe {
            n_cols: usize,
            n_test: usize,
            x_test: Vec<Option<f32>>,
        }
        let probe: Probe = serde_json::from_slice(include_bytes!(
            "../tests/data/lightgbm-4.7.0-binary.expected.json"
        ))?;
        if probe.n_cols != model.n_features() || probe.n_test != case.rows {
            return Err("imported probe shape differs from its case".into());
        }
        return Ok(probe
            .x_test
            .into_iter()
            .map(|x| x.unwrap_or(f32::NAN))
            .collect());
    }
    let mut categorical = vec![false; model.n_features()];
    for tree in model.trees() {
        for node in tree
            .nodes()
            .iter()
            .filter(|node| !node.is_leaf() && node.is_categorical)
        {
            categorical[node.split_feature as usize] = true;
        }
    }
    let missing = matches!(case.layout, Layout::Missing | Layout::CategoricalMissing);
    let mut state = SEED;
    Ok((0..case.rows * model.n_features())
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let row = index / model.n_features();
            let feature = index % model.n_features();
            if missing && ((row * 131 + feature * 17) % 11 < 3 || row % 67 == 66) {
                f32::NAN
            } else if categorical[feature] {
                ((state >> 32) % 8) as f32
            } else if index % 97 == 1 {
                -0.0
            } else if index.is_multiple_of(97) {
                0.0
            } else {
                (state >> 40) as f32 / 8_388_608.0 - 1.0
            }
        })
        .collect())
}

// The canonical NaN-missing rows stay unchanged for the scalar and borrowed-row
// oracles. Only the DMatrix representation changes, outside every timed call.
fn input_matrix(case: Case, model: &BoostedModel, values: &[f32]) -> BenchResult<DMatrix> {
    let width = model.n_features();
    Ok(match case.matrix_layout {
        MatrixLayout::Dense => DMatrix::from_dense(values, case.rows, width)?,
        MatrixLayout::RowView => {
            let source: Vec<f32> = values
                .chunks_exact(width)
                .rev()
                .flatten()
                .copied()
                .collect();
            let matrix = DMatrix::from_dense_vec(source, case.rows, width)?;
            // Reverse the reversed source via a view, not a materialized copy.
            // Noncontiguous row-map access exercises scratch-block staging.
            matrix.select_rows(&(0..case.rows).rev().collect::<Vec<_>>())?
        }
        MatrixLayout::FiniteSentinel => {
            let encoded: Vec<f32> = values
                .iter()
                .map(|&value| if value.is_nan() { -9999.0 } else { value })
                .collect();
            DMatrix::from_dense_with_missing(&encoded, case.rows, width, -9999.0)?
        }
        MatrixLayout::Csr => {
            let mut indptr = Vec::with_capacity(case.rows + 1);
            let mut indices = Vec::new();
            let mut present = Vec::new();
            indptr.push(0);
            for row in values.chunks_exact(width) {
                for (feature, &value) in row.iter().enumerate() {
                    if !value.is_nan() {
                        indices.push(feature as u32);
                        present.push(value);
                    }
                }
                indptr.push(indices.len());
            }
            DMatrix::from_csr(indptr, indices, present, width)?
        }
    })
}

// This plain, public-API tree walk does not use compact layouts or staging.
// It verifies all rows, not just a sample or a hash, before timing either API.
fn scalar_margins(model: &BoostedModel, values: &[f32]) -> BenchResult<Vec<f32>> {
    if model.shrinkage().is_some()
        || model.linear().is_some()
        || model
            .trees()
            .iter()
            .any(|tree| tree.linear_leaves().is_some())
    {
        return Err("the scalar benchmark oracle excludes shrinkage and linear models".into());
    }
    let mut out = Vec::with_capacity(values.len() / model.n_features() * model.n_outputs());
    for row in values.chunks_exact(model.n_features()) {
        let mut margins = model.base_scores().to_vec();
        for ((tree, weight), outputs) in model
            .trees()
            .iter()
            .zip(model.tree_weights())
            .zip(model.tree_outputs())
        {
            let leaf = tree.leaf_id_with(|feature| {
                let value = row[feature as usize];
                (!value.is_nan()).then_some(value)
            });
            if tree.is_vector_leaf() {
                for (output, &value) in outputs.zip(tree.leaf_vector(leaf)) {
                    margins[output] += weight * value;
                }
            } else {
                margins[outputs.start] += weight * tree.nodes()[leaf].leaf_value;
            }
        }
        out.extend(margins);
    }
    Ok(out)
}

fn assert_bits_equal(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "prediction length: {context}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "prediction bit mismatch: {context}, value {index}"
        );
    }
}

fn measure(
    options: &Options,
    iterations: u32,
    reference: &[f32],
    mut predict: impl FnMut() -> hessboost::error::Result<Predictions>,
) -> BenchResult<Vec<f64>> {
    let mut seconds = Vec::with_capacity(options.repeats);
    for run in 0..options.warmup + options.repeats {
        let start = Instant::now();
        let mut predictions = black_box(predict()?);
        for _ in 1..iterations {
            predictions = black_box(predict()?);
        }
        let elapsed = start.elapsed().as_secs_f64() / f64::from(iterations);
        assert_bits_equal(predictions.as_slice(), reference, "timed sample");
        if run >= options.warmup {
            seconds.push(elapsed);
        }
    }
    Ok(seconds)
}

fn measure_cold(
    options: &Options,
    bytes: &[u8],
    reference: &[f32],
    mut predict: impl FnMut(&BoostedModel) -> hessboost::error::Result<Predictions>,
) -> BenchResult<Vec<f64>> {
    let mut seconds = Vec::with_capacity(options.repeats);
    for run in 0..options.warmup + options.repeats {
        let model = BoostedModel::decode(bytes, ModelFormat::Json)?;
        let start = Instant::now();
        let predictions = black_box(predict(black_box(&model))?);
        let elapsed = start.elapsed().as_secs_f64();
        assert_bits_equal(predictions.as_slice(), reference, "cold-layout sample");
        if run >= options.warmup {
            seconds.push(elapsed);
        }
    }
    Ok(seconds)
}

fn run_case(case: Case, options: &Options) -> BenchResult<()> {
    let model = load_model(case)?;
    let values = input_rows(case, &model)?;
    let data = input_matrix(case, &model, &values)?;
    let reference = scalar_margins(&model, &values)?;
    // Initialize lazy layouts before timing, even when --warmup is zero.
    assert_bits_equal(
        model.predict_margin(&data, ..)?.as_slice(),
        &reference,
        case.name,
    );
    assert_bits_equal(
        model.predict_margin_rows(&values, ..)?.as_slice(),
        &reference,
        case.name,
    );
    let features: BTreeSet<u32> = model
        .trees()
        .iter()
        .flat_map(hessboost::tree::RegTree::nodes)
        .filter(|node| !node.is_leaf())
        .map(|node| node.split_feature)
        .collect();
    if let Source::Synthetic(config) = case.source {
        assert_eq!(features.len(), used_count(config));
    }
    let iterations = if options.cold_layout {
        1
    } else {
        options.iterations.unwrap_or(if case.rows < 16 {
            500
        } else if case.rows < 1024 {
            50
        } else {
            3
        })
    };
    let bytes = model.encode(ModelFormat::Json)?;
    let mut metadata = case.description();
    metadata["schema_version"] = json!(1);
    metadata["features"] = json!(model.n_features());
    metadata["used_features"] = json!(features.len());
    metadata["used_feature_fraction"] = json!(features.len() as f64 / model.n_features() as f64);
    metadata["max_used_feature"] = json!(features.last());
    metadata["trees"] = json!(model.num_trees());
    metadata["depth"] = json!(
        model
            .trees()
            .iter()
            .map(hessboost::tree::RegTree::depth)
            .max()
    );
    metadata["outputs"] = json!(model.n_outputs());
    metadata["threads"] = json!(rayon::current_num_threads());
    metadata["seed"] = json!(format!("{SEED:016x}"));
    metadata["warmup_samples"] = json!(options.warmup);
    metadata["measured_samples"] = json!(options.repeats);
    metadata["iterations_per_sample"] = json!(iterations);
    metadata["checksum_algorithm"] = json!("fnv1a64_le_f32_bits");
    metadata["checksum"] = json!(hash_floats(&reference));
    metadata["input_checksum"] = json!(hash_floats(&values));
    metadata["model_checksum"] = json!(hash_bytes(bytes.iter().copied()));
    metadata["layout_cache"] = json!(if options.cold_layout { "cold" } else { "warm" });
    metadata["scalar_oracle_verified"] = json!(true);
    metadata["missing_values"] = json!(values.iter().filter(|value| value.is_nan()).count());
    metadata["categorical_split_nodes"] = json!(
        model
            .trees()
            .iter()
            .flat_map(hessboost::tree::RegTree::nodes)
            .filter(|node| !node.is_leaf() && node.is_categorical)
            .count()
    );
    for api in ["matrix", "rows"] {
        if options.api != "both" && options.api != api {
            continue;
        }
        let predict = |model: &BoostedModel| match api {
            "matrix" => model.predict_margin(black_box(&data), ..),
            _ => model.predict_margin_rows(black_box(&values), ..),
        };
        let seconds = if options.cold_layout {
            measure_cold(options, &bytes, &reference, predict)?
        } else {
            measure(options, iterations, &reference, || {
                predict(black_box(&model))
            })?
        };
        metadata["api"] = json!(api);
        metadata["seconds"] = json!(seconds);
        println!("{metadata}");
    }
    Ok(())
}

fn main() -> BenchResult<()> {
    let options = Options::parse()?;
    for &case in CASES {
        if !options.cases.is_empty() && !options.cases.iter().any(|name| name == case.name) {
            continue;
        }
        if options.list {
            println!("{}", case.description());
        } else {
            run_case(case, &options)?;
        }
    }
    Ok(())
}
