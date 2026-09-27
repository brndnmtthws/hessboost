//! LightGBM text model import; user docs: `model` module, "LightGBM import".

use crate::error::{HessboostError, Result};
use crate::model::categories::{CategoryPool, PoolError};
use crate::model::{BoostedModel, ModelObjective, ModelSpec};
use crate::objective::{
    LambdaRank, Multiclass, Objective, PseudoHuber, Quantiles, RegLoss, Tweedie,
};
use crate::tree::{LinearLeaves, Node, RegTree};
use std::collections::HashMap;
use std::str::FromStr;

/// LightGBM's `kZeroThreshold` (`1e-35f`): with missing type `Zero`, values
/// of at most this magnitude take the default direction (`Tree::IsZero`).
const ZERO_THRESHOLD: f32 = 1e-35;

/// The model format version LightGBM 4.x writes (`kModelVersion`).
const MODEL_VERSION: &str = "v4";

/// `decision_type` bits (`include/LightGBM/tree.h`).
const CATEGORICAL_MASK: u8 = 1;
const DEFAULT_LEFT_MASK: u8 = 2;
/// Bits 2-3 hold the missing type; LightGBM defines no higher bit.
const KNOWN_DECISION_BITS: u8 = 0b1111;

/// Most bitset words one categorical split may span: categories stay below
/// `2^31`, the range LightGBM's `int` cast of a feature value covers.
const MAX_CATEGORY_WORDS: usize = 1 << 26;

/// Parse a LightGBM text model into a [`BoostedModel`].
pub(crate) fn import_lightgbm_text(text: &str) -> Result<BoostedModel> {
    let document = Document::parse(text)?;
    let header = &document.header;
    let n_features = header.n_features()?;
    let num_class: usize = header.number("num_class")?;
    let trees_per_iteration = match header.optional("num_tree_per_iteration") {
        Some(_) => header.number("num_tree_per_iteration")?,
        None => num_class,
    };
    // The output count sizes allocations (the intercepts, the objective's
    // per-class state) and comes from untrusted text: bound it by the parsed
    // trees, requiring at least one whole iteration (an empty model has no
    // outputs to check `num_class` against).
    if trees_per_iteration == 0 || document.trees.len() < trees_per_iteration {
        return Err(format_error(format!(
            "{} trees do not hold one whole iteration of {trees_per_iteration} trees",
            document.trees.len()
        )));
    }
    if header.values.contains_key("average_output") {
        return Err(format_error(
            "`average_output` (random forest, `boosting=rf`) models are not supported: \
             LightGBM averages their trees in predictions but sums them in raw scores and \
             SHAP values, which no single margin can express",
        ));
    }
    let objective = map_objective(
        header.optional("objective"),
        num_class,
        &document.parameters,
    )?;
    if trees_per_iteration != objective.n_outputs {
        return Err(format_error(format!(
            "`num_tree_per_iteration` {trees_per_iteration} does not match the {} outputs of \
             objective `{}`",
            objective.n_outputs,
            header.optional("objective").unwrap_or_default()
        )));
    }
    if !document.trees.len().is_multiple_of(trees_per_iteration) {
        return Err(format_error(format!(
            "{} trees do not fill whole iterations of {trees_per_iteration}",
            document.trees.len()
        )));
    }
    let mut trees = Vec::with_capacity(document.trees.len());
    for (index, fields) in document.trees.iter().enumerate() {
        let tree = convert_tree(fields, n_features)
            .map_err(|message| format_error(format!("tree {index}: {message}")))?;
        trees.push(tree);
    }
    let n_outputs = objective.n_outputs;
    // `boost_from_average` / `init_score` are already part of the first
    // iteration's trees (`Tree::AddBias`), so the intercepts are zero.
    let model = BoostedModel::from_parts(
        trees,
        Vec::new(),
        vec![0.0; n_outputs],
        ModelSpec {
            objective: ModelObjective::trained_with(&objective.objective),
            max_delta_step: objective.max_delta_step,
            num_class: objective.num_class,
            n_outputs,
            n_targets: objective.n_targets,
            n_features,
        },
    );
    model.validate_structure()?;
    Ok(model)
}

fn format_error(message: impl std::fmt::Display) -> HessboostError {
    HessboostError::model_format(format!("LightGBM model: {message}"))
}

// ---------------------------------------------------------------------------
// Document structure
// ---------------------------------------------------------------------------

/// A LightGBM text model split into its parts (`GBDT::SaveModelToString`):
/// the header, one key-value block per tree, and the training parameters.
struct Document<'a> {
    header: Header<'a>,
    trees: Vec<TreeFields<'a>>,
    /// `[key: value]` lines of the `parameters:` section.
    parameters: HashMap<&'a str, &'a str>,
}

/// The header's `key=value` lines (a bare key maps to `""`).
struct Header<'a> {
    values: HashMap<&'a str, &'a str>,
}

/// Header keys LightGBM 4.x writes; `tree` and `average_output` stand alone.
const HEADER_KEYS: &[&str] = &[
    "version",
    "num_class",
    "num_tree_per_iteration",
    "label_index",
    "max_feature_idx",
    "objective",
    "average_output",
    "feature_names",
    "monotone_constraints",
    "feature_infos",
    "tree_sizes",
];

/// Keys of a tree block (`Tree::ToString`).
const TREE_KEYS: &[&str] = &[
    "num_leaves",
    "num_cat",
    "split_feature",
    "split_gain",
    "threshold",
    "decision_type",
    "left_child",
    "right_child",
    "leaf_value",
    "leaf_weight",
    "leaf_count",
    "internal_value",
    "internal_weight",
    "internal_count",
    "cat_boundaries",
    "cat_threshold",
    "is_linear",
    "leaf_const",
    "num_features",
    "leaf_features",
    "leaf_coeff",
    "shrinkage",
];

/// One tree block's values by key.
struct TreeFields<'a> {
    values: HashMap<&'a str, &'a str>,
}

/// The lines of `text` with their byte offsets; a line ends at `\n`, and a
/// trailing `\r` is dropped.
fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut offset = 0;
    text.split_inclusive('\n').map(move |raw| {
        let start = offset;
        offset += raw.len();
        let line = raw.strip_suffix('\n').unwrap_or(raw);
        (start, line.strip_suffix('\r').unwrap_or(line))
    })
}

impl<'a> Document<'a> {
    fn parse(text: &'a str) -> Result<Self> {
        let mut lines = lines(text).peekable();
        // The model kind line, as `GBDT::SubModelName` writes it.
        match lines.by_ref().find(|(_, line)| !line.is_empty()) {
            Some((_, "tree")) => {}
            _ => {
                return Err(format_error(
                    "not a LightGBM text model (the first line is not `tree`)",
                ));
            }
        }
        let mut values = HashMap::new();
        while let Some(&(_, line)) = lines.peek() {
            if line.starts_with("Tree=") || line == "end of trees" {
                break;
            }
            lines.next();
            if line.is_empty() {
                continue;
            }
            let (key, value) = line.split_once('=').unwrap_or((line, ""));
            if !HEADER_KEYS.contains(&key) {
                return Err(format_error(format!(
                    "unknown header line `{}`",
                    clip(line)
                )));
            }
            if values.insert(key, value).is_some() {
                return Err(format_error(format!("duplicate header key `{key}`")));
            }
        }
        let header = Header { values };
        if header.optional("version") != Some(MODEL_VERSION) {
            return Err(format_error(format!(
                "model version {:?} is not LightGBM 4.x's `{MODEL_VERSION}`",
                header.optional("version").unwrap_or("(missing)")
            )));
        }

        // Tree blocks: `Tree=<i>`, then key=value lines up to a blank line.
        let mut trees = Vec::new();
        let mut starts = Vec::new();
        let end_of_trees = loop {
            let Some((offset, line)) = lines.next() else {
                return Err(format_error("truncated: no `end of trees` line"));
            };
            if line.is_empty() {
                continue;
            }
            if line == "end of trees" {
                break offset;
            }
            let expected = format!("Tree={}", trees.len());
            if line != expected {
                return Err(format_error(format!(
                    "expected `{expected}`, found `{}`",
                    clip(line)
                )));
            }
            starts.push(offset);
            let mut values = HashMap::new();
            for (_, line) in lines.by_ref() {
                if line.is_empty() {
                    break;
                }
                let Some((key, value)) = line.split_once('=') else {
                    return Err(format_error(format!(
                        "tree {}: malformed line `{}`",
                        trees.len(),
                        clip(line)
                    )));
                };
                if !TREE_KEYS.contains(&key) {
                    return Err(format_error(format!(
                        "tree {}: unknown key `{}`",
                        trees.len(),
                        clip(key)
                    )));
                }
                if values.insert(key, value).is_some() {
                    return Err(format_error(format!(
                        "tree {}: duplicate key `{key}`",
                        trees.len()
                    )));
                }
            }
            trees.push(TreeFields { values });
        };
        header.check_tree_sizes(text, &starts, end_of_trees)?;

        // Training parameters (only objective parameters are read).
        let mut parameters = HashMap::new();
        let mut in_parameters = false;
        for (_, line) in lines {
            match line {
                "parameters:" => in_parameters = true,
                "end of parameters" => break,
                _ if in_parameters => {
                    if let Some((key, value)) = line
                        .strip_prefix('[')
                        .and_then(|l| l.strip_suffix(']'))
                        .and_then(|l| l.split_once(": "))
                    {
                        parameters.insert(key, value);
                    }
                }
                _ => {}
            }
        }
        Ok(Document {
            header,
            trees,
            parameters,
        })
    }
}

/// At most 64 bytes of `line` (on a character boundary), for messages.
fn clip(line: &str) -> &str {
    let mut end = line.len().min(64);
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    &line[..end]
}

impl<'a> Header<'a> {
    fn optional(&self, key: &str) -> Option<&'a str> {
        self.values.get(key).copied()
    }

    fn required(&self, key: &str) -> Result<&'a str> {
        self.optional(key)
            .ok_or_else(|| format_error(format!("missing header key `{key}`")))
    }

    fn number<T: FromStr>(&self, key: &str) -> Result<T> {
        let value = self.required(key)?;
        value
            .trim()
            .parse()
            .map_err(|_| format_error(format!("invalid `{key}` value `{}`", clip(value))))
    }

    /// `max_feature_idx + 1`, checked against the per-feature lists (as
    /// LightGBM's loader does), which also bounds it by the text's size.
    fn n_features(&self) -> Result<usize> {
        let max_index: usize = self.number("max_feature_idx")?;
        let n_features = max_index
            .checked_add(1)
            .ok_or_else(|| format_error("`max_feature_idx` overflows"))?;
        for (key, required) in [
            ("feature_names", true),
            ("feature_infos", true),
            ("monotone_constraints", false),
        ] {
            let value = if required {
                self.required(key)?
            } else {
                match self.optional(key) {
                    Some(value) => value,
                    None => continue,
                }
            };
            let count = value.split(' ').filter(|token| !token.is_empty()).count();
            if count != n_features {
                return Err(format_error(format!(
                    "`{key}` lists {count} features, `max_feature_idx` implies {n_features}"
                )));
            }
        }
        // Parsed for validity only: the label column does not affect
        // predictions.
        let _: i64 = self.number("label_index")?;
        Ok(n_features)
    }

    /// Check `tree_sizes` (when present) against the tree blocks: LightGBM
    /// seeks each tree by these byte counts, so a file whose blocks
    /// disagree with them is one LightGBM reads differently.
    fn check_tree_sizes(&self, text: &str, starts: &[usize], end_of_trees: usize) -> Result<()> {
        let Some(sizes) = self.optional("tree_sizes") else {
            return Ok(());
        };
        let mut sizes = sizes.split(' ').filter(|token| !token.is_empty());
        for (i, &start) in starts.iter().enumerate() {
            let end = starts.get(i + 1).copied().unwrap_or(end_of_trees);
            match sizes.next().map(str::parse::<usize>) {
                Some(Ok(size)) if size == end - start => {}
                Some(Ok(size)) => {
                    // LightGBM writes `\n` line ends and seeks by these
                    // counts, so it too refuses a file converted to CRLF.
                    let crlf = if text.contains("\r\n") {
                        " (the file has CRLF line ends; LightGBM writes LF)"
                    } else {
                        ""
                    };
                    return Err(format_error(format!(
                        "tree {i} spans {} bytes but `tree_sizes` records {size}{crlf}",
                        end - start
                    )));
                }
                _ => {
                    return Err(format_error(format!(
                        "`tree_sizes` does not list the {} trees",
                        starts.len()
                    )));
                }
            }
        }
        if sizes.next().is_some() {
            return Err(format_error(format!(
                "`tree_sizes` lists more than the {} trees",
                starts.len()
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Objectives
// ---------------------------------------------------------------------------

/// The hessboost objective a LightGBM objective maps to.
struct MappedObjective {
    objective: Objective,
    /// XGBoost's `max_delta_step` for the objective: LightGBM's
    /// `poisson_max_delta_step` for `poisson`, else the objective's default.
    max_delta_step: f64,
    num_class: usize,
    n_outputs: usize,
    n_targets: usize,
}

/// A training parameter from the `parameters:` section, or LightGBM's
/// default when the section or the key is absent.
fn parameter(parameters: &HashMap<&str, &str>, key: &str, default: f64) -> Result<f64> {
    match parameters.get(key) {
        None => Ok(default),
        Some(value) => value
            .trim()
            .parse()
            .ok()
            .filter(|v: &f64| v.is_finite())
            .ok_or_else(|| {
                format_error(format!("invalid parameter `{key}` value `{}`", clip(value)))
            }),
    }
}

/// Map LightGBM's objective line (`ObjectiveFunction::ToString`: the name,
/// then `key:value` options or `sqrt`) to the hessboost objective with the
/// same prediction transform (`ConvertOutput`), or refuse it.
fn map_objective(
    line: Option<&str>,
    num_class: usize,
    parameters: &HashMap<&str, &str>,
) -> Result<MappedObjective> {
    let Some(line) = line else {
        return Err(format_error(
            "the model has no objective (trained with a custom objective); \
             hessboost cannot tell its prediction transform",
        ));
    };
    let mut tokens = line.split(' ').filter(|token| !token.is_empty());
    let name = tokens.next().unwrap_or_default();
    let mut sigmoid = None;
    let mut classes = None;
    for token in tokens {
        match token.split_once(':') {
            _ if token == "sqrt" => {
                return Err(format_error(format!(
                    "objective `{}` uses `reg_sqrt`, whose transform `sign(x) * x^2` no \
                     hessboost objective applies",
                    clip(line)
                )));
            }
            Some(("sigmoid", value)) => sigmoid = Some(value),
            Some(("num_class", value)) => classes = Some(value),
            _ => {
                return Err(format_error(format!(
                    "objective `{}` has an unknown option `{}`",
                    clip(line),
                    clip(token)
                )));
            }
        }
    }
    let multiclass = matches!(name, "multiclass" | "multiclassova");
    if multiclass {
        let classes = classes.and_then(|c| c.parse::<usize>().ok());
        if classes != Some(num_class) || num_class < 2 {
            return Err(format_error(format!(
                "objective `{}` does not match `num_class` {num_class}",
                clip(line)
            )));
        }
    } else if classes.is_some() || num_class != 1 {
        return Err(format_error(format!(
            "objective `{}` with `num_class` {num_class}",
            clip(line)
        )));
    }
    match (name, sigmoid) {
        ("binary" | "multiclassova", Some(value)) if value.parse::<f64>() == Ok(1.0) => {}
        ("binary" | "multiclassova", Some(value)) => {
            return Err(format_error(format!(
                "objective `{}`: `sigmoid` {value} scales the logistic transform, which \
                 hessboost's logistic objectives fix at 1",
                clip(line)
            )));
        }
        ("binary" | "multiclassova", None) => {
            return Err(format_error(format!(
                "objective `{}` has no `sigmoid`",
                clip(line)
            )));
        }
        (_, Some(_)) => {
            return Err(format_error(format!(
                "objective `{}` has an unknown option `sigmoid`",
                clip(line)
            )));
        }
        (_, None) => {}
    }
    let invalid = |e: HessboostError| format_error(format!("objective `{}`: {e}", clip(line)));
    let objective = match name {
        "regression" | "fair" => Objective::SquaredError(RegLoss::default()),
        "regression_l1" | "mape" => Objective::AbsoluteError,
        "huber" => Objective::PseudoHuber(
            PseudoHuber::new(parameter(parameters, "alpha", 0.9)?).map_err(invalid)?,
        ),
        "quantile" => Objective::Quantile(
            Quantiles::new([parameter(parameters, "alpha", 0.9)?]).map_err(invalid)?,
        ),
        "poisson" => Objective::Poisson,
        "gamma" => Objective::Gamma(RegLoss::default()),
        "tweedie" => Objective::Tweedie(
            Tweedie::new(parameter(parameters, "tweedie_variance_power", 1.5)?).map_err(invalid)?,
        ),
        "binary" | "multiclassova" => Objective::BinaryLogistic(RegLoss::default()),
        "cross_entropy" => Objective::RegLogistic(RegLoss::default()),
        "multiclass" => Objective::Softprob(Multiclass::new(num_class).map_err(invalid)?),
        "lambdarank" | "rank_xendcg" => Objective::RankNdcg(LambdaRank::default()),
        "cross_entropy_lambda" => {
            return Err(format_error(
                "objective `cross_entropy_lambda` predicts `log(1 + exp(x))`, a transform no \
                 hessboost objective applies",
            ));
        }
        other => {
            return Err(format_error(format!(
                "unsupported objective `{}`",
                clip(other)
            )));
        }
    };
    let max_delta_step = if name == "poisson" {
        parameter(parameters, "poisson_max_delta_step", 0.7)?
    } else {
        objective.default_max_delta_step()
    };
    let (num_class, n_outputs, n_targets) = match name {
        "multiclass" => (num_class, num_class, 1),
        // One-vs-all: an independent logistic output per class, which is a
        // multi-target `binary:logistic` model.
        "multiclassova" => (0, num_class, num_class),
        _ => (0, 1, 1),
    };
    Ok(MappedObjective {
        objective,
        max_delta_step,
        num_class,
        n_outputs,
        n_targets,
    })
}

// ---------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------

/// LightGBM's missing-value handling of a numeric split (`MissingType`).
#[derive(Clone, Copy)]
enum MissingType {
    /// `NaN` is read as `0.0`.
    None,
    /// `NaN` and values with `|x| <= kZeroThreshold` take the default
    /// direction.
    Zero,
    /// `NaN` takes the default direction.
    NaN,
}

/// The smallest `f32` greater than `t`: for every `f32` value `x`,
/// `f64::from(x) <= t` exactly when `x < f32_above(t)`, since no `f32`
/// lies strictly between `t` and it. `t` is not `NaN`.
fn f32_above(t: f64) -> f32 {
    let nearest = t as f32;
    // The nearest `f32` is above `t`, or else the next one up is the first:
    // an `f32` strictly between `t` and a nearest one above it would be
    // nearer.
    if f64::from(nearest) > t {
        nearest
    } else {
        nearest.next_up()
    }
}

/// How a numeric LightGBM split routes in hessboost: `x < split_cond` goes
/// left, missing values follow `default_left`, and with `swap` the node's
/// children trade places.
struct NumericRoute {
    split_cond: f32,
    default_left: bool,
    swap: bool,
}

/// hessboost's routing of LightGBM's `NumericalDecision` on the finite `f32`
/// values a [`DMatrix`](crate::data::DMatrix) holds (it refuses infinities
/// and reads `NaN` as missing), or why none exists.
fn numeric_split(
    threshold: f64,
    missing: MissingType,
    default_left: bool,
) -> Result<NumericRoute, String> {
    if threshold.is_nan() {
        return Err("threshold is NaN".to_string());
    }
    // `x <= threshold` exactly when `x < above`.
    let above = f32_above(threshold);
    let (cond, missing_left) = match missing {
        // `NaN` compares as `0.0`.
        MissingType::None => (above, 0.0 < above),
        MissingType::NaN => (above, default_left),
        // The band `[-z, z]` joins the default side. hessboost's single
        // threshold expresses that only when the band borders the
        // threshold's half-line on the default side.
        MissingType::Zero => {
            let z = ZERO_THRESHOLD;
            if default_left && above >= -z {
                (above.max(z.next_up()), true)
            } else if !default_left && above <= z.next_up() {
                (above.min(-z), false)
            } else {
                return Err(format!(
                    "zero_as_missing split at {threshold} sends zeros {} while values on both \
                     sides of zero go the other way, which a threshold cannot express",
                    if default_left { "left" } else { "right" }
                ));
            }
        }
    };
    if cond.is_finite() {
        return Ok(NumericRoute {
            split_cond: cond,
            default_left: missing_left,
            swap: false,
        });
    }
    // Every finite value goes left (LightGBM writes `inf` for the bin
    // above all values, splitting missing values from the rest). No finite
    // `split_cond` sends `f32::MAX` left, so the children swap and a
    // threshold no finite value is below sends everything right.
    Ok(NumericRoute {
        split_cond: -f32::MAX,
        default_left: !missing_left,
        swap: true,
    })
}

/// A whitespace-separated array field of `expected` entries. A missing
/// field is `None`; its entries are parsed as they are counted, so the
/// allocation is bounded by the text.
fn array<T: FromStr>(
    fields: &TreeFields,
    key: &str,
    expected: usize,
) -> Result<Option<Vec<T>>, String> {
    let Some(value) = fields.values.get(key) else {
        return Ok(None);
    };
    let values = value
        .split_ascii_whitespace()
        .map(|token| {
            token
                .parse()
                .map_err(|_| format!("`{key}` holds an invalid entry `{}`", clip(token)))
        })
        .collect::<Result<Vec<T>, String>>()?;
    if values.len() != expected {
        return Err(format!(
            "`{key}` has {} entries, expected {expected}",
            values.len()
        ));
    }
    Ok(Some(values))
}

fn required_array<T: FromStr>(
    fields: &TreeFields,
    key: &str,
    expected: usize,
) -> Result<Vec<T>, String> {
    array(fields, key, expected)?.ok_or_else(|| format!("missing `{key}`"))
}

fn scalar<T: FromStr>(fields: &TreeFields, key: &str) -> Result<Option<T>, String> {
    fields
        .values
        .get(key)
        .map(|value| {
            value
                .trim()
                .parse()
                .map_err(|_| format!("invalid `{key}` value `{}`", clip(value)))
        })
        .transpose()
}

/// Node counts (`leaf_count` / `internal_count`) as covers: LightGBM's
/// TreeSHAP weighs paths by data counts where hessboost reads `sum_hess`.
/// Absent counts are zero, as LightGBM reads them.
fn counts(fields: &TreeFields, key: &str, expected: usize) -> Result<Vec<f32>, String> {
    match array::<i64>(fields, key, expected)? {
        None => Ok(vec![0.0; expected]),
        Some(counts) if counts.iter().all(|&c| c >= 0) => {
            Ok(counts.into_iter().map(|c| c as f32).collect())
        }
        Some(_) => Err(format!("`{key}` holds a negative count")),
    }
}

/// The hessboost node id of LightGBM child reference `child` (internal
/// nodes `>= 0`, leaf `j` as `!j`): internal nodes keep their ids, leaves
/// follow them.
fn node_id(child: i64, n_internal: usize, n_leaves: usize) -> Result<i32, String> {
    let id = if child >= 0 {
        usize::try_from(child).ok().filter(|&c| c < n_internal)
    } else {
        usize::try_from(!child)
            .ok()
            .filter(|&j| j < n_leaves)
            .map(|j| n_internal + j)
    };
    id.and_then(|id| i32::try_from(id).ok())
        .ok_or_else(|| format!("child reference {child} is out of range"))
}

/// The categorical splits' category sets: set `i` holds the categories
/// whose bits are set in `cat_threshold[cat_boundaries[i]..cat_boundaries[i + 1]]`.
struct CategorySets {
    boundaries: Vec<usize>,
    words: Vec<u32>,
    used: Vec<bool>,
}

impl CategorySets {
    fn read(fields: &TreeFields) -> Result<Self, String> {
        let num_cat: usize = scalar(fields, "num_cat")?.ok_or("missing `num_cat`")?;
        if num_cat == 0 {
            return Ok(CategorySets {
                boundaries: vec![0],
                words: Vec::new(),
                used: Vec::new(),
            });
        }
        let boundaries_len = num_cat.checked_add(1).ok_or("`num_cat` overflows")?;
        let boundaries: Vec<usize> = required_array(fields, "cat_boundaries", boundaries_len)?;
        if boundaries[0] != 0 || !boundaries.is_sorted() {
            return Err("`cat_boundaries` does not ascend from 0".to_string());
        }
        let words = required_array(fields, "cat_threshold", boundaries[num_cat])?;
        if boundaries
            .windows(2)
            .any(|w| w[1] - w[0] > MAX_CATEGORY_WORDS)
        {
            return Err("a categorical split has categories of 2^31 or more".to_string());
        }
        Ok(CategorySets {
            boundaries,
            words,
            used: vec![false; num_cat],
        })
    }

    /// Append the categories of set `threshold` (the split's `threshold`
    /// holds the set's index) to `pool` as `node`'s set. Each set belongs to
    /// one split, as LightGBM writes them, which keeps the pool within 32
    /// categories per stored word.
    fn expand_into(
        &mut self,
        threshold: f64,
        node: &mut Node,
        pool: &mut CategoryPool,
    ) -> Result<(), String> {
        let index = (threshold >= 0.0
            && threshold.fract() == 0.0
            && threshold < self.used.len() as f64)
            .then_some(threshold as usize)
            .ok_or_else(|| format!("categorical threshold {threshold} names no category set"))?;
        if std::mem::replace(&mut self.used[index], true) {
            return Err(format!("category set {index} is used by two splits"));
        }
        let words = &self.words[self.boundaries[index]..self.boundaries[index + 1]];
        // Below 2^31: at most `MAX_CATEGORY_WORDS` words.
        let ids = words.iter().enumerate().flat_map(|(w, &word)| {
            (0..32)
                .filter(move |bit| word >> bit & 1 == 1)
                .map(move |bit| (w * 32 + bit) as u32)
        });
        pool.push_split(node, ids).map_err(|e| match e {
            PoolError::Empty => format!("category set {index} is empty"),
            PoolError::TooMany => "too many categories".to_string(),
        })
    }
}

/// Decode one tree block into a [`RegTree`] over `n_features` features.
fn convert_tree(fields: &TreeFields, n_features: usize) -> Result<RegTree, String> {
    let n_leaves: usize = scalar(fields, "num_leaves")?.ok_or("missing `num_leaves`")?;
    if n_leaves == 0 {
        return Err("`num_leaves` is 0".to_string());
    }
    let n_internal = n_leaves - 1;
    // Every array is checked against the counts before any node storage
    // (sized by them) is allocated.
    let leaf_values: Vec<f64> = required_array(fields, "leaf_value", n_leaves)?;
    let leaf_covers = counts(fields, "leaf_count", n_leaves)?;
    let _: Option<Vec<f64>> = array(fields, "leaf_weight", n_leaves)?;
    let (features, thresholds, decision_types, left, right) = if n_internal == 0 {
        Default::default()
    } else {
        (
            required_array::<i64>(fields, "split_feature", n_internal)?,
            required_array::<f64>(fields, "threshold", n_internal)?,
            array::<i64>(fields, "decision_type", n_internal)?
                .unwrap_or_else(|| vec![0; n_internal]),
            required_array::<i64>(fields, "left_child", n_internal)?,
            required_array::<i64>(fields, "right_child", n_internal)?,
        )
    };
    let gains: Vec<f32> =
        array(fields, "split_gain", n_internal)?.unwrap_or_else(|| vec![0.0; n_internal]);
    let internal_covers = counts(fields, "internal_count", n_internal)?;
    let _: Option<Vec<f64>> = array(fields, "internal_value", n_internal)?;
    let _: Option<Vec<f64>> = array(fields, "internal_weight", n_internal)?;
    // Informational: leaf values already include the learning rate.
    let _: Option<f64> = scalar(fields, "shrinkage")?;
    let mut sets = CategorySets::read(fields)?;

    let mut nodes = Vec::with_capacity(n_internal + n_leaves);
    let mut categories = CategoryPool::default();
    for i in 0..n_internal {
        let at = |message: String| format!("node {i}: {message}");
        let feature = u32::try_from(features[i])
            .ok()
            .filter(|&f| (f as usize) < n_features)
            .ok_or_else(|| at(format!("split feature {} is out of range", features[i])))?;
        let decision = u8::try_from(decision_types[i])
            .ok()
            .filter(|d| d & !KNOWN_DECISION_BITS == 0)
            .ok_or_else(|| at(format!("unknown decision_type {}", decision_types[i])))?;
        let mut node = Node::leaf(0.0, internal_covers[i]);
        node.split_feature = feature;
        let (left, right) = (
            node_id(left[i], n_internal, n_leaves).map_err(at)?,
            node_id(right[i], n_internal, n_leaves).map_err(at)?,
        );
        node.set_links(left, right);
        node.split_gain = gains[i];
        if decision & CATEGORICAL_MASK != 0 {
            // `CategoricalDecision`: NaN and negative values go right,
            // others by membership of their integer part.
            sets.expand_into(thresholds[i], &mut node, &mut categories)
                .map_err(at)?;
            node.is_categorical = true;
            node.default_left = false;
        } else {
            let missing = match decision >> 2 & 3 {
                0 => MissingType::None,
                1 => MissingType::Zero,
                2 => MissingType::NaN,
                _ => return Err(at("unknown missing type 3".to_string())),
            };
            let route = numeric_split(thresholds[i], missing, decision & DEFAULT_LEFT_MASK != 0)
                .map_err(at)?;
            node.split_cond = route.split_cond;
            node.default_left = route.default_left;
            if route.swap {
                node.set_links(right, left);
            }
        }
        nodes.push(node);
    }
    for (j, (&value, &cover)) in leaf_values.iter().zip(&leaf_covers).enumerate() {
        let value = value as f32;
        if !value.is_finite() {
            return Err(format!(
                "leaf {j} value {} is not a finite f32",
                leaf_values[j]
            ));
        }
        nodes.push(Node::leaf(value, cover));
    }
    let linear = match scalar::<u8>(fields, "is_linear")? {
        None | Some(0) => None,
        Some(1) => Some(linear_leaves(fields, n_internal, n_leaves, n_features)?),
        Some(other) => return Err(format!("`is_linear` is {other}")),
    };
    // Unchecked here: the model's `validate_structure` checks every tree.
    Ok(RegTree::from_parts(
        nodes,
        categories.finish(),
        0,
        Vec::new(),
        linear,
    ))
}

/// The per-leaf linear models of a `linear_tree` tree (`leaf_const`,
/// `num_features`, `leaf_features`, `leaf_coeff`), laid out over hessboost's
/// node ids. LightGBM predicts a leaf's model unless one of its features is
/// `NaN`, then the leaf's constant value: [`LinearLeaves`]' rule, since
/// hessboost reads `NaN` as missing.
fn linear_leaves(
    fields: &TreeFields,
    n_internal: usize,
    n_leaves: usize,
    n_features: usize,
) -> Result<LinearLeaves, String> {
    let constants: Vec<f64> =
        array(fields, "leaf_const", n_leaves)?.unwrap_or_else(|| vec![0.0; n_leaves]);
    let per_leaf: Vec<usize> =
        array(fields, "num_features", n_leaves)?.unwrap_or_else(|| vec![0; n_leaves]);
    let total = per_leaf
        .iter()
        .try_fold(0usize, |sum, &n| sum.checked_add(n))
        .filter(|&total| u32::try_from(total).is_ok())
        .ok_or("`num_features` overflows")?;
    let (features, coeffs): (Vec<i64>, Vec<f64>) = if total == 0 {
        Default::default()
    } else {
        (
            required_array(fields, "leaf_features", total)?,
            required_array(fields, "leaf_coeff", total)?,
        )
    };
    if !constants.iter().chain(&coeffs).all(|v| v.is_finite()) {
        return Err("a linear leaf coefficient is not finite".to_string());
    }
    let features = features
        .into_iter()
        .map(|f| {
            u32::try_from(f)
                .ok()
                .filter(|&f| (f as usize) < n_features)
                .ok_or_else(|| format!("linear leaf feature {f} is out of range"))
        })
        .collect::<Result<Vec<u32>, String>>()?;
    let mut offsets = vec![0u32; n_internal + 1];
    let mut end = 0u32;
    for &n in &per_leaf {
        // Within `u32`: the total fits.
        end += n as u32;
        offsets.push(end);
    }
    let mut intercepts = vec![0.0; n_internal];
    intercepts.extend(constants);
    Ok(LinearLeaves::from_parts(
        offsets, intercepts, features, coeffs,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DMatrix;
    use crate::model::Iterations;
    use serde_json::Value;

    /// LightGBM 4.7.0 models with LightGBM's own predictions on their test
    /// rows (`*.expected.json`), written by
    /// `scripts/gen_lightgbm_fixtures.py --test-data`: a binary model with
    /// categorical and `NaN`-missing splits (`categorical_binary` there),
    /// and a `linear_tree` model with missing values (`linear_tree`).
    const BINARY: &str = include_str!("../../tests/data/lightgbm-4.7.0-binary.txt");
    const BINARY_EXPECTED: &str =
        include_str!("../../tests/data/lightgbm-4.7.0-binary.expected.json");
    const LINEAR: &str = include_str!("../../tests/data/lightgbm-4.7.0-linear.txt");
    const LINEAR_EXPECTED: &str =
        include_str!("../../tests/data/lightgbm-4.7.0-linear.expected.json");

    fn floats(expected: &Value, key: &str) -> Vec<f64> {
        expected[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect()
    }

    fn assert_close(what: &str, got: &[f32], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                (f64::from(g) - w).abs() <= 1e-5 * w.abs().max(1.0),
                "{what}[{i}]: hessboost {g}, LightGBM {w}"
            );
        }
    }

    #[test]
    fn checked_in_models_predict_and_explain_like_lightgbm() {
        for (text, expected, objective) in [
            (BINARY, BINARY_EXPECTED, "binary:logistic"),
            (LINEAR, LINEAR_EXPECTED, "reg:squarederror"),
        ] {
            let expected: Value = serde_json::from_str(expected).unwrap();
            let model = BoostedModel::from_lightgbm_text(text).unwrap();
            assert_eq!(model.objective().name(), objective);
            let x: Vec<f32> = floats(&expected, "x_test")
                .iter()
                .map(|&v| v as f32)
                .collect();
            let n_cols = expected["n_cols"].as_u64().unwrap() as usize;
            let data = DMatrix::from_dense(&x, x.len() / n_cols, n_cols).unwrap();
            assert_close(
                "raw",
                model
                    .predict_margin(&data, Iterations::Best)
                    .unwrap()
                    .as_slice(),
                &floats(&expected, "raw"),
            );
            assert_close(
                "pred",
                model.predict(&data, Iterations::Best).unwrap().as_slice(),
                &floats(&expected, "pred"),
            );
            match expected["contribs"] {
                Value::Null => assert!(model.predict_contribs(&data, Iterations::Best).is_err()),
                _ => assert_close(
                    "contribs",
                    model
                        .predict_contribs(&data, Iterations::Best)
                        .unwrap()
                        .as_slice(),
                    &floats(&expected, "contribs"),
                ),
            }
            // LightGBM's leaf `j` is node `num_leaves - 1 + j`.
            let leaves = model.predict_leaf(&data, ..).unwrap();
            let expected_leaves = floats(&expected, "leaf");
            assert_eq!(leaves.width(), model.num_trees());
            assert_eq!(leaves.as_slice().len(), expected_leaves.len());
            for (row, want) in leaves
                .rows()
                .zip(expected_leaves.chunks_exact(leaves.width()))
            {
                for ((&node, &leaf), tree) in row.iter().zip(want).zip(model.trees()) {
                    assert_eq!(node as usize, tree.num_leaves() - 1 + leaf as usize);
                }
            }
            let half = expected["slice_iterations"].as_u64().unwrap() as usize;
            let sliced = model.slice(..half, 1).unwrap();
            assert_close(
                "raw_slice",
                sliced
                    .predict_margin(&data, Iterations::Best)
                    .unwrap()
                    .as_slice(),
                &floats(&expected, "raw_slice"),
            );
            let restored = BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
            assert_eq!(
                restored.predict(&data, Iterations::Best).unwrap(),
                model.predict(&data, Iterations::Best).unwrap()
            );
        }
    }

    /// A model header over one feature `x` with `classes` outputs
    /// (regression for one, else multiclass).
    fn header(classes: &str) -> String {
        let objective = if classes == "1" {
            "regression".to_string()
        } else {
            format!("multiclass num_class:{classes}")
        };
        format!(
            "tree\nversion=v4\nnum_class={classes}\nnum_tree_per_iteration={classes}\n\
             label_index=0\nmax_feature_idx=0\nobjective={objective}\nfeature_names=x\n\
             feature_infos=none\n\n"
        )
    }

    /// The block of a one-split tree on feature 0: left leaf `1`, right leaf
    /// `2`.
    fn stump_tree(threshold: &str, decision_type: u8, categories: Option<&str>) -> String {
        let cat_lines = categories.map_or(String::new(), |words| {
            let n = words.split(' ').count();
            format!("cat_boundaries=0 {n}\ncat_threshold={words}\n")
        });
        format!(
            "Tree=0\nnum_leaves=2\nnum_cat={}\nsplit_feature=0\nsplit_gain=1\n\
             threshold={threshold}\ndecision_type={decision_type}\nleft_child=-1\n\
             right_child=-2\nleaf_value=1 2\nleaf_weight=1 1\nleaf_count=1 1\n\
             internal_value=0\ninternal_weight=2\ninternal_count=2\n{cat_lines}\
             is_linear=0\nshrinkage=1\n\n\n",
            u8::from(categories.is_some())
        )
    }

    /// A one-split regression model on one feature: left leaf `1`, right
    /// leaf `2`.
    fn stump(threshold: &str, decision_type: u8, categories: Option<&str>) -> String {
        format!(
            "{}{}end of trees\n",
            header("1"),
            stump_tree(threshold, decision_type, categories)
        )
    }

    /// The leaf (`1` left, `2` right) each of `values` reaches.
    fn route(model: &str, values: &[f32]) -> Vec<f32> {
        let model = BoostedModel::from_lightgbm_text(model).unwrap();
        let data = DMatrix::from_dense(values, values.len(), 1).unwrap();
        model
            .predict_margin(&data, Iterations::Best)
            .unwrap()
            .into_vec() // one value per row
    }

    /// A header claiming `usize::MAX` classes with no trees is refused
    /// before anything is sized by the class count (it would otherwise
    /// allocate one intercept per claimed class).
    #[test]
    fn class_counts_are_bounded_by_the_parsed_trees() {
        for classes in [usize::MAX.to_string(), "3".to_string()] {
            let err =
                BoostedModel::from_lightgbm_text(&format!("{}end of trees\n", header(&classes)))
                    .unwrap_err()
                    .to_string();
            assert!(err.contains("whole iteration"), "{err}");
        }
        // A one-tree model claiming more outputs than it has trees too.
        let wide = format!(
            "{}{}end of trees\n",
            header("3"),
            stump_tree("0.5", NONE, None)
        );
        let err = BoostedModel::from_lightgbm_text(&wide)
            .unwrap_err()
            .to_string();
        assert!(err.contains("whole iteration"), "{err}");
    }

    const NONE: u8 = 0;
    const ZERO: u8 = 1 << 2;
    const NAN: u8 = 2 << 2;
    const LEFT: u8 = DEFAULT_LEFT_MASK;

    #[test]
    fn numeric_splits_route_as_lightgbm_compares_doubles() {
        // 0.1 lies between two f32 values: `x <= 0.1` holds for the lower.
        let tenth = 0.1f32;
        let below = tenth.next_down();
        assert!(f64::from(below) < 0.1 && f64::from(tenth) > 0.1);
        assert_eq!(
            route(&stump("0.1", NAN, None), &[below, tenth, f32::NAN]),
            [1.0, 2.0, 2.0]
        );
        // An f32-representable threshold includes itself.
        assert_eq!(
            route(
                &stump("0.5", NAN | LEFT, None),
                &[0.5, 0.5f32.next_up(), f32::NAN]
            ),
            [1.0, 2.0, 1.0]
        );
        // Missing type None compares NaN as 0, whatever the default bit.
        assert_eq!(
            route(&stump("-0.5", NONE | LEFT, None), &[f32::NAN, -0.5]),
            [2.0, 1.0]
        );
        assert_eq!(
            route(&stump("0.5", NONE, None), &[f32::NAN, 0.0]),
            [1.0, 1.0]
        );
        // LightGBM's `inf` bin bound sends every value left, missing ones by
        // the default bit.
        let values = [f32::MAX, -f32::MAX, 0.0, f32::NAN];
        assert_eq!(
            route(&stump("inf", NAN, None), &values),
            [1.0, 1.0, 1.0, 2.0]
        );
        assert_eq!(
            route(&stump("inf", NAN | LEFT, None), &values),
            [1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            route(&stump("-inf", NAN, None), &values),
            [2.0, 2.0, 2.0, 2.0]
        );
        assert_eq!(
            route(&stump("1e300", NONE, None), &values),
            [1.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn zero_as_missing_maps_where_zeros_join_the_default_side() {
        let z = ZERO_THRESHOLD;
        let values = [
            0.0,
            -0.0,
            z,
            -z,
            z.next_up(),
            (-z).next_down(),
            0.7,
            -1.0,
            f32::NAN,
        ];
        // Zeros default left and the threshold is above the band: the band
        // joins the left half-line.
        assert_eq!(
            route(&stump("0.5", ZERO | LEFT, None), &values),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0]
        );
        // Zeros default right below a negative threshold.
        assert_eq!(
            route(&stump("-0.5", ZERO, None), &values),
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0]
        );
        // A threshold inside the band: only values below it stay left.
        assert_eq!(
            route(&stump("0", ZERO, None), &values),
            [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
        // Zeros sent away from the small values of their own side.
        for (threshold, decision) in [("0.5", ZERO), ("-0.5", ZERO | LEFT)] {
            let error =
                BoostedModel::from_lightgbm_text(&stump(threshold, decision, None)).unwrap_err();
            assert!(error.to_string().contains("zero_as_missing"), "{error}");
        }
    }

    #[test]
    fn categorical_bitsets_route_by_integer_part() {
        // Categories 1, 3 (word 0) and 33 (word 1) go left; NaN goes right.
        let model = stump("0", CATEGORICAL_MASK | NAN, Some("10 2"));
        assert_eq!(
            route(
                &model,
                &[1.0, 3.0, 3.7, 33.0, 0.0, 2.0, 32.0, 64.0, 5e9, f32::NAN]
            ),
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        );
        let empty =
            BoostedModel::from_lightgbm_text(&stump("0", CATEGORICAL_MASK, Some("0"))).unwrap_err();
        assert!(empty.to_string().contains("empty"), "{empty}");
    }

    #[test]
    fn unsupported_or_malformed_models_are_refused() {
        // Without `tree_sizes`, which edits inside a tree would trip first.
        let sizes = &BINARY[BINARY.find("tree_sizes=").unwrap()..];
        let sizes = &sizes[..=sizes.find('\n').unwrap()];
        let base = BINARY.replacen(sizes, "", 1);
        let replace = |from: &str, to: &str| {
            assert!(base.contains(from), "{from}");
            base.replacen(from, to, 1)
        };
        // `key`'s first entry in the first tree set to `value`.
        let first_entry = |key: &str, value: &str| {
            let start = base.find(&format!("\n{key}=")).unwrap() + key.len() + 2;
            let end = start + base[start..].find([' ', '\n']).unwrap();
            format!("{}{value}{}", &base[..start], &base[end..])
        };
        let cases = [
            (
                replace("objective=", "average_output\nobjective="),
                "average_output",
            ),
            (replace("version=v4", "version=v3"), "version"),
            (
                replace("label_index=0", "label_index=0\nbest_iteration=3"),
                "unknown header",
            ),
            (replace("sigmoid:1", "sigmoid:2"), "sigmoid"),
            (
                replace("objective=binary sigmoid:1\n", ""),
                "custom objective",
            ),
            (
                replace("objective=binary sigmoid:1", "objective=regression sqrt"),
                "reg_sqrt",
            ),
            (
                BINARY.replacen("tree_sizes=782", "tree_sizes=781", 1),
                "tree_sizes",
            ),
            (BINARY.replace('\n', "\r\n"), "CRLF"),
            (
                base[..base.find("end of trees").unwrap()].to_string(),
                "end of trees",
            ),
            (
                replace("decision_type=9", "decision_type=25"),
                "decision_type",
            ),
            (first_entry("left_child", "99"), "out of range"),
            (first_entry("right_child", "-99"), "out of range"),
            (first_entry("split_feature", "6"), "split feature"),
            (first_entry("threshold", "nan"), "NaN"),
            (first_entry("leaf_value", "1e39"), "not a finite f32"),
            (replace("num_leaves=6", "num_leaves=6000000000"), "entries"),
            (replace("Tree=1", "Tree=7"), "Tree=1"),
            (replace("is_linear=0", "is_linear=2"), "is_linear"),
        ];
        for (text, needle) in cases {
            match BoostedModel::from_lightgbm_text(&text) {
                Err(HessboostError::ModelFormat(message)) => {
                    assert!(message.contains(needle), "`{needle}` not in `{message}`");
                }
                other => panic!("`{needle}`: {:?}", other.map(|_| ())),
            }
        }
    }
}
