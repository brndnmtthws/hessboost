//! Training parameters from a Python mapping with XGBoost's names.
//!
//! Every key must name a [`TrainingParams`] field (its serde name, which is
//! XGBoost's), an XGBoost alias of one, or an XGBoost option hessboost
//! implements at one setting only (accepted at exactly that value). Anything
//! else is refused, never ignored.

use crate::errors::{OrRaise, refuse};
use hessboost::config::{BoosterKind, TrainingParams};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyString, PyTuple};
use serde_json::{Map, Number, Value};

/// XGBoost aliases and the field each sets.
const ALIASES: &[(&str, &str)] = &[
    ("learning_rate", "eta"),
    ("min_split_loss", "gamma"),
    ("reg_lambda", "lambda"),
    ("reg_alpha", "alpha"),
    ("random_state", "seed"),
    ("n_jobs", "nthread"),
];

/// XGBoost options hessboost implements at one setting only (the crate
/// docs' "Not implemented"): accepted at exactly this value.
const FIXED: &[(&str, &str)] = &[
    ("updater", "\"coord_descent\""),
    ("feature_selector", "\"cyclic\""),
    ("lambdarank_pair_method", "\"topk\""),
    ("max_cat_to_onehot", "4"),
    ("max_cat_threshold", "64"),
];

/// Validated training parameters.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct Params {
    pub(crate) inner: TrainingParams,
}

/// The canonical field names: the serde names of every `TrainingParams`
/// field except `missing`, which belongs to the `DMatrix`.
fn field_names() -> PyResult<Vec<String>> {
    let defaults = serde_json::to_value(TrainingParams::default())
        .map_err(|error| refuse(format!("cannot list the training parameters: {error}")))?;
    let Value::Object(fields) = defaults else {
        return Err(refuse("training parameters are not a JSON object"));
    };
    Ok(fields.into_iter().map(|(name, _)| name).collect())
}

/// Levenshtein distance, for suggesting a parameter name.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substituted = previous + usize::from(ca != *cb);
            previous = row[j + 1];
            row[j + 1] = substituted.min(row[j] + 1).min(previous + 1);
        }
    }
    row[b.len()]
}

fn unknown(key: &str, fields: &[String]) -> PyErr {
    let candidates = fields
        .iter()
        .map(String::as_str)
        .chain(ALIASES.iter().map(|(alias, _)| *alias))
        .chain(FIXED.iter().map(|(name, _)| *name));
    let best = candidates
        .map(|name| (distance(key, name), name))
        .min_by_key(|(d, name)| (*d, *name))
        .filter(|(d, _)| *d <= (key.len() / 3).max(1));
    match best {
        Some((_, name)) => refuse(format!(
            "unknown parameter `{key}` (did you mean `{name}`?)"
        )),
        None => refuse(format!("unknown parameter `{key}`")),
    }
}

/// A JSON value from plain Python data (`None`, `bool`, `int`, `float`,
/// `str`, lists and tuples; numpy scalars and arrays through `tolist`).
fn to_value(key: &str, object: &Bound<'_, PyAny>) -> PyResult<Value> {
    if object.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(value) = object.cast::<PyBool>() {
        return Ok(Value::Bool(value.is_true()));
    }
    if let Ok(value) = object.cast::<PyInt>() {
        if let Ok(signed) = value.extract::<i64>() {
            return Ok(Value::Number(signed.into()));
        }
        if let Ok(unsigned) = value.extract::<u64>() {
            return Ok(Value::Number(unsigned.into()));
        }
        return Err(refuse(format!(
            "parameter `{key}`: {value} does not fit in 64 bits"
        )));
    }
    if let Ok(value) = object.cast::<PyFloat>() {
        let value = value.value();
        return Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| refuse(format!("parameter `{key}` must be finite, got {value}")));
    }
    if let Ok(value) = object.cast::<PyString>() {
        return Ok(Value::String(value.to_str()?.to_owned()));
    }
    if let Ok(list) = object.cast::<PyList>() {
        return list.iter().map(|item| to_value(key, &item)).collect();
    }
    if let Ok(tuple) = object.cast::<PyTuple>() {
        return tuple.iter().map(|item| to_value(key, &item)).collect();
    }
    if object.hasattr("tolist")? {
        let plain = object.call_method0("tolist")?;
        if !plain.is(object) {
            return to_value(key, &plain);
        }
    }
    Err(PyTypeError::new_err(format!(
        "parameter `{key}` has unsupported type {}",
        object.get_type().name()?
    )))
}

/// XGBoost's spellings of the fields whose serde form differs.
fn normalize(key: &str, value: Value) -> PyResult<Value> {
    Ok(match (key, value) {
        ("eval_metric", Value::String(name)) => Value::Array(vec![Value::String(name)]),
        ("quantile_alpha" | "expectile_alpha", Value::Number(alpha)) => {
            Value::Array(vec![Value::Number(alpha)])
        }
        ("monotone_constraints", Value::String(text)) => {
            let inner = text.trim().trim_start_matches('(').trim_end_matches(')');
            let entries = inner
                .split(',')
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(|entry| {
                    entry
                        .parse::<i64>()
                        .map(|v| Value::Number(v.into()))
                        .map_err(|_| {
                            refuse(format!(
                                "parameter `monotone_constraints`: bad entry `{entry}` in {text:?}"
                            ))
                        })
                })
                .collect::<PyResult<Vec<_>>>()?;
            normalize(key, Value::Array(entries))?
        }
        ("monotone_constraints", Value::Array(entries)) => entries
            .into_iter()
            .map(|entry| match entry.as_i64() {
                Some(1) => Ok(Value::from("increasing")),
                Some(-1) => Ok(Value::from("decreasing")),
                Some(0) => Ok(Value::from("none")),
                _ if entry.is_string() => Ok(entry),
                _ => Err(refuse(format!(
                    "parameter `monotone_constraints`: entries must be -1, 0 or 1, got {entry}"
                ))),
            })
            .collect::<PyResult<Value>>()?,
        ("interaction_constraints", Value::String(text)) => serde_json::from_str(&text)
            .map_err(|error| {
                refuse(format!(
                    "parameter `interaction_constraints`: {text:?} is not a list of index lists: {error}"
                ))
            })?,
        (_, value) => value,
    })
}

impl Params {
    fn parse(mapping: &Bound<'_, PyMapping>) -> PyResult<TrainingParams> {
        let fields = field_names()?;
        let mut settings = Map::new();
        let mut updater = false;
        for item in mapping.items()?.iter() {
            let (key, object): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.extract()?;
            let key: String = key.extract().map_err(|_| {
                PyTypeError::new_err(format!("parameter names must be str, got {key:?}"))
            })?;
            let value = to_value(&key, &object)?;
            if let Some((_, fixed)) = FIXED.iter().find(|(name, _)| *name == key) {
                let expected: Value = serde_json::from_str(fixed)
                    .map_err(|error| refuse(format!("bad fixed setting: {error}")))?;
                if value != expected {
                    return Err(refuse(format!(
                        "parameter `{key}` is only implemented as {fixed}, got {value}"
                    )));
                }
                updater |= key == "updater";
                continue;
            }
            let canonical = ALIASES
                .iter()
                .find(|(alias, _)| *alias == key)
                .map_or(key.as_str(), |(_, field)| field);
            if canonical == "missing" {
                return Err(refuse(
                    "`missing` is not a training parameter: pass it to DMatrix(data, missing=...)",
                ));
            }
            if !fields.iter().any(|field| field == canonical) {
                return Err(unknown(&key, &fields));
            }
            let value = normalize(canonical, value)?;
            // Deserialized alone first, so a type error names its key.
            let single = Value::Object(Map::from_iter([(canonical.to_owned(), value.clone())]));
            serde_json::from_value::<TrainingParams>(single)
                .map_err(|error| refuse(format!("parameter `{key}`: {error}")))?;
            if settings.insert(canonical.to_owned(), value).is_some() {
                return Err(refuse(format!(
                    "parameter `{canonical}` is set twice (through an alias)"
                )));
            }
        }
        let params: TrainingParams = serde_json::from_value(Value::Object(settings))
            .map_err(|error| refuse(format!("invalid parameters: {error}")))?;
        if updater && params.booster != BoosterKind::GbLinear {
            return Err(refuse(
                "parameter `updater`: `coord_descent` is gblinear's updater; tree boosters take no `updater`",
            ));
        }
        params.validate().or_raise()?;
        Ok(params)
    }
}

/// A Python object from a JSON value.
fn to_python<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(value) => PyBool::new(py, *value).to_owned().into_any(),
        Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                value.into_pyobject(py)?.into_any()
            } else if let Some(value) = number.as_i64() {
                value.into_pyobject(py)?.into_any()
            } else {
                PyFloat::new(py, number.as_f64().unwrap_or(f64::NAN)).into_any()
            }
        }
        Value::String(value) => PyString::new(py, value).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(to_python(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, to_python(py, item)?)?;
            }
            dict.into_any()
        }
    })
}

#[pymethods]
impl Params {
    /// Parses and validates `params`, a mapping of XGBoost parameter names.
    #[new]
    fn new(params: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mapping = params.cast::<PyMapping>().map_err(|_| {
            PyTypeError::new_err(format!(
                "params must be a mapping, got {}",
                params
                    .get_type()
                    .name()
                    .map_or_else(|_| "object".into(), |n| n.to_string())
            ))
        })?;
        Ok(Self {
            inner: Self::parse(mapping)?,
        })
    }

    /// Every setting under its canonical name, monotone constraints as
    /// `-1`/`0`/`1`.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut value = serde_json::to_value(&self.inner)
            .map_err(|error| refuse(format!("cannot serialize the parameters: {error}")))?;
        if let Value::Object(map) = &mut value {
            map.remove("missing");
            if let Some(Value::Array(entries)) = map.get_mut("monotone_constraints") {
                for entry in entries {
                    *entry = Value::from(match entry.as_str() {
                        Some("increasing") => 1,
                        Some("decreasing") => -1,
                        _ => 0,
                    });
                }
            }
        }
        to_python(py, &value)
    }
}
