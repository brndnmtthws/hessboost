//! Training parameters from a Python mapping with XGBoost's names.
//!
//! The keys and values go through the crate's XGBoost boundary,
//! [`TrainingParams::from_xgboost`]: XGBoost's names, aliases, and value
//! spellings, and the XGBoost options hessboost implements at one setting
//! only. Anything else is refused, never ignored.

use crate::errors::{OrRaise, refuse};
use hessboost::config::TrainingParams;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyString, PyTuple};
use serde_json::{Number, Value};

/// Validated training parameters.
#[pyclass(frozen, module = "hessboost._hessboost")]
pub struct Params {
    pub(crate) inner: TrainingParams,
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

impl Params {
    fn parse(mapping: &Bound<'_, PyMapping>) -> PyResult<TrainingParams> {
        let mut settings = Vec::new();
        for item in mapping.items()?.iter() {
            let (key, object): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.extract()?;
            let key: String = key.extract().map_err(|_| {
                PyTypeError::new_err(format!("parameter names must be str, got {key:?}"))
            })?;
            if key == "missing" {
                return Err(refuse(
                    "`missing` is not a training parameter: pass it to DMatrix(data, missing=...)",
                ));
            }
            let value = to_value(&key, &object)?;
            settings.push((key, value));
        }
        TrainingParams::from_xgboost(settings).or_raise()
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

    /// Every setting under its canonical XGBoost name, as
    /// `TrainingParams::to_xgboost` gives it.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = Value::Object(self.inner.to_xgboost().or_raise()?);
        to_python(py, &value)
    }
}
