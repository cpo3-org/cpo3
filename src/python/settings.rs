//! Conversion of the flat Python settings object to the Rust API.

use std::time::Duration;

use pyo3::{exceptions::PyValueError, prelude::*};

use super::parse_linear_solver;
use crate::{Initialization, Settings};

pub(super) fn from_python(values: &Bound<'_, PyAny>) -> PyResult<Settings> {
    let seconds: Option<f64> = values.getattr("time_limit")?.extract()?;
    let time_limit = seconds
        .map(|seconds| {
            Duration::try_from_secs_f64(seconds).map_err(|_| {
                PyValueError::new_err(
                    "time_limit must be finite, nonnegative, and representable as a duration",
                )
            })
        })
        .transpose()?;
    let linear_solver = values.getattr("linear_solver")?.extract::<String>()?;
    let linear_solver = parse_linear_solver(&linear_solver)?;
    let initialization = values.getattr("initialization")?.extract::<String>()?;
    let initialization = match initialization.as_str() {
        "canonical" => Initialization::Canonical,
        "cvxopt" => Initialization::Cvxopt,
        "certified_endpoint" => Initialization::CertifiedEndpoint,
        "positive_predictor" => Initialization::PositivePredictor,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown initialization {initialization:?}"
            )));
        }
    };
    let settings = Settings {
        max_iters: values.getattr("max_iters")?.extract()?,
        time_limit,
        verbose: values.getattr("verbose")?.extract()?,
        presolve: values.getattr("presolve")?.extract()?,

        tol_feas: values.getattr("tol_feas")?.extract()?,
        tol_gap_abs: values.getattr("tol_gap_abs")?.extract()?,
        tol_gap_rel: values.getattr("tol_gap_rel")?.extract()?,
        tol_infeas_abs: values.getattr("tol_infeas_abs")?.extract()?,
        tol_infeas_rel: values.getattr("tol_infeas_rel")?.extract()?,
        certificate_ktratio: values.getattr("certificate_ktratio")?.extract()?,

        force_warm_start: values.getattr("force_warm_start")?.extract()?,
        initialization,
        initial_mu: values.getattr("initial_mu")?.extract()?,
        ruiz_iters: values.getattr("ruiz_iters")?.extract()?,
        ruiz_tol: values.getattr("ruiz_tol")?.extract()?,

        linear_solver,
        threads: values.getattr("threads")?.extract()?,
        eliminate_bounds: values.getattr("eliminate_bounds")?.extract()?,
        refine_init_reg: values.getattr("refine_init_reg")?.extract()?,
        refine_iters: values.getattr("refine_iters")?.extract()?,
        refine_tol: values.getattr("refine_tol")?.extract()?,

        step_fraction: values.getattr("step_fraction")?.extract()?,
    };
    Ok(settings)
}

#[pyfunction(name = "_default_settings_match")]
fn default_settings_match(settings: &Bound<'_, PyAny>) -> PyResult<bool> {
    let rs_settings = Settings::default();
    let py_settings = from_python(settings)?;
    Ok(rs_settings == py_settings)
}

#[pyfunction(name = "_validate_settings")]
fn validate_settings(settings: &Bound<'_, PyAny>) -> PyResult<()> {
    from_python(settings)?
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(default_settings_match, module)?)?;
    module.add_function(wrap_pyfunction!(validate_settings, module)?)?;
    Ok(())
}
