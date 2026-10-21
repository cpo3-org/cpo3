//! Conversion of the flat Python settings object to the Rust API.

use std::time::Duration;

use pyo3::{exceptions::PyValueError, prelude::*};

use super::{parse_equilibration, parse_linear_solver};
use crate::Settings;

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
    let data_equilibration = values.getattr("data_equilibration")?.extract::<String>()?;
    let data_equilibration = parse_equilibration(&data_equilibration)?;
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
        cvxopt_init: values.getattr("cvxopt_init")?.extract()?,
        initial_mu: values.getattr("initial_mu")?.extract()?,
        data_equilibration,
        max_equilibration_passes: values.getattr("max_equilibration_passes")?.extract()?,
        equilibration_tolerance: values.getattr("equilibration_tolerance")?.extract()?,

        linear_solver,
        kkt_bound_elimination: values.getattr("kkt_bound_elimination")?.extract()?,
        initial_diagonal_floor: values.getattr("initial_diagonal_floor")?.extract()?,
        max_refinement_steps: values.getattr("max_refinement_steps")?.extract()?,
        refinement_tolerance: values.getattr("refinement_tolerance")?.extract()?,

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
