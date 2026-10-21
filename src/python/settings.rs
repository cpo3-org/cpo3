//! Conversion of the flat Python settings object to the Rust API.

use std::time::Duration;

use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use super::{parse_equilibration, parse_linear_solver};
use crate::Settings;

// Scalar fields use PyO3's native conversions in both directions. Keep this
// list in the same groups as Settings; defaults and constraints stay in Rust.
macro_rules! scalar_fields {
    ($apply:ident) => {
        $apply! {
            // Run controls.
            max_iters,
            verbose,
            presolve,
            // Termination and certificates.
            tol_feas,
            tol_gap_abs,
            tol_gap_rel,
            tol_infeas_abs,
            tol_infeas_rel,
            certificate_ktratio,
            // Initialization and equilibration.
            force_warm_start,
            cvxopt_init,
            initial_mu,
            max_equilibration_passes,
            equilibration_tolerance,
            // KKT solves.
            kkt_bound_elimination,
            initial_diagonal_floor,
            max_refinement_steps,
            refinement_tolerance,
            // Outer iterations.
            step_fraction,
        }
    };
}

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
    macro_rules! extract {
        ($($field:ident),* $(,)?) => {
            Settings {
                $($field: values.getattr(stringify!($field))?.extract()?,)*
                time_limit,
                linear_solver,
                data_equilibration,
            }
        };
    }
    Ok(scalar_fields!(extract))
}

#[pyfunction(name = "_settings_defaults")]
fn defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let settings = Settings::default();
    let values = PyDict::new(py);
    macro_rules! insert {
        ($($field:ident),* $(,)?) => {
            $(values.set_item(stringify!($field), settings.$field)?;)*
        };
    }
    scalar_fields!(insert);
    values.set_item(
        "time_limit",
        settings.time_limit.map(|limit| limit.as_secs_f64()),
    )?;
    values.set_item("linear_solver", settings.linear_solver.as_str())?;
    values.set_item("data_equilibration", settings.data_equilibration.as_str())?;
    Ok(values)
}

#[pyfunction(name = "_validate_settings")]
fn validate(settings: &Bound<'_, PyAny>) -> PyResult<()> {
    from_python(settings)?
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(defaults, module)?)?;
    module.add_function(wrap_pyfunction!(validate, module)?)?;
    Ok(())
}
