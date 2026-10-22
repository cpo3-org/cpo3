//! Public interface for the `cpo3` conic quadratic solver.

mod central_path;
mod cone;
mod io;
mod linear_algebra;
mod linear_solvers;
mod presolve;
mod problem;
mod ruiz;
mod settings;
mod solve;

#[cfg(feature = "python")]
mod python;

pub use crate::central_path::init::WarmStart;
pub use cone::Cone;
#[cfg(feature = "mps")]
pub use io::mps;
pub use linear_algebra::{CscMatrix, MatrixError};
pub use linear_solvers::{LinearAlgebraStats, LinearSolver};
pub use problem::{Problem, ProblemError};
pub use settings::Settings;
pub use solve::{Solution, SolutionInfo, SolveError, SolverStatus, solve, solve_from};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "_core")]
fn python_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(module)
}
