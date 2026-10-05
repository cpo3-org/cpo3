//! Public interface for the `cpo3` conic quadratic solver.

mod cone;
mod data_equilibration;
mod embedding;
mod equilibration;
mod governor;
mod init;
mod josephy;
mod linear_solver;
#[allow(dead_code)]
mod linear_system;
mod matrix;
mod mehrotra;
#[cfg(feature = "mps")]
pub mod mps;
mod presolve;
mod problem;
mod progress;
mod settings;
mod solution;

#[cfg(feature = "python")]
mod python;

pub use cone::Cone;
pub use equilibration::Equilibration;
pub use init::WarmStart;
pub use linear_solver::LinearSolver;
pub use linear_system::LinearAlgebraStats;
pub use matrix::{CscMatrix, MatrixError};
pub use problem::{Problem, ProblemError};
pub use settings::Settings;
pub use solution::{
    IterationStats, Solution, SolutionInfo, SolveError, SolverStatus, solve, solve_from,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "_core")]
fn python_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(module)
}
