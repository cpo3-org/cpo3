//! Public interface for the `cpo3` conic quadratic solver.

mod cone;
mod embedding;
mod init;
mod io;
mod josephy;
mod linear_algebra;
mod linear_solvers;
mod mehrotra;
mod presolve;
mod problem;
mod ruiz;
mod settings;
mod solve;

#[cfg(feature = "python")]
mod python;

pub use cone::Cone;
pub use init::WarmStart;
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
