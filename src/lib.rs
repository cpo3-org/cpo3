//! A sparse interior-point solver for convex linear, quadratic and
//! second-order cone programs.
//!
//! Solve `min ½ xᵀPx + cᵀx` subject to `Ax + s = b`, with positive
//! semidefinite `P` and slack `s` in a product of zero, nonnegative and
//! second-order cones. Positive-semidefinite cones are not yet supported.
//! The API is experimental.
//!
//! # Example
//!
//! ```
//! use cpo3::{Cone, CscMatrix, Problem, Settings, SolverStatus, solve};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let problem = Problem::new(
//!     Some(CscMatrix::identity(2)?),
//!     vec![-1.0, -2.0],
//!     CscMatrix::identity(2)?,
//!     vec![3.0, 4.0],
//!     vec![Cone::Nonnegative(2)],
//! )?;
//! let solution = solve(&problem, &Settings::default())?;
//! assert_eq!(solution.status(), SolverStatus::Solved);
//! for (actual, expected) in solution.x().iter().zip([1.0, 2.0]) {
//!     assert!((actual - expected).abs() < 1e-5);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Check [`Solution::status`] before interpreting the returned vectors.
//! Use [`solve_from`] and [`WarmStart`] for related problems. Compile numerical
//! workloads in release mode.
//!
//! # Features
//!
//! The default build uses Faer and needs no external numerical libraries.
//! - `mps`: read continuous LPs and QPs from MPS and QPS files.
//! - `accelerate`: Apple Accelerate on macOS 13 or later; requires `threads = 0`.
//! - `cholmod`: CHOLMOD 5.2.x or 5.3.x headers at build time and a compatible
//!   shared library at runtime.
//! - `pardiso`: Intel oneMKL shared runtime, loaded when selecting PARDISO.
//! - `python`: build the Python extension (normally managed by maturin).
//!
//! Select a backend using [`Settings::linear_solver`]. Optional runtime libraries
//! are not bundled. See the [README](https://github.com/cpo3-org/cpo3#optional-factorization-backends)
//! for installation and runtime discovery. Equality-only problems use Faer.

mod central_path;
mod cone;
mod equality;
mod io;
mod linear_algebra;
mod linear_solvers;
mod presolve;
mod problem;
mod recovery;
mod ruiz;
mod settings;
mod solution;
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
pub use solution::{Solution, SolutionInfo, SolverStatus};
pub use solve::{SolveError, solve, solve_from};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "bindings")]
fn python_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(module)
}
