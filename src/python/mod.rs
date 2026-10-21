mod settings;

#[cfg(feature = "mps")]
use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
#[cfg(feature = "mps")]
use pyo3::exceptions::PyOSError;
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyAny,
};

use crate::{
    Cone, CscMatrix, LinearSolver, Problem, Solution, SolutionInfo, SolveError, WarmStart,
    solve_from,
};

type PyCscParts<'py> = (
    usize,
    usize,
    PyReadonlyArray1<'py, usize>,
    PyReadonlyArray1<'py, usize>,
    PyReadonlyArray1<'py, f64>,
);

/// Copy borrowed NumPy buffers directly into the final Rust-owned matrix.
fn matrix_from_parts(parts: PyCscParts<'_>) -> PyResult<CscMatrix> {
    let (rows, columns, pointers, indices, values) = parts;
    CscMatrix::new(
        rows,
        columns,
        array_slice(&pointers, "CSC column pointers")?.to_vec(),
        array_slice(&indices, "CSC row indices")?.to_vec(),
        array_slice(&values, "CSC values")?.to_vec(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn matrix_parts(py: Python<'_>, matrix: &CscMatrix) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    (
        matrix.rows(),
        matrix.columns(),
        matrix.column_pointers().to_vec().into_pyarray(py),
        matrix.row_indices().to_vec().into_pyarray(py),
        matrix.values().to_vec().into_pyarray(py),
    )
        .into_py_any(py)
}

fn array_slice<'a, T: numpy::Element>(
    array: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))
}

#[pyclass(name = "_Problem", module = "cpo3._core", frozen)]
pub(crate) struct PyProblem {
    inner: Problem,
}

#[pymethods]
impl PyProblem {
    #[new]
    #[pyo3(signature = (p, c, a, b, cones))]
    fn new(
        p: Option<PyCscParts<'_>>,
        c: PyReadonlyArray1<'_, f64>,
        a: PyCscParts<'_>,
        b: PyReadonlyArray1<'_, f64>,
        cones: Vec<(String, usize)>,
    ) -> PyResult<Self> {
        let p = p.map(matrix_from_parts).transpose()?;
        let a = matrix_from_parts(a)?;
        let c = array_slice(&c, "c")?.to_vec();
        let b = array_slice(&b, "b")?.to_vec();
        let cones = cones
            .into_iter()
            .map(|(name, parameter)| parse_cone(&name, parameter))
            .collect::<PyResult<Vec<_>>>()?;
        let inner = Problem::new(p, c, a, b, cones)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn variable_count(&self) -> usize {
        self.inner.variable_count()
    }

    #[getter]
    fn constraint_count(&self) -> usize {
        self.inner.constraint_count()
    }

    #[getter]
    fn b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.b().to_vec().into_pyarray(py)
    }

    #[getter]
    fn c<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.c().to_vec().into_pyarray(py)
    }

    /// `(rows, columns, pointers, indices, values)` for a CSC matrix.
    fn _a_parts<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        matrix_parts(py, self.inner.a())
    }

    fn _p_parts<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        self.inner.p().map(|p| matrix_parts(py, p)).transpose()
    }

    #[getter]
    fn cones(&self) -> Vec<(&'static str, usize)> {
        self.inner
            .cones()
            .iter()
            .map(|cone| (cone.name(), cone.parameter()))
            .collect()
    }
}

fn parse_linear_solver(value: &str) -> PyResult<LinearSolver> {
    match value {
        "faer" => Ok(LinearSolver::Faer),
        "feral" => Ok(LinearSolver::Feral),
        _ => Err(PyValueError::new_err(format!(
            "unknown linear solver {value:?}"
        ))),
    }
}

fn parse_cone(name: &str, parameter: usize) -> PyResult<Cone> {
    match name {
        "zero" => Ok(Cone::Zero(parameter)),
        "nonnegative" => Ok(Cone::Nonnegative(parameter)),
        "second_order" => Ok(Cone::SecondOrder(parameter)),
        "positive_semidefinite" => Ok(Cone::PositiveSemidefinite(parameter)),
        _ => Err(PyValueError::new_err(format!("unknown cone type {name:?}"))),
    }
}

#[pyclass(name = "_SolutionInfo", module = "cpo3._core", frozen)]
struct PySolutionInfo {
    inner: SolutionInfo,
}

#[pymethods]
impl PySolutionInfo {
    #[getter]
    fn iters(&self) -> usize {
        self.inner.iters
    }

    #[getter]
    fn solve_time(&self) -> f64 {
        self.inner.solve_time.as_secs_f64()
    }

    #[getter]
    fn pobj(&self) -> f64 {
        self.inner.pobj
    }

    #[getter]
    fn dobj(&self) -> f64 {
        self.inner.dobj
    }

    #[getter]
    fn pres(&self) -> f64 {
        self.inner.pres
    }

    #[getter]
    fn dres(&self) -> f64 {
        self.inner.dres
    }

    #[getter]
    fn comp(&self) -> f64 {
        self.inner.comp
    }

    #[getter]
    fn warm_start_score(&self) -> Option<f64> {
        self.inner.warm_start_score
    }

    #[getter]
    fn warm_start_used(&self) -> Option<bool> {
        self.inner.warm_start_used
    }

    #[getter]
    fn cvxopt_init_used(&self) -> Option<bool> {
        self.inner.cvxopt_init_used
    }

    #[getter]
    fn presolve_rows_removed(&self) -> usize {
        self.inner.presolve_rows_removed
    }

    #[getter]
    fn presolve_columns_removed(&self) -> usize {
        self.inner.presolve_columns_removed
    }

    #[getter]
    fn presolve_time(&self) -> f64 {
        self.inner.presolve_time.as_secs_f64()
    }

    #[getter]
    fn presolve_parallel_comparisons(&self) -> Option<usize> {
        self.inner.presolve_parallel_comparisons
    }

    #[getter]
    fn primary_solves(&self) -> usize {
        self.inner.linear_algebra.primary_solves
    }

    #[getter]
    fn refinement_accurate(&self) -> usize {
        self.inner.linear_algebra.refinement_accurate
    }

    #[getter]
    fn refinement_forecast(&self) -> usize {
        self.inner.linear_algebra.refinement_forecast
    }

    #[getter]
    fn refinement_divergence(&self) -> usize {
        self.inner.linear_algebra.refinement_divergence
    }

    #[getter]
    fn refinement_budget(&self) -> usize {
        self.inner.linear_algebra.refinement_budget
    }

    #[getter]
    fn refinement_nonfinite(&self) -> usize {
        self.inner.linear_algebra.refinement_nonfinite
    }

    #[getter]
    fn clamp_increases(&self) -> usize {
        self.inner.linear_algebra.clamp_increases
    }

    #[getter]
    fn clamp_decreases(&self) -> usize {
        self.inner.linear_algebra.clamp_decreases
    }

    #[getter]
    fn refinement_clamp_limited(&self) -> usize {
        self.inner.linear_algebra.refinement_clamp_limited
    }

    #[getter]
    fn refinement_solve_limited(&self) -> usize {
        self.inner.linear_algebra.refinement_solve_limited
    }

    #[getter]
    fn refinement_ambiguous(&self) -> usize {
        self.inner.linear_algebra.refinement_ambiguous
    }

    #[getter]
    fn refinement_solves(&self) -> usize {
        self.inner.linear_algebra.refinement_solves
    }

    #[getter]
    fn refinement_settled(&self) -> usize {
        self.inner.linear_algebra.refinement_settled
    }

    #[getter]
    fn refinement_failures(&self) -> usize {
        self.inner.linear_algebra.refinement_failures
    }

    #[getter]
    fn factorizations(&self) -> usize {
        self.inner.linear_algebra.factorizations
    }

    #[getter]
    fn inertia_failures(&self) -> usize {
        self.inner.linear_algebra.inertia_failures
    }

    #[getter]
    fn linear_solver(&self) -> &'static str {
        self.inner.linear_solver.as_str()
    }

    #[getter]
    fn eliminate_bounds(&self) -> bool {
        self.inner.eliminate_bounds
    }
}

#[pyclass(name = "_Solution", module = "cpo3._core", frozen)]
struct PySolution {
    inner: Solution,
}

#[pymethods]
impl PySolution {
    #[getter]
    fn status(&self) -> &'static str {
        self.inner.status().as_str()
    }

    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.x().to_vec().into_pyarray(py)
    }

    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.y().to_vec().into_pyarray(py)
    }

    #[getter]
    fn s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.s().to_vec().into_pyarray(py)
    }

    #[getter]
    fn info(&self) -> PySolutionInfo {
        PySolutionInfo {
            inner: self.inner.info().clone(),
        }
    }
}

#[pyfunction(name = "_solve")]
#[pyo3(signature = (problem, settings, *, warm_x = None, warm_y = None, warm_s = None))]
fn solve_py(
    py: Python<'_>,
    problem: &PyProblem,
    settings: &Bound<'_, PyAny>,
    warm_x: Option<PyReadonlyArray1<'_, f64>>,
    warm_y: Option<PyReadonlyArray1<'_, f64>>,
    warm_s: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<PySolution> {
    let settings = settings::from_python(settings)?;
    // The frozen Python problem stays borrowed for this synchronous detach.
    let problem = &problem.inner;
    let warm = match (warm_x, warm_y, warm_s) {
        (Some(x), Some(y), Some(s)) => Some(WarmStart {
            x: array_slice(&x, "warm_start x")?.to_vec(),
            y: array_slice(&y, "warm_start y")?.to_vec(),
            s: array_slice(&s, "warm_start s")?.to_vec(),
        }),
        (None, None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "warm_start requires all of x, y, and s",
            ));
        }
    };

    match py.detach(move || solve_from(problem, &settings, warm.as_ref())) {
        Ok(solution) => Ok(PySolution { inner: solution }),
        Err(SolveError::InvalidSetting(error)) => Err(PyValueError::new_err(error)),
        Err(SolveError::Unimplemented(error)) => Err(PyNotImplementedError::new_err(error)),
        Err(error) => Err(PyRuntimeError::new_err(error.to_string())),
    }
}

#[cfg(feature = "mps")]
#[pyfunction(name = "_read_mps")]
#[pyo3(signature = (path, *, first_named_set = false))]
fn read_mps_py(py: Python<'_>, path: PathBuf, first_named_set: bool) -> PyResult<PyProblem> {
    let options = crate::mps::MpsOptions { first_named_set };
    match py.detach(move || crate::mps::read_with(path, options)) {
        Ok(inner) => Ok(PyProblem { inner }),
        Err(error @ crate::mps::MpsError::Read { .. }) => {
            Err(PyOSError::new_err(error.to_string()))
        }
        Err(error) => Err(PyValueError::new_err(error.to_string())),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    settings::register(module)?;
    module.add_class::<PyProblem>()?;
    module.add_class::<PySolution>()?;
    module.add_class::<PySolutionInfo>()?;
    module.add_function(wrap_pyfunction!(solve_py, module)?)?;
    #[cfg(feature = "mps")]
    module.add_function(wrap_pyfunction!(read_mps_py, module)?)?;
    Ok(())
}
