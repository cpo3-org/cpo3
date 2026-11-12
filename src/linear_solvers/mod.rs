//! Factorization adapters used by the optimization algorithm.

#[cfg(all(feature = "accelerate", target_os = "macos"))]
mod accelerate;
#[cfg(feature = "cholmod")]
mod cholmod;
mod faer;
#[cfg(feature = "pardiso")]
mod pardiso;

#[cfg(all(feature = "accelerate", target_os = "macos"))]
use self::accelerate::AccelerateFactorization;
#[cfg(feature = "cholmod")]
use self::cholmod::CholmodFactorization;
use self::faer::FaerFactorization;
#[cfg(feature = "pardiso")]
use self::pardiso::PardisoFactorization;
use std::fmt;
use thiserror::Error;

use crate::{CscMatrix, Settings};

#[cfg(any(feature = "cholmod", feature = "pardiso"))]
fn library_error(error: libloading::Error) -> String {
    use std::error::Error;
    error.source().unwrap_or(&error).to_string()
}

/// Linear-system implementation used by the numerical solver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LinearSolver {
    #[default]
    Faer,
    Accelerate,
    Cholmod,
    Pardiso,
}

impl LinearSolver {
    /// Whether this build and runtime provide the selected backend.
    pub fn is_available(self) -> bool {
        match self {
            Self::Faer => true,
            #[cfg(feature = "cholmod")]
            Self::Cholmod => cholmod::available(),
            #[cfg(not(feature = "cholmod"))]
            Self::Cholmod => false,
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Self::Accelerate => accelerate::available(),
            #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
            Self::Accelerate => false,
            #[cfg(feature = "pardiso")]
            Self::Pardiso => pardiso::available(),
            #[cfg(not(feature = "pardiso"))]
            Self::Pardiso => false,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Faer => "faer",
            Self::Accelerate => "accelerate",
            Self::Cholmod => "cholmod",
            Self::Pardiso => "pardiso",
        }
    }
}

impl fmt::Display for LinearSolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

// Per-pass contraction factor above which refinement may stop (no halving
// over two passes).
const STALL_RATE: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// How KKT systems are factorized and solved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KktSettings {
    pub(crate) linear_solver: LinearSolver,
    pub(crate) threads: usize,
    /// Fail a solve whose refinement diverges or stalls; otherwise return the
    /// best iterate seen.
    pub(crate) strict_refinement: bool,
    /// Fold nonnegative singleton rows of `A` into the `(1,1)` block.
    pub(crate) eliminate_bound_rows: bool,
    pub(crate) diagonal_floor: f64,
    pub(crate) refine_iters: usize,
    pub(crate) refine_tol: f64,
}

impl KktSettings {
    #[cfg(test)]
    pub(crate) fn new(linear_solver: LinearSolver) -> Self {
        Self::from(&Settings {
            linear_solver,
            ..Settings::default()
        })
    }
}

impl From<&Settings> for KktSettings {
    fn from(settings: &Settings) -> Self {
        Self {
            linear_solver: settings.linear_solver,
            threads: settings.threads,
            strict_refinement: false,
            eliminate_bound_rows: settings.eliminate_bounds,
            diagonal_floor: settings.refine_init_reg,
            refine_iters: settings.refine_iters,
            refine_tol: settings.refine_tol,
        }
    }
}

/// Linear-algebra counters accumulated over one solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearAlgebraStats {
    /// Numeric factorization attempts, including failures.
    pub factorizations: usize,
    /// Initial linear solves attempted for KKT right-hand sides, including failures.
    /// Includes initialization, retries, and equality-only LSMR preconditioner
    /// applications, but excludes refinement corrections.
    pub primary_solves: usize,
    /// Correction linear solves attempted by iterative refinement, including failures.
    /// Excludes the initial solve for each right-hand side and residual evaluations.
    pub refinement_solves: usize,
    /// Iterations of the equality-only preconditioned KKT least-squares solve.
    pub lsmr_iterations: usize,
    /// Refinements that failed (strict stall/divergence, or singular).
    pub refinement_failures: usize,
    /// Internal diagnostics used to adjust the diagonal clamp between steps.
    pub(crate) refinement_clamp_limited: usize,
    pub(crate) refinement_solve_limited: usize,
    /// Refinements that returned their best iterate after stalling or diverging.
    pub(crate) refinement_settled: usize,
}

impl LinearAlgebraStats {
    pub(crate) fn absorb(&mut self, other: Self) {
        self.factorizations += other.factorizations;
        self.primary_solves += other.primary_solves;
        self.refinement_solves += other.refinement_solves;
        self.lsmr_iterations += other.lsmr_iterations;
        self.refinement_failures += other.refinement_failures;
        self.refinement_clamp_limited += other.refinement_clamp_limited;
        self.refinement_solve_limited += other.refinement_solve_limited;
        self.refinement_settled += other.refinement_settled;
    }
}

/// A reusable symmetric factorization workspace. Numerical factors are valid
/// only after a successful factorization attempt. The backend owns matrix
/// storage; the wrapper retains the values of the entries that change
/// between factorizations for subsequent KKT updates.
pub(crate) struct Factorization {
    /// Values of the variable entries, in the order of their positions.
    entries: Vec<f64>,
    inner: FactorizationInner,
    settings: KktSettings,
    stats: LinearAlgebraStats,
}

enum FactorizationInner {
    Faer(Box<FaerFactorization>),
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    Accelerate(Box<AccelerateFactorization>),
    #[cfg(feature = "cholmod")]
    Cholmod(Box<CholmodFactorization>),
    #[cfg(feature = "pardiso")]
    Pardiso(Box<PardisoFactorization>),
}

impl Factorization {
    /// Prepare and retain backend storage before attempting numeric
    /// factorization. `variable_positions` lists the positions in the
    /// matrix's value storage of the entries that later updates rewrite.
    pub(crate) fn prepare(
        matrix: CscMatrix,
        variable_positions: &[usize],
        settings: &KktSettings,
    ) -> Result<Self, LinearSystemError> {
        validate_matrix(&matrix)?;
        if variable_positions
            .iter()
            .any(|&position| position >= matrix.values().len())
        {
            return Err(LinearSystemError::Factorization(
                "variable KKT entry lies outside the matrix storage".into(),
            ));
        }
        let entries = variable_positions
            .iter()
            .map(|&position| matrix.values()[position])
            .collect();
        let inner = match settings.linear_solver {
            LinearSolver::Faer => FactorizationInner::Faer(Box::new(FaerFactorization::new(
                matrix,
                variable_positions,
                settings.threads,
            )?)),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            LinearSolver::Accelerate => FactorizationInner::Accelerate(Box::new(
                AccelerateFactorization::new(matrix, variable_positions, settings.threads)?,
            )),
            #[cfg(feature = "cholmod")]
            LinearSolver::Cholmod => FactorizationInner::Cholmod(Box::new(
                CholmodFactorization::new(matrix, variable_positions, settings.threads)?,
            )),
            #[cfg(feature = "pardiso")]
            LinearSolver::Pardiso => FactorizationInner::Pardiso(Box::new(
                PardisoFactorization::new(matrix, variable_positions, settings.threads)?,
            )),
            #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
            LinearSolver::Accelerate => {
                return Err(LinearSystemError::Unavailable(
                    "Accelerate requires macOS and a cpo3 installation with Accelerate support"
                        .into(),
                ));
            }
            #[cfg(not(feature = "pardiso"))]
            LinearSolver::Pardiso => {
                return Err(LinearSystemError::Unavailable(
                    "This cpo3 installation does not include PARDISO support".into(),
                ));
            }
            #[cfg(not(feature = "cholmod"))]
            LinearSolver::Cholmod => {
                return Err(LinearSystemError::Unavailable(
                    "This cpo3 installation does not include CHOLMOD support".into(),
                ));
            }
        };
        Ok(Self {
            entries,
            inner,
            settings: *settings,
            stats: LinearAlgebraStats::default(),
        })
    }

    pub(crate) fn stats(&self) -> LinearAlgebraStats {
        self.stats
    }

    pub(crate) fn stats_mut(&mut self) -> &mut LinearAlgebraStats {
        &mut self.stats
    }

    pub(crate) fn set_clamp_floor(&mut self, floor: f64) {
        self.settings.diagonal_floor = floor;
    }

    pub(crate) fn set_strict_refinement(&mut self, strict: bool) {
        self.settings.strict_refinement = strict;
    }

    pub(crate) fn settings(&self) -> KktSettings {
        self.settings
    }

    /// Solve without refinement into a reusable buffer.
    pub(crate) fn solve_unrefined_into(
        &mut self,
        rhs: &[f64],
        out: &mut Vec<f64>,
    ) -> Result<(), LinearSystemError> {
        solve_into(&mut self.inner, rhs, out)
    }

    /// Solve a caller-owned RHS in place.
    pub(crate) fn solve_unrefined_in_place(
        &mut self,
        rhs: &mut [f64],
    ) -> Result<(), LinearSystemError> {
        self.inner.solve_in_place(rhs)
    }

    /// Apply the factors to two column-major right-hand sides together.
    pub(crate) fn solve_pair_in_place(&mut self, rhs: &mut [f64]) -> Result<(), LinearSystemError> {
        validate_rhs(2 * self.inner.dim(), rhs)?;
        match &mut self.inner {
            FactorizationInner::Faer(factor) => factor.solve_in_place(rhs, 2)?,
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            FactorizationInner::Accelerate(factor) => factor.solve_in_place(rhs, 2)?,
            #[cfg(feature = "cholmod")]
            FactorizationInner::Cholmod(factor) => factor.solve_in_place(rhs, 2)?,
            #[cfg(feature = "pardiso")]
            FactorizationInner::Pardiso(factor) => factor.solve_in_place(rhs, 2)?,
        }
        validate_solution(rhs)
    }

    /// Update the compact variable entries in the order of their positions.
    /// Each backend scatters them into its own matrix; no second KKT matrix
    /// is retained here.
    pub(crate) fn update_and_refactor(
        &mut self,
        factorizations: &mut usize,
        update: impl FnOnce(&mut [f64]),
    ) -> Result<(), LinearSystemError> {
        update(&mut self.entries);
        match &mut self.inner {
            FactorizationInner::Faer(factor) => {
                factor.refactor_entries(&self.entries, factorizations)
            }
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            FactorizationInner::Accelerate(factor) => {
                factor.refactor_entries(&self.entries, factorizations)
            }
            #[cfg(feature = "cholmod")]
            FactorizationInner::Cholmod(factor) => {
                factor.refactor_entries(&self.entries, factorizations)
            }
            #[cfg(feature = "pardiso")]
            FactorizationInner::Pardiso(factor) => {
                factor.refactor_entries(&self.entries, factorizations)
            }
        }
    }

    /// Factor the already assembled values without rewriting any entries.
    pub(crate) fn factor(&mut self, factorizations: &mut usize) -> Result<(), LinearSystemError> {
        match &mut self.inner {
            FactorizationInner::Faer(factor) => factor.factor(factorizations),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            FactorizationInner::Accelerate(factor) => factor.factor(factorizations),
            #[cfg(feature = "cholmod")]
            FactorizationInner::Cholmod(factor) => factor.factor(factorizations),
            #[cfg(feature = "pardiso")]
            FactorizationInner::Pardiso(factor) => factor.factor(factorizations),
        }
    }
}

fn solve_into(
    inner: &mut FactorizationInner,
    rhs: &[f64],
    out: &mut Vec<f64>,
) -> Result<(), LinearSystemError> {
    out.clear();
    out.extend_from_slice(rhs);
    inner.solve_in_place(out)
}

impl FactorizationInner {
    fn dim(&self) -> usize {
        match self {
            Self::Faer(factorization) => factorization.dim(),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Self::Accelerate(factorization) => factorization.dim(),
            #[cfg(feature = "cholmod")]
            Self::Cholmod(factorization) => factorization.dim(),
            #[cfg(feature = "pardiso")]
            Self::Pardiso(factorization) => factorization.dim(),
        }
    }

    fn solve_in_place(&mut self, vector: &mut [f64]) -> Result<(), LinearSystemError> {
        validate_rhs(self.dim(), vector)?;
        match self {
            Self::Faer(factorization) => factorization.solve_in_place(vector, 1),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Self::Accelerate(factorization) => factorization.solve_in_place(vector, 1),
            #[cfg(feature = "cholmod")]
            Self::Cholmod(factorization) => factorization.solve_in_place(vector, 1),
            #[cfg(feature = "pardiso")]
            Self::Pardiso(factorization) => factorization.solve_in_place(vector, 1),
        }?;
        validate_solution(vector)
    }
}

/// Reused by all right-hand sides and correction passes of a solve.
#[derive(Default)]
pub(crate) struct RefinementWorkspace {
    pub(crate) perturbation: Vec<f64>,
    residual: Vec<f64>,
    denominator: Vec<f64>,
    correction: Vec<f64>,
    best: Vec<f64>,
}

/// Refine against an implicit matrix. The initial callback writes the residual
/// and its fixed normalizer; correction solves overwrite reusable storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_with(
    rhs: &[f64],
    mut solution: Vec<f64>,
    settings: &KktSettings,
    workspace: &mut RefinementWorkspace,
    mut residual_into: impl FnMut(&[f64], &[f64], &mut [f64]),
    mut initialize: impl FnMut(&[f64], &[f64], &mut [f64], &mut [f64]),
    mut solve_correction: impl FnMut(&[f64], &mut Vec<f64>) -> Result<(), LinearSystemError>,
    stats: &mut LinearAlgebraStats,
) -> Result<Vec<f64>, LinearSystemError> {
    if solution.len() != rhs.len() {
        return Err(LinearSystemError::InvalidSolutionDimension {
            expected: rhs.len(),
            actual: solution.len(),
        });
    }
    // Even with no correction budget, evaluate the target residual before
    // applying the strict or best-effort acceptance rule.
    let strict = settings.strict_refinement;
    let RefinementWorkspace {
        perturbation,
        residual,
        denominator,
        correction,
        best,
    } = workspace;
    residual.resize(rhs.len(), 0.0);
    denominator.resize(rhs.len(), 0.0);
    initialize(&solution, rhs, residual, denominator);
    let mut best_error: Option<f64> = None;
    let mut errors = [f64::INFINITY; 2];
    let mut iteration = 0;
    loop {
        if iteration != 0 {
            residual_into(&solution, rhs, residual);
        }
        let backward_error = backward_error(residual, denominator);
        let improved = match best_error {
            Some(error) if !backward_error.is_finite() || backward_error > 10.0 * error => {
                break;
            }
            Some(error) => backward_error < error,
            None => true,
        };
        if backward_error <= settings.refine_tol {
            validate_solution(&solution)?;
            return Ok(solution);
        }
        // Strict refinement needs only the best error: failure never returns
        // a saved iterate. Best-effort mode keeps the original swap scheme.
        if improved {
            if !strict {
                // Keep the iterate by ownership, including the initial one.
                // The previous best buffer becomes the next update destination.
                std::mem::swap(best, &mut solution);
            }
            best_error = Some(backward_error);
        }
        if iteration >= 2 {
            let rate = (backward_error / errors[0]).sqrt();
            let remaining = i32::try_from(settings.refine_iters - iteration).unwrap_or(i32::MAX);
            if rate.is_nan()
                || (rate > STALL_RATE
                    && backward_error * rate.powi(remaining) > settings.refine_tol)
            {
                break;
            }
        }
        errors = [errors[1], backward_error];
        if iteration == settings.refine_iters {
            break;
        }
        stats.refinement_solves += 1;
        solve_correction(residual, correction)?;
        if correction.len() != rhs.len() {
            return Err(LinearSystemError::InvalidSolutionDimension {
                expected: rhs.len(),
                actual: correction.len(),
            });
        }
        if !strict && improved {
            // After a best-effort swap, `solution` is the stale output buffer.
            solution.resize(rhs.len(), 0.0);
            crate::linear_algebra::binary_into(
                &mut solution,
                best,
                correction,
                |base, correction| base + correction,
            );
        } else {
            crate::linear_algebra::add_assign(&mut solution, correction);
        }
        if validate_solution(&solution).is_err() {
            break;
        }
        iteration += 1;
    }
    if iteration > 0 {
        diagnose_correction(residual, correction, perturbation, denominator, stats);
    }
    settle_refinement(best_error, strict, best, solution, stats)
}

// Evaluated only on an inaccurate refinement exit. With the last correction d,
// q = r_new - (M-K)d also includes rounding in the solution update/residual.
fn diagnose_correction(
    r: &[f64],
    d: &[f64],
    diagonal: &[f64],
    scale: &[f64],
    stats: &mut LinearAlgebraStats,
) {
    let mut target: f64 = 0.0;
    let mut solve: f64 = 0.0;
    let mut regularization: f64 = 0.0;
    if d.len() != r.len() {
        return;
    }
    for i in 0..r.len() {
        let term = diagonal.get(i).copied().unwrap_or(0.0) * d[i];
        let q = r[i] - term;
        if !r[i].is_finite() || !q.is_finite() {
            return;
        }
        if scale[i] > 0.0 {
            target = target.max(r[i].abs() / scale[i]);
            solve = solve.max(q.abs() / scale[i]);
            regularization = regularization.max(term.abs() / scale[i]);
        } else if r[i] != 0.0 || q != 0.0 {
            return;
        }
    }
    if !target.is_finite() || !solve.is_finite() || !regularization.is_finite() || target == 0.0 {
        return;
    }
    if solve < regularization {
        stats.refinement_clamp_limited += 1;
    } else if solve > regularization {
        stats.refinement_solve_limited += 1;
    }
}

fn settle_refinement(
    best_error: Option<f64>,
    strict: bool,
    best: &mut Vec<f64>,
    mut solution: Vec<f64>,
    stats: &mut LinearAlgebraStats,
) -> Result<Vec<f64>, LinearSystemError> {
    match best_error {
        Some(error) if !strict && error.is_finite() => {
            stats.refinement_settled += 1;
            // Return the best iterate and retain the unused allocation.
            std::mem::swap(best, &mut solution);
            validate_solution(&solution)?;
            Ok(solution)
        }
        Some(error) => {
            stats.refinement_failures += 1;
            Err(LinearSystemError::RefinementDiverged(error))
        }
        None => {
            stats.refinement_failures += 1;
            Err(LinearSystemError::Singular)
        }
    }
}

fn backward_error(residual: &[f64], denominator: &[f64]) -> f64 {
    residual
        .iter()
        .zip(denominator)
        .map(|(residual, denominator)| {
            if !residual.is_finite() || !denominator.is_finite() {
                f64::INFINITY
            } else if *denominator == 0.0 {
                if *residual == 0.0 { 0.0 } else { f64::INFINITY }
            } else {
                residual.abs() / denominator
            }
        })
        .fold(0.0_f64, f64::max)
}

/// Backend slots for the variable entries of a factorized matrix.
#[derive(Default)]
struct TriangleSource {
    /// (backend slot, index into the variable entries) for each entry.
    entries: Vec<(usize, usize)>,
}

impl TriangleSource {
    /// Map each variable position of the input CSC storage to its backend slot.
    fn new(variable_positions: &[usize], slot: impl Fn(usize) -> usize) -> Self {
        Self {
            entries: variable_positions
                .iter()
                .enumerate()
                .map(|(index, &position)| (slot(position), index))
                .collect(),
        }
    }

    fn gather(&self, entries: &[f64], out: &mut [f64]) {
        for &(slot, index) in &self.entries {
            out[slot] = entries[index];
        }
    }
}

/// Positions of the diagonal entries of an upper CSC matrix, in column order.
#[cfg(test)]
fn diagonal_positions(matrix: &CscMatrix) -> Vec<usize> {
    (0..matrix.columns())
        .map(|column| {
            let end = matrix.column_pointers()[column + 1];
            assert_eq!(matrix.row_indices()[end - 1], column);
            end - 1
        })
        .collect()
}

/// Check the factorization contract once before selecting a backend.
fn validate_matrix(matrix: &CscMatrix) -> Result<(), LinearSystemError> {
    let (rows, columns) = (matrix.rows(), matrix.columns());
    if rows != columns {
        return Err(LinearSystemError::NotSquare { rows, columns });
    }
    if rows == 0 {
        return Err(LinearSystemError::Empty);
    }
    // Sorted CSC rows make the upper-triangle check O(n).
    for column in 0..columns {
        let end = matrix.column_pointers()[column + 1];
        if end > matrix.column_pointers()[column] && matrix.row_indices()[end - 1] > column {
            return Err(LinearSystemError::NotUpperTriangle);
        }
    }
    Ok(())
}

pub(crate) fn validate_rhs(dim: usize, rhs: &[f64]) -> Result<(), LinearSystemError> {
    if rhs.len() != dim {
        return Err(LinearSystemError::InvalidRightHandSide {
            expected: dim,
            actual: rhs.len(),
        });
    }
    if !rhs.iter().all(|value| value.is_finite()) {
        return Err(LinearSystemError::NonFiniteRightHandSide);
    }
    Ok(())
}

fn validate_solution(solution: &[f64]) -> Result<(), LinearSystemError> {
    if !solution.iter().all(|value| value.is_finite()) {
        return Err(LinearSystemError::Singular);
    }
    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum LinearSystemError {
    #[error("linear-system backend unavailable: {0}")]
    Unavailable(String),
    #[error("linear-system matrix must be square, received {rows} by {columns}")]
    NotSquare { rows: usize, columns: usize },
    #[error("linear-system matrix must not be empty")]
    Empty,
    #[error("linear-system matrix must store only its upper triangle")]
    NotUpperTriangle,
    #[error("linear-system matrix is singular")]
    Singular,
    #[error("iterative refinement diverged (best backward error {0:.1e})")]
    RefinementDiverged(f64),
    #[error("linear-system factorization failed: {0}")]
    Factorization(String),
    #[error("right-hand side must have length {expected}, received {actual}")]
    InvalidRightHandSide { expected: usize, actual: usize },
    #[error("right-hand side must contain only finite values")]
    NonFiniteRightHandSide,
    #[error("linear-system solution must have length {expected}, received {actual}")]
    InvalidSolutionDimension { expected: usize, actual: usize },
}

impl LinearSystemError {
    /// Failures eligible for retry with different numerical solve parameters.
    pub(crate) fn is_numerical(&self) -> bool {
        match self {
            Self::Singular
            | Self::RefinementDiverged(_)
            | Self::Factorization(_)
            | Self::NonFiniteRightHandSide => true,
            Self::Unavailable(_)
            | Self::NotSquare { .. }
            | Self::Empty
            | Self::NotUpperTriangle
            | Self::InvalidRightHandSide { .. }
            | Self::InvalidSolutionDimension { .. } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse_system(values: Vec<f64>) -> CscMatrix {
        CscMatrix::from_triplets(3, 3, vec![0, 0, 1, 1, 2], vec![0, 1, 1, 2, 2], values).unwrap()
    }

    #[test]
    fn factorization_validation_is_shared_by_backends() {
        for solver in [
            LinearSolver::Faer,
            LinearSolver::Accelerate,
            LinearSolver::Cholmod,
            LinearSolver::Pardiso,
        ]
        .into_iter()
        .filter(|solver| solver.is_available())
        {
            let settings = KktSettings::from(&Settings {
                linear_solver: solver,
                ..Settings::default()
            });
            let mut count = 0;
            for (matrix, error) in [
                (
                    CscMatrix::zeros(1, 2).unwrap(),
                    LinearSystemError::NotSquare {
                        rows: 1,
                        columns: 2,
                    },
                ),
                (CscMatrix::zeros(0, 0).unwrap(), LinearSystemError::Empty),
                (
                    CscMatrix::from_triplets(2, 2, vec![1], vec![0], vec![1.0]).unwrap(),
                    LinearSystemError::NotUpperTriangle,
                ),
            ] {
                assert_eq!(
                    Factorization::prepare(matrix, &[], &settings).err(),
                    Some(error)
                );
            }
            assert_eq!(count, 0);
            let mut factor =
                Factorization::prepare(CscMatrix::identity(1).unwrap(), &[0], &settings).unwrap();
            factor.factor(&mut count).unwrap();
            assert_eq!(
                factor.solve_unrefined_in_place(&mut []),
                Err(LinearSystemError::InvalidRightHandSide {
                    expected: 1,
                    actual: 0
                })
            );
            assert_eq!(
                factor.solve_unrefined_into(&[f64::NAN], &mut Vec::new()),
                Err(LinearSystemError::NonFiniteRightHandSide)
            );
        }
    }

    #[test]
    fn refinement_corrects_a_componentwise_inaccurate_solution() {
        let mut stats = LinearAlgebraStats::default();
        let solution = refine_with(
            &[1e12, 1.0],
            vec![1.0, 0.0],
            &KktSettings::from(&Settings::default()),
            &mut RefinementWorkspace::default(),
            |x, b, r| {
                r[0] = b[0] - 1e12 * x[0];
                r[1] = b[1] - x[1];
            },
            |x, b, r, d| {
                r[0] = b[0] - 1e12 * x[0];
                r[1] = b[1] - x[1];
                d[0] = b[0].abs() + (1e12 * x[0]).abs();
                d[1] = b[1].abs() + x[1].abs();
            },
            |r, out| {
                out.clear();
                out.extend([r[0] / 1e12, r[1]]);
                Ok(())
            },
            &mut stats,
        )
        .unwrap();
        assert_eq!(solution, vec![1.0, 1.0]);
        assert_eq!(stats.refinement_solves, 1);
    }

    #[test]
    fn refinement_counts_failed_corrections_and_successful_retries() {
        let mut stats = LinearAlgebraStats::default();
        let mut workspace = RefinementWorkspace::default();
        // Failed correction, successful retry, then an already accurate initial point.
        for (fail, initial, solves) in [(true, 0.0, 1), (false, 0.0, 2), (false, 1.0, 2)] {
            let result = refine_with(
                &[1.0],
                vec![initial],
                &KktSettings::from(&Settings::default()),
                &mut workspace,
                |x, b, r| r[0] = b[0] - x[0],
                |x, b, r, d| {
                    r[0] = b[0] - x[0];
                    d[0] = b[0].abs() + x[0].abs();
                },
                |r, out| {
                    if fail {
                        return Err(LinearSystemError::Singular);
                    }
                    out.clear();
                    out.extend_from_slice(r);
                    Ok(())
                },
                &mut stats,
            );
            if fail {
                assert_eq!(result, Err(LinearSystemError::Singular));
            } else {
                assert_eq!(result.unwrap(), vec![1.0]);
            }
            assert_eq!(stats.refinement_solves, solves);
        }
    }

    #[test]
    fn refinement_stops_once_the_rate_cannot_reach_tolerance() {
        for strict in [false, true] {
            let mut stats = LinearAlgebraStats::default();
            let result = refine_with(
                &[1.0],
                vec![0.0],
                &KktSettings {
                    strict_refinement: strict,
                    ..KktSettings::from(&Settings::default())
                },
                &mut RefinementWorkspace::default(),
                |x, b, r| r[0] = b[0] - x[0],
                |x, b, r, d| {
                    r[0] = b[0] - x[0];
                    d[0] = b[0].abs() + x[0].abs();
                },
                |_, out| {
                    out.clear();
                    out.push(0.0);
                    Ok(())
                },
                &mut stats,
            );
            if strict {
                assert_eq!(result, Err(LinearSystemError::RefinementDiverged(1.0)));
            } else {
                assert_eq!(result.unwrap(), vec![0.0]);
            }
            assert_eq!(stats.refinement_solves, 2);
            assert_eq!(stats.refinement_settled, usize::from(!strict));
            assert_eq!(stats.refinement_failures, usize::from(strict));
        }
    }

    #[test]
    fn zero_correction_budget_still_evaluates_target_residual() {
        for strict in [false, true] {
            let mut stats = LinearAlgebraStats::default();
            let result = refine_with(
                &[1.0],
                vec![0.5],
                &KktSettings {
                    strict_refinement: strict,
                    refine_iters: 0,
                    ..KktSettings::from(&Settings::default())
                },
                &mut RefinementWorkspace::default(),
                |_, _, _| unreachable!(),
                |x, b, r, d| {
                    r[0] = b[0] - x[0];
                    d[0] = b[0].abs() + x[0].abs();
                },
                |_, _| unreachable!(),
                &mut stats,
            );
            if strict {
                assert_eq!(
                    result,
                    Err(LinearSystemError::RefinementDiverged(1.0 / 3.0))
                );
            } else {
                assert_eq!(result.unwrap(), vec![0.5]);
            }
            assert_eq!(stats.refinement_solves, 0);
            assert_eq!(stats.refinement_failures, usize::from(strict));
            assert_eq!(stats.refinement_settled, usize::from(!strict));
        }
    }

    #[test]
    fn refinement_keeps_the_best_iterate_when_it_diverges() {
        for strict in [false, true] {
            let mut stats = LinearAlgebraStats::default();
            let result = refine_with(
                &[1.0],
                vec![0.9],
                &KktSettings {
                    strict_refinement: strict,
                    ..KktSettings::from(&Settings::default())
                },
                &mut RefinementWorkspace::default(),
                |x, b, r| r[0] = b[0] - x[0],
                |x, b, r, d| {
                    r[0] = b[0] - x[0];
                    d[0] = b[0].abs() + x[0].abs();
                },
                |_, out| {
                    out.clear();
                    out.push(100.0);
                    Ok(())
                },
                &mut stats,
            );
            if strict {
                assert!(matches!(
                    result,
                    Err(LinearSystemError::RefinementDiverged(_))
                ));
            } else {
                assert_eq!(result.unwrap(), vec![0.9]);
            }
        }
    }

    #[test]
    fn nonfinite_refinement_preserves_the_best_iterate_and_counts_the_exit() {
        for strict in [false, true] {
            // MAX overflows the next residual; INFINITY overflows the iterate itself.
            for correction in [f64::MAX, f64::INFINITY] {
                let mut stats = LinearAlgebraStats::default();
                let result = refine_with(
                    &[1.0],
                    vec![0.0],
                    &KktSettings {
                        strict_refinement: strict,
                        ..KktSettings::from(&Settings::default())
                    },
                    &mut RefinementWorkspace::default(),
                    |x, b, r| r[0] = b[0] - 2.0 * x[0],
                    |x, b, r, d| {
                        r[0] = b[0] - 2.0 * x[0];
                        d[0] = b[0].abs() + (2.0 * x[0]).abs();
                    },
                    |_, out| {
                        out.clear();
                        out.push(correction);
                        Ok(())
                    },
                    &mut stats,
                );
                if strict {
                    assert_eq!(result, Err(LinearSystemError::RefinementDiverged(1.0)));
                } else {
                    assert_eq!(result.unwrap(), vec![0.0]);
                }
                assert_eq!(
                    stats,
                    LinearAlgebraStats {
                        refinement_solves: 1,
                        refinement_failures: usize::from(strict),
                        refinement_settled: usize::from(!strict),
                        ..LinearAlgebraStats::default()
                    }
                );
            }
        }
    }

    #[test]
    fn diagonal_updates_preserve_off_diagonals_across_refactors() {
        for solver in [
            LinearSolver::Faer,
            LinearSolver::Accelerate,
            LinearSolver::Cholmod,
            LinearSolver::Pardiso,
        ]
        .into_iter()
        .filter(|solver| solver.is_available())
        {
            let settings = KktSettings::from(&Settings {
                linear_solver: solver,
                ..Settings::default()
            });
            let mut count = 0;
            let matrix = sparse_system(vec![4.0, 1.0, -3.0, 1.0, 2.0]);
            let positions = diagonal_positions(&matrix);
            let mut factorization = Factorization::prepare(matrix, &positions, &settings).unwrap();
            factorization.factor(&mut count).unwrap();
            let mut solution = Vec::new();
            for diagonal in [[4.0, -3.0, 2.0], [5.0, -4.0, 3.0], [7.0, -2.0, 6.0]] {
                factorization
                    .update_and_refactor(&mut count, |values| values.copy_from_slice(&diagonal))
                    .unwrap();
                let rhs = [
                    diagonal[0] + 2.0,
                    4.0 + 2.0 * diagonal[1],
                    2.0 + 3.0 * diagonal[2],
                ];
                factorization
                    .solve_unrefined_into(&rhs, &mut solution)
                    .unwrap();
                for (&actual, expected) in solution.iter().zip([1.0, 2.0, 3.0]) {
                    assert!((actual - expected).abs() < 1e-9);
                }
            }
            assert_eq!(count, 4);
        }
    }

    #[test]
    fn off_diagonal_updates_are_scattered_on_every_backend() {
        for solver in [
            LinearSolver::Faer,
            LinearSolver::Accelerate,
            LinearSolver::Cholmod,
            LinearSolver::Pardiso,
        ]
        .into_iter()
        .filter(|solver| solver.is_available())
        {
            let settings = KktSettings::from(&Settings {
                linear_solver: solver,
                ..Settings::default()
            });
            // Storage order (0,0), (0,1), (1,1), (1,2), (2,2); every entry varies,
            // listed diagonal first as in a KKT layout.
            let matrix = sparse_system(vec![4.0, 1.0, -3.0, 1.0, 2.0]);
            let mut count = 0;
            let mut factorization =
                Factorization::prepare(matrix, &[0, 2, 4, 1, 3], &settings).unwrap();
            factorization.factor(&mut count).unwrap();
            let mut solution = Vec::new();
            for [a, b, c, ab, bc] in [
                [4.0, -3.0, 2.0, 1.0, 1.0],
                [5.0, -4.0, 3.0, -2.0, 0.5],
                [6.0, -2.0, 5.0, 0.0, -1.5],
            ] {
                factorization
                    .update_and_refactor(&mut count, |values| {
                        values.copy_from_slice(&[a, b, c, ab, bc])
                    })
                    .unwrap();
                // K * (1, 2, 3) for K = [a, ab, 0; ab, b, bc; 0, bc, c].
                let rhs = [a + 2.0 * ab, ab + 2.0 * b + 3.0 * bc, 2.0 * bc + 3.0 * c];
                factorization
                    .solve_unrefined_into(&rhs, &mut solution)
                    .unwrap();
                for (&actual, expected) in solution.iter().zip([1.0, 2.0, 3.0]) {
                    assert!((actual - expected).abs() < 1e-9, "{solver}: {solution:?}");
                }
            }
            assert_eq!(count, 4);
        }
    }

    #[test]
    fn correction_diagnostic_separates_exact_regularized_solves_from_solve_error() {
        let mut stats = LinearAlgebraStats::default();
        // K=1, M=10, rhs=1, correction=.1: r_new=.9, q=0.
        diagnose_correction(&[0.9], &[0.1], &[9.0], &[1.0], &mut stats);
        assert_eq!(stats.refinement_clamp_limited, 1);
        diagnose_correction(&[0.9], &[0.1], &[], &[1.0], &mut stats);
        assert_eq!(stats.refinement_solve_limited, 1);
    }
}
