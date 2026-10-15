//! Factorization adapters used by the optimization algorithm.

mod faer;
mod feral;

use self::{faer::FaerFactorization, feral::FeralFactorization};
use thiserror::Error;

use crate::{CscMatrix, LinearSolver, Settings};

// Per-pass contraction below which refinement counts as stalled (no halving
// over two passes).
const STALL_RATE: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// How KKT systems are factorized and solved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KktSettings {
    pub(crate) linear_solver: LinearSolver,
    /// Fail a solve whose refinement diverges or stalls; otherwise return the
    /// best iterate seen.
    pub(crate) strict_refinement: bool,
    /// Fold nonnegative singleton rows of `A` into the `(1,1)` block.
    pub(crate) eliminate_bound_rows: bool,
    pub(crate) min_static_regularization: f64,
    pub(crate) max_refinement_steps: usize,
    pub(crate) refinement_tolerance: f64,
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
            strict_refinement: false,
            eliminate_bound_rows: settings.kkt_bound_elimination,
            min_static_regularization: settings.min_static_regularization,
            max_refinement_steps: settings.max_refinement_steps,
            refinement_tolerance: settings.refinement_tolerance,
        }
    }
}

/// Linear-algebra counters accumulated over one solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearAlgebraStats {
    /// Numeric factorization attempts, including failures.
    pub factorizations: usize,
    /// Initial linear solves attempted for KKT right-hand sides, including failures.
    /// Includes initialization and retries, but excludes refinement corrections.
    pub primary_solves: usize,
    /// Correction linear solves attempted by iterative refinement, including failures.
    /// Excludes the initial solve for each right-hand side and residual evaluations.
    pub refinement_solves: usize,
    /// Refinements that returned their best iterate after stalling or diverging.
    pub refinement_settled: usize,
    /// Refinements that failed (strict stall/divergence, or singular).
    pub refinement_failures: usize,
    /// Factorizations rejected before any solve because their inertia
    /// was not that of a quasi-definite matrix.
    pub inertia_failures: usize,
    /// Extra centrality correctors solved and accepted.
    pub corrector_tries: usize,
    pub corrector_accepts: usize,
}

/// Inertia and pivot extremes of the most recent factorization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PivotReport {
    pub(crate) positive: usize,
    pub(crate) negative: usize,
    pub(crate) zero: usize,
    pub(crate) min_abs: f64,
    pub(crate) max_abs: f64,
}

impl LinearAlgebraStats {
    pub(crate) fn absorb(&mut self, other: Self) {
        self.factorizations += other.factorizations;
        self.primary_solves += other.primary_solves;
        self.refinement_solves += other.refinement_solves;
        self.refinement_settled += other.refinement_settled;
        self.refinement_failures += other.refinement_failures;
        self.inertia_failures += other.inertia_failures;
        self.corrector_tries += other.corrector_tries;
        self.corrector_accepts += other.corrector_accepts;
    }
}

/// A factored symmetric system. The backend owns the matrix storage; the
/// wrapper retains only its diagonal for subsequent KKT updates.
pub(crate) struct Factorization {
    diagonal: Vec<f64>,
    inner: FactorizationInner,
    settings: KktSettings,
    stats: LinearAlgebraStats,
    /// Buffer lent to callers that solve reduced systems.
    scratch: Vec<f64>,
    refinement: RefinementWorkspace,
}

enum FactorizationInner {
    Faer(Box<FaerFactorization>),
    Feral(Box<FeralFactorization>),
}

impl Factorization {
    pub(crate) fn new(
        matrix: CscMatrix,
        settings: &KktSettings,
    ) -> Result<Self, LinearSystemError> {
        Self::new_counted(matrix, settings, &mut 0)
    }

    pub(crate) fn new_counted(
        matrix: CscMatrix,
        settings: &KktSettings,
        factorizations: &mut usize,
    ) -> Result<Self, LinearSystemError> {
        validate_square(matrix.rows(), matrix.columns())?;
        let diagonal = (0..matrix.rows())
            .map(|column| matrix.get(column, column).unwrap_or(0.0))
            .collect();
        let inner = match settings.linear_solver {
            LinearSolver::Faer => {
                FactorizationInner::Faer(Box::new(FaerFactorization::new(matrix, factorizations)?))
            }
            LinearSolver::Feral => FactorizationInner::Feral(Box::new(FeralFactorization::new(
                matrix,
                factorizations,
            )?)),
        };
        Ok(Self {
            diagonal,
            inner,
            settings: *settings,
            stats: LinearAlgebraStats::default(),
            scratch: Vec::new(),
            refinement: RefinementWorkspace::default(),
        })
    }

    pub(crate) fn dim(&self) -> usize {
        self.diagonal.len()
    }

    pub(crate) fn solve(
        &mut self,
        matrix: &CscMatrix,
        rhs: &[f64],
    ) -> Result<Vec<f64>, LinearSystemError> {
        let Self {
            inner,
            settings,
            stats,
            refinement,
            ..
        } = self;
        stats.primary_solves += 1;
        let initial = solve_owned(inner, rhs)?;
        refine_with(
            rhs,
            initial,
            settings,
            refinement,
            |solution, rhs, out| residual_into(matrix, solution, rhs, out),
            |solution, rhs, residual, denominator| {
                residual_and_denominator_into(matrix, solution, rhs, residual, denominator)
            },
            |residual, out| solve_into(inner, residual, out),
            stats,
        )
    }

    pub(crate) fn stats(&self) -> LinearAlgebraStats {
        self.stats
    }

    /// Inertia and pivot extremes of the current factor, when the backend
    /// exposes them.
    pub(crate) fn pivot_report(&self) -> Option<PivotReport> {
        self.inner.pivot_report()
    }

    /// Reject a factor whose inertia is not `(expected_positive, rest)`:
    /// a quasi-definite matrix has exactly one positive pivot per
    /// variable, so anything else is a rounding casualty that no
    /// refinement can repair.
    pub(crate) fn check_inertia(
        &mut self,
        expected_positive: usize,
    ) -> Result<(), LinearSystemError> {
        let Some(report) = self.pivot_report() else {
            return Ok(());
        };
        if report.positive != expected_positive || report.zero != 0 {
            self.stats.inertia_failures += 1;
            return Err(LinearSystemError::WrongInertia {
                positive: report.positive,
                expected: expected_positive,
            });
        }
        Ok(())
    }

    pub(crate) fn stats_mut(&mut self) -> &mut LinearAlgebraStats {
        &mut self.stats
    }

    pub(crate) fn set_strict_refinement(&mut self, strict: bool) {
        self.settings.strict_refinement = strict;
    }

    pub(crate) fn settings(&self) -> KktSettings {
        self.settings
    }

    /// One factored solve without refinement, for callers refining against
    /// a larger system.
    pub(crate) fn solve_unrefined(&mut self, rhs: &[f64]) -> Result<Vec<f64>, LinearSystemError> {
        solve_owned(&mut self.inner, rhs)
    }

    /// `solve_unrefined` into a reused buffer.
    pub(crate) fn solve_unrefined_into(
        &mut self,
        rhs: &[f64],
        out: &mut Vec<f64>,
    ) -> Result<(), LinearSystemError> {
        solve_into(&mut self.inner, rhs, out)
    }

    /// Solve a caller-owned RHS in place, with backend validation.
    pub(crate) fn solve_unrefined_in_place(
        &mut self,
        rhs: &mut [f64],
    ) -> Result<(), LinearSystemError> {
        self.inner.solve_in_place(rhs)
    }

    /// Borrow reduced-solve scratch; return it with `put_scratch`.
    pub(crate) fn take_scratch(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.scratch)
    }

    pub(crate) fn put_scratch(&mut self, scratch: Vec<f64>) {
        self.scratch = scratch;
    }

    /// Update the compact diagonal in original variable order. Each backend
    /// scatters it into its own matrix; no second KKT matrix is retained here.
    pub(crate) fn update_diagonal_and_refactor(
        &mut self,
        factorizations: &mut usize,
        update: impl FnOnce(&mut [f64]),
    ) -> Result<(), LinearSystemError> {
        update(&mut self.diagonal);
        match &mut self.inner {
            FactorizationInner::Faer(factor) => {
                factor.refactor_diagonal(&self.diagonal, factorizations)
            }
            FactorizationInner::Feral(factor) => {
                factor.refactor_diagonal(&self.diagonal, factorizations)
            }
        }
    }

    pub(crate) fn refactor(&mut self, matrix: CscMatrix) -> Result<(), LinearSystemError> {
        validate_square(matrix.rows(), matrix.columns())?;
        match &mut self.inner {
            FactorizationInner::Faer(factor) => factor.refactor(&matrix, &mut 0),
            FactorizationInner::Feral(factor) => factor.refactor(&matrix, &mut 0),
        }?;
        for (column, value) in self.diagonal.iter_mut().enumerate() {
            *value = matrix.get(column, column).unwrap_or(0.0);
        }
        Ok(())
    }
}

fn solve_owned(inner: &mut FactorizationInner, rhs: &[f64]) -> Result<Vec<f64>, LinearSystemError> {
    let mut out = Vec::with_capacity(rhs.len());
    solve_into(inner, rhs, &mut out)?;
    Ok(out)
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
    fn pivot_report(&self) -> Option<PivotReport> {
        match self {
            Self::Faer(_) => None,
            Self::Feral(factorization) => factorization.pivot_report(),
        }
    }

    fn solve_in_place(&mut self, vector: &mut [f64]) -> Result<(), LinearSystemError> {
        match self {
            Self::Faer(factorization) => factorization.solve_in_place(vector),
            Self::Feral(factorization) => factorization.solve_in_place(vector),
        }
    }
}

/// Refine `solution` against the true matrix, keeping the best iterate.
///
/// Refinement that diverges (backward error growing tenfold or non-finite)
/// or stalls at the pass limit returns the best iterate, or an error when
/// `strict` is set.
#[cfg(test)]
fn refine_solution(
    matrix: &CscMatrix,
    rhs: &[f64],
    solution: Vec<f64>,
    strict: bool,
    mut solve_correction: impl FnMut(&[f64]) -> Result<Vec<f64>, LinearSystemError>,
    stats: &mut LinearAlgebraStats,
) -> Result<Vec<f64>, LinearSystemError> {
    validate_rhs(matrix.rows(), rhs)?;
    refine_with(
        rhs,
        solution,
        &KktSettings {
            strict_refinement: strict,
            ..KktSettings::new(LinearSolver::Faer)
        },
        &mut RefinementWorkspace::default(),
        |solution, rhs, out| residual_into(matrix, solution, rhs, out),
        |solution, rhs, residual, denominator| {
            residual_and_denominator_into(matrix, solution, rhs, residual, denominator)
        },
        |residual, out| {
            *out = solve_correction(residual)?;
            Ok(())
        },
        stats,
    )
}

/// Reused by all right-hand sides and correction passes of a solve.
#[derive(Default)]
pub(crate) struct RefinementWorkspace {
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
    // The initial (possibly seeded) estimate already used one back-solve.
    // Disabling refinement bypasses its acceptance test and strict retries.
    if settings.max_refinement_steps == 0 {
        return validate_solution(solution);
    }
    let strict = settings.strict_refinement;
    let RefinementWorkspace {
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
    for iteration in 0..=settings.max_refinement_steps {
        if iteration != 0 {
            residual_into(&solution, rhs, residual);
        }
        let backward_error = backward_error(residual, denominator);
        let improved = match best_error {
            Some(error) if !backward_error.is_finite() || backward_error > 10.0 * error => {
                return settle_refinement(best_error, strict, best, stats);
            }
            Some(error) => backward_error < error,
            None => true,
        };
        if backward_error <= settings.refinement_tolerance {
            return validate_solution(solution);
        }
        // Strict refinement needs only the best error: failure never returns
        // a saved iterate. Best-effort mode keeps the original swap scheme.
        if improved {
            if !strict {
                if best_error.is_some() {
                    std::mem::swap(best, &mut solution);
                } else {
                    best.clear();
                    best.extend_from_slice(&solution);
                }
            }
            best_error = Some(backward_error);
        }
        if iteration >= 2 {
            let rate = (backward_error / errors[0]).sqrt();
            let remaining =
                i32::try_from(settings.max_refinement_steps - iteration).unwrap_or(i32::MAX);
            if rate.is_nan()
                || (rate > STALL_RATE
                    && backward_error * rate.powi(remaining) > settings.refinement_tolerance)
            {
                return settle_refinement(best_error, strict, best, stats);
            }
        }
        errors = [errors[1], backward_error];
        if iteration == settings.max_refinement_steps {
            return settle_refinement(best_error, strict, best, stats);
        }
        stats.refinement_solves += 1;
        solve_correction(residual, correction)?;
        if correction.len() != solution.len() {
            return Err(LinearSystemError::InvalidSolutionDimension {
                expected: solution.len(),
                actual: correction.len(),
            });
        }
        if !strict && improved {
            // After a best-effort swap, `solution` is the stale output buffer.
            crate::matrix::binary_into(&mut solution, best, correction, |base, correction| {
                base + correction
            });
        } else {
            crate::matrix::add_assign(&mut solution, correction);
        }
        if !solution.iter().all(|value| value.is_finite()) {
            return settle_refinement(best_error, strict, best, stats);
        }
    }
    unreachable!("the refinement loop returns after its final residual check")
}

fn settle_refinement(
    best_error: Option<f64>,
    strict: bool,
    best: &mut Vec<f64>,
    stats: &mut LinearAlgebraStats,
) -> Result<Vec<f64>, LinearSystemError> {
    match best_error {
        Some(_) if !strict => {
            stats.refinement_settled += 1;
            validate_solution(std::mem::take(best))
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

/// `residual <- rhs - matrix * solution`.
fn residual_into(matrix: &CscMatrix, solution: &[f64], rhs: &[f64], residual: &mut [f64]) {
    residual.copy_from_slice(rhs);
    crate::matrix::symmetric_accumulate(matrix, solution, [residual], |product| [-product]);
}

/// First refinement residual and its fixed normalizer in one matrix traversal.
fn residual_and_denominator_into(
    matrix: &CscMatrix,
    solution: &[f64],
    rhs: &[f64],
    residual: &mut [f64],
    denominator: &mut [f64],
) {
    residual.copy_from_slice(rhs);
    for (out, rhs) in denominator.iter_mut().zip(rhs) {
        *out = rhs.abs();
    }
    crate::matrix::symmetric_accumulate(matrix, solution, [residual, denominator], |product| {
        [-product, product.abs()]
    });
}

/// `out <- |matrix| |solution| + |rhs|`.
#[cfg(test)]
fn backward_denominator_into(matrix: &CscMatrix, solution: &[f64], rhs: &[f64], out: &mut [f64]) {
    for (out, rhs) in out.iter_mut().zip(rhs) {
        *out = rhs.abs();
    }
    crate::matrix::symmetric_accumulate(matrix, solution, [out], |product| [product.abs()]);
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

/// Where a fixed-pattern triangle's entries live in the source CSC triangle.
struct TriangleSource {
    matrix_nonzeros: usize,
    positions: Vec<usize>,
    /// (triangle slot, original column) for each stored diagonal entry.
    diagonal: Vec<(usize, usize)>,
}

impl TriangleSource {
    fn new(matrix: &CscMatrix) -> Self {
        let mut diagonal = Vec::with_capacity(matrix.columns());
        for column in 0..matrix.columns() {
            let end = matrix.column_pointers()[column + 1];
            if end > matrix.column_pointers()[column] && matrix.row_indices()[end - 1] == column {
                diagonal.push((end - 1, column));
            }
        }
        Self {
            matrix_nonzeros: matrix.values().len(),
            positions: (0..matrix.values().len()).collect(),
            diagonal,
        }
    }

    fn gather_diagonal(&self, diagonal: &[f64], out: &mut [f64]) {
        for &(slot, position) in &self.diagonal {
            out[slot] = diagonal[position];
        }
    }

    /// Gather unchanged-coordinate triangle values in backend order.
    fn gather_values(
        &self,
        matrix: &CscMatrix,
        out: &mut Vec<f64>,
    ) -> Result<(), LinearSystemError> {
        if matrix.values().len() != self.matrix_nonzeros {
            return Err(LinearSystemError::SparsityPatternChanged);
        }
        out.clear();
        out.extend(self.positions.iter().map(|&index| matrix.values()[index]));
        Ok(())
    }
}

fn upper_triangle_usize(matrix: &CscMatrix) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    (
        matrix.column_pointers().to_vec(),
        matrix.row_indices().to_vec(),
        matrix.values().to_vec(),
    )
}

fn validate_square(rows: usize, columns: usize) -> Result<(), LinearSystemError> {
    if rows != columns {
        return Err(LinearSystemError::NotSquare { rows, columns });
    }
    Ok(())
}

fn validate_rhs(dim: usize, rhs: &[f64]) -> Result<(), LinearSystemError> {
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

/// Internal factorization input is already a symmetric matrix stored by its
/// upper triangle. Sorted CSC rows make this check O(n), without reading values.
fn validate_upper_triangle(matrix: &CscMatrix) -> Result<(), LinearSystemError> {
    for column in 0..matrix.columns() {
        let end = matrix.column_pointers()[column + 1];
        if end > matrix.column_pointers()[column] && matrix.row_indices()[end - 1] > column {
            return Err(LinearSystemError::NotUpperTriangle);
        }
    }
    Ok(())
}

fn validate_solution(solution: Vec<f64>) -> Result<Vec<f64>, LinearSystemError> {
    validate_finite(&solution)?;
    Ok(solution)
}

fn validate_finite(solution: &[f64]) -> Result<(), LinearSystemError> {
    if !solution.iter().all(|value| value.is_finite()) {
        return Err(LinearSystemError::Singular);
    }
    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum LinearSystemError {
    #[error("linear-system matrix must be square, received {rows} by {columns}")]
    NotSquare { rows: usize, columns: usize },
    #[error("linear-system matrix must not be empty")]
    Empty,
    #[error("linear-system matrix must store only its upper triangle")]
    NotUpperTriangle,
    #[error("linear-system matrix is singular")]
    Singular,
    #[error("factorization has {positive} positive pivots, expected {expected}")]
    WrongInertia { positive: usize, expected: usize },
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
    #[error("matrix dimension changed from {expected} to {actual}")]
    DimensionChanged { expected: usize, actual: usize },
    #[error("matrix sparsity pattern changed during numeric refactorization")]
    SparsityPatternChanged,
    #[error("linear-system index exceeds the supported 64-bit range")]
    IndexOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_system(values: Vec<f64>) -> CscMatrix {
        crate::matrix::take_upper_triangle(crate::matrix::test_matrix(3, 3, values).unwrap())
    }

    fn sparse_system(values: Vec<f64>) -> CscMatrix {
        crate::matrix::take_upper_triangle(
            CscMatrix::from_triplets(
                3,
                3,
                vec![0, 1, 0, 1, 2, 1, 2],
                vec![0, 0, 1, 1, 1, 2, 2],
                values,
            )
            .unwrap(),
        )
    }

    fn assert_solution_close(actual: &[f64], expected: &[f64]) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1e-9 * (1.0 + expected.abs()),
                "{actual} != {expected}"
            );
        }
    }

    fn assert_solution(factorization: &mut Factorization, expected: &[f64]) {
        let solution = factorization.solve_unrefined(&[1.0, 2.0, 3.0]).unwrap();
        for (actual, expected) in solution.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    fn matrix_value(matrix: &CscMatrix, row: usize, column: usize) -> f64 {
        matrix.get(row.min(column), row.max(column)).unwrap_or(0.0)
    }

    fn backward_error_of(matrix: &CscMatrix, solution: &[f64], rhs: &[f64]) -> f64 {
        let mut residual = vec![0.0; rhs.len()];
        let mut denominator = vec![0.0; rhs.len()];
        residual_into(matrix, solution, rhs, &mut residual);
        backward_denominator_into(matrix, solution, rhs, &mut denominator);
        backward_error(&residual, &denominator)
    }

    #[test]
    fn fused_initial_residual_preserves_rounding_and_the_normalizer() {
        for matrix in [
            test_system(vec![1e16, 1.0, -1e16, -0.0, 3e-12, 2.0, 1.0, -7.0, 4.0]),
            sparse_system(vec![1e16, 1.0, -1e16, 3e-12, -7.0, 2.0, 4.0]),
        ] {
            for solution in [[1.0, 1.0, 1.0], [1e-12, -1e12, 0.0]] {
                let rhs = [1.0, -0.0, 3e-10];
                let mut expected_residual = vec![0.0; 3];
                let mut expected_denominator = vec![0.0; 3];
                residual_into(&matrix, &solution, &rhs, &mut expected_residual);
                backward_denominator_into(&matrix, &solution, &rhs, &mut expected_denominator);
                let mut residual = vec![f64::NAN; 3];
                let mut denominator = vec![f64::NAN; 3];
                residual_and_denominator_into(
                    &matrix,
                    &solution,
                    &rhs,
                    &mut residual,
                    &mut denominator,
                );
                for (actual, expected) in residual
                    .iter()
                    .chain(&denominator)
                    .zip(expected_residual.iter().chain(&expected_denominator))
                {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn componentwise_backward_error_detects_a_badly_scaled_row() {
        let matrix = crate::matrix::test_matrix(2, 2, vec![1e12, 0.0, 0.0, 1.0]).unwrap();
        let backward_error = backward_error_of(&matrix, &[1.0, 0.0], &[1e12, 1.0]);

        assert_eq!(backward_error, 1.0);
    }

    #[test]
    fn componentwise_backward_error_is_invariant_under_row_scaling() {
        let original: CscMatrix =
            crate::matrix::test_matrix(2, 2, vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let scaled: CscMatrix =
            crate::matrix::test_matrix(2, 2, vec![2e12, 0.0, 0.0, 3e-12]).unwrap();
        let solution = [0.75, 0.5];
        let original_error = backward_error_of(&original, &solution, &[2.0, 3.0]);
        let scaled_error = backward_error_of(&scaled, &solution, &[2e12, 3e-12]);

        assert!((original_error - scaled_error).abs() <= f64::EPSILON);
    }

    #[test]
    fn refinement_corrects_a_componentwise_inaccurate_solution() {
        let matrix: CscMatrix =
            crate::matrix::test_matrix(2, 2, vec![1e12, 0.0, 0.0, 1.0]).unwrap();
        let mut correction_count = 0;
        let solution = refine_solution(
            &matrix,
            &[1e12, 1.0],
            vec![1.0, 0.0],
            false,
            |residual| {
                correction_count += 1;
                Ok(vec![residual[0] / 1e12, residual[1]])
            },
            &mut LinearAlgebraStats::default(),
        )
        .unwrap();

        assert_eq!(solution, vec![1.0, 1.0]);
        assert_eq!(correction_count, 1);
    }

    #[test]
    fn refinement_counts_failed_corrections_and_successful_retries() {
        let matrix = crate::matrix::test_matrix(1, 1, vec![1.0]).unwrap();
        let mut stats = LinearAlgebraStats::default();
        let failed = refine_solution(
            &matrix,
            &[1.0],
            vec![0.0],
            true,
            |_| Err(LinearSystemError::Singular),
            &mut stats,
        );
        assert!(matches!(failed, Err(LinearSystemError::Singular)));
        assert_eq!(stats.refinement_solves, 1);

        let solution = refine_solution(
            &matrix,
            &[1.0],
            vec![0.0],
            true,
            |residual| Ok(residual.to_vec()),
            &mut stats,
        )
        .unwrap();
        assert_eq!(solution, vec![1.0]);
        assert_eq!(stats.refinement_solves, 2);

        // An accurate initial solve needs only a residual check, not a correction.
        refine_solution(
            &matrix,
            &[1.0],
            solution,
            true,
            |_| panic!("an accurate solution needs no correction solve"),
            &mut stats,
        )
        .unwrap();
        assert_eq!(stats.refinement_solves, 2);
    }

    #[test]
    fn refinement_stops_once_the_rate_cannot_reach_tolerance() {
        let matrix: CscMatrix = crate::matrix::test_matrix(1, 1, vec![1.0]).unwrap();
        let mut correction_count = 0;
        let mut stats = LinearAlgebraStats::default();
        let solution = refine_solution(
            &matrix,
            &[1.0],
            vec![0.0],
            false,
            |_| {
                correction_count += 1;
                Ok(vec![0.0])
            },
            &mut stats,
        )
        .unwrap();

        // Two stalled passes are enough to extrapolate a miss.
        assert_eq!(solution, vec![0.0]);
        assert_eq!(correction_count, 2);
        assert_eq!(stats.refinement_solves, 2);
        assert_eq!(stats.refinement_settled, 1);

        let mut stats = LinearAlgebraStats::default();
        assert!(matches!(
            refine_solution(
                &matrix,
                &[1.0],
                vec![0.0],
                true,
                |_| Ok(vec![0.0]),
                &mut stats,
            ),
            Err(LinearSystemError::RefinementDiverged(_))
        ));
        assert_eq!(stats.refinement_failures, 1);
    }

    #[test]
    fn refinement_keeps_the_best_iterate_when_it_diverges() {
        let matrix: CscMatrix = crate::matrix::test_matrix(1, 1, vec![1.0]).unwrap();
        // Corrections push the iterate away from the solution x = 1.
        let mut stats = LinearAlgebraStats::default();
        let solution = refine_solution(
            &matrix,
            &[1.0],
            vec![0.9],
            false,
            |_| Ok(vec![100.0]),
            &mut stats,
        )
        .unwrap();
        assert_eq!(solution, vec![0.9]);
        assert!(matches!(
            refine_solution(
                &matrix,
                &[1.0],
                vec![0.9],
                true,
                |_| Ok(vec![100.0]),
                &mut stats,
            ),
            Err(LinearSystemError::RefinementDiverged(_))
        ));
    }

    #[test]
    fn faer_regularizes_a_tiny_pivot_and_refines_to_the_true_solution() {
        // Quasi-definite with a 1e-14 leading pivot and determinant 1.
        let values = vec![1e-14, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0];
        let matrix = sparse_system(values);
        let mut factorization =
            Factorization::new(matrix.clone(), &KktSettings::new(LinearSolver::Faer)).unwrap();
        let solution = factorization.solve(&matrix, &[1.0, 2.0, 3.0]).unwrap();
        assert_solution_close(&solution, &[5.0, 1.0 - 5e-14, -2.0]);
    }

    #[test]
    fn numeric_failure_is_counted_but_shape_validation_is_not() {
        let settings = KktSettings::new(LinearSolver::Faer);
        let mut count = 0;
        let nonsquare = crate::matrix::test_matrix(1, 2, vec![1.0, 1.0]).unwrap();
        assert!(Factorization::new_counted(nonsquare, &settings, &mut count).is_err());
        assert_eq!(count, 0);
        let singular = crate::matrix::take_upper_triangle(
            crate::matrix::test_matrix(2, 2, vec![1.0; 4]).unwrap(),
        );
        assert!(Factorization::new_counted(singular, &settings, &mut count).is_err());
        assert_eq!(count, 1);
    }

    #[test]
    fn diagonal_updates_preserve_off_diagonals_across_refactors() {
        for solver in [LinearSolver::Faer, LinearSolver::Feral] {
            let mut factorization = Factorization::new(
                sparse_system(vec![4.0, 1.0, 1.0, -3.0, 1.0, 1.0, 2.0]),
                &KktSettings::new(solver),
            )
            .unwrap();
            // Exercise both the initially cached values and a subsequent
            // full refactor that changes the off-diagonal entries.
            for coupling in [1.0, 2.0] {
                if coupling == 2.0 {
                    factorization
                        .refactor(sparse_system(vec![
                            4.0, coupling, coupling, -3.0, coupling, coupling, 2.0,
                        ]))
                        .unwrap();
                }
                for diagonal in [[5.0, -4.0, 3.0], [7.0, -2.0, 6.0]] {
                    factorization
                        .update_diagonal_and_refactor(&mut 0, |values| {
                            values.copy_from_slice(&diagonal);
                        })
                        .unwrap();
                    // Use a known solution and no refinement: refinement
                    // must not conceal stale or misplaced factor entries.
                    let rhs = [
                        diagonal[0] + 2.0 * coupling,
                        4.0 * coupling + 2.0 * diagonal[1],
                        2.0 * coupling + 3.0 * diagonal[2],
                    ];
                    let solution = factorization.solve_unrefined(&rhs).unwrap();
                    assert_solution_close(&solution, &[1.0, 2.0, 3.0]);
                }
            }
        }
    }

    #[test]
    fn faer_factors_solves_and_refactors_sparse_systems() {
        let mut factorization = Factorization::new(
            sparse_system(vec![4.0, 1.0, 1.0, -3.0, 1.0, 1.0, 2.0]),
            &KktSettings::new(LinearSolver::Faer),
        )
        .unwrap();
        assert_solution(&mut factorization, &[4.0 / 15.0, -1.0 / 15.0, 23.0 / 15.0]);

        factorization
            .refactor(sparse_system(vec![5.0, 1.0, 1.0, -3.0, 1.0, 1.0, 2.0]))
            .unwrap();
        assert_solution(&mut factorization, &[8.0 / 37.0, -3.0 / 37.0, 57.0 / 37.0]);
    }

    #[test]
    fn feral_factors_solves_and_refactors_sparse_systems() {
        let mut factorization = Factorization::new(
            sparse_system(vec![4.0, 1.0, 1.0, -3.0, 1.0, 1.0, 2.0]),
            &KktSettings::new(LinearSolver::Feral),
        )
        .unwrap();
        assert_solution(&mut factorization, &[4.0 / 15.0, -1.0 / 15.0, 23.0 / 15.0]);

        factorization
            .refactor(sparse_system(vec![5.0, 1.0, 1.0, -3.0, 1.0, 1.0, 2.0]))
            .unwrap();
        assert_solution(&mut factorization, &[8.0 / 37.0, -3.0 / 37.0, 57.0 / 37.0]);
    }

    #[test]
    fn inertia_check_counts_positive_pivots() {
        // diag(4, -3, 2) minus couplings: two positive pivots, one negative.
        let mut factorization = Factorization::new(
            sparse_system(vec![4.0, 1.0, 1.0, -3.0, 1.0, 1.0, 2.0]),
            &KktSettings::new(LinearSolver::Feral),
        )
        .unwrap();
        let report = factorization.pivot_report().unwrap();
        assert_eq!((report.positive, report.negative, report.zero), (2, 1, 0));
        assert!(report.max_abs >= report.min_abs && report.min_abs > 0.0);
        assert!(factorization.check_inertia(2).is_ok());
        assert!(matches!(
            factorization.check_inertia(1),
            Err(LinearSystemError::WrongInertia {
                positive: 2,
                expected: 1
            })
        ));
        assert_eq!(factorization.stats().inertia_failures, 1);
    }
}
