//! Numerical realization of the paper's quasi-definite system (`eq:qd_system`).
//!
//! Singleton elimination implements the Schur complement in "Reduction of KKT
//! system". Refinement evaluates the full shifted system, including recovered
//! duals. Shared by initialization and the resolvent.

use crate::{
    CscMatrix, MatrixError, Problem,
    cone::{ConeBlock, ConeGeometry},
    linear_solvers::{
        Factorization, KktSettings, LinearAlgebraStats, LinearSystemError, RefinementWorkspace,
        refine_with, validate_rhs,
    },
};
use thiserror::Error;

/// Failures while assembling or solving a shifted KKT system.
#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum KktError {
    #[error(transparent)]
    LinearSystem(#[from] LinearSystemError),
    #[error(transparent)]
    Matrix(#[from] MatrixError),
    #[error("invalid KKT shift, regularization, or diagonal dimension")]
    InvalidParameters,
    #[error("the KKT diagonal must be finite and strictly positive")]
    InvalidDiagonal,
}

impl KktError {
    pub(crate) fn is_numerical(&self) -> bool {
        match self {
            Self::LinearSystem(error) => error.is_numerical(),
            Self::InvalidDiagonal => true,
            Self::Matrix(_) | Self::InvalidParameters => false,
        }
    }
}

/// Per-solve cache: the KKT sparsity pattern and a reusable factorization.
///
/// Only the diagonal of the KKT matrix changes between iterations, so the
/// layout and symbolic analysis are retained before the first numeric attempt
/// and reused on retries, including after numeric factorization failures.
#[derive(Default)]
pub(crate) struct KktCache {
    // Layout and backend storage survive failed numeric factorization attempts.
    pattern: Option<KktPattern>,
    workspace: KktWorkspace,
    factorization: Option<Factorization>,
    /// The constant `(-c, b)` right-hand side, built on first use.
    constant_rhs: Vec<f64>,
    /// Its latest solution, retained across outer iterations of this solve.
    direction: Vec<f64>,
    /// Retain ownership of the preceding seed until the outer step succeeds.
    direction_before_step: Option<Vec<f64>>,
    /// Adaptive diagonal clamp, applied only to the preconditioner.
    pub(crate) clamp: ClampPolicy,
    /// Numeric dispatches, including failed initial factorizations.
    factorizations: usize,
}

/// Persistent clamp state and its adjustment rules across outer steps.
#[derive(Default)]
pub(crate) struct ClampPolicy {
    floor: Option<f64>,
    increases: usize,
    decreases: usize,
}

impl ClampPolicy {
    const MINIMUM: f64 = 1e-20;
    const MAXIMUM: f64 = 1e-4;

    pub(crate) fn start_step(&mut self, initial: f64) -> f64 {
        let floor = self.floor.unwrap_or(initial).max(Self::MINIMUM);
        self.floor = Some(floor);
        floor
    }

    pub(crate) fn floor(&self) -> f64 {
        self.floor.expect("start_step initializes the clamp")
    }

    fn increase(&mut self) -> f64 {
        let floor = self.floor();
        let next = (floor * 10.0).min(Self::MAXIMUM);
        self.increases += usize::from(next > floor);
        self.floor = Some(next);
        next
    }

    /// Preserve the x10 grid while skipping clamps that cannot change the matrix.
    pub(crate) fn retry(&mut self, nonbinding_limit: f64) -> Option<f64> {
        while self.floor() < Self::MAXIMUM {
            let next = self.increase();
            if next > nonbinding_limit {
                return Some(next);
            }
        }
        None
    }

    pub(crate) fn accepted_step(
        &mut self,
        starting_floor: f64,
        before: LinearAlgebraStats,
        after: LinearAlgebraStats,
    ) {
        let clamp_limited = after.refinement_clamp_limited > before.refinement_clamp_limited;
        let solve_limited = after.refinement_solve_limited > before.refinement_solve_limited;
        if solve_limited && !clamp_limited {
            self.increase();
        } else if self.floor().to_bits() == starting_floor.to_bits()
            && ((clamp_limited && !solve_limited)
                || after.refinement_settled == before.refinement_settled)
        {
            // A retry prevents immediate relaxation after an accepted step.
            let floor = self.floor();
            let next = (floor / 10.0).max(Self::MINIMUM);
            self.decreases += usize::from(next < floor);
            self.floor = Some(next);
        }
    }
}

/// The reduced KKT matrix pattern: bound rows are eliminated by Schur
/// complement, so the matrix has `n + kept` rows.
struct KktPattern {
    variable_count: usize,
    /// Position of each diagonal entry in the matrix's value storage.
    diagonal_positions: Vec<usize>,
    /// Diagonal of `P`.
    p_diagonal: Vec<f64>,
    bound_rows: BoundRows,
}

/// Mutable buffers reused across factorizations and right-hand sides.
#[derive(Default)]
struct KktWorkspace {
    /// Reduced right-hand side and solution when bound rows are eliminated.
    reduced: Vec<f64>,
    /// Full-system residual used for the first correction of a warm start.
    initial_residual: Vec<f64>,
    refinement: RefinementWorkspace,
    /// The perturbation is shared by every RHS of the current factorization.
    perturbation_ready: bool,
    /// Scratch storage reused by predictor/corrector reconstruction within a solve.
    reconstruction: Vec<f64>,
}

/// Nonnegative singleton rows of `A`, eliminated from the factorized system.
///
/// For such a row `i` with entry `a` in column `j`, the block row
/// `a x_j - d_i y_i = r_i` gives `y_i = (a x_j - r_i) / d_i`, which adds
/// `a^2 / d_i` to the `(j, j)` diagonal and `a r_i / d_i` to the `x_j`
/// right-hand side.
#[derive(Clone, Debug, Default, PartialEq)]
struct BoundRows {
    /// `(row, column, coefficient)` of each eliminated row.
    eliminated: Vec<(usize, usize, f64)>,
    /// Original row index of each kept row.
    kept: Vec<usize>,
}

impl BoundRows {
    fn detect(problem: &Problem, geometry: &ConeGeometry, enabled: bool) -> Self {
        let m = problem.constraint_count();
        let mut singleton = vec![None; m];
        let mut counts = vec![0usize; m];
        if enabled {
            crate::linear_algebra::visit_entries(problem.a(), |row, column, value| {
                if value != 0.0 {
                    counts[row] += 1;
                    singleton[row] = Some((column, value));
                }
            });
            for block in geometry.blocks() {
                if let ConeBlock::Zero(range) = block {
                    counts[range.clone()].fill(0);
                }
            }
        }
        let mut eliminated = Vec::new();
        let mut kept = Vec::new();
        for row in 0..m {
            match singleton[row] {
                Some((column, coefficient)) if counts[row] == 1 => {
                    eliminated.push((row, column, coefficient));
                }
                _ => {
                    kept.push(row);
                }
            }
        }
        Self { eliminated, kept }
    }
}

impl KktCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Factorize the clamped preconditioner `[P + shift I, A^T; A, -diag(d)]`
    /// with `d = factor_diagonal`, reusing the cached pattern and symbolic
    /// analysis when present, and reject factors with the wrong inertia.
    pub(crate) fn prepare(
        &mut self,
        problem: &Problem,
        geometry: &ConeGeometry,
        shift: f64,
        factor_diagonal: &[f64],
        kkt: &KktSettings,
    ) -> Result<(), KktError> {
        self.workspace.perturbation_ready = false;
        if let (Some(factorization), Some(pattern)) =
            (self.factorization.as_mut(), self.pattern.as_ref())
        {
            factorization.set_strict_refinement(kkt.strict_refinement);
            factorization.set_clamp_floor(kkt.diagonal_floor);
            factorization.update_diagonal_and_refactor(&mut self.factorizations, |values| {
                pattern.write_diagonal(
                    values,
                    |index| index,
                    shift,
                    factor_diagonal,
                    kkt.diagonal_floor,
                );
            })?;
        } else {
            let (k, pattern) = assemble_preconditioner(
                problem,
                geometry,
                shift,
                factor_diagonal,
                kkt.diagonal_floor,
                kkt.eliminate_bound_rows,
            )?;
            self.pattern = Some(pattern);
            self.factorization = Some(Factorization::prepare(k, kkt)?);
            self.factorization
                .as_mut()
                .expect("the factorization workspace was just stored")
                .factor(&mut self.factorizations)?;
        }
        self.factorization
            .as_mut()
            .expect("the factorization was just stored")
            .check_inertia(problem.variable_count())
            .map_err(KktError::from)
    }

    /// Refine a right-hand side after `prepare` succeeds for this system.
    pub(crate) fn solve(
        &mut self,
        problem: &Problem,
        shift: f64,
        d_diagonal: &[f64],
        d_factor: &[f64],
        rhs: &[f64],
    ) -> Result<Vec<f64>, LinearSystemError> {
        self.workspace.solve(
            self.pattern
                .as_ref()
                .expect("prepare stores the KKT layout"),
            problem,
            self.factorization
                .as_mut()
                .expect("prepare stores the KKT factors"),
            shift,
            d_diagonal,
            d_factor,
            rhs,
            None,
        )
    }

    /// Refresh the constant column and solve the predictor together, retaining
    /// the accepted step's constant-column seed.
    pub(crate) fn prepare_direction(
        &mut self,
        problem: &Problem,
        shift: f64,
        d_diagonal: &[f64],
        d_factor: &[f64],
        predictor_rhs: &[f64],
    ) -> Result<Vec<f64>, LinearSystemError> {
        let n = problem.variable_count();
        if self.constant_rhs.is_empty() {
            self.constant_rhs.reserve(n + problem.constraint_count());
            self.constant_rhs
                .extend(problem.c().iter().map(|value| -value));
            self.constant_rhs.extend_from_slice(problem.b());
        }
        let initial = (!self.direction.is_empty()).then(|| self.direction.split_at(n));
        let pattern = self
            .pattern
            .as_ref()
            .expect("prepare stores the KKT layout");
        let factorization = self
            .factorization
            .as_mut()
            .expect("prepare stores the KKT factors");
        let (next, predictor) = self.workspace.solve_pair(
            pattern,
            problem,
            factorization,
            shift,
            d_diagonal,
            d_factor,
            &self.constant_rhs,
            initial,
            predictor_rhs,
        )?;
        self.direction_before_step = Some(std::mem::replace(&mut self.direction, next));
        Ok(predictor)
    }

    pub(crate) fn direction(&self) -> &[f64] {
        &self.direction
    }

    pub(crate) fn take_reconstruction(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.workspace.reconstruction)
    }

    pub(crate) fn put_reconstruction(&mut self, buffer: Vec<f64>) {
        self.workspace.reconstruction = buffer;
    }

    /// Commit the new constant column, or restore the seed of a failed step.
    pub(crate) fn finish_step(&mut self, accepted: bool) {
        if let Some(previous) = self.direction_before_step.take()
            && !accepted
        {
            self.direction = previous;
        }
    }

    pub(crate) fn nonbinding_floor_limit(&self, mu: f64) -> f64 {
        self.pattern.as_ref().map_or(0.0, |pattern| {
            pattern
                .p_diagonal
                .iter()
                .fold(mu, |bound, p| bound.min(p + mu))
        })
    }

    pub(crate) fn stats(&self) -> LinearAlgebraStats {
        let mut stats = self
            .factorization
            .as_ref()
            .map_or_else(LinearAlgebraStats::default, Factorization::stats);
        stats.factorizations = self.factorizations;
        stats.clamp_increases = self.clamp.increases;
        stats.clamp_decreases = self.clamp.decreases;
        stats
    }
}

/// Solve `[P + shift I, A^T; A, -(H + shift I)] z = rhs` for each stacked
/// right-hand side on one factorization. The factor adds `regularization`
/// to both diagonals so that a zero (1,1) block or zero rows of `H` stay
/// quasi-definite; refinement targets the unregularized shifted system.
/// The factorization stays cached for later resolvents to update in place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_shifted_systems(
    problem: &Problem,
    geometry: &ConeGeometry,
    h_diagonal: &[f64],
    shift: f64,
    regularization: f64,
    kkt: &KktSettings,
    cache: &mut KktCache,
    right_hand_sides: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, KktError> {
    if !shift.is_finite()
        || shift < 0.0
        || !regularization.is_finite()
        || regularization <= 0.0
        || h_diagonal.len() != problem.constraint_count()
    {
        return Err(KktError::InvalidParameters);
    }
    let d_diagonal = h_diagonal
        .iter()
        .map(|entry| entry + shift)
        .collect::<Vec<_>>();
    let d_factor = h_diagonal
        .iter()
        .map(|entry| (entry + shift + regularization).max(kkt.diagonal_floor))
        .collect::<Vec<_>>();
    cache.prepare(problem, geometry, shift + regularization, &d_factor, kkt)?;
    right_hand_sides
        .iter()
        .map(|rhs| Ok(cache.solve(problem, shift, &d_diagonal, &d_factor, rhs)?))
        .collect()
}

/// Assemble the clamped preconditioner with bound rows folded into the
/// `(1,1)` block when `eliminate_bound_rows`; `d_diagonal` is already clamped,
/// returning the initial matrix and its fixed layout.
/// Subsequent factorization updates use `KktPattern::write_diagonal` directly.
fn assemble_preconditioner(
    problem: &Problem,
    geometry: &ConeGeometry,
    shift: f64,
    d_diagonal: &[f64],
    diagonal_floor: f64,
    eliminate_bound_rows: bool,
) -> Result<(CscMatrix, KktPattern), KktError> {
    if d_diagonal.len() != problem.constraint_count()
        || !d_diagonal
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err(KktError::InvalidDiagonal);
    }
    // Assembly owns the only full CSC matrix; the cache retains just the
    // metadata needed for diagonal updates and bound-row reconstruction.
    // Numeric retries retain this layout and the prepared backend storage.
    let (mut matrix, pattern) = build_pattern(problem, geometry, eliminate_bound_rows)?;
    let values = matrix.values_mut();
    pattern.write_diagonal(
        values,
        |index| pattern.diagonal_positions[index],
        shift,
        d_diagonal,
        diagonal_floor,
    );
    Ok((matrix, pattern))
}

impl KktPattern {
    /// Clamp the full primal diagonal before adding singleton Schur terms.
    /// `d_diagonal` contains the already clamped factor dual diagonal.
    fn write_diagonal(
        &self,
        values: &mut [f64],
        position: impl Fn(usize) -> usize,
        shift: f64,
        d_diagonal: &[f64],
        delta: f64,
    ) {
        let n = self.variable_count;
        for index in 0..self.diagonal_positions.len() {
            values[position(index)] = if index < n {
                (self.p_diagonal[index] + shift).max(delta)
            } else {
                -d_diagonal[self.bound_rows.kept[index - n]]
            };
        }
        for &(row, column, coefficient) in &self.bound_rows.eliminated {
            values[position(column)] += coefficient * coefficient / d_diagonal[row];
        }
    }

    /// One reduced solve: fold the bound-row right-hand sides into `x`, solve
    /// `n + kept` rows, and recover the eliminated duals.
    fn solve_preconditioner(
        &self,
        factorization: &mut Factorization,
        d_diagonal: &[f64],
        rhs: &[f64],
        reduced: &mut Vec<f64>,
    ) -> Result<Vec<f64>, LinearSystemError> {
        let mut out = Vec::new();
        self.solve_preconditioner_into(factorization, d_diagonal, rhs, &mut out, reduced)?;
        Ok(out)
    }

    fn solve_preconditioner_into(
        &self,
        factorization: &mut Factorization,
        d_diagonal: &[f64],
        rhs: &[f64],
        full: &mut Vec<f64>,
        reduced: &mut Vec<f64>,
    ) -> Result<(), LinearSystemError> {
        if self.bound_rows.eliminated.is_empty() {
            return factorization.solve_unrefined_into(rhs, full);
        }
        reduced.clear();
        self.append_reduced_rhs(d_diagonal, rhs, reduced);
        factorization.solve_unrefined_in_place(reduced)?;
        self.expand_reduced_solution(d_diagonal, rhs, reduced, full);
        Ok(())
    }

    fn append_reduced_rhs(&self, d_diagonal: &[f64], rhs: &[f64], reduced: &mut Vec<f64>) {
        let n = self.variable_count;
        let start = reduced.len();
        reduced.extend_from_slice(&rhs[..n]);
        for &(row, column, coefficient) in &self.bound_rows.eliminated {
            reduced[start + column] += coefficient * rhs[n + row] / d_diagonal[row];
        }
        reduced.extend(self.bound_rows.kept.iter().map(|&row| rhs[n + row]));
    }

    fn expand_reduced_solution(
        &self,
        d_diagonal: &[f64],
        rhs: &[f64],
        reduced: &[f64],
        full: &mut Vec<f64>,
    ) {
        let n = self.variable_count;
        full.resize(rhs.len(), 0.0);
        full[..n].copy_from_slice(&reduced[..n]);
        for (index, &row) in self.bound_rows.kept.iter().enumerate() {
            full[n + row] = reduced[n + index];
        }
        for &(row, column, coefficient) in &self.bound_rows.eliminated {
            full[n + row] = (coefficient * reduced[column] - rhs[n + row]) / d_diagonal[row];
        }
    }

    fn solve_preconditioner_pair(
        &self,
        factorization: &mut Factorization,
        d_diagonal: &[f64],
        first: &[f64],
        second: &[f64],
        reduced: &mut Vec<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>), LinearSystemError> {
        reduced.clear();
        self.append_reduced_rhs(d_diagonal, first, reduced);
        self.append_reduced_rhs(d_diagonal, second, reduced);
        factorization.solve_pair_in_place(reduced)?;
        let (a, b) = reduced.split_at(reduced.len() / 2);
        let (mut first_solution, mut second_solution) = (Vec::new(), Vec::new());
        self.expand_reduced_solution(d_diagonal, first, a, &mut first_solution);
        self.expand_reduced_solution(d_diagonal, second, b, &mut second_solution);
        Ok((first_solution, second_solution))
    }
}

impl KktWorkspace {
    /// Refine against the full true system while applying the clamped
    /// preconditioner, with or without singleton elimination.
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        pattern: &KktPattern,
        problem: &Problem,
        factorization: &mut Factorization,
        shift: f64,
        d_diagonal: &[f64],
        d_factor: &[f64],
        rhs: &[f64],
        initial: Option<(&[f64], &[f64])>,
    ) -> Result<Vec<f64>, LinearSystemError> {
        validate_rhs(problem.variable_count() + problem.constraint_count(), rhs)?;
        factorization.stats_mut().primary_solves += 1;
        let reduced = &mut self.reduced;
        let initial = match initial {
            Some((x, y)) => {
                debug_assert_eq!(x.len(), problem.variable_count());
                debug_assert_eq!(y.len(), problem.constraint_count());
                self.initial_residual.clear();
                self.initial_residual.extend_from_slice(rhs);
                apply_full_system(
                    problem,
                    shift,
                    d_diagonal,
                    (x, y),
                    &mut self.initial_residual,
                );
                // z <- z + M^-1 (rhs - K z). Keeping z preserves components
                // that the diagonally clamped preconditioner corrects slowly.
                let mut solution = pattern.solve_preconditioner(
                    factorization,
                    d_factor,
                    &self.initial_residual,
                    reduced,
                )?;
                let (solution_x, solution_y) = solution.split_at_mut(x.len());
                crate::linear_algebra::add_assign(solution_x, x);
                crate::linear_algebra::add_assign(solution_y, y);
                solution
            }
            None => pattern.solve_preconditioner(factorization, d_factor, rhs, reduced)?,
        };
        self.refine(
            pattern,
            problem,
            factorization,
            shift,
            d_diagonal,
            d_factor,
            rhs,
            initial,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_pair(
        &mut self,
        pattern: &KktPattern,
        problem: &Problem,
        factorization: &mut Factorization,
        shift: f64,
        d_diagonal: &[f64],
        d_factor: &[f64],
        first_rhs: &[f64],
        first_seed: Option<(&[f64], &[f64])>,
        second_rhs: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), LinearSystemError> {
        let dim = problem.variable_count() + problem.constraint_count();
        validate_rhs(dim, first_rhs)?;
        validate_rhs(dim, second_rhs)?;
        factorization.stats_mut().primary_solves += 2;
        let first_correction = if let Some(seed) = first_seed {
            self.initial_residual.clear();
            self.initial_residual.extend_from_slice(first_rhs);
            apply_full_system(problem, shift, d_diagonal, seed, &mut self.initial_residual);
            self.initial_residual.as_slice()
        } else {
            first_rhs
        };
        let (mut first, second) = pattern.solve_preconditioner_pair(
            factorization,
            d_factor,
            first_correction,
            second_rhs,
            &mut self.reduced,
        )?;
        if let Some((x, y)) = first_seed {
            let (first_x, first_y) = first.split_at_mut(x.len());
            crate::linear_algebra::add_assign(first_x, x);
            crate::linear_algebra::add_assign(first_y, y);
        }
        let first = self.refine(
            pattern,
            problem,
            factorization,
            shift,
            d_diagonal,
            d_factor,
            first_rhs,
            first,
        )?;
        let second = self.refine(
            pattern,
            problem,
            factorization,
            shift,
            d_diagonal,
            d_factor,
            second_rhs,
            second,
        )?;
        Ok((first, second))
    }

    #[allow(clippy::too_many_arguments)]
    fn refine(
        &mut self,
        pattern: &KktPattern,
        problem: &Problem,
        factorization: &mut Factorization,
        shift: f64,
        d_diagonal: &[f64],
        d_factor: &[f64],
        rhs: &[f64],
        initial: Vec<f64>,
    ) -> Result<Vec<f64>, LinearSystemError> {
        let reduced = &mut self.reduced;
        let mut stats = LinearAlgebraStats::default();
        let workspace = &mut self.refinement;
        if !self.perturbation_ready {
            workspace.perturbation.clear();
            let floor = factorization.settings().diagonal_floor;
            if floor > shift {
                workspace.perturbation.extend(
                    pattern
                        .p_diagonal
                        .iter()
                        .map(|p| (floor - (p + shift)).max(0.0)),
                );
                workspace.perturbation.extend(
                    d_diagonal
                        .iter()
                        .zip(d_factor)
                        .map(|(target, factor)| target - factor),
                );
            }
            self.perturbation_ready = true;
        }
        let result = refine_with(
            rhs,
            initial,
            &factorization.settings(),
            workspace,
            |solution, rhs, out| {
                out.copy_from_slice(rhs);
                apply_full_system(
                    problem,
                    shift,
                    d_diagonal,
                    solution.split_at(problem.variable_count()),
                    out,
                );
            },
            |solution, rhs, residual, denominator| {
                full_residual_and_denominator(
                    problem,
                    shift,
                    d_diagonal,
                    solution,
                    rhs,
                    residual,
                    denominator,
                );
            },
            |residual, out| {
                pattern.solve_preconditioner_into(factorization, d_factor, residual, out, reduced)
            },
            &mut stats,
        );
        factorization.stats_mut().absorb(stats);
        result
    }
}

/// Assemble the unregularized equality KKT matrix using the shared CSC layout.
pub(crate) fn assemble_equality_kkt(
    problem: &Problem,
    geometry: &ConeGeometry,
) -> Result<CscMatrix, KktError> {
    let (mut matrix, pattern) = build_pattern(problem, geometry, false)?;
    for (column, value) in pattern.p_diagonal.iter().enumerate() {
        matrix.values_mut()[pattern.diagonal_positions[column]] = *value;
    }
    Ok(matrix)
}

/// Assemble fixed off-diagonal entries and retain zero-valued diagonal slots.
fn build_pattern(
    problem: &Problem,
    geometry: &ConeGeometry,
    eliminate_bound_rows: bool,
) -> Result<(CscMatrix, KktPattern), KktError> {
    let n = problem.variable_count();
    let bound_rows = BoundRows::detect(problem, geometry, eliminate_bound_rows);
    let m = bound_rows.kept.len();
    let dim = n.checked_add(m).ok_or(MatrixError::DimensionOverflow)?;
    // Every column has a diagonal. Count the remaining upper-triangle entries
    // before allocating CSC buffers, avoiding triplets and sorting temporaries.
    let mut pointers = vec![1usize; dim + 1];
    pointers[0] = 0;
    if let Some(p) = problem.p() {
        for column in 0..n {
            for at in p.column_pointers()[column]..p.column_pointers()[column + 1] {
                if p.row_indices()[at] != column && p.values()[at] != 0.0 {
                    pointers[column + 1] += 1;
                }
            }
        }
    }
    let mut reduced_rows = vec![usize::MAX; problem.constraint_count()];
    for (row, &source) in bound_rows.kept.iter().enumerate() {
        reduced_rows[source] = n + row;
    }
    crate::linear_algebra::visit_entries(problem.a(), |source, _, value| {
        let column = reduced_rows[source];
        if column != usize::MAX && value != 0.0 {
            pointers[column + 1] += 1;
        }
    });
    for column in 0..dim {
        pointers[column + 1] = pointers[column + 1]
            .checked_add(pointers[column])
            .ok_or(MatrixError::DimensionOverflow)?;
    }
    let mut rows = vec![0; pointers[dim]];
    let mut values = vec![0.0; pointers[dim]];
    let mut next = pointers[..dim].to_vec();
    let mut diagonal_positions = vec![0; dim];
    let mut p_diagonal = vec![0.0; n];
    for column in 0..n {
        if let Some(p) = problem.p() {
            for at in p.column_pointers()[column]..p.column_pointers()[column + 1] {
                let row = p.row_indices()[at];
                let value = p.values()[at];
                if row == column {
                    p_diagonal[column] = value;
                } else if value != 0.0 {
                    rows[next[column]] = row;
                    values[next[column]] = value;
                    next[column] += 1;
                }
            }
        }
        rows[next[column]] = column;
        diagonal_positions[column] = next[column];
    }
    // A is visited in primal-column order, so each column of A^T is sorted.
    crate::linear_algebra::visit_entries(problem.a(), |source, row, value| {
        let column = reduced_rows[source];
        if column != usize::MAX && value != 0.0 {
            rows[next[column]] = row;
            values[next[column]] = value;
            next[column] += 1;
        }
    });
    for column in n..dim {
        rows[next[column]] = column;
        diagonal_positions[column] = next[column];
    }
    let matrix = CscMatrix::new(dim, dim, pointers, rows, values)?;
    Ok((
        matrix,
        KktPattern {
            variable_count: n,
            diagonal_positions,
            p_diagonal,
            bound_rows,
        },
    ))
}

/// `out -= K u` for the full shifted matrix `K = [P + shift I, A^T; A, -diag(d)]`.
pub(crate) fn apply_full_system(
    problem: &Problem,
    shift: f64,
    d_diagonal: &[f64],
    (x, y): (&[f64], &[f64]),
    out: &mut [f64],
) {
    let n = problem.variable_count();
    let (out_x, out_y) = out.split_at_mut(n);
    if let Some(p) = problem.p() {
        crate::linear_algebra::symmetric_accumulate(p, x, [out_x], |product| [-product]);
    }
    crate::linear_algebra::update(out_x, x, |out, x| out - shift * x);
    crate::linear_algebra::constraint_accumulate(problem.a(), x, y, [out_x], [out_y], |product| {
        [-product]
    });
    faer::zip!(
        faer::ColMut::from_slice_mut(out_y),
        faer::ColRef::from_slice(y),
        faer::ColRef::from_slice(d_diagonal)
    )
    .for_each(|faer::unzip!(out, y, d)| *out += d * y);
}

/// Initialize the residual and its componentwise denominator in one sparse pass.
/// Faer has no fused signed/absolute product, so keep this traversal allocation-free.
pub(crate) fn full_residual_and_denominator(
    problem: &Problem,
    shift: f64,
    d_diagonal: &[f64],
    u: &[f64],
    rhs: &[f64],
    residual: &mut [f64],
    denominator: &mut [f64],
) {
    residual.copy_from_slice(rhs);
    for (out, rhs) in denominator.iter_mut().zip(rhs) {
        *out = rhs.abs();
    }
    let n = problem.variable_count();
    let (x, y) = u.split_at(n);
    if let Some(p) = problem.p() {
        crate::linear_algebra::symmetric_accumulate(
            p,
            x,
            [&mut residual[..n], &mut denominator[..n]],
            |product| [-product, product.abs()],
        );
    }
    for (index, x) in x.iter().enumerate() {
        let product = shift * x;
        residual[index] += -product;
        denominator[index] += product.abs();
    }
    let (residual_x, residual_y) = residual.split_at_mut(n);
    let (denominator_x, denominator_y) = denominator.split_at_mut(n);
    crate::linear_algebra::constraint_accumulate(
        problem.a(),
        x,
        y,
        [residual_x, denominator_x],
        [residual_y, denominator_y],
        |product| [-product, product.abs()],
    );
    for (index, (y, d)) in y.iter().zip(d_diagonal).enumerate() {
        let product = -d * y;
        residual[n + index] += -product;
        denominator[n + index] += product.abs();
    }
}

#[cfg(test)]
mod tests {
    use super::{ClampPolicy, KktCache};
    use crate::cone::ConeGeometry;
    use crate::linear_algebra::matrix;
    use crate::linear_solvers::{KktSettings, LinearAlgebraStats};
    use crate::{Cone, CscMatrix, LinearSolver, Problem, Settings};
    use approx::assert_relative_eq;
    use rstest::rstest;

    fn problem() -> Problem {
        Problem::new(
            Some(matrix(2, 2, &[2., 1., 1., 3.])),
            vec![-3., -7.],
            matrix(3, 2, &[1., 0., 0., -2., 1., 2.]),
            vec![1., -4., 5.],
            vec![Cone::Nonnegative(2), Cone::Zero(1)],
        )
        .unwrap()
    }

    fn settings(backend: LinearSolver) -> KktSettings {
        KktSettings::from(&Settings {
            threads: 1,
            linear_solver: backend,
            refine_tol: 1e-12,
            ..Settings::default()
        })
    }

    #[rstest]
    #[case::accurate(0, 0, 0, 1e-9)]
    #[case::clamp_limited(1, 0, 1, 1e-9)]
    #[case::solve_limited(0, 1, 1, 1e-7)]
    #[case::mixed_diagnostics(1, 1, 1, 1e-8)]
    #[case::ambiguous_settling(0, 0, 1, 1e-8)]
    fn accepted_step_adjusts_clamp_from_new_diagnostics(
        #[case] clamp_limited: usize,
        #[case] solve_limited: usize,
        #[case] settled: usize,
        #[case] expected: f64,
    ) {
        let mut policy = ClampPolicy::default();
        let start = policy.start_step(1e-8);
        // Prior steps' diagnostics must not determine this step's decision.
        let before = LinearAlgebraStats {
            refinement_clamp_limited: 3,
            refinement_solve_limited: 4,
            refinement_settled: 7,
            ..LinearAlgebraStats::default()
        };
        let after = LinearAlgebraStats {
            refinement_clamp_limited: before.refinement_clamp_limited + clamp_limited,
            refinement_solve_limited: before.refinement_solve_limited + solve_limited,
            refinement_settled: before.refinement_settled + settled,
            ..before
        };
        policy.accepted_step(start, before, after);
        assert_relative_eq!(policy.floor(), expected, epsilon = 1e-20);
        assert_eq!(policy.increases, usize::from(expected > start));
        assert_eq!(policy.decreases, usize::from(expected < start));
        // Subsequent steps retain the adjusted floor rather than resetting it.
        assert_eq!(policy.start_step(1e-6), policy.floor());
        assert_relative_eq!(policy.floor(), expected, epsilon = 1e-20);
    }

    #[rstest]
    fn accepted_retry_does_not_immediately_relax_the_clamp(
        #[values(false, true)] clamp_limited: bool,
    ) {
        let mut policy = ClampPolicy::default();
        let start = policy.start_step(1e-8);
        let retried = policy.retry(0.).unwrap();
        let before = LinearAlgebraStats::default();
        let after = LinearAlgebraStats {
            refinement_clamp_limited: usize::from(clamp_limited),
            refinement_settled: usize::from(clamp_limited),
            ..before
        };
        policy.accepted_step(start, before, after);
        assert_eq!(policy.floor(), retried);
        assert_eq!(policy.decreases, 0);
    }

    #[test]
    fn clamp_limits_bound_adjustments_and_do_not_count_noops() {
        let mut policy = ClampPolicy::default();
        let start = policy.start_step(0.);
        assert_eq!(start, 1e-20);
        let stats = LinearAlgebraStats::default();
        policy.accepted_step(start, stats, stats);
        assert_eq!(policy.floor(), 1e-20);
        assert_eq!(policy.decreases, 0);

        // No clamp can affect a matrix whose nonbinding limit is the maximum.
        assert_eq!(policy.retry(1e-4), None);
        assert_eq!(policy.floor(), 1e-4);
        let increases = policy.increases;
        assert_eq!(policy.retry(0.), None);
        let limited = LinearAlgebraStats {
            refinement_solve_limited: 1,
            refinement_settled: 1,
            ..stats
        };
        policy.accepted_step(1e-4, stats, limited);
        assert_eq!(policy.floor(), 1e-4);
        assert_eq!(policy.increases, increases);
    }

    fn kkt_product(x: &[f64], shift: f64, d: &[f64]) -> Vec<f64> {
        vec![
            (2. + shift) * x[0] + x[1] + x[2] + x[4],
            x[0] + (3. + shift) * x[1] - 2. * x[3] + 2. * x[4],
            x[0] - d[0] * x[2],
            -2. * x[1] - d[1] * x[3],
            x[0] + 2. * x[1] - d[2] * x[4],
        ]
    }

    #[rstest]
    fn reduced_cached_and_batched_kkt_solves_satisfy_full_equations(
        #[values(LinearSolver::Faer, LinearSolver::Feral)] backend: LinearSolver,
        #[values(false, true)] eliminate: bool,
    ) {
        let p = problem();
        let geometry = ConeGeometry::new(&p).unwrap();
        let expected = [1., -2., 3., -4., 5.];

        let mut settings = settings(backend);
        settings.eliminate_bound_rows = eliminate;
        let mut cache = KktCache::new();
        for (shift, d) in [(0.1, [1., 2., 0.1]), (0.3, [2., 1., 0.3])] {
            let rhs = kkt_product(&expected, shift, &d);
            cache.prepare(&p, &geometry, shift, &d, &settings).unwrap();
            let result = cache.solve(&p, shift, &d, &d, &rhs).unwrap();
            assert_relative_eq!(
                &result[..],
                &expected[..],
                epsilon = 1e-10,
                max_relative = 1e-10
            );
            let predictor = cache.prepare_direction(&p, shift, &d, &d, &rhs).unwrap();
            assert_relative_eq!(
                &predictor[..],
                &expected[..],
                epsilon = 1e-10,
                max_relative = 1e-10
            );
            assert_relative_eq!(
                &kkt_product(cache.direction(), shift, &d)[..],
                &[3., 7., 1., -4., 5.][..],
                epsilon = 1e-10,
                max_relative = 1e-10
            );
            cache.finish_step(true);
            // Factor a perturbed system, but refine against the original one.
            let factor_d = d.map(|v| v + 0.01);
            cache
                .prepare(&p, &geometry, shift + 0.01, &factor_d, &settings)
                .unwrap();
            assert_relative_eq!(
                &cache.solve(&p, shift, &d, &factor_d, &rhs).unwrap()[..],
                &expected[..],
                epsilon = 1e-10,
                max_relative = 1e-10
            );
        }
    }

    #[rstest]
    fn cache_recovers_after_failed_numeric_factorizations(
        #[values(LinearSolver::Faer, LinearSolver::Feral)] backend: LinearSolver,
    ) {
        let p = Problem::new(
            None,
            vec![-2.],
            CscMatrix::zeros(1, 1).unwrap(),
            vec![1.],
            vec![Cone::Zero(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&p).unwrap();
        let mut settings = settings(backend);
        settings.diagonal_floor = 0.;
        let mut cache = KktCache::new();
        for _ in 0..2 {
            assert!(cache.prepare(&p, &geometry, 0., &[1.], &settings).is_err());
        }
        cache.prepare(&p, &geometry, 1., &[1.], &settings).unwrap();
        let x = cache.solve(&p, 1., &[1.], &[1.], &[2., 1.]).unwrap();
        assert_relative_eq!(x.as_slice(), &[2., -1.][..], epsilon = 1e-12);
        assert_eq!(cache.stats().factorizations, 3);
    }
}
