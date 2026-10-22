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
        matches!(
            self,
            Self::InvalidDiagonal
                | Self::LinearSystem(
                    LinearSystemError::Singular
                        | LinearSystemError::WrongInertia { .. }
                        | LinearSystemError::RefinementDiverged(_)
                        | LinearSystemError::Factorization(_)
                        | LinearSystemError::NonFiniteRightHandSide
                )
        )
    }
}

/// Per-solve cache: the KKT sparsity pattern and a reusable factorization.
///
/// Only the diagonal of the KKT matrix changes between iterations, so the
/// layout and symbolic analysis are reused after the first successful
/// factorization. Failed initial attempts retain the layout for clamp decisions.
#[derive(Default)]
pub(crate) struct KktCache {
    // Layout may exist before the first successful numeric factorization.
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
    pub(crate) clamp_floor: Option<f64>,
    pub(crate) clamp_increases: usize,
    pub(crate) clamp_decreases: usize,
    /// Numeric dispatches, including failed initial factorizations.
    factorizations: usize,
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
        match (self.factorization.as_mut(), self.pattern.as_ref()) {
            (Some(factorization), Some(pattern)) => {
                factorization.set_strict_refinement(kkt.strict_refinement);
                factorization.set_clamp_floor(kkt.diagonal_floor);
                factorization.update_diagonal_and_refactor(&mut self.factorizations, |values| {
                    pattern.write_diagonal(
                        values,
                        |index| index,
                        shift,
                        factor_diagonal,
                        kkt.diagonal_floor,
                    )
                })?;
            }
            _ => {
                let (k, pattern) = assemble_preconditioner(
                    problem,
                    geometry,
                    shift,
                    factor_diagonal,
                    kkt.diagonal_floor,
                    kkt.eliminate_bound_rows,
                )?;
                self.pattern = Some(pattern);
                self.factorization = Some(Factorization::new_counted(
                    k,
                    kkt,
                    &mut self.factorizations,
                )?);
            }
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

    /// Refresh the constant column while retaining the accepted step's seed.
    pub(crate) fn prepare_direction(
        &mut self,
        problem: &Problem,
        shift: f64,
        d_diagonal: &[f64],
        d_factor: &[f64],
    ) -> Result<(), LinearSystemError> {
        let n = problem.variable_count();
        if self.constant_rhs.is_empty() {
            self.constant_rhs.reserve(n + problem.constraint_count());
            self.constant_rhs
                .extend(problem.c().iter().map(|value| -value));
            self.constant_rhs.extend_from_slice(problem.b());
        }
        let initial = (!self.direction.is_empty()).then(|| self.direction.split_at(n));
        let next = self.workspace.solve(
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
            &self.constant_rhs,
            initial,
        )?;
        self.direction_before_step = Some(std::mem::replace(&mut self.direction, next));
        Ok(())
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
        stats.clamp_increases = self.clamp_increases;
        stats.clamp_decreases = self.clamp_decreases;
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
    // A failed initial factorization can rebuild this same pattern on retry.
    let (mut matrix, pattern) =
        build_pattern(problem, geometry, shift, d_diagonal, eliminate_bound_rows)?;
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
    ) -> Result<Vec<f64>, LinearSystemError> {
        let mut out = Vec::new();
        self.solve_preconditioner_into(factorization, d_diagonal, rhs, &mut out)?;
        Ok(out)
    }

    fn solve_preconditioner_into(
        &self,
        factorization: &mut Factorization,
        d_diagonal: &[f64],
        rhs: &[f64],
        full: &mut Vec<f64>,
    ) -> Result<(), LinearSystemError> {
        if self.bound_rows.eliminated.is_empty() {
            return factorization.solve_unrefined_into(rhs, full);
        }
        let n = self.variable_count;
        let bound_rows = &self.bound_rows;
        let mut reduced = factorization.take_scratch();
        reduced.clear();
        reduced.extend_from_slice(&rhs[..n]);
        for &(row, column, coefficient) in &bound_rows.eliminated {
            reduced[column] += coefficient * rhs[n + row] / d_diagonal[row];
        }
        reduced.extend(bound_rows.kept.iter().map(|&row| rhs[n + row]));
        let solved = factorization.solve_unrefined_in_place(&mut reduced);
        let result = solved.map(|()| {
            full.resize(rhs.len(), 0.0);
            full[..n].copy_from_slice(&reduced[..n]);
            for (index, &row) in bound_rows.kept.iter().enumerate() {
                full[n + row] = reduced[n + index];
            }
            for &(row, column, coefficient) in &bound_rows.eliminated {
                full[n + row] = (coefficient * reduced[column] - rhs[n + row]) / d_diagonal[row];
            }
        });
        factorization.put_scratch(reduced);
        result
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
                    false,
                );
                // z <- z + M^-1 (rhs - K z). Keeping z preserves components
                // that the diagonally clamped preconditioner corrects slowly.
                let mut solution = pattern.solve_preconditioner(
                    factorization,
                    d_factor,
                    &self.initial_residual,
                )?;
                let (solution_x, solution_y) = solution.split_at_mut(x.len());
                crate::linear_algebra::add_assign(solution_x, x);
                crate::linear_algebra::add_assign(solution_y, y);
                solution
            }
            None => pattern.solve_preconditioner(factorization, d_factor, rhs)?,
        };
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
                    false,
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
                pattern.solve_preconditioner_into(factorization, d_factor, residual, out)
            },
            &mut stats,
        );
        factorization.stats_mut().absorb(stats);
        result
    }
}

fn build_pattern(
    problem: &Problem,
    geometry: &ConeGeometry,
    shift: f64,
    d_diagonal: &[f64],
    eliminate_bound_rows: bool,
) -> Result<(CscMatrix, KktPattern), KktError> {
    let n = problem.variable_count();
    let bound_rows = BoundRows::detect(problem, geometry, eliminate_bound_rows);
    let matrix = assemble_sparse_k(problem, shift, d_diagonal, &bound_rows)?;
    let dim = n + bound_rows.kept.len();
    let diagonal_positions = {
        let matrix = &matrix;
        (0..dim)
            .map(|column| {
                (matrix.column_pointers()[column]..matrix.column_pointers()[column + 1])
                    .find(|index| matrix.row_indices()[*index] == column)
                    .expect("assembly stores every diagonal entry")
            })
            .collect()
    };
    let mut p_diagonal = vec![0.0; n];
    if let Some(p) = problem.p() {
        for (index, value) in p_diagonal.iter_mut().enumerate() {
            *value = matrix_value(p, index, index);
        }
    }
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

fn assemble_sparse_k(
    problem: &Problem,
    shift: f64,
    d_diagonal: &[f64],
    bound_rows: &BoundRows,
) -> Result<CscMatrix, KktError> {
    let n = problem.variable_count();
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
    for column in 0..n {
        let mut diagonal = shift;
        if let Some(p) = problem.p() {
            for at in p.column_pointers()[column]..p.column_pointers()[column + 1] {
                let row = p.row_indices()[at];
                let value = p.values()[at];
                if row == column {
                    diagonal += value;
                } else if value != 0.0 {
                    rows[next[column]] = row;
                    values[next[column]] = value;
                    next[column] += 1;
                }
            }
        }
        rows[next[column]] = column;
        values[next[column]] = diagonal;
        next[column] += 1;
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
    for (row, &source) in bound_rows.kept.iter().enumerate() {
        let column = n + row;
        rows[next[column]] = column;
        values[next[column]] = -d_diagonal[source];
    }
    Ok(CscMatrix::new(dim, dim, pointers, rows, values)?)
}

/// `out -= K u` for the full shifted matrix `K = [P + shift I, A^T; A, -diag(d)]`,
/// or `out += |K| |u|` when `magnitudes`.
pub(crate) fn apply_full_system(
    problem: &Problem,
    shift: f64,
    d_diagonal: &[f64],
    (x, y): (&[f64], &[f64]),
    out: &mut [f64],
    magnitudes: bool,
) {
    let n = problem.variable_count();
    let (out_x, out_y) = out.split_at_mut(n);
    let sign = |value: f64| if magnitudes { value.abs() } else { -value };
    if let Some(p) = problem.p() {
        crate::linear_algebra::symmetric_accumulate(p, x, [out_x], |product| [sign(product)]);
    }
    crate::linear_algebra::update(out_x, x, |out, x| out + sign(shift * x));
    crate::linear_algebra::constraint_accumulate(problem.a(), x, y, [out_x], [out_y], |product| {
        [sign(product)]
    });
    faer::zip!(
        faer::ColMut::from_slice_mut(out_y),
        faer::ColRef::from_slice(y),
        faer::ColRef::from_slice(d_diagonal)
    )
    .for_each(|faer::unzip!(out, y, d)| *out += sign(-d * y));
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

pub(crate) fn matrix_value(matrix: &CscMatrix, row: usize, column: usize) -> f64 {
    matrix.get(row.min(column), row.max(column)).unwrap_or(0.0)
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::{Cone, LinearSolver};

    #[test]
    fn failed_factorization_retains_layout_and_can_be_retried() {
        let problem = Problem::new(
            None,
            vec![-2.0],
            CscMatrix::zeros(1, 1).unwrap(),
            vec![1.0],
            vec![Cone::Zero(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        for solver in [LinearSolver::Faer, LinearSolver::Feral] {
            let settings = KktSettings {
                diagonal_floor: 0.0,
                ..KktSettings::new(solver)
            };
            let mut cache = KktCache::new();
            let error = cache
                .prepare(&problem, &geometry, 0.0, &[1.0], &settings)
                .unwrap_err();
            assert!(cache.pattern.is_some());
            // Inertia rejection happens after construction; numeric failure does not.
            assert_eq!(
                cache.factorization.is_some(),
                matches!(
                    error,
                    KktError::LinearSystem(LinearSystemError::WrongInertia { .. })
                )
            );
            assert_eq!(cache.stats().factorizations, 1);
            assert_eq!(cache.nonbinding_floor_limit(0.1), 0.1);
            cache
                .prepare(&problem, &geometry, 1.0, &[1.0], &settings)
                .unwrap();
            let result = cache
                .solve(&problem, 1.0, &[1.0], &[1.0], &[2.0, 1.0])
                .unwrap();
            assert_eq!(result, vec![2.0, -1.0]);
            assert_eq!(cache.stats().factorizations, 2);
        }
    }

    #[test]
    fn rejected_step_restores_the_accepted_direction() {
        let problem = Problem::new(
            None,
            vec![-2.0],
            CscMatrix::zeros(1, 1).unwrap(),
            vec![1.0],
            vec![Cone::Zero(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let settings = KktSettings::new(LinearSolver::Faer);
        let mut cache = KktCache::new();
        for (shift, accepted, expected) in [
            (1.0, false, vec![]),
            (1.0, true, vec![2.0, -1.0]),
            (2.0, false, vec![2.0, -1.0]),
            (2.0, true, vec![1.0, -0.5]),
        ] {
            cache
                .prepare(&problem, &geometry, shift, &[shift], &settings)
                .unwrap();
            cache
                .prepare_direction(&problem, shift, &[shift], &[shift])
                .unwrap();
            assert_eq!(cache.direction(), &[2.0 / shift, -1.0 / shift]);
            cache.finish_step(accepted);
            assert_eq!(cache.direction(), expected);
        }
        assert_eq!(cache.stats().factorizations, 4);
        assert_eq!(cache.stats().primary_solves, 4);
    }

    #[test]
    fn refinement_solves_the_true_system_with_a_clamped_primal_diagonal() {
        let problem = Problem::new(
            Some(CscMatrix::from_triplets(1, 1, vec![0], vec![0], vec![1e-14]).unwrap()),
            vec![0.0],
            CscMatrix::identity(1).unwrap(),
            vec![0.0],
            vec![Cone::Nonnegative(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let settings = KktSettings::from(&crate::Settings {
            eliminate_bounds: false,
            refine_init_reg: 1e-6,
            refine_tol: 1e-12,
            ..crate::Settings::default()
        });
        let mut cache = KktCache::new();
        cache
            .prepare(&problem, &geometry, 0.0, &[1.0], &settings)
            .unwrap();
        let solution = cache
            .solve(&problem, 0.0, &[1.0], &[1.0], &[2.0 + 1e-14, -1.0])
            .unwrap();
        assert!((solution[0] - 1.0).abs() < 1e-10);
        assert!((solution[1] - 2.0).abs() < 1e-10);
        assert!(cache.stats().refinement_solves > 0);
        assert_eq!(cache.stats().refinement_accurate, 1);
    }

    #[test]
    fn perturbation_is_reused_and_invalidated_after_refactorization() {
        let problem = Problem::new(
            Some(CscMatrix::identity(1).unwrap()),
            vec![0.0],
            CscMatrix::identity(1).unwrap(),
            vec![0.0],
            vec![Cone::Nonnegative(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let shift = 0.1;
        let target = [0.15_f64];
        let mut cache = KktCache::new();
        for floor in [0.5, 0.3, 1e-12] {
            let settings = KktSettings {
                diagonal_floor: floor,
                ..KktSettings::new(LinearSolver::Faer)
            };
            let factor = [target[0].max(floor)];
            cache
                .prepare(&problem, &geometry, shift, &factor, &settings)
                .unwrap();
            assert!(!cache.workspace.perturbation_ready);
            for rhs in [[1.0, 2.0], [-3.0, 1.0]] {
                let result = cache
                    .solve(&problem, shift, &target, &factor, &rhs)
                    .unwrap();
                assert!((1.1 * result[0] + result[1] - rhs[0]).abs() < 1e-5);
                assert!((result[0] - 0.15 * result[1] - rhs[1]).abs() < 1e-5);
                assert!(cache.workspace.perturbation_ready);
                if floor > shift {
                    assert_eq!(
                        cache.workspace.refinement.perturbation,
                        vec![0.0, target[0] - factor[0]]
                    );
                } else {
                    assert!(cache.workspace.refinement.perturbation.is_empty());
                }
            }
        }
    }
}
