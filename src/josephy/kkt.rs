//! Numerical realization of the paper's quasi-definite system (`eq:qd_system`).
//!
//! Singleton elimination implements the Schur complement in "Reduction of KKT
//! system". Refinement evaluates the full shifted system, including recovered
//! duals. These transformations leave the Josephy equations unchanged.

use super::JosephyError;
use crate::{
    Cone, CscMatrix, MatrixError, Problem,
    linear_solvers::{
        Factorization, KktSettings, LinearAlgebraStats, LinearSystemError, RefinementWorkspace,
        refine_with,
    },
};

/// Per-solve cache: the KKT sparsity pattern and a reusable factorization.
///
/// Only the diagonal of the KKT matrix changes between iterations, so the
/// pattern is assembled once and the factorization's symbolic analysis is
/// reused through `refactor`.
#[derive(Default)]
pub(crate) struct KktCache {
    pub(super) pattern: Option<KktPattern>,
    pub(super) factorization: Option<Factorization>,
    /// The constant `(-c, b)` right-hand side, built on first use.
    pub(super) constant_rhs: Vec<f64>,
    /// Its latest solution, retained across outer iterations of this solve.
    pub(super) direction: Vec<f64>,
    /// Retain ownership of the preceding seed until the outer step succeeds.
    pub(super) direction_before_step: Option<Vec<f64>>,
    /// Adaptive diagonal clamp, applied only to the preconditioner.
    pub(crate) clamp_floor: Option<f64>,
    pub(crate) clamp_increases: usize,
    pub(crate) clamp_decreases: usize,
    /// Numeric dispatches, including failed initial factorizations.
    pub(super) factorizations: usize,
}

/// The reduced KKT matrix pattern: bound rows are eliminated by Schur
/// complement, so the matrix has `n + kept` rows.
pub(super) struct KktPattern {
    variable_count: usize,
    /// Position of each diagonal entry in the matrix's value storage.
    pub(super) diagonal_positions: Vec<usize>,
    /// Diagonal of `P`.
    p_diagonal: Vec<f64>,
    pub(super) bound_rows: BoundRows,
    /// Full-system residual used for the first correction of a warm start.
    initial_residual: Vec<f64>,
    refinement: RefinementWorkspace,
    /// The perturbation is shared by every RHS of the current factorization.
    perturbation_ready: bool,
    /// Scratch storage reused by predictor/corrector reconstruction within a solve.
    pub(super) reconstruction: Vec<f64>,
}

/// Nonnegative singleton rows of `A`, eliminated from the factorized system.
///
/// For such a row `i` with entry `a` in column `j`, the block row
/// `a x_j - d_i y_i = r_i` gives `y_i = (a x_j - r_i) / d_i`, which adds
/// `a^2 / d_i` to the `(j, j)` diagonal and `a r_i / d_i` to the `x_j`
/// right-hand side.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct BoundRows {
    /// `(row, column, coefficient)` of each eliminated row.
    pub(super) eliminated: Vec<(usize, usize, f64)>,
    /// Original row index of each kept row.
    pub(super) kept: Vec<usize>,
}

impl BoundRows {
    fn detect(problem: &Problem, enabled: bool) -> Self {
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
            let mut offset = 0;
            for cone in problem.cones() {
                let dim = cone.vector_dim();
                if !matches!(cone, Cone::Nonnegative(_)) {
                    counts[offset..offset + dim].fill(0);
                }
                offset += dim;
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

/// Factorize the clamped preconditioner `[P + shift I, A^T; A, -diag(d)]`
/// with `d = factor_diagonal`, reusing the cached pattern and symbolic
/// analysis when present, and reject factors with the wrong inertia.
pub(super) fn factorize(
    problem: &Problem,
    shift: f64,
    factor_diagonal: &[f64],
    kkt: &KktSettings,
    cache: &mut KktCache,
) -> Result<(), JosephyError> {
    if let Some(pattern) = cache.pattern.as_mut() {
        pattern.perturbation_ready = false;
    }
    match (cache.factorization.as_mut(), cache.pattern.as_ref()) {
        (Some(factorization), Some(pattern)) => {
            factorization.set_strict_refinement(kkt.strict_refinement);
            factorization.set_clamp_floor(kkt.diagonal_floor);
            factorization.update_diagonal_and_refactor(&mut cache.factorizations, |values| {
                pattern.write_diagonal(
                    values,
                    |index| index,
                    problem.variable_count(),
                    shift,
                    factor_diagonal,
                    kkt.diagonal_floor,
                )
            })?;
        }
        _ => {
            let k = assemble_preconditioner(
                problem,
                shift,
                factor_diagonal,
                kkt.diagonal_floor,
                kkt.eliminate_bound_rows,
                cache,
            )?;
            cache.factorization = Some(Factorization::new_counted(
                k,
                kkt,
                &mut cache.factorizations,
            )?);
        }
    }
    cache
        .factorization
        .as_mut()
        .expect("the factorization was just stored")
        .check_inertia(problem.variable_count())
        .map_err(JosephyError::from)
}

/// Assemble the clamped preconditioner with bound rows folded into the
/// `(1,1)` block when `eliminate_bound_rows`; `d_diagonal` is already clamped,
/// returning the initial matrix and retaining its update metadata in the cache.
/// Subsequent factorization updates use `KktPattern::write_diagonal` directly.
pub(crate) fn assemble_preconditioner(
    problem: &Problem,
    shift: f64,
    d_diagonal: &[f64],
    diagonal_floor: f64,
    eliminate_bound_rows: bool,
    cache: &mut KktCache,
) -> Result<CscMatrix, JosephyError> {
    if d_diagonal.len() != problem.constraint_count()
        || !d_diagonal
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err(JosephyError::InvalidTauPolynomial);
    }
    // Assembly owns the only full CSC matrix; the cache retains just the
    // metadata needed for diagonal updates and bound-row reconstruction.
    // A failed initial factorization can rebuild this same pattern on retry.
    let (mut matrix, pattern) = build_pattern(problem, shift, d_diagonal, eliminate_bound_rows)?;
    let values = matrix.values_mut();
    pattern.write_diagonal(
        values,
        |index| pattern.diagonal_positions[index],
        problem.variable_count(),
        shift,
        d_diagonal,
        diagonal_floor,
    );
    cache.pattern = Some(pattern);
    Ok(matrix)
}

impl KktPattern {
    /// Clamp the full primal diagonal before adding singleton Schur terms.
    /// `d_diagonal` contains the already clamped factor dual diagonal.
    pub(super) fn write_diagonal(
        &self,
        values: &mut [f64],
        position: impl Fn(usize) -> usize,
        n: usize,
        shift: f64,
        d_diagonal: &[f64],
        delta: f64,
    ) {
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

    /// Refine against the full true system while applying the clamped
    /// preconditioner, with or without singleton elimination.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve(
        &mut self,
        problem: &Problem,
        factorization: &mut Factorization,
        shift: f64,
        d_diagonal: &[f64],
        d_factor: &[f64],
        rhs: &[f64],
        initial: Option<(&[f64], &[f64])>,
    ) -> Result<Vec<f64>, LinearSystemError> {
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
                let mut solution =
                    self.solve_preconditioner(factorization, d_factor, &self.initial_residual)?;
                let (solution_x, solution_y) = solution.split_at_mut(x.len());
                crate::linear_algebra::add_assign(solution_x, x);
                crate::linear_algebra::add_assign(solution_y, y);
                solution
            }
            None => self.solve_preconditioner(factorization, d_factor, rhs)?,
        };
        let mut stats = LinearAlgebraStats::default();
        let mut workspace = std::mem::take(&mut self.refinement);
        if !self.perturbation_ready {
            workspace.perturbation.clear();
            let floor = factorization.settings().diagonal_floor;
            if floor > shift {
                workspace.perturbation.extend(
                    self.p_diagonal
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
            &mut workspace,
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
            |residual, out| self.solve_preconditioner_into(factorization, d_factor, residual, out),
            &mut stats,
        );
        self.refinement = workspace;
        factorization.stats_mut().absorb(stats);
        result
    }

    /// One reduced solve: fold the bound-row right-hand sides into `x`, solve
    /// `n + kept` rows, and recover the eliminated duals.
    pub(super) fn solve_preconditioner(
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

pub(super) fn build_pattern(
    problem: &Problem,
    shift: f64,
    d_diagonal: &[f64],
    eliminate_bound_rows: bool,
) -> Result<(CscMatrix, KktPattern), JosephyError> {
    let n = problem.variable_count();
    let bound_rows = BoundRows::detect(problem, eliminate_bound_rows);
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
            initial_residual: Vec::new(),
            refinement: RefinementWorkspace::default(),
            perturbation_ready: false,
            reconstruction: Vec::new(),
        },
    ))
}

fn assemble_sparse_k(
    problem: &Problem,
    shift: f64,
    d_diagonal: &[f64],
    bound_rows: &BoundRows,
) -> Result<CscMatrix, JosephyError> {
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
pub(super) fn apply_full_system(
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
pub(super) fn full_residual_and_denominator(
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

pub(super) fn matrix_value(matrix: &CscMatrix, row: usize, column: usize) -> f64 {
    matrix.get(row.min(column), row.max(column)).unwrap_or(0.0)
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::LinearSolver;

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
        let shift = 0.1;
        let target = [0.15_f64];
        let mut cache = KktCache::new();
        for floor in [0.5, 0.3, 1e-12] {
            let settings = KktSettings {
                diagonal_floor: floor,
                ..KktSettings::new(LinearSolver::Faer)
            };
            let factor = [target[0].max(floor)];
            factorize(&problem, shift, &factor, &settings, &mut cache).unwrap();
            let pattern = cache.pattern.as_mut().unwrap();
            let factorization = cache.factorization.as_mut().unwrap();
            assert!(!pattern.perturbation_ready);
            for rhs in [[1.0, 2.0], [-3.0, 1.0]] {
                let result = pattern
                    .solve(&problem, factorization, shift, &target, &factor, &rhs, None)
                    .unwrap();
                assert!((1.1 * result[0] + result[1] - rhs[0]).abs() < 1e-5);
                assert!((result[0] - 0.15 * result[1] - rhs[1]).abs() < 1e-5);
                assert!(pattern.perturbation_ready);
                if floor > shift {
                    assert_eq!(
                        pattern.refinement.perturbation,
                        vec![0.0, target[0] - factor[0]]
                    );
                } else {
                    assert!(pattern.refinement.perturbation.is_empty());
                }
            }
        }
    }
}
