//! Closed-form resolvent step for the Josephy--Newton subproblem.

use std::cell::OnceCell;

use crate::matrix::dot;
use thiserror::Error;

use crate::{
    Cone, CscMatrix, MatrixError, Problem,
    embedding::{EmbeddingError, State},
    linear_system::{
        Factorization, KktSettings, LinearAlgebraStats, LinearSystemError, RefinementWorkspace,
        refine_with,
    },
};

/// Failures while constructing a Josephy--Newton solve.
#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum JosephyError {
    #[error(transparent)]
    Embedding(#[from] EmbeddingError),
    #[error(transparent)]
    LinearSystem(#[from] LinearSystemError),
    #[error(transparent)]
    Matrix(#[from] MatrixError),
    #[error("the Josephy tau polynomial has invalid coefficients")]
    InvalidTauPolynomial,
    #[error("resolvent right-hand side {argument} has length {actual}, expected {expected}")]
    IncorrectRhsLength {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl JosephyError {
    /// Failures that a larger shift or best-effort refinement may cure.
    pub(crate) fn is_numerical(&self) -> bool {
        matches!(
            self,
            Self::InvalidTauPolynomial
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

/// One solution of the shifted system of Theorem "closed-form resolvent step".
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolventSolution {
    /// Contiguous primal/dual result, moved directly into an accepted State.
    pub(crate) xy: Vec<f64>,
    pub(crate) primal_len: usize,
    /// Nonnegative subproblem candidate; the accepted iterate stays positive.
    pub(crate) tau: f64,
    pub(crate) used_tau_fallback: bool,
}

impl ResolventSolution {
    pub(crate) fn x(&self) -> &[f64] {
        &self.xy[..self.primal_len]
    }

    pub(crate) fn y(&self) -> &[f64] {
        &self.xy[self.primal_len..]
    }
}

/// Per-solve cache: the KKT sparsity pattern and a reusable factorization.
///
/// Only the diagonal of the KKT matrix changes between iterations, so the
/// pattern is assembled once and the factorization's symbolic analysis is
/// reused through `refactor`.
#[derive(Default)]
pub(crate) struct KktCache {
    pattern: Option<KktPattern>,
    factorization: Option<Factorization>,
    /// The constant `(-c, b)` right-hand side, built on first use.
    constant_rhs: Vec<f64>,
    /// Its latest solution, retained across outer iterations of this solve.
    direction: Vec<f64>,
    /// Retain ownership of the preceding seed until the outer step succeeds.
    direction_before_step: Option<Vec<f64>>,
    /// Adaptive true-shift floor, distinct from factorization regularization.
    pub(crate) shift_floor: Option<f64>,
    /// Numeric dispatches, including failed initial factorizations.
    factorizations: usize,
    /// Extra correctors counted by `mehrotra::step`.
    pub(crate) corrector_tries: usize,
    pub(crate) corrector_accepts: usize,
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
    /// Full-system residual used for the first correction of a warm start.
    initial_residual: Vec<f64>,
    refinement: RefinementWorkspace,
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
    fn detect(problem: &Problem, enabled: bool) -> Self {
        let m = problem.constraint_count();
        let mut singleton = vec![None; m];
        let mut counts = vec![0usize; m];
        if enabled {
            crate::matrix::visit_entries(problem.a(), |row, column, value| {
                if value != 0.0 {
                    counts[row] += 1;
                    singleton[row] = Some((column, value));
                }
            });
            let mut offset = 0;
            for cone in problem.cones() {
                let dim = cone.vector_dim().unwrap_or(0);
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

    pub(crate) fn stats(&self) -> LinearAlgebraStats {
        let mut stats = self
            .factorization
            .as_ref()
            .map_or_else(LinearAlgebraStats::default, Factorization::stats);
        stats.factorizations = self.factorizations;
        stats.corrector_tries = self.corrector_tries;
        stats.corrector_accepts = self.corrector_accepts;
        stats
    }
}

/// The factorized shifted matrix and its target-independent column.
pub(crate) struct Resolvent<'a> {
    problem: &'a Problem,
    pattern: &'a mut KktPattern,
    factorization: &'a mut Factorization,
    shift: f64,
    d_diagonal: Vec<f64>,
    /// Present only when static regularization modifies the true diagonal.
    d_factor: Option<Vec<f64>>,
    /// `(x2, y2)`, the solution of the `(-c, b)` column.
    direction: &'a [f64],
    alpha: f64,
    /// Current iterate, not the corrector surrogate, for the tau fallback.
    state: &'a State,
    /// Computed only if a root fails; shared by subsequent RHS solves.
    fallback: OnceCell<TauFallback>,
}

struct TauFallback {
    q_direction: f64,
    px: Option<Vec<f64>>,
    q_z: f64,
    x_px: f64,
    px_kinv_q: f64,
}

impl<'a> Resolvent<'a> {
    /// Factorize a diagonally clamped preconditioner for the true system
    /// `[P + shift I, A^T; A, -(H + shift I)]`, then refine its `(-c, b)` solve.
    pub(crate) fn new(
        problem: &'a Problem,
        state: &'a State,
        h_diagonal: &[f64],
        shift: f64,
        kkt: &KktSettings,
        cache: &'a mut KktCache,
    ) -> Result<Self, JosephyError> {
        if !shift.is_finite() || shift <= 0.0 {
            return Err(EmbeddingError::InvalidMu.into());
        }
        let d_diagonal = h_diagonal
            .iter()
            .map(|entry| entry + shift)
            .collect::<Vec<_>>();
        let d_factor = (kkt.min_static_regularization > shift).then(|| {
            d_diagonal
                .iter()
                .map(|d| d.max(kkt.min_static_regularization))
                .collect::<Vec<_>>()
        });
        let factor_diagonal = d_factor.as_deref().unwrap_or(&d_diagonal);
        factorize(problem, shift, factor_diagonal, kkt, cache)?;
        let KktCache {
            pattern,
            factorization,
            constant_rhs,
            direction,
            direction_before_step,
            ..
        } = cache;
        let pattern = pattern.as_mut().expect("the pattern was just stored");
        let factorization = factorization
            .as_mut()
            .expect("the factorization was just stored");

        let n = problem.variable_count();
        if constant_rhs.is_empty() {
            constant_rhs.reserve(n + problem.constraint_count());
            constant_rhs.extend(problem.c().iter().map(|value| -value));
            constant_rhs.extend_from_slice(problem.b());
        }
        let initial = (!direction.is_empty()).then(|| direction.split_at(n));
        let next_direction = pattern.solve(
            problem,
            factorization,
            shift,
            &d_diagonal,
            factor_diagonal,
            constant_rhs,
            initial,
        )?;
        *direction_before_step = Some(std::mem::replace(direction, next_direction));
        let (x2, y2) = direction.split_at(n);
        // Positive identity from the KKT equations, with d = H + shift I.
        let alpha = tau_leading_coefficient(shift, x2, y2, &d_diagonal);

        Ok(Self {
            problem,
            pattern,
            factorization,
            shift,
            d_diagonal,
            d_factor,
            direction,
            alpha,
            state,
            fallback: OnceCell::new(),
        })
    }

    /// Solve the shifted system with right-hand side `(r_x, r_y, r_tau)` at
    /// target `mu`, accepting zero tau candidates without a boundary marker.
    #[allow(dead_code)]
    pub(crate) fn solve(
        &mut self,
        r_x: &[f64],
        r_y: &[f64],
        r_tau: f64,
        mu: f64,
    ) -> Result<ResolventSolution, JosephyError> {
        let n = self.problem.variable_count();
        let m = self.problem.constraint_count();
        ensure_rhs_length("r_x", n, r_x)?;
        ensure_rhs_length("r_y", m, r_y)?;
        let mut rhs = Vec::with_capacity(n + m);
        rhs.extend_from_slice(r_x);
        rhs.extend(r_y.iter().map(|value| -value));
        self.solve_rhs(&rhs, r_tau, mu)
    }

    /// `solve` with the stacked right-hand side `(r_x, -r_y)` prebuilt.
    /// Solve this changing right-hand side directly; only the constant data
    /// direction is carried between factorizations.
    pub(crate) fn solve_rhs(
        &mut self,
        rhs: &[f64],
        r_tau: f64,
        mu: f64,
    ) -> Result<ResolventSolution, JosephyError> {
        let n = self.problem.variable_count();
        let m = self.problem.constraint_count();
        ensure_rhs_length("rhs", n + m, rhs)?;
        if !r_tau.is_finite() || !mu.is_finite() || mu < 0.0 {
            return Err(JosephyError::InvalidTauPolynomial);
        }

        let mut constant = self.pattern.solve(
            self.problem,
            self.factorization,
            self.shift,
            &self.d_diagonal,
            self.d_factor.as_deref().unwrap_or(&self.d_diagonal),
            rhs,
            None,
        )?;
        let q_constant = q_dot(self.problem, &constant);
        let (tau, used_tau_fallback) = match self.solve_tau(&constant, q_constant, r_tau, mu) {
            Some(tau) => (tau, false),
            None => (self.linearized_tau(&constant, q_constant, r_tau, mu), true),
        };
        crate::matrix::update(&mut constant, self.direction, |constant, direction| {
            tau.mul_add(direction, constant)
        });
        Ok(ResolventSolution {
            xy: constant,
            primal_len: n,
            tau,
            used_tau_fallback,
        })
    }

    // Preserve the acceptance thresholds and fallback
    // policy, with a positive KKT identity and scaled root arithmetic.
    fn solve_tau(&self, constant: &[f64], q_constant: f64, r_tau: f64, mu: f64) -> Option<f64> {
        let n = self.problem.variable_count();
        let x1 = &constant[..n];
        let x2 = &self.direction[..n];
        let mut b = -r_tau - q_constant;
        let mut c = -mu;
        if let Some(px1) = self.problem.p_mul_opt(x1) {
            // Symmetry gives x1' P x2 = x2' P x1; reuse P x1 for both terms.
            b -= 2.0 * dot(x2, &px1);
            c -= dot(x1, &px1);
        }
        tau_root(self.alpha, b, c)
    }

    fn linearized_tau(&self, constant: &[f64], q_constant: f64, r_tau: f64, mu: f64) -> f64 {
        let n = self.problem.variable_count();
        let tau = self.state.tau();
        let cached = self.fallback.get_or_init(|| {
            let px = self.problem.p_mul_opt(self.state.x());
            TauFallback {
                q_direction: q_dot(self.problem, self.direction),
                q_z: dot(self.problem.c(), self.state.x()) + dot(self.problem.b(), self.state.y()),
                x_px: px.as_ref().map_or(0.0, |px| dot(self.state.x(), px)),
                px_kinv_q: px.as_ref().map_or(0.0, |px| -dot(px, &self.direction[..n])),
                px,
            }
        });
        let q_kinv_q = -cached.q_direction;
        let px_kinv_r = cached.px.as_ref().map_or(0.0, |px| dot(px, &constant[..n]));
        // Linearize x'Px at the current x but retain the quadratic in tau.
        // Unlike the primary solve, any finite root is clamped to the bounds.
        let candidate = quadratic_root(
            self.shift + q_kinv_q,
            -r_tau - q_constant + 2.0 * cached.px_kinv_q,
            -mu + cached.x_px - 2.0 * px_kinv_r,
            0.0,
        );
        if let Some(candidate) = candidate {
            return candidate.max(0.1 * tau).min(10.0 * tau);
        }

        // If that quadratic fails, take CPO3's first-order correction.
        let q_rz = q_constant - tau * q_kinv_q - cached.q_z;
        let px_rz = px_kinv_r - tau * cached.px_kinv_q - cached.x_px;
        // r_tau retains the caller's anchor even though we linearize at state.
        let g = self.shift * tau * tau - r_tau * tau - tau * cached.q_z - mu - cached.x_px;
        let num = g - tau * q_rz - 2.0 * px_rz;
        let den =
            2.0 * self.shift * tau - r_tau - cached.q_z + tau * q_kinv_q + 2.0 * cached.px_kinv_q;
        let candidate = tau + if den.abs() < 1e-16 { 0.0 } else { -num / den };
        if !candidate.is_finite() {
            return tau;
        }
        candidate.max(0.1 * tau).min(10.0 * tau)
    }
}

/// Factorize the clamped preconditioner `[P + shift I, A^T; A, -diag(d)]`
/// with `d = factor_diagonal`, reusing the cached pattern and symbolic
/// analysis when present, and reject factors with the wrong inertia.
fn factorize(
    problem: &Problem,
    shift: f64,
    factor_diagonal: &[f64],
    kkt: &KktSettings,
    cache: &mut KktCache,
) -> Result<(), JosephyError> {
    match (cache.factorization.as_mut(), cache.pattern.as_ref()) {
        (Some(factorization), Some(pattern)) => {
            factorization.set_strict_refinement(kkt.strict_refinement);
            factorization.update_diagonal_and_refactor(&mut cache.factorizations, |values| {
                pattern.write_diagonal(
                    values,
                    |index| index,
                    problem.variable_count(),
                    shift,
                    factor_diagonal,
                    kkt.min_static_regularization,
                )
            })?;
        }
        _ => {
            let k = assemble_preconditioner(
                problem,
                shift,
                factor_diagonal,
                kkt.min_static_regularization,
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

/// Solve `[P + shift I, A^T; A, -(H + shift I)] z = rhs` for each stacked
/// right-hand side on one factorization. The factor adds `regularization`
/// to both diagonals so that a zero (1,1) block or zero rows of `H` stay
/// quasi-definite; refinement targets the unregularized shifted system.
/// The factorization stays cached for later resolvents to update in place.
pub(crate) fn solve_shifted_systems(
    problem: &Problem,
    h_diagonal: &[f64],
    shift: f64,
    regularization: f64,
    kkt: &KktSettings,
    cache: &mut KktCache,
    right_hand_sides: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, JosephyError> {
    if !shift.is_finite()
        || shift < 0.0
        || !regularization.is_finite()
        || regularization <= 0.0
        || h_diagonal.len() != problem.constraint_count()
    {
        return Err(EmbeddingError::InvalidMu.into());
    }
    let d_diagonal = h_diagonal
        .iter()
        .map(|entry| entry + shift)
        .collect::<Vec<_>>();
    let d_factor = h_diagonal
        .iter()
        .map(|entry| (entry + shift + regularization).max(kkt.min_static_regularization))
        .collect::<Vec<_>>();
    factorize(problem, shift + regularization, &d_factor, kkt, cache)?;
    let KktCache {
        pattern,
        factorization,
        ..
    } = cache;
    let pattern = pattern.as_mut().expect("the pattern was just stored");
    let factorization = factorization
        .as_mut()
        .expect("the factorization was just stored");
    right_hand_sides
        .iter()
        .map(|rhs| {
            ensure_rhs_length(
                "rhs",
                problem.variable_count() + problem.constraint_count(),
                rhs,
            )?;
            Ok(pattern.solve(
                problem,
                factorization,
                shift,
                &d_diagonal,
                &d_factor,
                rhs,
                None,
            )?)
        })
        .collect()
}

/// rho + rho*||x||^2 + y'*(H + rho I)*y, without forming unweighted squares.
fn tau_leading_coefficient(shift: f64, x: &[f64], y: &[f64], d: &[f64]) -> f64 {
    // Multiplying the nonnegative weight first avoids an intermediate square
    // overflowing/underflowing when the weighted square is representable.
    let primal = x
        .iter()
        .fold(shift, |sum, &value| sum + (shift * value) * value);
    y.iter().zip(d).fold(primal, |sum, (&value, &weight)| {
        sum + (weight * value) * value
    })
}

fn q_dot(problem: &Problem, vector: &[f64]) -> f64 {
    let (x, y) = vector.split_at(problem.variable_count());
    dot(problem.c(), x) + dot(problem.b(), y)
}

fn ensure_rhs_length(
    argument: &'static str,
    expected: usize,
    vector: &[f64],
) -> Result<(), JosephyError> {
    if vector.len() != expected {
        return Err(JosephyError::IncorrectRhsLength {
            argument,
            expected,
            actual: vector.len(),
        });
    }
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(JosephyError::InvalidTauPolynomial);
    }
    Ok(())
}

/// Assemble the clamped preconditioner with bound rows folded into the
/// `(1,1)` block when `eliminate_bound_rows`; `d_diagonal` is already clamped,
/// returning the initial matrix and retaining its update metadata in the cache.
/// Subsequent factorization updates use `KktPattern::write_diagonal` directly.
pub(crate) fn assemble_preconditioner(
    problem: &Problem,
    shift: f64,
    d_diagonal: &[f64],
    min_static_regularization: f64,
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
        min_static_regularization,
    );
    cache.pattern = Some(pattern);
    Ok(matrix)
}

impl KktPattern {
    /// Clamp the full primal diagonal before adding singleton Schur terms.
    /// `d_diagonal` contains the already clamped factor dual diagonal.
    fn write_diagonal(
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
    fn solve(
        &mut self,
        problem: &Problem,
        factorization: &mut Factorization,
        shift: f64,
        d_diagonal: &[f64],
        d_factor: &[f64],
        rhs: &[f64],
        initial: Option<(&[f64], &[f64])>,
    ) -> Result<Vec<f64>, LinearSystemError> {
        factorization.stats_mut().kkt_solves += 1;
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
                crate::matrix::add_assign(solution_x, x);
                crate::matrix::add_assign(solution_y, y);
                solution
            }
            None => self.solve_preconditioner(factorization, d_factor, rhs)?,
        };
        let mut stats = LinearAlgebraStats::default();
        let mut workspace = std::mem::take(&mut self.refinement);
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

fn build_pattern(
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
    crate::matrix::visit_entries(problem.a(), |source, _, value| {
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
    crate::matrix::visit_entries(problem.a(), |source, row, value| {
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
fn apply_full_system(
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
        crate::matrix::symmetric_accumulate(p, x, [out_x], |product| [sign(product)]);
    }
    crate::matrix::update(out_x, x, |out, x| out + sign(shift * x));
    crate::matrix::constraint_accumulate(problem.a(), x, y, [out_x], [out_y], |product| {
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
fn full_residual_and_denominator(
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
        crate::matrix::symmetric_accumulate(
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
    crate::matrix::constraint_accumulate(
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

/// CPO3's admissible root; `None` requests the bounded fallback.
fn tau_root(a: f64, b: f64, c: f64) -> Option<f64> {
    let tau = quadratic_root(a, b, c, 1e-9)?;
    (tau >= -1e-10).then(|| tau.max(0.0))
}

/// CPO3's Muller branch for general coefficient signs, without squaring
/// unscaled coefficients. Tolerances remain in the original coefficient units.
fn quadratic_root(a: f64, b: f64, c: f64, negative_discriminant_tolerance: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return None;
    }
    let tau = if a.abs() < 1e-15 {
        if b.abs() < 1e-15 {
            return None;
        }
        -c / b
    } else if c == 0.0 {
        if b > 0.0 { 0.0 } else { -b / a }
    } else {
        let scale = a.abs().max(b.abs()).max(c.abs());
        let sqrt_scale = scale.sqrt();
        let sqrt_a = a.abs().sqrt();
        let sqrt_c = c.abs().sqrt();
        let sqrt_ac = (sqrt_a / sqrt_scale) * (sqrt_c / sqrt_scale);
        let abs_b = (b / scale).abs();
        let root = if a.is_sign_positive() != c.is_sign_positive() {
            // b^2 + 4 |ac|: hypot avoids both overflow and cancellation.
            abs_b.hypot(2.0 * sqrt_ac)
        } else {
            // Factoring the difference preserves its sign even when its
            // square would underflow. Compare sqrt(-D) in original units.
            let difference = abs_b - 2.0 * sqrt_ac;
            let magnitude = difference.abs().sqrt() * (abs_b + 2.0 * sqrt_ac).sqrt();
            if difference < 0.0 {
                if magnitude * scale > negative_discriminant_tolerance.sqrt() {
                    return None;
                }
                0.0
            } else {
                magnitude
            }
        };
        let q = 0.5 * (abs_b + root);
        if b > 0.0 {
            // c / (-q*scale), arranged so tiny c/scale is not erased.
            -c.signum() * (sqrt_c / sqrt_a) * (sqrt_ac / q)
        } else {
            // q*scale / a, without overflowing q*scale or underflowing a/scale.
            (q * sqrt_scale) / (a / sqrt_scale)
        }
    };
    tau.is_finite().then_some(tau)
}

fn matrix_value(matrix: &CscMatrix, row: usize, column: usize) -> f64 {
    matrix.get(row.min(column), row.max(column)).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cone, LinearSolver};

    const TOLERANCE: f64 = 2e-11;

    fn reference_state(problem: &Problem) -> State {
        crate::embedding::Embedding::new(problem)
            .unwrap()
            .canonical_initial_state(1.0)
            .unwrap()
    }

    fn problem() -> Problem {
        Problem::new(
            Some(crate::matrix::test_matrix(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap()),
            vec![0.5, -1.0],
            crate::matrix::test_matrix(2, 2, vec![1.0, -2.0, 3.0, 0.25]).unwrap(),
            vec![1.5, -0.75],
            vec![Cone::Zero(1), Cone::Nonnegative(1)],
        )
        .unwrap()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= TOLERANCE * (1.0 + actual.abs().max(expected.abs())),
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn fused_full_system_residual_preserves_rounding_and_the_normalizer() {
        {
            let problem = problem();
            for u in [[1.0, -2.0, 3.0, -4.0], [1e-12, -1e12, 1e8, 0.0]] {
                let rhs = [0.0, 1.0, -3.0, 7.0];
                let d = [1e-10, 3.5];
                let shift = 1e-12;
                let mut expected_residual = rhs.to_vec();
                let mut expected_denominator = rhs.map(f64::abs).to_vec();
                let initial = u.split_at(problem.variable_count());
                apply_full_system(&problem, shift, &d, initial, &mut expected_residual, false);
                apply_full_system(
                    &problem,
                    shift,
                    &d,
                    initial,
                    &mut expected_denominator,
                    true,
                );
                let mut residual = vec![f64::NAN; 4];
                let mut denominator = vec![f64::NAN; 4];
                full_residual_and_denominator(
                    &problem,
                    shift,
                    &d,
                    &u,
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
    fn assembles_sparse_quasi_definite_systems() {
        let expected = [
            [5.0, 0.5, 1.0, 3.0],
            [0.5, 4.0, -2.0, 0.25],
            [1.0, -2.0, -3.0, 0.0],
            [3.0, 0.25, 0.0, -6.0],
        ];
        {
            let problem = problem();
            let matrix = assemble_preconditioner(
                &problem,
                3.0,
                &[3.0, 6.0],
                0.0,
                true,
                &mut KktCache::new(),
            )
            .unwrap();
            for (row, expected_row) in expected.iter().enumerate() {
                for (column, expected) in expected_row.iter().enumerate() {
                    assert_close(matrix_value(&matrix, row, column), *expected);
                }
            }
        }
    }

    fn linear_program() -> Problem {
        Problem::new(
            None,
            vec![1.0, 2.0],
            crate::matrix::test_matrix(2, 2, vec![1.0, 1.0, -1.0, 2.0]).unwrap(),
            vec![1.0, 0.5],
            vec![Cone::Nonnegative(2)],
        )
        .unwrap()
    }

    fn a_mul(problem: &Problem, x: &[f64]) -> Vec<f64> {
        crate::matrix::matrix_mul(problem.a(), x)
    }

    fn at_mul(problem: &Problem, y: &[f64]) -> Vec<f64> {
        crate::matrix::matrix_transpose_mul(problem.a(), y)
    }

    /// Residual of the generic shifted system at a case (i) solution.
    fn generic_system_residual(
        problem: &Problem,
        h: &[f64],
        shift: f64,
        r: (&[f64], &[f64], f64),
        mu: f64,
        solution: &ResolventSolution,
    ) -> f64 {
        let (r_x, r_y, r_tau) = r;
        let px = problem.p_mul(solution.x());
        let aty = at_mul(problem, solution.y());
        let ax = a_mul(problem, solution.x());
        let mut worst: f64 = 0.0;
        for index in 0..problem.variable_count() {
            let row = px[index]
                + shift * solution.x()[index]
                + aty[index]
                + problem.c()[index] * solution.tau
                - r_x[index];
            worst = worst.max(row.abs());
        }
        for index in 0..problem.constraint_count() {
            let row = -ax[index]
                + (h[index] + shift) * solution.y()[index]
                + problem.b()[index] * solution.tau
                - r_y[index];
            worst = worst.max(row.abs());
        }
        let row = -dot(problem.c(), solution.x()) - dot(problem.b(), solution.y())
            + shift * solution.tau
            - (dot(solution.x(), &px) + mu) / solution.tau
            - r_tau;
        worst.max(row.abs())
    }

    #[test]
    fn resolvent_solves_the_shifted_system_for_every_solver() {
        let h = [0.0, 2.5];
        let shift = 0.7;
        let r_x = [0.3, -1.1];
        let r_y = [0.9, 0.4];
        let r_tau = -0.6;
        let cases = [LinearSolver::Faer, LinearSolver::Feral];
        for solver in cases {
            let problem = problem();
            let state = reference_state(&problem);
            let mut cache = KktCache::new();
            let mut resolvent = Resolvent::new(
                &problem,
                &state,
                &h,
                shift,
                &KktSettings::new(solver),
                &mut cache,
            )
            .unwrap();
            assert!(resolvent.alpha > 0.0);
            let (x2, y2) = resolvent.direction.split_at(problem.variable_count());
            let px2 = problem.p_mul(x2);
            let defining = shift - dot(problem.c(), x2) - dot(problem.b(), y2) - dot(x2, &px2);
            assert_close(resolvent.alpha, defining);
            for mu in [0.0, 0.25] {
                let solution = resolvent.solve(&r_x, &r_y, r_tau, mu).unwrap();
                assert!(solution.tau > 0.0);
                let residual = generic_system_residual(
                    &problem,
                    &h,
                    shift,
                    (&r_x, &r_y, r_tau),
                    mu,
                    &solution,
                );
                assert!(residual <= 1e-12, "{solver:?}: residual {residual}");
            }
        }
    }

    /// An equality, a general inequality, and two bound rows on `x`.
    fn bounded_problem() -> Problem {
        Problem::new(
            Some(crate::matrix::test_matrix(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap()),
            vec![0.5, -1.0],
            crate::matrix::test_matrix(4, 2, vec![1.0, -2.0, 3.0, 0.25, -4.0, 0.0, 0.0, 1.5])
                .unwrap(),
            vec![1.5, -0.75, 2.0, 0.5],
            vec![Cone::Zero(1), Cone::Nonnegative(3)],
        )
        .unwrap()
    }

    // Independently specified full matrices: one retained equality and one
    // singleton inequality, with both primal and dual clamps active.
    #[test]
    fn clamped_preconditioner_is_identical_with_or_without_bound_elimination() {
        let problem = Problem::new(
            Some(crate::matrix::test_matrix(2, 2, vec![0.1, 0.0, 0.0, 2.0]).unwrap()),
            vec![0.0; 2],
            crate::matrix::test_matrix(2, 2, vec![1.0, -1.0, 2.0, 0.0]).unwrap(),
            vec![0.0; 2],
            vec![Cone::Zero(1), Cone::Nonnegative(1)],
        )
        .unwrap();
        let k = [
            [0.2, 0.0, 1.0, 2.0],
            [0.0, 2.1, -1.0, 0.0],
            [1.0, -1.0, -0.1, 0.0],
            [2.0, 0.0, 0.0, -0.3],
        ];
        let m = [
            [0.5, 0.0, 1.0, 2.0],
            [0.0, 2.1, -1.0, 0.0],
            [1.0, -1.0, -0.5, 0.0],
            [2.0, 0.0, 0.0, -0.5],
        ];
        let expected = [0.4, -0.7, 1.1, -0.3];
        let multiply = |matrix: &[[f64; 4]; 4]| {
            matrix
                .iter()
                .map(|row| dot(row, &expected))
                .collect::<Vec<_>>()
        };
        let rhs = multiply(&m);
        for solver in [LinearSolver::Faer, LinearSolver::Feral] {
            for eliminate in [false, true] {
                let mut cache = KktCache::new();
                let matrix =
                    assemble_preconditioner(&problem, 0.1, &[0.5, 0.5], 0.5, eliminate, &mut cache)
                        .unwrap();
                assert_close(
                    matrix_value(&matrix, 0, 0),
                    if eliminate { 8.5 } else { 0.5 },
                );
                let mut settings = KktSettings::new(solver);
                // Request accuracy tighter than the solution assertions below;
                // the production backward-error tolerance is only 1e-8.
                settings.refinement_tolerance = 1e-12;
                let mut factor =
                    Factorization::new_counted(matrix, &settings, &mut cache.factorizations)
                        .unwrap();
                let pattern = cache.pattern.as_mut().unwrap();
                let solution = pattern
                    .solve_preconditioner(&mut factor, &[0.5, 0.5], &rhs)
                    .unwrap();
                for (actual, expected) in solution.iter().zip(expected) {
                    assert_close(*actual, expected);
                }
                let mut residual = [0.0; 4];
                let mut denominator = [0.0; 4];
                full_residual_and_denominator(
                    &problem,
                    0.1,
                    &[0.1, 0.3],
                    &solution,
                    &rhs,
                    &mut residual,
                    &mut denominator,
                );
                for i in 0..4 {
                    assert_close(residual[i], rhs[i] - dot(&k[i], &solution));
                    let normalizer = rhs[i].abs()
                        + k[i]
                            .iter()
                            .zip(&solution)
                            .map(|(a, x)| (a * x).abs())
                            .sum::<f64>();
                    assert_close(denominator[i], normalizer);
                }
                let true_rhs = multiply(&k);
                let refined = pattern
                    .solve(
                        &problem,
                        &mut factor,
                        0.1,
                        &[0.1, 0.3],
                        &[0.5, 0.5],
                        &true_rhs,
                        None,
                    )
                    .unwrap();
                for (actual, expected) in refined.iter().zip(expected) {
                    assert!(
                        (actual - expected).abs() < 1e-8,
                        "{solver:?}/{eliminate}: {actual} vs {expected}"
                    );
                }
                // A second diagonal assembly must clamp afresh, not add
                // another Schur contribution to the previous diagonal.
                factor
                    .update_diagonal_and_refactor(&mut cache.factorizations, |values| {
                        pattern.write_diagonal(values, |index| index, 2, 0.1, &[0.5, 0.5], 0.5)
                    })
                    .unwrap();
                let again = pattern
                    .solve_preconditioner(&mut factor, &[0.5, 0.5], &rhs)
                    .unwrap();
                for (actual, expected) in again.iter().zip(expected) {
                    assert_close(*actual, expected);
                }
                assert_eq!(cache.factorizations, 2);
            }
        }
    }

    #[test]
    fn best_effort_refines_true_system_even_without_eliminated_rows() {
        let problem = Problem::new(
            None,
            vec![0.0],
            crate::matrix::test_matrix(0, 1, vec![]).unwrap(),
            vec![],
            vec![],
        )
        .unwrap();
        let mut cache = KktCache::new();
        let matrix =
            assemble_preconditioner(&problem, 1e-14, &[], 1e-8, false, &mut cache).unwrap();
        let mut factor = Factorization::new(matrix, &KktSettings::new(LinearSolver::Faer)).unwrap();
        let solution = cache
            .pattern
            .as_mut()
            .unwrap()
            .solve(&problem, &mut factor, 1e-14, &[], &[], &[1.0], None)
            .unwrap();
        // The preconditioner alone solves to 1e8. True-system refinement must
        // attempt corrections, but cannot resolve this weak direction in 30.
        assert!(solution[0] > 1e8 && solution[0] < 1e14);
        assert!(factor.stats().refinement_passes > 0);
        assert_eq!(factor.stats().refinement_settled, 1);
        assert!((1.0 - 1e-14 * solution[0]).abs() > 0.99);
    }

    #[test]
    fn refinement_warm_start_preserves_weak_components_and_corrects_the_rest() {
        // The first variable is uncoupled: its true pivot is 1e-14, while
        // the factor pivot is 1e-8. The second variable has a singleton row.
        let problem = Problem::new(
            None,
            vec![0.0; 2],
            crate::matrix::test_matrix(1, 2, vec![0.0, 1.0]).unwrap(),
            vec![0.0],
            vec![Cone::Nonnegative(1)],
        )
        .unwrap();
        let shift = 1e-14;
        let d = [2.0];
        let rhs = [1.0, 2.0 * shift - 3.0, 8.0];
        let initial_x = [1e14, -1.0];
        let initial_y = [0.5];
        for solver in [LinearSolver::Faer, LinearSolver::Feral] {
            for eliminate in [false, true] {
                let mut cache = KktCache::new();
                let mut settings = KktSettings::new(solver);
                settings.min_static_regularization = 1e-8;
                // Resolve the corrected components to the asserted accuracy,
                // rather than stopping at the production tolerance of 1e-8.
                settings.refinement_tolerance = 1e-12;
                let matrix = assemble_preconditioner(
                    &problem,
                    shift,
                    &d,
                    settings.min_static_regularization,
                    eliminate,
                    &mut cache,
                )
                .unwrap();
                let mut factor = Factorization::new(matrix, &settings).unwrap();
                let pattern = cache.pattern.as_mut().unwrap();
                let fresh = pattern
                    .solve(&problem, &mut factor, shift, &d, &d, &rhs, None)
                    .unwrap();
                assert!((rhs[0] - shift * fresh[0]).abs() > 0.99);
                let solved = pattern
                    .solve(
                        &problem,
                        &mut factor,
                        shift,
                        &d,
                        &d,
                        &rhs,
                        Some((&initial_x, &initial_y)),
                    )
                    .unwrap();
                for (actual, expected) in solved.iter().zip([1e14, 2.0, -3.0]) {
                    assert_close(*actual, expected);
                }
                let mut residual = rhs;
                apply_full_system(
                    &problem,
                    shift,
                    &d,
                    solved.split_at(2),
                    &mut residual,
                    false,
                );
                assert!(residual.iter().all(|r| r.abs() < 1e-10));
            }
        }
    }

    #[test]
    fn failed_step_restores_the_previous_constant_column_without_copying() {
        let problem = Problem::new(
            None,
            vec![-1.0],
            crate::matrix::test_matrix(0, 1, vec![]).unwrap(),
            vec![],
            vec![],
        )
        .unwrap();
        let state = reference_state(&problem);
        let settings = KktSettings::new(LinearSolver::Feral);
        let mut cache = KktCache::new();
        Resolvent::new(&problem, &state, &[], 1.0, &settings, &mut cache).unwrap();
        cache.finish_step(true);
        let original = cache.direction.as_ptr();
        assert_close(cache.direction[0], 1.0);
        Resolvent::new(&problem, &state, &[], 2.0, &settings, &mut cache).unwrap();
        assert_close(cache.direction[0], 0.5);
        cache.finish_step(false);
        assert_eq!(cache.direction.as_ptr(), original);
        assert_close(cache.direction[0], 1.0);
        // Failure before solving the column must not discard the accepted seed.
        assert!(Resolvent::new(&problem, &state, &[], 0.0, &settings, &mut cache).is_err());
        cache.finish_step(false);
        assert_eq!(cache.direction.as_ptr(), original);
        Resolvent::new(&problem, &state, &[], 2.0, &settings, &mut cache).unwrap();
        cache.finish_step(true);
        assert_close(cache.direction[0], 0.5);
        assert!(cache.direction_before_step.is_none());
    }

    #[test]
    fn constant_column_warm_start_survives_refactorization_only_within_its_cache() {
        let problem = Problem::new(
            None,
            vec![-1e-14],
            crate::matrix::test_matrix(0, 1, vec![]).unwrap(),
            vec![],
            vec![],
        )
        .unwrap();
        let state = reference_state(&problem);
        let mut settings = KktSettings::new(LinearSolver::Feral);
        let mut cache = KktCache::new();
        // First compute an accurate column with an unclamped factorization.
        settings.min_static_regularization = 0.0;
        let first = Resolvent::new(&problem, &state, &[], 1e-14, &settings, &mut cache)
            .unwrap()
            .direction[0];
        assert_close(first, 1.0);
        // Clamping the next factor cannot erase the retained solution.
        settings.min_static_regularization = 1e-8;
        let next = Resolvent::new(&problem, &state, &[], 1e-14, &settings, &mut cache)
            .unwrap()
            .direction[0];
        assert_close(next, 1.0);
        assert_eq!(cache.stats().factorizations, 2);
        // A separate problem solve has no retained column.
        let fresh = Resolvent::new(
            &problem,
            &state,
            &[],
            1e-14,
            &settings,
            &mut KktCache::new(),
        )
        .unwrap()
        .direction[0];
        assert!(fresh < 1e-4);
    }

    #[test]
    fn bound_rows_fold_into_the_reduced_diagonal() {
        let d = [3.0, 6.0, 0.5, 4.0];
        // Rows 2 and 3 are singletons: 16 / 0.5 on x_0 and 2.25 / 4 on x_1.
        let expected = [
            [5.0 + 32.0, 0.5, 1.0, 3.0],
            [0.5, 4.0 + 0.5625, -2.0, 0.25],
            [1.0, -2.0, -3.0, 0.0],
            [3.0, 0.25, 0.0, -6.0],
        ];
        {
            let problem = bounded_problem();
            let mut cache = KktCache::new();
            let matrix = assemble_preconditioner(&problem, 3.0, &d, 0.0, true, &mut cache).unwrap();
            assert_eq!(matrix.rows(), 4);
            for (row, expected_row) in expected.iter().enumerate() {
                for (column, expected) in expected_row.iter().enumerate() {
                    assert_close(matrix_value(&matrix, row, column), *expected);
                }
            }
            let bound_rows = &cache.pattern.as_ref().unwrap().bound_rows;
            assert_eq!(bound_rows.eliminated, vec![(2, 0, -4.0), (3, 1, 1.5)]);
            assert_eq!(bound_rows.kept, vec![0, 1]);
            let full = assemble_preconditioner(&problem, 3.0, &d, 0.0, false, &mut KktCache::new())
                .unwrap();
            assert_eq!(full.rows(), 6);
        }
    }

    #[test]
    fn bound_row_elimination_leaves_the_resolvent_solution_unchanged() {
        let h = [0.0, 2.5, 0.8, 1e-6];
        let shift = 0.7;
        let r_x = [0.3, -1.1];
        let r_y = [0.9, 0.4, -0.2, 0.7];
        let r_tau = -0.6;
        let cases = [LinearSolver::Faer, LinearSolver::Feral];
        for solver in cases {
            let problem = bounded_problem();
            let state = reference_state(&problem);
            let mut solutions = Vec::new();
            for eliminate in [false, true] {
                let mut cache = KktCache::new();
                let kkt = KktSettings {
                    eliminate_bound_rows: eliminate,
                    ..KktSettings::new(solver)
                };
                let mut resolvent =
                    Resolvent::new(&problem, &state, &h, shift, &kkt, &mut cache).unwrap();
                assert_eq!(resolvent.factorization.dim(), if eliminate { 4 } else { 6 });
                let solution = resolvent.solve(&r_x, &r_y, r_tau, 0.25).unwrap();
                let residual = generic_system_residual(
                    &problem,
                    &h,
                    shift,
                    (&r_x, &r_y, r_tau),
                    0.25,
                    &solution,
                );
                assert!(
                    residual <= 1e-9,
                    "{solver:?}/{eliminate}: residual {residual}"
                );
                solutions.push(solution);
            }
            for (full, reduced) in solutions[0].x().iter().zip(solutions[1].x()) {
                assert_close(*full, *reduced);
            }
            for (full, reduced) in solutions[0].y().iter().zip(solutions[1].y()) {
                assert_close(*full, *reduced);
            }
            assert_close(solutions[0].tau, solutions[1].tau);
        }
    }

    #[test]
    fn resolvent_accepts_zero_tau_without_fallback() {
        let problem = linear_program();
        let state = reference_state(&problem);
        let h = [1.5, 0.5];
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            &problem,
            &state,
            &h,
            0.4,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();

        let boundary = resolvent.solve(&[0.0; 2], &[0.0; 2], -1.0, 0.0).unwrap();
        assert_eq!(boundary.tau, 0.0);
        assert_eq!(boundary.x(), [0.0; 2]);
        assert_eq!(boundary.y(), [0.0; 2]);
        assert!(resolvent.fallback.get().is_none());

        let interior = resolvent.solve(&[0.0; 2], &[0.0; 2], -1.0, 0.1).unwrap();
        assert!(interior.tau > 0.0);
        let residual = generic_system_residual(
            &problem,
            &h,
            0.4,
            (&[0.0; 2], &[0.0; 2], -1.0),
            0.1,
            &interior,
        );
        assert!(residual <= 1e-12, "residual {residual}");

        let positive_root = resolvent.solve(&[0.0; 2], &[0.0; 2], 1.0, 0.0).unwrap();
        assert!(positive_root.tau > 0.0);
    }

    #[test]
    fn resolvent_rejects_bad_inputs() {
        let problem = problem();
        let state = reference_state(&problem);
        assert!(matches!(
            Resolvent::new(
                &problem,
                &state,
                &[0.0, 1.0],
                0.0,
                &KktSettings::new(LinearSolver::Faer),
                &mut KktCache::new()
            ),
            Err(JosephyError::Embedding(EmbeddingError::InvalidMu))
        ));
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            &problem,
            &state,
            &[0.0, 1.0],
            1.0,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        assert_eq!(
            resolvent.solve(&[0.0], &[0.0; 2], 0.0, 0.0),
            Err(JosephyError::IncorrectRhsLength {
                argument: "r_x",
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            resolvent.solve(&[0.0; 2], &[0.0; 2], 0.0, -1.0),
            Err(JosephyError::InvalidTauPolynomial)
        );
    }

    #[test]
    fn assembly_rejects_a_nonpositive_diagonal() {
        let problem = problem();
        assert_eq!(
            assemble_preconditioner(&problem, 1.0, &[1.0, 0.0], 0.0, true, &mut KktCache::new()),
            Err(JosephyError::InvalidTauPolynomial)
        );
    }

    #[test]
    fn positive_tau_coefficient_handles_extreme_weights_before_squaring() {
        let large_x = tau_leading_coefficient(1e-300, &[1e200], &[], &[]);
        assert!((large_x / 1e100 - 1.0).abs() < 1e-14);
        let small_y = tau_leading_coefficient(1e-300, &[], &[1e-200], &[1e300]);
        assert!((small_y / 1e-100 - 1.0).abs() < 1e-14);
        let overflowing_norm = tau_leading_coefficient(1e-308, &[1e154, 1e154], &[], &[]);
        assert!((overflowing_norm - 2.0).abs() < 1e-14);
        assert_eq!(tau_leading_coefficient(0.25, &[0.0], &[0.0], &[1.0]), 0.25);
    }

    #[test]
    fn tau_root_accepts_zero_and_roundoff_but_rejects_invalid_roots() {
        assert_eq!(tau_root(2.0, 1.0, 0.0), Some(0.0));
        assert_eq!(tau_root(2.0, -2.0, 0.0), Some(1.0));
        assert_eq!(tau_root(0.0, 2.0, -4.0), Some(2.0));
        assert_eq!(tau_root(0.0, 1.0, 5e-11), Some(0.0));
        assert_eq!(tau_root(0.0, 1.0, 2e-10), None);
        assert_eq!(tau_root(0.0, 0.0, -1.0), None);
        assert_eq!(tau_root(1.0, 0.0, 1e-11), Some(0.0));
        assert_eq!(tau_root(1.0, 0.0, 1.0), None);
        assert!((tau_root(1.0, -1e308, -1.0).unwrap() / 1e308 - 1.0).abs() < 1e-14);
        let small = tau_root(1.0, 1e16, -1.0).unwrap();
        assert!((small - 1e-16).abs() <= 2.0 * f64::EPSILON * 1e-16);
    }

    #[test]
    fn scaled_roots_handle_extreme_coefficients_and_general_signs() {
        for (a, b, c, expected) in [
            (1e308, 0.0, -1e308, 1.0),
            (1e308, 1e-300, -1e-308, 1e-308),
            (1e-14, 0.0, -1e308, 1e161),
            (1.0, 1e308, -1.0, 1e-308),
            (-2.0, 1.0, 1.0, -0.5),
            (-1.0, 3.0, -2.0, 1.0),
        ] {
            let root = quadratic_root(a, b, c, 0.0).unwrap();
            assert!(
                (root / expected - 1.0).abs() < 1e-13,
                "{a}, {b}, {c}: {root}"
            );
        }
        assert_eq!(tau_root(-2.0, 1.0, 1.0), None);
        assert_eq!(quadratic_root(1e308, 0.0, 1e-308, 1e-9), None);
        assert_eq!(quadratic_root(1.0, 0.0, 1e-11, 0.0), None);
        assert_eq!(tau_root(f64::NAN, 1.0, -1.0), None);
        assert_eq!(tau_root(1.0, f64::INFINITY, -1.0), None);
        assert_eq!(tau_root(1.0, 1.0, f64::NEG_INFINITY), None);
    }

    #[test]
    fn root_thresholds_are_in_original_units() {
        // Normalizing first would incorrectly discard this nonzero a.
        assert_close(tau_root(1.0, 0.0, -1e20).unwrap() / 1e10, 1.0);
        // This a is deliberately treated as zero despite a large root.
        assert_eq!(tau_root(1e-16, 1.0, -1e16), Some(1e16));
        assert_eq!(tau_root(1e-16, 1e-16, -1.0), None);
        // Same scaled polynomial, different absolute discriminant acceptance.
        assert_eq!(tau_root(1.0, 0.0, 1e-11), Some(0.0));
        assert_eq!(tau_root(1e6, 0.0, 1e-5), None);
    }

    #[test]
    fn failed_root_uses_current_state_and_reuses_fallback_products() {
        // P = 1, c = 1, rho = 1 gives direction = -1/2. Linearizing
        // at x = 1 gives 1.5*tau^2 + (1-r_tau-x1)*tau + 1-mu-2*x1 = 0.
        // If it has no real root, the first-order step uses current tau = 2.
        let problem = Problem::new(
            Some(crate::matrix::test_matrix(1, 1, vec![1.0]).unwrap()),
            vec![1.0],
            crate::matrix::test_matrix(0, 1, vec![]).unwrap(),
            vec![],
            vec![],
        )
        .unwrap();
        let state = State::new(vec![1.0], vec![], vec![], 2.0).unwrap();
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            &problem,
            &state,
            &[],
            1.0,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        resolvent.solve(&[1.0], &[], 0.0, 1.0).unwrap();
        assert!(resolvent.fallback.get().is_none());

        // Force invalid quadratic coefficients to exercise the full solve's
        // recovery path, not just the standalone fallback helper.
        resolvent.alpha = f64::NAN;
        let solution = resolvent.solve(&[1.0], &[], 0.0, 1.0).unwrap();
        assert!(solution.used_tau_fallback);
        assert_close(solution.tau, 2.0 / 3.0);
        assert_close(solution.x()[0], 1.0 / 6.0);
        let px_pointer = resolvent
            .fallback
            .get()
            .unwrap()
            .px
            .as_ref()
            .unwrap()
            .as_ptr();
        let changed_rhs = resolvent.solve(&[2.0], &[], 1.0, 0.0).unwrap();
        assert_close(changed_rhs.tau, (1.0 + 7.0_f64.sqrt()) / 3.0);
        assert_eq!(
            resolvent
                .fallback
                .get()
                .unwrap()
                .px
                .as_ref()
                .unwrap()
                .as_ptr(),
            px_pointer
        );

        let lower = resolvent.solve(&[-10.0], &[], 0.0, 0.0).unwrap();
        assert_close(lower.tau, 0.2);
        // Negative discriminant: use the first-order correction, before clamping.
        let first_order = resolvent.solve(&[0.2], &[], 0.0, 0.0).unwrap();
        assert_close(first_order.tau, 0.28);
        let zero_denominator = resolvent.solve(&[-10.0], &[], 5.0, 0.0).unwrap();
        assert_eq!(zero_denominator.tau, state.tau());
        let upper = resolvent.solve(&[100.0], &[], 0.0, 0.0).unwrap();
        assert_close(upper.tau, 20.0);
        let singular = resolvent.solve(&[1.0], &[], 5.0, 0.0).unwrap();
        assert_close(singular.tau, 3.0);
        let nonfinite = resolvent.solve(&[1.0], &[], f64::MAX, 0.0).unwrap();
        assert_close(nonfinite.tau, 20.0);

        // Nonfinite fallback coefficients and correction preserve current tau.
        resolvent.fallback.get_mut().unwrap().q_direction = f64::NAN;
        let retained = resolvent.solve(&[1.0], &[], 0.0, 0.0).unwrap();
        assert_eq!(retained.tau, state.tau());
    }
}
