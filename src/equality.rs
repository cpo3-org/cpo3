//! Equality-only solver: dynamically regularized LDL and refinement against K,
//! then right-preconditioned LSMR if still unverified.
//! A least-squares residual lies in ker(K). Its primal block is a dual
//! infeasibility candidate; minus its equality block is a primal certificate.
use crate::{
    Problem, Settings, SolutionInfo, SolveError, SolverStatus,
    linear_algebra::{
        KktOperator, KktPreconditioner, assemble_unshifted_kkt,
        lsmr::{LsmrStop, lsmr_scratch, lsmr_with_control},
    },
    linear_solvers::LinearAlgebraStats,
    recovery::{CertificateWorkspace, Recovery, Termination},
    ruiz::EquilibratedProblem,
    solve::time_limit_reached,
};
use faer::{
    MatMut, MatRef,
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::{LinOp, Precond},
};
use std::{ops::ControlFlow, time::Instant};

/// Solve the equality KKT system, accepting only verified points or certificates.
pub(crate) fn solve(
    problem: &Problem,
    equilibrated: &EquilibratedProblem<'_>,
    settings: &Settings,
    recovery: &mut Recovery<'_>,
    started: Instant,
    info: &mut SolutionInfo,
) -> Result<Termination, SolveError> {
    let scaled = equilibrated.problem();
    let (m, n) = problem.dims();
    let zero_slack = vec![0.0; m];
    let rhs = scaled
        .c()
        .iter()
        .map(|c| -c)
        .chain(scaled.b().iter().copied())
        .collect::<Vec<_>>();

    let mut check_optimal_or_timeout = |point: &[f64]| {
        let (x, y) = point.split_at(n);
        let recovered = recovery.update_coordinates(x, y, &zero_slack, 1.0, 0.0);
        if recovered.is_solved(settings) {
            return Some(Termination::Point(SolverStatus::Solved));
        }
        time_limit_reached(started, settings.time_limit)
            .then_some(Termination::Point(SolverStatus::TimeLimit))
    };

    let mut workspace = EqualityWorkspace {
        point: vec![0.0; m + n],
        residual: Vec::new(),
    };
    // Check time limit before attempting factorization; use this opportunity
    // to check if 0 is an optimal point.
    if let Some(termination) = check_optimal_or_timeout(&workspace.point) {
        return Ok(termination);
    }

    let (operator, preconditioner) = prepare_kkt(scaled, settings.threads, info)?;
    let stats = &mut info.linear_algebra;
    // Try the direct solve with iterative refinement before falling back to LSMR.
    if let Some(termination) = solve_and_refine(
        &operator,
        &rhs,
        &preconditioner,
        stats,
        &mut workspace,
        settings.eq_refine_iters,
        &mut check_optimal_or_timeout,
    )? {
        return Ok(termination);
    }
    // Reuse the factors, but restart from zero to discard a growing nullspace
    // component left by refinement on an inconsistent system.
    let mut certs = CertificateWorkspace::default();
    solve_lsmr(
        &operator,
        &rhs,
        &preconditioner,
        stats,
        &mut workspace,
        settings.eq_lsmr_iters,
        |point, residual| {
            if let Some(cert) =
                certs.certificate_from_residual(problem, equilibrated, settings, residual)
            {
                return Some(Termination::Certificate(cert));
            }
            check_optimal_or_timeout(point)
        },
    )
}

/// Assemble and factor the KKT system, retaining only the operator and factors.
fn prepare_kkt<'a>(
    problem: &'a Problem,
    threads: usize,
    info: &mut SolutionInfo,
) -> Result<(KktOperator<'a>, KktPreconditioner), SolveError> {
    let matrix =
        assemble_unshifted_kkt(problem).map_err(|e| SolveError::Numerical(e.to_string()))?;
    info.linear_algebra.factorizations += 1;
    info.iters = 1;
    let preconditioner = KktPreconditioner::new(&matrix, problem.variable_count(), threads)
        .map_err(|e| SolveError::Numerical(e.to_string()))?;
    Ok((KktOperator::new(problem), preconditioner))
}

/// Point and residual storage shared by direct refinement and LSMR.
struct EqualityWorkspace {
    point: Vec<f64>,
    residual: Vec<f64>,
}

/// Try the direct solve and refinement, returning None for LSMR fallback.
fn solve_and_refine(
    operator: &KktOperator<'_>,
    rhs: &[f64],
    preconditioner: &KktPreconditioner,
    stats: &mut LinearAlgebraStats,
    workspace: &mut EqualityWorkspace,
    refine_iters: usize,
    mut check: impl FnMut(&[f64]) -> Option<Termination>,
) -> Result<Option<Termination>, SolveError> {
    let size = rhs.len();
    let par = preconditioner.parallelism();
    let mut buffer =
        MemBuffer::try_new(preconditioner.apply_in_place_scratch(1, par)).map_err(|_| {
            SolveError::Numerical("equality refinement workspace allocation failed".into())
        })?;
    let EqualityWorkspace { point, residual } = workspace;
    point.copy_from_slice(rhs);
    preconditioner.solve_in_place(point, MemStack::new(&mut buffer));
    stats.primary_solves += 1;
    // Refinement checks original-problem optimality rather than refine_tol.
    for correction in 0..=refine_iters {
        if !point.iter().all(|v| v.is_finite()) {
            break;
        }
        if let Some(termination) = check(point) {
            return Ok(Some(termination));
        }
        if correction == refine_iters {
            break;
        }
        residual.resize(size, 0.0);
        operator.residual(rhs, point, residual);
        preconditioner.solve_in_place(residual, MemStack::new(&mut buffer));
        stats.refinement_solves += 1;
        crate::linear_algebra::add_assign(point, residual);
    }
    Ok(None)
}

/// No damping or left preconditioning: preserve the original least-squares
/// residual and its nullspace certificate interpretation.
fn solve_lsmr(
    operator: &KktOperator<'_>,
    rhs: &[f64],
    preconditioner: &KktPreconditioner,
    stats: &mut LinearAlgebraStats,
    workspace: &mut EqualityWorkspace,
    max_iters: Option<usize>,
    mut check: impl FnMut(&[f64], &[f64]) -> Option<Termination>,
) -> Result<Termination, SolveError> {
    let size = operator.nrows();
    let EqualityWorkspace { point, residual } = workspace;
    let par = preconditioner.parallelism();
    let mut buffer = MemBuffer::try_new(lsmr_scratch(preconditioner, operator, 1, par))
        .map_err(|_| SolveError::Numerical("LSMR workspace allocation failed".into()))?;
    residual.resize(size, 0.0);
    let applications_before = preconditioner.applications();
    let result = lsmr_with_control(
        MatMut::from_column_major_slice_mut(point, size, 1),
        preconditioner,
        operator,
        MatRef::from_column_major_slice(rhs, size, 1),
        max_iters.unwrap_or_else(|| size.saturating_mul(4).max(32)),
        |x| {
            let point = x.col(0).try_as_col_major().unwrap().as_slice();
            if !point.iter().all(|v| v.is_finite()) {
                return ControlFlow::Break(Err(SolveError::Numerical(
                    "LSMR returned a nonfinite iterate".into(),
                )));
            }
            operator.residual(rhs, point, residual);
            match check(point, residual) {
                None => ControlFlow::Continue(()),
                Some(termination) => ControlFlow::Break(Ok(termination)),
            }
        },
        par,
        MemStack::new(&mut buffer),
    );
    stats.lsmr_iterations += result.iter_count;
    stats.primary_solves += preconditioner.applications() - applications_before;
    match result.stop {
        LsmrStop::Callback(termination) => termination,
        stop => Err(SolveError::Numerical(format!(
            "equality LSMR stopped without a verified solution or certificate after {} iterations ({stop:?})",
            result.iter_count,
        ))),
    }
}
