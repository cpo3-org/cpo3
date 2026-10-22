//! Equality-only solver: dynamically regularized LDL, at most ten refinement
//! corrections against K, then right-preconditioned LSMR if still unverified.
//! A least-squares residual lies in ker(K). Its primal block is a dual
//! infeasibility candidate; minus its equality block is a primal certificate.
use crate::CscMatrix;
use crate::central_path::kkt::{apply_full_system, assemble_equality_kkt};
use crate::{
    Problem, Settings, SolutionInfo, SolveError, SolverStatus,
    central_path::state::State,
    cone::ConeGeometry,
    linear_solvers::LinearAlgebraStats,
    recovery::{Ray, Recovery, Termination},
    ruiz::EquilibratedProblem,
    solve::time_limit_reached,
};
use faer::{
    Conj, Mat, MatMut, MatRef, Par, Side,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    matrix_free::{
        BiLinOp, BiPrecond, InitialGuessStatus, LinOp, Precond,
        lsmr::{LsmrParams, lsmr, lsmr_scratch},
    },
};
use faer::{
    linalg::cholesky::ldlt::factor::LdltRegularization, sparse::linalg::cholesky as sparse_cholesky,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// A fixed inverse of Faer's dynamically regularized LDL factorization.
/// Both the forward and transpose preconditioner use the same symmetric solve.
#[derive(Debug)]
struct DynamicLdlt {
    symbolic: sparse_cholesky::SymbolicCholesky<usize>,
    values: Vec<f64>,
    applications: AtomicUsize,
    parallelism: Par,
}

impl DynamicLdlt {
    fn new(matrix: &CscMatrix, primal_size: usize, threads: usize) -> Result<Self, SolveError> {
        let symbolic = sparse_cholesky::factorize_symbolic_cholesky(
            matrix.as_faer().symbolic(),
            Side::Upper,
            sparse_cholesky::SymmetricOrdering::Amd,
            Default::default(),
        )
        .map_err(|e| SolveError::Numerical(format!("equality LDL symbolic: {e:?}")))?;
        let parallelism = match symbolic.raw() {
            sparse_cholesky::SymbolicCholeskyRaw::Simplicial(_) => Par::Seq,
            sparse_cholesky::SymbolicCholeskyRaw::Supernodal(_) if threads == 1 => Par::Seq,
            sparse_cholesky::SymbolicCholeskyRaw::Supernodal(_) => Par::rayon(threads),
        };
        let mut values = vec![0.0; symbolic.len_val()];
        let signs: Vec<i8> = (0..matrix.rows())
            .map(|i| if i < primal_size { 1 } else { -1 })
            .collect();
        let mut buffer = MemBuffer::try_new(
            symbolic.factorize_numeric_ldlt_scratch::<f64>(parallelism, Default::default()),
        )
        .map_err(|_| SolveError::Numerical("equality LDL workspace allocation failed".into()))?;
        symbolic
            .factorize_numeric_ldlt(
                &mut values,
                matrix.as_faer(),
                Side::Upper,
                LdltRegularization {
                    dynamic_regularization_signs: Some(&signs),
                    dynamic_regularization_epsilon: 1e-12,
                    dynamic_regularization_delta: 1e-8,
                },
                parallelism,
                MemStack::new(&mut buffer),
                Default::default(),
            )
            .map_err(|e| SolveError::Numerical(format!("equality LDL numeric: {e:?}")))?;
        Ok(Self {
            symbolic,
            values,
            applications: AtomicUsize::new(0),
            parallelism,
        })
    }
}
impl LinOp<f64> for DynamicLdlt {
    fn nrows(&self) -> usize {
        self.symbolic.nrows()
    }
    fn ncols(&self) -> usize {
        self.nrows()
    }
    fn apply_scratch(&self, columns: usize, par: Par) -> StackReq {
        self.symbolic.solve_in_place_scratch::<f64>(columns, par)
    }
    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        out.copy_from(rhs);
        self.apply_in_place(out, par, stack);
    }
    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}
impl BiLinOp<f64> for DynamicLdlt {
    fn transpose_apply_scratch(&self, columns: usize, par: Par) -> StackReq {
        self.apply_scratch(columns, par)
    }
    fn transpose_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
    fn adjoint_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}
impl Precond<f64> for DynamicLdlt {
    fn apply_in_place_scratch(&self, columns: usize, par: Par) -> StackReq {
        self.apply_scratch(columns, par)
    }
    fn apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        self.applications.fetch_add(rhs.ncols(), Ordering::Relaxed);
        sparse_cholesky::LdltRef::new(&self.symbolic, &self.values).solve_in_place_with_conj(
            Conj::No,
            rhs,
            par,
            stack,
        );
    }
    fn conj_apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        self.apply_in_place(rhs, par, stack);
    }
}
impl BiPrecond<f64> for DynamicLdlt {
    fn transpose_apply_in_place_scratch(&self, columns: usize, par: Par) -> StackReq {
        self.apply_scratch(columns, par)
    }
    fn transpose_apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        self.apply_in_place(rhs, par, stack);
    }
    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        self.apply_in_place(rhs, par, stack);
    }
}

/// Matrix-free symmetric KKT products use the same sparse kernels as refinement.
#[derive(Debug)]
struct EqualityKkt<'a> {
    problem: &'a Problem,
    zero_diagonal: &'a [f64],
}
impl LinOp<f64> for EqualityKkt<'_> {
    fn nrows(&self) -> usize {
        self.problem.variable_count() + self.problem.constraint_count()
    }
    fn ncols(&self) -> usize {
        self.nrows()
    }
    fn apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::new::<f64>(2 * self.nrows())
    }
    fn apply(&self, mut out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, _: Par, stack: &mut MemStack) {
        let size = self.nrows();
        let (mut input, stack) = stack.make_with::<f64>(size, |_| 0.0);
        let (mut product, _) = stack.make_with::<f64>(size, |_| 0.0);
        for column in 0..rhs.ncols() {
            for i in 0..size {
                input[i] = rhs[(i, column)];
            }
            product.fill(0.0);
            apply_full_system(
                self.problem,
                0.0,
                self.zero_diagonal,
                input.split_at(self.problem.variable_count()),
                &mut product,
            );
            for i in 0..size {
                out[(i, column)] = -product[i];
            }
        }
    }
    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}
impl BiLinOp<f64> for EqualityKkt<'_> {
    fn transpose_apply_scratch(&self, columns: usize, par: Par) -> StackReq {
        self.apply_scratch(columns, par)
    }
    fn transpose_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
    fn adjoint_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

/// No damping or left preconditioning: preserve the original least-squares
/// residual and its nullspace certificate interpretation.
fn least_squares(
    operator: &EqualityKkt<'_>,
    rhs: &[f64],
    preconditioner: &DynamicLdlt,
    stats: &mut LinearAlgebraStats,
) -> Result<(Vec<f64>, Vec<f64>), SolveError> {
    let size = operator.nrows();
    let mut out = Mat::zeros(size, 1);
    let rhs_mat = Mat::from_fn(size, 1, |i, _| rhs[i]);
    let params = LsmrParams {
        initial_guess: InitialGuessStatus::Zero,
        rel_tolerance: 1e-14,
        abs_tolerance: 0.0,
        max_iters: size.saturating_mul(4).max(32),
        ..Default::default()
    };
    let par = preconditioner.parallelism;
    let scratch = lsmr_scratch(preconditioner, operator, 1, par);
    let mut buffer = MemBuffer::try_new(scratch)
        .map_err(|_| SolveError::Numerical("LSMR workspace allocation failed".into()))?;
    let mut iterations = 0;
    // Even NoConvergence can leave a verifiable solution or certificate.
    let _result = lsmr(
        out.as_mut(),
        preconditioner,
        operator,
        rhs_mat.as_ref(),
        params,
        |_| iterations += 1,
        par,
        MemStack::new(&mut buffer),
    );
    stats.lsmr_iterations += iterations;
    stats.primary_solves =
        preconditioner.applications.load(Ordering::Relaxed) - stats.refinement_solves;
    let point: Vec<_> = (0..size).map(|i| out[(i, 0)]).collect();
    if !point.iter().all(|v| v.is_finite()) {
        return Err(SolveError::Numerical(
            "LSMR returned a nonfinite iterate".into(),
        ));
    }
    let mut residual = rhs.to_vec();
    apply_full_system(
        operator.problem,
        0.0,
        operator.zero_diagonal,
        point.split_at(operator.problem.variable_count()),
        &mut residual,
    );
    Ok((point, residual))
}

/// Solve the equality KKT system, accepting only verified points or certificates.
pub(crate) fn solve(
    problem: &Problem,
    geometry: &ConeGeometry,
    equilibrated: &EquilibratedProblem<'_>,
    settings: &Settings,
    recovery: &mut Recovery<'_>,
    started: Instant,
    info: &mut SolutionInfo,
) -> Result<Termination, SolveError> {
    let scaled = equilibrated.problem();
    let n = problem.variable_count();
    let m = problem.constraint_count();
    let zero_diagonal = vec![0.0; m];
    let rhs: Vec<_> = scaled
        .c()
        .iter()
        .map(|c| -c)
        .chain(scaled.b().iter().copied())
        .collect();
    let size = rhs.len();
    // Preserve acceptance of an already optimal zero point, including with a
    // zero time limit, before attempting any factorization.
    let zero = State::from_stacked(vec![0.0; size], n, vec![0.0; m], 1.0, geometry)?;
    if recovery.update(&zero, 0.0).is_solved(settings) {
        return Ok(Termination::Point(SolverStatus::Solved));
    }
    if time_limit_reached(started, settings.time_limit) {
        return Ok(Termination::Point(SolverStatus::TimeLimit));
    }
    let matrix = assemble_equality_kkt(scaled, geometry)
        .map_err(|e| SolveError::Numerical(e.to_string()))?;
    let operator = EqualityKkt {
        problem: scaled,
        zero_diagonal: &zero_diagonal,
    };
    let stats = &mut info.linear_algebra;
    stats.factorizations += 1;
    info.iters = 1;
    let preconditioner = DynamicLdlt::new(&matrix, n, settings.threads)?;
    let par = preconditioner.parallelism;
    let mut buffer =
        MemBuffer::try_new(preconditioner.apply_in_place_scratch(1, par)).map_err(|_| {
            SolveError::Numerical("equality refinement workspace allocation failed".into())
        })?;
    let mut point = rhs.clone();
    preconditioner.apply_in_place(
        MatMut::from_column_major_slice_mut(&mut point, size, 1),
        par,
        MemStack::new(&mut buffer),
    );
    stats.primary_solves += 1;
    let mut delta = vec![0.0; size];
    // This path uses original-problem optimality, rather than refine_tol,
    // and a fixed ten-correction budget before the LSMR fallback.
    for correction in 0..=10 {
        if !point.iter().all(|v| v.is_finite()) {
            break;
        }
        let state = State::from_stacked(point.clone(), n, vec![0.0; m], 1.0, geometry)?;
        let recovered = recovery.update(&state, 0.0);
        if recovered.is_solved(settings) {
            return Ok(Termination::Point(SolverStatus::Solved));
        }
        if time_limit_reached(started, settings.time_limit) {
            return Ok(Termination::Point(SolverStatus::TimeLimit));
        }
        if correction == 10 {
            break;
        }
        delta.copy_from_slice(&rhs);
        apply_full_system(scaled, 0.0, &zero_diagonal, point.split_at(n), &mut delta);
        preconditioner.apply_in_place(
            MatMut::from_column_major_slice_mut(&mut delta, size, 1),
            par,
            MemStack::new(&mut buffer),
        );
        stats.refinement_solves += 1;
        crate::linear_algebra::add_assign(&mut point, &delta);
    }
    // Reuse the factors, but restart from zero to discard a growing nullspace
    // component left by refinement on an inconsistent system.
    let (point, residual) = least_squares(&operator, &rhs, &preconditioner, stats)?;
    let state = State::from_stacked(point, n, vec![0.0; m], 1.0, geometry)?;
    let recovered = recovery.update(&state, 0.0);
    if recovered.is_solved(settings) {
        return Ok(Termination::Point(SolverStatus::Solved));
    }
    if time_limit_reached(started, settings.time_limit) {
        return Ok(Termination::Point(SolverStatus::TimeLimit));
    }
    let y: Vec<_> = residual[n..].iter().map(|v| -v).collect();
    let ray = Ray::new(problem, equilibrated, &residual[..n], &y, &zero_diagonal);
    // The least-squares residual supplies a candidate directly, without a tau gate.
    if let Some(certificate) = ray.into_certificate(problem, settings, f64::INFINITY) {
        return Ok(Termination::Certificate(certificate));
    }
    Err(SolveError::Numerical(format!(
        "equality-only preconditioned LSMR did not produce a verified solution or certificate after a single LSMR solve ({} iterations)",
        stats.lsmr_iterations
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cone;

    #[test]
    fn assembled_triangle_and_matrix_free_products_match_kkt() {
        let p = CscMatrix::from_triplets(
            2,
            2,
            vec![0, 1, 0, 1],
            vec![0, 0, 1, 1],
            vec![2.0, 0.5, 0.5, 3.0],
        )
        .unwrap();
        let a = CscMatrix::from_triplets(
            2,
            2,
            vec![0, 1, 0, 1],
            vec![0, 0, 1, 1],
            vec![1.0, -3.0, -2.0, 4.0],
        )
        .unwrap();
        let problem =
            Problem::new(Some(p), vec![0.0; 2], a, vec![0.0; 2], vec![Cone::Zero(2)]).unwrap();
        let expected = [
            [2.0, 0.5, 1.0, -3.0],
            [0.5, 3.0, -2.0, 4.0],
            [1.0, -2.0, 0.0, 0.0],
            [-3.0, 4.0, 0.0, 0.0],
        ];
        let diagonal = [0.0; 2];
        let triangle =
            assemble_equality_kkt(&problem, &ConeGeometry::new(&problem).unwrap()).unwrap();
        for (i, row) in expected.iter().enumerate() {
            for (j, value) in row.iter().enumerate().skip(i) {
                assert_eq!(triangle.get(i, j).unwrap_or(0.0), *value);
            }
        }
        let operator = EqualityKkt {
            problem: &problem,
            zero_diagonal: &diagonal,
        };
        // A transposed view exercises strided inputs and multiple RHS columns.
        let rhs = Mat::from_fn(2, 4, |i, j| (1 + i * 4 + j) as f64);
        let mut output = Mat::zeros(4, 2);
        let mut buffer = MemBuffer::new(operator.apply_scratch(2, Par::Seq));
        for transpose in [false, true] {
            if transpose {
                operator.transpose_apply(
                    output.as_mut(),
                    rhs.as_ref().transpose(),
                    Par::Seq,
                    MemStack::new(&mut buffer),
                );
            } else {
                operator.apply(
                    output.as_mut(),
                    rhs.as_ref().transpose(),
                    Par::Seq,
                    MemStack::new(&mut buffer),
                );
            }
            for i in 0..4 {
                for j in 0..2 {
                    let value: f64 = (0..4).map(|k| expected[i][k] * rhs[(j, k)]).sum();
                    assert!((output[(i, j)] - value).abs() < 1e-12);
                }
            }
        }
    }
}
