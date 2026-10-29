//! Equality-only solver: dynamically regularized LDL, at most ten refinement
//! corrections against K, then right-preconditioned LSMR if still unverified.
//! Diagonal Hessians use elimination first, falling back to the full KKT system
//! if the reduction is unsuitable or fails to produce a verified result.
//! A least-squares residual lies in ker(K). Its primal block is a dual
//! infeasibility candidate; minus its equality block is a primal certificate.
use crate::CscMatrix;
use crate::central_path::kkt::{apply_full_system, assemble_equality_kkt};
use crate::{
    Problem, Settings, SolutionInfo, SolveError, SolverStatus,
    cone::ConeGeometry,
    linear_solvers::LinearAlgebraStats,
    recovery::{Ray, Recovery, Termination},
    ruiz::EquilibratedProblem,
    solve::time_limit_reached,
};
use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
use faer::{
    ColRef, Conj, Mat, MatMut, MatRef, Par, Side,
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

/// Eliminate positive diagonal Hessian entries, retaining every zero entry.
/// The reduced symmetric system is [A+ D^-1 A+^T, A0; A0^T, 0],
/// with unknowns (y, -x0). All residuals and certificates still use the full KKT.
#[derive(Debug)]
struct DiagonalElimination<'a> {
    a: &'a CscMatrix,
    inverse: Vec<f64>,
    retained: Vec<usize>,
}

impl<'a> DiagonalElimination<'a> {
    fn new(problem: &'a Problem) -> Option<(Self, SparseColMat<usize, f64>)> {
        let p = problem.p()?;
        let n = problem.variable_count();
        let m = problem.constraint_count();
        let mut inverse = vec![0.0; n];
        for (column, inverse) in inverse.iter_mut().enumerate() {
            for entry in p.column_pointers()[column]..p.column_pointers()[column + 1] {
                let value = p.values()[entry];
                if value == 0.0 {
                    continue;
                }
                if p.row_indices()[entry] != column || value < 0.0 {
                    return None;
                }
                *inverse = value.recip();
                if !inverse.is_finite() {
                    return None;
                }
            }
        }
        let retained: Vec<_> = (0..n).filter(|&j| inverse[j] == 0.0).collect();
        if retained.len() == n || m + retained.len() == 0 {
            return None;
        }
        let a = problem.a();
        // Avoid forming a dense Schur complement from long constraint columns.
        let limit = 8usize.saturating_mul(a.values().len().saturating_add(n).saturating_add(m));
        let mut entries = 0usize;
        for (j, &inv) in inverse.iter().enumerate() {
            let count = a.column_pointers()[j + 1] - a.column_pointers()[j];
            let added = if inv == 0.0 {
                count
            } else {
                count.checked_mul(count.checked_add(1)?)? / 2
            };
            entries = entries.checked_add(added)?;
            if entries > limit {
                return None;
            }
        }
        let mut triplets = Vec::with_capacity(entries);
        for (j, &inv) in inverse.iter().enumerate() {
            if inv == 0.0 {
                continue;
            }
            let start = a.column_pointers()[j];
            let end = a.column_pointers()[j + 1];
            for right in start..end {
                for left in start..=right {
                    triplets.push(Triplet::new(
                        a.row_indices()[left],
                        a.row_indices()[right],
                        (a.values()[left] * inv) * a.values()[right],
                    ));
                }
            }
        }
        for (k, &j) in retained.iter().enumerate() {
            for entry in a.column_pointers()[j]..a.column_pointers()[j + 1] {
                triplets.push(Triplet::new(
                    a.row_indices()[entry],
                    m + k,
                    a.values()[entry],
                ));
            }
        }
        let size = m + retained.len();
        let matrix =
            SparseColMat::<usize, f64>::try_new_from_triplets(size, size, &triplets).ok()?;
        if matrix.val().iter().any(|value| !value.is_finite()) {
            return None;
        }
        Some((
            Self {
                a,
                inverse,
                retained,
            },
            matrix,
        ))
    }

    fn size(&self) -> usize {
        self.a.rows() + self.retained.len()
    }

    fn accumulate_reduced_rhs(&self, rhs: MatRef<'_, f64>, mut reduced: MatMut<'_, f64>) {
        let n = self.a.columns();
        let m = self.a.rows();
        faer::with_dim!(rows, m);
        faer::with_dim!(cols, n);
        let a = self.a.as_faer().as_shape(rows, cols);
        let inverse = ColRef::from_slice(&self.inverse).as_row_shape(cols);
        for col in 0..rhs.ncols() {
            let x = rhs.col(col).subrows(0, n).as_row_shape(cols);
            let mut top = reduced
                .as_mut()
                .col_mut(col)
                .subrows_mut(0, m)
                .as_row_shape_mut(rows);
            for j in cols.indices() {
                let inv = inverse[j];
                if inv == 0.0 {
                    continue;
                }
                let value = inv * x[j];
                for (&i, &entry) in a.row_idx_of_col_raw(j).iter().zip(a.val_of_col(j)) {
                    top[i] += entry * value;
                }
            }
        }
    }

    fn expand(&self, mut rhs: MatMut<'_, f64>, reduced: MatRef<'_, f64>) {
        let n = self.a.columns();
        let m = self.a.rows();
        faer::with_dim!(rows, m);
        faer::with_dim!(cols, n);
        let a = self.a.as_faer().as_shape(rows, cols);
        let inverse = ColRef::from_slice(&self.inverse).as_row_shape(cols);
        for col in 0..rhs.ncols() {
            let y = reduced.col(col).subrows(0, m).as_row_shape(rows);
            let mut x = rhs
                .as_mut()
                .col_mut(col)
                .subrows_mut(0, n)
                .as_row_shape_mut(cols);
            for j in cols.indices() {
                let inv = inverse[j];
                if inv == 0.0 {
                    continue;
                }
                let mut value = x[j];
                for (&i, &entry) in a.row_idx_of_col_raw(j).iter().zip(a.val_of_col(j)) {
                    value -= entry * y[i];
                }
                x[j] = inv * value;
            }
            for (k, &j) in self.retained.iter().enumerate() {
                rhs[(j, col)] = -reduced[(m + k, col)];
            }
            for i in 0..m {
                rhs[(n + i, col)] = reduced[(i, col)];
            }
        }
    }
}

/// A fixed inverse of Faer's dynamically regularized LDL factorization.
/// Both the forward and transpose preconditioner use the same symmetric solve.
#[derive(Debug)]
struct DynamicLdlt<'a> {
    elimination: Option<DiagonalElimination<'a>>,
    symbolic: sparse_cholesky::SymbolicCholesky<usize>,
    values: Vec<f64>,
    applications: AtomicUsize,
    parallelism: Par,
}

impl<'a> DynamicLdlt<'a> {
    fn new(
        matrix: SparseColMatRef<'_, usize, f64>,
        primal_size: usize,
        threads: usize,
    ) -> Result<Self, SolveError> {
        let symbolic = sparse_cholesky::factorize_symbolic_cholesky(
            matrix.symbolic(),
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
        let signs: Vec<i8> = (0..matrix.nrows())
            .map(|i| if i < primal_size { 1 } else { -1 })
            .collect();
        let mut buffer = MemBuffer::try_new(
            symbolic.factorize_numeric_ldlt_scratch::<f64>(parallelism, Default::default()),
        )
        .map_err(|_| SolveError::Numerical("equality LDL workspace allocation failed".into()))?;
        symbolic
            .factorize_numeric_ldlt(
                &mut values,
                matrix,
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
            elimination: None,
            symbolic,
            values,
            applications: AtomicUsize::new(0),
            parallelism,
        })
    }
}
impl LinOp<f64> for DynamicLdlt<'_> {
    fn nrows(&self) -> usize {
        self.elimination
            .as_ref()
            .map_or(self.symbolic.nrows(), |e| e.a.rows() + e.a.columns())
    }
    fn ncols(&self) -> usize {
        self.nrows()
    }
    fn apply_scratch(&self, columns: usize, par: Par) -> StackReq {
        let solve = self.symbolic.solve_in_place_scratch::<f64>(columns, par);
        self.elimination.as_ref().map_or(solve, |e| {
            StackReq::new::<f64>(e.size() * columns).and(solve)
        })
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
impl BiLinOp<f64> for DynamicLdlt<'_> {
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
impl Precond<f64> for DynamicLdlt<'_> {
    fn apply_in_place_scratch(&self, columns: usize, par: Par) -> StackReq {
        self.apply_scratch(columns, par)
    }
    fn apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        self.applications.fetch_add(rhs.ncols(), Ordering::Relaxed);
        let factor = sparse_cholesky::LdltRef::new(&self.symbolic, &self.values);
        if let Some(elimination) = &self.elimination {
            let columns = rhs.ncols();
            let n = elimination.a.columns();
            let m = elimination.a.rows();
            let rhs_ref = rhs.as_ref();
            let initial = (0..columns).flat_map(|col| {
                let column = rhs_ref.col(col);
                (0..m)
                    .map(move |i| -column[n + i])
                    .chain(elimination.retained.iter().map(move |&j| column[j]))
            });
            let (mut buffer, stack) = stack.collect(initial);
            let mut reduced =
                MatMut::from_column_major_slice_mut(&mut buffer, elimination.size(), columns);
            elimination.accumulate_reduced_rhs(rhs.as_ref(), reduced.as_mut());
            factor.solve_in_place_with_conj(Conj::No, reduced.as_mut(), par, stack);
            elimination.expand(rhs, reduced.as_ref());
        } else {
            factor.solve_in_place_with_conj(Conj::No, rhs, par, stack);
        }
    }
    fn conj_apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        self.apply_in_place(rhs, par, stack);
    }
}
impl BiPrecond<f64> for DynamicLdlt<'_> {
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
    preconditioner: &DynamicLdlt<'_>,
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
    let applications = preconditioner.applications.load(Ordering::Relaxed);
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
    stats.primary_solves += preconditioner.applications.load(Ordering::Relaxed) - applications;
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
    let rhs: Vec<_> = scaled
        .c()
        .iter()
        .map(|c| -c)
        .chain(scaled.b().iter().copied())
        .collect();
    let size = rhs.len();
    // Preserve acceptance of an already optimal zero point, including with a
    // zero time limit, before attempting any factorization.
    let zero = vec![0.0; size];
    if recovery
        .update_from_parts(&zero[..n], &zero[n..], &zero[n..], 1.0, 0.0)
        .is_solved(settings)
    {
        return Ok(Termination::Point(SolverStatus::Solved));
    }
    if time_limit_reached(started, settings.time_limit) {
        return Ok(Termination::Point(SolverStatus::TimeLimit));
    }
    drop(zero);
    info.iters = 1;
    if let Some((elimination, matrix)) = DiagonalElimination::new(scaled) {
        info.linear_algebra.factorizations += 1;
        if let Ok(mut factor) = DynamicLdlt::new(matrix.as_ref(), m, settings.threads) {
            factor.elimination = Some(elimination);
            let result = solve_factored(
                problem,
                equilibrated,
                settings,
                recovery,
                started,
                &mut info.linear_algebra,
                &factor,
                &rhs,
            );
            match result {
                Ok(termination) => return Ok(termination),
                Err(SolveError::Numerical(_)) => {}
                Err(error) => return Err(error),
            }
        }
    }
    if time_limit_reached(started, settings.time_limit) {
        return Ok(Termination::Point(SolverStatus::TimeLimit));
    }
    let matrix = assemble_equality_kkt(scaled, geometry)
        .map_err(|e| SolveError::Numerical(e.to_string()))?;
    info.linear_algebra.factorizations += 1;
    let factor = DynamicLdlt::new(matrix.as_faer(), n, settings.threads)?;
    solve_factored(
        problem,
        equilibrated,
        settings,
        recovery,
        started,
        &mut info.linear_algebra,
        &factor,
        &rhs,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_factored(
    problem: &Problem,
    equilibrated: &EquilibratedProblem<'_>,
    settings: &Settings,
    recovery: &mut Recovery<'_>,
    started: Instant,
    stats: &mut LinearAlgebraStats,
    preconditioner: &DynamicLdlt<'_>,
    rhs: &[f64],
) -> Result<Termination, SolveError> {
    let scaled = equilibrated.problem();
    let n = problem.variable_count();
    let m = problem.constraint_count();
    let size = rhs.len();
    let zero_diagonal = vec![0.0; m];
    let operator = EqualityKkt {
        problem: scaled,
        zero_diagonal: &zero_diagonal,
    };
    let par = preconditioner.parallelism;
    let mut buffer =
        MemBuffer::try_new(preconditioner.apply_in_place_scratch(1, par)).map_err(|_| {
            SolveError::Numerical("equality refinement workspace allocation failed".into())
        })?;
    let mut point = rhs.to_vec();
    preconditioner.apply_in_place(
        MatMut::from_column_major_slice_mut(&mut point, size, 1),
        par,
        MemStack::new(&mut buffer),
    );
    stats.primary_solves += 1;
    let mut delta = Vec::new();
    // This path uses original-problem optimality, rather than refine_tol,
    // and a fixed ten-correction budget before the LSMR fallback.
    for correction in 0..=10 {
        if !point.iter().all(|v| v.is_finite()) {
            break;
        }
        let recovered =
            recovery.update_from_parts(&point[..n], &point[n..], &zero_diagonal, 1.0, 0.0);
        if recovered.is_solved(settings) {
            return Ok(Termination::Point(SolverStatus::Solved));
        }
        if time_limit_reached(started, settings.time_limit) {
            return Ok(Termination::Point(SolverStatus::TimeLimit));
        }
        // Regularization can expose a certificate in the iterate's nullspace component.
        let ray = Ray::new(
            problem,
            equilibrated,
            &point[..n],
            &point[n..],
            &zero_diagonal,
        );
        if let Some(certificate) = ray.into_certificate(problem, settings, f64::INFINITY) {
            return Ok(Termination::Certificate(certificate));
        }
        if correction == 10 {
            break;
        }
        delta.clear();
        delta.extend_from_slice(rhs);
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
    let (point, residual) = least_squares(&operator, rhs, preconditioner, stats)?;
    let recovered = recovery.update_from_parts(&point[..n], &point[n..], &zero_diagonal, 1.0, 0.0);
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
    use crate::{Cone, linear_algebra::matrix};
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    #[rstest]
    #[case::positive(3.0)]
    #[case::zero(0.0)]
    fn elimination_solves_full_kkt_for_multiple_rhs(#[case] diagonal: f64) {
        let problem = Problem::new(
            Some(matrix(2, 2, &[2., 0., 0., diagonal])),
            vec![0.; 2],
            matrix(1, 2, &[1., 2.]),
            vec![0.],
            vec![Cone::Zero(1)],
        )
        .unwrap();
        let (elimination, matrix) = DiagonalElimination::new(&problem).unwrap();
        let mut factor = DynamicLdlt::new(matrix.as_ref(), 1, 1).unwrap();
        factor.elimination = Some(elimination);
        let expected = faer::mat![[1., -2.], [3., 4.], [-1., 2.]];
        let mut rhs = Mat::from_fn(3, 2, |i, j| match i {
            0 => 2. * expected[(0, j)] + expected[(2, j)],
            1 => diagonal * expected[(1, j)] + 2. * expected[(2, j)],
            _ => expected[(0, j)] + 2. * expected[(1, j)],
        });
        let mut buffer = MemBuffer::new(factor.apply_in_place_scratch(2, Par::Seq));
        factor.apply_in_place(rhs.as_mut(), Par::Seq, MemStack::new(&mut buffer));
        for j in 0..2 {
            for i in 0..3 {
                assert_abs_diff_eq!(rhs[(i, j)], expected[(i, j)], epsilon = 1e-12);
            }
        }
    }

    #[rstest]
    #[case::off_diagonal([2., 1., 1., 2.])]
    #[case::zero([0.; 4])]
    #[case::overflowing_inverse([1e-320, 0., 0., 2.])]
    fn unsuitable_hessians_use_full_system(#[case] p: [f64; 4]) {
        let problem = Problem::new(
            Some(matrix(2, 2, &p)),
            vec![0.; 2],
            matrix(1, 2, &[1., 2.]),
            vec![1.],
            vec![Cone::Zero(1)],
        )
        .unwrap();
        assert!(DiagonalElimination::new(&problem).is_none());
    }
}
