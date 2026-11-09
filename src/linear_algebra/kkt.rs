//! Matrix-free KKT products, residuals, and a regularized LDL preconditioner.

use crate::{CscMatrix, Problem, SolveError};
use faer::{
    Conj, MatMut, MatRef, Par, Side,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    linalg::cholesky::ldlt::factor::LdltRegularization,
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    reborrow::ReborrowMut,
    sparse::linalg::cholesky as sparse_cholesky,
};

use std::sync::atomic::{AtomicUsize, Ordering};

/// Matrix-free K = [P + shift I, Aᵀ; A, -diag(d)], independent of its factorization.
#[derive(Debug)]
pub(crate) struct KktOperator<'a> {
    problem: &'a Problem,
    shift: f64,
    diagonal: Option<&'a [f64]>,
}

impl<'a> KktOperator<'a> {
    /// The unshifted operator [P, Aᵀ; A, 0].
    pub(crate) fn new(problem: &'a Problem) -> Self {
        Self {
            problem,
            shift: 0.0,
            diagonal: None,
        }
    }

    pub(crate) fn shifted(problem: &'a Problem, shift: f64, diagonal: &'a [f64]) -> Self {
        Self {
            problem,
            shift,
            diagonal: Some(diagonal),
        }
    }

    /// Subtract K * (x, y) from an existing accumulator.
    pub(crate) fn subtract(&self, point: (&[f64], &[f64]), out: &mut [f64]) {
        let (out_x, out_y) = out.split_at_mut(self.problem.variable_count());
        self.accumulate(point, [out_x], [out_y], |value| [-value]);
    }

    /// Overwrite out with rhs - K * point.
    pub(crate) fn residual(&self, rhs: &[f64], point: &[f64], out: &mut [f64]) {
        out.copy_from_slice(rhs);
        self.subtract(point.split_at(self.problem.variable_count()), out);
    }

    /// Compute the residual and its componentwise error denominator in one sparse pass.
    pub(crate) fn residual_and_denominator(
        &self,
        rhs: &[f64],
        point: &[f64],
        residual: &mut [f64],
        denominator: &mut [f64],
    ) {
        residual.copy_from_slice(rhs);
        for (out, rhs) in denominator.iter_mut().zip(rhs) {
            *out = rhs.abs();
        }
        let n = self.problem.variable_count();
        let (residual_x, residual_y) = residual.split_at_mut(n);
        let (denominator_x, denominator_y) = denominator.split_at_mut(n);
        self.accumulate(
            point.split_at(n),
            [residual_x, denominator_x],
            [residual_y, denominator_y],
            |value| [-value, value.abs()],
        );
    }

    fn product(&self, point: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        let n = self.problem.variable_count();
        let (out_x, out_y) = out.split_at_mut(n);
        self.accumulate(point.split_at(n), [out_x], [out_y], |value| [value]);
    }

    /// Accumulate signed products, optionally collecting absolute products in the same pass.
    fn accumulate<const N: usize>(
        &self,
        (x, y): (&[f64], &[f64]),
        mut out_x: [&mut [f64]; N],
        mut out_y: [&mut [f64]; N],
        map: impl Fn(f64) -> [f64; N] + Copy,
    ) {
        if let Some(p) = self.problem.p() {
            super::symmetric_accumulate(p, x, out_x.each_mut().map(|out| &mut **out), map);
        }
        if self.shift != 0.0 {
            for (i, &x) in x.iter().enumerate() {
                for (out, value) in out_x.iter_mut().zip(map(self.shift * x)) {
                    out[i] += value;
                }
            }
        }
        super::constraint_accumulate(
            self.problem.a(),
            x,
            y,
            out_x,
            out_y.each_mut().map(|out| &mut **out),
            map,
        );
        if let Some(diagonal) = self.diagonal {
            for (i, (&y, &d)) in y.iter().zip(diagonal).enumerate() {
                for (out, value) in out_y.iter_mut().zip(map(-d * y)) {
                    out[i] += value;
                }
            }
        }
    }
}

impl LinOp<f64> for KktOperator<'_> {
    fn apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::new::<f64>(2 * self.nrows())
    }

    fn nrows(&self) -> usize {
        self.problem.variable_count() + self.problem.constraint_count()
    }

    fn ncols(&self) -> usize {
        self.nrows()
    }

    fn apply(&self, mut out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, _: Par, stack: &mut MemStack) {
        if let (Some(rhs), Some(mut out)) =
            (rhs.try_as_col_major(), out.rb_mut().try_as_col_major_mut())
        {
            for column in 0..rhs.ncols() {
                let output = out.rb_mut().col_mut(column).as_slice_mut();
                self.product(rhs.col(column).as_slice(), output);
            }
            return;
        }

        // Strided views require contiguous scratch for the sparse kernels.
        let size = self.nrows();
        let (mut input, stack) = stack.make_with::<f64>(size, |_| 0.0);
        let (mut product, _) = stack.make_with::<f64>(size, |_| 0.0);
        for column in 0..rhs.ncols() {
            for i in 0..size {
                input[i] = rhs[(i, column)];
            }
            self.product(&input, &mut product);
            for i in 0..size {
                out[(i, column)] = product[i];
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

impl BiLinOp<f64> for KktOperator<'_> {
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

/// A fixed inverse of Faer's dynamically regularized LDL factorization.
/// Both the forward and transpose preconditioner use the same symmetric solve.
#[derive(Debug)]
pub(crate) struct KktPreconditioner {
    symbolic: sparse_cholesky::SymbolicCholesky<usize>,
    values: Vec<f64>,
    applications: AtomicUsize,
    parallelism: Par,
}

impl KktPreconditioner {
    pub(crate) fn new(
        matrix: &CscMatrix,
        primal_size: usize,
        threads: usize,
    ) -> Result<Self, SolveError> {
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

    /// Apply the fixed inverse to a single right-hand side using caller-owned scratch.
    pub(crate) fn solve_in_place(&self, values: &mut [f64], stack: &mut MemStack) {
        let size = values.len();
        self.apply_in_place(
            MatMut::from_column_major_slice_mut(values, size, 1),
            self.parallelism,
            stack,
        );
    }

    pub(crate) fn parallelism(&self) -> Par {
        self.parallelism
    }

    /// Number of right-hand sides solved, including refinement and LSMR applications.
    pub(crate) fn applications(&self) -> usize {
        self.applications.load(Ordering::Relaxed)
    }
}

impl LinOp<f64> for KktPreconditioner {
    fn apply_scratch(&self, columns: usize, par: Par) -> StackReq {
        self.symbolic.solve_in_place_scratch::<f64>(columns, par)
    }

    fn nrows(&self) -> usize {
        self.symbolic.nrows()
    }

    fn ncols(&self) -> usize {
        self.nrows()
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

impl BiLinOp<f64> for KktPreconditioner {
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

impl Precond<f64> for KktPreconditioner {
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

impl BiPrecond<f64> for KktPreconditioner {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cone, CscMatrix, central_path::kkt::assemble_unshifted_kkt};
    use faer::Mat;

    #[test]
    fn shifted_products_and_residuals_match_dense_system() {
        for quadratic in [false, true] {
            let p = quadratic.then(|| {
                CscMatrix::from_triplets(2, 2, vec![0, 0, 1], vec![0, 1, 1], vec![2.0, -0.5, 3.0])
                    .unwrap()
            });
            let a =
                CscMatrix::from_triplets(1, 2, vec![0, 0], vec![0, 1], vec![1.0, -2.0]).unwrap();
            let problem =
                Problem::from_upper_triangle(p, vec![0.0; 2], a, vec![0.0], vec![Cone::Zero(1)])
                    .unwrap();
            for shift in [0.0, 2.0] {
                for d in [0.0, 3.0] {
                    let diagonal = [d];
                    let operator = KktOperator::shifted(&problem, shift, &diagonal);
                    let q = f64::from(quadratic);
                    let matrix = [
                        [2.0 * q + shift, -0.5 * q, 1.0],
                        [-0.5 * q, 3.0 * q + shift, -2.0],
                        [1.0, -2.0, -d],
                    ];
                    let point = [2.0, -3.0, 4.0];
                    let rhs = [-1.0, 5.0, 2.0];
                    let mut product = [f64::NAN; 3];
                    operator.apply(
                        MatMut::from_column_major_slice_mut(&mut product, 3, 1),
                        MatRef::from_column_major_slice(&point, 3, 1),
                        Par::Seq,
                        MemStack::new(&mut []),
                    );
                    let mut residual = [f64::NAN; 3];
                    operator.residual(&rhs, &point, &mut residual);
                    let mut fused = [f64::NAN; 3];
                    let mut denominator = [f64::NAN; 3];
                    operator.residual_and_denominator(&rhs, &point, &mut fused, &mut denominator);
                    for i in 0..3 {
                        let terms: [f64; 3] = std::array::from_fn(|j| matrix[i][j] * point[j]);
                        assert_eq!(product[i], terms.iter().sum::<f64>());
                        assert_eq!(residual[i], rhs[i] - product[i]);
                        assert_eq!(fused[i], residual[i]);
                        assert_eq!(
                            denominator[i],
                            rhs[i].abs() + terms.iter().map(|v| v.abs()).sum::<f64>()
                        );
                    }
                }
            }
        }
    }

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
        let triangle = assemble_unshifted_kkt(&problem).unwrap();
        for (i, row) in expected.iter().enumerate() {
            for (j, value) in row.iter().enumerate().skip(i) {
                assert_eq!(triangle.get(i, j).unwrap_or(0.0), *value);
            }
        }
        let operator = KktOperator::new(&problem);
        // Exercise contiguous and strided inputs/outputs, including transpose application.
        let rhs = Mat::from_fn(4, 2, |i, j| (1 + i + 4 * j) as f64);
        let rhs_transposed = rhs.transpose().to_owned();
        for strided_input in [false, true] {
            for strided_output in [false, true] {
                for transpose in [false, true] {
                    let input = if strided_input {
                        rhs_transposed.as_ref().transpose()
                    } else {
                        rhs.as_ref()
                    };
                    let mut storage = if strided_output {
                        Mat::full(2, 4, f64::NAN)
                    } else {
                        Mat::full(4, 2, f64::NAN)
                    };
                    let mut output = if strided_output {
                        storage.as_mut().transpose_mut()
                    } else {
                        storage.as_mut()
                    };
                    // The contiguous path must not need any scratch storage.
                    let scratch = if strided_input || strided_output {
                        operator.apply_scratch(2, Par::Seq)
                    } else {
                        StackReq::EMPTY
                    };
                    let mut buffer = MemBuffer::new(scratch);
                    if transpose {
                        operator.transpose_apply(
                            output.rb_mut(),
                            input,
                            Par::Seq,
                            MemStack::new(&mut buffer),
                        );
                    } else {
                        operator.apply(
                            output.rb_mut(),
                            input,
                            Par::Seq,
                            MemStack::new(&mut buffer),
                        );
                    }
                    for i in 0..4 {
                        for j in 0..2 {
                            let value: f64 = (0..4).map(|k| expected[i][k] * rhs[(k, j)]).sum();
                            assert!((output[(i, j)] - value).abs() < 1e-12);
                        }
                    }
                }
            }
        }
        let point = [1.0, 2.0, 3.0, 4.0];
        let rhs = [5.0, 6.0, 7.0, 8.0];
        let mut residual = [f64::NAN; 4];
        operator.residual(&rhs, &point, &mut residual);
        for i in 0..4 {
            let value = rhs[i] - (0..4).map(|j| expected[i][j] * point[j]).sum::<f64>();
            assert_eq!(residual[i], value);
        }
    }
}
