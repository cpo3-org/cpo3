//! FAER factorization adapter.

#[cfg(test)]
use super::upper_triangle_usize;
use super::{
    LinearSystemError, TriangleSource, validate_finite, validate_rhs, validate_square,
    validate_upper_triangle,
};
use crate::CscMatrix;
use ::faer::{
    Conj, MatMut, Par, Side,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    linalg::cholesky::ldlt::factor::LdltRegularization,
    sparse::{SparseColMatRef, SymbolicSparseColMat, linalg::cholesky as sparse_cholesky},
};
// Disable Faer's dynamic pivot replacement. The solver still applies the
// true KKT shift and checks iterative refinement against that shifted system.
const FAER_REGULARIZATION_EPSILON: f64 = 0.0;
const FAER_REGULARIZATION_DELTA: f64 = 0.0;

pub(super) struct FaerFactorization {
    dim: usize,
    pattern: SymbolicSparseColMat<usize>,
    /// Expected pivot signs in factor order.
    signs: Vec<i8>,
    /// Source position of each entry in the permuted triangle.
    source: TriangleSource,
    values: Vec<f64>,
    symbolic: sparse_cholesky::SymbolicCholesky<usize>,
    factor_values: Vec<f64>,
    scratch: MemBuffer,
    parallelism: Par,
}

impl FaerFactorization {
    pub(super) fn new(
        matrix: CscMatrix,
        factorizations: &mut usize,
    ) -> Result<Self, LinearSystemError> {
        validate_square(matrix.rows(), matrix.columns())?;
        if matrix.rows() == 0 {
            return Err(LinearSystemError::Empty);
        }
        validate_upper_triangle(&matrix)?;
        let symbolic = sparse_cholesky::factorize_symbolic_cholesky(
            matrix.as_faer().symbolic(),
            Side::Upper,
            sparse_cholesky::SymmetricOrdering::Amd,
            sparse_cholesky::CholeskySymbolicParams::default(),
        )
        .map_err(|error| LinearSystemError::Factorization(format!("Faer symbolic: {error:?}")))?;
        let dim = matrix.rows();
        let source = TriangleSource::new(&matrix);
        let (column_pointers, row_indices, values) = matrix.into_parts();
        let pattern =
            SymbolicSparseColMat::new_checked(dim, dim, column_pointers, None, row_indices);
        // Faer's simplicial factorization and triangular solves are sequential.
        // Querying Rayon's thread count would start a pool they never use.
        let parallelism = match symbolic.raw() {
            sparse_cholesky::SymbolicCholeskyRaw::Simplicial(_) => Par::Seq,
            sparse_cholesky::SymbolicCholeskyRaw::Supernodal(_) => Par::rayon(0),
        };
        let factor_values = vec![0.0; symbolic.len_val()];
        let numeric_scratch = match symbolic.raw() {
            sparse_cholesky::SymbolicCholeskyRaw::Simplicial(_) => {
                sparse_cholesky::simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, f64>(
                    dim,
                )
            }
            sparse_cholesky::SymbolicCholeskyRaw::Supernodal(raw) => {
                sparse_cholesky::supernodal::factorize_supernodal_numeric_ldlt_scratch::<usize, f64>(
                    raw,
                    parallelism,
                    Default::default(),
                )
            }
        };
        let scratch_requirement = StackReq::any_of(&[
            numeric_scratch,
            symbolic.solve_in_place_scratch::<f64>(1, parallelism),
        ]);
        let scratch = MemBuffer::try_new(scratch_requirement).map_err(|_| {
            LinearSystemError::Factorization("Faer scratch allocation failed".to_owned())
        })?;
        let mut factorization = Self {
            dim,
            pattern,
            signs: Vec::new(),
            source,
            values: Vec::new(),
            symbolic,
            factor_values,
            scratch,
            parallelism,
        };
        factorization.signs = factorization.diagonal_signs(&values);
        let values = factorization.permute_input(values);
        factorization.factor(&values, factorizations)?;
        factorization.values = values;
        Ok(factorization)
    }

    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    pub(super) fn solve_in_place(&mut self, vector: &mut [f64]) -> Result<(), LinearSystemError> {
        validate_rhs(self.dim(), vector)?;
        let factor = sparse_cholesky::LdltRef::new(&self.symbolic, &self.factor_values);
        let view = MatMut::from_column_major_slice_mut(vector, self.dim, 1);
        let stack = MemStack::new(&mut self.scratch);
        factor.solve_in_place_with_conj(Conj::No, view, self.parallelism, stack);
        validate_finite(vector)
    }

    pub(super) fn refactor(
        &mut self,
        matrix: &CscMatrix,
        factorizations: &mut usize,
    ) -> Result<(), LinearSystemError> {
        if matrix.rows() != self.dim || matrix.columns() != self.dim {
            return Err(LinearSystemError::DimensionChanged {
                expected: self.dim,
                actual: matrix.rows(),
            });
        }
        let mut values = std::mem::take(&mut self.values);
        self.source.gather_values(matrix, &mut values)?;
        let result = self.factor(&values, factorizations);
        self.values = values;
        result
    }

    pub(super) fn refactor_diagonal(
        &mut self,
        diagonal: &[f64],
        factorizations: &mut usize,
    ) -> Result<(), LinearSystemError> {
        let mut values = std::mem::take(&mut self.values);
        self.source.gather_diagonal(diagonal, &mut values);
        let result = self.factor(&values, factorizations);
        self.values = values;
        result
    }

    fn diagonal_signs(&self, values: &[f64]) -> Vec<i8> {
        let mut signs = vec![0_i8; self.dim];
        for (column, sign) in signs.iter_mut().enumerate() {
            let range = self.pattern.col_ptr()[column]..self.pattern.col_ptr()[column + 1];
            for (row, value) in self.pattern.row_idx()[range.clone()]
                .iter()
                .zip(&values[range])
            {
                if *row == column {
                    *sign = if *value >= 0.0 { 1 } else { -1 };
                }
            }
        }
        signs
    }

    /// Cache the input layout expected by Faer's numeric kernel. Preserve the
    /// entry order of its unsorted permutation to preserve floating-point sums.
    fn permute_input(&mut self, values: Vec<f64>) -> Vec<f64> {
        let perm = self.symbolic.perm().expect("AMD supplies a permutation");
        let (forward, inverse) = perm.arrays();
        let upper = matches!(
            self.symbolic.raw(),
            sparse_cholesky::SymbolicCholeskyRaw::Simplicial(_)
        );
        let coordinates = |row: usize, column: usize| {
            let (row, column) = (inverse[row], inverse[column]);
            if upper {
                (row.min(column), row.max(column))
            } else {
                (row.max(column), row.min(column))
            }
        };
        let mut pointers = vec![0; self.dim + 1];
        for column in 0..self.dim {
            for &row in &self.pattern.row_idx()
                [self.pattern.col_ptr()[column]..self.pattern.col_ptr()[column + 1]]
            {
                pointers[coordinates(row, column).1 + 1] += 1;
            }
        }
        for column in 0..self.dim {
            pointers[column + 1] += pointers[column];
        }
        let mut next = pointers[..self.dim].to_vec();
        let mut rows = vec![0; values.len()];
        let mut permuted = vec![0.0; values.len()];
        let mut source = TriangleSource {
            matrix_nonzeros: self.source.matrix_nonzeros,
            positions: vec![0; values.len()],
            diagonal: Vec::with_capacity(self.dim),
        };
        for column in 0..self.dim {
            let range = self.pattern.col_ptr()[column]..self.pattern.col_ptr()[column + 1];
            for ((&row, &value), &position) in self.pattern.row_idx()[range.clone()]
                .iter()
                .zip(&values[range.clone()])
                .zip(&self.source.positions[range])
            {
                let (new_row, new_column) = coordinates(row, column);
                let slot = next[new_column];
                next[new_column] += 1;
                rows[slot] = new_row;
                permuted[slot] = value;
                source.positions[slot] = position;
                if row == column {
                    source.diagonal.push((slot, column));
                }
            }
        }
        self.signs = forward.iter().map(|&old| self.signs[old]).collect();
        self.pattern =
            SymbolicSparseColMat::new_unsorted_checked(self.dim, self.dim, pointers, None, rows);
        self.source = source;
        permuted
    }

    fn factor(
        &mut self,
        values: &[f64],
        factorizations: &mut usize,
    ) -> Result<(), LinearSystemError> {
        let matrix = SparseColMatRef::new(self.pattern.as_ref(), values);
        let stack = MemStack::new(&mut self.scratch);
        let regularization = LdltRegularization {
            dynamic_regularization_signs: Some(&self.signs),
            dynamic_regularization_delta: FAER_REGULARIZATION_DELTA,
            dynamic_regularization_epsilon: FAER_REGULARIZATION_EPSILON,
        };
        *factorizations += 1;
        match self.symbolic.raw() {
            sparse_cholesky::SymbolicCholeskyRaw::Simplicial(raw) => {
                sparse_cholesky::simplicial::factorize_simplicial_numeric_ldlt(
                    &mut self.factor_values,
                    matrix,
                    regularization,
                    raw,
                    stack,
                )
            }
            sparse_cholesky::SymbolicCholeskyRaw::Supernodal(raw) => {
                sparse_cholesky::supernodal::factorize_supernodal_numeric_ldlt(
                    &mut self.factor_values,
                    matrix,
                    regularization,
                    raw,
                    self.parallelism,
                    stack,
                    Default::default(),
                )
            }
        }
        .map_err(|error| LinearSystemError::Factorization(format!("Faer numeric: {error:?}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_permutation_matches_faer_for_both_numeric_kernels() {
        for dense in [false, true] {
            let n: usize = if dense { 128 } else { 8 };
            let mut rows = Vec::new();
            let mut columns = Vec::new();
            let mut values = Vec::new();
            let signs: Vec<i8> = (0..n).map(|i| if i < n / 2 { 1 } else { -1 }).collect();
            for column in 0..n {
                for (row, &sign) in signs.iter().enumerate() {
                    if dense || row.abs_diff(column) <= 1 {
                        rows.push(row);
                        columns.push(column);
                        values.push(if row == column {
                            2.0 * f64::from(sign)
                        } else {
                            0.001 * (1 + (row.min(column) + 3 * row.max(column)) % 7) as f64
                        });
                    }
                }
            }
            let matrix = crate::matrix::take_upper_triangle(
                CscMatrix::from_triplets(n, n, rows, columns, values).unwrap(),
            );
            let factor = FaerFactorization::new(matrix.clone(), &mut 0).unwrap();
            assert_eq!(
                matches!(
                    factor.symbolic.raw(),
                    sparse_cholesky::SymbolicCholeskyRaw::Supernodal(_)
                ),
                dense
            );
            let (pointers, rows, values) = upper_triangle_usize(&matrix);
            let pattern = SymbolicSparseColMat::new_checked(n, n, pointers, None, rows);
            let mut reference = vec![0.0; factor.factor_values.len()];
            let mut scratch = MemBuffer::new(
                factor
                    .symbolic
                    .factorize_numeric_ldlt_scratch::<f64>(factor.parallelism, Default::default()),
            );
            factor
                .symbolic
                .factorize_numeric_ldlt(
                    &mut reference,
                    SparseColMatRef::new(pattern.as_ref(), &values),
                    Side::Upper,
                    LdltRegularization {
                        dynamic_regularization_signs: Some(&signs),
                        dynamic_regularization_delta: FAER_REGULARIZATION_DELTA,
                        dynamic_regularization_epsilon: FAER_REGULARIZATION_EPSILON,
                    },
                    factor.parallelism,
                    MemStack::new(&mut scratch),
                    Default::default(),
                )
                .unwrap();
            assert_eq!(factor.factor_values, reference);
        }
    }
}
