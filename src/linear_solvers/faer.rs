//! FAER factorization adapter.

use super::{LinearSystemError, TriangleSource};
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
    /// Diagonal slots in the permuted triangle.
    source: TriangleSource,
    values: Vec<f64>,
    symbolic: sparse_cholesky::SymbolicCholesky<usize>,
    factor_values: Vec<f64>,
    scratch: MemBuffer,
    parallelism: Par,
}

impl FaerFactorization {
    pub(super) fn new(matrix: CscMatrix, threads: usize) -> Result<Self, LinearSystemError> {
        let symbolic = sparse_cholesky::factorize_symbolic_cholesky(
            matrix.as_faer().symbolic(),
            Side::Upper,
            sparse_cholesky::SymmetricOrdering::Amd,
            sparse_cholesky::CholeskySymbolicParams::default(),
        )
        .map_err(|error| LinearSystemError::Factorization(format!("Faer symbolic: {error:?}")))?;
        let dim = matrix.rows();
        let (column_pointers, row_indices, values) = matrix.into_parts();
        let pattern =
            SymbolicSparseColMat::new_checked(dim, dim, column_pointers, None, row_indices);
        // Faer's simplicial factorization and triangular solves are sequential.
        // Querying Rayon's thread count would start a pool they never use.
        let parallelism = match symbolic.raw() {
            sparse_cholesky::SymbolicCholeskyRaw::Simplicial(_) => Par::Seq,
            sparse_cholesky::SymbolicCholeskyRaw::Supernodal(_) if threads == 1 => Par::Seq,
            sparse_cholesky::SymbolicCholeskyRaw::Supernodal(_) => Par::rayon(threads),
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
            symbolic.solve_in_place_scratch::<f64>(2, parallelism),
        ]);
        let scratch = MemBuffer::try_new(scratch_requirement).map_err(|_| {
            LinearSystemError::Factorization("Faer scratch allocation failed".to_owned())
        })?;
        let mut factorization = Self {
            dim,
            pattern,
            source: TriangleSource::default(),
            values: Vec::new(),
            symbolic,
            factor_values,
            scratch,
            parallelism,
        };
        let values = factorization.permute_input(&values);
        factorization.values = values;
        Ok(factorization)
    }

    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    pub(super) fn solve_in_place(&mut self, vector: &mut [f64], columns: usize) {
        let factor = sparse_cholesky::LdltRef::new(&self.symbolic, &self.factor_values);
        let view = MatMut::from_column_major_slice_mut(vector, self.dim, columns);
        let stack = MemStack::new(&mut self.scratch);
        factor.solve_in_place_with_conj(Conj::No, view, self.parallelism, stack);
    }

    pub(super) fn refactor_diagonal(
        &mut self,
        diagonal: &[f64],
        factorizations: &mut usize,
    ) -> Result<(), LinearSystemError> {
        self.source.gather_diagonal(diagonal, &mut self.values);
        self.factor(factorizations)
    }

    /// Cache the input layout expected by Faer's numeric kernel. Preserve the
    /// entry order of its unsorted permutation to preserve floating-point sums.
    fn permute_input(&mut self, values: &[f64]) -> Vec<f64> {
        let perm = self.symbolic.perm().expect("AMD supplies a permutation");
        let (_forward, inverse) = perm.arrays();
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
            diagonal: Vec::with_capacity(self.dim),
        };
        for column in 0..self.dim {
            let range = self.pattern.col_ptr()[column]..self.pattern.col_ptr()[column + 1];
            for (&row, &value) in self.pattern.row_idx()[range.clone()]
                .iter()
                .zip(&values[range])
            {
                let (new_row, new_column) = coordinates(row, column);
                let slot = next[new_column];
                next[new_column] += 1;
                rows[slot] = new_row;
                permuted[slot] = value;
                if row == column {
                    source.diagonal.push((slot, column));
                }
            }
        }
        self.pattern =
            SymbolicSparseColMat::new_unsorted_checked(self.dim, self.dim, pointers, None, rows);
        self.source = source;
        permuted
    }

    pub(super) fn factor(&mut self, factorizations: &mut usize) -> Result<(), LinearSystemError> {
        let matrix = SparseColMatRef::new(self.pattern.as_ref(), &self.values);
        let stack = MemStack::new(&mut self.scratch);
        let regularization = LdltRegularization {
            dynamic_regularization_signs: None,
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
