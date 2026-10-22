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
    pub(super) fn new(
        matrix: CscMatrix,
        threads: usize,
        factorizations: &mut usize,
    ) -> Result<Self, LinearSystemError> {
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
        let values = factorization.permute_input(values);
        factorization.factor(&values, factorizations)?;
        factorization.values = values;
        Ok(factorization)
    }

    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    pub(super) fn solve_in_place(
        &mut self,
        vector: &mut [f64],
        columns: usize,
    ) -> Result<(), LinearSystemError> {
        let factor = sparse_cholesky::LdltRef::new(&self.symbolic, &self.factor_values);
        let view = MatMut::from_column_major_slice_mut(vector, self.dim, columns);
        let stack = MemStack::new(&mut self.scratch);
        factor.solve_in_place_with_conj(Conj::No, view, self.parallelism, stack);
        Ok(())
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

    /// Cache the input layout expected by Faer's numeric kernel. Preserve the
    /// entry order of its unsorted permutation to preserve floating-point sums.
    fn permute_input(&mut self, values: Vec<f64>) -> Vec<f64> {
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

    fn factor(
        &mut self,
        values: &[f64],
        factorizations: &mut usize,
    ) -> Result<(), LinearSystemError> {
        let matrix = SparseColMatRef::new(self.pattern.as_ref(), values);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_pivot_replacement_preserves_tiny_signed_pivots() {
        let matrix =
            CscMatrix::from_triplets(2, 2, vec![0, 1], vec![0, 1], vec![1e-20, -1e-20]).unwrap();
        let mut factor = FaerFactorization::new(matrix, 3, &mut 0).unwrap();
        assert_eq!(factor.parallelism, Par::Seq);
        let mut rhs = vec![1e-20, 1e-20];
        factor.solve_in_place(&mut rhs, 1).unwrap();
        assert!((rhs[0] - 1.0).abs() < 1e-12);
        assert!((rhs[1] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn configured_threads_apply_to_supernodal_factorization_and_solves() {
        let n = 128;
        let matrix = CscMatrix::from_triplets(
            n,
            n,
            (0..n).flat_map(|column| 0..=column).collect(),
            (0..n)
                .flat_map(|column| std::iter::repeat_n(column, column + 1))
                .collect(),
            (0..n)
                .flat_map(|column| {
                    (0..=column).map(move |row| if row == column { 2.0 } else { 0.001 })
                })
                .collect(),
        )
        .unwrap();
        for threads in [0, 1, 2, 3] {
            let settings = super::super::KktSettings::from(&crate::Settings {
                threads,
                ..crate::Settings::default()
            });
            let factor =
                super::super::Factorization::new_counted(matrix.clone(), &settings, &mut 0)
                    .unwrap();
            let super::super::FactorizationInner::Faer(mut factor) = factor.inner else {
                unreachable!()
            };
            assert!(matches!(
                factor.symbolic.raw(),
                sparse_cholesky::SymbolicCholeskyRaw::Supernodal(_)
            ));
            assert_eq!(
                factor.parallelism,
                if threads == 1 {
                    Par::Seq
                } else {
                    Par::rayon(threads)
                }
            );
            for diagonal in [2.0, 3.0] {
                factor
                    .refactor_diagonal(&vec![diagonal; n], &mut 0)
                    .unwrap();
                for columns in [1, 2] {
                    let mut rhs = vec![diagonal + 0.001 * (n - 1) as f64; n * columns];
                    factor.solve_in_place(&mut rhs, columns).unwrap();
                    assert!(rhs.iter().all(|value| (value - 1.0).abs() < 1e-12));
                }
            }
        }
    }

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
            let matrix = crate::linear_algebra::take_upper_triangle(
                CscMatrix::from_triplets(n, n, rows, columns, values).unwrap(),
            );
            let factor = FaerFactorization::new(matrix.clone(), 1, &mut 0).unwrap();
            assert_eq!(
                matches!(
                    factor.symbolic.raw(),
                    sparse_cholesky::SymbolicCholeskyRaw::Supernodal(_)
                ),
                dense
            );
            let (pointers, rows, values) = matrix.into_parts();
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
                        dynamic_regularization_signs: None,
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
