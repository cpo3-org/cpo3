use faer::sparse::{SparseColMatRef, SymbolicSparseColMat};
use presolve::matrix::{CscMatrix as NativeMatrix, CscMatrixRef, MatrixError};
use std::sync::Arc;

/// Owned CSC values with a shared, immutable sparsity pattern.
#[derive(Clone, Debug)]
pub struct CscMatrix {
    pattern: Arc<SymbolicSparseColMat<usize>>,
    values: Vec<f64>,
}

impl PartialEq for CscMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows() == other.rows()
            && self.columns() == other.columns()
            && self.column_pointers() == other.column_pointers()
            && self.row_indices() == other.row_indices()
            && self.values() == other.values()
    }
}

impl CscMatrix {
    /// Adopt CSC buffers, checking the pattern once before faer uses it.
    ///
    /// # Panics
    /// Panics if the CSC pattern is invalid or the index and value lengths differ.
    pub fn from_parts(
        rows: usize,
        columns: usize,
        pointers: Vec<usize>,
        indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        let symbolic = SymbolicSparseColMat::new_checked(rows, columns, pointers, None, indices);
        assert_eq!(symbolic.row_idx().len(), values.len());
        Self {
            pattern: Arc::new(symbolic),
            values,
        }
    }

    /// Construct a matrix from CSC buffers.
    ///
    /// # Errors
    /// Returns an error for invalid CSC storage or dimensions that exceed the supported range.
    pub fn new(
        rows: usize,
        columns: usize,
        pointers: Vec<usize>,
        indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, MatrixError> {
        if rows > isize::MAX as usize || columns > isize::MAX as usize {
            return Err(MatrixError::DimensionOverflow);
        }
        NativeMatrix::new(rows, columns, pointers, indices, values).map(Self::from_native)
    }

    /// Construct a matrix from row, column, and value triplets.
    ///
    /// # Errors
    /// Returns an error for mismatched lengths, invalid indices or values, or unsupported dimensions.
    pub fn from_triplets(
        rows: usize,
        columns: usize,
        row_indices: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, MatrixError> {
        if rows > isize::MAX as usize || columns > isize::MAX as usize {
            return Err(MatrixError::DimensionOverflow);
        }
        NativeMatrix::from_triplets(rows, columns, row_indices, column_indices, values)
            .map(Self::from_native)
    }

    /// Construct an identity matrix.
    ///
    /// # Errors
    /// Returns an error if the dimension exceeds the supported range.
    pub fn identity(size: usize) -> Result<Self, MatrixError> {
        if size > isize::MAX as usize {
            return Err(MatrixError::DimensionOverflow);
        }
        NativeMatrix::identity(size).map(Self::from_native)
    }

    /// Construct a matrix with no stored entries.
    ///
    /// # Errors
    /// Returns an error if a dimension exceeds the supported range.
    pub fn zeros(rows: usize, columns: usize) -> Result<Self, MatrixError> {
        if rows > isize::MAX as usize || columns > isize::MAX as usize {
            return Err(MatrixError::DimensionOverflow);
        }
        NativeMatrix::zeros(rows, columns).map(Self::from_native)
    }

    pub fn rows(&self) -> usize {
        self.pattern.nrows()
    }
    pub fn columns(&self) -> usize {
        self.pattern.ncols()
    }
    pub fn column_pointers(&self) -> &[usize] {
        self.pattern.col_ptr()
    }
    pub fn row_indices(&self) -> &[usize] {
        self.pattern.row_idx()
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
    pub fn get(&self, row: usize, column: usize) -> Option<f64> {
        self.as_ref().get(row, column)
    }

    pub fn as_ref(&self) -> CscMatrixRef<'_> {
        CscMatrixRef::from_parts(
            self.rows(),
            self.columns(),
            self.column_pointers(),
            self.row_indices(),
            self.values(),
        )
    }

    pub fn into_parts(self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let symbolic = Arc::unwrap_or_clone(self.pattern);
        let (_, _, pointers, _, indices) = symbolic.into_parts();
        (pointers, indices, self.values)
    }

    pub(crate) fn as_faer(&self) -> SparseColMatRef<'_, usize, f64> {
        SparseColMatRef::new(self.pattern.as_ref().as_ref(), &self.values)
    }

    /// Replace values without duplicating validated column pointers and indices.
    /// Panics if the value count changes; callers handle numerical validation.
    pub(crate) fn with_values(&self, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), self.values.len());
        Self {
            pattern: Arc::clone(&self.pattern),
            values,
        }
    }

    pub(crate) fn from_native(matrix: NativeMatrix) -> Self {
        let (rows, columns) = (matrix.rows(), matrix.columns());
        let (pointers, indices, values) = matrix.into_parts();
        Self::from_parts(rows, columns, pointers, indices, values)
    }

    pub(crate) fn into_native(self) -> NativeMatrix {
        let (rows, columns) = (self.rows(), self.columns());
        let (pointers, indices, values) = self.into_parts();
        NativeMatrix::from_parts(rows, columns, pointers, indices, values)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::CscMatrix;

    pub(crate) fn matrix(rows: usize, cols: usize, data: &[f64]) -> CscMatrix {
        assert_eq!(data.len(), rows * cols);
        matrix_from_fn(rows, cols, |i, j| data[i * cols + j])
    }

    fn matrix_from_fn(rows: usize, cols: usize, entry: impl Fn(usize, usize) -> f64) -> CscMatrix {
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut values = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                let value = entry(i, j);
                if value != 0.0 {
                    row.push(i);
                    col.push(j);
                    values.push(value);
                }
            }
        }
        CscMatrix::from_triplets(rows, cols, row, col, values).unwrap()
    }

    #[test]
    fn rejects_malformed_sparse_storage() {
        assert!(CscMatrix::new(2, 2, vec![0, 1], vec![0], vec![1.]).is_err());
    }
}
