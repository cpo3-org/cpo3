//! Apple Accelerate sparse LDLᵀ with retained symbolic analysis and workspaces.

use std::{ffi::c_void, ptr::NonNull};

use super::{LinearSystemError, TriangleSource};
use crate::CscMatrix;

unsafe extern "C" {
    fn cpo3_accelerate_available() -> bool;
    fn cpo3_accelerate_new(
        n: i32,
        columns: *mut i64,
        rows: *mut i32,
        status: *mut i32,
    ) -> *mut c_void;
    fn cpo3_accelerate_free(factor: *mut c_void);
    fn cpo3_accelerate_factor(factor: *mut c_void, values: *mut f64) -> i32;
    fn cpo3_accelerate_solve(factor: *mut c_void, columns: i32, rhs: *mut f64) -> i32;
}

pub(super) fn available() -> bool {
    // SAFETY: This only queries the running OS version.
    unsafe { cpo3_accelerate_available() }
}

pub(super) struct AccelerateFactorization {
    handle: NonNull<c_void>,
    // The C handle borrows these buffers until Drop; their allocations never move.
    _columns: Vec<i64>,
    _rows: Vec<i32>,
    values: Vec<f64>,
    source: TriangleSource,
    dim: usize,
}

impl AccelerateFactorization {
    pub(super) fn new(
        matrix: CscMatrix,
        variable_positions: &[usize],
        threads: usize,
    ) -> Result<Self, LinearSystemError> {
        if threads != 0 {
            return Err(LinearSystemError::Unavailable(
                "Accelerate supports only threads=0; its sparse solver has no per-solve thread cap"
                    .into(),
            ));
        }
        if !available() {
            return Err(LinearSystemError::Unavailable(
                "Accelerate requires macOS 13 or later".into(),
            ));
        }
        let dim = matrix.rows();
        let n = i32::try_from(dim).map_err(|_| {
            LinearSystemError::Factorization("Accelerate matrix dimension exceeds i32".into())
        })?;
        // Accelerate reads the CSC values in place.
        let source = TriangleSource::new(variable_positions, |position| position);
        let mut columns = matrix
            .column_pointers()
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>();
        let mut rows = matrix
            .row_indices()
            .iter()
            .map(|&v| v as i32)
            .collect::<Vec<_>>();
        let mut status = 0;
        // SAFETY: Validated upper CSC; all indices fit i32 after checking dim.
        // Column offsets fit i64 because Rust allocations cannot exceed isize::MAX.
        let handle =
            unsafe { cpo3_accelerate_new(n, columns.as_mut_ptr(), rows.as_mut_ptr(), &mut status) };
        let handle = NonNull::new(handle).ok_or_else(|| error(status))?;
        let values = matrix.into_parts().2;
        Ok(Self {
            handle,
            _columns: columns,
            _rows: rows,
            values,
            source,
            dim,
        })
    }

    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    pub(super) fn factor(&mut self, factorizations: &mut usize) -> Result<(), LinearSystemError> {
        *factorizations += 1;
        // SAFETY: The live handle and its fixed CSC buffers remain owned by self.
        check(unsafe { cpo3_accelerate_factor(self.handle.as_ptr(), self.values.as_mut_ptr()) })
    }

    pub(super) fn refactor_entries(
        &mut self,
        entries: &[f64],
        factorizations: &mut usize,
    ) -> Result<(), LinearSystemError> {
        self.source.gather(entries, &mut self.values);
        self.factor(factorizations)
    }

    pub(super) fn solve_in_place(
        &mut self,
        rhs: &mut [f64],
        columns: usize,
    ) -> Result<(), LinearSystemError> {
        // SAFETY: The dispatcher validates n * columns entries; columns is one or two.
        check(unsafe {
            cpo3_accelerate_solve(self.handle.as_ptr(), columns as i32, rhs.as_mut_ptr())
        })
    }
}

impl Drop for AccelerateFactorization {
    fn drop(&mut self) {
        // SAFETY: The handle is uniquely owned and its borrowed buffers are still live.
        unsafe { cpo3_accelerate_free(self.handle.as_ptr()) };
    }
}

fn error(status: i32) -> LinearSystemError {
    LinearSystemError::Factorization(format!("Accelerate returned status {status}"))
}

fn check(status: i32) -> Result<(), LinearSystemError> {
    if status == 0 {
        Ok(())
    } else {
        Err(error(status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_refactors_solve_after_diagonal_sign_changes() {
        if !available() {
            return;
        }
        // Include scalar factors (bound elimination can leave a 1x1 KKT matrix)
        // and sign changes on a retained symbolic factorization.
        for dim in [1, 2, 3] {
            let positions = (0..dim).collect::<Vec<_>>();
            let mut factor =
                AccelerateFactorization::new(CscMatrix::identity(dim).unwrap(), &positions, 0)
                    .unwrap();
            let mut count = 0;
            factor.factor(&mut count).unwrap();
            for pass in 0..6 {
                let diagonal: Vec<_> = (0..dim)
                    .map(|i| if (i + pass) % 2 == 0 { 2.0 } else { -3.0 })
                    .collect();
                factor.refactor_entries(&diagonal, &mut count).unwrap();
                let mut rhs = diagonal;
                factor.solve_in_place(&mut rhs, 1).unwrap();
                for x in rhs {
                    assert!((x - 1.0).abs() < 1e-12);
                }
            }
            assert_eq!(count, 7);
        }
    }

    #[test]
    fn repeated_refactors_solve_with_one_and_two_by_two_pivots() {
        if !available() {
            return;
        }
        // Tiny diagonals force a 2x2 pivot; later updates exercise both
        // positive and negative definite matrices with the same sparsity.
        let matrix =
            CscMatrix::new(2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![1e-12, 1.0, -1e-12]).unwrap();
        let mut factor = AccelerateFactorization::new(matrix, &[0, 2], 0).unwrap();
        let mut count = 0;
        factor.factor(&mut count).unwrap();
        for diagonal in [
            [1e-12, -1e-12],
            [3.0, -2.0],
            [-1e-12, 1e-12],
            [-2.0, -3.0],
            [4.0, 2.0],
            [1e-12, -1e-12],
        ] {
            factor.refactor_entries(&diagonal, &mut count).unwrap();
            let mut rhs = [
                diagonal[0] + 2.0,
                1.0 + 2.0 * diagonal[1],
                3.0 * diagonal[0] - 1.0,
                3.0 - diagonal[1],
            ];
            factor.solve_in_place(&mut rhs, 2).unwrap();
            for (x, expected) in rhs.into_iter().zip([1.0, 2.0, 3.0, -1.0]) {
                assert!((x - expected).abs() < 1e-12);
            }
        }
        assert_eq!(count, 7);
    }
}
