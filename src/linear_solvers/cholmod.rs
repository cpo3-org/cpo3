//! SuiteSparse CHOLMOD simplicial LDLᵀ with retained analysis and solve workspace.

use std::{ffi::c_void, ptr::NonNull};

use super::{LinearSystemError, PivotReport, TriangleSource};
use crate::CscMatrix;

unsafe extern "C" {
    fn cpo3_cholmod_new(
        n: usize,
        columns: *mut i64,
        rows: *mut i64,
        values: *mut f64,
        threads: i32,
        status: *mut i32,
    ) -> *mut c_void;
    fn cpo3_cholmod_free(solver: *mut c_void);
    fn cpo3_cholmod_factor(solver: *mut c_void, positive: *mut usize, negative: *mut usize) -> i32;
    fn cpo3_cholmod_solve(solver: *mut c_void, columns: usize, rhs: *mut f64) -> i32;
}

pub(super) struct CholmodFactorization {
    handle: NonNull<c_void>,
    // The C handle borrows these fixed buffers until Drop.
    _columns: Vec<i64>,
    _rows: Vec<i64>,
    values: Vec<f64>,
    source: TriangleSource,
    dim: usize,
    inertia: Option<PivotReport>,
}

impl CholmodFactorization {
    pub(super) fn new(matrix: CscMatrix, threads: usize) -> Result<Self, LinearSystemError> {
        let threads = i32::try_from(threads)
            .map_err(|_| LinearSystemError::Unavailable("CHOLMOD threads must fit i32".into()))?;
        let dim = matrix.rows();
        let source = TriangleSource::from_matrix(&matrix);
        let (columns, rows, mut values) = matrix.into_parts();
        let mut columns = columns.into_iter().map(|v| v as i64).collect::<Vec<_>>();
        let mut rows = rows.into_iter().map(|v| v as i64).collect::<Vec<_>>();
        let mut status = 0;
        // SAFETY: Validated upper CSC. Rust allocations are bounded by isize::MAX,
        // so indices fit i64. All borrowed allocations are retained until Drop.
        let handle = unsafe {
            cpo3_cholmod_new(
                dim,
                columns.as_mut_ptr(),
                rows.as_mut_ptr(),
                values.as_mut_ptr(),
                threads,
                &mut status,
            )
        };
        let handle = NonNull::new(handle).ok_or_else(|| error(status))?;
        Ok(Self {
            handle,
            _columns: columns,
            _rows: rows,
            values,
            source,
            dim,
            inertia: None,
        })
    }

    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    pub(super) fn factor(&mut self, factorizations: &mut usize) -> Result<(), LinearSystemError> {
        *factorizations += 1;
        self.inertia = None;
        let (mut positive, mut negative) = (0, 0);
        // SAFETY: The handle and its fixed CSC buffers are live and uniquely owned.
        check(unsafe { cpo3_cholmod_factor(self.handle.as_ptr(), &mut positive, &mut negative) })?;
        self.inertia = Some(PivotReport {
            positive,
            negative,
            zero: 0,
        });
        Ok(())
    }

    pub(super) fn refactor_diagonal(
        &mut self,
        diagonal: &[f64],
        factorizations: &mut usize,
    ) -> Result<(), LinearSystemError> {
        self.source.gather_diagonal(diagonal, &mut self.values);
        self.factor(factorizations)
    }

    pub(super) fn pivot_report(&self) -> Option<PivotReport> {
        self.inertia
    }

    pub(super) fn solve_in_place(
        &mut self,
        rhs: &mut [f64],
        columns: usize,
    ) -> Result<(), LinearSystemError> {
        // SAFETY: The dispatcher validates dim * columns entries; columns is one or two.
        check(unsafe { cpo3_cholmod_solve(self.handle.as_ptr(), columns, rhs.as_mut_ptr()) })
    }
}

impl Drop for CholmodFactorization {
    fn drop(&mut self) {
        // SAFETY: The handle is uniquely owned and its borrowed buffers are still live.
        unsafe { cpo3_cholmod_free(self.handle.as_ptr()) };
    }
}

fn error(status: i32) -> LinearSystemError {
    if status == 1 {
        LinearSystemError::Singular
    } else {
        LinearSystemError::Factorization(format!("CHOLMOD returned status {status}"))
    }
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
    fn refactor_recovers_after_a_zero_pivot() {
        let mut factor = CholmodFactorization::new(CscMatrix::identity(2).unwrap(), 1).unwrap();
        let mut count = 0;
        factor.factor(&mut count).unwrap();
        assert!(matches!(
            factor.refactor_diagonal(&[1.0, 0.0], &mut count),
            Err(LinearSystemError::Singular)
        ));
        assert!(factor.pivot_report().is_none());
        factor.refactor_diagonal(&[2.0, -3.0], &mut count).unwrap();
        let mut rhs = [4.0, 6.0];
        factor.solve_in_place(&mut rhs, 1).unwrap();
        assert_eq!(rhs, [2.0, -2.0]);
        assert_eq!(count, 3);
    }
}
