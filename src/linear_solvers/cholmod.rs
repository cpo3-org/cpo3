//! Runtime-loaded CHOLMOD simplicial LDLᵀ with retained analysis and solve workspace.

use std::{ffi::c_void, path::PathBuf, ptr::NonNull, sync::OnceLock};

use libloading::Library;

use super::{LinearSystemError, PivotReport, TriangleSource};
use crate::CscMatrix;

// Must match Cpo3CholmodApi in cholmod.c. CHOLMOD structs stay opaque to Rust.
#[repr(C)]
struct Api {
    start: unsafe extern "C" fn(*mut c_void) -> i32,
    finish: unsafe extern "C" fn(*mut c_void) -> i32,
    analyze: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void,
    factorize: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i32,
    solve2: unsafe extern "C" fn(
        i32,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
        *mut c_void,
    ) -> i32,
    free_dense: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> i32,
    free_factor: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> i32,
}

struct Cholmod {
    _library: Library,
    api: Api,
}

impl Cholmod {
    fn load() -> Result<Self, String> {
        let names: &[&str] = if cfg!(target_os = "windows") {
            &["cholmod.dll", "libcholmod.dll"]
        } else if cfg!(target_os = "macos") {
            &["libcholmod.5.dylib", "libcholmod.dylib"]
        } else {
            &["libcholmod.so.5", "libcholmod.so"]
        };
        let paths = if let Some(path) = std::env::var_os("CPO3_CHOLMOD") {
            vec![PathBuf::from(path)]
        } else {
            let mut paths = Vec::new();
            if let Some(prefix) = std::env::var_os("CONDA_PREFIX") {
                let directory = if cfg!(target_os = "windows") {
                    "Library/bin"
                } else {
                    "lib"
                };
                paths.extend(
                    names
                        .iter()
                        .map(|name| PathBuf::from(&prefix).join(directory).join(name)),
                );
            }
            paths.extend(names.iter().map(PathBuf::from));
            if cfg!(target_os = "macos") {
                for directory in [
                    "/opt/homebrew/opt/suite-sparse/lib",
                    "/usr/local/opt/suite-sparse/lib",
                    "/usr/local/lib",
                    "/opt/local/lib",
                ] {
                    paths.extend(names.iter().map(|name| PathBuf::from(directory).join(name)));
                }
            }
            paths
        };
        let mut load_error = String::new();
        let mut incompatible = None;
        for path in paths {
            // SAFETY: Load a user-installed library and retain it while its
            // symbols are in use. Check its version before passing any structs.
            let library = match unsafe { Library::new(&path) } {
                Ok(library) => library,
                Err(error) => {
                    load_error = format!("{}: {error}", path.display());
                    continue;
                }
            };
            match Self::from_library(library) {
                Ok(cholmod) => return Ok(cholmod),
                Err(error) => incompatible = Some(format!("{}: {error}", path.display())),
            }
        }
        Err(format!(
            "CHOLMOD runtime unavailable; install SuiteSparse with CHOLMOD 5.2.x or 5.3.x, or set CPO3_CHOLMOD to its shared library before starting the process ({})",
            incompatible.unwrap_or(load_error)
        ))
    }

    fn from_library(library: Library) -> Result<Self, String> {
        // SAFETY: The version function only writes three ints; unlike the
        // remaining API it does not depend on CHOLMOD's struct layouts.
        // 5.2 restored the ABI broken in 5.1; 5.3 retains those layouts.
        // Reject other series before calling start or exchanging structs.
        unsafe {
            let version = library
                .get::<unsafe extern "C" fn(*mut i32) -> i32>(b"cholmod_version\0")
                .map_err(|error| error.to_string())?;
            let mut runtime = [0; 3];
            version(runtime.as_mut_ptr());
            if !matches!(runtime, [5, 2..=3, _]) {
                return Err(format!(
                    "unsupported CHOLMOD version {}.{}.{}; expected 5.2.x or 5.3.x",
                    runtime[0], runtime[1], runtime[2]
                ));
            }
            let api = (|| -> Result<Api, libloading::Error> {
                Ok(Api {
                    start: *library.get(b"cholmod_l_start\0")?,
                    finish: *library.get(b"cholmod_l_finish\0")?,
                    analyze: *library.get(b"cholmod_l_analyze\0")?,
                    factorize: *library.get(b"cholmod_l_factorize\0")?,
                    solve2: *library.get(b"cholmod_l_solve2\0")?,
                    free_dense: *library.get(b"cholmod_l_free_dense\0")?,
                    free_factor: *library.get(b"cholmod_l_free_factor\0")?,
                })
            })()
            .map_err(|error| error.to_string())?;
            Ok(Self {
                _library: library,
                api,
            })
        }
    }
}

fn cholmod() -> Result<&'static Cholmod, LinearSystemError> {
    static CHOLMOD: OnceLock<Result<Cholmod, String>> = OnceLock::new();
    CHOLMOD
        .get_or_init(Cholmod::load)
        .as_ref()
        .map_err(|error| LinearSystemError::Unavailable(error.clone()))
}

pub(super) fn available() -> bool {
    cholmod().is_ok()
}

unsafe extern "C" {
    fn cpo3_cholmod_new(
        api: *const Api,
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
        let cholmod = cholmod()?;
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
        // The API and loaded library live in the process-wide OnceLock.
        let handle = unsafe {
            cpo3_cholmod_new(
                &cholmod.api,
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
        if !available() {
            return;
        }
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
