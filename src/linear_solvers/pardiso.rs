//! Intel oneMKL PARDISO, loaded at runtime with an explicit 64-bit integer ABI.

use std::{ffi::c_void, path::PathBuf, sync::OnceLock};

use libloading::Library;

use super::{LinearSystemError, PivotReport, TriangleSource};
use crate::CscMatrix;

type Pardiso = unsafe extern "C" fn(
    *mut *mut c_void,
    *const i64,
    *const i64,
    *const i64,
    *const i64,
    *const i64,
    *const f64,
    *const i64,
    *const i64,
    *mut i64,
    *const i64,
    *mut i64,
    *const i64,
    *mut f64,
    *mut f64,
    *mut i64,
);
type SetThreads = unsafe extern "C" fn(i32) -> i32;

struct Mkl {
    _library: Library,
    pardiso: Pardiso,
    set_threads: SetThreads,
}

impl Mkl {
    fn load() -> Result<Self, String> {
        let names: &[&str] = if cfg!(target_os = "windows") {
            &["mkl_rt.2.dll", "mkl_rt.dll"]
        } else if cfg!(target_os = "macos") {
            &["libmkl_rt.2.dylib", "libmkl_rt.dylib"]
        } else {
            &["libmkl_rt.so.2", "libmkl_rt.so"]
        };
        let paths = if let Some(path) = std::env::var_os("CPO3_MKL_RT") {
            vec![PathBuf::from(path)]
        } else {
            let mut paths = Vec::new();
            if let Some(root) = std::env::var_os("MKLROOT") {
                for directory in ["lib", "lib/intel64", "bin", "bin/intel64"] {
                    paths.extend(
                        names
                            .iter()
                            .map(|name| PathBuf::from(&root).join(directory).join(name)),
                    );
                }
            }
            paths.extend(names.iter().map(PathBuf::from));
            paths
        };
        let mut last_error = String::new();
        for path in paths {
            // SAFETY: Load the user-installed MKL runtime and retain it for the
            // lifetime of the function pointers. pardiso_64 always uses i64,
            // independently of MKL_INTERFACE_LAYER.
            let loaded = unsafe {
                Library::new(&path).and_then(|library| {
                    let pardiso = *library.get::<Pardiso>(b"pardiso_64\0")?;
                    let set_threads = *library
                        .get::<SetThreads>(b"mkl_set_num_threads_local\0")
                        .or_else(|_| library.get::<SetThreads>(b"MKL_Set_Num_Threads_Local\0"))?;
                    Ok(Self {
                        _library: library,
                        pardiso,
                        set_threads,
                    })
                })
            };
            match loaded {
                Ok(mkl) => return Ok(mkl),
                Err(error) => last_error = format!("{}: {error}", path.display()),
            }
        }
        Err(format!(
            "Intel MKL runtime unavailable; set CPO3_MKL_RT to its shared library before starting the process ({last_error})"
        ))
    }
}

fn mkl() -> Result<&'static Mkl, LinearSystemError> {
    static MKL: OnceLock<Result<Mkl, String>> = OnceLock::new();
    MKL.get_or_init(Mkl::load)
        .as_ref()
        .map_err(|error| LinearSystemError::Unavailable(error.clone()))
}

pub(super) fn available() -> bool {
    mkl().is_ok()
}

pub(super) struct PardisoFactorization {
    mkl: &'static Mkl,
    handle: [*mut c_void; 64],
    iparm: [i64; 64],
    n: i64,
    pointers: Vec<i64>,
    indices: Vec<i64>,
    values: Vec<f64>,
    source: TriangleSource,
    scratch: Vec<f64>,
    threads: i32,
    inertia: Option<PivotReport>,
}

impl PardisoFactorization {
    pub(super) fn new(matrix: CscMatrix, threads: usize) -> Result<Self, LinearSystemError> {
        let mkl = mkl()?;
        let threads = i32::try_from(threads)
            .map_err(|_| LinearSystemError::Unavailable("PARDISO threads must fit i32".into()))?;
        let n = matrix.rows();
        // PARDISO expects upper CSR, the transpose of our upper CSC. Include
        // every diagonal slot, even a structural zero, as required by MKL.
        let mut pointers = vec![0_i64; n + 1];
        for &row in matrix.row_indices() {
            pointers[row + 1] += 1;
        }
        for row in 0..n {
            if matrix.get(row, row).is_none() {
                pointers[row + 1] += 1;
            }
        }
        for row in 0..n {
            pointers[row + 1] += pointers[row];
        }
        let mut next = pointers[..n]
            .iter()
            .map(|&v| v as usize)
            .collect::<Vec<_>>();
        let nnz = pointers[n] as usize;
        let mut indices = vec![0; nnz];
        let mut values = vec![0.0; nnz];
        let mut source = TriangleSource {
            diagonal: Vec::with_capacity(n),
        };
        for (row, next_slot) in next.iter_mut().enumerate() {
            let slot = *next_slot;
            indices[slot] = row as i64;
            values[slot] = matrix.get(row, row).unwrap_or(0.0);
            source.diagonal.push((slot, row));
            *next_slot += 1;
        }
        for column in 0..n {
            for position in matrix.column_pointers()[column]..matrix.column_pointers()[column + 1] {
                let row = matrix.row_indices()[position];
                if row == column {
                    continue;
                }
                let slot = next[row];
                indices[slot] = column as i64;
                values[slot] = matrix.values()[position];
                next[row] += 1;
            }
        }
        let mut iparm = [0; 64];
        iparm[0] = 1; // Explicit controls.
        iparm[1] = 2; // METIS ordering, computed once from the pattern.
        iparm[7] = 0; // cpo3 performs and counts iterative refinement.
        iparm[9] = 13; // Perturb tiny pivots by 1e-13.
        iparm[20] = 3; // Bunch-Kaufman, without automatic refinement after perturbed pivots.
        iparm[26] = 1; // Check CSR during analysis.
        iparm[34] = 1; // Zero-based indices.
        // Scaling/matching stay disabled: symbolic analysis must remain valid
        // across diagonal updates, and Ruiz already scales the problem.
        let mut factor = Self {
            mkl,
            handle: [std::ptr::null_mut(); 64],
            iparm,
            n: n as i64,
            pointers,
            indices,
            values,
            source,
            scratch: Vec::new(),
            threads,
            inertia: None,
        };
        factor.call(11, 1, &mut 0.0, &mut 0.0)?;
        Ok(factor)
    }

    pub(super) fn dim(&self) -> usize {
        self.n as usize
    }

    fn call(
        &mut self,
        phase: i64,
        columns: i64,
        rhs: *mut f64,
        out: *mut f64,
    ) -> Result<(), LinearSystemError> {
        let mut error = 0;
        // SAFETY: All arrays have the documented PARDISO sizes and remain live
        // throughout the synchronous call. Only phases 11/22/33/-1 are used.
        // The solve dispatcher validates both RHS dimensions. MKL's local
        // thread setting is restored before returning, including on errors.
        unsafe {
            let previous = (self.mkl.set_threads)(self.threads);
            (self.mkl.pardiso)(
                self.handle.as_mut_ptr(),
                &1,
                &1,
                &-2,
                &phase,
                &self.n,
                self.values.as_ptr(),
                self.pointers.as_ptr(),
                self.indices.as_ptr(),
                std::ptr::null_mut(),
                &columns,
                self.iparm.as_mut_ptr(),
                &0,
                rhs,
                out,
                &mut error,
            );
            (self.mkl.set_threads)(previous);
        }
        match error {
            0 => Ok(()),
            -4 => Err(LinearSystemError::Singular),
            _ => Err(LinearSystemError::Factorization(format!(
                "PARDISO phase {phase} returned error {error}"
            ))),
        }
    }

    pub(super) fn factor(&mut self, factorizations: &mut usize) -> Result<(), LinearSystemError> {
        *factorizations += 1;
        self.inertia = None;
        self.call(22, 1, &mut 0.0, &mut 0.0)?;
        let (positive, negative) = (self.iparm[21] as usize, self.iparm[22] as usize);
        self.inertia = Some(PivotReport {
            positive,
            negative,
            zero: self.dim().saturating_sub(positive + negative),
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
        self.scratch.resize(rhs.len(), 0.0);
        let output = self.scratch.as_mut_ptr();
        self.call(33, columns as i64, rhs.as_mut_ptr(), output)?;
        rhs.copy_from_slice(&self.scratch);
        Ok(())
    }
}

impl Drop for PardisoFactorization {
    fn drop(&mut self) {
        let _ = self.call(-1, 1, &mut 0.0, &mut 0.0);
    }
}
