//! FERAL factorization adapter.

use super::{LinearSystemError, PivotReport, TriangleSource};
use crate::CscMatrix;
use ::feral::{
    CscMatrix as FeralCscMatrix, FactorStatus as FeralFactorStatus,
    NumericParams as FeralNumericParams, Solver as FeralSolver,
    scaling::ScalingStrategy as FeralScalingStrategy,
    symbolic::supernode::{
        OrderingPreprocess as FeralOrderingPreprocess, SupernodeParams as FeralSupernodeParams,
    },
};

pub(super) struct FeralFactorization {
    solver: FeralSolver,
    matrix: FeralCscMatrix,
    /// Diagonal slots in the lower triangle.
    source: TriangleSource,
    /// Output buffer for `solve_into`, which needs a distinct target.
    scratch: Vec<f64>,
}

impl FeralFactorization {
    pub(super) fn new(matrix: CscMatrix, threads: usize) -> Result<Self, LinearSystemError> {
        let (prepared, source) = to_feral(&matrix)?;
        drop(matrix);
        let matrix = prepared;
        let solver = feral_solver(threads);
        Ok(Self {
            solver,
            matrix,
            source,
            scratch: Vec::new(),
        })
    }

    pub(super) fn dim(&self) -> usize {
        self.matrix.n
    }

    pub(super) fn solve_in_place(&mut self, vector: &mut [f64]) -> Result<(), LinearSystemError> {
        self.scratch.resize(vector.len(), 0.0);
        self.solver
            .solve_into(vector, &mut self.scratch)
            .map_err(|error| LinearSystemError::Factorization(error.to_string()))?;
        vector.copy_from_slice(&self.scratch);
        Ok(())
    }

    pub(super) fn pivot_report(&self) -> Option<PivotReport> {
        let inertia = self.solver.inertia()?;
        Some(PivotReport {
            positive: inertia.positive,
            negative: inertia.negative,
            zero: inertia.zero,
        })
    }

    pub(super) fn refactor_diagonal(
        &mut self,
        diagonal: &[f64],
        factorizations: &mut usize,
    ) -> Result<(), LinearSystemError> {
        self.source
            .gather_diagonal(diagonal, &mut self.matrix.values);
        self.factor(factorizations)
    }

    pub(super) fn factor(&mut self, factorizations: &mut usize) -> Result<(), LinearSystemError> {
        *factorizations += 1;
        check_feral_status(self.solver.factor(&self.matrix, None))
    }
}

/// Feral with a pattern-only ordering, so its symbolic cache stays valid as
/// the KKT values change; the default value-dependent matching preprocess
/// computes a much worse ordering on the first IPM matrix and then caches it.
fn feral_solver(threads: usize) -> FeralSolver {
    let supernode = FeralSupernodeParams {
        preprocess: FeralOrderingPreprocess::None,
        ..FeralSupernodeParams::default()
    };
    let solver = FeralSolver::with_params(FeralNumericParams::default(), supernode)
        .with_scaling(FeralScalingStrategy::Identity);
    // Feral 0.21 creates its own pool and exposes only an on/off switch.
    // Use its sequential driver for any explicit cap so it cannot exceed it.
    if threads == 0 {
        solver
    } else {
        solver.with_parallel(false)
    }
}

fn to_feral(matrix: &CscMatrix) -> Result<(FeralCscMatrix, TriangleSource), LinearSystemError> {
    let n = matrix.rows();
    let nnz = matrix.values().len();
    let mut pointers = vec![0; n + 1];
    for &row in matrix.row_indices() {
        pointers[row + 1] += 1;
    }
    for column in 0..n {
        pointers[column + 1] += pointers[column];
    }
    let mut next = pointers[..n].to_vec();
    let mut rows = vec![0; nnz];
    let mut values = vec![0.0; nnz];
    let mut source = TriangleSource {
        diagonal: Vec::with_capacity(n),
    };
    for column in 0..n {
        for position in matrix.column_pointers()[column]..matrix.column_pointers()[column + 1] {
            let row = matrix.row_indices()[position];
            let slot = next[row];
            next[row] += 1;
            rows[slot] = column;
            values[slot] = matrix.values()[position];
            if row == column {
                source.diagonal.push((slot, column));
            }
        }
    }
    let matrix = FeralCscMatrix {
        n,
        col_ptr: pointers,
        row_idx: rows,
        values,
    };
    matrix
        .validate()
        .map_err(|error| LinearSystemError::Factorization(error.to_string()))?;
    Ok((matrix, source))
}

fn check_feral_status(status: FeralFactorStatus) -> Result<(), LinearSystemError> {
    match status {
        FeralFactorStatus::Success => Ok(()),
        FeralFactorStatus::Singular => Err(LinearSystemError::Singular),
        FeralFactorStatus::WrongInertia { actual, expected } => {
            Err(LinearSystemError::Factorization(format!(
                "Feral reported inertia {actual:?}, expected {expected:?}"
            )))
        }
        FeralFactorStatus::FatalError(error) => {
            Err(LinearSystemError::Factorization(error.to_string()))
        }
    }
}
