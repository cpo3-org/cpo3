use crate::{Cone, CscMatrix};
use thiserror::Error;

/// Errors detected while validating problem data.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ProblemError {
    #[error("P must be square with one row and column per element of c")]
    InvalidQuadraticShape,
    #[error("A must have one column per element of c")]
    InvalidConstraintColumnCount,
    #[error("A must have one row per element of b")]
    InvalidConstraintRowCount,
    #[error("problem matrices must contain only finite values")]
    NonFiniteMatrix,
    #[error("problem vectors must contain only finite values")]
    NonFiniteVector,
    #[error("P must be symmetric")]
    NonsymmetricQuadraticMatrix,
    #[error("cone dimensions must be strictly positive")]
    EmptyCone,
    #[error("a second-order cone must have dimension at least two")]
    InvalidSecondOrderCone,
    #[error("cone dimensions sum to {actual}, but A and b have {expected} rows")]
    ConeDimensionMismatch { expected: usize, actual: usize },
}

/// Validated, Rust-owned conic quadratic problem data.
#[derive(Clone, Debug)]
pub struct Problem {
    p: Option<CscMatrix>,
    c: Vec<f64>,
    a: CscMatrix,
    b: Vec<f64>,
    cones: Vec<Cone>,
}

impl Problem {
    /// Construct from validated data with upper-triangular P, without rechecking it.
    pub(crate) fn from_validated(
        p: Option<CscMatrix>,
        c: Vec<f64>,
        a: CscMatrix,
        b: Vec<f64>,
        cones: Vec<Cone>,
    ) -> Self {
        Self { p, c, a, b, cones }
    }

    /// Construct from a full symmetric `P`, retaining only its upper triangle.
    pub fn new(
        p: Option<CscMatrix>,
        c: Vec<f64>,
        a: CscMatrix,
        b: Vec<f64>,
        cones: Vec<Cone>,
    ) -> Result<Self, ProblemError> {
        let mut problem = Self { p, c, a, b, cones };
        problem.validate()?;
        if let Some(p) = &problem.p {
            validate_symmetry(p)?;
        }
        problem.p = problem.p.map(crate::matrix::take_upper_triangle);
        Ok(problem)
    }

    /// Construct from upper-triangular P, validating data without a symmetry check.
    #[cfg(any(feature = "mps", test))]
    pub(crate) fn from_upper_triangle(
        p: Option<CscMatrix>,
        c: Vec<f64>,
        a: CscMatrix,
        b: Vec<f64>,
        cones: Vec<Cone>,
    ) -> Result<Self, ProblemError> {
        let problem = Self { p, c, a, b, cones };
        problem.validate()?;
        Ok(problem)
    }

    fn validate(&self) -> Result<(), ProblemError> {
        let variable_count = self.c.len();
        if let Some(p) = &self.p
            && (p.rows() != variable_count || p.columns() != variable_count)
        {
            return Err(ProblemError::InvalidQuadraticShape);
        }
        if self.a.columns() != variable_count {
            return Err(ProblemError::InvalidConstraintColumnCount);
        }
        if self.a.rows() != self.b.len() {
            return Err(ProblemError::InvalidConstraintRowCount);
        }
        if self.a.values().iter().any(|value| !value.is_finite())
            || self
                .p
                .as_ref()
                .is_some_and(|p| p.values().iter().any(|value| !value.is_finite()))
        {
            return Err(ProblemError::NonFiniteMatrix);
        }
        if !self.c.iter().chain(&self.b).all(|value| value.is_finite()) {
            return Err(ProblemError::NonFiniteVector);
        }
        let mut cone_dim = 0usize;
        for cone in &self.cones {
            if cone.parameter() == 0 {
                return Err(ProblemError::EmptyCone);
            }
            if matches!(cone, Cone::SecondOrder(dim) if *dim < 2) {
                return Err(ProblemError::InvalidSecondOrderCone);
            }
            cone_dim = cone_dim.strict_add(cone.vector_dim());
        }
        if cone_dim != self.b.len() {
            return Err(ProblemError::ConeDimensionMismatch {
                expected: self.b.len(),
                actual: cone_dim,
            });
        }

        Ok(())
    }

    /// Upper triangle of the symmetric quadratic matrix, including its diagonal.
    pub fn p(&self) -> Option<&CscMatrix> {
        self.p.as_ref()
    }

    pub fn c(&self) -> &[f64] {
        &self.c
    }

    pub fn a(&self) -> &CscMatrix {
        &self.a
    }

    pub fn b(&self) -> &[f64] {
        &self.b
    }

    pub fn cones(&self) -> &[Cone] {
        &self.cones
    }

    pub fn variable_count(&self) -> usize {
        self.c.len()
    }

    pub fn constraint_count(&self) -> usize {
        self.b.len()
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.b.len(), self.c.len())
    }
}

fn validate_symmetry(matrix: &CscMatrix) -> Result<(), ProblemError> {
    let view = matrix.as_ref();
    if view.is_symmetric() {
        Ok(())
    } else {
        Err(ProblemError::NonsymmetricQuadraticMatrix)
    }
}

impl Problem {
    /// Multiply the quadratic matrix, returning None for a linear program.
    pub(crate) fn p_mul_opt(&self, vector: &[f64]) -> Option<Vec<f64>> {
        self.p().map(|p| {
            let mut product = Vec::new();
            crate::matrix::symmetric_mul_into(p, vector, &mut product);
            product
        })
    }

    pub(crate) fn p_mul(&self, vector: &[f64]) -> Vec<f64> {
        self.p_mul_opt(vector)
            .unwrap_or_else(|| vec![0.0; self.variable_count()])
    }
}

impl From<Problem> for ::presolve::Problem {
    fn from(p: Problem) -> Self {
        let (rows, cones) = crate::presolve::domains(&p.b, &p.cones);
        ::presolve::Problem {
            p: p.p.map(CscMatrix::into_native),
            c: p.c,
            c0: 0.,
            a: p.a.into_native(),
            rows,
            variable_bounds: vec![],
            cones,
        }
    }
}
