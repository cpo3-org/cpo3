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
    /// The presolve builder preserves dimensions, upper-triangle storage, and cone blocks.
    pub(crate) fn from_presolved(
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
        if let Some(p) = &p {
            if p.rows() != c.len() || p.columns() != c.len() {
                return Err(ProblemError::InvalidQuadraticShape);
            }
            validate_symmetry(p)?;
        }
        Self::from_upper_triangle(p.map(crate::matrix::take_upper_triangle), c, a, b, cones)
    }

    /// Internal builders supply a validated upper-triangle P; validate the
    /// remaining problem data without expanding or checking its symmetry again.
    pub(crate) fn from_upper_triangle(
        p: Option<CscMatrix>,
        c: Vec<f64>,
        a: CscMatrix,
        b: Vec<f64>,
        cones: Vec<Cone>,
    ) -> Result<Self, ProblemError> {
        let variable_count = c.len();
        if let Some(p) = &p
            && (p.rows() != variable_count || p.columns() != variable_count)
        {
            return Err(ProblemError::InvalidQuadraticShape);
        }
        if a.columns() != variable_count {
            return Err(ProblemError::InvalidConstraintColumnCount);
        }
        if a.rows() != b.len() {
            return Err(ProblemError::InvalidConstraintRowCount);
        }
        if !c.iter().chain(&b).all(|value| value.is_finite()) {
            return Err(ProblemError::NonFiniteVector);
        }
        let mut cone_dim = 0usize;
        for cone in &cones {
            if cone.parameter() == 0 {
                return Err(ProblemError::EmptyCone);
            }
            if matches!(cone, Cone::SecondOrder(dim) if *dim < 2) {
                return Err(ProblemError::InvalidSecondOrderCone);
            }
            cone_dim = cone_dim.strict_add(cone.vector_dim());
        }
        if cone_dim != b.len() {
            return Err(ProblemError::ConeDimensionMismatch {
                expected: b.len(),
                actual: cone_dim,
            });
        }

        Ok(Self { p, c, a, b, cones })
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
