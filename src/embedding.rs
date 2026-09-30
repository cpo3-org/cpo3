//! Independent-slack homogeneous embedding primitives.

use crate::matrix::dot;
use thiserror::Error;

use crate::{
    Problem,
    cone::{ConeGeometry, ConeGeometryError},
};

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum EmbeddingError {
    #[error("homogeneous x has length {actual}, expected {expected}")]
    IncorrectXLength { expected: usize, actual: usize },
    #[error("homogeneous s has length {actual}, expected {expected}")]
    IncorrectSLength { expected: usize, actual: usize },
    #[error("homogeneous y has length {actual}, expected {expected}")]
    IncorrectYLength { expected: usize, actual: usize },
    #[error("homogeneous state entries must be finite")]
    NonFiniteState,
    #[error("homogeneous tau must be finite and strictly positive")]
    NonPositiveTau,
    #[error("the slack or dual point is not in the relative interior of its cone")]
    NonInteriorPoint,
    #[error("mu must be finite and strictly positive")]
    InvalidMu,
    #[error(transparent)]
    ConeGeometry(#[from] ConeGeometryError),
    #[error("an embedding calculation produced a non-finite value")]
    NonFiniteResult,
}

/// Homogeneous coordinates `(x, s, y, tau)` for the lifted embedding.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct State {
    xy: Vec<f64>,
    primal_len: usize,
    s: Vec<f64>,
    tau: f64,
}

impl State {
    pub(crate) fn new(
        x: Vec<f64>,
        s: Vec<f64>,
        y: Vec<f64>,
        tau: f64,
    ) -> Result<Self, EmbeddingError> {
        let primal_len = x.len();
        let mut xy = x;
        xy.extend(y);
        Self::from_stacked(xy, primal_len, s, tau)
    }

    /// Adopt an accepted KKT point without splitting or copying its storage.
    pub(crate) fn from_stacked(
        xy: Vec<f64>,
        primal_len: usize,
        s: Vec<f64>,
        tau: f64,
    ) -> Result<Self, EmbeddingError> {
        if !tau.is_finite() || tau <= 0.0 {
            return Err(EmbeddingError::NonPositiveTau);
        }
        let (x, y) = xy.split_at(primal_len);
        if !x.iter().chain(&s).chain(y).all(|value| value.is_finite()) {
            return Err(EmbeddingError::NonFiniteState);
        }
        Ok(Self {
            xy,
            primal_len,
            s,
            tau,
        })
    }

    pub(crate) fn x(&self) -> &[f64] {
        &self.xy[..self.primal_len]
    }

    pub(crate) fn s(&self) -> &[f64] {
        &self.s
    }

    pub(crate) fn y(&self) -> &[f64] {
        &self.xy[self.primal_len..]
    }

    pub(crate) fn tau(&self) -> f64 {
        self.tau
    }

    /// Normalize all homogeneous coordinates using the floored sphere norm.
    /// Validate the scale before mutating so failure leaves the state intact.
    pub(crate) fn normalize_sphere(&mut self, nu: usize) -> Result<(), EmbeddingError> {
        let norm = faer::ColRef::from_slice(&self.xy).norm_l2().hypot(self.tau);
        let scale = if norm.is_infinite() {
            // Preserve representable normalized coordinates even if the norm
            // exceeds f64::MAX. This exceptional path needs scaled squares.
            let maximum = faer::ColRef::from_slice(&self.xy).norm_max().max(self.tau);
            let sum = self.xy.iter().map(|v| (v / maximum).powi(2)).sum::<f64>()
                + (self.tau / maximum).powi(2);
            ((nu as f64 + 1.0) / sum).sqrt() / maximum
        } else {
            (nu as f64 + 1.0).sqrt() / norm.max(1e-15_f64.sqrt())
        };
        if !scale.is_finite() || scale <= 0.0 {
            return Err(EmbeddingError::NonFiniteResult);
        }
        let representable = |value: &f64| {
            let scaled = value * scale;
            scaled.is_finite() && (scaled != 0.0 || *value == 0.0)
        };
        if !representable(&self.tau) || !self.xy.iter().chain(&self.s).all(representable) {
            return Err(EmbeddingError::NonFiniteResult);
        }
        crate::matrix::scale(&mut self.xy, scale);
        crate::matrix::scale(&mut self.s, scale);
        self.tau *= scale;
        Ok(())
    }
}

/// The homogeneous problem and its product-cone geometry.
pub(crate) struct Embedding<'a> {
    problem: &'a Problem,
    geometry: ConeGeometry,
}

impl<'a> Embedding<'a> {
    pub(crate) fn new(problem: &'a Problem) -> Result<Self, EmbeddingError> {
        let geometry = ConeGeometry::new(problem.cones(), problem.constraint_count())?;
        Ok(Self { problem, geometry })
    }

    pub(crate) fn problem(&self) -> &Problem {
        self.problem
    }

    pub(crate) fn geometry(&self) -> &ConeGeometry {
        &self.geometry
    }

    /// Barrier parameter `nu` of the product cone.
    pub(crate) fn nu(&self) -> usize {
        self.geometry.barrier_parameter()
    }

    /// Duality measure `mu = <s, y> / nu`, zero when the cone has no barrier.
    pub(crate) fn duality_measure(&self, state: &State) -> f64 {
        if self.nu() == 0 {
            return 0.0;
        }
        dot(&state.s, state.y()) / self.nu() as f64
    }

    pub(crate) fn ensure_domain(&self, state: &State) -> Result<(), EmbeddingError> {
        self.validate_state(state, true)
    }

    /// Validate newly computed coordinates once, before they become an iterate.
    /// Subsequent scaling and boundary calculations may trust this interior.
    pub(crate) fn accept_state(
        &self,
        xy: Vec<f64>,
        primal_len: usize,
        s: Vec<f64>,
        tau: f64,
    ) -> Result<State, EmbeddingError> {
        if !tau.is_finite() || tau <= 0.0 {
            return Err(EmbeddingError::NonPositiveTau);
        }
        debug_assert_eq!(primal_len, self.problem.variable_count());
        let (x, y) = xy.split_at(primal_len);
        if !x.iter().all(|value| value.is_finite()) {
            return Err(EmbeddingError::NonFiniteState);
        }
        if !self.geometry.is_primal_interior(&s)? || !self.geometry.is_dual_interior(y)? {
            return Err(EmbeddingError::NonInteriorPoint);
        }
        Ok(State {
            xy,
            primal_len,
            s,
            tau,
        })
    }

    /// Return the explicit lifted initial state `(0, mu e, e, 1)`.
    #[cfg(test)]
    pub(crate) fn canonical_initial_state(&self, mu: f64) -> Result<State, EmbeddingError> {
        if !mu.is_finite() || mu <= 0.0 {
            return Err(EmbeddingError::InvalidMu);
        }
        let mut identity = vec![0.0; self.problem.constraint_count()];
        self.geometry.canonical_identity(&mut identity)?;
        let slack = identity.iter().map(|entry| mu * entry).collect();
        let state = State::new(
            vec![0.0; self.problem.variable_count()],
            slack,
            identity,
            1.0,
        )?;
        self.validate_state(&state, true)?;
        Ok(state)
    }

    fn validate_state(&self, state: &State, require_interior: bool) -> Result<(), EmbeddingError> {
        if state.x().len() != self.problem.variable_count() {
            return Err(EmbeddingError::IncorrectXLength {
                expected: self.problem.variable_count(),
                actual: state.x().len(),
            });
        }
        if state.s.len() != self.problem.constraint_count() {
            return Err(EmbeddingError::IncorrectSLength {
                expected: self.problem.constraint_count(),
                actual: state.s.len(),
            });
        }
        if state.y().len() != self.problem.constraint_count() {
            return Err(EmbeddingError::IncorrectYLength {
                expected: self.problem.constraint_count(),
                actual: state.y().len(),
            });
        }
        if require_interior && !self.geometry.is_primal_interior(&state.s)? {
            return Err(EmbeddingError::NonInteriorPoint);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cone;

    #[test]
    fn sphere_projection_preserves_the_projective_solution_and_cone_geometry() {
        let problem = test_problem();
        let embedding = Embedding::new(&problem).unwrap();
        let original = State::new(vec![3.0, -4.0], vec![0.0, 7.0], vec![-2.0, 5.0], 0.5).unwrap();
        let mut state = original.clone();
        state.normalize_sphere(embedding.nu()).unwrap();
        let norm_squared =
            dot(state.x(), state.x()) + dot(state.y(), state.y()) + state.tau().powi(2);
        assert!((norm_squared - (embedding.nu() + 1) as f64).abs() < 1e-14);
        for (before, after) in original
            .x()
            .iter()
            .chain(original.y())
            .chain(original.s())
            .zip(state.x().iter().chain(state.y()).chain(state.s()))
        {
            assert!((before / original.tau() - after / state.tau()).abs() < 1e-13);
        }
        assert!((state.s()[1] / state.y()[1] - original.s()[1] / original.y()[1]).abs() < 1e-14);
        for factor in [1e-6, 1.0, 1e100] {
            let scaled = |values: &[f64]| values.iter().map(|v| factor * v).collect();
            let mut rescaled = State::new(
                scaled(original.x()),
                scaled(original.s()),
                scaled(original.y()),
                factor * original.tau(),
            )
            .unwrap();
            rescaled.normalize_sphere(embedding.nu()).unwrap();
            for (left, right) in rescaled
                .x()
                .iter()
                .chain(rescaled.y())
                .chain(rescaled.s())
                .chain(std::iter::once(&rescaled.tau()))
                .zip(
                    state
                        .x()
                        .iter()
                        .chain(state.y())
                        .chain(state.s())
                        .chain(std::iter::once(&state.tau())),
                )
            {
                assert!((left - right).abs() < 1e-14);
            }
        }
        let old_gap = embedding.duality_measure(&original) / original.tau().powi(2);
        let new_gap = embedding.duality_measure(&state) / state.tau().powi(2);
        assert!((old_gap - new_gap).abs() < 1e-12);
        embedding.ensure_domain(&state).unwrap();
        assert!(embedding.geometry().is_dual_interior(state.y()).unwrap());
    }

    #[test]
    fn sphere_projection_avoids_squaring_extreme_coordinates() {
        for magnitude in [1e-7, 1e300] {
            let mut state =
                State::new(vec![3.0 * magnitude], vec![], vec![], 4.0 * magnitude).unwrap();
            state.normalize_sphere(0).unwrap();
            assert!((state.x()[0] - 0.6).abs() < 1e-14);
            assert!((state.tau() - 0.8).abs() < 1e-14);
        }
    }

    #[test]
    fn unrepresentable_sphere_projection_leaves_the_state_unchanged() {
        for original in [
            State::new(vec![], vec![1e308], vec![1e-300], 1e-300).unwrap(),
            State::new(vec![1e300], vec![1e-300], vec![1.0], 1.0).unwrap(),
        ] {
            let mut state = original.clone();
            assert!(state.normalize_sphere(1).is_err());
            assert_eq!(state, original);
        }
    }

    #[test]
    fn normalization_uses_the_norm_floor_and_corrects_small_deviations() {
        let mut tiny = State::new(vec![], vec![], vec![], 1e-300).unwrap();
        tiny.normalize_sphere(0).unwrap();
        assert_eq!(tiny.tau(), 1e-300 / 1e-15_f64.sqrt());
        for radius in [0.5, 0.9, 1.0, 1.1, 2.0] {
            let mut state = State::new(vec![], vec![], vec![], radius).unwrap();
            state.normalize_sphere(0).unwrap();
            assert!((state.tau() - 1.0).abs() < 1e-14);
        }
    }

    fn test_problem() -> Problem {
        Problem::new(
            Some(crate::matrix::test_matrix(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap()),
            vec![0.5, -1.0],
            crate::matrix::test_matrix(2, 2, vec![1.0, -2.0, 3.0, 0.25]).unwrap(),
            vec![1.5, -0.75],
            vec![Cone::Zero(1), Cone::Nonnegative(1)],
        )
        .unwrap()
    }

    #[test]
    fn canonical_initial_state_has_unit_duality_measure_per_slack() {
        let problem = test_problem();
        let embedding = Embedding::new(&problem).unwrap();
        let initial = embedding.canonical_initial_state(3.0).unwrap();
        assert_eq!(initial.x(), [0.0, 0.0]);
        assert_eq!(initial.s(), [0.0, 3.0]);
        assert_eq!(initial.y(), [0.0, 1.0]);
        assert_eq!(initial.tau(), 1.0);
        assert_eq!(embedding.nu(), 1);
        assert_eq!(embedding.duality_measure(&initial), 3.0);
    }

    #[test]
    fn invalid_states_and_parameters_are_rejected() {
        let problem = test_problem();
        let embedding = Embedding::new(&problem).unwrap();
        assert_eq!(
            State::new(vec![0.0; 2], vec![0.0; 2], vec![0.0; 2], 0.0),
            Err(EmbeddingError::NonPositiveTau)
        );
        let boundary = State::new(vec![0.0; 2], vec![0.0; 2], vec![0.0; 2], 1.0).unwrap();
        assert_eq!(
            embedding.ensure_domain(&boundary),
            Err(EmbeddingError::NonInteriorPoint)
        );
        assert_eq!(
            embedding.canonical_initial_state(0.0),
            Err(EmbeddingError::InvalidMu)
        );
    }
}
