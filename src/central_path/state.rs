//! Homogeneous state and spherical normalization.

use crate::linear_algebra::dot;
use thiserror::Error;

use crate::{
    Problem,
    cone::{ConeGeometry, ConeGeometryError, INTERIOR_FLOOR},
};

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum StateError {
    #[error("homogeneous {variable} has length {actual}, expected {expected}")]
    IncorrectLength {
        variable: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("homogeneous state entries must be finite")]
    NonFiniteState,
    #[error("homogeneous tau must be finite and strictly positive")]
    NonPositiveTau,
    #[error("{variable} is not in the relative interior of its cone")]
    NonInteriorPoint { variable: &'static str },
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
    x_len: usize,
    s: Vec<f64>,
    tau: f64,
}

impl State {
    pub(crate) fn new(x: Vec<f64>, s: Vec<f64>, y: Vec<f64>, tau: f64) -> Result<Self, StateError> {
        let x_len = x.len();
        let mut xy = x;
        xy.extend(y);
        if !tau.is_finite() || tau <= 0.0 {
            return Err(StateError::NonPositiveTau);
        }
        if !xy.iter().chain(&s).all(|value| value.is_finite()) {
            return Err(StateError::NonFiniteState);
        }
        Ok(Self { xy, x_len, s, tau })
    }

    /// Adopt an accepted KKT point and floor its cone components in place.
    pub(crate) fn from_stacked(
        mut xy: Vec<f64>,
        x_len: usize,
        mut s: Vec<f64>,
        tau: f64,
        geometry: &ConeGeometry,
    ) -> Result<Self, StateError> {
        if !tau.is_finite() || tau <= 0.0 {
            return Err(StateError::NonPositiveTau);
        }
        if !xy.iter().chain(&s).all(|value| value.is_finite()) {
            return Err(StateError::NonFiniteState);
        }
        let (_, y) = xy.split_at_mut(x_len);
        geometry.floor_interior(&mut s, y);
        Ok(Self { xy, x_len, s, tau })
    }

    pub(crate) fn x(&self) -> &[f64] {
        &self.xy[..self.x_len]
    }

    pub(crate) fn s(&self) -> &[f64] {
        &self.s
    }

    pub(crate) fn y(&self) -> &[f64] {
        &self.xy[self.x_len..]
    }

    pub(crate) fn tau(&self) -> f64 {
        self.tau
    }

    /// Normalize all homogeneous coordinates using the floored sphere norm.
    /// Validate the scale before mutating so failure leaves the state intact.
    // Paper: eq:iterate_normalisation; scale s along with u = (x, y, tau).
    pub(crate) fn normalize_sphere(&mut self, cg: &ConeGeometry) -> Result<(), StateError> {
        let nu = cg.barrier_parameter();
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
            return Err(StateError::NonFiniteResult);
        }
        if !(self.tau * scale).is_finite()
            || !self
                .xy
                .iter()
                .chain(&self.s)
                .all(|value| (value * scale).is_finite())
        {
            return Err(StateError::NonFiniteResult);
        }
        crate::linear_algebra::scale(&mut self.xy, scale);
        crate::linear_algebra::scale(&mut self.s, scale);
        cg.floor_interior(&mut self.s, &mut self.xy[self.x_len..]);
        self.tau = (self.tau * scale).max(INTERIOR_FLOOR);
        Ok(())
    }

    /// Duality measure `mu = <s, y> / nu`, zero when the cone has no barrier.
    pub(crate) fn duality_measure(&self, geometry: &ConeGeometry) -> f64 {
        let nu = geometry.barrier_parameter();
        if nu == 0 {
            return 0.0;
        }
        dot(&self.s, self.y()) / nu as f64
    }

    pub(crate) fn validate(
        &self,
        problem: &Problem,
        geometry: &ConeGeometry,
    ) -> Result<(), StateError> {
        if self.x().len() != problem.variable_count() {
            return Err(StateError::IncorrectLength {
                variable: "x",
                expected: problem.variable_count(),
                actual: self.x().len(),
            });
        }
        if self.s.len() != problem.constraint_count() {
            return Err(StateError::IncorrectLength {
                variable: "s",
                expected: problem.constraint_count(),
                actual: self.s.len(),
            });
        }
        if self.y().len() != problem.constraint_count() {
            return Err(StateError::IncorrectLength {
                variable: "y",
                expected: problem.constraint_count(),
                actual: self.y().len(),
            });
        }
        if !geometry.is_primal_interior(&self.s)? {
            return Err(StateError::NonInteriorPoint { variable: "s" });
        }
        if !geometry.is_dual_interior(self.y())? {
            return Err(StateError::NonInteriorPoint { variable: "y" });
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
        let geometry = ConeGeometry::new(&problem).unwrap();
        let original = State::new(vec![3.0, -4.0], vec![0.0, 7.0], vec![-2.0, 5.0], 0.5).unwrap();
        let mut state = original.clone();
        state.normalize_sphere(&geometry).unwrap();
        let norm_squared =
            dot(state.x(), state.x()) + dot(state.y(), state.y()) + state.tau().powi(2);
        assert!((norm_squared - (geometry.barrier_parameter() + 1) as f64).abs() < 1e-14);
        for (before, after) in original
            .x()
            .iter()
            .chain(original.y())
            .chain(original.s())
            .zip(state.x().iter().chain(state.y()).chain(state.s()))
        {
            assert!((before / original.tau() - after / state.tau()).abs() < 1e-13);
        }
        let old_gap = original.duality_measure(&geometry) / original.tau().powi(2);
        let new_gap = state.duality_measure(&geometry) / state.tau().powi(2);
        assert!((old_gap - new_gap).abs() < 1e-12);
        state.validate(&problem, &geometry).unwrap();
    }

    #[test]
    fn sphere_projection_handles_extreme_scales_and_the_norm_floor() {
        let geometry = crate::cone::test_geometry(vec![Cone::Zero(2), Cone::Nonnegative(1)]);
        let mut underflow = State::new(
            vec![1e300, 1e-300],
            vec![0.0, 0.0, 1e-300],
            vec![-1e-300, 1e-300, 1e-300],
            1e-300,
        )
        .unwrap();
        underflow.normalize_sphere(&geometry).unwrap();
        assert_eq!(underflow.x()[1], 0.0);
        assert_eq!(underflow.s(), [0.0, 0.0, INTERIOR_FLOOR]);
        assert_eq!(underflow.y(), [0.0, 0.0, INTERIOR_FLOOR]);
        assert_eq!(underflow.tau(), INTERIOR_FLOOR);

        let unconstrained = crate::cone::test_geometry(vec![]);
        let mut tiny = State::new(vec![], vec![], vec![], 1e-300).unwrap();
        tiny.normalize_sphere(&unconstrained).unwrap();
        assert_eq!(tiny.tau(), INTERIOR_FLOOR);
        for magnitude in [1e-7, 1.0, 1e300] {
            let mut state =
                State::new(vec![3.0 * magnitude], vec![], vec![], 4.0 * magnitude).unwrap();
            state.normalize_sphere(&unconstrained).unwrap();
            assert!((state.x()[0] - 0.6).abs() < 1e-14);
            assert!((state.tau() - 0.8).abs() < 1e-14);
        }
    }

    #[test]
    fn overflowing_slack_scaling_leaves_the_state_unchanged() {
        let geometry = crate::cone::test_geometry(vec![Cone::Nonnegative(1)]);
        let original = State::new(vec![], vec![1e308], vec![1e-300], 1e-300).unwrap();
        let mut state = original.clone();
        assert!(state.normalize_sphere(&geometry).is_err());
        assert_eq!(state, original);
    }

    fn test_problem() -> Problem {
        Problem::new(
            Some(crate::linear_algebra::test_matrix(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap()),
            vec![0.5, -1.0],
            crate::linear_algebra::test_matrix(2, 2, vec![1.0, -2.0, 3.0, 0.25]).unwrap(),
            vec![1.5, -0.75],
            vec![Cone::Zero(1), Cone::Nonnegative(1)],
        )
        .unwrap()
    }
}
