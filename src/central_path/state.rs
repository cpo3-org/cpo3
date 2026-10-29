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
        Self::from_parts(xy, x_len, s, tau)
    }

    /// Adopt an accepted KKT point and floor its cone components in place.
    pub(crate) fn from_stacked(
        xy: Vec<f64>,
        x_len: usize,
        s: Vec<f64>,
        tau: f64,
        geometry: &ConeGeometry,
    ) -> Result<Self, StateError> {
        let mut state = Self::from_parts(xy, x_len, s, tau)?;
        geometry.floor_interior(&mut state.s, &mut state.xy[x_len..]);
        Ok(state)
    }

    fn from_parts(xy: Vec<f64>, x_len: usize, s: Vec<f64>, tau: f64) -> Result<Self, StateError> {
        if !tau.is_finite() || tau <= 0.0 {
            return Err(StateError::NonPositiveTau);
        }
        if !xy.iter().chain(&s).all(|value| value.is_finite()) {
            return Err(StateError::NonFiniteState);
        }
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
    use crate::{Cone, CscMatrix};
    use approx::assert_relative_eq;
    use rstest::rstest;

    fn problem(cones: Vec<Cone>, rows: usize) -> Problem {
        Problem::new(
            None,
            vec![0.; 2],
            CscMatrix::zeros(rows, 2).unwrap(),
            vec![0.; rows],
            cones,
        )
        .unwrap()
    }

    fn mixed_problem() -> Problem {
        problem(
            vec![Cone::Nonnegative(1), Cone::Zero(1), Cone::Nonnegative(1)],
            3,
        )
    }

    #[rstest]
    #[case::ordinary(1.)]
    #[case::small(1e-6)]
    #[case::large(1e150)]
    #[case::overflowing_norm(1.5e308)]
    fn normalization_preserves_homogeneous_coordinates_and_scales_slack(#[case] magnitude: f64) {
        let problem = mixed_problem();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let mut state = State::new(
            vec![magnitude, -0.5 * magnitude],
            vec![0.5 * magnitude, 0., magnitude],
            vec![0.25 * magnitude, -0.75 * magnitude, 0.5 * magnitude],
            0.5 * magnitude,
        )
        .unwrap();
        state.normalize_sphere(&geometry).unwrap();
        state.validate(&problem, &geometry).unwrap();
        let expected_scale = (3. / 2.375_f64).sqrt();
        let expected = [1., -0.5, 0.25, -0.75, 0.5];
        for (value, base) in state.xy.iter().zip(expected) {
            assert_relative_eq!(*value, base * expected_scale, epsilon = 1e-14);
            assert_relative_eq!(*value / state.tau(), 2. * base, epsilon = 1e-14);
        }
        for (value, base) in state.s().iter().zip([0.5, 0., 1.]) {
            assert_relative_eq!(*value, base * expected_scale, epsilon = 1e-14);
            assert_relative_eq!(*value / state.tau(), 2. * base, epsilon = 1e-14);
        }
        let squared_norm = state.xy.iter().map(|v| v * v).sum::<f64>() + state.tau().powi(2);
        assert_relative_eq!(squared_norm, 3., epsilon = 1e-14);
        assert_relative_eq!(
            state.duality_measure(&geometry),
            0.3125 * expected_scale.powi(2),
            epsilon = 1e-14
        );
        let normalized = state.clone();
        state.normalize_sphere(&geometry).unwrap();
        assert_relative_eq!(
            state.xy.as_slice(),
            normalized.xy.as_slice(),
            epsilon = 1e-14
        );
        assert_relative_eq!(state.s(), normalized.s(), epsilon = 1e-14);
        assert_relative_eq!(state.tau(), normalized.tau(), epsilon = 1e-14);
    }

    #[test]
    fn equality_only_normalization_uses_unit_sphere_and_unrestricted_multipliers() {
        let problem = problem(vec![Cone::Zero(2)], 2);
        let geometry = ConeGeometry::new(&problem).unwrap();
        let mut state = State::new(vec![2., -1.], vec![0., 0.], vec![-3., 4.], 2.).unwrap();
        state.normalize_sphere(&geometry).unwrap();
        let norm = 34_f64.sqrt();
        assert_relative_eq!(state.x(), &[2. / norm, -1. / norm][..], epsilon = 1e-14);
        assert_relative_eq!(state.y(), &[-3. / norm, 4. / norm][..], epsilon = 1e-14);
        assert_relative_eq!(state.tau(), 2. / norm, epsilon = 1e-14);
        assert_eq!(state.s(), &[0., 0.]);
        assert_eq!(state.duality_measure(&geometry), 0.);
        state.validate(&problem, &geometry).unwrap();
    }

    #[test]
    fn normalization_floors_only_constrained_coordinates_and_tau() {
        let problem = mixed_problem();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let mut state = State::new(
            vec![1., -1e-100],
            vec![1e-100, 0., 1e-100],
            vec![1e-100, -1e-100, 1e-100],
            1e-100,
        )
        .unwrap();
        state.normalize_sphere(&geometry).unwrap();
        let scale = 3_f64.sqrt();
        assert_relative_eq!(state.x()[0], scale, epsilon = 1e-14);
        for value in [state.x()[1], state.y()[1]] {
            assert_relative_eq!(value, -scale * 1e-100, epsilon = 0., max_relative = 1e-14);
        }
        assert_eq!(state.s(), &[INTERIOR_FLOOR, 0., INTERIOR_FLOOR]);
        assert_eq!(state.y()[0], INTERIOR_FLOOR);
        assert_eq!(state.y()[2], INTERIOR_FLOOR);
        assert_eq!(state.tau(), INTERIOR_FLOOR);
        state.validate(&problem, &geometry).unwrap();
    }

    #[test]
    fn tiny_norm_guard_limits_scaling_before_applying_interior_floors() {
        let problem = mixed_problem();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let mut state = State::new(
            vec![1e-100, 0.],
            vec![1e-100, 0., 1e-100],
            vec![1e-100, -1e-100, 1e-100],
            1e-100,
        )
        .unwrap();
        state.normalize_sphere(&geometry).unwrap();
        // Below the norm guard, the method deliberately does not scale all
        // the way to the sphere; unrestricted coordinates retain their sign.
        let expected = 1e-100 * (3e15_f64).sqrt();
        assert_relative_eq!(state.x()[0], expected, epsilon = 0., max_relative = 1e-14);
        assert_relative_eq!(state.y()[1], -expected, epsilon = 0., max_relative = 1e-14);
        assert_eq!(state.tau(), INTERIOR_FLOOR);
        state.validate(&problem, &geometry).unwrap();
    }

    #[test]
    fn failed_normalization_does_not_modify_the_state() {
        let problem = mixed_problem();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let mut state = State::new(
            vec![1e-8, 0.],
            vec![f64::MAX, 0., 1.],
            vec![1e-8, -1e-8, 1e-8],
            1e-8,
        )
        .unwrap();
        let before = state.clone();
        // The norm itself is safe, but the common scale would overflow s.
        assert_eq!(
            state.normalize_sphere(&geometry),
            Err(StateError::NonFiniteResult)
        );
        assert_eq!(state, before);
    }
}
