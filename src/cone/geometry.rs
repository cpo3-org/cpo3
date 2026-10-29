use std::ops::Range;

use faer::{ColMut, ColRef};

use thiserror::Error;

use super::{Cone, INTERIOR_FLOOR};
use crate::Problem;

/// Failures while laying out or evaluating product-cone geometry.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub(crate) enum ConeGeometryError {
    #[error("cone block {block} of type {name} is not implemented by the barrier geometry")]
    UnsupportedCone { block: usize, name: &'static str },
    #[error("{argument} has length {actual}, but cone geometry requires {expected}")]
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{argument} contains a non-finite value")]
    NonFiniteInput { argument: &'static str },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ConeBlock {
    Zero(Range<usize>),
    Nonnegative(Range<usize>),
}

impl ConeBlock {
    pub(crate) fn range(&self) -> Range<usize> {
        match self {
            Self::Zero(range) | Self::Nonnegative(range) => range.clone(),
        }
    }
}

/// Block layout of consecutive primal-cone rows.
///
/// A zero-cone block contributes no barrier parameter; its dual block is
/// unrestricted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConeGeometry {
    blocks: Vec<ConeBlock>,
    dim: usize,
    barrier_parameter: usize,
}

impl ConeGeometry {
    /// Build the supported cone layout from an already validated problem.
    pub(crate) fn new(problem: &Problem) -> Result<Self, ConeGeometryError> {
        let cones = problem.cones();
        let mut blocks = Vec::with_capacity(cones.len());
        let mut offset = 0usize;
        let mut barrier_parameter = 0usize;

        for (block, cone) in cones.iter().enumerate() {
            let dim = cone.vector_dim();
            let end = offset.strict_add(dim);
            let layout = match cone {
                Cone::Zero(_) => ConeBlock::Zero(offset..end),
                Cone::Nonnegative(_) => {
                    barrier_parameter = barrier_parameter.strict_add(dim);
                    ConeBlock::Nonnegative(offset..end)
                }
                unsupported => {
                    return Err(ConeGeometryError::UnsupportedCone {
                        block,
                        name: unsupported.name(),
                    });
                }
            };
            blocks.push(layout);
            offset = end;
        }

        Ok(Self {
            blocks,
            dim: offset,
            barrier_parameter,
        })
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn blocks(&self) -> &[ConeBlock] {
        &self.blocks
    }

    pub(crate) fn barrier_parameter(&self) -> usize {
        self.barrier_parameter
    }

    /// Test membership in the relative interior of the primal product cone.
    ///
    /// Zero-cone coordinates must be exactly zero, while nonnegative-cone
    /// coordinates must be strictly positive.
    pub(crate) fn is_primal_interior(&self, s: &[f64]) -> Result<bool, ConeGeometryError> {
        self.ensure_length("s", s)?;
        Ok(self.blocks.iter().all(|block| match block {
            ConeBlock::Zero(range) => s[range.clone()].iter().all(|entry| *entry == 0.0),
            ConeBlock::Nonnegative(range) => s[range.clone()]
                .iter()
                .all(|entry| entry.is_finite() && *entry > 0.0),
        }))
    }

    pub(crate) fn canonical_identity(&self, out: &mut [f64]) -> Result<(), ConeGeometryError> {
        self.ensure_length("out", out)?;
        out.fill(0.0);
        for block in &self.blocks {
            if let ConeBlock::Nonnegative(range) = block {
                out[range.clone()].fill(1.0);
            }
        }
        Ok(())
    }

    /// Test membership in the interior of the dual product cone.
    ///
    /// Zero-cone coordinates are unrestricted, while nonnegative-cone
    /// coordinates must be strictly positive.
    pub(crate) fn is_dual_interior(&self, y: &[f64]) -> Result<bool, ConeGeometryError> {
        self.ensure_length("y", y)?;
        Ok(self.blocks.iter().all(|block| match block {
            ConeBlock::Zero(range) => y[range.clone()].iter().all(|entry| entry.is_finite()),
            ConeBlock::Nonnegative(range) => y[range.clone()]
                .iter()
                .all(|entry| entry.is_finite() && *entry > 0.0),
        }))
    }

    /// Reconstruct slack from a precomputed `Ax`: zero equality rows and
    /// clamp `b - Ax` to zero on nonnegative-cone rows.
    pub(crate) fn reconstruct_slack(&self, b: &[f64], ax: &[f64], out: &mut Vec<f64>) {
        out.resize(self.dim, 0.0);
        for block in &self.blocks {
            match block {
                ConeBlock::Zero(range) => out[range.clone()].fill(0.0),
                ConeBlock::Nonnegative(range) => {
                    for i in range.clone() {
                        out[i] = (b[i] - ax[i]).max(0.0);
                    }
                }
            }
        }
    }

    /// Floor nonnegative-cone components, leaving equality rows unchanged.
    pub(crate) fn floor_interior(&self, s: &mut [f64], y: &mut [f64]) {
        for block in &self.blocks {
            if let ConeBlock::Nonnegative(range) = block {
                for value in s[range.clone()].iter_mut().chain(&mut y[range.clone()]) {
                    *value = value.max(INTERIOR_FLOOR);
                }
            }
        }
    }

    /// Compute scaling and inverse for an already accepted interior state.
    pub(crate) fn scaling_and_inverse(
        &self,
        s: &[f64],
        y: &[f64],
        h: &mut [f64],
        inverse: &mut [f64],
    ) -> Result<(), ConeGeometryError> {
        debug_assert!(self.is_primal_interior(s).unwrap_or(false));
        debug_assert!(self.is_dual_interior(y).unwrap_or(false));
        self.ensure_length("h", h)?;
        self.ensure_length("inverse", inverse)?;
        for block in &self.blocks {
            match block {
                ConeBlock::Zero(range) => {
                    h[range.clone()].fill(0.0);
                    inverse[range.clone()].fill(0.0);
                }
                ConeBlock::Nonnegative(range) => {
                    faer::zip!(
                        ColMut::from_slice_mut(&mut h[range.clone()]),
                        ColMut::from_slice_mut(&mut inverse[range.clone()]),
                        ColRef::from_slice(&s[range.clone()]),
                        ColRef::from_slice(&y[range.clone()])
                    )
                    .for_each(|faer::unzip!(h, inverse, s, y)| {
                        *h = s / y;
                        *inverse = y.recip();
                    });
                }
            }
        }
        Ok(())
    }

    /// Boundary from an accepted interior base toward a new target.
    pub(crate) fn step_to_boundary_toward(
        &self,
        v: &[f64],
        target: &[f64],
    ) -> Result<f64, ConeGeometryError> {
        let values = target;
        debug_assert!(self.is_dual_interior(v).unwrap_or(false));
        self.ensure_length("dv", values)?;
        let mut alpha: f64 = 1.0;
        for block in &self.blocks {
            match block {
                ConeBlock::Zero(range) => {
                    if !values[range.clone()].iter().all(|value| value.is_finite()) {
                        return Err(ConeGeometryError::NonFiniteInput { argument: "dv" });
                    }
                }
                ConeBlock::Nonnegative(range) => {
                    for (&value, &other) in v[range.clone()].iter().zip(&values[range.clone()]) {
                        if !other.is_finite() {
                            return Err(ConeGeometryError::NonFiniteInput { argument: "dv" });
                        }
                        let direction = other - value;
                        if direction < 0.0 {
                            alpha = alpha.min(-value / direction);
                        }
                    }
                }
            }
        }
        Ok(alpha)
    }

    fn ensure_length(
        &self,
        argument: &'static str,
        vector: &[f64],
    ) -> Result<(), ConeGeometryError> {
        if vector.len() != self.dim {
            return Err(ConeGeometryError::LengthMismatch {
                argument,
                expected: self.dim,
                actual: vector.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CscMatrix;
    use approx::assert_relative_eq;
    use rstest::rstest;

    fn mixed_geometry() -> ConeGeometry {
        let problem = Problem::new(
            None,
            vec![0.],
            CscMatrix::zeros(4, 1).unwrap(),
            vec![0.; 4],
            vec![Cone::Nonnegative(1), Cone::Zero(1), Cone::Nonnegative(2)],
        )
        .unwrap();
        ConeGeometry::new(&problem).unwrap()
    }

    #[rstest]
    #[case::unchanged([2., -3., 4., 2.], 1.)]
    #[case::interior([1., -100., 2., 6.], 1.)]
    #[case::boundary_target([0., -100., 2., 6.], 1.)]
    #[case::first_block([-6., -100., 2., 6.], 0.25)]
    #[case::last_block([1., -100., -4., -6.], 0.25)]
    #[case::tied_boundaries([-6., -100., -12., 0.], 0.25)]
    fn boundary_step_is_the_largest_feasible_fraction(
        #[case] target: [f64; 4],
        #[case] expected: f64,
    ) {
        let geometry = mixed_geometry();
        let base = [2., -3., 4., 2.];
        let alpha = geometry.step_to_boundary_toward(&base, &target).unwrap();
        assert_relative_eq!(alpha, expected, epsilon = 1e-14);
        for i in [0, 2, 3] {
            let at_boundary = (1. - alpha) * base[i] + alpha * target[i];
            assert!(at_boundary >= -1e-14);
            let damped = (1. - 0.99 * alpha) * base[i] + 0.99 * alpha * target[i];
            assert!(damped > 0.);
        }
        if alpha < 1. {
            let beyond = alpha + 1e-6;
            assert!(
                [0, 2, 3]
                    .iter()
                    .any(|&i| (1. - beyond) * base[i] + beyond * target[i] < 0.)
            );
        }
    }

    #[rstest]
    fn boundary_step_is_invariant_under_common_scaling(#[values(1e-300, 1., 1e300)] scale: f64) {
        let geometry = mixed_geometry();
        let base = [2., -3., 4., 2.].map(|v| v * scale);
        let target = [-6., 3., -4., -6.].map(|v| v * scale);
        assert_relative_eq!(
            geometry.step_to_boundary_toward(&base, &target).unwrap(),
            0.25,
            epsilon = 1e-14
        );
    }

    #[test]
    fn equality_rows_do_not_limit_the_step() {
        let problem = Problem::new(
            None,
            vec![0.],
            CscMatrix::zeros(2, 1).unwrap(),
            vec![0.; 2],
            vec![Cone::Zero(2)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        assert_eq!(
            geometry
                .step_to_boundary_toward(&[-1., 2.], &[1e100, -1e100])
                .unwrap(),
            1.
        );
    }

    #[rstest]
    fn boundary_rejects_nonfinite_targets_in_every_cone_block(
        #[values(0, 1, 2, 3)] index: usize,
        #[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY)] invalid: f64,
    ) {
        let geometry = mixed_geometry();
        let base = [2., -3., 4., 2.];
        let mut target = base;
        target[index] = invalid;
        assert_eq!(
            geometry.step_to_boundary_toward(&base, &target),
            Err(ConeGeometryError::NonFiniteInput { argument: "dv" })
        );
    }

    #[rstest]
    fn boundary_rejects_wrong_target_dimensions(#[values(3, 5)] length: usize) {
        let geometry = mixed_geometry();
        assert_eq!(
            geometry.step_to_boundary_toward(&[2., -3., 4., 2.], &vec![1.; length]),
            Err(ConeGeometryError::LengthMismatch {
                argument: "dv",
                expected: 4,
                actual: length
            })
        );
    }
}
