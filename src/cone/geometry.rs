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
        self.boundary_from_interior(v, target, |value, target| target - value)
    }

    /// Boundary from an accepted interior base along a new direction.
    pub(crate) fn step_to_boundary_from_interior(
        &self,
        v: &[f64],
        dv: &[f64],
    ) -> Result<f64, ConeGeometryError> {
        self.boundary_from_interior(v, dv, |_, direction| direction)
    }

    fn boundary_from_interior(
        &self,
        v: &[f64],
        values: &[f64],
        direction: impl Fn(f64, f64) -> f64,
    ) -> Result<f64, ConeGeometryError> {
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
                        let direction = direction(value, other);
                        if direction < 0.0 {
                            alpha = alpha.min(-value / direction);
                        }
                    }
                }
            }
        }
        Ok(alpha)
    }

    /// `-(a o b o c)` on nonnegative-cone rows, `-0.0` on zero-cone rows
    /// (Mehrotra's second-order correction, as negating a zero fill gives).
    pub(crate) fn negated_triple_product_in_place(
        &self,
        a: &[f64],
        b: &mut [f64],
        c: &[f64],
    ) -> Result<(), ConeGeometryError> {
        self.ensure_length("a", a)?;
        self.ensure_length("b", b)?;
        self.ensure_length("c", c)?;
        for block in &self.blocks {
            match block {
                ConeBlock::Zero(range) => b[range.clone()].fill(-0.0),
                ConeBlock::Nonnegative(range) => {
                    faer::zip!(
                        ColMut::from_slice_mut(&mut b[range.clone()]),
                        ColRef::from_slice(&a[range.clone()]),
                        ColRef::from_slice(&c[range.clone()])
                    )
                    .for_each(|faer::unzip!(b, a, c)| *b = -(a * (*b * c)));
                }
            }
        }
        Ok(())
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

    #[test]
    fn lays_out_mixed_blocks_and_builds_canonical_point() {
        let geometry = crate::cone::test_geometry(vec![
            Cone::Zero(2),
            Cone::Nonnegative(3),
            Cone::Zero(1),
            Cone::Nonnegative(1),
        ]);
        assert_eq!(geometry.dim(), 7);
        assert_eq!(geometry.barrier_parameter(), 4);

        let mut point = vec![f64::NAN; 7];
        geometry.canonical_identity(&mut point).unwrap();
        assert_eq!(point, vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]);
        assert!(
            geometry
                .is_primal_interior(&[0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 4.0])
                .unwrap()
        );
        assert!(
            !geometry
                .is_primal_interior(&[1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 4.0])
                .unwrap()
        );
    }

    #[test]
    fn reports_unsupported_cones_and_invalid_vectors() {
        let problem = Problem::new(
            None,
            vec![],
            crate::CscMatrix::zeros(3, 0).unwrap(),
            vec![0.0; 3],
            vec![Cone::SecondOrder(3)],
        )
        .unwrap();
        assert_eq!(
            ConeGeometry::new(&problem),
            Err(ConeGeometryError::UnsupportedCone {
                block: 0,
                name: "second_order"
            })
        );

        let geometry = crate::cone::test_geometry(vec![Cone::Nonnegative(2)]);
        assert_eq!(
            geometry.is_primal_interior(&[1.0]),
            Err(ConeGeometryError::LengthMismatch {
                argument: "s",
                expected: 2,
                actual: 1
            })
        );
        assert!(!geometry.is_primal_interior(&[1.0, 0.0]).unwrap());
        assert!(!geometry.is_dual_interior(&[1.0, 0.0]).unwrap());
        assert_eq!(
            geometry.step_to_boundary_from_interior(&[1.0, 1.0], &[0.0, f64::NAN]),
            Err(ConeGeometryError::NonFiniteInput { argument: "dv" })
        );
    }

    #[test]
    fn scaling_and_inverse_satisfy_both_self_scaled_identities() {
        let geometry = crate::cone::test_geometry(vec![Cone::Zero(1), Cone::Nonnegative(3)]);
        let s = [0.0, 0.5, 2.0, 8.0];
        let y = [-3.0, 4.0, 1.0, 0.25];
        let mut h = [f64::NAN; 4];
        let mut y_inverse = [f64::NAN; 4];
        geometry
            .scaling_and_inverse(&s, &y, &mut h, &mut y_inverse)
            .unwrap();
        assert_eq!(h, [0.0, 0.125, 2.0, 32.0]);
        assert_eq!(y_inverse, [0.0, 0.25, 1.0, 4.0]);

        let mut s_inverse = [f64::NAN; 4];
        s_inverse[0] = 0.0;
        for index in 1..4 {
            s_inverse[index] = s[index].recip();
        }
        for index in 0..4 {
            assert_eq!(h[index] * y[index], s[index]);
            assert_eq!(h[index] * s_inverse[index], y_inverse[index]);
        }
    }

    #[test]
    fn dual_interior_ignores_zero_rows() {
        let geometry = crate::cone::test_geometry(vec![Cone::Zero(1), Cone::Nonnegative(2)]);
        assert!(geometry.is_dual_interior(&[-7.0, 1.0, 2.0]).unwrap());
        assert!(!geometry.is_dual_interior(&[0.0, 1.0, 0.0]).unwrap());
        assert!(!geometry.is_dual_interior(&[0.0, f64::NAN, 1.0]).unwrap());
    }

    #[test]
    fn step_to_boundary_is_the_smallest_positive_ratio_capped_at_one() {
        let geometry = crate::cone::test_geometry(vec![Cone::Zero(1), Cone::Nonnegative(3)]);
        let v = [5.0, 1.0, 2.0, 3.0];
        assert_eq!(
            geometry
                .step_to_boundary_from_interior(&v, &[-100.0, -0.5, -8.0, 1.0])
                .unwrap(),
            0.25
        );
        assert_eq!(
            geometry
                .step_to_boundary_from_interior(&v, &[-100.0, 0.0, -1.0, -1.0])
                .unwrap(),
            1.0
        );
        assert_eq!(
            geometry
                .step_to_boundary_from_interior(&v, &[0.0; 4])
                .unwrap(),
            1.0
        );
    }
}
