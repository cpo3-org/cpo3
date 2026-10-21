use std::ops::Range;

use crate::linear_algebra::binary_into;
use faer::{ColMut, ColRef};

use thiserror::Error;

use super::Cone;

/// Failures while laying out or evaluating product-cone geometry.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub(crate) enum ConeGeometryError {
    #[error("cone block {block} has zero dimension")]
    EmptyBlock { block: usize },
    #[error("cone dimensions overflow usize")]
    DimensionOverflow,
    #[error("cone dimensions sum to {actual}, but the vector dimension is {expected}")]
    DimensionMismatch { expected: usize, actual: usize },
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
    #[error("{argument} is outside the cone relative interior in block {block}")]
    OutsideInterior {
        argument: &'static str,
        block: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind {
    Zero,
    Nonnegative,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ConeBlock {
    kind: BlockKind,
    range: Range<usize>,
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
    pub(crate) fn new(cones: &[Cone], expected_dim: usize) -> Result<Self, ConeGeometryError> {
        let mut blocks = Vec::with_capacity(cones.len());
        let mut offset = 0usize;
        let mut barrier_parameter = 0usize;

        for (block, cone) in cones.iter().enumerate() {
            let dim = cone.vector_dim();
            if dim == 0 {
                return Err(ConeGeometryError::EmptyBlock { block });
            }
            let end = offset
                .checked_add(dim)
                .ok_or(ConeGeometryError::DimensionOverflow)?;
            let kind = match cone {
                Cone::Zero(_) => BlockKind::Zero,
                Cone::Nonnegative(_) => {
                    barrier_parameter = barrier_parameter
                        .checked_add(dim)
                        .ok_or(ConeGeometryError::DimensionOverflow)?;
                    BlockKind::Nonnegative
                }
                unsupported => {
                    return Err(ConeGeometryError::UnsupportedCone {
                        block,
                        name: unsupported.name(),
                    });
                }
            };
            blocks.push(ConeBlock {
                kind,
                range: offset..end,
            });
            offset = end;
        }

        if offset != expected_dim {
            return Err(ConeGeometryError::DimensionMismatch {
                expected: expected_dim,
                actual: offset,
            });
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

    pub(crate) fn barrier_parameter(&self) -> usize {
        self.barrier_parameter
    }

    /// Test membership in the relative interior of the primal product cone.
    ///
    /// Zero-cone coordinates must be exactly zero, while nonnegative-cone
    /// coordinates must be strictly positive.
    pub(crate) fn is_primal_interior(&self, s: &[f64]) -> Result<bool, ConeGeometryError> {
        self.ensure_length("s", s)?;
        Ok(self.blocks.iter().all(|block| match block.kind {
            BlockKind::Zero => s[block.range.clone()].iter().all(|entry| *entry == 0.0),
            BlockKind::Nonnegative => s[block.range.clone()]
                .iter()
                .all(|entry| entry.is_finite() && *entry > 0.0),
        }))
    }

    pub(crate) fn canonical_identity(&self, out: &mut [f64]) -> Result<(), ConeGeometryError> {
        self.ensure_output("out", out)?;
        out.fill(0.0);
        for block in &self.blocks {
            if block.kind == BlockKind::Nonnegative {
                out[block.range.clone()].fill(1.0);
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
        Ok(self.blocks.iter().all(|block| match block.kind {
            BlockKind::Zero => y[block.range.clone()].iter().all(|entry| entry.is_finite()),
            BlockKind::Nonnegative => y[block.range.clone()]
                .iter()
                .all(|entry| entry.is_finite() && *entry > 0.0),
        }))
    }

    /// Nesterov--Todd scaling `H` with `H y = s`, stored as a diagonal.
    ///
    /// Zero-cone rows carry no scaling and return zero.
    pub(crate) fn nt_scaling(
        &self,
        s: &[f64],
        y: &[f64],
        out: &mut [f64],
    ) -> Result<(), ConeGeometryError> {
        self.ensure_primal_point(s)?;
        self.ensure_dual_point(y)?;
        self.ensure_output("out", out)?;
        out.fill(0.0);
        for block in &self.blocks {
            if block.kind == BlockKind::Nonnegative {
                binary_into(
                    &mut out[block.range.clone()],
                    &s[block.range.clone()],
                    &y[block.range.clone()],
                    |s, y| s / y,
                );
            }
        }
        Ok(())
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
        self.ensure_output("h", h)?;
        self.ensure_output("inverse", inverse)?;
        for block in &self.blocks {
            match block.kind {
                BlockKind::Zero => {
                    h[block.range.clone()].fill(0.0);
                    inverse[block.range.clone()].fill(0.0);
                }
                BlockKind::Nonnegative => {
                    let range = block.range.clone();
                    faer::zip!(
                        ColMut::from_slice_mut(&mut h[range.clone()]),
                        ColMut::from_slice_mut(&mut inverse[range.clone()]),
                        ColRef::from_slice(&s[range.clone()]),
                        ColRef::from_slice(&y[range])
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

    /// Jordan inverse `y^{-1}`, zero on zero-cone rows.
    pub(crate) fn jordan_inverse(
        &self,
        y: &[f64],
        out: &mut [f64],
    ) -> Result<(), ConeGeometryError> {
        self.ensure_dual_point(y)?;
        self.ensure_output("out", out)?;
        out.fill(0.0);
        for block in &self.blocks {
            if block.kind == BlockKind::Nonnegative {
                faer::zip!(
                    ColMut::from_slice_mut(&mut out[block.range.clone()]),
                    ColRef::from_slice(&y[block.range.clone()])
                )
                .for_each(|faer::unzip!(out, y)| *out = y.recip());
            }
        }
        Ok(())
    }

    /// Jordan product `a o b`, zero on zero-cone rows.
    pub(crate) fn jordan_product(
        &self,
        a: &[f64],
        b: &[f64],
        out: &mut [f64],
    ) -> Result<(), ConeGeometryError> {
        self.ensure_finite_vector("a", a)?;
        self.ensure_finite_vector("b", b)?;
        self.ensure_output("out", out)?;
        out.fill(0.0);
        for block in &self.blocks {
            if block.kind == BlockKind::Nonnegative {
                binary_into(
                    &mut out[block.range.clone()],
                    &a[block.range.clone()],
                    &b[block.range.clone()],
                    |a, b| a * b,
                );
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
            match block.kind {
                BlockKind::Zero => {
                    if !values[block.range.clone()]
                        .iter()
                        .all(|value| value.is_finite())
                    {
                        return Err(ConeGeometryError::NonFiniteInput { argument: "dv" });
                    }
                }
                BlockKind::Nonnegative => {
                    for (&value, &other) in v[block.range.clone()]
                        .iter()
                        .zip(&values[block.range.clone()])
                    {
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
            let range = block.range.clone();
            match block.kind {
                BlockKind::Zero => b[range].fill(-0.0),
                BlockKind::Nonnegative => {
                    faer::zip!(
                        ColMut::from_slice_mut(&mut b[range.clone()]),
                        ColRef::from_slice(&a[range.clone()]),
                        ColRef::from_slice(&c[range])
                    )
                    .for_each(|faer::unzip!(b, a, c)| *b = -(a * (*b * c)));
                }
            }
        }
        Ok(())
    }

    /// Largest `alpha` in `(0, 1]` keeping `v + alpha dv` inside the cone
    /// interior on nonnegative-cone rows; zero-cone rows are ignored.
    #[cfg(test)]
    pub(crate) fn step_to_boundary(&self, v: &[f64], dv: &[f64]) -> Result<f64, ConeGeometryError> {
        self.ensure_finite_vector("v", v)?;
        for block in &self.blocks {
            if block.kind == BlockKind::Nonnegative
                && v[block.range.clone()].iter().any(|value| *value <= 0.0)
            {
                return Err(ConeGeometryError::NonFiniteInput { argument: "v" });
            }
        }
        self.step_to_boundary_from_interior(v, dv)
    }

    fn ensure_primal_point(&self, s: &[f64]) -> Result<(), ConeGeometryError> {
        self.ensure_finite_vector("s", s)?;
        for (index, block) in self.blocks.iter().enumerate() {
            let inside = match block.kind {
                BlockKind::Zero => s[block.range.clone()].iter().all(|entry| *entry == 0.0),
                BlockKind::Nonnegative => s[block.range.clone()].iter().all(|entry| *entry > 0.0),
            };
            if !inside {
                return Err(ConeGeometryError::OutsideInterior {
                    argument: "s",
                    block: index,
                });
            }
        }
        Ok(())
    }

    fn ensure_dual_point(&self, y: &[f64]) -> Result<(), ConeGeometryError> {
        self.ensure_finite_vector("y", y)?;
        for (index, block) in self.blocks.iter().enumerate() {
            if block.kind == BlockKind::Nonnegative
                && y[block.range.clone()].iter().any(|entry| *entry <= 0.0)
            {
                return Err(ConeGeometryError::OutsideInterior {
                    argument: "y",
                    block: index,
                });
            }
        }
        Ok(())
    }

    fn ensure_finite_vector(
        &self,
        argument: &'static str,
        vector: &[f64],
    ) -> Result<(), ConeGeometryError> {
        self.ensure_length(argument, vector)?;
        if !vector.iter().all(|entry| entry.is_finite()) {
            return Err(ConeGeometryError::NonFiniteInput { argument });
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

    fn ensure_output(
        &self,
        argument: &'static str,
        vector: &mut [f64],
    ) -> Result<(), ConeGeometryError> {
        self.ensure_length(argument, vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lays_out_mixed_blocks_and_builds_canonical_point() {
        let geometry = ConeGeometry::new(
            &[
                Cone::Zero(2),
                Cone::Nonnegative(3),
                Cone::Zero(1),
                Cone::Nonnegative(1),
            ],
            7,
        )
        .unwrap();
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
    fn reports_layout_domain_and_extension_errors() {
        assert_eq!(
            ConeGeometry::new(&[Cone::Nonnegative(2)], 3),
            Err(ConeGeometryError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            ConeGeometry::new(&[Cone::SecondOrder(3)], 3),
            Err(ConeGeometryError::UnsupportedCone {
                block: 0,
                name: "second_order"
            })
        );

        let geometry = ConeGeometry::new(&[Cone::Nonnegative(2)], 2).unwrap();
        assert_eq!(
            geometry.is_primal_interior(&[1.0]),
            Err(ConeGeometryError::LengthMismatch {
                argument: "s",
                expected: 2,
                actual: 1
            })
        );
        assert!(!geometry.is_primal_interior(&[1.0, 0.0]).unwrap());
        assert_eq!(
            geometry.nt_scaling(&[1.0, 2.0], &[1.0, 0.0], &mut [0.0; 2]),
            Err(ConeGeometryError::OutsideInterior {
                argument: "y",
                block: 0
            })
        );
        assert_eq!(
            geometry.step_to_boundary(&[1.0, f64::NAN], &[0.0, 0.0]),
            Err(ConeGeometryError::NonFiniteInput { argument: "v" })
        );
    }

    #[test]
    fn nt_scaling_satisfies_both_self_scaled_identities() {
        let geometry = ConeGeometry::new(&[Cone::Zero(1), Cone::Nonnegative(3)], 4).unwrap();
        let s = [0.0, 0.5, 2.0, 8.0];
        let y = [-3.0, 4.0, 1.0, 0.25];
        let mut h = [f64::NAN; 4];
        geometry.nt_scaling(&s, &y, &mut h).unwrap();
        assert_eq!(h, [0.0, 0.125, 2.0, 32.0]);

        let mut s_inverse = [f64::NAN; 4];
        let mut y_inverse = [f64::NAN; 4];
        geometry.jordan_inverse(&y, &mut y_inverse).unwrap();
        s_inverse[0] = 0.0;
        for index in 1..4 {
            s_inverse[index] = s[index].recip();
        }
        for index in 0..4 {
            assert_eq!(h[index] * y[index], s[index]);
            assert_eq!(h[index] * s_inverse[index], y_inverse[index]);
        }

        let mut product = [f64::NAN; 4];
        geometry.jordan_product(&s, &y, &mut product).unwrap();
        assert_eq!(product, [0.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn dual_interior_ignores_zero_rows() {
        let geometry = ConeGeometry::new(&[Cone::Zero(1), Cone::Nonnegative(2)], 3).unwrap();
        assert!(geometry.is_dual_interior(&[-7.0, 1.0, 2.0]).unwrap());
        assert!(!geometry.is_dual_interior(&[0.0, 1.0, 0.0]).unwrap());
        assert!(!geometry.is_dual_interior(&[0.0, f64::NAN, 1.0]).unwrap());
    }

    #[test]
    fn step_to_boundary_is_the_smallest_positive_ratio_capped_at_one() {
        let geometry = ConeGeometry::new(&[Cone::Zero(1), Cone::Nonnegative(3)], 4).unwrap();
        let v = [5.0, 1.0, 2.0, 3.0];
        assert_eq!(
            geometry
                .step_to_boundary(&v, &[-100.0, -0.5, -8.0, 1.0])
                .unwrap(),
            0.25
        );
        assert_eq!(
            geometry
                .step_to_boundary(&v, &[-100.0, 0.0, -1.0, -1.0])
                .unwrap(),
            1.0
        );
        assert_eq!(geometry.step_to_boundary(&v, &[0.0; 4]).unwrap(), 1.0);
    }
}
