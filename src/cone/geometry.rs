use std::ops::Range;

use thiserror::Error;

use super::{ConeKernel, ConeScaling, ConeSide, INTERIOR_FLOOR, ScalingShape, SupportedCone};
use crate::{Problem, linear_algebra::DualCoupling};

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

/// One cone block and the consecutive rows it occupies.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Block {
    range: Range<usize>,
    cone: SupportedCone,
}

/// The product cone: block layout of consecutive primal-cone rows.
///
/// Each method splits its vectors by block and forwards to that block's
/// [`ConeKernel`]. A zero-cone block contributes no barrier parameter; its
/// dual block is unrestricted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConeGeometry {
    blocks: Vec<Block>,
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
            let end = offset.strict_add(cone.vector_dim());
            let Some(implementation) = SupportedCone::new(cone) else {
                return Err(ConeGeometryError::UnsupportedCone {
                    block,
                    name: cone.name(),
                });
            };
            barrier_parameter =
                barrier_parameter.strict_add(implementation.kernel().barrier_parameter());
            blocks.push(Block {
                range: offset..end,
                cone: implementation,
            });
            offset = end;
        }

        Ok(Self {
            blocks,
            dim: offset,
            barrier_parameter,
        })
    }

    fn kernels(&self) -> impl Iterator<Item = (Range<usize>, &dyn ConeKernel)> {
        self.blocks
            .iter()
            .map(|block| (block.range.clone(), block.cone.kernel()))
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn barrier_parameter(&self) -> usize {
        self.barrier_parameter
    }

    /// Row ranges with the structure of their KKT scaling blocks.
    pub(crate) fn scaling_layout(&self) -> impl Iterator<Item = (Range<usize>, ScalingShape)> {
        self.kernels()
            .map(|(range, kernel)| (range, kernel.scaling_shape()))
    }

    /// Test membership in the relative interior of the primal product cone:
    /// every block in the relative interior of its own cone, so equality
    /// rows must be exactly zero.
    pub(crate) fn is_primal_interior(&self, s: &[f64]) -> Result<bool, ConeGeometryError> {
        self.ensure_length("s", s)?;
        Ok(self
            .kernels()
            .all(|(range, kernel)| kernel.is_primal_interior(&s[range])))
    }

    /// Write the unit point `e` of the product cone.
    pub(crate) fn canonical_identity(&self, out: &mut [f64]) -> Result<(), ConeGeometryError> {
        self.ensure_length("out", out)?;
        for (range, kernel) in self.kernels() {
            kernel.unit_point(&mut out[range]);
        }
        Ok(())
    }

    /// The CVXOPT-style start's scaling: the identity on cone rows and zero
    /// on equality rows.
    pub(crate) fn identity_scaling(&self) -> ConeScaling<'_> {
        ConeScaling::identity(self)
    }

    /// The blocks whose KKT entries are not only diagonal, in block order.
    pub(crate) fn kkt_couplings(&self) -> Vec<DualCoupling> {
        self.scaling_layout()
            .filter_map(|(rows, shape)| match shape {
                ScalingShape::Zero | ScalingShape::Diagonal => None,
                ScalingShape::Dense => Some(DualCoupling::Dense(rows)),
                ScalingShape::Expanded { rank } => Some(DualCoupling::Expanded { rows, rank }),
            })
            .collect()
    }

    /// Replace the given blocks of a geometry by test cones with a coupled
    /// scaling of the given shape.
    #[cfg(test)]
    pub(crate) fn with_coupled_test_blocks(
        problem: &Problem,
        shapes: &[(usize, ScalingShape)],
    ) -> Self {
        let mut geometry = Self::new(problem).unwrap();
        for &(block, shape) in shapes {
            let dim = geometry.blocks[block].range.len();
            geometry.blocks[block].cone =
                SupportedCone::Coupled(super::coupled_test::CoupledTestCone::new(dim, shape));
        }
        geometry
    }

    /// Test membership in the interior of the dual product cone: every block
    /// in the interior of its own dual cone, so equality rows only need to be
    /// finite.
    pub(crate) fn is_dual_interior(&self, y: &[f64]) -> Result<bool, ConeGeometryError> {
        self.ensure_length("y", y)?;
        Ok(self
            .kernels()
            .all(|(range, kernel)| kernel.is_dual_interior(&y[range])))
    }

    /// Reconstruct slack from a precomputed `Ax`: project `b - Ax` onto the
    /// primal cone of each block, which zeros equality rows.
    pub(crate) fn reconstruct_slack(&self, b: &[f64], ax: &[f64], out: &mut Vec<f64>) {
        out.resize(self.dim, 0.0);
        for (range, kernel) in self.kernels() {
            kernel.reconstruct_slack(&b[range.clone()], &ax[range.clone()], &mut out[range]);
        }
    }

    /// Project a dual estimate onto the dual product cone.
    pub(crate) fn project_dual(&self, y: &mut [f64]) {
        for (range, kernel) in self.kernels() {
            kernel.project_dual(&mut y[range]);
        }
    }

    /// Keep an accepted iterate strictly interior despite rounding: move cone
    /// components of `s` and `y` at least [`INTERIOR_FLOOR`] inside.
    pub(crate) fn floor_interior(&self, s: &mut [f64], y: &mut [f64]) {
        self.floor_interior_with_margin(s, y, INTERIOR_FLOOR);
    }

    /// Move cone components of `s` and `y` at least `margin` into the interior;
    /// equality rows of `s` are zeroed and those of `y` left unchanged.
    pub(crate) fn floor_interior_with_margin(&self, s: &mut [f64], y: &mut [f64], margin: f64) {
        for (range, kernel) in self.kernels() {
            kernel.floor_interior(&mut s[range.clone()], &mut y[range], margin);
        }
    }

    /// Shift every block of `v` by one common multiple of its unit point so
    /// that the smallest margin reaches `margin`; primal equality rows are
    /// zeroed and dual equality rows left unchanged.
    pub(crate) fn shift_into_interior(&self, v: &mut [f64], side: ConeSide, margin: f64) {
        let minimum = self
            .kernels()
            .map(|(range, kernel)| kernel.margin(&v[range]))
            .fold(f64::INFINITY, f64::min);
        let shift = margin - minimum;
        let alpha = if shift > 0.0 { shift } else { 0.0 };
        for (range, kernel) in self.kernels() {
            kernel.unit_shift(&mut v[range], alpha, side);
        }
    }

    /// Compute the scaling for an already accepted interior state.
    pub(crate) fn scaling(
        &self,
        s: &[f64],
        y: &[f64],
    ) -> Result<ConeScaling<'_>, ConeGeometryError> {
        ConeScaling::new(self, s, y)
    }

    /// Total length of the blocks' scaling data beyond the diagonal.
    pub(super) fn scaling_len(&self) -> usize {
        self.kernels().map(|(_, kernel)| kernel.scaling_len()).sum()
    }

    /// Each block's rows, the range of its scaling data, and its kernel.
    pub(super) fn scaling_blocks(
        &self,
    ) -> impl Iterator<Item = (Range<usize>, Range<usize>, &dyn ConeKernel)> {
        self.kernels().scan(0, |offset, (rows, kernel)| {
            let start = *offset;
            *offset += kernel.scaling_len();
            Some((rows, start..*offset, kernel))
        })
    }

    /// The largest step in `[0, 1]` from an accepted interior base `v` along
    /// `dv` that stays in the cone.
    pub(crate) fn step_length(&self, v: &[f64], dv: &[f64]) -> Result<f64, ConeGeometryError> {
        debug_assert!(self.is_dual_interior(v).unwrap_or(false));
        self.ensure_length("dv", dv)?;
        let mut alpha: f64 = 1.0;
        for (range, kernel) in self.kernels() {
            alpha = alpha.min(kernel.step_length(&v[range.clone()], &dv[range])?);
        }
        Ok(alpha)
    }

    /// Boundary from an accepted interior base toward a new target.
    pub(crate) fn step_to_boundary_toward(
        &self,
        v: &[f64],
        target: &[f64],
    ) -> Result<f64, ConeGeometryError> {
        self.ensure_length("dv", target)?;
        let dv = target
            .iter()
            .zip(v)
            .map(|(target, base)| target - base)
            .collect::<Vec<_>>();
        self.step_length(v, &dv)
    }

    /// Complete the warm-start dual residual with each block's barrier
    /// gradient and accumulate its weighted squared norm into `sum`.
    pub(crate) fn accumulate_local_score(
        &self,
        y: &[f64],
        mu: f64,
        residual: &mut [f64],
        sum: &mut f64,
    ) {
        for (range, kernel) in self.kernels() {
            kernel.accumulate_local_score(&y[range.clone()], mu, &mut residual[range], sum);
        }
    }

    pub(super) fn ensure_length(
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
    use crate::Cone;

    #[test]
    fn scaling_layout_reports_each_block_shape() {
        let geometry =
            crate::cone::test_geometry(vec![Cone::Zero(2), Cone::Nonnegative(3), Cone::Zero(1)]);
        assert_eq!(
            geometry.scaling_layout().collect::<Vec<_>>(),
            [
                (0..2, ScalingShape::Zero),
                (2..5, ScalingShape::Diagonal),
                (5..6, ScalingShape::Zero),
            ]
        );
    }

    #[test]
    fn floor_and_projection_respect_equality_rows() {
        let geometry = crate::cone::test_geometry(vec![Cone::Zero(1), Cone::Nonnegative(2)]);
        let mut s = [3.0, -1.0, 5.0];
        let mut y = [-4.0, 0.0, 2.0];
        geometry.floor_interior_with_margin(&mut s, &mut y, 0.5);
        assert_eq!(s, [0.0, 0.5, 5.0]);
        assert_eq!(y, [-4.0, 0.5, 2.0]);

        let mut y = [-4.0, -1.0, 2.0];
        geometry.project_dual(&mut y);
        assert_eq!(y, [-4.0, 0.0, 2.0]);

        let identity = geometry.identity_scaling();
        assert_eq!(identity.kkt_diagonal(), [0.0, 1.0, 1.0]);
        assert!(geometry.kkt_couplings().is_empty());
    }

    #[test]
    fn local_score_adds_the_barrier_only_on_cone_rows() {
        let geometry = crate::cone::test_geometry(vec![Cone::Zero(1), Cone::Nonnegative(1)]);
        let (y, mu) = ([3.0, 2.0], 0.5);
        let mut residual = [1.0, 1.0];
        let mut sum = 0.0;
        geometry.accumulate_local_score(&y, mu, &mut residual, &mut sum);
        // r_1 = 1 - mu / 2 with weight mu + mu / 4.
        assert_eq!(residual, [1.0, 0.75]);
        assert_eq!(sum, 1.0 / mu + 0.75 * 0.75 / (mu + mu / 4.0));
    }

    #[test]
    fn slack_reconstruction_overwrites_mixed_cone_rows() {
        let geometry =
            crate::cone::test_geometry(vec![Cone::Zero(1), Cone::Nonnegative(2), Cone::Zero(1)]);
        let mut slack = vec![f64::NAN; 4];
        geometry.reconstruct_slack(&[5.0, 3.0, 1.0, -2.0], &[0.0, 1.0, 4.0, 0.0], &mut slack);
        assert_eq!(slack, [0.0, 2.0, 0.0, 0.0]);
        geometry.reconstruct_slack(&[0.0; 4], &[1.0, 2.0, -3.0, 4.0], &mut slack);
        assert_eq!(slack, [0.0, 0.0, 3.0, 0.0]);
    }

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
            geometry.step_to_boundary_toward(&[1.0, 1.0], &[1.0, f64::NAN]),
            Err(ConeGeometryError::NonFiniteInput { argument: "dv" })
        );
    }

    #[test]
    fn scaling_satisfies_both_self_scaled_identities() {
        let geometry = crate::cone::test_geometry(vec![Cone::Zero(1), Cone::Nonnegative(3)]);
        let s = [0.0, 0.5, 2.0, 8.0];
        let y = [-3.0, 4.0, 1.0, 0.25];
        let scaling = geometry.scaling(&s, &y).unwrap();
        assert_eq!(scaling.kkt_diagonal(), [0.0, 0.125, 2.0, 32.0]);
        let mut y_inverse = [f64::NAN; 4];
        scaling.centering_target(&y, 1.0, &mut y_inverse);
        assert_eq!(y_inverse, [0.0, 0.25, 1.0, 4.0]);

        // H y = s and H s^{-1} = y^{-1}, with zero on the equality row.
        let mut product = [f64::NAN; 4];
        scaling.mul(&y, &mut product);
        assert_eq!(product, [-0.0, 0.5, 2.0, 8.0]);
        let s_inverse = [0.0, 2.0, 0.5, 0.125];
        scaling.mul(&s_inverse, &mut product);
        assert_eq!(product, y_inverse);

        // The correction is -(ds dy) / y on cone rows and untouched elsewhere.
        let mut correction = [-0.0; 4];
        scaling.correction(
            &y,
            &[5.0, 1.0, -2.0, 4.0],
            &[7.0, 2.0, 3.0, -1.0],
            &mut correction,
        );
        assert_eq!(correction, [-0.0, -0.5, 6.0, 16.0]);
        assert!(correction[0].is_sign_negative());
    }

    #[test]
    fn step_length_follows_displacements_and_rejects_non_finite_ones() {
        let geometry = crate::cone::test_geometry(vec![Cone::Zero(1), Cone::Nonnegative(2)]);
        let v = [5.0, 1.0, 2.0];
        assert_eq!(
            geometry.step_length(&v, &[-100.0, -4.0, 1.0]).unwrap(),
            0.25
        );
        assert_eq!(geometry.step_length(&v, &[-100.0, 0.0, 1.0]).unwrap(), 1.0);
        assert_eq!(
            geometry.step_length(&v, &[f64::INFINITY, 0.0, 1.0]),
            Err(ConeGeometryError::NonFiniteInput { argument: "dv" })
        );
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
                .step_to_boundary_toward(&v, &[-95.0, 0.5, -6.0, 4.0])
                .unwrap(),
            0.25
        );
        assert_eq!(
            geometry
                .step_to_boundary_toward(&v, &[-95.0, 1.0, 1.0, 2.0])
                .unwrap(),
            1.0
        );
        assert_eq!(geometry.step_to_boundary_toward(&v, &v).unwrap(), 1.0);
    }
}
