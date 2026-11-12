//! Cone blocks: the public descriptors and one implementation file per cone.
//!
//! The interior-point method reaches cones only through [`ConeGeometry`], which
//! forwards each operation to the [`ConeKernel`] of every block. To add a cone,
//! implement [`ConeKernel`] in its own file, then register it in this file: a
//! [`Cone`] variant with its name and dimensions, a [`SupportedCone`] variant,
//! and its presolve conversions.

mod geometry;
mod nonnegative;
mod scaling;
mod zero;

use ::presolve as native;

pub(crate) use geometry::{ConeGeometry, ConeGeometryError};
use nonnegative::NonnegativeCone;
pub(crate) use scaling::ConeScaling;
use zero::ZeroCone;

pub(crate) const INTERIOR_FLOOR: f64 = 1e-32;

/// A cone block occupying consecutive rows of `A` and `b`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Cone {
    /// Equality constraints.
    Zero(usize),
    /// The nonnegative orthant of the given dimension.
    Nonnegative(usize),
    /// A second-order cone of the given vector dimension.
    SecondOrder(usize),
    /// A positive-semidefinite cone with matrices of the given order.
    PositiveSemidefinite(usize),
}

impl Cone {
    /// Return the number of rows occupied by this cone block.
    #[must_use]
    pub fn vector_dim(&self) -> usize {
        match *self {
            Self::Zero(dim) | Self::Nonnegative(dim) | Self::SecondOrder(dim) => dim,
            Self::PositiveSemidefinite(order) => order.strict_mul(order.strict_add(1)) / 2,
        }
    }

    pub(crate) fn parameter(&self) -> usize {
        match *self {
            Self::Zero(value)
            | Self::Nonnegative(value)
            | Self::SecondOrder(value)
            | Self::PositiveSemidefinite(value) => value,
        }
    }

    /// Return the stable lowercase name of this cone type.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Zero(_) => "zero",
            Self::Nonnegative(_) => "nonnegative",
            Self::SecondOrder(_) => "second_order",
            Self::PositiveSemidefinite(_) => "positive_semidefinite",
        }
    }

    /// Inverse of [`Cone::name`] and [`Cone::parameter`].
    #[cfg(feature = "python")]
    pub(crate) fn from_name(name: &str, parameter: usize) -> Option<Self> {
        Some(match name {
            "zero" => Self::Zero(parameter),
            "nonnegative" => Self::Nonnegative(parameter),
            "second_order" => Self::SecondOrder(parameter),
            "positive_semidefinite" => Self::PositiveSemidefinite(parameter),
            _ => return None,
        })
    }

    /// How this block is expressed to the native presolver.
    pub(crate) fn presolve_form(&self) -> PresolveForm {
        match *self {
            Self::Zero(_) => PresolveForm::Equality,
            Self::Nonnegative(_) => PresolveForm::UpperBound,
            Self::SecondOrder(dim) => PresolveForm::Conic(native::problem::Cone::SecondOrder(dim)),
            Self::PositiveSemidefinite(order) => {
                PresolveForm::Conic(native::problem::Cone::PositiveSemidefinite { order })
            }
        }
    }

    /// Convert a cone block emitted by the native presolver.
    pub(crate) fn from_presolve(cone: native::problem::Cone) -> Option<Self> {
        Some(match cone {
            native::problem::Cone::Zero(dim) => Self::Zero(dim),
            native::problem::Cone::Nonnegative(dim) => Self::Nonnegative(dim),
            native::problem::Cone::SecondOrder(dim) => Self::SecondOrder(dim),
            native::problem::Cone::PositiveSemidefinite { order } => {
                Self::PositiveSemidefinite(order)
            }
            _ => return None,
        })
    }
}

/// How a cone block crosses the native presolve boundary.
pub(crate) enum PresolveForm {
    /// Each row is the linear equality `(Ax)_i = b_i`.
    Equality,
    /// Each row is the linear inequality `(Ax)_i <= b_i`.
    UpperBound,
    /// The block passes through as one native cone.
    Conic(native::problem::Cone),
}

/// Which member of the primal–dual pair a vector belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConeSide {
    Primal,
    Dual,
}

/// Structure of a block's scaling matrix in the KKT system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalingShape {
    /// Identically zero, as for equality rows.
    Zero,
    /// Diagonal, with strictly positive entries at interior points.
    Diagonal,
}

/// The operations the interior-point method needs from one cone block.
///
/// Every slice argument covers exactly this block's rows; [`ConeGeometry`]
/// checks lengths and splits product-cone vectors before calling these.
/// `F` is the cone's barrier and `F*` the barrier of its dual cone.
trait ConeKernel: std::fmt::Debug {
    /// Contribution to the barrier parameter `nu` of the product cone.
    fn barrier_parameter(&self) -> usize;

    /// Structure of this block's scaling in the KKT system.
    fn scaling_shape(&self) -> ScalingShape;

    /// Whether `s` lies in the relative interior of the primal cone.
    fn is_primal_interior(&self, s: &[f64]) -> bool;

    /// Whether `y` lies in the interior of the dual cone.
    fn is_dual_interior(&self, y: &[f64]) -> bool;

    /// The unit point `e` of the canonical start: `-∇F(e) = e`, and
    /// `<e, e>` equals the barrier parameter.
    fn unit_point(&self, out: &mut [f64]);

    /// The scaling used by the CVXOPT-style start: the identity on cone rows.
    fn identity_scaling(&self, out: &mut [f64]);

    /// The largest `t` with `v - t e` in the closed cone, or `+∞` when the
    /// cone imposes no margin.
    fn margin(&self, v: &[f64]) -> f64;

    /// Shift `v` by `alpha e` towards the interior of the cone on `side`. A
    /// cone without an interior projects onto itself instead.
    fn unit_shift(&self, v: &mut [f64], alpha: f64, side: ConeSide);

    /// Write the slack for a given `Ax`: `b - Ax` projected onto the primal cone.
    fn reconstruct_slack(&self, b: &[f64], ax: &[f64], out: &mut [f64]);

    /// Project `y` onto the dual cone.
    fn project_dual(&self, y: &mut [f64]);

    /// Move `s` and `y` at least `floor` inside the primal and dual cones.
    fn floor_interior(&self, s: &mut [f64], y: &mut [f64], floor: f64);

    /// Length of this block's scaling data beyond the diagonal of `H`.
    fn scaling_len(&self) -> usize {
        0
    }

    /// Compute the scaling `H`, with `H y = s` and `H s^{-1} = y^{-1}`, at an
    /// interior pair: write the diagonal of `H` and the block's further data.
    /// Blocks without a barrier have `H = 0`.
    fn update_scaling(&self, s: &[f64], y: &[f64], diagonal: &mut [f64], data: &mut [f64]);

    /// Write `H v`. The default applies a diagonal `H`.
    fn mul_scaling(&self, scaling: BlockScaling<'_>, v: &[f64], out: &mut [f64]) {
        for ((out, h), v) in out.iter_mut().zip(scaling.diagonal).zip(v) {
            *out = h * v;
        }
    }

    /// Write the centering target `mu y^{-1}` of linearized complementarity.
    fn centering_target(&self, y: &[f64], mu: f64, out: &mut [f64]);

    /// Write the Mehrotra second-order correction, in slack coordinates, for
    /// the affine displacements `ds` and `dy`. `out` starts at `-0.0`, the
    /// exact additive identity; blocks without complementarity leave it.
    fn correction(
        &self,
        scaling: BlockScaling<'_>,
        y: &[f64],
        ds: &[f64],
        dy: &[f64],
        out: &mut [f64],
    );

    /// The largest step in `[0, 1]` from the interior point `v` along `dv`
    /// that stays in the cone. Non-finite displacements are an error.
    fn step_length(&self, v: &[f64], dv: &[f64]) -> Result<f64, ConeGeometryError>;

    /// Complete the warm-start dual residual `r <- r + mu ∇F*(y)` and add
    /// `r' (mu (I + ∇²F*(y)))^{-1} r` to `sum`.
    fn accumulate_local_score(&self, y: &[f64], mu: f64, residual: &mut [f64], sum: &mut f64);
}

/// One block's view of a [`ConeScaling`]: the diagonal of its `H` and the
/// further data written by [`ConeKernel::update_scaling`].
#[derive(Clone, Copy)]
struct BlockScaling<'a> {
    diagonal: &'a [f64],
    #[expect(
        dead_code,
        reason = "read by cones with a non-diagonal scaling, such as the second-order cone"
    )]
    data: &'a [f64],
}

/// The implemented cones, one variant per implementation file.
#[derive(Clone, Debug, PartialEq, Eq)]
enum SupportedCone {
    Zero(ZeroCone),
    Nonnegative(NonnegativeCone),
}

impl SupportedCone {
    /// The implementation for a public cone descriptor, if one exists.
    fn new(cone: &Cone) -> Option<Self> {
        match *cone {
            Cone::Zero(_) => Some(Self::Zero(ZeroCone)),
            Cone::Nonnegative(dim) => Some(Self::Nonnegative(NonnegativeCone::new(dim))),
            Cone::SecondOrder(_) | Cone::PositiveSemidefinite(_) => None,
        }
    }

    fn kernel(&self) -> &dyn ConeKernel {
        match self {
            Self::Zero(cone) => cone,
            Self::Nonnegative(cone) => cone,
        }
    }
}

#[cfg(test)]
pub(crate) fn test_geometry(cones: Vec<Cone>) -> ConeGeometry {
    let dim = cones.iter().map(Cone::vector_dim).sum();
    let problem = crate::Problem::new(
        None,
        vec![],
        crate::CscMatrix::zeros(dim, 0).unwrap(),
        vec![0.0; dim],
        cones,
    )
    .unwrap();
    ConeGeometry::new(&problem).unwrap()
}
