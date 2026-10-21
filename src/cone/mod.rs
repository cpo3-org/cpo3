// Scaling helpers are added here as the predictor--corrector step lands.
#[allow(dead_code)]
mod geometry;

pub(crate) use geometry::{ConeGeometry, ConeGeometryError};

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
}
