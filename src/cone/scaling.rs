//! The scaling `H` of the product cone at one iterate.

use super::{BlockScaling, ConeGeometry, ConeGeometryError};

/// The scaling `H`, with `H y = s` and `H s^{-1} = y^{-1}`, of every cone
/// block at one interior iterate.
///
/// The diagonal of `H` is stored for all rows, since the KKT system consumes
/// it directly; blocks with a non-diagonal `H` keep further data alongside.
pub(crate) struct ConeScaling<'a> {
    geometry: &'a ConeGeometry,
    diagonal: Vec<f64>,
    data: Vec<f64>,
}

impl<'a> ConeScaling<'a> {
    /// Compute the scaling at an interior pair `(s, y)`.
    pub(super) fn new(
        geometry: &'a ConeGeometry,
        s: &[f64],
        y: &[f64],
    ) -> Result<Self, ConeGeometryError> {
        geometry.ensure_length("s", s)?;
        geometry.ensure_length("y", y)?;
        debug_assert!(geometry.is_primal_interior(s).unwrap_or(false));
        debug_assert!(geometry.is_dual_interior(y).unwrap_or(false));
        let mut diagonal = vec![0.0; geometry.dim()];
        let mut data = vec![0.0; geometry.scaling_len()];
        for (rows, block_data, kernel) in geometry.scaling_blocks() {
            kernel.update_scaling(
                &s[rows.clone()],
                &y[rows.clone()],
                &mut diagonal[rows],
                &mut data[block_data],
            );
        }
        Ok(Self {
            geometry,
            diagonal,
            data,
        })
    }

    pub(crate) fn geometry(&self) -> &'a ConeGeometry {
        self.geometry
    }

    /// The diagonal of `H`; zero on equality rows.
    pub(crate) fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }

    /// Write `H v`.
    pub(crate) fn mul(&self, v: &[f64], out: &mut [f64]) {
        for (rows, block_data, kernel) in self.geometry.scaling_blocks() {
            kernel.mul_scaling(
                self.block(rows.clone(), block_data),
                &v[rows.clone()],
                &mut out[rows],
            );
        }
    }

    /// Write the centering target `mu y^{-1}`; zero on equality rows.
    pub(crate) fn centering_target(&self, y: &[f64], mu: f64, out: &mut [f64]) {
        for (rows, _, kernel) in self.geometry.scaling_blocks() {
            kernel.centering_target(&y[rows.clone()], mu, &mut out[rows]);
        }
    }

    /// Write the Mehrotra second-order correction for the affine slack and
    /// dual displacements; equality rows of `out` are left unchanged.
    pub(crate) fn correction(&self, y: &[f64], ds: &[f64], dy: &[f64], out: &mut [f64]) {
        for (rows, block_data, kernel) in self.geometry.scaling_blocks() {
            kernel.correction(
                self.block(rows.clone(), block_data),
                &y[rows.clone()],
                &ds[rows.clone()],
                &dy[rows.clone()],
                &mut out[rows],
            );
        }
    }

    fn block(
        &self,
        rows: std::ops::Range<usize>,
        data: std::ops::Range<usize>,
    ) -> BlockScaling<'_> {
        BlockScaling {
            diagonal: &self.diagonal[rows],
            data: &self.data[data],
        }
    }
}
