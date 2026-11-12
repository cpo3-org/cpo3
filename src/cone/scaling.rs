//! The scaling `H` of the product cone at one iterate.

use std::ops::Range;

use super::{BlockScaling, ConeGeometry, ConeGeometryError};

/// The scaling `H`, with `H y = s` and `H s^{-1} = y^{-1}`, of every cone
/// block at one interior iterate.
///
/// The diagonal that the KKT matrix stores for `H` is kept for all rows; see
/// [`super::ScalingShape`]. Blocks with a non-diagonal `H` keep further data
/// alongside.
#[derive(Debug)]
pub(crate) struct ConeScaling<'a> {
    geometry: &'a ConeGeometry,
    diagonal: Vec<f64>,
    data: Vec<f64>,
    /// Whether any block stores KKT entries off the diagonal.
    coupled: bool,
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
        Ok(Self::build(geometry, |kernel, rows, diagonal, data| {
            kernel.update_scaling(&s[rows.clone()], &y[rows], diagonal, data);
        }))
    }

    /// The scaling of the CVXOPT-style start: the identity on cone rows and
    /// zero on equality rows.
    pub(super) fn identity(geometry: &'a ConeGeometry) -> Self {
        Self::build(geometry, |kernel, _, diagonal, data| {
            kernel.identity_scaling(diagonal, data);
        })
    }

    fn build(
        geometry: &'a ConeGeometry,
        mut write: impl FnMut(&dyn super::ConeKernel, Range<usize>, &mut [f64], &mut [f64]),
    ) -> Self {
        let mut diagonal = vec![0.0; geometry.dim()];
        let mut data = vec![0.0; geometry.scaling_len()];
        let mut coupled = false;
        for (rows, block_data, kernel) in geometry.scaling_blocks() {
            coupled |= kernel.scaling_shape().is_coupled();
            write(
                kernel,
                rows.clone(),
                &mut diagonal[rows],
                &mut data[block_data],
            );
        }
        Self {
            geometry,
            diagonal,
            data,
            coupled,
        }
    }

    /// A scaling with explicitly given diagonal and block data.
    #[cfg(test)]
    pub(crate) fn from_parts(
        geometry: &'a ConeGeometry,
        diagonal: Vec<f64>,
        data: Vec<f64>,
    ) -> Self {
        assert_eq!(diagonal.len(), geometry.dim());
        assert_eq!(data.len(), geometry.scaling_len());
        let coupled = geometry
            .scaling_layout()
            .any(|(_, shape)| shape.is_coupled());
        Self {
            geometry,
            diagonal,
            data,
            coupled,
        }
    }

    pub(crate) fn geometry(&self) -> &'a ConeGeometry {
        self.geometry
    }

    /// The diagonal `D` that the KKT matrix stores for `H`: the diagonal of
    /// `H`, except on expanded blocks. Zero on equality rows.
    pub(crate) fn kkt_diagonal(&self) -> &[f64] {
        &self.diagonal
    }

    /// Write the KKT entries of every coupled block beyond its diagonal, in
    /// block order, as laid out by [`ConeGeometry::kkt_couplings`].
    pub(crate) fn write_kkt_coupling(
        &self,
        off_diagonal: &mut Vec<f64>,
        expansion_diagonal: &mut Vec<f64>,
    ) {
        off_diagonal.clear();
        expansion_diagonal.clear();
        if !self.coupled {
            return;
        }
        for (rows, block_data, kernel) in self.geometry.scaling_blocks() {
            let (off, expansion) = kernel.scaling_shape().coupling_len(rows.len());
            if off == 0 && expansion == 0 {
                continue;
            }
            let (off_start, expansion_start) = (off_diagonal.len(), expansion_diagonal.len());
            off_diagonal.resize(off_start + off, 0.0);
            expansion_diagonal.resize(expansion_start + expansion, 0.0);
            kernel.write_kkt_coupling(
                self.block(rows, block_data),
                &mut off_diagonal[off_start..],
                &mut expansion_diagonal[expansion_start..],
            );
        }
    }

    /// Visit each row `i` of `-(H + shift I) v` with a bound on the same row
    /// of `|H + shift I| |v|`. Rows of diagonal blocks use `d = D + shift`
    /// directly, so their terms are exactly `-d_i v_i`.
    pub(crate) fn visit_shifted_product(
        &self,
        shift: f64,
        d: &[f64],
        v: &[f64],
        mut visit: impl FnMut(usize, f64, f64),
    ) {
        let diagonal_rows = |rows: Range<usize>, visit: &mut dyn FnMut(usize, f64, f64)| {
            for i in rows {
                let term = -d[i] * v[i];
                visit(i, term, term.abs());
            }
        };
        if !self.coupled {
            diagonal_rows(0..v.len(), &mut visit);
            return;
        }
        let (mut product, mut magnitude) = (Vec::new(), Vec::new());
        for (rows, block_data, kernel) in self.geometry.scaling_blocks() {
            if !kernel.scaling_shape().is_coupled() {
                diagonal_rows(rows, &mut visit);
                continue;
            }
            let block = self.block(rows.clone(), block_data);
            product.resize(rows.len(), 0.0);
            magnitude.resize(rows.len(), 0.0);
            kernel.mul_scaling(block, &v[rows.clone()], &mut product);
            kernel.abs_mul_scaling(block, &v[rows.clone()], &mut magnitude);
            for (offset, i) in rows.enumerate() {
                visit(
                    i,
                    -shift.mul_add(v[i], product[offset]),
                    shift.mul_add(v[i].abs(), magnitude[offset]),
                );
            }
        }
    }

    /// `initial + v' (H + shift I) v`, with `d = D + shift` on diagonal blocks.
    /// Multiplying the nonnegative weight first avoids an intermediate square
    /// overflowing or underflowing when the weighted square is representable.
    pub(crate) fn shifted_quadratic_form(
        &self,
        initial: f64,
        shift: f64,
        d: &[f64],
        v: &[f64],
    ) -> f64 {
        if !self.coupled {
            return v.iter().zip(d).fold(initial, |sum, (&value, &weight)| {
                sum + (weight * value) * value
            });
        }
        let mut sum = initial;
        let mut product = Vec::new();
        for (rows, block_data, kernel) in self.geometry.scaling_blocks() {
            if !kernel.scaling_shape().is_coupled() {
                for i in rows {
                    sum += (d[i] * v[i]) * v[i];
                }
                continue;
            }
            product.resize(rows.len(), 0.0);
            kernel.mul_scaling(
                self.block(rows.clone(), block_data),
                &v[rows.clone()],
                &mut product,
            );
            for (offset, i) in rows.enumerate() {
                sum += shift.mul_add(v[i], product[offset]) * v[i];
            }
        }
        sum
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

    fn block(&self, rows: Range<usize>, data: Range<usize>) -> BlockScaling<'_> {
        BlockScaling {
            diagonal: &self.diagonal[rows],
            data: &self.data[data],
        }
    }
}
