//! A test cone with an explicitly given coupled scaling, used to exercise
//! dense and expanded KKT blocks independently of any real cone.

use super::{BlockScaling, ConeGeometryError, ConeKernel, ConeSide, ScalingShape};

/// A block whose scaling `H` is given directly rather than computed from an
/// iterate. Only the operations used by KKT assembly and solves are defined.
///
/// For a `Dense` shape the data is `H` in column-major order. For an
/// `Expanded { rank }` shape the diagonal is `D` and the data is
/// `e_1, g_1, ..., e_rank, g_rank`, with `H = D + sum_k e_k e_k' / g_k`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CoupledTestCone {
    dim: usize,
    shape: ScalingShape,
}

impl CoupledTestCone {
    pub(super) fn new(dim: usize, shape: ScalingShape) -> Self {
        assert!(shape.is_coupled());
        Self { dim, shape }
    }
}

/// The dense `H` represented by a test block's diagonal and data.
pub(crate) fn dense_scaling(shape: ScalingShape, diagonal: &[f64], data: &[f64]) -> Vec<Vec<f64>> {
    let dim = diagonal.len();
    match shape {
        ScalingShape::Dense => (0..dim)
            .map(|i| (0..dim).map(|j| data[i + j * dim]).collect())
            .collect(),
        ScalingShape::Expanded { rank } => (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        let low_rank: f64 = (0..rank)
                            .map(|k| {
                                let (e, g) = expansion(data, dim, k);
                                e[i] * e[j] / g
                            })
                            .sum();
                        low_rank + if i == j { diagonal[i] } else { 0.0 }
                    })
                    .collect()
            })
            .collect(),
        ScalingShape::Zero | ScalingShape::Diagonal => unreachable!(),
    }
}

fn expansion(data: &[f64], dim: usize, k: usize) -> (&[f64], f64) {
    let start = k * (dim + 1);
    (&data[start..start + dim], data[start + dim])
}

impl ConeKernel for CoupledTestCone {
    fn barrier_parameter(&self) -> usize {
        1
    }

    fn scaling_shape(&self) -> ScalingShape {
        self.shape
    }

    fn scaling_len(&self) -> usize {
        match self.shape {
            ScalingShape::Dense => self.dim * self.dim,
            ScalingShape::Expanded { rank } => rank * (self.dim + 1),
            ScalingShape::Zero | ScalingShape::Diagonal => unreachable!(),
        }
    }

    fn identity_scaling(&self, diagonal: &mut [f64], data: &mut [f64]) {
        diagonal.fill(1.0);
        data.fill(0.0);
        match self.shape {
            ScalingShape::Dense => {
                for i in 0..self.dim {
                    data[i * (self.dim + 1)] = 1.0;
                }
            }
            ScalingShape::Expanded { rank } => {
                for k in 0..rank {
                    data[k * (self.dim + 1) + self.dim] = 1.0;
                }
            }
            ScalingShape::Zero | ScalingShape::Diagonal => unreachable!(),
        }
    }

    fn mul_scaling(&self, scaling: BlockScaling<'_>, v: &[f64], out: &mut [f64]) {
        let h = dense_scaling(self.shape, scaling.diagonal, scaling.data);
        for (out, row) in out.iter_mut().zip(&h) {
            *out = row.iter().zip(v).map(|(h, v)| h * v).sum();
        }
    }

    fn abs_mul_scaling(&self, scaling: BlockScaling<'_>, v: &[f64], out: &mut [f64]) {
        let h = dense_scaling(self.shape, scaling.diagonal, scaling.data);
        for (out, row) in out.iter_mut().zip(&h) {
            *out = row.iter().zip(v).map(|(h, v)| h.abs() * v.abs()).sum();
        }
    }

    fn write_kkt_coupling(
        &self,
        scaling: BlockScaling<'_>,
        off_diagonal: &mut [f64],
        expansion_diagonal: &mut [f64],
    ) {
        let dim = self.dim;
        match self.shape {
            ScalingShape::Dense => {
                let mut next = 0;
                for j in 0..dim {
                    for i in 0..j {
                        off_diagonal[next] = -scaling.data[i + j * dim];
                        next += 1;
                    }
                }
            }
            ScalingShape::Expanded { rank } => {
                for k in 0..rank {
                    let (e, g) = expansion(scaling.data, dim, k);
                    off_diagonal[k * dim..(k + 1) * dim].copy_from_slice(e);
                    expansion_diagonal[k] = g;
                }
            }
            ScalingShape::Zero | ScalingShape::Diagonal => unreachable!(),
        }
    }

    fn update_scaling(&self, _s: &[f64], _y: &[f64], _diagonal: &mut [f64], _data: &mut [f64]) {
        unreachable!("test scalings are given explicitly")
    }

    fn is_primal_interior(&self, _s: &[f64]) -> bool {
        unreachable!()
    }

    fn is_dual_interior(&self, _y: &[f64]) -> bool {
        unreachable!()
    }

    fn unit_point(&self, _out: &mut [f64]) {
        unreachable!()
    }

    fn margin(&self, _v: &[f64]) -> f64 {
        unreachable!()
    }

    fn unit_shift(&self, _v: &mut [f64], _alpha: f64, _side: ConeSide) {
        unreachable!()
    }

    fn reconstruct_slack(&self, _b: &[f64], _ax: &[f64], _out: &mut [f64]) {
        unreachable!()
    }

    fn project_dual(&self, _y: &mut [f64]) {
        unreachable!()
    }

    fn floor_interior(&self, _s: &mut [f64], _y: &mut [f64], _floor: f64) {
        unreachable!()
    }

    fn centering_target(&self, _y: &[f64], _mu: f64, _out: &mut [f64]) {
        unreachable!()
    }

    fn correction(
        &self,
        _scaling: BlockScaling<'_>,
        _y: &[f64],
        _ds: &[f64],
        _dy: &[f64],
        _out: &mut [f64],
    ) {
        unreachable!()
    }

    fn step_length(&self, _v: &[f64], _dv: &[f64]) -> Result<f64, ConeGeometryError> {
        unreachable!()
    }

    fn accumulate_local_score(&self, _y: &[f64], _mu: f64, _residual: &mut [f64], _sum: &mut f64) {
        unreachable!()
    }
}
