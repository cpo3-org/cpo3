//! The nonnegative orthant `{s : s >= 0}`, which is self-dual, with barrier
//! `F(s) = -sum(log s_i)`.

use faer::{ColMut, ColRef};

use super::{ConeGeometryError, ConeKernel, ConeSide, ScalingShape};

/// Inequality rows `(Ax)_i <= b_i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct NonnegativeCone {
    dim: usize,
}

impl NonnegativeCone {
    pub(super) fn new(dim: usize) -> Self {
        Self { dim }
    }
}

fn is_interior(v: &[f64]) -> bool {
    v.iter().all(|entry| entry.is_finite() && *entry > 0.0)
}

impl ConeKernel for NonnegativeCone {
    fn barrier_parameter(&self) -> usize {
        self.dim
    }

    fn scaling_shape(&self) -> ScalingShape {
        ScalingShape::Diagonal
    }

    fn is_primal_interior(&self, s: &[f64]) -> bool {
        is_interior(s)
    }

    fn is_dual_interior(&self, y: &[f64]) -> bool {
        is_interior(y)
    }

    fn unit_point(&self, out: &mut [f64]) {
        out.fill(1.0);
    }

    fn identity_scaling(&self, out: &mut [f64]) {
        out.fill(1.0);
    }

    fn margin(&self, v: &[f64]) -> f64 {
        v.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn unit_shift(&self, v: &mut [f64], alpha: f64, _side: ConeSide) {
        for value in v {
            *value += alpha;
        }
    }

    fn reconstruct_slack(&self, b: &[f64], ax: &[f64], out: &mut [f64]) {
        for ((out, b), ax) in out.iter_mut().zip(b).zip(ax) {
            *out = (b - ax).max(0.0);
        }
    }

    fn project_dual(&self, y: &mut [f64]) {
        for value in y {
            *value = value.max(0.0);
        }
    }

    fn floor_interior(&self, s: &mut [f64], y: &mut [f64], floor: f64) {
        for value in s.iter_mut().chain(y) {
            *value = value.max(floor);
        }
    }

    fn scaling_and_inverse(&self, s: &[f64], y: &[f64], h: &mut [f64], inverse: &mut [f64]) {
        faer::zip!(
            ColMut::from_slice_mut(h),
            ColMut::from_slice_mut(inverse),
            ColRef::from_slice(s),
            ColRef::from_slice(y)
        )
        .for_each(|faer::unzip!(h, inverse, s, y)| {
            *h = s / y;
            *inverse = y.recip();
        });
    }

    fn step_to_boundary(&self, v: &[f64], target: &[f64]) -> Result<f64, ConeGeometryError> {
        let mut alpha: f64 = 1.0;
        for (&value, &other) in v.iter().zip(target) {
            if !other.is_finite() {
                return Err(ConeGeometryError::NonFiniteInput { argument: "dv" });
            }
            let direction = other - value;
            if direction < 0.0 {
                alpha = alpha.min(-value / direction);
            }
        }
        Ok(alpha)
    }

    fn accumulate_local_score(&self, y: &[f64], mu: f64, residual: &mut [f64], sum: &mut f64) {
        for (&value, residual) in y.iter().zip(residual) {
            *residual -= mu / value;
            let weight = mu + mu / (value * value);
            *sum += *residual * *residual / weight;
        }
    }
}
