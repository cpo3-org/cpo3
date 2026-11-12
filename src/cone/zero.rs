//! The zero cone `{0}` of equality rows. Its dual cone is the whole space.

use super::{ConeGeometryError, ConeKernel, ConeSide, ScalingShape};

/// Equality rows: the slack is exactly zero and the multiplier is free.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ZeroCone;

impl ConeKernel for ZeroCone {
    fn barrier_parameter(&self) -> usize {
        0
    }

    fn scaling_shape(&self) -> ScalingShape {
        ScalingShape::Zero
    }

    fn is_primal_interior(&self, s: &[f64]) -> bool {
        s.iter().all(|entry| *entry == 0.0)
    }

    fn is_dual_interior(&self, y: &[f64]) -> bool {
        y.iter().all(|entry| entry.is_finite())
    }

    fn unit_point(&self, out: &mut [f64]) {
        out.fill(0.0);
    }

    fn identity_scaling(&self, out: &mut [f64]) {
        out.fill(0.0);
    }

    fn margin(&self, _v: &[f64]) -> f64 {
        f64::INFINITY
    }

    fn unit_shift(&self, v: &mut [f64], _alpha: f64, side: ConeSide) {
        if side == ConeSide::Primal {
            v.fill(0.0);
        }
    }

    fn reconstruct_slack(&self, _b: &[f64], _ax: &[f64], out: &mut [f64]) {
        out.fill(0.0);
    }

    fn project_dual(&self, _y: &mut [f64]) {}

    fn floor_interior(&self, s: &mut [f64], _y: &mut [f64], _floor: f64) {
        s.fill(0.0);
    }

    fn scaling_and_inverse(&self, _s: &[f64], _y: &[f64], h: &mut [f64], inverse: &mut [f64]) {
        h.fill(0.0);
        inverse.fill(0.0);
    }

    fn step_to_boundary(&self, _v: &[f64], target: &[f64]) -> Result<f64, ConeGeometryError> {
        if target.iter().all(|value| value.is_finite()) {
            Ok(1.0)
        } else {
            Err(ConeGeometryError::NonFiniteInput { argument: "dv" })
        }
    }

    fn accumulate_local_score(&self, _y: &[f64], mu: f64, residual: &mut [f64], sum: &mut f64) {
        for residual in residual {
            *sum += *residual * *residual / mu;
        }
    }
}
