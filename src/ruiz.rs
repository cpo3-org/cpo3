//! One-time equilibration of the homogeneous problem data.

use faer::{Col, ColRef};
use std::borrow::Cow;

use crate::{CscMatrix, Problem, Settings, SolveError};

struct Scales {
    dx: Vec<f64>,
    dy: Vec<f64>,
    dt: f64,
}

/// A problem in equilibrated homogeneous coordinates and its inverse mapping.
///
/// `dx` and `dy` store cumulative positive Ruiz factors for primal variables
/// and constraints; `dt` is the scalar factor for the homogeneous coordinate.
/// With `Dx = diag(dx)` and `Dy = diag(dy)`, the equilibrated data are:
///
/// ```text
/// P_scaled = Dx * P * Dx       A_scaled = Dy * A * Dx
/// c_scaled = dt * Dx * c       b_scaled = dt * Dy * b
/// ```
/// Identity equilibration stores no factors (`scales: None`).
pub(crate) struct EquilibratedProblem<'a> {
    problem: Cow<'a, Problem>,
    scales: Option<Scales>,
}

impl<'a> EquilibratedProblem<'a> {
    fn unscaled(problem: &'a Problem) -> Self {
        Self {
            problem: Cow::Borrowed(problem),
            scales: None,
        }
    }

    pub(crate) fn problem(&self) -> &Problem {
        &self.problem
    }

    pub(crate) fn scale_x(&self, values: &[f64]) -> Vec<f64> {
        let Some(scales) = &self.scales else {
            return values.to_vec();
        };
        values
            .iter()
            .zip(&scales.dx)
            .map(|(&x, &dx)| x * scales.dt / dx)
            .collect()
    }

    pub(crate) fn scale_y(&self, values: &[f64]) -> Vec<f64> {
        let Some(scales) = &self.scales else {
            return values.to_vec();
        };
        values
            .iter()
            .zip(&scales.dy)
            .map(|(&y, &dy)| y * scales.dt / dy)
            .collect()
    }

    pub(crate) fn scale_s(&self, values: &[f64]) -> Vec<f64> {
        let Some(scales) = &self.scales else {
            return values.to_vec();
        };
        values
            .iter()
            .zip(&scales.dy)
            .map(|(&s, &dy)| dy * (s * scales.dt))
            .collect()
    }

    pub(crate) fn unscale_x(&self, values: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        self.unscale_x_into(values, 1.0, &mut out);
        out
    }

    pub(crate) fn unscale_y(&self, values: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        self.unscale_y_into(values, 1.0, &mut out);
        out
    }

    pub(crate) fn unscale_s(&self, values: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        self.unscale_s_into(values, 1.0, &mut out);
        out
    }

    /// Write `unscale_x(values * inverse_tau)` to `out`.
    pub(crate) fn unscale_x_into(&self, values: &[f64], inverse_tau: f64, out: &mut Vec<f64>) {
        let Some(scales) = &self.scales else {
            unscale_identity_into(values, inverse_tau, out);
            return;
        };
        let factor = inverse_tau / scales.dt;
        out.clear();
        out.extend(
            values
                .iter()
                .zip(&scales.dx)
                .map(|(&x, &dx)| dx * (x * factor)),
        );
    }

    pub(crate) fn unscale_y_into(&self, values: &[f64], inverse_tau: f64, out: &mut Vec<f64>) {
        let Some(scales) = &self.scales else {
            unscale_identity_into(values, inverse_tau, out);
            return;
        };
        let factor = inverse_tau / scales.dt;
        out.clear();
        out.extend(
            values
                .iter()
                .zip(&scales.dy)
                .map(|(&y, &dy)| dy * (y * factor)),
        );
    }

    pub(crate) fn unscale_s_into(&self, values: &[f64], inverse_tau: f64, out: &mut Vec<f64>) {
        let Some(scales) = &self.scales else {
            unscale_identity_into(values, inverse_tau, out);
            return;
        };
        out.clear();
        out.extend(
            values
                .iter()
                .zip(&scales.dy)
                .map(|(&s, &dy)| (s * inverse_tau) / (scales.dt * dy)),
        );
    }

    pub(crate) fn new(problem: &'a Problem, settings: &Settings) -> Result<Self, SolveError> {
        if settings.ruiz_iters == 0 {
            return Ok(Self::unscaled(problem));
        }
        let (m, n) = problem.dims();
        let mut dx = vec![1.0_f64; n];
        let mut dy = vec![1.0_f64; m];
        let mut dt = 1.0_f64;

        let mut max_x = Col::zeros(n);
        let mut max_y = Col::zeros(m);
        for i in 0..settings.ruiz_iters {
            let max_t = if i == 0 {
                row_norms::<true>(problem, &dx, &dy, dt, &mut max_x, &mut max_y)
            } else {
                row_norms::<false>(problem, &dx, &dy, dt, &mut max_x, &mut max_y)
            };

            if converged(&max_x, &max_y, max_t, settings.ruiz_tol) {
                break;
            }

            update_scales(&mut dx, &max_x);
            update_scales(&mut dy, &max_y);
            if max_t > 0.0 {
                dt /= max_t.sqrt();
            }
        }

        if !valid(&dx, &dy, dt) {
            return Err(SolveError::Numerical(
                "data equilibration produced invalid numerical values".into(),
            ));
        }
        if is_identity(&dx, &dy, dt) {
            return Ok(Self::unscaled(problem));
        }
        let p = problem.p().map(|p| scale_matrix(p, &dx, &dx)).transpose()?;
        let a = scale_matrix(problem.a(), &dy, &dx)?;
        let c = scale_vector(problem.c(), &dx, dt)?;
        let b = scale_vector(problem.b(), &dy, dt)?;
        let scaled = Problem::from_validated(p, c, a, b, problem.cones().to_vec());
        Ok(Self {
            problem: Cow::Owned(scaled),
            scales: Some(Scales { dx, dy, dt }),
        })
    }
}

fn unscale_identity_into(values: &[f64], inverse_tau: f64, out: &mut Vec<f64>) {
    out.clear();
    if inverse_tau == 1.0 {
        out.extend_from_slice(values);
    } else {
        out.extend(values.iter().map(|&value| value * inverse_tau));
    }
}

#[inline]
fn update_scales(scales: &mut [f64], norms: &Col<f64>) {
    let norms = norms.try_as_col_major().unwrap().as_slice();
    assert_eq!(scales.len(), norms.len());
    pulp::Arch::new().dispatch(ScaleUpdate { scales, norms });
}

struct ScaleUpdate<'a> {
    scales: &'a mut [f64],
    norms: &'a [f64],
}

impl pulp::WithSimd for ScaleUpdate<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let (scales, scale_tail) = S::as_mut_simd_f64s(self.scales);
        let (norms, norm_tail) = S::as_simd_f64s(self.norms);
        let zero = simd.splat_f64s(0.0);
        let one = simd.splat_f64s(1.0);
        // Inactive lanes retain their original factors and use a safe divisor.
        for (scale, &norm) in scales.iter_mut().zip(norms) {
            let mask = simd.greater_than_f64s(norm, zero);
            let denominator = simd.sqrt_f64s(simd.select_f64s(mask, norm, one));
            let updated = simd.div_f64s(*scale, denominator);
            *scale = simd.select_f64s(mask, updated, *scale);
        }
        for (scale, &norm) in scale_tail.iter_mut().zip(norm_tail) {
            if norm > 0.0 {
                *scale /= norm.sqrt();
            }
        }
    }
}

#[inline]
fn converged(max_x: &Col<f64>, max_y: &Col<f64>, max_t: f64, tol: f64) -> bool {
    max_x
        .iter()
        .chain(max_y.iter())
        .chain(std::iter::once(&max_t))
        .all(|&norm| norm == 0.0 || (norm - 1.0).abs() <= tol)
}

#[inline]
fn is_identity(dx: &[f64], dy: &[f64], dt: f64) -> bool {
    dx.iter()
        .chain(dy)
        .chain(std::iter::once(&dt))
        .all(|&scale| scale == 1.0)
}

#[inline]
fn valid(dx: &[f64], dy: &[f64], dt: f64) -> bool {
    // Non-short-circuiting `&` enables SIMD vectorization; benchmarks showed
    // this reduction is faster than the short-circuiting `any` check.
    dx.iter()
        .chain(dy)
        .chain(std::iter::once(&dt))
        .fold(true, |valid, &scale| {
            valid & (scale > 0.0) & (scale <= f64::MAX)
        })
}

fn scale_vector(values: &[f64], scales: &[f64], dt: f64) -> Result<Vec<f64>, SolveError> {
    let mut finite = true;
    let scaled = values
        .iter()
        .zip(scales)
        .map(|(&value, &scale)| {
            let scaled = dt * scale * value;
            finite &= scaled.is_finite();
            scaled
        })
        .collect();
    if !finite {
        return Err(SolveError::Numerical("non-finite scaled vector".into()));
    }
    Ok(scaled)
}

/// Apply the cumulative factors once to the original matrix coefficients.
fn scale_matrix(matrix: &CscMatrix, rows: &[f64], cols: &[f64]) -> Result<CscMatrix, SolveError> {
    faer::with_dim!(m, matrix.rows());
    faer::with_dim!(n, matrix.columns());
    let rows = ColRef::from_slice(rows).as_row_shape(m);
    let cols = ColRef::from_slice(cols).as_row_shape(n);
    let orig = matrix.as_faer().as_shape(m, n);
    let mut vals = Vec::with_capacity(matrix.values().len());
    let mut finite = true;
    for j in n.indices() {
        let col_scale = cols[j];
        vals.extend(
            orig.row_idx_of_col_raw(j)
                .iter()
                .zip(orig.val_of_col(j))
                .map(|(&i, &value)| {
                    let scaled = value * (rows[i] * col_scale);
                    finite &= scaled.is_finite();
                    scaled
                }),
        );
    }
    if !finite {
        return Err(SolveError::Numerical("non-finite scaled matrix".into()));
    }
    Ok(matrix.with_values(vals))
}

/// Row infinity norms of the cumulatively scaled [P Aᵀ c; A 0 b; cᵀ bᵀ 0].
/// Overwrites `max_x` and `max_y` and returns `max_t`, without changing the original data.
/// The first iteration uses unit factors and skips their multiplications.
fn row_norms<const UNIT_SCALING: bool>(
    problem: &Problem,
    dx: &[f64],
    dy: &[f64],
    dt: f64,
    max_x: &mut Col<f64>,
    max_y: &mut Col<f64>,
) -> f64 {
    let mut max_t = 0.0_f64;
    for ((norm, &scale), &value) in max_x.iter_mut().zip(dx).zip(problem.c()) {
        let magnitude = if UNIT_SCALING {
            value.abs()
        } else {
            value.abs() * scale * dt
        };
        *norm = magnitude;
        max_t = max_t.max(magnitude);
    }
    for ((norm, &scale), &value) in max_y.iter_mut().zip(dy).zip(problem.b()) {
        let magnitude = if UNIT_SCALING {
            value.abs()
        } else {
            value.abs() * scale * dt
        };
        *norm = magnitude;
        max_t = max_t.max(magnitude);
    }
    let (m, n) = problem.dims();
    faer::with_dim!(m, m);
    faer::with_dim!(n, n);
    {
        let dx = ColRef::from_slice(dx).as_row_shape(n);
        let dy = ColRef::from_slice(dy).as_row_shape(m);
        let mut max_x = max_x.as_row_shape_mut(n);
        let mut max_y = max_y.as_row_shape_mut(m);
        // P stores only its upper triangle, so each entry contributes to both rows.
        if let Some(p) = problem.p() {
            let p = p.as_faer().as_shape(n, n);
            for j in n.indices() {
                let col_scale = dx[j];
                let mut col_max = max_x[j];
                for (&i, &value) in p.row_idx_of_col_raw(j).iter().zip(p.val_of_col(j)) {
                    let magnitude = if UNIT_SCALING {
                        value.abs()
                    } else {
                        value.abs() * dx[i] * col_scale
                    };
                    max_x[i] = max_x[i].max(magnitude);
                    col_max = col_max.max(magnitude);
                }
                max_x[j] = col_max;
            }
        }
        let a = problem.a().as_faer().as_shape(m, n);
        for j in n.indices() {
            let col_scale = dx[j];
            let mut col_max = max_x[j];
            for (&i, &value) in a.row_idx_of_col_raw(j).iter().zip(a.val_of_col(j)) {
                let magnitude = if UNIT_SCALING {
                    value.abs()
                } else {
                    value.abs() * col_scale * dy[i]
                };
                max_y[i] = max_y[i].max(magnitude);
                col_max = col_max.max(magnitude);
            }
            max_x[j] = col_max;
        }
    }
    max_t
}
