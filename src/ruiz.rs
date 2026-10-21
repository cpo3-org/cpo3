//! One-time equilibration of the homogeneous problem data.

use faer::{Col, ColMut};
use std::borrow::Cow;

use crate::{Cone, CscMatrix, Problem, Settings, SolveError};

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
pub(crate) struct EquilibratedProblem<'a> {
    problem: Cow<'a, Problem>,
    dx: Col<f64>,
    dy: Col<f64>,
    dt: f64,
}

impl<'a> EquilibratedProblem<'a> {
    fn unscaled(problem: &'a Problem) -> Self {
        let (m, n) = problem.dims();
        Self {
            problem: Cow::Borrowed(problem),
            dx: Col::ones(n),
            dy: Col::ones(m),
            dt: 1.0,
        }
    }

    pub(crate) fn problem(&self) -> &Problem {
        &self.problem
    }

    pub(crate) fn scale_x(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(self.dx.iter())
            .map(|(&x, &dx)| x * self.dt / dx)
            .collect()
    }

    pub(crate) fn scale_y(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(self.dy.iter())
            .map(|(&y, &dy)| y * self.dt / dy)
            .collect()
    }

    pub(crate) fn scale_s(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(self.dy.iter())
            .map(|(&s, &dy)| dy * (s * self.dt))
            .collect()
    }

    pub(crate) fn unscale_x(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(self.dx.iter())
            .map(|(&x, &dx)| dx * x / self.dt)
            .collect()
    }

    pub(crate) fn unscale_y(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(self.dy.iter())
            .map(|(&y, &dy)| dy * y / self.dt)
            .collect()
    }

    /// Write `unscale_x(values * factor)` to `out`.
    pub(crate) fn unscale_x_into(&self, values: &[f64], factor: f64, out: &mut Vec<f64>) {
        out.clear();
        out.extend(
            values
                .iter()
                .zip(self.dx.iter())
                .map(|(&x, &dx)| dx * (x * factor) / self.dt),
        );
    }

    pub(crate) fn unscale_y_into(&self, values: &[f64], factor: f64, out: &mut Vec<f64>) {
        out.clear();
        out.extend(
            values
                .iter()
                .zip(self.dy.iter())
                .map(|(&y, &dy)| dy * (y * factor) / self.dt),
        );
    }

    pub(crate) fn unscale_s_by(&self, values: &[f64], factor: f64) -> Vec<f64> {
        let mut out = Vec::new();
        self.unscale_s_into(values, factor, &mut out);
        out
    }

    pub(crate) fn unscale_s_into(&self, values: &[f64], factor: f64, out: &mut Vec<f64>) {
        out.clear();
        out.extend(
            values
                .iter()
                .zip(self.dy.iter())
                .map(|(&s, &dy)| (s * factor) / (self.dt * dy)),
        );
    }

    pub(crate) fn new(problem: &'a Problem, settings: &Settings) -> Result<Self, SolveError> {
        if settings.ruiz_iters == 0 {
            return Ok(Self::unscaled(problem));
        }
        let (m, n) = problem.dims();
        let mut dx = Col::<f64>::ones(n);
        let mut dy = Col::<f64>::ones(m);
        let mut dt = 1.0_f64;

        let mut nx = Col::zeros(n);
        let mut ny = Col::zeros(m);
        for i in 0..settings.ruiz_iters {
            let nt = if i == 0 {
                row_norms::<true>(problem, &dx, &dy, dt, &mut nx, &mut ny)
            } else {
                row_norms::<false>(problem, &dx, &dy, dt, &mut nx, &mut ny)
            };
            equalize_cone_norms(problem.cones(), ny.as_mut());

            if converged(&nx, &ny, nt, settings.ruiz_tol) {
                break;
            }

            faer::zip!(&mut dx, &nx).for_each(|faer::unzip!(scale, norm)| {
                if *norm > 0.0 {
                    *scale /= norm.sqrt();
                }
            });
            faer::zip!(&mut dy, &ny).for_each(|faer::unzip!(scale, norm)| {
                if *norm > 0.0 {
                    *scale /= norm.sqrt();
                }
            });
            if nt > 0.0 {
                dt /= nt.sqrt();
            }
        }

        if !valid(&dx, &dy, dt) {
            return Err(SolveError::Numerical(
                "data equilibration produced invalid numerical values".into(),
            ));
        }
        if is_identity(&dx, &dy, dt) {
            return Ok(Self {
                problem: Cow::Borrowed(problem),
                dx,
                dy,
                dt,
            });
        }
        let p = problem.p().map(|p| scale_matrix(p, &dx, &dx)).transpose()?;
        let a = scale_matrix(problem.a(), &dy, &dx)?;
        let c = problem
            .c()
            .iter()
            .zip(dx.iter())
            .map(|(&c, &dx)| dt * dx * c)
            .collect();
        let b = problem
            .b()
            .iter()
            .zip(dy.iter())
            .map(|(&b, &dy)| dt * dy * b)
            .collect();
        let scaled = Problem::from_upper_triangle(p, c, a, b, problem.cones().to_vec())
            .map_err(|error| SolveError::Numerical(error.to_string()))?;
        Ok(Self {
            problem: Cow::Owned(scaled),
            dx,
            dy,
            dt,
        })
    }
}

#[inline]
fn converged(nx: &Col<f64>, ny: &Col<f64>, nt: f64, tol: f64) -> bool {
    nx.iter()
        .chain(ny.iter())
        .chain(std::iter::once(&nt))
        .all(|&norm| norm == 0.0 || (norm - 1.0).abs() <= tol)
}

#[inline]
fn is_identity(dx: &Col<f64>, dy: &Col<f64>, dt: f64) -> bool {
    dx.iter()
        .chain(dy.iter())
        .chain(std::iter::once(&dt))
        .all(|&scale| scale == 1.0)
}

#[inline]
fn valid(dx: &Col<f64>, dy: &Col<f64>, dt: f64) -> bool {
    // Non-short-circuiting `&` enables SIMD vectorization; benchmarks showed
    // this reduction is faster than the short-circuiting `any` check.
    dx.iter()
        .chain(dy.iter())
        .chain(std::iter::once(&dt))
        .fold(true, |valid, &scale| {
            valid & (scale > 0.0) & (scale <= f64::MAX)
        })
}

/// Apply the cumulative factors once to the original matrix coefficients.
fn scale_matrix(
    matrix: &CscMatrix,
    rows: &Col<f64>,
    cols: &Col<f64>,
) -> Result<CscMatrix, SolveError> {
    faer::with_dim!(m, matrix.rows());
    faer::with_dim!(n, matrix.columns());
    let rows = rows.as_row_shape(m);
    let cols = cols.as_row_shape(n);
    let orig = matrix.as_faer().as_shape(m, n);
    let mut vals = Vec::with_capacity(matrix.values().len());
    for j in n.indices() {
        let col_scale = cols[j];
        vals.extend(
            orig.row_idx_of_col_raw(j)
                .iter()
                .zip(orig.val_of_col(j))
                .map(|(&i, &value)| value * (rows[i] * col_scale)),
        );
    }
    matrix
        .with_values(vals)
        .map_err(|error| SolveError::Numerical(error.to_string()))
}

/// Row infinity norms of the cumulatively scaled [P Aᵀ c; A 0 b; cᵀ bᵀ 0].
/// Overwrites `nx` and `ny` and returns `nt`, without changing the original data.
/// The first iteration uses unit factors and skips their multiplications.
fn row_norms<const UNIT_SCALING: bool>(
    problem: &Problem,
    dx: &Col<f64>,
    dy: &Col<f64>,
    dt: f64,
    nx: &mut Col<f64>,
    ny: &mut Col<f64>,
) -> f64 {
    nx.fill(0.0);
    ny.fill(0.0);
    let (m, n) = problem.dims();
    faer::with_dim!(m, m);
    faer::with_dim!(n, n);
    {
        let dx = dx.as_row_shape(n);
        let dy = dy.as_row_shape(m);
        let mut nx = nx.as_row_shape_mut(n);
        let mut ny = ny.as_row_shape_mut(m);
        // P stores only its upper triangle, so each entry contributes to both rows.
        if let Some(p) = problem.p() {
            let p = p.as_faer().as_shape(n, n);
            for j in n.indices() {
                let col_scale = dx[j];
                let mut col_max = nx[j];
                for (&i, &value) in p.row_idx_of_col_raw(j).iter().zip(p.val_of_col(j)) {
                    let magnitude = if UNIT_SCALING {
                        value.abs()
                    } else {
                        value.abs() * dx[i] * col_scale
                    };
                    nx[i] = nx[i].max(magnitude);
                    col_max = col_max.max(magnitude);
                }
                nx[j] = col_max;
            }
        }
        let a = problem.a().as_faer().as_shape(m, n);
        for j in n.indices() {
            let col_scale = dx[j];
            let mut col_max = nx[j];
            for (&i, &value) in a.row_idx_of_col_raw(j).iter().zip(a.val_of_col(j)) {
                let magnitude = if UNIT_SCALING {
                    value.abs()
                } else {
                    value.abs() * col_scale * dy[i]
                };
                ny[i] = ny[i].max(magnitude);
                col_max = col_max.max(magnitude);
            }
            nx[j] = col_max;
        }
    }
    let mut nt = 0.0_f64;
    for ((norm, &scale), &value) in nx.iter_mut().zip(dx.iter()).zip(problem.c()) {
        let magnitude = if UNIT_SCALING {
            value.abs()
        } else {
            value.abs() * scale * dt
        };
        *norm = norm.max(magnitude);
        nt = nt.max(magnitude);
    }
    for ((norm, &scale), &value) in ny.iter_mut().zip(dy.iter()).zip(problem.b()) {
        let magnitude = if UNIT_SCALING {
            value.abs()
        } else {
            value.abs() * scale * dt
        };
        *norm = norm.max(magnitude);
        nt = nt.max(magnitude);
    }
    nt
}

fn equalize_cone_norms(cones: &[Cone], mut norms: ColMut<'_, f64>) {
    let mut offset = 0;
    for cone in cones {
        let dim = cone.vector_dim();
        if matches!(cone, Cone::SecondOrder(_) | Cone::PositiveSemidefinite(_)) {
            let mut block = norms.as_mut().subrows_mut(offset, dim);
            let norm_max = block.norm_max();
            block.fill(norm_max);
        }
        offset += dim;
    }
}
