//! Closed-form resolvent for the Josephy--Newton step (`subsec:josephy_newton`, `thm:resolvent`).
//!
//! `Resolvent::new` factors the preconditioner and caches the
//! target-system solve `eq:cached_data_solve`.
//! Each target/surrogate supplies the changing right-hand side of
//! `eq:surrogate_rhs`. `solve_increment_rhs` evaluates the same equations in
//! increments about the current iterate to avoid cancellation, solves the scalar
//! quadratic for tau, and reconstructs x and y. Sparse assembly, elimination,
//! and full-system refinement live in `kkt`.

use std::cell::OnceCell;

use crate::linear_algebra::dot;
use thiserror::Error;

use super::{
    kkt::{KktCache, KktError},
    state::{State, StateError},
};

use crate::{
    Problem,
    cone::ConeGeometry,
    linear_solvers::{KktSettings, LinearSystemError},
};

/// Failures while constructing or evaluating the resolvent.
#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum ResolventError {
    #[error(transparent)]
    State(#[from] StateError),
    #[error(transparent)]
    LinearSystem(#[from] LinearSystemError),
    #[error(transparent)]
    Kkt(#[from] KktError),
    #[error("the resolvent tau polynomial has invalid coefficients")]
    InvalidTauPolynomial,
}

impl ResolventError {
    /// Failures that a larger shift or best-effort refinement may cure.
    pub(crate) fn is_numerical(&self) -> bool {
        match self {
            Self::LinearSystem(error) => error.is_numerical(),
            Self::Kkt(error) => error.is_numerical(),
            Self::InvalidTauPolynomial => true,
            Self::State(_) => false,
        }
    }
}

/// One solution of the shifted system of Theorem "closed-form resolvent step".
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolventSolution {
    /// Contiguous primal/dual result, moved directly into an accepted State.
    pub(crate) xy: Vec<f64>,
    pub(crate) x_len: usize,
    /// Nonnegative subproblem candidate; the accepted iterate stays positive.
    pub(crate) tau: f64,
}

impl ResolventSolution {
    pub(crate) fn x(&self) -> &[f64] {
        &self.xy[..self.x_len]
    }

    pub(crate) fn y(&self) -> &[f64] {
        &self.xy[self.x_len..]
    }
}

/// The factorized shifted matrix and its target-independent column.
pub(crate) struct Resolvent<'a> {
    problem: &'a Problem,
    cache: &'a mut KktCache,
    shift: f64,
    d_diagonal: Vec<f64>,
    /// Present only when the preconditioner modifies the true diagonal.
    d_factor: Option<Vec<f64>>,
    alpha: f64,
    /// Anchor contribution tau_a (-c, b) - K z_a, shared by every RHS.
    #[cfg(test)]
    rhs_offset: Vec<f64>,
    /// P x_a + A' y_a + c tau_a, without any resolvent shift.
    dual_residual: Vec<f64>,
    /// b tau_a - A x_a, without forming and then subtracting the slack.
    primal_rhs: Vec<f64>,
    /// d_x' P d_x, shared by every increment polynomial.
    direction_quadratic: f64,
    /// Current iterate, not the corrector surrogate, for the tau fallback.
    state: &'a State,
    /// Computed only if a root fails; shared by subsequent RHS solves.
    fallback: OnceCell<TauFallback>,
}

struct TauFallback {
    q_direction: f64,
    px: Option<Vec<f64>>,
    q_z: f64,
    x_px: f64,
    px_kinv_q: f64,
}

impl<'a> Resolvent<'a> {
    /// Factorize a diagonally clamped preconditioner for the true system
    /// `[P + shift I, A^T; A, -(H + shift I)]`, then refine its `(-c, b)` solve.
    pub(crate) fn new(
        problem: &'a Problem,
        geometry: &ConeGeometry,
        state: &'a State,
        h_diagonal: &[f64],
        shift: f64,
        kkt: &KktSettings,
        cache: &'a mut KktCache,
    ) -> Result<Self, ResolventError> {
        if !shift.is_finite() || shift < 0.0 || (shift == 0.0 && kkt.diagonal_floor <= 0.0) {
            return Err(StateError::InvalidMu.into());
        }
        let d_diagonal = h_diagonal
            .iter()
            .map(|entry| entry + shift)
            .collect::<Vec<_>>();
        let d_factor = (kkt.diagonal_floor > shift).then(|| {
            h_diagonal
                .iter()
                .map(|h| (h + shift).max(kkt.diagonal_floor))
                .collect::<Vec<_>>()
        });
        let factor_diagonal = d_factor.as_deref().unwrap_or(&d_diagonal);
        cache.prepare(problem, geometry, shift, factor_diagonal, kkt)?;
        cache.prepare_direction(problem, shift, &d_diagonal, factor_diagonal)?;
        let n = problem.variable_count();
        let (x2, y2) = cache.direction().split_at(n);
        // Positive identity from the KKT equations, with d = H + shift I.
        let alpha = tau_leading_coefficient(shift, x2, y2, &d_diagonal);
        let direction_quadratic = problem
            .p_mul_opt(x2)
            .as_ref()
            .map_or(0.0, |px2| dot(x2, px2));
        let mut dual_residual = problem.p_mul(state.x());
        for (out, c) in dual_residual.iter_mut().zip(problem.c()) {
            *out += state.tau() * c;
        }
        let mut primal_rhs: Vec<f64> = problem.b().iter().map(|b| -state.tau() * b).collect();
        crate::linear_algebra::constraint_accumulate(
            problem.a(),
            state.x(),
            state.y(),
            [&mut dual_residual[..]],
            [&mut primal_rhs[..]],
            |product| [product],
        );
        crate::linear_algebra::scale(&mut primal_rhs, -1.0);
        #[cfg(test)]
        let rhs_offset = {
            let mut rhs = Vec::with_capacity(n + problem.constraint_count());
            rhs.extend(
                dual_residual
                    .iter()
                    .zip(state.x())
                    .map(|(rd, x)| -rd - shift * x),
            );
            rhs.extend(
                primal_rhs
                    .iter()
                    .zip(&d_diagonal)
                    .zip(state.y())
                    .map(|((primal, d), y)| primal + d * y),
            );
            rhs
        };

        Ok(Self {
            problem,
            cache,
            shift,
            d_diagonal,
            d_factor,
            alpha,
            #[cfg(test)]
            rhs_offset,
            dual_residual,
            primal_rhs,
            direction_quadratic,
            state,
            fallback: OnceCell::new(),
        })
    }

    /// Solve the shifted system with right-hand side `(r_x, r_y, r_tau)` at
    /// target `mu`, accepting zero tau candidates without a boundary marker.
    #[cfg(test)]
    pub(crate) fn solve(
        &mut self,
        r_x: &[f64],
        r_y: &[f64],
        r_tau: f64,
        mu: f64,
    ) -> Result<ResolventSolution, ResolventError> {
        let n = self.problem.variable_count();
        let m = self.problem.constraint_count();
        crate::linear_solvers::validate_rhs(n, r_x)?;
        crate::linear_solvers::validate_rhs(m, r_y)?;
        let mut rhs = Vec::with_capacity(n + m);
        rhs.extend_from_slice(r_x);
        rhs.extend(r_y.iter().map(|value| -value));
        self.solve_rhs(&rhs, r_tau, mu)
    }

    /// `solve` with the stacked right-hand side `(r_x, -r_y)` prebuilt.
    /// Solve this changing right-hand side directly; only the constant data
    /// direction is carried between factorizations.
    #[cfg(test)]
    pub(crate) fn solve_rhs(
        &mut self,
        rhs: &[f64],
        r_tau: f64,
        mu: f64,
    ) -> Result<ResolventSolution, ResolventError> {
        let n = self.problem.variable_count();
        let m = self.problem.constraint_count();
        crate::linear_solvers::validate_rhs(n + m, rhs)?;

        let mut residual = rhs.to_vec();
        crate::linear_algebra::add_assign(&mut residual, &self.rhs_offset);
        self.solve_increment_rhs(&residual, r_tau, mu)
    }

    /// The cached mu-target solution of K z = (-c, b).
    pub(crate) fn constant_direction(&self) -> &[f64] {
        self.cache.direction()
    }

    pub(crate) fn dual_residual(&self) -> &[f64] {
        &self.dual_residual
    }

    pub(crate) fn primal_rhs(&self) -> &[f64] {
        &self.primal_rhs
    }

    /// Solve K u = residual about the current iterate and recover
    /// z = z_a + u + (tau - tau_a) d. Callers can form residual directly,
    /// avoiding cancellation of H y with s and shifted anchor terms.
    pub(crate) fn solve_increment_rhs(
        &mut self,
        residual: &[f64],
        r_tau: f64,
        mu: f64,
    ) -> Result<ResolventSolution, ResolventError> {
        let n = self.problem.variable_count();
        if !r_tau.is_finite() || !mu.is_finite() || mu < 0.0 {
            return Err(ResolventError::InvalidTauPolynomial);
        }
        let anchor_tau = self.state.tau();
        let mut increment = self.cache.solve(
            self.problem,
            self.shift,
            &self.d_diagonal,
            self.d_factor.as_deref().unwrap_or(&self.d_diagonal),
            residual,
        )?;
        let mut base = self.cache.take_reconstruction();
        base.clear();
        base.extend_from_slice(&increment);
        let (base_x, base_y) = base.split_at_mut(n);
        crate::linear_algebra::add_assign(base_x, self.state.x());
        crate::linear_algebra::add_assign(base_y, self.state.y());
        let (delta_tau, tau) = match self.solve_tau_increment(&base, r_tau, mu) {
            Some(solution) => solution,
            None => {
                // The fallback expects the point-coordinate particular solution.
                crate::linear_algebra::update(
                    &mut base,
                    self.cache.direction(),
                    |base, direction| (-anchor_tau).mul_add(direction, base),
                );
                let q_constant = q_dot(self.problem, &base);
                let tau = self.linearized_tau(&base, q_constant, r_tau, mu);
                (tau - anchor_tau, tau)
            }
        };
        self.cache.put_reconstruction(base);
        crate::linear_algebra::update(
            &mut increment,
            self.cache.direction(),
            |increment, direction| delta_tau.mul_add(direction, increment),
        );
        let (x, y) = increment.split_at_mut(n);
        crate::linear_algebra::add_assign(x, self.state.x());
        crate::linear_algebra::add_assign(y, self.state.y());
        Ok(ResolventSolution {
            xy: increment,
            x_len: n,
            tau,
        })
    }

    /// Substitute z = base + delta_tau * direction and
    /// tau = anchor_tau + delta_tau into the same scalar resolvent equation.
    /// Both LP and QP use the existing scaled quadratic-root routine.
    fn solve_tau_increment(&self, base: &[f64], r_tau: f64, mu: f64) -> Option<(f64, f64)> {
        let n = self.problem.variable_count();
        let anchor_tau = self.state.tau();
        let q_base = q_dot(self.problem, base);
        let g = self.shift * anchor_tau - r_tau - q_base;
        // h(d) = shift - alpha - d_x' P d_x from the KKT equations.
        // Use the same positive identity as the leading coefficient, rather
        // than separately summing the data column's potentially large terms.
        let mut b = self.alpha * anchor_tau + g + anchor_tau * self.direction_quadratic;
        let mut c = anchor_tau * g - mu;
        if let Some(p_base) = self.problem.p_mul_opt(&base[..n]) {
            b -= 2.0 * dot(&self.cache.direction()[..n], &p_base);
            c -= dot(&base[..n], &p_base);
        }
        let delta_tau = quadratic_root(self.alpha, b, c, 1e-9)?;
        let tau = anchor_tau + delta_tau;
        if !tau.is_finite() || tau < -1e-10 {
            return None;
        }
        // Preserve the existing acceptance tolerance and zero clamp, while
        // retaining the computed increment when no clamp was needed.
        if tau < 0.0 {
            Some((-anchor_tau, 0.0))
        } else {
            Some((delta_tau, tau))
        }
    }

    fn linearized_tau(&self, constant: &[f64], q_constant: f64, r_tau: f64, mu: f64) -> f64 {
        let n = self.problem.variable_count();
        let tau = self.state.tau();
        let cached = self.fallback.get_or_init(|| {
            let px = self.problem.p_mul_opt(self.state.x());
            TauFallback {
                q_direction: q_dot(self.problem, self.cache.direction()),
                q_z: dot(self.problem.c(), self.state.x()) + dot(self.problem.b(), self.state.y()),
                x_px: px.as_ref().map_or(0.0, |px| dot(self.state.x(), px)),
                px_kinv_q: px
                    .as_ref()
                    .map_or(0.0, |px| -dot(px, &self.cache.direction()[..n])),
                px,
            }
        });
        let q_kinv_q = -cached.q_direction;
        let px_kinv_r = cached.px.as_ref().map_or(0.0, |px| dot(px, &constant[..n]));
        // Linearize x'Px at the current x but retain the quadratic in tau.
        // Unlike the primary solve, any finite root is clamped to the bounds.
        let candidate = quadratic_root(
            self.shift + q_kinv_q,
            -r_tau - q_constant + 2.0 * cached.px_kinv_q,
            -mu + cached.x_px - 2.0 * px_kinv_r,
            0.0,
        );
        if let Some(candidate) = candidate {
            return candidate.max(0.1 * tau).min(10.0 * tau);
        }

        // If that quadratic fails, take CPO3's first-order correction.
        let q_rz = q_constant - tau * q_kinv_q - cached.q_z;
        let px_rz = px_kinv_r - tau * cached.px_kinv_q - cached.x_px;
        // r_tau retains the caller's anchor even though we linearize at state.
        let g = self.shift * tau * tau - r_tau * tau - tau * cached.q_z - mu - cached.x_px;
        let num = g - tau * q_rz - 2.0 * px_rz;
        let den =
            2.0 * self.shift * tau - r_tau - cached.q_z + tau * q_kinv_q + 2.0 * cached.px_kinv_q;
        let candidate = tau + if den.abs() < 1e-16 { 0.0 } else { -num / den };
        if !candidate.is_finite() {
            return tau;
        }
        candidate.max(0.1 * tau).min(10.0 * tau)
    }
}

/// shift + shift*||x||^2 + y'*(H + shift I)*y, without forming unweighted squares.
fn tau_leading_coefficient(shift: f64, x: &[f64], y: &[f64], d: &[f64]) -> f64 {
    // Multiplying the nonnegative weight first avoids an intermediate square
    // overflowing/underflowing when the weighted square is representable.
    let primal = x
        .iter()
        .fold(shift, |sum, &value| sum + (shift * value) * value);
    y.iter().zip(d).fold(primal, |sum, (&value, &weight)| {
        sum + (weight * value) * value
    })
}

fn q_dot(problem: &Problem, vector: &[f64]) -> f64 {
    let (x, y) = vector.split_at(problem.variable_count());
    dot(problem.c(), x) + dot(problem.b(), y)
}

/// CPO3's Muller branch for general coefficient signs, without squaring
/// unscaled coefficients. Tolerances remain in the original coefficient units.
fn quadratic_root(a: f64, b: f64, c: f64, negative_discriminant_tolerance: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return None;
    }
    let tau = if a.abs() < 1e-15 {
        if b.abs() < 1e-15 {
            return None;
        }
        -c / b
    } else if c == 0.0 {
        if b > 0.0 { 0.0 } else { -b / a }
    } else {
        let scale = a.abs().max(b.abs()).max(c.abs());
        let sqrt_scale = scale.sqrt();
        let sqrt_a = a.abs().sqrt();
        let sqrt_c = c.abs().sqrt();
        let sqrt_ac = (sqrt_a / sqrt_scale) * (sqrt_c / sqrt_scale);
        let abs_b = (b / scale).abs();
        let root = if a.is_sign_positive() != c.is_sign_positive() {
            // b^2 + 4 |ac|: hypot avoids both overflow and cancellation.
            abs_b.hypot(2.0 * sqrt_ac)
        } else {
            // Factoring the difference preserves its sign even when its
            // square would underflow. Compare sqrt(-D) in original units.
            let difference = abs_b - 2.0 * sqrt_ac;
            let magnitude = difference.abs().sqrt() * (abs_b + 2.0 * sqrt_ac).sqrt();
            if difference < 0.0 {
                if magnitude * scale > negative_discriminant_tolerance.sqrt() {
                    return None;
                }
                0.0
            } else {
                magnitude
            }
        };
        let q = 0.5 * (abs_b + root);
        if b > 0.0 {
            // c / (-q*scale), arranged so tiny c/scale is not erased.
            -c.signum() * (sqrt_c / sqrt_a) * (sqrt_ac / q)
        } else {
            // q*scale / a, without overflowing q*scale or underflowing a/scale.
            (q * sqrt_scale) / (a / sqrt_scale)
        }
    };
    tau.is_finite().then_some(tau)
}
