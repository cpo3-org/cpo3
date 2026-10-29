//! Closed-form resolvent for the Josephy--Newton step (`subsec:josephy_newton`, `thm:resolvent`).
//!
//! `Resolvent::new` factors the preconditioner and caches the
//! target-system solve `eq:cached_data_solve`, returning the predictor increment
//! computed alongside it.
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
    /// `[P + shift I, A^T; A, -(H + shift I)]`, then refine its `(-c, b)` and
    /// predictor solves. Return the predictor increment for reconstruction.
    pub(crate) fn new(
        problem: &'a Problem,
        geometry: &ConeGeometry,
        state: &'a State,
        h_diagonal: &[f64],
        shift: f64,
        kkt: &KktSettings,
        cache: &'a mut KktCache,
    ) -> Result<(Self, Vec<f64>), ResolventError> {
        if !shift.is_finite() || shift <= 0.0 {
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
        let n = problem.variable_count();
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
        let predictor_rhs = dual_residual
            .iter()
            .map(|rd| -rd)
            .chain(primal_rhs.iter().copied())
            .collect::<Vec<_>>();
        let predictor_increment = cache.prepare_direction(
            problem,
            shift,
            &d_diagonal,
            factor_diagonal,
            &predictor_rhs,
        )?;
        let (x2, y2) = cache.direction().split_at(n);
        // Positive identity from the KKT equations, with d = H + shift I.
        let alpha = tau_leading_coefficient(shift, x2, y2, &d_diagonal);
        let direction_quadratic = problem
            .p_mul_opt(x2)
            .as_ref()
            .map_or(0.0, |px2| dot(x2, px2));
        Ok((
            Self {
                problem,
                cache,
                shift,
                d_diagonal,
                d_factor,
                alpha,
                dual_residual,
                primal_rhs,
                direction_quadratic,
                state,
                fallback: OnceCell::new(),
            },
            predictor_increment,
        ))
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
        if !r_tau.is_finite() || !mu.is_finite() || mu < 0.0 {
            return Err(ResolventError::InvalidTauPolynomial);
        }
        let increment = self.cache.solve(
            self.problem,
            self.shift,
            &self.d_diagonal,
            self.d_factor.as_deref().unwrap_or(&self.d_diagonal),
            residual,
        )?;
        self.reconstruct_increment(increment, r_tau, mu)
    }

    pub(crate) fn reconstruct_increment(
        &mut self,
        mut increment: Vec<f64>,
        r_tau: f64,
        mu: f64,
    ) -> Result<ResolventSolution, ResolventError> {
        if !r_tau.is_finite() || !mu.is_finite() || mu < 0.0 {
            return Err(ResolventError::InvalidTauPolynomial);
        }
        let n = self.problem.variable_count();
        let anchor_tau = self.state.tau();
        let mut base = self.cache.take_reconstruction();
        base.clear();
        base.extend_from_slice(&increment);
        let (base_x, base_y) = base.split_at_mut(n);
        crate::linear_algebra::add_assign(base_x, self.state.x());
        crate::linear_algebra::add_assign(base_y, self.state.y());
        let (delta_tau, tau) = if let Some(solution) = self.solve_tau_increment(&base, r_tau, mu) {
            solution
        } else {
            // The fallback expects the point-coordinate particular solution.
            crate::linear_algebra::update(&mut base, self.cache.direction(), |base, direction| {
                (-anchor_tau).mul_add(direction, base)
            });
            let q_constant = q_dot(self.problem, &base);
            let tau = self.linearized_tau(&base, q_constant, r_tau, mu);
            (tau - anchor_tau, tau)
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
        let root = if a.is_sign_positive() == c.is_sign_positive() {
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
        } else {
            // b^2 + 4 |ac|: hypot avoids both overflow and cancellation.
            abs_b.hypot(2.0 * sqrt_ac)
        };
        let q = abs_b.midpoint(root);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_algebra::matrix;
    use crate::{Cone, LinearSolver, Settings};
    use approx::assert_relative_eq;
    use rstest::rstest;

    fn settings(backend: LinearSolver) -> KktSettings {
        KktSettings::from(&Settings {
            threads: 1,
            linear_solver: backend,
            refine_tol: 1e-12,
            ..Settings::default()
        })
    }

    // Deliberately use dense, explicit equations instead of the production
    // sparse products and scalar polynomial to construct and check the oracle.
    fn px(x: &[f64], quadratic: bool) -> [f64; 2] {
        if quadratic {
            [2. * x[0] + x[1], x[0] + 3. * x[1]]
        } else {
            [0.; 2]
        }
    }

    fn kkt_product(z: &[f64], quadratic: bool) -> [f64; 5] {
        let p = px(z, quadratic);
        // shift = 0.5, H = diag(1, 2, 0).
        [
            p[0] + 0.5 * z[0] + z[2] + z[4],
            p[1] + 0.5 * z[1] - 2. * z[3] + 2. * z[4],
            z[0] - 1.5 * z[2],
            -2. * z[1] - 2.5 * z[3],
            z[0] + 2. * z[1] - 0.5 * z[4],
        ]
    }

    fn scalar_equation(z: &[f64], tau: f64, r_tau: f64, mu: f64, quadratic: bool) -> f64 {
        let p = px(z, quadratic);
        0.5 * tau * tau
            - tau * (-3. * z[0] - 7. * z[1] + z[2] - 4. * z[3] + 5. * z[4] + r_tau)
            - z[0] * p[0]
            - z[1] * p[1]
            - mu
    }

    #[rstest]
    fn resolvent_satisfies_vector_and_scalar_equations(
        #[values(LinearSolver::Faer, LinearSolver::Feral)] backend: LinearSolver,
        #[values(false, true)] eliminate: bool,
        #[values(false, true)] quadratic: bool,
    ) {
        let problem = Problem::new(
            quadratic.then(|| matrix(2, 2, &[2., 1., 1., 3.])),
            vec![-3., -7.],
            matrix(3, 2, &[1., 0., 0., -2., 1., 2.]),
            vec![1., -4., 5.],
            vec![Cone::Nonnegative(2), Cone::Zero(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let state = State::new(vec![0.5, -0.25], vec![1., 4., 0.], vec![1., 2., -1.], 1.).unwrap();
        let anchor = [0.5, -0.25, 1., 2., -1.];
        let mut settings = settings(backend);
        settings.eliminate_bound_rows = eliminate;
        let mut cache = KktCache::new();
        let (mut resolvent, predictor) = Resolvent::new(
            &problem,
            &geometry,
            &state,
            &[1., 2., 0.],
            0.5,
            &settings,
            &mut cache,
        )
        .unwrap();
        let p = px(&anchor, quadratic);
        let predictor_rhs = [3. - p[0], 13. - p[1], 0.5, -4.5, 5.];
        assert_relative_eq!(
            kkt_product(&predictor, quadratic).as_slice(),
            predictor_rhs.as_slice(),
            epsilon = 1e-11
        );

        // The true nonlinear root is not subject to the fallback interval.
        // Reuse the same factorization for multiple targets, including mu = 0.
        for (tau, mu) in [(0.05, 0.), (1.3, 0.1), (20., 1.)] {
            let expected = [0.2, -0.4, 0.7, 1.1, -0.3];
            let delta: Vec<_> = expected.iter().zip(anchor).map(|(z, a)| z - a).collect();
            let q = [-3., -7., -1., 4., -5.];
            let residual: Vec<_> = kkt_product(&delta, quadratic)
                .iter()
                .zip(q)
                .map(|(kz, q)| kz + q * (tau - 1.))
                .collect();
            let r_tau = scalar_equation(&expected, tau, 0., mu, quadratic) / tau;
            let result = resolvent.solve_increment_rhs(&residual, r_tau, mu).unwrap();
            assert!(resolvent.fallback.get().is_none());
            assert_relative_eq!(result.tau, tau, epsilon = 1e-10);
            assert_relative_eq!(result.xy.as_slice(), expected.as_slice(), epsilon = 1e-10);
            let delta: Vec<_> = result.xy.iter().zip(anchor).map(|(z, a)| z - a).collect();
            for ((kz, q), rhs) in kkt_product(&delta, quadratic).iter().zip(q).zip(&residual) {
                assert_relative_eq!(kz + q * (result.tau - 1.), *rhs, epsilon = 1e-10);
            }
            assert_relative_eq!(
                scalar_equation(&result.xy, result.tau, r_tau, mu, quadratic),
                0.,
                epsilon = 1e-9
            );
        }
    }

    #[rstest]
    #[case(1., -3., 2., 2.)]
    #[case(1., 2., 0., 0.)]
    #[case(1., -2., 0., 2.)]
    #[case(0., 2., -4., 2.)]
    #[case(1e-16, 2., -4., 2.)]
    #[case(1., 1e16, -1., 1e-16)]
    #[case(1., -1e16, -1., 1e16)]
    #[case(1e200, -3e200, 2e200, 2.)]
    #[case(1e200, 0., -1e-200, 1e-200)]
    fn scalar_root_handles_degeneracy_cancellation_and_extreme_scales(
        #[case] a: f64,
        #[case] b: f64,
        #[case] c: f64,
        #[case] expected: f64,
    ) {
        let root = quadratic_root(a, b, c, 0.).unwrap();
        assert_relative_eq!(root, expected, epsilon = 0., max_relative = 2e-14);
    }

    #[rstest]
    #[case(0., 0., 1.)]
    #[case(1., 0., 1.)]
    #[case(f64::NAN, 1., -1.)]
    #[case(1., f64::INFINITY, -1.)]
    #[case(1., 1., f64::NEG_INFINITY)]
    #[case(0., 1e-14, f64::MAX)]
    fn scalar_root_rejects_unsolvable_or_nonfinite_cases(
        #[case] a: f64,
        #[case] b: f64,
        #[case] c: f64,
    ) {
        assert_eq!(quadratic_root(a, b, c, 0.), None);
    }

    #[test]
    fn negative_discriminant_tolerance_is_in_original_coefficient_units() {
        let c = 1. + 1e-12;
        assert!(quadratic_root(1., -2., c, 0.).is_none());
        assert_relative_eq!(
            quadratic_root(1., -2., c, 1e-9).unwrap(),
            1.,
            epsilon = 1e-11
        );
        // Scaling all coefficients by 100 increases |D| by 10,000.
        assert!(quadratic_root(100., -200., 100. * c, 1e-9).is_none());
    }

    #[test]
    fn repeated_root_has_small_polynomial_residual() {
        let root = quadratic_root(1., -2., 1., 0.).unwrap();
        // A double root has only O(sqrt(epsilon)) forward accuracy when
        // coefficient arithmetic is perturbed by O(epsilon).
        assert_relative_eq!(root, 1., epsilon = 2. * f64::EPSILON.sqrt());
        assert!((root - 1.).powi(2) <= 4. * f64::EPSILON);
    }

    #[test]
    fn sub_ulp_tau_increment_still_updates_x() {
        let problem = Problem::new(
            None,
            vec![-1e8],
            matrix(1, 1, &[0.]),
            vec![0.],
            vec![Cone::Zero(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let state = State::new(vec![1e-8], vec![0.], vec![0.], 1.).unwrap();
        let mut cache = KktCache::new();
        let (mut resolvent, _) = Resolvent::new(
            &problem,
            &geometry,
            &state,
            &[0.],
            1.,
            &settings(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        let result = resolvent.solve_increment_rhs(&[0., 0.], 2., 0.1).unwrap();
        // delta_tau is about 1e-17: storing tau rounds it away, but multiplying
        // the increment by d_x = 1e8 must retain the 1e-9 correction to x.
        assert_eq!(result.tau, state.tau());
        assert_relative_eq!(result.x()[0], 1.1e-8, epsilon = 0., max_relative = 1e-14);
        assert_relative_eq!(
            result.tau.powi(2) - result.tau * (-1e8 * result.x()[0] + 2.) - 0.1,
            0.,
            epsilon = 1e-15
        );
        assert!(resolvent.fallback.get().is_none());
    }

    #[test]
    fn leading_coefficient_preserves_representable_weighted_squares() {
        assert_relative_eq!(
            tau_leading_coefficient(1e-300, &[1e200], &[1e200], &[2e-300]),
            3e100,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            tau_leading_coefficient(1e-300, &[], &[1e-200], &[1e300]),
            1e-100,
            epsilon = 0.,
            max_relative = 1e-14
        );
    }

    // P = 2, x_k = tau_k = 1, and zero data column give x(tau) = constant.
    // The first fallback is tau^2 - r_tau*tau + 2 - 4*x = 0.
    // If that fails, the full first-order equation is
    // (2 - r_tau)*tau + 1 - 4*x = 0.
    fn scalar_problem() -> Problem {
        Problem::new(
            Some(matrix(1, 1, &[2.])),
            vec![0.],
            matrix(1, 1, &[0.]),
            vec![0.],
            vec![Cone::Zero(1)],
        )
        .unwrap()
    }

    #[rstest]
    #[case::quadratic(1., 0., 2_f64.sqrt())]
    #[case::quadratic_lower_bound(0.500025, 0., 0.1)]
    #[case::quadratic_upper_bound(2500.5, 0., 10.)]
    #[case::negative_quadratic_root(0., -3., 0.1)]
    #[case::first_order(0.375, 0., 0.25)]
    #[case::first_order_lower_bound(0., 0., 0.1)]
    #[case::first_order_upper_bound(0., 2.01, 10.)]
    #[case::small_denominator(0., 2., 1.)]
    #[case::nonfinite_correction(f64::MAX, 0., 1.)]
    fn scalar_fallback_layers_and_bounds(
        #[case] constant_x: f64,
        #[case] r_tau: f64,
        #[case] expected: f64,
    ) {
        let problem = scalar_problem();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let state = State::new(vec![1.], vec![0.], vec![0.], 1.).unwrap();
        let mut cache = KktCache::new();
        let (resolvent, _) = Resolvent::new(
            &problem,
            &geometry,
            &state,
            &[0.],
            1.,
            &settings(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        let tau = resolvent.linearized_tau(&[constant_x, 0.], 0., r_tau, 0.);
        assert_relative_eq!(tau, expected, epsilon = 1e-12);
    }

    #[test]
    fn failed_primary_root_recovers_xy_using_the_fallback_tau() {
        let problem = Problem::new(
            Some(matrix(1, 1, &[2.])),
            vec![1.],
            matrix(1, 1, &[1.]),
            vec![2.],
            vec![Cone::Zero(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let state = State::new(vec![1.], vec![0.], vec![0.], 1.).unwrap();
        let mut cache = KktCache::new();
        let (mut resolvent, _) = Resolvent::new(
            &problem,
            &geometry,
            &state,
            &[0.],
            1.,
            &settings(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        // Inject failure only into the primary polynomial; K and its cached
        // direction stay valid. K^{-1}(-c,b) = (1/4, -7/4).
        resolvent.alpha = f64::NAN;
        for (constant, r_tau, expected_tau) in [
            ([1., 0.], -2., (2. / 4.25_f64).sqrt()),
            ([0.375, 1.], -3.375, 1.875 / 6.625),
        ] {
            let increment = vec![constant[0] - 1. + 0.25, constant[1] - 1.75];
            let result = resolvent
                .reconstruct_increment(increment, r_tau, 0.)
                .unwrap();
            assert!(resolvent.fallback.get().is_some());
            assert_relative_eq!(result.tau, expected_tau, epsilon = 1e-12);
            assert_relative_eq!(
                result.x()[0],
                constant[0] + 0.25 * result.tau,
                epsilon = 1e-12
            );
            assert_relative_eq!(
                result.y()[0],
                constant[1] - 1.75 * result.tau,
                epsilon = 1e-12
            );
        }
    }
}
