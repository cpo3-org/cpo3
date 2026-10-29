//! Modified Mehrotra iteration (`alg:ipm`, `subsec:mehrotra` in the paper).
//!
//! A step forms D_k = diag(s_k / y_k), shares one shifted KKT factorization,
//! probes with target zero, then solves at sigma * mu_k with the full predictor
//! as surrogate and the second-order complementarity correction. The centering
//! rule is sigma = (1 - alpha_affine)^3. Numerical retries wrap this same step.

use thiserror::Error;

use super::{
    kkt::KktCache,
    resolvent::{Resolvent, ResolventError, ResolventSolution},
    state::{State, StateError},
};

use crate::{
    Problem,
    cone::{ConeGeometry, ConeGeometryError, INTERIOR_FLOOR},
    linear_solvers::KktSettings,
    settings::Settings,
};

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum MehrotraError {
    #[error(transparent)]
    State(#[from] StateError),
    #[error(transparent)]
    ConeGeometry(#[from] ConeGeometryError),
    #[error(transparent)]
    Resolvent(#[from] ResolventError),
    #[error("the duality measure {0} must be finite and strictly positive")]
    InvalidDualityMeasure(f64),
}

impl MehrotraError {
    /// Failures eligible for another step with a larger diagonal clamp.
    fn is_numerical(&self) -> bool {
        match self {
            Self::Resolvent(error) => error.is_numerical(),
            Self::State(_) | Self::ConeGeometry(_) | Self::InvalidDualityMeasure(_) => false,
        }
    }
}

/// One accepted outer iterate and the scalars that produced it.
pub(crate) struct Step {
    pub(crate) state: State,
    pub(crate) sigma: f64,
    pub(crate) alpha_affine: f64,
    pub(crate) alpha: f64,
}

/// One outer iteration, sharing a factorization unless a numerical retry is needed.
pub(crate) fn step(
    problem: &Problem,
    geometry: &ConeGeometry,
    state: &State,
    kkt: &KktSettings,
    settings: &Settings,
    cache: &mut KktCache,
) -> Result<Step, MehrotraError> {
    debug_assert!(geometry.barrier_parameter() > 0);
    let scaling = Scaling::new(problem, geometry, state)?;
    with_clamp_retries(kkt, scaling.mu, cache, |trial, cache| {
        step_attempt(problem, geometry, state, trial, &scaling, settings, cache)
    })
}

/// Retry numerical failures with a larger diagonal clamp, keeping K(mu)
/// as the target. Accepted steps adjust the clamp using refinement diagnostics;
/// a numerical retry prevents immediate relaxation.
fn with_clamp_retries<T>(
    kkt: &KktSettings,
    mu: f64,
    cache: &mut KktCache,
    mut attempt: impl FnMut(&KktSettings, &mut KktCache) -> Result<T, MehrotraError>,
) -> Result<T, MehrotraError> {
    let starting_floor = cache.clamp.start_step(kkt.diagonal_floor);
    let before = cache.stats();
    let mut trial = *kkt;
    trial.strict_refinement = false;
    trial.diagonal_floor = starting_floor;
    loop {
        let result = attempt(&trial, cache);
        cache.finish_step(result.is_ok());
        match result {
            Err(error) if error.is_numerical() => {
                // h_i is nonnegative at an interior point. Also account for
                // negative primal diagonals rather than assuming convex input.
                let limit = cache.nonbinding_floor_limit(mu);
                let Some(next) = cache.clamp.retry(limit) else {
                    return Err(error);
                };
                trial.diagonal_floor = next;
            }
            Ok(step) => {
                let after = cache.stats();
                cache.clamp.accepted_step(starting_floor, before, after);
                return Ok(step);
            }
            Err(error) => return Err(error),
        }
    }
}

fn step_attempt(
    problem: &Problem,
    geometry: &ConeGeometry,
    state: &State,
    kkt: &KktSettings,
    scaling: &Scaling,
    settings: &Settings,
    cache: &mut KktCache,
) -> Result<Step, MehrotraError> {
    let (mut resolvent, predictor_increment) =
        Resolvent::new(problem, geometry, state, &scaling.d, scaling.mu, kkt, cache)?;
    let first = predictor(state, scaling, predictor_increment, &mut resolvent)?;
    let sigma = first.sigma;
    let alpha_affine = first.alpha_affine;
    let second = corrector(state, scaling, first, &mut resolvent)?;
    let boundary = geometry
        .step_to_boundary_toward(state.s(), &second.s_centered)?
        .min(geometry.step_to_boundary_toward(state.y(), second.centered.y())?);
    let mut centered = second.centered;
    let mut slack = second.s_centered;
    let alpha = (settings.step_fraction * boundary).min(1.0);

    // `base + alpha (target - base)`, overwriting the corrector's buffers.
    let step_toward = |base: &[f64], target: &mut [f64]| {
        crate::linear_algebra::update(target, base, |target, base| {
            alpha.mul_add(target - base, base)
        });
    };
    let (x, y) = centered.xy.split_at_mut(centered.x_len);
    step_toward(state.x(), x);
    step_toward(state.s(), &mut slack);
    step_toward(state.y(), y);
    let next = State::from_stacked(
        centered.xy,
        centered.x_len,
        slack,
        accepted_tau(state.tau(), centered.tau, alpha)?,
        geometry,
    )?;

    Ok(Step {
        state: next,
        sigma,
        alpha_affine,
        alpha,
    })
}

/// Keep zero subproblem roots admissible while the updated iterate stays
/// strictly positive. Check finiteness before `max`, which would hide NaNs.
fn accepted_tau(current: f64, candidate: f64, alpha: f64) -> Result<f64, StateError> {
    let tau = current + alpha * (candidate - current);
    if !tau.is_finite() {
        return Err(StateError::NonFiniteResult);
    }
    Ok(tau.max(INTERIOR_FLOOR))
}

/// Diagonal barrier scaling `D_k` and Jordan inverse `y_k^{-1}` at one iterate.
pub(crate) struct Scaling {
    /// Complementarity in the current spherical coordinates.
    mu: f64,
    pub(crate) d: Vec<f64>,
    pub(crate) y_inverse: Vec<f64>,
}

impl Scaling {
    pub(crate) fn new(
        problem: &Problem,
        geometry: &ConeGeometry,
        state: &State,
    ) -> Result<Self, MehrotraError> {
        let mu = state.duality_measure(geometry);
        if !mu.is_finite() || mu <= 0.0 {
            return Err(MehrotraError::InvalidDualityMeasure(mu));
        }
        // Initialization and acceptance establish the interior invariant;
        // checked sphere normalization preserves it between iterations.
        debug_assert!(state.validate(problem, geometry).is_ok());
        let mut d = vec![0.0; state.s().len()];
        let mut y_inverse = vec![0.0; state.y().len()];
        geometry.scaling_and_inverse(state.s(), state.y(), &mut d, &mut y_inverse)?;
        Ok(Self { mu, d, y_inverse })
    }
}

/// The affine trial point and the centering parameter it induces.
pub(crate) struct Predictor {
    pub(crate) affine: ResolventSolution,
    /// Second-order correction, reusing the slack-displacement allocation.
    pub(crate) correction: Vec<f64>,
    pub(crate) alpha_affine: f64,
    pub(crate) sigma: f64,
}

/// Solve the surrogate system at target zero with the current iterate as
/// surrogate, then evaluate the step-length centering heuristic.
///
/// The current complementarity is both the KKT shift and the anchor weight.
pub(crate) fn predictor(
    state: &State,
    scaling: &Scaling,
    increment: Vec<f64>,
    resolvent: &mut Resolvent<'_>,
) -> Result<Predictor, MehrotraError> {
    let r_tau = scaling.mu * state.tau();
    let affine = resolvent.reconstruct_increment(increment, r_tau, 0.0)?;

    let mut correction = vec![-0.0; state.s().len()];
    let mut alpha_affine = 1.0_f64;
    let mut finite = true;
    // Each affine displacement is needed only for its boundary and correction.
    faer::zip!(
        faer::ColMut::from_slice_mut(&mut correction),
        faer::ColRef::from_slice(state.s()),
        faer::ColRef::from_slice(state.y()),
        faer::ColRef::from_slice(affine.y()),
        faer::ColRef::from_slice(&scaling.d),
        faer::ColRef::from_slice(&scaling.y_inverse)
    )
    .for_each(|faer::unzip!(correction, s, y, target_y, h, inverse)| {
        let dy = target_y - y;
        let ds = -h * dy - s;
        finite &= ds.is_finite() & dy.is_finite();
        if *inverse > 0.0 {
            if ds < 0.0 {
                alpha_affine = alpha_affine.min(-s / ds);
            }
            if dy < 0.0 {
                alpha_affine = alpha_affine.min(-y / dy);
            }
            *correction = -(inverse * (ds * dy));
        }
    });
    if !finite {
        return Err(ConeGeometryError::NonFiniteInput { argument: "dv" }.into());
    }
    // A long feasible affine step supports a smaller centering target.
    // Keep mu as current complementarity; only the dimensionless sigma changes.
    let sigma = (1.0 - alpha_affine).powi(3);
    if !sigma.is_finite() {
        return Err(StateError::NonFiniteResult.into());
    }

    Ok(Predictor {
        affine,
        correction,
        alpha_affine,
        sigma,
    })
}

/// The centered iterate and its slack, both at target `mu_c = sigma mu_k`.
pub(crate) struct Corrector {
    pub(crate) centered: ResolventSolution,
    pub(crate) s_centered: Vec<f64>,
}

/// Consume the predictor to solve at target `sigma * mu`, using its affine
/// point as the anchor and reusing its second-order correction buffer for slack.
pub(crate) fn corrector(
    state: &State,
    scaling: &Scaling,
    predictor: Predictor,
    resolvent: &mut Resolvent<'_>,
) -> Result<Corrector, MehrotraError> {
    let Predictor {
        affine,
        mut correction,
        sigma,
        ..
    } = predictor;
    let mu = scaling.mu;
    let mu_centered = sigma * mu;
    let residual_weight = mu - mu_centered;
    let (anchor_x, anchor_y) = (affine.x(), affine.y());
    let m = state.y().len();
    // Form the increment RHS from actual feasibility residuals. This
    // cancels H y = s and common shifted anchor terms algebraically.
    let mu_over_y = |index: usize| {
        if scaling.y_inverse[index] > 0.0 {
            mu_centered / state.y()[index]
        } else {
            0.0
        }
    };
    let mut rhs = Vec::with_capacity(state.x().len() + m);
    rhs.extend(
        anchor_x
            .iter()
            .zip(state.x())
            .zip(resolvent.dual_residual())
            .map(|((anchor, current), rd)| -rd + mu * (anchor - current) - mu_centered * anchor),
    );
    rhs.extend((0..m).map(|index| {
        resolvent.primal_rhs()[index] - mu * (anchor_y[index] - state.y()[index])
            + mu_centered * anchor_y[index]
            - mu_over_y(index)
            - correction[index]
    }));
    let r_tau = residual_weight * affine.tau;
    let centered = resolvent.solve_increment_rhs(&rhs, r_tau, mu_centered)?;
    // The second-order correction is no longer needed after the solve.
    // Reuse its allocation for slack, preserving the scalar evaluation order.
    for (index, value) in correction.iter_mut().enumerate() {
        let second_order_correction = *value;
        *value = mu_over_y(index) - scaling.d[index] * (centered.y()[index] - state.y()[index])
            + second_order_correction;
    }
    Ok(Corrector {
        centered,
        s_centered: correction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::central_path::kkt::KktError;
    use crate::linear_algebra::matrix;
    use crate::linear_solvers::LinearSystemError;
    use crate::{Cone, LinearSolver};
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

    #[rstest]
    #[case::no_step(2., 0., 0., 2.)]
    #[case::partial_step(2., 0., 0.5, 1.)]
    #[case::near_full_step(2., 0., 0.999, 0.002)]
    #[case::zero_root(2., 0., 1., INTERIOR_FLOOR)]
    #[case::positive_root(2., 4., 0.25, 2.5)]
    fn accepted_tau_interpolates_and_keeps_zero_roots_interior(
        #[case] current: f64,
        #[case] candidate: f64,
        #[case] alpha: f64,
        #[case] expected: f64,
    ) {
        let tau = accepted_tau(current, candidate, alpha).unwrap();
        assert!(tau >= INTERIOR_FLOOR);
        assert_relative_eq!(tau, expected, epsilon = 0., max_relative = 1e-14);
    }

    #[rstest]
    fn accepted_tau_rejects_nonfinite_candidates_before_flooring(
        #[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY)] candidate: f64,
    ) {
        assert_eq!(
            accepted_tau(1., candidate, 0.5),
            Err(StateError::NonFiniteResult)
        );
    }

    #[rstest]
    #[case::slack_limited(10., 1., 0)]
    #[case::dual_limited(0.01, 10., 1)]
    fn accepted_step_damps_all_coordinates_by_the_primal_dual_boundary(
        #[values(LinearSolver::Faer, LinearSolver::Feral)] backend: LinearSolver,
        #[values(0.5, 0.999)] fraction: f64,
        #[case] second_slack: f64,
        #[case] second_multiplier: f64,
        #[case] limiting_side: usize,
    ) {
        let problem = Problem::new(
            Some(matrix(2, 2, &[2., 1., 1., 3.])),
            vec![-3., -7.],
            matrix(3, 2, &[1., 0., 0., -2., 1., 2.]),
            vec![1., -4., 5.],
            vec![Cone::Nonnegative(2), Cone::Zero(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let state = State::new(
            vec![0.5, -0.25],
            vec![0.01, second_slack, 0.],
            vec![0.01, second_multiplier, -1.],
            1.2,
        )
        .unwrap();
        let scaling = Scaling::new(&problem, &geometry, &state).unwrap();
        let kkt = settings(backend);
        let mut cache = KktCache::new();
        let (mut resolvent, increment) = Resolvent::new(
            &problem, &geometry, &state, &scaling.d, scaling.mu, &kkt, &mut cache,
        )
        .unwrap();
        let first = predictor(&state, &scaling, increment, &mut resolvent).unwrap();
        let sigma = first.sigma;
        let alpha_affine = first.alpha_affine;
        let second = corrector(&state, &scaling, first, &mut resolvent).unwrap();

        // Independently intersect the four nonnegative-coordinate constraints.
        let mut boundaries = [1.0_f64; 2];
        for (side, (base, target)) in [
            (state.s(), second.s_centered.as_slice()),
            (state.y(), second.centered.y()),
        ]
        .into_iter()
        .enumerate()
        {
            for i in 0..2 {
                if target[i] < 0. {
                    boundaries[side] = boundaries[side].min(base[i] / (base[i] - target[i]));
                }
            }
        }
        assert!(
            boundaries[limiting_side] < 0.95,
            "fixture must exercise a binding cone boundary"
        );
        assert_eq!(boundaries[1 - limiting_side], 1.);
        let expected_alpha = fraction * boundaries[0].min(boundaries[1]);
        let settings = Settings {
            step_fraction: fraction,
            ..Settings::default()
        };
        let mut fresh_cache = KktCache::new();
        let accepted = step(
            &problem,
            &geometry,
            &state,
            &kkt,
            &settings,
            &mut fresh_cache,
        )
        .unwrap();
        assert_relative_eq!(accepted.alpha, expected_alpha, epsilon = 1e-12);
        assert_relative_eq!(accepted.alpha_affine, alpha_affine, epsilon = 1e-12);
        assert_relative_eq!(accepted.sigma, sigma, epsilon = 1e-12);
        for (base, target, result) in [
            (state.x(), second.centered.x(), accepted.state.x()),
            (state.y(), second.centered.y(), accepted.state.y()),
            (state.s(), second.s_centered.as_slice(), accepted.state.s()),
        ] {
            for ((base, target), result) in base.iter().zip(target).zip(result) {
                assert_relative_eq!(
                    *result,
                    (1. - expected_alpha) * base + expected_alpha * target,
                    epsilon = 1e-11
                );
            }
        }
        assert_relative_eq!(
            accepted.state.tau(),
            (1. - expected_alpha) * state.tau() + expected_alpha * second.centered.tau,
            epsilon = 1e-11
        );
        accepted.state.validate(&problem, &geometry).unwrap();
        assert!(
            accepted.state.s()[..2]
                .iter()
                .chain(&accepted.state.y()[..2])
                .all(|v| *v > 0.)
        );
        assert!(accepted.state.tau() > 0.);
        assert_eq!(accepted.state.s()[2], 0.);
    }

    #[rstest]
    fn predictor_corrector_satisfy_surrogate_and_complementarity_equations(
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
        let state = State::new(vec![0.5, -0.25], vec![1., 4., 0.], vec![1., 2., -1.], 1.2).unwrap();
        let scaling = Scaling::new(&problem, &geometry, &state).unwrap();
        let mu = 4.5;
        assert_eq!(scaling.mu, mu);
        assert_eq!(scaling.d, [1., 2., 0.]);
        assert_eq!(scaling.y_inverse, [1., 0.5, 0.]);
        let mut kkt = settings(backend);
        kkt.eliminate_bound_rows = eliminate;
        let mut cache = KktCache::new();
        let (mut resolvent, increment) = Resolvent::new(
            &problem, &geometry, &state, &scaling.d, mu, &kkt, &mut cache,
        )
        .unwrap();
        let first = predictor(&state, &scaling, increment, &mut resolvent).unwrap();

        // Evaluate the original surrogate equations directly, without the
        // production KKT assembly, increment RHS, or scalar polynomial.
        let px = |x: &[f64]| {
            if quadratic {
                [2. * x[0] + x[1], x[0] + 3. * x[1]]
            } else {
                [0.; 2]
            }
        };
        let ax = |x: &[f64]| [x[0], -2. * x[1], x[0] + 2. * x[1]];
        let check_surrogate = |point: &ResolventSolution,
                               slack: &[f64],
                               anchor: &[f64],
                               anchor_tau: f64,
                               target: f64| {
            let x = point.x();
            let y = point.y();
            let p = px(x);
            let dual = [
                p[0] + y[0] + y[2] - 3. * point.tau,
                p[1] - 2. * y[1] + 2. * y[2] - 7. * point.tau,
            ];
            for i in 0..2 {
                assert_relative_eq!(
                    dual[i] + mu * x[i],
                    (mu - target) * anchor[i],
                    epsilon = 1e-10
                );
            }
            let a = ax(x);
            for i in 0..3 {
                assert_relative_eq!(
                    a[i] + slack[i] - problem.b()[i] * point.tau - mu * y[i],
                    -(mu - target) * anchor[2 + i],
                    epsilon = 1e-10
                );
            }
            let scalar = mu * point.tau.powi(2)
                - point.tau
                    * (-3. * x[0] - 7. * x[1] + y[0] - 4. * y[1]
                        + 5. * y[2]
                        + (mu - target) * anchor_tau)
                - x[0] * p[0]
                - x[1] * p[1]
                - target;
            assert_relative_eq!(scalar, 0., epsilon = 1e-10);
        };
        // Recover affine slack from the primal surrogate equation, then check
        // the independent linearized complementarity and second-order product.
        let a = ax(first.affine.x());
        let affine_slack: Vec<_> = (0..3)
            .map(|i| {
                problem.b()[i] * first.affine.tau - a[i] + mu * (first.affine.y()[i] - state.y()[i])
            })
            .collect();
        check_surrogate(
            &first.affine,
            &affine_slack,
            &[0.5, -0.25, 1., 2., -1.],
            1.2,
            0.,
        );
        let mut boundary = 1.0_f64;
        let mut products = [0.; 2];
        for i in 0..2 {
            let ds = affine_slack[i] - state.s()[i];
            let dy = first.affine.y()[i] - state.y()[i];
            assert_relative_eq!(
                state.y()[i] * affine_slack[i] + state.s()[i] * dy,
                0.,
                epsilon = 1e-11
            );
            products[i] = ds * dy;
            assert_relative_eq!(
                state.y()[i] * first.correction[i],
                -products[i],
                epsilon = 1e-11
            );
            for (base, displacement) in [(state.s()[i], ds), (state.y()[i], dy)] {
                if displacement < 0. {
                    boundary = boundary.min(-base / displacement);
                }
            }
        }
        assert_relative_eq!(affine_slack[2], 0., epsilon = 1e-11);
        assert_eq!(first.correction[2], 0.);
        assert_relative_eq!(first.alpha_affine, boundary, epsilon = 1e-12);
        assert_relative_eq!(first.sigma, (1. - boundary).powi(3), epsilon = 1e-12);
        assert!(boundary > 0. && boundary < 1.);
        let target = mu * first.sigma;
        let anchor = first.affine.clone();
        let second = corrector(&state, &scaling, first, &mut resolvent).unwrap();
        check_surrogate(
            &second.centered,
            &second.s_centered,
            &anchor.xy,
            anchor.tau,
            target,
        );
        for (i, product) in products.iter().enumerate() {
            assert_relative_eq!(
                state.y()[i] * second.s_centered[i]
                    + state.s()[i] * (second.centered.y()[i] - state.y()[i])
                    + product,
                target,
                epsilon = 1e-11
            );
        }
        assert_eq!(second.s_centered[2], 0.);
    }

    #[rstest]
    fn retries_restore_the_accepted_direction_and_skip_ineffective_clamps(
        #[values(LinearSolver::Faer, LinearSolver::Feral)] backend: LinearSolver,
    ) {
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
        let mut kkt = settings(backend);
        kkt.diagonal_floor = 1e-8;
        kkt.strict_refinement = true;
        Resolvent::new(&problem, &geometry, &state, &[0.], 1., &kkt, &mut cache).unwrap();
        cache.finish_step(true);
        let accepted = cache.direction().to_vec();
        let mu = 2e-6;
        let mut floors = Vec::new();
        let result = with_clamp_retries(&kkt, mu, &mut cache, |trial, cache| {
            assert_eq!(cache.direction(), accepted);
            assert!(!trial.strict_refinement);
            assert_eq!(trial.refine_tol, kkt.refine_tol);
            floors.push(trial.diagonal_floor);
            Resolvent::new(&problem, &geometry, &state, &[0.], mu, trial, cache)?;
            let d = cache.direction();
            // Each attempt must still solve K(mu)d = (-c,b), even when the
            // preconditioner's floor exceeds mu on the second attempt.
            assert_relative_eq!((2. + mu) * d[0] + d[1], -1., epsilon = 1e-10);
            assert_relative_eq!(d[0] - mu * d[1], 2., epsilon = 1e-10);
            if floors.len() == 1 {
                Err(ResolventError::InvalidTauPolynomial.into())
            } else {
                Ok(d.to_vec())
            }
        })
        .unwrap();
        assert_relative_eq!(floors.as_slice(), &[1e-8, 1e-5][..], epsilon = 1e-18);
        assert_eq!(cache.direction(), result);
        assert_ne!(cache.direction(), accepted);
        assert_relative_eq!(cache.clamp.floor(), 1e-5, epsilon = 1e-18);
        assert_eq!(cache.stats().clamp_increases, 3);
        assert_eq!(cache.stats().clamp_decreases, 0);
    }

    #[test]
    fn numerical_retries_stop_at_the_maximum_and_return_the_last_error() {
        let mut kkt = settings(LinearSolver::Faer);
        kkt.diagonal_floor = 2e-6;
        let mut cache = KktCache::new();
        let mut floors = Vec::new();
        let result: Result<(), _> = with_clamp_retries(&kkt, 1e-8, &mut cache, |trial, _| {
            floors.push(trial.diagonal_floor);
            assert!(
                floors.len() <= 3,
                "retry must terminate at the maximum clamp"
            );
            Err(
                ResolventError::LinearSystem(LinearSystemError::RefinementDiverged(
                    floors.len() as f64
                ))
                .into(),
            )
        });
        assert_relative_eq!(floors.as_slice(), &[2e-6, 2e-5, 1e-4][..], epsilon = 1e-18);
        assert_eq!(
            result,
            Err(ResolventError::LinearSystem(LinearSystemError::RefinementDiverged(3.)).into())
        );
        assert_eq!(cache.clamp.floor(), 1e-4);
    }

    #[rstest]
    #[case(StateError::NonFiniteResult.into())]
    #[case(MehrotraError::InvalidDualityMeasure(0.))]
    #[case(ResolventError::Kkt(KktError::InvalidParameters).into())]
    #[case(ResolventError::LinearSystem(LinearSystemError::InvalidRightHandSide { expected: 2, actual: 1 }).into())]
    fn invalid_inputs_are_not_retried(#[case] error: MehrotraError) {
        let kkt = settings(LinearSolver::Faer);
        let mut cache = KktCache::new();
        let mut calls = 0;
        let result: Result<(), _> = with_clamp_retries(&kkt, 1., &mut cache, |_, _| {
            calls += 1;
            Err(error.clone())
        });
        assert_eq!(result, Err(error));
        assert_eq!(calls, 1);
        assert_eq!(cache.stats().clamp_increases, 0);
        assert_eq!(cache.stats().clamp_decreases, 0);
    }
}
