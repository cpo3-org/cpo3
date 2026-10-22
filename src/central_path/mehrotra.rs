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
    const MAX_CLAMP: f64 = 1e-4;
    let minimum = 1e-20;
    let mut floor = cache.clamp_floor.unwrap_or(kkt.diagonal_floor).max(minimum);
    cache.clamp_floor = Some(floor);
    let starting_floor = floor;
    let before = cache.stats();
    let settled_before = before.refinement_settled;
    let mut trial = *kkt;
    trial.strict_refinement = false;
    loop {
        trial.diagonal_floor = floor;
        let result = attempt(&trial, cache);
        cache.finish_step(result.is_ok());
        match result {
            Err(error) if retryable(&error) && floor < MAX_CLAMP => {
                // h_i is nonnegative at an interior point. Also account for
                // negative primal diagonals rather than assuming convex input.
                let limit = cache.nonbinding_floor_limit(mu);
                // Preserve the x10 grid while skipping identical matrices.
                loop {
                    floor = (floor * 10.0).min(MAX_CLAMP);
                    cache.clamp_increases += 1;
                    if floor > limit || floor >= MAX_CLAMP {
                        break;
                    }
                }
                cache.clamp_floor = Some(floor);
                if floor <= limit {
                    return Err(error);
                }
            }
            Ok(step) => {
                let after = cache.stats();
                let clamp_limited =
                    after.refinement_clamp_limited > before.refinement_clamp_limited;
                let solve_limited =
                    after.refinement_solve_limited > before.refinement_solve_limited;
                if solve_limited && !clamp_limited {
                    let next = (floor * 10.0).min(MAX_CLAMP);
                    cache.clamp_increases += usize::from(next > floor);
                    cache.clamp_floor = Some(next);
                } else if floor == starting_floor
                    && ((clamp_limited && !solve_limited)
                        || after.refinement_settled == settled_before)
                {
                    let next = (floor / 10.0).max(minimum);
                    cache.clamp_decreases += usize::from(next < floor);
                    cache.clamp_floor = Some(next);
                }
                return Ok(step);
            }
            Err(error) => return Err(error),
        }
    }
}

fn retryable(error: &MehrotraError) -> bool {
    matches!(error, MehrotraError::Resolvent(inner) if inner.is_numerical())
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
    let mut resolvent =
        Resolvent::new(problem, geometry, state, &scaling.d, scaling.mu, kkt, cache)?;
    if geometry.barrier_parameter() == 0 {
        // At mu=0 with only equality rows, the constant KKT column is the
        // original primal-dual solution. Its homogeneous scale is arbitrary.
        let point = State::from_stacked(
            resolvent.constant_direction().to_vec(),
            state.x().len(),
            vec![0.0; state.s().len()],
            1.0,
            geometry,
        )?;
        return Ok(Step {
            state: point,
            sigma: 0.0,
            alpha_affine: 1.0,
            alpha: 1.0,
        });
    }
    let mut first = predictor(geometry, state, scaling, &mut resolvent)?;
    let second = corrector(state, scaling, &mut first, &mut resolvent)?;
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
        sigma: first.sigma,
        alpha_affine: first.alpha_affine,
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
        if !mu.is_finite() || mu < 0.0 {
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
    #[cfg(test)]
    pub(crate) s_affine: Vec<f64>,
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
    geometry: &ConeGeometry,
    state: &State,
    scaling: &Scaling,
    resolvent: &mut Resolvent<'_>,
) -> Result<Predictor, MehrotraError> {
    // Current-anchor predictor RHS: (-r_dual, b tau - A x).
    // H y = s cancels algebraically, before floating-point evaluation.
    let mut rhs = Vec::with_capacity(state.x().len() + state.y().len());
    rhs.extend(resolvent.dual_residual().iter().map(|rd| -rd));
    rhs.extend_from_slice(resolvent.primal_rhs());
    let r_tau = scaling.mu * state.tau();
    let affine = resolvent.solve_increment_rhs(&rhs, r_tau, 0.0)?;

    let dy = difference(affine.y(), state.y());
    let mut ds = dy
        .iter()
        .zip(&scaling.d)
        .zip(state.s())
        .map(|((dy, h), s)| -h * dy - s)
        .collect::<Vec<_>>();
    #[cfg(test)]
    let s_affine = dy.iter().zip(&scaling.d).map(|(dy, h)| -h * dy).collect();

    let alpha_affine = geometry
        .step_to_boundary_from_interior(state.s(), &ds)?
        .min(geometry.step_to_boundary_from_interior(state.y(), &dy)?);
    // A long feasible affine step supports a smaller centering target.
    // Keep mu as current complementarity; only the dimensionless sigma changes.
    let sigma = if scaling.mu > 0.0 {
        (1.0 - alpha_affine).powi(3)
    } else {
        0.0
    };
    if !sigma.is_finite() {
        return Err(StateError::NonFiniteResult.into());
    }

    // The affine boundary is complete. Reuse
    // ds for the correction and release dy before any centered solve.
    if geometry.barrier_parameter() > 0 {
        geometry.negated_triple_product_in_place(&scaling.y_inverse, &mut ds, &dy)?;
    } else {
        ds.fill(0.0);
    }
    drop(dy);

    Ok(Predictor {
        affine,
        #[cfg(test)]
        s_affine,
        correction: ds,
        alpha_affine,
        sigma,
    })
}

/// The centered iterate and its slack, both at target `mu_c = sigma mu_k`.
pub(crate) struct Corrector {
    pub(crate) centered: ResolventSolution,
    pub(crate) s_centered: Vec<f64>,
}

/// Solve the surrogate system at target `sigma * complementarity` with the given surrogate
/// and Mehrotra's second-order correction in the dual block.
pub(crate) fn corrector(
    state: &State,
    scaling: &Scaling,
    predictor: &mut Predictor,
    resolvent: &mut Resolvent<'_>,
) -> Result<Corrector, MehrotraError> {
    let mu_centered = predictor.sigma * scaling.mu;

    let correction = std::mem::take(&mut predictor.correction);

    let (centered, s_centered) = centered_solve(
        state,
        scaling,
        (
            predictor.affine.x(),
            predictor.affine.y(),
            predictor.affine.tau,
        ),
        mu_centered,
        correction,
        resolvent,
    )?;

    Ok(Corrector {
        centered,
        s_centered,
    })
}

/// Solve the surrogate system at target `mu_centered` with `correction` in
/// the dual block; returns the point and its slack.
fn centered_solve(
    state: &State,
    scaling: &Scaling,
    (anchor_x, anchor_y, anchor_tau): (&[f64], &[f64], f64),
    mu_centered: f64,
    mut correction: Vec<f64>,
    resolvent: &mut Resolvent<'_>,
) -> Result<(ResolventSolution, Vec<f64>), MehrotraError> {
    let mu = scaling.mu;
    let residual_weight = mu - mu_centered;
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
    let r_tau = residual_weight * anchor_tau;
    let centered = resolvent.solve_increment_rhs(&rhs, r_tau, mu_centered)?;
    // The second-order correction is no longer needed after the solve.
    // Reuse its allocation for slack, preserving the scalar evaluation order.
    for (index, value) in correction.iter_mut().enumerate() {
        let second_order_correction = *value;
        *value = mu_over_y(index) - scaling.d[index] * (centered.y()[index] - state.y()[index])
            + second_order_correction;
    }
    Ok((centered, correction))
}

fn difference(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left - right)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cone, LinearSolver, Problem, linear_solvers::LinearSystemError};

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn adaptive_clamp_retries_then_relaxes_without_changing_mu() {
        for mu in [0.0, 1e-14, 1e-3] {
            let settings = KktSettings::new(LinearSolver::Faer);
            let mut cache = KktCache::new();
            let mut floors = Vec::new();
            with_clamp_retries(&settings, mu, &mut cache, |trial, _| {
                assert!(!trial.strict_refinement);
                floors.push(trial.diagonal_floor);
                if floors.len() < 3 {
                    Err(ResolventError::InvalidTauPolynomial.into())
                } else {
                    Ok(())
                }
            })
            .unwrap();
            assert_eq!(floors.len(), 3);
            assert!((floors[0] - 1e-12).abs() < 1e-25);
            assert!((floors[2] - 1e-10).abs() < 1e-23);
            assert_eq!(cache.clamp_floor, Some(floors[2]));
            with_clamp_retries(&settings, mu, &mut cache, |trial, _| {
                assert_eq!(trial.diagonal_floor, floors[2]);
                Ok(())
            })
            .unwrap();
            assert_eq!(cache.clamp_floor, Some(floors[2] / 10.0));
            let mut fresh = KktCache::new();
            with_clamp_retries(&settings, mu, &mut fresh, |trial, _| {
                assert_eq!(trial.diagonal_floor, 1e-12);
                Ok(())
            })
            .unwrap();
            assert_eq!(fresh.clamp_floor, Some(1e-13));
        }
    }

    #[test]
    fn adaptive_clamp_is_bounded_and_skips_nonnumerical_failures() {
        let settings = KktSettings::new(LinearSolver::Faer);
        let mut cache = KktCache::new();
        let error = MehrotraError::Resolvent(LinearSystemError::Singular.into());
        let mut floors = Vec::new();
        let result: Result<(), _> = with_clamp_retries(&settings, 1e-14, &mut cache, |trial, _| {
            floors.push(trial.diagonal_floor);
            Err(error.clone())
        });
        assert_eq!(result, Err(error));
        assert_eq!(*floors.last().unwrap(), 1e-4);
        assert!(floors.len() <= 10);
        assert!(floors.windows(2).all(|w| w[1] > w[0]));
        let error = MehrotraError::Resolvent(ResolventError::IncorrectRhsLength {
            argument: "test",
            expected: 2,
            actual: 1,
        });
        let mut count = 0;
        let result: Result<(), _> =
            with_clamp_retries(&settings, 1e-14, &mut KktCache::new(), |_, _| {
                count += 1;
                Err(error.clone())
            });
        assert_eq!(count, 1);
        assert_eq!(result, Err(error));
    }

    #[test]
    fn finite_inaccurate_refinement_retains_clamp_without_retry() {
        let problem = Problem::new(
            None,
            vec![-1.0],
            crate::linear_algebra::test_matrix(0, 1, vec![]).unwrap(),
            vec![],
            vec![],
        )
        .unwrap();
        let state = State::new(vec![0.0], vec![], vec![], 1.0).unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let settings = KktSettings {
            refine_iters: 0,
            ..KktSettings::new(LinearSolver::Faer)
        };
        let mut cache = KktCache::new();
        cache.clamp_floor = Some(1e-8);
        let mut calls = 0;
        with_clamp_retries(&settings, 1e-14, &mut cache, |trial, cache| {
            calls += 1;
            let _resolvent = Resolvent::new(&problem, &geometry, &state, &[], 1e-14, trial, cache)?;
            Ok(())
        })
        .unwrap();
        assert!(cache.stats().refinement_settled > 0);
        assert_eq!(calls, 1);
        assert_eq!(cache.clamp_floor, Some(1e-8));
    }

    #[test]
    fn corrector_uses_full_affine_anchor() {
        let problem = quadratic_problem();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let state = interior_state(&problem);
        let scaling = Scaling::new(&problem, &geometry, &state).unwrap();
        let mu = scaling.mu;
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            &problem,
            &geometry,
            &state,
            &scaling.d,
            mu,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        let mut first = predictor(&geometry, &state, &scaling, &mut resolvent).unwrap();
        // The corrector must use the full anchor independently of the
        // reachable fraction (even when this particular predictor is interior).
        first.alpha_affine = 0.25;
        let result = corrector(&state, &scaling, &mut first, &mut resolvent).unwrap();
        let target = first.sigma * state.duality_measure(&geometry);
        let weight = mu - target;
        let rx = first
            .affine
            .x()
            .iter()
            .map(|x| weight * x)
            .collect::<Vec<_>>();
        let ry = first
            .affine
            .y()
            .iter()
            .enumerate()
            .map(|(i, y)| {
                weight * y + state.s()[i] + target * scaling.y_inverse[i]
                    - scaling.y_inverse[i]
                        * (first.s_affine[i] - state.s()[i])
                        * (first.affine.y()[i] - state.y()[i])
            })
            .collect::<Vec<_>>();
        let expected = resolvent
            .solve(&rx, &ry, weight * first.affine.tau, target)
            .unwrap();
        for (actual, expected) in result
            .centered
            .x()
            .iter()
            .chain(result.centered.y())
            .chain(std::iter::once(&result.centered.tau))
            .zip(
                expected
                    .x()
                    .iter()
                    .chain(expected.y())
                    .chain(std::iter::once(&expected.tau)),
            )
        {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn predictor_corrector_share_one_factorization_and_three_solves() {
        let problem = quadratic_problem();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let mut state = interior_state(&problem);
        state.normalize_sphere(&geometry).unwrap();
        let mut cache = KktCache::new();
        let result = step(
            &problem,
            &geometry,
            &state,
            &KktSettings::new(LinearSolver::Faer),
            &Settings::default(),
            &mut cache,
        )
        .unwrap();
        assert_eq!(cache.stats().factorizations, 1);
        assert_eq!(cache.stats().primary_solves, 3);
        assert_eq!(result.sigma, (1.0 - result.alpha_affine).powi(3));
    }

    #[test]
    fn accepted_tau_damps_zero_candidates_and_rejects_nonfinite_updates() {
        assert!((accepted_tau(1.0, 0.0, 0.99).unwrap() - 0.01).abs() < 1e-15);
        assert_eq!(accepted_tau(INTERIOR_FLOOR, 0.0, 0.99), Ok(INTERIOR_FLOOR));
        assert_eq!(accepted_tau(1.0, 0.0, 1.0), Ok(INTERIOR_FLOOR));
        for candidate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                accepted_tau(1.0, candidate, 0.99),
                Err(StateError::NonFiniteResult)
            );
        }
    }

    fn quadratic_problem() -> Problem {
        Problem::new(
            Some(crate::linear_algebra::test_matrix(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap()),
            vec![0.5, -1.0],
            crate::linear_algebra::test_matrix(3, 2, vec![1.0, -2.0, 3.0, 0.25, -1.0, 1.0])
                .unwrap(),
            vec![1.5, -0.75, 2.0],
            vec![Cone::Zero(1), Cone::Nonnegative(2)],
        )
        .unwrap()
    }

    fn linear_problem() -> Problem {
        Problem::new(
            None,
            vec![1.0, 2.0],
            crate::linear_algebra::test_matrix(2, 2, vec![1.0, 1.0, -1.0, 2.0]).unwrap(),
            vec![1.0, 0.5],
            vec![Cone::Nonnegative(2)],
        )
        .unwrap()
    }

    fn interior_state(problem: &Problem) -> State {
        let s = problem
            .cones()
            .iter()
            .flat_map(|cone| match cone {
                Cone::Zero(dim) => vec![0.0; *dim],
                _ => (0..cone.vector_dim())
                    .map(|index| 0.5 + 0.25 * index as f64)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let y = (0..problem.constraint_count())
            .map(|index| 1.5 - 0.25 * index as f64)
            .collect();
        let x = (0..problem.variable_count())
            .map(|index| 0.3 * (index as f64 + 1.0))
            .collect();
        State::new(x, s, y, 0.8).unwrap()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= TOLERANCE * (1.0 + actual.abs().max(expected.abs())),
            "expected {expected}, got {actual}"
        );
    }

    fn run_predictor(problem: &Problem) -> (State, f64, Scaling, Predictor) {
        let geometry = ConeGeometry::new(problem).unwrap();
        let state = interior_state(problem);
        let mu = state.duality_measure(&geometry);
        let scaling = Scaling::new(problem, &geometry, &state).unwrap();
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            problem,
            &geometry,
            &state,
            &scaling.d,
            mu,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        let result = predictor(&geometry, &state, &scaling, &mut resolvent).unwrap();
        (state, mu, scaling, result)
    }

    fn run_both(problem: &Problem) -> (State, f64, Predictor, Corrector) {
        let geometry = ConeGeometry::new(problem).unwrap();
        let state = interior_state(problem);
        let mu = state.duality_measure(&geometry);
        let scaling = Scaling::new(problem, &geometry, &state).unwrap();
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            problem,
            &geometry,
            &state,
            &scaling.d,
            mu,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        let mut first = predictor(&geometry, &state, &scaling, &mut resolvent).unwrap();
        let correction_allocation = first.correction.as_ptr();
        let second = corrector(&state, &scaling, &mut first, &mut resolvent).unwrap();
        assert_eq!(second.s_centered.as_ptr(), correction_allocation);
        (state, mu, first, second)
    }

    #[test]
    fn predictor_slack_is_the_newton_update_on_affine_complementarity() {
        for problem in [quadratic_problem(), linear_problem()] {
            let (state, _, _, result) = run_predictor(&problem);
            for index in 0..problem.constraint_count() {
                let (s, y) = (state.s()[index], state.y()[index]);
                let lhs = result.s_affine[index] * y + s * result.affine.y()[index];
                assert_close(lhs, s * y);
            }
            assert!(result.alpha_affine > 0.0 && result.alpha_affine <= 1.0);
            assert!((0.0..=1.0).contains(&result.sigma));
            assert_close(result.sigma, (1.0 - result.alpha_affine).powi(3));
        }
    }

    #[test]
    fn predictor_slack_vanishes_on_zero_rows() {
        let problem = quadratic_problem();
        let (_, _, _, result) = run_predictor(&problem);
        assert_eq!(result.s_affine[0], 0.0);
    }

    #[test]
    fn corrector_slack_carries_the_second_order_correction() {
        for problem in [quadratic_problem(), linear_problem()] {
            let (state, mu, first, second) = run_both(&problem);
            // A zero centering target can place the trial on the boundary.
            // The accepted (damped) iterate is tested separately for interiority.
            assert!(second.centered.tau.is_finite() && second.centered.tau >= 0.0);
            let nonneg = |index: usize| problem.cones()[0] != Cone::Zero(1) || index > 0;
            for index in 0..problem.constraint_count() {
                let (s, y) = (state.s()[index], state.y()[index]);
                let ds = first.s_affine[index] - s;
                let dy = first.affine.y()[index] - y;
                let lhs = second.s_centered[index] * y + s * second.centered.y()[index];
                let expected = if nonneg(index) {
                    s * y + (first.sigma * mu) - ds * dy
                } else {
                    0.0
                };
                assert_close(lhs, expected);
            }
        }
    }

    #[test]
    fn corrector_with_unit_sigma_and_zero_displacement_is_pure_centering() {
        let problem = linear_problem();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let state = interior_state(&problem);
        let mu = state.duality_measure(&geometry);
        let scaling = Scaling::new(&problem, &geometry, &state).unwrap();
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            &problem,
            &geometry,
            &state,
            &scaling.d,
            mu,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        let first = predictor(&geometry, &state, &scaling, &mut resolvent).unwrap();
        // Force sigma = 1 with a zero affine displacement: the corrector must
        // then be the pure centering step at target mu, i.e. r = (0, s + mu y^-1, 0).
        let mut pure = Predictor {
            affine: ResolventSolution {
                xy: state.x().iter().chain(state.y()).copied().collect(),
                x_len: state.x().len(),
                tau: state.tau(),
                used_tau_fallback: false,
            },
            s_affine: state.s().to_vec(),
            correction: vec![-0.0; state.s().len()],
            alpha_affine: 1.0,
            sigma: 1.0,
        };
        let second = corrector(&state, &scaling, &mut pure, &mut resolvent).unwrap();
        let r_y = state
            .s()
            .iter()
            .zip(&scaling.y_inverse)
            .map(|(s, y_inverse)| s + mu * y_inverse)
            .collect::<Vec<_>>();
        let direct = resolvent.solve(&[0.0; 2], &r_y, 0.0, mu).unwrap();
        assert_close(second.centered.tau, direct.tau);
        for index in 0..2 {
            assert_close(second.centered.x()[index], direct.x()[index]);
            assert_close(second.centered.y()[index], direct.y()[index]);
        }
        assert!(first.sigma < 1.0);
    }

    #[test]
    fn step_stays_interior_and_reduces_the_duality_measure() {
        for problem in [quadratic_problem(), linear_problem()] {
            let geometry = ConeGeometry::new(&problem).unwrap();
            let mut state = interior_state(&problem);
            let mut cache = KktCache::new();
            for _ in 0..5 {
                state.normalize_sphere(&geometry).unwrap();
                let mu = state.duality_measure(&geometry);
                let result = step(
                    &problem,
                    &geometry,
                    &state,
                    &KktSettings::new(LinearSolver::Faer),
                    &Settings::default(),
                    &mut cache,
                )
                .unwrap();
                assert!(result.alpha > 0.0 && result.alpha <= 1.0);
                assert!(result.alpha_affine > 0.0 && result.alpha_affine <= 1.0);
                assert!((0.0..=1.0).contains(&result.sigma));
                result.state.validate(&problem, &geometry).unwrap();
                assert!(geometry.is_dual_interior(result.state.y()).unwrap());
                assert!(result.state.tau() > 0.0);
                let next_mu = result.state.duality_measure(&geometry);
                assert!(next_mu < mu, "mu {mu} -> {next_mu}");
                state = result.state;
            }
        }
    }

    #[test]
    fn resolvent_rejects_invalid_shifts() {
        let problem = linear_problem();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let state = interior_state(&problem);
        let scaling = Scaling::new(&problem, &geometry, &state).unwrap();
        for shift in [-1.0, f64::NAN, f64::INFINITY] {
            let mut cache = KktCache::new();
            assert_eq!(
                Resolvent::new(
                    &problem,
                    &geometry,
                    &state,
                    &scaling.d,
                    shift,
                    &KktSettings::new(LinearSolver::Faer),
                    &mut cache,
                )
                .err(),
                Some(ResolventError::State(StateError::InvalidMu))
            );
            assert_eq!(cache.stats().factorizations, 0);
        }
    }
    #[test]
    fn accurate_steps_relax_to_a_positive_nonbinding_floor() {
        let settings = KktSettings::new(LinearSolver::Faer);
        let mut cache = KktCache::new();
        for _ in 0..30 {
            with_clamp_retries(&settings, 1e-3, &mut cache, |trial, _| {
                assert!(trial.diagonal_floor > 0.0);
                assert!(trial.diagonal_floor < 1e-3);
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(cache.clamp_floor, Some(1e-20));
        assert_eq!(cache.clamp_increases, 0);
        assert!(cache.clamp_decreases >= 8);
    }
}
