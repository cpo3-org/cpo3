//! Mehrotra-style predictor--corrector substeps on the shared resolvent.

use thiserror::Error;

use crate::{
    cone::ConeGeometryError,
    embedding::{Embedding, EmbeddingError, State},
    josephy::{JosephyError, KktCache, Resolvent, ResolventSolution},
    linear_system::KktSettings,
    settings::Settings,
};

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum MehrotraError {
    #[error(transparent)]
    Embedding(#[from] EmbeddingError),
    #[error(transparent)]
    ConeGeometry(#[from] ConeGeometryError),
    #[error(transparent)]
    Josephy(#[from] JosephyError),
    #[error("the duality measure {0} must be finite and strictly positive")]
    InvalidDualityMeasure(f64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum StepKind {
    Ordinary { sigma_floor: f64 },
    Recenter,
}

/// One accepted outer iterate and the scalars that produced it.
pub(crate) struct Step {
    /// True KKT shift used for this step (legacy diagnostic name).
    pub(crate) mu_step: f64,
    /// Complementarity of the state used to construct this step.
    pub(crate) mu_before: f64,
    pub(crate) recentered: bool,
    pub(crate) state: State,
    pub(crate) sigma: f64,
    pub(crate) alpha_affine: Option<f64>,
    pub(crate) alpha: f64,
}

/// One outer iteration, sharing a factorization unless a numerical retry is needed.
pub(crate) fn step(
    embedding: &Embedding<'_>,
    state: &State,
    kkt: &KktSettings,
    requested_shift_floor: f64,
    kind: StepKind,
    settings: &Settings,
    cache: &mut KktCache,
) -> Result<Step, MehrotraError> {
    let scaling = Scaling::new(embedding, state)?;
    let raw_mu = scaling.mu;
    with_shift_retries(
        kkt,
        raw_mu,
        requested_shift_floor,
        cache,
        |trial, rho, cache| {
            step_attempt(
                embedding, state, trial, rho, &scaling, kind, settings, cache,
            )
        },
    )
}

/// Retry with a larger true KKT shift, leaving factor regularization unchanged.
/// Strict refinement protects the outer step; after bounded shift increases,
/// allow one attempt using the best finite refinement iterate.
fn with_shift_retries<T>(
    kkt: &KktSettings,
    raw_mu: f64,
    requested_shift_floor: f64,
    cache: &mut KktCache,
    mut attempt: impl FnMut(&KktSettings, f64, &mut KktCache) -> Result<T, MehrotraError>,
) -> Result<T, MehrotraError> {
    const MAX_RETRY_SHIFT: f64 = 1e-4;
    let mut rho = raw_mu.max(*cache.shift_floor.get_or_insert(requested_shift_floor));
    let requested_rho = rho;
    let mut trial = *kkt;
    trial.strict_refinement = true;
    loop {
        let result = attempt(&trial, rho, cache);
        cache.finish_step(result.is_ok());
        match result {
            Err(error) if retryable(&error) => {
                let next = rho * 10.0;
                if next <= MAX_RETRY_SHIFT {
                    rho = next;
                    cache.shift_floor = Some(rho);
                } else if trial.strict_refinement {
                    trial.strict_refinement = false;
                } else {
                    return Err(error);
                }
            }
            Ok(step) => {
                // Relax only after an accurate step that needed no escalation.
                if trial.strict_refinement && rho == requested_rho && raw_mu < rho {
                    cache.shift_floor = Some((rho / 10.0).max(requested_shift_floor));
                }
                return Ok(step);
            }
            Err(error) => return Err(error),
        }
    }
}

fn retryable(error: &MehrotraError) -> bool {
    matches!(error, MehrotraError::Josephy(inner) if inner.is_numerical())
}

#[allow(clippy::too_many_arguments)]
fn step_attempt(
    embedding: &Embedding<'_>,
    state: &State,
    kkt: &KktSettings,
    rho: f64,
    scaling: &Scaling,
    kind: StepKind,
    settings: &Settings,
    cache: &mut KktCache,
) -> Result<Step, MehrotraError> {
    let raw_mu = scaling.mu;
    let mut resolvent = Resolvent::new(embedding.problem(), state, &scaling.h, rho, kkt, cache)?;
    let geometry = embedding.geometry();
    let recentered = kind == StepKind::Recenter;
    let (mut centered, mut slack, sigma, alpha_affine, boundary) = match kind {
        StepKind::Recenter => {
            let (centered, slack) = centered_solve(
                state,
                rho,
                scaling,
                (state.x(), state.y(), state.tau()),
                raw_mu,
                None,
                &mut resolvent,
            )?;
            let boundary = geometry
                .step_to_boundary_toward(state.s(), &slack)?
                .min(geometry.step_to_boundary_toward(state.y(), centered.y())?);
            (centered, slack, 1.0, None, boundary)
        }
        StepKind::Ordinary { sigma_floor } => {
            let mut first = predictor(embedding, state, rho, scaling, &mut resolvent)?;
            first.sigma = first.sigma.max(sigma_floor);
            let second = corrector(state, rho, scaling, &mut first, &mut resolvent)?;

            let mut second = second;
            let mut boundary = geometry
                .step_to_boundary_toward(state.s(), &second.s_centered)?
                .min(geometry.step_to_boundary_toward(state.y(), second.centered.y())?);

            // Gondzio centrality correctors on the same factorization: push the
            // complementarity products of an aspirational trial point back into
            // [0.1 mu_c, 10 mu_c] through the dual block, and keep the direction only
            // if it lengthens the step. Skip the extra solve once the boundary
            // reaches the configured gate. Also stop after accepting a tau fallback.
            let mu_c = second.mu_centered;
            let mut tries = 0;
            let mut accepts = 0;
            for _ in 0..if embedding.nu() > 0 {
                settings.max_centrality_correctors
            } else {
                0
            } {
                if second.centered.used_tau_fallback || boundary >= settings.corrector_gate {
                    break;
                }
                let aspiration = (1.5 * boundary + 0.3).min(1.0);
                let mut changed = false;
                for (i, correction) in second.correction.iter_mut().enumerate() {
                    let y_inverse = scaling.y_inverse[i];
                    if y_inverse <= 0.0 {
                        continue;
                    }
                    let y = state.y()[i] + aspiration * (second.centered.y()[i] - state.y()[i]);
                    let s = state.s()[i] + aspiration * (second.s_centered[i] - state.s()[i]);
                    let v = y * s;
                    let t = v.clamp(0.1 * mu_c, 10.0 * mu_c) - v;
                    if t != 0.0 {
                        changed = true;
                        *correction += y_inverse * t;
                    }
                }
                if !changed {
                    break;
                }
                tries += 1;
                let (centered, s_centered) = centered_solve(
                    state,
                    rho,
                    scaling,
                    (first.affine.x(), first.affine.y(), first.affine.tau),
                    mu_c,
                    Some(&second.correction),
                    &mut resolvent,
                )?;
                let trial = geometry
                    .step_to_boundary_toward(state.s(), &s_centered)?
                    .min(geometry.step_to_boundary_toward(state.y(), centered.y())?);
                if trial <= boundary + 0.1 * (aspiration - boundary) {
                    break;
                }
                accepts += 1;
                boundary = trial;
                second.centered = centered;
                second.s_centered = s_centered;
            }
            cache.corrector_tries += tries;
            cache.corrector_accepts += accepts;
            (
                second.centered,
                second.s_centered,
                first.sigma,
                Some(first.alpha_affine),
                boundary,
            )
        }
    };
    let alpha = (settings.step_fraction * boundary).min(1.0);

    // `base + alpha (target - base)`, overwriting the corrector's buffers.
    let step_toward = |base: &[f64], target: &mut [f64]| {
        crate::matrix::update(target, base, |target, base| {
            alpha.mul_add(target - base, base)
        });
    };
    let (x, y) = centered.xy.split_at_mut(centered.primal_len);
    step_toward(state.x(), x);
    step_toward(state.s(), &mut slack);
    step_toward(state.y(), y);
    let next = embedding.accept_state(
        centered.xy,
        centered.primal_len,
        slack,
        accepted_tau(state.tau(), centered.tau, alpha)?,
    )?;

    Ok(Step {
        state: next,
        mu_step: rho,
        mu_before: raw_mu,
        recentered,
        sigma,
        alpha_affine,
        alpha,
    })
}

/// Keep zero subproblem roots admissible while the updated iterate stays
/// strictly positive. Check finiteness before `max`, which would hide NaNs.
fn accepted_tau(current: f64, candidate: f64, alpha: f64) -> Result<f64, EmbeddingError> {
    let tau = current + alpha * (candidate - current);
    if !tau.is_finite() {
        return Err(EmbeddingError::NonFiniteResult);
    }
    Ok(tau.max(1e-30))
}

/// Nesterov--Todd scaling `H_k` and Jordan inverse `y_k^{-1}` at one iterate.
pub(crate) struct Scaling {
    mu: f64,
    pub(crate) h: Vec<f64>,
    pub(crate) y_inverse: Vec<f64>,
}

impl Scaling {
    pub(crate) fn new(embedding: &Embedding<'_>, state: &State) -> Result<Self, MehrotraError> {
        let mu = embedding.duality_measure(state);
        if !mu.is_finite() || mu < 0.0 {
            return Err(MehrotraError::InvalidDualityMeasure(mu));
        }
        // Initialization and acceptance establish the interior invariant;
        // checked sphere normalization preserves it between iterations.
        debug_assert!(embedding.ensure_domain(state).is_ok());
        let mut h = vec![0.0; state.s().len()];
        let mut y_inverse = vec![0.0; state.y().len()];
        embedding
            .geometry()
            .scaling_and_inverse(state.s(), state.y(), &mut h, &mut y_inverse)?;
        Ok(Self { mu, h, y_inverse })
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
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) mu_affine: f64,
    pub(crate) sigma: f64,
}

/// Solve the surrogate system at target zero with the current iterate as
/// surrogate, then evaluate Mehrotra's centering heuristic.
///
/// The numerical shift `rho` is also the predictor anchor weight.
/// Centering uses the actual complementarity, independently of this shift.
pub(crate) fn predictor(
    embedding: &Embedding<'_>,
    state: &State,
    rho: f64,
    scaling: &Scaling,
    resolvent: &mut Resolvent<'_>,
) -> Result<Predictor, MehrotraError> {
    if !rho.is_finite() || rho <= 0.0 {
        return Err(MehrotraError::InvalidDualityMeasure(rho));
    }
    // Stacked right-hand side (r_x, -r_y) = (rho x, -(rho y + s)).
    let mut rhs = Vec::with_capacity(state.x().len() + state.y().len());
    rhs.extend(state.x().iter().map(|x| rho * x));
    rhs.extend(state.y().iter().zip(state.s()).map(|(y, s)| -(rho * y + s)));
    let r_tau = rho * state.tau();
    let affine = resolvent.solve_rhs(&rhs, r_tau, 0.0, Some((state.x(), state.y())))?;

    let dy = difference(affine.y(), state.y());
    let mut ds = dy
        .iter()
        .zip(&scaling.h)
        .zip(state.s())
        .map(|((dy, h), s)| -h * dy - s)
        .collect::<Vec<_>>();
    #[cfg(test)]
    let s_affine = dy.iter().zip(&scaling.h).map(|(dy, h)| -h * dy).collect();

    let geometry = embedding.geometry();
    let alpha_affine = geometry
        .step_to_boundary_from_interior(state.s(), &ds)?
        .min(geometry.step_to_boundary_from_interior(state.y(), &dy)?);
    let mu_affine = affine_trial_mu(state, &affine, &ds, &dy, alpha_affine, embedding.nu())?;
    let raw_mu = scaling.mu;
    let sigma = if raw_mu > 0.0 {
        (mu_affine / raw_mu).powi(3).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if !sigma.is_finite() {
        return Err(EmbeddingError::NonFiniteResult.into());
    }

    // The affine boundary and trial complementarity are complete. Reuse
    // ds for the correction and release dy before any centered solve.
    if embedding.nu() > 0 {
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
        mu_affine,
        sigma,
    })
}

/// Complementarity of the boundary-limited predictor probe after projection
/// onto ||(x, y, tau)|| = sqrt(nu + 1). Leave the full predictor unchanged.
fn affine_trial_mu(
    state: &State,
    affine: &ResolventSolution,
    ds: &[f64],
    dy: &[f64],
    alpha: f64,
    nu: usize,
) -> Result<f64, EmbeddingError> {
    if nu == 0 {
        return Ok(0.0);
    }
    let x = state
        .x()
        .iter()
        .zip(affine.x())
        .map(|(x, target)| x + alpha * (target - x));
    let y = state.y().iter().zip(dy).map(|(y, dy)| y + alpha * dy);
    let tau = state.tau() + alpha * (affine.tau - state.tau());
    let norm = crate::matrix::stable_norm(x.chain(y).chain(std::iter::once(tau)));
    if !norm.is_finite() || norm <= 0.0 {
        return Err(EmbeddingError::NonFiniteResult);
    }
    // Divide before multiplying to avoid overflow/underflow in the raw
    // complementarity and squared norm of a homogeneously scaled probe.
    let mu = state
        .s()
        .iter()
        .zip(ds)
        .zip(state.y())
        .zip(dy)
        .map(|(((s, ds), y), dy)| ((s + alpha * ds) / norm) * ((y + alpha * dy) / norm))
        .sum::<f64>()
        * ((nu as f64 + 1.0) / nu as f64);
    if !mu.is_finite() {
        return Err(EmbeddingError::NonFiniteResult);
    }
    Ok(mu)
}

/// The centered iterate and its slack, both at target `mu_c = sigma mu_k`.
pub(crate) struct Corrector {
    pub(crate) centered: ResolventSolution,
    pub(crate) s_centered: Vec<f64>,
    pub(crate) mu_centered: f64,
    /// Dual-block correction the solve used (Mehrotra's, plus any accepted
    /// centrality terms).
    pub(crate) correction: Vec<f64>,
}

/// Solve the surrogate system at target `sigma * complementarity` with the given surrogate
/// and Mehrotra's second-order correction in the dual block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn corrector(
    state: &State,
    rho: f64,
    scaling: &Scaling,
    predictor: &mut Predictor,
    resolvent: &mut Resolvent<'_>,
) -> Result<Corrector, MehrotraError> {
    if !rho.is_finite() || rho <= 0.0 {
        return Err(MehrotraError::InvalidDualityMeasure(rho));
    }
    let mu_centered = predictor.sigma * scaling.mu;

    let correction = std::mem::take(&mut predictor.correction);

    let (centered, s_centered) = centered_solve(
        state,
        rho,
        scaling,
        (
            predictor.affine.x(),
            predictor.affine.y(),
            predictor.affine.tau,
        ),
        mu_centered,
        Some(&correction),
        resolvent,
    )?;

    Ok(Corrector {
        centered,
        s_centered,
        mu_centered,
        correction,
    })
}

/// Solve the surrogate system at target `mu_centered` with `correction` in
/// the dual block; returns the point and its slack.
fn centered_solve(
    state: &State,
    rho: f64,
    scaling: &Scaling,
    (anchor_x, anchor_y, anchor_tau): (&[f64], &[f64], f64),
    mu_centered: f64,
    correction: Option<&[f64]>,
    resolvent: &mut Resolvent<'_>,
) -> Result<(ResolventSolution, Vec<f64>), MehrotraError> {
    let residual_weight = rho - mu_centered;
    let m = state.y().len();
    // Stacked right-hand side (r_x, -r_y).
    let mut rhs = Vec::with_capacity(state.x().len() + m);
    rhs.extend(anchor_x.iter().map(|x| residual_weight * x));
    rhs.extend((0..m).map(|index| {
        -(residual_weight * anchor_y[index]
            + state.s()[index]
            + mu_centered * scaling.y_inverse[index]
            + correction.map_or(0.0, |values| values[index]))
    }));
    let r_tau = residual_weight * anchor_tau;
    let initial = (anchor_x, anchor_y);
    let centered = resolvent.solve_rhs(&rhs, r_tau, mu_centered, Some(initial))?;
    let s_centered: Vec<f64> = (0..m)
        .map(|index| {
            mu_centered * scaling.y_inverse[index]
                - scaling.h[index] * (centered.y()[index] - state.y()[index])
                + correction.map_or(0.0, |values| values[index])
        })
        .collect();
    Ok((centered, s_centered))
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
    use crate::{Cone, LinearSolver, Problem, linear_system::LinearSystemError, matrix::dot};

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn retries_increase_the_true_shift_and_decay_after_accurate_steps() {
        let settings = KktSettings::new(LinearSolver::Faer);
        let mut cache = KktCache::new();
        let mut shifts = Vec::new();
        with_shift_retries(&settings, 1e-12, 1e-8, &mut cache, |trial, rho, _| {
            assert_eq!(
                trial.min_static_regularization,
                settings.min_static_regularization
            );
            assert!(trial.strict_refinement);
            shifts.push(rho);
            if shifts.len() == 1 {
                Err(JosephyError::InvalidTauPolynomial.into())
            } else {
                Ok(())
            }
        })
        .unwrap();
        assert_eq!(shifts, [1e-8, 1e-7]);
        assert_eq!(cache.shift_floor, Some(1e-7));
        with_shift_retries(&settings, 1e-12, 1e-8, &mut cache, |_, rho, _| {
            assert_eq!(rho, 1e-7);
            Ok(())
        })
        .unwrap();
        assert_eq!(cache.shift_floor, Some(1e-8));
    }

    #[test]
    fn retries_are_bounded_and_skip_nonnumerical_failures() {
        for floor in [1e-8, 1e-3] {
            let settings = KktSettings::new(LinearSolver::Faer);
            let mut cache = KktCache::new();
            let mut attempts = Vec::new();
            let error = MehrotraError::Josephy(LinearSystemError::Singular.into());
            let result: Result<(), _> =
                with_shift_retries(&settings, 0.0, floor, &mut cache, |trial, rho, _| {
                    attempts.push((rho, trial.strict_refinement));
                    Err(error.clone())
                });
            assert_eq!(result, Err(error));
            assert!(!attempts.last().unwrap().1);
            assert!(
                attempts[..attempts.len() - 1]
                    .iter()
                    .all(|(_, strict)| *strict)
            );
            if floor == 1e-8 {
                assert!(attempts.last().unwrap().0 <= 1e-4);
                assert!(attempts.len() <= 7);
                assert!(attempts.windows(2).all(|pair| pair[0].0 <= pair[1].0));
            } else {
                assert_eq!(attempts, [(floor, true), (floor, false)]);
            }
        }
        let mut cache = KktCache::new();
        let settings = KktSettings::new(LinearSolver::Faer);
        let mut calls = 0;
        let error = MehrotraError::Josephy(JosephyError::IncorrectRhsLength {
            argument: "test",
            expected: 2,
            actual: 1,
        });
        let result: Result<(), _> =
            with_shift_retries(&settings, 0.0, 1e-8, &mut cache, |_, _, _| {
                calls += 1;
                Err(error.clone())
            });
        assert_eq!(result, Err(error));
        assert_eq!(calls, 1);
    }

    #[test]
    fn affine_complementarity_matches_a_fixed_reference_and_handles_no_barrier() {
        // At alpha = 1/4, (x, y, tau) = (1, 1/2, 3/2) and s = 1/2.
        // Thus mu_affine = 2 * (1/4) / (7/2) = 1/7 at every common scale.
        for scale in [1e-200, 1.0, 1e200] {
            let state = State::new(vec![0.0], vec![scale], vec![scale], scale).unwrap();
            let affine = ResolventSolution {
                xy: vec![4.0 * scale, -scale],
                primal_len: 1,
                tau: 3.0 * scale,
                used_tau_fallback: false,
            };
            let ds = [-2.0 * scale];
            let dy = [-2.0 * scale];
            let mu = affine_trial_mu(&state, &affine, &ds, &dy, 0.25, 1).unwrap();
            assert_close(mu, 1.0 / 7.0);
            assert_close(mu.powi(3), 1.0 / 343.0);
            assert_eq!(affine_trial_mu(&state, &affine, &ds, &dy, 0.25, 0), Ok(0.0));
        }
    }

    #[test]
    fn corrector_uses_full_affine_anchor_even_when_working_mu_is_floored() {
        let problem = quadratic_problem();
        let embedding = Embedding::new(&problem).unwrap();
        let state = interior_state(&problem);
        let mu = 0.7; // Different from the state's raw complementarity.
        let scaling = Scaling::new(&embedding, &state).unwrap();
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            &problem,
            &state,
            &scaling.h,
            mu,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        let mut first = predictor(&embedding, &state, mu, &scaling, &mut resolvent).unwrap();
        // The corrector must use the full anchor independently of the
        // reachable fraction (even when this particular predictor is interior).
        first.alpha_affine = 0.25;
        let result = corrector(&state, mu, &scaling, &mut first, &mut resolvent).unwrap();
        let target = first.sigma * embedding.duality_measure(&state);
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
    fn ordinary_and_recenter_share_one_factorization_with_expected_solve_counts() {
        let problem = quadratic_problem();
        let embedding = Embedding::new(&problem).unwrap();
        let mut state = interior_state(&problem);
        state.normalize_sphere(embedding.nu()).unwrap();
        for kind in [StepKind::Ordinary { sigma_floor: 0.0 }, StepKind::Recenter] {
            let mut cache = KktCache::new();
            let result = step(
                &embedding,
                &state,
                &KktSettings::new(LinearSolver::Faer),
                1e-14,
                kind,
                &Settings {
                    max_centrality_correctors: 0,
                    ..Settings::default()
                },
                &mut cache,
            )
            .unwrap();
            let recentered = kind == StepKind::Recenter;
            assert_eq!(cache.stats().factorizations, 1);
            assert_eq!(cache.stats().kkt_solves, if recentered { 2 } else { 3 });
            assert_eq!(result.recentered, recentered);
            assert_eq!(result.alpha_affine.is_none(), recentered);
            if recentered {
                assert_eq!(result.sigma, 1.0);
                let mu = embedding.duality_measure(&state);
                let scaling = Scaling::new(&embedding, &state).unwrap();
                let mut cache = KktCache::new();
                let mut resolvent = Resolvent::new(
                    &problem,
                    &state,
                    &scaling.h,
                    mu,
                    &KktSettings::new(LinearSolver::Faer),
                    &mut cache,
                )
                .unwrap();
                // Independent reference RHS: no predictor, anchor, or cross term.
                let ry = state
                    .s()
                    .iter()
                    .zip(&scaling.y_inverse)
                    .map(|(s, v)| s + mu * v)
                    .collect::<Vec<_>>();
                let centered = resolvent
                    .solve(&vec![0.0; state.x().len()], &ry, 0.0, mu)
                    .unwrap();
                for (i, x) in result.state.x().iter().enumerate() {
                    assert_close(
                        *x,
                        state.x()[i] + result.alpha * (centered.x()[i] - state.x()[i]),
                    );
                }
                for (i, s) in result.state.s().iter().enumerate() {
                    let target =
                        mu * scaling.y_inverse[i] - scaling.h[i] * (centered.y()[i] - state.y()[i]);
                    assert_close(*s, state.s()[i] + result.alpha * (target - state.s()[i]));
                }
            }
        }
    }

    #[test]
    fn accepted_tau_damps_zero_candidates_and_rejects_nonfinite_updates() {
        assert!((accepted_tau(1.0, 0.0, 0.99).unwrap() - 0.01).abs() < 1e-15);
        assert_eq!(accepted_tau(1e-30, 0.0, 0.99), Ok(1e-30));
        assert_eq!(accepted_tau(1.0, 0.0, 1.0), Ok(1e-30));
        for candidate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                accepted_tau(1.0, candidate, 0.99),
                Err(EmbeddingError::NonFiniteResult)
            );
        }
    }

    fn quadratic_problem() -> Problem {
        Problem::new(
            Some(crate::matrix::test_matrix(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap()),
            vec![0.5, -1.0],
            crate::matrix::test_matrix(3, 2, vec![1.0, -2.0, 3.0, 0.25, -1.0, 1.0]).unwrap(),
            vec![1.5, -0.75, 2.0],
            vec![Cone::Zero(1), Cone::Nonnegative(2)],
        )
        .unwrap()
    }

    fn linear_problem() -> Problem {
        Problem::new(
            None,
            vec![1.0, 2.0],
            crate::matrix::test_matrix(2, 2, vec![1.0, 1.0, -1.0, 2.0]).unwrap(),
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
                _ => (0..cone.vector_dim().unwrap())
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
        let embedding = Embedding::new(problem).unwrap();
        let state = interior_state(problem);
        let mu = embedding.duality_measure(&state);
        let scaling = Scaling::new(&embedding, &state).unwrap();
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            problem,
            &state,
            &scaling.h,
            mu,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        let result = predictor(&embedding, &state, mu, &scaling, &mut resolvent).unwrap();
        (state, mu, scaling, result)
    }

    fn run_both(problem: &Problem) -> (State, f64, Predictor, Corrector) {
        let embedding = Embedding::new(problem).unwrap();
        let state = interior_state(problem);
        let mu = embedding.duality_measure(&state);
        let scaling = Scaling::new(&embedding, &state).unwrap();
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            problem,
            &state,
            &scaling.h,
            mu,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        let mut first = predictor(&embedding, &state, mu, &scaling, &mut resolvent).unwrap();
        let second = corrector(&state, mu, &scaling, &mut first, &mut resolvent).unwrap();
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
            assert!(result.mu_affine >= 0.0);
            let probe = |base: &[f64], target: &[f64]| {
                base.iter()
                    .zip(target)
                    .map(|(base, target)| base + result.alpha_affine * (target - base))
                    .collect::<Vec<_>>()
            };
            let x = probe(state.x(), result.affine.x());
            let y = probe(state.y(), result.affine.y());
            let s = probe(state.s(), &result.s_affine);
            let tau = state.tau() + result.alpha_affine * (result.affine.tau - state.tau());
            let norm_sq = dot(&x, &x) + dot(&y, &y) + tau * tau;
            let embedding = Embedding::new(&problem).unwrap();
            let nu = embedding.nu() as f64;
            let expected_mu = (nu + 1.0) * dot(&s, &y) / (nu * norm_sq);
            assert_close(result.mu_affine, expected_mu);
            assert_close(
                result.sigma,
                (expected_mu / embedding.duality_measure(&state))
                    .powi(3)
                    .clamp(0.0, 1.0),
            );
        }
    }

    #[test]
    fn predictor_slack_vanishes_on_zero_rows_and_is_exact_when_the_step_is_full() {
        let problem = quadratic_problem();
        let (_, mu, _, result) = run_predictor(&problem);
        assert_eq!(result.s_affine[0], 0.0);
        if result.alpha_affine == 1.0 {
            let nu = problem.constraint_count() as f64 - 1.0;
            let norm_sq = dot(result.affine.x(), result.affine.x())
                + dot(result.affine.y(), result.affine.y())
                + result.affine.tau.powi(2);
            assert_close(
                result.mu_affine,
                (nu + 1.0) * dot(&result.s_affine, result.affine.y()) / (nu * norm_sq),
            );
        }
        assert!(result.mu_affine < mu, "the affine step must reduce mu");
    }

    #[test]
    fn corrector_slack_carries_the_second_order_correction() {
        for problem in [quadratic_problem(), linear_problem()] {
            let (state, mu, first, second) = run_both(&problem);
            assert_close(second.mu_centered, first.sigma * mu);
            assert!(second.centered.tau > 0.0);
            let nonneg = |index: usize| problem.cones()[0] != Cone::Zero(1) || index > 0;
            for index in 0..problem.constraint_count() {
                let (s, y) = (state.s()[index], state.y()[index]);
                let ds = first.s_affine[index] - s;
                let dy = first.affine.y()[index] - y;
                let lhs = second.s_centered[index] * y + s * second.centered.y()[index];
                let expected = if nonneg(index) {
                    s * y + second.mu_centered - ds * dy
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
        let embedding = Embedding::new(&problem).unwrap();
        let state = interior_state(&problem);
        let mu = embedding.duality_measure(&state);
        let scaling = Scaling::new(&embedding, &state).unwrap();
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            &problem,
            &state,
            &scaling.h,
            mu,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        let first = predictor(&embedding, &state, mu, &scaling, &mut resolvent).unwrap();
        // Force sigma = 1 with a zero affine displacement: the corrector must
        // then be the pure centering step at target mu, i.e. r = (0, s + mu y^-1, 0).
        let mut pure = Predictor {
            affine: ResolventSolution {
                xy: state.x().iter().chain(state.y()).copied().collect(),
                primal_len: state.x().len(),
                tau: state.tau(),
                used_tau_fallback: false,
            },
            s_affine: state.s().to_vec(),
            correction: vec![-0.0; state.s().len()],
            alpha_affine: 1.0,
            mu_affine: mu,
            sigma: 1.0,
        };
        let second = corrector(&state, mu, &scaling, &mut pure, &mut resolvent).unwrap();
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
            let embedding = Embedding::new(&problem).unwrap();
            let mut state = interior_state(&problem);
            let mut cache = KktCache::new();
            for _ in 0..5 {
                state.normalize_sphere(embedding.nu()).unwrap();
                let mu = embedding.duality_measure(&state);
                let result = step(
                    &embedding,
                    &state,
                    &KktSettings::new(LinearSolver::Faer),
                    1e-14,
                    StepKind::Ordinary { sigma_floor: 0.0 },
                    &Settings::default(),
                    &mut cache,
                )
                .unwrap();
                assert!(result.alpha > 0.0 && result.alpha <= 1.0);
                assert!(result.alpha_affine.is_some_and(|a| a > 0.0 && a <= 1.0));
                assert!((0.0..=1.0).contains(&result.sigma));
                embedding.ensure_domain(&result.state).unwrap();
                assert!(
                    embedding
                        .geometry()
                        .is_dual_interior(result.state.y())
                        .unwrap()
                );
                assert!(result.state.tau() > 0.0);
                let next_mu = embedding.duality_measure(&result.state);
                assert!(next_mu < mu, "mu {mu} -> {next_mu}");
                state = result.state;
            }
        }
    }

    #[test]
    fn predictor_rejects_a_nonpositive_duality_measure() {
        let problem = linear_problem();
        let embedding = Embedding::new(&problem).unwrap();
        let state = interior_state(&problem);
        let scaling = Scaling::new(&embedding, &state).unwrap();
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            &problem,
            &state,
            &scaling.h,
            1.0,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        assert_eq!(
            predictor(&embedding, &state, 0.0, &scaling, &mut resolvent).err(),
            Some(MehrotraError::InvalidDualityMeasure(0.0))
        );
    }
}
