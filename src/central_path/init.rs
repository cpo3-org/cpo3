//! Spherical, geometric, and CVXOPT-style cold starts, and warm screening.

use super::{
    kkt::{KktCache, solve_shifted_systems},
    state::{State, StateError},
};

use crate::{
    Initialization, Problem, Settings,
    cone::{ConeBlock, ConeGeometry, INTERIOR_FLOOR},
    linear_algebra::{dot, matrix_mul, matrix_transpose_mul, stable_norm},
    linear_solvers::KktSettings,
    ruiz::EquilibratedProblem,
};

/// A caller-supplied starting point in original units.
#[derive(Clone, Debug, PartialEq)]
pub struct WarmStart {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub s: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InitialStart {
    Cold(Initialization),
    Warm,
}

/// The selected start and the separate warm-screening results.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InitReport {
    pub(crate) start: InitialStart,
    pub(crate) warm_start_supplied: bool,
    /// Local-norm score used to accept or reject the best warm candidate.
    pub(crate) warm_start_score: Option<f64>,
}

const WARM_FLOORS: [f64; 3] = [1e-12, 1e-6, 1e-3];
const WARM_THRESHOLD: f64 = 100.0;
const LOCAL_EPS: f64 = 1e-15;
/// The local-norm theorem applies below one. Use a fixed margin for both
/// geometric starts; this is a proximity target, not a convergence tolerance.
const INITIAL_PROXIMITY: f64 = 0.5;
const PREDICTOR_TRIALS: usize = 16;
/// Additive factor regularization of the CVXOPT initialization system,
/// as in cpo3's standalone initialization solve.
const CVXOPT_REGULARIZATION: f64 = 1e-8;
/// Largest true shift tried before the CVXOPT solve gives up, matching the
/// main loop's retry ceiling.
const MAX_CVXOPT_SHIFT: f64 = 1e-4;
/// cpo3's interior margin: cone components are shifted to at least this.
const INTERIOR_MARGIN: f64 = 1.0;

/// Try cpo3's three interior floors and keep the lowest finite local score.
/// The supplied vectors are in original units; every candidate is formed in
/// the solver's operating scale and projected onto the embedding sphere.
fn warm_state(
    geometry: &ConeGeometry,
    equilibrated: &EquilibratedProblem<'_>,
    warm: &WarmStart,
) -> Option<(State, f64)> {
    let problem = equilibrated.problem();
    let x = equilibrated.scale_x(&warm.x);
    let y = equilibrated.scale_y(&warm.y);
    let s = equilibrated.scale_s(&warm.s);
    // Flooring changes only y and s, so share the primal matrix products.
    let ax = matrix_mul(problem.a(), &x);
    let px = problem.p_mul(&x);
    let mut best = None;
    let mut best_score = f64::INFINITY;
    for floor in WARM_FLOORS {
        let mut cy = y.clone();
        let mut cs = s.clone();
        for block in geometry.blocks() {
            match block {
                ConeBlock::Zero(range) => cs[range.clone()].fill(0.0),
                ConeBlock::Nonnegative(range) => {
                    for i in range.clone() {
                        cy[i] = cy[i].max(floor);
                        cs[i] = cs[i].max(floor);
                    }
                }
            }
        }
        let Ok(mut candidate) = State::new(x.clone(), cs, cy, 1.0) else {
            continue;
        };
        if candidate.normalize_sphere(geometry).is_err()
            || candidate.validate(problem, geometry).is_err()
        {
            continue;
        }
        let score = local_score(problem, geometry, &candidate, &ax, &px);
        if score < best_score {
            best_score = score;
            best = Some(candidate);
        }
    }
    best.map(|candidate| (candidate, best_score))
}

/// cpo3's guarded local residual metric divided by the square root of the candidate's own `mu`.
fn local_score(
    problem: &Problem,
    geometry: &ConeGeometry,
    state: &State,
    ax: &[f64],
    px: &[f64],
) -> f64 {
    let mu = state.duality_measure(geometry);
    if !mu.is_finite() || mu <= 0.0 {
        return f64::INFINITY;
    }
    let (x, y, tau) = (state.x(), state.y(), state.tau());
    let aty = matrix_transpose_mul(problem.a(), y);
    let x_sq = x
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let residual = tau * px[i] + aty[i] + problem.c()[i] * tau + mu * value;
            residual * residual
        })
        .sum::<f64>()
        / mu.max(LOCAL_EPS);
    let mut y_sq = 0.0;
    for block in geometry.blocks() {
        for i in block.range() {
            let value = y[i];
            let mut residual = -tau * ax[i] + problem.b()[i] * tau + mu * value;
            let mut weight = mu;
            if matches!(block, ConeBlock::Nonnegative(_)) {
                residual -= mu / value;
                weight += mu / (value * value);
            }
            y_sq += residual * residual / weight;
        }
    }
    let quadratic = tau * dot(x, px);
    let tau_residual = -dot(problem.c(), x) - dot(problem.b(), y) - quadratic / tau.max(LOCAL_EPS)
        + mu * (tau - 1.0 / tau.max(LOCAL_EPS));
    let tau_weight = mu + mu / (tau * tau).max(LOCAL_EPS);
    let score =
        (x_sq + y_sq + tau_residual * tau_residual / tau_weight.max(LOCAL_EPS)).sqrt() / mu.sqrt();
    if score.is_finite() {
        score
    } else {
        f64::INFINITY
    }
}

/// Factor placing `(x, s, y, 1)` on the embedding sphere
/// `||(x, y, tau)|| = sqrt(nu + 1)`, as `State::normalize_sphere` does for
/// warm starts. The canonical point already
/// lies on the sphere, so cold starts of either strategy enter the loop at
/// the same radius.
fn sphere_factor(geometry: &ConeGeometry, x: &[f64], y: &[f64]) -> f64 {
    let norm = stable_norm(x.iter().chain(y).copied().chain(std::iter::once(1.0)));
    (geometry.barrier_parameter() as f64 + 1.0).sqrt() / norm
}

/// A point on the embedding sphere with inequality complementarity beta.
fn canonical_state(
    problem: &Problem,
    geometry: &ConeGeometry,
    beta: f64,
) -> Result<State, StateError> {
    let mut point = vec![0.0; geometry.dim()];
    geometry.canonical_identity(&mut point)?;
    for value in &mut point {
        *value *= beta.sqrt();
    }
    let tau = (1.0 + geometry.barrier_parameter() as f64 * (1.0 - beta)).sqrt();
    State::new(
        vec![0.0; problem.variable_count()],
        point.clone(),
        point,
        tau,
    )
}

/// Start at the large-mu endpoint, optionally predicting along the path.
/// The sphere fixes u, not s: s = mu / y sets complementarity independently.
fn geometric_state(
    problem: &Problem,
    geometry: &ConeGeometry,
    predict: bool,
) -> Result<(State, Initialization), StateError> {
    let mut e = vec![0.0; geometry.dim()];
    geometry.canonical_identity(&mut e)?;
    // Q(0,e,1) = (A'e + c, b, -b'e); P contributes only away from the endpoint.
    let mut qx = matrix_transpose_mul(problem.a(), &e);
    for (q, c) in qx.iter_mut().zip(problem.c()) {
        *q += c;
    }
    let qt = -dot(problem.b(), &e);
    let norm = stable_norm(
        qx.iter()
            .copied()
            .chain(
                problem
                    .b()
                    .iter()
                    .zip(&e)
                    .map(|(&b, &e)| b / (1.0 + e).sqrt()),
            )
            .chain(std::iter::once(qt / 2.0_f64.sqrt())),
    );
    if !norm.is_finite() {
        return Err(StateError::NonFiniteResult);
    }
    let mut mu = if norm == 0.0 {
        1.0
    } else {
        (norm / INITIAL_PROXIMITY).max(INTERIOR_FLOOR)
    };
    if !mu.is_finite() {
        return Err(StateError::InvalidMu);
    }
    let endpoint = State::new(
        vec![0.0; problem.variable_count()],
        e.iter().map(|&v| mu * v).collect(),
        e,
        1.0,
    )?;
    if !endpoint.duality_measure(geometry).is_finite() {
        return Err(StateError::InvalidMu);
    }
    if !predict || norm == 0.0 {
        return Ok((endpoint, Initialization::CertifiedEndpoint));
    }

    let mut best = endpoint;
    let mut accepted = false;
    let endpoint_mu = mu;
    // Try the endpoint's certified parameter first. If accepted, double 1/mu
    // until the first rejected trial; otherwise backtrack toward the endpoint.
    // Bound the search so initialization cannot turn into a separate solve.
    for _ in 0..PREDICTOR_TRIALS {
        let candidate = positive_predictor(problem, geometry, &qx, qt, mu);
        if let Ok(candidate) = candidate
            && path_proximity(problem, geometry, &candidate) <= INITIAL_PROXIMITY
        {
            best = candidate;
            if !accepted && mu > endpoint_mu {
                return Ok((best, Initialization::PositivePredictor));
            }
            accepted = true;
            mu *= 0.5;
        } else if accepted {
            break;
        } else {
            mu *= 2.0;
        }
        if !mu.is_finite() || mu < INTERIOR_FLOOR {
            break;
        }
    }
    Ok((
        best,
        if accepted {
            Initialization::PositivePredictor
        } else {
            Initialization::CertifiedEndpoint
        },
    ))
}

/// Positive root of w - 1/w = -a, without cancellation or squaring a.
fn barrier_root(a: f64) -> f64 {
    let half_norm = 0.5 * a.hypot(2.0);
    if a >= 0.0 {
        1.0 / (half_norm + 0.5 * a)
    } else {
        half_norm - 0.5 * a
    }
}

/// Freeze Q at the endpoint, solve w + grad(phi)(w) = -Q(u_inf)/mu,
/// normalize w, then impose exact complementarity at that same mu.
fn positive_predictor(
    problem: &Problem,
    geometry: &ConeGeometry,
    qx: &[f64],
    qt: f64,
    mu: f64,
) -> Result<State, StateError> {
    let mut x: Vec<_> = qx.iter().map(|&q| -q / mu).collect();
    let mut y: Vec<_> = problem.b().iter().map(|&b| -b / mu).collect();
    for block in geometry.blocks() {
        if let ConeBlock::Nonnegative(range) = block {
            for i in range.clone() {
                y[i] = barrier_root(problem.b()[i] / mu);
            }
        }
    }
    let mut tau = barrier_root(qt / mu);
    let norm = stable_norm(x.iter().chain(&y).copied().chain(std::iter::once(tau)));
    let scale = (geometry.barrier_parameter() as f64 + 1.0).sqrt() / norm;
    crate::linear_algebra::scale(&mut x, scale);
    crate::linear_algebra::scale(&mut y, scale);
    tau *= scale;
    let mut s = vec![0.0; y.len()];
    for block in geometry.blocks() {
        if let ConeBlock::Nonnegative(range) = block {
            for i in range.clone() {
                s[i] = mu / y[i];
                // Avoid a candidate that the solver's interior floor would
                // immediately change, invalidating its proximity score.
                if y[i] < INTERIOR_FLOOR || s[i] < INTERIOR_FLOOR {
                    return Err(StateError::NonInteriorPoint { variable: "y or s" });
                }
            }
        }
    }
    if tau < INTERIOR_FLOOR {
        return Err(StateError::NonPositiveTau);
    }
    State::new(x, s, y, tau)
}

/// Exact local-norm score for a homogeneous state. Unlike warm screening,
/// certification must not floor mu or the metric weights.
fn path_proximity(problem: &Problem, geometry: &ConeGeometry, state: &State) -> f64 {
    let mu = state.duality_measure(geometry);
    if !mu.is_finite() || mu <= 0.0 {
        return f64::INFINITY;
    }
    let (x, y, tau) = (state.x(), state.y(), state.tau());
    let px = problem.p_mul(x);
    let mut rx = matrix_transpose_mul(problem.a(), y);
    for i in 0..x.len() {
        rx[i] = (rx[i] + px[i] + problem.c()[i] * tau) / mu + x[i];
    }
    let mut ry = matrix_mul(problem.a(), x);
    for i in 0..y.len() {
        ry[i] = (-ry[i] + problem.b()[i] * tau) / mu + y[i];
    }
    for block in geometry.blocks() {
        if let ConeBlock::Nonnegative(range) = block {
            for i in range.clone() {
                ry[i] = (ry[i] * y[i] - 1.0) / y[i].hypot(1.0);
            }
        }
    }
    let qt = -dot(problem.c(), x) - dot(problem.b(), y) - dot(x, &px) / tau;
    let rt = ((qt / mu + tau) * tau - 1.0) / tau.hypot(1.0);
    stable_norm(rx.iter().chain(&ry).copied().chain(std::iter::once(rt)))
}

/// cpo3's CVXOPT-style start in the solver's operating scale.
///
/// Solve `[P, A'; A, -H][x; y] = [-c; b]` with `H` the identity on cone
/// rows and zero on equality rows. Without a numerical retry shift, equality
/// rows hold exactly and cone rows give `y = A x - b`. Linear programs
/// split the solve on one factorization: `(0, b)` for the primal and
/// `(-c, 0)` for the dual, matching CVXOPT's independent least-squares
/// initializations. Then `s = b - A x`; vectors too close to the cone
/// boundary are shifted to have minimum one, as in CVXOPT.
///
/// The shifted duals can be orders of magnitude larger than the sphere
/// radius `sqrt(nu + 1)` (the CONT family reaches `||y|| ~ 1e5` against a
/// radius of a few hundred). Project onto the embedding sphere, as for warm
/// starts, so both cold strategies begin at the same radius.
fn cvxopt_state(
    problem: &Problem,
    geometry: &ConeGeometry,
    kkt: &KktSettings,
    cache: &mut KktCache,
) -> Result<State, String> {
    let (n, m) = (problem.variable_count(), problem.constraint_count());
    let mut h = vec![0.0; geometry.dim()];
    geometry
        .canonical_identity(&mut h)
        .map_err(|error| error.to_string())?;
    let linear = problem
        .p()
        .is_none_or(|p| p.values().iter().all(|value| *value == 0.0));
    let right_hand_sides = if linear {
        let mut primal = vec![0.0; n + m];
        primal[n..].copy_from_slice(problem.b());
        let mut dual = vec![0.0; n + m];
        for (out, value) in dual[..n].iter_mut().zip(problem.c()) {
            *out = -value;
        }
        vec![primal, dual]
    } else {
        let mut both = problem.c().iter().map(|value| -value).collect::<Vec<_>>();
        both.extend_from_slice(problem.b());
        vec![both]
    };
    let solutions = cvxopt_solve(problem, geometry, &h, kkt, cache, &right_hand_sides)?;
    let (mut x, mut y) = if linear {
        (solutions[0][..n].to_vec(), solutions[1][n..].to_vec())
    } else {
        let (x, y) = solutions[0].split_at(n);
        (x.to_vec(), y.to_vec())
    };
    // Equality slacks are exactly zero; the solve only makes them small.
    let mut s = matrix_mul(problem.a(), &x);
    for block in geometry.blocks() {
        match block {
            ConeBlock::Zero(range) => s[range.clone()].fill(0.0),
            ConeBlock::Nonnegative(range) => {
                for (s, b) in s[range.clone()].iter_mut().zip(&problem.b()[range.clone()]) {
                    *s = b - *s;
                }
            }
        }
    }
    shift_into_interior(&mut s, geometry);
    shift_into_interior(&mut y, geometry);
    let scale = sphere_factor(geometry, &x, &y);
    crate::linear_algebra::scale(&mut x, scale);
    crate::linear_algebra::scale(&mut s, scale);
    crate::linear_algebra::scale(&mut y, scale);
    let state = State::new(x, s, y, scale).map_err(|error| error.to_string())?;
    state
        .validate(problem, geometry)
        .map_err(|error| error.to_string())?;
    Ok(state)
}

/// Solve the initialization system exactly first, then with increasing true
/// shifts, and finally with best-effort refinement, like the main loop.
fn cvxopt_solve(
    problem: &Problem,
    geometry: &ConeGeometry,
    h: &[f64],
    kkt: &KktSettings,
    cache: &mut KktCache,
    right_hand_sides: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let mut trial = *kkt;
    trial.strict_refinement = true;
    let mut shift = 0.0;
    loop {
        match solve_shifted_systems(
            problem,
            geometry,
            h,
            shift,
            CVXOPT_REGULARIZATION,
            &trial,
            cache,
            right_hand_sides,
        ) {
            Ok(solutions) => return Ok(solutions),
            Err(error) if error.is_numerical() => {
                if shift == 0.0 {
                    shift = CVXOPT_REGULARIZATION;
                } else if shift * 10.0 <= MAX_CVXOPT_SHIFT {
                    shift *= 10.0;
                } else if trial.strict_refinement {
                    trial.strict_refinement = false;
                } else {
                    return Err(error.to_string());
                }
            }
            Err(error) => return Err(error.to_string()),
        }
    }
}

/// CVXOPT's coneprog.py shifts only vectors at or near the cone boundary.
/// Equality components contribute neither to the norm nor to the shift.
fn shift_into_interior(values: &mut [f64], geometry: &ConeGeometry) {
    let cone_values = geometry
        .blocks()
        .iter()
        .filter_map(|block| match block {
            ConeBlock::Nonnegative(range) => Some(&values[range.clone()]),
            ConeBlock::Zero(_) => None,
        })
        .flatten()
        .copied();
    let minimum = cone_values.clone().fold(f64::INFINITY, f64::min);
    let norm = stable_norm(cone_values);
    if minimum <= 1e-8 * norm.max(1.0) {
        let shift = INTERIOR_MARGIN - minimum;
        for block in geometry.blocks() {
            if let ConeBlock::Nonnegative(range) = block {
                for value in &mut values[range.clone()] {
                    *value += shift;
                }
            }
        }
    }
}

/// The configured cold start, falling back to the canonical point when the
/// CVXOPT solve fails.
fn cold_state(
    problem: &Problem,
    geometry: &ConeGeometry,
    settings: &Settings,
    kkt: &KktSettings,
    cache: &mut KktCache,
) -> Result<(State, InitialStart), StateError> {
    let alternative = match settings.initialization {
        Initialization::Canonical => None,
        Initialization::Cvxopt => Some(
            cvxopt_state(problem, geometry, kkt, cache)
                .map(|state| (state, Initialization::Cvxopt)),
        ),
        Initialization::CertifiedEndpoint | Initialization::PositivePredictor => Some(
            geometric_state(
                problem,
                geometry,
                settings.initialization == Initialization::PositivePredictor,
            )
            .map_err(|error| error.to_string()),
        ),
    };
    if let Some(result) = alternative {
        match result {
            Ok((state, initialization)) => return Ok((state, InitialStart::Cold(initialization))),
            Err(reason) => eprintln!(
                "cpo3: {} initialization failed ({reason}); using the canonical point",
                settings.initialization
            ),
        }
    }
    Ok((
        canonical_state(problem, geometry, settings.initial_mu)?,
        InitialStart::Cold(Initialization::Canonical),
    ))
}

/// Choose and prepare the starting point according to the settings.
pub(crate) fn initial_state(
    geometry: &ConeGeometry,
    equilibrated: &EquilibratedProblem<'_>,
    warm: Option<&WarmStart>,
    settings: &Settings,
    kkt: &KktSettings,
    cache: &mut KktCache,
) -> Result<(State, InitReport), StateError> {
    debug_assert!(geometry.barrier_parameter() > 0);
    let problem = equilibrated.problem();
    let mut warm_start_score = None;
    if let Some(warm) = warm {
        if let Some((candidate, score)) = warm_state(geometry, equilibrated, warm) {
            warm_start_score = Some(score);
            if settings.force_warm_start || score <= WARM_THRESHOLD {
                return Ok((
                    candidate,
                    InitReport {
                        start: InitialStart::Warm,
                        warm_start_supplied: true,
                        warm_start_score,
                    },
                ));
            }
            eprintln!(
                "cpo3: warm start ignored: local score {score:.3e} exceeds {WARM_THRESHOLD:.3e}"
            );
        } else {
            eprintln!("cpo3: warm start ignored: no finite interior candidate");
        }
    }
    let (cold, start) = cold_state(problem, geometry, settings, kkt, cache)?;
    Ok((
        cold,
        InitReport {
            start,
            warm_start_supplied: warm.is_some(),
            warm_start_score,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cone, LinearSolver, Problem};

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "expected {expected}, got {actual}"
        );
    }

    /// `||(x, y, tau)||`, the embedding sphere radius of a state.
    fn sphere_radius(state: &State) -> f64 {
        stable_norm(
            state
                .x()
                .iter()
                .chain(state.y())
                .copied()
                .chain(std::iter::once(state.tau())),
        )
    }

    fn cvxopt_settings() -> Settings {
        Settings {
            ruiz_iters: 0,
            initialization: Initialization::Cvxopt,
            ..Settings::default()
        }
    }

    /// The CVXOPT start of an unscaled problem and its report.
    fn cvxopt_start(problem: &Problem) -> (State, InitReport) {
        let settings = cvxopt_settings();
        let equilibrated = EquilibratedProblem::new(problem, &settings).unwrap();
        let geometry = ConeGeometry::new(problem).unwrap();
        initial_state(
            &geometry,
            &equilibrated,
            None,
            &settings,
            &KktSettings::new(LinearSolver::Faer),
            &mut KktCache::new(),
        )
        .unwrap()
    }

    #[test]
    fn cold_start_is_centered_on_the_sphere_after_data_scaling() {
        let problem = Problem::new(
            None,
            vec![1000.0],
            crate::linear_algebra::test_matrix(2, 1, vec![100.0, 0.001]).unwrap(),
            vec![1e4, 0.02],
            vec![Cone::Zero(1), Cone::Nonnegative(1)],
        )
        .unwrap();
        let scaled = EquilibratedProblem::new(&problem, &crate::Settings::default()).unwrap();
        let geometry = ConeGeometry::new(scaled.problem()).unwrap();
        for beta in [Settings::default().initial_mu, 0.25, 1.0] {
            let settings = Settings {
                initial_mu: beta,
                ..Settings::default()
            };
            let (state, report) = initial_state(
                &geometry,
                &scaled,
                None,
                &settings,
                &KktSettings::new(LinearSolver::Faer),
                &mut KktCache::new(),
            )
            .unwrap();
            assert_eq!(state.x(), [0.0]);
            assert_eq!(state.y(), [0.0, beta.sqrt()]);
            assert_eq!(state.s(), [0.0, beta.sqrt()]);
            assert_close(state.duality_measure(&geometry), beta, 1e-12);
            assert_close(sphere_radius(&state), 2.0_f64.sqrt(), 1e-12);
            state.validate(&problem, &geometry).unwrap();
            assert_eq!(report.start, InitialStart::Cold(Initialization::Canonical));
            assert_ne!(scaled.unscale_s(state.s()), state.s());
        }
    }

    #[test]
    fn cvxopt_linear_program_splits_the_primal_and_dual_solves() {
        // With A = I the primal solve gives x = b and the dual solve gives
        // y = -c. A coupled (-c, b) solve would instead give x = b - c.
        // Healthy positive multipliers below one must remain unchanged.
        let problem = Problem::new(
            None,
            vec![-0.25, -0.5],
            crate::linear_algebra::test_matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
            vec![3.0, 4.0],
            vec![Cone::Nonnegative(2)],
        )
        .unwrap();
        let (state, report) = cvxopt_start(&problem);
        assert_eq!(report.start, InitialStart::Cold(Initialization::Cvxopt));
        // The unscaled point (3, 4 | .25, .5 | 1) has squared norm 26.3125.
        let tau = state.tau();
        assert_close(tau, (3.0_f64 / 26.3125).sqrt(), 1e-6);
        assert_close(sphere_radius(&state), 3.0_f64.sqrt(), 1e-12);
        for (actual, expected) in state.x().iter().zip([3.0, 4.0]) {
            assert_close(actual / tau, expected, 1e-6);
        }
        for (actual, expected) in state.s().iter().zip([1.0, 1.0]) {
            assert_close(actual / tau, expected, 1e-6);
        }
        for (actual, expected) in state.y().iter().zip([0.25, 0.5]) {
            assert_close(actual / tau, expected, 1e-6);
        }
    }

    #[test]
    fn cvxopt_quadratic_start_solves_the_initial_kkt_system_and_shifts_cone_rows() {
        let problem = Problem::new(
            Some(crate::linear_algebra::test_matrix(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap()),
            vec![0.5, -1.0],
            crate::linear_algebra::test_matrix(3, 2, vec![1.0, -2.0, 3.0, 0.25, -1.0, 1.0])
                .unwrap(),
            vec![1.5, -0.75, 2.0],
            vec![Cone::Zero(1), Cone::Nonnegative(2)],
        )
        .unwrap();
        let (state, report) = cvxopt_start(&problem);
        assert_eq!(report.start, InitialStart::Cold(Initialization::Cvxopt));
        let tau = state.tau();
        assert!(tau > 0.0 && tau < 1.0);
        assert_close(sphere_radius(&state), 3.0_f64.sqrt(), 1e-12);
        let x = state.x().iter().map(|v| v / tau).collect::<Vec<_>>();
        let s = state.s().iter().map(|v| v / tau).collect::<Vec<_>>();
        let y = state.y().iter().map(|v| v / tau).collect::<Vec<_>>();
        let ax = matrix_mul(problem.a(), &x);
        // The equality row holds exactly and its slack is exactly zero.
        assert_close(ax[0], problem.b()[0], 1e-6);
        assert_eq!(state.s()[0], 0.0);
        // Cone rows: the raw least-squares point has y = A x - b = -s, and
        // only the negative multipliers need an interior shift in this case.
        let raw_s = [problem.b()[1] - ax[1], problem.b()[2] - ax[2]];
        let s_shift = [s[1] - raw_s[0], s[2] - raw_s[1]];
        let y_shift = [y[1] + raw_s[0], y[2] + raw_s[1]];
        assert_close(s_shift[0], s_shift[1], 1e-6);
        assert_close(y_shift[0], y_shift[1], 1e-6);
        assert_close(s_shift[0], 0.0, 1e-6);
        assert!(y_shift[0] > 0.0);
        assert_close(y[1].min(y[2]), 1.0, 1e-6);
        // The unshifted duals satisfy the first block, P x + A' y + c = 0.
        let raw_y = [y[0], -raw_s[0], -raw_s[1]];
        let px = problem.p_mul(&x);
        let aty = matrix_transpose_mul(problem.a(), &raw_y);
        for i in 0..2 {
            assert_close(px[i] + aty[i] + problem.c()[i], 0.0, 1e-6);
        }
    }

    #[test]
    fn cvxopt_shift_uses_the_cone_norm_and_boundary_threshold() {
        let geometry = crate::cone::test_geometry(vec![
            Cone::Zero(1),
            Cone::Nonnegative(1),
            Cone::Zero(1),
            Cone::Nonnegative(2),
        ]);
        let mut values = vec![-3.0, 0.5, -7.0, 2.0, -1.0];
        shift_into_interior(&mut values, &geometry);
        assert_eq!(values, [-3.0, 2.5, -7.0, 4.0, 1.0]);
        let geometry = crate::cone::test_geometry(vec![Cone::Nonnegative(2), Cone::Zero(1)]);
        let mut values = vec![0.5, 2.0, 1e20];
        shift_into_interior(&mut values, &geometry);
        assert_eq!(values, [0.5, 2.0, 1e20]);
        let mut values = vec![1e-10, 2.0, -7.0];
        shift_into_interior(&mut values, &geometry);
        assert_eq!(values, [1.0, 3.0 - 1e-10, -7.0]);
        let mut values = vec![2.0, 1e10, -7.0];
        shift_into_interior(&mut values, &geometry);
        assert_eq!(values, [1.0, 1e10 - 1.0, -7.0]);
        let geometry = crate::cone::test_geometry(vec![Cone::Zero(1)]);
        let mut values = vec![-2.0];
        shift_into_interior(&mut values, &geometry);
        assert_eq!(values, [-2.0]);
    }

    #[test]
    fn local_score_matches_the_guarded_metric_with_an_equality_row() {
        let problem = Problem::new(
            Some(crate::linear_algebra::test_matrix(1, 1, vec![2.0]).unwrap()),
            vec![3.0],
            crate::linear_algebra::test_matrix(2, 1, vec![1.0, 2.0]).unwrap(),
            vec![4.0, 5.0],
            vec![Cone::Zero(1), Cone::Nonnegative(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let state = State::new(vec![1.0], vec![0.0, 3.0], vec![-1.0, 2.0], 2.0).unwrap();
        // Here mu=6, residuals are (17, [1,17], -1), and the metric
        // weights are (6, [6,7.5], 7.5). The squared local norm is 87,
        // so the squared screening score is 87 / mu = 14.5.
        let score = local_score(&problem, &geometry, &state, &[0.5, 1.0], &[1.0]);
        assert!((score * score - 14.5).abs() < 1e-12);
        let exact = path_proximity(&problem, &geometry, &state);
        assert!((exact * exact - 14.5).abs() < 1e-12);
    }

    #[test]
    fn geometric_starts_are_spherical_centered_and_close_to_the_path() {
        // Include stiff quadratic curvature: the predictor must check the
        // full QP residual and backtrack, not just the endpoint's frozen Q.
        for curvature in [0.0, 2.0, 1e6] {
            let problem = Problem::new(
                Some(crate::linear_algebra::test_matrix(1, 1, vec![curvature]).unwrap()),
                vec![-1.0],
                crate::linear_algebra::test_matrix(3, 1, vec![3.0, -2.0, 1.0]).unwrap(),
                vec![4.0, -1.0, 2.0],
                vec![Cone::Nonnegative(1), Cone::Zero(1), Cone::Nonnegative(1)],
            )
            .unwrap();
            let geometry = ConeGeometry::new(&problem).unwrap();
            for predict in [false, true] {
                let (mut state, _) = geometric_state(&problem, &geometry, predict).unwrap();
                // Check the state as the solve loop receives it, including
                // the loop's normalization and interior floor.
                state.normalize_sphere(&geometry).unwrap();
                state.validate(&problem, &geometry).unwrap();
                assert_close(sphere_radius(&state), 3.0_f64.sqrt(), 1e-12);
                assert_eq!(state.s()[1], 0.0);
                let mu = state.duality_measure(&geometry);
                for i in [0, 2] {
                    assert_close(state.s()[i] * state.y()[i], mu, 1e-12);
                }
                assert!(path_proximity(&problem, &geometry, &state) <= INITIAL_PROXIMITY + 1e-12);
                if !predict {
                    // q=(3, [4,-1,2], -6), ||q||^2_H^-1 = 38.
                    assert_close(mu, 38.0_f64.sqrt() / INITIAL_PROXIMITY, 1e-12);
                }
            }
        }
    }

    #[test]
    fn positive_predictor_has_second_order_endpoint_accuracy() {
        let problem = Problem::new(
            Some(crate::linear_algebra::test_matrix(1, 1, vec![2.0]).unwrap()),
            vec![-1.0],
            crate::linear_algebra::test_matrix(1, 1, vec![3.0]).unwrap(),
            vec![4.0],
            vec![Cone::Nonnegative(1)],
        )
        .unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let first = positive_predictor(&problem, &geometry, &[2.0], -4.0, 100.0).unwrap();
        let second = positive_predictor(&problem, &geometry, &[2.0], -4.0, 200.0).unwrap();
        let ratio = path_proximity(&problem, &geometry, &first)
            / path_proximity(&problem, &geometry, &second);
        assert!((3.9..4.1).contains(&ratio));
        // The stable root must remain positive even when a*a would overflow.
        for a in [-1e200, -1.0, 0.0, 1.0, 1e200] {
            let w = barrier_root(a);
            assert!(w > 0.0 && w.is_finite());
            assert_close(w - 1.0 / w, -a, 1e-14);
        }
    }

    #[test]
    fn geometric_start_handles_zero_and_overflowing_endpoint_residuals() {
        let problem = |c| {
            Problem::new(
                None,
                vec![c],
                crate::linear_algebra::test_matrix(2, 1, vec![0.0; 2]).unwrap(),
                vec![0.0; 2],
                vec![Cone::Nonnegative(2)],
            )
            .unwrap()
        };
        let zero = problem(0.0);
        let geometry = ConeGeometry::new(&zero).unwrap();
        let (state, used) = geometric_state(&zero, &geometry, true).unwrap();
        assert_eq!(used, Initialization::CertifiedEndpoint);
        assert_eq!(path_proximity(&zero, &geometry, &state), 0.0);
        assert_eq!(
            geometric_state(&problem(5e307), &geometry, false),
            Err(StateError::InvalidMu)
        );
    }

    #[test]
    fn screening_can_select_each_of_the_three_floors() {
        for (rhs, expected_floor) in [(1e-12, 1e-12), (1e-6, 1e-6), (1e-3, 1e-3)] {
            let problem = Problem::new(
                None,
                vec![0.0],
                crate::linear_algebra::test_matrix(1, 1, vec![0.0]).unwrap(),
                vec![rhs],
                vec![Cone::Nonnegative(1)],
            )
            .unwrap();
            let equilibrated = EquilibratedProblem::new(
                &problem,
                &crate::Settings {
                    ruiz_iters: 0,
                    ..crate::Settings::default()
                },
            )
            .unwrap();
            let geometry = ConeGeometry::new(&problem).unwrap();
            let warm = WarmStart {
                x: vec![0.0],
                y: vec![0.0],
                s: vec![0.0],
            };
            let (candidate, score) = warm_state(&geometry, &equilibrated, &warm).unwrap();
            assert!(score.is_finite());
            assert!((candidate.y()[0] / candidate.tau() - expected_floor).abs() < 1e-12);
            assert!((candidate.s()[0] / candidate.tau() - expected_floor).abs() < 1e-12);
        }
    }

    #[test]
    fn warm_start_uses_interleaved_cone_blocks() {
        let problem = Problem::new(
            None,
            vec![0.0],
            crate::linear_algebra::test_matrix(5, 1, vec![0.0; 5]).unwrap(),
            vec![1.0, 0.0, 1.0, 1.0, 0.0],
            vec![
                Cone::Nonnegative(1),
                Cone::Zero(1),
                Cone::Nonnegative(2),
                Cone::Zero(1),
            ],
        )
        .unwrap();
        let settings = Settings {
            ruiz_iters: 0,
            ..Settings::default()
        };
        let equilibrated = EquilibratedProblem::new(&problem, &settings).unwrap();
        let geometry = ConeGeometry::new(&problem).unwrap();
        let warm = WarmStart {
            x: vec![0.0],
            y: vec![-3.0, -2.0, 0.0, 4.0, 7.0],
            s: vec![-1.0, 9.0, 2.0, 0.0, 8.0],
        };
        let (state, score) = warm_state(&geometry, &equilibrated, &warm).unwrap();
        assert!(score.is_finite());
        state.validate(&problem, &geometry).unwrap();
        assert_eq!(state.s()[1], 0.0);
        assert_eq!(state.s()[4], 0.0);
        for (i, expected) in [(1, -2.0), (3, 4.0), (4, 7.0)] {
            assert_close(state.y()[i] / state.tau(), expected, 1e-12);
        }
        assert_close(state.s()[2] / state.tau(), 2.0, 1e-12);
    }
}
