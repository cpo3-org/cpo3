//! Initial points: canonical and CVXOPT-style cold starts, and cpo3-style
//! warm screening.

use super::{
    kkt::{KktCache, solve_shifted_systems},
    state::{State, StateError},
};

use crate::{
    Problem, Settings,
    cone::{ConeBlock, ConeGeometry},
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

/// What the initialization decided and measured.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InitReport {
    /// `Some(true)` when a supplied warm start was used.
    pub(crate) warm_start_used: Option<bool>,
    /// Local-norm score used to accept or reject the best warm candidate.
    pub(crate) warm_start_score: Option<f64>,
    /// `Some(true)` when the CVXOPT-style point started the solve and
    /// `Some(false)` for the canonical point, including after a fallback;
    /// `None` when an accepted warm start was used instead.
    pub(crate) cvxopt_init_used: Option<bool>,
}

const WARM_FLOORS: [f64; 3] = [1e-12, 1e-6, 1e-3];
const WARM_THRESHOLD: f64 = 100.0;
const LOCAL_EPS: f64 = 1e-15;
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
/// warm starts; one for equality-only problems. The canonical point already
/// lies on the sphere, so cold starts of either strategy enter the loop at
/// the same radius.
fn sphere_factor(geometry: &ConeGeometry, x: &[f64], y: &[f64]) -> f64 {
    if geometry.barrier_parameter() == 0 {
        return 1.0;
    }
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

/// cpo3's CVXOPT-style start in the solver's operating scale.
///
/// Solve `[P, A'; A, -H][x; y] = [-c; b]` with `H` the identity on cone
/// rows and zero on equality rows, so equality rows hold exactly and cone
/// rows give the bounded least-squares point `y = A x - b`. Linear programs
/// split the solve on one factorization: `(0, b)` for the primal and
/// `(-c, 0)` for the dual, since coupling both is ill-posed without a
/// `(1,1)` block. Then `s = b - A x`, and cone components of `s` and `y`
/// are shifted so that their minimum is at least the interior margin.
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

/// Add one constant to every cone component so that their minimum reaches
/// the interior margin; equality components are left unchanged.
fn shift_into_interior(values: &mut [f64], geometry: &ConeGeometry) {
    let minimum = geometry
        .blocks()
        .iter()
        .filter_map(|block| match block {
            ConeBlock::Nonnegative(range) => Some(&values[range.clone()]),
            ConeBlock::Zero(_) => None,
        })
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let shift = INTERIOR_MARGIN - minimum;
    if shift > 0.0 {
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
/// CVXOPT solve fails, and whether the CVXOPT-style point was used.
fn cold_state(
    problem: &Problem,
    geometry: &ConeGeometry,
    settings: &Settings,
    kkt: &KktSettings,
    cache: &mut KktCache,
) -> Result<(State, bool), StateError> {
    if settings.cvxopt_init {
        match cvxopt_state(problem, geometry, kkt, cache) {
            Ok(state) => return Ok((state, true)),
            Err(reason) => {
                eprintln!(
                    "cpo3: cvxopt initialization failed ({reason}); using the canonical point"
                );
            }
        }
    }
    Ok((
        canonical_state(problem, geometry, settings.initial_mu)?,
        false,
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
    let problem = equilibrated.problem();
    let mut report = InitReport {
        warm_start_used: None,
        warm_start_score: None,

        cvxopt_init_used: None,
    };
    if let Some(warm) = warm {
        // The local score is undefined without inequality pairs.
        if geometry.barrier_parameter() == 0 {
            report.warm_start_used = Some(false);
        } else if let Some((candidate, score)) = warm_state(geometry, equilibrated, warm) {
            let accept = settings.force_warm_start || score <= WARM_THRESHOLD;
            report.warm_start_used = Some(accept);
            report.warm_start_score = Some(score);
            if accept {
                return Ok((candidate, report));
            }
            eprintln!(
                "cpo3: warm start ignored: local score {score:.3e} exceeds {WARM_THRESHOLD:.3e}"
            );
        } else {
            report.warm_start_used = Some(false);
            eprintln!("cpo3: warm start ignored: no finite interior candidate");
        }
    }
    let (cold, cvxopt_init_used) = cold_state(problem, geometry, settings, kkt, cache)?;
    report.cvxopt_init_used = Some(cvxopt_init_used);
    Ok((cold, report))
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
            cvxopt_init: true,
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
            assert_eq!(report.cvxopt_init_used, Some(false));
            assert_ne!(scaled.unscale_s(state.s()), state.s());
        }
    }

    #[test]
    fn cvxopt_linear_program_splits_the_primal_and_dual_solves() {
        // With A = I the primal solve gives x = b and the dual solve gives
        // y = -c up to the 1e-8 regularization; a coupled (-c, b) solve would
        // give x = b - c and a shifted slack of (2, 1) instead of (1, 1).
        let problem = Problem::new(
            None,
            vec![-1.0, -2.0],
            crate::linear_algebra::test_matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
            vec![3.0, 4.0],
            vec![Cone::Nonnegative(2)],
        )
        .unwrap();
        let (state, report) = cvxopt_start(&problem);
        assert_eq!(report.cvxopt_init_used, Some(true));
        // The unscaled point (3, 4 | 1, 2 | 1) has norm sqrt(31); the sphere
        // of the two-dimensional orthant has radius sqrt(3).
        let tau = state.tau();
        assert_close(tau, (3.0_f64 / 31.0).sqrt(), 1e-6);
        assert_close(sphere_radius(&state), 3.0_f64.sqrt(), 1e-12);
        for (actual, expected) in state.x().iter().zip([3.0, 4.0]) {
            assert_close(actual / tau, expected, 1e-6);
        }
        for (actual, expected) in state.s().iter().zip([1.0, 1.0]) {
            assert_close(actual / tau, expected, 1e-6);
        }
        for (actual, expected) in state.y().iter().zip([1.0, 2.0]) {
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
        assert_eq!(report.cvxopt_init_used, Some(true));
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
        // both vectors are shifted by one constant each to a minimum of one.
        let raw_s = [problem.b()[1] - ax[1], problem.b()[2] - ax[2]];
        let s_shift = [s[1] - raw_s[0], s[2] - raw_s[1]];
        let y_shift = [y[1] + raw_s[0], y[2] + raw_s[1]];
        assert_close(s_shift[0], s_shift[1], 1e-6);
        assert_close(y_shift[0], y_shift[1], 1e-6);
        assert!(s_shift[0] >= 0.0 && y_shift[0] >= 0.0);
        assert_close(s[1].min(s[2]), 1.0, 1e-6);
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
    fn cvxopt_equality_only_start_is_the_kkt_point_at_unit_scale() {
        // x + A' y = -c and A x = b give x = (1, 0), y = -2.
        let problem = Problem::new(
            Some(crate::linear_algebra::test_matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap()),
            vec![1.0, 2.0],
            crate::linear_algebra::test_matrix(1, 2, vec![1.0, 1.0]).unwrap(),
            vec![1.0],
            vec![Cone::Zero(1)],
        )
        .unwrap();
        let (state, report) = cvxopt_start(&problem);
        assert_eq!(report.cvxopt_init_used, Some(true));
        assert_eq!(state.tau(), 1.0);
        assert_close(state.x()[0], 1.0, 1e-6);
        assert_close(state.x()[1], 0.0, 1e-6);
        assert_close(state.y()[0], -2.0, 1e-6);
        assert_eq!(state.s(), [0.0]);
    }

    #[test]
    fn interior_shift_moves_only_cone_rows_and_only_upward() {
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
        let mut values = vec![7.0, 1.5, 4.0];
        shift_into_interior(&mut values, &geometry);
        assert_eq!(values, [7.0, 1.5, 4.0]);
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

    #[test]
    fn all_equality_warm_point_is_ignored() {
        let problem = Problem::new(
            None,
            vec![0.0],
            crate::linear_algebra::test_matrix(1, 1, vec![1.0]).unwrap(),
            vec![1.0],
            vec![Cone::Zero(1)],
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
            x: vec![1.0],
            y: vec![2.0],
            s: vec![0.0],
        };
        let settings = Settings {
            force_warm_start: true,
            ..Settings::default()
        };
        let (state, report) = initial_state(
            &geometry,
            &equilibrated,
            Some(&warm),
            &settings,
            &KktSettings::new(LinearSolver::Faer),
            &mut KktCache::new(),
        )
        .unwrap();
        assert_eq!(report.warm_start_used, Some(false));
        assert_eq!(report.cvxopt_init_used, Some(false));
        assert_eq!(
            state,
            canonical_state(&problem, &geometry, settings.initial_mu).unwrap()
        );
    }
}
