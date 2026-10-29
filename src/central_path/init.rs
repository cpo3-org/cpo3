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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InitialStart {
    Canonical,
    Cvxopt,
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
/// CVXOPT solve fails.
fn cold_state(
    problem: &Problem,
    geometry: &ConeGeometry,
    settings: &Settings,
    kkt: &KktSettings,
    cache: &mut KktCache,
) -> Result<(State, InitialStart), StateError> {
    if settings.cvxopt_init {
        match cvxopt_state(problem, geometry, kkt, cache) {
            Ok(state) => return Ok((state, InitialStart::Cvxopt)),
            Err(reason) => {
                eprintln!(
                    "cpo3: cvxopt initialization failed ({reason}); using the canonical point"
                );
            }
        }
    }
    Ok((
        canonical_state(problem, geometry, settings.initial_mu)?,
        InitialStart::Canonical,
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
    use crate::central_path::{init, kkt::KktCache};
    use crate::cone::ConeGeometry;
    use crate::linear_algebra::matrix;
    use crate::linear_solvers::KktSettings;
    use crate::ruiz::EquilibratedProblem;
    use crate::{Cone, LinearSolver, Problem, Settings, WarmStart};
    use approx::assert_relative_eq;
    use rstest::rstest;

    fn problem() -> Problem {
        Problem::new(
            Some(matrix(2, 2, &[2., 1., 1., 3.])),
            vec![-3., -7.],
            matrix(3, 2, &[1., 0., 0., -2., 1., 2.]),
            vec![1., -4., 5.],
            vec![Cone::Nonnegative(2), Cone::Zero(1)],
        )
        .unwrap()
    }

    #[rstest]
    #[case::default(None)]
    #[case::small(Some(1e-12))]
    #[case::one_e_minus_four(Some(1e-4))]
    #[case::intermediate(Some(0.25))]
    #[case::endpoint(Some(1.))]
    fn canonical_initialization_uses_requested_complementarity(#[case] beta: Option<f64>) {
        let p = problem();
        let mut settings = Settings {
            threads: 1,
            ..Settings::default()
        };
        if let Some(beta) = beta {
            settings.initial_mu = beta;
        }
        let beta = settings.initial_mu;
        let eq = EquilibratedProblem::new(&p, &settings).unwrap();
        let geometry = ConeGeometry::new(eq.problem()).unwrap();
        let mut cache = KktCache::new();
        let (state, report) = init::initial_state(
            &geometry,
            &eq,
            None,
            &settings,
            &KktSettings::from(&settings),
            &mut cache,
        )
        .unwrap();
        assert_eq!(report.start, init::InitialStart::Canonical);
        assert!(!report.warm_start_supplied);
        assert_eq!(report.warm_start_score, None);
        assert_eq!(state.x(), &[0., 0.]);
        assert_relative_eq!(
            state.s(),
            &[beta.sqrt(), beta.sqrt(), 0.][..],
            epsilon = 0.,
            max_relative = 1e-14
        );
        assert_eq!(state.y(), state.s());
        assert_relative_eq!(state.tau(), (3. - 2. * beta).sqrt(), epsilon = 1e-14);
        assert_relative_eq!(
            state.duality_measure(&geometry),
            beta,
            epsilon = 0.,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            state.y().iter().map(|v| v * v).sum::<f64>() + state.tau().powi(2),
            3.,
            epsilon = 1e-14
        );
        state.validate(eq.problem(), &geometry).unwrap();
        assert_eq!(cache.stats().factorizations, 0);
    }

    #[rstest]
    #[case::smallest_floor(1e-12)]
    #[case::middle_floor(1e-6)]
    #[case::largest_floor(1e-3)]
    fn automatic_warm_screening_selects_the_best_interior_floor(#[case] floor: f64) {
        // min -f*x, x+s=f. At x=0, the candidate y=s=f satisfies
        // primal and dual feasibility; the other floors disturb both.
        let p = Problem::new(
            None,
            vec![-floor],
            matrix(1, 1, &[1.]),
            vec![floor],
            vec![Cone::Nonnegative(1)],
        )
        .unwrap();
        let settings = Settings {
            threads: 1,
            ruiz_iters: 0,
            initial_mu: 0.25,
            force_warm_start: false,
            ..Settings::default()
        };
        let eq = EquilibratedProblem::new(&p, &settings).unwrap();
        let geometry = ConeGeometry::new(eq.problem()).unwrap();
        let warm = WarmStart {
            x: vec![0.],
            y: vec![0.],
            s: vec![0.],
        };
        let before = warm.clone();
        let mut cache = KktCache::new();
        let (state, report) = init::initial_state(
            &geometry,
            &eq,
            Some(&warm),
            &settings,
            &KktSettings::from(&settings),
            &mut cache,
        )
        .unwrap();
        assert_eq!(report.start, init::InitialStart::Warm);
        assert!(report.warm_start_supplied);
        let score = report.warm_start_score.unwrap();
        assert!(score.is_finite() && score <= init::WARM_THRESHOLD);
        // Ratios recover the winning floor independently of normalization.
        assert_relative_eq!(
            state.y()[0] / state.tau(),
            floor,
            epsilon = 0.,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            state.s()[0] / state.tau(),
            floor,
            epsilon = 0.,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            state.tau(),
            (2. / (1. + floor * floor)).sqrt(),
            epsilon = 1e-14
        );
        assert_eq!(state.x(), &[0.]);
        state.validate(&p, &geometry).unwrap();
        assert_eq!(warm, before);
        assert_eq!(cache.stats().factorizations, 0);
    }

    #[rstest]
    fn poor_warm_start_is_rejected_unless_forced(#[values(false, true)] force: bool) {
        let p = Problem::new(
            None,
            vec![-1.],
            matrix(1, 1, &[1.]),
            vec![1.],
            vec![Cone::Nonnegative(1)],
        )
        .unwrap();
        let settings = Settings {
            threads: 1,
            ruiz_iters: 0,
            initial_mu: 0.25,
            force_warm_start: force,
            ..Settings::default()
        };
        let eq = EquilibratedProblem::new(&p, &settings).unwrap();
        let geometry = ConeGeometry::new(eq.problem()).unwrap();
        let warm = WarmStart {
            x: vec![1e6],
            y: vec![1.],
            s: vec![1.],
        };
        let (state, report) = init::initial_state(
            &geometry,
            &eq,
            Some(&warm),
            &settings,
            &KktSettings::from(&settings),
            &mut KktCache::new(),
        )
        .unwrap();
        assert!(report.warm_start_supplied);
        let score = report.warm_start_score.unwrap();
        assert!(score.is_finite() && score > init::WARM_THRESHOLD);
        if force {
            assert_eq!(report.start, init::InitialStart::Warm);
            assert_relative_eq!(state.x()[0] / state.tau(), 1e6, max_relative = 1e-14);
        } else {
            assert_eq!(report.start, init::InitialStart::Canonical);
            assert_eq!(state.x(), &[0.]);
            assert_eq!(state.y(), &[0.5]);
            assert_eq!(state.s(), &[0.5]);
            assert_relative_eq!(state.tau(), 1.75_f64.sqrt(), epsilon = 1e-14);
        }
        state.validate(&p, &geometry).unwrap();
    }

    #[rstest]
    fn failed_cvxopt_initialization_falls_back_to_configured_canonical_start(
        #[values(LinearSolver::Faer, LinearSolver::Feral)] backend: LinearSolver,
        #[values(1e-4, 0.25)] beta: f64,
    ) {
        // Finite LP data, but the uncoupled dual solve requires dividing
        // 1e308 by a diagonal no larger than roughly 1e-4. Every allowed
        // CVXOPT shift therefore overflows, on either backend.
        let p = Problem::new(
            None,
            vec![1e308, 1e308],
            matrix(1, 2, &[0., 0.]),
            vec![1.],
            vec![Cone::Nonnegative(1)],
        )
        .unwrap();
        let settings = Settings {
            threads: 1,
            ruiz_iters: 0,
            cvxopt_init: true,
            initial_mu: beta,
            linear_solver: backend,
            ..Settings::default()
        };
        let eq = EquilibratedProblem::new(&p, &settings).unwrap();
        let geometry = ConeGeometry::new(eq.problem()).unwrap();
        let kkt = KktSettings::from(&settings);
        assert!(init::cvxopt_state(&p, &geometry, &kkt, &mut KktCache::new()).is_err());
        let mut cache = KktCache::new();
        let (state, report) =
            init::initial_state(&geometry, &eq, None, &settings, &kkt, &mut cache).unwrap();
        assert!(
            cache.stats().factorizations > 1,
            "CVXOPT must actually try and fail before fallback"
        );
        assert_eq!(report.start, init::InitialStart::Canonical);
        assert!(!report.warm_start_supplied);
        assert_eq!(report.warm_start_score, None);
        assert_eq!(state.x(), &[0., 0.]);
        assert_relative_eq!(
            state.s()[0],
            beta.sqrt(),
            epsilon = 0.,
            max_relative = 1e-14
        );
        assert_eq!(state.y(), state.s());
        assert_relative_eq!(state.tau(), (2. - beta).sqrt(), epsilon = 1e-14);
        assert_relative_eq!(
            state.duality_measure(&geometry),
            beta,
            epsilon = 0.,
            max_relative = 1e-14
        );
        state.validate(&p, &geometry).unwrap();
    }

    #[rstest]
    #[case::canonical(false)]
    #[case::cvxopt(true)]
    fn initialization_preserves_sphere_and_cone_interior(#[case] cvxopt_init: bool) {
        let p = problem();

        let settings = Settings {
            threads: 1,
            cvxopt_init,
            ..Settings::default()
        };
        let eq = EquilibratedProblem::new(&p, &settings).unwrap();
        let geometry = ConeGeometry::new(eq.problem()).unwrap();
        let (state, report) = init::initial_state(
            &geometry,
            &eq,
            None,
            &settings,
            &KktSettings::from(&settings),
            &mut KktCache::new(),
        )
        .unwrap();
        assert_eq!(
            report.start,
            if cvxopt_init {
                init::InitialStart::Cvxopt
            } else {
                init::InitialStart::Canonical
            }
        );
        let squared: f64 = state
            .x()
            .iter()
            .chain(state.y())
            .map(|v| v * v)
            .sum::<f64>()
            + state.tau().powi(2);
        assert_relative_eq!(squared, 3., epsilon = 1e-10, max_relative = 1e-10);
        assert!(state.tau() > 0.);
        assert_eq!(state.s()[2], 0.);
        assert!(
            state.s()[..2]
                .iter()
                .chain(&state.y()[..2])
                .all(|v| *v > 0.)
        );
    }
}
