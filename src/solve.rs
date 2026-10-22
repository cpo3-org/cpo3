use std::time::Duration;
use std::time::Instant;

use thiserror::Error;

use crate::{
    Problem, Settings, Solution, SolutionInfo, SolverStatus,
    central_path::{
        init::{self, WarmStart},
        kkt::KktCache,
        mehrotra::{self, MehrotraError},
        state::{State, StateError},
    },
    cone::{ConeGeometry, ConeGeometryError},
    io::progress::{IterationRecord, ProgressReporter},
    linear_algebra::{dot, matrix_mul},
    linear_solvers::KktSettings,
    presolve::{self, PresolveOutcome, Presolved},
    recovery::{Ray, RecoveredSolution, Recovery, Termination},
    ruiz::EquilibratedProblem,
    solution::{build_certificate, build_solution, direct_solution, expand_solution},
};

/// Exceptional failures that prevent a solution object from being returned.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SolveError {
    #[error("invalid solver setting: {0}")]
    InvalidSetting(&'static str),
    #[error("not implemented: {0}")]
    Unimplemented(String),
    #[error("numerical solver failure: {0}")]
    Numerical(String),
}

impl From<StateError> for SolveError {
    fn from(error: StateError) -> Self {
        Self::Numerical(error.to_string())
    }
}

impl From<ConeGeometryError> for SolveError {
    fn from(error: ConeGeometryError) -> Self {
        match error {
            ConeGeometryError::UnsupportedCone { block, name } => {
                Self::Unimplemented(format!("{name} cone in block {block}"))
            }
            error => Self::Numerical(error.to_string()),
        }
    }
}

impl From<MehrotraError> for SolveError {
    fn from(error: MehrotraError) -> Self {
        Self::Numerical(error.to_string())
    }
}

/// Solve a validated conic quadratic problem.
pub fn solve(problem: &Problem, settings: &Settings) -> Result<Solution, SolveError> {
    solve_from(problem, settings, None)
}

/// Solve from an optional warm start in original units.
pub fn solve_from(
    problem: &Problem,
    settings: &Settings,
    warm_start: Option<&WarmStart>,
) -> Result<Solution, SolveError> {
    settings.validate()?;
    validate_warm_start(problem, warm_start)?;
    // Establish the supported layout before presolve can return a solution.
    let geometry = ConeGeometry::new(problem)?;
    let started = Instant::now();
    if !settings.presolve {
        return solve_core(
            problem, &geometry, settings, warm_start, None, None, started,
        );
    }
    let outcome = presolve::presolve(problem, settings.time_limit)
        .map_err(|error| SolveError::Numerical(error.to_string()))?;
    let presolve_time = started.elapsed();
    let direct = |status, point| {
        let elapsed = started.elapsed();
        let info = SolutionInfo {
            solve_time: elapsed,
            presolve_rows_removed: problem.constraint_count(),
            presolve_columns_removed: problem.variable_count(),
            ..SolutionInfo::new(settings, None)
        };
        let solution = direct_solution(problem, status, point, info);
        let reporter = ProgressReporter::new(settings.verbose);
        reporter.header(
            problem,
            settings,
            settings.linear_solver,
            Some((problem.constraint_count(), problem.variable_count())),
        );
        reporter.finish(status, 0, elapsed, solution.info.pobj);
        (solution, None)
    };
    let (mut solution, comparisons) = match outcome {
        PresolveOutcome::Solved(x, y) => {
            let ax = matrix_mul(problem.a(), &x);
            let mut s = Vec::new();
            geometry.reconstruct_slack(problem.b(), &ax, &mut s);
            direct(SolverStatus::Solved, (x, y, s))
        }
        PresolveOutcome::PrimalInfeasible(y) => direct(
            SolverStatus::PrimalInfeasible,
            (
                vec![f64::NAN; problem.variable_count()],
                y,
                vec![f64::NAN; problem.constraint_count()],
            ),
        ),
        PresolveOutcome::DualInfeasible(x) => {
            let s = matrix_mul(problem.a(), &x).iter().map(|v| -v).collect();
            direct(
                SolverStatus::DualInfeasible,
                (x, vec![f64::NAN; problem.constraint_count()], s),
            )
        }
        PresolveOutcome::Unchanged(comparisons) => (
            solve_core(
                problem,
                &geometry,
                settings,
                warm_start,
                Some((0, 0)),
                None,
                started,
            )?,
            Some(comparisons),
        ),
        PresolveOutcome::Reduced(presolved) => {
            if presolved.problem().variable_count() == 0 {
                let (x, y) = presolved.expand_point(&[], &[]);
                let ax = matrix_mul(problem.a(), &x);
                let s = presolved.expand_s_from_ax(problem, &geometry, &[], &ax);
                direct(SolverStatus::Solved, (x, y, s))
            } else {
                let reduced_geometry = ConeGeometry::new(presolved.problem())?;
                let warm =
                    warm_start.map(|warm| presolved.reduce_warm_start(&reduced_geometry, warm));
                let reduced = solve_core(
                    presolved.problem(),
                    &reduced_geometry,
                    settings,
                    warm.as_ref(),
                    Some((presolved.rows_removed(), presolved.columns_removed())),
                    Some((problem, &presolved, &geometry)),
                    started,
                )?;
                (
                    expand_solution(problem, &geometry, &presolved, reduced),
                    Some(presolved.parallel_comparisons),
                )
            }
        }
    };
    solution.info.presolve_time = presolve_time;
    solution.info.presolve_parallel_comparisons = comparisons;
    solution.info.solve_time = started.elapsed();
    Ok(solution)
}

fn validate_warm_start(
    problem: &Problem,
    warm_start: Option<&WarmStart>,
) -> Result<(), SolveError> {
    if let Some(warm) = warm_start {
        if warm.x.len() != problem.variable_count()
            || warm.y.len() != problem.constraint_count()
            || warm.s.len() != problem.constraint_count()
        {
            return Err(SolveError::InvalidSetting(
                "warm start dimensions do not match the problem",
            ));
        }
        if !warm
            .x
            .iter()
            .chain(&warm.y)
            .chain(&warm.s)
            .all(|v| v.is_finite())
        {
            return Err(SolveError::InvalidSetting("warm start must be finite"));
        }
    }
    Ok(())
}

/// Solve `problem` as given, without presolve.
fn solve_core(
    problem: &Problem,
    geometry: &ConeGeometry,
    settings: &Settings,
    warm_start: Option<&WarmStart>,
    presolve_summary: Option<(usize, usize)>,
    original: Option<(&Problem, &Presolved, &ConeGeometry)>,
    started: Instant,
) -> Result<Solution, SolveError> {
    let linear_solver = settings.linear_solver;
    let kkt = KktSettings::from(settings);
    let mut cache = KktCache::new();
    let equilibrated = EquilibratedProblem::new(problem, settings)?;
    let scaled_problem = equilibrated.problem();
    let (mut state, init_report) = init::initial_state(
        geometry,
        &equilibrated,
        warm_start,
        settings,
        &kkt,
        &mut cache,
    )?;
    let mut diagnostics = StepDiagnostics::new(geometry, &state);
    let mut recovery = Recovery::new(problem, &equilibrated, geometry, original);
    let mut recovered = recovery.update(&state, diagnostics.sy);
    let mut reporter = ProgressReporter::new(settings.verbose);
    reporter.header(problem, settings, linear_solver, presolve_summary);
    reporter.iteration(
        recovered.iteration_record(0, started.elapsed(), diagnostics),
        true,
    );

    let mut iteration = 0;
    let termination = if recovered.is_solved(settings) {
        Termination::Point(SolverStatus::Solved)
    } else {
        loop {
            if iteration == settings.max_iters {
                break Termination::Point(SolverStatus::MaxIterations);
            }
            if time_limit_reached(started, settings.time_limit) {
                break Termination::Point(SolverStatus::TimeLimit);
            }

            // Restore ||(x, y, tau)|| = sqrt(nu + 1) before evaluating
            // complementarity, scaling s by the same factor.
            state.normalize_sphere(geometry)?;
            let step = match mehrotra::step(
                scaled_problem,
                geometry,
                &state,
                &kkt,
                settings,
                &mut cache,
            ) {
                Ok(step) => step,
                Err(error) => {
                    let error = SolveError::from(error);
                    reporter.iteration(
                        recovered.iteration_record(iteration, started.elapsed(), diagnostics),
                        true,
                    );
                    reporter.failure(iteration + 1, &error.to_string());
                    return Err(error);
                }
            };
            diagnostics = StepDiagnostics {
                sigma: Some(step.sigma),
                alpha_aff: Some(step.alpha_affine),
                alpha: Some(step.alpha),
                ..StepDiagnostics::new(geometry, &step.state)
            };
            state = step.state;
            iteration += 1;
            recovered = recovery.update(&state, diagnostics.sy);
            // The ray is only examined past the kappa / tau gate.
            let ray = (recovered.ktratio > settings.certificate_ktratio)
                .then(|| Ray::new(problem, &equilibrated, state.x(), state.y(), state.s()));
            if let Some(termination) = recovered.termination(settings, ray) {
                break termination;
            }
            reporter.iteration(
                recovered.iteration_record(iteration, started.elapsed(), diagnostics),
                false,
            );
        }
    };

    let elapsed = started.elapsed();
    reporter.iteration(
        recovered.iteration_record(iteration, elapsed, diagnostics),
        true,
    );
    let info = SolutionInfo {
        solve_time: elapsed,
        iters: iteration,
        linear_algebra: cache.stats(),
        ..SolutionInfo::new(settings, Some(init_report))
    };
    let solution = match termination {
        Termination::Certificate(certificate) => build_certificate(certificate, info),
        Termination::Point(status) => build_solution(status, recovery.into_point(), info),
    };
    reporter.finish(solution.status(), iteration, elapsed, solution.info.pobj);
    Ok(solution)
}

#[derive(Clone, Copy, Debug)]
struct StepDiagnostics {
    /// Keep the numerator to preserve recovery's division order.
    sy: f64,
    mu: f64,
    tau: f64,
    sigma: Option<f64>,
    alpha_aff: Option<f64>,
    alpha: Option<f64>,
}

impl StepDiagnostics {
    fn new(geometry: &ConeGeometry, state: &State) -> Self {
        let nu = geometry.barrier_parameter();
        let sy = if nu == 0 {
            0.0
        } else {
            dot(state.s(), state.y())
        };
        Self {
            sy,
            mu: if nu == 0 { 0.0 } else { sy / nu as f64 },
            tau: state.tau(),
            sigma: None,
            alpha_aff: None,
            alpha: None,
        }
    }
}

impl RecoveredSolution {
    fn iteration_record(
        &self,
        iteration: usize,
        elapsed: Duration,
        diagnostics: StepDiagnostics,
    ) -> IterationRecord {
        IterationRecord {
            iteration,
            elapsed,
            mu: diagnostics.mu,
            tau: Some(diagnostics.tau),
            sigma: diagnostics.sigma,
            alpha_aff: diagnostics.alpha_aff,
            alpha: diagnostics.alpha,
            pobj: Some(self.measures.pobj),
            dobj: Some(self.measures.dobj),
            pres: Some(self.measures.pres),
            dres: Some(self.measures.dres),
            comp: Some(self.measures.comp),
        }
    }
}

fn time_limit_reached(started: Instant, time_limit: Option<Duration>) -> bool {
    time_limit.is_some_and(|limit| started.elapsed() >= limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cone, CscMatrix, linear_algebra::matrix_transpose_mul};

    #[test]
    fn presolve_metadata_and_coordinates_survive_core_solves() {
        let matrix = crate::linear_algebra::test_matrix;
        let problem = Problem::new(
            Some(matrix(3, 3, vec![2., 1., 0., 1., 2., 0., 0., 0., 1.]).unwrap()),
            vec![-1., -1., 0.],
            matrix(3, 3, vec![1., 1., 0., -1., 1., 0., 0., 0., 1.]).unwrap(),
            vec![4., 1., 2.],
            vec![Cone::Nonnegative(2), Cone::Zero(1)],
        )
        .unwrap();
        let PresolveOutcome::Reduced(reduced) = presolve::presolve(&problem, None).unwrap() else {
            panic!("expected the fixed variable to be removed");
        };
        assert!(matches!(
            presolve::presolve(reduced.problem(), None).unwrap(),
            PresolveOutcome::Unchanged(_)
        ));
        for (problem, removed, expected_x) in [
            (&problem, 1, vec![1. / 3., 1. / 3., 2.]),
            (reduced.problem(), 0, vec![1. / 3., 1. / 3.]),
        ] {
            let solution = solve(problem, &Settings::default()).unwrap();
            assert_eq!(solution.status(), SolverStatus::Solved);
            assert_eq!(solution.x().len(), expected_x.len());
            assert!(
                solution
                    .x()
                    .iter()
                    .zip(expected_x)
                    .all(|(actual, expected)| (actual - expected).abs() < 1e-6)
            );
            let info = solution.info();
            assert_eq!(
                (info.presolve_rows_removed, info.presolve_columns_removed),
                (removed, removed)
            );
            assert!(info.presolve_parallel_comparisons.is_some());
            assert!(info.solve_time >= info.presolve_time);
        }
    }

    #[test]
    fn convergence_takes_precedence_over_time_and_iteration_limits() {
        for (target, time_limit, expected_iters) in [(0.0, Some(Duration::ZERO), 0), (1.0, None, 1)]
        {
            let problem = Problem::new(
                Some(CscMatrix::identity(1).unwrap()),
                vec![-target],
                CscMatrix::identity(1).unwrap(),
                vec![target],
                vec![Cone::Zero(1)],
            )
            .unwrap();
            let solution = solve(
                &problem,
                &Settings {
                    presolve: false,
                    max_iters: 1,
                    time_limit,
                    ..Settings::default()
                },
            )
            .unwrap();
            assert_eq!(solution.status(), SolverStatus::Solved);
            assert_eq!(solution.info().iters, expected_iters);
            assert!((solution.x()[0] - target).abs() < 1e-8);
        }
    }

    #[test]
    fn the_core_keeps_time_already_spent_in_presolve() {
        let problem = lp(vec![1.0, -1.0], 2, vec![2.0, 0.0], vec![-1.0]);
        let settings = Settings {
            time_limit: Some(Duration::from_secs(1)),
            ..Settings::default()
        };
        let started = Instant::now() - Duration::from_secs(2);
        let geometry = ConeGeometry::new(&problem).unwrap();
        let result = solve_core(
            &problem,
            &geometry,
            &settings,
            None,
            Some((0, 0)),
            None,
            started,
        )
        .unwrap();
        assert_eq!(result.status(), SolverStatus::TimeLimit);
        assert_eq!(result.info().iters, 0);
        assert!(result.info().solve_time >= Duration::from_secs(2));
    }

    fn lp(a: Vec<f64>, rows: usize, b: Vec<f64>, c: Vec<f64>) -> Problem {
        let columns = c.len();
        Problem::new(
            None,
            c,
            crate::linear_algebra::test_matrix(rows, columns, a).unwrap(),
            b,
            vec![Cone::Nonnegative(rows)],
        )
        .unwrap()
    }

    #[test]
    fn primal_infeasible_lp_returns_a_farkas_certificate() {
        // x <= -1 and x >= 0: A x + s = b with s >= 0.
        let problem = lp(vec![1.0, -1.0], 2, vec![-1.0, 0.0], vec![1.0]);
        for presolve in [true, false] {
            let solution = solve(
                &problem,
                &Settings {
                    presolve,
                    ..Settings::default()
                },
            )
            .unwrap();
            assert_eq!(solution.info().iters == 0, presolve);
            assert_eq!(solution.info().linear_algebra.factorizations == 0, presolve);
            assert_eq!(
                solution.info().cvxopt_init_used,
                if presolve { None } else { Some(false) }
            );
            assert!(solution.info().pobj.is_nan() && solution.info().dobj.is_nan());
            assert_eq!(solution.info().presolve_parallel_comparisons, None);
            assert!(solution.info().solve_time >= solution.info().presolve_time);
            assert_eq!(solution.status(), SolverStatus::PrimalInfeasible);
            assert!(solution.x().iter().all(|v| v.is_nan()));
            let bty = dot(problem.b(), solution.y());
            assert!((bty + 1.0).abs() < 1e-6);
            let aty = matrix_transpose_mul(problem.a(), solution.y());
            assert!(aty.iter().all(|v| v.abs() < 1e-6));
            assert!(solution.y().iter().all(|v| *v >= 0.0));
        }
    }

    #[test]
    fn dual_infeasible_lp_returns_an_unbounded_ray() {
        // minimize -x subject to x >= 0.
        let problem = lp(vec![-1.0], 1, vec![0.0], vec![-1.0]);
        for presolve in [true, false] {
            let solution = solve(
                &problem,
                &Settings {
                    presolve,
                    ..Settings::default()
                },
            )
            .unwrap();
            assert_eq!(solution.info().iters == 0, presolve);
            assert_eq!(solution.info().linear_algebra.factorizations == 0, presolve);
            assert_eq!(
                solution.info().cvxopt_init_used,
                if presolve { None } else { Some(false) }
            );
            assert!(solution.info().pobj.is_nan() && solution.info().dobj.is_nan());
            assert_eq!(solution.info().presolve_parallel_comparisons, None);
            assert!(solution.info().solve_time >= solution.info().presolve_time);
            assert_eq!(solution.status(), SolverStatus::DualInfeasible);
            assert!(solution.y().iter().all(|v| v.is_nan()));
            assert!((dot(problem.c(), solution.x()) + 1.0).abs() < 1e-6);
            let ax = matrix_mul(problem.a(), solution.x());
            assert!(
                ax.iter()
                    .zip(solution.s())
                    .all(|(ax, s)| (ax + s).abs() < 1e-6)
            );
        }
    }

    #[test]
    fn feasible_lp_is_not_declared_infeasible() {
        // minimize x subject to x >= 1.
        let problem = lp(vec![-1.0], 1, vec![-1.0], vec![1.0]);
        let solution = solve(&problem, &Settings::default()).unwrap();
        assert_eq!(solution.info().iters, 0);
        assert_eq!(solution.info().presolve_parallel_comparisons, None);
        assert!(solution.info().solve_time >= solution.info().presolve_time);
        assert_eq!(solution.status(), SolverStatus::Solved);
        assert!((solution.x()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn warm_start_can_be_forced() {
        // minimize x1 + x2 subject to x1 + x2 >= 1, x >= 0.
        let problem = lp(
            vec![-1.0, -1.0, -1.0, 0.0, 0.0, -1.0],
            3,
            vec![-1.0, 0.0, 0.0],
            vec![1.0, 1.0],
        );
        // Test the initialization policy even when presolve could solve this
        // LP without entering the interior-point method.
        let settings = Settings {
            presolve: false,
            ..Settings::default()
        };
        let cold = solve(&problem, &settings).unwrap();
        assert_eq!(cold.info.warm_start_used, None);
        assert_eq!(cold.info.warm_start_score, None);
        assert_eq!(cold.info.cvxopt_init_used, Some(false));
        let warm = WarmStart {
            x: cold.x().to_vec(),
            y: cold.y().to_vec(),
            s: cold.s().to_vec(),
        };
        let again = solve_from(&problem, &settings, Some(&warm)).unwrap();
        assert_eq!(again.status(), SolverStatus::Solved);
        assert_eq!(again.info.warm_start_used, Some(true));
        assert!(again.info.warm_start_score.unwrap().is_finite());
        assert_eq!(again.info.cvxopt_init_used, None);
        assert!(again.info.iters <= cold.info.iters);

        let junk = WarmStart {
            x: vec![1e6, -1e6],
            y: vec![-1.0, 1e-9, 5e3],
            s: vec![1e3, -7.0, 1e-9],
        };
        let rejected = solve_from(&problem, &settings, Some(&junk)).unwrap();
        assert_eq!(rejected.info.warm_start_used, Some(false));
        assert!(rejected.info.warm_start_score.unwrap().is_finite());
        assert_eq!(rejected.info.cvxopt_init_used, Some(false));
        assert_eq!(rejected.info.iters, cold.info.iters);

        let forced = Settings {
            force_warm_start: true,
            presolve: false,
            ..Settings::default()
        };
        let used = solve_from(&problem, &forced, Some(&junk)).unwrap();
        assert_eq!(used.info.warm_start_used, Some(true));
        assert_eq!(used.info.warm_start_score, rejected.info.warm_start_score);
        assert_eq!(used.info.cvxopt_init_used, None);
        assert_eq!(used.status(), SolverStatus::Solved);

        let without_warm = solve_from(&problem, &forced, None).unwrap();
        assert_eq!(without_warm.info.warm_start_used, None);
        assert_eq!(without_warm.info.iters, cold.info.iters);

        let short = WarmStart {
            x: vec![0.0],
            y: vec![0.0; 3],
            s: vec![0.0; 3],
        };
        assert!(matches!(
            solve_from(&problem, &settings, Some(&short)),
            Err(SolveError::InvalidSetting(_))
        ));
    }
}
