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
///
/// # Errors
/// Returns an error for invalid settings, unsupported cones, or numerical failures
/// that prevent returning a solution.
pub fn solve(problem: &Problem, settings: &Settings) -> Result<Solution, SolveError> {
    solve_from(problem, settings, None)
}

/// Solve from an optional warm start in original units.
///
/// # Errors
/// Returns the errors described in [`solve`], or an error if the warm start has
/// incompatible dimensions or nonfinite values.
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
    let equilibrated = EquilibratedProblem::new(problem, settings)?;
    let mut recovery = Recovery::new(problem, &equilibrated, geometry, original);
    if geometry.barrier_parameter() == 0 {
        let settings = Settings {
            linear_solver: crate::LinearSolver::Faer,
            ..settings.clone()
        };
        let reporter = ProgressReporter::new(settings.verbose);
        reporter.header(problem, &settings, presolve_summary);
        let mut info = SolutionInfo::new(&settings, None);
        info.warm_start_used = warm_start.map(|_| false);
        let termination = crate::equality::solve(
            problem,
            geometry,
            &equilibrated,
            &settings,
            &mut recovery,
            started,
            &mut info,
        )
        .inspect_err(|error| reporter.failure(info.iters, &error.to_string()))?;
        info.solve_time = started.elapsed();
        let solution = match termination {
            Termination::Certificate(certificate) => build_certificate(certificate, info),
            Termination::Point(status) => build_solution(status, recovery.into_point(), info),
        };
        reporter.finish(
            solution.status(),
            solution.info.iters,
            solution.info.solve_time,
            solution.info.pobj,
        );
        return Ok(solution);
    }
    let kkt = KktSettings::from(settings);
    let mut cache = KktCache::new();
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
    let mut recovered = recovery.update(&state, diagnostics.sy);
    let mut reporter = ProgressReporter::new(settings.verbose);
    reporter.header(problem, settings, presolve_summary);
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
            let certificate = (recovered.ktratio > settings.certificate_ktratio)
                .then(|| Ray::new(problem, &equilibrated, state.x(), state.y(), state.s()))
                .and_then(|ray| ray.into_certificate(problem, settings, recovered.ktratio));
            if let Some(termination) = recovered.termination(settings, certificate) {
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
            tau: diagnostics.tau,
            sigma: diagnostics.sigma,
            alpha_aff: diagnostics.alpha_aff,
            alpha: diagnostics.alpha,
            pobj: self.measures.pobj,
            dobj: self.measures.dobj,
            pres: self.measures.pres,
            dres: self.measures.dres,
            comp: self.measures.comp,
        }
    }
}

pub(crate) fn time_limit_reached(started: Instant, time_limit: Option<Duration>) -> bool {
    time_limit.is_some_and(|limit| started.elapsed() >= limit)
}

#[cfg(test)]
mod tests {
    use crate::linear_algebra::matrix;
    use crate::{
        Cone, CscMatrix, Problem, Settings, SolveError, SolverStatus, WarmStart, solve, solve_from,
    };
    use approx::assert_relative_eq;
    use rstest::rstest;

    fn problem() -> Problem {
        Problem::new(
            Some(matrix(2, 2, &[2., 1., 1., 2.])),
            vec![-5., -5.],
            matrix(2, 2, &[1., 0., 1., 1.]),
            vec![0.5, 2.],
            vec![Cone::Nonnegative(2)],
        )
        .unwrap()
    }

    #[rstest]
    #[case::zero_iterations(Settings { max_iters:0, ..Settings::default() })]
    #[case::nan_tolerance(Settings { tol_feas:f64::NAN, ..Settings::default() })]
    #[case::negative_refinement(Settings { refine_tol:-1., ..Settings::default() })]
    #[case::infinite_mu(Settings { initial_mu:f64::INFINITY, ..Settings::default() })]
    #[case::boundary_step(Settings { step_fraction:1., ..Settings::default() })]
    fn rejects_invalid_settings_at_solve_entry(#[case] settings: Settings) {
        assert!(matches!(
            solve(&problem(), &settings),
            Err(SolveError::InvalidSetting(_))
        ));
    }

    #[rstest]
    #[case::second_order(Cone::SecondOrder(2))]
    #[case::semidefinite(Cone::PositiveSemidefinite(1))]
    fn unsupported_cones_return_an_error_before_presolve(#[case] cone: Cone) {
        let m = cone.vector_dim();
        let p = Problem::new(
            None,
            vec![0.],
            CscMatrix::zeros(m, 1).unwrap(),
            vec![0.; m],
            vec![cone],
        )
        .unwrap();
        assert!(matches!(
            solve(&p, &Settings::default()),
            Err(SolveError::Unimplemented(_))
        ));
    }

    #[test]
    fn solves_qp_through_public_api() {
        let solution = solve(
            &problem(),
            &Settings {
                threads: 1,
                ..Settings::default()
            },
        )
        .unwrap();
        assert_eq!(solution.status(), SolverStatus::Solved);
        assert_relative_eq!(solution.x(), &[0.5, 1.5][..], epsilon = 1e-7);
    }

    #[rstest]
    #[case::dimension(vec![0.])]
    #[case::nonfinite(vec![0.,f64::NAN])]
    fn rejects_invalid_warm_starts(#[case] x: Vec<f64>) {
        let warm = WarmStart {
            x,
            y: vec![1.; 2],
            s: vec![1.; 2],
        };
        assert!(matches!(
            solve_from(&problem(), &Settings::default(), Some(&warm)),
            Err(SolveError::InvalidSetting(_))
        ));
    }
}
