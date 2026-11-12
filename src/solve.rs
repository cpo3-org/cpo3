use std::time::Duration;
use std::time::Instant;

use thiserror::Error;

use crate::{
    Problem, Settings, Solution, SolutionInfo, SolverStatus,
    central_path::{
        init::{self, WarmStart},
        kkt_solver::KktCache,
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
            Some((problem.constraint_count(), problem.variable_count())),
        );
        reporter.finish(status, 0, elapsed, solution.info.pobj);
        solution
    };
    let mut solution = match outcome {
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
        PresolveOutcome::Unchanged => solve_core(
            problem,
            &geometry,
            settings,
            warm_start,
            Some((0, 0)),
            None,
            started,
        )?,
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
                expand_solution(problem, &geometry, &presolved, reduced)
            }
        }
    };
    solution.info.presolve_time = presolve_time;
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
    let equilibrated = EquilibratedProblem::new(problem, geometry, settings)?;
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
    use super::*;
    use crate::{Cone, CscMatrix};

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

    /// min -sum(x) subject to ||x|| <= 1, with optimum x = 1 / sqrt(dim).
    fn ball(dim: usize) -> Problem {
        let rows = (1..=dim).collect();
        let columns = (0..dim).collect();
        Problem::new(
            None,
            vec![-1.0; dim],
            CscMatrix::from_triplets(dim + 1, dim, rows, columns, vec![-1.0; dim]).unwrap(),
            std::iter::once(1.0).chain(vec![0.0; dim]).collect(),
            vec![Cone::SecondOrder(dim + 1)],
        )
        .unwrap()
    }

    #[test]
    fn second_order_cones_solve_in_dense_and_expanded_layouts() {
        for dim in [1, 2, 3, 5, 12] {
            for presolve in [false, true] {
                for cvxopt_init in [false, true] {
                    let settings = Settings {
                        presolve,
                        cvxopt_init,
                        ..Settings::default()
                    };
                    let solution = solve(&ball(dim), &settings).unwrap();
                    assert_eq!(solution.status(), SolverStatus::Solved, "{dim}");
                    let root = (dim as f64).sqrt();
                    for x in solution.x() {
                        assert!((x - 1.0 / root).abs() < 1e-6, "{dim}: {:?}", solution.x());
                    }
                    assert!((solution.info().pobj + root).abs() < 1e-6);
                    assert!((solution.y()[0] - root).abs() < 1e-5);
                }
            }
        }
    }

    /// A deterministic generator for random test data.
    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (u, v) = (self.uniform().max(1e-300), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    /// The projection of `z` onto the second-order cone.
    fn project_soc(z: &[f64]) -> Vec<f64> {
        let norm = z[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= z[0] {
            return z.to_vec();
        }
        if norm <= -z[0] {
            return vec![0.0; z.len()];
        }
        let scale = 0.5 * (z[0] + norm);
        std::iter::once(scale)
            .chain(z[1..].iter().map(|v| v * scale / norm))
            .collect()
    }

    #[test]
    fn random_mixed_cone_programs_reach_their_known_optimum() {
        // Optimal (x*, s*, y*) by construction: s* = proj(z), y* = proj(-z)
        // per block, b = A x* + s*, c = -(P x* + A' y*).
        for seed in 0..12_u64 {
            let mut rng = Lcg(seed + 1);
            let cones = vec![
                Cone::Zero(2),
                Cone::Nonnegative(4),
                Cone::SecondOrder(3),
                Cone::SecondOrder(2),
                Cone::SecondOrder(6),
                Cone::SecondOrder(11),
            ];
            let m: usize = cones.iter().map(Cone::vector_dim).sum();
            let n = 8;
            let mut a = vec![0.0; m * n];
            for (index, value) in a.iter_mut().enumerate() {
                if rng.uniform() < 0.5 || index % n == (index / n) % n {
                    *value = rng.normal();
                }
            }
            let (mut s, mut y) = (vec![0.0; m], vec![0.0; m]);
            let mut row = 0;
            for cone in &cones {
                let dim = cone.vector_dim();
                let mut z = (0..dim).map(|_| rng.normal()).collect::<Vec<_>>();
                match cone {
                    Cone::Zero(_) => y[row..row + dim].copy_from_slice(&z),
                    Cone::Nonnegative(_) => {
                        for i in 0..dim {
                            s[row + i] = z[i].max(0.0);
                            y[row + i] = (-z[i]).max(0.0);
                        }
                    }
                    _ => {
                        let tail = z[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
                        z[0] += [-1.0, 0.0, 1.0][(seed % 3) as usize] * tail;
                        s[row..row + dim].copy_from_slice(&project_soc(&z));
                        let negated = z.iter().map(|v| -v).collect::<Vec<_>>();
                        y[row..row + dim].copy_from_slice(&project_soc(&negated));
                    }
                }
                row += dim;
            }
            let x = (0..n).map(|_| rng.normal()).collect::<Vec<_>>();
            let quadratic = seed % 2 == 1;
            let p = quadratic.then(|| CscMatrix::identity(n).unwrap());
            let b = (0..m)
                .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum::<f64>() + s[i])
                .collect::<Vec<_>>();
            let c = (0..n)
                .map(|j| {
                    let px = if quadratic { x[j] } else { 0.0 };
                    -(px + (0..m).map(|i| a[i * n + j] * y[i]).sum::<f64>())
                })
                .collect::<Vec<_>>();
            let quadratic_value = if quadratic {
                0.5 * x.iter().map(|v| v * v).sum::<f64>()
            } else {
                0.0
            };
            let objective = quadratic_value + c.iter().zip(&x).map(|(c, x)| c * x).sum::<f64>();
            let problem = Problem::new(
                p,
                c,
                crate::linear_algebra::test_matrix(m, n, a).unwrap(),
                b,
                cones,
            )
            .unwrap();
            for presolve in [false, true] {
                let settings = Settings {
                    presolve,
                    ..Settings::default()
                };
                let solution = solve(&problem, &settings).unwrap();
                assert_eq!(solution.status(), SolverStatus::Solved, "seed {seed}");
                let error = (solution.info().pobj - objective).abs() / (1.0 + objective.abs());
                assert!(
                    error < 1e-6,
                    "seed {seed}: {} vs {objective}",
                    solution.info().pobj
                );
            }
        }
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
}
