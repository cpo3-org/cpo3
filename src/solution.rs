//! Solution types and construction of solver results.

use std::time::Duration;

use crate::{
    LinearSolver, Problem, Settings,
    central_path::init,
    cone::ConeGeometry,
    io::progress::ProgressReporter,
    linear_algebra::matrix_mul,
    linear_solvers::LinearAlgebraStats,
    presolve::Presolved,
    recovery::{Measures, Ray, RecoveredSolution},
};

/// An ordinary solver termination status. Numerical failures return `SolveError` instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolverStatus {
    Solved,
    PrimalInfeasible,
    DualInfeasible,
    MaxIterations,
    TimeLimit,
}

impl SolverStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Solved => "solved",
            Self::PrimalInfeasible => "primal_infeasible",
            Self::DualInfeasible => "dual_infeasible",
            Self::MaxIterations => "max_iterations",
            Self::TimeLimit => "time_limit",
        }
    }
}

/// Summary information for a completed solve.
#[derive(Clone, Debug)]
pub struct SolutionInfo {
    pub iters: usize,
    pub solve_time: Duration,
    pub pobj: f64,
    pub dobj: f64,
    pub pres: f64,
    pub dres: f64,
    pub comp: f64,
    pub linear_solver: LinearSolver,
    pub eliminate_bounds: bool,
    /// Whether a supplied warm start was used; `None` when none was supplied.
    pub warm_start_used: Option<bool>,
    /// Score used to screen the best warm candidate; `None` without one.
    pub warm_start_score: Option<f64>,
    /// `Some(true)` when the CVXOPT-style point started the solve and
    /// `Some(false)` for the canonical point, including after a fallback;
    /// `None` when an accepted warm start was used or no interior-point
    /// iterate was formed.
    pub cvxopt_init_used: Option<bool>,
    /// Constraints and variables removed by presolve.
    pub presolve_rows_removed: usize,
    pub presolve_columns_removed: usize,
    /// Presolve duration, excluding core solve and postsolve.
    pub presolve_time: Duration,
    /// None when presolve is disabled or returns a direct solution or certificate.
    pub presolve_parallel_comparisons: Option<usize>,
    /// Linear-algebra counters over the solve.
    pub linear_algebra: LinearAlgebraStats,
}

impl SolutionInfo {
    fn new(
        measures: Measures,
        linear_solver: LinearSolver,
        settings: &Settings,
        init_report: Option<init::InitReport>,
    ) -> Self {
        let init = init_report.unwrap_or(init::InitReport {
            warm_start_used: None,
            warm_start_score: None,
            cvxopt_init_used: None,
        });
        Self {
            iters: 0,
            solve_time: Duration::ZERO,
            pobj: measures.pobj,
            dobj: measures.dobj,
            pres: measures.pres,
            dres: measures.dres,
            comp: measures.comp,
            linear_solver,
            eliminate_bounds: settings.eliminate_bounds,
            warm_start_used: init.warm_start_used,
            warm_start_score: init.warm_start_score,
            cvxopt_init_used: init.cvxopt_init_used,
            presolve_rows_removed: 0,
            presolve_columns_removed: 0,
            presolve_time: Duration::ZERO,
            presolve_parallel_comparisons: None,
            linear_algebra: LinearAlgebraStats::default(),
        }
    }
}

/// Solver result vectors and termination information.
#[derive(Clone, Debug)]
pub struct Solution {
    status: SolverStatus,
    x: Vec<f64>,
    y: Vec<f64>,
    s: Vec<f64>,
    pub(crate) info: SolutionInfo,
}

impl Solution {
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    pub fn x(&self) -> &[f64] {
        &self.x
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }

    pub fn s(&self) -> &[f64] {
        &self.s
    }

    pub fn info(&self) -> &SolutionInfo {
        &self.info
    }
}

/// Map a solution of the presolved problem back to `problem`.
pub(crate) fn expand_solution(
    problem: &Problem,
    geometry: &ConeGeometry,
    presolved: &Presolved,
    reduced: Solution,
) -> Solution {
    let nan = |len: usize| vec![f64::NAN; len];
    let mut info = reduced.info;
    info.presolve_rows_removed = presolved.rows_removed();
    info.presolve_columns_removed = presolved.columns_removed();
    let (x, y, s) = match reduced.status {
        SolverStatus::PrimalInfeasible => (
            nan(problem.variable_count()),
            presolved.expand_certificate(&reduced.y),
            nan(problem.constraint_count()),
        ),
        SolverStatus::DualInfeasible => {
            let x = presolved.expand_ray(&reduced.x);
            let s = matrix_mul(problem.a(), &x).iter().map(|v| -v).collect();
            (x, nan(problem.constraint_count()), s)
        }
        _ => {
            let (x, y) = presolved.expand_point(&reduced.x, &reduced.y);
            let ax = matrix_mul(problem.a(), &x);
            let s = presolved.expand_s_from_ax(problem, geometry, &reduced.s, &ax);
            let measures = Measures::with_ax(problem, &x, &y, &s, &ax);
            info.pobj = measures.pobj;
            info.dobj = measures.dobj;
            info.pres = measures.pres;
            info.dres = measures.dres;
            info.comp = measures.comp;
            (x, y, s)
        }
    };
    Solution {
        status: reduced.status,
        x,
        y,
        s,
        info,
    }
}

/// A solution decided by presolve alone.
pub(crate) fn direct_solution(
    problem: &Problem,
    settings: &Settings,
    status: SolverStatus,
    (x, y, s): (Vec<f64>, Vec<f64>, Vec<f64>),
    solve_time: Duration,
    linear_solver: LinearSolver,
) -> Solution {
    let measures = if status == SolverStatus::Solved {
        Measures::new(problem, &x, &y, &s)
    } else {
        Measures::nan()
    };
    let reporter = ProgressReporter::new(settings.verbose);
    reporter.header(
        problem,
        settings,
        linear_solver,
        Some((problem.constraint_count(), problem.variable_count())),
    );
    reporter.finish(status, 0, solve_time, measures.pobj);
    Solution {
        status,
        x,
        y,
        s,
        info: SolutionInfo {
            solve_time,
            presolve_rows_removed: problem.constraint_count(),
            presolve_columns_removed: problem.variable_count(),
            ..SolutionInfo::new(measures, linear_solver, settings, None)
        },
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_solution(
    status: SolverStatus,
    recovered: RecoveredSolution,
    iters: usize,
    solve_time: Duration,
    linear_solver: LinearSolver,
    settings: &Settings,
    init_report: init::InitReport,
    stats: LinearAlgebraStats,
) -> Solution {
    Solution {
        status,
        x: recovered.x,
        y: recovered.y,
        s: recovered.s,
        info: SolutionInfo {
            solve_time,
            iters,
            linear_algebra: stats,
            ..SolutionInfo::new(
                recovered.measures,
                linear_solver,
                settings,
                Some(init_report),
            )
        },
    }
}

/// Package a certificate: the ray scaled to unit slope, the other side NaN.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_certificate(
    status: SolverStatus,
    ray: Ray,
    iters: usize,
    solve_time: Duration,
    linear_solver: LinearSolver,
    settings: &Settings,
    init_report: init::InitReport,
    stats: LinearAlgebraStats,
) -> Solution {
    let nan = |len: usize| vec![f64::NAN; len];
    let scaled = |v: &[f64], slope: f64| v.iter().map(|e| e / slope.abs()).collect::<Vec<_>>();
    let (x, y, s) = match status {
        SolverStatus::PrimalInfeasible => {
            (nan(ray.x.len()), scaled(&ray.y, ray.bty), nan(ray.s.len()))
        }
        _ => (
            scaled(&ray.x, ray.ctx),
            nan(ray.y.len()),
            scaled(&ray.s, ray.ctx),
        ),
    };
    Solution {
        status,
        x,
        y,
        s,
        info: SolutionInfo {
            solve_time,
            iters,
            linear_algebra: stats,
            ..SolutionInfo::new(Measures::nan(), linear_solver, settings, Some(init_report))
        },
    }
}
