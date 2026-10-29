//! Solution types and construction of solver results.

use std::time::Duration;

use crate::{
    LinearSolver, Problem, Settings,
    central_path::init,
    cone::ConeGeometry,
    linear_algebra::matrix_mul,
    linear_solvers::LinearAlgebraStats,
    presolve::Presolved,
    recovery::{Certificate, Measures, RecoveredSolution},
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
    pub(crate) fn new(settings: &Settings, init_report: Option<init::InitReport>) -> Self {
        let (warm_start_used, warm_start_score, cvxopt_init_used) = match init_report {
            Some(init) => (
                init.warm_start_supplied
                    .then_some(init.start == init::InitialStart::Warm),
                init.warm_start_score,
                match init.start {
                    init::InitialStart::Canonical => Some(false),
                    init::InitialStart::Cvxopt => Some(true),
                    init::InitialStart::Warm => None,
                },
            ),
            None => (None, None, None),
        };
        Self {
            iters: 0,
            solve_time: Duration::ZERO,
            pobj: f64::NAN,
            dobj: f64::NAN,
            pres: f64::NAN,
            dres: f64::NAN,
            comp: f64::NAN,
            linear_solver: settings.linear_solver,
            eliminate_bounds: settings.eliminate_bounds,
            warm_start_used,
            warm_start_score,
            cvxopt_init_used,
            presolve_rows_removed: 0,
            presolve_columns_removed: 0,
            presolve_time: Duration::ZERO,
            presolve_parallel_comparisons: None,
            linear_algebra: LinearAlgebraStats::default(),
        }
    }

    fn set_measures(&mut self, measures: &Measures) {
        self.pobj = measures.pobj;
        self.dobj = measures.dobj;
        self.pres = measures.pres;
        self.dres = measures.dres;
        self.comp = measures.comp;
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
    fn new(
        status: SolverStatus,
        (x, y, s): (Vec<f64>, Vec<f64>, Vec<f64>),
        measures: &Measures,
        mut info: SolutionInfo,
    ) -> Self {
        info.set_measures(measures);
        Self {
            status,
            x,
            y,
            s,
            info,
        }
    }

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
            let measures = Measures::new(problem, &x, &y, &s, ax);
            info.set_measures(&measures);
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
    status: SolverStatus,
    (x, y, s): (Vec<f64>, Vec<f64>, Vec<f64>),
    info: SolutionInfo,
) -> Solution {
    let measures = if status == SolverStatus::Solved {
        let ax = matrix_mul(problem.a(), &x);
        Measures::new(problem, &x, &y, &s, ax)
    } else {
        Measures::nan()
    };
    Solution::new(status, (x, y, s), &measures, info)
}

pub(crate) fn build_solution(
    status: SolverStatus,
    recovered: RecoveredSolution,
    info: SolutionInfo,
) -> Solution {
    Solution::new(
        status,
        (recovered.x, recovered.y, recovered.s),
        &recovered.measures,
        info,
    )
}

/// Package a certificate: the ray scaled to unit slope, the other side NaN.
pub(crate) fn build_certificate(certificate: Certificate, info: SolutionInfo) -> Solution {
    let nan = |len: usize| vec![f64::NAN; len];
    let scaled = |v: &[f64], slope: f64| v.iter().map(|e| e / slope.abs()).collect::<Vec<_>>();
    let (status, point) = match certificate {
        Certificate::Primal(ray) => (
            SolverStatus::PrimalInfeasible,
            (nan(ray.x.len()), scaled(&ray.y, ray.bty), nan(ray.s.len())),
        ),
        Certificate::Dual(ray) => (
            SolverStatus::DualInfeasible,
            (
                scaled(&ray.x, ray.ctx),
                nan(ray.y.len()),
                scaled(&ray.s, ray.ctx),
            ),
        ),
    };
    Solution::new(status, point, &Measures::nan(), info)
}
