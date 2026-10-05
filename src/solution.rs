use std::time::Duration;
use std::time::Instant;

use thiserror::Error;

use crate::{
    Equilibration, LinearSolver, Problem, Settings,
    data_equilibration::EquilibratedProblem,
    embedding::{Embedding, EmbeddingError, State},
    governor::{Governor, Observation},
    init::{self, WarmStart},
    josephy::KktCache,
    linear_system::{KktSettings, LinearAlgebraStats},
    matrix::{dot, matrix_mul, matrix_products_into, matrix_transpose_mul, stable_norm},
    mehrotra::{self, MehrotraError, StepKind},
    presolve::{self, PresolveOutcome, Presolved},
    progress::{IterationRecord, ProgressReporter},
};

// CPO3's floor when recovering homogeneous iterates and measuring rays.
const RECOVERY_EPS: f64 = 1e-15;

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

/// Linear-solve work for one outer iteration, including any retries.
/// Iteration zero records initialization separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(
        name = "_IterationStats",
        module = "cpo3._core",
        get_all,
        frozen,
        skip_from_py_object
    )
)]
pub struct IterationStats {
    pub iteration: usize,
    /// KKT right-hand-side solves requested, excluding refinement corrections.
    pub kkt_solves: usize,
    /// Correction back-solves performed by iterative refinement.
    pub refinement_passes: usize,
}

/// Summary information for a completed solve.
#[derive(Clone, Debug)]
pub struct SolutionInfo {
    pub iters: usize,
    pub recenter_steps: usize,
    pub solve_time: Duration,
    pub pobj: f64,
    pub dobj: f64,
    pub pres: f64,
    pub dres: f64,
    pub comp: f64,
    pub linear_solver: LinearSolver,
    pub data_equilibration: Equilibration,
    pub kkt_bound_elimination: bool,
    /// Path level at which the starting point was closest to the central path.
    pub init_level: f64,
    /// Local-norm proximity of the starting point to the path before iteration one.
    pub init_proximity: f64,
    /// Proximity after initialization (equal to `init_proximity` without centering).
    pub init_proximity_after: f64,
    /// Whether a supplied warm start was used; `None` when none was supplied.
    pub warm_start_used: Option<bool>,
    /// Best level and proximity of the supplied warm start (NaN when none).
    pub warm_level: f64,
    pub warm_proximity: f64,
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
    /// KKT solve and refinement counts for initialization and each outer iteration.
    /// Entry zero is initialization; subsequent entries are outer iterations 1..=iters.
    /// Empty when presolve returns without a core solve.
    pub iteration_stats: Vec<IterationStats>,
}

impl SolutionInfo {
    fn new(
        measures: Measures,
        linear_solver: LinearSolver,
        settings: &Settings,
        init_report: Option<init::InitReport>,
    ) -> Self {
        let init = init_report.unwrap_or(init::InitReport {
            level: f64::NAN,
            proximity: f64::NAN,
            proximity_after: f64::NAN,
            warm_start_used: None,
            warm_level: f64::NAN,
            warm_proximity: f64::NAN,
            cvxopt_init_used: None,
        });
        Self {
            iters: 0,
            recenter_steps: 0,
            solve_time: Duration::ZERO,
            pobj: measures.pobj,
            dobj: measures.dobj,
            pres: measures.pres,
            dres: measures.dres,
            comp: measures.comp,
            linear_solver,
            data_equilibration: settings.data_equilibration,
            kkt_bound_elimination: settings.kkt_bound_elimination,
            init_level: init.level,
            init_proximity: init.proximity,
            init_proximity_after: init.proximity_after,
            warm_start_used: init.warm_start_used,
            warm_level: init.warm_level,
            warm_proximity: init.warm_proximity,
            cvxopt_init_used: init.cvxopt_init_used,
            presolve_rows_removed: 0,
            presolve_columns_removed: 0,
            presolve_time: Duration::ZERO,
            presolve_parallel_comparisons: None,
            linear_algebra: LinearAlgebraStats::default(),
            iteration_stats: Vec::new(),
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
    info: SolutionInfo,
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

/// Exceptional failures that prevent a solution object from being returned.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SolveError {
    #[error("invalid solver setting: {0}")]
    InvalidSetting(&'static str),
    #[error("numerical solver failure: {0}")]
    Numerical(String),
}

impl From<EmbeddingError> for SolveError {
    fn from(error: EmbeddingError) -> Self {
        Self::Numerical(error.to_string())
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
    // Presolve's direct exits must respect the same cone support as the
    // embedding, even when fixing variables would bypass its construction.
    for (block, cone) in problem.cones().iter().enumerate() {
        if !matches!(cone, crate::Cone::Zero(_) | crate::Cone::Nonnegative(_)) {
            return Err(SolveError::Numerical(
                crate::cone::ConeGeometryError::UnsupportedCone {
                    block,
                    name: cone.name(),
                }
                .to_string(),
            ));
        }
    }
    let started = Instant::now();
    if !settings.presolve {
        return solve_core(problem, settings, warm_start, None, None, started);
    }
    let linear_solver = settings.linear_solver;
    let outcome = presolve::presolve(problem, settings.time_limit)
        .map_err(|error| SolveError::Numerical(error.to_string()))?;
    let presolve_time = started.elapsed();
    let presolved = match outcome {
        PresolveOutcome::Solved(x, y) => {
            let ax = matrix_mul(problem.a(), &x);
            let mut s: Vec<_> = problem.b().iter().zip(ax).map(|(b, ax)| b - ax).collect();
            let mut offset = 0;
            for cone in problem.cones() {
                let dim = cone.vector_dim().unwrap();
                match cone {
                    crate::Cone::Zero(_) => s[offset..offset + dim].fill(0.0),
                    crate::Cone::Nonnegative(_) => {
                        for value in &mut s[offset..offset + dim] {
                            *value = value.max(0.0);
                        }
                    }
                    _ => {}
                }
                offset += dim;
            }
            return Ok(direct_solution(
                problem,
                settings,
                SolverStatus::Solved,
                (x, y, s),
                started.elapsed(),
                presolve_time,
                linear_solver,
            ));
        }
        PresolveOutcome::PrimalInfeasible(y) => {
            let nan = |len: usize| vec![f64::NAN; len];
            return Ok(direct_solution(
                problem,
                settings,
                SolverStatus::PrimalInfeasible,
                (
                    nan(problem.variable_count()),
                    y,
                    nan(problem.constraint_count()),
                ),
                started.elapsed(),
                presolve_time,
                linear_solver,
            ));
        }
        PresolveOutcome::DualInfeasible(x) => {
            let s = matrix_mul(problem.a(), &x).iter().map(|v| -v).collect();
            return Ok(direct_solution(
                problem,
                settings,
                SolverStatus::DualInfeasible,
                (x, vec![f64::NAN; problem.constraint_count()], s),
                started.elapsed(),
                presolve_time,
                linear_solver,
            ));
        }
        PresolveOutcome::Unchanged(comparisons) => {
            let mut solution =
                solve_core(problem, settings, warm_start, Some((0, 0)), None, started)?;
            solution.info.presolve_time = presolve_time;
            solution.info.presolve_parallel_comparisons = Some(comparisons);
            solution.info.solve_time = started.elapsed();
            return Ok(solution);
        }
        PresolveOutcome::Reduced(presolved) => presolved,
    };
    let summary = (presolved.rows_removed(), presolved.columns_removed());
    if presolved.problem().variable_count() == 0 {
        let (x, y) = presolved.expand_point(&[], &[]);
        let ax = matrix_mul(problem.a(), &x);
        let s = presolved.expand_s_from_ax(problem, &[], &ax);
        return Ok(direct_solution(
            problem,
            settings,
            SolverStatus::Solved,
            (x, y, s),
            started.elapsed(),
            presolve_time,
            linear_solver,
        ));
    }
    let warm = warm_start.map(|warm| presolved.reduce_warm_start(warm));
    let reduced = solve_core(
        presolved.problem(),
        settings,
        warm.as_ref(),
        Some(summary),
        Some((problem, &presolved)),
        started,
    )?;
    let mut solution = expand_solution(problem, &presolved, reduced);
    solution.info.presolve_time = presolve_time;
    solution.info.presolve_parallel_comparisons = Some(presolved.parallel_comparisons);
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

/// Map a solution of the presolved problem back to `problem`.
fn expand_solution(problem: &Problem, presolved: &Presolved, reduced: Solution) -> Solution {
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
            let s = presolved.expand_s_from_ax(problem, &reduced.s, &ax);
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
fn direct_solution(
    problem: &Problem,
    settings: &Settings,
    status: SolverStatus,
    (x, y, s): (Vec<f64>, Vec<f64>, Vec<f64>),
    solve_time: Duration,
    presolve_time: Duration,
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
            presolve_time,
            ..SolutionInfo::new(measures, linear_solver, settings, None)
        },
    }
}

/// Three products already needed for original-coordinate stopping checks.
#[derive(Debug, Default)]
struct Products {
    ax: Vec<f64>,
    px: Vec<f64>,
    aty: Vec<f64>,
}

impl Products {
    #[cfg(test)]
    fn new(problem: &Problem, x: &[f64], y: &[f64]) -> Self {
        let mut products = Self::default();
        products.update(problem, x, y);
        products
    }

    fn update(&mut self, problem: &Problem, x: &[f64], y: &[f64]) {
        matrix_products_into(problem.a(), x, y, &mut self.ax, &mut self.aty);
        if let Some(p) = problem.p() {
            crate::matrix::symmetric_mul_into(p, x, &mut self.px);
        } else {
            self.px.resize(problem.variable_count(), 0.0);
            self.px.fill(0.0);
        }
    }

    /// Euclidean residual of the central-path equations at raw complementarity.
    /// Equality rows have no barrier term; slack is not a separate term in Ty.
    fn delta_path(&self, problem: &Problem, state: &State, cone_rows: &[bool], raw_mu: f64) -> f64 {
        let tau = state.tau().max(RECOVERY_EPS);
        let t_tau = -dot(problem.c(), state.x())
            - dot(problem.b(), state.y())
            - dot(state.x(), &self.px) / tau
            + raw_mu * (state.tau() - 1.0 / tau);
        let tx = self
            .px
            .iter()
            .zip(&self.aty)
            .zip(problem.c())
            .zip(state.x())
            .map(|(((px, aty), c), x)| px + aty + c * state.tau() + raw_mu * x);
        let ty = self
            .ax
            .iter()
            .zip(problem.b())
            .zip(state.y())
            .zip(cone_rows)
            .map(|(((ax, b), y), cone)| {
                -ax + b * state.tau() + raw_mu * y - if *cone { raw_mu / y } else { 0.0 }
            });
        stable_norm(tx.chain(ty).chain(std::iter::once(t_tau))) / raw_mu.max(1e-15)
    }
}

/// Objective values and absolute residuals of `(x, y, s)` on `problem`.
#[derive(Debug)]
struct Measures {
    pobj: f64,
    dobj: f64,
    pres: f64,
    dres: f64,
    comp: f64,
    gap: f64,
    primal_scale: f64,
    dual_scale: f64,
    objective_scale: f64,
}

/// Norms of immutable data used in every stopping check.
#[derive(Clone, Copy)]
struct DataNorms {
    b: f64,
    c: f64,
}

impl DataNorms {
    fn new(problem: &Problem) -> Self {
        Self {
            b: infinity_norm(problem.b()),
            c: infinity_norm(problem.c()),
        }
    }
}

impl Measures {
    fn new(problem: &Problem, x: &[f64], y: &[f64], s: &[f64]) -> Self {
        let ax = matrix_mul(problem.a(), x);
        Self::with_ax(problem, x, y, s, &ax)
    }

    /// Evaluate the returned point without first evaluating the reduced one.
    #[cfg(test)]
    fn postsolve(
        problem: &Problem,
        presolved: &Presolved,
        x: &[f64],
        y: &[f64],
        s: &[f64],
    ) -> Self {
        let (x, y) = presolved.expand_point(x, y);
        let ax = matrix_mul(problem.a(), &x);
        let s = presolved.expand_s_from_ax(problem, s, &ax);
        Self::with_ax(problem, &x, &y, &s, &ax)
    }

    fn with_ax(problem: &Problem, x: &[f64], y: &[f64], s: &[f64], ax: &[f64]) -> Self {
        let px = problem.p_mul(x);
        let aty = matrix_transpose_mul(problem.a(), y);
        Self::with_products(problem, x, y, s, ax, &px, &aty)
    }

    fn with_products(
        problem: &Problem,
        x: &[f64],
        y: &[f64],
        s: &[f64],
        ax: &[f64],
        px: &[f64],
        aty: &[f64],
    ) -> Self {
        Self::with_norms(problem, x, y, s, ax, px, aty, DataNorms::new(problem))
    }

    #[allow(clippy::too_many_arguments)]
    fn with_norms(
        problem: &Problem,
        x: &[f64],
        y: &[f64],
        s: &[f64],
        ax: &[f64],
        px: &[f64],
        aty: &[f64],
        norms: DataNorms,
    ) -> Self {
        let (mut pres, mut ax_norm, mut s_norm) = (0.0_f64, 0.0_f64, 0.0_f64);
        for ((ax, s), b) in ax.iter().zip(s).zip(problem.b()) {
            let value = (ax + s - b).abs();
            pres = if pres.is_nan() || value.is_nan() {
                f64::NAN
            } else {
                pres.max(value)
            };
            ax_norm = ax_norm.max(ax.abs());
            s_norm = s_norm.max(s.abs());
        }
        let (mut dres, mut px_norm, mut aty_norm) = (0.0_f64, 0.0_f64, 0.0_f64);
        for ((px, aty), c) in px.iter().zip(aty).zip(problem.c()) {
            let value = (px + aty + c).abs();
            dres = if dres.is_nan() || value.is_nan() {
                f64::NAN
            } else {
                dres.max(value)
            };
            px_norm = px_norm.max(px.abs());
            aty_norm = aty_norm.max(aty.abs());
        }
        let quadratic = dot(x, px);
        let ctx = dot(problem.c(), x);
        let bty = dot(problem.b(), y);
        let pobj = 0.5 * quadratic + ctx;
        let dobj = -0.5 * quadratic - bty;
        Self {
            pobj,
            dobj,
            pres,
            dres,
            comp: dot(s, y).abs(),
            gap: (ctx + bty + quadratic).abs(),
            primal_scale: 1.0 + ax_norm.max(s_norm).max(norms.b),
            dual_scale: 1.0 + px_norm.max(aty_norm).max(norms.c),
            objective_scale: pobj.abs().min(dobj.abs()).max(1.0),
        }
    }

    fn nan() -> Self {
        Self {
            pobj: f64::NAN,
            dobj: f64::NAN,
            pres: f64::NAN,
            dres: f64::NAN,
            comp: f64::NAN,
            gap: f64::NAN,
            primal_scale: f64::NAN,
            dual_scale: f64::NAN,
            objective_scale: f64::NAN,
        }
    }

    // Equation-relative infinity-norm feasibility, with tol_feas used for both
    // the absolute and relative terms. Keep the absolute-or-relative gap test.
    fn score(&self, tol_feas: f64, gap_abs: f64, gap_rel: f64) -> f64 {
        let ratios = [
            self.pres / self.primal_scale / tol_feas,
            self.dres / self.dual_scale / tol_feas,
            (self.gap / gap_abs).min(self.gap / self.objective_scale / gap_rel),
        ];
        if ratios.iter().any(|ratio| !ratio.is_finite()) {
            return f64::INFINITY;
        }
        ratios.into_iter().fold(0.0, f64::max)
    }
}

/// Solve `problem` as given, without presolve.
fn solve_core(
    problem: &Problem,
    settings: &Settings,
    warm_start: Option<&WarmStart>,
    presolve_summary: Option<(usize, usize)>,
    original: Option<(&Problem, &Presolved)>,
    started: Instant,
) -> Result<Solution, SolveError> {
    let linear_solver = settings.linear_solver;
    let kkt = KktSettings::from(settings);
    let mut cache = KktCache::new();
    let mut governor = Governor::new(settings);
    let equilibrated = EquilibratedProblem::new(problem, settings)
        .map_err(|error| SolveError::Numerical(error.to_string()))?;
    let embedding = Embedding::new(equilibrated.problem())?;
    let governor_enabled = settings.governor && settings.max_recenters > 0 && embedding.nu() > 0;
    let mut identity = vec![0.0; problem.constraint_count()];
    embedding
        .geometry()
        .canonical_identity(&mut identity)
        .map_err(|error| SolveError::Numerical(error.to_string()))?;
    let cone_rows = identity.iter().map(|e| *e > 0.0).collect::<Vec<_>>();
    drop(identity);
    let (mut state, init_report) = init::initial_state(
        &embedding,
        &equilibrated,
        &cone_rows,
        warm_start,
        settings,
        &kkt,
        &mut cache,
    )?;
    let mut previous_stats = cache.stats();
    let mut iteration_stats = vec![IterationStats {
        iteration: 0,
        kkt_solves: previous_stats.kkt_solves,
        refinement_passes: previous_stats.refinement_passes,
    }];
    let mut diagnostics = StepDiagnostics::initial(&embedding, &state);
    let mut recovery = RecoveryWorkspace::new(problem, original);
    let mut recovered = RecoveredSolution::empty();
    recover_into(
        problem,
        &equilibrated,
        &state,
        embedding.nu(),
        diagnostics.sy,
        original,
        &mut recovered,
        &mut recovery,
    );
    let mut reporter = ProgressReporter::new(settings.verbose);
    reporter.header(problem, settings, linear_solver, presolve_summary);
    reporter.iteration(
        recovered.iteration_record(0, started.elapsed(), diagnostics),
        true,
    );

    if recovered.is_solved(settings) {
        let elapsed = started.elapsed();
        reporter.finish(SolverStatus::Solved, 0, elapsed, recovered.measures.pobj);
        return Ok(build_solution(
            SolverStatus::Solved,
            recovered,
            0,
            elapsed,
            linear_solver,
            settings,
            init_report,
            cache.stats(),
            iteration_stats,
            governor.recenter_steps(),
        ));
    }

    // The requested floor may relax when it limits outer convergence. The
    // cached safety floor decays separately, only after accurate KKT solves.
    let mut requested_floor = settings.initial_shift_floor;
    let minimum_floor = if embedding.nu() > 0 {
        0.0
    } else {
        // With no barrier, retain a positive shift for the resolvent.
        settings.initial_shift_floor * 0.01
    };
    let mut previous_progress = f64::INFINITY;
    let mut stalled_progress = 0;
    for iteration in 1..=settings.max_iters {
        if time_limit_reached(started, settings.time_limit) {
            let elapsed = started.elapsed();
            reporter.iteration(
                recovered.iteration_record(iteration - 1, elapsed, diagnostics),
                true,
            );
            reporter.finish(
                SolverStatus::TimeLimit,
                iteration - 1,
                elapsed,
                recovered.measures.pobj,
            );
            return Ok(build_solution(
                SolverStatus::TimeLimit,
                recovered,
                iteration - 1,
                elapsed,
                linear_solver,
                settings,
                init_report,
                cache.stats(),
                iteration_stats,
                governor.recenter_steps(),
            ));
        }

        // Keep the small homogeneous scale while complementarity sets the
        // shift. Once its safety floor is active, restore the sphere so radial
        // drift cannot stall the tau update.
        if diagnostics.mu < cache.shift_floor.unwrap_or(requested_floor) {
            state.normalize_sphere(embedding.nu())?;
        }
        let (next, next_diagnostics) = match predictor_corrector_step(
            &embedding,
            &state,
            &kkt,
            requested_floor,
            if governor_enabled {
                governor.next_step()
            } else {
                StepKind::Ordinary { sigma_floor: 0.0 }
            },
            settings,
            &mut cache,
        ) {
            Ok(step) => step,
            Err(error) => {
                let error = SolveError::from(error);
                reporter.iteration(
                    recovered.iteration_record(iteration - 1, started.elapsed(), diagnostics),
                    true,
                );
                reporter.failure(iteration, &error.to_string());
                return Err(error);
            }
        };
        let stats = cache.stats();
        iteration_stats.push(IterationStats {
            iteration,
            kkt_solves: stats.kkt_solves - previous_stats.kkt_solves,
            refinement_passes: stats.refinement_passes - previous_stats.refinement_passes,
        });
        previous_stats = stats;
        let raw_mu_before_step = next_diagnostics.mu_before;
        state = next;
        diagnostics = next_diagnostics;
        recover_into(
            problem,
            &equilibrated,
            &state,
            embedding.nu(),
            diagnostics.sy,
            original,
            &mut recovered,
            &mut recovery,
        );
        let progress = recovered.stopping_score(settings);
        // Compare in the same coordinates as the KKT solve.
        let floor_active = diagnostics
            .mu_step
            .expect("an accepted step has a KKT shift")
            > raw_mu_before_step;
        if floor_active
            && recovered.ktratio < 1.0
            && progress.is_finite()
            && progress > 1.0
            && progress >= 0.9 * previous_progress
        {
            stalled_progress += 1;
        } else {
            stalled_progress = 0;
        }
        if stalled_progress >= 5 && requested_floor > minimum_floor {
            requested_floor = (requested_floor * 0.1).max(minimum_floor);
            if embedding.nu() > 0 {
                requested_floor = requested_floor.min(0.1 * raw_mu_before_step);
            }
            stalled_progress = 0;
        }
        previous_progress = progress;

        // The ray is only examined past the kappa / tau gate.
        let ray = (recovered.ktratio > settings.certificate_ktratio)
            .then(|| Ray::new(problem, &equilibrated, state.x(), state.y(), state.s()));
        let status = recovered.terminal_status(settings, ray.as_ref());
        if let Some(status) = status.filter(|status| *status != SolverStatus::Solved) {
            let ray = ray.expect("a certificate status requires a ray");
            let elapsed = started.elapsed();
            reporter.iteration(
                recovered.iteration_record(iteration, elapsed, diagnostics),
                true,
            );
            reporter.finish(status, iteration, elapsed, f64::NAN);
            return Ok(build_certificate(
                status,
                ray,
                iteration,
                elapsed,
                linear_solver,
                settings,
                init_report,
                cache.stats(),
                iteration_stats,
                governor.recenter_steps(),
            ));
        }
        let solved = status == Some(SolverStatus::Solved);
        reporter.iteration(
            recovered.iteration_record(iteration, started.elapsed(), diagnostics),
            solved || diagnostics.recentered,
        );
        if solved {
            let elapsed = started.elapsed();
            reporter.finish(
                SolverStatus::Solved,
                iteration,
                elapsed,
                recovered.measures.pobj,
            );
            return Ok(build_solution(
                SolverStatus::Solved,
                recovered,
                iteration,
                elapsed,
                linear_solver,
                settings,
                init_report,
                cache.stats(),
                iteration_stats,
                governor.recenter_steps(),
            ));
        }
        if governor_enabled && iteration < settings.max_iters && governor.needs_observation() {
            let mut products = recovered.products.take().unwrap_or_default();
            if original.is_some() {
                products.update(embedding.problem(), state.x(), state.y());
            } else {
                equilibrated.scale_products(
                    &mut products.ax,
                    &mut products.px,
                    &mut products.aty,
                    state.tau().max(RECOVERY_EPS),
                );
            }
            governor.observe(Observation {
                mu_step: diagnostics
                    .mu_step
                    .expect("an accepted step has a working mu"),
                delta_path: products.delta_path(
                    embedding.problem(),
                    &state,
                    &cone_rows,
                    diagnostics.mu,
                ),
                stopping_score: recovered.stopping_score(settings),
            });
            recovered.products = Some(products);
        }
    }

    let elapsed = started.elapsed();
    reporter.iteration(
        recovered.iteration_record(settings.max_iters, elapsed, diagnostics),
        true,
    );
    reporter.finish(
        SolverStatus::MaxIterations,
        settings.max_iters,
        elapsed,
        recovered.measures.pobj,
    );
    Ok(build_solution(
        SolverStatus::MaxIterations,
        recovered,
        settings.max_iters,
        elapsed,
        linear_solver,
        settings,
        init_report,
        cache.stats(),
        iteration_stats,
        governor.recenter_steps(),
    ))
}

/// One Mehrotra predictor--corrector iteration (paper, Section 4.4).
fn predictor_corrector_step(
    embedding: &Embedding<'_>,
    state: &State,
    kkt: &KktSettings,
    requested_shift_floor: f64,
    kind: StepKind,
    settings: &Settings,
    cache: &mut KktCache,
) -> Result<(State, StepDiagnostics), MehrotraError> {
    let step = mehrotra::step(
        embedding,
        state,
        kkt,
        requested_shift_floor,
        kind,
        settings,
        cache,
    )?;
    let sy = if embedding.nu() == 0 {
        0.0
    } else {
        dot(step.state.s(), step.state.y())
    };
    let diagnostics = StepDiagnostics {
        sy,
        mu: if embedding.nu() == 0 {
            0.0
        } else {
            sy / embedding.nu() as f64
        },
        mu_before: step.mu_before,
        tau: step.state.tau(),
        sigma: Some(step.sigma),
        alpha_aff: step.alpha_affine,
        mu_step: Some(step.mu_step),
        recentered: step.recentered,
        alpha: Some(step.alpha),
    };
    Ok((step.state, diagnostics))
}

#[derive(Debug)]
struct RecoveredSolution {
    products: Option<Products>,
    x: Vec<f64>,
    y: Vec<f64>,
    s: Vec<f64>,
    ktratio: f64,
    measures: Measures,
}

#[derive(Clone, Copy, Debug)]
struct StepDiagnostics {
    /// Keep the numerator to preserve recovery's division order.
    sy: f64,
    mu_before: f64,
    mu_step: Option<f64>,
    recentered: bool,
    mu: f64,
    tau: f64,
    sigma: Option<f64>,
    alpha_aff: Option<f64>,
    alpha: Option<f64>,
}

impl StepDiagnostics {
    fn initial(embedding: &Embedding<'_>, state: &State) -> Self {
        let sy = if embedding.nu() == 0 {
            0.0
        } else {
            dot(state.s(), state.y())
        };
        Self {
            sy,
            mu: if embedding.nu() == 0 {
                0.0
            } else {
                sy / embedding.nu() as f64
            },
            mu_before: 0.0,
            mu_step: None,
            recentered: false,
            tau: state.tau(),
            sigma: None,
            alpha_aff: None,
            alpha: None,
        }
    }
}

impl RecoveredSolution {
    fn empty() -> Self {
        Self {
            products: None,
            x: Vec::new(),
            y: Vec::new(),
            s: Vec::new(),
            ktratio: 0.0,
            measures: Measures::nan(),
        }
    }

    fn stopping_score(&self, settings: &Settings) -> f64 {
        self.measures.score(
            settings.tol_feas,
            settings.tol_gap_abs,
            settings.tol_gap_rel,
        )
    }

    fn is_solved(&self, settings: &Settings) -> bool {
        self.stopping_score(settings) < 1.0
    }

    /// Preserve certificate-first precedence if approximate tests overlap.
    fn terminal_status(&self, settings: &Settings, ray: Option<&Ray>) -> Option<SolverStatus> {
        ray.and_then(|ray| ray.certificate(settings, self.ktratio))
            .or_else(|| self.is_solved(settings).then_some(SolverStatus::Solved))
    }

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
            mu_step: diagnostics.mu_step,
            recentered: diagnostics.recentered,
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

/// Certificate measures of a homogeneous ray `(x, y, s)` in original units.
#[derive(Debug)]
struct Ray {
    x: Vec<f64>,
    y: Vec<f64>,
    s: Vec<f64>,
    bty: f64,
    ctx: f64,
    pinfeas: f64,
    dinfeas: f64,
}

impl Ray {
    fn new(
        problem: &Problem,
        equilibrated: &EquilibratedProblem<'_>,
        x: &[f64],
        y: &[f64],
        s: &[f64],
    ) -> Self {
        let x = equilibrated.unscale_x(x);
        let y = equilibrated.unscale_y(y);
        let s = equilibrated.unscale_s_by(s, 1.0);
        let bty = dot(problem.b(), &y);
        let ctx = dot(problem.c(), &x);
        let aty = matrix_transpose_mul(problem.a(), &y);
        let mut ax = matrix_mul(problem.a(), &x);
        let px = problem.p_mul(&x);
        let (norm_x, norm_y) = (two_norm(&x), two_norm(&y));
        let pinfeas = two_norm(&aty) / bty.abs().max(norm_y).max(RECOVERY_EPS);
        crate::matrix::add_assign(&mut ax, &s);
        let violation = two_norm(&ax);
        let dinfeas = (violation / ctx.abs().max(norm_x + two_norm(&s)).max(RECOVERY_EPS))
            .max(two_norm(&px) / ctx.abs().max(norm_x).max(RECOVERY_EPS));
        Self {
            x,
            y,
            s,
            bty,
            ctx,
            pinfeas,
            dinfeas,
        }
    }

    /// Lemma "tau_kappa_cond": a certificate once `kappa / tau` exceeds the
    /// gate and the ray's violations are negligible.
    fn certificate(&self, settings: &Settings, ktratio: f64) -> Option<SolverStatus> {
        if ktratio <= settings.certificate_ktratio {
            return None;
        }
        if self.bty < -settings.tol_infeas_abs && self.pinfeas < settings.tol_infeas_rel {
            return Some(SolverStatus::PrimalInfeasible);
        }
        if self.ctx < -settings.tol_infeas_abs && self.dinfeas < settings.tol_infeas_rel {
            return Some(SolverStatus::DualInfeasible);
        }
        None
    }
}

struct RecoveryWorkspace {
    expanded: Option<presolve::RecoveryWorkspace>,
    products: Products,
    slack: Vec<f64>,
    norms: DataNorms,
}

impl RecoveryWorkspace {
    fn new(problem: &Problem, original: Option<(&Problem, &Presolved)>) -> Self {
        Self {
            expanded: original.map(|(_, presolved)| presolved.recovery_workspace()),
            products: Products::default(),
            slack: Vec::new(),
            norms: DataNorms::new(original.map_or(problem, |(problem, _)| problem)),
        }
    }
}

#[cfg(test)]
fn recover(
    problem: &Problem,
    equilibrated: &EquilibratedProblem<'_>,
    state: &State,
    nu: usize,
    sy: f64,
    original: Option<(&Problem, &Presolved)>,
) -> RecoveredSolution {
    let mut out = RecoveredSolution::empty();
    recover_into(
        problem,
        equilibrated,
        state,
        nu,
        sy,
        original,
        &mut out,
        &mut RecoveryWorkspace::new(problem, original),
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn recover_into(
    problem: &Problem,
    equilibrated: &EquilibratedProblem<'_>,
    state: &State,
    nu: usize,
    sy: f64,
    original: Option<(&Problem, &Presolved)>,
    out: &mut RecoveredSolution,
    workspace: &mut RecoveryWorkspace,
) {
    // Keep the original denominator evaluation order.
    out.ktratio = if nu == 0 {
        0.0
    } else {
        sy / (nu as f64 * state.tau() * state.tau())
    };
    let inverse_tau = state.tau().max(RECOVERY_EPS).recip();
    equilibrated.unscale_x_into(state.x(), inverse_tau, &mut out.x);
    equilibrated.unscale_y_into(state.y(), inverse_tau, &mut out.y);
    equilibrated.unscale_s_into(state.s(), inverse_tau, &mut out.s);
    out.measures = match original {
        Some((problem, presolved)) => {
            let expanded = workspace
                .expanded
                .as_mut()
                .expect("presolve recovery workspace");
            let (x, y) = presolved.expand_point_into(&out.x, &out.y, expanded);
            workspace.products.update(problem, x, y);
            let products = &workspace.products;
            presolved.expand_s_into(problem, &out.s, &products.ax, &mut workspace.slack);
            Measures::with_norms(
                problem,
                x,
                y,
                &workspace.slack,
                &products.ax,
                &products.px,
                &products.aty,
                workspace.norms,
            )
        }
        None => {
            let products = out.products.get_or_insert_with(Products::default);
            products.update(problem, &out.x, &out.y);
            Measures::with_norms(
                problem,
                &out.x,
                &out.y,
                &out.s,
                &products.ax,
                &products.px,
                &products.aty,
                workspace.norms,
            )
        }
    };
}

#[allow(clippy::too_many_arguments)]
fn build_solution(
    status: SolverStatus,
    recovered: RecoveredSolution,
    iters: usize,
    solve_time: Duration,
    linear_solver: LinearSolver,
    settings: &Settings,
    init_report: init::InitReport,
    stats: LinearAlgebraStats,
    iteration_stats: Vec<IterationStats>,
    recenter_steps: usize,
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
            iteration_stats,
            recenter_steps,
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
fn build_certificate(
    status: SolverStatus,
    ray: Ray,
    iters: usize,
    solve_time: Duration,
    linear_solver: LinearSolver,
    settings: &Settings,
    init_report: init::InitReport,
    stats: LinearAlgebraStats,
    iteration_stats: Vec<IterationStats>,
    recenter_steps: usize,
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
            iteration_stats,
            recenter_steps,
            ..SolutionInfo::new(Measures::nan(), linear_solver, settings, Some(init_report))
        },
    }
}

fn time_limit_reached(started: Instant, time_limit: Option<Duration>) -> bool {
    time_limit.is_some_and(|limit| started.elapsed() >= limit)
}

fn infinity_norm(vector: &[f64]) -> f64 {
    faer::ColRef::from_slice(vector).norm_max()
}

fn two_norm(vector: &[f64]) -> f64 {
    faer::ColRef::from_slice(vector).norm_l2()
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
        let result = solve_core(&problem, &settings, None, Some((0, 0)), None, started).unwrap();
        assert_eq!(result.status(), SolverStatus::TimeLimit);
        assert_eq!(result.info().iters, 0);
        assert!(result.info().solve_time >= Duration::from_secs(2));
    }

    #[test]
    fn measures_use_infinity_norms_and_equation_scales() {
        let problem = Problem::new(
            Some(crate::matrix::test_matrix(2, 2, vec![2.0, 0.0, 0.0, 4.0]).unwrap()),
            vec![-3.0, 2.0],
            crate::matrix::test_matrix(2, 2, vec![1.0, 2.0, -3.0, 1.0]).unwrap(),
            vec![1.0, -2.0],
            vec![Cone::Nonnegative(2)],
        )
        .unwrap();
        let measures = Measures::new(&problem, &[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]);
        assert_eq!(measures.pres, 9.0);
        assert_eq!(measures.dres, 20.0);
        assert_eq!(measures.primal_scale, 7.0);
        assert_eq!(measures.dual_scale, 11.0);
        assert_eq!(measures.pobj, 10.0);
        assert_eq!(measures.dobj, -4.0);
        assert_eq!(measures.gap, 14.0);
        assert_eq!(measures.objective_scale, 4.0);
        assert_eq!(measures.comp, 39.0);
    }

    #[test]
    fn large_nullspace_iterates_do_not_relax_stationarity() {
        let problem = lp(vec![1.0, -1.0], 2, vec![0.0, 0.0], vec![1e-4]);
        let small = Measures::new(&problem, &[0.0], &[1.0, 1.0], &[0.0, 0.0]);
        let large = Measures::new(&problem, &[0.0], &[1e10, 1e10], &[0.0, 0.0]);
        assert_eq!(small.dres, large.dres);
        assert_eq!(small.dual_scale, large.dual_scale);
        assert!(large.score(1e-8, 1e-8, 1e-8) > 1.0);
        let invalid = Measures::new(&problem, &[0.0], &[f64::NAN, 1.0], &[0.0, 0.0]);
        assert_eq!(invalid.score(1e-8, 1e-8, 1e-8), f64::INFINITY);
    }

    #[test]
    fn presolved_acceptance_uses_original_equation_scales() {
        // Eliminating x0 from x0 + 1e6*x1 = 0 changes the objective scale
        // from 1 to 1e6. A stationarity defect of 1e-4 must not be hidden by it.
        let problem = Problem::new(
            Some(
                crate::matrix::test_matrix(3, 3, vec![0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0])
                    .unwrap(),
            ),
            vec![1.0, 0.0, 0.0],
            crate::matrix::test_matrix(
                4,
                3,
                vec![1.0, 1e6, 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
            )
            .unwrap(),
            vec![0.0, 1.0, 0.0, 0.0],
            vec![Cone::Zero(1), Cone::Nonnegative(3)],
        )
        .unwrap();
        let PresolveOutcome::Reduced(presolved) = presolve::presolve(&problem, None).unwrap()
        else {
            panic!("expected singleton substitution");
        };
        assert_eq!(presolved.problem().variable_count(), 2);
        let WarmStart { x, y, s } = presolved.reduce_warm_start(&WarmStart {
            x: vec![-1e6, 1.0, 0.0],
            y: vec![-1.0, 1e6 - 2.0 + 1e-4, 0.0, 1e6 - 1.0 + 1e-4],
            s: vec![0.0, 0.0, 1.0, 0.0],
        });
        let mut point = RecoveredSolution {
            products: None,
            measures: Measures::new(presolved.problem(), &x, &y, &s),
            x,
            y,
            s,
            ktratio: 0.0,
        };
        assert!(point.is_solved(&Settings::default()));
        point.measures = Measures::postsolve(&problem, &presolved, &point.x, &point.y, &point.s);
        assert!(!point.is_solved(&Settings::default()));
        assert!(point.measures.dres / point.measures.dual_scale > 1e-6);
    }

    #[test]
    fn shared_products_match_direct_path_residual_with_scaling_and_guarded_tau() {
        let problem = Problem::new(
            Some(crate::matrix::test_matrix(2, 2, vec![200.0, 0.0, 0.0, 0.03]).unwrap()),
            vec![8.0, -0.2],
            crate::matrix::test_matrix(4, 2, vec![10.0, 2.0, -0.1, 3.0, 2.0, -7.0, 4.0, 0.0])
                .unwrap(),
            vec![30.0, -2.0, 4.0, 0.1],
            vec![
                crate::Cone::Zero(1),
                crate::Cone::Nonnegative(1),
                crate::Cone::Zero(1),
                crate::Cone::Nonnegative(1),
            ],
        )
        .unwrap();
        let cone_rows = [false, true, false, true];
        for method in [Equilibration::None, Equilibration::Ruiz] {
            let scaled = EquilibratedProblem::new(
                &problem,
                &crate::Settings {
                    data_equilibration: method,
                    ..crate::Settings::default()
                },
            )
            .unwrap();
            for tau in [0.8, 1e-20] {
                let state = State::new(
                    vec![0.3, -0.4],
                    vec![0.0, 0.8, 0.0, 1.2],
                    vec![0.2, 1.3, -0.6, 0.9],
                    tau,
                )
                .unwrap();
                let mut recovered = recover(
                    &problem,
                    &scaled,
                    &state,
                    2,
                    dot(state.s(), state.y()),
                    None,
                );
                let products = recovered.products.as_mut().unwrap();
                scaled.scale_products(
                    &mut products.ax,
                    &mut products.px,
                    &mut products.aty,
                    tau.max(RECOVERY_EPS),
                );
                let direct = Products::new(scaled.problem(), state.x(), state.y());
                for (actual, expected) in products
                    .ax
                    .iter()
                    .chain(&products.px)
                    .chain(&products.aty)
                    .zip(direct.ax.iter().chain(&direct.px).chain(&direct.aty))
                {
                    assert!((actual - expected).abs() < 1e-12 * (1.0 + expected.abs()));
                }
                let mu = 1.06;
                let actual = products.delta_path(scaled.problem(), &state, &cone_rows, mu);
                let expected = direct.delta_path(scaled.problem(), &state, &cone_rows, mu);
                if method == Equilibration::None && tau == 0.8 {
                    // Independently expanded path equations for this fixed state.
                    let residual = [
                        70.988,
                        7.904,
                        22.012,
                        1.008 - 1.06 / 1.3,
                        -0.836,
                        -0.166 - 1.06 / 0.9,
                        -26.553,
                    ];
                    let reference = residual.iter().map(|v| v * v).sum::<f64>().sqrt() / 1.06;
                    assert!((actual - reference).abs() < 1e-12 * reference);
                }
                assert!((actual - expected).abs() < 1e-12 * expected);
            }
        }
    }

    #[test]
    fn cpo3_stopping_uses_gap_or_relative_gap_and_strict_feasibility() {
        let settings = Settings::default();
        let mut point = RecoveredSolution {
            products: None,
            x: vec![],
            y: vec![],
            s: vec![],
            ktratio: 0.0,
            measures: Measures {
                pobj: 1.0,
                dobj: 1.0,
                pres: 0.0,
                dres: 0.0,
                comp: 1e6,
                gap: 0.5e-8,
                primal_scale: 1.0,
                dual_scale: 1.0,
                objective_scale: 1.0,
            },
        };
        assert!(point.is_solved(&settings)); // No separate complementarity test.
        point.measures.gap = 2e-8;
        point.measures.comp = 0.0;
        assert!(!point.is_solved(&settings));
        point.measures.objective_scale = 4.0;
        assert!(point.is_solved(&settings)); // Relative gap alone passes.
        point.measures.pres = settings.tol_feas;
        assert!(!point.is_solved(&settings)); // Strict inequality.
        point.measures.primal_scale = 2.0;
        assert!(point.is_solved(&settings));
        point.ktratio = f64::INFINITY;
        assert!(point.is_solved(&settings)); // No solved-status homogeneous gate.
        let ray = Ray {
            x: vec![],
            y: vec![],
            s: vec![],
            bty: -1.0,
            ctx: 0.0,
            pinfeas: 0.0,
            dinfeas: 1.0,
        };
        assert_eq!(
            point.terminal_status(&settings, Some(&ray)),
            Some(SolverStatus::PrimalInfeasible)
        );
        assert_eq!(
            point.terminal_status(&settings, None),
            Some(SolverStatus::Solved)
        );
        point.ktratio = 0.0;
        point.measures.dres = f64::NAN;
        assert!(!point.is_solved(&settings));
    }

    #[test]
    fn cpo3_dual_certificate_uses_supplied_slack_and_sum_of_norms() {
        let problem = lp(vec![-1.0], 1, vec![0.0], vec![-1.0]);
        let equilibrated = EquilibratedProblem::new(
            &problem,
            &crate::Settings {
                data_equilibration: Equilibration::None,
                ..crate::Settings::default()
            },
        )
        .unwrap();
        let settings = Settings::default();
        let ray = Ray::new(&problem, &equilibrated, &[1.0], &[0.0], &[2.0]);
        assert_eq!(ray.dinfeas, 1.0 / 3.0);
        assert_eq!(ray.certificate(&settings, f64::INFINITY), None);
        let ray = Ray::new(&problem, &equilibrated, &[1.0], &[0.0], &[1.0]);
        assert_eq!(
            ray.certificate(&settings, settings.certificate_ktratio),
            None
        );
        assert_eq!(
            ray.certificate(&settings, f64::INFINITY),
            Some(SolverStatus::DualInfeasible)
        );
        assert_eq!(ray.s, vec![1.0]);
    }

    fn lp(a: Vec<f64>, rows: usize, b: Vec<f64>, c: Vec<f64>) -> Problem {
        let columns = c.len();
        Problem::new(
            None,
            c,
            crate::matrix::test_matrix(rows, columns, a).unwrap(),
            b,
            vec![Cone::Nonnegative(rows)],
        )
        .unwrap()
    }

    #[test]
    fn primal_infeasible_lp_returns_a_farkas_certificate() {
        // x <= -1 and x >= 0: A x + s = b with s >= 0.
        let problem = lp(vec![1.0, -1.0], 2, vec![-1.0, 0.0], vec![1.0]);
        let solution = solve(&problem, &Settings::default()).unwrap();
        assert_eq!(solution.status, SolverStatus::PrimalInfeasible);
        assert!(solution.x.iter().all(|v| v.is_nan()));
        let bty = dot(problem.b(), &solution.y);
        assert!((bty + 1.0).abs() < 1e-6);
        let aty = matrix_transpose_mul(problem.a(), &solution.y);
        assert!(aty.iter().all(|v| v.abs() < 1e-6));
        assert!(solution.y.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn dual_infeasible_lp_returns_an_unbounded_ray() {
        // minimize -x subject to x >= 0.
        let problem = lp(vec![-1.0], 1, vec![0.0], vec![-1.0]);
        let solution = solve(&problem, &Settings::default()).unwrap();
        assert_eq!(solution.status, SolverStatus::DualInfeasible);
        assert!(solution.y.iter().all(|v| v.is_nan()));
        assert!((dot(problem.c(), &solution.x) + 1.0).abs() < 1e-6);
        let ax = matrix_mul(problem.a(), &solution.x);
        assert!(
            ax.iter()
                .zip(&solution.s)
                .all(|(ax, s)| (ax + s).abs() < 1e-6)
        );
    }

    #[test]
    fn feasible_lp_is_not_declared_infeasible() {
        // minimize x subject to x >= 1.
        let problem = lp(vec![-1.0], 1, vec![-1.0], vec![1.0]);
        let solution = solve(&problem, &Settings::default()).unwrap();
        assert_eq!(solution.status, SolverStatus::Solved);
        assert!((solution.x[0] - 1.0).abs() < 1e-6);
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
        assert_eq!(cold.info.cvxopt_init_used, Some(false));
        let warm = WarmStart {
            x: cold.x.clone(),
            y: cold.y.clone(),
            s: cold.s.clone(),
        };
        let again = solve_from(&problem, &settings, Some(&warm)).unwrap();
        assert_eq!(again.status, SolverStatus::Solved);
        assert_eq!(again.info.warm_start_used, Some(true));
        assert_eq!(again.info.cvxopt_init_used, None);
        assert!(again.info.iters <= cold.info.iters);

        let junk = WarmStart {
            x: vec![1e6, -1e6],
            y: vec![-1.0, 1e-9, 5e3],
            s: vec![1e3, -7.0, 1e-9],
        };
        let rejected = solve_from(&problem, &settings, Some(&junk)).unwrap();
        assert_eq!(rejected.info.warm_start_used, Some(false));
        assert_eq!(rejected.info.iters, cold.info.iters);

        let forced = Settings {
            force_warm_start: true,
            presolve: false,
            ..Settings::default()
        };
        let used = solve_from(&problem, &forced, Some(&junk)).unwrap();
        assert_eq!(used.info.warm_start_used, Some(true));
        assert_eq!(used.status, SolverStatus::Solved);

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

    #[test]
    fn recovery_uses_the_tracked_slack() {
        let problem = Problem::new(
            Some(CscMatrix::identity(1).unwrap()),
            vec![0.0],
            CscMatrix::identity(1).unwrap(),
            vec![1.0],
            vec![Cone::Nonnegative(1)],
        )
        .unwrap();
        let equilibrated = EquilibratedProblem::new(
            &problem,
            &crate::Settings {
                data_equilibration: Equilibration::None,
                ..crate::Settings::default()
            },
        )
        .unwrap();
        let state = State::new(vec![0.25], vec![0.5], vec![1.0], 1.0).unwrap();

        let recovered = recover(
            &problem,
            &equilibrated,
            &state,
            1,
            dot(state.s(), state.y()),
            None,
        );

        assert_eq!(recovered.s, [0.5]);
        // The equality-derived slack would be b - A x = 0.75.
        assert_eq!(recovered.measures.pres, 0.25);
    }
}
