use std::time::Duration;

use crate::{LinearSolver, SolveError};

/// Cold-start strategy used when no warm start is accepted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Initialization {
    /// Spherical point with complementarity set by `Settings::initial_mu`.
    #[default]
    Canonical,
    /// CVXOPT-style linear-system initialization, projected onto the sphere.
    Cvxopt,
    /// Large-parameter endpoint with a certified central-path proximity bound.
    CertifiedEndpoint,
    /// Positive spherical predictor from the large-parameter endpoint.
    PositivePredictor,
}

impl Initialization {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Cvxopt => "cvxopt",
            Self::CertifiedEndpoint => "certified_endpoint",
            Self::PositivePredictor => "positive_predictor",
        }
    }
}

impl std::fmt::Display for Initialization {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// User-facing solver settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    // Run controls.
    pub max_iters: usize,
    pub time_limit: Option<Duration>,
    pub verbose: bool,
    /// Reduce the problem before solving (empty, singleton, and parallel
    /// rows; columns constrained only by bounds) and map the solution back.
    pub presolve: bool,

    // Termination and certificates.
    /// Absolute and relative feasibility tolerance: ||r||_inf < tol_feas *
    /// (1 + max infinity norm of the terms in the corresponding equation).
    pub tol_feas: f64,
    /// Absolute duality-gap tolerance.
    pub tol_gap_abs: f64,
    /// Relative duality-gap tolerance.
    pub tol_gap_rel: f64,
    /// Certificate slope threshold: `b'y < -tol_infeas_abs` or `c'x < -tol_infeas_abs`.
    pub tol_infeas_abs: f64,
    /// Certificate violation threshold, relative to the ray's slope and norm.
    pub tol_infeas_rel: f64,
    /// Certificates are considered only once `kappa / tau = mu / tau^2` exceeds this.
    pub certificate_ktratio: f64,

    // Initialization and equilibration.
    /// Use the best finite warm candidate even when its local score exceeds
    /// the screening threshold. No effect without a point or inequality rows.
    pub force_warm_start: bool,
    /// Cold-start strategy when no warm start is accepted. The positive
    /// predictor can fall back to the certified endpoint; failed alternative
    /// initializations ultimately fall back to the canonical point.
    /// Equality-only problems bypass cold-start initialization.
    pub initialization: Initialization,
    /// Initial complementarity beta for the canonical cold start.
    /// Defaults to 1e-3, with inequality slacks and multipliers sqrt(beta).
    /// Must be finite and in (0, 1]. Tau is chosen to keep
    /// ||(x, y, tau)||^2 = nu + 1. Also used after a failed alternative start;
    /// has no effect on accepted warm or alternative starts, or equality-only problems.
    pub initial_mu: f64,
    /// Maximum Ruiz sweeps on the augmented problem data (P, A, b, c).
    /// Zero leaves all problem data unscaled.
    pub ruiz_iters: usize,
    /// Nonnegative tolerance on augmented row infinity norms relative to one.
    pub ruiz_tol: f64,

    // KKT solves.
    /// Factorization backend for the interior-point path. Equality-only
    /// problems always use Faer with dynamically regularized LDL and LSMR.
    pub linear_solver: LinearSolver,
    /// Maximum parallelism for factorization and solves. Zero uses the
    /// backend default; one runs sequentially. Feral currently runs sequentially
    /// for every positive cap because it cannot configure its private pool size.
    pub threads: usize,
    /// Eliminate singleton nonnegative-cone rows from the KKT factorization.
    pub eliminate_bounds: bool,
    /// Initial adaptive floor for full-system factor diagonal magnitudes.
    /// Outer steps adjust this floor; zero starts at the positive lower bound
    /// 1e-20. Refinement always targets the original K(mu) system.
    /// Unused by the dedicated equality-only solver.
    pub refine_init_reg: f64,
    /// Additional correction solves per KKT solve; zero returns the initial solve without refinement.
    /// Equality-only problems instead use ten corrections before falling back to LSMR.
    pub refine_iters: usize,
    /// Positive componentwise backward-error target for iterative refinement.
    /// Unused for equality-only problems, which check optimality directly.
    pub refine_tol: f64,

    // Outer iterations.
    /// Fraction of the cone-boundary step to take, strictly between zero and one.
    /// The default is 0.999, leaving a relative margin of 1e-3.
    pub step_fraction: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_iters: 100,
            time_limit: None,
            verbose: false,
            presolve: true,

            tol_feas: 1e-8,
            tol_gap_abs: 1e-8,
            tol_gap_rel: 1e-8,
            tol_infeas_abs: 1e-8,
            tol_infeas_rel: 1e-8,
            certificate_ktratio: 1e9,

            force_warm_start: false,
            initialization: Initialization::Canonical,
            initial_mu: 1e-3,
            ruiz_iters: 2,
            ruiz_tol: 1e-2,

            linear_solver: LinearSolver::Faer,
            threads: 0,
            eliminate_bounds: true,
            refine_init_reg: 1e-12,
            refine_iters: 30,
            refine_tol: 1e-7,

            step_fraction: 0.999,
        }
    }
}

impl Settings {
    /// Check numerical ranges before solving. Defaults satisfy these invariants.
    pub fn validate(&self) -> Result<(), SolveError> {
        let beta = self.initial_mu;
        if !beta.is_finite() || beta <= 0.0 || beta > 1.0 {
            return Err(SolveError::InvalidSetting(
                "initial_mu must be finite and in (0, 1]",
            ));
        }
        if !self.refine_tol.is_finite() || self.refine_tol <= 0.0 {
            return Err(SolveError::InvalidSetting(
                "refine_tol must be finite and positive",
            ));
        }
        if !self.ruiz_tol.is_finite() || self.ruiz_tol < 0.0 {
            return Err(SolveError::InvalidSetting(
                "ruiz_tol must be finite and nonnegative",
            ));
        }

        if !self.step_fraction.is_finite() || self.step_fraction <= 0.0 || self.step_fraction >= 1.0
        {
            return Err(SolveError::InvalidSetting(
                "step_fraction must be finite and strictly between zero and one",
            ));
        }

        if self.max_iters == 0 {
            return Err(SolveError::InvalidSetting("max_iters must be positive"));
        }
        if !self.tol_gap_abs.is_finite() || self.tol_gap_abs <= 0.0 {
            return Err(SolveError::InvalidSetting(
                "tol_gap_abs must be finite and positive",
            ));
        }
        if !self.tol_feas.is_finite() || self.tol_feas <= 0.0 {
            return Err(SolveError::InvalidSetting(
                "tol_feas must be finite and positive",
            ));
        }
        if !self.tol_gap_rel.is_finite() || self.tol_gap_rel <= 0.0 {
            return Err(SolveError::InvalidSetting(
                "tol_gap_rel must be finite and positive",
            ));
        }
        if !self.refine_init_reg.is_finite() || self.refine_init_reg < 0.0 {
            return Err(SolveError::InvalidSetting(
                "refine_init_reg must be finite and nonnegative",
            ));
        }
        if !self.tol_infeas_abs.is_finite() || self.tol_infeas_abs < 0.0 {
            return Err(SolveError::InvalidSetting(
                "tol_infeas_abs must be finite and nonnegative",
            ));
        }
        if !self.tol_infeas_rel.is_finite() || self.tol_infeas_rel < 0.0 {
            return Err(SolveError::InvalidSetting(
                "tol_infeas_rel must be finite and nonnegative",
            ));
        }
        if !self.certificate_ktratio.is_finite() || self.certificate_ktratio < 1.0 {
            return Err(SolveError::InvalidSetting(
                "certificate_ktratio must be finite and at least one",
            ));
        }
        Ok(())
    }
}
