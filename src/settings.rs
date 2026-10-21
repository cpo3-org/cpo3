use std::time::Duration;

use crate::{LinearSolver, SolveError};

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
    /// Start from cpo3's CVXOPT/Clarabel-style point instead of the canonical
    /// spherical cold start when no warm start is supplied or accepted: one
    /// solve of `[P, A'; A, -H][x; y] = [-c; b]` with `H` the identity on
    /// cone rows and zero on equality rows (two solves on one factorization
    /// for linear programs), `s = b - A x`, and cone components of `s` and
    /// `y` shifted so that their minimum is at least one. A failed solve
    /// falls back to the canonical point.
    pub cvxopt_init: bool,
    /// Initial complementarity beta for the canonical cold start.
    /// Defaults to 1e-3, with inequality slacks and multipliers sqrt(beta).
    /// Must be finite and in (0, 1]. Tau is chosen to keep
    /// ||(x, y, tau)||^2 = nu + 1. Also used after a failed CVXOPT start;
    /// has no effect on accepted warm/CVXOPT starts or equality-only problems.
    pub initial_mu: f64,
    /// Maximum Ruiz sweeps on the augmented problem data (P, A, b, c).
    /// Zero leaves all problem data unscaled.
    pub ruiz_iters: usize,
    /// Nonnegative tolerance on augmented row infinity norms relative to one.
    pub ruiz_tol: f64,

    // KKT solves.
    pub linear_solver: LinearSolver,
    /// Eliminate singleton nonnegative-cone rows from the KKT factorization.
    pub eliminate_bounds: bool,
    /// Initial adaptive floor for full-system factor diagonal magnitudes.
    /// Outer steps adjust this floor; zero starts at the positive lower bound
    /// 1e-20. Refinement always targets the original K(mu) system.
    pub refine_init_reg: f64,
    /// Additional correction solves per KKT solve; zero returns the initial solve without refinement.
    pub refine_iters: usize,
    /// Positive componentwise backward-error target for iterative refinement.
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
            cvxopt_init: false,
            initial_mu: 1e-3,
            ruiz_iters: 2,
            ruiz_tol: 1e-2,

            linear_solver: LinearSolver::Faer,
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
