use std::time::Duration;

use crate::{Equilibration, LinearSolver, SolveError};

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
    /// point `(0, e, e, 1)` when no warm start is supplied or accepted: one
    /// solve of `[P, A'; A, -H][x; y] = [-c; b]` with `H` the identity on
    /// cone rows and zero on equality rows (two solves on one factorization
    /// for linear programs), `s = b - A x`, and cone components of `s` and
    /// `y` shifted so that their minimum is at least one. A failed solve
    /// falls back to the canonical point.
    pub cvxopt_init: bool,
    pub data_equilibration: Equilibration,
    /// Maximum equilibration sweeps: primal columns, then constraint rows.
    /// The linear data is normalized once after the matrix sweeps.
    /// Zero leaves all problem data unscaled.
    pub max_equilibration_passes: usize,
    /// Nonnegative tolerance on scaled P/A row infinity norms relative to one.
    pub equilibration_tolerance: f64,

    // KKT solves.
    pub linear_solver: LinearSolver,
    /// Eliminate singleton nonnegative-cone rows from the KKT factorization.
    pub kkt_bound_elimination: bool,
    /// Minimum magnitude of full-system factor diagonals; refinement targets
    /// the true shifted KKT system. Zero disables static regularization;
    /// the backend can still regularize pivots dynamically.
    pub min_static_regularization: f64,
    /// Additional correction solves per KKT solve; zero returns the initial solve without refinement.
    pub max_refinement_steps: usize,
    /// Positive componentwise backward-error target for iterative refinement.
    pub refinement_tolerance: f64,

    // Outer iterations.
    /// Initial requested KKT shift floor for problems with no barrier. Numerical failures raise
    /// the shift; accurate steps relax the numerical-retry floor. Stalled outer
    /// progress can also relax the requested floor for barrier-free problems.
    /// Centering targets use complementarity in the current spherical coordinates.
    pub initial_shift_floor: f64,
    /// Fraction of the cone-boundary step to take, strictly between zero and one.
    /// The default leaves a relative margin of sqrt(machine epsilon).
    pub step_fraction: f64,
    /// Most Gondzio centrality correctors per iteration, each one extra solve
    /// on the iteration's factorization; 0 disables them.
    pub max_centrality_correctors: usize,
    /// Skip extra centrality correctors when the undamped boundary step reaches
    /// this threshold in [0, 1]. Zero skips all extra correctors.
    pub corrector_gate: f64,
    /// Recenter and pace complementarity reduction when progress stalls; off by default.
    pub governor: bool,
    /// Maximum governor-triggered recentering steps; zero disables the governor.
    pub max_recenters: usize,
    /// Persistent centering floor in [0, 1] after the governor first triggers.
    pub recenter_sigma_floor: f64,
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
            data_equilibration: Equilibration::Ruiz,
            max_equilibration_passes: 1,
            equilibration_tolerance: 1e-2,

            linear_solver: LinearSolver::Faer,
            kkt_bound_elimination: true,
            min_static_regularization: 0.0,
            max_refinement_steps: 30,
            refinement_tolerance: 1e-8,

            initial_shift_floor: 1e-10,
            step_fraction: 1.0 - f64::EPSILON.sqrt(),
            max_centrality_correctors: 0,
            corrector_gate: 0.9,
            governor: false,
            max_recenters: 8,
            recenter_sigma_floor: 1.0 / 30.0,
        }
    }
}

impl Settings {
    /// Check numerical ranges before solving. Defaults satisfy these invariants.
    pub fn validate(&self) -> Result<(), SolveError> {
        if !self.refinement_tolerance.is_finite() || self.refinement_tolerance <= 0.0 {
            return Err(SolveError::InvalidSetting(
                "refinement_tolerance must be finite and positive",
            ));
        }
        if !self.equilibration_tolerance.is_finite() || self.equilibration_tolerance < 0.0 {
            return Err(SolveError::InvalidSetting(
                "equilibration_tolerance must be finite and nonnegative",
            ));
        }
        if !self.recenter_sigma_floor.is_finite()
            || !(0.0..=1.0).contains(&self.recenter_sigma_floor)
        {
            return Err(SolveError::InvalidSetting(
                "recenter_sigma_floor must be finite and between zero and one inclusive",
            ));
        }
        if !self.step_fraction.is_finite() || self.step_fraction <= 0.0 || self.step_fraction >= 1.0
        {
            return Err(SolveError::InvalidSetting(
                "step_fraction must be finite and strictly between zero and one",
            ));
        }
        if !self.corrector_gate.is_finite() || !(0.0..=1.0).contains(&self.corrector_gate) {
            return Err(SolveError::InvalidSetting(
                "corrector_gate must be finite and between zero and one inclusive",
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
        if !self.min_static_regularization.is_finite() || self.min_static_regularization < 0.0 {
            return Err(SolveError::InvalidSetting(
                "min_static_regularization must be finite and nonnegative",
            ));
        }
        if !self.initial_shift_floor.is_finite() || self.initial_shift_floor <= 0.0 {
            return Err(SolveError::InvalidSetting(
                "initial_shift_floor must be finite and positive",
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
