//! Recover unscaled points and evaluate stopping measures and infeasibility rays.

use crate::{
    Problem, Settings, SolverStatus,
    central_path::state::State,
    cone::ConeGeometry,
    linear_algebra::{dot, matrix_mul, matrix_products_into, matrix_transpose_mul},
    presolve::{self, Presolved},
    ruiz::EquilibratedProblem,
};

// CPO3's floor when recovering homogeneous iterates and measuring rays.
const RECOVERY_EPS: f64 = 1e-15;

/// Three products already needed for original-coordinate stopping checks.
#[derive(Debug, Default)]
struct Products {
    ax: Vec<f64>,
    px: Vec<f64>,
    aty: Vec<f64>,
}

impl Products {
    fn update(&mut self, problem: &Problem, x: &[f64], y: &[f64]) {
        matrix_products_into(problem.a(), x, y, &mut self.ax, &mut self.aty);
        if let Some(p) = problem.p() {
            crate::linear_algebra::symmetric_mul_into(p, x, &mut self.px);
        } else {
            self.px.resize(problem.variable_count(), 0.0);
            self.px.fill(0.0);
        }
    }
}

/// Objective values and absolute residuals of `(x, y, s)` on `problem`.
#[derive(Debug)]
pub(crate) struct Measures {
    pub(crate) pobj: f64,
    pub(crate) dobj: f64,
    pub(crate) pres: f64,
    pub(crate) dres: f64,
    pub(crate) comp: f64,
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
    /// Evaluate a point using its already computed `Ax`.
    pub(crate) fn new(problem: &Problem, x: &[f64], y: &[f64], s: &[f64], ax: Vec<f64>) -> Self {
        let products = Products {
            ax,
            px: problem.p_mul(x),
            aty: matrix_transpose_mul(problem.a(), y),
        };
        Self::from_products(problem, x, y, s, &products, DataNorms::new(problem))
    }

    fn from_products(
        problem: &Problem,
        x: &[f64],
        y: &[f64],
        s: &[f64],
        products: &Products,
        norms: DataNorms,
    ) -> Self {
        let Products { ax, px, aty } = products;
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

    pub(crate) fn nan() -> Self {
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

/// Unscaled solve coordinates, with stopping measures evaluated on original data.
#[derive(Debug)]
pub(crate) struct RecoveredSolution {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) s: Vec<f64>,
    pub(crate) ktratio: f64,
    pub(crate) measures: Measures,
}

impl RecoveredSolution {
    fn empty() -> Self {
        Self {
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

    pub(crate) fn is_solved(&self, settings: &Settings) -> bool {
        self.stopping_score(settings) < 1.0
    }

    /// Preserve certificate-first precedence if approximate tests overlap.
    pub(crate) fn termination(
        &self,
        settings: &Settings,
        certificate: Option<Certificate>,
    ) -> Option<Termination> {
        certificate.map(Termination::Certificate).or_else(|| {
            self.is_solved(settings)
                .then_some(Termination::Point(SolverStatus::Solved))
        })
    }
}

/// An ordinary exit returns the current point or an identified certificate.
#[derive(Debug)]
pub(crate) enum Termination {
    Point(SolverStatus),
    Certificate(Certificate),
}

/// A ray that passed the corresponding infeasibility test.
#[derive(Debug)]
pub(crate) enum Certificate {
    Primal(Ray),
    Dual(Ray),
}

/// Homogeneous ray `(x, y, s)` in original units, with its objective slopes.
#[derive(Debug)]
pub(crate) struct Ray {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) s: Vec<f64>,
    pub(crate) bty: f64,
    pub(crate) ctx: f64,
}

impl Ray {
    pub(crate) fn new(
        problem: &Problem,
        equilibrated: &EquilibratedProblem<'_>,
        x: &[f64],
        y: &[f64],
        s: &[f64],
    ) -> Self {
        let x = equilibrated.unscale_x(x);
        let y = equilibrated.unscale_y(y);
        let s = equilibrated.unscale_s(s);
        let bty = dot(problem.b(), &y);
        let ctx = dot(problem.c(), &x);
        Self { x, y, s, bty, ctx }
    }

    /// Lemma "tau_kappa_cond": a certificate once `kappa / tau` exceeds the
    /// gate and the ray's violations are negligible.
    pub(crate) fn into_certificate(
        self,
        problem: &Problem,
        settings: &Settings,
        ktratio: f64,
    ) -> Option<Certificate> {
        if ktratio <= settings.certificate_ktratio {
            return None;
        }
        if self.bty < -settings.tol_infeas_abs {
            let aty = matrix_transpose_mul(problem.a(), &self.y);
            let pinfeas = two_norm(&aty) / self.bty.abs().max(two_norm(&self.y)).max(RECOVERY_EPS);
            if pinfeas < settings.tol_infeas_rel {
                return Some(Certificate::Primal(self));
            }
        }
        if self.ctx < -settings.tol_infeas_abs {
            let mut ax = matrix_mul(problem.a(), &self.x);
            let px = problem.p_mul(&self.x);
            let norm_x = two_norm(&self.x);
            crate::linear_algebra::add_assign(&mut ax, &self.s);
            let violation = two_norm(&ax);
            let dinfeas = (violation
                / self
                    .ctx
                    .abs()
                    .max(norm_x + two_norm(&self.s))
                    .max(RECOVERY_EPS))
            .max(two_norm(&px) / self.ctx.abs().max(norm_x).max(RECOVERY_EPS));
            if dinfeas < settings.tol_infeas_rel {
                return Some(Certificate::Dual(self));
            }
        }
        None
    }
}

/// Fixed recovery context and reusable buffers for one solve.
pub(crate) struct Recovery<'a> {
    problem: &'a Problem,
    equilibrated: &'a EquilibratedProblem<'a>,
    nu: usize,
    presolve: Option<PresolveRecovery<'a>>,
    products: Products,
    norms: DataNorms,
    point: RecoveredSolution,
}

/// Original-data mapping and its buffers, present only after presolve reduction.
struct PresolveRecovery<'a> {
    problem: &'a Problem,
    presolved: &'a Presolved,
    geometry: &'a ConeGeometry,
    workspace: presolve::RecoveryWorkspace,
    slack: Vec<f64>,
}

impl<'a> Recovery<'a> {
    pub(crate) fn new(
        problem: &'a Problem,
        equilibrated: &'a EquilibratedProblem<'a>,
        geometry: &ConeGeometry,
        original: Option<(&'a Problem, &'a Presolved, &'a ConeGeometry)>,
    ) -> Self {
        Self {
            problem,
            equilibrated,
            nu: geometry.barrier_parameter(),
            presolve: original.map(|(problem, presolved, geometry)| PresolveRecovery {
                problem,
                presolved,
                geometry,
                workspace: presolved.recovery_workspace(),
                slack: Vec::new(),
            }),
            products: Products::default(),
            norms: DataNorms::new(original.map_or(problem, |(problem, _, _)| problem)),
            point: RecoveredSolution::empty(),
        }
    }

    pub(crate) fn update(&mut self, state: &State, sy: f64) -> &RecoveredSolution {
        let point = &mut self.point;
        // Keep the original denominator evaluation order.
        point.ktratio = if self.nu == 0 {
            0.0
        } else {
            sy / (self.nu as f64 * state.tau() * state.tau())
        };
        let inverse_tau = state.tau().max(RECOVERY_EPS).recip();
        self.equilibrated
            .unscale_x_into(state.x(), inverse_tau, &mut point.x);
        self.equilibrated
            .unscale_y_into(state.y(), inverse_tau, &mut point.y);
        self.equilibrated
            .unscale_s_into(state.s(), inverse_tau, &mut point.s);

        let (problem, x, y, s) = if let Some(presolve) = &mut self.presolve {
            let (x, y) =
                presolve
                    .presolved
                    .expand_point_into(&point.x, &point.y, &mut presolve.workspace);
            self.products.update(presolve.problem, x, y);
            presolve.presolved.expand_s_into(
                presolve.problem,
                presolve.geometry,
                &point.s,
                &self.products.ax,
                &mut presolve.slack,
            );
            (presolve.problem, x, y, presolve.slack.as_slice())
        } else {
            self.products.update(self.problem, &point.x, &point.y);
            (
                self.problem,
                point.x.as_slice(),
                point.y.as_slice(),
                point.s.as_slice(),
            )
        };
        point.measures = Measures::from_products(problem, x, y, s, &self.products, self.norms);
        point
    }

    pub(crate) fn into_point(self) -> RecoveredSolution {
        self.point
    }
}

fn infinity_norm(vector: &[f64]) -> f64 {
    faer::ColRef::from_slice(vector).norm_max()
}

fn two_norm(vector: &[f64]) -> f64 {
    faer::ColRef::from_slice(vector).norm_l2()
}

#[cfg(test)]
mod tests {
    use super::Recovery;
    use crate::central_path::state::State;
    use crate::cone::ConeGeometry;
    use crate::linear_algebra::matrix;
    use crate::presolve::{PresolveOutcome, presolve};
    use crate::ruiz::EquilibratedProblem;
    use crate::{Cone, Problem, Settings, WarmStart};
    use approx::assert_relative_eq;
    use rstest::rstest;

    #[test]
    fn presolve_does_not_hide_original_stationarity_error() {
        // Eliminating x0 turns c=(1,0,0) into an objective coefficient of magnitude 1e6.
        let p = Problem::new(
            Some(matrix(3, 3, &[0., 0., 0., 0., 2., 1., 0., 1., 2.])),
            vec![1., 0., 0.],
            matrix(4, 3, &[1., 1e6, 0., 0., 1., 1., 0., -1., 0., 0., 0., -1.]),
            vec![0., 1., 0., 0.],
            vec![Cone::Zero(1), Cone::Nonnegative(3)],
        )
        .unwrap();
        let PresolveOutcome::Reduced(reduced) = presolve(&p, None).unwrap() else {
            panic!("expected a reduced problem")
        };
        let geometry = ConeGeometry::new(reduced.problem()).unwrap();
        let settings = Settings {
            ruiz_iters: 0,
            ..Settings::default()
        };
        let eq = EquilibratedProblem::new(reduced.problem(), &settings).unwrap();
        let original_geometry = ConeGeometry::new(&p).unwrap();
        let mut recovery = Recovery::new(
            reduced.problem(),
            &eq,
            &geometry,
            Some((&p, &reduced, &original_geometry)),
        );
        for defect in [1e-4, 0., 1e-4] {
            let warm = reduced.reduce_warm_start(
                &geometry,
                &WarmStart {
                    x: vec![-1e6, 1., 0.],
                    y: vec![-1., 1e6 - 2. + defect, 0., 1e6 - 1. + defect],
                    s: vec![0., 0., 1., 0.],
                },
            );
            let state = State::new(warm.x, warm.s, warm.y, 1.).unwrap();
            let point = recovery.update(&state, 0.);
            // Original stationarity is (0, defect, 0), regardless of the reduced scale.
            assert_relative_eq!(point.measures.dres, defect, epsilon = 1e-9);
            assert_eq!(point.is_solved(&settings), defect == 0.);
        }
    }

    #[rstest]
    fn large_cancelling_multipliers_do_not_relax_stationarity(#[values(1., 1e10)] multiplier: f64) {
        let p = Problem::new(
            None,
            vec![1e-4],
            matrix(2, 1, &[1., -1.]),
            vec![0., 0.],
            vec![Cone::Nonnegative(2)],
        )
        .unwrap();
        let settings = Settings {
            ruiz_iters: 0,
            ..Settings::default()
        };
        let geometry = ConeGeometry::new(&p).unwrap();
        let eq = EquilibratedProblem::new(&p, &settings).unwrap();
        let mut recovery = Recovery::new(&p, &eq, &geometry, None);
        let state = State::new(vec![0.], vec![0., 0.], vec![multiplier, multiplier], 1.).unwrap();
        let point = recovery.update(&state, 0.);
        assert_relative_eq!(point.measures.dres, 1e-4, epsilon = 1e-15);
        assert!(!point.is_solved(&settings));
    }
}
