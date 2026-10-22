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
    pub(crate) fn termination(&self, settings: &Settings, ray: Option<Ray>) -> Option<Termination> {
        ray.and_then(|ray| ray.into_certificate(settings, self.ktratio))
            .map(Termination::Certificate)
            .or_else(|| {
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

/// Certificate measures of a homogeneous ray `(x, y, s)` in original units.
#[derive(Debug)]
pub(crate) struct Ray {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) s: Vec<f64>,
    pub(crate) bty: f64,
    pub(crate) ctx: f64,
    pinfeas: f64,
    dinfeas: f64,
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
        let aty = matrix_transpose_mul(problem.a(), &y);
        let mut ax = matrix_mul(problem.a(), &x);
        let px = problem.p_mul(&x);
        let (norm_x, norm_y) = (two_norm(&x), two_norm(&y));
        let pinfeas = two_norm(&aty) / bty.abs().max(norm_y).max(RECOVERY_EPS);
        crate::linear_algebra::add_assign(&mut ax, &s);
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
    fn into_certificate(self, settings: &Settings, ktratio: f64) -> Option<Certificate> {
        if ktratio <= settings.certificate_ktratio {
            return None;
        }
        if self.bty < -settings.tol_infeas_abs && self.pinfeas < settings.tol_infeas_rel {
            return Some(Certificate::Primal(self));
        }
        if self.ctx < -settings.tol_infeas_abs && self.dinfeas < settings.tol_infeas_rel {
            return Some(Certificate::Dual(self));
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

        let (problem, x, y, s) = match &mut self.presolve {
            Some(presolve) => {
                let (x, y) = presolve.presolved.expand_point_into(
                    &point.x,
                    &point.y,
                    &mut presolve.workspace,
                );
                self.products.update(presolve.problem, x, y);
                presolve.presolved.expand_s_into(
                    presolve.problem,
                    presolve.geometry,
                    &point.s,
                    &self.products.ax,
                    &mut presolve.slack,
                );
                (presolve.problem, x, y, presolve.slack.as_slice())
            }
            None => {
                self.products.update(self.problem, &point.x, &point.y);
                (
                    self.problem,
                    point.x.as_slice(),
                    point.y.as_slice(),
                    point.s.as_slice(),
                )
            }
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
    use super::*;
    use crate::{Cone, CscMatrix, WarmStart, presolve::PresolveOutcome};

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
    fn measures_use_infinity_norms_and_equation_scales() {
        let problem = Problem::new(
            Some(crate::linear_algebra::test_matrix(2, 2, vec![2.0, 0.0, 0.0, 4.0]).unwrap()),
            vec![-3.0, 2.0],
            crate::linear_algebra::test_matrix(2, 2, vec![1.0, 2.0, -3.0, 1.0]).unwrap(),
            vec![1.0, -2.0],
            vec![Cone::Nonnegative(2)],
        )
        .unwrap();
        let x = [1.0, 2.0];
        let ax = matrix_mul(problem.a(), &x);
        let measures = Measures::new(&problem, &x, &[3.0, 4.0], &[5.0, 6.0], ax);
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
        let small = Measures::new(
            &problem,
            &[0.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            matrix_mul(problem.a(), &[0.0]),
        );
        let large = Measures::new(
            &problem,
            &[0.0],
            &[1e10, 1e10],
            &[0.0, 0.0],
            matrix_mul(problem.a(), &[0.0]),
        );
        assert_eq!(small.dres, large.dres);
        assert_eq!(small.dual_scale, large.dual_scale);
        assert!(large.score(1e-8, 1e-8, 1e-8) > 1.0);
        let invalid = Measures::new(
            &problem,
            &[0.0],
            &[f64::NAN, 1.0],
            &[0.0, 0.0],
            matrix_mul(problem.a(), &[0.0]),
        );
        assert_eq!(invalid.score(1e-8, 1e-8, 1e-8), f64::INFINITY);
    }

    #[test]
    fn presolved_acceptance_uses_original_equation_scales() {
        // Eliminating x0 from x0 + 1e6*x1 = 0 changes the objective scale
        // from 1 to 1e6. A stationarity defect of 1e-4 must not be hidden by it.
        let problem = Problem::new(
            Some(
                crate::linear_algebra::test_matrix(
                    3,
                    3,
                    vec![0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0],
                )
                .unwrap(),
            ),
            vec![1.0, 0.0, 0.0],
            crate::linear_algebra::test_matrix(
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
        let geometry = ConeGeometry::new(presolved.problem()).unwrap();
        let WarmStart { x, y, s } = presolved.reduce_warm_start(
            &geometry,
            &WarmStart {
                x: vec![-1e6, 1.0, 0.0],
                y: vec![-1.0, 1e6 - 2.0 + 1e-4, 0.0, 1e6 - 1.0 + 1e-4],
                s: vec![0.0, 0.0, 1.0, 0.0],
            },
        );
        let settings = Settings {
            ruiz_iters: 0,
            ..Settings::default()
        };
        let equilibrated = EquilibratedProblem::new(presolved.problem(), &settings).unwrap();
        let state = State::new(x, s, y, 1.0).unwrap();
        let sy = dot(state.s(), state.y());
        let mut reduced = Recovery::new(presolved.problem(), &equilibrated, &geometry, None);
        assert!(reduced.update(&state, sy).is_solved(&settings));

        let original_geometry = ConeGeometry::new(&problem).unwrap();
        let mut recovery = Recovery::new(
            presolved.problem(),
            &equilibrated,
            &geometry,
            Some((&problem, &presolved, &original_geometry)),
        );
        for _ in 0..2 {
            let point = recovery.update(&state, sy);
            assert!(!point.is_solved(&settings));
            assert!(point.measures.dres / point.measures.dual_scale > 1e-6);
        }
    }

    #[test]
    fn cpo3_stopping_uses_gap_or_relative_gap_and_strict_feasibility() {
        let settings = Settings::default();
        let mut point = RecoveredSolution {
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
        assert!(matches!(
            point.termination(&settings, Some(ray)),
            Some(Termination::Certificate(Certificate::Primal(_)))
        ));
        assert!(matches!(
            point.termination(&settings, None),
            Some(Termination::Point(SolverStatus::Solved))
        ));
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
                ruiz_iters: 0,
                ..crate::Settings::default()
            },
        )
        .unwrap();
        let settings = Settings::default();
        let ray = Ray::new(&problem, &equilibrated, &[1.0], &[0.0], &[2.0]);
        assert_eq!(ray.dinfeas, 1.0 / 3.0);
        assert!(ray.into_certificate(&settings, f64::INFINITY).is_none());
        let ray = Ray::new(&problem, &equilibrated, &[1.0], &[0.0], &[1.0]);
        assert!(
            ray.into_certificate(&settings, settings.certificate_ktratio)
                .is_none()
        );
        let ray = Ray::new(&problem, &equilibrated, &[1.0], &[0.0], &[1.0]);
        let Some(Certificate::Dual(ray)) = ray.into_certificate(&settings, f64::INFINITY) else {
            panic!("expected a dual infeasibility certificate");
        };
        assert_eq!(ray.s, vec![1.0]);
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
                ruiz_iters: 0,
                ..crate::Settings::default()
            },
        )
        .unwrap();
        let state = State::new(vec![0.25], vec![0.5], vec![1.0], 1.0).unwrap();

        let geometry = ConeGeometry::new(&problem).unwrap();
        let mut recovery = Recovery::new(&problem, &equilibrated, &geometry, None);
        let recovered = recovery.update(&state, dot(state.s(), state.y()));

        assert_eq!(recovered.s, [0.5]);
        // The equality-derived slack would be b - A x = 0.75.
        assert_eq!(recovered.measures.pres, 0.25);
    }
}
