//! cpo3's Ax+s=b convention at the standalone presolve boundary.
use crate::{
    Cone, Problem, WarmStart,
    cone::{ConeGeometry, PresolveForm},
};
use ::presolve as native;
use std::time::Duration;

pub(crate) enum PresolveOutcome {
    Reduced(Box<Presolved>),
    Solved(Vec<f64>, Vec<f64>),
    Unchanged,
    PrimalInfeasible(Vec<f64>),
    DualInfeasible(Vec<f64>),
}
pub(crate) struct Presolved {
    problem: Problem,
    postsolve: native::postsolve::Postsolve,
    map: native::problem::ConicMap,
    original_rows: usize,
    original_columns: usize,
}
/// Reusable native mapping and postsolve buffers for iteration recovery.
pub(crate) struct RecoveryWorkspace {
    dual: native::postsolve::PrimalCertificate,
    work: native::postsolve::Workspace,
    conic_slack: Vec<f64>,
    y: Vec<f64>,
}

impl From<&Problem> for native::Problem {
    fn from(p: &Problem) -> Self {
        let (rows, cones) = domains(p.b(), p.cones());
        native::Problem {
            p: p.p().cloned().map(crate::CscMatrix::into_native),
            c: p.c().to_vec(),
            c0: 0.0,
            a: p.a().clone().into_native(),
            rows,
            variable_bounds: vec![],
            cones,
        }
    }
}
pub(crate) fn domains(
    b: &[f64],
    input_cones: &[Cone],
) -> (Vec<native::problem::Constraint>, Vec<native::problem::Cone>) {
    let mut rows = Vec::with_capacity(b.len());
    let mut cones = Vec::new();
    let mut offset = 0;
    for cone in input_cones {
        let n = cone.vector_dim();
        match cone.presolve_form() {
            PresolveForm::Equality => {
                rows.extend(b[offset..offset + n].iter().map(|&b| {
                    native::problem::Constraint::Linear(native::problem::Bounds::fixed(b))
                }))
            }
            PresolveForm::UpperBound => rows.extend(b[offset..offset + n].iter().map(|&b| {
                native::problem::Constraint::Linear(native::problem::Bounds {
                    lower: f64::NEG_INFINITY,
                    upper: b,
                })
            })),
            PresolveForm::Conic(native_cone) => {
                let block = cones.len();
                cones.push(native_cone);
                rows.extend(
                    b[offset..offset + n]
                        .iter()
                        .map(|&rhs| native::problem::Constraint::Cone { rhs, block }),
                );
            }
        }
        offset += n;
    }
    (rows, cones)
}
fn restore_dual(linear: &[usize], conic: &[usize], y: &[f64], w: &[f64]) -> Vec<f64> {
    let mut out = Vec::new();
    restore_dual_into(
        linear,
        conic,
        y.iter().copied(),
        w.iter().copied(),
        &mut out,
    );
    out
}

fn restore_dual_into(
    linear: &[usize],
    conic: &[usize],
    y: impl IntoIterator<Item = f64>,
    w: impl IntoIterator<Item = f64>,
    out: &mut Vec<f64>,
) {
    out.resize(linear.len() + conic.len(), 0.0);
    out.fill(0.0);
    for (&i, v) in linear.iter().zip(y) {
        out[i] = -v;
    }
    for (&i, v) in conic.iter().zip(w) {
        out[i] = v;
    }
}

fn original_dual(original: &Problem, y: &[f64], w: &[f64]) -> Vec<f64> {
    let mut y = y.iter();
    let mut w = w.iter();
    let mut out = Vec::with_capacity(original.constraint_count());
    for cone in original.cones() {
        let n = cone.vector_dim();
        match cone.presolve_form() {
            PresolveForm::Equality | PresolveForm::UpperBound => {
                out.extend(y.by_ref().take(n).map(|v| -v));
            }
            PresolveForm::Conic(_) => out.extend(w.by_ref().take(n).copied()),
        }
    }
    out
}
pub(crate) fn presolve(
    original: &Problem,
    time_limit: Option<Duration>,
) -> Result<PresolveOutcome, native::InitError> {
    let linear = original.p().is_none();
    // Global reduction searches must amortize their setup and matrix scans.
    // Keep small models on the existing inexpensive presolve path.
    let extended_lp = linear && original.variable_count() >= 1_000;
    let mut settings = native::Settings {
        time_limit: time_limit
            .unwrap_or(Duration::from_secs(60))
            .min(Duration::from_secs(60)),
        rules: native::settings::Rules {
            sparsification: linear,
            implied_free_equalities: extended_lp,
            bound_shift: false,
            lp_folding: extended_lp,
            ..native::settings::Rules::default()
        },
        ..native::Settings::default()
    };
    if linear {
        // Broaden bounded-variable substitution without risking Hessian fill.
        // Keep the default largest pivot and single attempt to bound growth and work.
        settings.equalities.require_free_variable = false;
        settings.equalities.min_column_length = 1;
        settings.equalities.max_row_length = 16;
        settings.equalities.max_column_length = 16;
        settings.equalities.cost_aware = true;
        settings.sparsification.allow_auxiliary_variables = false;
    }
    let result = native::Presolver::new(settings)?.presolve(original.into());
    Ok(match result.outcome {
        native::Outcome::Unchanged(_) => PresolveOutcome::Unchanged,
        native::Outcome::Solved(p) => {
            PresolveOutcome::Solved(p.x, original_dual(original, &p.y, &p.conic_dual))
        }
        native::Outcome::Infeasible(p) => {
            PresolveOutcome::PrimalInfeasible(original_dual(original, &p.y, &p.conic_dual))
        }
        native::Outcome::Unbounded(c) => PresolveOutcome::DualInfeasible(c.ray),
        native::Outcome::Reduced(r) => {
            let export = r.problem.into_conic();
            let p = export.problem;
            let cones = p
                .cones
                .iter()
                .map(|&c| {
                    Cone::from_presolve(c).expect("presolve emitted an unsupported cpo3 cone")
                })
                .collect();
            let problem = Problem::from_validated(
                p.p.map(crate::CscMatrix::from_native),
                p.c,
                crate::CscMatrix::from_native(p.a),
                p.b,
                cones,
            );
            PresolveOutcome::Reduced(Box::new(Presolved {
                problem,
                map: export.map,
                postsolve: r.postsolve,
                original_rows: original.constraint_count(),
                original_columns: original.variable_count(),
            }))
        }
    })
}
impl Presolved {
    pub(crate) fn problem(&self) -> &Problem {
        &self.problem
    }
    pub(crate) fn rows_removed(&self) -> usize {
        self.original_rows
            .saturating_sub(self.problem.constraint_count())
    }
    pub(crate) fn columns_removed(&self) -> usize {
        self.original_columns
            .saturating_sub(self.problem.variable_count())
    }
    pub(crate) fn recovery_workspace(&self) -> RecoveryWorkspace {
        RecoveryWorkspace {
            dual: native::postsolve::PrimalCertificate {
                y: vec![0.0; self.map.linear_row_count()],
                z: vec![0.0; self.map.variable_count()],
                conic_dual: vec![0.0; self.map.conic_row_count()],
            },
            work: self.postsolve.workspace(),
            conic_slack: vec![0.0; self.postsolve.reduced_conic_dimension()],
            y: Vec::new(),
        }
    }

    pub(crate) fn expand_point(&self, x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut workspace = self.recovery_workspace();
        let (x, _) = self.expand_point_into(x, y, &mut workspace);
        let x = x.to_vec();
        (x, workspace.y)
    }

    pub(crate) fn expand_point_into<'a>(
        &'a self,
        x: &[f64],
        y: &[f64],
        workspace: &'a mut RecoveryWorkspace,
    ) -> (&'a [f64], &'a [f64]) {
        let RecoveryWorkspace {
            dual,
            work,
            conic_slack,
            y: full_y,
        } = workspace;
        self.map
            .dual_into(y, &mut dual.y, &mut dual.z, &mut dual.conic_dual);
        let point = self.postsolve.recover_borrowed(
            native::postsolve::SolutionRef {
                x,
                y: &dual.y,
                z: &dual.z,
                conic_dual: &dual.conic_dual,
                conic_slack,
            },
            work,
        );
        restore_dual_into(
            self.postsolve.original_linear_rows(),
            self.postsolve.original_conic_rows(),
            point.y(),
            point.conic_dual(),
            full_y,
        );
        (point.x(), full_y)
    }

    pub(crate) fn expand_ray(&self, x: &[f64]) -> Vec<f64> {
        self.postsolve.recover_primal_ray(x)
    }
    pub(crate) fn expand_certificate(&self, y: &[f64]) -> Vec<f64> {
        let p = self
            .postsolve
            .recover_primal_certificate(self.map.dual(y).as_ref());
        restore_dual(
            self.postsolve.original_linear_rows(),
            self.postsolve.original_conic_rows(),
            &p.y,
            &p.conic_dual,
        )
    }
    pub(crate) fn expand_s_from_ax(
        &self,
        original: &Problem,
        geometry: &ConeGeometry,
        s: &[f64],
        ax: &[f64],
    ) -> Vec<f64> {
        let mut full = Vec::new();
        self.expand_s_into(original, geometry, s, ax, &mut full);
        full
    }

    pub(crate) fn expand_s_into(
        &self,
        original: &Problem,
        geometry: &ConeGeometry,
        s: &[f64],
        ax: &[f64],
        full: &mut Vec<f64>,
    ) {
        geometry.reconstruct_slack(original.b(), ax, full);
        for &row in self.postsolve.original_conic_rows() {
            full[row] = original.b()[row] - ax[row];
        }
        let conic_start = self.map.conic_offset();
        for &(original, reduced) in self.postsolve.surviving_conic_coordinates() {
            full[self.postsolve.original_conic_rows()[original]] = s[conic_start + reduced];
        }
    }
    pub(crate) fn reduce_warm_start(&self, geometry: &ConeGeometry, warm: &WarmStart) -> WarmStart {
        let y: Vec<_> = self
            .postsolve
            .original_linear_rows()
            .iter()
            .map(|&i| -warm.y[i])
            .collect();
        let z = vec![0.0; self.original_columns];
        let conic_dual: Vec<_> = self
            .postsolve
            .original_conic_rows()
            .iter()
            .map(|&i| warm.y[i])
            .collect();
        let conic_slack: Vec<_> = self
            .postsolve
            .original_conic_rows()
            .iter()
            .map(|&i| warm.s[i])
            .collect();
        let p = self
            .postsolve
            .reduce_warm_start(native::postsolve::SolutionRef {
                x: &warm.x,
                y: &y,
                z: &z,
                conic_dual: &conic_dual,
                conic_slack: &conic_slack,
            });
        let mut y = self.map.warm_dual(&p);
        let ax = crate::linear_algebra::matrix_mul(self.problem.a(), &p.x);
        let mut s = Vec::new();
        geometry.reconstruct_slack(self.problem.b(), &ax, &mut s);
        geometry.project_dual(&mut y);
        WarmStart { x: p.x, y, s }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Cone, CscMatrix, Problem};

    #[test]
    fn native_presolve_converts_cpo3_cones_and_recovers_original_coordinates() {
        let cases = [
            (
                Cone::SecondOrder(3),
                vec![0],
                vec![-1.0],
                vec![0., 1., 0.],
                1.0,
            ),
            (
                Cone::PositiveSemidefinite(2),
                vec![0],
                vec![-1.0],
                vec![0., 0., 1.],
                0.0,
            ),
            (
                Cone::PositiveSemidefinite(2),
                vec![1],
                vec![-1.0],
                vec![0., 0., 0.],
                0.0,
            ),
        ];
        for (cone, rows, values, b, expected) in cases {
            let a = CscMatrix::new(3, 1, vec![0, values.len()], rows, values).unwrap();
            let p = Problem::new(None, vec![1.0], a, b, vec![cone]).unwrap();
            let result = ::presolve::Presolver::default().presolve((&p).into());
            let ::presolve::Outcome::Solved(s) = result.outcome else {
                panic!("expected presolve solution")
            };
            assert!((s.x[0] - expected).abs() < 1e-7);
            assert_eq!(s.conic_dual.len(), 3);
            assert_eq!(s.conic_slack.len(), 3);
        }
    }
}
