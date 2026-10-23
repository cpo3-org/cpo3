mod common;
use cpo3::{Cone, CscMatrix, Problem, Settings, Solution, SolveError, SolverStatus, solve};

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(x, y)| x * y).sum()
}
fn mul(a: &CscMatrix, x: &[f64], transpose: bool) -> Vec<f64> {
    let mut out = vec![0.0; if transpose { a.columns() } else { a.rows() }];
    for j in 0..a.columns() {
        for k in a.column_pointers()[j]..a.column_pointers()[j + 1] {
            let i = a.row_indices()[k];
            if transpose {
                out[j] += a.values()[k] * x[i]
            } else {
                out[i] += a.values()[k] * x[j]
            }
        }
    }
    out
}
fn norm(x: &[f64]) -> f64 {
    x.iter().map(|x| x.abs()).fold(0.0, f64::max)
}
fn p_mul(p: &Problem, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.; x.len()];
    if let Some(a) = p.p() {
        for j in 0..x.len() {
            for k in a.column_pointers()[j]..a.column_pointers()[j + 1] {
                let i = a.row_indices()[k];
                out[i] += a.values()[k] * x[j];
                if i != j {
                    out[j] += a.values()[k] * x[i];
                }
            }
        }
    }
    out
}
fn verify(p: &Problem, s: &Solution, expected: SolverStatus) {
    assert_eq!(s.status(), expected);
    assert_eq!(s.info().linear_algebra.factorizations, 1);
    assert!(s.info().linear_algebra.refinement_solves <= 10);
    match expected {
        SolverStatus::Solved => {
            let ax = mul(p.a(), s.x(), false);
            let aty = mul(p.a(), s.y(), true);
            let px = p_mul(p, s.x());
            let rp: Vec<_> = ax.iter().zip(p.b()).map(|(a, b)| a - b).collect();
            let rd: Vec<_> = px
                .iter()
                .zip(aty.iter())
                .zip(p.c())
                .map(|((p, a), c)| p + a + c)
                .collect();
            assert!(
                norm(&rp) <= 1e-7 * (1. + norm(p.b()).max(norm(&ax))),
                "primal {rp:?}"
            );
            assert!(
                norm(&rd) <= 1e-7 * (1. + norm(&px).max(norm(&aty)).max(norm(p.c()))),
                "dual {rd:?}"
            );
            assert!(s.x().iter().chain(s.y()).all(|v| v.is_finite()));
        }
        SolverStatus::PrimalInfeasible => {
            assert!((dot(p.b(), s.y()) + 1.).abs() < 1e-7);
            assert!(
                norm(&mul(p.a(), s.y(), true)) < 1e-7,
                "bad primal certificate {:?}",
                s.y()
            );
        }
        SolverStatus::DualInfeasible => {
            assert!((dot(p.c(), s.x()) + 1.).abs() < 1e-7);
            assert!(norm(&mul(p.a(), s.x(), false)) < 1e-7);
            assert!(norm(&p_mul(p, s.x())) < 1e-7);
        }
        _ => panic!("unexpected status"),
    }
}
fn problem(p: Vec<f64>, c: Vec<f64>, m: usize, a: Vec<f64>, b: Vec<f64>) -> Problem {
    let n = c.len();
    Problem::new(
        Some(common::matrix(n, n, p).unwrap()),
        c,
        common::matrix(m, n, a).unwrap(),
        b,
        if m == 0 { vec![] } else { vec![Cone::Zero(m)] },
    )
    .unwrap()
}
fn settings() -> Settings {
    Settings {
        presolve: false,
        ..Settings::default()
    }
}

#[test]
fn equality_lsmr_inconsistent_equalities() {
    for eps in [1., 1e-4, 1e-8] {
        // Identical left sides; contradictory right sides. Scale a column
        // to make the nonzero part of the KKT system nearly singular.
        let p = problem(
            vec![1., 0., 0., eps],
            vec![0., 0.],
            2,
            vec![eps, 1., eps, 1.],
            vec![1., 2.],
        );
        let s = solve(&p, &settings()).unwrap();
        verify(&p, &s, SolverStatus::PrimalInfeasible);
        assert!(s.info().linear_algebra.lsmr_iterations > 0);
        assert_eq!(s.info().linear_algebra.refinement_solves, 10);
    }
}
#[test]
fn equality_lsmr_unbounded() {
    for eps in [1., 1e-4, 1e-8, 1e-12] {
        let p = problem(
            vec![eps, 0., 0., 0.],
            vec![0., -1.],
            1,
            vec![1., 0.],
            vec![1.],
        );
        let s = solve(&p, &settings()).unwrap();
        verify(&p, &s, SolverStatus::DualInfeasible);
        assert!(s.info().linear_algebra.lsmr_iterations > 0);
        assert_eq!(s.info().linear_algebra.refinement_solves, 10);
    }
}
#[test]
fn equality_lsmr_singular_consistent() {
    for eps in [1e-4, 1e-8, 1e-12] {
        // Redundant equalities and a completely free variable; nonunique optimum.
        let p = problem(
            vec![eps, 0., 0., 0., 0., 0., 0., 0., 0.],
            vec![-1., 0., 0.],
            2,
            vec![0., 1., 0., 0., 2., 0.],
            vec![1., 2.],
        );
        let cfg = Settings {
            refine_init_reg: 1e-4,
            refine_iters: 0,
            ruiz_iters: 0,
            ..settings()
        };
        let s = solve(&p, &cfg).unwrap();
        verify(&p, &s, SolverStatus::Solved);
        assert!((s.x()[0] * eps - 1.).abs() < 1e-7);
        if eps == 1e-12 {
            // Slow Richardson convergence exhausts the budget; LSMR succeeds
            // with the existing factorization, despite refine_iters=0 above.
            assert_eq!(s.info().linear_algebra.refinement_solves, 10);
            assert!(s.info().linear_algebra.lsmr_iterations > 0);
        }
    }
}
#[test]
fn equality_lsmr_nearly_dependent_but_feasible() {
    for eps in [1e-3, 1e-4] {
        // The apparent inconsistent equalities have a large but valid solution.
        let p = problem(
            vec![1., 0., 0., 1.],
            vec![0., 0.],
            2,
            vec![1., 0., 1., eps],
            vec![1., 2.],
        );
        let s = solve(&p, &settings()).unwrap();
        verify(&p, &s, SolverStatus::Solved);
    }
}
#[test]
fn equality_lsmr_zero_operator() {
    let p = problem(vec![0.], vec![0.], 1, vec![0.], vec![1.]);
    let s = solve(&p, &settings()).unwrap();
    verify(&p, &s, SolverStatus::PrimalInfeasible);
    let p = problem(vec![0.], vec![-1.], 0, vec![], vec![]);
    let s = solve(&p, &settings()).unwrap();
    verify(&p, &s, SolverStatus::DualInfeasible);
}
#[test]
fn equality_lsmr_regular_system_uses_one_factorization() {
    let p = problem(
        vec![2., 0., 0., 2.],
        vec![-2., -4.],
        1,
        vec![1., 1.],
        vec![3.],
    );
    let s = solve(&p, &settings()).unwrap();
    verify(&p, &s, SolverStatus::Solved);
    assert_eq!(s.info().linear_algebra.factorizations, 1);
    assert!(s.info().linear_algebra.refinement_solves <= 10);
    assert_eq!(s.info().linear_algebra.lsmr_iterations, 0);
}

#[test]
fn equality_lsmr_ill_conditioned_solution() {
    // The regularized factors improve convergence on this difficult feasible system.
    let p = problem(
        vec![1., 0., 0., 1.],
        vec![0., 0.],
        2,
        vec![1., 0., 1., 1e-6],
        vec![1., 2.],
    );
    let s = solve(&p, &settings()).unwrap();
    verify(&p, &s, SolverStatus::Solved);
}

#[test]
fn equality_lsmr_extreme_near_dependence_returns_numerical_failure() {
    let p = problem(
        vec![1., 0., 0., 1.],
        vec![0., 0.],
        2,
        vec![1., 0., 1., 1e-8],
        vec![1., 2.],
    );
    assert!(matches!(
        solve(&p, &settings()),
        Err(SolveError::Numerical(_))
    ));
}

#[test]
fn equality_refinement_ignores_refine_tol() {
    let feasible = problem(
        vec![1., 0., 0., 1.],
        vec![0., 0.],
        2,
        vec![1., 0., 1., 1e-6],
        vec![1., 2.],
    );
    let infeasible = problem(vec![1.], vec![0.], 2, vec![1., 1.], vec![1., 2.]);
    for (p, status) in [
        (&feasible, SolverStatus::Solved),
        (&infeasible, SolverStatus::PrimalInfeasible),
    ] {
        let mut reference = None;
        for refine_tol in [1e-30, 1e-7, 1e2] {
            let cfg = Settings {
                refine_tol,
                ..settings()
            };
            let s = solve(p, &cfg).unwrap();
            verify(p, &s, status);
            let stats = s.info().linear_algebra;
            if status == SolverStatus::PrimalInfeasible {
                assert_eq!(stats.refinement_solves, 10);
            }
            let result = (
                s.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                s.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                stats.refinement_solves,
                stats.lsmr_iterations,
            );
            if let Some(expected) = &reference {
                assert_eq!(&result, expected);
            } else {
                reference = Some(result);
            }
        }
    }
}

#[test]
fn equality_presolve_recovers_points_and_bypasses_initialization() {
    let n = 32;
    let p = problem(
        (0..n)
            .flat_map(|i| (0..n).map(move |j| if i == j { 2. } else { 0.001 }))
            .collect(),
        vec![-(2. + 0.001 * (n - 1) as f64); n],
        2,
        (0..n)
            .map(|j| if j == 0 { 1. } else { 0. })
            .chain((0..n).map(|j| if j == 0 { 0. } else { 1. }))
            .collect(),
        vec![1., (n - 1) as f64],
    );
    let warm = cpo3::WarmStart {
        x: vec![7.; n],
        y: vec![2.; 2],
        s: vec![0.; 2],
    };
    for presolve in [false, true] {
        let cfg = Settings {
            presolve,
            cvxopt_init: true,
            force_warm_start: true,
            threads: 1,
            linear_solver: cpo3::LinearSolver::Feral,
            ..settings()
        };
        let s = cpo3::solve_from(&p, &cfg, Some(&warm)).unwrap();
        verify(&p, &s, SolverStatus::Solved);
        assert!(s.x().iter().all(|x| (x - 1.).abs() < 1e-7));
        assert_eq!(s.info().linear_solver, cpo3::LinearSolver::Faer);
        assert_eq!(s.info().warm_start_used, Some(false));
        assert_eq!(s.info().cvxopt_init_used, None);
        if presolve {
            assert!(s.info().presolve_columns_removed > 0);
        }
    }
}

#[test]
fn equality_dense_kkt_uses_configured_parallelism() {
    let n = 128;
    let p = problem(
        (0..n)
            .flat_map(|i| (0..n).map(move |j| if i == j { 2. } else { 0.001 }))
            .collect(),
        vec![-(2. + 0.001 * (n - 1) as f64); n],
        1,
        vec![1.; n],
        vec![n as f64],
    );
    for threads in [1, 2] {
        let s = solve(
            &p,
            &Settings {
                threads,
                ..settings()
            },
        )
        .unwrap();
        verify(&p, &s, SolverStatus::Solved);
        assert!(s.x().iter().all(|x| (x - 1.).abs() < 1e-7));
    }
}
