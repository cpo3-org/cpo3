mod common;
use cpo3::{
    Cone, CscMatrix, Equilibration, LinearSolver, Problem, ProblemError, Settings, SolverStatus,
    solve,
};

#[test]
fn cones_report_their_vector_dim() {
    assert_eq!(Cone::Nonnegative(4).vector_dim(), Some(4));
    assert_eq!(Cone::SecondOrder(3).vector_dim(), Some(3));
    assert_eq!(Cone::PositiveSemidefinite(3).vector_dim(), Some(6));
}

#[test]
fn settings_use_short_names() {
    let settings = Settings::default();

    assert_eq!(settings.max_iters, 100);
    assert_eq!(settings.tol_gap_abs, 1e-8);
    assert_eq!(settings.tol_gap_rel, 1e-8);
    assert_eq!(settings.linear_solver, LinearSolver::Faer);
    assert_eq!(LinearSolver::default(), LinearSolver::Faer);
    assert_eq!(settings.data_equilibration, Equilibration::Ruiz);
    assert!(settings.kkt_bound_elimination);
    assert_eq!(settings.initial_diagonal_floor, 1e-12);
    assert!(!settings.cvxopt_init);
    assert_eq!(settings.initial_mu, 1e-3);
    assert_eq!(settings.step_fraction, 0.999);
}

fn sparse_identity(size: usize) -> CscMatrix {
    CscMatrix::new(
        size,
        size,
        (0..=size).collect(),
        (0..size).collect(),
        vec![1.0; size],
    )
    .unwrap()
}

#[test]
fn sparse_problem_preserves_dimensions() {
    let problem = Problem::new(
        Some(sparse_identity(2)),
        vec![1.0, 2.0],
        sparse_identity(2),
        vec![3.0, 4.0],
        vec![Cone::Nonnegative(2)],
    )
    .unwrap();
    assert_eq!(problem.variable_count(), 2);
    assert_eq!(problem.constraint_count(), 2);
}

#[test]
fn cone_rows_must_match_constraint_rows() {
    let error = Problem::new(
        None,
        vec![1.0, 2.0],
        sparse_identity(2),
        vec![3.0, 4.0],
        vec![Cone::PositiveSemidefinite(2)],
    )
    .unwrap_err();

    assert_eq!(
        error,
        ProblemError::ConeDimensionMismatch {
            expected: 2,
            actual: 3,
        }
    );
}

#[test]
fn quadratic_matrix_must_be_symmetric() {
    let p = common::matrix(2, 2, vec![1.0, 1.0, 0.0, 1.0]).unwrap();
    let error = Problem::new(
        Some(p),
        vec![1.0, 2.0],
        sparse_identity(2),
        vec![3.0, 4.0],
        vec![Cone::Nonnegative(2)],
    )
    .unwrap_err();

    assert_eq!(error, ProblemError::NonsymmetricQuadraticMatrix);
}

fn solve_settings(linear_solver: LinearSolver) -> Settings {
    Settings {
        max_iters: 400,
        linear_solver,
        ..Settings::default()
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-6 * (1.0 + actual.abs().max(expected.abs())),
        "expected {expected}, got {actual}"
    );
}

fn bounded_quadratic() -> Problem {
    Problem::new(
        Some(sparse_identity(1)),
        vec![-2.0],
        sparse_identity(1),
        vec![1.0],
        vec![Cone::Nonnegative(1)],
    )
    .unwrap()
}

fn assert_bounded_quadratic_solution(problem: &Problem, linear_solver: LinearSolver) {
    let solution = solve(problem, &solve_settings(linear_solver)).unwrap();

    assert_eq!(solution.status(), SolverStatus::Solved);
    assert_eq!(solution.info().data_equilibration, Equilibration::Ruiz);
    assert!(solution.info().kkt_bound_elimination);
    assert_close(solution.x()[0], 1.0);
    assert_close(solution.y()[0], 1.0);
    assert_close(solution.s()[0], 0.0);
    assert_close(solution.info().pobj, -1.5);
    assert_close(solution.info().dobj, -1.5);
    assert!(solution.info().pres <= 2e-8);
    assert!(solution.info().dres <= 3e-8);
    assert!(solution.info().comp <= 3e-8);
}

#[test]
fn solves_a_sparse_bounded_quadratic_with_every_sparse_solver() {
    let problem = bounded_quadratic();
    for linear_solver in [LinearSolver::Faer, LinearSolver::Feral] {
        assert_bounded_quadratic_solution(&problem, linear_solver);
    }
}

#[test]
fn bound_elimination_is_optional_with_every_sparse_solver() {
    // Coupled curvature and a general inequality keep this a numerical solve.
    // The singleton inequality becomes a variable bound during presolve.
    let problem = Problem::new(
        Some(common::matrix(2, 2, vec![2.0, 1.0, 1.0, 2.0]).unwrap()),
        vec![-5.0, -5.0],
        common::matrix(2, 2, vec![1.0, 0.0, 1.0, 1.0]).unwrap(),
        vec![0.5, 2.0],
        vec![Cone::Nonnegative(2)],
    )
    .unwrap();
    for linear_solver in [LinearSolver::Faer, LinearSolver::Feral] {
        for presolve in [false, true] {
            for kkt_bound_elimination in [false, true] {
                let solution = solve(
                    &problem,
                    &Settings {
                        presolve,
                        kkt_bound_elimination,
                        ..solve_settings(linear_solver)
                    },
                )
                .unwrap();
                assert_eq!(solution.status(), SolverStatus::Solved);
                assert!(solution.info().linear_algebra.primary_solves > 0);
                assert_eq!(solution.info().kkt_bound_elimination, kkt_bound_elimination);
                assert_close(solution.x()[0], 0.5);
                assert_close(solution.x()[1], 1.5);
                assert_close(solution.y()[0], 1.0);
                assert_close(solution.y()[1], 1.5);
                assert_close(solution.s()[0], 0.0);
                assert_close(solution.s()[1], 0.0);
            }
        }
    }
}

#[test]
fn solves_a_linear_program() {
    let problem = Problem::new(
        None,
        vec![1.0],
        common::matrix(1, 1, vec![-1.0]).unwrap(),
        vec![-1.0],
        vec![Cone::Nonnegative(1)],
    )
    .unwrap();
    let solution = solve(&problem, &solve_settings(LinearSolver::Faer)).unwrap();

    assert_eq!(solution.status(), SolverStatus::Solved);
    assert_close(solution.x()[0], 1.0);
    assert_close(solution.y()[0], 1.0);
    assert_close(solution.s()[0], 0.0);
    assert_close(solution.info().pobj, 1.0);
    assert_close(solution.info().dobj, 1.0);
}

#[test]
fn solves_a_quadratic_with_an_inactive_nonnegative_constraint() {
    let problem = Problem::new(
        Some(sparse_identity(1)),
        vec![-3.0],
        common::matrix(1, 1, vec![-1.0]).unwrap(),
        vec![0.0],
        vec![Cone::Nonnegative(1)],
    )
    .unwrap();
    let solution = solve(&problem, &solve_settings(LinearSolver::Faer)).unwrap();

    assert_eq!(solution.status(), SolverStatus::Solved);
    assert_close(solution.x()[0], 3.0);
    assert_close(solution.y()[0], 0.0);
    assert_close(solution.s()[0], 3.0);
    assert_close(solution.info().pobj, -4.5);
    assert_close(solution.info().dobj, -4.5);
    assert!(solution.info().pres <= 4e-8);
    assert!(solution.info().dres <= 4e-8);
    assert!(solution.info().comp <= 6e-8);
}

#[test]
fn solves_mixed_equality_and_inequality_constraints() {
    let problem = Problem::new(
        Some(sparse_identity(1)),
        vec![0.0],
        common::matrix(2, 1, vec![1.0, -1.0]).unwrap(),
        vec![3.0, 0.0],
        vec![Cone::Zero(1), Cone::Nonnegative(1)],
    )
    .unwrap();
    let solution = solve(&problem, &solve_settings(LinearSolver::Faer)).unwrap();

    assert_eq!(solution.status(), SolverStatus::Solved);
    assert_close(solution.x()[0], 3.0);
    assert_close(solution.y()[0], -3.0);
    assert_close(solution.y()[1], 0.0);
    assert_close(solution.s()[0], 0.0);
    assert_close(solution.s()[1], 3.0);
    assert_close(solution.info().pobj, 4.5);
    assert_close(solution.info().dobj, 4.5);
}

#[test]
fn solves_an_equality_constrained_quadratic() {
    let problem = Problem::new(
        Some(sparse_identity(1)),
        vec![0.0],
        sparse_identity(1),
        vec![2.0],
        vec![Cone::Zero(1)],
    )
    .unwrap();
    let solution = solve(&problem, &solve_settings(LinearSolver::Faer)).unwrap();

    assert_eq!(solution.status(), SolverStatus::Solved);
    assert_close(solution.x()[0], 2.0);
    assert_close(solution.y()[0], -2.0);
    assert_eq!(solution.s(), [0.0]);
    assert!(solution.info().pres <= 3e-8);
    assert!(solution.info().dres <= 3e-8);
    assert_eq!(solution.info().comp, 0.0);
}

/// The problems above, with their expected primal solutions.
fn reference_problems() -> Vec<(Problem, Vec<f64>)> {
    vec![
        (bounded_quadratic(), vec![1.0]),
        (
            Problem::new(
                None,
                vec![1.0],
                common::matrix(1, 1, vec![-1.0]).unwrap(),
                vec![-1.0],
                vec![Cone::Nonnegative(1)],
            )
            .unwrap(),
            vec![1.0],
        ),
        (
            Problem::new(
                Some(common::matrix(2, 2, vec![2.0, 1.0, 1.0, 2.0]).unwrap()),
                vec![-5.0, -5.0],
                common::matrix(2, 2, vec![1.0, 0.0, 1.0, 1.0]).unwrap(),
                vec![0.5, 2.0],
                vec![Cone::Nonnegative(2)],
            )
            .unwrap(),
            vec![0.5, 1.5],
        ),
        (
            Problem::new(
                Some(sparse_identity(1)),
                vec![0.0],
                common::matrix(2, 1, vec![1.0, -1.0]).unwrap(),
                vec![3.0, 0.0],
                vec![Cone::Zero(1), Cone::Nonnegative(1)],
            )
            .unwrap(),
            vec![3.0],
        ),
        (
            Problem::new(
                Some(sparse_identity(1)),
                vec![0.0],
                sparse_identity(1),
                vec![2.0],
                vec![Cone::Zero(1)],
            )
            .unwrap(),
            vec![2.0],
        ),
    ]
}

#[test]
fn cvxopt_initialization_solves_the_reference_problems_with_every_solver() {
    for (problem, expected) in reference_problems() {
        for linear_solver in [LinearSolver::Faer, LinearSolver::Feral] {
            for presolve in [false, true] {
                let solution = solve(
                    &problem,
                    &Settings {
                        presolve,
                        cvxopt_init: true,
                        ..solve_settings(linear_solver)
                    },
                )
                .unwrap();
                assert_eq!(solution.status(), SolverStatus::Solved);
                let info = solution.info();
                assert_ne!(info.cvxopt_init_used, Some(false));
                if info.cvxopt_init_used.is_some() {
                    assert!(info.linear_algebra.factorizations >= 1);
                    assert!(info.linear_algebra.primary_solves >= 1);
                }
                for (actual, expected) in solution.x().iter().zip(&expected) {
                    assert_close(*actual, *expected);
                }
            }
        }
    }
}

/// Refinement stops at a 1e-8 backward error, so the starting point lands
/// just outside the stopping test and one step finishes the solve.
#[test]
fn cvxopt_initialization_solves_an_equality_constrained_quadratic_in_one_step() {
    let (problem, expected) = reference_problems().pop().unwrap();
    let solution = solve(
        &problem,
        &Settings {
            presolve: false,
            cvxopt_init: true,
            ..Settings::default()
        },
    )
    .unwrap();
    assert_eq!(solution.status(), SolverStatus::Solved);
    assert_eq!(solution.info().cvxopt_init_used, Some(true));
    assert!(solution.info().iters <= 1, "{}", solution.info().iters);
    assert_close(solution.x()[0], expected[0]);
    assert_close(solution.y()[0], -2.0);
}

#[test]
fn solves_equality_chain_without_homogeneous_scale_stagnation() {
    // min ||x||^2 / 2, subject to x_i - x_(i+1) = 1. The solution is the
    // centered ramp. Without sphere renormalization the homogeneous scale
    // stalls, leaving a primal residual around 0.2 even after 200 iterations.
    let n = 100;
    let mut rows = Vec::new();
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for row in 0..n - 1 {
        rows.extend([row, row]);
        columns.extend([row, row + 1]);
        values.extend([1.0, -1.0]);
    }
    let problem = Problem::new(
        Some(sparse_identity(n)),
        vec![0.0; n],
        CscMatrix::from_triplets(n - 1, n, rows, columns, values).unwrap(),
        vec![1.0; n - 1],
        vec![Cone::Zero(n - 1)],
    )
    .unwrap();
    let solution = solve(&problem, &Settings::default()).unwrap();
    assert_eq!(solution.status(), SolverStatus::Solved);
    assert!(solution.info().iters < 20);
    for (i, &x) in solution.x().iter().enumerate() {
        assert!((x - (49.5 - i as f64)).abs() < 1e-6);
    }
    assert!(solution.info().pres <= 2e-8);
    assert!(solution.info().dres <= 1e-8);

    // The CVXOPT start solves the equality-constrained KKT system outright.
    let solution = solve(
        &problem,
        &Settings {
            cvxopt_init: true,
            ..Settings::default()
        },
    )
    .unwrap();
    assert_eq!(solution.status(), SolverStatus::Solved);
    assert!(solution.info().iters <= 1, "{}", solution.info().iters);
    for (i, &x) in solution.x().iter().enumerate() {
        assert!((x - (49.5 - i as f64)).abs() < 1e-6);
    }
}

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

#[test]
fn owned_conversion_moves_cpo3_matrix_and_vector_buffers() {
    let p = CscMatrix::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![2., 1., 1., 3.]).unwrap();
    let problem = Problem::new(
        Some(p),
        vec![1., 2.],
        sparse_identity(2),
        vec![3., 4.],
        vec![Cone::Nonnegative(2)],
    )
    .unwrap();
    let a = problem.a().values().as_ptr();
    let c = problem.c().as_ptr();
    let p = problem.p().unwrap().values().as_ptr();
    let native: ::presolve::Problem = problem.into();
    assert_eq!(native.a.values().as_ptr(), a);
    assert_eq!(native.c.as_ptr(), c);
    assert_eq!(native.p.as_ref().unwrap().values().as_ptr(), p);
    assert_eq!(native.p.unwrap().values(), [2., 1., 3.]);
}

#[test]
fn equality_zero_target_does_not_falsely_accept_inconsistent_or_unbounded_problems() {
    // The baseline does not generate equality-only certificates with presolve
    // off (its certificate gate is zero). Preserve rejection as non-solutions.
    for solver in [LinearSolver::Faer, LinearSolver::Feral] {
        for (c, b) in [(0.0, 1.0), (-1.0, 0.0)] {
            let problem = Problem::new(
                None,
                vec![c],
                common::matrix(1, 1, vec![0.0]).unwrap(),
                vec![b],
                vec![Cone::Zero(1)],
            )
            .unwrap();
            let solution = solve(
                &problem,
                &Settings {
                    presolve: false,
                    linear_solver: solver,
                    ..Settings::default()
                },
            )
            .unwrap();
            assert_eq!(solution.status(), SolverStatus::MaxIterations);
        }
    }
}
