use super::kkt::{
    apply_full_system, assemble_preconditioner, full_residual_and_denominator, matrix_value,
};
use super::*;
use crate::{Cone, LinearSolver};

const TOLERANCE: f64 = 2e-11;

fn reference_state(problem: &Problem) -> State {
    crate::embedding::Embedding::new(problem)
        .unwrap()
        .canonical_initial_state(1.0)
        .unwrap()
}

fn problem() -> Problem {
    Problem::new(
        Some(crate::matrix::test_matrix(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap()),
        vec![0.5, -1.0],
        crate::matrix::test_matrix(2, 2, vec![1.0, -2.0, 3.0, 0.25]).unwrap(),
        vec![1.5, -0.75],
        vec![Cone::Zero(1), Cone::Nonnegative(1)],
    )
    .unwrap()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= TOLERANCE * (1.0 + actual.abs().max(expected.abs())),
        "expected {expected}, got {actual}"
    );
}

#[test]
fn increment_resolvent_matches_point_equations_for_lp_and_qp() {
    for quadratic in [false, true] {
        let original = problem();
        let problem = Problem::from_upper_triangle(
            if quadratic {
                original.p().cloned()
            } else {
                None
            },
            original.c().to_vec(),
            original.a().clone(),
            original.b().to_vec(),
            original.cones().to_vec(),
        )
        .unwrap();
        let state = State::new(vec![0.3, -0.2], vec![0.0, 1.4], vec![0.1, 1.2], 0.9).unwrap();
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            &problem,
            &state,
            &[0.0, 1.4 / 1.2],
            0.3,
            &KktSettings::new(LinearSolver::Faer),
            &mut cache,
        )
        .unwrap();
        for (rhs, r_tau, mu) in [
            ([1.0, -2.0, 0.4, -0.7], 0.7, 0.3),
            ([-0.2, 0.6, -0.3, 0.9], -0.2, 0.0),
        ] {
            let mut expected = resolvent
                .pattern
                .solve(
                    &problem,
                    resolvent.factorization,
                    resolvent.shift,
                    &resolvent.d_diagonal,
                    &resolvent.d_diagonal,
                    &rhs,
                    None,
                )
                .unwrap();
            let expected_tau = resolvent
                .solve_tau(&expected, q_dot(&problem, &expected), r_tau, mu)
                .unwrap();
            crate::matrix::update(&mut expected, resolvent.direction, |value, direction| {
                expected_tau.mul_add(direction, value)
            });
            let actual = resolvent.solve_rhs(&rhs, r_tau, mu).unwrap();
            assert!(!actual.used_tau_fallback);
            assert_close(actual.tau, expected_tau);
            for (&actual, &expected) in actual.xy.iter().zip(&expected) {
                assert_close(actual, expected);
            }
        }
    }
}

#[test]
fn increment_resolvent_preserves_a_small_point_between_large_columns() {
    // K = rho, d = -1/rho, and the point-coordinate particular solution
    // is 1/rho. Recovering x near one by adding these columns loses its
    // significant digits. The displacement equation remains well scaled.
    let rho = 1e-16;
    let r_tau = rho - 1.0;
    let problem = Problem::new(
        None,
        vec![1.0],
        crate::matrix::test_matrix(0, 1, vec![]).unwrap(),
        vec![],
        vec![],
    )
    .unwrap();
    let state = State::new(vec![1.0], vec![], vec![], 1.0).unwrap();
    let mut cache = KktCache::new();
    let mut resolvent = Resolvent::new(
        &problem,
        &state,
        &[],
        rho,
        &KktSettings::new(LinearSolver::Faer),
        &mut cache,
    )
    .unwrap();
    let actual = resolvent.solve_rhs(&[1.0], r_tau, 0.0).unwrap();
    // Solve the two scalar equations without subtracting large columns.
    let expected_x = (rho - r_tau) / (1.0 + rho * rho);
    let expected_tau = (1.0 + rho * r_tau) / (1.0 + rho * rho);
    assert!(!actual.used_tau_fallback);
    assert_close(actual.x()[0], expected_x);
    assert_close(actual.tau, expected_tau);
}

#[test]
fn fused_full_system_residual_preserves_rounding_and_the_normalizer() {
    {
        let problem = problem();
        for u in [[1.0, -2.0, 3.0, -4.0], [1e-12, -1e12, 1e8, 0.0]] {
            let rhs = [0.0, 1.0, -3.0, 7.0];
            let d = [1e-10, 3.5];
            let shift = 1e-12;
            let mut expected_residual = rhs.to_vec();
            let mut expected_denominator = rhs.map(f64::abs).to_vec();
            let initial = u.split_at(problem.variable_count());
            apply_full_system(&problem, shift, &d, initial, &mut expected_residual, false);
            apply_full_system(
                &problem,
                shift,
                &d,
                initial,
                &mut expected_denominator,
                true,
            );
            let mut residual = vec![f64::NAN; 4];
            let mut denominator = vec![f64::NAN; 4];
            full_residual_and_denominator(
                &problem,
                shift,
                &d,
                &u,
                &rhs,
                &mut residual,
                &mut denominator,
            );
            for (actual, expected) in residual
                .iter()
                .chain(&denominator)
                .zip(expected_residual.iter().chain(&expected_denominator))
            {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
}

#[test]
fn assembles_sparse_quasi_definite_systems() {
    let expected = [
        [5.0, 0.5, 1.0, 3.0],
        [0.5, 4.0, -2.0, 0.25],
        [1.0, -2.0, -3.0, 0.0],
        [3.0, 0.25, 0.0, -6.0],
    ];
    {
        let problem = problem();
        let matrix =
            assemble_preconditioner(&problem, 3.0, &[3.0, 6.0], 0.0, true, &mut KktCache::new())
                .unwrap();
        for (row, expected_row) in expected.iter().enumerate() {
            for (column, expected) in expected_row.iter().enumerate() {
                assert_close(matrix_value(&matrix, row, column), *expected);
            }
        }
    }
}

fn linear_program() -> Problem {
    Problem::new(
        None,
        vec![1.0, 2.0],
        crate::matrix::test_matrix(2, 2, vec![1.0, 1.0, -1.0, 2.0]).unwrap(),
        vec![1.0, 0.5],
        vec![Cone::Nonnegative(2)],
    )
    .unwrap()
}

fn a_mul(problem: &Problem, x: &[f64]) -> Vec<f64> {
    crate::matrix::matrix_mul(problem.a(), x)
}

fn at_mul(problem: &Problem, y: &[f64]) -> Vec<f64> {
    crate::matrix::matrix_transpose_mul(problem.a(), y)
}

/// Residual of the generic shifted system at a case (i) solution.
fn generic_system_residual(
    problem: &Problem,
    h: &[f64],
    shift: f64,
    r: (&[f64], &[f64], f64),
    mu: f64,
    solution: &ResolventSolution,
) -> f64 {
    let (r_x, r_y, r_tau) = r;
    let px = problem.p_mul(solution.x());
    let aty = at_mul(problem, solution.y());
    let ax = a_mul(problem, solution.x());
    let mut worst: f64 = 0.0;
    for index in 0..problem.variable_count() {
        let row = px[index]
            + shift * solution.x()[index]
            + aty[index]
            + problem.c()[index] * solution.tau
            - r_x[index];
        worst = worst.max(row.abs());
    }
    for index in 0..problem.constraint_count() {
        let row = -ax[index]
            + (h[index] + shift) * solution.y()[index]
            + problem.b()[index] * solution.tau
            - r_y[index];
        worst = worst.max(row.abs());
    }
    let row = -dot(problem.c(), solution.x()) - dot(problem.b(), solution.y())
        + shift * solution.tau
        - (dot(solution.x(), &px) + mu) / solution.tau
        - r_tau;
    worst.max(row.abs())
}

#[test]
fn resolvent_solves_the_shifted_system_for_every_solver() {
    let h = [0.0, 2.5];
    let shift = 0.7;
    let r_x = [0.3, -1.1];
    let r_y = [0.9, 0.4];
    let r_tau = -0.6;
    let cases = [LinearSolver::Faer, LinearSolver::Feral];
    for solver in cases {
        let problem = problem();
        let state = reference_state(&problem);
        let mut cache = KktCache::new();
        let mut resolvent = Resolvent::new(
            &problem,
            &state,
            &h,
            shift,
            &KktSettings::new(solver),
            &mut cache,
        )
        .unwrap();
        assert!(resolvent.alpha > 0.0);
        let (x2, y2) = resolvent.direction.split_at(problem.variable_count());
        let px2 = problem.p_mul(x2);
        let defining = shift - dot(problem.c(), x2) - dot(problem.b(), y2) - dot(x2, &px2);
        assert_close(resolvent.alpha, defining);
        for mu in [0.0, 0.25] {
            let solution = resolvent.solve(&r_x, &r_y, r_tau, mu).unwrap();
            assert!(solution.tau > 0.0);
            let residual =
                generic_system_residual(&problem, &h, shift, (&r_x, &r_y, r_tau), mu, &solution);
            assert!(residual <= 1e-12, "{solver:?}: residual {residual}");
        }
    }
}

/// An equality, a general inequality, and two bound rows on `x`.
fn bounded_problem() -> Problem {
    Problem::new(
        Some(crate::matrix::test_matrix(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap()),
        vec![0.5, -1.0],
        crate::matrix::test_matrix(4, 2, vec![1.0, -2.0, 3.0, 0.25, -4.0, 0.0, 0.0, 1.5]).unwrap(),
        vec![1.5, -0.75, 2.0, 0.5],
        vec![Cone::Zero(1), Cone::Nonnegative(3)],
    )
    .unwrap()
}

// Independently specified full matrices: one retained equality and one
// singleton inequality, with both primal and dual clamps active.
#[test]
fn clamped_preconditioner_is_identical_with_or_without_bound_elimination() {
    let problem = Problem::new(
        Some(crate::matrix::test_matrix(2, 2, vec![0.1, 0.0, 0.0, 2.0]).unwrap()),
        vec![0.0; 2],
        crate::matrix::test_matrix(2, 2, vec![1.0, -1.0, 2.0, 0.0]).unwrap(),
        vec![0.0; 2],
        vec![Cone::Zero(1), Cone::Nonnegative(1)],
    )
    .unwrap();
    let k = [
        [0.2, 0.0, 1.0, 2.0],
        [0.0, 2.1, -1.0, 0.0],
        [1.0, -1.0, -0.1, 0.0],
        [2.0, 0.0, 0.0, -0.3],
    ];
    let m = [
        [0.5, 0.0, 1.0, 2.0],
        [0.0, 2.1, -1.0, 0.0],
        [1.0, -1.0, -0.5, 0.0],
        [2.0, 0.0, 0.0, -0.5],
    ];
    let expected = [0.4, -0.7, 1.1, -0.3];
    let multiply = |matrix: &[[f64; 4]; 4]| {
        matrix
            .iter()
            .map(|row| dot(row, &expected))
            .collect::<Vec<_>>()
    };
    let rhs = multiply(&m);
    for solver in [LinearSolver::Faer, LinearSolver::Feral] {
        for eliminate in [false, true] {
            let mut cache = KktCache::new();
            let matrix =
                assemble_preconditioner(&problem, 0.1, &[0.5, 0.5], 0.5, eliminate, &mut cache)
                    .unwrap();
            assert_close(
                matrix_value(&matrix, 0, 0),
                if eliminate { 8.5 } else { 0.5 },
            );
            let mut settings = KktSettings::new(solver);
            // Request accuracy tighter than the solution assertions below;
            // the production backward-error tolerance is only 1e-8.
            settings.refinement_tolerance = 1e-12;
            let mut factor =
                Factorization::new_counted(matrix, &settings, &mut cache.factorizations).unwrap();
            let pattern = cache.pattern.as_mut().unwrap();
            let solution = pattern
                .solve_preconditioner(&mut factor, &[0.5, 0.5], &rhs)
                .unwrap();
            for (actual, expected) in solution.iter().zip(expected) {
                assert_close(*actual, expected);
            }
            let mut residual = [0.0; 4];
            let mut denominator = [0.0; 4];
            full_residual_and_denominator(
                &problem,
                0.1,
                &[0.1, 0.3],
                &solution,
                &rhs,
                &mut residual,
                &mut denominator,
            );
            for i in 0..4 {
                assert_close(residual[i], rhs[i] - dot(&k[i], &solution));
                let normalizer = rhs[i].abs()
                    + k[i]
                        .iter()
                        .zip(&solution)
                        .map(|(a, x)| (a * x).abs())
                        .sum::<f64>();
                assert_close(denominator[i], normalizer);
            }
            let true_rhs = multiply(&k);
            let refined = pattern
                .solve(
                    &problem,
                    &mut factor,
                    0.1,
                    &[0.1, 0.3],
                    &[0.5, 0.5],
                    &true_rhs,
                    None,
                )
                .unwrap();
            for (actual, expected) in refined.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-8,
                    "{solver:?}/{eliminate}: {actual} vs {expected}"
                );
            }
            // A second diagonal assembly must clamp afresh, not add
            // another Schur contribution to the previous diagonal.
            factor
                .update_diagonal_and_refactor(&mut cache.factorizations, |values| {
                    pattern.write_diagonal(values, |index| index, 2, 0.1, &[0.5, 0.5], 0.5)
                })
                .unwrap();
            let again = pattern
                .solve_preconditioner(&mut factor, &[0.5, 0.5], &rhs)
                .unwrap();
            for (actual, expected) in again.iter().zip(expected) {
                assert_close(*actual, expected);
            }
            assert_eq!(cache.factorizations, 2);
        }
    }
}

#[test]
fn best_effort_refines_true_system_even_without_eliminated_rows() {
    let problem = Problem::new(
        None,
        vec![0.0],
        crate::matrix::test_matrix(0, 1, vec![]).unwrap(),
        vec![],
        vec![],
    )
    .unwrap();
    let mut cache = KktCache::new();
    let matrix = assemble_preconditioner(&problem, 1e-14, &[], 1e-8, false, &mut cache).unwrap();
    let mut factor = Factorization::new(matrix, &KktSettings::new(LinearSolver::Faer)).unwrap();
    let solution = cache
        .pattern
        .as_mut()
        .unwrap()
        .solve(&problem, &mut factor, 1e-14, &[], &[], &[1.0], None)
        .unwrap();
    // The preconditioner alone solves to 1e8. True-system refinement must
    // attempt corrections, but cannot resolve this weak direction in 30.
    assert!(solution[0] > 1e8 && solution[0] < 1e14);
    assert!(factor.stats().refinement_solves > 0);
    assert_eq!(factor.stats().refinement_settled, 1);
    assert!((1.0 - 1e-14 * solution[0]).abs() > 0.99);
}

#[test]
fn refinement_warm_start_preserves_weak_components_and_corrects_the_rest() {
    // The first variable is uncoupled: its true pivot is 1e-14, while
    // the factor pivot is 1e-8. The second variable has a singleton row.
    let problem = Problem::new(
        None,
        vec![0.0; 2],
        crate::matrix::test_matrix(1, 2, vec![0.0, 1.0]).unwrap(),
        vec![0.0],
        vec![Cone::Nonnegative(1)],
    )
    .unwrap();
    let shift = 1e-14;
    let d = [2.0];
    let rhs = [1.0, 2.0 * shift - 3.0, 8.0];
    let initial_x = [1e14, -1.0];
    let initial_y = [0.5];
    for solver in [LinearSolver::Faer, LinearSolver::Feral] {
        for eliminate in [false, true] {
            let mut cache = KktCache::new();
            let mut settings = KktSettings::new(solver);
            settings.min_static_regularization = 1e-8;
            // Resolve the corrected components to the asserted accuracy,
            // rather than stopping at the production tolerance of 1e-8.
            settings.refinement_tolerance = 1e-12;
            let matrix = assemble_preconditioner(
                &problem,
                shift,
                &d,
                settings.min_static_regularization,
                eliminate,
                &mut cache,
            )
            .unwrap();
            let mut factor = Factorization::new(matrix, &settings).unwrap();
            let pattern = cache.pattern.as_mut().unwrap();
            let fresh = pattern
                .solve(&problem, &mut factor, shift, &d, &d, &rhs, None)
                .unwrap();
            assert!((rhs[0] - shift * fresh[0]).abs() > 0.99);
            let solved = pattern
                .solve(
                    &problem,
                    &mut factor,
                    shift,
                    &d,
                    &d,
                    &rhs,
                    Some((&initial_x, &initial_y)),
                )
                .unwrap();
            for (actual, expected) in solved.iter().zip([1e14, 2.0, -3.0]) {
                assert_close(*actual, expected);
            }
            let mut residual = rhs;
            apply_full_system(
                &problem,
                shift,
                &d,
                solved.split_at(2),
                &mut residual,
                false,
            );
            assert!(residual.iter().all(|r| r.abs() < 1e-10));
        }
    }
}

#[test]
fn failed_step_restores_the_previous_constant_column_without_copying() {
    let problem = Problem::new(
        None,
        vec![-1.0],
        crate::matrix::test_matrix(0, 1, vec![]).unwrap(),
        vec![],
        vec![],
    )
    .unwrap();
    let state = reference_state(&problem);
    let settings = KktSettings::new(LinearSolver::Feral);
    let mut cache = KktCache::new();
    Resolvent::new(&problem, &state, &[], 1.0, &settings, &mut cache).unwrap();
    cache.finish_step(true);
    let original = cache.direction.as_ptr();
    assert_close(cache.direction[0], 1.0);
    Resolvent::new(&problem, &state, &[], 2.0, &settings, &mut cache).unwrap();
    assert_close(cache.direction[0], 0.5);
    cache.finish_step(false);
    assert_eq!(cache.direction.as_ptr(), original);
    assert_close(cache.direction[0], 1.0);
    // Failure before solving the column must not discard the accepted seed.
    assert!(Resolvent::new(&problem, &state, &[], 0.0, &settings, &mut cache).is_err());
    cache.finish_step(false);
    assert_eq!(cache.direction.as_ptr(), original);
    Resolvent::new(&problem, &state, &[], 2.0, &settings, &mut cache).unwrap();
    cache.finish_step(true);
    assert_close(cache.direction[0], 0.5);
    assert!(cache.direction_before_step.is_none());
}

#[test]
fn constant_column_warm_start_survives_refactorization_only_within_its_cache() {
    let problem = Problem::new(
        None,
        vec![-1e-14],
        crate::matrix::test_matrix(0, 1, vec![]).unwrap(),
        vec![],
        vec![],
    )
    .unwrap();
    let state = reference_state(&problem);
    let mut settings = KktSettings::new(LinearSolver::Feral);
    let mut cache = KktCache::new();
    // First compute an accurate column with an unclamped factorization.
    settings.min_static_regularization = 0.0;
    let first = Resolvent::new(&problem, &state, &[], 1e-14, &settings, &mut cache)
        .unwrap()
        .direction[0];
    assert_close(first, 1.0);
    // Clamping the next factor cannot erase the retained solution.
    settings.min_static_regularization = 1e-8;
    let next = Resolvent::new(&problem, &state, &[], 1e-14, &settings, &mut cache)
        .unwrap()
        .direction[0];
    assert_close(next, 1.0);
    assert_eq!(cache.stats().factorizations, 2);
    // A separate problem solve has no retained column.
    let fresh = Resolvent::new(
        &problem,
        &state,
        &[],
        1e-14,
        &settings,
        &mut KktCache::new(),
    )
    .unwrap()
    .direction[0];
    assert!(fresh < 1e-4);
}

#[test]
fn bound_rows_fold_into_the_reduced_diagonal() {
    let d = [3.0, 6.0, 0.5, 4.0];
    // Rows 2 and 3 are singletons: 16 / 0.5 on x_0 and 2.25 / 4 on x_1.
    let expected = [
        [5.0 + 32.0, 0.5, 1.0, 3.0],
        [0.5, 4.0 + 0.5625, -2.0, 0.25],
        [1.0, -2.0, -3.0, 0.0],
        [3.0, 0.25, 0.0, -6.0],
    ];
    {
        let problem = bounded_problem();
        let mut cache = KktCache::new();
        let matrix = assemble_preconditioner(&problem, 3.0, &d, 0.0, true, &mut cache).unwrap();
        assert_eq!(matrix.rows(), 4);
        for (row, expected_row) in expected.iter().enumerate() {
            for (column, expected) in expected_row.iter().enumerate() {
                assert_close(matrix_value(&matrix, row, column), *expected);
            }
        }
        let bound_rows = &cache.pattern.as_ref().unwrap().bound_rows;
        assert_eq!(bound_rows.eliminated, vec![(2, 0, -4.0), (3, 1, 1.5)]);
        assert_eq!(bound_rows.kept, vec![0, 1]);
        let full =
            assemble_preconditioner(&problem, 3.0, &d, 0.0, false, &mut KktCache::new()).unwrap();
        assert_eq!(full.rows(), 6);
    }
}

#[test]
fn bound_row_elimination_leaves_the_resolvent_solution_unchanged() {
    let h = [0.0, 2.5, 0.8, 1e-6];
    let shift = 0.7;
    let r_x = [0.3, -1.1];
    let r_y = [0.9, 0.4, -0.2, 0.7];
    let r_tau = -0.6;
    let cases = [LinearSolver::Faer, LinearSolver::Feral];
    for solver in cases {
        let problem = bounded_problem();
        let state = reference_state(&problem);
        let mut solutions = Vec::new();
        for eliminate in [false, true] {
            let mut cache = KktCache::new();
            let kkt = KktSettings {
                eliminate_bound_rows: eliminate,
                ..KktSettings::new(solver)
            };
            let mut resolvent =
                Resolvent::new(&problem, &state, &h, shift, &kkt, &mut cache).unwrap();
            assert_eq!(resolvent.factorization.dim(), if eliminate { 4 } else { 6 });
            let solution = resolvent.solve(&r_x, &r_y, r_tau, 0.25).unwrap();
            let residual =
                generic_system_residual(&problem, &h, shift, (&r_x, &r_y, r_tau), 0.25, &solution);
            assert!(
                residual <= 1e-9,
                "{solver:?}/{eliminate}: residual {residual}"
            );
            solutions.push(solution);
        }
        for (full, reduced) in solutions[0].x().iter().zip(solutions[1].x()) {
            assert_close(*full, *reduced);
        }
        for (full, reduced) in solutions[0].y().iter().zip(solutions[1].y()) {
            assert_close(*full, *reduced);
        }
        assert_close(solutions[0].tau, solutions[1].tau);
    }
}

#[test]
fn resolvent_accepts_zero_tau_without_fallback() {
    let problem = linear_program();
    let state = reference_state(&problem);
    let h = [1.5, 0.5];
    let mut cache = KktCache::new();
    let mut resolvent = Resolvent::new(
        &problem,
        &state,
        &h,
        0.4,
        &KktSettings::new(LinearSolver::Faer),
        &mut cache,
    )
    .unwrap();

    let boundary = resolvent.solve(&[0.0; 2], &[0.0; 2], -1.0, 0.0).unwrap();
    assert_eq!(boundary.tau, 0.0);
    for &value in boundary.x().iter().chain(boundary.y()) {
        assert_close(value, 0.0);
    }
    assert!(resolvent.fallback.get().is_none());

    let interior = resolvent.solve(&[0.0; 2], &[0.0; 2], -1.0, 0.1).unwrap();
    assert!(interior.tau > 0.0);
    let residual = generic_system_residual(
        &problem,
        &h,
        0.4,
        (&[0.0; 2], &[0.0; 2], -1.0),
        0.1,
        &interior,
    );
    assert!(residual <= 1e-12, "residual {residual}");

    let positive_root = resolvent.solve(&[0.0; 2], &[0.0; 2], 1.0, 0.0).unwrap();
    assert!(positive_root.tau > 0.0);
}

#[test]
fn resolvent_rejects_bad_inputs() {
    let problem = problem();
    let state = reference_state(&problem);
    assert!(matches!(
        Resolvent::new(
            &problem,
            &state,
            &[0.0, 1.0],
            0.0,
            &KktSettings::new(LinearSolver::Faer),
            &mut KktCache::new()
        ),
        Err(JosephyError::Embedding(EmbeddingError::InvalidMu))
    ));
    let mut cache = KktCache::new();
    let mut resolvent = Resolvent::new(
        &problem,
        &state,
        &[0.0, 1.0],
        1.0,
        &KktSettings::new(LinearSolver::Faer),
        &mut cache,
    )
    .unwrap();
    assert_eq!(
        resolvent.solve(&[0.0], &[0.0; 2], 0.0, 0.0),
        Err(JosephyError::IncorrectRhsLength {
            argument: "r_x",
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        resolvent.solve(&[0.0; 2], &[0.0; 2], 0.0, -1.0),
        Err(JosephyError::InvalidTauPolynomial)
    );
}

#[test]
fn assembly_rejects_a_nonpositive_diagonal() {
    let problem = problem();
    assert_eq!(
        assemble_preconditioner(&problem, 1.0, &[1.0, 0.0], 0.0, true, &mut KktCache::new()),
        Err(JosephyError::InvalidTauPolynomial)
    );
}

#[test]
fn positive_tau_coefficient_handles_extreme_weights_before_squaring() {
    let large_x = tau_leading_coefficient(1e-300, &[1e200], &[], &[]);
    assert!((large_x / 1e100 - 1.0).abs() < 1e-14);
    let small_y = tau_leading_coefficient(1e-300, &[], &[1e-200], &[1e300]);
    assert!((small_y / 1e-100 - 1.0).abs() < 1e-14);
    let overflowing_norm = tau_leading_coefficient(1e-308, &[1e154, 1e154], &[], &[]);
    assert!((overflowing_norm - 2.0).abs() < 1e-14);
    assert_eq!(tau_leading_coefficient(0.25, &[0.0], &[0.0], &[1.0]), 0.25);
}

#[test]
fn tau_root_accepts_zero_and_roundoff_but_rejects_invalid_roots() {
    assert_eq!(tau_root(2.0, 1.0, 0.0), Some(0.0));
    assert_eq!(tau_root(2.0, -2.0, 0.0), Some(1.0));
    assert_eq!(tau_root(0.0, 2.0, -4.0), Some(2.0));
    assert_eq!(tau_root(0.0, 1.0, 5e-11), Some(0.0));
    assert_eq!(tau_root(0.0, 1.0, 2e-10), None);
    assert_eq!(tau_root(0.0, 0.0, -1.0), None);
    assert_eq!(tau_root(1.0, 0.0, 1e-11), Some(0.0));
    assert_eq!(tau_root(1.0, 0.0, 1.0), None);
    assert!((tau_root(1.0, -1e308, -1.0).unwrap() / 1e308 - 1.0).abs() < 1e-14);
    let small = tau_root(1.0, 1e16, -1.0).unwrap();
    assert!((small - 1e-16).abs() <= 2.0 * f64::EPSILON * 1e-16);
}

#[test]
fn scaled_roots_handle_extreme_coefficients_and_general_signs() {
    for (a, b, c, expected) in [
        (1e308, 0.0, -1e308, 1.0),
        (1e308, 1e-300, -1e-308, 1e-308),
        (1e-14, 0.0, -1e308, 1e161),
        (1.0, 1e308, -1.0, 1e-308),
        (-2.0, 1.0, 1.0, -0.5),
        (-1.0, 3.0, -2.0, 1.0),
    ] {
        let root = quadratic_root(a, b, c, 0.0).unwrap();
        assert!(
            (root / expected - 1.0).abs() < 1e-13,
            "{a}, {b}, {c}: {root}"
        );
    }
    assert_eq!(tau_root(-2.0, 1.0, 1.0), None);
    assert_eq!(quadratic_root(1e308, 0.0, 1e-308, 1e-9), None);
    assert_eq!(quadratic_root(1.0, 0.0, 1e-11, 0.0), None);
    assert_eq!(tau_root(f64::NAN, 1.0, -1.0), None);
    assert_eq!(tau_root(1.0, f64::INFINITY, -1.0), None);
    assert_eq!(tau_root(1.0, 1.0, f64::NEG_INFINITY), None);
}

#[test]
fn root_thresholds_are_in_original_units() {
    // Normalizing first would incorrectly discard this nonzero a.
    assert_close(tau_root(1.0, 0.0, -1e20).unwrap() / 1e10, 1.0);
    // This a is deliberately treated as zero despite a large root.
    assert_eq!(tau_root(1e-16, 1.0, -1e16), Some(1e16));
    assert_eq!(tau_root(1e-16, 1e-16, -1.0), None);
    // Same scaled polynomial, different absolute discriminant acceptance.
    assert_eq!(tau_root(1.0, 0.0, 1e-11), Some(0.0));
    assert_eq!(tau_root(1e6, 0.0, 1e-5), None);
}

#[test]
fn failed_root_uses_current_state_and_reuses_fallback_products() {
    // P = 1, c = 1, rho = 1 gives direction = -1/2. Linearizing
    // at x = 1 gives 1.5*tau^2 + (1-r_tau-x1)*tau + 1-mu-2*x1 = 0.
    // If it has no real root, the first-order step uses current tau = 2.
    let problem = Problem::new(
        Some(crate::matrix::test_matrix(1, 1, vec![1.0]).unwrap()),
        vec![1.0],
        crate::matrix::test_matrix(0, 1, vec![]).unwrap(),
        vec![],
        vec![],
    )
    .unwrap();
    let state = State::new(vec![1.0], vec![], vec![], 2.0).unwrap();
    let mut cache = KktCache::new();
    let mut resolvent = Resolvent::new(
        &problem,
        &state,
        &[],
        1.0,
        &KktSettings::new(LinearSolver::Faer),
        &mut cache,
    )
    .unwrap();
    resolvent.solve(&[1.0], &[], 0.0, 1.0).unwrap();
    assert!(resolvent.fallback.get().is_none());

    // Force invalid quadratic coefficients to exercise the full solve's
    // recovery path, not just the standalone fallback helper.
    resolvent.alpha = f64::NAN;
    let solution = resolvent.solve(&[1.0], &[], 0.0, 1.0).unwrap();
    assert!(solution.used_tau_fallback);
    assert_close(solution.tau, 2.0 / 3.0);
    assert_close(solution.x()[0], 1.0 / 6.0);
    let px_pointer = resolvent
        .fallback
        .get()
        .unwrap()
        .px
        .as_ref()
        .unwrap()
        .as_ptr();
    let changed_rhs = resolvent.solve(&[2.0], &[], 1.0, 0.0).unwrap();
    assert_close(changed_rhs.tau, (1.0 + 7.0_f64.sqrt()) / 3.0);
    assert_eq!(
        resolvent
            .fallback
            .get()
            .unwrap()
            .px
            .as_ref()
            .unwrap()
            .as_ptr(),
        px_pointer
    );

    let lower = resolvent.solve(&[-10.0], &[], 0.0, 0.0).unwrap();
    assert_close(lower.tau, 0.2);
    // Negative discriminant: use the first-order correction, before clamping.
    let first_order = resolvent.solve(&[0.2], &[], 0.0, 0.0).unwrap();
    assert_close(first_order.tau, 0.28);
    let zero_denominator = resolvent.solve(&[-10.0], &[], 5.0, 0.0).unwrap();
    assert_eq!(zero_denominator.tau, state.tau());
    let upper = resolvent.solve(&[100.0], &[], 0.0, 0.0).unwrap();
    assert_close(upper.tau, 20.0);
    let singular = resolvent.solve(&[1.0], &[], 5.0, 0.0).unwrap();
    assert_close(singular.tau, 3.0);
    let nonfinite = resolvent.solve(&[1.0], &[], f64::MAX, 0.0).unwrap();
    assert_close(nonfinite.tau, 20.0);

    // Nonfinite fallback coefficients and correction preserve current tau.
    resolvent.fallback.get_mut().unwrap().q_direction = f64::NAN;
    let retained = resolvent.solve(&[1.0], &[], 0.0, 0.0).unwrap();
    assert_eq!(retained.tau, state.tau());
}
