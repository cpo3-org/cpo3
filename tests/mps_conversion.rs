#![cfg(feature = "mps")]

use cpo3::mps::MpsError;
use cpo3::{Cone, CscMatrix};
use mps::Parser;

fn sparse_value(matrix: &CscMatrix, row: usize, column: usize) -> f64 {
    matrix.get(row, column).unwrap_or(0.0)
}

#[test]
fn converts_rows_ranges_and_continuous_bounds_to_canonical_cones() {
    let parsed = Parser::<f64>::parse(include_str!("fixtures/parser_compatibility.mps"))
        .expect("the compatibility MPS fixture should parse");
    let problem = cpo3::mps::from_parser(&parsed).expect("the LP should convert");

    assert_eq!(problem.variable_count(), 6);
    assert_eq!(problem.constraint_count(), 11);
    assert!(problem.p().is_none());
    assert_eq!(problem.c(), &[1.0, -2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(problem.cones(), &[Cone::Zero(2), Cone::Nonnegative(9)]);
    assert_eq!(
        problem.b(),
        &[5.0, 2.0, -5.5, 7.0, 2.0, 1.0, 1.0, 10.0, 0.0, -0.0, -0.0]
    );

    let a = problem.a();
    assert_eq!(sparse_value(a, 0, 0), 2.0);
    assert_eq!(sparse_value(a, 0, 1), -1.0);
    assert_eq!(sparse_value(a, 0, 4), 6.0);
    assert_eq!(sparse_value(a, 1, 1), 1.0);
    assert_eq!(sparse_value(a, 2, 0), -3.0);
    assert_eq!(sparse_value(a, 2, 2), -1.0);
    assert_eq!(sparse_value(a, 2, 5), 1.0);
    assert_eq!(sparse_value(a, 3, 0), 3.0);
    assert_eq!(sparse_value(a, 4, 0), -4.0);
    assert_eq!(sparse_value(a, 4, 3), 2.0);
    assert_eq!(sparse_value(a, 5, 0), 4.0);
    assert_eq!(sparse_value(a, 5, 3), -2.0);
    assert_eq!(sparse_value(a, 6, 0), -1.0);
    assert_eq!(sparse_value(a, 7, 0), 1.0);
    assert_eq!(sparse_value(a, 8, 3), 1.0);
    assert_eq!(sparse_value(a, 9, 4), -1.0);
    assert_eq!(sparse_value(a, 10, 5), -1.0);
}

#[test]
fn converts_quadratic_objective_and_default_variable_bounds() {
    let parsed = Parser::<f64>::parse(include_str!("fixtures/parser_compatibility.qps"))
        .expect("the compatibility QPS fixture should parse");
    let problem = cpo3::mps::from_parser(&parsed).expect("the QP should convert");

    assert_eq!(problem.c(), &[-1.0, -2.0]);
    assert_eq!(problem.cones(), &[Cone::Zero(1), Cone::Nonnegative(2)]);
    assert_eq!(problem.b(), &[3.0, -0.0, -0.0]);

    let p = problem.p().expect("QPS conversion should produce P");
    assert_eq!(sparse_value(p, 0, 0), 2.0);
    assert_eq!(sparse_value(p, 0, 1), 0.5);
    assert_eq!(sparse_value(p, 1, 0), 0.0);
    assert_eq!(sparse_value(p, 1, 1), 4.0);
    assert_eq!(sparse_value(problem.a(), 0, 0), 1.0);
    assert_eq!(sparse_value(problem.a(), 0, 1), 1.0);
    assert_eq!(sparse_value(problem.a(), 1, 0), -1.0);
    assert_eq!(sparse_value(problem.a(), 2, 1), -1.0);
}

#[test]
fn normalizes_maximization_objectives_to_minimization() {
    let contents = include_str!("fixtures/parser_compatibility.qps").replace(" MIN", " MAX");
    let parsed = Parser::<f64>::parse(&contents).expect("the maximization QPS should parse");
    let problem = cpo3::mps::from_parser(&parsed).expect("the maximization QP should convert");

    assert_eq!(problem.c(), &[1.0, 2.0]);
    let p = problem.p().expect("QPS conversion should produce P");
    assert_eq!(sparse_value(p, 0, 0), -2.0);
    assert_eq!(sparse_value(p, 0, 1), -0.5);
    assert_eq!(sparse_value(p, 1, 0), 0.0);
    assert_eq!(sparse_value(p, 1, 1), -4.0);
}

#[test]
fn rejects_discrete_variable_bounds() {
    let parsed = Parser::<f64>::parse(
        "NAME INTEGER\nROWS\n N COST\nCOLUMNS\n X1 COST 1\nBOUNDS\n BV BND1 X1\nENDATA\n",
    )
    .expect("the integer MPS should parse");
    let error = cpo3::mps::from_parser(&parsed).expect_err("integer bounds are unsupported");

    assert_eq!(
        error,
        MpsError::UnsupportedBoundType {
            bound_type: "BV",
            variable: "X1".to_string(),
        }
    );
}
