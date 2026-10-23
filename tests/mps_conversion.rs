#![cfg(feature = "mps")]

use cpo3::mps::{MpsError, MpsOptions, from_parser, from_parser_with};
use mps::Parser;

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

const ROW_MODEL: &str = "NAME ROWS\nROWS\n N COST\n N UNUSED\n L R\nCOLUMNS\n X COST 1 R 1\n";

#[test]
fn rhs_and_ranges_define_the_row_interval_and_ignore_n_rows() {
    let source = format!(
        "{ROW_MODEL}RHS\n SET COST 10 R 5\n SET COST 20 UNUSED 30\n\
         RANGES\n SET UNUSED 10 R -2\n SET COST 20 UNUSED 30\n\
         BOUNDS\n FR BND X\nENDATA\n"
    );
    let parsed = Parser::<f64>::parse(&source).unwrap();
    let problem = from_parser(&parsed).unwrap();
    assert_eq!(problem.b(), [-3.0, 5.0]);
    assert_eq!(problem.a().values(), [-1.0, 1.0]);
    assert_eq!(problem.c(), [1.0]);
}

#[test]
fn rhs_and_ranges_select_only_the_first_named_set_when_requested() {
    for section in ["RHS", "RANGES"] {
        let source = format!(
            "{ROW_MODEL}{section}\n FIRST COST 10\n SECOND MISSING 99\n FIRST R 2\n\
             BOUNDS\n FR BND X\nENDATA\n"
        );
        let parsed = Parser::<f64>::parse(&source).unwrap();
        assert_eq!(
            from_parser(&parsed).unwrap_err(),
            MpsError::MultipleSets { section }
        );
        let problem = from_parser_with(
            &parsed,
            MpsOptions {
                first_named_set: true,
            },
        )
        .unwrap();
        let expected = if section == "RHS" {
            vec![2.0]
        } else {
            vec![2.0, 0.0]
        };
        assert_eq!(problem.b(), expected);
    }
}

#[test]
fn rhs_and_ranges_reject_duplicate_unknown_and_nonfinite_values() {
    for section in ["RHS", "RANGES"] {
        for (entries, expected) in [
            (
                " SET R 0 R 0",
                MpsError::DuplicateValue {
                    section,
                    name: "R".into(),
                },
            ),
            (" SET MISSING 1", MpsError::UnknownRow("MISSING".into())),
            (
                " SET COST 1",
                MpsError::NonFiniteValue {
                    section,
                    name: "COST".into(),
                },
            ),
        ] {
            let source = format!("{ROW_MODEL}{section}\n{entries}\nENDATA\n");
            let mut parsed = Parser::<f64>::parse(&source).unwrap();
            if matches!(expected, MpsError::NonFiniteValue { .. }) {
                let lines = if section == "RHS" {
                    parsed.rhs.as_mut()
                } else {
                    parsed.ranges.as_mut()
                };
                lines.unwrap()[0].first_pair.value = f64::INFINITY;
            }
            assert_eq!(from_parser(&parsed).unwrap_err(), expected);
        }
    }
}
