#![cfg(feature = "mps")]

use mps::Parser;
use mps::types::{BoundType, ObjectiveSense, RowType, WideLine};

fn pairs<'a>(lines: &'a [WideLine<'a, f64>]) -> Vec<(&'a str, &'a str, f64)> {
    lines
        .iter()
        .flat_map(|line| {
            let first = (line.name, line.first_pair.row_name, line.first_pair.value);
            std::iter::once(first).chain(
                line.second_pair
                    .as_ref()
                    .map(|pair| (line.name, pair.row_name, pair.value)),
            )
        })
        .collect()
}

#[test]
fn parser_exposes_linear_mps_sections_without_losing_semantics() {
    let parsed = Parser::<f64>::parse(include_str!("fixtures/parser_compatibility.mps"))
        .expect("the compatibility MPS fixture should parse");

    assert_eq!(parsed.name, "PARSERLP");
    assert_eq!(parsed.objective_sense, Some(ObjectiveSense::Min));
    assert_eq!(parsed.objective_name, Some("COST"));
    assert_eq!(
        parsed
            .rows
            .iter()
            .map(|row| (&row.row_type, row.row_name))
            .collect::<Vec<_>>(),
        vec![
            (&RowType::Nr, "COST"),
            (&RowType::Eq, "EQROW"),
            (&RowType::Leq, "LEROW"),
            (&RowType::Geq, "GEROW"),
        ]
    );
    assert_eq!(
        pairs(&parsed.columns),
        vec![
            ("X1", "COST", 1.0),
            ("X1", "EQROW", 2.0),
            ("X1", "LEROW", 3.0),
            ("X1", "GEROW", 4.0),
            ("X2", "COST", -2.0),
            ("X2", "EQROW", -1.0),
            ("X3", "COST", 3.0),
            ("X3", "LEROW", 1.0),
            ("X4", "COST", 4.0),
            ("X4", "GEROW", -2.0),
            ("X5", "COST", 5.0),
            ("X5", "EQROW", 6.0),
            ("X6", "COST", 6.0),
            ("X6", "LEROW", -1.0),
        ]
    );
    assert_eq!(
        pairs(parsed.rhs.as_deref().expect("fixture has an RHS section")),
        vec![
            ("RHS1", "EQROW", 5.0),
            ("RHS1", "LEROW", 7.0),
            ("RHS1", "GEROW", -2.0),
        ]
    );
    assert_eq!(
        pairs(
            parsed
                .ranges
                .as_deref()
                .expect("fixture has a RANGES section")
        ),
        vec![("RNG1", "LEROW", 1.5), ("RNG1", "GEROW", -3.0),]
    );

    let bounds = parsed.bounds.expect("fixture has a BOUNDS section");
    assert_eq!(bounds.len(), 7);
    assert_eq!(
        bounds
            .iter()
            .map(|bound| (
                &bound.bound_type,
                bound.bound_name,
                bound.column_name,
                bound.value,
            ))
            .collect::<Vec<_>>(),
        vec![
            (&BoundType::Lo, "BND1", "X1", Some(-1.0)),
            (&BoundType::Up, "BND1", "X1", Some(10.0)),
            (&BoundType::Fx, "BND1", "X2", Some(2.0)),
            (&BoundType::Fr, "BND1", "X3", None),
            (&BoundType::Mi, "BND1", "X4", None),
            (&BoundType::Pl, "BND1", "X5", None),
            (&BoundType::Lo, "BND1", "X6", Some(0.0)),
        ]
    );
}

#[test]
fn parser_exposes_qps_quadratic_terms_in_file_orientation() {
    let contents = include_str!("fixtures/parser_compatibility.qps");
    let parsed =
        Parser::<f64>::parse(contents).expect("the compatibility QPS fixture should parse");

    assert_eq!(parsed.name, "PARSERQP");
    assert_eq!(parsed.objective_sense, Some(ObjectiveSense::Min));
    assert_eq!(
        pairs(&parsed.columns),
        vec![
            ("X1", "COST", -1.0),
            ("X1", "BALANCE", 1.0),
            ("X2", "COST", -2.0),
            ("X2", "BALANCE", 1.0),
        ]
    );

    let quadratic = parsed
        .quadratic_objective
        .expect("fixture has a QUADOBJ section");
    assert_eq!(quadratic.len(), 3);
    assert_eq!(quadratic[0].var1, "X1");
    assert_eq!(quadratic[0].var2, "X1");
    assert_eq!(quadratic[0].coefficient, 2.0);
    assert_eq!(quadratic[1].var1, "X1");
    assert_eq!(quadratic[1].var2, "X2");
    assert_eq!(quadratic[1].coefficient, 0.5);
    assert_eq!(quadratic[2].var1, "X2");
    assert_eq!(quadratic[2].var2, "X2");
    assert_eq!(quadratic[2].coefficient, 4.0);

    let windows_contents = contents.replace('\n', "\r\n");
    Parser::<f64>::parse(&windows_contents)
        .expect("the QPS parser should accept the collection's Windows line endings");
}
