#![cfg(feature = "mps")]

use cpo3::mps::MpsError;
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
