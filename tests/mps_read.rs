#![cfg(feature = "mps")]

use std::path::{Path, PathBuf};

use cpo3::Cone;
use cpo3::mps::{MpsError, MpsOptions};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn reads_and_converts_an_mps_file() {
    let problem =
        cpo3::mps::read(fixture("parser_compatibility.mps")).expect("the MPS fixture should load");

    assert_eq!(problem.variable_count(), 6);
    assert_eq!(problem.constraint_count(), 11);
    assert_eq!(problem.cones(), &[Cone::Zero(2), Cone::Nonnegative(9)]);
}

#[test]
fn reads_and_converts_a_qps_file() {
    let problem =
        cpo3::mps::read(fixture("parser_compatibility.qps")).expect("the QPS fixture should load");

    assert_eq!(problem.variable_count(), 2);
    assert_eq!(problem.c(), &[-1.0, -2.0]);
    assert_eq!(problem.p().unwrap().rows(), 2);
}

#[test]
fn reports_the_path_when_a_file_cannot_be_read() {
    let path = fixture("does_not_exist.mps");
    let error = cpo3::mps::read(&path).expect_err("the missing file should fail");

    assert!(matches!(
        error,
        MpsError::Read {
            path: error_path,
            ..
        } if error_path == path
    ));
}

#[test]
fn reports_the_path_when_a_file_cannot_be_parsed() {
    let path = fixture("malformed.mps");
    let error = cpo3::mps::read(&path).expect_err("the malformed file should fail");

    assert!(matches!(
        error,
        MpsError::Parse {
            path: error_path,
            ..
        } if error_path == path
    ));
}

#[test]
fn several_named_bound_sets_are_rejected_unless_the_first_is_kept() {
    let path = fixture("two_bound_sets.mps");
    let error = cpo3::mps::read(&path).expect_err("two bound sets should fail by default");
    assert!(matches!(
        error,
        MpsError::MultipleSets { section: "BOUNDS" }
    ));

    let options = MpsOptions {
        first_named_set: true,
    };
    let problem = cpo3::mps::read_with(&path, options).expect("the first set should load");
    // ROW1, X1 <= 2 from the first set, and the two default lower bounds.
    assert_eq!(problem.constraint_count(), 4);
    assert!(problem.b().contains(&2.0));
    assert!(!problem.b().contains(&3.0));
}
