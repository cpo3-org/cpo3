#![cfg(feature = "mps")]

use std::path::{Path, PathBuf};

use cpo3::mps::MpsError;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
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
