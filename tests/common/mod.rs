use cpo3::{CscMatrix, MatrixError};

pub fn matrix(rows: usize, columns: usize, values: Vec<f64>) -> Result<CscMatrix, MatrixError> {
    assert_eq!(values.len(), rows * columns);
    let mut ri = Vec::new();
    let mut ci = Vec::new();
    let mut data = Vec::new();
    for (i, v) in values.into_iter().enumerate() {
        if v != 0.0 {
            ri.push(i / columns);
            ci.push(i % columns);
            data.push(v);
        }
    }
    CscMatrix::from_triplets(rows, columns, ri, ci, data)
}
