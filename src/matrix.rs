mod csc;
pub use ::presolve::matrix::MatrixError;
pub use csc::CscMatrix;
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::{Accum, ColMut, ColRef, Conj, Par, RowRef};

/// Retain the upper triangle of an owned symmetric CSC matrix.
pub(crate) fn take_upper_triangle(p: CscMatrix) -> CscMatrix {
    let n = p.columns();
    let (mut pointers, mut rows, mut values) = p.into_parts();
    let mut write = 0;
    let mut start = 0;
    for j in 0..n {
        let end = pointers[j + 1];
        pointers[j] = write;
        for at in start..end {
            if rows[at] <= j {
                rows[write] = rows[at];
                values[write] = values[at];
                write += 1;
            }
        }
        start = end;
    }
    pointers[n] = write;
    rows.truncate(write);
    values.truncate(write);
    rows.shrink_to_fit();
    values.shrink_to_fit();
    CscMatrix::from_parts(n, n, pointers, rows, values)
}
/// Apply one symmetric product to N output accumulators. Keep each column's
/// mirrored contribution local, and handle its diagonal outside the inner loop.
/// Sorted upper-triangle CSC preserves ascending-column accumulation order.
#[inline]
pub(crate) fn symmetric_accumulate<const N: usize>(
    matrix: &CscMatrix,
    x: &[f64],
    outputs: [&mut [f64]; N],
    map: impl Fn(f64) -> [f64; N],
) {
    faer::with_dim!(dim, matrix.columns());
    let matrix = matrix.as_faer().as_shape(dim, dim);
    let x = ColRef::from_slice(x).as_row_shape(dim);
    let mut outputs = outputs.map(|out| ColMut::from_slice_mut(out).as_row_shape_mut(dim));
    for column in dim.indices() {
        let rows = matrix.row_idx_of_col_raw(column);
        let values = matrix.val_of_col(column);
        if rows.is_empty() {
            continue;
        }
        let xc = x[column];
        let diagonal = rows.last() == Some(&column);
        let stop = rows.len() - usize::from(diagonal);
        if stop == 0 {
            let values = map(values[0] * xc);
            for (out, value) in outputs.iter_mut().zip(values) {
                out[column] += value;
            }
            continue;
        }
        let mut sums: [f64; N] = std::array::from_fn(|i| outputs[i][column]);
        for (&row, &value) in rows[..stop].iter().zip(&values[..stop]) {
            let scatter = map(value * xc);
            let gather = map(value * x[row]);
            for i in 0..N {
                outputs[i][row] += scatter[i];
                sums[i] += gather[i];
            }
        }
        if diagonal {
            let values = map(values[stop] * xc);
            for (sum, value) in sums.iter_mut().zip(values) {
                *sum += value;
            }
        }
        for (out, sum) in outputs.iter_mut().zip(sums) {
            out[column] = sum;
        }
    }
}

pub(crate) fn symmetric_mul_into(matrix: &CscMatrix, x: &[f64], out: &mut Vec<f64>) {
    out.resize(matrix.rows(), 0.0);
    out.fill(0.0);
    symmetric_accumulate(matrix, x, [out], |product| [product]);
}

pub(crate) fn matrix_mul(matrix: &CscMatrix, vector: &[f64]) -> Vec<f64> {
    let mut product = Vec::new();
    matrix_mul_into(matrix, vector, &mut product);
    product
}

pub(crate) fn matrix_mul_into(matrix: &CscMatrix, vector: &[f64], product: &mut Vec<f64>) {
    product.resize(matrix.rows(), 0.0);
    sparse_dense_matmul(
        ColMut::from_slice_mut(product).as_mat_mut(),
        Accum::Replace,
        matrix.as_faer(),
        ColRef::from_slice(vector).as_mat(),
        1.0,
        Par::Seq,
    );
}

/// Compute both constraint products in one traversal of dimension-bound faer views.
/// Two sparse_dense_matmul calls reread every coefficient and index.
pub(crate) fn matrix_products_into(
    matrix: &CscMatrix,
    x: &[f64],
    y: &[f64],
    ax: &mut Vec<f64>,
    aty: &mut Vec<f64>,
) {
    ax.resize(matrix.rows(), 0.0);
    ax.fill(0.0);
    aty.resize(matrix.columns(), 0.0);
    aty.fill(0.0);
    constraint_accumulate(matrix, x, y, [aty], [ax], |product| [product]);
}

/// Accumulate the off-diagonal KKT blocks without temporary vectors.
#[inline]
pub(crate) fn constraint_accumulate<const N: usize>(
    matrix: &CscMatrix,
    x: &[f64],
    y: &[f64],
    out_x: [&mut [f64]; N],
    out_y: [&mut [f64]; N],
    map: impl Fn(f64) -> [f64; N],
) {
    faer::with_dim!(rows, matrix.rows());
    faer::with_dim!(cols, matrix.columns());
    let matrix = matrix.as_faer().as_shape(rows, cols);
    let x = ColRef::from_slice(x).as_row_shape(cols);
    let y = ColRef::from_slice(y).as_row_shape(rows);
    let mut out_x = out_x.map(|out| ColMut::from_slice_mut(out).as_row_shape_mut(cols));
    let mut out_y = out_y.map(|out| ColMut::from_slice_mut(out).as_row_shape_mut(rows));
    for column in cols.indices() {
        let xc = x[column];
        let mut sums: [f64; N] = std::array::from_fn(|i| out_x[i][column]);
        for (&row, &value) in matrix
            .row_idx_of_col_raw(column)
            .iter()
            .zip(matrix.val_of_col(column))
        {
            let gather = map(value * y[row]);
            let scatter = map(value * xc);
            for i in 0..N {
                sums[i] += gather[i];
                out_y[i][row] += scatter[i];
            }
        }
        for (out, sum) in out_x.iter_mut().zip(sums) {
            out[column] = sum;
        }
    }
}

pub(crate) fn matrix_transpose_mul(matrix: &CscMatrix, vector: &[f64]) -> Vec<f64> {
    let mut product = Vec::new();
    matrix_transpose_mul_into(matrix, vector, &mut product);
    product
}

pub(crate) fn matrix_transpose_mul_into(
    matrix: &CscMatrix,
    vector: &[f64],
    product: &mut Vec<f64>,
) {
    product.resize(matrix.columns(), 0.0);
    sparse_dense_matmul(
        ColMut::from_slice_mut(product).as_mat_mut(),
        Accum::Replace,
        matrix.as_faer().transpose(),
        ColRef::from_slice(vector).as_mat(),
        1.0,
        Par::Seq,
    );
}

pub(crate) fn dot(left: &[f64], right: &[f64]) -> f64 {
    faer::linalg::matmul::dot::inner_prod(
        RowRef::from_slice(left),
        Conj::No,
        ColRef::from_slice(right),
        Conj::No,
    )
}

/// Apply an elementwise update over contiguous, equally sized faer views.
#[inline]
pub(crate) fn update(out: &mut [f64], input: &[f64], f: impl Fn(f64, f64) -> f64) {
    faer::zip!(ColMut::from_slice_mut(out), ColRef::from_slice(input))
        .for_each(|faer::unzip!(out, input)| *out = f(*out, *input));
}

#[inline]
pub(crate) fn binary_into(
    out: &mut [f64],
    left: &[f64],
    right: &[f64],
    f: impl Fn(f64, f64) -> f64,
) {
    faer::zip!(
        ColMut::from_slice_mut(out),
        ColRef::from_slice(left),
        ColRef::from_slice(right)
    )
    .for_each(|faer::unzip!(out, left, right)| *out = f(*left, *right));
}

pub(crate) fn scale(values: &mut [f64], factor: f64) {
    let mut view = ColMut::from_slice_mut(values);
    view *= factor;
}

pub(crate) fn add_assign(out: &mut [f64], input: &[f64]) {
    let mut view = ColMut::from_slice_mut(out);
    view += ColRef::from_slice(input);
}

/// Visit every stored entry in column order.
pub(crate) fn visit_entries(matrix: &CscMatrix, mut visit: impl FnMut(usize, usize, f64)) {
    for column in 0..matrix.columns() {
        for index in matrix.column_pointers()[column]..matrix.column_pointers()[column + 1] {
            visit(matrix.row_indices()[index], column, matrix.values()[index]);
        }
    }
}

/// Build a small sparse fixture from row-major values. Test code only.
#[cfg(test)]
pub(crate) fn test_matrix(
    rows: usize,
    columns: usize,
    values: Vec<f64>,
) -> Result<CscMatrix, MatrixError> {
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

#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    #[test]
    fn rectangular_products_handle_explicit_sparse_zeros() {
        let canonical: CscMatrix =
            crate::matrix::test_matrix(2, 3, vec![2.0, 0.0, -1.0, 0.0, 3.0, 4.0]).unwrap();
        let sparse: CscMatrix = CscMatrix::new(
            2,
            3,
            vec![0, 2, 3, 5],
            vec![0, 1, 1, 0, 1],
            vec![2.0, 0.0, 3.0, -1.0, 4.0],
        )
        .unwrap();
        for matrix in [&canonical, &sparse] {
            assert_eq!(matrix_mul(matrix, &[1.0, 2.0, 3.0]), vec![-1.0, 18.0]);
            assert_eq!(
                matrix_transpose_mul(matrix, &[2.0, -1.0]),
                vec![4.0, -3.0, -6.0]
            );
        }
        let mut entries = Vec::new();
        visit_entries(&sparse, |row, col, value| entries.push((row, col, value)));
        assert_eq!(
            entries,
            vec![
                (0, 0, 2.0),
                (1, 0, 0.0),
                (1, 1, 3.0),
                (0, 2, -1.0),
                (1, 2, 4.0)
            ]
        );
    }
}

/// Euclidean norm with a cheap ordinary path and scaled extreme-value fallback.
/// A clonable iterator lets callers measure trial/residual coordinates without
/// allocating a vector. Unlike a hypot fold, the common path takes one sqrt.
pub(crate) fn stable_norm(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let squared = values.clone().map(|v| v * v).sum::<f64>();
    if squared.is_finite() && squared > 0.0 {
        return squared.sqrt();
    }
    if squared.is_nan() {
        return f64::NAN;
    }
    let scale = values.clone().map(f64::abs).fold(0.0, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    scale * values.map(|v| (v / scale).powi(2)).sum::<f64>().sqrt()
}

#[cfg(test)]
mod norm_tests {
    use super::stable_norm;

    #[test]
    fn ordinary_extreme_and_nonfinite_norms() {
        for scale in [1.0, 1e-200, 1e200] {
            let norm = stable_norm([3.0 * scale, -4.0 * scale].into_iter());
            assert!((norm / scale - 5.0).abs() < 1e-14);
        }
        assert_eq!(stable_norm([0.0, 0.0].into_iter()), 0.0);
        assert_eq!(stable_norm([1.0, f64::INFINITY].into_iter()), f64::INFINITY);
        assert!(stable_norm([f64::NAN].into_iter()).is_nan());
    }
}
