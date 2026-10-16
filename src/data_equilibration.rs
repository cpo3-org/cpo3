//! One-time equilibration of the homogeneous problem data.

use std::borrow::Cow;

use thiserror::Error;

use crate::{Cone, CscMatrix, Equilibration, MatrixError, Problem, ProblemError, Settings};

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum DataEquilibrationError {
    #[error(transparent)]
    Matrix(#[from] MatrixError),
    #[error(transparent)]
    Problem(#[from] ProblemError),
    #[error("data equilibration produced invalid numerical values")]
    InvalidNumericalResult,
}

/// A problem in equilibrated homogeneous coordinates and its inverse mapping.
pub(crate) struct EquilibratedProblem<'a> {
    problem: Cow<'a, Problem>,
    x: Vec<f64>,
    y: Vec<f64>,
    tau: f64,
}

impl<'a> EquilibratedProblem<'a> {
    pub(crate) fn new(
        problem: &'a Problem,
        settings: &Settings,
    ) -> Result<Self, DataEquilibrationError> {
        match settings.data_equilibration {
            Equilibration::None => Ok(Self::unscaled(problem)),
            Equilibration::Ruiz => Self::ruiz(problem, settings),
        }
    }

    fn unscaled(problem: &'a Problem) -> Self {
        Self {
            problem: Cow::Borrowed(problem),
            x: vec![1.0; problem.variable_count()],
            y: vec![1.0; problem.constraint_count()],
            tau: 1.0,
        }
    }

    pub(crate) fn problem(&self) -> &Problem {
        &self.problem
    }

    pub(crate) fn scale_x(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(&self.x)
            .map(|(value, scale)| value * self.tau / scale)
            .collect()
    }

    pub(crate) fn scale_y(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(&self.y)
            .map(|(value, scale)| value * self.tau / scale)
            .collect()
    }

    pub(crate) fn scale_s(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(&self.y)
            .map(|(value, scale)| value * self.tau * scale)
            .collect()
    }

    pub(crate) fn unscale_x(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(&self.x)
            .map(|(value, scale)| scale * value / self.tau)
            .collect()
    }

    pub(crate) fn unscale_y(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(&self.y)
            .map(|(value, scale)| scale * value / self.tau)
            .collect()
    }

    /// `unscale_x(values * factor)` in one pass.
    pub(crate) fn unscale_x_into(&self, values: &[f64], factor: f64, out: &mut Vec<f64>) {
        out.clear();
        out.extend(
            values
                .iter()
                .zip(&self.x)
                .map(|(value, scale)| scale * (value * factor) / self.tau),
        );
    }

    pub(crate) fn unscale_y_into(&self, values: &[f64], factor: f64, out: &mut Vec<f64>) {
        out.clear();
        out.extend(
            values
                .iter()
                .zip(&self.y)
                .map(|(value, scale)| scale * (value * factor) / self.tau),
        );
    }

    pub(crate) fn unscale_s_by(&self, values: &[f64], factor: f64) -> Vec<f64> {
        let mut out = Vec::new();
        self.unscale_s_into(values, factor, &mut out);
        out
    }

    pub(crate) fn unscale_s_into(&self, values: &[f64], factor: f64, out: &mut Vec<f64>) {
        out.clear();
        out.extend(
            values
                .iter()
                .zip(&self.y)
                .map(|(value, scale)| (value * factor) / (self.tau * scale)),
        );
    }

    fn ruiz(problem: &'a Problem, settings: &Settings) -> Result<Self, DataEquilibrationError> {
        if settings.max_equilibration_passes == 0 {
            return Ok(Self::unscaled(problem));
        }
        let n = problem.variable_count();
        let m = problem.constraint_count();
        let mut scaling = vec![1.0; n + m + 1];

        let mut norms = vec![0.0_f64; scaling.len()];
        for _ in 0..settings.max_equilibration_passes {
            norms.fill(0.0);
            accumulate_augmented_norms(problem, &scaling, &mut norms);
            cone_compatible_norms(problem.cones(), &mut norms[n..n + m]);
            let deviation = norms
                .iter()
                .filter(|norm| **norm > 0.0)
                .map(|norm| (norm - 1.0).abs())
                .fold(0.0_f64, f64::max);
            if deviation <= settings.equilibration_tolerance {
                break;
            }
            for (scale, norm) in scaling.iter_mut().zip(&norms) {
                if *norm > 0.0 {
                    *scale /= norm.sqrt();
                }
                if !scale.is_finite() || *scale <= 0.0 {
                    return Err(DataEquilibrationError::InvalidNumericalResult);
                }
            }
        }

        // Unit scaling needs neither a matrix copy nor repeated validation.
        if scaling.iter().all(|&scale| scale == 1.0) {
            return Ok(Self::unscaled(problem));
        }

        let x = scaling[..n].to_vec();
        let y = scaling[n..n + m].to_vec();
        let tau = scaling[n + m];
        let p = problem
            .p()
            .map(|matrix| scale_matrix(matrix, &x, &x))
            .transpose()?;
        let a = scale_matrix(problem.a(), &y, &x)?;
        let c = problem
            .c()
            .iter()
            .zip(&x)
            .map(|(value, scale)| tau * scale * value)
            .collect();
        let b = problem
            .b()
            .iter()
            .zip(&y)
            .map(|(value, scale)| tau * scale * value)
            .collect();
        let scaled = Problem::from_upper_triangle(p, c, a, b, problem.cones().to_vec())?;
        Ok(Self {
            problem: Cow::Owned(scaled),
            x,
            y,
            tau,
        })
    }
}

fn accumulate_augmented_norms(problem: &Problem, scaling: &[f64], norms: &mut [f64]) {
    let n = problem.variable_count();
    let m = problem.constraint_count();
    if let Some(p) = problem.p() {
        crate::matrix::visit_entries(p, |row, column, value| {
            accumulate_edge(row, column, value, scaling, norms);
        });
    }
    crate::matrix::visit_entries(problem.a(), |row, column, value| {
        accumulate_edge(column, n + row, value, scaling, norms);
    });
    let tau = n + m;
    for (column, value) in problem.c().iter().enumerate() {
        accumulate_edge(column, tau, *value, scaling, norms);
    }
    for (row, value) in problem.b().iter().enumerate() {
        accumulate_edge(n + row, tau, *value, scaling, norms);
    }
}

fn accumulate_edge(left: usize, right: usize, value: f64, scaling: &[f64], norms: &mut [f64]) {
    let magnitude = value.abs() * scaling[left] * scaling[right];
    norms[left] = norms[left].max(magnitude);
    norms[right] = norms[right].max(magnitude);
}

fn cone_compatible_norms(cones: &[Cone], norms: &mut [f64]) {
    let mut offset = 0;
    for cone in cones {
        let dim = cone
            .vector_dim()
            .expect("validated cone dimensions cannot overflow");
        if matches!(cone, Cone::SecondOrder(_) | Cone::PositiveSemidefinite(_)) {
            let block_norm = norms[offset..offset + dim]
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);
            norms[offset..offset + dim].fill(block_norm);
        }
        offset += dim;
    }
}

fn scale_matrix(
    matrix: &CscMatrix,
    row_scaling: &[f64],
    column_scaling: &[f64],
) -> Result<CscMatrix, MatrixError> {
    let mut values = matrix.values().to_vec();
    for (column, column_scale) in column_scaling
        .iter()
        .copied()
        .enumerate()
        .take(matrix.columns())
    {
        for index in matrix.column_pointers()[column]..matrix.column_pointers()[column + 1] {
            values[index] *= row_scaling[matrix.row_indices()[index]] * column_scale;
        }
    }
    matrix.with_values(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> Problem {
        Problem::new(
            Some(crate::matrix::test_matrix(2, 2, vec![4.0, 1.0, 1.0, 9.0]).unwrap()),
            vec![8.0, -0.25],
            crate::matrix::test_matrix(2, 2, vec![2.0, -3.0, 0.5, 7.0]).unwrap(),
            vec![5.0, 0.125],
            vec![Cone::Zero(1), Cone::Nonnegative(1)],
        )
        .unwrap()
    }

    fn matrix_vector(matrix: &CscMatrix, vector: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; matrix.rows()];
        crate::matrix::visit_entries(matrix, |row, column, value| {
            result[row] += value * vector[column];
        });
        result
    }

    fn matrix_transpose_vector(matrix: &CscMatrix, vector: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; matrix.columns()];
        crate::matrix::visit_entries(matrix, |row, column, value| {
            result[column] += value * vector[row];
        });
        result
    }

    fn dot(left: &[f64], right: &[f64]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-12 * (1.0 + expected.abs()));
    }

    #[test]
    fn none_borrows_the_original_problem_and_uses_identity_maps() {
        let problem = problem();
        let equilibrated = EquilibratedProblem::new(
            &problem,
            &crate::Settings {
                data_equilibration: Equilibration::None,
                ..crate::Settings::default()
            },
        )
        .unwrap();
        assert!(std::ptr::eq(equilibrated.problem(), &problem));
        assert_eq!(equilibrated.unscale_x(&[2.0, -1.0]), [2.0, -1.0]);
        assert_eq!(equilibrated.unscale_y(&[3.0, 4.0]), [3.0, 4.0]);
        assert_eq!(equilibrated.unscale_s_by(&[0.0, 5.0], 1.0), [0.0, 5.0]);
    }

    #[test]
    fn unit_ruiz_scaling_preserves_original_data_and_maps() {
        {
            let matrix: CscMatrix =
                crate::matrix::test_matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
            let problem = Problem::new(
                Some(matrix.clone()),
                vec![0.0, 0.0],
                matrix,
                vec![0.0, 0.0],
                vec![Cone::Zero(2)],
            )
            .unwrap();
            let equilibrated = EquilibratedProblem::new(
                &problem,
                &crate::Settings {
                    data_equilibration: Equilibration::Ruiz,
                    ..crate::Settings::default()
                },
            )
            .unwrap();
            assert!(std::ptr::eq(equilibrated.problem(), &problem));
            assert_eq!(equilibrated.scale_x(&[2.0, -3.0]), [2.0, -3.0]);
            assert_eq!(equilibrated.unscale_y(&[4.0, -5.0]), [4.0, -5.0]);
        }
    }

    #[test]
    fn homogeneous_ruiz_preserves_the_qp_under_coordinate_mapping() {
        {
            let original = problem();
            let equilibrated = EquilibratedProblem::new(
                &original,
                &crate::Settings {
                    data_equilibration: Equilibration::Ruiz,
                    ..crate::Settings::default()
                },
            )
            .unwrap();
            let scaled = equilibrated.problem();
            let scaled_x = [0.75, -1.25];
            let scaled_y = [-0.5, 2.0];
            let scaled_s = [0.0, 1.5];
            let x = equilibrated.unscale_x(&scaled_x);
            let y = equilibrated.unscale_y(&scaled_y);
            let s = equilibrated.unscale_s_by(&scaled_s, 1.0);

            let scaled_px = scaled.p_mul(&scaled_x);
            let original_px = original.p_mul(&x);
            let scaled_objective = 0.5 * dot(&scaled_x, &scaled_px) + dot(scaled.c(), &scaled_x);
            let original_objective = 0.5 * dot(&x, &original_px) + dot(original.c(), &x);
            assert_close(
                scaled_objective,
                equilibrated.tau.powi(2) * original_objective,
            );
            assert_close(
                dot(&scaled_y, &scaled_s),
                equilibrated.tau.powi(2) * dot(&y, &s),
            );

            let scaled_ax = matrix_vector(scaled.a(), &scaled_x);
            let original_ax = matrix_vector(original.a(), &x);
            for row in 0..original.constraint_count() {
                assert_close(
                    scaled_ax[row] - scaled.b()[row],
                    equilibrated.tau * equilibrated.y[row] * (original_ax[row] - original.b()[row]),
                );
            }

            let scaled_aty = matrix_transpose_vector(scaled.a(), &scaled_y);
            let original_aty = matrix_transpose_vector(original.a(), &y);
            for column in 0..original.variable_count() {
                assert_close(
                    scaled_px[column] + scaled_aty[column] + scaled.c()[column],
                    equilibrated.tau
                        * equilibrated.x[column]
                        * (original_px[column] + original_aty[column] + original.c()[column]),
                );
            }
        }
    }

    #[test]
    fn homogeneous_ruiz_balances_the_augmented_data() {
        let original = problem();
        let equilibrated = EquilibratedProblem::new(
            &original,
            &crate::Settings {
                data_equilibration: Equilibration::Ruiz,
                max_equilibration_passes: 20,
                ..crate::Settings::default()
            },
        )
        .unwrap();
        let scaled = equilibrated.problem();
        let mut norms = vec![0.0; scaled.variable_count() + scaled.constraint_count() + 1];

        accumulate_augmented_norms(scaled, &vec![1.0; norms.len()], &mut norms);

        for norm in norms {
            assert!(
                (norm - 1.0).abs() <= Settings::default().equilibration_tolerance,
                "row norm is {norm}"
            );
        }
    }
}
