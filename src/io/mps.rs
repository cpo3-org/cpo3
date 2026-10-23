//! Conversion from parsed MPS/QPS models into cpo3's canonical problem form.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use ::mps::Parser;
use ::mps::types::{BoundType, ObjectiveSense, RowType, WideLine};
use thiserror::Error;

use crate::{Cone, CscMatrix, MatrixError, Problem, ProblemError};

/// Errors produced while converting a parsed MPS/QPS model.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum MpsError {
    #[error("failed to read MPS/QPS file {path:?}: {message}")]
    Read { path: PathBuf, message: String },
    #[error("failed to parse MPS/QPS file {path:?}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("the MPS model does not define an objective row")]
    MissingObjectiveRow,
    #[error("objective row {0:?} is missing or is not an N row")]
    InvalidObjectiveRow(String),
    #[error("row {0:?} is defined more than once")]
    DuplicateRow(String),
    #[error("row {0:?} is referenced but not defined")]
    UnknownRow(String),
    #[error("variable {0:?} is referenced but not defined in COLUMNS")]
    UnknownVariable(String),
    #[error("the {section} section defines {name:?} more than once")]
    DuplicateValue { section: &'static str, name: String },
    #[error("bound type {bound_type} for variable {variable:?} requires a finite value")]
    MissingBoundValue {
        bound_type: &'static str,
        variable: String,
    },
    #[error("bound type {bound_type} for variable {variable:?} is not continuous")]
    UnsupportedBoundType {
        bound_type: &'static str,
        variable: String,
    },
    #[error("the {section} section contains more than one named set")]
    MultipleSets { section: &'static str },
    #[error("variable {variable:?} has inconsistent bounds [{lower}, {upper}]")]
    InconsistentBounds {
        variable: String,
        lower: f64,
        upper: f64,
    },
    #[error("the MPS model uses unsupported {0}")]
    UnsupportedExtension(&'static str),
    #[error("the {section} section contains a non-finite value for {name:?}")]
    NonFiniteValue { section: &'static str, name: String },
    #[error(
        "quadratic entries ({first}, {second}) and ({second}, {first}) disagree: {forward} != {reverse}"
    )]
    AsymmetricQuadraticEntry {
        first: String,
        second: String,
        forward: f64,
        reverse: f64,
    },
    #[error(transparent)]
    Matrix(#[from] MatrixError),
    #[error(transparent)]
    Problem(#[from] ProblemError),
}

#[derive(Clone, Copy)]
enum ConstraintSense {
    Equal,
    Less,
    Greater,
}

struct SourceConstraint {
    name: String,
    sense: ConstraintSense,
    coefficients: Vec<(usize, f64)>,
    rhs: f64,
    range: Option<f64>,
}

#[derive(Clone, Copy)]
enum SourceRow {
    Objective,
    Constraint(usize),
    Ignored,
}

#[derive(Clone, Copy)]
struct VariableBounds {
    lower: f64,
    upper: f64,
}

struct CanonicalRow {
    coefficients: Vec<(usize, f64)>,
    rhs: f64,
}

#[derive(Default)]
struct QuadraticPair {
    forward: f64,
    reverse: f64,
    has_forward: bool,
    has_reverse: bool,
}

/// Reader options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpsOptions {
    /// Use the first named RHS, RANGES, or BOUNDS set when several are
    /// present instead of rejecting the file.
    pub first_named_set: bool,
}

/// Read an MPS/QPS file and convert it into a sparse cpo3 problem.
pub fn read(path: impl AsRef<Path>) -> Result<Problem, MpsError> {
    read_with(path, MpsOptions::default())
}

/// [`read`] with explicit reader options.
pub fn read_with(path: impl AsRef<Path>, options: MpsOptions) -> Result<Problem, MpsError> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path).map_err(|error| MpsError::Read {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let parsed = Parser::<f64>::parse(&contents).map_err(|error| MpsError::Parse {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    from_parser_with(&parsed, options)
}

/// Convert an already parsed MPS/QPS model into a sparse cpo3 problem.
///
/// The source model is normalized to minimization form. Its linear rows,
/// ranges, and continuous variable bounds are converted to zero and
/// nonnegative cone rows. Several named RHS, RANGES, or BOUNDS sets are
/// rejected unless `MpsOptions::first_named_set` keeps the first one.
/// Objective constants do not affect the optimizer and are therefore
/// omitted from [`Problem`].
pub fn from_parser(parsed: &Parser<'_, f64>) -> Result<Problem, MpsError> {
    from_parser_with(parsed, MpsOptions::default())
}

/// [`from_parser`] with explicit reader options.
pub fn from_parser_with(
    parsed: &Parser<'_, f64>,
    options: MpsOptions,
) -> Result<Problem, MpsError> {
    reject_unsupported_extensions(parsed)?;

    let objective_name = objective_name(parsed)?;
    let (mut constraints, row_lookup) = collect_rows(parsed, objective_name)?;
    let (variable_names, variable_lookup) = collect_variables(parsed);
    let mut c = vec![0.0; variable_names.len()];

    collect_columns(
        parsed,
        &row_lookup,
        &variable_lookup,
        &mut constraints,
        &mut c,
    )?;
    collect_row_values(
        "RHS",
        parsed.rhs.as_deref(),
        &row_lookup,
        &mut constraints,
        options,
        |row, value| row.rhs = value,
    )?;
    collect_row_values(
        "RANGES",
        parsed.ranges.as_deref(),
        &row_lookup,
        &mut constraints,
        options,
        |row, value| row.range = Some(value),
    )?;
    let bounds = collect_bounds(parsed, &variable_names, &variable_lookup, options)?;

    let objective_sign = match parsed.objective_sense.unwrap_or(ObjectiveSense::Min) {
        ObjectiveSense::Min => 1.0,
        ObjectiveSense::Max => -1.0,
    };
    for value in &mut c {
        *value *= objective_sign;
    }
    let p = collect_quadratic_objective(parsed, &variable_lookup, objective_sign)?;

    let (zero_rows, nonnegative_rows) = canonical_rows(constraints, bounds)?;
    let zero_dim = zero_rows.len();
    let nonnegative_dim = nonnegative_rows.len();
    let row_count = zero_dim + nonnegative_dim;

    let mut row_indices = Vec::new();
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    let mut b = Vec::with_capacity(row_count);
    for (row, canonical) in zero_rows.into_iter().chain(nonnegative_rows).enumerate() {
        b.push(canonical.rhs);
        for (column, value) in canonical.coefficients {
            row_indices.push(row);
            column_indices.push(column);
            values.push(value);
        }
    }

    let a = CscMatrix::from_triplets(
        row_count,
        variable_names.len(),
        row_indices,
        column_indices,
        values,
    )?;
    let mut cones = Vec::with_capacity(2);
    if zero_dim != 0 {
        cones.push(Cone::Zero(zero_dim));
    }
    if nonnegative_dim != 0 {
        cones.push(Cone::Nonnegative(nonnegative_dim));
    }

    Problem::from_upper_triangle(p, c, a, b, cones).map_err(Into::into)
}

fn reject_unsupported_extensions(parsed: &Parser<'_, f64>) -> Result<(), MpsError> {
    let extensions = [
        (
            parsed
                .user_cuts
                .as_ref()
                .is_some_and(|value| !value.is_empty()),
            "user cuts",
        ),
        (
            parsed
                .special_ordered_sets
                .as_ref()
                .is_some_and(|value| !value.is_empty()),
            "special ordered sets",
        ),
        (
            parsed
                .quadratic_constraints
                .as_ref()
                .is_some_and(|value| !value.is_empty()),
            "quadratic constraints",
        ),
        (
            parsed
                .indicators
                .as_ref()
                .is_some_and(|value| !value.is_empty()),
            "indicator constraints",
        ),
        (
            parsed
                .lazy_constraints
                .as_ref()
                .is_some_and(|value| !value.is_empty()),
            "lazy constraints",
        ),
        (
            parsed
                .cone_constraints
                .as_ref()
                .is_some_and(|value| !value.is_empty()),
            "cone constraints",
        ),
        (
            parsed
                .branch_priorities
                .as_ref()
                .is_some_and(|value| !value.is_empty()),
            "branch priorities",
        ),
    ];
    if let Some((_, name)) = extensions.into_iter().find(|(present, _)| *present) {
        return Err(MpsError::UnsupportedExtension(name));
    }
    Ok(())
}

fn objective_name<'a>(parsed: &'a Parser<'_, f64>) -> Result<&'a str, MpsError> {
    if let Some(name) = parsed.objective_name {
        if parsed
            .rows
            .iter()
            .any(|row| row.row_name == name && row.row_type == RowType::Nr)
        {
            return Ok(name);
        }
        return Err(MpsError::InvalidObjectiveRow(name.to_string()));
    }
    parsed
        .rows
        .iter()
        .find(|row| row.row_type == RowType::Nr)
        .map(|row| row.row_name)
        .ok_or(MpsError::MissingObjectiveRow)
}

fn collect_rows(
    parsed: &Parser<'_, f64>,
    objective_name: &str,
) -> Result<(Vec<SourceConstraint>, HashMap<String, SourceRow>), MpsError> {
    let mut constraints = Vec::new();
    let mut lookup = HashMap::new();
    for row in &parsed.rows {
        let kind = if row.row_name == objective_name {
            SourceRow::Objective
        } else {
            match row.row_type {
                RowType::Eq => {
                    push_constraint(&mut constraints, row.row_name, ConstraintSense::Equal)
                }
                RowType::Leq => {
                    push_constraint(&mut constraints, row.row_name, ConstraintSense::Less)
                }
                RowType::Geq => {
                    push_constraint(&mut constraints, row.row_name, ConstraintSense::Greater)
                }
                RowType::Nr => SourceRow::Ignored,
            }
        };
        if lookup.insert(row.row_name.to_string(), kind).is_some() {
            return Err(MpsError::DuplicateRow(row.row_name.to_string()));
        }
    }
    Ok((constraints, lookup))
}

fn push_constraint(
    constraints: &mut Vec<SourceConstraint>,
    name: &str,
    sense: ConstraintSense,
) -> SourceRow {
    let index = constraints.len();
    constraints.push(SourceConstraint {
        name: name.to_string(),
        sense,
        coefficients: Vec::new(),
        rhs: 0.0,
        range: None,
    });
    SourceRow::Constraint(index)
}

fn collect_variables<'a>(parsed: &'a Parser<'_, f64>) -> (Vec<&'a str>, HashMap<&'a str, usize>) {
    let mut names = Vec::new();
    let mut lookup = HashMap::new();
    for column in &parsed.columns {
        if !lookup.contains_key(column.name) {
            let index = names.len();
            names.push(column.name);
            lookup.insert(column.name, index);
        }
    }
    (names, lookup)
}

fn collect_columns(
    parsed: &Parser<'_, f64>,
    row_lookup: &HashMap<String, SourceRow>,
    variable_lookup: &HashMap<&str, usize>,
    constraints: &mut [SourceConstraint],
    c: &mut [f64],
) -> Result<(), MpsError> {
    for column in &parsed.columns {
        let variable = variable_lookup[column.name];
        for pair in line_pairs(column) {
            finite("COLUMNS", pair.row_name, pair.value)?;
            match row_lookup.get(pair.row_name).copied() {
                Some(SourceRow::Objective) => c[variable] += pair.value,
                Some(SourceRow::Constraint(row)) => {
                    constraints[row].coefficients.push((variable, pair.value));
                }
                Some(SourceRow::Ignored) => {}
                None => return Err(MpsError::UnknownRow(pair.row_name.to_string())),
            }
        }
    }
    for (index, value) in c.iter().enumerate() {
        finite("COLUMNS", &format!("objective coefficient {index}"), *value)?;
    }
    Ok(())
}

/// Apply a named section's values once per constraint, ignoring N rows.
fn collect_row_values(
    section: &'static str,
    lines: Option<&[WideLine<'_, f64>]>,
    row_lookup: &HashMap<String, SourceRow>,
    constraints: &mut [SourceConstraint],
    options: MpsOptions,
    mut assign: impl FnMut(&mut SourceConstraint, f64),
) -> Result<(), MpsError> {
    let Some(lines) = lines else {
        return Ok(());
    };
    let mut assigned = vec![false; constraints.len()];
    for line in wide_set(section, lines, options)? {
        for pair in line_pairs(line) {
            finite(section, pair.row_name, pair.value)?;
            match row_lookup.get(pair.row_name).copied() {
                Some(SourceRow::Constraint(row)) => {
                    if std::mem::replace(&mut assigned[row], true) {
                        return Err(MpsError::DuplicateValue {
                            section,
                            name: pair.row_name.to_string(),
                        });
                    }
                    assign(&mut constraints[row], pair.value);
                }
                Some(SourceRow::Objective | SourceRow::Ignored) => {}
                None => return Err(MpsError::UnknownRow(pair.row_name.to_string())),
            }
        }
    }
    Ok(())
}

/// The lines of a wide section's first named set; several sets are an error
/// unless the options keep the first.
fn wide_set<'l, 'a>(
    section: &'static str,
    lines: &'l [WideLine<'a, f64>],
    options: MpsOptions,
) -> Result<impl Iterator<Item = &'l WideLine<'a, f64>>, MpsError> {
    let first = lines.first().map(|line| line.name);
    if !options.first_named_set && lines.iter().any(|line| Some(line.name) != first) {
        return Err(MpsError::MultipleSets { section });
    }
    Ok(lines.iter().filter(move |line| Some(line.name) == first))
}

fn collect_bounds(
    parsed: &Parser<'_, f64>,
    variable_names: &[&str],
    variable_lookup: &HashMap<&str, usize>,
    options: MpsOptions,
) -> Result<Vec<VariableBounds>, MpsError> {
    let mut bounds = vec![
        VariableBounds {
            lower: 0.0,
            upper: f64::INFINITY
        };
        variable_names.len()
    ];
    let Some(lines) = &parsed.bounds else {
        return Ok(bounds);
    };
    let first = lines.first().map(|bound| bound.bound_name);
    if !options.first_named_set && lines.iter().any(|bound| Some(bound.bound_name) != first) {
        return Err(MpsError::MultipleSets { section: "BOUNDS" });
    }
    for bound in lines.iter().filter(|bound| Some(bound.bound_name) == first) {
        let Some(&variable) = variable_lookup.get(bound.column_name) else {
            return Err(MpsError::UnknownVariable(bound.column_name.to_string()));
        };
        let current = &mut bounds[variable];
        match bound.bound_type {
            BoundType::Lo => current.lower = bound_value(bound, "LO")?,
            BoundType::Up => current.upper = bound_value(bound, "UP")?,
            BoundType::Fx => {
                let value = bound_value(bound, "FX")?;
                current.lower = value;
                current.upper = value;
            }
            BoundType::Fr => {
                current.lower = f64::NEG_INFINITY;
                current.upper = f64::INFINITY;
            }
            BoundType::Mi => {
                current.lower = f64::NEG_INFINITY;
                current.upper = 0.0;
            }
            BoundType::Pl => {
                current.lower = 0.0;
                current.upper = f64::INFINITY;
            }
            BoundType::Bv => return unsupported_bound(bound, "BV"),
            BoundType::Li => return unsupported_bound(bound, "LI"),
            BoundType::Ui => return unsupported_bound(bound, "UI"),
            BoundType::Sc => return unsupported_bound(bound, "SC"),
        }
    }
    for (name, bound) in variable_names.iter().zip(&bounds) {
        if bound.lower > bound.upper {
            return Err(MpsError::InconsistentBounds {
                variable: (*name).to_string(),
                lower: bound.lower,
                upper: bound.upper,
            });
        }
    }
    Ok(bounds)
}

fn bound_value(
    bound: &::mps::types::BoundsLine<'_, f64>,
    bound_type: &'static str,
) -> Result<f64, MpsError> {
    let Some(value) = bound.value.filter(|value| value.is_finite()) else {
        return Err(MpsError::MissingBoundValue {
            bound_type,
            variable: bound.column_name.to_string(),
        });
    };
    Ok(value)
}

fn unsupported_bound<T>(
    bound: &::mps::types::BoundsLine<'_, f64>,
    bound_type: &'static str,
) -> Result<T, MpsError> {
    Err(MpsError::UnsupportedBoundType {
        bound_type,
        variable: bound.column_name.to_string(),
    })
}

fn collect_quadratic_objective(
    parsed: &Parser<'_, f64>,
    variable_lookup: &HashMap<&str, usize>,
    objective_sign: f64,
) -> Result<Option<CscMatrix>, MpsError> {
    let Some(terms) = &parsed.quadratic_objective else {
        return Ok(None);
    };
    let mut pairs: HashMap<(usize, usize), QuadraticPair> = HashMap::new();
    for term in terms {
        finite(
            "quadratic objective",
            &format!("{} * {}", term.var1, term.var2),
            term.coefficient,
        )?;
        let Some(&first) = variable_lookup.get(term.var1) else {
            return Err(MpsError::UnknownVariable(term.var1.to_string()));
        };
        let Some(&second) = variable_lookup.get(term.var2) else {
            return Err(MpsError::UnknownVariable(term.var2.to_string()));
        };
        let (key, forward) = if first <= second {
            ((first, second), true)
        } else {
            ((second, first), false)
        };
        let pair = pairs.entry(key).or_default();
        if forward {
            pair.forward += term.coefficient;
            pair.has_forward = true;
        } else {
            pair.reverse += term.coefficient;
            pair.has_reverse = true;
        }
    }

    let mut rows = Vec::new();
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for ((first, second), pair) in pairs {
        let value = if first == second || !pair.has_reverse {
            pair.forward
        } else if !pair.has_forward {
            pair.reverse
        } else {
            let scale = 1.0 + pair.forward.abs().max(pair.reverse.abs());
            if (pair.forward - pair.reverse).abs() > 1e-12 * scale {
                return Err(MpsError::AsymmetricQuadraticEntry {
                    first: variable_name(variable_lookup, first).to_string(),
                    second: variable_name(variable_lookup, second).to_string(),
                    forward: pair.forward,
                    reverse: pair.reverse,
                });
            }
            0.5 * (pair.forward + pair.reverse)
        } * objective_sign;
        rows.push(first);
        columns.push(second);
        values.push(value);
    }
    CscMatrix::from_triplets(
        variable_lookup.len(),
        variable_lookup.len(),
        rows,
        columns,
        values,
    )
    .map(Some)
    .map_err(Into::into)
}

fn variable_name<'a>(lookup: &'a HashMap<&str, usize>, index: usize) -> &'a str {
    lookup
        .iter()
        .find_map(|(name, &candidate)| (candidate == index).then_some(*name))
        .expect("quadratic variable indices originate in the lookup")
}

fn canonical_rows(
    constraints: Vec<SourceConstraint>,
    bounds: Vec<VariableBounds>,
) -> Result<(Vec<CanonicalRow>, Vec<CanonicalRow>), MpsError> {
    let mut zero = Vec::new();
    let mut nonnegative = Vec::new();
    for constraint in constraints {
        let (lower, upper) = row_interval(&constraint)?;
        append_interval(
            constraint.coefficients,
            lower,
            upper,
            &mut zero,
            &mut nonnegative,
        );
    }
    for (variable, bound) in bounds.into_iter().enumerate() {
        append_interval(
            vec![(variable, 1.0)],
            bound.lower,
            bound.upper,
            &mut zero,
            &mut nonnegative,
        );
    }
    Ok((zero, nonnegative))
}

fn row_interval(constraint: &SourceConstraint) -> Result<(f64, f64), MpsError> {
    let rhs = constraint.rhs;
    let (lower, upper) = match (constraint.sense, constraint.range) {
        (ConstraintSense::Equal, None | Some(0.0)) => (rhs, rhs),
        (ConstraintSense::Equal, Some(range)) if range > 0.0 => (rhs, rhs + range),
        (ConstraintSense::Equal, Some(range)) => (rhs + range, rhs),
        (ConstraintSense::Less, None) => (f64::NEG_INFINITY, rhs),
        (ConstraintSense::Less, Some(range)) => (rhs - range.abs(), rhs),
        (ConstraintSense::Greater, None) => (rhs, f64::INFINITY),
        (ConstraintSense::Greater, Some(range)) => (rhs, rhs + range.abs()),
    };
    if constraint.range.is_some() && (!lower.is_finite() || !upper.is_finite()) {
        return Err(MpsError::NonFiniteValue {
            section: "RANGES",
            name: constraint.name.clone(),
        });
    }
    Ok((lower, upper))
}

fn append_interval(
    coefficients: Vec<(usize, f64)>,
    lower: f64,
    upper: f64,
    zero: &mut Vec<CanonicalRow>,
    nonnegative: &mut Vec<CanonicalRow>,
) {
    if lower == upper {
        zero.push(CanonicalRow {
            coefficients,
            rhs: upper,
        });
        return;
    }
    if lower.is_finite() {
        nonnegative.push(CanonicalRow {
            coefficients: coefficients
                .iter()
                .map(|&(column, value)| (column, -value))
                .collect(),
            rhs: -lower,
        });
    }
    if upper.is_finite() {
        nonnegative.push(CanonicalRow {
            coefficients,
            rhs: upper,
        });
    }
}

fn line_pairs<'a>(
    line: &'a WideLine<'a, f64>,
) -> impl Iterator<Item = &'a ::mps::types::RowValuePair<'a, f64>> {
    std::iter::once(&line.first_pair).chain(line.second_pair.as_ref())
}

fn finite(section: &'static str, name: &str, value: f64) -> Result<(), MpsError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(MpsError::NonFiniteValue {
            section,
            name: name.to_string(),
        })
    }
}
