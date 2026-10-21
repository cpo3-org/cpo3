use std::{env, io::IsTerminal, time::Duration};

use crate::{LinearSolver, Problem, Settings, SolverStatus};

const FRAME_WIDTH: usize = 62;
const TABLE_LABELS: [&str; 13] = [
    "iter",
    "time",
    "mu_raw",
    "rho",
    "tau",
    "sigma",
    "alpha_aff",
    "alpha",
    "pobj",
    "dobj",
    "pres",
    "dres",
    "comp",
];
const TABLE_WIDTHS: [usize; 13] = [5, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10];

#[derive(Clone, Copy, Debug)]
pub(crate) struct IterationRecord {
    pub iteration: usize,
    pub elapsed: Duration,
    pub mu: f64,
    pub rho: Option<f64>,
    pub tau: Option<f64>,
    pub sigma: Option<f64>,
    pub alpha_aff: Option<f64>,
    pub alpha: Option<f64>,
    pub pobj: Option<f64>,
    pub dobj: Option<f64>,
    pub pres: Option<f64>,
    pub dres: Option<f64>,
    pub comp: Option<f64>,
}

pub(crate) struct ProgressReporter {
    enabled: bool,
    unicode: bool,
    last_iteration: Option<usize>,
}

impl ProgressReporter {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            unicode: enabled && std::io::stderr().is_terminal() && locale_supports_unicode(),
            last_iteration: None,
        }
    }

    pub(crate) fn header(
        &self,
        problem: &Problem,
        settings: &Settings,
        linear_solver: LinearSolver,
        presolve: Option<(usize, usize)>,
    ) {
        if !self.enabled {
            return;
        }

        for line in banner(self.unicode) {
            eprintln!("{line}");
        }
        eprintln!();
        eprintln!(
            "  problem        {} variables {} {} constraints",
            problem.variable_count(),
            separator(self.unicode),
            problem.constraint_count(),
        );
        match presolve {
            Some((rows, columns)) => eprintln!(
                "  presolve       removed {rows} constraints {} {columns} variables",
                separator(self.unicode),
            ),
            None => eprintln!("  presolve       off"),
        }
        eprintln!("  linear solver  {linear_solver}");
        eprintln!("  scaling passes {}", settings.ruiz_iters);
        eprintln!("  cvxopt init    {}", settings.cvxopt_init);
        eprintln!("  KKT bound elimination {}", settings.eliminate_bounds);
        eprintln!("  initial diagonal floor {:.1e}", settings.refine_init_reg);
        eprintln!("  force warm start {}", settings.force_warm_start);
        eprintln!(
            "  tolerances     feas {:.1e} {} gap abs {:.1e} {} gap rel {:.1e}",
            settings.tol_feas,
            separator(self.unicode),
            settings.tol_gap_abs,
            separator(self.unicode),
            settings.tol_gap_rel,
        );
        eprintln!("  limits         {} iterations", settings.max_iters);
        eprintln!();
        eprintln!("{}", table_header(self.unicode));
        eprintln!("{}", table_rule(self.unicode));
    }

    pub(crate) fn wants_iteration(&self, iteration: usize) -> bool {
        self.enabled && (iteration <= 10 || iteration.is_multiple_of(25))
    }

    pub(crate) fn iteration(&mut self, record: IterationRecord, force: bool) {
        if !self.enabled
            || self.last_iteration == Some(record.iteration)
            || (!force && !self.wants_iteration(record.iteration))
        {
            return;
        }
        eprintln!("{}", format_iteration(record, self.unicode));
        self.last_iteration = Some(record.iteration);
    }

    pub(crate) fn finish(
        &self,
        status: SolverStatus,
        iteration: usize,
        elapsed: Duration,
        objective: f64,
    ) {
        if !self.enabled {
            return;
        }
        eprintln!();
        let details = format!(
            "{iteration} iterations {} {:.2}s {} objective {objective:.6e}",
            separator(self.unicode),
            elapsed.as_secs_f64(),
            separator(self.unicode),
        );
        for line in status_box(status_title(status), &[details], self.unicode) {
            eprintln!("{line}");
        }
    }

    pub(crate) fn failure(&self, iteration: usize, message: &str) {
        if !self.enabled {
            return;
        }
        eprintln!();
        let details = [format!("iteration {iteration}: {message}")];
        for line in status_box("numerical error", &details, self.unicode) {
            eprintln!("{line}");
        }
    }
}

fn locale_supports_unicode() -> bool {
    for name in ["LC_ALL", "LC_CTYPE", "LANG"] {
        if let Some(value) = env::var_os(name) {
            let locale = value.to_string_lossy().to_ascii_lowercase();
            return locale.contains("utf-8") || locale.contains("utf8");
        }
    }
    true
}

fn separator(unicode: bool) -> &'static str {
    if unicode { "·" } else { "|" }
}

fn status_title(status: SolverStatus) -> &'static str {
    match status {
        SolverStatus::Solved => "solved",
        SolverStatus::PrimalInfeasible => "primal infeasible",
        SolverStatus::DualInfeasible => "dual infeasible",
        SolverStatus::MaxIterations => "maximum iterations",
        SolverStatus::TimeLimit => "time limit",
    }
}

fn banner(unicode: bool) -> Vec<String> {
    let mut lines = frame(
        &[
            "cpo3  ·  Conic Quadratic Optimization".to_string(),
            format!("v{}", env!("CARGO_PKG_VERSION")),
        ],
        unicode,
    );
    if !unicode {
        lines[1] = lines[1].replace('·', "|");
    }
    lines
}

fn status_box(title: &str, details: &[String], unicode: bool) -> Vec<String> {
    let mut contents = vec![title.to_string()];
    contents.extend(details.iter().cloned());
    frame(&contents, unicode)
}

fn frame(contents: &[String], unicode: bool) -> Vec<String> {
    let (top_left, horizontal, top_right, vertical, bottom_left, bottom_right) = if unicode {
        ('╭', '─', '╮', '│', '╰', '╯')
    } else {
        ('+', '-', '+', '|', '+', '+')
    };
    let mut lines = Vec::with_capacity(contents.len() + 2);
    lines.push(format!(
        "{top_left}{}{top_right}",
        horizontal.to_string().repeat(FRAME_WIDTH + 2)
    ));
    for content in contents {
        let clipped = clip(content, FRAME_WIDTH, unicode);
        let padding = FRAME_WIDTH - clipped.chars().count();
        lines.push(format!(
            "{vertical} {clipped}{}{vertical}",
            " ".repeat(padding + 1)
        ));
    }
    lines.push(format!(
        "{bottom_left}{}{bottom_right}",
        horizontal.to_string().repeat(FRAME_WIDTH + 2)
    ));
    lines
}

fn clip(value: &str, width: usize, unicode: bool) -> String {
    if value.chars().count() <= width {
        return value.to_string();
    }
    if unicode {
        value.chars().take(width - 1).collect::<String>() + "…"
    } else {
        value.chars().take(width - 3).collect::<String>() + "..."
    }
}

fn table_header(unicode: bool) -> String {
    let separator = if unicode { " │ " } else { " | " };
    TABLE_LABELS
        .iter()
        .zip(TABLE_WIDTHS)
        .map(|(label, width)| format!("{label:^width$}"))
        .collect::<Vec<_>>()
        .join(separator)
}

fn table_rule(unicode: bool) -> String {
    let horizontal = if unicode { '─' } else { '-' };
    let intersection = if unicode { "┼" } else { "+" };
    let last = TABLE_WIDTHS.len() - 1;
    TABLE_WIDTHS
        .iter()
        .enumerate()
        .map(|(index, width)| {
            let margin = if index == 0 || index == last { 1 } else { 2 };
            horizontal.to_string().repeat(width + margin)
        })
        .collect::<Vec<_>>()
        .join(intersection)
}

fn format_iteration(record: IterationRecord, unicode: bool) -> String {
    let separator = if unicode { '│' } else { '|' };
    let missing = if unicode { "—" } else { "-" };
    let tau = optional_number(record.tau, missing);
    let sigma = optional_number(record.sigma, missing);
    let alpha_aff = optional_number(record.alpha_aff, missing);
    let alpha = optional_number(record.alpha, missing);
    format!(
        "{iteration:>5} {separator} {elapsed:>6.2}s {separator} {mu:>10.3e} {separator} {rho} {separator} {tau} {separator} {sigma} {separator} {alpha_aff} {separator} {alpha} {separator} {pobj} {separator} {dobj} {separator} {pres} {separator} {dres} {separator} {comp}",
        iteration = record.iteration,
        elapsed = record.elapsed.as_secs_f64(),
        mu = record.mu,
        rho = optional_number(record.rho, missing),
        pobj = optional_number(record.pobj, missing),
        dobj = optional_number(record.dobj, missing),
        pres = optional_number(record.pres, missing),
        dres = optional_number(record.dres, missing),
        comp = optional_number(record.comp, missing),
    )
}

fn optional_number(value: Option<f64>, missing: &str) -> String {
    value.map_or_else(
        || format!("{missing:>10}"),
        |value| format!("{value:>10.3e}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_uses_the_minimal_signature() {
        let banner = banner(true).join("\n");
        assert!(banner.contains("cpo3  ·  Conic Quadratic Optimization"));
        assert!(!banner.contains("Matteo Santamaria"));
        assert!(!banner.contains("Written by"));
    }

    #[test]
    fn cadence_is_dense_at_first_and_periodic_afterward() {
        let reporter = ProgressReporter {
            enabled: true,
            unicode: true,
            last_iteration: None,
        };
        for iteration in 0..=10 {
            assert!(reporter.wants_iteration(iteration));
        }
        assert!(!reporter.wants_iteration(11));
        assert!(reporter.wants_iteration(25));
        assert!(!reporter.wants_iteration(26));
    }

    #[test]
    fn redirected_output_has_an_ascii_fallback() {
        let banner = banner(false).join("\n");
        assert!(banner.starts_with('+'));
        assert!(banner.contains("cpo3  |  Conic Quadratic Optimization"));
        assert!(!banner.contains('╭'));
    }

    #[test]
    fn iteration_rows_include_both_objectives() {
        let row = format_iteration(
            IterationRecord {
                iteration: 1,
                elapsed: Duration::ZERO,
                mu: 1.0,
                rho: Some(1.0),
                tau: Some(1.0),
                sigma: Some(0.1),
                alpha_aff: Some(0.2),
                alpha: Some(0.3),
                pobj: Some(1.25),
                dobj: Some(-2.5),
                pres: Some(0.0),
                dres: Some(0.0),
                comp: Some(0.0),
            },
            false,
        );
        let primal = row.find("1.250e0").unwrap();
        let dual = row.find("-2.500e0").unwrap();
        assert!(primal < dual);
        assert!(table_header(false).contains("pobj    |    dobj"));
    }

    #[test]
    fn table_columns_are_aligned() {
        let row = format_iteration(
            IterationRecord {
                iteration: 1,
                elapsed: Duration::ZERO,
                mu: 1.0,
                rho: Some(1.0),
                tau: Some(1.0),
                sigma: Some(0.1),
                alpha_aff: Some(0.2),
                alpha: Some(0.3),
                pobj: Some(1.25),
                dobj: Some(-2.5),
                pres: Some(0.0),
                dres: Some(0.0),
                comp: Some(0.0),
            },
            false,
        );
        let header = table_header(false);
        let rule = table_rule(false);
        let separators = |line: &str, separator| {
            line.char_indices()
                .filter_map(|(index, character)| (character == separator).then_some(index))
                .collect::<Vec<_>>()
        };

        assert_eq!(row.len(), header.len());
        assert_eq!(rule.len(), header.len());
        assert_eq!(separators(&row, '|'), separators(&header, '|'));
        assert_eq!(separators(&rule, '+'), separators(&header, '|'));
    }
}
