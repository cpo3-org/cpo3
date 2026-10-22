use std::{env, io::IsTerminal, time::Duration};

use tabled::{
    settings::{Alignment, Padding, Style},
    tables::CompactTable,
};

use crate::{Problem, Settings, SolverStatus};

const TABLE_LABELS: [&str; 12] = [
    "iter",
    "time",
    "primal obj",
    "dual obj",
    "primal res",
    "dual res",
    "comp",
    "step",
    "mu",
    "tau",
    "sigma",
    "aff step",
];
const TABLE_WIDTHS: [usize; 12] = [5, 8, 14, 14, 12, 12, 12, 8, 12, 12, 12, 12];

#[derive(Clone, Copy, Debug)]
pub(crate) struct IterationRecord {
    pub iteration: usize,
    pub elapsed: Duration,
    pub mu: f64,
    pub tau: f64,
    pub sigma: Option<f64>,
    pub alpha_aff: Option<f64>,
    pub alpha: Option<f64>,
    pub pobj: f64,
    pub dobj: f64,
    pub pres: f64,
    pub dres: f64,
    pub comp: f64,
}

pub(crate) struct ProgressReporter {
    enabled: bool,
    unicode: bool,
    details: bool,
    last_iteration: Option<usize>,
}

impl ProgressReporter {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            unicode: enabled && std::io::stderr().is_terminal() && locale_supports_unicode(),
            details: enabled && env::var("CPO3_PROGRESS_DETAILS").is_ok_and(|value| value == "1"),
            last_iteration: None,
        }
    }

    pub(crate) fn header(
        &self,
        problem: &Problem,
        settings: &Settings,
        presolve: Option<(usize, usize)>,
    ) {
        if !self.enabled {
            return;
        }
        eprintln!("{}\n", banner(self.unicode));
        let separator = self.separator();
        let (m, n) = problem.dims();
        let p_nnz = problem.p().map_or(0, |p| {
            let diagonal = p
                .column_pointers()
                .windows(2)
                .enumerate()
                .filter(|(column, range)| {
                    p.row_indices()[range[0]..range[1]].last() == Some(column)
                })
                .count();
            2 * p.values().len() - diagonal
        });
        let a_nnz = problem.a().values().len();
        let fill = |nnz: usize, slots: f64| {
            if slots == 0.0 {
                0.0
            } else {
                100.0 * nnz as f64 / slots
            }
        };
        eprintln!("Problem   {n} variables {separator} {m} constraints");
        eprintln!(
            "          nnz(P) = {p_nnz} {separator} fill = {:.2}%",
            fill(p_nnz, n as f64 * n as f64),
        );
        eprintln!(
            "          nnz(A) = {a_nnz} {separator} fill = {:.2}%",
            fill(a_nnz, m as f64 * n as f64),
        );
        eprintln!(
            "Solver    {} {separator} presolve {} {separator} {} Ruiz passes",
            settings.linear_solver,
            if settings.presolve { "on" } else { "off" },
            settings.ruiz_iters,
        );
        if let Some((rows, columns)) = presolve {
            eprintln!("Presolve  removed {rows} constraints {separator} {columns} variables");
        }
        eprintln!(
            "Accuracy  feasibility {:e} {separator} absolute gap {:e} {separator} relative gap {:e}",
            settings.tol_feas, settings.tol_gap_abs, settings.tol_gap_rel,
        );
        match settings.time_limit {
            Some(limit) => eprintln!(
                "Limits    {} iterations {separator} {:.2} s",
                settings.max_iters,
                limit.as_secs_f64(),
            ),
            None => eprintln!("Limits    {} iterations", settings.max_iters),
        }
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
        if self.last_iteration.is_none() {
            eprintln!("\n{}", self.table_row(self.labels(), true));
        }
        eprintln!("{}", self.format_iteration(record));
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
        eprint!(
            "\n{} {} {iteration} iterations {} {:.2} s",
            status_title(status),
            self.separator(),
            self.separator(),
            elapsed.as_secs_f64(),
        );
        // Presolve can finish before there is an iteration row with an objective.
        if self.last_iteration.is_none() && status == SolverStatus::Solved {
            eprint!(" {} objective {objective:.6e}", self.separator());
        }
        eprintln!();
    }

    pub(crate) fn failure(&self, iteration: usize, message: &str) {
        if self.enabled {
            eprintln!("\nNumerical error at iteration {iteration}: {message}");
        }
    }

    fn separator(&self) -> &'static str {
        if self.unicode { "·" } else { "|" }
    }

    fn labels(&self) -> Vec<String> {
        TABLE_LABELS[..if self.details { 12 } else { 8 }]
            .iter()
            .map(|label| (*label).into())
            .collect()
    }

    fn table_row(&self, cells: Vec<String>, header: bool) -> String {
        let columns = cells.len();
        let table = CompactTable::new([cells])
            .columns(columns)
            .width(TABLE_WIDTHS)
            .with(Padding::zero())
            .with(Alignment::right())
            .with(Style::blank());
        if header {
            table
                .with(
                    Style::blank()
                        .bottom(if self.unicode { '─' } else { '-' })
                        .intersection_bottom(' '),
                )
                .to_string()
        } else {
            table.to_string()
        }
    }

    fn format_iteration(&self, record: IterationRecord) -> String {
        let optional_number = |value: Option<f64>| {
            value.map_or_else(
                || if self.unicode { "—" } else { "-" }.into(),
                |value| format!("{value:.3e}"),
            )
        };
        let mut cells = vec![
            record.iteration.to_string(),
            format!("{:.2}s", record.elapsed.as_secs_f64()),
            format!("{:.6e}", record.pobj),
            format!("{:.6e}", record.dobj),
            format!("{:.3e}", record.pres),
            format!("{:.3e}", record.dres),
            format!("{:.3e}", record.comp),
            record.alpha.map_or_else(
                || if self.unicode { "—" } else { "-" }.into(),
                |value| format!("{value:.3}"),
            ),
        ];
        if self.details {
            cells.extend([
                format!("{:.3e}", record.mu),
                format!("{:.3e}", record.tau),
                optional_number(record.sigma),
                optional_number(record.alpha_aff),
            ]);
        }
        self.table_row(cells, false)
    }
}

fn banner(unicode: bool) -> String {
    format!(
        "cpo3 v{} {} Conic Quadratic Optimization\n{}",
        env!("CARGO_PKG_VERSION"),
        if unicode { "·" } else { "|" },
        env!("CARGO_PKG_AUTHORS").replace(':', ", "),
    )
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

fn status_title(status: SolverStatus) -> &'static str {
    match status {
        SolverStatus::Solved => "Solved",
        SolverStatus::PrimalInfeasible => "Primal infeasible",
        SolverStatus::DualInfeasible => "Dual infeasible",
        SolverStatus::MaxIterations => "Maximum iterations",
        SolverStatus::TimeLimit => "Time limit",
    }
}
