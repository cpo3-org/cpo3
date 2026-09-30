use std::fmt;

/// Linear-system implementation used by the numerical solver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LinearSolver {
    #[default]
    Faer,
    Feral,
}

impl LinearSolver {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Faer => "faer",
            Self::Feral => "feral",
        }
    }
}

impl fmt::Display for LinearSolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}
