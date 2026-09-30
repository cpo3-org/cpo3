//! Diagonal equilibration choices.

use std::fmt::{Display, Formatter};

/// Diagonal equilibration method for problem data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Equilibration {
    /// Do not apply equilibration.
    None,
    /// Apply symmetric Ruiz infinity-norm equilibration.
    #[default]
    Ruiz,
}

impl Equilibration {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ruiz => "ruiz",
        }
    }
}

impl Display for Equilibration {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}
