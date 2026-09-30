# cpo3

`cpo3` is an interior-point solver for convex quadratic and linear programs,
implemented in Rust with a Python interface. It solves

$$
\min_{x,s}\; \tfrac12 x^\top P x + c^\top x
\quad\text{subject to}\quad Ax+s=b,\quad s\in\mathcal K,
$$

where $P$ is positive semidefinite. Currently, $\mathcal K$ supports zero and
nonnegative cone blocks, representing linear equalities and inequalities.

`cpo3` is under active development. Backward compatibility is not guaranteed;
the API and solver behavior may change between releases.

## Install from source

Requires Rust 1.89 or later, Python 3.10 or later, and [uv](https://docs.astral.sh/uv/).
From the repository root:

```console
uv sync
```

Run Python examples with `uv run python` to use this environment.

## Python example

```python
import cpo3
import numpy as np
from scipy import sparse

problem = cpo3.Problem(
    P=sparse.eye(2, format="csc"),
    c=np.array([-1.0, -2.0]),
    A=sparse.eye(2, format="csc"),
    b=np.array([3.0, 4.0]),
    cones=[cpo3.NonnegativeCone(2)],
)
solution = cpo3.solve(problem)
print(solution.status)
print(solution.x)  # approximately [1.0, 2.0]
```

Use SciPy sparse matrices for `A` and `P`; set `P=None` for a linear program.
Options can be passed with `settings=cpo3.Settings(...)`.

## Rust example

```rust
use cpo3::{Cone, CscMatrix, Problem, Settings, solve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = Problem::new(
        Some(CscMatrix::identity(2)?),
        vec![-1.0, -2.0],
        CscMatrix::identity(2)?,
        vec![3.0, 4.0],
        vec![Cone::Nonnegative(2)],
    )?;
    let solution = solve(&problem, &Settings::default())?;
    println!("{:?}: {:?}", solution.status(), solution.x());
    Ok(())
}
```

## Development

Run the Rust and Python tests:

```console
cargo test --features mps
uv run pytest
```

Format, lint, and type-check:

```console
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

See [benchmarks/README.md](benchmarks/README.md) for the included benchmark data.

## Authors and license

Matteo Santamaria, Michael Cummins, and Brendan O'Donoghue.

Licensed under [MIT](LICENSE).
