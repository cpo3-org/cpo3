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

Requires Rust 1.91 or later, Python 3.10 or later, and [uv](https://docs.astral.sh/uv/).
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

`initial_mu` sets the canonical cold-start complementarity beta
(default `1e-3`, range `(0, 1]`). Inequality slacks and multipliers start at
sqrt(beta), and tau is adjusted to preserve the embedding sphere radius.
For example, `cpo3.Settings(initial_mu=0.01)` starts with
inequality slacks and multipliers `0.1`. Accepted warm starts and CVXOPT starts are
unaffected; a failed CVXOPT start falls back to this configured cold start.

The default `step_fraction` is `0.999`. KKT solves use an adaptive diagonal
floor, initially `refine_init_reg=1e-12`, applied before singleton
elimination. Refinement always targets the original system at the current
complementarity. Numerical failures increase the floor and retry; accepted
steps increase or decrease it according to the dominant source of refinement
error. Accurate steps relax it, and a retry prevents immediate relaxation.
Faer pivot replacement is disabled.

Refinement allows up to `refine_iters=30` corrections, with tolerance
`1e-7`. It retains the best finite iterate when it cannot reach tolerance.
The existing forecast can stop earlier: after two corrections, it projects
whether the recent contraction rate can reach tolerance within the remaining
budget. It stops when this projection misses tolerance and the per-correction
rate exceeds `1/sqrt(2)`.

Work counters are available on `solution.info`:

```python
info = solution.info
print(info.iters, info.factorizations)
print(info.primary_solves, info.refinement_solves)
total_linear_solves = info.primary_solves + info.refinement_solves
```

`primary_solves` counts the initial solve for each KKT right-hand side,
throughout initialization and the outer iterations. `refinement_solves`
counts additional correction solves using the existing factorization. Both count
attempts, including internally failed attempts; residual evaluations are excluded.

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

## Connection to the paper

The numerical iteration follows the modified Mehrotra algorithm (`alg:ipm`).
Equation labels below refer to the paper source, `mpc_main.tex`.

| Code | Mathematical role |
| --- | --- |
| `src/embedding.rs` | Homogeneous state, complementarity, and spherical normalization (`eq:iterate_normalisation`). |
| `src/init.rs` | Canonical or CVXOPT cold starts and local-norm warm-start screening (`subsec:warm-start`). |
| `src/mehrotra.rs` | Predictor, centering, second-order corrector, and damped boundary step (`subsec:mehrotra`). |
| `src/josephy.rs` | Surrogate right-hand sides, cached data solve, scalar quadratic, and reconstruction (`eq:surrogate_rhs`, `eq:cached_data_solve`, `thm:resolvent`). |
| `src/josephy/kkt.rs` | Sparse quasi-definite system (`eq:qd_system`), singleton Schur complement, and full-system refinement. |
| `src/linear_system/` | Faer and Feral factorization adapters. |
| `src/solution.rs` | Preparation, outer loop, original-unit stopping tests, certificates, and recovery (`subsec:termination`). |

The implemented centering rule is `sigma = (1 - alpha_affine)^3`; the paper's
normalized affine-complementarity formula is awaiting revision. `mu` denotes
current complementarity or a target, while `rho` denotes the true KKT shift.
Factor-diagonal clamping is distinct from `rho`: refinement targets the full
system with shift `rho`. The resolvent uses increment equations to preserve
small updates when point-coordinate terms would cancel.

Each ordinary iteration shares one factorization and three primary solves:
the constant data column, predictor, and corrector. Numerical retries and
refinement can add work. Spherical normalization, shift retries, the homogeneous
root fallback, and CVXOPT initialization remain supported. Governor recentering
and additional Gondzio centrality correctors are not part of this iteration.

`SolutionInfo.warm_start_score` reports the score used to accept or reject the
best warm candidate (`None` if no finite candidate was screened). The former
best-level initialization diagnostics and governor/MCC settings and counters
have been removed from both the Rust and Python interfaces.

## Authors and license

Matteo Santamaria, Michael Cummins, and Brendan O'Donoghue.

Licensed under [MIT](LICENSE).
