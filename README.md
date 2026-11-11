# cpo3

cpo3 is a sparse interior-point solver for convex linear and quadratic programs,
written in Rust with a Python interface. It solves

$$
\min_{x,s} \tfrac12 x^\top P x + c^\top x
\quad\text{subject to}\quad Ax+s=b,\quad s\in\mathcal K,
$$

where $P$ is symmetric positive semidefinite. The supported cone blocks are
zero cones for equalities and nonnegative cones for inequalities. Thus,
nonnegative-cone rows represent $Ax\le b$; variable bounds can be written as
additional constraint rows.

- **Solutions and certificates:** recover primal and dual solutions or
  certificates of primal or dual infeasibility.
- **Warm starts:** supply a previous solution when solving related problems.
- **Sparse linear algebra:** Faer, Apple Accelerate, SuiteSparse CHOLMOD, and Intel MKL PARDISO backends, with
  presolve and Ruiz equilibration.
- **File loading:** read continuous LPs and QPs from MPS and QPS files.

The method uses a regularized homogeneous embedding with a canonical central
path. Its predictor–corrector steps retain the nonlinear embedding equations
while linearizing complementarity.

The name **cpo3** combines *Conic Programming* with **O3**, a double reference to
the compiler optimization flag `-O3` and the trioxide in rust—a nod to its
implementation in Rust.

cpo3 is under active development. The API and solver behavior may change between
releases.

## Installation

cpo3 is not yet published on PyPI or crates.io. Install it from source using
the instructions below.

To build the Python interface from source, install Rust 1.91 or later,
Python 3.10 or later, and [uv](https://docs.astral.sh/uv/), then run:

```sh
git clone https://github.com/cpo3-org/cpo3.git
cd cpo3
uv sync
```

Use `uv run python` from this checkout to run Python with the built package.

For a Rust application, add the Git dependency to `Cargo.toml`:

```toml
[dependencies]
cpo3 = { git = "https://github.com/cpo3-org/cpo3.git" }
```

## Optional factorization backends

Faer is the default and requires no external numerical runtime. The `accelerate`,
`cholmod`, and `pardiso` Cargo features enable additional backends. All retain symbolic
analysis across diagonal updates and solve paired right-hand sides together.
Equality-only problems continue to use Faer for their LDL/LSMR path.

For Python development, build and install the selected backend into the checkout's
environment. For example, on macOS:

```sh
uv sync --config-setting "build-args=--features=python,mps,accelerate"
uv run --no-sync python
```

Use `python,mps,cholmod` for CHOLMOD or `python,mps,pardiso` for PARDISO, or combine
the features. Install the requirements below before building. Run subsequent
Python commands with `uv run --no-sync` so uv does not replace the selected
build with the default configuration. Repeat the configured `uv sync` command
after changing Rust code to rebuild the extension.

Rust applications can enable the corresponding features in their Cargo dependency.

- **Accelerate:** requires macOS 13 or later and the Apple SDK at build time.
  Select `LinearSolver.ACCELERATE` with `threads=0`. Positive thread caps are
  rejected: Apple's sparse solver has no per-solve threading control. Its
  process-wide `VECLIB_MAXIMUM_THREADS` setting must be configured before startup
  if needed; cpo3 never modifies it.
- **CHOLMOD:** loads a separately installed SuiteSparse runtime when first
  requested. With a wheel that includes the `cholmod` feature, install the
  runtime with `brew install suite-sparse` on macOS,
  `sudo apt-get install libcholmod5` on Ubuntu 24.04, or
  `conda install -c conda-forge suitesparse` in a Conda environment.
  No rebuild is needed. The loader searches the system library path, standard
  Homebrew/MacPorts locations, and the active `CONDA_PREFIX`.
  Set `CPO3_CHOLMOD` to the full shared-library path before starting Python to
  override discovery; the library's dependencies must also be discoverable by
  the operating system. Availability is cached for the process lifetime.
  Supported CHOLMOD versions are 5.2.x and 5.3.x; other versions are rejected
  before accessing version-dependent structs. A missing or incompatible runtime
  disables only CHOLMOD, so importing cpo3 and using Faer still works.
  Select `LinearSolver.CHOLMOD`. The adapter uses simplicial LDLᵀ, since CHOLMOD's
  supernodal LLᵀ cannot factor our indefinite KKT systems. It does not pivot;
  failed factorizations and incorrect inertia use cpo3's existing recovery path.
  Positive `threads` settings cap CHOLMOD's per-instance OpenMP work; the
  simplicial factorization itself is sequential. No process-wide thread settings
  are changed.
  Building cpo3 from source with this feature still requires compatible headers.
  Set `CHOLMOD_INCLUDE_DIR` to the directory containing `cholmod.h`, or use
  `pkg-config` discovery: install `suite-sparse pkg-config` with Homebrew or
  `libsuitesparse-dev pkg-config` with apt. Set `PKG_CONFIG_PATH` to the directory
  containing `cholmod.pc` or `CHOLMOD.pc` if needed. Conda's Windows package has
  no pkg-config metadata; use `%CONDA_PREFIX%\Library\include\suitesparse` as
  `CHOLMOD_INCLUDE_DIR`. The build neither links nor bundles SuiteSparse libraries.
- **PARDISO:** requires a compatible Intel oneMKL shared runtime installed
  separately. Select `LinearSolver.PARDISO`. Set `CPO3_MKL_RT` to the runtime's
  full path before starting Python if it is not on the system library path.
  The loader also searches `MKLROOT/lib`, `MKLROOT/lib/intel64`, and the
  corresponding `bin` directories. MKL's dependent libraries must also be on the
  loader path (for example, source Intel's `setvars.sh` on Linux). Explicit
  thread caps use MKL's thread-local setting and restore it after every call.
  The adapter uses `pardiso_64`, independently of `MKL_INTERFACE_LAYER`, and
  disables MKL's internal refinement so cpo3 owns correction counting.

Check availability before selecting a backend:

```python
cpo3.LinearSolver.ACCELERATE.is_available()
cpo3.LinearSolver.CHOLMOD.is_available()
cpo3.LinearSolver.PARDISO.is_available()
```

Availability is also exposed as `LinearSolver::is_available()` in Rust. A missing
feature or runtime produces an error when that backend is needed; it does not
silently fall back to Faer. Feral is no longer supported.

Backend references: [Apple Sparse Solvers](https://developer.apple.com/documentation/accelerate/sparse-solvers-library),
[CHOLMOD guide](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Doc/CHOLMOD_UserGuide.tex),
[Intel PARDISO controls](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/pardiso-iparm-parameter.html).

## Python quick start

This problem minimizes $\tfrac12(x_1^2+x_2^2)-x_1-2x_2$ subject to
$x_1\le3$ and $x_2\le4$. The solution is $x=(1,2)$, with objective $-2.5$.

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

if solution.status == cpo3.SolverStatus.SOLVED:
    print(solution.x)  # approximately [1.0, 2.0]
    print(solution.info.pobj)  # approximately -2.5
```

Use SciPy sparse matrices for `P` and `A`, and set `P=None` for an LP.
Cone blocks correspond to consecutive rows of `A` and `b`; their dimensions must
sum to the number of constraints. Use `cpo3.ZeroCone(dim)` for equality rows.

Pass options through `Settings`:

```python
settings = cpo3.Settings(verbose=True, time_limit=30.0, threads=1)
solution = cpo3.solve(problem, settings=settings)
```

`verbose=True` prints progress and a termination summary. `threads=1` selects
sequential factorization and solves; the default, `threads=0`, uses the backend's
automatic thread selection. See the
[`Settings` docstring](python/cpo3/settings.py) for the available controls.

For a related problem, pass `warm_start=(solution.x, solution.y, solution.s)`
to `solve`. The solver screens the point before deciding whether to use it.
Load a problem from disk with `cpo3.io.read_mps("problem.mps")`.

## Rust quick start

The same problem in Rust:

```rust
use cpo3::{Cone, CscMatrix, Problem, Settings, SolverStatus, solve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = Problem::new(
        Some(CscMatrix::identity(2)?),
        vec![-1.0, -2.0],
        CscMatrix::identity(2)?,
        vec![3.0, 4.0],
        vec![Cone::Nonnegative(2)],
    )?;
    let solution = solve(&problem, &Settings::default())?;
    println!("{:?}", solution.status());

    if solution.status() == SolverStatus::Solved {
        println!("{:?}", solution.x()); // approximately [1.0, 2.0]
    }
    Ok(())
}
```

Run numerical workloads with `cargo run --release`. Enable the `mps` Cargo
feature to read MPS and QPS files. Use `solve_from` with a `WarmStart` to supply a
starting point for a related problem.

## Understanding results

Always check the termination status before interpreting the returned vectors.

| Status | Meaning |
| --- | --- |
| `solved` | A solution satisfies the feasibility and duality-gap tolerances. |
| `primal_infeasible` | A certificate of primal infeasibility was found. |
| `dual_infeasible` | A certificate of dual infeasibility was found. |
| `max_iterations` | The iteration limit was reached. |
| `time_limit` | The time limit was reached. |

For a solved problem, `x` contains primal variables, `y` dual multipliers, and `s`
constraint slacks. For primal infeasibility, `y` contains the certificate; for
dual infeasibility, `x` and `s` contain the certificate direction. A limit status
does not certify optimality or infeasibility.

`solution.info` includes objectives, residuals, elapsed time, iteration counts,
and linear-algebra statistics. Numerical failures raise `RuntimeError` in Python
or return `SolveError` in Rust.

## Development

After installing the checkout with `uv sync` (or the configured command above), run:

```sh
cargo test --features mps
uv run --no-sync pytest -ra
cargo fmt --check
cargo clippy --all-targets --features python,mps -- -D warnings
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
```

Python tests exercise every available backend and skip unavailable ones. To match
CI's Rust checks, install the CHOLMOD development headers described above, use
`--features mps,accelerate,cholmod,pardiso` for `cargo test`, and use `--all-features`
for Clippy.

See [benchmarks/README.md](benchmarks/README.md) for the benchmark data.

## Authors and license

Matteo Santamaria, Michael Cummins, and Brendan O'Donoghue.

Licensed under [MIT](LICENSE).
