<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cpo3-org/.github/main/assets/logo-dark.png">
    <img src="https://raw.githubusercontent.com/cpo3-org/.github/main/assets/logo-light.png" alt="cpo³ — sparse optimization solver" width="320">
  </picture>
</p>

<p align="center">
  A sparse interior-point solver for convex linear and quadratic programs.
</p>

<p align="center">
  <a href="https://pypi.org/project/cpo3/"><img src="https://img.shields.io/pypi/v/cpo3?color=527a9a" alt="PyPI"></a>
  <a href="https://crates.io/crates/cpo3"><img src="https://img.shields.io/crates/v/cpo3?color=527a9a" alt="crates.io"></a>
  <a href="https://arxiv.org/abs/2611.18403"><img src="https://img.shields.io/badge/arXiv-2611.18403-b31b1b" alt="arXiv"></a>
  <a href="https://github.com/cpo3-org/cpo3/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://cpo3-org.github.io/site/">Documentation</a> ·
  <a href="https://github.com/cpo3-org/cpo3/blob/main/CHANGELOG.md">Changelog</a>
</p>

cpo3 solves convex optimization problems of the form

```text
minimize      ½ xᵀ P x + cᵀ x
subject to    Ax + s = b,  s ∈ K
```

Here, `P` is symmetric positive semidefinite, and `K` is a product of zero and
nonnegative cones representing linear equality and inequality constraints.
Setting `P = 0` gives a linear program.

The solver uses an infeasible-start, primal–dual interior-point method with
predictor–corrector steps based on a novel strongly monotone central path equation.
It handles quadratic objectives directly and uses a homogeneous embedding to
detect primal and dual infeasibility.

In the paper's [benchmark results](https://arxiv.org/pdf/2611.18403#page=41),
cpo3 returned verified solutions for **640 of 675 problems**, the highest count
among all tested open-source and commercial solvers. It also recorded the lowest
penalized shifted geometric mean runtime among the tested open-source solvers.

> [!NOTE]
> cpo3 is under active development. Feedback and contributions are welcome.

## Quick start

### Python

Requires Python 3.10 or later.

```sh
python -m pip install cpo3==0.0.1
```

Solve a two-variable quadratic program with upper-bound constraints:

```python
import cpo3
import numpy as np
import scipy as sp

problem = cpo3.Problem(
    P=sp.sparse.eye(2, format="csc"),
    c=np.array([-1.0, -2.0]),
    A=sp.sparse.eye(2, format="csc"),
    b=np.array([3.0, 4.0]),
    cones=[cpo3.NonnegativeCone(2)],
)
solution = cpo3.solve(problem)
print(solution.status)

if solution.status == cpo3.SolverStatus.SOLVED:
    print(solution.x)  # approximately [1.0, 2.0]
```

### Rust

Requires Rust 1.91 or later. Add to `Cargo.toml`:

```toml
[dependencies]
cpo3 = "0.0.1"
```

### Further reading

The [getting-started guide](https://cpo3-org.github.io/site/getting-started/)
includes complete Python and Rust examples, with details on constructing problem
data and interpreting solver results.

cpo3 includes the Faer backend by default, with no external numerical libraries
required. For Accelerate, CHOLMOD, or PARDISO support, or to build from source, see
the [installation guide](https://cpo3-org.github.io/site/installation/).

## Roadmap

Planned extensions to the solver include:

- [ ] Additional cones, including second-order and positive semidefinite cones.
- [ ] Iterative linear solvers.
- [ ] GPU support.

## Citation

If you use cpo3 in your research, please cite our [paper](https://arxiv.org/abs/2611.18403):

```bibtex
@misc{cpo3,
  title         = {Strongly Monotone Central Path Equations for Quadratic Cone Programs},
  author        = {Michael Cummins and Matteo Santamaria and Brendan O'Donoghue},
  year          = {2026},
  eprint        = {2611.18403},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC},
  url           = {https://arxiv.org/abs/2611.18403}
}
```
