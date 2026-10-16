# First-pass refactor validation

Compare the `c74a982` baseline with the refactor using isolated release builds.
Both builds use the existing cubic step-length centering rule. The baseline's
default governor is off and its additional centrality-corrector budget is zero.

The completed local comparison is documented in
[results/report.md](results/report.md), with profiles and all final run records.

`compare.py` runs all 486 bundled MPS problems and the locally available MPC
and DC-OPF collections. It decompresses MPS input before calling `read_mps`,
and uses the existing `solver_benchmarks` transformations for MPC/DC-OPF.
The harness does not download data. Each build runs in its own persistent
Python process with numerical thread counts set to one. Build order alternates
by problem and repeat; the processes never solve concurrently.

Settings match the current default iteration budget and tolerances, with
presolve disabled to expose the interior-point method and a 60-second solve
limit. A 120-second external limit also covers loading and verification.
Loading and verification are excluded from elapsed solver time. A small
unmeasured solve initializes library dispatch before measured runs.

Each result records exact solution-vector hashes, hexadecimal objective and
residual values, work counters, and independent KKT verification at `1e-6`.
The raw result log is append-only and resumable; later successful loader
attempts supersede earlier harness errors. Analysis requires all requested
runs and rejects unresolved harness errors, worker failures, or external
timeouts. Wall-clock-censored trajectories are reported separately.

Run with a Python interpreter containing NumPy, SciPy, and the dependencies
of the local `solver_benchmarks` checkout:

```sh
python benchmarks/refactor/compare.py compare \
  --baseline /absolute/path/to/baseline-package \
  --candidate /absolute/path/to/candidate-package \
  --harness /absolute/path/to/solver_benchmarks \
  --data /absolute/path/to/cpo3/benchmarks/data \
  --output /absolute/path/to/run --repeats 2
```

Each package directory contains `cpo3/` with that revision's Python sources
and native extension. Build both revisions with the same Rust toolchain,
locked dependencies, release profile, and `python,mps` features. On macOS,
plain Cargo builds of the extension require dynamic Python symbol lookup:

```sh
cargo rustc --offline --locked --release --features python,mps --lib -- \
  -C link-arg=-undefined -C link-arg=dynamic_lookup
```

Copy the resulting `libcpo3.dylib` into the isolated package as
`cpo3/_core.abi3.so`. Maturin can also build/install isolated wheels.

Generate the audit and figures with NumPy and Matplotlib:

```sh
python benchmarks/refactor/analyze.py /absolute/path/to/run \
  benchmarks/refactor/results
```

Dolan–Moré profiles use the better of the two builds as the per-problem
reference, retain all failures in the denominator, and count a solve only
when it passes the independent KKT check. Runtime profiles use median elapsed
time across repeats; iteration profiles use median iterations (zero mapped
to one). Timing variation is compared against repeated baseline solves,
separately identifying cases whose solve time is at least 10 ms.

These local profiles assess this refactor against the current branch. They
are not a rerun of historical cluster measurements or configurations that
enabled the removed governor or additional correctors.
