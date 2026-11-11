# Changelog

## 0.0.1 — 2026-11-11

Initial experimental release. APIs and numerical behavior may change between
releases.

- Sparse convex LP and QP solver with zero and nonnegative cone constraints.
- Rust API and typed Python interface, including warm starts and infeasibility
  certificates.
- Presolve, Ruiz equilibration, and continuous MPS/QPS input.
- Faer factorization, with optional Accelerate, CHOLMOD, and MKL PARDISO backends.
- Python wheels for Linux x86-64, Windows x86-64, and macOS Intel/Apple Silicon.
  Wheels use external CHOLMOD/MKL runtimes; macOS wheels include Accelerate support.

Second-order and semidefinite cone solving, integer optimization, and
free-threaded Python wheels are outside this release's supported scope.
