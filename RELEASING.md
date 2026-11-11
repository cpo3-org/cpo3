# Releasing cpo3

`Cargo.toml` is the version source for both crates.io and Python packages.
`pyproject.toml` obtains its version dynamically through maturin. Version 0.0.1
is prepared here but has not been published by this change.

## Build and verify without publishing

Open a pull request or run **Release packages** (`release.yml`) manually on the
release branch. Both paths build and test only. They cannot enter the publish
jobs, even when manually run against a tag.

The workflow:

1. Tests default Rust and `mps` builds on Rust 1.91 and stable, builds docs, and
   runs formatting, linting, type checks, and native-backend tests.
2. Runs `cargo publish --dry-run --locked` and builds an sdist, checking the
   native sources, lockfile, Python modules, typing files, and licenses.
3. Builds every wheel from that sdist: Linux x86-64 (manylinux 2.28), Windows
   x86-64, macOS Intel, and macOS Apple Silicon (macOS 13 minimum).
4. Installs each wheel into fresh Python 3.10, 3.14, and 3.15 environments and runs
   the test suite outside the checkout. It requires Faer (plus Accelerate on
   macOS) with CHOLMOD/MKL discovery pointed at missing libraries, then installs
   compatible runtimes and requires the optional backends too.
5. Uploads the checked crate, sdist, and wheels as workflow artifacts.

Wheel policy: Faer everywhere; Accelerate on macOS; CHOLMOD on every wheel
platform; PARDISO on Linux/Windows x86-64. CHOLMOD/MKL are dynamically loaded,
never bundled. Headers come from SuiteSparse v7.10.1 (CHOLMOD 5.3.1), while the
runtime tests require CHOLMOD 5.2.x or 5.3.x. Source installations default to Faer
(and `mps` for Python); optional backends require explicit Cargo features.

For local checks, use:

```sh
cargo test --locked
cargo test --locked --features mps
cargo doc --locked --no-deps --features mps
cargo publish --dry-run --locked
maturin sdist --out dist
```

A dirty working tree needs `--allow-dirty` for the Cargo publishing dry run.
Inspect the workflow artifacts and results before tagging. The full OS matrix
requires GitHub Actions; a successful local build does not replace it.

## One-time registry setup

- Create GitHub environments `pypi` and `crates-io`, with required reviewers if
  desired. Restrict deployment to release tags.
- On PyPI, configure a Trusted Publisher for owner `cpo3-org`, repository `cpo3`,
  workflow filename `release.yml`, environment `pypi`. For the initial release,
  create a pending publisher for project `cpo3`.
- crates.io requires a token for a crate's first publication. Set a short-lived
  crates.io publishing token as the `CARGO_REGISTRY_TOKEN` secret in the
  `crates-io` environment for that first run. After the first release, configure
  a Trusted Publisher for `cpo3-org/cpo3`, `release.yml`, environment `crates-io`;
  delete/revoke the bootstrap secret. Subsequent runs obtain temporary tokens
  through `rust-lang/crates-io-auth-action`.

References: [PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [crates.io Trusted Publishing](https://crates.io/docs/trusted-publishing).

## Publish when ready

1. Update the version in `Cargo.toml` and regenerate `Cargo.lock` when changing
   versions. Date the changelog entry and run the build-only workflow.
2. Merge the reviewed changes and tag that exact commit as `v<version>` (for this
   release, `v0.0.1`). Push the tag only when both registries are configured and
   you intend to publish.
3. The tag run rejects a mismatch with `Cargo.toml`. After all checks and wheel
   tests pass, separate environment-gated jobs publish the Python artifacts and
   the Rust crate. No GitHub Release is created automatically.
4. Verify installation from both registries in fresh environments.

Registries are independent: one upload can succeed while the other fails. Fix
credentials/configuration and rerun only the failed job, retaining the original
release commit and artifacts. Never move a published tag or try to replace an
already-published version. Changes to the package itself need a new version.
