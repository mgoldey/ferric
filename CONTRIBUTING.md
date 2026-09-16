# Contributing to ferric

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| Rust       | 1.75+   | stable toolchain |
| libint2    | 2.7+    | build from the [mpqc4 tarball](https://github.com/evaleev/libint/releases) |
| OpenBLAS   | any     | with LAPACK support (`libopenblas-dev` on Debian/Ubuntu) |
| libxc      | 6+      | `libxc-dev` on Debian/Ubuntu |
| cmake      | 3.14+   | needed to build the vendored libecpint |
| Python     | 3.10+   | matches `requires-python` in `pyproject.toml`; needed for `uv`, pytest, and the Python bindings |

**Non-standard install locations.** If libint2 or libxc are not in
`$HOME/.local` or `/usr/local`, set these environment variables before building:

```
export LIBINT2_PREFIX=/path/to/libint2   # expects $LIBINT2_PREFIX/include and $LIBINT2_PREFIX/lib
export LIBXC_DIR=/path/to/libxc
```

## Building

```
cargo build --workspace
```

For a release build (required for Python bindings):

```
cargo build --release --workspace
```

## Testing

**Critical:** set `OPENBLAS_NUM_THREADS=1` for all test runs. OpenBLAS with
multiple threads under rayon causes segfaults and non-deterministic slowdowns.

```bash
# Full Rust test suite
OPENBLAS_NUM_THREADS=1 cargo test --workspace

# Single crate
OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf

# Python bindings (requires a release build of ferric-python)
cargo build --release -p ferric-python
OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest crates/ferric-python/tests/ -q
```

The `--no-sync` flag matters: a bare `uv run` rebuilds and reinstalls the wheel,
replacing the site-packages symlink. See CLAUDE.md for recovery instructions.

Plain `uv sync` now works, for the first time: before this branch,
`requires-python` was `>=3.9` while the `dev` extra's `ipython>=9.13.0`
required `>=3.11`, so `uv lock`/`uv sync`/`uv run` failed for everyone,
not just on old Python. The floor is now `>=3.10` (matching the wheel
matrix, which never actually built for 3.9) and the `dev` extra's `ipython`
floor is `>=8.18`, which resolves. Use it to pull in the `dev` extra's
dependencies (numpy, scipy, pytest, ipython, ...) without touching the
compiled extension:

```bash
uv sync --extra dev --no-install-project
```

## Code quality

CI enforces several gates. Some block a PR, some only report; know which is
which.

**Blocking:**

```bash
# Rust: compiles clean, clippy is warning-free, formatting is untouched
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all --check

# Python: formatting and security
uvx ruff@0.15.8 format --check .
uvx bandit@1.9.4 -ll -q -r . -x terf-tables/generate_interpolation_tables.py

# pre-commit hook set, run over the whole tree (see below)
pre-commit run --all-files --show-diff-on-failure
```

Run the formatters themselves, not just the checkers, before pushing:

```bash
cargo fmt --all
uvx ruff@0.15.8 format .
```

The `ruff` version pinned above matches CI's `python-quality` job. `ruff
format`'s output is not stable across releases, so formatting locally with a
different version can produce a diff that fails CI's `--check`, or pass
locally against a diff CI would reject.

The workspace Cargo.toml allows specific clippy lints that fire on reviewed
numerical code (e.g. `excessive_precision` for verbatim quadrature constants,
`needless_range_loop` for flat-index arithmetic in integral kernels). See the
`[workspace.lints.clippy]` section for the full list and rationale.

**Report-only** (CI runs these and posts the results, but a PR is not blocked
on them):

- `ruff check .` (default rule set): 282 remaining findings as of this
  writing, tracked as a backlog, not yet gated.
- `ty check`: type-check debt, not yet gated.
- Rust coverage and the `ferric-python` bindings coverage (nightly-only, via
  `schedule`/`workflow_dispatch`): no coverage threshold is set.
- The `tools/`/`experiments/` pytest suite run under coverage in
  `python-quality`: provisional, `continue-on-error` until a run has been
  observed green in CI.

**`--locked`.** Every `cargo` invocation in CI passes `--locked`, so a
`Cargo.lock` that has drifted from the manifests fails the build instead of
being silently repaired in-job (this bit the repo once: the lockfile was
stale from commit `9a0b955` until it was caught and regenerated). If you add
or change a dependency, regenerate the lockfile and commit it alongside the
manifest change:

```bash
cargo build --locked   # or `cargo update -p <crate>`; commit the resulting Cargo.lock diff
```

**pre-commit hooks.** `.pre-commit-config.yaml` runs bandit (medium+
severity), a critical-only subset of ruff (`E9,F63,F7,F82`: syntax errors,
undefined names), shellcheck, detect-secrets, and check-yaml/check-toml.
Install once per clone:

```bash
uv tool install pre-commit
pre-commit install
```

`scripts/install-hooks.sh` also installs this hook automatically if
`pre-commit` is already on your PATH. Either way, CI runs the same hook set
with `pre-commit run --all-files`, so skipping the local install only moves
the failure from your machine to CI, not away from it.

**git blame.** Two commits reformatted the whole tree in one mechanical pass
(`cargo fmt --all`, then `ruff format`) and are listed in
`.git-blame-ignore-revs` so they don't obscure real authorship locally. Opt
in once per clone:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Reliability conventions

ferric follows strict experimental protocols for numerical claims. The key
principles:

- **Exactness anchor first.** Every approximation has a trivial limit where it
  does nothing. Write `..._matches_exact_in_the_trivial_limit` before measuring
  anything else.
- **Consistency is not corroboration.** A construction bug is deterministic and
  reproduces across systems. Agreement distinguishes signal from noise, never
  signal from systematic error.
- **A test you have never seen fail is an assumption.** Mutation-test new tests
  against a deliberately broken version.
- **Too clean is a stop condition.** An exact coincidence at every system size
  is a fingerprint of arithmetic, not chemistry.

The full contributor guide lives in the project wiki under `docs/guide/dev/`:
architecture, adding a method, testing conventions, common pitfalls, and
workflow.

## Commit and PR guidelines

- Keep commits focused: one logical change per commit.
- Run the full test suite before pushing.
- The repo uses a pre-push hook (`scripts/install-hooks.sh`) that runs the CI
  gate, and (if `pre-commit` is on your PATH) also installs the pre-commit
  hooks described in "Code quality" above. Install it after cloning.

## Third-party licenses

ferric itself is dual-licensed under MIT OR Apache-2.0. It links against
external C/C++ libraries with their own licenses:

| Library   | License      | Linking   |
|-----------|-------------|-----------|
| libint2   | LGPL-3.0    | static    |
| libxc     | MPL-2.0     | dynamic   |
| OpenBLAS  | BSD-3-Clause| dynamic   |
| libecpint | BSD-3-Clause| static (vendored) |
| xtb       | LGPL-3.0    | dynamic (optional, feature-gated) |

These licenses govern redistribution of compiled binaries. In particular, static
linking of LGPL libraries requires that downstream users can relink against their
own version of the library, or that the combined work is distributed under a
GPL-compatible license. See each library's license for details.
