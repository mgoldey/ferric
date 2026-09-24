# Contributing to ferric

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| Rust       | 1.75+   | stable toolchain |
| libint2    | 2.7+    | build from the [mpqc4 tarball](https://github.com/evaleev/libint/releases) |
| OpenBLAS   | any     | with LAPACK support (`libopenblas-dev` on Debian/Ubuntu) |
| libxc      | as packaged | `libxc-dev` on Debian/Ubuntu (CI builds against Ubuntu 22.04's) |
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
replacing the site-packages symlink with a copied `.so`. To restore it:

```bash
ln -sf "$(pwd)/target/release/libferric.so" \
  .venv/lib/python3.11/site-packages/ferric/ferric.cpython-311-x86_64-linux-gnu.so
```

Adjust the Python version in both paths to match your `.venv`.

To pull in the `dev` extra (numpy, scipy, pytest, ipython, ...) without touching
the compiled extension:

```bash
uv sync --extra dev --no-install-project --inexact
```

`--inexact` matters: `uv sync` otherwise removes packages the lockfile does not
list, and with `--no-install-project` that includes the `ferric` you installed
or symlinked above.

### The Python development loop

```bash
uv run maturin develop --release   # compile and install the .so into .venv
uv run python scripts/foo.py       # later runs reuse it, no recompile
```

Always use `uv run maturin develop`, not a bare `maturin develop`. Without
`uv run`, maturin targets whatever Python is first on `$PATH` and installs there
instead of into the project `.venv`, and `uv run python` keeps loading the stale
build.

For zero-copy updates, replace the installed `.so` with the symlink above. It
points at **`target/release/libferric.so`**, which a plain `cargo build
--release -p ferric-python` refreshes. Do not symlink to
`target/maturin/libferric.so`: only `maturin develop` writes it, so it goes
stale, and a failed `maturin develop` can leave it truncated to 0 bytes. A later
`uv run maturin develop` replaces the symlink with a copy; re-run `ln -sf` to
restore it.

### Documentation

User documentation is the mdBook under `site/src` (table of contents:
`site/src/SUMMARY.md`), published by `.github/workflows/docs.yml`. Build it
locally with [mdBook 0.4.40](https://github.com/rust-lang/mdBook/releases/tag/v0.4.40),
the version CI pins:

```bash
mdbook serve site
```

`ruff format` also formats Python code blocks inside Markdown, so run
`uvx ruff@0.15.8 format .` after editing a page with Python examples. Every
number on a page should be traceable to a test, an example header or a measured
run; if it is not, leave the cell empty rather than estimating.

The docs are tested, in four places:

| Check | What fails it | Where it runs |
|---|---|---|
| `mdbook build` + `scripts/check_doc_links.py` | a missing page, a link to a page or heading that does not exist | `docs-check.yml`, on every PR touching Markdown |
| `input_reference_documents_exactly_the_accepted_keys` | a config key with no row in `reference/input.md`, or a row for a key the CLI rejects | `ferric-cli` unit tests (`src/config_doc_tests.rs`) |
| `every_toml_block_in_the_docs_parses` | a ```` ```toml ```` block the CLI would reject | same |
| `scripts/check_doc_snippets.py` | a marked Python block that raises, or prints something other than the ```` ```text ```` block the page shows | wheel smoke test (nightly, release tags) and `crates/ferric-python/tests/test_doc_snippets.py` |

`ci.yml` skips docs-only PRs (its `paths-ignore` covers `**/*.md` and
`site/**`), so a PR that edits only `input.md` or a TOML example runs the two
`ferric-cli` tests nightly, not before merge. Run them yourself when you edit
either:

```bash
OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --lib doc_tests
```

Adding a config field therefore means adding its row to
`site/src/reference/input.md`; the test names the missing key. A TOML block that
is deliberately invalid goes after a `<!-- doctest: skip -->` line.

To make a Python block checked, put `<!-- doctest -->` on the line above it.
Marked blocks on a page run in order in one interpreter, like a notebook, in an
empty directory, so they must not need a clone. A ```` ```text ```` block
directly after a marked block (or anywhere below it, marked
`<!-- doctest-output -->`) is its expected output; numbers are compared to
`atol=1e-8` unless the marker says otherwise (`<!-- doctest: atol=1e-6 -->`).
Run the check locally against a release build:

```bash
cargo build --release -p ferric-python
OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest crates/ferric-python/tests/test_doc_snippets.py
```

Contributor-only analysis lives next to the book but outside its table of
contents, for example `site/src/reference/ci-timing-analysis.md` (test sharding
and CI timing).

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
being silently repaired in-job. If you add
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

The crate layout and cross-cutting conventions (threading, determinism,
memory, errors) are described in the book's
[Architecture](https://matthew.thegoldeys.com/ferric/reference/architecture.html)
page. Doc comments in the code record the reasoning behind non-obvious
decisions, including optimizations that were tried and rejected.

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

### Derived data (not linked)

`ferric-d3` links against nothing, but its reference tables are DERIVED from an
LGPL-3.0 source, which is a separate obligation from the linking rows above and
is recorded here so it is not missed:

| Source | License | Relationship |
|--------|---------|--------------|
| [simple-dftd3](https://github.com/dftd3/simple-dftd3) | LGPL-3.0-or-later | D3 reference data transcribed into `crates/ferric-d3/src/tables.rs` |

`crates/ferric-d3/src/tables.rs` is generated by
`crates/ferric-d3/generate_tables.py`, which parses simple-dftd3's Fortran
sources and emits Rust. The numerical content (reference C6 coefficients,
reference coordination numbers, r4/r2 expectation values, covalent and vdW
radii) is the D3 parameterisation published by Grimme et al., JCP 132, 154104
(2010) and JCC 32, 1456 (2011); simple-dftd3 is the machine-readable form it
was taken from. No simple-dftd3 CODE is copied or linked, and `ferric-d3` has
no build-time or run-time dependency on it -- the generator is run by hand when
the tables need regenerating, and is not part of the build.
