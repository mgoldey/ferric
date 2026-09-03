# Contributing to ferric

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| Rust       | 1.75+   | stable toolchain |
| libint2    | 2.7+    | build from the [mpqc4 tarball](https://github.com/evaleev/libint/releases) |
| OpenBLAS   | any     | with LAPACK support (`libopenblas-dev` on Debian/Ubuntu) |
| libxc      | 6+      | `libxc-dev` on Debian/Ubuntu |
| cmake      | 3.14+   | needed to build the vendored libecpint |

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

## Code quality

Clippy must pass on all targets:

```
cargo clippy --workspace --all-targets
```

The workspace Cargo.toml allows specific clippy lints that fire on reviewed
numerical code (e.g. `excessive_precision` for verbatim quadrature constants,
`needless_range_loop` for flat-index arithmetic in integral kernels). See the
`[workspace.lints.clippy]` section for the full list and rationale.

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
  gate. Install it after cloning.

### The complexity gate

`scripts/ci-gate.sh` includes a CC/MI regression gate
(`scripts/complexity_gate.py`) that compares each function against a
checked-in snapshot, `scripts/complexity_baseline.json`. It fails only on a
regression versus that snapshot, not on an absolute threshold — several
SCF/RPA kernels are legitimately complex and are not targets for mechanical
splitting.

Two things about it are worth knowing before it blocks a push:

- **A stale baseline blocks everyone, not just you.** The snapshot is a
  checked-in file, so it goes stale as unrelated commits land. When it does,
  the gate reports regressions in code your branch never touched. Before
  assuming your diff is at fault, run the gate on an unmodified checkout of
  `main` **at the same filesystem path** and compare the two regression sets.
  Regenerate with `python3 scripts/complexity_gate.py --update-baseline`, and
  commit the refreshed JSON on its own so what it absorbs is visible in
  review rather than folded into an unrelated change.
- **Compare at the same path.** Baseline keys are repo-relative, so the gate
  works from a worktree — but a baseline generated before that fix stored
  absolute paths, matched nothing from any other directory, and passed
  vacuously while comparing against zero entries. The gate now refuses to
  report `PASS` when a non-empty baseline shares no keys with the scan; if you
  see that error, regenerate the baseline. When in doubt, prefer a same-path
  control: a gate run that measured nothing looks exactly like a clean one.

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
