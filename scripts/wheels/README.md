# scripts/wheels: macOS arm64 and Windows x86_64 wheels

Used by `.github/workflows/native-deps-macwin.yml` (the native libraries) and
`.github/workflows/wheels-macwin.yml` (the wheels). Both are reusable
workflows. `wheels.yml` calls them for releases and nightly runs. The Linux
wheels are built by `wheels.yml` itself and do not use anything here.

| File | Used by | What it does |
|---|---|---|
| `build-libint2.sh` | native-deps | Builds ferric's small libint2 2.7.2 export as a static library (`libint2.a` on macOS, `int2.lib` on Windows) |
| `build-blas-xc.sh` | native-deps | Builds OpenBLAS 0.3.27 (fixed TARGET, C LAPACK, pthreads, `NUM_THREADS=64`) and libxc 6.2.2, and copies in the Eigen 3.4 headers |
| `stamp_version.py` | wheels (tags only) | Writes the tag's version into `pyproject.toml` and `crates/ferric-python/Cargo.toml`, then verifies the change with `git diff` (the Linux job does the same in shell) |
| `add_licenses.py` | wheels | Adds the bundled libraries' licence texts to `dist-info/licenses/third-party/` and updates RECORD |
| `smoke.py` | wheels (smoke job) | Checks an installed wheel: the Linux smoke cases, plus energies compared with Linux, each case in its own process |

## The native-deps prefix

There are two artifacts per platform `P` (`macos-arm64`, `windows-x86_64`),
kept for 7 days:

- `native-libint2-P`: libint2 in `include/` and `lib/`, plus `licenses/`.
- `native-blas-xc-P`: OpenBLAS, libxc and Eigen. It has `include/` (with
  `eigen3/`), `lib/` (with `pkgconfig/openblas.pc`), `bin/` (Windows only,
  holding `openblas.dll`) and `licenses/`.

Download both into one directory and use it as one prefix:

```yaml
- uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093 # v4.3.0
  with:
    pattern: native-*-macos-arm64     # or native-*-windows-x86_64
    merge-multiple: true
    path: native
    # From another run, e.g. while this branch's cache is cold:
    # run-id: <run id>
    # github-token: ${{ github.token }}
```

To build ferric against the prefix `P=$GITHUB_WORKSPACE/native`
(`cygpath -m` it on Windows):

```bash
export LIBINT2_PREFIX="$P"                 # libint2 headers + lib, and include/eigen3
export FERRIC_REQUIRE_PATCHED_BOYS=1       # the 2.7.2 boys.h override must apply
export PKG_CONFIG_PATH="$P/lib/pkgconfig"  # openblas.pc, relocatable (${pcfiledir})
export RUSTFLAGS="-L native=$P/lib"        # openblas / xc for every crate's link line
# Windows only, in an MSVC developer environment (ilammy/msvc-dev-cmd):
export PATH="$(dirname "$(which cl)"):/c/Program Files/LLVM/bin:$PATH"
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER="$(dirname "$(which cl)")/link.exe"
export CC=clang-cl CXX=clang-cl
```

The build also needs these tools on PATH: CMake, Ninja and pkg-config. The
workflows install them with micromamba from conda-forge. Nothing from that
environment is linked into ferric.

To run what you built:

- macOS: `DYLD_LIBRARY_PATH="$P/lib"` (`libopenblas.0.dylib` and
  `libxc.*.dylib` have `@rpath` install names).
- Windows: add `$P/bin` to PATH (for `openblas.dll`). libint2 and libxc are
  static there.

## Caches

Each platform has two `actions/cache` entries, written only by the
native-deps jobs:

```
native-macwin-v1-libint2-<P>-<os>-<compiler major>-libint2.7.2-<export sha256>-eigen3.4.0-script<sha256[:16] of build-libint2.sh>
native-macwin-v1-blas-xc-<P>-<os>-<compiler major>-openblas0.3.27-libxc6.2.2-eigen3.4.0-script<sha256[:16] of build-blas-xc.sh>
```

`<os>` is `macos15-arm64` or `win10.0.26100`. `<compiler major>` is
`appleclang17` or `clangcl20`. Bumping `NATIVE_MACWIN_CACHE_VERSION` in
native-deps-macwin.yml drops every entry.

A branch can restore its own caches and main's, but not another branch's.
Until a run on main has saved them, other branches must build cold (about
90 min for libint2) or reuse a run's artifacts by run id.
