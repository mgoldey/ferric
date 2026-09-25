# Installation

There are two ways in. Most people want the first.

| You want to | Do this | Time |
|---|---|---|
| Run calculations | [Install the prebuilt wheel](#fastest-the-prebuilt-wheel) | about a minute |
| Change ferric's Rust code, or use MPI | [Build from source](#building-from-source) | about 10 minutes, mostly compiling ferric |

## Fastest: the prebuilt wheel

Wheels are published to PyPI for **Linux x86_64** (`manylinux_2_28`),
CPython 3.10–3.13. libint2 is statically linked into the extension, and libxc
and OpenBLAS ship inside the wheel as bundled shared libraries, so nothing
needs compiling.

```bash
pip install ferric        # or: uv pip install ferric
```

The only releases so far are pre-releases (`0.1.0rc*`). pip and uv both select
them when no final release exists, so the plain command above works. Pin a
version (`ferric==0.1.0rc5`) if you need a reproducible environment.

The wheel gives you two things:

- the Python module: `import ferric`
- a `ferric` command on `PATH` that runs TOML input files (the same CLI as
  `cargo run --bin ferric`)

### Check it works

```bash
python -c "import ferric; print(ferric.__file__)"
```

Then run [your first calculation](./quickstart.md). It takes under a second.

### What the wheel does *not* contain

The wheel holds the compiled library and nothing else. The `examples/` input
files, the `testdata/` molecules and the `tools/` pipeline scripts live in the
git repository. Pages that use them say so. To have them locally:

```bash
git clone https://github.com/mgoldey/ferric
cd ferric
ferric examples/water-rhf.toml
```

The CLI resolves `[molecule] xyz = "..."` relative to the **current directory**,
so run example files from the repository root.

## Building from source

Build from source if you are changing ferric, need the MPI build, or are on a
platform without a wheel. `ferric` links **libint2**, a C++ integral library;
`scripts/install-libint.sh` installs a prebuilt copy in seconds.

### Prerequisites

- **Rust 1.75+**: install via [rustup](https://rustup.rs/)
- **libint2 2.13.1**: the conda-forge Linux x86-64 build, which includes
  second-derivative integrals; `scripts/install-libint.sh` downloads it (checksum
  pinned) and needs `curl`, `python3`, `zstd` and `patchelf`
- **libxc**, **OpenBLAS**, **LAPACK**, **Eigen3** and **Boost** headers
- **Python 3.10+ and maturin**: optional, for the Python bindings

### Steps (Ubuntu 22.04+)

The apt list matches what CI installs (`.github/workflows/ci.yml`).

```bash
# 1. System dependencies
sudo apt-get install -y build-essential cmake g++ gfortran wget git \
    libeigen3-dev libopenblas-dev liblapack-dev pkg-config \
    libxc-dev libboost-dev patchelf zstd \
    python3-dev python3-pip python3-venv

# 2. Get ferric and install libint2 (seconds)
git clone https://github.com/mgoldey/ferric
cd ferric
scripts/install-libint.sh ~/.local/libint2-2.13.1
export LIBINT2_PREFIX=~/.local/libint2-2.13.1

# 3. Build ferric
cargo build --release

# 4. Check it: seconds, not minutes
OPENBLAS_NUM_THREADS=1 ./target/release/ferric examples/water-rhf.toml
```

The last line should end with `converged  = true` and
`energy     = -74.9631468000 Hartree`, as shown in the
[first calculation](./quickstart.md).

`LIBINT2_PREFIX` is read by `crates/ferric-integrals/build.rs`; unset, it
defaults to the installer's `~/.local/libint2-2.13.1` when that exists, and to
`~/.local` otherwise, so the `export` above is optional for the default path. The install script records the library's absolute path
in every binary that links it, so moving or deleting that directory breaks
the build's executables until they are rebuilt. A prefix holding a static
`libint2.a` (a from-source libint2 build) also works, but a library generated
without second derivatives has no analytic Hessians; frequencies then use
finite differences of the analytic gradient.

The workspace has three binaries (`ferric`, `ferric-cli` and `ferric-batch`),
so `cargo run` needs to be told which one: `cargo run --release --bin ferric -- input.toml`.

### Running the test suite

```bash
OPENBLAS_NUM_THREADS=1 cargo test --workspace
```

`.cargo/config.toml` sets `OPENBLAS_NUM_THREADS=1` for cargo-invoked
processes only when the variable is unset (its `[env]` entry has no
`force = true`, so an exported value wins). The prefix above makes this command
use 1 regardless of your shell. Do not raise it above 1; see
[Threading](#threading).
The Python binding tests are pytest, not cargo; see
[CONTRIBUTING.md](https://github.com/mgoldey/ferric/blob/main/CONTRIBUTING.md).

### Debug vs release

- **Debug** (`cargo build`): fast to compile, slow to run. Use it while
  iterating on Rust code. It catches `debug_assert!` violations and integer
  overflow that release builds silently allow.
- **Release** (`cargo build --release`): slow to compile, fast to run. Use it
  for anything you will actually wait on: real molecules, benchmarks, and any
  RPA/GW/CC job.

### Python bindings from source

```bash
uv sync
uv run maturin develop --release
uv run python -c "import ferric; print(ferric.__file__)"
```

Use `uv run maturin develop`, not a bare `maturin develop`. A bare one can
install into a different interpreter from the one `uv run python` loads, and the
stale build keeps getting imported.

### What the libint2 build carries

Integral classes, derivative orders and angular-momentum limits are fixed when
libint2's source is *generated*, so no build flag changes them. The build
`scripts/install-libint.sh` installs (conda-forge 2.13.1) carries:

| Integrals | Highest angular momentum for energy / 1st / 2nd derivatives | Needed for |
|---|---|---|
| 4-centre ERI | 7 / 6 / 3 | SCF, gradients, analytic Hessians |
| One-electron (overlap, kinetic, nuclear) | 7 / 6 / 3 | the same |
| 3- and 2-centre ERI | 7 / 7 / 4 | RI-MP2, RPA, GW and their gradients |
| G12 geminal | 4 (energy only) | F12 / geminal integrals |

Analytic Hessians therefore cover orbital bases up to f functions; a basis with
g or higher functions uses finite differences of the analytic gradient. The
upstream mpqc4 tarball (libint2 2.7.2) has no second derivatives and no G12
class: built against it, ferric uses finite-difference Hessians and the G12
tests skip with an explicit message.

## MPI

MPI is a **source build only**. There is no MPI wheel on PyPI: the wheels
workflow deliberately does not publish one.

```bash
sudo apt-get install -y libopenmpi-dev openmpi-bin libclang-dev
cargo build --release --workspace --features mpi
mpirun -np 4 -x OPENBLAS_NUM_THREADS=1 ./target/release/ferric examples/water-rhf.toml
```

`libclang-dev` is needed by `mpi-sys`'s bindgen step. MPICH and Intel MPI are
not supported: their `libmpi.so.12` ABI differs from OpenMPI 4.x's
`libmpi.so.40`.

### The CLI is the MPI entry point; the Python API is not

The CLI is SPMD by design: `mpirun -np N ferric input.toml` runs one rank per
process.

`mpirun -np N python script.py` is **not supported**. The bindings expose no
rank or world-size accessor, so `if rank == 0` cannot be written. Every rank
runs the whole script, prints N times, and races on the same output files.

## Threading

The `ferric` binary and `import ferric` pin OpenBLAS to one thread when
`OPENBLAS_NUM_THREADS` is unset, and honour the variable when it is set
(`cargo` sets it to 1 through `.cargo/config.toml`, but only when it is unset,
so an exported value reaches cargo-launched processes). Leave it unset or at 1.
ferric uses rayon for its parallelism; a threaded OpenBLAS on top of that
oversubscribes the machine, and its LU routines can crash when called from
rayon workers. For throughput
across many independent jobs, prefer many single-threaded processes over one
multi-threaded job.
