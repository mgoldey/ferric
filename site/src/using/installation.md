# Installation

`ferric` links against **libint2**, a C++ integral library that must be built
from source. That build takes roughly 30 minutes and is the main cost of getting
started.

## Prerequisites

- **Rust 1.75+** — install via [rustup](https://rustup.rs/)
- **libint2 2.7+** — from the
  [mpqc4 tarball](https://github.com/evaleev/libint/releases/download/v2.7.2/libint-2.7.2-mpqc4.tgz)
- **OpenBLAS and LAPACK**
- **Eigen3** headers
- **Python 3.10+ and maturin** — optional, for the Python bindings

### What the mpqc4 export does and does not carry

These capabilities are fixed when the tarball is *generated*, so no `cmake` flag
changes them. `compiler.config` inside the tarball records the exact settings.

| Capability | mpqc4 export | Needed for |
|---|---|---|
| 1st derivatives | yes | analytical **gradients**, geometry optimization |
| RI / 3- and 2-center ERI | yes | RI-MP2, RPA, GW |
| 2nd derivatives | **no** | analytical **Hessians** / frequencies |
| G12 geminal | **no** | F12 / geminal integrals |

Building against this tarball is correct for everything `ferric` currently
validates. The G12-dependent tests detect its absence at run time and **skip
with an explicit message** rather than failing.

Getting either missing capability requires re-generating libint2 from the
upstream source repo with the corresponding `--enable-*` flags — a substantially
longer build.

## Building from source

```bash
# System dependencies (Ubuntu 22.04+)
sudo apt-get install -y build-essential cmake g++ gfortran wget \
    libeigen3-dev libopenblas-dev liblapack-dev pkg-config \
    python3-dev python3-pip python3-venv

# Build and install libint2 (~30 min)
wget https://github.com/evaleev/libint/releases/download/v2.7.2/libint-2.7.2-mpqc4.tgz
tar xzf libint-2.7.2-mpqc4.tgz
cd libint-2.7.2-mpqc4
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
make install
cd ../..

# Build ferric
cargo build --release

# Run tests
OPENBLAS_NUM_THREADS=1 cargo test --workspace
```

## Debug vs release

The difference is large enough to matter in practice:

- **Debug** (`cargo build`) — fast to compile, slow to run. Use it while
  iterating on Rust code; it catches `debug_assert!` violations and integer
  overflow that release builds silently permit.
- **Release** (`cargo build --release`) — slow to compile, fast to run. Use it
  for anything you will actually wait on: real molecules, benchmarks, and any
  RPA/GW/CC job.

## Python bindings

```bash
# Set up the venv and install the extension in editable/develop mode
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release

# Verify
python -c "import ferric; print(ferric.__file__)"
```

See [Python bindings](./python.md).

## Optional: MPI

The `mpi` feature additionally needs an MPI implementation (OpenMPI or MPICH)
**and** libclang (`libclang-dev`, for `mpi-sys`'s bindgen step).

## Threading

Set `OPENBLAS_NUM_THREADS=1`. `ferric` uses rayon for outer parallelism and pins
BLAS to a single thread inside rayon workers; letting OpenBLAS thread on top of
that oversubscribes the machine and produces unstable timings.
