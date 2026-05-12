# ferric

Rust-native quantum chemistry engine wrapping libint2 for electron integrals, with pyo3 Python bindings.

<!-- ![build](https://img.shields.io/badge/build-passing-brightgreen) ![tests](https://img.shields.io/badge/tests-passing-brightgreen) -->

## Features

- **Closed-shell RHF** with DIIS convergence acceleration and Schwarz screening
- **Analytical RHF nuclear gradients** validated against finite differences
- **RI-MP2** (density-fitted MP2) via 3-center/2-center Coulomb integrals
- **OO-RI-MP2** (orbital-optimized RI-MP2) with level-shifted Newton, DIIS, Cayley rotations, and backtracking
- **Canonical MP2** for cross-validation against RI-MP2
- **Spherical and Cartesian** basis set support (BSE-JSON and Gaussian-94 parsers)
- **Bundled basis sets**: STO-3G, 6-31G, cc-pVDZ, def2-SVP
- **Bundled RI auxiliary bases**: cc-pVDZ-RI, def2-SVP-RIFIT through def2-QZVPP-RIFIT
- **Python bindings** via pyo3 (Molecule, BasisSet, run_rhf, run_rimp2)
- **TOML-driven CLI** for RHF, RI-MP2, and OO-RI-MP2 calculations

## Quick Example

### CLI

```bash
# RHF on water with STO-3G
cargo run --release -- examples/water-rhf.toml

# RI-MP2 on water with cc-pVDZ / cc-pVDZ-RI
cargo run --release -- examples/water-rimp2.toml
```

### Python

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs  = ferric.BasisSet.bundled("sto-3g")

rhf = ferric.run_rhf(mol, bs)
print(f"RHF energy: {rhf.energy:.10f} Ha")  # -74.9631468000 Ha

aux = ferric.BasisSet.bundled("cc-pvdz-ri")
bs2 = ferric.BasisSet.bundled("cc-pvdz")
mp2 = ferric.run_rimp2(mol, bs2, aux)
print(f"RI-MP2 total: {mp2.total_energy:.10f} Ha")  # -76.2308014548 Ha
```

## Architecture

```
                  +------------------+
                  |   ferric-cli     |   TOML config -> RHF / RI-MP2 / OO-RI-MP2
                  +--------+---------+
                           |
         +-----------------+------------------+
         |                 |                  |
+--------v------+  +-------v-------+  +-------v--------+
|  ferric-scf   |  |  ferric-mp2   |  | ferric-python  |
|  RHF, DIIS,   |  |  RI-MP2,      |  | pyo3 bindings  |
|  gradients,   |  |  OO-RI-MP2,   |  +-------+--------+
|  Schwarz      |  |  canonical    |          |
+-------+-------+  +------+--------+          |
        |                  |                   |
        +--------+---------+-------------------+
                 |
        +--------v--------+
        | ferric-integrals |   libint2 FFI: overlap, kinetic, nuclear,
        | shim/shim.cc     |   2e ERIs, 3-center, 2-center, derivatives
        +--------+--------+
                 |
        +--------v--------+
        |   ferric-core    |   Molecule, BasisSet, Shell, elements,
        |                  |   BSE-JSON / G94 parsers, bundled bases
        +------------------+
```

## Installation

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- libint2 2.7+ built from the [mpqc4 tarball](https://github.com/evaleev/libint/releases/download/v2.7.2/libint-2.7.2-mpqc4.tgz) (includes derivative and RI support)
- OpenBLAS and LAPACK
- Eigen3 headers
- Python 3.10+ and maturin (for Python bindings, optional)

### Building from Source

```bash
# Install system dependencies (Ubuntu 22.04+)
sudo apt-get install -y build-essential cmake g++ gfortran wget \
    libeigen3-dev libopenblas-dev liblapack-dev pkg-config \
    python3-dev python3-pip python3-venv

# Build and install libint2 (takes ~30 min)
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
cargo test --workspace
```

### Python Bindings

```bash
cd crates/ferric-python
python3 -m venv .venv
source .venv/bin/activate
pip install maturin numpy
maturin develop --release
python -c "import ferric; print('OK')"
```

## Testing

```bash
# All workspace tests
cargo test --workspace

# Specific crate
cargo test -p ferric-scf

# With output (shows energies and convergence info)
cargo test --workspace -- --nocapture
```

Reference energies validated against PySCF to at least 1e-8 Hartree:

| System | Basis | Method | Energy (Ha) |
|--------|-------|--------|-------------|
| H2O | STO-3G | RHF | -74.9631468000 |
| H2O | cc-pVDZ | RI-MP2 | -76.2308014548 |
| CH4 | cc-pVDZ | RHF | -40.1987085425 |

## Project Structure

```
ferric/
  Cargo.toml                    # Workspace root
  crates/
    ferric-core/                # Molecule, BasisSet, elements, parsers
      src/basis/bundled/        # Embedded BSE-JSON basis set files
    ferric-integrals/           # libint2 FFI: 1e, 2e, 3-center, derivatives
      shim/shim.{h,cc}         # C++ shim calling libint2 API
    ferric-scf/                 # RHF solver, DIIS, Fock build, gradients
    ferric-mp2/                 # RI-MP2, OO-RI-MP2, canonical MP2
    ferric-cli/                 # TOML-driven command-line driver
    ferric-python/              # pyo3 Python bindings
  testdata/
    molecules/                  # XYZ files (water, methane)
    reference/                  # PySCF reference energies (JSON)
  examples/                     # TOML input files
```

## Roadmap

- [ ] Rayon-parallel J/K build (LinK for exchange)
- [ ] CFMM (continuous fast multipole) for Coulomb
- [ ] AO-Laplace-Transform MP2
- [ ] MPI distributed RI-MP2 via 2D block-cyclic tensor distribution
- [ ] Geometry optimization using analytical gradients
- [ ] DFT (LDA/GGA) via numerical quadrature

## License

Apache-2.0

## References

- [libint2](https://github.com/evaleev/libint) -- Obara-Saika integral engine
- [pyo3](https://pyo3.rs/) -- Rust/Python interop
- [ndarray](https://docs.rs/ndarray) -- N-dimensional arrays for Rust
- [ndarray-linalg](https://docs.rs/ndarray-linalg) -- LAPACK bindings for ndarray
- Szabo & Ostlund, *Modern Quantum Chemistry* (1996)
- Pulay, Chem. Phys. Lett. 73, 393 (1980) -- DIIS convergence acceleration
- Weigend, Phys. Chem. Chem. Phys. 4, 4285 (2002) -- RI-MP2 auxiliary basis sets
- Bozkaya & Sherrill, J. Chem. Phys. 135, 104103 (2011) -- Orbital-optimized MP2
