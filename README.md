# ferric

Rust-native quantum chemistry engine wrapping libint2 for electron integrals, with pyo3 Python bindings.

<!-- ![build](https://img.shields.io/badge/build-passing-brightgreen) ![tests](https://img.shields.io/badge/tests-passing-brightgreen) -->

## Features

- **Closed-shell RHF** with DIIS convergence acceleration and Schwarz screening
- **Analytical RHF nuclear gradients** validated against finite differences
- **RI-MP2** (density-fitted MP2) via 3-center/2-center Coulomb integrals
- **OO-RI-MP2** (orbital-optimized RI-MP2) with level-shifted Newton, DIIS, Cayley rotations, and backtracking
- **Attenuated RI-MP2** with erfc(ωr)/r operator (Goldey & Head-Gordon, JPCL 2012)
- **SCS-MP2** spin-component scaled MP2 (Grimme, JCP 2003)
- **SCS-MP2(2terfc)** dual-attenuated SCS-MP2 (Goldey, Dutoi, Head-Gordon, PCCP 2013)
- **Canonical MP2** for cross-validation against RI-MP2
- **QQR screening** distance-dependent integral bounds with operator-aware decay (Maurer/Lambrecht/Ochsenfeld 2012)
- **LinK exchange** linear-scaling K matrix builder via pair-list-driven loops (Ochsenfeld/White/Head-Gordon 1998)
- **Spherical and Cartesian** basis set support (BSE-JSON and Gaussian-94 parsers)
- **Bundled basis sets**: STO-3G, 6-31G, cc-pVDZ, def2-SVP
- **Bundled RI auxiliary bases**: cc-pVDZ-RI, def2-SVP-RIFIT through def2-QZVPP-RIFIT
- **Python bindings** via pyo3 (RHF, RI-MP2, attenuated MP2, SCS-MP2, SCS-MP2(2terfc))
- **TOML-driven CLI** for all methods

## Quick Example

### CLI

```bash
# RHF on water with STO-3G
cargo run --release -- examples/water-rhf.toml

# RI-MP2 on water with cc-pVDZ / cc-pVDZ-RI
cargo run --release -- examples/water-rimp2.toml

# Attenuated RI-MP2 (short-range correlation only, r0=1.05 A)
cargo run --release -- examples/water-attmp2.toml

# SCS-MP2 (Grimme spin-component scaling)
cargo run --release -- examples/water-scs-mp2.toml

# SCS-MP2(2terfc) (dual-attenuated, Goldey/Dutoi/Head-Gordon 2013)
cargo run --release -- examples/water-scs-mp2-2terfc.toml
```

### Python

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs  = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

# Standard RI-MP2
mp2 = ferric.run_rimp2(mol, bs, aux)
print(f"RI-MP2 total: {mp2.total_energy:.10f} Ha")

# Attenuated RI-MP2 (r0 in Angstrom)
att = ferric.run_attenuated_rimp2(mol, bs, aux, r0=1.05)
print(f"Att-MP2 total: {att.total_energy:.10f} Ha (E_OS={att.e_os:.6f}, E_SS={att.e_ss:.6f})")

# SCS-MP2 (Grimme defaults)
scs = ferric.run_scs_mp2(mol, bs, aux)
print(f"SCS-MP2 total: {scs.total_energy:.10f} Ha")

# SCS-MP2(2terfc) (thesis defaults: r0_1=0.75A, r0_2=1.05A, c_OS=1.27, c_SS=4.05)
terfc = ferric.run_scs_mp2_2terfc(mol, bs, aux)
print(f"SCS-MP2(2terfc) total: {terfc.total_energy:.10f} Ha")
```

## Architecture

```
                  +------------------+
                  |   ferric-cli     |   TOML config -> all methods
                  +--------+---------+
                           |
         +-----------------+------------------+
         |                 |                  |
+--------v------+  +-------v-------+  +-------v--------+
|  ferric-scf   |  |  ferric-mp2   |  | ferric-python  |
|  RHF, DIIS,   |  |  RI-MP2,      |  | pyo3 bindings  |
|  gradients,   |  |  OO-RI-MP2,   |  +-------+--------+
|  QQR, LinK,   |  |  attenuated,  |          |
|  Schwarz      |  |  SCS, 2terfc  |          |
+-------+-------+  +------+--------+          |
        |                  |                   |
        +--------+---------+-------------------+
                 |
        +--------v--------+
        | ferric-integrals |   libint2 FFI: Coulomb, erf, erfc operators,
        | shim/shim.cc     |   1e/2e/3-center/2-center, derivatives
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
| H2O | cc-pVDZ | Att-RI-MP2 (r₀=1.05Å) | -76.2102635714 |
| H2O | cc-pVDZ | SCS-MP2 | -76.2268940016 |
| H2O | cc-pVDZ | SCS-MP2(2terfc) | -76.2151715715 |
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
    ferric-scf/                 # RHF solver, DIIS, Fock build, gradients, QQR, LinK
    ferric-mp2/                 # RI-MP2, OO-RI-MP2, attenuated, SCS, canonical
    ferric-cli/                 # TOML-driven command-line driver
    ferric-python/              # pyo3 Python bindings
  testdata/
    molecules/                  # XYZ files (water, methane)
    reference/                  # PySCF reference energies (JSON)
  examples/                     # TOML input files
```

## Roadmap

- [ ] Rayon-parallel LinK exchange (single-threaded LinK implemented, parallelism pending)
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
- Goldey & Head-Gordon, J. Phys. Chem. Lett. 3, 3592 (2012) -- Attenuated MP2
- Goldey, Dutoi, Head-Gordon, Phys. Chem. Chem. Phys. 15, 15869 (2013) -- SCS-MP2(2terfc)
- Grimme, J. Chem. Phys. 118, 9095 (2003) -- SCS-MP2
- Maurer, Lambrecht, Ochsenfeld, J. Chem. Phys. 136, 144107 (2012) -- QQR screening
- Ochsenfeld, White, Head-Gordon, J. Chem. Phys. 109, 1663 (1998) -- LinK exchange
