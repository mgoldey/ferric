# ferric

Rust-native quantum chemistry engine wrapping libint2 for electron integrals, with pyo3 Python bindings.

<!-- ![build](https://img.shields.io/badge/build-passing-brightgreen) ![tests](https://img.shields.io/badge/tests-passing-brightgreen) -->

## Features

- **Closed-shell RHF** with DIIS convergence acceleration and Schwarz screening
- **Analytical RHF nuclear gradients** validated against finite differences
- **RI-MP2** (density-fitted MP2) via 3-center/2-center Coulomb integrals
- **OO-RI-MP2** (orbital-optimized RI-MP2) with level-shifted Newton, DIIS, Cayley rotations, and backtracking
- **Attenuated RI-MP2** with erfc(ωr)/r and terfc operators (Goldey & Head-Gordon, JPCL 2012)
- **SCS-MP2** spin-component scaled MP2 (Grimme, JCP 2003)
- **SCS-MP2(2terfc)** dual-attenuated SCS-MP2 (Goldey, Dutoi, Head-Gordon, PCCP 2013)
- **Canonical MP2** for cross-validation against RI-MP2
- **QQR screening** distance-dependent integral bounds with operator-aware decay (Maurer/Lambrecht/Ochsenfeld 2012)
- **LinK exchange** linear-scaling K matrix builder via pair-list-driven loops (Ochsenfeld/White/Head-Gordon 1998)
- **Spherical and Cartesian** basis set support (BSE-JSON and Gaussian-94 parsers)
- **Bundled basis sets**: STO-3G, 6-31G, cc-pVDZ, def2-SVP
- **Bundled RI auxiliary bases**: cc-pVDZ-RI, def2-SVP-RIFIT through def2-QZVPP-RIFIT
- **Python bindings** via pyo3 (RHF, RI-MP2, attenuated MP2, SCS-MP2, SCS-MP2(2terfc), RI-Laplace MP2, CCSD(T))
- **Coupled Cluster suite**: RI-CCD, RI-CCSD, and perturbative triples (T) correction
- **TOML-driven CLI** for all methods

## Mathematical Principles

### RI-Laplace MP2
The canonical MP2 correlation energy is given by:
$$E_{corr} = -\sum_{iajb} \frac{(ia|jb)[2(ia|jb) - (ib|ja)]}{\epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j}$$

Using the **Laplace transform** identity:
$$\frac{1}{x} = \int_0^\infty e^{-tx} dt \approx \sum_k w_k e^{-t_k x}$$

We can express the energy as a sum over quadrature points $t_k$. In the AO basis, we define **pseudo-density matrices** $P(t)$ and $Q(t)$:
$$P(t)_{\mu\nu} = \sum_{i \in occ} C_{\mu i} e^{t \epsilon_i} C_{\nu i}, \quad Q(t)_{\mu\nu} = \sum_{a \in vir} C_{\mu a} e^{-t \epsilon_a} C_{\nu a}$$

The energy is then computed via trace contractions of the 3-center RI integrals $B^P_{\mu\nu}$:
$$E_{corr} \approx -\sum_k w_k \sum_{PQ} \left[ 2 \text{Tr}(M^P N^Q) \text{Tr}(M^Q N^P) - \text{Tr}(M^P N^Q M^Q N^P) \right]$$
where $M^P = B^P P(t)$ and $N^P = B^P Q(t)$. This formulation enables **linear scaling** $O(N)$ when combined with sparse matrix algebra.

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

# SCS-MP2(2terfc) (dual-attenuated, Goldey/Head-Gordon 2013)
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

# Attenuated RI-MP2 (omega in Å⁻¹)
att = ferric.run_attenuated_rimp2(mol, bs, aux, omega=0.420)
print(f"Att-MP2 total: {att.total_energy:.10f} Ha (E_OS={att.e_os:.6f}, E_SS={att.e_ss:.6f})")

# SCS-MP2 (Grimme defaults)
scs = ferric.run_scs_mp2(mol, bs, aux)
print(f"SCS-MP2 total: {scs.total_energy:.10f} Ha")

# SCS-MP2(2terfc) (thesis defaults: r0_1=0.75A, r0_2=1.05A, c_OS=1.27, c_SS=4.05)
terfc = ferric.run_scs_mp2_2terfc(mol, bs, aux)
print(f"SCS-MP2(2terfc) total: {terfc.total_energy:.10f} Ha")

# Coupled Cluster (CCSD(T))
cc = ferric.run_ccsd_t(mol, bs, aux)
print(f"CCSD correlation: {cc.correlation_energy:.10f} Ha, (T) corr: {cc.t_correction:.10f} Ha")
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
# Set up the venv and install the extension in editable/develop mode
uv sync
uv run maturin develop --release

# Verify
uv run python -c "import ferric; print('OK')"
```

After `maturin develop`, the compiled `.so` is installed into `.venv/`. The `pyproject.toml` sets `[tool.uv] no-build-isolation-package = ["ferric"]` so that `uv run` skips re-invoking cargo and uses the existing `.so`.

**Important:** always use `uv run maturin develop --release`, not bare `maturin develop`. Without `uv run`, maturin targets whatever Python is on `$PATH` (e.g. pyenv's) and installs the `.so` into that Python's site-packages instead of the project `.venv`. The two copies are unrelated, so `uv run python` will keep loading the stale build.

**Normal dev loop:**
```bash
uv run maturin develop --release   # recompile and install .so into .venv
uv run python scripts/foo.py       # fast on subsequent runs — no recompile
```

**Optional: symlink for zero-copy updates**

If you want `cargo build --release` alone to update what Python sees (without running `maturin develop`), replace the installed `.so` with a symlink to `target/maturin/`:

```bash
ln -sf "$(pwd)/target/maturin/libferric.so" \
  .venv/lib/python3.11/site-packages/ferric/ferric.cpython-311-x86_64-linux-gnu.so
```

With this symlink, `cargo build --release` is sufficient — the `.so` in `.venv` always reflects the latest build. Note: `uv run maturin develop --release` overwrites the symlink with a copy; re-run `ln -sf` to restore it.

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

- [x] Rayon-parallel LinK exchange (Implemented)
- [x] CFMM (continuous fast multipole) for linear-scaling Coulomb
- [x] AO-Laplace-Transform MP2 (Linear Scaling via Sparse Tensors)
- [x] MPI distributed parallelization (Integrated across SCF/Gradients/MP2)
- [x] Geometry optimization using analytical gradients (RHF, RI-MP2, SCS-MP2)
- [x] ferric-tensors: Sparse tensor support implemented for linear correlation
- [x] DFT (LDA/GGA) via numerical quadrature
- [x] Coupled Cluster (CCD, CCSD, CCSD(T)) with RI-integral dressing

### Performance & Scaling Verification

`ferric` is designed for linear scaling ($O(N)$) for large systems using the LinK exchange builder and the CFMM Coulomb builder. You can verify these scaling properties using the provided benchmarking script:

```bash
# Run the alkane scaling benchmark (C10 to C50)
.venv/bin/python scripts/scaling_bench.py
```

The script benchmarks the scaling of **computed quartets vs system size** for the standard Coulomb operator, demonstrating how physical screening (Schwarz/QQR) reduces the computational effort from $O(N^4)$ toward $O(N)$ for large systems.

For correlation methods (like AO-Laplace-MP2), `ferric` leverages **attenuated Coulomb operators** to reduce systematic model errors (e.g., dispersion overestimation and BSSE) and to enable aggressive AO-based sparsity, which is the key to achieving $O(N)$ correlation scaling.

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
- Bartlett & Musiał, Rev. Mod. Phys. 79, 291 (2007) -- Coupled-cluster theory
- Scuseria, Janssen, Schaefer, J. Chem. Phys. 89, 7382 (1988) -- CCSD methods
- Raghavachari et al., Chem. Phys. Lett. 157, 479 (1989) -- CCSD(T) triples correction
