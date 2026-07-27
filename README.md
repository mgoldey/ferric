# ferric

Rust-native quantum chemistry engine wrapping libint2 for electron integrals, with pyo3 Python bindings.

<!-- ![build](https://img.shields.io/badge/build-passing-brightgreen) ![tests](https://img.shields.io/badge/tests-passing-brightgreen) -->

## The idea behind it

`ferric` is organized around **electronic response** — how the density reacts to a
perturbation, the object that appears as the polarizability α, the dielectric function ε,
and the susceptibility χ. The methods here are, at heart, three faces of getting that one
object right where standard methods get it wrong:

- **Attenuated MP2** — MP2 builds dispersion from an *uncoupled* polarizability that
  over-polarizes (too-large C₆, overestimated π-stacking); attenuating the correlation
  operator tames that response error with a single tunable parameter.
- **PDEP-RPA / GW** — the dielectric matrix *is* the density–density response; PDEP keeps
  only its dominant low-rank eigenmodes, so RPA correlation and the GW screened
  interaction need no explicit sum over empty states.
- **Constrained DFT** — a constraint couples to the density and reads its response
  (∂N/∂λ is a susceptibility), building charge-localized diabatic states and their
  electron-transfer couplings.

The computational payoff falls out of the physics: **response is local in real space and
low-rank in its eigenspectrum, so organizing around it makes the computation sparse** —
attenuate the operator, keep the dominant dielectric modes, exploit sparse Laplace
pseudo-densities. Get the physics right, and the cheap method follows from its structure.

## Features

**Self-consistent field**
- **RHF** (closed-shell) with DIIS, Schwarz/QQR screening, and a choice of direct or density-fitted (RI-J / RI-K) Fock builds
- **UHF / ROHF** (open-shell) with per-spin DIIS, virtual-space level shifting, augmented-Hessian Newton, and Maximum-Overlap-Method (MOM) orbital tracking for near-degenerate cases
- **Kohn–Sham DFT** — closed- and open-shell (RKS / UKS / ROKS) via libxc: LDA/GGA/hybrid/range-separated-hybrid functionals (LDA, PBE, B3LYP, ωB97X-V) on Becke–Lebedev grids, with VV10 nonlocal correlation
- **Analytical nuclear gradients** for RHF, UHF, ROHF and KS-DFT (incl. grid response), validated against finite differences

**Correlation (MP2 family)**
- **RI-MP2** (density-fitted) via 3-center/2-center integrals; **canonical MP2** for cross-validation
- **OO-RI-MP2** (orbital-optimized) with level-shifted Newton, orbital DIIS, Cayley rotations, backtracking
- **Attenuated RI-MP2** with erfc(ωr)/r and terfc operators (Goldey & Head-Gordon, JPCL 2012)
- **SCS-MP2** (Grimme, JCP 2003) and **SCS-MP2(2terfc)** dual-attenuated (Goldey, Dutoi, Head-Gordon, PCCP 2013)
- **RI-Laplace MP2** — O(N) AO-Laplace formulation via sparse pseudo-density tensors

**Coupled cluster**
- **RI-CCD, RI-CCSD, and the perturbative triples (T)** correction — all
  validated against exact-integral / PySCF references (H2O/cc-pVDZ (T) matches
  PySCF to ~1e-6). See [docs/VALIDATION.md](docs/VALIDATION.md).
- CLI: only `method.kind = "ccsd"` is currently wired (see
  `examples/water-ccsd.toml`). **CCD and CCSD(T) are library/Python-only, not
  yet CLI-wired** — use `ferric.run_ccd` / `ferric.run_ccsd_t` from Python (see
  the Quick Example below) until a CLI arm is added.

**Many-body response (RPA & GW)**
- **PDEP-RPA** — RPA correlation via projective dielectric-eigenpotentials (a low-rank W basis in Gaussians), closed- and open-shell (U-PDEP-RPA over a spin-summed dielectric)
- **GW quasiparticle energies** — G0W0, COHSEX, evGW0, evGW (and unrestricted U-GW); G0W0@HF matches MOLGW to ~5 meV
- **Attenuated RPA** (short-range correlation via erfc)

**Constrained DFT (electron transfer)**
- **cDFT** — fragment charge/spin constraints via a grid-Becke weight operator and a nested Lagrange-multiplier solve (Wu–Van Voorhis)
- **Electron-transfer coupling H_ab** — diabatic-state coupling via non-orthogonal-determinant overlap (Löwdin biorthogonalization)

**Properties & ML export**
- ESP-at-nuclei, electric field, static and atom-partitioned polarizabilities, Hirshfeld and Löwdin charges, density matrices
- **NPZ export** of ML-ready features (MO coefficients, orbital energies, PDEP eigenvectors, ESP, polarizability tensors, charges) for downstream generative-model conditioning

**Infrastructure**
- **QQR screening** (Maurer/Lambrecht/Ochsenfeld 2012) and **LinK exchange** (Ochsenfeld/White/Head-Gordon 1998) for linear-scaling Fock builds; CFMM Coulomb
- **Spherical and Cartesian** basis support (BSE-JSON and Gaussian-94 parsers); bundled orbital bases (STO-3G, 6-31G, cc-pVDZ, def2-SVP) + RI/JK auxiliary bases (cc-pVDZ-RI, def2-\*-RIFIT, def2-universal-jkfit)
- **Python bindings** (pyo3) and a **TOML-driven CLI** for all methods

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

# CCSD (H2/STO-3G; CCD and CCSD(T) are not yet CLI-wired, use Python)
cargo run --release -- examples/water-ccsd.toml

# LinLCCD(hh) -- linearized hole-hole ladder CCD (closed-shell only)
cargo run --release -- examples/water-linlccd.toml

# wB97X-L-V -- double hybrid built on LinLCCD(hh) instead of MP2.
# Converges its own wB97X-L-V KS reference, then adds the short-range
# LinLCCD(hh) correction. [dft] lambda/omega override the published
# 0.6 / 0.1 Bohr^-1; omitting them gives the published values.
cargo run --release -- examples/water-wb97xlv.toml
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

# Coupled Cluster — RI-CCSD(T) (validated vs exact-integral / PySCF refs).
cc = ferric.run_ccsd_t(mol, bs, aux)
print(f"CCSD correlation: {cc.correlation_energy:.10f} Ha")
print(f"(T) correction:   {cc.t_correction:.10f} Ha")
print(f"CCSD(T) total:    {cc.correlation_energy + cc.t_correction:.10f} Ha")
```

## Tutorials

Step-by-step, runnable walkthroughs (CLI + Python, with verified output and
per-method maturity badges) live in [`docs/guide/tutorials/`](docs/guide/tutorials/00-index.md):

1. [Your first calculation: RHF on water](docs/guide/tutorials/01-first-calculation.md)
2. [Energies you can trust: the MP2 family](docs/guide/tutorials/02-mp2-family.md)
3. [DFT calculations](docs/guide/tutorials/03-dft.md)
4. [Open-shell systems](docs/guide/tutorials/04-open-shell.md)
5. [Geometry optimization](docs/guide/tutorials/05-geometry-optimization.md)
6. [Dispersion C6 and polarizabilities](docs/guide/tutorials/06-dispersion-c6.md)
7. [Exporting ML features (NPZ)](docs/guide/tutorials/07-ml-feature-export.md)
8. [Batches and scaling](docs/guide/tutorials/08-batches-and-scaling.md)

For the **theory behind the methods** — Hartree–Fock, MP2/RI, why MP2 fails for
non-covalent interactions, attenuated MP2 and the terfc operator,
SCS-MP2(2terfc), and the DFT/RPA/GW response methods (drawing on the developer's
dissertation) — see the [methods guide](docs/guide/methods/00-index.md).

## Architecture

```
                          +------------------+
                          |   ferric-cli     |   TOML config -> all methods
                          +--------+---------+   (+ ferric-python: pyo3 bindings)
                                   |
   +-----------+-----------+-------+------+-----------+------------+
   |           |           |              |           |            |
+--v----+ +----v----+ +----v----+   +-----v----+ +----v-----+ +---v------+
|ferric | |ferric   | |ferric   |   |ferric    | |ferric    | |ferric    |
|-scf   | |-mp2     | |-dft     |   |-rpa      | |-gw       | |-cc       |
|RHF/UHF| |RI-MP2,  | |RKS/UKS/ |   |PDEP-RPA, | |G0W0,     | |CCD/CCSD/ |
|/ROHF, | |OO,att,  | |ROKS,    |   |U-PDEP,   | |COHSEX,   | |(T)       |
|KS-DFT,| |SCS,     | |libxc,   |   |response  | |evGW,     | +----------+
|DIIS,  | |2terfc,  | |Becke    |   |props,    | |U-GW      |
|MOM,AH,| |Laplace  | |grids,   |   |ESP/Hirsh/| +-----+----+
|cDFT,  | +----+----+ |VV10     |   |NPZ export|       |
|grads  |      |      +----+----+   +-----+----+       |
+---+---+      |           |              |            |
    |          +-----+-----+------+-------+------------+
    |                |     |      |
    |   +------------v--+ +v------v-----+   ferric-tensors (sparse), 
    |   |ferric-export | |ferric-      |   ferric-quadrature (Laplace/grid roots)
    |   |cube,NPZ,GTO  | |integrals    |   support crates
    |   +--------------+ |libint2 FFI  |
    |                    |shim/shim.cc |   Coulomb/erf/erfc, 1e/2e/3c/2c, derivs
    +--------+-----------+------+------+
             |                  |
        +----v------------------v----+
        |        ferric-core         |   Molecule, BasisSet, Shell, elements,
        |                            |   BSE-JSON / G94 parsers, bundled bases
        +----------------------------+
```

## Installation

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- libint2 2.7+ built from the [mpqc4 tarball](https://github.com/evaleev/libint/releases/download/v2.7.2/libint-2.7.2-mpqc4.tgz) (includes derivative and RI support)
- OpenBLAS and LAPACK
- Eigen3 headers
- Python 3.10+ and maturin (for Python bindings, optional)
- For the optional `mpi` feature only: an MPI implementation (OpenMPI/MPICH) **and**
  libclang (`libclang-dev`, for `mpi-sys`'s bindgen step) — see
  [Optional: distributed-memory MPI](#optional-distributed-memory-mpi---features-mpi) below

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

### Debug vs. Release builds

```bash
# Debug build (fast to compile, slow to run) -- for iterating on Rust code
# and catching debug_assert!/overflow bugs during development
cargo build --workspace
cargo run -- examples/water-rhf.toml

# Release build (slow to compile, fast to run) -- for anything you'll
# actually wait on: real molecules, benchmarks, RPA/GW/CC jobs
cargo build --release --workspace
cargo run --release -- examples/water-rhf.toml
```

A debug SCF/RPA/GW run can be one to two orders of magnitude slower than
release (no LTO/opt, plus active overflow and `debug_assert!` checks) --
debug builds are for compile-edit-test loops on small systems (H2/STO-3G,
water), not for anything you'd want a real energy from. `cargo test
--workspace` runs against debug builds by default; add `--release` if a slow
test needs it. Always set `OPENBLAS_NUM_THREADS=1` for both build kinds when
running tests or jobs (see [Testing](#testing) below) -- BLAS>1 under this
crate's rayon parallelism is known to segfault or slow down significantly.

### Optional: distributed-memory MPI (`--features mpi`)

MPI support (distributed DF-JK aux-band striping across ranks/nodes) is behind
the optional `mpi` Cargo feature and is **off by default** — a normal build
needs none of the packages below.

To build `--features mpi` you need **two** things:

1. **An MPI implementation** (OpenMPI or MPICH) providing `mpicc`, `mpirun`,
   `mpi.h`, and `libmpi.so`. On Ubuntu/Mint:
   ```bash
   sudo apt-get install -y libopenmpi-dev openmpi-bin
   ```
   (A user-local OpenMPI on `PATH` works too — `rsmpi`/`mpi-sys` discovers it via
   `mpicc`.)

2. **libclang** — the `mpi-sys` crate runs `bindgen` over `mpi.h` at build time,
   which needs libclang's shared library. **This is the piece most often
   missing** (the MPI runtime can be present while libclang is not, giving
   `Unable to find libclang` from the `mpi-sys` build script). Prefer the distro
   package — it ships both the library and clang's builtin headers:
   ```bash
   sudo apt-get install -y libclang-dev
   ```
   If bindgen still can't find the library, point it at it explicitly:
   ```bash
   export LIBCLANG_PATH=$(dirname "$(find /usr/lib -name 'libclang.so*' | head -1)")
   ```
   *No sudo?* The `pip install --user libclang` wheel provides `libclang.so`, but
   it does **not** bundle clang's builtin headers, so bindgen then fails with
   `'stddef.h' file not found`. Point it at GCC's builtin headers to fix that:
   ```bash
   pip install --user libclang
   export LIBCLANG_PATH="$(python3 -c 'import clang,os;print(os.path.join(os.path.dirname(clang.__file__),"native"))')"
   export BINDGEN_EXTRA_CLANG_ARGS="-I$(dirname "$(find /usr/lib/gcc -name stddef.h | head -1)")"
   ```

Then build and run under `mpirun` (keep OpenBLAS single-threaded; see
`docs/superpowers/mpi.md` for thread-layout guidance):

```bash
OPENBLAS_NUM_THREADS=1 cargo build --release --workspace --features mpi
mpirun -np 4 -x OPENBLAS_NUM_THREADS=1 -x RAYON_NUM_THREADS=4 \
    target/release/ferric input.toml
```

**A user-local (non-system) OpenMPI install needs `LD_LIBRARY_PATH`**, or the
built binary fails at launch with `error while loading shared libraries:
libmpi.so.40: cannot open shared object file` even though it linked and
compiled fine (the linker found `libmpi.so` via `mpicc`'s search path at
build time; the dynamic loader does not use that same path at run time). If
`mpirun`/`mpicc` resolve to a path under your home directory (e.g.
`~/.local/bin/mpirun`, check with `which mpirun`) rather than
`/usr/bin/mpirun`, set:
```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
```
before running `mpirun` (both for `target/release/ferric` and for any
`--features mpi`-gated test binary launched directly, e.g.
`mpi_dfjk_banding`). Not needed for a distro-packaged
`/usr/lib/.../libopenmpi` install, where the loader already knows the path.

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

If you want `cargo build --release` alone to update what Python sees (without running `maturin develop`), replace the installed `.so` with a symlink to **`target/release/libferric.so`** — the artifact `cargo build` actually refreshes:

```bash
ln -sf "$(pwd)/target/release/libferric.so" \
  .venv/lib/python3.11/site-packages/ferric/ferric.cpython-311-x86_64-linux-gnu.so
```

With this symlink, `cargo build --release` is sufficient — the `.so` in `.venv` always reflects the latest build.

> **Do not symlink to `target/maturin/libferric.so`.** That directory is only
> written by `maturin develop`, *not* by `cargo build`, so a symlink there goes
> stale after a plain `cargo build` — and if a `maturin develop` run fails, the
> file there can be truncated to 0 bytes, breaking the import. Symlink to
> `target/release/` instead.

Note: `uv run maturin develop --release` overwrites the symlink with a fresh copy of the build; re-run the `ln -sf` above to restore the symlink if you want zero-copy updates again.

## Testing

```bash
# All workspace tests (OPENBLAS_NUM_THREADS=1 -- see note above)
OPENBLAS_NUM_THREADS=1 cargo test --workspace

# Specific crate
OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf

# With output (shows energies and convergence info)
OPENBLAS_NUM_THREADS=1 cargo test --workspace -- --nocapture
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
      src/basis/bundled/        # Embedded BSE-JSON basis set files (orbital + RI/JK aux)
    ferric-integrals/           # libint2 FFI: 1e, 2e, 3-center, 2-center, derivatives
      shim/shim.{h,cc}         # C++ shim calling libint2 API
    ferric-scf/                 # RHF/UHF/ROHF + KS-DFT solvers, DIIS, MOM, AH-Newton,
                                #   Fock builds (direct/DF), gradients, QQR, LinK, cDFT
    ferric-dft/                 # libxc bridge, Becke-Lebedev grids, Vxc, VV10, cDFT weights
    ferric-mp2/                 # RI-MP2, OO-RI-MP2, attenuated, SCS, canonical, Laplace
    ferric-cc/                  # RI-CCD, RI-CCSD, CCSD(T) perturbative triples
    ferric-rpa/                 # PDEP-RPA (closed/open-shell), response properties,
                                #   ESP / Hirshfeld / Löwdin / polarizability
    ferric-gw/                  # G0W0, COHSEX, evGW0, evGW, U-GW (PDEP-as-W)
    ferric-tensors/             # Sparse tensor support (linear-scaling correlation)
    ferric-quadrature/          # Laplace / grid quadrature roots and weights
    ferric-export/              # Cube files, NPZ ML-feature export, GTO grid eval
    ferric-cli/                 # TOML-driven command-line driver
    ferric-python/              # pyo3 Python bindings
  testdata/
    molecules/                  # XYZ files (water, methane, ...)
    reference/                  # PySCF/MOLGW reference values (JSON)
  examples/                     # TOML input files
  docs/superpowers/             # design specs + implementation plans
```

## Roadmap

> **Implemented ≠ validated.** A checked box means the code exists and runs. For
> how strongly each capability's *numbers* are checked against ground truth —
> and where they are known to fail — see [docs/VALIDATION.md](docs/VALIDATION.md).

- [x] Rayon-parallel LinK exchange
- [x] CFMM (continuous fast multipole) for linear-scaling Coulomb
- [x] AO-Laplace-Transform MP2 (linear scaling via sparse tensors)
- [x] MPI distributed parallelization (SCF DF-J/K, RI-MP2, RPA frequency quadrature; GW QP loop not yet distributed — see `docs/superpowers/mpi.md`)
- [x] Geometry optimization via analytical gradients (RHF, RI-MP2, SCS-MP2)
- [x] Sparse tensor support (ferric-tensors) for linear correlation
- [x] KS-DFT (LDA/GGA/hybrid/RSH) via libxc + Becke-Lebedev quadrature; VV10 nonlocal
- [x] Coupled Cluster: RI-CCD, RI-CCSD, CCSD(T) (validated vs exact-integral / PySCF refs)
- [x] Open-shell SCF: UHF, ROHF, UKS, ROKS (with MOM + augmented-Hessian Newton)
- [x] Open-shell analytical gradients (UHF/ROHF/KS-DFT, incl. grid response)
- [x] PDEP-RPA correlation (closed- and open-shell) + attenuated RPA
- [x] GW quasiparticle energies (G0W0, COHSEX, evGW0, evGW, U-GW)
- [x] Constrained DFT + electron-transfer couplings (Wu-Van Voorhis H_ab)
- [x] Response properties + NPZ ML-feature export (ESP, polarizability, charges, PDEP)

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
