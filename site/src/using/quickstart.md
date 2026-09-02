# Quick start

Both interfaces cover most methods: a TOML-driven CLI and Python bindings.

> Build first — see [Installation](./installation.md). `ferric` needs libint2
> built and on the linker path.

## CLI

Calculations are described by a TOML file. The `examples/` directory has one per
method.

```bash
# RHF on water with STO-3G
cargo run --release -- examples/water-rhf.toml

# RI-MP2 on water with cc-pVDZ / cc-pVDZ-RI
cargo run --release -- examples/water-rimp2.toml

# Attenuated RI-MP2 (short-range correlation only, r0 = 1.05 Å)
cargo run --release -- examples/water-attmp2.toml

# SCS-MP2 (Grimme spin-component scaling)
cargo run --release -- examples/water-scs-mp2.toml

# SCS-MP2(2terfc) (dual-attenuated, Goldey/Head-Gordon 2013)
cargo run --release -- examples/water-scs-mp2-2terfc.toml

# CCSD (H2/STO-3G)
cargo run --release -- examples/water-ccsd.toml

# LinLCCD(hh) — linearized hole-hole ladder CCD (closed-shell only)
cargo run --release -- examples/water-linlccd.toml

# wB97X-L-V — a double hybrid built on LinLCCD(hh) instead of MP2
cargo run --release -- examples/water-wb97xlv.toml
```

**CLI coverage is not complete.** Only `method.kind = "ccsd"` is wired for
coupled cluster; **CCD and CCSD(T) are library/Python-only**. Use
`ferric.run_ccd` / `ferric.run_ccsd_t` from Python until a CLI arm is added.

## Python

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
print(f"Att-MP2 total: {att.total_energy:.10f} Ha "
      f"(E_OS={att.e_os:.6f}, E_SS={att.e_ss:.6f})")

# SCS-MP2 (Grimme defaults)
scs = ferric.run_scs_mp2(mol, bs, aux)

# SCS-MP2(2terfc) — thesis defaults r0_1=0.75Å, r0_2=1.05Å, c_OS=1.27, c_SS=4.05
terfc = ferric.run_scs_mp2_2terfc(mol, bs, aux)

# Coupled cluster — RI-CCSD(T)
cc = ferric.run_ccsd_t(mol, bs, aux)
print(f"CCSD(T) total: {cc.correlation_energy + cc.t_correction:.10f} Ha")
```

See [Python bindings](./python.md) for the full surface and threading notes.

## Threading

Set `OPENBLAS_NUM_THREADS=1` when running tests or benchmarks. `ferric` uses
rayon for outer parallelism and pins BLAS to one thread inside rayon workers;
letting OpenBLAS thread on top of that oversubscribes the box and can produce
unstable timings.

```bash
OPENBLAS_NUM_THREADS=1 cargo test --workspace
```

For throughput across many independent jobs, prefer many single-threaded
processes over one multi-threaded job.
