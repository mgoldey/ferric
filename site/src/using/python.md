# Python bindings

The pyo3 bindings expose most of the library. Build them with
`maturin develop --release` — see [Installation](./installation.md).

## Basics

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs  = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")
```

Bundled orbital bases include STO-3G, 6-31G, cc-pVDZ and def2-SVP; bundled
auxiliary bases include cc-pVDZ-RI, the def2-\*-RIFIT family, and
def2-universal-jkfit. Both BSE-JSON and Gaussian-94 basis files can be parsed
from disk.

## Ground state

```python
rhf = ferric.run_rhf(mol, bs)
print(f"RHF: {rhf.energy:.10f} Ha, converged={rhf.converged}")

uhf  = ferric.run_uhf(mol, bs, multiplicity=3)
rohf = ferric.run_rohf(mol, bs, multiplicity=3)
dft  = ferric.run_dft(mol, bs, functional="b3lyp")
```

**Always check `.converged`.** These functions return a result whether or not the
SCF converged — a non-converged result is not an error, it is a result with
`converged = False`. Treating it as success is a common way to get a plausible,
wrong number.

## Correlation

```python
mp2   = ferric.run_rimp2(mol, bs, aux)
att   = ferric.run_attenuated_rimp2(mol, bs, aux, omega=0.420)   # Å⁻¹
scs   = ferric.run_scs_mp2(mol, bs, aux)
terfc = ferric.run_scs_mp2_2terfc(mol, bs, aux)

ccd    = ferric.run_ccd(mol, bs, aux)
ccsd   = ferric.run_ccsd(mol, bs, aux)
ccsd_t = ferric.run_ccsd_t(mol, bs, aux)
```

`run_ccd` and `run_ccsd_t` are **Python-only** — they are not yet CLI-wired.

## Response and excited states

```python
gw    = ferric.run_gw(mol, bs, aux)      # G0W0 quasiparticle energies
tddft = ferric.run_tddft(mol, bs, aux)   # TDA or Casida
```

`run_tddft` warns on stderr when the reference is not pure Hartree–Fock: the
`(ia|f_xc|jb)` XC-kernel response is unimplemented, so with a DFT reference the
excitation energies omit a physical term and are approximate. Only
`c_hf = 1.0` (CIS/TDHF) is exact.

## Memory budgets

Most drivers accept `memory_budget_gb`. The budget is enforced, not advisory:
a job whose predicted peak exceeds it is **refused** with a breakdown naming the
dominant term, rather than being OOM-killed partway through.

```python
mp2 = ferric.run_rimp2(mol, bs, aux, memory_budget_gb=8.0)
```

## Threading and concurrency

The compute drivers release the GIL, so independent jobs submitted from a
`ThreadPoolExecutor` genuinely run in parallel rather than serializing at the
FFI boundary.

Set `OPENBLAS_NUM_THREADS=1`. `ferric` uses rayon for outer parallelism and pins
BLAS to one thread inside rayon workers; for throughput across many jobs, prefer
many single-threaded processes over one wide job.

## Property export

ESP at nuclei, electric fields, static and atom-partitioned polarizabilities,
Hirshfeld and Löwdin charges, and density matrices are available, with **NPZ
export** of ML-ready features (MO coefficients, orbital energies, PDEP
eigenvectors, ESP, polarizability tensors, charges) for downstream model
conditioning.

For the full signature list, see the [API documentation](../reference/api.md).
