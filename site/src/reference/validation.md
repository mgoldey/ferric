# What is validated

> **Implemented ≠ validated.** Working code is not a checked number.

Read this page before trusting any result. The other pages say what *exists*;
this one says how far each capability's numbers have been checked against an
independent reference, and where they are known to fail.

## Grades

Every CLI `method.kind` has a grade. The same grades are printed as a warning by
the CLI at run time for anything below Proven. The per-method table, with
Python-only capabilities included, is in [Capabilities](./capabilities.md).

| Grade | Meaning |
|---|---|
| **Proven** | Total energies (or the stated property) agree with an independent code or an exact limit, pinned by tests |
| **Proven (narrow)** | As Proven, but only on the stated class of systems |
| **Smoke** | Runs end to end, and its parts or limits are checked, but no independent reference for the headline number, or only one loose one |
| **Spike** | New code with no comparison to a reference yet; for exploration only |

**Proven:** `rhf`, `uhf`, `rohf`, `ksdft`, `rimp2`, `mp3`, `att-rimp2`,
`scs-mp2`, `scs-mp2-2terfc`, `laplace-mp2`, `pdep-rpa`, `ccsd`, `linlccd`.

**Smoke or Spike**, with the caveat the CLI prints:

| `method.kind` | Grade | What is and is not checked |
|---|---|---|
| `gw` | Smoke | ~5 meV vs MOLGW on one H2O/cc-pVDZ case; most assertions are range bands. Treat results as ±0.3 eV. |
| `bse-tda` | Smoke | Only excitation ordering and a physicality gate; inherits the GW gap error. |
| `tdhf-static-polarizability` | Smoke | Static α close to DOSD for water (one case); the same kernel gives C6 ~63% low. |
| `rs-mp2-rpa` | Smoke | The ω→0 and ω→∞ limits reduce exactly to MP2 and MP2+dRPA; production ω is unproven on new systems. |
| `mp2-v` | Smoke | VV10 half bit-identical to the ωB97X-V path; no published MP2-V total energy to compare against. Defaults are fitted for aug-cc-pVTZ, no counterpoise, frozen core. |
| `oo-rimp2` | Smoke | Stationarity and vanishing orbital gradient checked; no external absolute reference. |
| `wb97x-l-v` | Smoke | Components and limits checked; no reference for the total energy. |
| `b2plyp`, `dsd-pbep86` | Spike | No comparison to a reference code yet. |
| `tda`, `tddft` | Spike | Exact for an HF reference (CIS / TDHF); DFT references omit the f<sub>xc</sub> kernel. |

## Anchors

Where numbers are checked, they are checked against external references or
exact limits, not against ferric's own earlier output. "Stated agreement" is
the measured difference recorded in the repository; "test tolerance" is what
the pinning test actually asserts, which is often looser.

| Capability | System / basis | Reference | Stated agreement | Test tolerance | Pinned by |
|---|---|---|---:|---:|---|
| Closed-shell (T) | H2O / cc-pVDZ | PySCF `ccsd_t()` | ~1e-6 Ha | 1e-4 Ha | `ferric-cc` `closed_shell_t_h2o_ccpvdz_matches_pyscf` |
| G0W0@HF | H2O / cc-pVDZ | MOLGW | ~5 meV | 0.30 eV | `ferric-gw/tests/h2o_g0w0_cohsex.rs` |
| RI-MP2 size-extensivity | H2 dimer at large separation | 2 × monomer | 2e-12 Ha | 1e-7 Ha | `ferric-mp2/tests/rimp2_size_extensivity.rs` |
| RHF/UHF/ROHF/KS gradients | several | finite differences of the energy | — | per test | `ferric-scf` gradient tests |
| COSX exchange, dense-grid limit | water / cc-pVDZ | direct K | 3.3e-7 | — | see [SCF: choosing how exchange is built](../methods/scf.md) |
| COSX SCF energy, (50,110)+fit | water / cc-pVDZ | direct K | 4.9e-6 Ha | — | ″ |
| COSX SCF energy, (50,110)+fit | butane / def2-SVP | direct K | 1.7e-4 Ha | — | ″ |
| COSX SCF energy, (50,110)+fit | butane / def2-TZVP | direct K | 1.2e-4 Ha | — | ″ |
| COSX, open shell | CH3 doublet / cc-pVDZ | direct K | 1.96e-5 Ha (UHF), 1.97e-5 Ha (ROHF) | — | ″ |

A "—" means the repository states no number for that cell. It is left empty on
purpose rather than filled with an estimate.

Benchmark sweeps (GW100 and others) are kept in the project's working notes
and are not reproduced here. Quoting a benchmark statistic from memory rather
than from the record is the kind of unchecked claim this page exists to
prevent.

## Known limits and negatives

Reported rather than omitted:

- **Analytic Hessians**: not implemented. The mpqc4 libint2 export has no
  second-derivative integrals. Harmonic **frequencies are available** by
  central finite differences of the analytic gradient (`task = "frequencies"`,
  `ferric.run_frequencies`).
- **Local MP2 (amplitude threshold)**: the J build is still dense (from RI), so
  no scaling claim is made; only counters are reported.
- **Laplace SOS-MP2, AO-sparse variant**: the domain truncation is accurate
  and the radius it needs does **not** grow with the molecule (octane is exact
  at the radius that suffices for butane; a 71-atom drug molecule is within
  0.05% at 4 Bohr, about 13% of its diameter). The algebra is still dense, so
  **no speedup is claimed**. An earlier "measured negative" for this variant
  was an indexing bug and has been retracted; the regression test
  `sos_ao_sparse_truncation_radius_is_transferable_across_sizes` records the
  history.
- **TDHF/RPAx C6**: ~60% low regardless of gap. Use it for static
  polarizabilities, not dispersion.
- **TDDFT / TDA with a DFT reference**: the f<sub>xc</sub> kernel term is
  omitted in the CLI/Python path; the run warns.
- **COSX at large N**: the one-thread tail over alkanes (C12–C20, def2-SVP)
  fits A-build ~N<sup>1.5</sup> and full K ~N<sup>2.2</sup>, slower than
  analytic exchange at that basis. COSX is Coulomb-only and has no gradients.

## Why the distinction is drawn so sharply

A quantum chemistry code can produce a plausible number in many ways that are
wrong:

- a **non-converged SCF** returned as an ordinary result, because convergence is
  a flag rather than an error
- a method missing a **physical term** it does not mention (TDDFT's f<sub>xc</sub>
  kernel is exactly this case, which is why it now warns)
- a **fallback model** silently substituted for one atom in a molecule, changing
  a partitioning without changing the shape of the output
- a **screening or truncation threshold** that happens to be safe for the test
  system and not for yours

None of these look like failures. All of them have occurred in this codebase and
been fixed. The remedy is to grade each capability separately and say which ones
are checked against ground truth.

## Testing discipline

Some properties are pinned by tests rather than asserted in prose:

- **Bit-identity across thread counts** for reductions and permutations:
  results do not depend on `RAYON_NUM_THREADS`
- **ERI 8-fold permutational symmetry**: verified against the engine, not
  assumed, since the MP2 code relies on it to compute only ~1/8 of quartets
- **Size-extensivity** and **rotational invariance** of total energies
- **Memory guards in both directions**: a starved budget must be refused *and*
  an ample budget must still run, because an over-estimating guard is also a bug

New guards are **mutation-tested**: a deliberate defect is injected and the test
confirmed to fail before the guard is trusted. This has caught guards that
passed while proving nothing.
