# The MP2 family

The largest method family here. Every variant is density-fitted (RI): each
needs an orbital basis and a matching RI auxiliary basis (`[mp2] auxbasis`, or
the `auxbasis` argument in Python). Formal cost is O(N⁵) for the RI-MP2
transformation. Grades per `method.kind` are on
[Capabilities and validation](../reference/validation.md). The `[mp2]` keys are in
[Input file](../reference/input.md#mp2).

## RI-MP2

**What it is.** Second-order Møller–Plesset theory on an RHF reference (UHF
for an open shell, below) with 3-centre/2-centre density fitting. Canonical (non-RI) MP2 is also implemented,
for cross-validation, not production.

**Open shells.** `rimp2` and `oo-rimp2` accept an open-shell molecule
(`multiplicity > 1`) from the CLI, for `task = "energy"` only: they solve the
same plain UHF that `kind = "uhf"` runs, then take unrestricted RI-MP2 (UMP2,
as PySCF `mp.MP2(uhf)`) or unrestricted OO-RI-MP2. There is no unrestricted
MP2 nuclear gradient, and `[mp2] kappa` is refused on an open shell. From
Python, `run_rimp2` does the same (UHF + UMP2 when `multiplicity > 1`;
`result.reference` says which, and `kappa` raises); `run_oo_rimp2` is closed
shell only. Both unrestricted methods are validated on the CH3 doublet at
cc-pVDZ: U-RI-MP2 against ORCA `RI-MP2 NoRI`, U-OO-RI-MP2 against an
independent numpy OO-RI-MP2 (see the
[anchors](../reference/validation.md#anchors)).

The other MP2-family kinds (`mp3`, `att-rimp2`, `scs-mp2`, `scs-mp2-2terfc`,
`laplace-mp2`, `laplace-sos-mp2`, `rs-mp2-rpa`) and their Python drivers are
closed shell only: set `multiplicity = 1`. `mp2-v` switches to a UHF reference
when `multiplicity > 1` (CLI, energy only); Python `run_mp2_v` is closed shell
only.

**The reference SCF: exact, RI-JK or RIJCOSX.** The `auxbasis` above fits the
correlation step only. The reference SCF is a separate choice, `scf_jk` on
every MP2-family Python driver (`[scf] jk` in the CLI, task `energy`):
`"exact"` (four-centre J and K, the default), `"rijk"` (RI-J + RI-K) or
`"rijcosx"` (RI-J + COSX K). The fitted modes use `jk_auxbasis` (`[scf]
jk_aux`), a JK-fit basis with default `def2-universal-jkfit`. It is never the
MP2 `auxbasis`, because an MP2-fit set used for K costs mHa. `rs-mp2-rpa` is the
exception: it has always run RI-JK, so its default is `"rijk"`. Every result
echoes `scf_jk` and `jk_auxbasis`. An unknown mode, `jk_auxbasis` with
`"exact"`, or a `k_builder` the mode would override is an error. At aug-cc-pVTZ
the four-centre SCF is most of an MP2 single point, so a fitted reference is the
usual choice for interaction-energy benchmarks. The fitting error is small and
largely cancels in a counterpoise-corrected interaction energy. Measured with
PySCF DF-MP2 (frozen core, `aug-cc-pvXz-ri` for MP2) on the S66 water dimer:

| Basis | JK aux | ΔE_total(dimer) | ΔIE (CP) |
|---|---|---|---|
| aug-cc-pVDZ | def2-universal-jkfit | +98 µHa | +0.0059 kcal/mol |
| aug-cc-pVDZ | aug-cc-pvdz-jkfit (not bundled) | +58 µHa | +0.0001 kcal/mol |
| aug-cc-pVTZ | def2-universal-jkfit | +69 µHa | +0.0043 kcal/mol |
| aug-cc-pVTZ | cc-pvtz-jkfit | +78 µHa | +0.0026 kcal/mol |
| aug-cc-pVTZ | aug-cc-pvtz-jkfit (not bundled) | +63 µHa | <0.00001 kcal/mol |

**Run it.** `method.kind = "rimp2"` (`examples/water-rimp2.toml`,
`examples/water-rimp2-frozen-core.toml`); Python `ferric.run_rimp2(mol, bs, aux)`.

**Accuracy.** Proven. RI-MP2 is size-extensive to **2e-12 Ha** for a
well-separated dimer against twice the monomer
(`ferric-mp2/tests/rimp2_size_extensivity.rs`, which asserts 1e-7).

**Knobs.** `frozen_core` defaults to **0** (all electrons correlated). Several
published parameterizations below assume frozen core, so set it when you use
them.

## Attenuated MP2

**What it is.** MP2 with the Coulomb operator in the correlation energy
replaced by a short-range one. Two forms are implemented:

- **erfc**: \\( \mathrm{erfc}(\omega r)/r \\), which is \\( 1/r \\) at short
  range and decays to zero beyond roughly \\( 1/\omega \\).
- **terfc**: a short-range operator that is Coulombic inside a cutoff radius
  \\( r_0 \\) and attenuated outside it, with a controlled curvature
  (Dutoi & Head-Gordon 2008). It needs precomputed interpolation tables
  (`FERRIC_TERF_TABLE_DIR`, generated under `terf-tables/`); without them the
  run errors rather than substituting another operator.

**Why.** MP2's error for non-covalent interactions has two parts that partly
cancel in small basis sets: basis-set superposition error, and an error in the
long-range part of the correlation energy. Attenuation removes the long-range
part. That also means attenuated MP2 has **no long-range dispersion at all**:
its asymptotic \\( C_6 \\) is zero. The method works in the basis and on the
systems it was fitted for; MP2-V (below) adds long-range dispersion back
through VV10.

**Parameterization.** The attenuation parameters are **fitted**, not derived,
and each fit belongs to one basis. The aug-cc-pVTZ sets that ferric ships as
defaults were fitted on S66 **without counterpoise and with frozen core**
(stated in `crates/ferric-mp2/src/scs.rs`):

| Variant | Parameters | Basis of the fit | Source |
|---|---|---|---|
| erfc | ω; ferric's default is 0.420 Å⁻¹, recorded in the code as the dissertation's erfc optimum | aug-cc-pVDZ in the 2012 paper | Goldey & Head-Gordon 2012 |
| terfc | one cutoff \\( r_0 \\) | aug-cc-pVTZ | Goldey, Dutoi & Head-Gordon 2013 |
| SCS-MP2(2terfc) | \\( r_0(1) \\) = 0.75 Å, \\( r_0(2) \\) = 1.05 Å, c<sub>OS</sub> = 1.27, c<sub>SS</sub> = 4.05 | aug-cc-pVTZ, no counterpoise, frozen core | Goldey & Head-Gordon 2014 |
| MP2-V | \\( r_0 \\) = 1.00 Å, b = 11.0, C = 0.0089, terfc | aug-cc-pVTZ, no counterpoise, frozen core | Goldey, Belzunces & Head-Gordon 2015 |

Running a fitted variant in another basis, with counterpoise, or with
`frozen_core = 0` (the default) is extrapolation outside the fit.

**Run it.**

| Variant | `method.kind` | Example | Python |
|---|---|---|---|
| erfc | `att-rimp2` | `examples/water-attmp2.toml` | `run_attenuated_rimp2(..., omega=0.420)` (Å⁻¹) |
| terfc | `att-rimp2` with `[mp2] att_operator = "terfc"`, `att_r0` (Å) | `examples/water-attmp2-terfc.toml` | `run_terfc_rimp2(..., r0=...)` |
| SCS-MP2 (Grimme) | `scs-mp2` | `examples/water-scs-mp2.toml` | `run_scs_mp2(..., c_os=, c_ss=)` |
| SCS-MP2(2terfc) | `scs-mp2-2terfc` | `examples/water-scs-mp2-2terfc.toml` | `run_scs_mp2_2terfc(...)` |
| MP2-V | `mp2-v` | `examples/water-mp2v.toml` | `run_mp2_v(...)` |

ω is in Å⁻¹ in the CLI and Python and in Bohr⁻¹ in the Rust API. The SCS-MP2
defaults are Grimme's c<sub>OS</sub> = 1.2, c<sub>SS</sub> = 0.333.

**Accuracy.** `att-rimp2`, `scs-mp2` and `scs-mp2-2terfc` are Proven (the erfc
energy is pinned against PySCF/libcint in
`testdata/reference/h2o_cc-pvdz_attenuated-rimp2-erfc0p420.json`). `mp2-v`'s
implementation is Proven (narrow): the damped VV10 half, the terfc-attenuated MP2
half and the erfc control each match an independent PySCF/numpy reference. Its
parameterization is unbenchmarked: there is no published MP2-V total energy to
compare with, only an aug-cc-pVTZ S66 subset run (n=5, H-bonded/mixed, S66x8 geometry, no CP-anchored reference), and ferric evaluates the VV10 term on the converged HF
density (the post-HF variant) on its own 50×50 grid rather than SG-1. Read the header of `examples/water-mp2v.toml` before quoting a number.

**Robust fitting.** When the RI metric differs from the operator in the
integrals, as it does for attenuated operators, robust (Dunlap) density fitting
is required, not optional.

## RS-MP2 + LR-RPA

**What it is.** Short-range MP2 plus long-range direct RPA: the attenuated-MP2
idea with the missing long-range correlation supplied by response rather than
by a fitted dispersion term. Two formulations:

- `delta-lr` (default): \\( E_{MP2}[\text{Coulomb}] + (E_{dRPA}[\text{erf}] - 2E_{OS}[\text{erf}]) \\), one dRPA call.
- `coupled-rings`: \\( E_{MP2}[\text{Coulomb}] + \Delta dRPA[\text{Coulomb}] - \Delta dRPA[\text{erfc}] \\), two dRPA calls; includes the mixed short/long-range ring diagrams.

**Run it.** `method.kind = "rs-mp2-rpa"` (`examples/water-rs-mp2-rpa.toml`,
`[mp2] formulation`); Python `run_rs_mp2_rpa(..., omega=0.420, formulation="delta-lr")`.

**Accuracy.** Smoke. The ω→0 and ω→∞ limits reduce exactly to MP2 and to
MP2 + dRPA; numbers at production ω are not established on new systems.

## Orbital-optimized MP2 and MP3

**OO-RI-MP2** (`oo-rimp2`, `examples/water-oo-rimp2.toml`, `run_oo_rimp2`):
orbitals optimized for the MP2 Lagrangian, with a level-shifted Newton step,
orbital DIIS, Cayley rotations and backtracking. Proven (narrow): the energy
matches an independent numpy OO-RI-MP2 to 7.5e-13 Ha (H2O, NH3, UHF CH3 at
cc-pVDZ), and the closed-shell analytic gradient matches a finite difference of
its own energy to 8e-9 Ha/Bohr. ORCA 6.1.1's OO-RI-MP2 stops 3.7e-8 to
7.0e-8 Ha above the same minimum, so it is only a loose cross-check.

**MP3** (`mp3`, `examples/water-mp3.toml`, `run_mp3`): spin-orbital
third-order Møller–Plesset through the `einsum!` framework. Proven.

## Laplace formulations

**RI-Laplace MP2** (`laplace-mp2`, `examples/water-laplace-rimp2.toml`,
`run_laplace_mp2`): MP2 through a Laplace transform of the energy denominator,
using pseudo-density matrices in the AO basis. The implementation is **dense**.
It is the correctness reference for that formulation, not a reduced-scaling
path, and no reduced scaling has been measured.

**Laplace SOS-MP2** (`laplace-sos-mp2`, `examples/water-laplace-sos-mp2.toml`,
`run_laplace_sos_mp2`): opposite-spin-only MP2 (Jung et al. 2004,
c<sub>OS</sub> = 1.3 by default) with a minimax Laplace quadrature
(`n_quad` must be 3, 5 or 7; anything else is an error). There is deliberately
no c<sub>SS</sub>: dropping the same-spin term is what lets the denominator
factorize. With c<sub>OS</sub> = 1.0 it reproduces the opposite-spin component
of RI-MP2 to quadrature error, which is the test anchor. Three formulations
(`sos_formulation`):

- `mo` (default) and `ao` are both exact and agree to round-off.
- `ao-sparse` restricts each Boys-localized orbital's pseudo-density to an AO
  domain of radius `domain_cutoff_bohr` (required, no default). It is the one
  approximate variant.

**What is measured for `ao-sparse`:**

- Against the exact AO path on n-alkanes (cc-pVDZ, c<sub>OS</sub> = 1,
  `n_quad` = 7, 2026-07-28), chemical accuracy (absolute error below 1.6 mHa)
  needs a domain radius of 3, 3, 3, 4, 5 and 5 Bohr for C2, C4, C6, C8, C10
  and C12. The diameter grows fivefold over that
  series, so radius/diameter falls from 0.52 to 0.17.
- In the STO-3G tests, a 12 Bohr domain that is exact for butane (10.5 Bohr
  across) is also exact for octane (19.9 Bohr across), and a 4 Bohr domain on
  butane is already within 0.1%.
- A 71-atom drug molecule (danuglipron, 31.3 Bohr across, STO-3G) is within
  0.05% at 4 Bohr.

The test `sos_ao_sparse_truncation_radius_is_transferable_across_sizes` in
`crates/ferric-mp2/src/laplace.rs` pins the STO-3G butane/octane comparison:
at 12 Bohr butane is exact (relative error below 1e-9) and octane is
within 1e-6, and octane at 3 Bohr is worse than at 12 Bohr. The
4 Bohr butane figure and the danuglipron run (recorded in the test's doc
comment) are measurements, not assertions. The alkane sweep is kept in the project's working notes.

**How to read it (provisional):** the radius needed grows, but far more slowly
than the molecule. That points to a finite decay length rather than strict
saturation: no single radius is shown to suffice at every size.

**What is not claimed:** any speedup. The domains discard contributions but
the tensor algebra is still dense, so there are no timings to report.

## Exact and local MP2

`method.kind = "rimp2"` is the method; it is computed exactly unless a
[`[local]`](../reference/input.md#local) section asks for a local
approximation (see
[Exact and local correlation](./index.md#exact-and-local-correlation)).
The local approximation is closed shell and energy only.

**Amplitude-threshold local MP2** (`examples/water-rimp2-local.toml`;
`[local] scheme = "amplitude-threshold"`, `eps = 1e-4`; Python
`run_rimp2(..., local="amplitude-threshold", eps=1e-4)`): the
single-threshold local MP2 of Wang, Aldossary, Shi, Liu, Li & Head-Gordon
(2023), with localized virtuals and per-pair domain-local RI fits. `eps` has
no default: it is part of the model. `eps = 0` reproduces RI-MP2 exactly; a
finite `eps` carries a one-sided truncation error that grows faster than
linearly in `eps`.
Every printout and run-log record of a local run states `eps` and the
fraction of amplitudes kept.

The exact RI-MP2 reference that measures that error is **opt-in**. It is a
full canonical RI-MP2 and forms the global `(naux, nocc·nvir)` tensor, so a
run with it switched on is not reduced-cost. In the CLI, `[local] reference =
true` computes it and prints the local error against it; without it the
output reads
`E_corr(canonical RI)  = not computed (opt-in: set [local] reference = true)`
and the run log's `e_corr_canonical_ri` is null. In Python,
`compute_reference=True` computes it; by default
`result.local["e_corr_canonical_ri"]` is `None`.

**Integral-direct local MP2** (`[local] integral_direct = true`,
`examples/alkane8-rimp2-local-direct.toml`; Python `integral_direct=True`) is
the reduced-cost path. Its correlation assembly never forms the global
3-index tensor; only the opt-in reference does. Locality comes from an
integral-free pair gate (`gate_cal`), per-occupied auxiliary-fit and virtual
domains (`aux_radius`, `virt_radius`), and truncation of each orbital's AO
support (`ao_tail`). With every map at its trivial setting it reproduces the
global 3-index path and canonical RI-MP2 (`tests/lmp2_direct.rs`).

**What is measured for the integral-direct local MP2** (n-alkanes C20 → C48, 6-31G with
cc-pVDZ-RI, a quiet machine, 2026-09-07; benchmark
`bench_direct_alkane_series` in `crates/ferric-mp2/tests/lmp2_direct.rs`).
The run froze the carbon cores (`frozen_core` = number of carbons) and
calibrated the pair gate (0.7 Coulomb, 0.02 erfc with ω = 1.0) at
`eps = 1e-4`; the defaults are all-electron with no pair gate. Timings are the correlation
stage (assembly + solve) and exclude the canonical reference:

- Correlation-stage cost grows as about N<sup>1.24</sup> with the erfc
  kernel and N<sup>1.4</sup> with Coulomb, fitted to the last three sizes.
- It is at about parity with canonical RI-MP2 at C20 (about 1.1× slower)
  and about 6× faster at C32 (5.7–6.3×).

**How to read it (provisional):** sub-quadratic on this series, but the fit
has three points on one family of molecules in one basis. It is not shown
to be linear, and it is not measured on 3-D or diffuse systems.

**Schwarz triple cut (`schwarz_skip`, default `1e-5`)** applies to the Coulomb, erfc and
terfc kernels alike. A shell triple is skipped when
`sqrt(max (P|P)) * sqrt(max (ab|ab)) < schwarz_skip`, both factors from the run's own
operator; `schwarz_skip = 0` evaluates every triple. For terfc the factors come from the
table engine's own diagonals and carry a 1e-5 relative inflation for table error, and the
cut is only available inside the checked box: orbital shells up to g, r0 between 0.3 and
2.5 Å and r0·ω ≤ 8.9 (outside it the run errors and names `schwarz_skip = 0.0`). Measured
for terfc (n-alkanes, 6-31G with cc-pVDZ-RI, `eps = 5e-5`, default maps, plain Schwarz,
2026-10-11, a loaded machine so ratios only, order alternated, two repeats):

| System | operator | wall vs `schwarz_skip = 0` at `1e-6` | at `1e-5` | energy change at `1e-5` |
|---|---|---|---|---|
| C12 | linked (r0 = 1.0 Å, r0·ω = 0.707) | 2.3–2.5× faster | 2.7× | 1.5e-7 relative |
| C12 | NCP (r0 = 0.991 Å, ω = 1.3663 Å⁻¹) | 2.4–2.5× | 2.6–2.9× | 3.6e-8 |
| C16 | linked | 3.3–3.5× | 3.5–3.8× | 1.7e-7 |
| C16 | NCP | 3.1–3.9× | 3.3–4.1× | 3.6e-8 |

At `1e-5` the cut skips 55% of the shell triples at C12 and 67% at C16 (51% and 63% at `1e-6`). The pair gate
(`gate_cal = 0.7`) leaves the C12 result unchanged. At `1e-6` the energy changes by
1e-10 (C12 linked) to 1.8e-9 (C16 NCP).

**Local MP2 without `integral_direct` makes no scaling claim.** Its assembly
is pair-local (no dense J is formed), but it still builds the global 3-index
tensor. Use it as the reference implementation and for small systems.

## Cite

RI-MP2 auxiliary sets: Weigend et al. 1998. SCS-MP2: Grimme 2003. SOS-MP2:
Jung et al. 2004. OO-MP2: Lochan & Head-Gordon 2007; Bozkaya et al. 2011.
Laplace MP2: Häser & Almlöf 1992. Attenuated MP2: Goldey & Head-Gordon 2012;
terfc: Dutoi & Head-Gordon 2008, Goldey, Dutoi & Head-Gordon 2013;
SCS-MP2(2terfc): Goldey & Head-Gordon 2014; MP2-V: Goldey, Belzunces &
Head-Gordon 2015. LMP2: Wang et al. 2023. Robust fitting: Dunlap 2000. Full
entries in [References](../reference/references.md).
