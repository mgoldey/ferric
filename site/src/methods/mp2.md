# The MP2 family

The largest method family here. Every variant is density-fitted (RI): each
needs an orbital basis and a matching RI auxiliary basis (`[mp2] auxbasis`, or
the `auxbasis` argument in Python). Formal cost is O(N⁵) for the RI-MP2
transformation. Grades per `method.kind` are on
[What is validated](../reference/validation.md) and
[Capabilities](../reference/capabilities.md). The `[mp2]` keys are in
[Input file](../reference/input.md#mp2).

## RI-MP2

**What it is.** Closed- and open-shell second-order Møller–Plesset theory with
3-centre/2-centre density fitting. Canonical (non-RI) MP2 is also implemented,
for cross-validation, not production.

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
| terfc | Python only | — | `run_terfc_rimp2(..., r0=...)` |
| SCS-MP2 (Grimme) | `scs-mp2` | `examples/water-scs-mp2.toml` | `run_scs_mp2(..., c_os=, c_ss=)` |
| SCS-MP2(2terfc) | `scs-mp2-2terfc` | `examples/water-scs-mp2-2terfc.toml` | `run_scs_mp2_2terfc(...)` |
| MP2-V | `mp2-v` | `examples/water-mp2v.toml` | `run_mp2_v(...)` |

ω is in Å⁻¹ in the CLI and Python and in Bohr⁻¹ in the Rust API. The SCS-MP2
defaults are Grimme's c<sub>OS</sub> = 1.2, c<sub>SS</sub> = 0.333.

**Accuracy.** `att-rimp2`, `scs-mp2` and `scs-mp2-2terfc` are Proven (the erfc
energy is pinned against PySCF/libcint in
`testdata/reference/h2o_cc-pvdz_attenuated-rimp2-erfc0p420.json`). `mp2-v` is
Smoke: its VV10 half is bit-identical to the ωB97X-V code path, but there is no
published MP2-V total energy to compare with, and ferric evaluates the VV10 term
on the converged HF density (the post-HF variant) on its own 50×50 grid rather
than SG-1. Read the header of `examples/water-mp2v.toml` before quoting a number.

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
orbital DIIS, Cayley rotations and backtracking. Smoke: stationarity and a
vanishing orbital gradient are checked, but there is no external
absolute-energy reference.

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

- Against the exact AO path on n-alkanes (c<sub>OS</sub> = 1, `n_quad` = 7,
  2026-07-28), chemical accuracy needs a domain radius of 3, 3, 3, 4, 5 and
  5 Bohr for C2, C4, C6, C8, C10 and C12. The diameter grows fivefold over that
  series, so radius/diameter falls from 0.52 to 0.17.
- In the STO-3G tests, a 12 Bohr domain that is exact for butane (10.5 Bohr
  across) is also exact for octane (19.9 Bohr across), and a 4 Bohr domain on
  butane is already within 0.1%.
- A 71-atom drug molecule (danuglipron, 31.3 Bohr across, STO-3G) is within
  0.05% at 4 Bohr.

The butane/octane figures are pinned by the test
`sos_ao_sparse_truncation_radius_is_transferable_across_sizes` in
`crates/ferric-mp2/src/laplace.rs`, whose doc comment also records the
danuglipron run. The alkane sweep is kept in the project's working notes.

**How to read it (provisional):** the radius needed grows, but far more slowly
than the molecule. That points to a finite decay length rather than strict
saturation: no single radius is shown to suffice at every size.

**What is not claimed:** any speedup. The domains discard contributions but
the tensor algebra is still dense, so there are no timings to report.

> An earlier version of these docs called `ao-sparse` a "measured negative"
> (radius tracking the molecular diameter). That result came from an indexing
> bug that masked canonical orbitals with localized orbitals' domains, and it
> has been retracted; the test above records the history.

## Local MP2

**Amplitude-threshold LMP2** (`lmp2`, `examples/water-lmp2.toml`,
`run_lmp2(..., eps=1e-4)`): the single-threshold local MP2 of Wang, Aldossary,
Shi, Liu, Li & Head-Gordon (2023), closed shell, with localized virtuals and
per-pair domain-local RI fits. `eps = 0` reproduces RI-MP2 exactly; the
default `1e-4` carries a one-sided truncation error, which the output prints
against the canonical RI reference.

**No scaling claim is made.** The J build is still dense (from RI), so the
amplitude machinery is anchored but end-to-end cost is not reduced. Counters
are reported instead.

## Cite

RI-MP2 auxiliary sets: Weigend et al. 1998. SCS-MP2: Grimme 2003. SOS-MP2:
Jung et al. 2004. OO-MP2: Lochan & Head-Gordon 2007; Bozkaya et al. 2011.
Laplace MP2: Häser & Almlöf 1992. Attenuated MP2: Goldey & Head-Gordon 2012;
terfc: Dutoi & Head-Gordon 2008, Goldey, Dutoi & Head-Gordon 2013;
SCS-MP2(2terfc): Goldey & Head-Gordon 2014; MP2-V: Goldey, Belzunces &
Head-Gordon 2015. LMP2: Wang et al. 2023. Robust fitting: Dunlap 2000. Full
entries in [References](../reference/references.md).
