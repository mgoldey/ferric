# ECP / pseudopotential support in ferric — feasibility assessment

Date: 2026-06-17. Scope: research only, no implementation. Question: what would it take to
support the 7 GW100 molecules whose elements require ECPs (Z ≥ 37)?

## Blocked molecules (7)

Xe, Rb-containing, I and iodides (CH3I, CHI3-class, AlI3), Ag-containing — anything with a
heavy element (Z ≥ 37: Rb, Ag, I, Xe) that, in the GW100 reference protocol, is treated with
a small-core relativistic pseudopotential (def2-ECP / cc-pVnZ-PP) rather than all-electron.
For these, the all-electron Dunning/def2 orbital basis either does not exist or is
prohibitively large, and the standard GW100 numbers use ECPs.

## Current state in ferric: NONE

- **No ECP code anywhere.** `grep -rin "ecp|pseudo|effective_core|core_potential"` over
  `crates/**/*.rs` returns zero ECP hooks. The only "core" concept is `frozen_core` — that
  is a post-SCF correlation-window offset (which occupied MOs to skip in MP2/CC/RPA), wholly
  unrelated to replacing core electrons with a potential.
- **No data path.** `Molecule`/`Atom` carry only `symbol, z, coords`; there is no field for
  core-electron count, ECP parameters, or an effective nuclear charge. `nelec()` is derived
  straight from Z, so even loading an ECP basis would give the wrong electron count without
  a parallel "n_core removed" bookkeeping layer.

## libint2 cannot do ECP integrals

The C-ABI shim (`crates/ferric-integrals/shim/shim.{h,cc}`) exposes exactly these libint2
operators: `coulomb, erf_coulomb, erfc_coulomb` (2e); `overlap, kinetic, nuclear` (1e);
`cgtg, cgtg_x_coulomb, delcgtg2` (geminal). **There is no ECP/pseudopotential operator** —
because libint2 itself does not implement ECP integrals. This is a known, long-standing
limitation of libint2 (ECP semilocal + spin-orbit projectors are out of scope for the
library; upstream has never shipped them). So there is no "flip a flag in the shim" path.

ECP integrals would have to come from a **separate library**. The standard choice is
**libecpint** (Shaw & Hill, the same engine PySCF/Psi4/ORCA-class codes use). It computes
the type-1 (local) and type-2 (semilocal angular-projector) radial ECP integrals over
Gaussian shells.

## What full ECP support would require (concrete)

1. **New integral dependency.** Vendor/build **libecpint** and add a second C-ABI shim
   (`ecp_shim.cc`) plus FFI bindings in `ferric-integrals`. This is a new build-system
   surface (CMake/linking) parallel to the libint2 shim. Non-trivial but bounded — libecpint
   is a focused, self-contained C++ library.

2. **ECP parameter ingestion.** Parse ECP definitions from BSE (`def2-ECP`, `cc-pVnZ-PP`
   come bundled with their ECP block in the same JSON under an `ecp_electrons` /
   `ecp_potentials` key). ferric's basis parser currently reads only `electron_shells` and
   ignores everything else — it would need a new struct + loader for the ECP block.

3. **Molecule / electron-count plumbing.** Add `n_core_ecp` per atom; subtract it from
   `nelec()`; reduce the effective nuclear charge seen by the `nuclear` 1e operator (the
   point charge for an ECP atom becomes Z − n_core); thread this through SCF, the guess, and
   every correlation driver's occupied-count logic.

4. **Hcore assembly.** Add the ECP matrix `V_ECP` (from libecpint) into the core Hamiltonian
   alongside kinetic + nuclear. Localized change once 1–3 exist.

5. **Gradients (if needed).** ECP integral derivatives for geometry optimization — libecpint
   supports them, but it is extra wiring. NOT needed for single-point GW100 IPs.

6. **Relativistic reference data / validation.** GW100 heavy-element numbers use specific
   ECP+basis combos; reproducing them requires matching the exact def2-ECP / aug-cc-pVnZ-PP
   pairing PySCF/MOLGW use, then validating against published GW100 values.

## Effort verdict: LARGE, not small

This is a multi-week feature, not an afternoon. The expensive parts are (a) standing up a
**second integral library** (libecpint) with its own build/FFI/shim surface, and (b)
threading **reduced electron count + effective charge** through every electron-count-aware
code path (SCF occupation, frozen-core windows, all MP2/RPA/GW drivers). Items 2 and 4 alone
are easy; items 1 and 3 are where the real work and risk live.

It is genuinely feasible (libecpint is the proven path and integrates cleanly with
Gaussian-basis codes), but it is a distinct subsystem, not a patch. For GW100 specifically it
buys only 7 of 100 molecules.

## Cheaper alternatives to consider first

- **All-electron the lighter blocked elements where tractable.** Rb/I/Xe all-electron with a
  relativistic all-electron basis (e.g. x2c / ANO-RCC) would ALSO require a scalar-relativistic
  correction (X2C/DKH) that ferric likewise lacks — so this is not actually cheaper.
- **Scope GW100 to the all-electron-tractable subset (93 molecules)** and document the 7
  ECP molecules as out-of-scope pending libecpint. This is the recommended near-term posture:
  it unblocks the expansion (the alkali/d-block aux gaps above are the real GW100 blocker,
  and those are def2-aux / pairing decisions, not ECP) without a large new subsystem.
