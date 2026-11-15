# Choosing a method

Start from what you want to compute. Each row names a method that ferric
implements for that task, where to run it (CLI `method.kind` or Python only),
how far it is validated, and a shipped example to copy. Grades, and the anchors behind them, are on
[Capabilities and validation](../reference/validation.md). Cost is given as formal scaling
with system size N, not as timings.

This page only recommends what the code and its tests support. Where ferric
has no validated option for a task, the row says so.

## By task

### General tasks

| Task | Recommended | Where | Grade | Scaling | Example |
|---|---|---|---|---|---|
| Geometry optimization | KS-DFT (e.g. PBE, B3LYP), RHF, UHF or ROHF, `task = "optimize"`; with `[dft] functional`, `uhf` and `rohf` run UKS and ROKS | CLI; Python `run_optimize` (`reference=`, `xc=`, `dispersion=`, `solvent=`; also `method="rimp2"` / `"oo-rimp2"`) | Proven | N⁴ | `h2-lda-opt.toml`, `h2_opt.toml` |
| Harmonic frequencies | KS-DFT or RHF/UHF/ROHF, `task = "frequencies"`. The analytic Hessian covers RHF/UHF/ROHF (exact J/K, or RI-J and/or RI-K) and RKS and UKS LDA/GGA, pure or global hybrid (moving-grid Hessian); both are the default under `hessian = "auto"`. Otherwise finite differences of the analytic gradient, 6N gradients. `rimp2`, `scs-mp2`, `oo-rimp2` and `pdep-rpa` take the same finite-difference route on their own analytic gradients | CLI, Python `run_frequencies` | energies Proven; check the printed Hessian asymmetry | 6N × N⁴ | `water-frequencies.toml` |
| Transition state | `run_saddle` (P-RFO), then `run_irc` to confirm which minima it connects | Python only, closed shell (RHF or RKS) | see [Capabilities and validation](../reference/validation.md#python-entry-points) | ~2(6N+1) gradients + steps | none |
| Routine energies, conformer or reaction energies | KS-DFT + D3(BJ) | CLI `ksdft` + `[dft] dispersion = "d3bj"`, Python `run_dft(dispersion=...)` | DFT Proven; D3(BJ) (with or without ATM) matches simple-dftd3 to <1e-12 Ha | N⁴ (DFT) | `water-pbe-d3bj.toml` |
| Open-shell molecules | UKS or ROKS for DFT, UHF or ROHF for Hartree-Fock; UMP2 on a UHF reference | CLI `ksdft` with `multiplicity > 1` (UKS), `uhf`/`rohf` with `[dft] functional`; Python `run_uks`, `run_roks`, `run_uhf`, `run_rohf` | Proven; see [open shells in the CLI](../reference/validation.md#open-shells-in-the-cli) | N⁴ | `h_uhf.toml` |
| Correlated energies, small to medium | RI-MP2 | CLI `rimp2`, Python `run_rimp2` | Proven | N⁵ | `water-rimp2.toml` |
| Correlated energies, benchmark quality, small | CCSD(T) (closed shell) | Python `run_ccsd_t` only; CCSD also CLI `ccsd` | CCSD Proven; (T) matches PySCF ~1e-6 Ha on H2O/cc-pVDZ | N⁶ / N⁷ | `water-ccsd.toml` (H2) |
| Non-covalent interaction energies | KS-DFT + D3(BJ) is the routine choice. Attenuated MP2 is the higher-accuracy option, in the basis its parameters were fitted for: SCS-MP2(2terfc) or MP2-V in aug-cc-pVTZ, erfc att-MP2 in aug-cc-pVDZ (the basis of the 2012 paper) | CLI `ksdft`; `scs-mp2-2terfc`, `mp2-v`, `att-rimp2` | DFT Proven; att-MP2, 2terfc and MP2-V Proven (narrow) as implementations; the MP2-V parameterization is unbenchmarked | N⁴ (DFT); N⁵ (MP2) | `water-pbe-d3bj.toml`; `water-scs-mp2-2terfc.toml`, `water-mp2v.toml`, `water-attmp2.toml` |
| Excitation energies | Singlet or triplet TDA or TDDFT on a closed-shell RHF (CIS/TDHF) or RKS reference (global or range-separated hybrids); spin-conserving TDA or TDDFT on an open-shell UHF or UKS reference, with `<S^2>` per root; BSE-TDA on G0W0@HF | CLI `tda`, `tddft`, `bse-tda`; Python `run_tddft` (`solver` = `"dense"`, `"davidson"` or `"auto"`), `run_bse_tda` | TDA/TDDFT Proven (narrow, closed shell); BSE-TDA Smoke | — | `water-tda.toml`, `water-tddft-pbe.toml`, `water-tddft-cam-b3lyp-triplet.toml`, `nh2-tddft-uks.toml`, `water-bse-tda.toml` |
| Implicit solvation | IEF-PCM (CLI `[pcm]`, Python `solvent=`; analytic gradient, so it optimizes) or COSMO (CLI `[cosmo]`, no gradient) | see [SCF and DFT](../methods/scf.md#implicit-solvation) | both cross-checked against PySCF on water | SCF cost | none |
| Embedding in a protein or solvent | QM/MM | Python `run_qmmm`; CLI `[qmmm]` (point charges) | see [QM/MM](./qmmm.md) | SCF cost | `water-qmmm.toml` |
| Multireference, static correlation | CASSCF; SC-NEVPT2 on top of it | CLI `casscf`, `nevpt2` with `[cas]`; Python `run_casscf`, `run_nevpt2` | Proven; the optimizers do not check the orbital Hessian, see [CAS-CI and CASSCF](../methods/ci.md#casscf) | — | `water-casscf.toml`, `n2-nevpt2.toml` |
| Periodic systems | Gamma-point or k-point RHF/UHF/ROHF, RKS/UKS/ROKS, MP2, dRPA | CLI `[cell]`; Python `run_<method>_gamma`, `_kpts` | see [Periodic systems](../methods/pbc.md) | see the page | none |
| Atomic charges, ESP | Löwdin, Hirshfeld, CHELPG, RESP; ESP at nuclei or on the surface | Python | see [Capabilities and validation](../reference/validation.md#python-entry-points) | SCF cost | none |

### Specialist methods

| Task | Recommended | Where | Grade | Scaling | Example |
|---|---|---|---|---|---|
| Long-range correlation from response | RS-MP2 + LR-RPA, or PDEP-RPA | CLI `rs-mp2-rpa`, `pdep-rpa` | RS-MP2+RPA Proven at the default ω (the ω choice is unbenchmarked); PDEP-RPA Proven | N⁵ (MP2 part); N⁴ (RI-RPA) | `water-rs-mp2-rpa.toml`, `water-pdep-rpa.toml` |
| Ionization potentials / electron affinities | G0W0@PBE (or @HF); U-GW for open shells | CLI `gw`, Python `run_gw`, `run_u_gw` | Smoke, about ±0.3 eV | — | `water-g0w0-pbe.toml`, `oh-ugw.toml` |
| Static polarizability | PDEP-RPA properties, or RPAx@KS (static α only) | CLI `pdep-rpa` with `compute_polarizability`, `tdhf-static-polarizability` | PDEP-RPA Proven (energy); RPAx Smoke | N⁴ (RI) | `h2o-pdep-rpa-props.toml`, `water-tdhf-static-alpha.toml` |
| \\( C_6 \\) dispersion coefficients | PDEP-RPA dynamic α (`c6_source = "pdep"`) or TS/MBD, in an augmented basis. Not RPAx: its \\( C_6 \\) is ~63% low | CLI `pdep-rpa` + `[rpa] compute_c6` | which source is better is not established | N⁴ (RI) | `water-c6-pdep.toml`, `argon-c6-rpa-pbe.toml` |
| Electron-transfer coupling | constrained DFT + Wu–Van Voorhis \\( H_{ab} \\) | Python `run_cdft`, `cdft_coupling`; CLI `[cdft]` for a single constrained state (the coupling is Python only) | no external reference | SCF cost × outer λ loop | none |

Notes on the table:

- "Proven" for an attenuated MP2 method means the implementation reproduces
  its reference, not that the method is accurate on your system. Its
  parameters are fitted to interaction-energy benchmarks in one basis (the
  aug-cc-pVTZ sets without counterpoise, with frozen core); in another basis,
  with counterpoise correction, or with `frozen_core = 0` (the default), you
  are extrapolating. See
  [The MP2 family](../methods/mp2.md#attenuated-mp2).
- KS-DFT uses density fitting by default (`def2-universal-jkfit`), and the
  fitting error grows with system size. Turn it off or match it before
  comparing with an exact-Coulomb code; see
  [SCF and DFT](../methods/scf.md#kohnsham-dft).
- CC results are sensitive to the aux basis: at CC accuracy the RI error
  depends on the auxiliary set. See [Coupled cluster](../methods/cc.md).

## Pairing an orbital basis with its auxiliary basis

Every correlated method needs an RI auxiliary basis (`[mp2] auxbasis` or
`[rpa] auxbasis`; the `auxbasis` argument in Python). Use the set built for
your orbital basis. The bundled pairs are:

| Orbital basis | RI (correlation) aux | JK aux (for `df_j_aux` / `df_k_aux`) |
|---|---|---|
| cc-pVDZ / TZ / QZ | `cc-pvdz-rifit` (alias `cc-pvdz-ri`), `cc-pvtz-rifit`, `cc-pvqz-rifit` | `def2-universal-jkfit`; `cc-pvtz-jkfit` / `cc-pvqz-jkfit` (H, B-F, Al-Cl, Ga-Br only) |
| aug-cc-pVDZ / TZ / QZ | `aug-cc-pvdz-rifit`, `aug-cc-pvtz-rifit`, `aug-cc-pvqz-rifit` | `aug-cc-pvdz-jkfit`, `aug-cc-pvtz-jkfit`, `aug-cc-pvqz-jkfit` (H, B-Ne, Al-Ar, Ga-Kr; Ne/Ar/Kr extrapolated), else `def2-universal-jkfit` |
| cc-pVXZ-PP, aug-cc-pVXZ-PP | `<orbital>-pp-rifit` (H-Ar, without Li and Na in the aug sets; Sc-Ni at TZ and QZ only; Y-Pd; Hf-Pt. K-Ni at DZ, K and Ca at TZ/QZ, Cu-Sr, Ag-Lu and Au-Rn error) | `def2-universal-jkfit` |
| def2-SVP(D), TZVP(P)(D), QZVP(P)(D) | `def2-<x>-rifit` for each | `def2-universal-jkfit` |

Each RIFIT set in the table also loads under ORCA's `<orbital>/c` name (for
example `def2-tzvp/c`), each JKFIT set under `<orbital>/jk` (for example
`aug-cc-pvtz/jk`), and `def2-universal-jkfit` as `def2/jk`. STO-3G, the
Pople sets and the pcseg family have no matching RI set; pair them with a
`def2` or `cc` RIFIT set of similar quality. Any other orbital set goes in
through `[basis] path` as a Gaussian-94 file (orbital basis only).
`def2-universal-jkfit` is the default JK set for KS-DFT.

## When nothing here fits

- **Analytic Hessians** cover RHF, UHF and ROHF (exact J/K, or RI-J / RI-K with no
  external potential; an ECP is fine; basis up to f functions; RHF and UHF with
  exact J/K also allow an external potential, ROHF allows none on any route), and
  RKS and UKS LDA/GGA (pure or global hybrid, moving-grid Hessian). ROKS, meta-GGA,
  range-separated hybrids and correlated methods' frequencies are by finite
  differences of the analytic gradient.
- **Open-shell coupled cluster and open-shell TS search** are not available
  (the one open-shell CC-type method, LinLCCD(hh), is library-only). `run_saddle`
  and `run_irc` refuse a multiplicity other than 1. Open-shell
  KS-DFT is available from the CLI: `ksdft` with `multiplicity > 1` runs UKS,
  and `uhf`/`rohf` with `[dft] functional` run UKS/ROKS. See
  [Capabilities and validation: open shells in the CLI](../reference/validation.md#open-shells-in-the-cli).
- **TDDFT** on an open-shell molecule (`multiplicity > 1`) runs on a UHF/UKS
  reference: spin-conserving excitations with `<S^2>` reported per root.
  ROHF/ROKS references and `spin = "triplet"` on an open shell are refused.
  With a functional the XC kernel is included; VV10 functionals are refused.
  See [RPA, GW and excited states](../methods/rpa-gw.md#tddft-and-tda).
- **Other front ends.** The [ASE calculator](./ase.md) and
  [QCSchema](./qcschema.md) route energies and gradients (QCSchema also the
  Hessian) to the drivers above.
