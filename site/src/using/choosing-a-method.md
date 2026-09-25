# Choosing a method

Start from what you want to compute. Each row names a method that ferric
implements for that task, where to run it (CLI `method.kind` or Python only),
how far it is validated, and a shipped example to copy. Grades, and the anchors behind them, are on
[Capabilities and validation](../reference/validation.md). Cost is given as formal scaling
with system size N, not as timings.

This page only recommends what the code and its tests support. Where ferric
has no validated option for a task, the row says so.

## By task

| Task | Recommended | Where | Grade | Scaling | Example |
|---|---|---|---|---|---|
| Geometry optimization | KS-DFT (e.g. PBE, B3LYP) or RHF, `task = "optimize"` | CLI, Python (`run_optimize` is RHF) | Proven | N⁴ | `h2-lda-opt.toml`, `h2_opt.toml` |
| Harmonic frequencies | KS-DFT or RHF/UHF/ROHF, `task = "frequencies"` (finite differences of the analytic gradient, 6N gradients) | CLI, Python `run_frequencies` | energies Proven; check the printed Hessian asymmetry | 6N × N⁴ | `water-frequencies.toml` |
| Transition state | `run_saddle` (P-RFO), then `run_irc` to confirm which minima it connects | Python only, closed shell | see [Capabilities and validation](../reference/validation.md#python-entry-points) | ~2(6N+1) gradients + steps | — |
| Conformer or reaction energies, routine | KS-DFT + D3(BJ) | CLI `ksdft` + `[dft] dispersion = "d3bj"`, Python `run_dft(dispersion=...)` | DFT Proven; D3(BJ) matches simple-dftd3 to <1e-12 Ha | N⁴ (DFT) | `water-pbe-d3bj.toml` |
| Correlated energies, small to medium | RI-MP2 | CLI `rimp2`, Python `run_rimp2` | Proven | N⁵ | `water-rimp2.toml` |
| Correlated energies, benchmark quality, small | CCSD(T) (closed shell) | Python `run_ccsd_t` only; CCSD also CLI `ccsd` | CCSD Proven; (T) matches PySCF ~1e-6 Ha on H2O/cc-pVDZ | N⁶ / N⁷ | `water-ccsd.toml` (H2) |
| Non-covalent interaction energies | Attenuated MP2 in the basis its parameters were fitted for: SCS-MP2(2terfc) or MP2-V in aug-cc-pVTZ, erfc att-MP2 in aug-cc-pVDZ (the basis of the 2012 paper); or KS-DFT + D3(BJ) | CLI `scs-mp2-2terfc`, `mp2-v`, `att-rimp2`, `ksdft` | att-MP2 and 2terfc Proven (as implementations); MP2-V Smoke | N⁵ | `water-scs-mp2-2terfc.toml`, `water-mp2v.toml`, `water-attmp2.toml` |
| Long-range correlation from response | RS-MP2 + LR-RPA, or PDEP-RPA | CLI `rs-mp2-rpa`, `pdep-rpa` | RS-MP2+RPA Smoke; PDEP-RPA Proven | N⁵ (MP2 part); N⁴ (RI-RPA) | `water-rs-mp2-rpa.toml`, `water-pdep-rpa.toml` |
| Ionization potentials / electron affinities | G0W0@PBE (or @HF); U-GW for open shells | CLI `gw`, Python `run_gw`, `run_u_gw` | Smoke, about ±0.3 eV | — | `water-g0w0-pbe.toml`, `oh-ugw.toml` |
| Excitation energies | BSE-TDA on G0W0@HF; or TDA/TDDFT (CIS/TDHF on an HF reference) | CLI `bse-tda`, `tda`, `tddft`; Python `run_bse_tda`, `run_tddft` | BSE-TDA Smoke; TDA/TDDFT Proven (narrow, closed shell) | — | `water-bse-tda.toml`, `water-tda.toml` |
| Static polarizability | PDEP-RPA properties, or RPAx@KS (static α only) | CLI `pdep-rpa` with `compute_polarizability`, `tdhf-static-polarizability` | PDEP-RPA Proven (energy); RPAx Smoke | N⁴ (RI) | `h2o-pdep-rpa-props.toml`, `water-tdhf-static-alpha.toml` |
| \\( C_6 \\) dispersion coefficients | PDEP-RPA dynamic α (`c6_source = "pdep"`) or TS/MBD, in an augmented basis. Not RPAx: its \\( C_6 \\) is ~63% low | CLI `pdep-rpa` + `[rpa] compute_c6` | which source is better is not established | N⁴ (RI) | `water-c6-pdep.toml`, `argon-c6-rpa-pbe.toml` |
| Implicit solvation | IEF-PCM (CLI `[pcm]`, Python `solvent=`) or COSMO (CLI `[cosmo]`) | see [SCF and DFT](../methods/scf.md#implicit-solvation) | both cross-checked against PySCF on water | SCF cost | — |
| Embedding in a protein or solvent | QM/MM | Python `run_qmmm`; CLI `[qmmm]` (point charges) | see [QM/MM](./qmmm.md) | SCF cost | `water-qmmm.toml` |
| Electron-transfer coupling | constrained DFT + Wu–Van Voorhis \\( H_{ab} \\) | Python `run_cdft`, `cdft_coupling` (no CLI) | no external reference | SCF cost × outer λ loop | — |
| Atomic charges, ESP | Löwdin, Hirshfeld, CHELPG, RESP; ESP at nuclei or on the surface | Python | see [Capabilities and validation](../reference/validation.md#python-entry-points) | SCF cost | — |

Notes on the table:

- **"Proven" for an attenuated MP2 method means the implementation reproduces
  its reference**, not that the method is accurate on your system. Its
  parameters are fitted to interaction-energy benchmarks in one basis (the
  aug-cc-pVTZ sets without counterpoise, with frozen core); in another basis,
  with counterpoise correction, or with `frozen_core = 0` (the default), you
  are extrapolating. See
  [The MP2 family](../methods/mp2.md#attenuated-mp2).
- **KS-DFT uses density fitting by default** (`def2-universal-jkfit`), and the
  fitting error grows with system size. Turn it off or match it before
  comparing with an exact-Coulomb code; see
  [SCF and DFT](../methods/scf.md#kohnsham-dft).
- **CC results are sensitive to the aux basis**: at CC accuracy the RI error
  depends on the auxiliary set. See [Coupled cluster](../methods/cc.md).

## Pairing an orbital basis with its auxiliary basis

Every correlated method needs an RI auxiliary basis (`[mp2] auxbasis` or
`[rpa] auxbasis`; the `auxbasis` argument in Python). Use the set built for
your orbital basis. The bundled pairs are:

| Orbital basis | RI (correlation) aux | JK aux (for `df_j_aux` / `df_k_aux`) |
|---|---|---|
| cc-pVDZ | `cc-pvdz-ri` (alias `cc-pvdz-rifit`) | `def2-universal-jkfit` |
| cc-pVTZ | `cc-pvtz-rifit` | `def2-universal-jkfit` |
| aug-cc-pVDZ / TZ / QZ | `aug-cc-pvdz-rifit`, `aug-cc-pvtz-rifit`, `aug-cc-pvqz-rifit` | `def2-universal-jkfit` |
| def2-SVP, def2-TZVP, def2-QZVP | `def2-svp-rifit`, `def2-tzvp-rifit`, `def2-qzvp-rifit` | `def2-universal-jkfit` |

`def2-tzvpp-rifit` and `def2-qzvpp-rifit` are also bundled as auxiliary sets.
The ECP orbital sets `aug-cc-pvdz-pp` and `aug-cc-pvtz-pp` have no bundled
`-pp-rifit` partner. STO-3G and 6-31G have no matching RI set; the examples
pair them with `cc-pvdz-ri` or a larger RI set. Only the names above (plus
`cc-pvdz-f12` and `cc-pvdz-f12-optri`) load by name; any other set goes in
through `[basis] path` as a Gaussian-94 file (orbital basis only).
`def2-universal-jkfit` is the default JK set for KS-DFT.

## When nothing here fits

- **Analytic Hessians** are not implemented; frequencies are by finite
  differences.
- **Open-shell coupled cluster and open-shell TS search** are not available
  (the one open-shell CC-type method, LinLCCD(hh), is library-only). Open-shell
  KS-DFT is available from the CLI: `ksdft` with `multiplicity > 1` runs UKS,
  and `uhf`/`rohf` with `[dft] functional` run UKS/ROKS. See
  [Capabilities and validation: open shells in the CLI](../reference/validation.md#open-shells-in-the-cli).
- **TDDFT with the XC kernel** exists only as a library-only spike; the CLI
  and Python TDDFT omit it. See
  [RPA, GW and excited states](../methods/rpa-gw.md#tddft-and-tda).
