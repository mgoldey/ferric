# RPA, GW and excited states

Methods built on the density–density response function: RPA correlation
energies, GW quasiparticle energies, BSE and TDDFT excitations, and
polarizabilities and \\( C_6 \\) coefficients. All need an RI auxiliary basis
(`[rpa] auxbasis`). The `[rpa]`, `[gw]` and `[tddft]` keys are in
[Input file](../reference/input.md#rpa); grades are on
[Capabilities and validation](../reference/validation.md).

## What PDEP does in ferric

The independent-particle response \\( \chi_0(i\omega) \\) is built in the RI
auxiliary basis from the three-centre B tensors, as an explicit
(Adler–Wiser) sum over every occupied–virtual pair \\( ia \\), weighted by
\\( 4\varepsilon_{ia}/(\omega^2 + \varepsilon_{ia}^2) \\)
(`sternheimer::dielectric_matrix`). **The sum over empty states is still
there.**

**PDEP** (projective dielectric eigenpotentials) then works in the
eigenbasis of the static dielectric matrix in that RI space. Eigenpotentials
whose eigenvalue is within `trunc_thresh` of 1 (default 1e-4) carry almost no
screening and are dropped, so the frequency-dependent work runs in a smaller
basis. That compression, not the removal of the empty-state sum, is what PDEP
contributes here. How much it saves depends on the threshold; runs that need
the full-rank answer set `trunc_thresh = 0.0`, as the GW, BSE and \\( C_6 \\)
examples do.

## RPA correlation energy

**What it is.** Direct RPA (dRPA) correlation from the dielectric eigenvalues
on an imaginary-frequency quadrature.

- **PDEP-RPA**, closed shell: `method.kind = "pdep-rpa"`
  (`examples/water-pdep-rpa.toml`); Python `ferric.run_pdep_rpa`. Proven.
- **U-PDEP-RPA**, open shell over a spin-summed dielectric. From the CLI, set
  `method.kind = "pdep-rpa"` with `multiplicity > 1` and `task = "energy"`: the
  CLI solves UHF (UKS with `[rpa] xc`) with MOM after 5 iterations and runs
  U-PDEP-RPA on it. It is CLI-only: Python `run_pdep_rpa` is closed shell only.
  The library (`ferric_rpa::run_u_pdep_rpa`) also accepts a ROHF (or ROKS)
  reference, which it semi-canonicalizes first: each spin uses the orbitals and
  energies of its own Fock matrix, diagonalized in its occupied and virtual
  blocks (see the [anchors](../reference/validation.md#anchors)).
- **Attenuated RPA**: short-range correlation with an erfc operator.
- **RS-MP2 + LR-RPA**: short-range MP2 plus long-range dRPA, on the
  [MP2 page](./mp2.md#rs-mp2--lr-rpa).

The static eigensolve defaults to **Lanczos**, with a dense path for small
problems. Geometry optimization with `pdep-rpa` is supported
(`task = "optimize"`) on a closed-shell RHF reference.

## GW

**What it is.** Quasiparticle energies from the GW self-energy: **G0W0**,
**COHSEX**, **evGW0** and **evGW**, closed shell, plus unrestricted **U-GW**.
The starting point is HF by default or a KS functional (`[rpa] xc`). From a
KS starting point the static term Σx − v_xc enters the G0W0, evGW₀ and evGW
quasiparticle equation (for U-GW, each spin's own equation with that spin's
v_xc), so Σc is evaluated at the shifted root; U-COHSEX, which is static, adds
it to the quasiparticle energy.

**Run it.** `method.kind = "gw"` with `[gw] method = "g0w0"`
(`examples/water-g0w0-pbe.toml`, open shell `examples/oh-ugw.toml`); Python
`ferric.run_gw`, `run_u_gw`. The open-shell reference is UHF by default;
`[gw] reference = "rohf"` (Python `run_u_gw(reference="rohf")`) uses ROHF
instead, or ROKS with `[rpa] xc` (`examples/oh-ugw-rohf.toml`), semi-canonicalized
per spin as for U-PDEP-RPA.

**Accuracy.** Smoke; treat results as about ±0.3 eV.

| Quantity | System / basis | Reference | Pinned by |
|---|---|---|---|
| G0W0@HF, G0W0@PBE, U-G0W0@UHF, ECP, COHSEX, evGW₀, evGW quasiparticle energies (HOMO−2 to LUMO+2) | H2O, NH3, N2, OH, CH3, NH2, O2, CH2, I2, Xe, Ag2 / cc-pVDZ, aug-cc-pVDZ(-PP) | PySCF `gw_ac`/`ugw_ac` at matched settings (`[rpa] n_quad = 100`, `trunc_thresh = 0`); see [What is validated](../reference/validation.md) | `ferric-gw/tests/validation_gw.rs` |
| G0W0@PBE HOMO IP | H2O / cc-pVDZ | PySCF `gw_ac`, 11.1714 eV; asserted to <0.1 eV | `ferric-gw/tests/g0w0_pbe_h2o.rs` |
| U-G0W0@UKS/PBE quasiparticle energies, Σx − v_xc inside each spin's equation | OH, CH3, NH2 / cc-pVDZ | PySCF `ugw_ac` Σc(ef + iω), quasiparticle equation solved in numpy; MEASURE | `ferric-gw/tests/validation_gw.rs` |
| Σx − v_xc placement in U-GW (inside the quasiparticle equation; none for a UHF reference) | OH / STO-3G | internal: shifted residual, bit-identity of the no-shift path | `ferric-gw/tests/u_gw_ks_shift.rs` |
| U-G0W0@UHF α-HOMO IP | OH / cc-pVDZ | ~13–14 eV window, brackets experiment 13.02 eV | `ferric-gw/tests/oh_u_g0w0.rs` |

**Limits.** The quasiparticle equation is solved by a Newton root search on
the self-energy, which is fragile near \\( \Sigma_c \\) poles. Runs report
whether each root and each eigenvalue-self-consistency loop converged, and warn
when one did not; check those flags.

## BSE-TDA

**What it is.** Bethe–Salpeter excitation energies in the Tamm–Dancoff
approximation on top of G0W0@HF quasiparticle energies, closed shell.

**Run it.** `method.kind = "bse-tda"` (`examples/water-bse-tda.toml`, and a set
of `*-bse-tda-augdz.toml` examples for small organics); Python
`ferric.run_bse_tda`.

**Accuracy.** Smoke. Given the same quasiparticle energies, the lowest five
singlet excitation energies match an independent numpy BSE-TDA (PySCF
density-fitted integrals, static RPA W) to 1.8e-10 Ha and their oscillator
strengths to 2.6e-9, for H2O at cc-pVDZ and aug-cc-pVDZ and NH3 and CH2O at
cc-pVDZ. The quasiparticle energies come from the internal G0W0@HF,
which matches PySCF only at `[rpa] n_quad = 100` and `trunc_thresh = 0`; the
defaults are coarser. The core and high-virtual quasiparticle energies are
ill-conditioned, which moves the lowest five excitations by at most
2.2e-7 Ha.

## TDDFT and TDA

**What it is.** Linear-response excitations in the Tamm–Dancoff approximation
(TDA, which is CIS for an HF reference) and the full Casida equations, closed
shell.

**Run it.** `method.kind = "tda"` or `"tddft"` with `[tddft] n_roots` and
optionally `xc` (`examples/water-tda.toml`, `examples/water-tddft-pbe.toml`); Python
`ferric.run_tddft(mol, bs, aux, functional=..., method="tda")` or
`method="casida"`.

**Scope.** Closed-shell references, singlet excitations. With a DFT
reference the \\( (ia|f_{xc}|jb) \\) XC-kernel term is included; with no
functional the result is CIS/TDHF. Meta-GGA, VV10 and range-separated
functionals are refused. Grade: Proven (narrow, closed shell): water,
formaldehyde and NH3 at 6-31G and aug-cc-pVDZ with HF, LDA, PBE and B3LYP match
PySCF `TDA`/`TDDFT` to at most 6.5e-4 eV, test bar 1e-3 eV
(`ferric-tddft/tests/validation_tddft.rs`).

A separate, **library-only** TDA-DFT in `ferric-gw/src/tddft.rs` uses the same
kernel (`ferric_dft::lr_kernel`) and is pinned against PySCF
(`ferric-gw/tests/tda_dft_vs_pyscf.rs`). The user-facing TDA reproduces it to
1e-8 Ha for HF, PBE and B3LYP.

## Polarizabilities and dispersion coefficients

**What exists.** Static molecular and atom-partitioned polarizabilities,
Casimir–Polder \\( C_6 \\) coefficients from per-atom dynamic
polarizabilities \\( \alpha^A(i\omega) \\), and many-body dispersion (MBD).
Three sources feed the \\( C_6 \\) contraction (`[rpa] c6_source`):

- `ts`: the Tkatchenko–Scheffler single-pole model (default).
- `mbd`: many-body (coupled-dipole) screening on top of the TS polarizabilities.
- `pdep`: dynamic PDEP-RPA polarizabilities (`examples/water-c6-pdep.toml`,
  `examples/argon-c6-rpa-pbe.toml`).

The `argon-c6-rpa-pbe.toml` header records C6(Ar–Ar) = 56.4 a.u. at
RPA@PBE/aug-cc-pVTZ against the DOSD value 64.3 (−12%). **Which of TS and
PDEP-RPA gives better molecular \\( C_6 \\) is not established.**

Use an augmented basis for any polarizability or \\( C_6 \\): without diffuse
functions the dipole response is badly underestimated.

**TDHF/RPAx \\( C_6 \\) is a measured negative.** \\( C_6 \\) built on the
RPAx@PBE kernel stays about 63% low regardless of the gap. Its static
polarizability is not established either: at a physical scissor (0.36 Ha)
water/cc-pVDZ gives an isotropic α of 5.20 a.u. against the DOSD 9.64 a.u.
(−46%). The 9.24 a.u. quoted in the example's header comes from
`scissor = 0`, where the tensor has a negative diagonal component, and that
setting is refused. `method.kind = "tdhf-static-polarizability"` computes
static α only. It needs a KS reference (`[rpa] xc`), and at the default
`[gw] scissor = 0` it can hit an excitonic instability, which is reported as
an error rather than a negative α; `examples/water-tdhf-static-alpha.toml`
hits it as shipped, so set `scissor` to about 0.3–0.4 Ha.

## Cite

PDEP: Wilson, Gygi & Galli 2008. RI-RPA quadrature: Eshuis, Yarkony & Furche
2010; minimax grids: Kaltak, Klimeš & Kresse 2014. GW: Hedin 1965; GW100:
van Setten et al. 2015. TDDFT review: Dreuw & Head-Gordon 2005. TS:
Tkatchenko & Scheffler 2009; MBD: Tkatchenko et al. 2012. Full entries in
[References](../reference/references.md).
