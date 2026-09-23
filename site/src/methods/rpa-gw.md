# RPA, GW and excited states

Methods built on the density–density response function: RPA correlation
energies, GW quasiparticle energies, BSE and TDDFT excitations, and
polarizabilities and \\( C_6 \\) coefficients. All need an RI auxiliary basis
(`[rpa] auxbasis`). The `[rpa]`, `[gw]` and `[tddft]` keys are in
[Input file](../reference/input.md#rpa); grades are on
[What is validated](../reference/validation.md).

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
- **U-PDEP-RPA**, open shell over a spin-summed dielectric, from a UHF or ROHF
  reference.
- **Attenuated RPA**: short-range correlation with an erfc operator.
- **RS-MP2 + LR-RPA**: short-range MP2 plus long-range dRPA, on the
  [MP2 page](./mp2.md#rs-mp2--lr-rpa).

The static eigensolve defaults to **Lanczos**, with a dense path for small
problems. Geometry optimization with `pdep-rpa` is supported
(`task = "optimize"`).

## GW

**What it is.** Quasiparticle energies from the GW self-energy: **G0W0**,
**COHSEX**, **evGW0** and **evGW**, closed shell, plus unrestricted **U-GW**.
The starting point is HF by default or a KS functional (`[rpa] xc`).

**Run it.** `method.kind = "gw"` with `[gw] method = "g0w0"`
(`examples/water-g0w0-pbe.toml`, open shell `examples/oh-ugw.toml`); Python
`ferric.run_gw`, `run_u_gw`.

**Accuracy.** Smoke; treat results as about ±0.3 eV.

| Quantity | System / basis | Reference | Pinned by |
|---|---|---|---|
| G0W0@HF HOMO IP | H2O / cc-pVDZ | MOLGW 11.97 eV (van Setten et al. 2015); test tolerance ±0.30 eV | `ferric-gw/tests/h2o_g0w0_cohsex.rs` |
| G0W0@PBE HOMO IP | H2O / cc-pVDZ | PySCF `gw_ac`, 11.1714 eV; asserted to <0.1 eV | `ferric-gw/tests/g0w0_pbe_h2o.rs` |
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

**Accuracy.** Smoke. Only excitation ordering and a physicality gate are
checked, and the excitation energies inherit the GW gap error. The
`water-bse-tda.toml` header records one measured lowest singlet (8.457 eV)
against a PySCF-integral BSE cross-check (8.46 eV).

## TDDFT and TDA

**What it is.** Linear-response excitations in the Tamm–Dancoff approximation
(TDA, which is CIS for an HF reference) and the full Casida equations, closed
shell.

**Run it.** `method.kind = "tda"` or `"tddft"` with `[tddft] n_roots` and
optionally `xc` (`examples/water-tda.toml`, `examples/water-tddft-pbe.toml`;
both need `[mp2] auxbasis = "cc-pvdz-ri"` added to run, see
[Examples](../reference/examples.md)); Python
`ferric.run_tddft(mol, bs, aux, functional=..., method="tda")` or
`method="casida"`.

**Important limitation.** This path (the `ferric-tddft` crate) does **not**
include the \\( (ia|f_{xc}|jb) \\) XC-kernel term. With a pure Hartree–Fock
reference and no correlation functional that term is zero and the result is
exactly CIS/TDHF. With a DFT reference the excitation energies omit it and are
approximate; the code warns on stderr. Grade: Spike.

A separate, **library-only** TDA-DFT in `ferric-gw/src/tddft.rs` does include
a GGA \\( f_{xc} \\) kernel. It covers closed-shell singlets with LDA, GGA and
global hybrids (LDA, PBE and B3LYP are tested), is pinned against PySCF (`ferric-gw/tests/tda_dft_vs_pyscf.rs`),
and rejects triplets, open shells, meta-GGAs, VV10 and range-separated hybrids.
It is not wired into the CLI or Python, so the gap for users is wiring, not
missing physics.

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
PDEP-RPA gives better molecular \\( C_6 \\) is not established**: an earlier
claim that dRPA@PBE was about 3× better than TS predates a fix to TS's
free-atom volumes and is withdrawn.

Use an augmented basis for any polarizability or \\( C_6 \\): without diffuse
functions the dipole response is badly underestimated.

**TDHF/RPAx \\( C_6 \\) is a measured negative.** The RPAx@PBE static
polarizability of water is close to the DOSD value, but \\( C_6 \\) built on
the same kernel stays about 60% low regardless of the gap. Use
`method.kind = "tdhf-static-polarizability"`
(`examples/water-tdhf-static-alpha.toml`) for static α only. It needs a KS
reference (`[rpa] xc`), and at the default scissor it can hit an excitonic
instability, which is reported as an error rather than a negative α.

## Cite

PDEP: Wilson, Gygi & Galli 2008. RI-RPA quadrature: Eshuis, Yarkony & Furche
2010; minimax grids: Kaltak, Klimeš & Kresse 2014. GW: Hedin 1965; GW100:
van Setten et al. 2015. TDDFT review: Dreuw & Head-Gordon 2005. TS:
Tkatchenko & Scheffler 2009; MBD: Tkatchenko et al. 2012. Full entries in
[References](../reference/references.md).
