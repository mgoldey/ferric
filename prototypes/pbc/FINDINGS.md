# PBC for ferric — review + Python prototype (2026-09-23, spike/pbc-prototype off origin/main e7e21b24)

## What was built
`pbc_gamma.py`: all-electron **Gamma-point periodic RHF** assembled only from *molecular*
integral primitives (PySCF `gto` intor standing in for libint2) plus ONE new primitive, the
analytic Fourier transform of Gaussian pair densities (McMurchie–Davidson E^{ij}_t × (−iG)^t).
PySCF `pbc` is used only as the oracle (`run_h2.py`, `run_h2_oracle.py`, `run_triclinic_p.py`).

Ewald split 1/r = erfc(ωr)/r + erf(ωr)/r:
- SR: real-space lattice sums of ordinary erfc ERIs / Gaussian-nucleus 3c integrals over image shells
- LR: Σ_{G≠0} 4π/G² e^{−G²/4ω²} with analytic pair FTs P_mn(G) = Σ_L FT[φ_m φ_n(·−L)]
- G=0: real-space erfc sums carry π/(ω²Ω); subtracted → standard convention (cancels in J−V for a
  neutral cell; for K it equals PySCF exxdiv=None)
- ω=None ⇒ pure analytic-FT (no SR) — an independent construction of the same energy.

At Gamma everything collapses to a real 8-fold-symmetric nao⁴ tensor I = Σ_{L,M,N}(m0 nL|lM sN):
**the SCF is literally the molecular RHF on (S, h, I, E_nn)**.

## Measured (Bohr/Hartree, cart basis, exxdiv=None)
| check | result |
|---|---|
| E_nn Ewald vs `cell.energy_nuc()` at ω=0.8/1.5/3 | 0 / 1e-16 / 6e-16 |
| lattice S, T vs `pbc_intor` | 5e-15, 1.2e-14 |
| pair FT at G=0 vs lattice S (all elements, after diagonal calibration) | 5.5e-13 |
| H2/STO-3G cubic a=4, pure-AFT vs PySCF AFTDF (mesh 61³) | E −0.9490026912 both; ε identical to 8 digits |
| triclinic, 4 H, s+p basis, pure-AFT vs PySCF AFTDF | ΔE −7.8e-13 |
| ω=3 split, SR cutoffs (bra/ket) 9.5 / 12,13 / 16,18 Bohr | ΔE −1.6e-6 / −5.5e-8 / <1e-10 |
| ω=1.5, 1.0 at default cutoffs | ΔE −3.4e-6, −7.8e-7 (unconverged SR, same mechanism) |
| PySCF GDF (default aux) | ΔE −3.8e-6 (its own fitting error) |

Notes: PySCF `jk_method('RS')` + AFTDF hcore gave identical eigenvalues but a total E of +0.393 —
a PySCF energy-bookkeeping quirk with that combination, not used as oracle.
**Artifact hypothesis tested**: if the ω-dependence were a G=0/convention bug it would NOT shrink
with SR cutoff; it shrank monotonically to <1e-10 ⇒ truncation, not construction. Two
independent constructions (pure-AFT vs Ewald-split) agree.
**Not measured**: speed (this is numpy/PySCF; no claim), k-points, DFT, spherical basis, ECPs.

## The lesson for the Rust design
SR lattice sums through a "supermolecule of images" scale badly: STO-3G H pair densities reach
~20 Bohr, so bra/ket image sets need ~16–18 Bohr, and the ω=3 SR build dominated runtime
(minutes, nao=2). Production needs **per-shell-pair lattice lists with overlap screening** (what
PySCF RSJK/CP2K do), not a ghost-atom supercell `PreparedBasis`. Also note Schwarz is blind to
erfc attenuation (existing memory) — SR screening needs distance-aware bounds.

## What it would take in ferric (from code survey of origin/main)
Reusable as-is: `Operator::erfc/erf` (libint2), ghost-atom image placement (site_basis.rs),
`set_point_charges_extra` (arbitrary positions), `md3c1e.rs:e_table` (+ prim_norm, cart2sph) for
the pair FT, `KBuilder` plug-in point, the whole molecular SCF/DIIS for Gamma.

Needs generalization:
- `Molecule` → add `Cell { mol, lattice, (kpts) }` (keep Molecule untouched for callers)
- `nuclear_repulsion` → Ewald (prototype `ewald_nn`, 30 lines)
- 1e engine: no erf/erfc nuclear op (new_1e hardcodes ω=0) — add libint `erfc_nuclear` op code or
  use Gaussian-nucleus eri3 as the prototype does
- shim: single-basis API only; need bra/ket shells at *translated* centers per call (lattice-pair
  quartets) rather than a giant image basis
- J is concrete `DirectJ/DfJ` in rhf.rs:1117-1160 → needs `dyn JBuilder` to plug a periodic J
- Becke grid / AO-on-grid → periodic images + periodic Becke weights (for KS-DFT)

Missing entirely: lattice/G-vector types, analytic pair FT kernel (Rust port of `pair_ft`),
periodic GDF (3c2e lattice sums + aux FT) for anything bigger than toy, k-points (complex Hermitian
matrices through JBuilder/KBuilder/DIIS/eigh — num-complex exists only in ferric-gw), exchange
divergence treatment beyond exxdiv=None (Madelung/probe-charge or truncated Coulomb),
no FFT dep (only needed for GPW; the AFT/GDF route here needs none).

## Suggested staging
1. Gamma-point HF, all-electron, GDF-free: Rust `pair_ft` + erfc lattice-pair quartets + Ewald →
   reuses molecular SCF unchanged (it is just (S,h,I) as shown). Oracle: this prototype + PySCF AFTDF.
2. Periodic GDF (RSGDF-style: SR 3c2e lattice sums + LR aux FT) → makes Gamma MP2/RPA free via
   existing B-tensor code (supercell Gamma correlation).
3. Gamma KS-DFT: periodic Becke grid.
4. k-points: complex SCF, Bloch-summed integrals, exxdiv='ewald'. Largest refactor; do last.

## Adversarial review (Fable subagent, 2026-09-23) — corrections to the above
- **exxdiv=None is not a physics target.** H2/STO-3G box vs molecular RHF (−1.1167143): None gives
  E−E_mol = +0.168/+0.351/+0.326 at a=4/6/8 (does not converge); exxdiv='ewald' gives
  −0.542/−0.122/−0.028 (converges). Madelung correction (K += v_M·S·D·S, ~10 lines) moves to stage 1.
- **1e-12 vs PySCF AFTDF proves the pair-FT kernel, not the convention** — same algorithm and
  convention. The independent checks are the ω-sweep and a ferric molecular-limit box sweep (add as test).
- **"Reuse molecular SCF unchanged" is FALSE for ferric**: solve_rhf builds S/hcore/E_nn inside
  driver.rs:92-99; K builders are resolved by name (fock_assembly.rs:199,251), J is concrete; no dense
  nao⁴ SCF path exists. Stage 1 needs a Cell-aware prepare + dyn JBuilder + KBuilder injection.
- **nao⁴ tensor and pure-AFT are toy/oracle only** (Si cc-pVDZ 8-atom I = 4.3 GB; pure-AFT G-count
  4e11+). Production: integral-direct Ewald split, **K via periodic RS-GDF from the start** (skip 4-center
  SR lattice quartets).
- Missing items: periodic ECP lattice sums, lindep filtering of diffuse functions at Cell build,
  gradients/stress, low-dim/dipole correction, memory budgeting, Python/CLI Cell surface, charged-cell
  hard error/warn. Code refs: e_table is private (md3c1e.rs:389); shim quartet API is single-basis (shim.h:101).
- RSJK +0.393 offset is undiagnosed, not proven a PySCF quirk.

Revised staging: 0) Cell/G types, Ewald + Madelung, Rust pair_ft anchored P(0)≡S (~1 wk);
1) Gamma HF, exxdiv=ewald default, Cell-aware prepare, J = AFT-LR + SR, K via RS-GDF (~4-6 wk);
2) Gamma KS-DFT GGA + periodic ECP + lindep (~3 wk); 3) k-points (~8-12 wk, biggest refactor).
Biggest risk: SR lattice-sum screening + two-basis shim path — do a measured screening study first.

## Iteration 1 (Python) — 2026-09-23: exxdiv='ewald', molecular-limit sweep, RSJK anomaly

### Code changes
- `pbc_gamma.py`: `_ewald(cell, Z, R, w)` (general point-charge Ewald, background + self term),
  `ewald_nn` now wraps it; `madelung(cell, w=1.0) = -2 * _ewald(cell, [1], [0])` (own Ewald sum;
  PySCF convention from `pbc/tools/pbc.py:madelung` omega=0 branch and
  `df/df_jk.py:_ewald_exxdiv_for_G0`: `vk += madelung * S D S`). `build_integrals(..., exxdiv=None|'none'|'ewald')`
  returns `madelung` (0 for none; other values raise); `rhf(..., kshift=v_M)` does K -> K + v_M S D S.
- New: `run_h2_exxdiv.py` (oracle vs PySCF AFTDF), `run_molecular_limit.py` (box sweep),
  `test_prototype.py` (11 fast tests, 32 s; 3 more under PBC_SLOW=1, 133 s; all pass).

### Measured
| check | result |
|---|---|
| madelung vs `tools.pbc.madelung` (cubic a=4, a=10, triclinic; w=0.5/1/2) | <= 1.9e-15 |
| v_M * a, simple cubic | 2.8372974794806 (literature const) to 1e-11 |
| H2/STO-3G a=4 pure-AFT vs PySCF AFTDF mesh 61^3, exxdiv=None | E -0.949002691179 both, dE -2.6e-14, d eps 1.2e-14 |
| same, exxdiv='ewald' (v_M = 0.709324369870) | E -1.658327061049 both, dE -2.7e-14, d eps 1.3e-14 |
| pair FT at 12 random G != 0, triclinic 4H s+p vs PySCF `ft_aopair` | < 1e-10 (test) |

Molecular-limit sweep (pure-AFT, H2/STO-3G, cubic box a; E_mol = -1.116714325063, pyscf.scf.RHF cart):

| a | nG | v_M | E-E_mol none | E-E_mol ewald | wall |
|---|---|---|---|---|---|
| 4 | 28424 | 0.7093243699 | +1.677e-01 | -5.416e-01 | 22 s |
| 6 | 95768 | 0.4728829132 | +3.5106630420e-01 | -1.2181660904e-01 | 19 s |
| 8 | 227094 | 0.3546621849 | +3.2617770046e-01 | -2.8484484474e-02 | 19 s |
| 10 | 443190 | 0.2837297479 | +2.7294215100e-01 | -1.0787596952e-02 | 22 s |
| 12 | 766432 | 0.2364414566 | +2.3059351114e-01 | -5.8479454843e-03 | 29 s |
| 14 | 1216732 | 0.2026641057 | +1.9900466022e-01 | -3.6594454566e-03 | 20 s |
| 16 | 1815230 | 0.1773310925 | +1.7488076330e-01 | -2.4503291683e-03 | 20 s |
| 20 | 3547596 | 0.1418648740 | +1.4061068304e-01 | -1.2541909319e-03 | 73 s |
| 24 | 6128662 | 0.1182207283 | +1.1749504017e-01 | -7.2568813897e-04 | 170 s |

(pure-AFT runtime is ~flat to a=16: G count grows a^3 but pair-FT image count shrinks a^-3.)
- none - ewald == v_M to ~1e-10 at every a (Gamma: v_M S D S shifts occupied levels by -v_M, D unchanged).
- ewald local exponents d ln|dE| / d ln a: 5.05 (8), 4.35 (10), 3.36 (12), 3.04 (14), 3.004 (16), 3.001 (20), 3.001 (24).
- Tail fit (a=16,20,24) dE = c3/a^3 + c5/a^5: c3 = -10.0282, c5 = -2.14. 3-pt power fit p = 3.0011.
- Independent prediction (exchange Makov-Payne second term, unit-charge pair density |phi_occ|^2):
  c3 = -(4 pi/3) sigma^2, sigma^2 = second central moment of the molecular occupied orbital = 2.3940672
  => c3 = -10.028245. Fitted/predicted agree to 6e-6 relative. Pinned at a=16 (slow test, 1%).

RSJK anomaly (PySCF 2.13, H2 a=4, `RHF(exxdiv=None)` + AFTDF + `jk_method('RS')`):
- `jk_method('RS')` REPLACES with_df by GDF (hf.py:910-911, `not isinstance(with_df, GDF)`).
- vs AFTDF: |dD| 4.4e-13, |dh| 4.8e-13, |dJ| 7.2e-11, |dK| 7.5e-11; E1/EJ/EK identical to 1e-10.
- `mf.energy_nuc()` = 0.7142857143 (= 1/1.4, bare molecular Z_A Z_B/R) vs `cell.energy_nuc()` = -0.6281049380.
- Replacing it: E_RSJK - E_nn(mol) + E_nn(Ewald) - E_AFTDF = -1.8e-11 (exxdiv=None) and -1.8e-11 (ewald;
  RSJK ewald raw E = -0.3159364088).

### Interpretation (provisional)
- Madelung/exxdiv='ewald' reproduces PySCF bit-for-bit-level (1e-14) and the convention is confirmed
  INDEPENDENTLY by the molecular limit: the ewald residual is exactly the predicted O(a^-3) Makov-Payne
  exchange term, with the predicted coefficient. Earlier onset region (a<=12) is not power-law — don't fit it.
- RSJK root cause: operator-precedence bug in `pyscf/pbc/scf/hf.py:760` `SCF.energy_nuc`:
  `if (cell.dimension == 0 and isinstance(self.with_df, df.GDF) or self.rsjk is not None)` parses as
  `(dim==0 and GDF) or rsjk`, so ANY 3-D cell with RSJK gets the non-periodic classical nuclear repulsion
  instead of the Ewald sum; J/K/hcore/eigenvalues are correct. It is a PySCF bug (total-energy only),
  not an RSJK G=0 convention. RSJK totals from PySCF 2.13 must be corrected by cell.energy_nuc() - mf.energy_nuc().

### Corrections to earlier sections
- "RSJK +0.393 ... energy-bookkeeping quirk" / "undiagnosed": now diagnosed (above); it IS bookkeeping, of E_nn only.
- Adversarial review said exxdiv=None "does not converge": it converges, but only as v_M ~ 2.837/a
  (+0.117 at a=24); ewald converges as a^-3. The review's a=4/6/8 numbers are reproduced exactly.
- Mutation testing: flipping (-iG)^t -> (+iG)^t in pair_ft is INVISIBLE to every H2/STO-3G test (s-only:
  t=0 only) and to P(0)==S; only the new G!=0 p-function `ft_aopair` test catches it. Dropping the
  Madelung term (0*kshift) fails the ewald energy + level-shift tests. Both mutations performed and seen to fail.

## Iteration 2 (Rust, main agent) — 2026-09-23
- Commits 1–2 of stage1-design.md built and pass: SCF injection (injected molecular LinK pair ≡ solve_rhf
  bitwise; DirectJ+DirectK vs default DirectJK |dE| 1.8e-13), shifted 1e shells (shift=0 bitwise;
  Σ_L shifted S ≡ pair_ft(G=0) 1.1e-13 for H2O STO-3G/cc-pVDZ).
- **Commit 3 DROPPED — libint2 2.7.2 erf_nuclear/erfc_nuclear are WRONG.** engine.impl.h:895,1057 passes
  rhop = α1α2/(α1+α2) (pair reduced exponent) to erf_coulomb_gm_eval where a point charge needs γp = α1+α2.
  Measured single-s α=1.3 (γp/rhop = 4): libint V_erf(ω) = closed form at 2ω for ω = 0.5/1.5/4
  (−0.95894/−1.60263/−1.78359 vs PySCF int3c2e −0.53888/−1.23926/−1.68751). erf+erfc ≡ nuclear still holds
  to 9e-14, so the sum identity is BLIND to it; only the closed-form test caught it. Reproducer + patch
  kept in reference/pbc/libint-erf-nuclear-bug/. SR nuclear attraction goes via Gaussian-nucleus
  eri3 with Operator::erfc (design option b) instead.

## Iteration 2 (Python, RS-GDF) — 2026-09-23

### Code
- New `pbc_gdf.py` (~250 lines): `build_gdf(cell, auxbasis, w=1.0, ...)` -> B[P,m,n] with
  I ~= sum_P B^P_mn B^P_ls in exactly the pbc_gamma G=0 convention; `jk_from_B`, `eri_from_B`,
  `ferric_basis(name, symbols)` (reads ferric's bundled BSE JSON), `even_tempered(...)`, `aux_ft`.
  Only PySCF *molecular* intor (int2c2e/int3c2e under `with_range_coulomb(-w)`, int1e_ovlp) +
  `pair_ft`/`hermite_E` from pbc_gamma. Aux spherical by default (cart2sph applied after), orbital cart.
- `pbc_gamma.py`: `rhf(..., jk=None)` hook (D -> J, K) so SCF can run from B tensors; `pair_ft`
  gets a per-primitive-pair G window (e^{-G^2/4p} < thresh*e^-10 dropped) — needed to make the
  LiH pure-AFT reference affordable (289k G); existing tests unchanged (still 1e-12).
- `test_prototype.py`: +7 fast tests (suite 18 passed / 3 slow skipped, ~56 s).

### Method (what the numbers below were measured with)
All quantities in the G=0-dropped kernel v'(G)=4pi/G^2, G!=0 (the target I's own kernel):
J2[P,Q]=(P|Q)', J3[mn,P]=(mn|P)', B = s^-1/2 U^T J3^T, J2 = U s U^T, keep s > 1e-10 (PySCF's
LINEAR_DEP_THR). Evaluation: SR = sum_T (P_0|Q_T)_erfc and sum_{L,T} (m_0 n_L|P_T)_erfc over
molecular image shells; LR = (1/Omega) sum_{G!=0} 4pi/G^2 e^{-G^2/4w^2} conj(A(G)) X_P(G);
G=0: subtract pi/(w^2 Omega) q_P q_Q from J2 and pi/(w^2 Omega) S_mn q_P from J3 (q_P = int chi_P,
nonzero for s and for Cartesian d/f r^2 components). Same scheme as PySCF RSGDF (rsdf_builder.py
get_2c2e g0_fac, gen_j3c_loader vbar). SR cutoffs from erfc(sqrt(theta)R)/R = prec/q_max^k
(theta = 1/(1/a_aux + 1/2a_orb + 1/w^2)), prec 1e-13; gcut = 2w sqrt(ln 1/prec).
Reference = exact pure-AFT I from pbc_gamma (same S, h, E_nn in both; only the ERI differs).

### Measured
Exactness anchor (H2, one primitive s per H, al=0.5, cubic a=4). Aux = 24 s functions of exponent
2al at the 8 half-lattice classes (A_m+A_n+h.a)/2, h in {0,1}^3, for (m,n) in {11,22,12} — every
periodic pair product lies exactly in this span:

| variant | max\|I_fit - I\| | dE (Ha) |
|---|---|---|
| anchor, w=0.8 / 1.2 | 1.6e-12 / 1.2e-11 | -1.2e-12 / -8.0e-12 (none and ewald identical) |
| MUTANT G=0 removed from J2 only | 5.3 | +4.3 |
| MUTANT G=0 removed from neither | 6.1e-2 | +4.9e-2 |
| MUTANT SR aux images truncated to T=0 | 2.9e-2 | +1.1e-2 |
| MUTANT aux FT phase e^{+iG.A} | 7.1e-2 | -4.0e-2 |
| aux incomplete: only h=0 class (3 aux) | 6.9e-4 | -2.5e-4 |
| aux incomplete: 7/8 classes (drop h=(1,1,1)) | 7.7e-10 | -2.4e-10 |

Oracle cross-checks: aux FT (def2-universal-jkfit H, cart d) vs PySCF `ft_ao` at 16 G incl. G=0:
<1e-12. H2/STO-3G a=4, cart aux: ours vs PySCF RSGDF (same aux) E difference -6.1e-13
(cc-pvdz-ri), -2.4e-12 (def2-universal-jkfit); PySCF GDF (compensated-charge scheme) = RSGDF to
2e-12. LiH, cart cc-pvdz-ri: ours -1.7872e-5 vs PySCF RSGDF -1.774e-5 (diff 1.3e-7; ours kept
79/81 aux at 1e-10 — cause of the 1.3e-7 not isolated). w-independence of B B^T (w=0.7 vs 1.4,
cc-pvdz-ri): <1e-9 (test).

Fitting error dE = E_GDF - E_exact (Hartree, w=1, spherical aux unless noted). exxdiv=none and
exxdiv=ewald give the SAME dE in every row to all printed digits (the Madelung term is v_M S D S,
fit-independent):

| aux | H2/STO-3G a=4 (nao 2) naux, dE | triclinic 4H s+p (nao 16) naux, dE | LiH/STO-3G a=4 (nao 6) naux, dE |
|---|---|---|---|
| def2-universal-jkfit | 36, -4.47e-6 | 72, -1.72e-5 | 65 kept/69, -8.37e-5 |
| cc-pvdz-ri (ferric's cc-pvdz-rifit alias) | 28, -1.97e-6 | 56, -2.66e-5 | 70, -1.94e-5 |
| def2-svp-rifit | 28, -6.18e-6 | 56, -8.99e-5 | 38/39, -9.74e-4 |
| ET l<=0 b=2.2 | 18, -5.75e-5 | 36, -4.66e-3 | – |
| ET l<=1 b=2.2 | 69/72, -3.69e-6 | 139/144, -2.46e-5 | (PySCF RSGDF only, b=2.0, cart) 96, +7.4e-6 |
| ET l<=2 b=3.0 / 2.2 / 1.7 | -3.0e-7 / -2.9e-8 / -1.3e-8 | -1.1e-6 / -2.2e-7 / -1.3e-7 | b=2.0 cart, 197/240: +2.3e-8 (ours); PySCF RSGDF same aux: +6.5e-4 |
| ET l<=3 b=2.2 | 252/288, -2.8e-9 | – | b=2.0 cart: ours not run (cost); PySCF RSGDF 480 aux: +4.2e-3 |

(ET: amin 0.1 (0.04 LiH), up to 40 (80 LiH); "kept/naux" where the 1e-10 lindep filter dropped
functions.) Cart aux has smaller error than spherical (H2 cc-pvdz-ri -1.71e-6 vs -1.97e-6; jkfit
-3.84e-6 vs -4.47e-6) — Cartesian d carry an extra s-type function.

Energy split at the exact density (triclinic): cc-pvdz-ri dE_J -7.6e-5, dE_K +4.9e-5;
def2-universal-jkfit dE_J -6.5e-5, dE_K +4.8e-5 (max|dK| 4-6e-4). H2 (one occupied orbital) has
dE_K = -dE_J/2 identically, so it says nothing about K separately.

LiH ET l<=2 (cart, amin 0.04, 240 aux) lindep scan on ONE saved (J2, J3): metric eig min -1.4e-8
(noise), max 85; lindep 1e-12/1e-10/1e-8/1e-6/1e-4 -> kept 216/197/181/168/155, dE +2.8e-8 / +2.3e-8 /
+1.5e-8 / -8.5e-9 / -3.4e-7. PySCF RSGDF (2.13, defaults, cell.precision 1e-12) on the same aux
gives +6.5e-4 (l<=2) and +4.2e-3 (l<=3): it DEGRADES as the aux set grows, ours does not. Cause in
PySCF not isolated (candidates: Cholesky path accepting a noisy near-singular j2c; its j2c mesh).

Metric noise: largest ET set, J2 vs an independent pure-AFT J2 (488k G): 3.7e-9 before scaling the
SR cutoff by q_max (normalised diffuse aux carry |q| up to ~22), 1.0e-10 after; negative metric
eigenvalues -4.9e-9 -> -1.8e-10. dE unchanged (-1.27e-8) across w=0.7/1.0/1.4, prec 1e-13/1e-15,
lindep 1e-12/1e-10/1e-8 (1e-6: -1.50e-8) => the ET plateau is basis incompleteness, not noise.

Counts (w=1, prec 1e-13; the only cost statements made here):
| system | nG (LR) | pair images L (rcut_pair) | aux images T (3c / 2c) | SR 3c shell triplets | B size |
|---|---|---|---|---|---|
| H2 a=4, cc-pvdz-ri | 1364 | 675 (20.8 Bohr) | 269 | – | 28 x 2 x 2 |
| tri, cc-pvdz-ri | 2102 | 567 | 289 | – | 56 x 16 x 16 |
| LiH a=4, cart cc-pvdz-ri | 1364 | 3825 (37.3 Bohr) | 1631 / 2201 | 2.4e9 | 79 x 6 x 6 |
| pure-AFT reference (for contrast) | 28424 (H2) / 289090 (LiH) | | | | nao^4 |
The image counts are set by the most diffuse ORBITAL primitive (Li 2sp 0.048 -> 37 Bohr) and
most diffuse AUX primitive; the SR 3c triplet count (nb0 x nb0*|L| x naux_sh*|T|, unscreened) is
the dominant cost. LiH builds took 12-14 min each in this unscreened numpy/PySCF form (not a claim).

### Artifact hypotheses tested
- "The fit reproduces I only because the G=0 errors cancel between J2 and J3 by accident" vs
  "the convention is consistent": if the latter, the anchor is exact at every w AND the two
  one-sided mutations fail by O(1). Observed: 1e-12 at w=0.8 and 1.2; mutants 5.3 / 6e-2.
- "Error is SR/LR truncation, not fitting": then dE would move with w/prec/lindep. It does not
  (table above), and it falls monotonically with aux completeness (l and beta) to 2.8e-9.
- "Same-algorithm agreement with PySCF proves only the algorithm": the anchor is the independent
  check (no PySCF, exact limit); PySCF agreement (6e-13) confirms normalisation/cutoffs.
- "The anchor is vacuous (would pass a broken build)": 4 mutations each fail it by >= 1e-2.
- Side finding: pbc_gamma's default rcut_1e=22 truncates S/T for Li STO-3G (2sp exponent 0.048):
  LiH exact E moved by 6.4e-5 and P(0)-vs-S calibration check read 8.9e-5 until rcut_1e=44
  (then 5.6e-9). Iteration 0/1 H-only results are unaffected.

### Interpretation (provisional, 2026-09-23)
- G=0 / charged aux: fit in the G=0-dropped kernel, removing pi/(w^2 Omega) q q^T and
  pi/(w^2 Omega) S q^T from the erfc real-space sums (RSGDF). No compensating charges needed; the
  metric stays positive (semi)definite (min eig ~1e-10-level noise only) because Gaussians cannot
  build a constant. PySCF's compensated-charge GDF gives the same number (2e-12) — the two are
  evaluation schemes for the same primed metric, and RS is the one that needs no extra smooth
  functions or eta choice.
- Accuracy: ferric-bundled JK/RI aux bases give 2e-6..8e-5 Ha on these toy cells (µHa to tens of
  µHa), the same size as molecular DF errors, with J and K errors partially cancelling. K via GDF is
  NOT µHa-exact with stock aux sets; it is converged-by-aux-basis (even-tempered l<=2-3 reaches
  1e-7..3e-9). def2-svp-rifit (an MP2 set) is unsuitable for JK (1e-3 on LiH).
- Diffuse aux in a small cell makes the periodic metric near-singular (LiH: 43/240 eigenvalues
  below 1e-10); an eigen-decomposition with a lindep cut is stable over 8 decades of threshold, and
  PySCF's default path on the same aux is not. The Rust metric solve must be eig/pivoted-Cholesky
  with an explicit drop count, never plain Cholesky.
- exxdiv=ewald adds no fitting error at Gamma (identical dE).
- NOT measured: spherical orbital basis, anything beyond 16 AOs, k-points, DFT, wall time.

## Future stages (added 2026-09-24): open shell, MP2, RPA

Current plan: 0 Cell/Ewald/pair-FT ✅ · 1 Gamma RHF + RS-GDF (nearly done) · 2 Gamma KS-DFT GGA + periodic
ECP + lindep · 3 k-points. The stages below slot in after their dependencies; Gamma versions are
"supercell" methods (sample the BZ by enlarging the cell), k-point versions come after stage 3.

What the code survey says (2026-09-24): `ri_mp2`, `u_ri_mp2`, `run_pdep_rpa`, `run_u_pdep_rpa` all take
(mol, obs, dfbs) and build B internally through `ThreeIndexSource` (rimp2.rs:1300, u_rimp2.rs:149,
ferric-rpa lib.rs:493/1046). RPA already has an injection seam: `run_pdep_rpa_from_intermediates`
(lib.rs:895) consumes precomputed `RpaIntermediates`. MP2 has none. The common prerequisite is therefore
**one seam: a `ThreeIndexSource` (or B-tensor) built from the periodic RS-GDF B**, fitted in the periodic
Coulomb metric, instead of from molecular (P|Q)/(mn|P).

| # | Stage | Depends on | Work | Oracle / anchors |
|---|---|---|---|---|
| 4 | **Gamma UHF** | 1 | `solve_uhf_injected` mirroring RHF injection (uhf.rs:34 builds S/h/E_nn in `prepare`, same seam); RS-GDF K per spin (K_α, K_β from one B), Madelung shift per spin | PySCF `pbc.scf.UHF` AFTDF/GDF; closed-shell UHF ≡ RHF bitwise-ish; H-atom lattice / O2 in a box vs molecular UHF box-limit sweep |
| 5 | **Gamma UKS / ROKS** | 2, 4 | reuse stage-2 periodic grid; spin-polarized XC already exists molecularly | PySCF `pbc.dft.UKS`; box limit vs molecular UKS |
| 6 | **Gamma RI-MP2** | 1 (RS-GDF) | periodic `ThreeIndexSource` from RS-GDF B → MO transform → existing `ri_mp2` kernels; frozen core via `active_occ` | anchor: dense AFT (ia\|jb) MP2 in the trivial-aux limit; PySCF `pbc.mp.RMP2` at Gamma; box limit → molecular RI-MP2 |
| 7 | **Gamma dRPA / PDEP-RPA** | 1 (RS-GDF) | build `RpaIntermediates` from the periodic B and call `run_pdep_rpa_from_intermediates` (seam exists) | PySCF `pbc` RPA at Gamma if available, else dense-AFT dRPA oracle; box limit → molecular `run_pdep_rpa` |
| 8 | **Gamma UMP2 / URPA** | 4, 6, 7 | per-spin periodic B (`compute_rpa_intermediates_spin` path, `run_u_pdep_rpa` from UHF/ROHF) | closed-shell U ≡ R; PySCF `pbc.mp.UMP2`; box limit |
| 8b | **Gamma LMP2 (local MP2)** | 6 | periodic orbital localization at Gamma (Boys/Pipek-Mezey on the supercell = Wannier functions at Gamma; localization must be translation-aware, i.e. an LMO and its lattice images are the same function, so domains/pairs are defined modulo L), then the existing `amplitude_lmp2` machinery (ferric-mp2 lmp2_amplitude.rs: VV-HV virtuals, Eq-8 mask, pair gate, per-pair RI fit) with pair lists over (i, j+L) instead of (i, j); the per-pair local RI fit must use the periodic Coulomb metric (RS-GDF), not the molecular one | anchor: eps=0 ≡ Gamma RI-MP2 (stage 6) to 1e-9, same as the molecular lmp2 anchor; box limit → molecular `amplitude_lmp2`; PySCF has no periodic LMP2, so the canonical Gamma MP2 is the oracle. Payoff: this is where periodicity HELPS locality — a large supercell is exactly the regime where local correlation should win, so it is also a clean test of the local-correlation lane (read the local-correlation-rollup memory before claiming any scaling) |
| 9 | **k-point MP2 / RPA / UHF** | 3 + above | complex B^P_{ia}(k,k'), momentum conservation, k-point Madelung; the Gamma code becomes the k=0 anchor | PySCF KMP2/KRPA; Gamma-supercell ≡ k-mesh equivalence (NxNxN supercell at Gamma = NxNxN k-mesh) |

**Open physics questions to MEASURE before claiming anything (not assumed):**
- **Exchange divergence in correlation.** Stage 1 treats it in K (exxdiv=ewald); the Madelung shift also
  moves occupied orbital energies by −v_M and so changes the MP2/RPA denominators. Which orbital energies
  the correlation step should use (shifted or unshifted) is a convention that decides finite-size
  convergence. Settle it with the box-limit sweep against molecular MP2/RPA, the same independent
  oracle that confirmed exxdiv=ewald for HF.
- **Finite-size convergence.** Gamma-supercell correlation energies converge slowly with cell size
  (N_k^(-1) class). Report the scaling fit alongside every number, and do not quote a
  single-cell value as a solid-state result.
- **Aux quality.** Molecular RI aux (cc-pvdz-ri) gave µHa-level HF fitting error at toy scale. Correlation
  is more aux-sensitive, so remeasure with the stage-6 anchor.
- **Cost.** SR 3c triplet counts are already 17–21M for the 4-atom triclinic toy. MP2/RPA add
  O(N⁴)/O(N⁴ log) on top, so the step-8 screening study must land first.

Prototype order (Python first, per convention): 6 (Gamma MP2 from pbc_gdf.py B) → 7 (dRPA from the same B)
→ 8b (LMP2, eps=0 anchored to 6) → 4 (UHF) → 8. Each gets its trivial-limit anchor before any sweep.

## Iteration 3 (Python, Gamma MP2) — 2026-09-24

### Code
- New `pbc_mp2.py`: `gamma_mp2(C, eps, nocc, eri=|B=, frozen=)`, `bia_from_B` (B^P_mn -> B^P_ia),
  `mp2_energy` (os/ss split), `denominators(eps_none, nocc, v_M, 'shifted'|'unshifted')` (strict),
  `dipole_prediction_h2_minimal` (a^-3 moment model, below). Real arithmetic, closed shell.
- `pbc_gamma.rhf(..., return_mo=True)` also returns C (default return unchanged).
- Drivers: `run_mp2_anchor.py`, `run_mp2_oracle.py` (PySCF pbc), `run_mp2_box_limit.py BASIS a...`,
  `run_mp2_fit_error.py`.
- `test_prototype.py`: +9 fast tests (whole suite 27 passed / 3 slow skipped, 51 s).

### PySCF conventions (read from source, PySCF 2.13)
- `pbc.mp.RMP2` = molecular `mp2.RMP2` + `with_df.ao2mo`; denominators from `mf.mo_energy`, i.e. from
  whatever exxdiv the HF used. exxdiv=None -> unshifted; exxdiv='ewald' -> occupied shifted by -v_M.
  `KMP2` the same (`self.mo_energy = mf.mo_energy`, no Madelung code in kmp2.py).
- `pbc.cc` (RCCSD/UCCSD/KRCCSD/KCCSD/KUCCSD) rebuilds the Fock with exxdiv=None and then ALWAYS applies
  `_adjust_occ(eps, nocc, -madelung)` ("Without the correction, MP2 energy may be largely off").
  So PySCF CC is always 'shifted'; PySCF MP2 follows the HF exxdiv. `ccsd(mbpt2=True)` is NOT the CC
  convention (it calls RMP2(mf) with mf.mo_energy).
- The ov integrals carry no G=0 term in either code (ov pair densities are neutral: C_o^T S C_v = 0).
  At Gamma C is identical under none/ewald, so the ONLY difference between conventions is eps_occ -= v_M.

### Measured
Anchor (trivial-aux limit, one s primitive al=0.5 per H, aux = all periodic pair products):

| cell | nocc/nvir | naux | |dMP2| B vs dense AFT (same C / B-SCF C), none & ewald |
|---|---|---|---|
| H2 a=4 | 1/1 | 24 | 1.2e-13 / 1.3e-13, 7.2e-14 / 7.6e-14 |
| triclinic 4H | 2/2 | 80 | 2.1e-13 / 1.9e-13, 9.8e-14 / 7.0e-14 |

Mutations (tri 4H unless noted): exchange term dropped +1.4e-6 (H2: -7e-14, invisible, nocc=nvir=1);
Cv for both ov indices -6.6e-3 (H2 +9.2e-4); aux 7/8 classes +1.1e-8 (H2 +3.7e-13, invisible);
Madelung sign flipped, B_ia with wrong C: caught by the pinned-PySCF / box-limit / formula tests
(all four code mutations run, each failed >= 3 tests). G=0 mutants of pbc_gdf: ALL invisible to MP2 in
the anchor (|dMP2| < 1e-13 while max|dI| = 5.3 / 7.7e+1 / 3.5e-2): the J3 term c0 S_mn q_P is killed by
C_o^T S C_v = 0 (structural), and the J2 term acts only through the fitted charge of ov densities,
which is exactly 0 in the anchor. With cc-pvdz-ri (tri): fitted ov charge up to 0.87, J2-side mutant
moves MP2 by 7.8e-7 (fit error 3.3e-5); J3-side mutant 0; H2: both 0.

Oracle, PySCF 2.13 AFTDF mesh 61^3 (ours = pure-AFT dense ERI):

| system | ours unshifted | PySCF RMP2 exxdiv=None | ours shifted | PySCF RMP2 exxdiv=ewald | PySCF CC-eris MP2 (none / ewald) |
|---|---|---|---|---|---|
| H2/STO-3G a=4 | -5.891222456423e-3 | +4.2e-16 | -3.881428851328e-3 | +2.9e-16 | shifted +2.9e-16 / +2.9e-16 |
| tri 4H s+p | -6.080626811746e-2 | -1.2e-10 | -4.009219916836e-2 | +3.7e-11 | shifted -5.7e-11 / +3.8e-11 |

(v_M = 0.7093243699 / 0.6224368790. The conventions differ by 34% / 34% of E_corr at these cells.)

Box limit vs molecular MP2 (pyscf.mp.MP2, cart, same geometry), H2 in cubic boxes. ERI by Ewald split
w = min(1, 8/a), bra images 18 Bohr, ket 18+6/w; checked at a=8 vs pure-AFT (STO-3G, 1e-14) and vs
wider cutoffs / w=0.5 (6-31G, 4e-16 in E_MP2). E_MP2(mol) = -1.315787005264e-2 (STO-3G),
-1.739045734672e-2 (6-31G). Residual dE = E_corr(pbc) - E_corr(mol):

| a | STO-3G shifted | STO-3G unshifted | 6-31G shifted | 6-31G unshifted |
|---|---|---|---|---|
| 6 | +3.959e-3 | -6.181e-4 | +3.576e-3 | -4.089e-3 |
| 8 | +1.415e-3 | -3.032e-3 | +1.503e-3 | -4.435e-3 |
| 10 | +6.804e-4 | -2.932e-3 | +7.823e-4 | -3.852e-3 |
| 12 | +3.909e-4 | -2.568e-3 | +4.537e-4 | -3.329e-3 |
| 14 | +2.459e-4 | -2.244e-3 | +2.860e-4 | -2.893e-3 |
| 16 | +1.646e-4 | -1.980e-3 | +1.917e-4 | -2.542e-3 |
| 20 | +8.419e-5 | -1.589e-3 | +9.815e-5 | -2.031e-3 |
| 24 | +4.868e-5 | -1.321e-3 | +5.679e-5 | -1.684e-3 |
| 32 | +2.052e-5 | -9.835e-4 | +2.395e-5 | -1.249e-3 |
| 40 | +1.050e-5 | -7.813e-4 | +1.226e-5 | -9.909e-4 |

- Local exponents d ln|dE|/d ln a, shifted: STO-3G 3.58 (8), 3.28, 3.04, 3.007, 3.005 ... 3.003 (40);
  6-31G 3.01 (8), 2.93, 2.99, 2.995 ... 3.002 (40). 3-pt tail (24,32,40): 3.0032 / 3.0017.
  Fit c3/a^3 + c5/a^5 (24,32,40): c3 = 0.671358 (STO-3G), 0.784083 (6-31G).
- Unshifted local exponents: STO-3G -5.5 (8), 0.15, 0.73, 0.88, 0.94, 0.99, 1.01, 1.03, 1.03 (40);
  6-31G similar, 1.04 at 40 (approaching 1 from above: a c2/a^2 term of the same sign).
  Fit c1/a + c2/a^2 + c3/a^3 (24,32,40): c1 = -0.029835 / -0.037655.
- (unshifted - shifted) / (c1/a + c2/a^2 predicted) = 0.9985..1.0077 for a >= 12, both bases.
- Denominator check at a=40, STO-3G: measured 2(d eps_occ - d eps_vir) a^3 = -27.33 vs moment model
  -2(4pi/3)(sigma^2 + |d_ia|^2) = -27.32. Individual eps carry an extra common constant
  (+k R_tot/2, Bethe-type potential of the neutral cell) that cancels in every denominator.
- Wall time per box point 1-8 s for a >= 8 (w-scaled Ewald split); a=6 55 s / 315 s (6-31G).

Hypotheses stated before the sweep (predictions are from MOLECULAR quantities only):
- physics, shifted: dE = c3/a^3 + O(a^-5); for a one-occupied/one-virtual system at frozen (symmetry-fixed)
  orbitals, delta(1|2) = -(4pi/3 Omega)[d1.d2 - (q1 R2 + q2 R1)/2] (cubic, G=0-dropped kernel, after
  Madelung) gives d(ia|ia) = -k|d_ia|^2, d eps_i = -k sigma_i^2, d eps_a = +k|d_ia|^2, k = 4pi/3a^3,
  => c3 = 0.67109405 (STO-3G). MEASURED 0.671358 (4e-4 rel; the c5 fit absorbs the rest).
- physics, unshifted: D -> D + 2 v_M, so dE = sum N/(D + 2v_M) - sum N/D + O(a^-3):
  c1 = -2 (v_M a) sum N/D^2 = -0.029903 (STO-3G) / -0.037746 (6-31G). MEASURED -0.029835 / -0.037655.
- artifact (a) broken transform / missing images: plateau at nonzero dE — not seen (both converge to 0).
- artifact (b) Madelung sign flipped: 'shifted' would converge as 1/a with ~2 c1 — not seen (exponent 3).
Distinguishable by construction (exponent 3 vs 1 vs 0), and the coefficients are predicted, not fitted.

Aux fitting error dMP2 = E_MP2(B) - E_MP2(dense AFT), spherical aux, w=1; "same C" = exact SCF orbitals,
"own SCF" = SCF also on B (the full pipeline):

| aux | H2/STO-3G a=4 shifted: same C / own SCF (rel) | unshifted same C | tri 4H s+p shifted: same C / own SCF (rel) | unshifted same C |
|---|---|---|---|---|
| cc-pvdz-ri (28 / 56) | +6.31e-7 / +7.02e-7 (-1.6e-4) | +9.58e-7 | +3.33e-5 / +3.65e-5 (-8.3e-4) | +4.85e-5 |
| def2-universal-jkfit (36 / 72) | +2.73e-6 / +2.85e-6 (-7.0e-4) | +4.15e-6 | +7.61e-5 / +7.92e-5 (-1.9e-3) | +1.04e-4 |
| def2-svp-rifit (28 / 56) | +2.19e-6 / +2.21e-6 (-5.6e-4) | +3.32e-6 | +4.85e-5 / +5.25e-5 (-1.2e-3) | +7.38e-5 |
| ET l<=2 b=2.2 amin 0.1 (146/162, 291/324) | +5.1e-10 / +1.4e-9 (-1.3e-7) | +7.7e-10 | +2.44e-6 / +2.57e-6 (-6.1e-5) | +3.52e-6 |

Same-molecule molecular DF-MP2 (pyscf DFMP2, H2/STO-3G, same geometry): cc-pvdz-ri +1.73e-6 (-1.3e-4),
def2-universal-jkfit +3.40e-6 (-2.6e-4), def2-svp-rifit +6.26e-6 (-4.8e-4). All fit errors positive
(underbinding); the relative error is convention-independent (same rel in both columns).

### Interpretation (provisional, 2026-09-24)
- **Shifted denominators (Madelung-corrected occupied energies) converge to the molecular limit, as a^-3,
  with a coefficient predicted from molecular moments; unshifted converge as 1/a (c1 = -2 v_M a sum N/D^2,
  also predicted).** At a=40 the unshifted error is still -7.8e-4 / -9.9e-4 Ha (4.5-5.7% of E_corr); the
  shifted error is 1.0e-5 / 1.2e-5. This is the same a^-3 mechanism (Makov-Payne second moment, now
  also the ov-dipole tinfoil term) that Iteration 1 found for HF with exxdiv=ewald.
- Small-box trap: at a=6 (STO-3G) unshifted is 6x CLOSER to the molecule than shifted (-6e-4 vs +4e-3) —
  the two error sources cross there. A single small-cell comparison would pick the wrong convention; only
  the tail (a >= 12, where the shifted exponent has settled at 3) decides it.
- At Gamma the choice is purely "eps_occ from the ewald Fock" — PySCF CC does this always, PySCF MP2 only
  when the HF was run with exxdiv='ewald'. Correlation should NOT inherit exxdiv=None from the HF.
- The trivial-aux MP2 anchor proves transform + assembly (1e-13), but is structurally blind to both
  G=0 bookkeeping terms of the fit (J3 via S-orthogonality, J2 via zero fitted charge); those remain
  guarded only by the HF ERI anchor of Iteration 2. With a real aux the J2 term leaks weakly (7.8e-7).
- Aux error: periodic RS-GDF MP2 fitting error with cc-pvdz-ri is 1.6e-4 relative on H2, matching the
  molecular DF-MP2 error for the same molecule/aux (1.3e-4) — the periodic metric adds no new error class.
  tri 4H s+p (p on H, small cell, overlapping images) is 8e-4 relative; the ET l<=2 set only gets to 6e-5
  there (it needs higher l for p-orbital products), vs 1e-7 on the s-only H2. cc-pvdz-ri is the best
  stock set here (as molecularly: it is the MP2 RI set); def2-universal-jkfit is 2-4x worse.
- NOT measured: finite-size in a real solid (a molecule in a box has no band dispersion; the a^-3 law is
  the ISOLATED-molecule limit, not the N_k^-1 solid-state law), spherical orbital basis, frozen core in a
  periodic run, anything > 16 AOs, cost, UHF/UMP2, k-points.

### For the Rust port (stage 6)
- Denominators: shifted (eps_occ from the exxdiv=ewald Fock, = none-eigenvalues - v_M). If the stage-1
  SCF result already came from exxdiv=ewald, ScfResult eps are correct as-is; if it came from exxdiv=None,
  the MP2 driver must apply the shift itself (PySCF-CC style) or refuse. Pin it with the a^-3 box test.
- Seam: `ri_mp2_spin_components` builds its own molecular metric (`coulomb_metric_2c` +
  `metric_inverse_sqrt`) and `ThreeIndexSource::build_band_screened` from (obs, dfbs); there is no
  from-array constructor. The periodic B is already metric-dressed (eig + lindep drop). Least new surface:
  split the kernel after `b_flat` into `ri_mp2_from_b_ov(b_ov, eps_occ, eps_vir)` and have the periodic
  path transform the SAME B^P_mn used for SCF K to B^P_ia. A periodic ThreeIndexSource would also need an
  injected metric solve (and must never Cholesky: Iteration 2's LiH metric) — larger change, only needed
  once B must stream/spill.

## Iteration 4 (Python, Gamma dRPA) — 2026-09-24

### Code
- New `pbc_rpa.py`: `drpa_quad(Bia, eo, ev, n=|None)` (frequency integral; n=None doubles GL n until
  |dE| < 1e-12), `drpa_plasmon` / `drpa_riccati` (dense (ia|jb), no quadrature, no aux), `direct_mp2`,
  `drpa_second_order_quad` (-(1/2pi) int tr Pi^2/2), `gamma_drpa(C, eps, nocc, eri=|B=, method=)`,
  `ov_factor`, `drpa_moment_prediction_h2_minimal` (nov=1 closed form), `r2_kernel_c3(mol)` (general
  a^-3 predictor, below). Denominators reuse `pbc_mp2.denominators` (shifted/unshifted).
- Formula (ferric energy.rs and PySCF gw/rpa.py, identical): Pi = 4 B diag(e_ia/(w^2+e_ia^2)) B^T >= 0,
  E_c = (1/2pi) int_0^inf dw [ln det(1+Pi) - tr Pi], GL mapped w = x0(1+x)/(1-x) (same map as ferric
  `gauss_legendre_nodes` / PySCF `_get_scaled_legendre_roots`). Summand evaluated as sum_k log1p(l_k) - l_k
  over eigenvalues of Pi: `slogdet(1+Pi) - tr Pi` cancels catastrophically at large-w nodes (w^2 weights;
  measured 1.9e-10 drift between n=512 and n=1024 on H2O/6-31G, naux ~100, gone after the change).
- Drivers: `run_rpa_formula.py` (molecular), `run_rpa_anchor.py`, `run_rpa_oracle.py`,
  `run_rpa_box_limit.py BASIS a...`, `run_rpa_fit_error.py`, `run_rpa_quadrature.py`, `run_rpa_c3_prediction.py`.
- `test_prototype.py`: +10 fast tests (suite 37 passed / 3 slow skipped, 79 s).
- PySCF 2.13 has NO periodic RPA: `pyscf/pbc` has gw (krgw_ac/cd, kgw_slow) and tdscf but no rpa module and no
  RPA correlation energy. Molecular `pyscf.gw.rpa.RPA` (= dRPA, DF-only) is the convention oracle.

### Measured
Molecular formula checks (run_rpa_formula.py; exact ERI unless noted):

| check | H2O/6-31G (nocc 5, nvir 8) | H2/STO-3G |
|---|---|---|
| E_dRPA (plasmon) | -1.37546830507603e-1 | -2.06589071750135e-2 |
| Riccati - plasmon / converged quad - plasmon | 3.3e-14 / -1.0e-14 (n=128) | 2.2e-15 / -1.2e-16 (n=64) |
| -(1/2pi) int tr Pi^2/2 - direct MP2 | -4.0e-15 | -5.1e-16 |
| (E(lam V)/lam^2 - dMP2)/lam, lam = 1e-2, 1e-3 | 0.09614, 0.09676 (O(V^3) term) | 0.00761, 0.00764 |
| ours (GL n=40) - pyscf.gw.rpa.RPA (nw=40), PySCF's own DF, cc-pvdz-ri / jkfit | -6.3e-13 / -8.0e-13 | -9.0e-14 / +9.1e-14 |
| GL x0=0.5 error n = 10 / 20 / 40 / 80 | 1.9e-5 / -2.3e-6 / 7.4e-10 / -1.1e-14 | 6.7e-8 / 5.4e-14 / ~0 / ~0 |

Exactness anchor (a): trivial-aux RS-GDF B + frequency quadrature vs dense pure-AFT (ia|jb) + plasmon.
Same cells/aux as the MP2 anchor (one s primitive al=0.5 per H; aux = every periodic pair product):

| cell | conv | E_dRPA (plasmon) | Riccati | B-quad same C | B-quad own B-SCF | (b) B trPi^2 term - dMP2 | dRPA - dMP2 |
|---|---|---|---|---|---|---|---|
| H2 a=4 | shifted | -7.5076621772474e-3 | -1e-16 | -1.2e-13 | -1.3e-13 | -1.4e-13 | +9.9e-4 |
| H2 a=4 | unshifted | -1.1393259292548e-2 | -6e-16 | -1.8e-13 | -1.9e-13 | -2.3e-13 | +2.4e-3 |
| tri 4H | shifted | -2.1056730916346e-2 | +5e-16 | -1.7e-13 | -1.3e-13 | -1.9e-13 | +3.8e-3 |
| tri 4H | unshifted | -3.4654484691117e-2 | +3e-15 | -3.3e-13 | -3.0e-13 | -4.1e-13 | +1.2e-2 |

Mutations (shifted, exact orbitals), dE vs plasmon: spin factor 4->2 in Pi +5.5e-3 (H2) / +1.5e-2 (tri);
1/pi instead of 1/2pi -7.5e-3 / -2.1e-2; exchange-type K (A = D + 2K - K_x) +5.5e-3 (H2: identical to the
factor-2 mutant, nocc=nvir=1) / +1.5e-2; unshifted eps fed as shifted -3.9e-3 / -1.4e-2; Cv for both ov
indices +1.6e-3 / -6.3e-3; aux 7/8 classes +6e-13 (H2, invisible) / +2.1e-8 (tri). The three G=0 mutants of
pbc_gdf (J2 only / J3 only / kept in both): ALL invisible to dRPA (|dE| <= 1.7e-13), same structural reason
as MP2 (C_o^T S C_v = 0; zero fitted ov charge in the anchor). Code mutations of the tests (factor 4->2 in
`_summand`, 1/pi in `drpa_quad`, 4K->2K in the plasmon, sign of |d|^2 in the moment model): each fails >= 1
test (6/6/7/1 failures).

PySCF oracles (run_rpa_oracle.py; PySCF 2.13 pbc RHF on AFTDF mesh 61^3, exxdiv None -> unshifted,
'ewald' -> shifted; ours = pure-AFT ERI + plasmon, and our RS-GDF with cart cc-pvdz-ri):

| system | conv | ours exact | (i) PySCF AFTDF (ia|jb) + our plasmon | ours RS-GDF | (ii) PySCF RSGDF + pyscf.gw.rpa loop, nw=200 | nw=40 vs our n=40 |
|---|---|---|---|---|---|---|
| H2/STO-3G a=4 | unshifted | -1.000052013959e-2 | +6.7e-16 | -9.999017017904e-3 | +9.6e-13 | +3.0e-13 |
| H2/STO-3G a=4 | shifted | -6.938130614440e-3 | +2.2e-16 | -6.937063950377e-3 | -1.1e-13 | +3.2e-14 |
| tri 4H s+p | unshifted | -9.631964613214e-2 | -1.5e-10 | -9.624188539521e-2 | -1.4e-10 | -1.5e-10 |
| tri 4H s+p | shifted | -6.846932539148e-2 | +5.7e-11 | -6.841208596393e-2 | +5.7e-11 | +5.7e-11 |

(The tri ~1e-10 is the same in every column, i.e. SCF orbitals ours-vs-PySCF; Iteration 3 MP2 saw -1.2e-10/+3.7e-11.)

Box limit vs molecular dRPA (exact ERI, plasmon, cart, same geometry); same ERI construction as Iteration 3
(w = min(1, 8/a), bra 18 Bohr, ket 18+6/w). E_dRPA(mol) = -2.065890717501e-2 (STO-3G), -2.832712047898e-2
(6-31G). Residual dE = E_dRPA(pbc) - E_dRPA(mol); quadrature route agreed with plasmon to <= 6e-16 at every a.

| a | STO-3G shifted | STO-3G unshifted | 6-31G shifted | 6-31G unshifted |
|---|---|---|---|---|
| 6 | +5.539e-3 | -2.366e-4 | +5.089e-3 | -4.894e-3 |
| 8 | +1.948e-3 | -3.394e-3 | +2.087e-3 | -5.471e-3 |
| 10 | +9.386e-4 | -3.394e-3 | +1.089e-3 | -4.794e-3 |
| 12 | +5.391e-4 | -3.026e-3 | +6.313e-4 | -4.189e-3 |
| 14 | +3.388e-4 | -2.674e-3 | +3.974e-4 | -3.670e-3 |
| 16 | +2.267e-4 | -2.377e-3 | +2.662e-4 | -3.245e-3 |
| 20 | +1.159e-4 | -1.927e-3 | +1.362e-4 | -2.613e-3 |
| 24 | +6.697e-5 | -1.612e-3 | +7.876e-5 | -2.178e-3 |
| 32 | +2.822e-5 | -1.208e-3 | +3.320e-5 | -1.626e-3 |
| 40 | +1.444e-5 | -9.632e-4 | +1.699e-5 | -1.294e-3 |

- Shifted local exponents: STO-3G 3.48 (8), 3.17, 3.03, 3.012, 3.009, 3.007, 3.005, 3.004 (32);
  6-31G 3.02 (8), 2.95, 2.996, 3.002 ... 3.003 (32). 3-pt tail (24,32,40): 3.0038 / 3.0025.
  Fit c3/a^3 + c5/a^5 (24,32,40): c3 = 0.923006 (STO-3G), 1.086677 (6-31G).
- Unshifted: local exponents -5.2 (8), 0.28, 0.71 ... 1.008 (32) (STO-3G), similar for 6-31G;
  fit c1/a + c2/a^2 + c3/a^3 (24,32,40): c1 = -0.037272 / -0.049804.
  (unshifted - shifted) / [E_mol(e_ia - v_M) - E_mol(e_ia)] = 0.947 (10), 0.970, 0.981, 0.988, 0.994, 0.996,
  0.9985, 0.9992 (40) STO-3G; 0.942 ... 0.9991 (40) 6-31G.
- The MP2 residuals recomputed in the same runs reproduce Iteration 3 to all printed digits.
- dRPA/MP2 shifted-residual ratio at a=40: 1.375 (STO-3G), 1.386 (6-31G).

Predictions written into run_rpa_box_limit.py BEFORE the sweep (molecular quantities only):

| quantity | predicted | measured (tail fit) |
|---|---|---|
| STO-3G shifted c3, nov=1 closed form (dE/dD k(sigma^2+\|d\|^2) + dE/dK (-k\|d\|^2); dE/dD 0.013161, dE/dK -0.204628) | 0.922741 | 0.923006 (2.9e-4 rel) |
| STO-3G unshifted c1 = -(v_M a) dE_mol/de_ia (uniform, molecular finite difference) | -0.037343 | -0.037272 |
| 6-31G unshifted c1 | -0.049903 | -0.049804 |
| artifact (b) Madelung sign flipped | shifted ~ +c1-magnitude/a, exponent 1 | exponent 3.00 — not seen |
| artifacts (a)/(c) missing images / normalisation | plateau at nonzero dE | both -> 0 — not seen |

General a^-3 predictor (`r2_kernel_c3`, run_rpa_c3_prediction.py). Written AFTER the 6-31G sweep had been
fitted (so not a blind prediction for 6-31G), but it has no adjustable parameter and uses no periodic code: after the Madelung term the cubic-lattice G=0-dropped kernel differs from
1/r at O(a^-3) by the harmonic kernel (k/2)|r-r'|^2, k = 4pi/3a^3 (it IS pbc_mp2's delta(1|2), since
|r-r'|^2 = r^2 + r'^2 - 2r.r'). Adding it to every ee/en/nn interaction of the MOLECULE, re-running RHF (orbitals
relax), MP2 and dRPA, and taking d/dk (central difference h=1e-3 and 1e-4 agree to 1.5e-5 abs):

| basis | HF c3 pred / meas | MP2 c3 pred / meas (Iter. 3) | dRPA c3 pred / meas |
|---|---|---|---|
| STO-3G | -10.028245 / -10.0282 (Iter. 1) | 0.671094 / 0.671358 | 0.922741 / 0.923006 |
| 6-31G | -10.940500 / not measured | 0.783666 / 0.784083 | 1.086250 / 1.086677 |

(STO-3G predictions equal the closed-form frozen-orbital models to 1e-6: orbitals are symmetry-fixed there.)

Aux fitting error dE = E(B) - E(dense AFT, plasmon), spherical aux, w=1 (run_rpa_fit_error.py):

| aux | H2/STO-3G a=4 shifted same C / own SCF (rel) | unshifted same C (rel) | MP2 same C (rel) | molecular DF-dRPA, same aux (rel) | tri 4H s+p shifted same C / own SCF (rel) | unshifted same C (rel) | MP2 same C (rel) |
|---|---|---|---|---|---|---|---|
| cc-pvdz-ri (28 / 56) | +1.07e-6 / +1.18e-6 (-1.5e-4) | +1.51e-6 (-1.5e-4) | +6.3e-7 (-1.6e-4) | +2.45e-6 (-1.2e-4) | +6.00e-5 / +6.49e-5 (-8.8e-4) | +8.16e-5 (-8.5e-4) | +3.33e-5 (-8.3e-4) |
| def2-universal-jkfit (36 / 72) | +4.63e-6 / +4.82e-6 (-6.7e-4) | +6.53e-6 (-6.5e-4) | +2.73e-6 (-7.0e-4) | +1.94e-5 (-9.4e-4) | +1.39e-4 / +1.44e-4 (-2.0e-3) | +1.80e-4 (-1.9e-3) | +7.61e-5 (-1.9e-3) |
| def2-svp-rifit (28 / 56) | +3.70e-6 / +3.74e-6 (-5.3e-4) | +5.22e-6 (-5.2e-4) | +2.19e-6 (-5.6e-4) | +8.83e-6 (-4.3e-4) | +8.72e-5 / +9.32e-5 (-1.3e-3) | +1.20e-4 (-1.2e-3) | +4.85e-5 (-1.2e-3) |
| ET l<=2 b=2.2 (146/162, 291/324) | +8.6e-10 / +2.3e-9 (-1.2e-7) | +1.2e-9 (-1.2e-7) | +5.1e-10 (-1.3e-7) | +5.2e-9 (-2.5e-7) | +4.50e-6 / +4.69e-6 (-6.6e-5) | +6.11e-6 (-6.3e-5) | +2.44e-6 (-6.1e-5) |

(Molecular column: spherical aux via cart ints + cart2sph, eig metric; with PySCF's default Cartesian aux the
molecular H2 errors are +2.44e-6 (cc-pvdz-ri) / +4.80e-6 (jkfit), run_rpa_formula.py.)

Frequency quadrature, ferric's DEFAULT grid (PdepRpaConfig: MiniMax, n_points 20 -> GL with u0 =
optimized_u0(20) = 0.5, the same nodes as gl_quadrature(20, 0.5)), error vs plasmon (run_rpa_quadrature.py):

| system | gap / max e_ia | n=10 | n=20 (ferric default) | n=40 | n=80 |
|---|---|---|---|---|---|
| H2/STO-3G a=4 shifted | 2.079 / 2.079 | -5.3e-7 | +1.2e-12 | ~0 | ~0 |
| H2/STO-3G a=4 unshifted | 1.370 / 1.370 | +9.7e-8 | +8.5e-14 | 0 | ~0 |
| tri 4H s+p shifted | 1.195 / 5.37 | -1.7e-6 | +8.7e-10 | 6e-15 | 6e-15 |
| tri 4H s+p unshifted | 0.573 / 4.74 | +4.8e-7 | -2.6e-10 | -1e-15 | -2e-15 |
| H2O/6-31G (molecular) | core e_ia ~ 21 | +1.9e-5 | -2.3e-6 | +7.4e-10 | -1e-14 |

### Interpretation (provisional, 2026-09-24)
- The dRPA formula, spin factor 4 and half-line 1/(2pi) are pinned three independent ways: plasmon/Riccati/
  quadrature agree to 1e-14; the O(Pi^2) term is exactly direct MP2 (2 sum (ia|jb)^2/Delta; 4e-15), so the
  normalisation does not rest on PySCF; and PySCF's RPA arithmetic agrees to 1e-13 on the same tensors.
- Periodic B -> dRPA is exact in the trivial-aux limit (1e-13, both conventions, own SCF too). Like MP2, this
  anchor is structurally blind to both G=0 terms of the fit (ov quantities only); the HF ERI anchor of Iteration 2
  remains their only guard.
- **Shifted denominators again converge as a^-3, and the coefficient is predicted a priori** (STO-3G 2.9e-4 rel,
  6-31G 3.9e-4 rel — the residual is the c5 truncation of a 3-point fit, same size as for MP2); unshifted converge
  as 1/a with the predicted c1 = -(v_M a) dE/de_ia. At a=40 unshifted is still -9.6e-4 / -1.3e-3 Ha (4.7% / 4.6%
  of E_c), shifted 1.4e-5 / 1.7e-5. Same small-box trap as MP2: at a=6 (STO-3G) unshifted is 23x closer to the
  molecule than shifted. dRPA's c3 is ~1.38x MP2's in both bases.
- The harmonic-kernel predictor generalises the moment model to any molecule/basis/method with orbital relaxation,
  reproduced HF (Iter. 1), MP2 (Iter. 3) and dRPA box tails to <= 5e-4 relative. It is the cheap independent
  finite-size oracle a Rust port can use for a box test on real molecules (one molecular calc at +-k).
- Aux: dRPA relative fitting error tracks MP2's on the same B to within ~10% in every row (cc-pvdz-ri -1.5e-4 H2,
  -8.8e-4 tri); cc-pvdz-ri is the best stock set, jkfit 2-4x worse, as for MP2. The periodic metric adds no new
  error class (H2 periodic cc-pvdz-ri -1.5e-4 vs molecular -1.2e-4). The own-SCF column adds 4-10% on top (x2.6 for ET on H2, where the same-C error is only 9e-10).
- Quadrature: ferric's default 20-point grid is <= 1e-9 on these small-e_ia cells but 2.3e-6 on H2O/6-31G, where
  core excitations (e_ia ~ 21 Ha) are far above u0=0.5. 1e-8 needs n >= 40 once any core or high virtual is
  present — this is a property of the default grid, not of periodicity. The unshifted convention has smaller
  gaps (0.57 vs 1.20 on tri) but converged no worse here.
- NOT measured: anything > 16 AOs, spherical orbital basis, frozen core, real solids (a^-3 is the isolated-molecule
  law, not N_k^-1), open shell, PDEP truncation (trunc_thresh > 0) on a periodic B, cost.

### For the Rust port (stage 7) via run_pdep_rpa_from_intermediates
- Build `RpaIntermediates` from the SAME periodic RS-GDF B^P_mn used for SCF J/K:
  `b_ov` = B^P_ia reshaped (naux_kept, nocc*nvir) — B is already metric-dressed (s^-1/2 U^T J3', eig + lindep
  drop), so NO further V^{-1/2}; `naux` = naux_KEPT (it sizes the eigensolve); `v_inv_sqrt` = U_kept s_kept^-1/2,
  shape (naux_ao, naux_kept) — non-square, used only for `eigenpotentials_aux = v_inv_sqrt . eigenvectors`
  (lib.rs:845), fine for an energy; `nocc`, `nvir`, `nocc_total`, `first_occ` as molecular (frozen core via
  `active_occ`).
- Denominators come from `rhf.eps_r()` (lib.rs:909-910): the ScfResult must carry the exxdiv='ewald' eigenvalues
  (shifted). If the Gamma SCF ran with exxdiv=None the driver must shift eps_occ by -v_M itself or refuse. Pin it
  with the a^-3 box test (predicted c3 from r2_kernel_c3).
- `mol/obs/dfbs` are only read by (1) `build_atom_seed` (when trunc_thresh > 0 and naux > 4 natoms; it sizes the
  seed with dfbs.nbasis(), which is WRONG when the periodic lindep filter dropped aux functions) and (2) the
  Boys-screened chi0 path (molecular 3-index build; default is Dense). Periodic callers must use
  `chi0_sparsity: Dense` and `trunc_thresh: 0` (full rank — also the production rule), or better, split
  `run_pdep_rpa_from_intermediates` so the energy path takes (inter, eps_occ, eps_vir, config) only.
- Quadrature: set n_points >= 40 (GL u0 0.5) for a 1e-8 target; ferric's log-det summand `ln det(eps) + tr(I-eps)`
  has the same large-w cancellation measured here (1.9e-10 at n=1024, negligible at n <= 80).
- Tests to port: trivial-aux anchor vs dense plasmon (1e-11), tr Pi^2 term == direct MP2, shifted box residual vs
  r2_kernel_c3 prediction at a=24/32 (<1%), unshifted - shifted == E_mol(e_ia - v_M) - E_mol(e_ia).

## Iteration 6 (Python, Gamma UHF) — 2026-09-24

### Code
- New `pbc_uhf.py`: `uhf(S, h, I, enn, na, nb, kshift=v_M, jk=, guess=, mix=, level_shift=)` (per-spin DIIS on
  the concatenated alpha/beta commutators), `spin_square` (lattice S), `core_guess`, `dense_jk`,
  `uhf_c3_closed_form` and `uhf_r2_kernel_c3` (box-limit predictors). Seams `_jk_spin` / `_madelung_term`
  are module functions so tests mutate them.
- `pbc_gamma.Cell`: Mole `spin` = electron-count parity (odd-electron cells failed PySCF's check; nothing reads it).
- Drivers: `run_uhf_anchor.py`, `run_uhf_oracle.py`, `run_uhf_box_limit.py {H|O2} a...`, `run_uhf_guess.py`.
- `test_prototype.py`: +10 fast UHF tests, +1 slow (whole suite 56 passed / 4 skipped, 153 s).

### Convention (read from PySCF 2.13 source, then measured)
`pbc/scf/uhf.py get_veff`: vhf = vj[0]+vj[1]-vk with (D_a, D_b); `df_jk._ewald_exxdiv_for_G0` adds
v_M S dm S to vk[i] for EACH dm. So F_s = h + J[D_a+D_b] - K[D_s] - v_M S D_s S: the SAME v_M as RHF, on the
single-spin density, coefficient 1. RHF's "K += v_M S D S, F = h + J - K/2" is identical because D = 2 D_s;
the Madelung wrapper is LINEAR in D, so one injected KBuilder serves both. Energy shift -v_M (N_a+N_b)/2.

### Measured
Anchors (run_uhf_anchor.py):

| check | none | ewald |
|---|---|---|
| (a) closed shell via UHF - RHF, H2/STO-3G a=4 pure-AFT | 0 (d eps 2e-16) | 0 (1e-15) |
| (a) same, tri 4H s+p (nao 16) | +2.7e-15 | +1.3e-15 |
| (b) open shell dense AFT vs trivial-aux RS-GDF, tri one-s TRIPLET (<S2> 2.00018) | -1.1e-11 | -1.1e-11 |
| (b) same, singlet | -5.3e-12 | -5.3e-12 |
| MUTANT K[D_total]: (a) H2 / tri; (b) via B, triplet / singlet | -4.5e-2 / -3.9e-1; -2.4e-1 / -3.9e-1 | same |
| MUTANT Madelung v_M/2 per spin: (a) H2 / tri | – | +3.5e-1 / +6.2e-1 (= v_M N/4) |
| (b) aux 7/8 classes, triplet / singlet | +6.7e-7 / +1.8e-7 | |
Blind spot: (b) applies the Madelung term identically on both sides, so it cannot see a wrong factor.

PySCF oracle (run_uhf_oracle.py; pbc.scf.UHF, AFTDF mesh 61^3, conv 1e-12; ours = pure-AFT dense ERI):

| system | exxdiv | ours E | <S2> | PySCF default guess dE / d<S2> | PySCF from our D |
|---|---|---|---|---|---|
| H atom a=4 doublet | none | -0.402177788224 | 0.75 | +7.4e-14 / 0 | same |
| | ewald | -0.756839973159 | 0.75 | +8.2e-14 / 0 | same |
| H2/STO-3G a=4 triplet (na 2, nb 0) | none | +0.322229103842 | 2.0 | +1.4e-14 / 0 | same |
| | ewald | -0.387095266028 | 2.0 | +5.7e-15 / 0 | same |
| tri 4H s+p triplet (na 3, nb 1) | none | -0.583125965387 | 2.001814801 | +7.1e-13 / -6.5e-10 | +1.3e-12 |
| | ewald, our core guess | -1.812958714837 | 2.002889744 | **-1.5e-2** / -1.1e-3 (different state) | +6.6e-13 / -3.1e-10 |
| | ewald, our none-first | -1.827999723359 | 2.001814801 | 0 (same state) | |
Pinned systems have nb = 0, so K[D_total] == K[D_a] there: the oracle cannot see that mutant (the anchors and the
O2 box test do).

Box limit vs PySCF MOLECULAR UHF (cart STO-3G, same geometry; Ewald split w = min(1, 8/a), bra 18, ket 18+6/w;
molecular UHF density as guess). Predictions written into run_uhf_box_limit.py BEFORE the sweep:
- physics, ewald: dE = c3/a^3 + O(a^-5), c3 = -(2pi/3)(|d|^2 + Omega_a + Omega_b), Omega_s = sum_i <i|r^2|i> -
  sum_ij |<i|r|j>|^2 over s-occupied (Foster-Boys invariant spread). Derivation: after Madelung the cubic G=0-dropped
  kernel is 1/r + (k/2)|r-r'|^2, k = 4pi/3a^3; Hartree+en+nn of a neutral cell give -(k/2)|d|^2, exchange of spin
  s gives -(k/2) Omega_s; first order in k, so orbital relaxation does not enter. RHF (Omega_a = Omega_b = sigma^2,
  one orbital) recovers Iteration 1's -(4pi/3) sigma^2. Per spin the exchange coefficient is HALF the RHF
  per-orbital one: the H atom has c3 = -(2pi/3) sigma^2, not -(4pi/3) sigma^2.
- physics, none: dE_none - dE_ewald = +v_M (N_a+N_b)/2 exactly -> 1/a, coefficient +2.8373 N/2.
- artifacts: per-spin Madelung factor 1/2 -> exponent 1 (+0.709 N/a); missing images -> plateau; K[D_total] -> O(1).

Predicted c3: H -4.081081 (Omega_a 1.948573); O2 triplet -30.622429 (Omega_a 6.823932, Omega_b 7.797201, |d| 1e-14).
Independent construction `uhf_r2_kernel_c3` (harmonic kernel added to every interaction of the molecular UHF,
relaxed, central difference): -4.081081 / -30.622429 (agree to 1e-6).
E_mol(UHF) = -0.466581849557 (H), -147.633950632184 (O2, <S2> 2.0034108656).

| a | H dE_ewald | H dE*a^3 | O2 dE_ewald | O2 dE*a^3 | O2 <S2> | O2 wall |
|---|---|---|---|---|---|---|
| 8 | -1.1804780731e-2 | -6.044048 | -6.8568398238e-2 | -35.107020 | 2.00316835 | 302 s |
| 10 | -4.3444294260e-3 | -4.344429 | -3.1192324469e-2 | -31.192324 | 2.00332353 | 34 s |
| 12 | -2.3715247834e-3 | -4.097995 | -1.7880107655e-2 | -30.896826 | 2.00336085 | 13 s |
| 14 | -1.4874498980e-3 | -4.081563 | -1.1232034317e-2 | -30.820702 | 2.00337918 | 4 s |
| 16 | -9.9635921039e-4 | -4.081087 | -7.5129705739e-3 | -30.773127 | 2.00338954 | 5 s |
| 20 | -5.1013514216e-4 | -4.081081 | -3.8397648415e-3 | -30.718119 | 2.00339988 | 3 s |
| 24 | -2.9521709615e-4 | -4.081081 | -2.2199515489e-3 | -30.688610 | 2.00340449 | 3 s |
| 32 | -1.2454471244e-4 | -4.081081 | -9.3565382184e-4 | -30.659504 | 2.00340817 | 4 s |
| 40 | -6.3766892768e-5 | -4.081081 | -4.7884545873e-4 | -30.646109 | 2.00340948 | 6 s |

- none - ewald - v_M N/2: |.| <= 1e-16 (H), <= 1e-13 (O2) at every a. With exxdiv=None, dE is still +0.567 Ha for O2
  at a=40 (= 22.698/a - 4.8e-4).
- O2 local exponents 3.53 (8), 3.05, 3.016, 3.012, 3.008, 3.005, 3.003, 3.002 (32->40). Fit c3/a^3 + c5/a^5
  (24,32,40): c3 = -30.622127 (-9.9e-6 relative to the prediction), c5 = -38.3; adding c7: c3 = -30.622416 (4e-7).
- H: dE*a^3 is flat at -4.081081 from a=20 on, with no a^-5 term. This looked too clean, so I checked it. It is
  expected: for one electron, Hartree == self-exchange in ANY kernel. Beyond r^2, the cubic kernel expansion has
  only l >= 4 cubic harmonics, which vanish on a spherical density. So the residual is exactly c3/a^3 plus
  image-overlap tails, which decay exponentially (a=10..16 above). O2 (non-spherical, 16 e) shows the ordinary
  c5 tail.
- Mutations of the source (not only monkeypatch): Madelung x0.5 fails 5 tests (incl. both box tests); K[D_total]
  fails 5; replacing Omega_b by Omega_a in the predictor fails both box tests.

SCF convergence / guesses (run_uhf_guess.py):
- **Ewald trap (new).** At Gamma, v_M S D_s S = v_M x (occupied projector), so ewald and None have IDENTICAL
  stationary densities, with energies offset by the constant -v_M N/2. But ewald lowers every occupied level by
  v_M, so a state with a hole below the Fermi level under None can become aufbau-self-consistent under ewald.
  tri 4H s+p triplet, ewald from the core guess (also with beta mix 0.3): E -1.812958714837, alpha gap
  0.617 < v_M 0.622. Evaluated under None, the same D is stationary (|[F,D]| 3e-8), with E exactly +v_M N/2
  (1e-15) and alpha eps (occ) 0.922 > (vir) 0.916. None first, then ewald from that D: -1.827999723359 in 1
  iteration (= PySCF default guess, 1.5e-2 lower). Necessary condition for the HF minimum under ewald
  (single-swap second variation, positive kernel): per-spin gap >= v_M. It is necessary, not sufficient.
- **O2 from the core (hcore) guess lands in the wrong state** in both conventions: 0.256 Ha high, <S2> 2.0121, gaps
  > v_M, so the gap criterion does not flag it. A 0.5 level shift gives the same state. This is NOT periodic: the
  molecular hcore guess finds the same state (ours -147.378615604497 == PySCF init_guess='1e'). The molecular UHF
  density (cell-0 AOs == Gamma AOs) as guess reaches the right state in 5 iterations.
- Closed shells: with na == nb and no mixing, UHF stays RHF (tri 4H, stretched H2). With a beta HOMO/LUMO mix of
  0.3 or 0.7, stretched H2 (R=4, a=10) breaks symmetry to <S2> 0.9449, 0.1438 Ha below RHF, in both conventions
  (the difference is again exactly v_M). tri 4H s+p is UHF-stable (returns to RHF from a 0.7 mix).

### Interpretation (provisional, 2026-09-24)
- Gamma UHF is the molecular UHF on (S, h, I or B, E_nn), exact to 1e-11 against both anchors and 1e-12 against
  PySCF AFTDF (energies and <S2>, using the lattice S).
- Per-spin Madelung: same v_M, on D_s, coefficient 1. It needs no special handling if the Madelung term lives
  in a linear KBuilder wrapper and UHF calls it with D_s (RHF calls it with D_total and uses -K/2).
- exxdiv=ewald converges to the molecule as a^-3, with c3 predicted a priori from molecular spreads (H exact,
  O2 1e-5 relative). exxdiv=None converges as 1/a with the exactly predicted +v_M N/2. The UHF finite-size
  term is a sum over spins of the exchange spread (per spin half the RHF per-orbital factor).
- The Gamma Madelung term makes the SCF landscape stickier. An ewald SCF should either start from a converged
  None density or check "per-spin gap >= v_M" at convergence. Guess quality (hcore vs molecular/SAD)
  matters exactly as it does molecularly.
- NOT measured: RS-GDF with a real aux for UHF (only the trivial-aux anchor; RHF Iteration 2 errors should carry
  over per spin), ROHF, spherical basis, nao > 16 in the pure-AFT oracle, k-points, cost.

### For the Rust port (stage 4): `solve_uhf_injected`
- Mirror `PeriodicInjection` + `validate_injected`. uhf.rs:438 gets S/h/V_nn from `driver::prepare`; take them
  from the injection as solve_rhf_impl does. J = injected JBuilder on D_a + D_b; K_s = the SAME injected
  KBuilder called per spin (`update_density(D_s)` then `build(D_s)`, as uhf.rs:858 already does). The Madelung
  wrapper `K += v_M S D S` is linear, so no per-spin factor code is needed; do NOT add a 1/2 "for spin".
- If a periodic RS-GDF KBuilder overrides `build_from_occ` (DfK-style half-transform), the override must also add
  v_M S C C^T S. The default impl reconstructs D and is safe; an override that forgets it drops the Madelung term
  silently on the occupied path only.
- Guess: `uhf_guess_mos` builds its guess Fock with molecular `rhf::build_jk(bounds)`, and MINAO projects onto
  molecular atoms. Both must be rejected or rerouted on the injected path: reject `use_sad_guess`, and build the
  guess Fock from the INJECTED J/K when `init_guess_density` is given, or accept per-spin MOs via the
  existing `initial_mos` argument. Also reject `scf_stability_descent` / `check_stability`: they rebuild
  molecular J/K. Recommended default for exxdiv=ewald: converge with Madelung off, then switch it on, or assert
  per-spin gap >= v_M at convergence and warn.
- Tests to port: closed-shell UHF == injected RHF (1e-11, both exxdiv); open-shell trivial-aux anchor (1e-10);
  ewald - none == -v_M N/2 (1e-11); H-atom box residual == -(2pi/3) Omega_a / a^3 at a=20 (1e-5); O2 a=24/32
  vs the closed-form c3 (0.5%), exponent 3 +- 0.02; tri 4H s+p ewald trap (slow).

## Iteration 5 (Python, Gamma LMP2) — 2026-09-24

### Code
- New `pbc_supercell.py`: `Supercell(a_prim, atoms, basis, (n1,n2,n3), wrap=)` + `build_supercell(...)` -> S, T, V, h,
  E_nn, v_M, RS-GDF B/J2/J3, Resta matrices Z_k, minimal-basis cross overlap, aux centres. Supercell quantities are
  built by FOLDING primitive-cell lattice sums by residue mod n (SR real-space integrals computed once on the
  primitive cell; LR through per-residue pair FTs Q^r(G) on the supercell G lattice, P^sc_{(c,m),(c',n)}(G) =
  e^{-iG.t_c} Q^{res(c'-c)}_mn(G)). Reason: `build_gdf` on an explicit supercell is an unscreened
  nb_sc x nb_sc|L| x naux|T| block (8.8M shell triplets already at 2 cells); the fold's SR cost is N-independent.
  1x1x48 (nao 192, naux 1344) builds in 49 s.
- New `pbc_lmp2.py`: Gamma RHF on the supercell B (C/eps re-diagonalised from the ewald Fock of the final D, so
  canonical and local paths share one F), Berghold/Resta Jacobi localisation, periodic centroids/spreads,
  minimum-image distances, periodic VV-HV virtuals, (ia|jb) from B, optional minimum-image occupied pair cutoff,
  per-pair domain fit in the PERIODIC metric (J2, J3), translation-equivalence analysis, uniform-field transition
  dipoles. Solver/energy/pair energies/pivoted Cholesky/Loewdin are IMPORTED unchanged from
  scripts/amplitude_lmp2_proto.py (dense masked CG, and the ragged per-pair CG for large N).
- Drivers: `run_lmp2_anchor.py` (anchor + mutations), `run_lmp2_translation.py`, `run_lmp2_sweep.py` (hypotheses
  written in its docstring before the sweep), `run_lmp2_farfield.py` (analytic uniform-field pair energies).
- `test_prototype.py`: +9 fast tests (the LMP2 subset runs in ~35 s).

System for everything below: primitive cubic a0 = 7 Bohr, one tilted H2 (R = 1.4 Bohr) per cell, 6-31G (cart;
s only on H), cc-pvdz-ri spherical aux, RS-GDF w = 0.5, exxdiv='ewald' / shifted denominators, eps on the Eq-8
swap-closed integral mask. Supercells 1x1xN ("needles", N = 2..48), 2x2x2, 3x3x3.

### Why the Berghold (Resta) functional
At Gamma the orbitals are supercell-periodic; <mu|r|nu> lattice-summed is undefined and the L=0 molecular
integrals see the supercell boundary. The Resta operator e^{i b_k.r} (b_k = supercell reciprocal vectors) is
periodic, its AO matrix is the lattice-summed pair FT at -b_k (the primitive the integrals already use), and
maximising sum_k w_k sum_i |z_k,ii|^2 (w_k = (|a_k|/2pi)^2, orthorhombic only; general cells raise
NotImplementedError) is Boys' Jacobi sweep with Re/Im z_k as six weighted coordinates. It yields periodic
centroids arg(z_ii)/2pi (needed for minimum-image pair/fit domains) and spreads (needed for the HV weights).
Pipek-Mezey on the lattice-summed S would also be periodic-safe; not implemented here.

### Measured: validation of the new construction
| check | result |
|---|---|
| fold vs `build_gdf` on explicit supercell, (1,1,1) / (1,1,2): max dJ2, dJ3 | 1.3e-15, 3.6e-14 / 1.8e-14, 1.2e-13 |
| fold S, T, E_nn vs PySCF pbc_intor / energy_nuc, (1,1,2) | 1.4e-15, 7.3e-15, 4.1e-15 |
| folded (1,1,2) HF / MP2 (cart aux) vs PySCF pbc RHF(exxdiv='ewald')+RSGDF, RMP2 | dE_HF -3.3e-13, dMP2 +2.4e-11, dh 5.3e-14 |
| fold Z_k vs direct `pair_ft` on an explicit (1,1,3) supercell with its last atom WRAPPED across the boundary | 1.4e-13 |

### Measured: exactness anchor (eps = 0, full mask) and mutations
| supercell | nocc/nvir (VV/HV) | E_MP2(canonical, shifted) | anchor dE | M1 drop 1 HV | M2 pair cutoff at trivial R*: min-image / raw | M3 periodic domain fit at trivial R*: min-image max\|dJ\| / raw | M4 molecular Boys, eps=0 |
|---|---|---|---|---|---|---|---|
| 1x1x4 | 4/12 (4/8) | -7.437860250848e-2 | +1.8e-16 | +3.4e-3 | +1.8e-16 / +1.6e-4 (14/16 pairs) | 3.3e-16 / 5.8e-5 (dE +1.6e-6) | +1.8e-16 |
| 2x2x2 | 8/24 (8/16) | -1.377289905714e-1 | -9.2e-16 | +3.1e-3 | -9.2e-16 / **-9.2e-16 (64/64)** | 5.4e-16 / 2.5e-6 | -6.1e-16 |
| 1x1x8 | 8/24 (8/16) | -1.528153223461e-1 | +5.6e-17 | +3.4e-3 | +5.6e-17 / +2.2e-4 (52/64) | 1.3e-15 / 9.1e-5 | +1.9e-16 |
Anchor also +1e-14..+1.1e-13 on 1x1x16/24/32 (dense), and the ragged vs dense solver agree to 1.1e-16 on identical
masks (1x1x16, 1x1x4). VV-HV: orthonormality <= 4e-15, span of canonical virtuals <= 9e-15 everywhere.
Blind spots (pinned in tests): M2 cannot fail when n = 2 along every axis (each raw distance IS a minimum image);
the fold's residue SIGN is invisible at n = 2 (c'-c == c-c' mod 2) — the (1,1,2) fold test PASSED a flipped sign,
so that test now uses n = 3; M4 (non-periodic localisation) is invisible to the eps = 0 anchor (unitary invariance).
Test mutations run (each restored): min_image->raw, Jacobi angle sign, fold residue sign, Z not conjugated: each
failed >= 1 test. Flipping the transition-dipole sign fails nothing — correctly: mu enters as mu_i (x) mu_j.

### Measured: translation equivalence (item 4)
Deviation = 1 - max_j |<phi_j|S|T phi_i>| over every LMO/virtual, T = one primitive translation (an exact AO
permutation at Gamma); pair_dev = max |e_ij - e_T(i)T(j)|. "wrapped" = last atom moved by -a_sc(z), so one
molecule straddles the supercell boundary.

| supercell | localisation | occ_dev | vir_dev | pair_dev (eps 0 / 1e-4) | spread_dev |
|---|---|---|---|---|---|
| 1x1x6, 2x2x2, 3x3x3, 1x1x16 (straight + wrapped) | Berghold, 3 starts (canonical + 2 random) | <= 2.3e-15 | <= 4.0e-15 | <= 1.5e-15 | <= 3.5e-12 |
| 1x1x4 wrapped / straight | molecular Boys (MUTANT) | 3.2e-4 / 4.5e-5 | 4.8e-4 / 1.1e-4 | 1.2e-5 (eps 1e-4) / 4.4e-6 | 1.2e-2 / 2.4e-3 |
| 1x1x6, 2x2x2, 3x3x3, 1x1x16 | molecular Boys (MUTANT) | 1.5e-5 .. 1.4e-4 | 7e-5 .. 4.7e-4 | 2.4e-6 .. 1.1e-5 | 7e-4 .. 4.6e-3 |
Berghold functional values agree across starts to 1e-8 (|grad| <= 4e-11); E(eps=1e-4) spread over starts
<= 1.2e-15. Molecular Boys shifts E(eps=1e-4) by -1.8e-5 (1x1x6) to +3.2e-6 (1x1x16). Partner counts were equal
for every molecule in every row of every sweep below (partners min == max).

### Hypotheses stated before the sweep (run_lmp2_sweep.py docstring, verbatim in substance)
- P1 (claim under test): past an onset, kept pairs ~ linear in N, error at fixed eps size-intensive per molecule.
- P2 (derived for Gamma): every pair carries a distance-INDEPENDENT coupling. In the needle the G_par = 0 Fourier
  components of j's image lattice are periodic dipole sheets with a uniform field, so far pairs have
  (ia|jb) -> -(4pi/Omega_sc) mu_ia,z mu_jb,z (G_par != 0 terms decay as exp(-2pi d/7)). Predictions: (a) the eps mask
  keeps ALL pairs for N < N* = 4pi mu*^2/(Omega_prim eps); (b) far-pair energies ~N^-2, ~N^2 of them: O(1) total,
  so the error past onset is a N + b (+ c/N) with b = O(1); (c) a distance cutoff R_c gives partners
  2 floor(R_c/a0) + 1 once N a0 > 2 R_c. The molecular R^-3 picture predicts an eps onset at N ~ 2 instead.
- Artifacts: A1 non-periodic localisation/distances -> unequal partner counts/pair energies (checked every row);
  A2 translation symmetry makes E_err = N x (per molecule) at EVERY N, so extensivity is automatic, not evidence;
  A3 below the onset all pairs are kept: no locality statement either way.

### Measured: the uniform-field coupling (P2)
mu*_z = max|mu_ia,z| = 0.735 (N=2) .. 0.7964 (N >= 32, converging as the Resta estimate's O((b sigma)^2) error falls).
| N | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 40 | 48 |
|---|---|---|---|---|---|---|---|---|---|
| farthest pair: rel \|J - J_unif\| | 4.1e-2 | 1.8e-2 | 1.0e-2 | 4.6e-3 | 2.6e-3 | 1.2e-3 | 6.5e-4 | 5.0e-4 | 4.6e-4 |
| farthest pair: N max\|J\| | 0.0233 | 0.0233 | 0.0233 | 0.0233 | 0.0232 | 0.0232 | 0.0232 | 0.0232 | 0.0232 |
| sum e_ij over pairs d > 13.9 (eps=0 solve) | -2.73e-4 | -5.45e-4 | -6.81e-4 | -8.17e-4 | -8.85e-4 | -9.53e-4 | -9.87e-4 | – | – |
| same, predicted (uniform J, Sylvester with full Fvv) | -2.4e-4 | -5.1e-4 | -6.57e-4 | -8.03e-4 | -8.76e-4 | -9.47e-4 | -9.83e-4 | -1.004e-3 | -1.018e-3 |
(A first predictor with diagonal-Fvv denominators was 1.57-1.74x too small: the VV-HV basis is non-canonical. The
Sylvester version is the one quoted; its N=4/6 entries are the Rc 10.5/17.5 far-field column of run_lmp2_farfield.)

eps-mask onset, partners per molecule (all N molecules identical):
| eps | N* predicted | N=2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 40 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1e-2 | 2.3 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 3e-3 | 7.7 | 2 | 4 | 6 | 3 | 1 | 1 | 1 | 1 | 1 | 1 |
| 1e-3 | 23.2 | 2 | 4 | 6 | 8 | 12 | 16 | 3 | 3 | 3 | 1 |
| 3e-4 | 77 | all | all | all | all | all | all | all | all | – | – |
| 1e-4 | 232 | all | all | all | all | all | all | all | all | – | – |
Distance cutoff R_c = 3.5 / 10.5 / 17.5: partners 1 / 3 / 5 exactly once N a0 > 2 R_c (N >= 2 / 4 / 6).

### Measured: truncation error dE = E_LMP2 - E_canonical (Hartree; >0 = correlation lost)
| N | eps 1e-2 | eps 3e-3 | eps 1e-3 | eps 3e-4 | eps 1e-4 | R_c 3.5 | R_c 10.5 | R_c 17.5 |
|---|---|---|---|---|---|---|---|---|
| 2 | +4.05e-4 | +1.50e-4 | +2.07e-5 | +1.25e-7 | +1.25e-7 | +6.04e-4 | 0 | 0 |
| 4 | +1.05e-3 | +6.88e-4 | +1.87e-4 | +3.18e-5 | +2.00e-5 | +8.98e-4 | +2.73e-4 | 0 |
| 8 | +1.34e-3 | +1.20e-3 | +2.48e-4 | +8.37e-5 | +1.62e-5 | +1.05e-3 | +6.81e-4 | +4.09e-4 |
| 16 | +1.74e-3 | +1.74e-3 | +8.16e-4 | +2.67e-4 | +2.71e-5 | +1.15e-3 | +8.85e-4 | +7.49e-4 |
| 24 | +2.09e-3 | +2.09e-3 | +1.30e-3 | +3.19e-4 | +4.64e-5 | +1.20e-3 | +9.53e-4 | +8.62e-4 |
| 32 | +2.43e-3 | +2.43e-3 | +1.40e-3 | +7.98e-4 | +4.98e-5 | +1.23e-3 | +9.87e-4 | +9.19e-4 |
| 40 | +2.77e-3 | +2.77e-3 | +1.50e-3 | – | – | +1.27e-3 | +1.008e-3 | +9.53e-4 |
| 48 | +3.10e-3 | +3.10e-3 | +1.65e-3 | – | – | +1.30e-3 | +1.021e-3 | +9.76e-4 |
(also N = 6, 12 in the logs; N = 40, 48 by the ragged solver, eps >= 1e-3 only.) Kept element fraction at N=48:
2e-5 (eps 1e-2, 1e-3) .. 0.10 (R_c 17.5); at N=32 eps 1e-4 keeps 0.22% of elements but 100% of pairs.
Canonical E_MP2/N: -1.75430e-2 (2) .. -1.95171e-2 (48); fit E = e_inf N + b on N = 16..48: e_inf = -1.95997e-2,
b = +3.964e-3, max residual 1.9e-6 over 5 points (Gamma finite size is 1/N per molecule, 20% of E_c at N = 2).

Tail fits (last three N, 32/40/48), and the residual after subtracting the analytic uniform-field energy of the
entirely dropped pairs (run_lmp2_farfield.py; resid = dE - farfield):
| setting | a (per molecule) from dE = aN + b + c/N | b | resid at N = 16 / 32 / 48 | a from resid = aN + b |
|---|---|---|---|---|
| R_c 17.5 | +1.3e-9 | +1.089e-3 | 7.8e-6 / 3.8e-6 / 2.9e-6 | -5.8e-8 |
| R_c 10.5 | -1.3e-8 | +1.090e-3 | 9.5e-6 / 4.3e-6 / 3.3e-6 | -6.4e-8 |
| R_c 3.5 | +3.31e-6 | +1.163e-3 | 1.35e-4 / 1.83e-4 / 2.35e-4 | +3.28e-6 |
| eps 1e-2 = 3e-3 (self pairs only, N >= 12) | +4.11e-5 | +1.152e-3 | 7.3e-4 / 1.38e-3 / 2.04e-3 | +4.10e-5 |
| eps 1e-3 | not a clean tail: partners change 3 -> 1 between N = 40 and 48 | | 3.48e-4 (24) / 4.17e-4 / 5.91e-4 | – |

### Interpretation (provisional, 2026-09-24; one toy crystal, s-only basis, needle supercells)
- **Construction:** the periodic ingredients are right and the anchor is exact: eps=0 == canonical Gamma MP2 to
  1e-16..1e-13 up to 1x1x32, mutations of the three new periodic pieces (HV span, minimum-image pair cutoff,
  periodic-metric domain fit) each break it by 1e-6..3e-3; Berghold LMOs are translation-equivalent to 1e-15
  and start-independent, while a non-periodic (molecular) Boys operator breaks equivalence at 1e-5..5e-4 even
  without a wrapped molecule. The eps=0 anchor is blind to localisation; the translation check is its guard.
- **P1 holds only after an onset set by VOLUME for the integral threshold, not by distance.** At Gamma every pair
  carries the uniform-field coupling -(4pi/Omega_sc) mu mu (measured: the farthest-pair max|J| x N is 0.0233 at every
  N, and agrees with the formula to 4.6e-4 relative at N=48). The Eq-8 mask therefore keeps all N^2 pairs until
  N > N* = 4pi mu*^2/(Omega_prim eps): observed onsets bracket the prediction for eps = 1e-2 (2.3), 3e-3 (7.7) and
  1e-3 (23.2); the molecular R^-3 picture (onset at N ~ 2) is refuted. For eps <= 3e-4 (N* >= 77) every measured
  size is below the onset: kept pairs grow as N^2 there. By A3 this is NOT a negative for locality; the onset is
  simply beyond 48 cells of 343 Bohr^3 (Omega_sc > 2.6e4 Bohr^3 for eps = 3e-4).
  Past the onset, partners per molecule do saturate (kept pairs = N k), but for eps the plateau is not unique:
  the near pairs are kept by near-field + uniform, so k steps down again as the uniform part shrinks (eps 1e-3:
  3 at N = 24..40, 1 at 48).
- **Error decomposition.** dE(N) = a N + b with b ~ +1.09e-3 Ha, O(1) and the same for every cutoff: it is the
  uniform-field energy of the dropped pairs (predicted analytically to within 3e-6 of dE at N = 48 for R_c >= 10.5).
  The intensive part a is: indistinguishable from 0 (|a| < 1e-7) for R_c >= 10.5; 3.3e-6 Ha/molecule for R_c = 3.5
  (dropping the +-1 neighbours at 7 Bohr); 4.1e-5 Ha/molecule for eps >= 3e-3 (element truncation inside self pairs).
  So at every reachable N the per-molecule error is dominated by b/N (e.g. R_c 10.5: 2.1e-5/molecule at N=48,
  of which only ~7e-8 is a locality loss). E_err/N IS size-intensive in the limit, approached as 1/N.
- The uniform-field part of both the canonical energy's 1/N finite-size term and the LMP2 truncation error is the
  same G->0 physics that gave the a^-3 box law (Iterations 1/3/4). In the thermodynamic limit it contributes 0 per
  molecule, so the truncation "error" b is a finite-size term removed, not correlation lost; whether LMP2 then
  converges to e_inf FASTER than canonical is NOT established here: E_LMP2/N drifts 1.42e-4 vs canonical 1.24e-4
  between N = 16 and 32 (R_c 17.5).
- Caveats: needle geometry maximises the effect (lateral images screen the R^-3 near field to exp(-0.9 d), leaving
  the uniform term); in cubic supercells the uniform term is (4pi/3Omega) mu.mu and the farthest-pair dipole term
  scales the same way, so the eps onset there is also ~ a volume, with a different constant (3x3x3 at eps 1e-4:
  25/27 partners; no cubic size sweep was run). 6-31G on H has no polarisation functions, so dispersion
  (hence a) is tiny; a real molecular crystal will have larger a and larger mu. NOT measured: cost/timing claims,
  k-points, open shell, the integral-free R^-6 pair gate, domain-fit accuracy at finite radius, frozen core.

### For the Rust port onto ferric's amplitude_lmp2
1. Localisation: replace `boys_localize` (dipole integrals) by a Berghold/Resta Jacobi on Re/Im z_k, z_k = lattice-summed
   pair FT at -b_k (stage-0 `pair_ft` already provides it); weights (|a_k|/2pi)^2 (orthorhombic; Silvestrelli weights for
   general cells, or refuse). Centroids = arg(z_ii)/2pi (fractional). Spreads for the HV weights from 1 - |z_ii|^2.
   Port test: translation equivalence (AO permutation) to 1e-12, plus a molecule wrapped across the boundary.
2. VV-HV: lattice-summed S and minimal-basis cross overlap; everything else is the molecular code on those matrices.
3. Every distance (pair gate R^-6 estimator, fit domains, any pair cutoff) must be minimum-image; test with n >= 3 along
   an axis (n = 2 cannot see the bug). The Eq-8 integral mask itself needs no distance (Gamma integrals already sum images).
4. Integrals and fits: (ia|jb) from the periodic RS-GDF B of stage 6; per-pair domain fits must use the periodic J2/J3
   (G=0-dropped, eig/lindep pseudo-inverse), anchored at the trivial radius == global B.B (<1e-12 here).
5. Denominators: Foo/Fvv from the exxdiv='ewald' Fock (shifted), as stage 6.
6. **Before trusting any eps at Gamma**: the uniform G->0 dipole coupling makes the integral mask non-local below
   Omega_sc ~ 4pi mu*^2/eps. Options to evaluate (not tested here): threshold on J - J_unif and add the dropped pairs'
   uniform-field energy analytically (the Sylvester dipole formula reproduced it to <= 3e-6 Ha here), use a
   minimum-image distance/energy pair screen (the R_c rows lose < 1e-7 Ha/molecule beyond 10.5 Bohr), or move to
   k-points where G->0 is handled by the mesh. Report every per-molecule error with its a + b/N split.

## Iteration 5b (Python, LMP2 uniform coupling) — 2026-09-24

Follow-up to Iteration 5 item 6. Same system and construction as Iteration 5 (1x1xN needles, a0 = 7, tilted H2,
6-31G cart, cc-pvdz-ri, RS-GDF w = 0.5, shifted denominators, Berghold LMOs + periodic VV-HV). ONE toy crystal,
s-only basis, needle supercells only, N = 4..32 (dense solver; N = 40/48 not run: the dense closed-form references
hold ~6 copies of the no^2 nv^2 tensor, over the 2.5 GB budget at N = 48; N = 32 peaked at 1.36 GB).

### Code
- `pbc_lmp2.py`: `solve(..., gate="J"|"J-unif", J_unif=, pair_ecut=, addback=)` (defaults unchanged; strict
  `gate` values); `resta_z_at` (Resta matrix at m b_k by the same residue fold; m = 1 == data Zk to 1e-16);
  `transition_dipoles_richardson` ((b, 2b) Richardson of the Resta estimate: O(b^4) instead of O(b^2));
  `uniform_coupling` (needle form, W = zz); `uniform_pair_energies` (the Iteration-5 Sylvester add-back, now a
  library function); `pair_energy_estimate` (semicanonical pair energy from the gating tensor, energy screen);
  `canonical_mp2_from_local` / `mp2_local_closed_form` (closed-form MP2 of a modified local J rotated to the
  canonical / modified-Fock eigenbasis: the independent reference for the head-restored energies);
  `fock_head_correction` (the same head in the Fock exchange: Fvv, diag Foo).
- `run_lmp2_uniform.py` (hypotheses H1-H4, X1-X3 in its docstring; H1b added after the N <= 16 run, marked).
- `test_prototype.py`: +3 fast tests at the END (~34 s together, N = 4 wrapped + N = 8).

### The physics question, settled first (derivation, then measured)
A Gamma supercell is a 1x1xN k-mesh on the primitive cell; for the needle every momentum transfer q is along z.
For q != 0 the G=0 head of (ia|jb) is (4pi/Omega_sc)(q.mu_ia)(q.mu_jb)/q^2 = (4pi/Omega_sc) mu_z mu_z, constant;
at q = 0 it is dropped. In real space sum_{q!=0} e^{iqd} = N delta_d0 - 1, so every LMO pair carries
-(4pi/Omega_sc) mu mu: **the uniform coupling is exactly minus the omitted q=0 head**, a quadrature hole of weight
1/N. It is O(1) in total and O(1/N) per molecule, so in the thermodynamic limit it contributes nothing. It is a
finite-size ARTIFACT like Madelung, not long-range correlation. Restoring it (J' = J - J_unif) makes far-pair
couplings vanish and gives the self pair its N-independent (4pi/Omega_prim) mu mu. **It is part of Iteration 3's
c3**: c3's d(ia|ia) = -(4pi/3 Omega)|d_ia|^2 is this head for one molecule per cubic box (depolarisation W = I/3
instead of the needle's zz). c3 also contains the Fock-side heads (d eps_i, d eps_a), which J' does not touch.

H1 (discriminating: a physical term would make |b_head| > |b_can|). Fits E = e N + b + c/N on N = 16/24/32:

| reference | e_inf | b | c | b(N) = E - N e_inf, N = 4 .. 32 |
|---|---|---|---|---|
| canonical Gamma MP2 (J, F) | -1.9599301e-2 | +3.940e-3 | +3.1e-4 | +4.02e-3 .. +3.94e-3 |
| head in ERIs (J', F) = E_head | -1.9599298e-2 | +1.061e-3 | -4.0e-5 | +1.06e-3 .. +1.05e-3 |
| head in ERIs + Fock (J', Foo', Fvv') | -1.9599299e-2 | +1.43e-4 | -1.7e-5 | +1.5e-4 .. +1.3e-4 |

e_inf agrees to 3e-9 (guaranteed by the 1/Omega scaling: a consistency check, not evidence). b falls 3.7x with the
ERI head and 28x with ERI + Fock heads: 73% of the Gamma finite-size term is the ERI head, 23% the Fock heads,
3.6% (+1.4e-4) something not identified here (Foo off-diagonal heads, SCF relaxation). E_can - E_head =
+2.955e-3 (N=4) .. +2.890e-3 (32), converging to a constant, positive (the missing head underbinds), as predicted.
X1 (mu accuracy): farthest pair max|J - J_unif| 3.7e-5 / 1.3e-6 / 3.3e-7 at N = 4 / 8 / 32 with Richardson, vs
2.4e-4 / 3.0e-5 / 4.7e-7 with the single-b Resta estimate (max|J| 5.8e-3 / 2.9e-3 / 7.3e-4).

### Measured: candidates (dE in Hartree, >0 = correlation lost; partners/molecule equal for all molecules, every row)
Anchors at eps = 0 (every N): J-unif gate + add-back vs E_can <= 1.1e-13 (add-back exactly 0, nothing dropped);
CG on J' vs closed-form E_head <= 1.9e-14.

| setting (reference) | partners N=4/8/16/32 | dE N=8 | dE N=16 | dE N=32 | tail fit a (/molecule) | b | c (/N) |
|---|---|---|---|---|---|---|---|
| base eps 1e-4, gate J (E_can) | 4/8/16/32 (all) | 1.6e-5 | 2.7e-5 | 5.0e-5 | below onset N* = 232: no fit | | |
| A-add eps 1e-4: gate J', solve J, add-back (E_can) | 3/3/3/3 | 1.21e-4 | 7.40e-5 | 5.39e-5 | +2.5e-7 | +2.2e-5 | +7.7e-4 |
| A-add eps 3e-5 (E_can) | 4/3/3/3 | 2.86e-5 | 1.79e-5 | 1.34e-5 | +5.5e-8 | +6.3e-6 | +1.7e-4 |
| A-add eps 1e-4, NO add-back (mutation) | 3/3/3/3 | 8.0e-4 | 9.6e-4 | 1.04e-3 | +2.7e-7 | **+1.108e-3** | -2.5e-3 |
| A-drop eps 1e-3: gate+solve J' (E_head) | 1/1/1/1 | 1.01e-4 | 1.80e-4 | 3.53e-4 | +1.08e-5 | +6.7e-6 | +9.6e-6 |
| A-drop eps 3e-4 (E_head) | 3/3/3/3 | 5.49e-5 | 1.05e-4 | 2.06e-4 | +6.3e-6 | +3.7e-6 | +6.7e-6 |
| A-drop eps 1e-4 (E_head) | 3/3/3/3 | 2.28e-6 | 4.25e-6 | 8.25e-6 | +2.51e-7 | +2.2e-7 | +4e-7 |
| A-drop eps 3e-5 (E_head) | 4/3/3/3 | 4.6e-7 | 8.5e-7 | 1.65e-6 | +5.0e-8 | +3.7e-8 | +1e-7 |
| B R_c 10.5 min-image + add-back (E_can) | 3/3/3/3 | 2.36e-6 | 2.14e-6 | 2.26e-6 | +3.9e-9 | +2.19e-6 | -1.7e-6 |
| B R_c 10.5, no add-back (E_can) | 3/3/3/3 | 6.8e-4 | 8.9e-4 | 9.9e-4 | +5.0e-9 | **+1.089e-3** | -3.3e-3 |
| B energy screen 1e-6 / 1e-7 from J (E_can) | 4/8/16/32 (all) | 0 | 0 | 1e-13 | (onset N ~ sqrt(1e-3/T): ~32 / ~100) | | |
| B energy screen 1e-6 / 1e-7 from J' + add-back (E_can) | 3/3/3/3 | = R_c 10.5 row, same pair set, to 1e-12 | | | | | |

(Also N = 6, 12, 24 in the log; A-add and A-drop at eps 1e-3 keep self pairs only, partners 1.)
Mutations of the new code, each run against the 3 new tests in a scratch copy (8/8 caught): Richardson -> single-b
Resta (3 fail), J_unif volume Omega/8 (3), add-back not applied (1), gate on J + J_unif (1), energy screen from J
instead of the gating tensor (1), Fvv head sign (1), Foo head sign (1), add-back factor 2 dropped (1).

### Interpretation (provisional, 2026-09-24; one toy crystal, s-only basis, needles only)
- **The onset is gone with either J' gate**: partners saturate at 3/molecule (the R_c 10.5 set) from N = 6 at
  eps <= 3e-4, and at 1 from N = 4 at eps 1e-3. With the plain J gate, eps 1e-4 kept all N^2 pairs through N = 32
  (N* = 232), and so did an energy screen computed from J (far uniform pair energies ~1e-3/N^2 each: an onset
  again, at N ~ sqrt(1e-3/T)). **Energy-screening J does not remove the onset. Screening on J' does.**
- **A-drop (restore the head in the integrals, gate and solve on J') is the clean candidate.** Its error against
  its own reference is purely extensive and eps-controlled: a = 1.1e-5 / 6.3e-6 / 2.5e-7 / 5.0e-8 per molecule at
  eps 1e-3 / 3e-4 / 1e-4 / 3e-5, with |b| <= 7e-6. There is nothing to add back (far pairs carry no energy once
  the head is restored), and adding the Sylvester term would double-count ~-1e-3. Its reference E_head is also
  the better Gamma estimator of the TDL (b 1.06e-3 vs 3.94e-3).
- **A-add as specified (element gate on J', solve on J, add back entirely dropped pairs) is NOT clean.** It has
  the SAME a as A-drop, but also a +7.7e-4/N term (eps 1e-4) from the uniform part of elements masked INSIDE kept
  pairs, which the pair-level add-back cannot see. At eps 1e-4 it is 53x (N=8) to 6.5x (N=32) the A-drop error.
- **To reproduce canonical Gamma MP2 (e.g. the PySCF number), screen PAIRS (min-image R_c or the energy
  estimate from J'), keep all elements inside kept pairs, and add the analytic uniform energy back.** Residual
  b = +2.2e-6, a < 1e-8. Mutation: without the add-back b = +1.09e-3, so the add-back is needed exactly when
  the reference is E_can, and never when it is E_head.
- Answer to "remove or keep": the term is a finite-size artifact (a quadrature hole). Remove it (restore the head)
  for anything aimed at the TDL. Add it back only to match a canonical Gamma-point number.
- NOT established: the needle W = zz form only (cubic W = I/3 is Iteration 3's analogue, but no cubic supercell
  sweep was run); mu via Resta needs orthorhombic cells and LMO spread << L; the 1.4e-4 residual of the full head
  restoration is unexplained; Foo off-diagonal heads not derived; no polarisation functions (a is tiny here);
  N <= 32; no timing claims; k-points (candidate C, the principled fix: the mesh treats q -> 0 itself) not
  implemented.

### For the Rust port (hook in gamma_lmp2)
1. Implement **A-drop**. J' = J + (4pi/Omega_sc) mu mu^T is a POSITIVE rank-1 update per pair, so it is exactly one
   extra RI column: append B^{extra}_ia = sqrt(4pi/Omega_sc) mu_ia,z to the periodic B_ia. The eps mask, pair
   gate, solver and pair energies then need no change. Test: the eps = 0 anchor against closed-form canonical MP2
   on the augmented B, to 1e-12 (it is an independent construction).
2. mu_ia: the Resta matrix at b_z AND 2 b_z (one extra pair-FT, same fold), Richardson (4 f(b) - f(2b))/3.
   Port test: farthest-pair |J'| / |J| <= 2e-3 at 1x1x8 (single-b fails at 1e-2). Refuse non-needle supercells
   until W is derived for them (cubic W = I/3 by Iteration 3's c3).
3. Optional, same shape: the Fock heads (Fvv -= c mu^T mu, Foo_ii += c(sigma_i^2 - sum_k mu_ik^2)) take b down
   another ~7x. That is a finite-size correction on top of the LMP2, so flag it as a separate, opt-in option.
4. If a caller needs the canonical Gamma number, provide the pair-level screen + Sylvester add-back
   (`uniform_pair_energies`), never an element-level J' gate with a pair-level add-back.

## Iteration 6b (Rust, step-8 SR screening study) — measured 2026-09-24
`sr_screening_table` (release, 16177 s), SR Gaussian-nucleus 3-centre attraction in periodic_hcore.
Unscreened references: H2/STO-3G, triclinic 4H s+p at ω=0.3885 (1.63e9 triplets) and ω=1.3 (9.87e8).
| run | Derived bound violations | predicted/actual at 1e-10 | triplets at 1e-10 vs unscreened | mutant controls |
|---|---|---|---|---|
| H2 | 0 | — | — | 20 violations (control fires) |
| tri ω=0.3885 | 0 | 1.2e-6 / 1.07e-9 (~1100x) | 2.6M / 1634M | 0 violations: control SILENT |
| tri ω=1.3 | 0 | 4.2e-7 / 2.7e-10 (~1600x) | 0.78M / 987M | 80 (NoGaussianExtent up to 1.1e6x under-predicted) |
Interpretation (provisional): the Derived bound is conservative on every measured row (never under-predicts),
by roughly 3 orders of magnitude, so the default threshold is safe but over-computes by an unknown factor.
The ω=0.3885 row cannot certify the bound because its negative controls never fire there. Only toy cells
(≤16 AOs) were measured, so no cost or size claim.

## Iteration 7 (Python, Gamma UMP2/URPA) — 2026-09-24

### Code
- New `pbc_ump2.py`: `u_denominators` (per-spin shifted/unshifted), `u_bia` / `u_ovov` (alpha and beta B_ia from ONE
  B^P_mn, shared metric), `gamma_ump2` (E_aa, E_bb, E_ab), `direct_ump2`, `gamma_urpa` (`quad`: spin-summed
  Pi = sum_s 2 B_s diag(e/(w^2+e^2)) B_s^T, one ln det; `plasmon`: spin-orbital direct-RPA plasmon on the joint
  alpha+beta ov space, no aux, no quadrature), `urpa_second_order_quad`, `u_r2_kernel_c3` (box-limit predictor:
  harmonic kernel on the molecular UHF, relaxed; `second=True` also gives c6 = 1/2 E''(k) (4pi/3)^2),
  `uniform_occ_shift_slope` (unshifted c1). Seams `SPIN_FACTOR`, `u_bia` for mutation tests.
- Drivers: `run_ump2_anchor.py`, `run_ump2_formula.py` (molecular pins), `run_ump2_oracle.py` (PySCF pbc),
  `run_ump2_box_limit.py {H|NH} a...` (predictions + denominator mutants).
- `test_prototype.py`: +8 fast tests (~40 s), +1 slow (H2/6-31G + tri s+p PySCF pins, 173 s, passes). Whole default
  suite 67 passed / 5 skipped, 190 s.

### Measured: exactness anchors (run_ump2_anchor.py; mutants must be LARGE where the anchor can see them)
| anchor | shifted | unshifted | mutants |
|---|---|---|---|
| (a) closed shell U - R, H2/STO-3G a=4, same C: UMP2 / URPA plasmon / URPA quad | 0 / 2e-16 / 2e-16 | 9e-19 / 2e-16 / 4e-17 | per-spin factor 4: -1.8e-2 / -2.5e-2 |
| (a) same, tri one-s (dense + trivial-aux B-quad); UHF's own orbitals | <= 2.4e-16; 3.7e-12..1.0e-11 | same | factor 4 -5.3e-2 / -7.9e-2; ss 1/2 dropped +1.2e-2; ab x1/2 +6.2e-3 |
| (b) trivial-aux B - dense, tri one-s TRIPLET (3,1; ab only): UMP2 / URPA (own B-SCF) | 9.6e-15 / -1.1e-13 (-2.1e-13) | 9.3e-15 / -2.1e-13 (-3.4e-13) | alpha C for beta B: UMP2 +9.7e-4, URPA +2.2e-4; factor 4 -2.7e-2 |
| (b) same, penta one-s DOUBLET (3,2; aa, bb, ab all nonzero) | 6.1e-15 / 1.1e-14 (1.3e-14) | 1.2e-14 / 1.8e-14 (2.5e-14) | alpha-for-beta -6.3e-4 / -1.2e-3; factor 4 -5.5e-2; exchange dropped -1.3e-2; ss 1/2 dropped -1.2e-5 |
| (c) -(1/2pi) int tr Pi^2/2 - direct UMP2 (tri / penta) | -1.2e-13 / 1.2e-14 | -2.5e-13 / 2.4e-14 | factor 4: -3.4e-2 / -7.4e-2 |
Penta needs RS-GDF `prec=1e-15`: at 1e-13 the SR truncation leaves max|dI| 3.5e-10 -> dUMP2 1e-11. The triplet's
aa block is identically 0 (one alpha virtual), so it cannot see a same-spin 1/2 error: penta was added for that.
Molecular formula pins (run_ump2_formula.py; NH triplet, OH doublet, STO-3G and 6-31G, cart): UMP2 - pyscf.mp.UMP2
<= 5.6e-17 (ss and os separately); URPA quad (joint-ov eigen-factor) - plasmon <= 1.1e-13; tr Pi^2 - direct UMP2
<= 7.6e-17; our quadrature (n=40) on PySCF's own DF tensors - pyscf.gw.urpa <= 2.7e-13.
Source mutations (sed on pbc_ump2.py, all caught by the new fast tests): beta occupied unshifted (3 fail), same-spin
1/2 -> 1 (4), ab x1/2 (4), plasmon tr A with K/2 (6).

### Measured: PySCF oracle (run_ump2_oracle.py; pbc.scf.UHF + AFTDF mesh 61^3; PySCF started from our D)
| system | conv. | ours UMP2 (pure-AFT) | pbc.mp.UMP2 - ours | pbc UCCSD MP2 - ours shifted | URPA ours | PySCF AFTDF ov + our plasmon - ours |
|---|---|---|---|---|---|---|
| H2/6-31G a=4 triplet (2,0) | unshifted | -8.878036761406e-04 | +2.3e-14 | | -3.225957844666e-02 | -9.2e-12 |
| | shifted | -5.712132378835e-04 | +1.1e-14 | +1.1e-14 | -1.133223577926e-02 | +2.1e-13 |
| penta one-s doublet (3,2) | unshifted | -2.167566950379e-02 | +1.2e-12 | | -3.602910454578e-02 | +1.1e-12 |
| | shifted | -1.179101239478e-02 | +4.6e-13 | +5.3e-13 | -2.150209066750e-02 | +5.7e-13 |
| tri 4H s+p triplet (3,1) | unshifted | -3.332802555901e-02 | -3.3e-10 | | -6.935114479735e-02 | -4.5e-10 |
| | shifted | -2.278123418384e-02 | -1.4e-10 | -1.3e-10 | -4.834534729507e-02 | -1.1e-10 |
- PySCF convention confirmed per spin: pbc.mp.UMP2 on exxdiv=None == our unshifted, on exxdiv='ewald' == our shifted;
  pbc UCCSD's MP2 == shifted for both HFs. eps(ewald) - eps(None) = -v_M on EACH spin's occupied (<= 1.8e-10),
  0 on virtuals (<= 3.1e-10); UMP2 from the ewald-SCF eigenvalues - shifted(None eps) <= 1.6e-10.
- H2/STO-3G triplet would be a vacuous pin (no alpha virtual, no beta electron: UMP2 == URPA == 0), hence 6-31G.
- tri s+p 1e-10 residual is SCF orbitals (pure-AFT vs AFTDF integrals; dE_HF 7.7e-13), not the formula.
- URPA with a real aux: PySCF RSGDF (cart cc-pvdz-ri) + pyscf.gw.urpa arithmetic vs our RS-GDF B + quad (n=40):
  9e-12 (H2), 1.5e-12 (penta), 4.5e-10 (tri, same SCF floor). Our RS-GDF fit error vs exact, cc-pvdz-ri:
  +4.7e-5 / +1.8e-5 (H2 uns/sh, 1.5e-3 / 1.6e-3 rel), +1.9e-5 / +9.3e-6 (penta, 5e-4 / 4e-4), +9.5e-5 / +6.3e-5
  (tri s+p, 1.4e-3 / 1.3e-3; the restricted dRPA tri s+p was 8.8e-4).
- The H2 triplet URPA is 36x its UMP2 (unshifted): direct RPA has no same-spin exchange, so the aa ring self-
  correlation that UMP2's exchange cancels survives. This is expected for dRPA, not a defect.

### Box limit (run_ump2_box_limit.py; cubic box vs molecular UHF/UMP2/URPA, 6-31G cart; independent of PySCF pbc)
Predictions were printed BEFORE the sweep, from molecular quantities only:
- physics, shifted (eps_occ,s - v_M on BOTH spins): dE = c3/a^3 + O(a^-5), c3 from `u_r2_kernel_c3`.
  NH triplet (5,3), R 1.95: c3 HF -24.164001, UMP2 0.761014, URPA 1.502486. H doublet: HF -6.005259, UMP2 0
  (one electron, identically), URPA 0.046533.
- physics, unshifted: dE_uns - dE_sh = c1/a, c1 = -(v_M a) dE/ds: NH UMP2 -0.107139, URPA -0.125099; H URPA -0.007086.
- artifacts: per-spin chi0 factor 4, alpha C for beta B, missing images -> O(1) plateau; Madelung v_M/2 per spin or
  alpha-only shift -> 1/a in 'shifted'. Exponent 3 with the predicted c3 vs 1 vs 0 separates them.

| a | NH UMP2 sh dE*a^3 | NH URPA sh dE*a^3 | NH HF dE*a^3 | NH UMP2 uns dE | H URPA sh dE*a^3 | H URPA uns dE |
|---|---|---|---|---|---|---|
| 12 | 0.839248 | 1.598522 | -24.32721 | -1.007518e-2 | 0.046600 | -6.609e-4 |
| 16 | 0.797184 | 1.550901 | -24.38095 | -7.425312e-3 | 0.046726 | -4.859e-4 |
| 20 | 0.785177 | 1.534821 | -24.29949 | -5.845585e-3 | 0.046632 | -3.830e-4 |
| 24 | 0.778237 | 1.525530 | -24.25655 | -4.812011e-3 | 0.046590 | -3.157e-4 |
| 32 | 0.770999 | 1.515846 | -24.21502 | -3.548925e-3 | | |
| 40 | 0.767684 | 1.511312 | -24.19626 | -2.808723e-3 | | |
Wall 2-16 s per box. <S2> 2.012882 at a=40 (molecule 2.01288577). quad - plasmon <= 9e-14 at every a.
- NH shifted local exponents 3.049, 3.032, 3.019 (UMP2) and 3.033, 3.022, 3.013 (URPA) from 16->20 to 32->40.
  c3+c5 fit (32,40): UMP2 0.761792 (+1.0e-3 rel), URPA 1.503251 (+5.1e-4), HF -24.162919 (-4.5e-5). With c6 fixed at
  the kernel's second-order value (below): +6.4e-4 / +2.4e-4 / -2.9e-6. Iterations 3/4 got 3-4e-4 for the same kind of fit.
- Unshifted - shifted, c1/a + c2/a^2 fit (32,40): UMP2 -0.106870 (-2.5e-3 rel), URPA -0.124881 (-1.7e-3); with c3 term
  (24,32,40): -9e-4 / -1.1e-3. H URPA -0.007067 (-2.7e-3). At a=40 unshifted is still -2.8e-3 Ha (5% of E_UMP2).
- Denominator MUTANTS (same boxes, NH, dE*a at 24/32/40): v_M/2 per spin UMP2 -0.05447/-0.05452/-0.05446,
  URPA -0.0622/-0.0628/-0.0630; alpha-only shift UMP2 -0.0438/-0.0439/-0.0439, URPA -0.0623/-0.0623/-0.0621.
  These are flat 1/a plateaus, 30-60x the shifted residual at a=40. **The per-spin shift is therefore established
  by the box limit, not assumed: only the same v_M on both spins' occupied levels gives a^-3.**
- H atom: UMP2 == 0 exactly at every a. URPA dE*a^3 is NOT flat (unlike the minimal-basis UHF H of Iteration 6):
  after subtracting c3/a^3 the residual scales as a^-6 (local exponent 3.0 of (dE a^3 - c3) at 16->20->24),
  and the HF residual does the same. **This was found after the sweep and explained afterwards.** 6-31G lets the
  orbital relax in the harmonic en field, which is second order in k ~ a^-3, so a^-6. The spherical density
  still gives no c5. The kernel's second derivative, computed after the sweep but from the molecule only, gives
  c6 = -32.871 (HF) and 0.79013 (URPA). The measured (dE - c3/a^3)a^6 is about -33 and 0.79. The residual after
  both terms is 7e-7 of c3 (HF) and 4.5-5.7e-6 of c3 (URPA) at a=20-24; the a=10-12 rows are still image-overlap
  dominated. NH has c6 too (UMP2 -27.2, URPA -37.0, HF -93.2), which is why fixing c6 tightens the NH fits above.

### Interpretation (provisional, 2026-09-24)
- Gamma UMP2/URPA from the periodic B is the molecular UMP2/URPA on the per-spin transforms of ONE shared B^P_mn.
  It is exact to 1e-14 (UMP2) / 2e-13 (URPA) in the trivial-aux limit, and closed shell reduces to the restricted
  code to 1e-16 (same C). The per-spin chi0 factor 2 (restricted 4 = 2 spins x 2) is pinned without PySCF by
  tr Pi^2 == direct UMP2 and by the plasmon formula.
- Denominators: shifted means the same v_M on every spin's occupied levels, which equals the exxdiv='ewald'
  UHF eigenvalues. PySCF agrees (pbc.mp.UMP2 on an ewald UHF, pbc UCCSD always), and the box limit confirms it
  independently: a^-3 with the predicted c3 to 1e-3 (UMP2) and 5e-4 (URPA), while v_M/2 or an alpha-only shift
  leaves 1/a. Unshifted converges as 1/a with the predicted c1.
- The harmonic-kernel predictor extends to second order (c6) and explained a residual that looked like an
  artifact (the H atom "not flat"). The explanation came after the measurement, so it counts as an
  interpretation, not a pre-registered prediction. It was checked quantitatively on two systems.
- The RS-GDF fit error for URPA with cc-pvdz-ri is 4e-4 to 1.6e-3 relative, the same class as restricted dRPA.
- NOT measured: ROHF reference, frozen core in a periodic run (code path exists, not exercised), spherical basis,
  nao > 16 in the pure-AFT oracle, real solids (a^-3 is the isolated-molecule law), k-points, cost, spin-
  contamination effects beyond <S2> 2.013.

### For the Rust port (stage 8)
- `gamma_ump2`: `u_ri_mp2` builds its own molecular metric and B. Add `u_ri_mp2_from_parts(inter_a, inter_b,
  eps_a_full, eps_b_full)` that calls the existing `same_spin_pair_energy` (x2) and `opposite_spin_pair_energy`
  (u_rimp2.rs:1270/1285). They already index eps by `first_occ`/`nocc_total`, so the periodic caller builds two
  `RpaIntermediates` from the SAME periodic B^P_mn (naux = naux_KEPT, no further V^-1/2, as in stage 7) and passes
  full eps arrays with the occupied part already lowered by v_M, or the ewald-UHF eigenvalues unchanged.
  Refuse exxdiv=None eigenvalues unless the caller asks for unshifted explicitly.
- `gamma_urpa`: **yes, ferric-rpa needs a U-variant of `run_pdep_rpa_from_parts`.** `run_u_pdep_rpa` (lib.rs:1257)
  takes (mol, obs, dfbs, rhf) only to build `inter_a`/`inter_b` via `compute_rpa_intermediates_spin` (l.1341-1342)
  and to slice eps (l.1347-1360). From l.1361 on it reads only `inter_a`, `inter_b`, the four eps slices and
  `config`. Split it into `run_u_pdep_rpa_from_parts(inter_a, inter_b, eps_occ_a, eps_vir_a, eps_occ_b,
  eps_vir_b, config)` with the same refusals as the R version (trunc_thresh 0, Dense chi0, Lanczos, shape and
  e_ia > 0 checks), plus `inter_a.naux == inter_b.naux` as a hard error (it is only a debug_assert today).
  run_u_pdep_rpa then becomes a thin wrapper, which is bit-identical by construction. Quadrature n >= 40 as in stage 7.
- Tests to port: closed shell U == R (1e-12, both conventions); trivial-aux open-shell anchor on a DOUBLET with all
  three blocks nonzero (penta, 1e-11; a triplet with one alpha virtual cannot see same-spin errors);
  tr Pi^2 == direct UMP2 (1e-12); PySCF pins (UMP2_REF in test_prototype.py); NH box at a=32/40 vs c3 (3e-3) plus
  the v_M/2 mutant plateau; H atom UMP2 == 0 and URPA == c3/a^3 + c6/a^6 at a=20 (2e-5).

## Iteration 8 (Python, Gamma KS-DFT) — 2026-09-24

### Code
- `pbc_dft.py`: periodic grid **A2** (`PeriodicGrid`): TA-M4 radial × Lebedev atomic grids for the CELL's atoms only
  (ferric `build_atomic_grid` formula + ξ table), each weighted by the home atom's fuzzy-cell weight computed over
  IMAGE atoms (all R_B+L within D of the point). Why this is exact: w_{A,L}(r) = w_{A,0}(r−L) (translation
  covariance) and Σ_{A,L} w_{A,L} = 1, so ∫_cell f = Σ_A ∫_{R³} w_{A,0} f for lattice-periodic f — the full atomic
  grids with crystal weights, no cut at the cell boundary. The D-truncation keeps it an exact partition of unity
  (same neighbour set for every home at a given r) but D must exceed the covering radius of the atom lattice.
  Partitions: `becke` (ferric becke.rs, 3 iterations + Bragg size adjust), `becke1/2`, `ssf` (compact |ν|<0.64),
  `exp` (Hirshfeld-like e^{−2r} weights). AOs lattice-summed χ^Γ(r)=Σ_L χ(r−L) (molecular `eval_gto` on an image
  supermolecule). `uniform_grid` (**B**, oracle) and `pyscf_a1_grid` (**A1** = PySCF pbc BeckeGrids points:
  grids on image atoms, cut to the parallelepiped). `rks`: DIIS RKS on (S, h, jk, grid); hybrids use
  K_eff = hyb·(K + v_M S D S). J/K exact: dense pure-AFT I (pbc_gamma). libxc via pyscf.dft.libxc.
- Scripts: `run_dft_anchor.py` (N, S_latt, Slater E_x vs uniform), `run_dft_partition.py` (partition × grid incl.
  RKS energy vs uniform), `run_dft_box_limit.py` (A: grid identity vs molecular grid; B: KS box limit),
  `run_dft_oracle.py` (PySCF pbc.dft.RKS AFTDF 61³, exxdiv=ewald). Fix: `molecular_rks` grids.cutoff 0 → 1e-100
  (0 crashes PySCF numint binning).
- 5 tests at the end of test_prototype.py (1 PBC_SLOW).

### Predictions stated before each sweep (verbatim substance in the script docstrings)
- Box limit B: LDA/PBE no a⁻³ term (no exact exchange, H2 dipole 0) → a⁻⁵; PBE0 c3 = −hyb(4π/3)σ²; artifact:
  Madelung on full K → 4×. Huge-box grid A: artifact = plateau at grid-error level. Partition sweep: error shrinks
  faster for smoother partitions; artifact = Σw−Ω not shrinking and identical across schemes.

### Measured
Numint/SCF/Madelung plumbing (same points as PySCF): ours on A1 points − PySCF pbc.dft.RKS = −2.5e-14 (H2 a=4,
LDA/PBE/PBE0, 50/75/100 grids), −5.2e-13..−5.8e-13 (triclinic 4H s+p), Δε ≤ 2e-9. Our uniform-48³ RKS ≡ PySCF
UniformGrids to 1e-12 (H2 LDA −1.521492168321, PBE0 −1.576992652865; tri LDA −2.205431956735, 48³ ≡ 64³).

Grid error vs the spectrally converged uniform reference, E(grid) − E(uniform), Ha:
| system | grid | Becke A2 LDA | PySCF A1 LDA | exp A2 LDA | Becke A2 PBE0 | npts A2 |
|---|---|---|---|---|---|---|
| H2 a=4 | 50×146 | −7.8e-4 | −7.9e-4 | +1.0e-4 | −6.3e-4 | 14032 |
| H2 a=4 | 75×302 | −4.2e-4 | −4.4e-4 | +2.4e-6 | −3.3e-4 | 43702 |
| H2 a=4 | 100×590 | −6.0e-5 | −6.1e-5 | −3.6e-6 | −4.8e-5 | 113686 |
| H2 a=4 | 150×974 | — | — | −1.1e-6 | — | 282460 |
| tri 4H sp | 50×146 | −2.0e-4 | −2.0e-4 | −4.5e-4 | −1.6e-4 | 28389 |
| tri 4H sp | 75×302 | −2.9e-5 | −3.0e-5 | +5.3e-5 | −2.3e-5 | 87501 |
| tri 4H sp | 100×590 | −3.9e-6 | −3.8e-6 | −9.4e-6 | −3.1e-6 | 227408 |
- H2 a=4 anchors (probe density, D=10): |∫ρ−N| Becke 2.5e-3/1.05e-3/1.6e-4 (50/75×302/100×590), 1.5e-4 at 150×590
  (angular-limited: 100×302 ≈ 75×302, 150×590 ≈ 100×590); Σw−Ω +0.43/+0.20/+0.029. becke2/becke1 WORSE (1.4e-3/
  1.65e-3 at 75×302), ssf ≈ becke. exp: Σw−Ω 1.4e-5, |dN| 1.2e-5 at 75×302; D=10 vs 14 identical (not D-limited).
- Huge-box grid identity (LiH/STO-3G, 75×302, D=0.9a, point-by-point vs PySCF molecular grid within 0.08a):
  SSF max|Δw| 1.4e-8 (a=12), 5.7e-17 (20), 9.7e-17 (30), 3.9e-16 (60); Becke 1.5e-5/1.4e-6/4.7e-7/4.8e-11
  (algebraic, as predicted: Becke tails are polynomial in 1/a); control adjust=False: 0.29–0.30. E_xc(PBE, mol D)
  − molecular: SSF −3.1e-3 (12), +1.1e-6 (20), −4.9e-9 (30), −2.8e-11 (60).
  **Surprise, measured:** the first comparison within 0.15a FAILED for SSF (2.4e-3 at a=60): the Bragg size
  adjustment (|a|=½ clip for Li–H) maps ν → −1+4r/L, so a Li IMAGE 51 Bohr away enters the SSF support for an H
  point 8.7 Bohr out (ν = −0.445). The compact-support radius is ~0.09L with size adjustment, not 0.18L.
- KS box limit (H2/STO-3G, 75×302, pure-AFT, vs pyscf.dft.RKS molecular on the identical grid):
| a | dE LDA | dE PBE | dE PBE0 | dE PBE0·a³ |
|---|---|---|---|---|
| 12 | −3.31e-5 | −3.29e-5 | −1.486e-3 | −2.5685 |
| 16 | +8.73e-7 | +8.75e-7 | −6.119e-4 | −2.50645 |
| 20 | +2.91e-7 | +2.92e-7 | −3.133e-4 | −2.50663 |
| 24 | +1.15e-7 | +1.15e-7 | −1.813e-4 | −2.50679 |
  LDA local exponents 4.92 (16→20), 5.11 (20→24): a⁻⁵ as predicted. PBE0 two-term fit (20, 24) c3 = −2.50715 vs
  predicted −0.25·(4π/3)·2.39406725 = −2.50706 (3.6e-5 rel.). σ² equals the HF value because the minimal-basis
  σg orbital is symmetry-fixed.
- Costs (H2 a=4): image atoms per point in the Becke product 7/31/64/138/362 at D = 4/6/8/10/14 (pair products
  O(n_nb²) per point; exp is O(n_nb) and built 10–30× faster here); live image cells per 448-point AO chunk
  260–500 (thresh 1e-15); LiH box a=12: 68, a=30: 4. A2 point counts ≈ A1 (43702 vs 44267 at 75×302).

### Interpretation (provisional, 2026-09-24; two toy cells, H/s+p only, no cusp-heavy atoms)
- The periodic construction is right: exact plumbing vs PySCF, exact molecular limit (SSF 1e-16), hybrid box limit
  = hyb × the HF Makov-Payne term with the predicted coefficient, semilocal a⁻⁵.
- The dominant error is the **Becke partition in a dense lattice**, not the domain cut: A2 (no cut) and PySCF A1
  (cut) agree with each other to ~1e-6 and are both 4e-4 Ha off at 75×302 for H2 a=4. Error is angular-limited.
  ferric's molecular default (75,110) is not adequate periodically (|dN| 1.9e-3 on H2 a=4).
- "Smoother partition converges faster" is REFUTED as stated: becke1/2 are worse. exp (Hirshfeld-like) wins 175× on
  H2 at 75×302 but LOSES 2× on the triclinic cell — no partition winner is established; do not port exp on this
  evidence. Grid choice needs a measured sweep on a real solid (with core electrons) before any default is set.
- Uniform grids are spectrally exact for these all-Gaussian H densities; that says nothing about all-electron
  cores (tight exponents), which is why atom-centred grids are needed at all — untested here.

### For the Rust port (stage 2)
- **XC kernel is reusable unchanged**: `semilocal_vxc_closed(grid, chi, dchi, dens, tau, xc)` takes points,
  weights and AO tables only. Feed it periodic GridPoints and lattice-summed χ/∇χ.
- **Becke must change**: `becke_weight(mol, a_idx, r)` loops over `mol.atoms`. Needs a variant over an image-atom
  neighbour list (xyz, Z, home index) from a cell list, with truncation D ≥ covering radius; prefer SSF for finite
  exact lists, remembering size adjustment widens the support to ~0.09L. `build_atomic_grid` radial/angular reused.
- **AO**: `eval_basis_and_grad_on_points` on an image-shell supermolecule, summed into cell AO indices, screened by a
  per-shell extent (ao_rcut); budget npts×nao×4 through check_ao_grid_budget.
- **Injection hook**: keep `validate_injected` rejecting `xc` UNLESS the injection carries a grid:
  add `PeriodicInjection.xc: Option<Box<dyn XcBuilder>>` with `build(&D) -> (E_xc, V_xc)` (owning grid + AO tables),
  and have `solve_rhf_impl` use it in place of the molecular grid path when present. Hybrid: the injected K
  builder keeps the Madelung term and the SCF applies hyb (k_mix) to it (measured: Madelung on hyb·K gives c3 =
  hyb × HF). Still reject: RSH (needs attenuated periodic K), meta-GGA (not prototyped), grid response/gradients,
  newton/fxc (molecular grid rebuild), and grid pruning.

## Iteration 10 (Python, Gamma UKS) — 2026-09-24

### Code
- New `pbc_uks.py`: `uks(S, h, jk, enn, na, nb, grid, xc, kshift=v_M, guess=, mix=, staged=, level_shift=, diis_start=)`
  (xc='HF' = UHF through the same loop), `roks` (Roothaan effective Fock in PySCF's projector form, DIIS on
  [F_eff, D_a+D_b]), `eval_vxc_uks` (libxc spin=1: vrho (N,2), vsigma (N,3) = aa, ab, bb), `occ_gap`
  (occupation-aware gap: NEGATIVE for a hole state; the sorted-eigenvalue gap cannot see a trap), `uks_c3_closed_form`,
  `uks_r2_kernel_c3(..., second=True)` (c3 AND the relaxation c6), `MolGrid` (molecular UKS on a fixed molecular grid).
  Mutation seam `_MUTANT` in {unpolarized, madelung_full_k, hyb_half_per_spin}.
- Drivers: `run_uks_anchor.py {h2|tri}`, `run_uks_oracle.py [H|H2|tri]`, `run_uks_box_limit.py BASIS a...`,
  `run_uks_trap.py [alpha...]` (env CORR='' for exchange-only hybrids).
- test_prototype.py: +7 tests at the end (6 fast, 57 s; 1 PBC_SLOW: tri pins + trap).

### Convention (PySCF 2.13 pbc/dft/uks.py, then measured)
F_s = h + J[D_a+D_b] − α(K[D_s] + v_M S D_s S) + V_xc^s;  E = Σ_s tr D_s h + ½ tr D J − (α/2) Σ_s tr D_s(K[D_s] + v_M S D_s S)
+ E_xc + E_nn. Madelung rides on the exact-exchange part only (Iteration 8) and per spin with coefficient 1 on D_s
(Iteration 6). Closed shell reduces to Iteration 8's RKS (−(α/2)(K[D] + v_M S D S)). E_ewald − E_none = −α v_M N/2.

### Measured
Anchors (run_uks_anchor.py; A1 50×146 grid, pure-AFT I):

| check | H2 a=4 | tri 4H s+p |
|---|---|---|
| (a) na=nb UKS − RKS, LDA/PBE/PBE0 × none/ewald | ≤ 4.4e-16 | ≤ 1.8e-15 |
| (a) ROKS(na=nb) − RKS | 0 | — |
| MUTANT hyb/2 per spin, PBE0 none / ewald | +5.6e-3 / +9.4e-2 | +4.7e-2 / +2.0e-1 |
| MUTANT Madelung on full K, ewald (predicted −(1−α) v_M N/2) | LDA −0.70932 (pred −0.70932), PBE0 −0.53199 (−0.53199) | −1.2449 / −0.93366 (exact) |
| MUTANT unpolarized XC, closed shell | 0 (blind, as predicted) | ≤ 3e-15 (blind) |
| (b) open shell UKS[xc=HF] − pbc_uhf.uhf, none / ewald | 1e-16 / 2e-16 (triplet) | 1.1e-14 / −2.9e-14, d<S2> 2e-10 |
| (c) tr(V_s dD) vs FD of E_xc, PBE, alpha / beta (relative) | — (nb=0: beta FD meaningless at rho_b=0) | 4.9e-9 / 2.1e-9 |
| (d) open shell (E_ewald − E_none) + α v_M N/2 | ≤ 5e-16 | ≤ 1.2e-14 |
| ROKS(triplet, 2 e in 2 AOs) − UKS | ≤ 8e-16 (fully determined) | |
MUTANT unpolarized XC on the H2 triplet: +0.120 / +0.129 / +0.090 (LDA/PBE/PBE0).

PySCF oracle (run_uks_oracle.py; pbc.dft.UKS/ROKS AFTDF 61³, exxdiv=ewald, BeckeGrids (50,146) treutler prune None,
small_rho_cutoff 0; ours on PySCF's own A1 points; PySCF from its DEFAULT guess; from our D in parentheses):

| system | xc | ours E (UKS) | <S2> | PySCF dE | ROKS E | PySCF dE |
|---|---|---|---|---|---|---|
| H a=4 doublet | LDA | −0.668127813328 | 0.75 | +7.2e-14 | | |
| | PBE | −0.677787718838 | 0.75 | +7.2e-14 | | |
| | PBE0 | −0.700202408672 | 0.75 | +7.5e-14 | | |
| H2 a=4 triplet | LDA | −0.261697156130 | 2.0 | +9.5e-15 | | |
| | PBE | −0.307603027052 | 2.0 | +1.0e-14 | | |
| | PBE0 | −0.332094904636 | 2.0 | +8.3e-15 | | |
| tri 4H s+p triplet | LDA | −1.694673507916 | 2.0003319270 (d 4.6e-12) | +4.8e-13 | −1.694361887534 | +4.8e-13 (1.6e-11) |
| | PBE | −1.731772052782 | 2.0005413886 (d −8.8e-11) | +5.1e-13 | −1.731308266751 | +3.5e-12 |
| | PBE0 | −1.777429192569 | 2.0006805370 (d 7.9e-11) | +5.6e-13 | −1.776801906487 | +5.3e-13 |
All PySCF runs converged to the same state from their own guess. Our molecular UKS (MolGrid) == PySCF molecular
UKS on the same grid to ≤ 4e-15 (H/STO-3G, H/6-31G, LDA/PBE/PBE0).

Box limit (run_uks_box_limit.py; H atom doublet, cubic box, periodic SSF 75×302 D=0.9a vs molecular UKS on the
identical molecular grid). Predictions in the docstring, written before the sweep: c3 = −(2π/3) α Ω_a(KS orbital)
(= α × the UHF coefficient; semilocal XC is local); spherical density → no a⁻⁵; relaxation → a⁻⁶ only if the
basis can relax; LDA/PBE: no a⁻³ and no relaxation (J + e-n harmonic potentials cancel for a neutral centred atom).
Artifacts: Madelung on full K → 1/a with −(1−α)·2.8373/2; UHF coefficient → 4×.

| a | STO-3G PBE0 dE·a³ | STO-3G LDA dE | 6-31G PBE0 dE·a³ | 6-31G c3 + c6/a³ | 6-31G LDA dE |
|---|---|---|---|---|---|
| 10 | | | −2.223183 | | −7.8e-4 |
| 12 | −1.036738 | −9.8e-6 | −1.551756 | | −3.4e-5 |
| 14 | | | −1.497230 | | −3.1e-7 |
| 16 | −1.019298 | +2.2e-7 | −1.492950 | | +8.6e-7 |
| 20 | −1.020201 | +8.4e-9 | −1.495931 | −1.496230 | +3.9e-8 |
| 24 | −1.020268 | +1.8e-10 | −1.496113 | −1.496125 | +9.4e-10 |
| 28 | | | −1.496071 | −1.496071 | +1.7e-11 |
| 32 | −1.020270 | +3.1e-14 | −1.496041 | −1.496041 | +2.1e-13 |
| 40 | | | −1.496011 | −1.496011 | +4.4e-16 |
- Predicted c3: STO-3G −1.020270 (Ω_a 1.948573 fixed by the single AO = 0.25 × UHF's −4.081081); 6-31G −1.495980
  (Ω_a 2.857112 of the PBE0 orbital; HF/LDA/PBE orbitals give different Ω, e.g. LDA 2.964759). The relaxed r2-kernel
  FD reproduces both to 1e-6 and gives c6 = ½E''(k)(4π/3)² = −2.0020 (6-31G), 0 (STO-3G).
- 6-31G: c3 + c6/a³ matches dE·a³ to ≤ 1e-6 from a=28; c3 alone is off 9e-5 (28), 6e-5 (32). The a⁻⁶ relaxation
  term is REAL for the hybrid (UHF H had none because Hartree == self-exchange for one electron; with α < 1 the
  harmonic self-interaction (1−α) is uncancelled). Below a≈20 the residual is image/grid tails (LDA column).
- (none − ewald) − α v_M/2: ≤ 8e-15 at every a. MUTANT Madelung on full K at a=40: −2.662e-2 = −0.75 v_M/2 (1/a).

Ewald trap for hybrids (run_uks_trap.py; tri 4H s+p triplet, α·HF + (1−α)·PBE exchange; starts: staged
none→ewald, core guess, and the UHF TRAP density):
- Identity checked on every converged ewald state (27 + 15 runs): re-evaluated under none it is stationary
  (|[F,D]| ≤ 2e-7) with E offset exactly α v_M N/2 (≤ 6e-15), and ewald occ-gap = none occ-gap + α v_M (e.g. α 0.25:
  0.1774 = 0.0218 + 0.1556). So the gap criterion scales with α exactly as predicted: per-spin gap ≥ α v_M.
- With PBE correlation, α = 0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0: every start converges to the staged state
  (≤ 1.3e-12); no trap, flag never set. Ground-state none-gaps are small (alpha 0.0059–0.038) but positive.
- Exchange-only (CORR=''), α = 0.5, 0.75, 0.9, 0.95: same (no trap, including from the UHF trap density). α = 1.0
  (== HF) reproduces Iteration 6: core guess and trap-density start give −1.8129587148 (+1.50e-2 above staged),
  none occ-gap −0.0057 (hole), flag True.
- DIIS stagnated (energy frozen, commutator 3e-4..2e-2) from the core guess for α ≥ 0.75 under ewald; plain
  level-shifted Roothaan (shift 0.5, no DIIS) converged to the staged state. Staged starts never stagnated.

### Interpretation (provisional, 2026-09-24; three toy H cells, s/sp bases)
- Gamma UKS is the molecular UKS on (S, h, I, E_nn, periodic grid): exact vs RKS (closed shell), vs UHF (xc=HF),
  and vs PySCF pbc.dft.UKS/ROKS to ≤ 3.5e-12 with <S2> to 1e-10.
- The box limit is α × the UHF Makov-Payne term evaluated with the KS orbitals, plus a relaxation a⁻⁶ term that
  the r2-kernel construction predicts to 6 digits. Semilocal parts contribute no power law.
- The Ewald trap is possible for any α > 0 in principle (window α v_M), but on the one cell that traps UHF, the hole
  state is not a stationary point of any hybrid tested (α ≤ 0.95, or α = 1 with PBE correlation). This is ONE cell;
  do not read "hybrids never trap". The smaller window (α v_M) makes it less likely, not impossible.
- Energy tests are blind to potential errors that move the variational energy only at second order or act on an
  empty spin: source mutations dropping vsigma_ab, or feeding beta the alpha vrho / vsigma_aa, survived every energy
  test; only the FD test tr(V_s dD) == dE_xc (both spins populated, rho_a ≠ rho_b) caught them. Madelung ×0.5 on
  the hybrid part fails all 5 energy tests.
- NOT measured: meta-GGA, RSH, spherical AOs, cores, RS-GDF for UKS, k-points, ROKS box limit (ROKS pinned on tri only).

### For the Rust port: periodic UKS in `solve_uhf_injected`
- Today `solve_uhf_injected` refuses `PeriodicInjection.xc` by name (uhf.rs:137). Add a polarized builder:
  `trait UksXcBuilder { fn build(&mut self, d_a, d_b) -> Result<(f64, Array2, Array2)>; fn exact_exchange_fraction(&self) -> f64; }`
  (or `XcBuilder::build_polarized` with a default that errors) and a `PeriodicInjection.xc_uks` slot, accepted
  only by the UHF path. Implement it over the periodic grid + lattice-summed AO tables with ferric's existing
  `semilocal_vxc_polarized` (vxc.rs:521; same libxc (N,2)/(N,3) layout) — the kernel needs no change.
- In the loop: F_s = h + J − a·K_inj(D_s) + V_s with a = exact_exchange_fraction (the injected K carries the
  Madelung term, linear in D_s: NO per-spin ½, NO Madelung on (1−a)); for a = 0 do not build K. E_xc outside the
  trace, exactly the molecular `UksXcContribution::add_xc_uks` convention. ROKS: same builder, Roothaan F_eff.
- Keep refusing RSH/meta-GGA/VV10/fxc-Newton/stability/SAD on this path (Iteration 8 list + Iteration 6 list).
- Ewald trap: report the OCCUPATION-AWARE per-spin gap (min ε_unocc − max ε_occ of the undamped Fock with
  occupations from D) against a·v_M at convergence and warn if below; offer the staged start (a·v_M off, then on).
  The staged start is always safe and never stagnated here; recommend it as the default for a > 0.
- Tests to port: UKS(na=nb) == injected RKS for LDA/PBE/PBE0 × both exxdiv (1e-11); UKS xc=None == injected UHF
  (1e-11); ewald − none == −a v_M N/2 (1e-11); PySCF pins above (1e-10); H/STO-3G a=24 PBE0 dE·a³ == −1.020270
  (2e-5); **FD test tr(V_s dD) vs E_xc on an open shell with both spins populated (1e-6)** — the only test that
  sees vsigma_ab / per-spin column mistakes.

## Iteration 9 (Python, k-point RHF) — 2026-09-24

### Code
- New `pbc_kpts.py` (~300 lines): `build_k(cell, n, exxdiv, gcut, thresh)` (Gamma-centred n1 x n2 x n3 mesh, pure AFT,
  dense Nk^2 nao^4 kernels Jker/Kker, complex S(k), T(k), V(k)), `krhf(kb, nelec, kshift=)` (complex RHF per k, global
  aufbau as PySCF KRHF get_occ, DIIS over the stacked k blocks with Gram Re sum_k <e_i,e_j>), `kmesh_madelung`,
  `gamma_aft` (pbc_gamma's pure-AFT Gamma build, G-chunked, explicit gcut: the independent supercell reference),
  `supercell_cell`, `omega_I` (Marzari-Vanderbilt gauge-invariant spread from the same pair FT), `_MUTANT` switch.
- Drivers: `run_kpts_anchor.py` (a/b/c + mutants), `run_kpts_oracle.py` (PySCF KRHF+AFTDF, env PYSCF_MESH),
  `run_kpts_convergence.py nmax [a] [nmin]` (hypotheses in its docstring, written before the run).
- `test_prototype.py`: +8 tests at the END (~2 min under load; supercell anchor at a loose gcut, see below).

Conventions (derived, then checked): chi_mk = sum_L e^{ik.L} phi_m(r-L) (PySCF's: S(k) equals `pbc_intor(kpts=)` to
5e-15; the conjugate convention differs by 0.8). Pair FT P^{kk'}_mn(K) = sum_L e^{ik'.L} FT[phi_m phi_n(.-L)](K) on
K = G + (k'-k); ERI (m k1 n k2|l k3 s k4) = (1/Omega) sum_{K!=0} 4pi/|K|^2 P^{k1k2}_mn(K) conj P^{k4k3}_sl(K).
Only K = 0 is dropped, i.e. the q = 0 (k = k') G = 0 head of exchange; J and V_ne drop G = 0 as at Gamma.
exxdiv='ewald': K(k) += v_M S_k dm_k S_k with v_M = the Gamma Madelung constant of the diag(n) supercell
(`tools.pbc.madelung(cell, kpts)` builds exactly that lattice; `df_jk._ewald_exxdiv_for_G0` adds it per k with no
1/Nk). Build: e^{ik'.L} depends only on L mod the mesh, so one residue-resolved pair FT per q
(`pbc_supercell.pair_ft_residues` on the K = G+q set) gives P^{k'-q,k'} for every k' by an (Nk x R) phase product;
time reversal Kker[-k,-k'] = conj Kker[k,k'] skips the -q passes (vs brute force 1.3e-15). Cost ~ (Nk/2) Gamma pair FTs.

### Measured: exactness anchors (run_kpts_anchor.py)
| anchor | system | result |
|---|---|---|
| (a) 1x1x1 mesh vs pbc_gamma build_integrals + rhf | H2/STO-3G a=4, none / ewald | dE 8.4e-15 / 6.7e-15, d eps 9.9e-14; = pinned PySCF (Iteration 1) |
| (b) k-mesh E/cell vs explicit-supercell Gamma E/N | H2 1x1x3 (prec 1e-8 gcut), none / ewald | -4.0e-14 / -4.1e-14; all Nk*nmo eigenvalues 1e-13 |
| | H2 2x2x2 (prec 1e-6) | 4.9e-15 / 4.2e-15 |
| | triclinic 4H s+p 1x1x3 (prec 1e-6) | -8.5e-13 / -8.5e-13; eps 3-6e-13 |
| | H2 1x1x3 at prec 1e-4, pair thresh 1e-8 (the test config) | -3.8e-14 / -3.7e-14 |
| v_M(k-mesh) vs madelung(supercell) vs PySCF madelung(cell, kpts) | 1x1x3, 2x2x2, 2x2x3, 1x2x3 tri | 0.0 / <= 9.4e-16 |
| (c) max\|S-S^H\|, \|S(-k)-S(k)^*\|, \|V(-k)-V(k)^*\|, \|J-J^H\|, \|K-K^H\| (Hermitian test dm) | H2 2x2x3, tri 1x2x3 | <= 4.4e-16, 5.2e-16, 6.8e-15, 4.2e-16, 4.5e-16 |
The supercell anchor is exact at ANY gcut, because {G + q : q in mesh} IS the supercell reciprocal lattice and both
sides sum the same |K| <= gcut sphere with the same pair-screening test (measured: 4e-14 at prec 1e-4 as at 1e-8).
That is what makes it cheap enough to be a fast test. It needs n >= 3 on some axis: at n = 2, e^{ik.L} = e^{-ik.L}.
Mutations (H2 1x1x3, all caught by (b), each also a fast test): pair-FT phase e^{-ik'.L} with S,T unchanged -0.219 Ha;
exchange kernel v(G) instead of v(G+q) +0.377 (both exxdiv); primitive-cell v_M instead of the supercell v_M -0.520
(ewald only; none row unchanged to 4e-14, as it must be). Blind spot: flipping the phase EVERYWHERE is k -> -k,
an exact relabelling by time reversal; no energy anchor can see it (the PySCF S(k) comparison does).

### Measured: PySCF oracle (pbc.scf.KRHF + AFTDF, cell.precision 1e-12, started from our dm, conv 1e-11)
| system, mesh | exxdiv | E/cell (ours) | dE vs PySCF | max\|d eps\| |
|---|---|---|---|---|
| H2/STO-3G a=4, 1x1x2 (mesh 61^3) | none / ewald | -0.902683427348 / -1.354143879961 | -1.9e-14 / -2.7e-14 | 1.1e-13 |
| H2/STO-3G a=4, 2x2x2 (61^3) | none / ewald | -0.700885391756 / -1.055547576692 | -1.9e-14 / -1.9e-14 | 1.6e-13 |
| tri 4H s+p, 1x1x2 (61^3 and 41^3) | none / ewald | -1.587649533398 / -2.327120141714 | -9.6e-13 / -9.6e-13 | 2.2e-9 (same at 41^3 and 61^3) |
tri 2x2x2 NOT compared: the first run's process was killed silently under the 1.5 GB cap during the PySCF
step (ours had taken 31 min); a rerun (2 GB cap, PySCF mesh 41^3) was still running when this entry was written.
The tri eps residual 2.2e-9 does not move with the PySCF mesh (41^3 vs 61^3), so it is not the AFT mesh; it is below
the energy-relevant level (dE 1e-12, quadratic) and not isolated (candidates: 1e lattice-sum ranges, rcut_1e 22 vs
PySCF precision 1e-12). Wall time (loaded box, load ~20-28): ours tri 2x2x2 31 min, PySCF H2 2x2x2 ~8 min per exxdiv.

### Measured: mesh convergence (run_kpts_convergence.py; H2/STO-3G, gcut prec 1e-10, same K sphere at every n)
Hypotheses (docstring, before the run): H0 identity E_none - E_ewald = nocc v_M(n) = 2.8372974795/(n a) (occupied
levels shift rigidly by -v_M, D unchanged in an insulator) => none converges as N_k^(-1/3) with a KNOWN coefficient;
H1 ewald leaves the q^2 term of the q = 0 head: E_ewald(n) - E_inf ~ -(4pi/3) Omega_I / (n a)^3 (N_k^-1), Omega_I = the
MV gauge-invariant spread, the crystal form of Iteration 1's c3 = -(4pi/3) sigma^2 (equal for a flat band).

| n | a=4 E_ewald | a=4 gap | a=4 Omega_I | a=6 E_ewald | a=6 gap | a=6 Omega_I |
|---|---|---|---|---|---|---|
| 1 | -1.658327061048 | 2.079 | – | -1.238530934105 | 1.423 | – |
| 2 | -1.055547576690 | 0.586 | – | -1.120361720418 | 1.098 | – |
| 3 | -1.099878133431 | 0.909 | 2.0227 | -1.117695348104 | 1.158 | 2.2609 |
| 4 | -1.086067029552 | 0.508 | 2.1321 | -1.116650366716 | 1.086 | 2.3195 |
| 5 | -1.086911915158 | 0.656 | 2.2545 | -1.116295989516 | 1.111 | 2.3481 |
| 6 | -1.085799824217 | 0.495 | 2.2961 | (running) | | |
E_none - E_ewald - v_M(n) <= 1.5e-15 at every n, both cells (H0 holds to machine precision; v_M n a = 2.8372974795).
So exxdiv=none is off by +nocc 2.837/(n a): +0.142 Ha at n = 5 (a=4), +0.095 (a=6).
- a=4 (the Iteration-1 cell: H2 units 2.6 Bohr apart along z, a dispersive band, gap 0.5-2.1 swinging with n): E_ewald
  oscillates even/odd through n = 6 (n = 4/5/6: -1.08607, -1.08691, -1.08580; steps 8e-4, 1.1e-3).
  Band-sampling (smooth-integrand quadrature of a dispersive band) dominates, not the Coulomb head; no power-law tail exists in this range and none was fitted. H1 is untestable here.
- a=6 (gap ~1.1 for n >= 2): two-point (4,5) fit c3/n^3: c3 = -0.04648, E_inf = -1.115924. Predicted c3 =
  -(4pi/3) Omega_I/216 = -0.04554 (Omega_I(n=5) = 2.348, still rising with n: finite-difference O(b^2)) or -0.04643 with
  the molecular sigma^2 = 2.39407 (the flat-band limit). Three-point (3,4,5) c3 + c5 fit: c3 = -0.0432, c5 = -0.038
  (unstable: n = 3 is not asymptotic). c3-only fit on (3,4,5): c3 = -0.0483, E_inf = -1.115903; local exponents with that E_inf: 2.25 (2->3), 3.04 (3->4), 2.88 (4->5).

### Interpretation (provisional, 2026-09-24; H2/STO-3G only, two cubic cells, n <= 6 at a=4, n <= 5 at a=6)
- **k-mesh RHF is the Gamma supercell, term by term.** The anchor is exact to 1e-14..1e-12 (H2 1x1x3, 2x2x2; triclinic
  s+p 1x1x3) and PySCF KRHF/AFTDF agrees to 2e-14 (H2) / 1e-12 (tri). The only k-specific physics is which K is
  dropped (K = 0, the k = k' head) and which v_M is added (the supercell's). They are consistent by construction:
  **the supercell v_M and the k-mesh v_M are the same number** (PySCF computes the k-mesh one by building the supercell),
  measured equal to 0 / 1e-15. Iteration 5b's "missing q = 0 head" is the same K = 0 hole, weight 1/Nk.
- exxdiv=none vs ewald is not an empirical race: none = ewald + nocc v_M(n) exactly, so none converges as N_k^(-1/3)
  with the Madelung constant as coefficient. ewald converges as N_k^-1 once the band is sampled (a=6: exponent ~3 in n,
  coefficient within 2% of the Omega_I prediction from two points). On the dispersive a=4 cell even n = 5 is not in the
  asymptotic regime; the finite-size ANALYSIS (which power) must be done on the tail of a mesh sweep, never on n <= 3.
- The dense AFT kernel is an oracle only: tri 2x2x2 (nao 16) took 31 min of pair FTs; Nk^2 nao^4 memory.
- NOT measured: shifted (non-Gamma-centred) MP meshes, metals / partial occupation (global aufbau is implemented but
  every run here had nocc per k constant), RS-GDF with k (complex B^P(k,k')), spherical basis, nao > 16, UHF/k,
  k-point MP2/RPA, cost at scale.

### For the Rust port (stage 3)
What becomes complex (file:line from this worktree):
- `ScfResult` (ferric-scf result.rs:37: densities, `mos_*` Array2<f64>, `eps_*` Vec<f64>) -> per-k Vec<Array2<Complex64>>;
  eps stay real (Vec<Vec<f64>>). Recommendation: do NOT generify ScfResult; add `KScfResult` in ferric-pbc.
- `Diis` (diis.rs:136, RingHistory<Array2<f64>> + f64 GramCache) -> history of Nk complex blocks; Gram entry
  Re sum_k <e_i,e_j>, extrapolation coefficients stay REAL (as in pbc_kpts.krhf). Either a small `KDiis` in ferric-pbc or a
  `DiisVector` trait (dot + axpy) so the f64 instantiation stays byte-identical.
- `canonical_orthogonalizer` (rhf.rs:2547, pub(crate), real) -> a Complex64 Hermitian version (zheevd via ndarray-linalg,
  already a dependency of ferric-pbc), same lindep threshold, per k.
- `JBuilder` / `KBuilder` (fock.rs:10/16, &Array2<f64>) and `PeriodicInjection` (rhf.rs:780, real s/h + boxed builders)
  -> new `KPointInjection { s, h: Vec<Array2<C64>>, vnn, madelung, jk: Box<dyn KPointJk> }` with
  `KPointJk::build(&mut self, dm: &[Array2<C64>], j: &mut [..], k: &mut [..])`: J needs only rho(G) (q = 0), K needs the
  Nk^2 (k,k') pairs, so one trait with both is the natural seam.
Minimal design: a separate `ferric_pbc::kscf::solve_krhf(cell, mesh, inj, cfg) -> KScfResult` loop (~300 lines: per-k
orthogonalizer + eigh, global aufbau with a hard error when occupation per k changes between iterations or the gap
closes, complex DIIS, E = (1/Nk) sum_k Re tr[(h+F) dm]/2 + E_nn, K += v_M S dm S). Leave `solve_rhf_impl` (CC 176,
byte-identity contract) untouched; reject the same config features `validate_injected` rejects. Reuse:
(i) Nk = 1 dispatches to `solve_rhf_injected` (real path), and the anchor (a) test pins KScf-at-Gamma == real driver;
(ii) time reversal: only one of each (k, -k) pair is diagonalised and built, C(-k) = C(k)^*; TRIM k (2k in G) have real
S(k), h(k) and can use the real eigensolver; (iii) v_M = the existing Gamma Madelung of the diag(n)-scaled lattice
(no new code; test it against the explicit supercell, as here).
Integral side (the bulk of the work): `pair_ft.rs` already accepts arbitrary vectors, so K = G + q needs no change;
add a residue-bucketed variant (accumulate lattice image L into bucket L mod mesh, return [R, nbf, nbf, nK]) so one
pass per q serves every k' via an (Nk x R) phase GEMM; the dense-k AFT kernel is the oracle (like dense_aft.rs), and
production K needs complex RS-GDF: per q an aux FT at G+q, a Hermitian metric J2(q) (eig + lindep per q; only q = 0
carries Iteration 2's G = 0 bookkeeping), and SR 3c lattice sums weighted by e^{ik'.L}.
Tests to port: 1x1x1 == Gamma driver (1e-12); 1x1x3 k-mesh == explicit 1x1x3 supercell via the dense-AFT oracle at a
LOOSE gcut (exact at any gcut) + the three mutants; S(k) vs PySCF pbc_intor(kpts) (convention) and S(-k) = S(k)^*;
E_none - E_ewald = nocc v_M; PySCF pins (test_prototype.py KPT_REF_H2_112, and the tables above).
