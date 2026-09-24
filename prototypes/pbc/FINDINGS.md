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
