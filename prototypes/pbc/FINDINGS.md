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
