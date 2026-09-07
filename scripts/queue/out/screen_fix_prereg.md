# COSX shell-pair screen fix — sparsity re-measurement pre-registration

Date: 2026-09-07. Branch `feat/cosx-screen-fix` (off `feat/3c1e-md-kernel` 7f4e51cb).
Written and committed BEFORE any kept-pair number with the new bound was
produced (the new bound module compiles; its anchors have been run only
against the OLD bound, where they are RED as designed).

## What was wrong

`PairBounds` bounded a shell pair by `sqrt(max|S_block|) / d` from the SIGNED
overlap block. Same-centre s-p / s-d / p-d overlaps vanish by symmetry, so
those pairs were dropped while `|A^g|` was up to 0.233 — water/cc-pVDZ
(50,110), t=1e-7: `max|K_scr − K_unscr| = 0.71` (grid error 4.8e-5). My own
independent K harness (hcore-guess density, ferric-integrals only) reproduces
it as 1.06 with the old bound; the trivial-limit test also catches the old
bound at t=1e-300 (21/1485 pairs dropped because their magnitude is exactly 0).
Every "screened" number in the COSX record (Stage 2 kept fractions 0.96→0.57
DZ; L-axis scr columns) is therefore invalid. Unscreened numbers stand.

## The replacement bound (theory only at this point)

Hölder on the primitive expansion, per primitive pair `(a@A, b@B)`,
`p=a+b`, `P` the product centre, `R=|P−r_g|`, `K_AB = exp(−ab/p |A−B|²)`:

    |A^g|_block ≤ Σ_ij |c_i c_j| K_ij s_a s_b [ (2π/p) q_0 F0⁺(pR²) + (4π/p) Σ_{k≥1} q_k M_k F0⁺(pR²/2) ]

with `q_k` the coefficients of `(ρ+d_A)^{l_a}(ρ+d_B)^{l_b}`, `M_k=(k/(pe))^{k/2}`,
`F0⁺(T)=min(1, ½√(π/T))`, `s` the max |column sum| of cart→sph. Every step is
a pointwise inequality. It decays as `K_AB` in pair separation and `1/R` in
grid distance. The diagonal Cauchy-Schwarz alternative was REJECTED on paper:
`sqrt(A_μμ A_νν)` carries no `K_AB`, so it is `~1/√(R_A R_B) ≫ 1e-7` at every
grid point — valid but vacuous.

## Consequence for sparsity, reasoned before measuring

The `1/R` factor never reaches 1e-7 inside a Becke grid (that needs R≈1e7
Bohr), so ALL of the screen's bite comes from `K_AB`. The most diffuse def2-SVP
s primitives (C ≈0.15, H ≈0.12 Bohr⁻²) give `ab/p ≈ 0.07` and a prefactor
`|cc|(2π/p) ≈ 0.6`, so a pair survives at 1e-7 while `exp(−0.07 R_AB²) ≳ 1.7e-7`,
i.e. `R_AB ≲ 15 Bohr` (≈16 Bohr at 1e-8). Along an alkane chain (≈2.5 Bohr per
carbon projected) that is ≈6 carbons each way.

The OLD bound's pair-separation decay was `sqrt(K_AB)` (sqrt of the overlap),
i.e. SLOWER than the truth; the new bound decays at the correct `K_AB` rate but
with a larger polynomial prefactor. So the new screen can drop MORE distant
pairs than the old one while dropping NONE of the same-centre pairs (which are
O(nsh) of O(nsh²) — negligible in the tail).

### Measurement

alkane_4/8/12/16/20 at def2-SVP (def2-TZVP if time), ≥ 2000 points sampled
(fixed-seed LCG 20260907) from the (50,110) grid positions, t = 1e-7 and 1e-8.
Report per molecule: nshells, total pairs, mean kept COUNT per point, kept
fraction, and the per-point min/max. Counts only — load-independent — via
`PairBounds::estimate`, no A-build. Fit the TAIL (C12, C16, C20) of
`log(kept count)` vs `log(natoms)`.

### Pre-registered expectations (t = 1e-7, def2-SVP)

| system | kept fraction expected | note |
|---|---|---|
| C4  | ≥ 0.97 | chain ≈ 10 Bohr < 15: nothing separable |
| C8  | 0.85–0.95 | chain ≈ 20 Bohr, onset |
| C12 | 0.65–0.85 | |
| C16 | 0.55–0.75 | |
| C20 | 0.45–0.65 | chain ≈ 48 Bohr ≈ 3× the 15 Bohr radius |

Kept-COUNT tail exponent (C12→C20): **1.3–1.7** — past onset but well
short of the asymptotic 1.0 because 15 Bohr is still a third of the C20
chain. At 1e-8 every fraction is a few points higher, exponent slightly
larger.

**"Sparsity survives"** = kept fraction at C20 ≤ 0.70 AND tail exponent < 1.8.
**"Sparsity does NOT survive a valid screen"** = C20 kept fraction ≥ 0.85 OR
tail exponent ≥ 1.9 (count grows like the pair count). Either result is
reported as measured; the invalid Stage 2 numbers (0.96→0.57) are quoted next
to it for the record, not as a target.

### Artifact hypotheses

* If kept fractions are IDENTICAL to Stage 2's at every size, the new bound
  is not actually in the loop (wiring bug) — the same-centre pairs alone
  must raise C4's fraction.
* If the fraction at C4 is < 0.95, the bound is dropping something on a
  molecule where nothing is separable — suspect an underestimate that the
  anchor's probe set missed; STOP and extend the anchor before quoting.
* If the count exponent is < 1.0 in the tail, the sample's points are biased
  (e.g. all far from the chain); check the per-point min/max spread.

### Screen cost

Per point: `estimate` over all pairs vs `Md3c1e::for_each_pair` per point at
B=256, unscreened, on alkane_4 and alkane_8 / def2-SVP. Expected: the screen
is 1–5 % of the A-build per point (one sqrt + one div per primitive pair vs
hundreds of flops per primitive pair per point in the kernel).
