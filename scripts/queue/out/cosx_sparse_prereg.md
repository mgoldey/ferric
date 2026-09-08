# Pre-registration: shell-sparse COSX half transforms (F = D X, Ktilde += X G^T, S_num)

Date: 2026-09-08. Branch `feat/cosx-sparse-gemm` (from main a131b69f). Written
and committed BEFORE the implementation and BEFORE any number below was taken.

## Baseline being changed (measured, `cosx_scaling_results.md`, one thread, def2-SVP)

Full COSX K at C4..C20 has tail exponent (C12,C16,C20) **2.16** while its
A-build alone is **1.54**; the gap is the three DENSE block GEMMs per
1024-point block — `F = D X` (nbf x nbf x B), `Ktilde += X G^T` (same shape)
and, once per geometry, `S_num += X X^T` — which are 2% of the build at C4 and
**32% at C20 (54.6 of 168.5 s)**, with a GEMM tail exponent of 4.55 (the
measured dgemm rate also fell 19 -> 9 GFlop/s from nbf 298 to 490 on the
(nbf x 1024) planes, i.e. part of that 4.55 is a cache effect, not FLOPs).

## What is implemented (the standard COSX/sn-LinK structure)

Per 1024-point block, shell-granular so it composes with the kernel's
shell-pair loop:

1. **A** (active AOs) = shells `s` with `max_{mu in s, g in block} |X_{mu g}| >= eps_ao`,
   `X = sqrt(w) chi`. The Becke partition weight is INSIDE `X`, so a far
   radial shell of atom k whose points sit inside another atom's cell is
   damped by `sqrt(w_cell)` (~1e-4 at 20 Bohr, NOT exponentially small — the
   Becke step function is a polynomial), which is why outer-shell blocks stay
   partly delocalized (see the reachability expectation).
2. **Lambda** (output rows of F) = shells `l` with `max_{s in A} dmax[l][s] >= eps_d`,
   where `dmax[l][s] = max |D_{l s}|` over the shell block, computed ONCE per
   `build` (O(nbf^2), a single pass over D). `F[Lambda, blk] = D[Lambda, A] X[A, blk]`;
   rows outside Lambda are exactly zero, which the existing density-driven
   screen sees as `fmax = 0` (pairs with both shells outside Lambda are
   dropped for `t > 0`, kept for `t = 0`) — no second screen is added.
3. **B** (output columns of Ktilde) = union over the block's sub-batches of the
   shells touched by SURVIVING pairs (both `s1` and `s2`, because of the
   mirror fold). Rows of `G` outside B are never written (exactly zero).
   `Ktilde[A, B] += X[A, blk] G[B, blk]^T`, scattered into the full Ktilde.
   The cosx_a cross-check backend has no per-pair information, so B = all
   shells there (still correct, less sparse).
4. Fit: `S_num[A, A] += X[A, blk] X[A, blk]^T` (first build only, as before).
5. `build_from_occ(C)` on the sparse path forms `D_occ = C C^T` once per build
   (nbf^2 nocc, the SCF already forms D anyway) and uses (2): canonical MOs are
   delocalized so a C-based row restriction would keep every row, while
   `D = C C^T` is what decays. The dense path keeps its `C (C^T X)` transform
   unchanged.

The dense path stays selectable (`CosxHalfTransform::Dense`) and is the
cross-check; it must be BYTE-IDENTICAL to the pre-change builder. The comparison
rule is `>=` throughout, so `eps_ao = eps_d = 0` keeps every shell (including
exact zeros) and the sparse path degenerates to the same dgemm calls on
fresh standard-layout copies of the same operands — the vacuous-mask trivial
limit, expected BITWISE equal to dense.

REUSE CHECK (done before writing this): `ferric_dft::density_on_grid` /
`vxc.rs` / `ks.rs` / `gradient.rs` contain NO per-block active-AO screening —
`eval_density_closed` is a dense `d.dot(chi)` plus a full-`nbf` row reduction
and there is no AO-value cutoff anywhere in ferric-dft. Nothing to reuse or
mirror; the mask machinery is new and lives in `cosx_k.rs`.

## Thresholds (chosen from an error model, NOT tuned to any measurement)

`eps_ao = 1e-10`, `eps_d = 1e-10` (both on absolute matrix elements).

Error model: dropping X rows with `|X| < eps_ao` perturbs each `F` element by
`< eps_ao * ||D||_inf` (row 1-norm of D, ~10), each `G` element by that times
`||A^g||_inf` (~few), and each `Ktilde` element by `sum_g |X_{mu g}|` (<=
`sqrt(npts)` ~ 5e2 by Cauchy-Schwarz since `sum_g w chi^2 = 1`) times that:
a worst-case `~1e4 * eps_ao`. The density-driven screen's own bound
overshot reality by only ~10x on butane (t = 1e-7 -> 7.6e-7), so a 1e-6
target with `eps = 1e-10` has 1e2 of margin at the bound and ~1e3 in
practice. Because the AO reach depends on `sqrt(ln(1/eps)/alpha_min)`, going
from 1e-8 to 1e-10 costs only ~12% in reach (11.0 -> 12.4 Bohr for the most
diffuse def2-SVP exponent 0.15); the conservative value is nearly free.
Literature anchor: PySCF's `non0tab` AO mask defaults to 1e-15 on the AO
value (reach ~15 Bohr), i.e. the field runs even tighter than this.

If anchor (b) FAILS at these values the sparsification is over-aggressive and
the thresholds are tightened (the anchor decides, not the other way round).

## Expectations (physics hypothesis)

Onset geometry: the most diffuse def2-SVP exponent (0.15, H s / C p) gives an
AO reach of ~12 Bohr at 1e-10; the alkane density matrix decays over ~30 Bohr
(CLAUDE.md). So |A| is O(1) only once the molecule is >> 24 Bohr and |Lambda|
only once it is >> 60 Bohr. The C4..C20 series (7..50 Bohr) is therefore
PRE-asymptotic for Lambda everywhere and for A from ~C12 on.

* **Reachability counts (alkane_8 / def2-SVP, ~20 Bohr long, anchor (c)).**
  Inner-shell blocks (r < 3 Bohr) see every AO within ~12 Bohr — half to
  all of octane; outer-shell blocks (r > 8 Bohr) are spherical shells that
  intersect the chain in <= 2 places but their `sqrt(w)`-damped points near
  other atoms stay above 1e-10. Expected mean `|A|/nbf` **0.6 - 0.9** and
  mean `|Lambda|/nbf` **~1.0** (D has not decayed below 1e-10 anywhere on a
  20-Bohr molecule). The anchor's stated bars (`< 0.5` and `< 0.7`) are
  therefore expected to be MISSED at alkane_8 with a sound eps; per the brief
  the numbers are reported, eps is NOT tuned. The bars are reachable by
  construction on a water dimer 28 Bohr apart (each block's A is one
  monomer, D is block-diagonal to round-off): expected `|A|/nbf ~ 0.55-0.65`
  (0.5 for the ~84% of blocks inside r ~ 14 Bohr, 1.0 for the outer ones)
  and `|Lambda|/nbf` the same — that anchor is set at `< 0.75` for both,
  derived from this count, and is what the "skip the Lambda restriction"
  mutant must turn RED (it gives exactly 1.0).
* **Accuracy (anchor (b)).** `max|K_sparse - K_dense|` **< 1e-8** on
  water/cc-pVDZ and **< 1e-7** on butane/def2-SVP at production eps (bar 1e-6;
  grid errors 5.6e-5 / 2.9e-4); converged SCF `|dE| < 1e-9` Ha on water (bar
  1e-7).
* **Scaling (to be measured in a LATER pass — not this one).** With
  `|A| ~ 0.5 nbf`, `|Lambda| ~ 0.9 nbf`, `|B| <= nbf` at C20 the two
  per-iteration GEMMs drop to ~0.45x of dense and the smaller operands
  (X[A] ~ 2 MB instead of 4 MB planes) should recover part of the collapsed
  dgemm rate: GEMM seconds at C20 **54.6 -> 15-30 s**, GEMM fraction
  **32% -> 10-20%**, GEMM tail exponent **4.55 -> <= 2.5**, full-K tail
  exponent **2.16 -> 1.6-1.9** (bounded below by the A-build's 1.54 plus the
  still-dense O(N^2) terms: AO evaluation 8.3 s at C20 (5%), the per-block
  Lambda scan, the per-sub-batch `nbf x 256` fold buffer memset). The
  butane/def2-QZVP full K (137.04 s) should be UNCHANGED within noise: at
  7 Bohr every AO is active in every block and the GEMMs were 2% there.

## Artifact hypotheses (what "broken" looks like)

* Exponent unchanged (>= 2.1) AND `|A|/nbf ~ 1` at C20: the mask is not
  biting — eps applied to the wrong quantity (chi instead of X would keep
  MORE, not fewer; a weight-only quantity would keep the far shells), or the
  active set is computed over the wrong axis.
* Exponent unchanged with `|A|/nbf ~ 0.5`: the GEMMs were not the O(N^2)
  term after all (the 4.55 was entirely cache) — then the gather/scatter
  overhead is O(nbf x B) per block and dominates; look at the `gather_s`
  counter.
* Anchor (a) not bitwise at eps = 0: a `>` instead of `>=` somewhere (drops
  exact zeros — invisible on full-scale water, visible on the 1e-8-scaled
  density via the counters), or an operand handed to dgemm in a different
  layout than the dense path.
* Anchor (b) fails (> 1e-6): over-aggressive eps, OR a scatter-offset bug
  (that one is O(1), not 1e-6..1e-5 — the two are distinguishable by size).
* Too-clean check: `|A|/nbf` must VARY across blocks (inner vs outer radial
  shells); a constant fraction at every block is a fingerprint of a mask
  that depends on the shell list only.
