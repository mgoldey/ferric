# eps-linked (magnitude-based) refinements of the integral-direct LMP2 maps (2026-09-05, worktree eps-linked-maps, PROTOTYPE PHASE)

Task: can magnitude (eps-linked) rules BEAT the validated distance rules of
`ferric_mp2::lmp2_direct` (§27-§31) — smaller domains at matched energy
error, or less error at matched size? Python prototype FIRST
(`scripts/queue/proto_eps_linked_maps.py` + `_atom.py`, PySCF 2.13, reusing
`amplitude_lmp2_proto.py` machinery: Boys occ + VV-HV virtuals, same-kernel
RI A=(ia|P), V=(P|Q), 6-31G, mp2fit aux). Raw rows:
`scripts/queue/out/eps_linked_maps.txt` and `eps_linked_maps_atom.txt`
(promoted with `git add -f`).

## Anchors + mutations (all BEFORE any sweep; every arm SEEN to fail)

Per system/op: A1 dist(1e9) fit == global fit (max|dJ| 6.5e-14 water …
2.3e-12 C8); A2 mag(tau=0) same; A3 trivial candidate masks all-true
(masked E bit-identical); A4 closed-form pseudo-canonical eps=0 solve vs
masked-CG (independent algebra) |dE| ≤ 6.5e-15. Mutations: M1 drop-top-aux
moved max|dJ| to 8.6e-5…7.6e-4 (anchor FAILED as required); M2 zeroed-top-q
produced 7-35 escaping Eq-8 elements (conservativeness check FAILED as
required).

## C1: eps-linked i→aux fit domains — measured frontier

Rules (per-i, pair domain D_ij = D_i ∪ D_j, same domain-local same-kernel
fit as lmp2_direct stage 5): `dist` = radius r on Boys centroids (the
production rule); `mag` = keep P with s_i(P) = max_a|(P|ia)| ≥ tau (the
unwhitened strip magnitude, available in Rust AFTER stage 3);
`magm` = s_i(P)/√V_PP; `tail` = §26-style L1 tail (drop smallest-s while
Σ ≤ budget). Error metric: FULL (eps=0) MP2 energy of the domain-fitted J
minus global-fit J — map error in isolation. dE in Ha; pairdom = mean over
unique pairs; locmax/loc95 = max/p95 distance (Bohr) of selected aux
functions from their centroid.

alkane_8 / coulomb (naux=700), matched-size brackets:

| dist r | pairdom | dE       | nearest magnitude row | pairdom | dE       |
|--------|---------|----------|-----------------------|---------|----------|
| 5      | 473.6   | +2.04e-5 | mag 3e-3              | 503.8   | +5.79e-6 |
| 8      | 574.9   | +2.72e-6 | tail 0.1              | 559.9   | +2.74e-6 |
| 10     | 644.7   | +3.81e-7 | mag 3e-4 / tail 1e-2  | 647.4 / 650.8 | +2.80e-7 / +2.18e-7 |
| 12     | 664.2   | +1.49e-7 | tail 3e-3             | 672.1   | +7.97e-8 |

alkane_8 / erfc(1.0):

| dist r | pairdom | dE       | nearest magnitude row | pairdom | dE       |
|--------|---------|----------|-----------------------|---------|----------|
| 5      | 473.6   | +6.01e-6 | tail 0.1              | 456.6   | +4.02e-6 |
| 8      | 574.9   | +9.63e-7 | mag 3e-4              | 569.6   | +7.80e-7 |
| 10     | 644.7   | +1.20e-7 | mag 1e-4              | 632.8   | +1.72e-7 (dist WINS) |
| 12     | 664.2   | +4.35e-8 | tail 1e-3             | 671.2   | +2.83e-8 |

alkane_4 showed larger magnitude wins (up to ~25x error at matched size,
coulomb small-domain end); at C8 the win is ~1.2-3.5x with a crossover
(erfc r=10). The frontiers CONVERGE with system size — consistent with the
magnitudes of a same-kernel local fit being distance-driven past the onset,
not with a construction artifact (error is monotone in every knob, zero
V_DD solve failures anywhere, trivial limits exact).

Structural negative: magnitude-selected domains are broader-but-sparser —
locmax 19.3 Bohr (the whole C8 length) vs 9.9 at dist r=10, p95 13.5 vs
9.2. Per-i magnitude sets are all DISTINCT, so the §31 pair-domain sharing
(6-9x pairs per distinct domain, the grouped d³ Cholesky hoist) is
destroyed at function granularity. Atom-granular re-measurement + stage-5
flop model: see below.

## C2: bound-based pair virtual candidates — measured frontier

Rules (single pair set C_ij, both Eq-8 orientations): `rv` = distance ball
r_v on dipole centroids, C_ij = V_i ∪ V_j (the production rule); `kap` =
Schwarz-type screen a ∈ C_ij iff q_ia·qmax_j ≥ κ·eps or q_ja·qmax_i ≥
κ·eps with q_ia = √((ia|ia)_fit); `kapl` = same with STRIP-LOCAL q (V_DD
restricted to the r=10 distance domain — the Rust-implementable variant;
max dev vs global q ≤ 4e-13 at C4, rows indistinguishable at C8). Error
metric: masked-CG energy with candidates ANDed into the Eq-8 mask, minus
the eps-only masked energy, both on the SAME global-fit J. nv=75 at C8.

alkane_8 / coulomb:

| eps  | rule    | mean C / max | dE       | escapes | worst esc |
|------|---------|--------------|----------|---------|-----------|
| 1e-3 | rv 8    | 61.7/75      | +2.53e-5 | 44      | 1.4e-3    |
| 1e-3 | rv 10   | 68.7/75      | 0 (exact)| 0       | —         |
| 1e-3 | kap 1   | 65.9/75      | 0 (exact)| 0       | —         |
| 1e-3 | kap 3   | 58.7/75      | +1.84e-6 | 4       | 1.3e-3    |
| 1e-4 | rv 12   | 70.7/75      | +5.01e-6 | 808     | 4.0e-4    |
| 1e-4 | kap 1   | 74.2/75      | 0 (exact)| 0       | —         |
| 1e-4 | kap 3   | 71.9/75      | +2.09e-8 | 10      | 1.3e-4    |

alkane_8 / erfc(1.0):

| eps  | rule    | mean C / max | dE       | escapes |
|------|---------|--------------|----------|---------|
| 1e-3 | rv 8    | 61.7/75      | 0 (exact)| 0       |
| 1e-3 | rv 12 (prod) | 70.7/75 | 0 (exact)| 0       |
| 1e-3 | kap 1   | 61.2/75      | 0 (exact)| 0       |
| 1e-3 | kap 3   | 42.3/75      | +2.90e-6 | 12      |
| 1e-4 | rv 12   | 70.7/75      | +3.49e-7 | 110     |
| 1e-4 | kap 1   | 73.1/75      | 0 (exact)| 0       |
| 1e-4 | kap 3   | 68.5/75      | +1.21e-7 | 34      |

Findings:
- κ=1 is EXACTLY conservative on the global-fit J at every system/op/eps
  (0 escapes, dE bit-zero). Not a coincidence: the global fit J = ÃᵀÃ is a
  Gram matrix, so |J_iajb| ≤ q_ia·q_jb is Cauchy-Schwarz, exact — same
  theorem the whitened path's §23 screen uses. NOT too-clean; structural.
- At matched size the Schwarz rule error is 14-200x (coulomb) / ~30x
  (erfc) below the distance ball's; at matched exactness it is never
  larger and up to 40% smaller (erfc/1e-3: 61.2 vs 70.7 production).
- The distance rule's zero-error radius is eps-DEPENDENT (rv=10 exact at
  coulomb/1e-3 but leaks +2.3e-5 at 1e-4 with worst escape 6.3e-4 > eps);
  κ auto-tunes with eps by construction. At tight eps the screen honestly
  refuses to trim (74.2/75 at coulomb/1e-4) — fewer virtuals ARE
  droppable there.
- CAVEAT measured before porting: the direct path's J is the DOMAIN-fitted
  J (per-pair D_ij), not a global Gram — the exactness theorem does not
  transfer (§23 said so). Empirical bound-slack vs J_dom(r_aux=10): see
  c2dom rows below.

## Atom-granular pass + stage-5 cost model (C1 confirmation)

Rust's distance rule is atom-quantized (shell membership by shell-ATOM
distance), which is what creates the §31 pair-domain sharing the grouped
d³ Cholesky exploits. `proto_eps_linked_maps_atom.py` re-measures both
rules at atom granularity (magnitude score s_i(A) = max_{P∈A} s_i(P)) with
the stage-5 flop model chol = Σ_distinct d³/3, gemm = Σ_pairs (2d²nv +
2dnv²). C8, ungated, selected rows:

| op   | rule       | pairdom | dE       | distinct/325 | tot GF |
|------|------------|---------|----------|--------------|--------|
| coul | dist r=10  | 644.7   | +3.81e-7 | 11           | 23.6   |
| coul | dist r=12  | 664.2   | +1.49e-7 | 15           | 25.2   |
| coul | magt 0.03  | 680.9   | +1.21e-7 | 9            | 25.9   |
| coul | magt ≤0.01 | 700.0 (COLLAPSE: keeps everything) | 7.6e-12 | 1 | 26.6 |
| coul | tail 0.3   | 630.5   | +8.06e-7 | 33           | 24.4   |
| coul | tail 0.1   | 676.7   | +7.55e-8 | 15           | 26.2   |
| erfc | dist r=10  | 644.7   | +1.20e-7 | 11           | 23.6   |
| erfc | magt 1e-3  | 623.2   | +2.81e-7 | 38           | 24.0   |
| erfc | tail 3e-3  | 627.0   | +3.38e-7 | 44           | 24.9   |
| erfc | tail 1e-3  | 656.8   | +1.27e-7 | 35           | 26.1   |

- COULOMB THRESHOLD COLLAPSE: per-atom max |(P|ia)| does not discriminate
  under Coulomb — every atom's max stays ≥1e-2 across the whole C8 chain
  (the Schwarz-bound distance-independence of the Coulomb kernel, seen as
  a map). The threshold rule cannot shrink Coulomb domains at all below
  knob 0.03.
- At atom granularity the magnitude frontier COINCIDES with or sits
  OUTSIDE the distance frontier (erfc tail 1e-3: 656.8 @ 1.27e-7 vs dist
  r=10: 644.7 @ 1.20e-7 — distance strictly wins; coulomb similar), with
  2-4x WORSE domain sharing and no total-flop advantage at matched error.
- The function-granularity wins (C4 up to ~25x, C8 down to ~1.2-3.5x)
  therefore do not survive the granularity Rust needs; they also shrink
  with system size, i.e. past the onset the strip magnitudes are
  distance-driven.

## C2 on the DOMAIN-fitted J (direct-path object)

The global-Gram Cauchy-Schwarz theorem does not transfer to the per-pair
domain fit (§23). Empirical bound slack, C8, J_dom at r_aux=10, q from the
strip-local fit: escapes = 0 at κ=0.5 AND κ=1.0, both operators, both
eps ∈ {1e-3, 1e-4}. The escape counter is a reachable fail (M2 mutation
produced 29-35 escapes through the same code path).

## Verdicts (2026-09-05, prototype phase; provisional as always)

- C1 eps-linked aux fit domains: **NO-GO.** At the atom granularity the
  Rust maps use, the magnitude frontier does not beat the distance
  frontier (coincides or loses; Coulomb thresholds collapse outright),
  while costing 2-4x of the pair-domain sharing that §31's grouped
  Cholesky exploits. The C4-scale wins fade monotonically with system
  size — a distance-driven-magnitudes signature, not an artifact (anchors
  exact, error monotone in every knob, zero V_DD failures, locality
  witness bounded). Artifact-hypothesis check: none of the pre-registered
  Y-fingerprints appeared; the measured outcome is the pre-registered
  honest-NO-GO branch of X ("frontiers coincide"), with the twist that
  magnitude picks are broader-but-sparser (locmax 19.3 Bohr), not
  distance balls in disguise — either way, no exploitable win. Do not
  re-propose without a qualitatively different score (e.g. a whitened /
  metric-contracted quantity with a proven Gram bound, which is the §26
  whitened-path construct that does NOT apply to the unwhitened fit).
- C2 Schwarz virtual candidates: **GO.** κ=1 is exactly conservative on
  the global-fit J (Gram/Cauchy-Schwarz, the §23 theorem) and empirically
  escape-free against the domain-fitted J at κ≤1; at matched candidate
  size it beats the distance ball by 14-200x in error; at matched
  exactness it is up to 40% smaller (erfc/1e-3: 42.3 vs production
  rv=12's 70.7 at +2.9e-6, 1600x below the eps-truncation error); and it
  is eps-LINKED where the distance rule's zero-error radius silently
  depends on eps (rv=12 leaks 3.5e-7..5.0e-6 at eps=1e-4 with escapes up
  to 6.3e-4 > eps). Strip-local q ≡ global q (dev ≤ 4e-13), so the rule
  is implementable AFTER strips are built, refining C_ij = V_i ∪ V_j
  before the pair fit. Rust port: `DirectConfig.virt_schwarz_kappa:
  Option<f64>` (None = off = trivial limit), q_ia from the per-i
  D_i-local metric Cholesky over strip columns, screen threshold
  κ·eps/qmax with the scale folded like the whitened path's √scale.

