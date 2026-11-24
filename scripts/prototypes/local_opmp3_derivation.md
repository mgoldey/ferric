# Local OP-MP3 with the WSHG23 amplitude threshold: working equations

Prototype: `local_opmp3_proto.py`. Canonical OP-MP3: `opmp3_derivation.md`
and `crates/ferric-mp2/theory.md` §6.1–6.2. Local MP2: §12 there and
`scripts/amplitude_lmp2_proto.py`.

## 1. Closed-shell equations

RHF reference, frozen core, spatial orbitals. The amplitude is the αβ block,
stored `t[i,a,j,b]` = $c^{\alpha\beta}_{iJaB}$; the same-spin block is
$t - t^{s}$ with $t^{s}_{iajb} = t_{ibja}$. With $J_{iajb} = (ia|jb)$ and the
Fock superoperator

$$ (\mathcal A t)_{iajb} = \sum_c F_{ac} t_{icjb} + \sum_c t_{iajc} F_{cb} - \sum_k F_{ik} t_{kajb} - \sum_k t_{iakb} F_{kj}, $$

the first- and second-order amplitudes solve

$$ \mathcal A c^{(1)} = -J, \qquad \mathcal A c^{(2)} = -R(c^{(1)}), $$

$R$ the αβ MP3 residual of theory.md §6.1 with α = β (two ladders, the
same-spin and mixed rings, and the two αβ exchange rings; `residual`). In
canonical orbitals $\mathcal A$ is diagonal ($\Delta$) and both are explicit
divisions. The energies are $E^{(n)} = \sum (2c^{(n-1)} - c^{(n-1)s})\cdot J$
and the α-block W matrices (the β block is equal) are

$$ W^{OS}_{ij} = \tfrac14\sum_{kab} c_{iakb} J_{jakb} + (i\leftrightarrow j), \qquad
W^{SS}_{ij} = \tfrac14\sum_{kab} (c - c^{s})_{iakb} J_{jakb} + (i\leftrightarrow j). $$

Each model is twice its α trace: OP-MP3 $= 2\operatorname{Tr}P(W^{(2)}, W^{(3)})$,
SOP-MP3 $= 2[\operatorname{Tr}P(W^{(2)SS}, W^{(3)SS}) + \operatorname{Tr}P(W^{(2)OS}, W^{(3)OS})]$,
SSD-OPMP3 the same with $x_{ss}W^{(3)SS}$ ($x_{ss} = 0$ for an HF reference,
ferric's `PAPER_X_SS`; 0.5 is also reported). These formulas equal ferric's
spin-blocked `run_opmp3` to 1e-14 Ha when both are fed ferric's orbitals and
integrals (`ferric_cross_check`).

## 2. Is "local OP-MP3" well defined? (occupied-rotation invariance)

Let the localized active occupied orbitals be $C_{loc} = C_{can}U$, $U$
orthogonal (Boys acts inside the active block; the frozen core is untouched),
and the local virtuals $C^v_{loc} = C^v_{can}V$. The amplitude equations are
covariant: if $c$ solves them in canonical orbitals, then $c$ transformed by
$U$ on both occupied and $V$ on both virtual indices solves them in local
orbitals, because $\mathcal A$, $J$ and $R$ all transform as tensors and the
full Fock matrix (off-diagonal blocks included) is kept. So, with no
truncation,

$$ W_{loc} = U^{\mathsf T} W_{can} U \quad\text{for each order and channel,} $$

since $W$ sums the virtual indices and $k$ (invariant) and leaves $i, j$ as
free indices. Then
$P(U^{\mathsf T}AU, U^{\mathsf T}BU) = U^{\mathsf T}AU\,(U^{\mathsf T}(A-B)U)^{-1}U^{\mathsf T}AU = U^{\mathsf T}P(A,B)U$,
and its trace is invariant. The SS/OS split is by the spin of the summed $k$,
which a spatial rotation does not change, and the frozen core is excluded
from both $U$ and the W matrices, so neither the channel split nor the frozen
core breaks this. The prototype verifies it two ways: a random orthogonal
rotation of the canonical W set changes no model by more than 1e-16 Ha
(`anchor_invariance`), and the untruncated local W rotated back equals the
canonical W elementwise (§4).

Consequence: local OP-MP3 is "OP-MP3 evaluated on locally truncated
first- and second-order amplitudes"; no extra choice of basis enters at
$\varepsilon = 0$.

## 3. Truncation (the consistent choice)

Mask: WSHG23 Eq. 8 on the local integrals, keep $(i,a,j,b)$ iff
$|(ia|jb)| > \varepsilon$ or $|(ib|ja)| > \varepsilon$ (as in the LMP2 port).
It is closed under $a\leftrightarrow b$ and under $(ia)\leftrightarrow(jb)$,
so $c - c^{s}$ stays inside it.

1. $c^{(1)}$ solves $P\mathcal A P c^{(1)} = -PJ$ (the LMP2 amplitudes).
2. $R(c^{(1)})$ is evaluated exactly from the masked $c^{(1)}$, with
   untruncated integrals.
3. $c^{(2)}$ solves $P\mathcal A P c^{(2)} = -P R(c^{(1)})$ on the SAME mask.
4. $W^{(2)}, W^{(3)}$ and the models from these amplitudes.

A dropped amplitude is zero at both orders. In the ring terms of another
pair, e.g. $X_{iajb} = \sum_{kc} M_{kc,jb}\,c_{iakc}$, a dropped
$(i,a,k,c)$ (in particular every element of a dropped pair $(ik)$)
contributes nothing to $R_{iajb}$; the integrals $M$ are not truncated. The
same holds for the hole ladder ($c_{kalb}$ of dropped $(kl)$) and the
exchange rings. Pair $(ij)$'s residual is then needed only on its own
retained pattern (step 3 projects it), so the particle ladder is a
domain-local contraction:
$R_{ij}[a,b] = \sum_{(c,d)\in S_{ij}} (ac|bd)\,c_{ij}[c,d]$ for
$(a,b)\in S_{ij}$. The dense masked evaluation in the prototype is
numerically identical to that domain evaluation; the cost proxies count the
domain work (§6).

Why this choice: with one mask $P$ for both orders, the Wigner 2n+1
identity survives truncation,

$$ E^{(3)} = \sum(2c^{(2)} - c^{(2)s})\cdot J = \sum (2c^{(1)} - c^{(1)s})\cdot R(c^{(1)}), $$

because $c^{(2)}\cdot PJ = -c^{(2)}\cdot P\mathcal A c^{(1)} = -c^{(1)}\cdot P\mathcal A c^{(2)} = c^{(1)}\cdot PR$.
The prototype checks it at every $\varepsilon$ against a bound derived from
the two CG residuals, $|{\rm gap}| \le 3(\|c^{(1)}\|\|r_2\| + \|c^{(2)}\|\|r_1\|)$.
A $c^{(2)}$ mask different from the $c^{(1)}$ mask breaks it (mutation
`c2_mask_shift`). The diagnostic mode `full` solves $c^{(2)}$ unmasked; it
isolates the first-order truncation and breaks the identity by design.

Rescue variants named in advance (not run unless the decision needs them):
a tighter threshold for $c^{(2)}$ than for $c^{(1)}$; filling dropped
$c^{(1)}$ elements with the diagonal estimate $-J/D$ for the ring terms only.

### 3.1 Density-fitted ladder (per pair, inside the domain)

With same-metric factors $B^P_{pq} = \sum_Q (L^{-1})_{PQ}(Q|pq)$, $V = LL^{\mathsf T}$,
every integral is $\sum_P B^P_{pq}B^P_{rs}$. For pair $(ij)$ with retained
pattern $S_{ij}\subset D^a_{ij}\times D^b_{ij}$, $c_{ij}$ vanishes outside the
domain block, so on that block

$$ R^{\rm lad}_{ij} = \sum_P B^P[D^a,D^a]\;c_{ij}[D^a,D^b]\;B^P[D^b,D^b]^{\mathsf T}, $$

exact, at cost $n_{aux}(|D^a|^2|D^b| + |D^a||D^b|^2)$ per pair (`df_ladder`;
$R_{ji} = R_{ij}^{\mathsf T}$). This is the contraction an integral-direct
port would run. Canonical and local $B$ are both transforms of one AO
3-index tensor, so the $\varepsilon=0$ anchor stays exact under DF.

Checks specific to the DF route:

- $\varepsilon=0$: as in §4, with the canonical ladder formed as the dense
  $(ac|bd) = B\cdot B$ when it fits (an algebra independent of `df_ladder`).
- Finite $\varepsilon$ (`--df-check`): on the mask, `df_ladder` equals the
  dense $B\cdot B$ ladder applied to the same masked $c^{(1)}$, bar
  $10^{-11}|R|$. The vacuous-mask anchor cannot see a domain mix-up
  (every domain is full); this check can. Mutation `pair_domain_ii` reads
  pair $(ij)$'s domain from pair $(ii)$ and must fail it. On propane
  STO-3G every domain is full, so the mutation could not fire there; the
  check runs on butane 6-31G at $\varepsilon=10^{-3}$, where domains are
  about half the virtual space.
- DF vs exact (`--compare-exact`): the RI error of each canonical model,
  and the difference of the truncation errors between the two routes, are
  reported separately.

## 4. Anchors (run before any sweep)

- $\varepsilon = 0$ (mask all true): local W rotated to the canonical frame
  equals the canonical W elementwise, eigenvalues of $W^{(2)}, W^{(3)}$ agree,
  and every model energy agrees, all to ≤ 1e-10 Ha. Measured: ≤ 2e-16.
  This needs the local zeroth-order Fock to be the one the canonical
  reference uses, $F = SC\,{\rm diag}(\varepsilon)C^{\mathsf T}S$; PySCF's
  `get_fock()` from the final density differs by the SCF residual and moved
  the anchor to 7e-11.
- Mutation `semicanonical` (drop the off-diagonal Fock couplings in the local
  solve): the anchor fails at 5e-3.
- The canonical formulas vs ferric `run_opmp3` (exact integrals, same frozen
  core, ferric's orbitals and integrals into this file's formulas): ≤ 1e-14
  Ha on every model and W eigenvalue. On PySCF's orbitals the difference is
  1e-9 Ha, the PySCF-vs-ferric SCF difference (E_HF differs by 3e-9).
  Mutation: dropping residual term 7 must fail this check. The
  local-vs-canonical anchor cannot see a residual error (both sides share
  `residual`); this cross-check is the one that can.
- Finite $\varepsilon$: $E^{(2)}$ of the masked $c^{(1)}$ equals the
  `amplitude_lmp2_proto.py` masked solver's energy (independent code) to
  ≤ 1e-9 Ha; the E3 identity of §3 holds within its CG bound.

## 5. Hypotheses (written before the sweeps)

Written 2026-11-16 after smoke runs on propane and butane STO-3G only.

Padé linearization. With $M = A - B$, $P = AM^{-1}A$:

$$ \delta\operatorname{Tr}P = \operatorname{Tr}[G_A\,\delta A] + \operatorname{Tr}[G_B\,\delta B], \quad
G_A = M^{-1}A + AM^{-1} - M^{-1}A^2M^{-1}, \quad G_B = M^{-1}A^2M^{-1}. $$

On an eigen-channel with $\zeta = b/a$ the gains are
$g_A = (1-2\zeta)/(1-\zeta)^2$ and $g_B = 1/(1-\zeta)^2$; for MP3 both are 1.

- **Physics hypothesis.** For gapped, closed-shell systems ($\zeta_{max}\lesssim 0.3$)
  the OP-type errors are the W errors weighted by gains near 1:
  $|\delta E_{OP}| \lesssim 2(|\delta E^{(2)}| + |\delta E^{(3)}|)$, smooth and
  one-signed in $\varepsilon$ like the MP3 error, and the linearized
  prediction matches the measured OP error.
- **Padé amplification (a real property of the method, to be measured).**
  Where $\zeta_{max}\to 1$ (stretched bond), $g_B$ grows as $(1-\zeta)^{-2}$ and
  $g_A$ changes sign; truncation errors in $W^{(3)}$ along that channel are
  amplified. Fingerprint: amplification grows with $\zeta_{max}$ and the
  linearized prediction still matches the measurement.
- **Artifact: mask mixing two index sets** (CLAUDE.md worked example; e.g. a
  ring term reading pair $(ij)$'s mask for amplitude $(ik)$, or a
  canonical-index mask on local amplitudes). Fingerprint: the $\varepsilon = 0$
  anchor still passes (mask all true) but the finite-$\varepsilon$ error does not
  track the LMP2 prototype's $E^{(2)}$, or violates the E3 identity, or does
  not fall smoothly with $\varepsilon$.
- **Artifact: inconsistent orders** ($c^{(2)}$ on another mask, or W built from
  amplitudes of different solves). Fingerprint: the E3 identity fails.
- **Redesign forced by these.** The amplification ratio alone cannot detect a
  W-construction bug: a wrong $\delta W$ enters MP3 and OP-MP3 alike, so the
  ratio stays near 1 either way. That is why the finite-$\varepsilon$ checks of
  §4 (independent LMP2 energy, E3 identity) are part of every sweep, and why
  the linearized prediction is reported (it tests the Padé step on its own).

### 5.1 Size-consistency mismatch: hypotheses (written 2026-11-17, before the rerun)

Measured so far: butane x2 at 100 Bohr (STO-3G and 6-31G). The gap
E(dimer) - 2E(mono) is below 1e-9 Ha at eps = 0, but at finite eps it is of
the order of the truncation error and changes sign with eps. The separability
diagnostic shows occupied local orbitals separate exactly (overlap 1 - 1e-13),
while each copy's local virtuals sit wholly on that copy but match the
monomer's only to 0.707 = 1/sqrt(2), not one-to-one. STO-3G (no hard
virtuals) shows it too, so it comes from the Boys localization of the
projected (VV) virtuals. The eps-mask lives in that basis, so the two runs
keep different amplitude sets.

- **(A) Saddle point or local minimum.** One of the two runs stops at a
  stationary point that is not the minimum. Expect: a different total Boys
  spread for the same virtual space (subspace deviation ~0), a negative or
  clearly nonzero Boys Hessian along the mismatched rotation in the run with
  the higher spread, and monomer runs from random starts that reach the lower
  value. Fix: a better optimizer or start point.
- **(B) Degenerate minima.** Both runs reach the same spread to optimizer
  tolerance and the 1/sqrt(2) overlaps are a 45 degree rotation inside a pair
  whose Boys objective is flat along that rotation (e.g. symmetry-equivalent
  C-H sigma* combinations). Expect: equal spreads, a Boys Hessian along the
  rotation ~0, the two orbitals related by the molecule's local symmetry
  (equal individual spreads, mirror-image centroids), and monomer runs from
  random starts that land on different rotations with the same spread.
  Fix: a deterministic tie-break, or a mask that is invariant inside
  near-degenerate blocks (keep the block if any element passes).

Either way the defect sits in the VV-HV construction that ferric's production
amplitude-threshold LMP2 also uses, so the follow-up applies there too. This
lane does not change production code.

## 6. Measurements and the decision rule (stated before the data)

Per system and $\varepsilon$: errors of $E^{(2)}$, $E^{(3)}$, MP3, OP-MP3,
SOP-MP3, SSD-OPMP3 (absolute, and relative to $E^{(2)}$); kept amplitude and
pair fractions; the eigenvalue of $A-B$ nearest zero and $\zeta_{max}$,
canonical vs truncated; the linearized/measured OP ratio; cost proxies:
ladder $\sum_{ij}(|D^a_{ij}||D^b_{ij}|)^2 / (o^2v^4)$ and ring
$\sum_{ij,ik}|D^a_{ij}||D^b_{ij}||D^b_{ik}| / (o^3v^3)$ over retained pairs.

Amplification: $a = |\delta E_{OP}| / (|\delta E^{(2)}| + |\delta E^{(3)}|)$. The raw
ratio $\delta E_{OP}/\delta E_{MP3}$ is also reported, but $\delta E^{(2)}$ and
$\delta E^{(3)}$ can cancel in $\delta E_{MP3}$ (propane STO-3G at 1e-5:
+7.4e-7 and −4.5e-7), which makes the raw ratio meaningless there; this is
why $a$ uses the no-cancellation scale. (This refinement was made after the
propane smoke run.)

Decision rule:

- **GO (recommend a Rust port)** if, for the gapped systems at every
  $\varepsilon\in[10^{-6},10^{-4}]$, $a \le 3$ for OP-MP3, SOP-MP3 and SSD-OPMP3,
  $\delta E_{OP}$ is one-signed and falls monotonically with $\varepsilon$, the
  linearized prediction matches the measured OP error within 10%, and no
  truncated $A-B$ changes sign of the eigenvalue nearest zero. Measured past
  the onset (n-alkanes ≥ C20) at least at one basis.
- **NO-GO for the stretched case only** if $a > 3$ there and the
  linearization explains it; the recommendation is then conditional on
  $\zeta_{max}$, with the rescue variants of §3.
- **NO-GO** if the gapped systems fail, with the same evidence bar: anchors
  passing, the pass condition reachable (it is: the smoke runs met it), the
  artifact fingerprints above ruled out.

## 7. v2 (2026-11-18): the fixed LMP2 maps

Since the tables above, ferric's amplitude-threshold LMP2 gained (book
section 13.1): the Fock-propagated screen, on by default with κ = 1 on the
integral-direct path; λ-Boys occupied orbitals with λ = 10 (both paths);
integral-direct map defaults virt_radius 20 Bohr / ao_tail 1e-4; and a
memory-bounded direct solve. Only the first two change the amplitude
pattern this dense oracle can model. The AO/aux maps are integral-direct
assembly approximations that the oracle does not have (it holds the full
MO integrals); they are measured at first order through ferric's own
`run_rimp2` (`--rust-lmp2 AUX --rust-maps old,old-direct,new`).

Arms (`--arms`, canonical W computed once and shared):

| arm | mask | occupied orbitals |
|---|---|---|
| `old` | Eq 8 | pyscf.lo.Boys as returned (the 2026-11-16/17 tables) |
| `old-stable` | Eq 8 | converged Boys (`boys_stabilize`, occ and VV) |
| `screen` | Eq 8 ∪ screen κ = 1 | converged Boys |
| `lambda` | Eq 8 | converged Boys, then λ-Boys λ = 10 |
| `new` | Eq 8 ∪ screen κ = 1 | converged Boys, then λ-Boys λ = 10 |

Differences from production: the screen's $F'$ here is the whole
off-diagonal virtual block, production's is restricted to the pair's
candidate square $C_{ij}$ (a subset, so the oracle's pattern is a superset;
for molecules shorter than virt_radius the two coincide). λ-Boys is a port
of `boys.rs::lambda_boys_rotation` (same angle, pair order, stop rule
1e-8), started from converged Boys as production does.

Anchors for v2, before any sweep: ε = 0 with the `new` arm (λ-Boys
orbitals, screen inert at ε = 0) reproduces canonical OP-MP3 ≤ 1e-10 Ha;
λ-Boys converges and raises G; a sign-flipped λ-Boys step lowers G and does
not converge (mutation).

Hypotheses (written before the sweep):

- **Physics (carry-over).** The earlier tables found the Padé a linear
  passenger: pred/OP = 1.000 and a ≈ 0.3–1.2. Then any reduction of the
  first-order truncation error carries over, dOP_new/dOP_old ≈
  dMP3_new/dMP3_old, with E3 following E2 (c2 sits on the same mask and is
  driven through the same Fock couplings). Expect the screen to give
  the LMP2 ratios (0.02–0.66 at matched ε, < 1 at matched keep) on OP-MP3
  too; λ-Boys to give nothing on alkanes and a gain on naphthalene.
- **Physics (no carry-over at third order).** The screen ranks entries by a
  first-order estimate. If c2's dropped weight sits elsewhere, dE2 falls but
  dE3 does not, and dOP stops tracking dE2. Fingerprint: |dE3| / |dE2|
  grows in the new arm.
- **Artifact: the gain is just more amplitudes.** The screen only adds
  elements. Fingerprint: the gain at matched ε vanishes at matched keep.
  Matched-keep ratios (log-log interpolation in keep) are reported for that
  reason.
- **Artifact: λ-Boys port wrong or not applied.** The ε = 0 anchor cannot see
  it (any occupied rotation passes). Fingerprints: λ-Boys gradient check
  fails, G not raised, or `lambda` arm keep/error identical to `old-stable`
  to all digits. Separately, the finite-ε E2 of the masked c1 still matches
  the independent ragged LMP2 solver for no ≤ 30.
- **Artifact: inconsistent orders.** The screen mask is built once and used
  for both c1 and c2 (`make_mask` in `local_run`); the E3 identity checks it
  at every ε in every arm.
