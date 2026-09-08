# LinK pairwise density screen + density-pair criterion — design, written before the fix

Date: 2026-09-08. Branch `fix/link-pairwise-screen` off origin/main `e2cd1898`.
Input measurement: `link_fixed_counts.md` (same directory), post-#50 counts at
thresh 1e-12 / def2-SVP on a converged direct-SCF density.

This note states, BEFORE any code change, what each bound actually bounds, what
I expect a correct fix to do, and what I expect to see if my implementation is
broken instead. Per the repo's experimental protocol, the artifact hypothesis is
written next to the physics hypothesis so the anchors can distinguish them.

## The measured problem (restated, not re-derived)

| system / def2-SVP | DirectK quartets | fixed LinK | `build_jk` (pairwise D) | LinK/`build_jk` |
|---|---|---|---|---|
| alkane_8  | 8,775,381  | 8,775,277  | 8,107,064  | 1.082 |
| alkane_16 | 48,644,165 | 48,642,325 | 42,916,434 | 1.133 |

Kept pair fractions, same density: sp `Q·Qmax > t` keeps 0.855 (C8) / 0.560
(C16); dp `|D|·qmax(j)·qmax(σ) > t` keeps **1.000 / 1.000**.

K matches DirectK to 2.4e-14 and LinK is a strict subset of DirectK, so #50 is
sound. Two separate defects remain.

## (B) The per-quartet screen uses a global scalar

`link_k.rs` screens each quartet with

```text
    estimate(cs1,cs2,cs3,cs4) · max|D|  <  thresh      ->  skip
```

where `max|D|` is ONE scalar over the whole density matrix
(`link_k.rs`, `let max_d = d.iter()...fold(f64::max)`).

`build_jk` / `DirectJK` instead pass `DensityScreen::SixPair(&d_max_shell)`,
which evaluates `max(d12,d34,d13,d14,d23,d24)` from the shell-blocked
`build_d_max_shell` table. That is why the DEFAULT builder evaluates 7.6% (C8) /
11.8% (C16) FEWER quartets than the global-scalar screen at the identical
threshold.

### What the exchange quartet actually needs

LinK's scatter (link_k.rs, the 8-fold block) contracts the quartet `(cs1 cs2 |
cs3 cs4)` against D through exactly these blocks:

```text
    k[mu,la] += d[nu,sg]      k[nu,la] += d[mu,sg]
    k[mu,sg] += d[nu,la]      k[nu,sg] += d[mu,la]
```

plus, under `sym1234`, their transposes. In shell terms those are the FOUR
pairings `d13, d14, d23, d24` — and no others. The J-type pairings `d12` and
`d34` bound nothing that a K-only builder touches.

**Decision: add `DensityScreen::FourPairK`, `max(d13,d14,d23,d24)`, to the same
enum, reusing the same `build_d_max_shell` table.** Rationale:

* Reusing `SixPair` verbatim would be VALID (it is `>=` the four-pairing max, so
  it can only admit more) but strictly looser. It would cap LinK's quartet count
  at exactly `build_jk`'s rather than below it — and "LinK below `build_jk`" is
  the very thing anchor (c) has to demonstrate. A screen that merely ties the
  builder it is supposed to beat is not a fix.
* Writing a second max-table implementation is what the ticket forbids;
  `FourPairK` is a new *variant* on the existing enum consuming the existing
  `build_d_max_shell` output, so there remains exactly ONE table implementation.
* Validity: every block the LinK scatter reads is one of the four, so
  `max(d13,d14,d23,d24)` is an elementwise upper bound on every density element
  this quartet multiplies. Transposes are covered because `build_d_max_shell` is
  built from `|D|` and D is symmetric in every path LinK serves (RHF total
  density, UHF per-spin density) — the table is symmetric, `t[(a,b)] == t[(b,a)]`.

Physics hypothesis: LinK's quartet count drops BELOW `build_jk`'s, because LinK
screens on four pairings where `build_jk` screens on six (a max over fewer terms
is <=), while both start from the same Schwarz product.

Artifact hypothesis: if I get the index mapping wrong (e.g. pair `d13` as
`t[(s1,s3)]` when the scatter actually needs `t[(s2,s4)]`), the screen is no
longer an upper bound and K CHANGES — anchor (b) goes red at 1e-12, and the
thresh->0 anchor (a) stays green because at thresh 0 nothing is screened at all.
The two anchors therefore separate "wrong bound" from "wrong plumbing", which a
single test could not.

Note X == Y check: a correct fix and a too-tight (invalid) fix BOTH reduce the
quartet count. Count alone cannot distinguish them. That is exactly why (c)'s
count bars are paired with (b)'s K-value bars — neither is sufficient alone.

## (A) What the density-pair list bounds, and why it prunes nothing

Current criterion (`pairs.rs`):

```text
    max|D[j,σ]| · qmax(j) · qmax(σ)  >  thresh ,     qmax(x) = max_y Q(x,y)
```

This is a valid NECESSARY condition — the ticket's measurement confirms LinK
stays a subset of DirectK — but it is nearly vacuous, and the reason is
structural rather than a coding error:

`qmax(x)` is a maximum over ALL partners y in the molecule. For any alkane the
largest `Q(x,y)` is realized by a tight core s-shell pair, and its value is
essentially independent of x. So `qmax(j)·qmax(σ)` is not a per-pair quantity at
all: it is a molecule-wide constant ~`Qmax²`. The criterion degenerates to

```text
    max|D[j,σ]|  >  thresh / Qmax²
```

With `Qmax²` of order 1 and 1e-12 as the threshold, this asks whether any
density-matrix element between shells j and σ exceeds ~1e-12. Across ~50 Bohr of
alkane the density tail is still above that (consistent with the ~30 Bohr
density-matrix decay length in repo memory), so the answer is "yes" for every
pair — hence the measured 1.000 / 1.000.

**The two independent maxima are the defect.** No single quartet realizes both:
the quartet carrying `D[j,σ]` is `(i j | σ l)`, and LinK reaches it only with
`l ∈ sp(σ)` and with i constrained by the bra pair it is iterating. Taking the
global max over i and over l *separately* discards all locality on both sides at
once.

### What the dp list SHOULD bound

The dp list gates the ket loop: `σ ∈ dp(ish) ∪ dp(jsh)`. What it must bound is
the largest exchange contribution the pair `(j,σ)` can make to K through ANY
quartet LinK will actually visit from bra shell j:

```text
    max|D[j,σ]| · Q(j)_bra · qmax(σ)  >  thresh
```

where the first Schwarz factor is the one attached to the BRA pair containing j
that LinK is iterating, not a molecule-wide max. The tightest correct form that
is still cheap and pair-local keeps `qmax(σ)` (the ket partner l genuinely
ranges over sp(σ), so its max is the honest bound there) but replaces the
free-floating `qmax(j)` with the bra factor actually in play.

**Threshold semantics chosen: the SAME `thresh` the per-quartet screen
enforces.** Not a raised threshold. The ticket explicitly forbids raising it to
force pruning, and it would be wrong: the dp list is a necessary condition
feeding a per-quartet screen at `thresh`, so any pair it drops must be one no
surviving quartet could need. Keeping the thresholds equal is what makes the
list a pure accelerator with no accuracy cost — the property anchor (b) checks.

### The honest expectation, stated before measuring

With fix (B) in place the per-quartet screen already carries the pairwise density
information. The dp list then stops being the primary pruner and becomes a
ket-loop length bound. **I therefore expect (A) alone to prune modestly, and I
expect most of the count reduction to come from (B).** If the corrected dp
criterion still keeps ~100% at C16 while (B) delivers the count win, that is a
reportable negative about the dp list specifically — not a failure of the fix —
and anchor (c) must be written so it can SAY that rather than hide it.

This is the pre-registered position: I am not claiming the dp list will bite. I
am claiming the per-quartet screen will, and measuring whether the dp list adds
anything on top.

## Anchor plan (written before the fix compiles)

* (a) thresh -> 0 == dense K. EXISTING `link_k_matches_dense_in_the_trivial_limit`
  at 1.1e-15; must stay green. This is the exactness anchor and it must pass
  before any sweep runs.
* (b) production-threshold correctness, LinK K == DirectK K <= 1e-12, on
  water/cc-pVDZ, butane/def2-SVP (existing) AND alkane_16/def2-SVP (new — C16 is
  where the screen must bite, so it is where a too-tight screen would first show).
* (c) REACHABILITY — the assert whose absence let this ship. Past the ~30 Bohr
  onset (C16, C20), at the production threshold: the dp list must prune a stated
  non-trivial fraction, AND LinK's quartet count must be BELOW `build_jk`'s on
  the same density. Both counts, both deterministic. Bars set from what the
  corrected screen actually achieves — recorded as measured, and if the honest
  number is small it is stated as small.
* (d) SCF-level: butane/def2-SVP link SCF == direct SCF to 1e-9 Ha, same
  iteration count (existing `link_scf_anchor.rs`), extended to alkane_8.

Every new anchor is mutation-tested: revert to global-max|D| -> (c) red; revert
the dp criterion -> (c) red; loosen a bound so K changes -> (b) red. A bar I have
never seen fail is an assumption, not a measurement.
