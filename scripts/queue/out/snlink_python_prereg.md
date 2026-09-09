# Pre-registration: a Python reference for sn-LinK-style screening of seminumerical K

Date: 2026-09-08. Branch `spec/snlink-python`. Written and committed BEFORE
`scripts/snlink_proto.py` exists and BEFORE any number in
`snlink_python_results.md` was taken. No CPU-heavy work: numpy/PySCF, water-
and methane-sized systems, one thread, counts not timings.

---

## 0. The question, stated so it can come out "no"

**Does sn-LinK's screening structure keep LESS (shell-pair, grid-batch) work
than ferric's current density-driven batch screen, at equal K accuracy, on the
same molecule and grid?**

The deliverable is a kept-work table for three configurations at matched K
error — unscreened / ferric-current / sn-LinK — plus a recommendation. Per
`wiki/cosx-scaling-whitepaper.md` §4.1 (density-derived masks are vacuous below
~62 Bohr, established from three independent directions), the *expected* answer
at water/methane scale is **"cannot distinguish"**, and that is a complete
deliverable. This pre-registration is written so that outcome is reportable
without it looking like a failure to measure.

---

## 1. What is being compared (both screens, precisely)

Both sit in the same loop. Per grid batch `b` (a contiguous chunk of grid
points), for each shell pair `(s1,s2)`, decide keep/drop, then accumulate

```
G[s2, b] += A^b[s2, s1] F[s1, b]      (and the mirror s1<->s2)
Ktilde   += X G^T ;  K = ½(Ktilde + Ktilde^T)
X_{mu,g} = sqrt(w_g) chi_mu(r_g) ;   F = D X
```

**Screen FE ("ferric current", `crates/ferric-scf/src/cosx_k.rs:1096-1105`):**

```
keep(s1,s2,b)  iff  bound_A(s1,s2,sphere(b)) * max(fmax[s1,b], fmax[s2,b]) >= t
fmax[s,b] = max_{mu in s, g in b} |F_{mu,g}|
```

one threshold `t`, default 1e-7; `bound_A` is the Hölder primitive-pair bound of
`crates/ferric-integrals/src/cosx_screen.rs`, evaluated over the batch's
bounding sphere.

**Screen SN ("sn-LinK-style", the thing under test):** two branches, keep if
EITHER fires —

```
keep(s1,s2,b)  iff   bound_A(s1,s2,sphere(b)) * xmax_pair(s1,s2,b) >= eps_E     (E-branch)
                or   bound_A(s1,s2,sphere(b)) * dweight(s1,s2,b)   >= eps_K     (K-branch)
```

with, for the K-branch, the *pair-resolved* density weight that is the
seminumerical analogue of LinK's density-pair list

```
dweight(s1,s2,b) = max_{l in Lambda(b)} ( dmax[l,s1] * xmax[l,b] )   (and s1<->s2, take the max)
dmax[l,s]        = max_{mu in l, nu in s} |D_{mu,nu}|
xmax[l,b]        = max_{mu in l, g in b} |X_{mu,g}|
Lambda(b)        = shells with xmax[l,b] > 0
```

and, for the E-branch, the density-free integral-estimate gate
`xmax_pair(s1,s2,b) = max(xmax[s1,b], xmax[s2,b])`, which retains a pair on the
strength of its *integral* alone regardless of the density.

### 1.1 What is sourced and what is reconstructed (READ THIS)

Every attempt to obtain the primary sources failed on paywalls; the session's
web-search budget was also exhausted before this file was written. What was
actually verified, and from where:

| claim | status | source actually read |
|---|---|---|
| sn-LinK is a seminumerical counterpart of LinK, same `K = X G^T` core, differing in screening | **verified** | Ochsenfeld group review, Pure Appl. Chem. 2025, DOI 10.1515/pac-2025-0603, OA full text via EuropePMC PMC12645584 |
| the CPU sn-LinK paper is Laqua, Kussmann, Ochsenfeld, *JCTC* **14**, 3451 (2018); GPU paper is *JCTC* **16**, 1456 (2020) | **verified** | Crossref + OpenAlex metadata; the 2020 abstract names the 2018 paper by page range |
| the 2018 method combines **preLinK** (Kussmann & Ochsenfeld, *JCP* **138**, 134114 (2013)) with **explicit screening of integrals for batches of grid points** | **verified (abstract only)** | OpenAlex abstract of DOI 10.1021/acs.jctc.8b00062 — quote: "combining the preLinK method ... with explicit screening of integrals for batches of grid points to minimize the screening overhead" |
| the 2020 GPU paper adds the **integral partition bounds** of Thompson & Ochsenfeld, *JCP* **150**, 044101 (2019) (DOI 10.1063/1.5048491) | **verified (abstract only)** | OpenAlex abstract of DOI 10.1021/acs.jctc.9b00860 |
| preLinK is a **Schwarz-estimate-based preselection of the significant elements of K, before evaluation** | **verified (abstract only)** | OpenAlex abstract of DOI 10.1063/1.4796441 |
| the specific **two-threshold `eps^E` / `eps^K` structure** and its exact formulas | **NOT VERIFIED — reconstructed** | no primary text obtainable |
| the exact form of the **integral partition bound** applied to 3c1e | **NOT VERIFIED — substituted** | ferric's own Hölder bound is used instead (see below) |

**Consequence, stated up front so it cannot be over-claimed later.** Screen SN
as implemented here is *not* a transcription of Laqua's screen. It is the
screening *structure* the sources establish — a density-free integral branch
ORed with a density-weighted branch, both evaluated per grid batch, with the
density entering **pair-resolved** rather than as a per-shell scalar —
instantiated on ferric's own Hölder integral bound. Two deliberate
substitutions:

- **The integral estimate.** Thompson & Ochsenfeld's IPB is unobtainable, so
  both screens use the SAME `bound_A`. This is *deliberate and load-bearing for
  the experiment*: holding the integral estimate fixed isolates the **screening
  structure** (one threshold on a shell-scalar vs two thresholds with a
  pair-resolved density weight), which is the question asked. A different bound
  would confound the comparison. It also means this study **cannot** speak to
  how much of Laqua's win comes from a tighter bound — recorded as a scope
  limit, not resolved.
- **`eps^E` / `eps^K` naming.** The two-branch structure follows from
  "preLinK (density-driven) + explicit per-batch integral screening"; the
  *names* are this document's, not quoted.

Anyone with library access should re-derive Screen SN from the 2018 paper §2
before any Rust is written. This file is a structural probe, not a citation.

---

## 2. Anchors, pre-registered (each with the mutation that must break it)

Written before the implementation. Every one is a hard assert in
`scripts/snlink_proto.py`; the script exits nonzero if any fails.

### A1 — Trivial limit (EXACTNESS ANCHOR; nothing else means anything without it)

With all thresholds set to 0 (FE: `t=0`; SN: `eps_E=eps_K=0`) **every** pair is
kept in **every** batch, and `K_screened` reproduces `K_unscreened` to
`<= 1e-14` absolute (both computed by the same accumulation in the same order,
so the realistic expectation is bitwise 0.0; the bar is 1e-14 to allow for
`-ffast-math`-free numpy reassociation if any appears).

- **Bar:** `max|K_screen(0) - K_unscreened| <= 1e-14`, AND kept-pair count ==
  total pair count for both screens.
- **Mutation that must break it:** make the keep rule `>` instead of `>=` at
  threshold 0 with a zero-valued bound (drops same-centre pairs whose bound
  underflows), and separately, drop the mirror `s1<->s2` accumulation. Both
  must fail A1.

### A2 — Correctness at production thresholds, against the GRID error

The screen's K error must sit BELOW the error the grid itself already carries.
The grid error is measured, not assumed:

```
E_grid = max| K_unscreened(grid) - K_analytic |     K_analytic from PySCF ERIs
E_scr  = max| K_screened - K_unscreened |
```

- **Bar:** `E_scr <= 0.1 * E_grid` for both screens at their production
  thresholds, on water/cc-pVDZ **and** on a second, larger case
  (methane/cc-pVDZ and, if it runs in seconds, ethane/cc-pVDZ).
- **Rationale for 0.1 rather than "below":** a screen error equal to the grid
  error would double the total; an order of magnitude below is the standard the
  whitepaper's §4 layer-2 result already meets (K error 1.6e-10 on water
  against a grid error of 2.4e-5, i.e. 5 orders).
- **Mutation that must break it:** raise the thresholds by 6 orders
  (`t = 1e-1`). If A2 still passes at `t=1e-1`, the bar is not measuring
  anything and must be reported as such.

### A3 — Reachability (the pass condition must be REACHABLE)

At the production thresholds the screen must actually drop work. State the
expectation BEFORE measuring:

- **Expected at water/methane scale: it does NOT bite.** Whitepaper §4.1 and
  §6.2: density-derived masks are vacuous at ≤62 Bohr; water is ~3 Bohr across.
  So the pre-registered expectation for the *density* branch is
  `kept_fraction ≈ 1.000`, and any measured drop at these sizes must come from
  the *geometric* decay of `bound_A` (the 1/R and `K_AB` factors), which does
  bite even on small molecules because the grid extends 10-30 Bohr from the
  nuclei.
- **Reported, not asserted:** `kept_pair_batches / total_pair_batches` for
  every configuration. The script asserts only the weaker, genuinely reachable
  statement: **at production thresholds at least one pair-batch is dropped**
  (i.e. the screen is not vacuous), and it separately reports whether the
  DENSITY branch alone drops anything beyond what the geometric branch drops.
- **If the density branch drops nothing** — the expected outcome — the honest
  report is "cannot distinguish at this scale", plus the size at which our own
  data says it would begin (~62 Bohr, i.e. alkane_20 and beyond), which Python
  cannot reach in this task's CPU budget.
- **Mutation that must break it:** set thresholds to 0 — the "at least one
  dropped" assert must fail (this is the same check as A1 from the other side,
  and it proves the reachability assert is not tautological).

### A4 — The screens are actually different

If Screen SN and Screen FE keep exactly the same set on every system tested,
the comparison is uninformative and must say so rather than reporting "equal".

- **Reported:** the symmetric difference `|kept_SN Δ kept_FE|` per system, and
  the count of pair-batches kept by SN's density branch but not by FE, and vice
  versa.
- **Mutation that must break the harness's ability to see a difference:**
  set `eps_E = eps_K = t` and make `dweight` = `max(fmax[s1],fmax[s2])`; SN then
  reduces to FE exactly and the symmetric difference must be 0. This is a
  *positive* control on the difference-counter itself.

### A5 — The kept-work counter counts the work

The counter must be the thing that drives cost, not a proxy. Pre-registered
definition: **one unit = one (shell pair, grid batch) that reaches the kernel**,
weighted by `ncart(s1)*ncart(s2)*len(batch)` in a second, cost-weighted column
(a d-d pair costs 36x an s-s pair per point). Both raw and weighted counts are
reported; the raw count is the headline because it is what the Rust screen
skips, and the weighted one is what actually costs.

- **Mutation that must break it:** skip the accumulation for a kept pair
  without decrementing the counter — A1/A2 must then fail (wrong K) while the
  counter is unchanged, demonstrating the counter and the K are independently
  observable.

---

## 3. Artifact hypothesis, next to the physics hypothesis

Required by the repo's Experimental Protocol: state what a broken
implementation would look like, and check it differs from the real result.

| | if sn-LinK's structure is genuinely better here | if my implementation is broken |
|---|---|---|
| kept fraction, SN vs FE at matched K error | SN keeps measurably fewer pair-batches, and the gap GROWS with system size | SN keeps fewer because its bound is being *underestimated* — which shows up as `E_scr > E_grid` in A2, i.e. the accuracy anchor breaks |
| density branch | drops pair-batches that the geometric branch keeps | drops nothing at all (vacuous, the §4.1 expectation) **or** drops everything (a `dmax` indexing bug zeroing the weight) |
| trivial limit | exact | A1 fails |

**These predictions differ**, so the experiment can distinguish them. The one
case it CANNOT distinguish: SN keeping fewer pairs *and* K error staying below
the grid error *because the grid error is large*. Guarded by reporting `E_scr`
and `E_grid` as separate numbers, never only their ratio.

A second artifact channel, specific to this comparison: **if SN and FE differ
only because SN's `dweight` happens to be numerically larger/smaller than FE's
`fmax` by a constant factor, the comparison is a threshold rescaling, not a
structural result.** Guarded by A4's positive control and by reporting each
screen's kept fraction as a function of its own threshold over a sweep, so a
pure rescaling shows up as two curves that superimpose after a shift.

---

## 4. Systems, grid, thresholds (fixed before measuring)

- **Systems:** water/cc-pVDZ (24 bf) as the primary; methane/cc-pVDZ (34 bf)
  as the second case; ethane/cc-pVDZ (58 bf) only if it completes in seconds.
- **Grid:** PySCF `Grids(level=0)` Becke grid, atom-partitioned, weights folded
  into `X`. Batch size 256 points (matching ferric's `COSX_SUB_BATCH_POINTS`).
- **Density:** the converged RHF density at the same basis (PySCF `scf.RHF`),
  so the density-dependence is the physical one, not a guess matrix.
- **Production thresholds:** FE `t = 1e-7` (ferric's default,
  `COSX_DEFAULT_SCREEN_THRESH`). SN `eps_E = eps_K = 1e-7` for the headline row,
  with a sweep 1e-5..1e-11 reported for both.
- **What is NOT done:** no overlap fit (`S_num`), because it changes K by a
  system-dependent amount (whitepaper §5.2: 16.9x on water, 0.5x on ethane) and
  would confound the screen comparison. Both screens are compared against the
  same unfitted unscreened K.

---

## 5. What would flip the recommendation

Pre-registered so the negative, if it comes, is not open-ended:

1. The density branch dropping a measurable fraction at ≤ 60 Bohr would
   contradict §4.1 and make sn-LinK's structure worth Rust immediately.
2. A tighter integral estimate (the IPB) is a SEPARATE lever this study cannot
   measure, and a negative here does NOT close it. If someone obtains
   *JCP* **150**, 044101 (2019), the right follow-up is to swap `bound_A` for
   the IPB in BOTH screens and re-run this harness — the code is structured to
   make the bound a single injectable function for exactly that reason.
3. The onset test (alkane_20/alkane_32, ~62-98 Bohr) is a Rust-side experiment,
   not a Python one; §6.2 of the whitepaper already briefs it.
