# Specification: a from-scratch 3-center-1-electron kernel for COSX

Companion reference implementation: `scripts/3c1e_reference.py` (numpy, validated
against PySCF to ~1e-13 including degenerate probe geometries).

Target quantity, for contracted Cartesian/spherical AOs `chi` and a probe point `r_g`:

```
A^g_{mu,nu} = \int chi_mu(r) chi_nu(r) / |r - r_g| dr
```

---

## 0. VERDICT FIRST: this kernel does NOT close the COSX gap

**A well-vectorized McMurchie-Davidson or Rys 3c1e kernel is worth an estimated
4-7x over libint2's per-point path, with an absolute ceiling around 30-55x that
requires sustained peak AVX-512/AVX2 FMA throughput on the contraction step. The
measured COSX deficit is ~150x. This kernel does not close it, and should not be
built on the expectation that it will.**

The reasoning is in §7 and rests on one measured number: on water/cc-pVDZ,
libint2's per-point arithmetic cost is `b = 7.315e-5 s` (Phase 1 linear fit,
`scripts/queue/out/phase1_prereg.md`), while the McMurchie-Davidson operation
count for the same shell-pair set is **41,161 FLOPs per grid point**. libint2 is
therefore delivering the equivalent of **0.56 GFLOP/s** — roughly 15% of what a
single scalar core can do and ~2% of one core's AVX2 FMA peak.

That inefficiency is the entire opportunity, and it is bounded:

| system | measured `b` (s/pt) | MD FLOPs/pt | libint2 effective | speedup @4 GF/s | speedup @32 GF/s |
|---|---|---|---|---|---|
| water/cc-pVDZ | 7.315e-5 | 41,161 | 0.56 GFLOP/s | **7.1x** | 56.9x |
| water/cc-pVTZ | 2.597e-4 | 270,854 | 1.04 GFLOP/s | **3.8x** | 30.7x |

Note the direction: the achievable win **shrinks as the basis grows** (7.1x → 3.8x
at realistic throughput), because the contraction step's FLOP count grows faster
than libint2's overhead does. cc-pVTZ is where COSX would need the win most. This
is the same qualitative signature the Phase 1 batching study found, for the same
underlying reason.

**What would have to be true for this to be worth building.** Closing 150x needs
the arithmetic win *multiplied* by something else — screening that removes most
shell pairs at most grid points, and it must improve with system size. The Stage 2
kill gate measured that surviving-pair fraction across an alkane series reaching
past the ~30 Bohr locality onset and found it **flat**. A flat fraction plus a
≤7x arithmetic win does not reach 150x at any system size. If a future screening
result changes that — a genuinely falling pair fraction — then this spec becomes
worth implementing, and §1-§6 are written so it can be. Absent that, **do not
build this kernel.**

The honest framing: §1-§6 are a complete, implementable, validated specification.
§7 is the reason to leave it on the shelf.

---

## 1. Sign convention (load-bearing — get this wrong and errors are 2x, not small)

| source | operator returned |
|---|---|
| PySCF `mol.intor('int1e_grids', grids=pts)` | `+1/|r-r_g|` (repulsive) |
| libint2 nuclear operator, unit probe charge | `-1/|r-r_g|` (attractive) |
| ferric `cosx_a::a_matrix_at_point` | `+1/|r-r_g|` — it **negates** libint2 (`cosx_a.rs:237-238`) |
| `scripts/3c1e_reference.py` | `+1/|r-r_g|` |

A from-scratch kernel replaces the libint2 call **and** the negation together, so
it must produce `+1/|r-r_g|` directly and the caller's negation must be removed.

Note the asymmetry already present in the tree: `cosx_a`'s anchor test compares
against `esp_at_points`, which consumes the *un*-negated convention, so the anchor
flips the sign back (`cosx_a.rs:28-30`). A replacement kernel must keep that
anchor working.

Historical evidence this trap is live: the Stage 0 prototype got it backwards, and
the symptom was `max|dK|` plateauing at `2*||K||_max` rather than an obvious blowup.

---

## 2. Gaussian product theorem, and exactly what is grid-independent

For primitives on `A` (exponent `a`) and `B` (exponent `b`):

```
p   = a + b                                  total exponent
mu  = a*b/p                                  reduced exponent
P   = (a*A + b*B)/p                          product centre
Q   = A - B
K_AB = exp(-mu * |Q|^2)                      pair prefactor
```

**This is the whole basis of the vectorization strategy, so state it precisely:**

| quantity | depends on grid point `r_g`? |
|---|---|
| `p`, `mu`, `P`, `K_AB` | **NO** |
| E-coefficients `E_t^{ij}` (§3) | **NO** |
| contraction coefficients, normalization | **NO** |
| `PC = P - r_g` | YES |
| `T = p |PC|^2` | YES |
| Boys `F_n(T)` | YES |
| R-tensor `R^0_{tuv}` | YES |

Everything in the top block is hoisted **out** of the grid loop and computed once
per primitive pair. Only the bottom block runs per grid point.

**This is precisely what the failed batching attempt could not exploit.** The
Phase 1 study established (by reading `engine.h:650` and `engine.impl.h:261-268`)
that libint2's `set_params` is a cheap assignment, and that `compute_primdata` —
the primitive-pair work — is called *inside* the per-charge loop. So libint2
recomputes the grid-independent block for **every charge**, and no amount of
batching at the API level can hoist it, because the hoisting has to happen inside
the recursion. A from-scratch kernel hoists it by construction. That, not setup
amortization, is where the 4-7x comes from.

Empirical confirmation, from the numpy reference itself (water/cc-pVDZ, per-point
cost as the grid batch grows):

| npts | per-point (ms) | vs npts=1 |
|---|---|---|
| 1 | 271.0 | 1.0x |
| 16 | 38.8 | 7.0x |
| 256 | 2.84 | 95.5x |
| 1024 | 0.99 | 273.5x |
| 4096 | 0.48 | **562.6x** |

Even in pure Python the grid-independent work amortizes away almost entirely. (This
is a *structural* demonstration, not a performance claim against libint2 — Python's
constant factors dominate the absolute numbers. The FLOP-based estimate in §7 is
the honest performance statement.)

---

## 3. Chosen recursion: McMurchie-Davidson

### The three candidates

**(a) McMurchie-Davidson — RECOMMENDED.**
Expand the Cartesian AO pair density in Hermite Gaussians on `P`, then apply the
Coulomb kernel to each Hermite function analytically.

```
chi_a chi_b = sum_{t,u,v} E_t^{ij} E_u^{kl} E_v^{mn} * Lambda_{tuv}(r; P, p)
A = (2*pi/p) * sum_{tuv} E_t E_u E_v * R^0_{tuv}(p, P - r_g)
```

The decisive property for **this specific shape**: the E-coefficients carry all the
angular-momentum bookkeeping and are **grid-independent**, while the grid point
enters only through `R^0_{tuv}`, whose recursion is a handful of FMAs per element
and vectorizes trivially over the grid axis. The split is exactly the §2 table.

**(b) Rys quadrature.** What ORCA-class codes use, and it does batch naturally.
But for 3c1e specifically it is the wrong trade: Rys needs roots and weights of a
`(L_total/2 + 1)`-point quadrature *per grid point*, and root-finding does not
vectorize as cleanly as an FMA recursion. Rys earns its keep on 4-center ERIs where
it caps the intermediate growth; here the intermediates are small, and the
grid-independence structure MD offers is worth more than the intermediate-size
control Rys offers. Rys also has no analogue of "the E-coefficients don't depend on
the grid point" — its roots do.

**(c) Obara-Saika / Head-Gordon-Pople.** Mentioned for completeness. OS builds
angular momentum by a recursion whose intermediates depend on `PC`, i.e. on the
grid point, so the grid-independent hoisting that motivates MD is not available in
the same form. HGP's contraction-cost reduction targets 4-center integrals with
many primitive combinations; for 3c1e with one probe point it is not the binding
constraint.

**Recommendation: McMurchie-Davidson, justified by the vectorization structure —
the E-coefficients are grid-independent and the R-tensor is a pure FMA recursion
over the grid axis.** Not by aesthetics or familiarity.

### E-coefficient recursion (grid-independent)

Per Cartesian direction, with `Qx = Ax - Bx`:

```
E_0^{00} = exp(-mu * Qx^2)
E_t^{i+1,j} = (1/(2p)) E_{t-1}^{ij} - (mu*Qx/a) E_t^{ij} + (t+1) E_{t+1}^{ij}
E_t^{i,j+1} = (1/(2p)) E_{t-1}^{ij} + (mu*Qx/b) E_t^{ij} + (t+1) E_{t+1}^{ij}
E_t^{ij} = 0   for t < 0 or t > i+j
```

Three independent 1-D tables (x, y, z), each `(la+1) x (lb+1) x (la+lb+1)`.

### R-tensor recursion (grid-dependent — the only thing inside the grid loop)

```
R^n_{000}   = (-2p)^n * F_n(T),        T = p |PC|^2
R^n_{t+1,u,v} = t * R^{n+1}_{t-1,u,v} + PCx * R^{n+1}_{t,u,v}
R^n_{u+1} , R^n_{v+1}  analogously with PCy, PCz
```

Build by increasing total order `t+u+v`; each element is one FMA plus an optional
scaled add. **Every element is a vector over the grid axis.**

### Assembly

```
A_{ab} = (2*pi/p) * c_a * c_b * sum_{tuv} Ex[ax,bx,t] Ey[ay,by,u] Ez[az,bz,v] R^0_{tuv}
```

The `2*pi/p` prefactor already contains the Hermite-Coulomb normalization; note it
is `2*pi/p`, **not** `2*pi^{5/2}/(p*q*sqrt(p+q))` — that is the 4-center form, and
substituting it is a classic transcription error when adapting ERI code.

---

## 4. Boys function `F_n(T) = \int_0^1 t^{2n} exp(-T t^2) dt`

### Required order

`n = 0 .. L_total` where `L_total = la + lb`. For COSX with cc-pVTZ (f functions),
`L_total <= 6`; d-only bases need `n <= 4`. A kernel supporting up to g needs
`n <= 8`.

### Evaluation strategy, with measured breakpoints

**Small/moderate `T` (`T < 35`):** Taylor series for the **top** order `n = nmax`,
then recur **downward**.

```
F_nmax(T) = exp(-T) * sum_{k>=0} (2T)^k / (2*nmax + 2k + 1)!!
F_n(T)    = (2T * F_{n+1}(T) + exp(-T)) / (2n + 1)
```

Downward recursion is numerically **stable**; upward recursion in this regime is
not (it subtracts comparable quantities and loses relative precision at high `n`).
This direction choice is not stylistic — getting it backwards is a standard way to
lose 6+ digits at large `n` and small `T`.

**Large `T` (`T >= 35`):** asymptotic `F_0`, then recur **upward** (stable here):

```
F_0(T) = (1/2) sqrt(pi/T)
F_{n+1}(T) = ((2n+1) F_n(T) - exp(-T)) / (2T)
```

**Why the breakpoint is 35 and not the commonly-used 25.** The asymptotic `F_0`
drops the `erfc` tail. Measured relative size of the neglected term:

| T | 25 | 30 | 35 | >=40 |
|---|---|---|---|---|
| rel. error | 1.5e-12 | 9.4e-15 | 1.9e-16 | 0 (double) |

35 is the smallest breakpoint at which the asymptotic form is correct to full
double precision. **A code that switches at 25 silently carries ~1e-12 relative
error in `F_0`.** This is below a 1e-11 integral tolerance, so it does not show up
in an end-to-end integral test — verified by mutation: moving the switch to 25
passed every integral case in the reference's suite. `3c1e_reference.py` therefore
guards the breakpoint *directly*.

**Achieved precision.** The reference's Boys implementation agrees with mpmath at
50 digits to **1.7e-16 relative** across `T = 0 .. 80`, `n = 0..7`.

> Oracle warning for whoever validates the Rust version: `scipy.integrate.quad` is
> **not** an adequate reference here. It carries up to 1.7e-11 relative error on
> `t^14 exp(-80 t^2)`. An earlier draft of the reference's self-test used `quad`
> and reported a *failure of the Boys function* that was actually a failure of the
> oracle. Use mpmath, or closed forms.

### Vectorization note

`F_n(T)` is evaluated for a whole batch of grid points at once. The two branches
are data-dependent on `T`, so a SIMD implementation should either mask both
branches or partition the batch by `T` before evaluating. Partitioning is
preferable: the branches have very different costs (the Taylor loop iterates,
the asymptotic form does not).

---

## 5. ferric's actual conventions (READ THIS — several differ from textbook defaults)

All verified against the source, not assumed. Citations are to this branch.

### 5.1 Shell ordering (global)

Shells are emitted **atom-major, then basis-set order within the atom**, one
libint2 shell per `ferric_core::basis::Shell` (`basis_bridge.rs:78-100`). The C++
side preserves that order 1:1 (`shim.cc:94-107`) — **no sorting by angular
momentum**. Note this differs from libint2's own `BasisSet(name, atoms)`
constructor path, which is *not* used. AO offsets are a prefix sum of shell
dimensions (`basis_bridge.rs:129-132`).

### 5.2 Cartesian component ordering

libint2 is compiled with `LIBINT_CGSHELL_ORDERING_STANDARD` (CCA)
(`config.h:165,271`): `lx` descends from `l`, then `ly` descends from `l-lx`.

- d: `xx, xy, xz, yy, yz, zz`
- f: `xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz`

This **matches PySCF**. ferric's own terfc engine re-uses libint's `FOR_CART`
macro rather than hardcoding a table (`shim.cc:1044,1050-1056`), and `ao_grid.rs`
hardcodes exactly this order (`ao_grid.rs:216-222, 247-259`).

### 5.3 Spherical ordering

`LIBINT_SHGSHELL_ORDERING_STANDARD` (`config.h:280,359`), i.e.
**`m = -l, ..., +l`** (`shgshell_ordering.h:50,60`). This matches PySCF. It is
**not** the Gaussian/Molden `0,+1,-1,+2,-2,...` order (that is libint's
`GAUSSIAN` ordering, not compiled in here).

### 5.4 pure vs Cartesian

Per-basis **and** l-gated with an `l >= 2` floor:

```rust
// basis.rs:137 and :157
let shell_pure = pure && l >= 2;
```

BSE JSON `function_type` decides, but **l < 2 is forced Cartesian regardless**.
The G94 loader hardcodes `pure = l >= 2` (`basis.rs:283-284`).

**The cart→sph transform happens inside libint2**, not in ferric, for every
mainline integral — ferric passes the `pure` flag (`shim.cc:99`) and gets pure
blocks back. A from-scratch kernel must therefore do this transform **itself**,
and must reproduce libint2's convention (§5.6).

> **TRAP — pure p shells.** libint2 orders a *pure* p shell as `m = -1,0,+1 =
> y,z,x`, flagged twice in ferric's own C++ (`shim.cc:724`, `shim.cc:1465`). But
> `ao_grid.rs:194-199` returns `x,y,z` and its comment claims libint2 does too —
> **that comment is wrong**. It is currently harmless *only* because the `l >= 2`
> gate guarantees no basis-loaded l=1 shell ever has `pure == true`. Do not rely
> on the comment: `site_basis.rs:123-127` constructs shells with `pure: true`
> unconditionally, which is safe only for l >= 2.

### 5.5 Normalization — three layers, and they interact

**Layer A — ferric renormalizes each contraction to unit self-overlap at load**
(`basis.rs:220-233`, called at `:143,:156,:288-294`):

```
S = sum_pq c_p c_q (2 sqrt(a_p a_q)/(a_p + a_q))^(l + 3/2);   c <- c / sqrt(S)
```

This exists to fix the **grid path**, which has no contraction-level normalization
step (`basis.rs:209-212`). def2-* (Turbomole-raw) files need it; cc-pV*Z
(Gaussian-prenormalized) files are already unit.

**Layer B — the stored coefficients do NOT include the primitive norm `N(alpha,l)`.**
`ao_grid.rs:155-175` applies it explicitly at evaluation time:

```
N(a,l) = (2a/pi)^{3/4} * (4a)^{l/2} / sqrt((2l-1)!!)
```

**Layer C — libint2 applies the primitive norm AND its own unit-normalization**
(`shell.h:128-141` → `renorm()` at `shell.h:277-315`). ferric never disables
`do_enforce_unit_normalization`, so it runs — but it is a **no-op** because Layer A
already made `S = 1` (`basis.rs:212-213`).

> **Consequence for a from-scratch kernel:** consume
> `BasisSet::Shell::coefficients` as coefficients of *primitive-normalization-free*
> Gaussians, apply `N(alpha,l)` per primitive yourself, and you may skip the
> contraction renormalization (already unity). If you reproduce ferric's stored
> coefficients but skip Layer A, libint2 will still fix it up — but the **grid path
> will not**, and the two will silently disagree. That asymmetry is exactly what
> `basis.rs:209-212` documents.

### 5.6 Cartesian shells are UNNORMALIZED per component

**This is the single most likely thing to get wrong, and it is invisible on any
s/p-only test.**

In ferric/libint2/PySCF, a Cartesian shell carries **one shell-wide** normalization
constant. So within a d shell, `<xx|xx> != <xy|xy>` — the Cartesian overlap matrix
is **not** unit-diagonal for `l >= 2`. Do **not** apply the textbook per-component
factor `sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!)`.

libint2's `solidharmonics::coeff()` carries a compensating factor
(`solidharmonics.h:157`)

```
sqrt( (2l-1)!! / ((2lx-1)!!(2ly-1)!!(2lz-1)!!) )
```

which maps **unnormalized** Cartesians onto unit-normalized real solid harmonics
(reference given in-header: IJQC 54, 83 (1995) eqn 15; plus a `sqrt(2)` for
`m != 0`, `solidharmonics.h:159`). Normalizing the Cartesians per component
double-counts this and breaks **both** the Cartesian block and the cart→sph
transform.

Measured, and pinned in the reference's self-test — the `(l,0,0)` Cartesian
self-overlap in PySCF is:

| l | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `<g\|g>` | 1 | 1 | `4pi/5` | `4pi/7` | `4pi/9` |

i.e. **exactly 1 for `l < 2`, and `4pi/(2l+1)` for `l >= 2`**. s and p are
normalized as unit Cartesians; d and above as solid harmonics. PySCF's `gto_norm`
is the `l >= 2` branch for *all* l, which is why naively round-tripping through it
corrupts s and p. This discontinuity at `l = 2` is why a validation done only on
water/STO-3G proves nothing about this class of bug.

### 5.7 General contractions

ferric, libint2 and PySCF all permit `nctr > 1` — one shell sharing an exponent set
across several contractions (e.g. the 8-primitive s shell on oxygen in cc-pVDZ has
`nctr = 2`). Each contraction is a **separate basis function**, and they are
**adjacent** in the AO ordering.

Reading only the first contraction silently drops basis functions. It cost exactly
one AO (nbf 23 vs 24) on water/cc-pVDZ while writing the reference — the kind of
error that produces a wrong-shaped matrix rather than a wrong number, so it is
caught immediately *if* you test on a generally-contracted basis, and never
otherwise.

A production kernel should keep the shared exponents and **reuse the E-tables
across contractions** — a real optimization, since the E-coefficients depend only
on `(l_a, l_b, a, b, A, B)`, all shared.

### 5.8 `Z_eff`, not `Z`

`basis_bridge.rs:60-72` passes `Z - n_core` (and 0 for ghosts) to libint2, while
the **basis lookup uses the real `atom.z`** (`basis_bridge.rs:79`). This affects
nuclear attraction, not the 3c1e probe integral — but a kernel that also builds
`V_ne` must replicate it.

---

## 6. Loop structure and memory layout

The grid axis is the **fast/inner, contiguous** dimension throughout.

```
for each shell pair (s1, s2)          # screening decision lives here
    for each primitive pair (a, b)    # GRID-INDEPENDENT block
        p, mu, P, K_AB
        Ex[la+1][lb+1][la+lb+1]       # three small 1-D tables
        Ey[...], Ez[...]
        # ---- grid-dependent block ----
        for each grid chunk G (e.g. 256-1024 points, cache-resident):
            PC[3][G]                  # contiguous over G
            T[G]        = p * |PC|^2
            F[nmax+1][G]              # Boys, batched; partition by T
            R[(t,u,v)][G]             # FMA recursion, vector over G
            for (m, n) in cartesian components:
                acc[G] = sum_tuv Ex*Ey*Ez * R[(t,u,v)][G]     # FMA over G
                out[m][n][G] += pref * acc[G]
    cart -> sph transform on out       # small dense GEMM per shell pair
```

**Layout rules:**

- Store `R` as `[(t,u,v)][G]` — Hermite index outer, grid inner. The recursion
  then reads and writes contiguous vectors, and every operation is a fused
  multiply-add over `G`.
- Store the output block as `[ncart_a][ncart_b][G]`, transposing to
  `[G][nbf][nbf]` only at the end. Writing `[G][m][n]` inside the inner loop
  strides the hot accumulator and is the natural-looking layout that kills
  performance.
- Chunk `G` so `R` fits in L1/L2: `R` holds `(L+1)(L+2)(L+3)/6` vectors of length
  `G`. For `L = 6` that is 84 vectors; at `G = 512` doubles that is ~344 KB, so
  `G` in the 256-1024 range is the right order for typical L2.
- The E-tables are tiny and grid-independent — keep them in registers/L1 across
  the whole grid sweep for the pair.

**Screening** sits at the shell-pair level, outside everything, exactly where
`cosx_a.rs` already puts it. The existing `PairBounds`/`CosxScreen` structure and
its trivial-limit anchor (`cosx_a_zero_threshold_matches_unscreened`) carry over
unchanged; a replacement kernel must keep that anchor passing.

---

## 7. FLOP and vectorization analysis — is this worth writing?

### Operation counts

Counted directly from the recursion structure over the real segmented shell-pair
list (not estimated from `nbf`):

| system | seg. shells | shell pairs | R-tensor FLOPs/pt | contraction FLOPs/pt | total/pt |
|---|---|---|---|---|---|
| water/cc-pVDZ | 12 | 78 | 13,360 | 27,801 | **41,161** |
| water/cc-pVTZ | 22 | 253 | 39,908 | 230,946 | **270,854** |

The contraction step (`sum_tuv E*E*E*R`) dominates — 68% at DZ, 85% at TZ. It is
also the most vectorizable part: pure FMA over the grid axis, no branches, no
gathers. The R-tensor build is smaller and also FMA-dominated. The Boys function is
the only branchy component and is a small minority of the work.

### The comparison

Measured libint2 per-point arithmetic (Phase 1 linear fit, `R^2 >= 0.9997`):

```
water/cc-pVDZ:  b = 7.315e-5 s/point
water/cc-pVTZ:  b = 2.597e-4 s/point
```

Dividing the FLOP count by the measured time gives libint2's **effective**
throughput on this shape:

| system | effective throughput |
|---|---|
| water/cc-pVDZ | **0.56 GFLOP/s** |
| water/cc-pVTZ | **1.04 GFLOP/s** |

A modern core does ~4 GFLOP/s scalar and ~32 GFLOP/s with AVX2 FMA (~64 with
AVX-512). So libint2 is running this shape at roughly **2-3% of one core's vector
peak**. That is the headroom, and it is real — it is the direct consequence of
recomputing the grid-independent block per charge (§2).

### Achievable speedup

| system | at 4 GF/s (scalar-equivalent) | at 32 GF/s (sustained AVX2 FMA) |
|---|---|---|
| water/cc-pVDZ | **7.1x** | 56.9x |
| water/cc-pVTZ | **3.8x** | 30.7x |

**The realistic number is the left column.** Reaching 32 GF/s sustained would
require the contraction loop to hit near-peak FMA with perfect cache behaviour,
no loop overhead, and no time in Boys or the cart→sph transform — achievable in a
micro-benchmark of the inner loop, not across a real shell-pair sweep with
short inner dimensions (many pairs have `ncart_a * ncart_b` of 1-9 and `L <= 2`,
where loop overhead dominates and the vector units idle). A well-engineered kernel
landing at 25-50% of vector peak would give roughly **8-15x**.

Even taking the optimistic ceiling at face value: **30.7x at cc-pVTZ, against a
150x requirement.**

And the trend is adverse: the win *falls* from 7.1x to 3.8x (realistic) and 56.9x
to 30.7x (peak) going DZ → TZ, because contraction FLOPs grow faster than
libint2's per-call overhead. COSX needs the win most at larger bases, which is
exactly where this kernel delivers least.

### Verdict

**Do not build this to close the COSX gap.** A 4-7x realistic (≤30x optimistic)
arithmetic win does not overcome a ~150x deficit, and the residual would still be
5-20x slower than ferric's own analytic K — while costing a full Rys/MD kernel
with its own correctness surface (§5's traps are all live for a reimplementation).

This is consistent with, and independent of, the two prior negative results:
Stage 2 found the surviving-pair fraction flat past the locality onset, and Phase 1
found the batching ceiling at 1.03-1.36x. Three independent routes — screening,
API batching, and now raw arithmetic — each fall short by more than an order of
magnitude. They do not compose into a win: 7x arithmetic times a flat pair fraction
is still ~7x.

**What would change the verdict.** Only a screening result showing the surviving
shell-pair count per grid point *saturating* with system size (so the fraction
falls as `1/nsh^2`). That is a locality question, not an arithmetic one, and it was
already measured flat. If someone re-opens it with a better screen and it falls,
return here: §1-§6 are ready to implement, and the reference in
`scripts/3c1e_reference.py` is the correctness oracle for doing so.

---

## 8. Degenerate cases

| case | effect on `T = p|PC|^2` | handling |
|---|---|---|
| probe **on** a nucleus | `T = 0` exactly when `P` coincides with `r_g` | `F_n(0) = 1/(2n+1)` exactly. The reference short-circuits below `T < 1e-12` to avoid `0/0` in both recursions. Validated: 1.3e-14. |
| probe **1e-4 Bohr** away | `T ~ 1e-8 * p`, tiny but nonzero | Taylor branch. The downward recursion's `exp(-T)` → 1 and no cancellation occurs. Validated: 1.3e-14. |
| probe **1e-8 Bohr** away | `T ~ 1e-16 * p` | Below the `1e-12` cutoff for typical `p`; falls into the exact `T=0` limit. Validated: 2.5e-14. |
| probe **50-200 Bohr** away | `T` large (1e3-1e5) | Asymptotic branch, upward recursion. `exp(-T)` underflows to 0 harmlessly — the `((2n+1)F_n - exp(-T))/(2T)` recursion is then exact. Validated: 1.4e-17 absolute (values themselves are ~1e-2). |
| highly **contracted** primitives (large `a`) | large `p`, so `T` large even for nearby probes | Asymptotic branch; fine. Note `K_AB = exp(-mu Q^2)` underflows for distant tight pairs — that is a *correct* zero and should be used to skip the pair. |
| very **diffuse** primitives (small `a`) | small `p`, `T` small over a wide region | Taylor branch, many grid points share it. Partitioning the batch by `T` (§4) keeps these on one code path. |

The `T = 0` case deserves emphasis: it is not exotic. COSX grids are atom-centred,
so probe points land **exactly on nuclei** routinely, and `P` coincides with a
nucleus for any pair of primitives on that atom. A kernel that divides by `T` or by
`|PC|` without guarding will produce NaN on the very first realistic grid.

---

## 9. Validation

`scripts/3c1e_reference.py` self-test, elementwise vs
`mol.intor('int1e_grids')`. Every case at **~1e-13 absolute** (bar: 1e-11):

| case | lmax | nbf | worst probe set | max abs err |
|---|---|---|---|---|
| water/cc-pVDZ (sph) | 2 | 24 | generic | 7.1e-14 |
| water/cc-pVTZ (sph) | 3 | 58 | generic | 4.8e-14 |
| water/cc-pVDZ (cart) | 2 | 25 | generic | 1.5e-13 |
| methane/cc-pVDZ (sph) | 2 | 34 | generic | 5.2e-14 |
| water/STO-3G (sph) | 1 | 7 | generic | 1.7e-14 |

Each case is run against six probe sets: generic, on-nucleus (`T=0`),
near-nucleus 1e-4, near-nucleus 1e-8, far 50 Bohr, far 200 Bohr — 30 integral
checks total, all passing. Plus: Boys vs closed form, both-branch agreement
(6.5e-16), Boys vs mpmath at 50 dps (1.7e-16), the asymptotic breakpoint guard,
and the per-l AO normalization pin.

### Mutation testing

Per the repo rule that *a test you have never seen fail is an assumption*, every
guard was mutation-tested:

| mutation | caught? |
|---|---|
| flip the integral sign | **yes** — 30 failures |
| drop the `l >= 2` shell normalization factor | **yes** — 24 failures |
| reverse Cartesian component order | **yes** — 30 failures |
| read only the first general contraction | **yes** — shape mismatch (23 vs 24) |
| move the Boys breakpoint 35 → 25 | **initially NO** — added a direct breakpoint guard; now **yes** |

The Boys-breakpoint escape is worth noting: a ~1e-12 error in `F_0` is invisible to
a 1e-11 integral bar. Without the dedicated guard, that class of precision
regression would pass silently.

### For a Rust implementer

Run the same cases against your kernel. `run_self_test()` prints a pass/fail table
in exactly this form; the probe geometries are defined in `_probe_sets()` and the
molecules at the top of the file. Match to 1e-11 absolute and the traps in §5 are
all covered — in particular, **use a basis with d functions and a generally
contracted shell**, or §5.6 and §5.7 will not be exercised at all.
