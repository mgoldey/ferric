# Unrestricted CCSD and CCSD(T) {#uccsd}

<!-- status: ready; registry: u-ccsd, u-ccsd-t -->

In an open-shell molecule the α and β electrons feel different exchange
fields, and a UHF reference gives them different orbitals, so the
closed-shell spin adaptation of sections @sec:closed-shell-ccsd and @sec:triples does not apply. Spin
is still conserved: a spin-orbital amplitude vanishes unless each
electron keeps its spin. The spin-blocked solver keeps only the five nonzero
amplitude blocks and the integral blocks they touch. The CLI kinds `ccsd`
and `ccsd(t)`, and Python `run_ccsd` and `run_ccsd_t`, take this route for a
molecule with multiplicity > 1.

The spin-orbital solver of sections @sec:ccd-ccsd-spin-orbital and @sec:triples is the cross-check. Fed
the same UHF orbitals as one α-then-β spin-orbital space, it has no spin
bookkeeping at all. The section derives the blocks (@sec:uccsd-equations), lays out the
loop (@sec:uccsd-algorithm) and lists what is anchored (@sec:uccsd-numerical-anchors).

| Symbol | Meaning | Code name |
|---|---|---|
| $i,j,a,b$ | α occupied, α virtual | lower case in `ein` specs |
| $I,J,A,B$ | β occupied, β virtual | upper case in `ein` specs |
| $t_i^a$, $t_I^A$ | singles per spin | `t1a`, `t1b` |
| $t_{ij}^{ab}$, $t_{IJ}^{AB}$ | same-spin doubles, antisymmetric in each pair | `t2aa`, `t2bb` |
| $t_{iJ}^{aB}$ | opposite-spin doubles, $t_{i\alpha J\beta}^{a\alpha B\beta}$ | `t2ab` |
| $B^P_{pq,\sigma}$ | dressed RI factors of spin $\sigma$ | `URiBlocks` |
| $n_\beta$ | number of β occupied indices in a (T) triple | class aaa, aab, abb, bbb |
| $w(n_\beta)$ | (T) class weight | `CLASS_WEIGHTS` |

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/u_ccsd.rs::u_ccsd`, `ferric-cc/src/u_ccsd_t.rs::u_ccsd_t` |
| Cross-check | `ferric-cc/src/ccsd.rs::ccsd_uhf_spin_orbital`, `ferric-cc/src/ccsd_t.rs::ccsd_t_uhf_spin_orbital` |
| CLI | `method.kind = "ccsd"` or `"ccsd(t)"` with `multiplicity > 1`, `task = "energy"` |
| Python | `run_ccsd`, `run_ccsd_t` (open-shell molecule) |

## The equations {#uccsd-equations}

**The amplitude blocks.** With lower-case indices α and upper-case β, the
nonzero amplitudes are

$$t_i^a,\quad t_I^A,\quad t_{ij}^{ab},\quad t_{iJ}^{aB} = t_{i\alpha J\beta}^{a\alpha B\beta},\quad t_{IJ}^{AB}.$$

`t2aa` and `t2bb` are antisymmetric in $(i,j)$ and in $(a,b)$. These are
PySCF's `uccsd` conventions. The other mixed-spin spin-orbital elements
follow from antisymmetry: swapping the two particle labels, or the two hole
labels, changes the sign.

| spins of $(q,r,s,e)$ | $t_{qr}^{se}$ |
|---|---|
| $\alpha\alpha\alpha\alpha$ | $t_{qr}^{se}$ (`t2aa`) |
| $\beta\beta\beta\beta$ | $t_{QR}^{SE}$ (`t2bb`) |
| $\alpha\beta\alpha\beta$ | $t_{qR}^{sE}$ |
| $\alpha\beta\beta\alpha$ | $-t_{qR}^{eS}$ |
| $\beta\alpha\alpha\beta$ | $-t_{rQ}^{sE}$ |
| $\beta\alpha\beta\alpha$ | $t_{rQ}^{eS}$ |

Every other spin pattern is zero.

**Integrals.** Each spin has its own dressed RI factors $B^P_{pq,\sigma}$
over its active occupied and virtual orbitals (`URiBlocks`), so $(pq|rs)$
with $p,q$ of spin $\sigma$ and $r,s$ of spin $\tau$ is
$\sum_P B^P_{pq,\sigma}B^P_{rs,\tau}$. A chemist integral is zero unless each
bra and each ket pair shares a spin, and

$$\langle pq\|rs\rangle = (pr|qs)\,\delta_{\sigma_p\sigma_r}\delta_{\sigma_q\sigma_s} - (ps|qr)\,\delta_{\sigma_p\sigma_s}\delta_{\sigma_q\sigma_r} .$$

`UEris` holds the chemist blocks of PySCF's `_ChemistsERIs` for UCCSD, all
three `vvvv` blocks dense. The Fock matrix is taken as diagonal (canonical
UHF orbital energies), as on the restricted route.

**Amplitude equations.** The residuals are the spin-orbital CCSD residuals
of section @sec:ccd-ccsd-spin-orbital with every index restricted to the spin-allowed blocks.
`update_amps` evaluates them in the factorization of PySCF
`uccsd.update_amps` (with $f_{ia}=0$), line for line:

- the $F_{oo}$, $F_{vv}$, $F_{ov}$ intermediates per spin;
- the six $W_{ovvo}$ ring blocks (`wovvo`, `wOVVO`, `woVvO`, `woVVo`,
  `wOvVo`, `wOvvO`), one per spin pattern of the particle–hole pair a ring
  passes through;
- the hole–hole and particle–particle ladders on $\tau$;
- the singles.

The update is $t \leftarrow r/(-\Delta)$ with the per-spin denominators, and
DIIS runs on all five blocks jointly.

**Energy.** With $\tau_{ij}^{ab} = t_{ij}^{ab} + t_i^at_j^b - t_j^at_i^b$
the spin-orbital energy $\tfrac14\sum\langle ij\|ab\rangle\tau_{ij}^{ab}$
splits by the spins of $(i,j)$. For two α electrons the integral is
$(ia|jb)-(ib|ja)$, and its antisymmetry in $a,b$ turns the two singles
products of $\tau$ into twice one of them. For one α and one β electron the
exchange half is zero, and the four spin orderings
($i\alpha J\beta$ or $J\beta i\alpha$, times the two placements of $a,B$)
each give a quarter of the same sum:

$$
\begin{aligned}
E_{\alpha\alpha} &= \tfrac14\sum_{ijab}\big[(ia|jb)-(ib|ja)\big]t_{ij}^{ab}
 + \tfrac12\sum_{ijab}\big[(ia|jb)-(ib|ja)\big]t_i^at_j^b, \\
E_{\alpha\beta} &= \sum_{iJaB}(ia|JB)\big(t_{iJ}^{aB} + t_i^at_J^B\big),
\end{aligned}
$$

and $E_{\beta\beta}$ is $E_{\alpha\alpha}$ with β blocks. The αβ term
already counts both $(i\alpha,J\beta)$ and $(J\beta,i\alpha)$ orderings.

<!-- excerpt: ferric-cc/src/u_ccsd.rs::energy -->
```rust
    let same = |t1: &ArrayD<f64>, t2: &ArrayD<f64>, ovov: &ArrayD<f64>| {
        0.25 * ein0("ijab,iajb->", &[t2, ovov]) - 0.25 * ein0("ijab,ibja->", &[t2, ovov])
            + 0.5 * ein0("ia,jb,iajb->", &[t1, t1, ovov])
            - 0.5 * ein0("ia,jb,ibja->", &[t1, t1, ovov])
    };
    let e_aa = same(&t.t1a, &t.t2aa, &e.ovov);
    let e_bb = same(&t.t1b, &t.t2bb, &e.ovov_b);
    let e_ab = ein0("iJaB,iaJB->", &[&t.t2ab, &e.ov_ov])
        + ein0("ia,JB,iaJB->", &[&t.t1a, &t.t1b, &e.ov_ov]);
```

- `same` is $E_{\alpha\alpha}$ term by term, applied once to the α blocks
  and once to the β blocks.
- `ovov` holds $(ia|jb)$ at `[i,a,j,b]`; the spec `ibja` reads the same
  block as $(ib|ja)$ without a transpose.
- `ov_ov` is the mixed block $(ia|JB)$, $\sum_P B^P_{ia,\alpha}B^P_{JB,\beta}$.

> **Verified equation {#eq:cc-u-energy-split}** · the CCSD energy split
>
> $E_{\alpha\alpha} + E_{\alpha\beta} + E_{\beta\beta}$, with the
> $\alpha\beta$ term counting both spin orderings once, equals the dense
> spin-orbital $\tfrac14\sum\langle ij\|ab\rangle\tau_{ij}^{ab}$ on the same
> random blocks. Claim 4 checks the closed-shell limit: RHF orbitals for
> both spins give $E_{\alpha\alpha} = E_{\beta\beta}$, aaa $=$ bbb and
> aab $=$ abb.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_uccsd_spin_blocks_sympy.py`, claims 3 and 4 |
> | Negative control | claim 5 (a dropped exchange term, a dropped $t_I^A$); the runner's mutation dropping the same-spin exchange term |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/u_ccsd.rs::{energy,update_amps,u_ccsd}` |
> | Code test | `closed_shell_through_uhf_matches_closed_shell` in `crates/ferric-cc/tests/uccsd_anchors.rs` |
>
> <!-- verified: id=cc.u.energy-split; script=scripts/cc_uccsd_spin_blocks_sympy.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/u_ccsd.rs::{energy,update_amps,u_ccsd}; rust_test=ferric-cc/tests/uccsd_anchors.rs::closed_shell_through_uhf_matches_closed_shell; registry=u-ccsd -->

**(T).** `u_ccsd_t` evaluates the spin-orbital formula of section @sec:triples-spin-orbital-t on
the spin blocks. The summand $f = (W+V)W/\Delta$ is invariant under any
permutation of $(i,j,k)$ and, separately, any permutation of $(a,b,c)$. It
vanishes unless the occupied and virtual triples carry the same spins, and
unless $i,j,k$ are distinct. Classify the occupied triple by its number of β
indices, $n_\beta = 0,1,2,3$ (classes aaa, aab, abb, bbb). For each class
the loop visits canonical occupied triples ($i<j<k$ within one spin) and one
canonical virtual spin layout (α positions first), with each same-spin
virtual index running over its full range:

| class | occupied loop | virtual layout | weight |
|---|---|---|---|
| aaa | $i<j<k$ | $(\alpha,\alpha,\alpha)$ | $1/6$ |
| aab | $i<j$, $K$ | $(\alpha,\alpha,\beta)$ | $1/2$ |
| abb | $i$, $J<K$ | $(\alpha,\beta,\beta)$ | $1/2$ |
| bbb | $I<J<K$ | $(\beta,\beta,\beta)$ | $1/6$ |

**The weights.** Count how many terms of the fully ordered sum each visited
term stands for. A canonical occupied triple stands for its $3!$ orderings.
The virtual labels already run over their full ranges within each spin, so
they cover the orderings among same-spin positions; what the canonical
layout does not cover is where the $n_\beta$ β labels sit among the three
positions, $\binom{3}{n_\beta}$ choices. The fully ordered sum carries
$\tfrac1{36}$, so

$$w(n_\beta) = \frac{3!\binom{3}{n_\beta}}{36},$$

which is $\tfrac{6}{36} = \tfrac16$ for aaa and bbb and $\tfrac{18}{36} = \tfrac12$
for aab and abb. Then $E_{(T)} = -\sum_{\text{class}} w \sum_{\text{canonical}} (W+V)W/\Delta$.
The closed-shell check: with RHF orbitals for both spins, aaa $=$ bbb and
aab $=$ abb.

The constant in `ferric-cc/src/u_ccsd_t.rs::CLASS_WEIGHTS`, in class order
aaa, aab, abb, bbb:

```rust
pub(crate) const CLASS_WEIGHTS: [f64; 4] = [1.0 / 6.0, 0.5, 0.5, 1.0 / 6.0];
```

For each occupied triple the kernel builds $X$ and $Y$ block by block: a
term enters only when the spins of its $t_2$ element (table above) and its
integral both match, and $P(i/jk)P(a/bc)$ acts on the three occupied labels
and on the three virtual positions with their spins. The loop runs over
canonical triples of all four classes in parallel and folds them serially in
a fixed order. The class energies are reported.

> **Verified equation {#eq:cc-ut-class-weights}** · the (T) class weights
>
> $w(n_\beta) = 3!\binom{3}{n_\beta}/36$ is $1/6, 1/2, 1/2, 1/6$. A
> brute-force count of ordered spin-orbital triples equals the sum over
> classes of canonical triples times $3!\binom{3}{n_\beta}$.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_uccsd_spin_blocks_sympy.py`, claim 1 |
> | Negative control | claim 5 (an aab weight of $1/6$ fails); the runner's mutation `W_AAB = 1.0 / 6.0` |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/u_ccsd_t.rs::{CLASS_WEIGHTS,u_ccsd_t}` |
> | Code test | `oh_spin_blocked_matches_spin_orbital` in `crates/ferric-cc/tests/uccsd_anchors.rs` (each class against the spin-orbital (T)) |
>
> <!-- verified: id=cc.ut.class-weights; script=scripts/cc_uccsd_spin_blocks_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/u_ccsd_t.rs::{CLASS_WEIGHTS,u_ccsd_t}; rust_test=ferric-cc/tests/uccsd_anchors.rs::oh_spin_blocked_matches_spin_orbital; registry=u-ccsd-t -->

> **Verified equation {#eq:cc-ut-classes}** · spin-blocked (T) per class
>
> On random spin-blocked amplitudes and per-spin RI factors with unequal α
> and β spaces ($n_o = 4, 3$; $n_v = 3, 4$), the spin-blocked (T) of each
> class, built with the $t_2$ sign table, equals the dense spin-orbital (T)
> restricted to that class, to $1.2\times10^{-16}$ relative. Every class is
> nonzero on these sizes.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_uccsd_spin_blocks_sympy.py`, claim 2 |
> | Negative control | claim 5; the runner's mutations: aab weight $1/6$, a dropped sign in the $t_2$ table |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/u_ccsd_t.rs::u_ccsd_t` |
> | Code test | `nh2_spin_blocked_matches_spin_orbital` in `crates/ferric-cc/tests/uccsd_anchors.rs` |
>
> <!-- verified: id=cc.ut.classes; script=scripts/cc_uccsd_spin_blocks_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/u_ccsd_t.rs::u_ccsd_t; rust_test=ferric-cc/tests/uccsd_anchors.rs::nh2_spin_blocked_matches_spin_orbital; registry=u-ccsd-t -->

**Frozen core.** `frozen_core` drops the lowest `frozen_core` orbitals of
each spin, as `rimp2::active_occ` does for U-RI-MP2.

**Two limits.**

- Closed-shell molecule, RHF orbitals given to both spins: $t_{ij}^{ab}$
  (`t2aa`) $= t_{iJ}^{aB} - t_{iJ}^{bA}$, the same-spin block of section
  @sec:closed-shell-ccsd. Then $E_{\alpha\alpha} = E_{\beta\beta}$, aaa $=$ bbb and
  aab $=$ abb, and the CCSD energy, $t_{iJ}^{aB}$ and $E_{(T)}$ equal those
  of `ccsd_closed_shell` and `ccsd_t_closed_shell`.
- On the same UHF orbitals, the spin-orbital solver of `ccsd.rs` and
  `ccsd_t.rs`, fed the α-then-β spin-orbital space
  (`ccsd_uhf_spin_orbital`, `ccsd_t_uhf_spin_orbital`), has no spin
  bookkeeping and must give the same amplitudes, energy and (T) classes.

**What the proofs cover.** `scripts/cc_uccsd_spin_blocks_sympy.py` proves
the class weights with sympy and a brute-force count (claim 1). On random
spin-blocked amplitudes and per-spin RI factors with unequal α and β spaces
it checks that the transcribed spin-blocked (T) equals the dense
spin-orbital (T) per class (claim 2), that the energy split above equals the
dense spin-orbital energy (claim 3) and the closed-shell limit (claim 4).
Claim 5 shows that an aab weight of $1/6$, a wrong sign in the $t_2$ table,
swapped α/β virtual energies, a dropped $t_I^A$ and a dropped exchange term
each fail. The `update_amps` residual is a transcription of PySCF; its
evidence is the Rust comparisons below. In Rust, `tests/uccsd_anchors.rs`
runs both limits above and `tests/validation_uccsd.rs` compares with PySCF
DF-UCCSD and UCCSD(T). The two Rust solvers share the per-spin RI factors
and orbital energies, so a fault in those inputs is caught only by the PySCF
comparison.

## The algorithm {#uccsd-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | UHF with exact J/K; refuse a restricted reference | | `solve_uhf`, `uhf_counts` |
| 2 | dressed $B^P_{pq,\sigma}$ over active occupied and all virtual orbitals of each spin | $N_{\rm aux}n^2$ per spin | `URiBlocks::build` |
| 3 | chemist blocks of PySCF `_ChemistsERIs`, the three `vvvv` blocks dense | $N_{\rm aux}n_v^4$ | `UEris` |
| 4 | residuals in the factorization of PySCF `uccsd.update_amps`, $f_{ia} = 0$ | $n_o^2n_v^4$ | `update_amps` |
| 5 | $t \leftarrow r/(-\Delta)$, DIIS on all five blocks jointly | | `solve` |
| 6 | $E_{\alpha\alpha}$, $E_{\alpha\beta}$, $E_{\beta\beta}$ | $n_o^2n_v^2$ | `energy` |
| 7 | (T): canonical occupied triples of the four classes in parallel, folded serially in a fixed order | $n_o^3n_v^4$ | `u_ccsd_t` |

The memory pre-flight (`u_ccsd_memory_plan`) sizes the dense `vvvv`
blocks; there is no integral-direct ladder on this route. Frozen core drops
the lowest `frozen_core` orbitals of each spin (`active_occ`).

## Numerical anchors {#uccsd-numerical-anchors}

| Quantity | Systems | Error | Bar | Test |
|---|---|---|---|---|
| E_UHF vs stability-followed PySCF UHF | OH, NH2, O2 triplet / cc-pVDZ | $\le 5.4\times10^{-13}$ Ha | $3\times10^{-11}$ Ha | `crates/ferric-cc/tests/validation_uccsd.rs` |
| E_CCSD and each αα / αβ / ββ block vs PySCF DF-UCCSD (aux cc-pVDZ-RI), frozen core 0 and 1 | same | $\le 3.4\times10^{-11}$ Ha | $2\times10^{-10}$ Ha | `crates/ferric-cc/tests/validation_uccsd.rs` |
| E_(T) and each class vs PySCF `uccsd_t` | same | $\le 2.2\times10^{-12}$ Ha | $2\times10^{-11}$ Ha | `crates/ferric-cc/tests/validation_uccsd.rs` |
| spin-blocked vs spin-orbital: energy | OH (frozen core 0, 1), NH2, CH3 / 6-31G | $\le 1.1\times10^{-11}$ Ha | $10^{-10}$ Ha | `crates/ferric-cc/tests/uccsd_anchors.rs` |
| spin-blocked vs spin-orbital: each (T) class | same | $\le 2.3\times10^{-13}$ Ha | $10^{-10}$ Ha | `crates/ferric-cc/tests/uccsd_anchors.rs` |
| spin-blocked vs spin-orbital: each amplitude block | same | $\le 2.3\times10^{-10}$ | $10^{-6}$ | `crates/ferric-cc/tests/uccsd_anchors.rs` |
| closed shell through the UHF route vs `ccsd_closed_shell` and `ccsd_t_closed_shell` | H2O / 6-31G, frozen core 0, 1 | $\le 1.3\times10^{-13}$ Ha | $10^{-10}$ Ha | `crates/ferric-cc/tests/uccsd_anchors.rs` |

The PySCF references come from `scripts/validation/gen_uccsd.py`
(`testdata/reference/validation/uccsd/`). It builds DF-UCCSD on the same
aux basis and checks the built aux set. The spin split and the (T) classes
come from the `uccsd_t_slow` formulas, each summed back to PySCF's total to
$10^{-10}$ Ha.

**Recorded mutations**, each failing on an assertion: a dropped
$t_{iJ}^{aB}$ ring term ($1.1\times10^{-2}$ Ha), an aab weight of $1/6$
($2.2\times10^{-4}$ Ha), an α orbital energy on a β (T) index
($6.2\times10^{-6}$ Ha) and a dropped $t_I^A$ ($1.6\times10^{-4}$ Ha) fail
the spin-orbital comparison. The two Rust solvers share the RI factors and
orbital energies. So giving the α virtual block the β orbital energies
passes the spin-orbital comparison; only the PySCF test catches it
($2.3\times10^{-3}$ Ha).

## Sources {#uccsd-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "Unrestricted CCSD and CCSD(T)" | the spin blocks, the $t_2$ sign table, the energy split and the class weights |
| PySCF `cc/uccsd.py::update_amps`, `cc/uccsd_t_slow.py` | the residual factorization and the per-class (T) reference |
| `scripts/cc_uccsd_spin_blocks_sympy.py` | claims 1–5 |
| `scripts/validation/gen_uccsd.py` | the PySCF DF-UCCSD and UCCSD(T) references |

**Known limits (2026-11-16).** UHF reference only: no ROHF-based CC. Energy
only: there is no UCCSD nuclear gradient, so `optimize` and `frequencies`
refuse an open-shell `ccsd` or `ccsd(t)`. The three `vvvv` blocks are held
dense and the `[cc] ladder` choice of section @sec:closed-shell-ccsd does not apply (the CLI
warns; Python refuses a ladder other than `"auto"`). `ccd` and `eom-ccsd`
have no open-shell route. The PySCF comparison is an ignored test and runs
only on request.
