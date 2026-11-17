# (T) in spin orbitals and in closed-shell form {#triples}

<!-- status: ready; registry: ccsd-t -->

CCSD lets electrons correlate in pairs and through orbital relaxation, but
it has no amplitude for three electrons moving together. Iterating triples amplitudes (CCSDT) costs $O(N^8)$ per iteration. The
perturbative triples correction instead estimates the triples once from the
converged CCSD amplitudes, in one non-iterative $O(N^7)$ pass. ferric has it
twice: a spin-orbital form that streams one occupied triple at a time, and a
spin-adapted closed-shell form that bands over virtual triples as PySCF's
`ccsd_t.c` does; the CLI and Python run the closed-shell form, and the
spin-orbital one is its cross-check.

The two forms band differently, and copying one banding into the other gives
a wrong energy (the negative control of equation @eq:cc-t-band-weights). In spin orbitals an
antisymmetrized amplitude with a repeated index vanishes, so the loop can
visit $i<j<k$ once with weight 6. In spatial orbitals each orbital holds two
electrons, $i = j$ and $a = b$ are physical, and the band $a\le b\le c$
needs multiplicity weights $1, 3, 6$ with divisor 3. The section derives
both rules:

1. the spin-orbital energy and its divisor 6 (@sec:triples-spin-orbital-t);
2. the closed-shell energy, its pair invariance and the banding (@sec:triples-closed-shell-t);
3. the two loops (@sec:triples-algorithms).

| Index | Range | Code name |
|---|---|---|
| $i,j,k$ (spin-orbital) | $0..2n_o$, visited $i<j<k$ | `unique_occ_triples` |
| $i,j,k$ (closed shell) | $0..n_o$, all orderings at once | the `[no,no,no]` block |
| $a,b,c$ (closed shell) | $0..n_v$, banded $a\le b\le c$ | `virtual_pairs`, then $c = b..n_v$ |
| $m(a,b,c)$ | 1, 3, 6 | `triple_multiplicity` |

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/ccsd_t_closed_shell.rs::ccsd_t_closed_shell`, `ferric-cc/src/ccsd_t.rs::ccsd_t` |
| CLI | `method.kind = "ccsd(t)"` (`examples/water-ccsd-t.toml`) |
| Python | `run_ccsd_t` |

## Spin-orbital (T) {#triples-spin-orbital-t}

**Where the triples come from.** Take a converged doubles amplitude
$t_{jk}^{ae}$ and let one integral act on it. $\langle bc\|ei\rangle$ moves
the particle $e$ into the pair $b,c$ and opens a hole in $i$: three holes
$i,j,k$ and three particles $a,b,c$, a triple excitation. The hole-side
analogue uses $t_{im}^{bc}$ and $\langle ma\|jk\rangle$. Antisymmetrized
over which hole and which particle play which role, the result is the connected
block (`ferric-cc/src/ccsd_t.rs`, module doc; PySCF `gccsd_t_slow`
convention)

$$W_{ijk}^{abc} = P(i/jk)\,P(a/bc)\Big[\sum_e t_{jk}^{ae}\langle bc\|ei\rangle - \sum_m t_{im}^{bc}\langle ma\|jk\rangle\Big],$$

with $P(i/jk)x = x - x_{i\leftrightarrow j} - x_{i\leftrightarrow k}$.
Dividing by the triples denominator gives the triples amplitude the doubles
drive, $W/(-\Delta)$ (the code's `t3c`). A singles amplitude times a doubles
integral gives a second, disconnected block (the code's `t3d` is
$V/(-\Delta)$):

$$V_{ijk}^{abc} = P(i/jk)\,P(a/bc)\big[t_i^a \langle bc\|jk\rangle\big].$$

The energy pairs both with the connected block over the denominator:

$$E_{(T)} = -\frac{1}{36}\sum_{ijkabc} \frac{(W + V)_{ijk}^{abc}\, W_{ijk}^{abc}}{\Delta_{ijk}^{abc}} .$$

**The divisor 6.** Swap two occupied labels. $W$ and $V$ each change sign,
because both are antisymmetrized, and $\Delta$ does not depend on the
order. The summand has two such factors, so it is invariant. The same holds
for any permutation of $(i,j,k)$ and, separately, of $(a,b,c)$. Two more
facts follow:

- A distinct occupied triple and a distinct virtual triple appear in
  $6\times6 = 36$ orderings, all with the same value. The $\tfrac1{36}$
  counts each unordered pair of triples once.
- If two occupied labels are equal, the antisymmetric $W$ is zero there, and
  so is the summand.

So the loop visits each $i<j<k$ once, lets $a,b,c$ run over the full axis
(six orderings of each virtual triple), and divides by 6 instead of 36.
A two-electron system has $2n_o = 2$ and no triple $i<j<k$ at all, so its
(T) is zero (`test_ccsd_t_h2_sto3g_is_zero`), as it must be when CCSD is
already exact (section @sec:ccd-ccsd-spin-orbital).

**Streaming.** For each $i<j<k$ the loop forms only the
$[2n_v,2n_v,2n_v]$ blocks of $W$, $V$ and $\Delta$: two GEMMs per raw block
(`raw_w_block`) and an outer product (`raw_v_block`), antisymmetrized by the
three signed occupied and three signed virtual permutations
(`triple_block`). The precomputed integral blocks are $O(n_on_v^3)$; nothing
of size $(n_on_v)^3$ is held. The final fold and the divisor, as they stand
in `ccsd_t` (the comment above the return removed):

<!-- excerpt: ferric-cc/src/ccsd_t.rs::ccsd_t -->
```rust
    // Serial fold in ascending triple order (the determinism anchor).
    let mut et = 0.0f64;
    for p in partials {
        et += p;
    }
```

followed by `Ok(et / 6.0)`. Each `partial` is one $i<j<k$ triple summed over
all $a,b,c$.

## Closed-shell (T) {#triples-closed-shell-t}

`ferric-cc/src/ccsd_t_closed_shell.rs` computes the same correction for a
closed-shell RHF-reference CCSD in spatial orbitals (Rendell, Lee and
Komornicki, CPL 178, 462 (1991); index convention of PySCF
`cc/ccsd_t_slow.py`). It needs the spatial amplitudes of
`ccsd_closed_shell` and rejects spin-orbital ones. Indices $i,j,k,l$ run
over the $n_o$ active occupied orbitals, $a,b,c,d$ over the $n_v$ virtuals,
and all integrals are chemist RI integrals.

**Definition.** Write $o=(i,j,k)$, $v=(a,b,c)$ and, for a permutation
$\pi\in S_3$, $o_\pi = (o_{\pi(0)},o_{\pi(1)},o_{\pi(2)})$, likewise $v_\pi$.
The raw connected block is

$$w^{(0)}_{ijk}{}^{abc} = \sum_d (ia|bd)\,t_{kj}^{cd} - \sum_l (ia|lj)\,t_{lk}^{bc}.$$

It plays the role of the bracket in the spin-orbital $W$: a doubles
amplitude and one integral. In spatial orbitals there is no
antisymmetrizer. Instead $W$ sums the raw block over the six simultaneous
permutations of the (occupied, virtual) pairs $(i,a),(j,b),(k,c)$, and $V$
adds the singles:

$$W_{o}^{v} = \sum_{\pi\in S_3} w^{(0)}{}_{o_\pi}^{v_\pi},\qquad
V_{ijk}^{abc} = W_{ijk}^{abc} + (jb|kc)\,t_i^a + (ia|kc)\,t_j^b + (ia|jb)\,t_k^c .$$

The $f_{ai}\,t_2$ piece of PySCF's `get_v` is absent because $f_{ai}=0$ for
canonical HF orbitals. Spin enters through PySCF's `r3`, a fixed combination
of the virtual labels at fixed $o$:

$$\tilde W_{ijk}^{abc} = 4W_{ijk}^{abc} + W_{ijk}^{bca} + W_{ijk}^{cab}
 - 2W_{ijk}^{cba} - 2W_{ijk}^{acb} - 2W_{ijk}^{bac},$$

and the energy is the unrestricted sum

$$E_{(T)} = \frac13 \sum_{ijk}\sum_{abc} f_{ijk}^{abc},\qquad
f_{ijk}^{abc} = -\frac{\tilde W_{ijk}^{abc}\,V_{ijk}^{abc}}{\Delta_{ijk}^{abc}} .$$

The divisor is 3, not $1/36$ or $1/6$: the spin summation behind the
$4,1,1,-2,-2,-2$ coefficients has already absorbed most of the
spin-orbital antisymmetrizer's redundancy. This sum equals the spin-orbital
$(T)$ of section @sec:triples-spin-orbital-t on the spin-orbital expansion of the same amplitudes
and integrals (claim 4 of the box below).

**Pair invariance.** Let $\sigma\in S_3$ act on both triples at once.

- $W_{o_\sigma}^{v_\sigma} = W_o^v$, because $\pi\mapsto\pi\circ\sigma$ is a
  bijection of $S_3$: the sum visits the same six raw blocks.
- $V$: the three singles terms map onto each other, using $(pq|rs)=(rs|pq)$.
  For $(i,a)\leftrightarrow(j,b)$, $(jb|kc)t_i^a \leftrightarrow (ia|kc)t_j^b$
  and $(ia|jb)t_k^c$ is fixed.
- $\Delta$ is symmetric in $o$ and in $v$ separately.
- $\tilde W_o^v = \sum_\tau c_\tau W_o^{v_\tau}$ with $c_\tau = 4$ for the
  identity, $1$ for the two 3-cycles and $-2$ for the three transpositions.
  Then $\tilde W_{o_\sigma}^{v_\sigma} = \sum_\tau c_\tau W_o^{v_{\sigma\tau\sigma^{-1}}}$
  by pair invariance of $W$, and this is $\tilde W_o^v$ because $c_\tau$ is
  constant on conjugacy classes.

So $f_{o_\sigma}^{v_\sigma} = f_o^v$. Pair invariance also moves the `r3`
pattern from the virtual labels to the occupied ones, which is the form the
kernel uses at a fixed virtual triple:
$W_{ijk}^{bca} = W_{kij}^{abc}$, $W_{ijk}^{cab} = W_{jki}^{abc}$,
$W_{ijk}^{cba} = W_{kji}^{abc}$, $W_{ijk}^{acb} = W_{ikj}^{abc}$,
$W_{ijk}^{bac} = W_{jik}^{abc}$, so

$$\tilde W_{ijk}^{abc} = 4W_{ijk}^{abc} + W_{kij}^{abc} + W_{jki}^{abc}
 - 2W_{kji}^{abc} - 2W_{ikj}^{abc} - 2W_{jik}^{abc}.$$

**Virtual-triple banding.** Define $g(a,b,c) = \sum_{ijk} f_{ijk}^{abc}$, with
$i,j,k$ over the full occupied axes. Relabel the dummy $(i,j,k)$ by the
same $\sigma$: $g$ is symmetric in $(a,b,c)$ alone. Each multiset
$\{a,b,c\}$ therefore contributes $m(a,b,c)$ equal terms to $\sum_{abc} g$,
where $m$ counts its distinct orderings:

$$\sum_{abc} g(a,b,c) = \sum_{a\le b\le c} m(a,b,c)\, g(a,b,c),\qquad
m(a,b,c) = \begin{cases} 1 & a=b=c \\ 3 & \text{exactly two equal} \\ 6 & \text{all distinct,}\end{cases}$$

and $\sum_{a\le b\le c<n_v} m = n_v^3$. Hence

$$E_{(T)} = -\frac13 \sum_{a\le b\le c} m(a,b,c) \sum_{ijk}
\frac{\tilde W_{ijk}^{abc}\,V_{ijk}^{abc}}{\Delta_{ijk}^{abc}} .$$

**Why the spin-orbital banding does not transfer.** Here $a,b,c$ (and
$i,j,k$) label spatial orbitals that each hold two electrons, so $a=b$ or
$i=j$ gives a nonzero contribution: an α and a β electron excited into the
same spatial orbital. In the spin-orbital formulation the antisymmetrized
amplitude vanishes for a repeated index, which is why that path can band
over $i<j<k$ with a single weight. Transplanting that banding here (strict
inequalities, weight 6) drops these terms. On a repeated virtual triple the
kernel still sums all six permutations in $W$, some of them identical, as
the definition requires; $m$ only counts how many orderings of the multiset
the unrestricted sum would visit.

**The all-equal limit.** At $a=b=c$ all six $W^{v_\tau}$ in $\tilde W$ are
the same number, and the coefficients sum to $4+1+1-2-2-2 = 0$, so
$\tilde W = 0$ and $g(a,a,a) = 0$. The same argument on the occupied-axis
form gives $f_{iii}^{abc} = 0$. Three electrons cannot be excited into, or
out of, one spatial orbital, and the spin-adapted weights encode that. The
weight 1 therefore multiplies zero. The code evaluates the term anyway, so
the band reproduces the unrestricted sum term for term, which is what the box
below checks. A single occupied orbital ($n_o = 1$, two electrons) has only the
triple $i=j=k$, so this form also gives zero (T) there. (Both zeros were
checked on the random model of the proof script: $|f_{iii}|$ and
$|g(a,a,a)|$ below $3\times10^{-17}$, against $|f_{001}| = 0.022$.)

The excerpt below is the weight:

<!-- excerpt: ferric-cc/src/ccsd_t_closed_shell.rs::triple_multiplicity -->
```rust
if a == c {
    1.0 // a == b == c
} else if a == b || b == c {
    3.0
} else {
    6.0
}
```

`a == c` with $a\le b\le c$ forces all three equal; the band order makes the
two-equal test two comparisons.

> **Verified equation {#eq:cc-t-band-weights}** · the band weights and the divisor 3
>
> Symbolically, $\sum_{a\le b\le c<n} m(a,b,c) = n^3$. On random
> closed-shell amplitudes and 8-fold-symmetric RI integrals, the per-triple
> sum is the same for all orderings of its labels, the $m$-weighted band
> equals the unrestricted sum, and the unrestricted sum over 3 equals the
> dense spin-orbital (T), $\tfrac1{36}\sum(W+V)W/\Delta$, on the spin-orbital
> expansion of the same amplitudes and integrals. The spin-orbital banding
> transplanted ($i<j<k$, weight 6), divisor 6, unit weights and a $\tilde W$
> without the cyclic terms each fail.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`, claims 1, 2, 3, 4, 5 |
> | Negative control | claim 5 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/ccsd_t_closed_shell.rs::{triple_multiplicity,virtual_pairs,pair_job_energy}`, `ferric-cc/src/dlpno_ccsd_t.rs::occ_triple_multiplicity` |
> | Code test | `multiplicity_weights_sum_to_the_unrestricted_count` in `crates/ferric-cc/src/ccsd_t_closed_shell.rs` |
>
> <!-- verified: id=cc.t.band-weights; script=scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/ccsd_t_closed_shell.rs::{triple_multiplicity,virtual_pairs,pair_job_energy}; rust_test=ferric-cc/src/ccsd_t_closed_shell.rs::multiplicity_weights_sum_to_the_unrestricted_count; registry=ccsd-t -->

**The kernel.** The code maps onto these equations as follows.

- Layouts (`ccsd_t_closed_shell`, built once as `TripleIntegrals`):
  `ovvv[i,a,b,d]` $=(ia|bd)$, `vooo[a,i,j,l]` $=(ia|lj)$,
  `ovov[i,a,j,b]` $=(ia|jb)$, `t2x[c,d,j,k]` $=t_{kj}^{cd}$,
  `t1t[a,i]` $=t_i^a$. The denominators use the canonical HF orbital
  energies of the active occupied and virtual orbitals.
- `raw_w_pair` evaluates $r_\pi[n] = w^{(0)}{}_n^{v_\pi}$ for every occupied
  $n=(n_0,n_1,n_2)$ at once, for two permutations that share the third
  virtual label (`PERM_PAIRS`). Term 1 is one GEMM of the stacked
  `[ovvv[:,x0,y0,:]; ovvv[:,x1,y1,:]]` with `t2x[z]`; term 2 is `vooo[x]`
  times the transpose of `t2x[y,z]`, scaled by $-1$.
- `triple_energy(a,b,c)` accumulates $W_n = \sum_\pi r_\pi[n_\pi]$, the raw
  block read with its occupied axes permuted by $\pi^{-1}$. One fused pass
  over $(i,j,k)$ then forms $V$, the occupied-axis $\tilde W$ above and
  $-\tilde W V/\Delta$ (as $\tilde W V$ over the stored $-\Delta$), and
  returns $g(a,b,c)$.
- `pair_job_energy(a,b)` sums `triple_multiplicity(a,b,c)`
  $\times\,g(a,b,c)$ over $c=b,\dots,n_v-1$. Jobs are the pairs $a\le b$
  (`virtual_pairs`), folded serially in ascending order, so the result does
  not depend on the thread count.
- `ccsd_t_closed_shell` returns the folded sum divided by 3.

The fused pass of `ferric-cc/src/ccsd_t_closed_shell.rs::triple_energy`, at
one occupied triple (the loop headers and the two lines that set the flat
index `n` and `w_ijk = w[n]` removed; comments added):

```rust
let wt = 4.0 * w_ijk + w[kk * nn + i * no + j] + w[j * nn + kk * no + i]  // 4W_ijk + W_kij + W_jki
    - 2.0
        * (w[kk * nn + j * no + i]                                         // -2(W_kji
            + w[i * nn + kk * no + j]                                      //   + W_ikj
            + w[j * nn + i * no + kk]);                                    //   + W_jik)
let v = w_ijk
    + s_bc[j * no + kk] * t1a[i]                                           // (jb|kc) t_i^a
    + s_ac[i * no + kk] * t1b[j]                                           // (ia|kc) t_j^b
    + s_ab[i * no + j] * t1c[kk];                                          // (ia|jb) t_k^c
partial += wt * v / (k.eijk[n] - e_abc);                                   // W~ V / (-Delta)
```

- `w` is $W$ at the fixed $(a,b,c)$ as a flat `[no,no,no]` array, so each
  occupied-axis permutation of $\tilde W$ is a different stride pattern into
  the same block.
- `k.eijk[n] - e_abc` is $\varepsilon_i+\varepsilon_j+\varepsilon_k-\varepsilon_a-\varepsilon_b-\varepsilon_c = -\Delta$.
  The minus sign of $f$ is in this denominator; `partial` is $g(a,b,c)$.
- `kk` is the occupied index $k$, renamed because `k` holds the kernel
  inputs.

> **Verified equation {#eq:cc-t-kernel}** · the production kernel
>
> A port of the virtual-banded kernel (raw blocks per pair permutation,
> $W$ accumulated with its occupied axes permuted back, $V$ and the
> occupied-axis $\tilde W$ at fixed $(a,b,c)$, weights $m$, divisor 3)
> equals the dense spin-orbital (T) to a few $10^{-16}$ relative.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`, claim 4b |
> | Negative control | the runner's mutations: $2W$ for $4W$ in $\tilde W$; $(jc|kb)$ for $(jb|kc)$ in $V$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/ccsd_t_closed_shell.rs::{raw_w_pair,triple_energy,pair_job_energy,ccsd_t_closed_shell}` |
> | Code test | `closed_shell_t_matches_spin_orbital_t_h2o_ccpvdz` in `crates/ferric-cc/src/ccsd_t_closed_shell.rs` (both forms fed the same amplitudes) |
>
> <!-- verified: id=cc.t.kernel; script=scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py; claim=4b; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/ccsd_t_closed_shell.rs::{raw_w_pair,triple_energy,pair_job_energy,ccsd_t_closed_shell}; rust_test=ferric-cc/src/ccsd_t_closed_shell.rs::closed_shell_t_matches_spin_orbital_t_h2o_ccpvdz; registry=ccsd-t -->

**DLPNO-(T).** `ferric-cc/src/dlpno_ccsd_t.rs::screened_triple_energy` keeps
the occupied-banded form of the same identity, because it screens occupied
triples and so needs $f$ summed per occupied triple. With
$h(i,j,k) = \sum_{abc} f_{ijk}^{abc}$ (the `r3` pattern on the virtual
labels; `dlpno_ccsd_t_kernel.rs` evaluates it per triple), $h$ is symmetric
in $(i,j,k)$ by the same argument, and `screened_triple_energy` forms
$\frac13\sum_{i\le j\le k} m(i,j,k)\,h(i,j,k)$ over the retained triples,
with `occ_triple_multiplicity` giving the same $1,3,6$ weights. The screen
retains or drops whole multisets $\{i,j,k\}$ and always keeps the
repeated-index classes, so the weights stay exact (section @sec:local-cc-dlpno-t-triple-screen).

**What the proofs cover.** The script proves $\sum_{a\le b\le c} m = n^3$
symbolically (claim 1). On random closed-shell amplitudes and
8-fold-symmetric RI integrals it checks that $h$ is the same for all
orderings of $(i,j,k)$ (claim 2), that the occupied banding equals the
unrestricted sum (claim 3), and that the unrestricted sum over 3 equals the
dense spin-orbital $(T)$ (claim 4); together these cover the DLPNO-(T) form.
Claim 4b ports the dense kernel and claim 5 shows that the spin-orbital
banding, divisor 6, unit weights and a $\tilde W$ without the cyclic terms
all fail. In Rust, the `ccsd_t_closed_shell` tests compare with the
spin-orbital `ccsd_t` given the same amplitudes and with PySCF. The
spin-orbital divisor 6 has no sympy claim of its own; its evidence is the
streaming-versus-dense regression `streaming_matches_dense_h2o_ccpvdz` and
the PySCF comparisons below.

## The algorithms {#triples-algorithms}

**Closed shell** (`ccsd_t_closed_shell`).

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | spin-adapted CCSD (section @sec:closed-shell-ccsd); spin-orbital amplitudes are refused with a typed error | | `ccsd_closed_shell` |
| 2 | `ovvv`, `vooo`, `ovov`, `t2x`, `t1t` built once | $N_{\rm aux}n_on_v^3$ | `TripleIntegrals` |
| 3 | per job $(a,b)$, $a \le b$, for $c = b..n_v$: two stacked GEMMs per permutation pair | $n_o^3n_v$ per triple | `raw_w_pair` |
| 4 | one fused pass over $(i,j,k)$: $W$, $V$, $\tilde W$, $-\tilde WV/\Delta$; $g(a,b,c)$ | $n_o^3$ | `triple_energy` |
| 5 | $\sum_c m(a,b,c)\,g(a,b,c)$ | | `pair_job_energy` |
| 6 | fold jobs in ascending order, divide by 3 | | `ccsd_t_closed_shell` |

The jobs run in parallel and are folded serially, so the result does not
depend on the thread count (`thread_count_bit_identical_h2o_ccpvdz`). The
band width is set from the memory budget (`t_band_width`); it changes the
time, never the energy.

**Why band over virtuals.** When $n_v$ is several times $n_o$, a virtual
triple carries an $[n_o,n_o,n_o]$ block, small enough to stay in cache
while the six pair permutations, $V$, $\tilde W$ and the denominator are
applied in passes over it. Banding over occupied triples would carry an
$[n_v,n_v,n_v]$ block instead, which is what the spin-orbital loop does.

**Spin orbitals** (`ccsd_t`): the same shape over $i<j<k$ with divisor 6,
from spin-orbital amplitudes (section @sec:ccd-ccsd-spin-orbital, or the closed-shell ones expanded
by `expand_amplitudes_to_spin_orbital`).

## Numerical anchors {#triples-numerical-anchors}

| Quantity | Systems | Error | Test |
|---|---|---|---|
| (T), spin-orbital and spin-adapted, vs PySCF `ccsd_t` on the same DF integrals | H2O / cc-pVDZ, aug-cc-pVDZ; HCN / cc-pVDZ; C2H6 / cc-pVDZ (spin-adapted only) | $\le 6.9\times10^{-12}$ Ha | `crates/ferric-cc/tests/validation_cc.rs` |
| the two (T) codes against each other | same | $8.9\times10^{-12}$ Ha | `crates/ferric-cc/tests/validation_cc.rs` |
| anions, all-electron and frozen core | CH3⁻, Te²⁻ / def2-SVP | $\le 2.7\times10^{-12}$ Ha | `crates/ferric-cc/tests/cc_anion_occupied_count.rs` |

## Sources {#triples-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, the two (T) sections | the definitions, pair invariance and the banding |
| Rendell, Lee and Komornicki, CPL 178, 462 (1991); PySCF `cc/ccsd_t_slow.py` | the closed-shell algorithm and its index convention |
| `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py` | claims 1–5 and 4b |

**Known limits (2026-11-15).** Closed shell only; the open-shell (T) on a
UHF reference is section @sec:uccsd. The
DLPNO-(T) screen of section @sec:local-cc keeps the occupied banding of the same
identity.
