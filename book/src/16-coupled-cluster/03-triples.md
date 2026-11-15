# 16.3 (T) in spin orbitals and in closed-shell form

<!-- section: 16.3; status: ready; registry: ccsd-t;
     theory: crates/ferric-cc/theory.md#"CCSD(T), spin-orbital", "CCSD(T), closed shell" -->

The perturbative triples correction is one non-iterative $O(N^7)$ pass over
the converged CCSD amplitudes. ferric has it twice: a spin-orbital form that
streams one occupied triple at a time, and a spin-adapted closed-shell form
that bands over virtual triples as PySCF's `ccsd_t.c` does. The CLI and
Python run the closed-shell form; the spin-orbital one is its cross-check.

The two forms band differently, and copying one banding into the other gives
a wrong energy (the negative control of equation 16.4). In spin orbitals an antisymmetrized amplitude with a repeated index
vanishes, so the loop can visit $i<j<k$ once with weight 6. In spatial
orbitals each orbital holds two electrons, $i = j$ and $a = b$ are physical,
and the band $a\le b\le c$ needs multiplicity weights $1, 3, 6$ with divisor
3.

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

## 16.3.1 Spin-orbital (T)

<!-- include: crates/ferric-cc/theory.md#"CCSD(T), spin-orbital" -->

The streaming loop forms, for each $i<j<k$, only the $[2n_v,2n_v,2n_v]$
blocks of $W$, $V$ and $\Delta$ from two GEMMs (`raw_w_block`) and an outer
product (`raw_v_block`), antisymmetrized by the three signed occupied and
three signed virtual permutations (`triple_block`). The precomputed
integral blocks are $O(n_on_v^3)$; nothing of size $(n_on_v)^3$ is held.

## 16.3.2 Closed-shell (T)

<!-- include: crates/ferric-cc/theory.md#"CCSD(T), closed shell" -->

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

> **Verified equation 16.4** · the band weights and the divisor 3
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

> **Verified equation 16.5** · the production kernel
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

## 16.3.3 The algorithms

**Closed shell** (`ccsd_t_closed_shell`).

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | spin-adapted CCSD (section 16.2); spin-orbital amplitudes are refused with a typed error | | `ccsd_closed_shell` |
| 2 | `ovvv`, `vooo`, `ovov`, `t2x`, `t1t` built once | $N_{\rm aux}n_on_v^3$ | `TripleIntegrals` |
| 3 | per job $(a,b)$, $a \le b$, for $c = b..n_v$: two stacked GEMMs per permutation pair | $n_o^3n_v$ per triple | `raw_w_pair` |
| 4 | one fused pass over $(i,j,k)$: $W$, $V$, $\tilde W$, $-\tilde WV/\Delta$; $g(a,b,c)$ | $n_o^3$ | `triple_energy` |
| 5 | $\sum_c m(a,b,c)\,g(a,b,c)$ | | `pair_job_energy` |
| 6 | fold jobs in ascending order, divide by 3 | | `ccsd_t_closed_shell` |

The jobs run in parallel and are folded serially, so the result does not
depend on the thread count (`thread_count_bit_identical_h2o_ccpvdz`). The
band width is set from the memory budget (`t_band_width`); it changes the
time, never the energy.

**Spin orbitals** (`ccsd_t`): the same shape over $i<j<k$ with divisor 6, from
spin-orbital amplitudes (section 16.1, or the closed-shell ones expanded by
`expand_amplitudes_to_spin_orbital`).

## 16.3.4 Numerical anchors

| Quantity | Systems | Error | Test |
|---|---|---|---|
| (T), spin-orbital and spin-adapted, vs PySCF `ccsd_t` on the same DF integrals | H2O / cc-pVDZ, aug-cc-pVDZ; HCN / cc-pVDZ; C2H6 / cc-pVDZ (spin-adapted only) | $\le 6.9\times10^{-12}$ Ha | `crates/ferric-cc/tests/validation_cc.rs` |
| the two (T) codes against each other | same | $8.9\times10^{-12}$ Ha | `crates/ferric-cc/tests/validation_cc.rs` |
| anions, all-electron and frozen core | CH3⁻, Te²⁻ / def2-SVP | $\le 2.7\times10^{-12}$ Ha | `crates/ferric-cc/tests/cc_anion_occupied_count.rs` |

## 16.3.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, the two (T) sections | the definitions, pair invariance and the banding |
| Rendell, Lee and Komornicki, CPL 178, 462 (1991); PySCF `cc/ccsd_t_slow.py` | the closed-shell algorithm and its index convention |
| `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py` | claims 1–5 and 4b |

**Known limits (2026-11-15).** Closed shell only; no open-shell (T). The
DLPNO-(T) screen of section 16.7 keeps the occupied banding of the same
identity.
