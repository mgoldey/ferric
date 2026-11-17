# Closed-shell CCSD and the particle-particle ladder {#closed-shell-ccsd}

<!-- status: ready; registry: ccsd -->

In a closed-shell molecule every spatial orbital holds an α and a β
electron, and the two spins are excited in mirror-image ways. The
spin-orbital amplitudes of section @sec:ccd-ccsd-spin-orbital therefore repeat themselves: all of
them follow from $n_o n_v$ singles and $n_o^2n_v^2$ doubles over spatial
orbitals. The spin-adapted solver iterates only those, which makes every
rank-4 tensor 16 times smaller than its spin-orbital counterpart, and it is
what `ccsd`, `ccsd(t)` and `eom-ccsd` run.

| Index | Range | Code name |
|---|---|---|
| $i, j, k, l$ | active occupied spatial orbitals | `0..no` |
| $a, b, c, d$ | virtual spatial orbitals | `0..nv` |
| $t_i^a$, $t_{ij}^{ab}$ | spatial amplitudes, $t_{ij}^{ab} = t_{ji}^{ba}$ | `t1[i,a]`, `t2[i,j,a,b]` |

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/ccsd_closed_shell.rs::{ccsd_closed_shell,expand_amplitudes_to_spin_orbital,plan_ladder}`, `ferric-cc/src/ri_ladder.rs::ri_ladder_contract` |
| CLI | `method.kind = "ccsd"` (`examples/water-ccsd.toml`), `[cc] ladder` |
| Python | `run_ccsd(ladder=)` |

## Spin adaptation {#closed-shell-ccsd-spin-adaptation}

**The map.** For an RHF reference the spin-orbital amplitudes are fixed by
spatial ones. `expand_amplitudes_to_spin_orbital` writes them out, the
expansion (@eq:spin-adaptation-4) of section @sec:spin-adaptation-closed-shell-amplitudes:

$$ t_{i\sigma}^{a\tau} = \delta_{\sigma\tau}t_i^a,\qquad
t_{i\sigma j\tau}^{a\sigma' b\tau'} = \delta_{\sigma\sigma'}\delta_{\tau\tau'}t_{ij}^{ab} - \delta_{\sigma\tau'}\delta_{\tau\sigma'}t_{ij}^{ba},\qquad
t_{ij}^{ab} = t_{ji}^{ba} . $$

Read off the blocks. The singles are the same for α and β, and a spin-flip
single is zero. The opposite-spin block,
$t_{i\alpha j\beta}^{a\alpha b\beta}$, is $t_{ij}^{ab}$ itself: only the
first delta pair holds. The same-spin block is $t_{ij}^{ab} - t_{ij}^{ba}$,
antisymmetric in $a,b$ as it must be. So $t_{ij}^{ab}$ is the amplitude of
an α electron going $i\to a$ while a β electron goes $j\to b$. The pairing
is Coulomb order, $i\leftrightarrow a$ and $j\leftrightarrow b$: the MP2
guess is $(ia|jb)/(-\Delta)$, and the opposite-spin amplitude with $a$ on
the α electron is $+t_{ij}^{ab}$.

**The residual keeps the pattern.** Feed amplitudes of this form into the
spin-orbital residual of section @sec:ccd-ccsd-spin-orbital. The result has the same structure:
the α and β singles are equal, the opposite-spin block
$R_{ij}^{ab} = R_{i\alpha j\beta}^{a\alpha b\beta}$ has
$R_{ij}^{ab} = R_{ji}^{ba}$, and the same-spin block is
$R_{ij}^{ab} - R_{ij}^{ba}$. So one spatial singles residual $R_i^a$ and one
spatial doubles residual $R_{ij}^{ab}$ carry all of it, and solving
$R_i^a = 0$, $R_{ij}^{ab} = 0$ is solving CCSD.

**The energy.** Combine the singles into the doubles as in section @sec:ccd-ccsd-spin-orbital:
the spin-orbital energy is $\tfrac14\sum\langle ij\|ab\rangle\tau_{ij}^{ab}$.
On the spatial map the opposite-spin $\tau$ is
$t_{ij}^{ab} + t_i^at_j^b$ (the spin-flip singles that would subtract are
zero), and the same-spin one is
$(t_{ij}^{ab} - t_{ij}^{ba}) + t_i^at_j^b - t_i^bt_j^a$. Write
$\tau_{ij}^{ab} = t_{ij}^{ab} + t_i^at_j^b$ for the spatial quantity. Then
the same-spin block is $\tau_{ij}^{ab} - \tau_{ij}^{ba}$, so the spatial
$\tau$ expands by the same map as $t$. The count of section @sec:spin-adaptation-closed-shell-amplitudes then
applies with $\tau$ in place of $t$:

- the four opposite-spin patterns give $\sum(ia|jb)\,\tau_{ij}^{ab}$;
- the two same-spin patterns give $\sum[(ia|jb) - (ib|ja)]\,\tau_{ij}^{ab}$,

and their sum is

$$E_c = \sum_{ijab}\big(2(ia|jb) - (ib|ja)\big)\big(t_{ij}^{ab} + t_i^at_j^b\big).$$

At $t_1 = 0$ and $t_{ij}^{ab} = (ia|jb)/(-\Delta_{ij}^{ab})$ the energy
reduces to the closed-shell MP2 energy (@eq:closed-shell-ri-mp2-3), with the opposite-spin and same-spin pieces
of (@eq:mp2-cs-spin-adaptation) as the two bullets.

> **Verified equation {#eq:cc-ccsd-spin-adaptation}** · the closed-shell residual is the opposite-spin block
>
> For random spatial $t_1$ and $t_2 = t_2^{(ij)(ab)}$, expanded to spin
> orbitals by the map of `expand_amplitudes_to_spin_orbital`, the brute-force
> projections $\langle\Phi_\mu|e^{-\hat T}\hat He^{\hat T}|\Phi_0\rangle$ give equal
> α and β singles and no spin-flip singles, an opposite-spin block with
> $R_{ij}^{ab} = R_{ji}^{ba}$, a same-spin block equal to
> $R_{ij}^{ab} - R_{ij}^{ba}$, and the energy
> $\sum(2(ia|jb) - (ib|ja))(t_{ij}^{ab} + t_i^at_j^b)$.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_ccsd_eom_brute_force_sympy.py`, check `spin-adaptation` |
> | Negative control | the runner's mutation: the exchange integral dropped from the energy |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/ccsd_closed_shell.rs::{ccsd_closed_shell,expand_amplitudes_to_spin_orbital}` |
> | Code test | `ccsd_spin_adapted_matches_df_reference` in `crates/ferric-cc/tests/validation_cc.rs` |
>
> <!-- verified: id=cc.ccsd.spin-adaptation; script=scripts/cc_ccsd_eom_brute_force_sympy.py; claim=spin-adaptation; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/ccsd_closed_shell.rs::{ccsd_closed_shell,expand_amplitudes_to_spin_orbital}; rust_test=ferric-cc/tests/validation_cc.rs::ccsd_spin_adapted_matches_df_reference; registry=ccsd -->

**The factorization the code evaluates.**
`ferric-cc/src/ccsd_closed_shell.rs::ccsd_closed_shell` does not derive the
spatial residual from the spin-orbital one at run time. It evaluates the
closed-shell equations of Hirata, Podeszwa, Tobita and Bartlett, JCP 120,
2581 (2004), Eqs. 35–45, in the form of PySCF `rccsd.update_amps` and
`rintermediates`:

- the one-body intermediates $F_{oo}$, $F_{vv}$, $F_{ov}$ and $L_{oo}$,
  $L_{vv}$ (Eqs. 37–41). There is no orbital-energy shift, so $\varepsilon$
  enters only through $\Delta$;
- the two-body intermediates $W_{oooo}$, $W_{voov}$, $W_{vovo}$ (Eqs. 42–45);
- the T1 and T2 equations (Eqs. 35, 36). The T2 part is built as one half
  $x$ and symmetrized as $x + x^{(ij)(ab)}$, which enforces
  $t_{ij}^{ab} = t_{ji}^{ba}$.

$(t_1, t_2)$ are extrapolated jointly by DIIS (`diis_step_two`).

The box proves that solving the spatial equations is solving CCSD. It does
not reach the Hirata factorization the code evaluates: that is a
transcription of PySCF `rccsd`, and its evidence is the Rust comparison with
the spin-orbital `ccsd` on the same integrals and with PySCF `cc.CCSD`
(`tests/validation_cc.rs`).

## The algorithm {#closed-shell-ccsd-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF; $B^P_{ov}, B^P_{oo}, B^P_{vv}$ | $N_{\rm aux}N^2n$ | `build_b` |
| 2 | chemist blocks $(oo|oo)$, $(ov|oo)$, $(ov|ov)$, $(oo|vv)$, $(ov|vo)$, $(ov|vv)$; $(ac|bd)$ only on the dense route | $N_{\rm aux}n_on_v^3$ | |
| 3 | per iteration: $F$, $L$ and $W$ intermediates (Eqs. 37–45) | $n_o^3n_v^3$ | |
| 4 | the pp ladder $\sum_{cd}W_{abcd}\tau_{ij}^{cd}$ | $n_o^2n_v^4$ | `dense_ladder` or `ri_ladder_contract` |
| 5 | $t \leftarrow r/(-\Delta)$, joint DIIS on $(t_1, t_2)$ | | `diis_step_two` |
| 6 | energy; stop when it changes by less than $\min(\texttt{energy\_conv}, 10^{-9})$ | $n_o^2n_v^2$ | |

**The ladder.** Step 4 is the $n_o^2n_v^4$ term and the only place an
$n_v^4$ object appears. Its vertex is

$$ W_{abcd} = (ac|bd) - \sum_k (kd|ac)\,t_k^b - \sum_k (kc|bd)\,t_k^a . $$

It carries no $t_1^2$ term; that piece enters through $\tau$. The code never
forms $W_{abcd}$. The dense route contracts $(ac|bd)$ with $\tau$, and the
two $t_1$ terms with $\tau$ separately:

<!-- excerpt: ferric-cc/src/ccsd_closed_shell.rs::dense_ladder -->
```rust
    // Σ_cd (ac|bd) τ_ijcd: left-free (a,b), right-free (i,j); permute to (i,j,a,b).
    let mut z = perm(&einsum!("abcd,ijcd->abij", vacbd, tau_t), &[2, 3, 0, 1]);
    // ovvv_t holds (kd|ac) at [k,d,a,c]; Y_kaij = Σ_cd (kd|ac) τ_ijcd.
    let y = lbl4(
        einsum!("kdac,ijcd->kaij", ovvv_t, tau_t),
        [Axis::O, Axis::V, Axis::O, Axis::O],
    );
    // −Σ_k t1_kb Y_kaij: left-free (a,i,j) -> 'aijb'; permute to (i,j,a,b).
    let x1 = perm(&einsum!("kaij,kb->aijb", &y, t1_t), &[1, 2, 0, 3]);
```

- The first GEMM is the $n_o^2n_v^4$ one: $\sum_{cd}(ac|bd)\tau_{ij}^{cd}$.
- The $t_1$ terms are reassociated. $\sum_{cd}(kd|ac)\tau_{ij}^{cd}$ is
  formed first, an $n_on_v^3$ integral against $\tau$ at
  $n_o^3n_v^3$ cost, and only then multiplied by $t_k^b$. Contracting
  $t_1$ into the integral first would build the $n_v^4$ vertex.
- The function continues with the mirror term $\sum_k (kc|bd)\,t_k^a$,
  read from the same `ovvv_t` with relabelled axes (excerpt cut there).

**The RI route.** The dressed RI factors make the $t_1$ terms part of the
integral. Define the $t_1$-dressed block
$B^P_{x,bd} = -\sum_k B^P_{kd}t_k^b$. Substitute $(pq|rs) = \sum_P B^P_{pq}B^P_{rs}$
into each term of $W_{abcd}$:

$$
\begin{aligned}
(ac|bd) &= \textstyle\sum_P B^P_{vv,ac}B^P_{vv,bd}, \\
-\textstyle\sum_k (kc|bd)\,t_k^a &= \textstyle\sum_P B^P_{x,ac}\,B^P_{vv,bd}, \\
-\textstyle\sum_k (kd|ac)\,t_k^b &= \textstyle\sum_P B^P_{vv,ac}\,B^P_{x,bd},
\end{aligned}
$$

and add them:

$$ W_{abcd} = \sum_P\big[(B_{vv}+B_x)^P_{ac}B^P_{vv,bd} + B^P_{vv,ac}B^P_{x,bd}\big] . $$

There is no $B_xB_x$ term, which would be the absent $t_1^2$ piece. This is
an exact rewriting over the same $B$, so the two routes agree to round-off.
`ri_ladder_contract` rebuilds $W_{abcd}$ in panels of the first index $a$
and contracts each with $\tau$ before the next. Both routes contract the
same algebra over the same $B$:

| Route | Holds | Rebuilds per iteration |
|---|---|---|
| `dense` | $(ac|bd)$, one $n_v^4$ buffer (two during setup) | nothing; the $t_1$ terms are contracted with $\tau$ separately |
| `ri` | $B_{vv}$ and $B_x = -\sum_k B_{ov,kd}t_k^b$, $N_{\rm aux}n_v^2$ each | $(ac|bd)$ in $a_{\rm blk}\cdot n_v^3$ panels, one rayon job per $a$ |
| `auto` (default) | dense when its `MemoryPlan` fits the budget, otherwise RI | |

The term-by-term factorization is also in the module documentation of
`ri_ladder.rs`. The route taken is printed (`ladder = dense (auto)`), logged
and returned in `CcResult::ladder_route`.

**Why two routes.** The dense route holds an $n_v^4$ block, a ceiling set
by the basis size. The RI route lowers that ceiling to one panel and pays
for it in time, since it rebuilds the integrals every iteration (measured
numbers in `site/src/methods/cc.md`). `auto` takes the faster route
whenever it fits.

**Reproducibility.** Under `auto` the route depends on the memory budget,
and without an explicit `[memory] budget_gb` that is 0.8 times the RAM
available at start-up. The energy agrees to round-off either way; the time
does not. Set `ladder` or the budget for reproducible timings. The RI ladder
is bit-identical at any thread count: each job owns its own panel rows and
its own slice of the result, so nothing is reduced across threads.

## Options {#closed-shell-ccsd-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cc] ladder`, `ladder=` | `auto`, `dense`, `ri` | `auto` | pp-ladder route; unknown spellings are errors |
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | frozen occupied orbitals |
| `[memory] budget_gb`, `memory_budget_gb=` | GB | auto | preflight and the `auto` decision |

## Numerical anchors {#closed-shell-ccsd-numerical-anchors}

| Quantity | Systems | Error | Test |
|---|---|---|---|
| RI-CCSD vs PySCF `CCSD` on the same DF integrals (both solvers) | H2O, NH3, CH4, HF / cc-pVDZ, def2-SVP; HCN / cc-pVDZ; H2O / aug-cc-pVDZ; C2H6 / cc-pVDZ | $\le 4.7\times10^{-11}$ Ha | `crates/ferric-cc/tests/validation_cc.rs` |
| the same, anions whose RHF HOMO lies above zero | CH3⁻, Te²⁻ / def2-SVP | $\le 6.0\times10^{-12}$ Ha | `crates/ferric-cc/tests/cc_anion_occupied_count.rs` |

## Sources {#closed-shell-ccsd-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "CCSD, closed shell" | the spin adaptation and the energy |
| Hirata, Podeszwa, Tobita and Bartlett, JCP 120, 2581 (2004), Eqs. 35–45 | the factorization the code transcribes |
| `site/src/methods/cc.md`, "Particle-particle ladder: dense or RI" | measured memory and time of the two routes |

**Known limits (2026-11-15).** Closed shell only; the RI route still holds
the $n_on_v^3$ $(ia|bc)$ block, so it lowers the $n_v^4$ ceiling without
removing every large tensor.
