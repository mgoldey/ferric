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

## The equations the code evaluates {#closed-shell-ccsd-equations}

`ferric-cc/src/ccsd_closed_shell.rs::ccsd_closed_shell` does not derive the
spatial residual from the spin-orbital one at run time. It evaluates the
closed-shell equations of Hirata, Podeszwa, Tobita and Bartlett, JCP 120,
2581 (2004), Eqs. 35–45, in the form of PySCF `rccsd.update_amps` and
`rintermediates`. They are written here with chemist integrals over
spatial orbitals and $\tau_{ij}^{ab} = t_{ij}^{ab} + t_i^at_j^b$. The code
keeps the $t_1t_1$ pieces as separate contractions; collecting them into
$\tau$ changes nothing but the grouping.

**One-body intermediates** (Eqs. 37–41). With $f_{ia} = 0$ and the Fock
diagonal moved into $D$ below, they carry no Fock term and no
orbital-energy shift:

$$
\begin{aligned}
F_{ki} &= \textstyle\sum_{lcd}\big[2(kc|ld) - (kd|lc)\big]\tau_{il}^{cd}, \\
F_{ac} &= -\textstyle\sum_{kld}\big[2(kc|ld) - (kd|lc)\big]\tau_{kl}^{ad}, \\
F_{kc} &= \textstyle\sum_{ld}\big[2(kc|ld) - (kd|lc)\big]t_l^d, \\
L_{ki} &= F_{ki} + \textstyle\sum_{lc}\big[2(lc|ki) - (kc|li)\big]t_l^c, \\
L_{ac} &= F_{ac} + \textstyle\sum_{kd}\big[2(kd|ac) - (kc|ad)\big]t_k^d .
\end{aligned} \label{eq:cc-rccsd-one-body}
$$

**Two-body intermediates** (Eqs. 42–45):

$$
\begin{aligned}
W_{klij} &= (ki|lj) + \textstyle\sum_c (lc|ki)\,t_j^c + \sum_c (kc|lj)\,t_i^c + \sum_{cd}(kc|ld)\,\tau_{ij}^{cd}, \\
W_{akic} &= (kc|ai) + \textstyle\sum_d (kc|ad)\,t_i^d - \sum_l (kc|li)\,t_l^a
 + \sum_{ld}(ld|kc)\big(t_{il}^{ad} - \tfrac12t_{il}^{da} - t_i^dt_l^a\big) - \tfrac12\sum_{ld}(lc|kd)\,t_{il}^{ad}, \\
W_{akci} &= (ki|ac) + \textstyle\sum_d (kd|ac)\,t_i^d - \sum_l (lc|ki)\,t_l^a
 - \sum_{ld}(lc|kd)\big(\tfrac12t_{il}^{da} + t_i^dt_l^a\big),
\end{aligned} \label{eq:cc-rccsd-two-body}
$$

and $W_{abcd}$ of (@eq:cc-rccsd-ladder) below.

**Singles.** The non-Fock part of the singles residual is

$$
\begin{aligned}
r_i^a ={}& \textstyle\sum_c F_{ac}t_i^c - \sum_k F_{ki}t_k^a
 + \sum_{kc}F_{kc}\big(2t_{ki}^{ca} - t_{ik}^{ca} + t_i^ct_k^a\big)
 + \sum_{kc}\big[2(kc|ai) - (ki|ac)\big]t_k^c \\
 &+ \textstyle\sum_{kcd}\big[2(kd|ac) - (kc|ad)\big]\tau_{ik}^{cd}
 - \sum_{klc}\big[2(lc|ki) - (kc|li)\big]\tau_{kl}^{ac} .
\end{aligned} \label{eq:cc-rccsd-singles}
$$

**Doubles.** Write $\mathcal P x_{ij}^{ab} = x_{ij}^{ab} + x_{ji}^{ba}$.
The code builds the half $x$ and symmetrizes it, which enforces
$r_{ij}^{ab} = r_{ji}^{ba}$:

$$
\begin{aligned}
r_{ij}^{ab} ={}& (ia|jb) + \textstyle\sum_{kl}W_{klij}\tau_{kl}^{ab} + \sum_{cd}W_{abcd}\tau_{ij}^{cd}
 + \mathcal P\Big\{\textstyle\sum_c\big[(ia|cb) - \sum_k (ki|bc)\,t_k^a\big]t_j^c
 - \sum_k\big[(ia|jk) + \sum_c (kc|ai)\,t_j^c\big]t_k^b \\
 &+ \textstyle\sum_c L_{ac}t_{ij}^{cb} - \sum_k L_{ki}t_{kj}^{ab}
 + \sum_{kc}\big(2W_{akic} - W_{akci}\big)t_{kj}^{cb}
 - \sum_{kc}W_{akic}t_{kj}^{bc} - \sum_{kc}W_{bkci}t_{kj}^{ac}\Big\} .
\end{aligned} \label{eq:cc-rccsd-doubles}
$$

**The update.** With $D_i^a = \varepsilon_i - \varepsilon_a$ and
$D_{ij}^{ab} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b$,
the full residuals are $R = r - Dt$, and one iteration sets $t \leftarrow r/D$,
the fixed point of $R = 0$. $(t_1, t_2)$ are then extrapolated jointly by
DIIS (`diis_step_two`).

> **Verified equation {#eq:cc-rccsd-factorization}** · the transcribed factorization is the CCSD residual
>
> For random spatial $t_1$ and $t_2 = t_2^{(ij)(ab)}$, the residuals
> $r_i^a - D_i^at_i^a$ and $r_{ij}^{ab} - D_{ij}^{ab}t_{ij}^{ab}$ of
> (@eq:cc-rccsd-singles) and (@eq:cc-rccsd-doubles), with the intermediates
> of (@eq:cc-rccsd-one-body), (@eq:cc-rccsd-two-body) and
> (@eq:cc-rccsd-ladder), equal the α singles and the opposite-spin doubles
> of the brute-force projections
> $\langle\Phi_\mu|e^{-\hat T}\hat He^{\hat T}|\Phi_0\rangle$ on the amplitudes
> expanded by `expand_amplitudes_to_spin_orbital`.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_ccsd_eom_brute_force_sympy.py`, check `rccsd-factorization` |
> | Negative control | the runner's mutations: $\mathcal P$ replaced by $2x$; the sign of the $\tfrac12t_{il}^{da}$ term of $W_{akci}$ flipped |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/ccsd_closed_shell.rs::{ccsd_closed_shell,dense_ladder}` |
> | Code test | `ccsd_spin_adapted_matches_df_reference` in `crates/ferric-cc/tests/validation_cc.rs` |
>
> <!-- verified: id=cc.ccsd.rccsd-factorization; script=scripts/cc_ccsd_eom_brute_force_sympy.py; claim=rccsd-factorization; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/ccsd_closed_shell.rs::{ccsd_closed_shell,dense_ladder}; rust_test=ferric-cc/tests/validation_cc.rs::ccsd_spin_adapted_matches_df_reference; registry=ccsd -->

### What is proven and what is not {#closed-shell-ccsd-proven}

The box above (@eq:cc-ccsd-spin-adaptation) proves that solving the spatial
equations is solving CCSD. The box (@eq:cc-rccsd-factorization) proves that
the equations displayed here are those spatial equations. The script
checks a numpy transcription of the Rust loop, not the Rust loop itself;
that the Rust evaluates the same expressions is shown by its comparison
with the spin-orbital `ccsd` on the same integrals and with PySCF
`cc.CCSD` (`tests/validation_cc.rs`).

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

$$ W_{abcd} = (ac|bd) - \sum_k (kd|ac)\,t_k^b - \sum_k (kc|bd)\,t_k^a . \label{eq:cc-rccsd-ladder} $$

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

The two routes on water in cc-pVDZ, from the `589b50832` build. The
correlation energies agree to $10^{-16}$ Ha; `ladder_block` is the panel
width the RI route chose:

```python
# Water/cc-pVDZ CCSD with the dense and the RI particle-particle ladder: the
# two routes rewrite the same W_abcd over the same B, so the energies agree
# to round-off.
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")
dense = ferric.run_ccsd(mol, bs, aux, ladder="dense")
ri = ferric.run_ccsd(mol, bs, aux, ladder="ri")
print(f"dense: {dense.correlation_energy:.12f}  route={dense.ladder_route}")
print(f"ri:    {ri.correlation_energy:.12f}  route={ri.ladder_route} block={ri.ladder_block}")
print(f"difference: {abs(dense.correlation_energy - ri.correlation_energy):.2e} Ha")
```

```text
closed-shell CCSD converged in 10 iterations. E_corr = -0.2135061893
closed-shell CCSD converged in 10 iterations. E_corr = -0.2135061893
dense: -0.213506189269  route=dense
ri:    -0.213506189269  route=ri block=19
difference: 1.39e-16 Ha
```

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
| Hirata, Podeszwa, Tobita and Bartlett, JCP 120, 2581 (2004), Eqs. 35–45 | the factorization the code transcribes |
| `site/src/methods/cc.md`, "Particle-particle ladder: dense or RI" | measured memory and time of the two routes |

**Known limits (2026-11-15).** Closed shell only; the RI route still holds
the $n_on_v^3$ $(ia|bc)$ block, so it lowers the $n_v^4$ ceiling without
removing every large tensor.
