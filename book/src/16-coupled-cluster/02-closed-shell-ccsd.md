# 16.2 Closed-shell CCSD and the particle-particle ladder

<!-- section: 16.2; status: ready; registry: ccsd;
     theory: crates/ferric-cc/theory.md#"CCSD, closed shell" -->

For an RHF reference the spin-orbital amplitudes of section 16.1 are fixed
by $n_o n_v$ singles and $n_o^2n_v^2$ doubles over spatial orbitals. The
spin-adapted solver iterates only those, which makes every rank-4 tensor 16
times smaller than its spin-orbital counterpart. It is what `ccsd`,
`ccsd(t)` and `eom-ccsd` run.

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

## 16.2.1 Spin adaptation

<!-- include: crates/ferric-cc/theory.md#"CCSD, closed shell" -->

The pairing of $t_{ij}^{ab}$ is Coulomb order, $i\leftrightarrow a$ and
$j\leftrightarrow b$: the MP2 guess is $(ia|jb)/(-\Delta)$, and the
opposite-spin amplitude with $a$ on the α electron is $+t_{ij}^{ab}$.

> **Verified equation 16.3** · the closed-shell residual is the opposite-spin block
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

The box proves that solving the spatial equations is solving CCSD. It does
not reach the Hirata factorization the code evaluates: that is a
transcription of PySCF `rccsd`, and the Rust test above and the spin-orbital
comparison are its evidence.

## 16.2.2 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF; $B^P_{ov}, B^P_{oo}, B^P_{vv}$ | $N_{\rm aux}N^2n$ | `build_b` |
| 2 | chemist blocks $(oo|oo)$, $(ov|oo)$, $(ov|ov)$, $(oo|vv)$, $(ov|vo)$, $(ov|vv)$; $(ac|bd)$ only on the dense route | $N_{\rm aux}n_on_v^3$ | |
| 3 | per iteration: $F$, $L$ and $W$ intermediates (Eqs. 37–45) | $n_o^3n_v^3$ | |
| 4 | the pp ladder $\sum_{cd}W_{abcd}\tau_{ij}^{cd}$ | $n_o^2n_v^4$ | `dense_ladder` or `ri_ladder_contract` |
| 5 | $t \leftarrow r/(-\Delta)$, joint DIIS on $(t_1, t_2)$ | | `diis_step_two` |
| 6 | energy; stop when it changes by less than $\min(\texttt{energy\_conv}, 10^{-9})$ | $n_o^2n_v^2$ | |

**The ladder.** $W_{abcd} = (ac|bd) - \sum_k (kd|ac)t_k^b - \sum_k (kc|bd)t_k^a$
carries no $t_1^2$ term; that piece enters through $\tau$. Both routes
contract the same algebra over the same $B$:

| Route | Holds | Rebuilds per iteration |
|---|---|---|
| `dense` | $(ac|bd)$, one $n_v^4$ buffer (two during setup) | nothing; the $t_1$ terms are contracted with $\tau$ separately |
| `ri` | $B_{vv}$ and $B_x = -\sum_k B_{ov,kd}t_k^b$, $N_{\rm aux}n_v^2$ each | $(ac|bd)$ in $a_{\rm blk}\cdot n_v^3$ panels, one rayon job per $a$ |
| `auto` (default) | dense when its `MemoryPlan` fits the budget, otherwise RI | |

The RI factorization of $W_{abcd}$, term by term, is in the module
documentation of `ri_ladder.rs`; it is exact, not an approximation, and the
two routes agree to round-off. The route taken is printed
(`ladder = dense (auto)`), logged and returned in `CcResult::ladder_route`.

**Reproducibility.** Under `auto` the route depends on the memory budget,
and without an explicit `[memory] budget_gb` that is 0.8 times the RAM
available at start-up. The energy agrees to round-off either way; the time
does not. Set `ladder` or the budget for reproducible timings. The RI ladder
is bit-identical at any thread count.

## 16.2.3 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cc] ladder`, `ladder=` | `auto`, `dense`, `ri` | `auto` | pp-ladder route; unknown spellings are errors |
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | frozen occupied orbitals |
| `[memory] budget_gb`, `memory_budget_gb=` | GB | auto | preflight and the `auto` decision |

## 16.2.4 Numerical anchors

| Quantity | Systems | Error | Test |
|---|---|---|---|
| RI-CCSD vs PySCF `CCSD` on the same DF integrals (both solvers) | H2O, NH3, CH4, HF / cc-pVDZ, def2-SVP; HCN / cc-pVDZ; H2O / aug-cc-pVDZ; C2H6 / cc-pVDZ | $\le 4.7\times10^{-11}$ Ha | `crates/ferric-cc/tests/validation_cc.rs` |
| the same, anions whose RHF HOMO lies above zero | CH3⁻, Te²⁻ / def2-SVP | $\le 6.0\times10^{-12}$ Ha | `crates/ferric-cc/tests/cc_anion_occupied_count.rs` |

## 16.2.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "CCSD, closed shell" | the spin adaptation and the energy |
| Hirata, Podeszwa, Tobita and Bartlett, JCP 120, 2581 (2004), Eqs. 35–45 | the factorization the code transcribes |
| `site/src/methods/cc.md`, "Particle-particle ladder: dense or RI" | measured memory and time of the two routes |

**Known limits (2026-11-15).** Closed shell only; the RI route still holds
the $n_on_v^3$ $(ia|bc)$ block, so it lowers the $n_v^4$ ceiling without
removing every large tensor.
