# 16.8 Unrestricted CCSD and CCSD(T)

<!-- section: 16.8; status: ready; registry: u-ccsd, u-ccsd-t;
     theory: crates/ferric-cc/theory.md#"Unrestricted CCSD and CCSD(T)" -->

On a UHF reference the α and β orbitals differ, so the closed-shell spin
adaptation of sections 16.2 and 16.3 does not apply. The spin-orbital
amplitudes still vanish unless spin is conserved. The spin-blocked solver
keeps only the five nonzero amplitude blocks and the integral blocks they
touch. The CLI kinds `ccsd` and `ccsd(t)`, and Python `run_ccsd` and
`run_ccsd_t`, take this route for a molecule with multiplicity > 1.

The spin-orbital solver of sections 16.1 and 16.3 is the cross-check. Fed
the same UHF orbitals as one α-then-β spin-orbital space, it has no spin
bookkeeping at all.

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

## 16.8.1 The equations

<!-- include: crates/ferric-cc/theory.md#"Unrestricted CCSD and CCSD(T)" -->

> **Verified equation 16.12** · the (T) class weights
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

> **Verified equation 16.13** · spin-blocked (T) per class
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

> **Verified equation 16.14** · the CCSD energy split
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

## 16.8.2 The algorithm

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

## 16.8.3 Numerical anchors

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

Recorded mutations, each failing on an assertion: a dropped $t_{iJ}^{aB}$
ring term ($1.1\times10^{-2}$ Ha), an aab weight of $1/6$
($2.2\times10^{-4}$ Ha), an α orbital energy on a β (T) index
($6.2\times10^{-6}$ Ha) and a dropped $t_I^A$ ($1.6\times10^{-4}$ Ha) fail
the spin-orbital comparison. The two Rust solvers share the RI factors and
orbital energies. So giving the α virtual block the β orbital energies
passes the spin-orbital comparison; only the PySCF test catches it
($2.3\times10^{-3}$ Ha).

## 16.8.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "Unrestricted CCSD and CCSD(T)" | the spin blocks, the $t_2$ sign table, the energy split and the class weights |
| PySCF `cc/uccsd.py::update_amps`, `cc/uccsd_t_slow.py` | the residual factorization and the per-class (T) reference |
| `scripts/cc_uccsd_spin_blocks_sympy.py` | claims 1–5 |
| `scripts/validation/gen_uccsd.py` | the PySCF DF-UCCSD and UCCSD(T) references |

**Known limits (2026-11-16).** UHF reference only: no ROHF-based CC. Energy
only: there is no UCCSD nuclear gradient, so `optimize` and `frequencies`
refuse an open-shell `ccsd` or `ccsd(t)`. The three `vvvv` blocks are held
dense and the `[cc] ladder` choice of section 16.2 does not apply (the CLI
warns; Python refuses a ladder other than `"auto"`). `ccd` and `eom-ccsd`
have no open-shell route. The PySCF comparison is an ignored test and runs
only on request.
