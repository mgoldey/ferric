# 16.5 LinLCCD and LinLCCD(hh)

<!-- section: 16.5; status: ready; registry: linlccd;
     theory: crates/ferric-cc/theory.md#"LinLCCD and LinLCCD(hh)" -->

Linearized ladder CCD keeps three kinds of term from the CCD residual of
section 16.1: the driver $\langle ij\|ab\rangle$, the Fock term and the
ladders. Rings, crossed rings and every quadratic term are dropped, so the
amplitude equations are linear. LinLCCD(hh) keeps only the hole–hole ladder,
which costs $O(n_o^4n_v^2)$ and never forms the $n_v^4$ block. It is the
correlation half of the ωB97X-L-V double hybrid (section 16.6) and a CLI
kind of its own.

| Index | Range | Code name |
|---|---|---|
| $i, j, k, l$ / $a, b, c, d$ | occupied / virtual (spin orbitals in the spin-orbital solver) | `0..2*no`, `0..2*nv` |
| $T_{iajb}$ | spatial amplitude, $= t_{i\alpha j\beta}^{a\alpha b\beta}$ | closed-shell form |

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/linlccd.rs::{linlccd,LadderVariant}`, `ferric-cc/src/linlccd_u.rs::u_linlccd`, `ferric-cc/src/linlccd_exact.rs::linlccd_exact` |
| CLI | `method.kind = "linlccd"`, `[mp2] linlccd_variant` (`examples/water-linlccd.toml`) |
| Python | `run_linlccd(variant=)` |

## 16.5.1 The equations

<!-- include: crates/ferric-cc/theory.md#"LinLCCD and LinLCCD(hh)" -->

The include ends with the ωB97X-L-V scaling, which section 16.6 takes up.

> **Verified equation 16.8** · the spatial form is the spin-orbital form
>
> On random 8-fold-symmetric RI integrals, the spin-orbital LinLCCD of
> `linlccd.rs` is solved as one dense linear system in $(2n_o)^2(2n_v)^2$
> unknowns, and the spatial closed-shell equations as another in
> $n_o^2n_v^2$. For all three variants the correlation energies agree, and
> $T_{iajb}$ equals the $\alpha\beta$ block $t_{i\alpha j\beta}^{a\alpha b\beta}$
> of the spin-orbital solution.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`, claim 6 |
> | Negative control | the runner's mutation: the closed-shell energy with $(2T_{iajb} - T_{ibja})$ replaced |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/linlccd.rs::linlccd`, `ferric-cc/src/linlccd_amplitude.rs::amplitude_linlccd` |
> | Code test | `linlccd_hh_h2o_ccpvdz_vs_numpy` in `crates/ferric-cc/tests/validation_linlccd.rs` |
>
> <!-- verified: id=cc.linlccd.spin-adaptation; script=scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/linlccd.rs::linlccd; rust_test=ferric-cc/tests/validation_linlccd.rs::linlccd_hh_h2o_ccpvdz_vs_numpy; registry=linlccd -->

The box covers the two forms against each other. The Rust tests compare both
with an independent numpy linear solve on PySCF density-fitted integrals,
and with the ladder off with PySCF `DFMP2`.

## 16.5.2 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF (or UHF; ROHF semicanonicalized first); $B^P_{pq}$ | $N_{\rm aux}N^2n$ | |
| 2 | antisymmetrized OOVV and OOOO blocks; VVVV only for `Full` | $N_{\rm aux}n_o^4$ | `needs_vvvv` |
| 3 | MP2 guess $t = -\langle ij\|ab\rangle/\Delta$ | | |
| 4 | residual: driver, hh ladder, and for `Full` the pp ladder | $n_o^4n_v^2$ (hh), $n_o^2n_v^4$ (pp) | `ferric-cc/src/helpers.rs::contract_hh_ladder` |
| 5 | $t \leftarrow -R/\Delta$ with DIIS until the energy settles | | |
| 6 | $E_c = \tfrac14\sum\langle ij\|ab\rangle t$ | | |

The unrestricted solver (`u_linlccd`) builds the same blocks from per-spin
orbitals; a slot where one spin has fewer orbitals than the other gets
$\varepsilon_i = 10^6$, $\varepsilon_a = -10^6$, so its zero integrals give a
zero amplitude. `linlccd_exact` solves the same equations from exact
four-centre integrals.

## 16.5.3 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[mp2] linlccd_variant`, `variant=` | `hh`, `drivers-only`, `full` | `hh` | ladder terms kept; unknown spellings are errors |
| `[mp2] auxbasis`, `auxbasis` | basis name | | RI basis of the integrals |
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | frozen occupied orbitals |
| `[local]` section, `local=`, `eps=` | see section 16.7 | exact | amplitude-threshold approximation |

`drivers-only` is the MP2 amplitude equation, so it reproduces RI-MP2.

## 16.5.4 Numerical anchors

| Quantity | Systems | Error | Test |
|---|---|---|---|
| LinLCCD(hh) vs a numpy linear solve on PySCF DF integrals | H2O, NH3 (RHF) / 6-31G, cc-pVDZ | $\le 4.4\times10^{-13}$ Ha | `crates/ferric-cc/tests/validation_linlccd.rs` |
| the same, unrestricted | OH (UHF) / 6-31G, cc-pVDZ | $1.2\times10^{-12}$ Ha | `crates/ferric-cc/tests/validation_linlccd.rs` |
| ladder off vs RI-MP2 | same | $1.1\times10^{-16}$ | `crates/ferric-cc/tests/validation_linlccd.rs` |
| exact canonical LinLCCD, every variant | H2O, NH3 / 6-31G, cc-pVDZ | $\le 4.4\times10^{-13}$ Ha | `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |

## 16.5.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "LinLCCD and LinLCCD(hh)" | the equations and both solvers |
| Carter-Fenk, JPCA 129, 7251 (2025), eqs. 7 and 14 | the method and the hh truncation |
| `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py` | claim 6 |

**Known limits (2026-11-15).** The CLI and Python run closed shell only; the
open-shell solver `u_linlccd` is library-only. No other code implements
LinLCCD(hh), so the anchors are against an independent numpy solve rather
than another program.
