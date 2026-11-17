# LinLCCD and LinLCCD(hh) {#linlccd}

<!-- status: ready; registry: linlccd;
     theory: crates/ferric-cc/theory.md#"LinLCCD and LinLCCD(hh)" -->

MP2 divides each pair's driving integral by the orbital-energy gap, so it
diverges as the gap closes. Coupled cluster at first order in $\hat T_2$
(LinCCD) does no better, and Carter-Fenk traces that divergence to the
exchange part of the ring and crossed-ring terms, not to small denominators
as such (module documentation of `linlccd.rs`). Linearized ladder CCD keeps
only the driver, the Fock term and the ladders of the CCD residual of
section @sec:ccd-ccsd-spin-orbital, so the amplitude equations are linear, and LinLCCD(hh) keeps
only the hole–hole ladder, which costs $O(n_o^4n_v^2)$ and never forms the
$n_v^4$ block. It is the correlation half of the ωB97X-L-V double hybrid
(section @sec:wb97x-l-v) and a CLI kind of its own.

| Index | Range | Code name |
|---|---|---|
| $i, j, k, l$ / $a, b, c, d$ | occupied / virtual (spin orbitals in the spin-orbital solver) | `0..2*no`, `0..2*nv` |
| $T_{iajb}$ | spatial amplitude, $= t_{i\alpha j\beta}^{a\alpha b\beta}$ | closed-shell form |

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/linlccd.rs::{linlccd,LadderVariant}`, `ferric-cc/src/linlccd_u.rs::u_linlccd`, `ferric-cc/src/linlccd_exact.rs::linlccd_exact` |
| CLI | `method.kind = "linlccd"`, `[mp2] linlccd_variant` (`examples/water-linlccd.toml`) |
| Python | `run_linlccd(variant=)` |

## The equations {#linlccd-equations}

**From CCD to LinLCCD.** Start from the CCD residual of section @sec:ccd-ccsd-spin-orbital and
delete the ring term (in spin orbitals it holds the rings and crossed rings
together) and the four quadratic terms (the products of two pair
excitations). What is left is the driver, the Fock part $\Delta t$ and the two ladders,
equation 7 of Carter-Fenk, JPCA 129, 7251 (2025):

$$0 = \langle ij\|ab\rangle + \Delta_{ij}^{ab}\, t_{ij}^{ab}
+ \tfrac12 \sum_{kl}\langle kl\|ij\rangle\, t_{kl}^{ab}
+ \tfrac12 \sum_{cd}\langle ab\|cd\rangle\, t_{ij}^{cd}.$$

Every term is linear in $t$, so the amplitudes solve one linear system.
`LadderVariant` (`linlccd.rs`) selects the ladder terms:

| Variant | Terms kept | Blocks built |
|---|---|---|
| `DriversOnly` | driver + Fock (= MP2 amplitude equations) | OOVV |
| `Hh`: LinLCCD(hh), eq. 14 | + hole–hole ladder; $O(n_o^4 n_v^2)$ | OOVV, OOOO |
| `Full`: LinLCCD | + particle–particle ladder; $O(n_o^2 n_v^4)$ | OOVV, OOOO, VVVV |

The hh truncation is the absence of the pp term: `Hh` never forms the VVVV
block (`needs_vvvv`, `linlccd.rs`).

**Why the hh ladder stays finite.** Move the hole–hole term to the left,
next to the Fock part. It then acts as an amplitude-independent operator on
the occupied pair, $f_{ik}\delta_{jl} + \delta_{ik}f_{jl} + \tfrac12 v_{ij}^{kl}$,
which shifts the effective pair energies. The paper's eq. 15 writes the
result as $(\varepsilon_a + \varepsilon_b - \tilde\eta_{ij})\,\tilde t_{ij}^{ab} = -\tilde v_{ij}^{ab}$:
the hh ladder widens the effective gap. ferric solves the equations
self-consistently, as the paper's implementation does, and does not
diagonalize the $n_o^2\times n_o^2$ hh operator; the dressed form is the
interpretation. The module documentation also records that LinLCCD(hh) is
size-consistent, size-extensive and orbital-invariant.

**Limits.**

- `DriversOnly` is the MP2 amplitude equation, so it reproduces RI-MP2. The
  solver starts from the MP2 guess, which is already its answer.
- `Full` is not exact for two electrons, unlike CCSD (section @sec:ccd-ccsd-spin-orbital): it
  drops the ring terms, and on H2/cc-pVDZ it recovers about half of the CCD
  correlation energy (`scripts/validation/gen_linlccd.py`, quoted in the
  documentation of `LadderVariant::Full`).

**Spin-orbital solver** (`linlccd.rs`, restricted orbitals; `linlccd_u.rs`,
unrestricted). With canonical orbitals the Fock part is $\Delta t$, so the
solver is the CCD loop with terms removed: Jacobi iteration with DIIS on
$t \leftarrow -R/\Delta$, starting from the MP2 guess
$t = -\langle ij\|ab\rangle/\Delta$, where

$$R_{ij}^{ab} = \langle ij\|ab\rangle + \tfrac12\sum_{kl}\langle kl\|ij\rangle t_{kl}^{ab}
+ \tfrac12\sum_{cd} t_{ij}^{cd}\langle ab\|cd\rangle\ (\text{Full only}),$$

$$E_c = \tfrac14\sum_{ijab}\langle ij\|ab\rangle\, t_{ij}^{ab}.$$

The residual in `ferric-cc/src/linlccd.rs::linlccd`, verbatim:

<!-- excerpt: ferric-cc/src/linlccd.rs::linlccd -->
```rust
        let mut r = v_oovv.clone();

        // hh ladder: 0.5 <kl||ij> t_klab
        if let Some(oooo) = &oooo_t {
            let x: ArrayD<f64> = einsum!("klij,klab->ijab", oooo, &t_t);
            r = r + 0.5 * x;
        }
        // pp ladder: 0.5 <ab||cd> t_ijcd   (Full only)
        if let Some(vvvv) = &vvvv_t {
            let x: ArrayD<f64> = einsum!("ijcd,abcd->ijab", &t_t, vvvv);
            r = r + 0.5 * x;
        }
```

- The terms are the CCD ones of `ccd_spinorbital` with the same signs and
  factors, so LinLCCD is CCD minus terms.
- The integral blocks are `Option`s. `oooo_t` exists for `Hh` and `Full`,
  `vvvv_t` only for `Full`, so the variant decides at setup which blocks are
  built, and the loop never branches on the variant name.
- The hh contraction runs over $k,l$ for each $(i,j)$ and all $(a,b)$:
  $n_o^4n_v^2$ in spin orbitals.

`linlccd_u.rs` builds the same blocks from per-spin orbitals
(`ferric_mp2::spinorbital_u`). A slot where one spin has fewer orbitals than
the other gets $\varepsilon_i = 10^6$, $\varepsilon_a = -10^6$, so its zero
integrals give a zero amplitude without a $0/0$. A raw ROHF reference is
rejected; it must be semicanonicalized first. `linlccd_exact.rs` solves the
same equations from exact four-centre integrals.

**Spatial closed-shell form** (`linlccd_amplitude.rs`, module doc). Write
$T_{iajb} = t_{i\alpha j\beta}^{a\alpha b\beta}$, the opposite-spin block.
Take the hole–hole term for that block. The sum over spin orbitals $k,l$
has two spin orderings. With $k$ α and $l$ β the integral is $(ki|lj)$ and
the amplitude $T_{kalb}$. With $k$ β and $l$ α the integral is
$-(kj|li)$ and the amplitude $-T_{lakb}$; relabel $k\leftrightarrow l$ and
it equals the first. The two orderings cancel the $\tfrac12$, and the pp
ladder goes the same way. The result, with the energy in the form (@eq:so-cc-amplitude-expansion):

$$0 = (ia|jb) + \Delta_{ij}^{ab} T_{iajb} + \sum_{kl}(ik|jl)\, T_{kalb} + \sum_{cd}(ac|bd)\, T_{icjd},\qquad
E_c = \sum_{iajb}(ia|jb)\,(2T_{iajb} - T_{ibja}).$$

It gives the same $E_c$ and amplitudes as the spin-orbital equations for all
three variants (`scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`,
claim 6). The hole–hole coefficient is $(ik|jl)$: hole $i$ pairs with $k$
and $j$ with $l$. Writing $(ij|kl)$ instead is the negative control of claim
7 (section @sec:local-cc).

> **Verified equation {#eq:cc-linlccd-spin-adaptation}** · the spatial form is the spin-orbital form
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
and with the ladder off with PySCF `DFMP2`. The ωB97X-L-V scaling of the same
equations is section @sec:wb97x-l-v.

## The algorithm {#linlccd-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF (or UHF; ROHF semicanonicalized first); $B^P_{pq}$ | $N_{\rm aux}N^2n$ | |
| 2 | antisymmetrized OOVV and OOOO blocks; VVVV only for `Full` | $N_{\rm aux}n_o^4$ | `needs_vvvv` |
| 3 | MP2 guess $t = -\langle ij\|ab\rangle/\Delta$ | | |
| 4 | residual: driver, hh ladder, and for `Full` the pp ladder | $n_o^4n_v^2$ (hh), $n_o^2n_v^4$ (pp) | `linlccd` (`einsum!` on the stored blocks) |
| 5 | $t \leftarrow -R/\Delta$ with DIIS until the energy changes by less than `energy_conv` | | |
| 6 | $E_c = \tfrac14\sum\langle ij\|ab\rangle t$ | | |

Unlike `ccd`, whose stopping threshold is fixed, `linlccd` reads
`CcConfig::energy_conv`.

## Options {#linlccd-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[mp2] linlccd_variant`, `variant=` | `hh`, `drivers-only`, `full` | `hh` | ladder terms kept; unknown spellings are errors |
| `[mp2] auxbasis`, `auxbasis` | basis name | | RI basis of the integrals |
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | frozen occupied orbitals |
| `[local]` section, `local=`, `eps=` | see section @sec:local-cc | exact | amplitude-threshold approximation |

`drivers-only` is the MP2 amplitude equation, so it reproduces RI-MP2.

## Numerical anchors {#linlccd-numerical-anchors}

| Quantity | Systems | Error | Test |
|---|---|---|---|
| LinLCCD(hh) vs a numpy linear solve on PySCF DF integrals | H2O, NH3 (RHF) / 6-31G, cc-pVDZ | $\le 4.4\times10^{-13}$ Ha | `crates/ferric-cc/tests/validation_linlccd.rs` |
| the same, unrestricted | OH (UHF) / 6-31G, cc-pVDZ | $1.2\times10^{-12}$ Ha | `crates/ferric-cc/tests/validation_linlccd.rs` |
| ladder off vs RI-MP2 | same | $1.1\times10^{-16}$ | `crates/ferric-cc/tests/validation_linlccd.rs` |
| exact canonical LinLCCD, every variant | H2O, NH3 / 6-31G, cc-pVDZ | $\le 4.4\times10^{-13}$ Ha | `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |

## Sources {#linlccd-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "LinLCCD and LinLCCD(hh)" | the equations and both solvers |
| Carter-Fenk, JPCA 129, 7251 (2025), eqs. 7 and 14 | the method and the hh truncation |
| `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py` | claim 6 |

**Known limits (2026-11-15).** The CLI and Python run closed shell only; the
open-shell solver `u_linlccd` is library-only. No other code implements
LinLCCD(hh), so the anchors are against an independent numpy solve rather
than another program.
