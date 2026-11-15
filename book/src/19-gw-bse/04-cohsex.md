# 19.4 COHSEX

<!-- section: 19.4; status: ready; registry: gw;
     theory: crates/ferric-gw/theory.md#"5. COHSEX: the static limit" -->

When $W$ does not depend on frequency, the integral of section 19.3 is
elementary and the self-energy splits into a screened-exchange part and a
Coulomb-hole part. COHSEX is that static limit: no frequency grid for the
integral, no continuation and no root search, at the cost of treating the
screening as instantaneous. It is also the starting point of the BSE kernel of
section 19.8, which uses the same static $W$.

| Symbol | Meaning | Code name |
|---|---|---|
| $\Delta\Sigma^{\rm SEX}_p$ | screened exchange beyond the bare exchange | `delta_sex` in `cohsex_pieces` |
| $\Sigma^{\rm COH}_p$ | Coulomb hole | `coh` in `cohsex_pieces` |
| $w_\xi = 1/\lambda_\xi(0)-1$ | static weights | `static_weights` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/cohsex.rs::{run_cohsex,cohsex_pieces}` |
| CLI | `method.kind = "gw"`, `[gw] method = "cohsex"` |
| Python | `run_gw(method="cohsex")` |

## 19.4.1 The static limit

<!-- include: crates/ferric-gw/theory.md#"5. COHSEX: the static limit" -->

The quasiparticle energy needs no root search because the self-energy is
energy-independent: the renormalization factor is fixed at $Z = 1$ and
`GwResult.z_factor` is all ones. `run_cohsex` assembles
$\epsilon_p + \Delta\Sigma^{\rm SEX}_p + \Sigma^{\rm COH}_p$ and, for a KS
reference, adds $\Sigma^x_p - v^{xc}_p$ as a plain additive term.

> **Verified equation 19.6** · the static-$W$ limit of the integral is COHSEX
>
> A frequency-independent $W_{pq}$ in the integral of box 19.4 gives
> $-\tfrac12W\,\mathrm{sgn}(\mathrm{Re}(z-\epsilon_q))$, which at $z$ in the
> gap is $-\sum_iW_{pi} + \tfrac12\sum_qW_{pq}$. The plasmon model of
> section 19.2 reaches the same limit as $\Omega\to\infty$ at fixed $W(0)$.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_self_energy_sympy.py`, claim 3 |
> | Negative control | a Coulomb-hole weight of 1 instead of $\tfrac12$ must miss |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/cohsex.rs::cohsex_pieces` |
> | Code test | `sigma_c_static_w_in_gap_equals_cohsex_pieces` in `crates/ferric-gw/src/sigma.rs` |
>
> <!-- verified: id=gw.cohsex.static-limit; script=scripts/gw_self_energy_sympy.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/cohsex.rs::cohsex_pieces; rust_test=ferric-gw/src/sigma.rs::sigma_c_static_w_in_gap_equals_cohsex_pieces; registry=gw -->

## 19.4.2 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $B$, $\Sigma^x$ | section 19.1 | `build_full_b`, `sigma_x_diag` |
| 2 | $w_\xi$ from the static PDEP eigenvalues, $B^\xi$ | $N_{\rm pdep}N_{\rm aux}n_{\rm act}^2$ | `static_weights`, `project_b_into_pdep` |
| 3 | $\Delta\Sigma^{\rm SEX}_p$, $\Sigma^{\rm COH}_p$ for every state, parallel over $p$ | $N_{\rm pdep}n_{\rm act}^2$ per state | `cohsex_pieces` |
| 4 | $\epsilon^{\rm QP}_p$, KS shift if any | $O(n_{\rm act})$ | `run_cohsex` |

The two sums are independent per state: each $(\Delta\Sigma^{\rm SEX}_p, \Sigma^{\rm COH}_p)$
pair is written once, so the parallel loop is bit-identical to the serial one.

## 19.4.3 Options and limits

`sc-COHSEX`, the self-consistent variant, is not implemented: `GwMethod::ScCohsex`
returns an error from both `run_gw` and the open-shell driver.

## 19.4.4 Numerical anchors

From `site/src/reference/validation.md` (the COHSEX row):

| Quantity | Systems | Error | Test |
|---|---|---|---|
| COHSEX@HF against numpy on PySCF density-fitted integrals | H2O, N2 / cc-pVDZ | $9.4\times10^{-10}$ Ha | `cohsex_hf_h2o_n2_vs_numpy` in `crates/ferric-gw/tests/validation_gw.rs` |
| COHSEX@PBE, same, with the static $\Sigma^x - v^{xc}$ | H2O, N2 / cc-pVDZ | $1.2\times10^{-9}$ Ha | `cohsex_pbe_h2o_n2_vs_numpy` in `crates/ferric-gw/tests/validation_gw.rs` |

## 19.4.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, "COHSEX: the static limit" | the derivation |
| `scripts/gw_self_energy_sympy.py`, claim 3 | the proof |
| Hedin, *Phys. Rev.* **139**, A796 (1965) | GW and COHSEX |
