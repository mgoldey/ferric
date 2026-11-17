# Ring amplitudes with a threshold: local dRPA {#local-drpa}

<!-- status: verify; registry: drpa;
     theory: crates/ferric-rpa/theory.md#"RPA Correlation Energy" -->

In Boys-localized orbitals each occupied orbital sits on a bond or a lone
pair, and each local virtual sits near some atom. Two particle-hole
fluctuations far apart interact weakly, so most elements of the ring
coupling $K_{ia,jb} = 2(ia|jb)$ are small, as in local MP2 (section @sec:amplitude-lmp2).
The local approximation drops the amplitudes whose coupling element is below
a threshold $\varepsilon$ and solves the Riccati equation (@eq:drpa-three-ways-2) on the
pattern that is left. At $\varepsilon = 0$ nothing is dropped and the energy
is exact dRPA. At finite $\varepsilon$ it is an approximation with a
measured, not a validated, error.

The plan: the masked equation and why its error is first order (@sec:local-drpa-masked-riccati-equation), the
algorithm (@sec:local-drpa-algorithm), then options and the measured error map.

| Index | Range | Code name |
|---|---|---|
| $i, j$ | Boys-localized active occupied orbitals | |
| $a, b$ | localized virtuals (projected valence virtuals and hard virtuals, as in local MP2) | |
| $\varepsilon$ | amplitude threshold, Hartree | `eps` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/drpa_amplitude.rs::{amplitude_drpa,amplitude_drpa_direct,amplitude_drpa_scan,riccati_masked_solve}` |
| CLI | `method.kind = "drpa"` with `[local] scheme = "amplitude-threshold"` and `eps` or `eps_sweep` (`examples/water-drpa-local.toml`) |
| Python | `run_drpa(local="amplitude-threshold", eps=...)`, `run_drpa_scan(eps_list=...)` |

## The masked Riccati equation {#local-drpa-masked-riccati-equation}

In the localized basis $K_{ia,jb} = 2(ia|jb)$ is symmetric under
$(ia) \leftrightarrow (jb)$. The kept pattern is

$$ \mathcal M = \{(ia, jb) : |K_{ia,jb}| > \varepsilon\}, $$

closed under the swap because $\mathbf K$ is symmetric. The amplitudes are
restricted to $\mathcal M$, and the damped fixed point of section @sec:drpa-three-ways is
projected onto $\mathcal M$ after each step:

$$ \mathbf T \leftarrow P_{\mathcal M}\Big[\mathbf T - \mathbf R(\mathbf T)/\mathbf D\Big], \qquad E_c(\varepsilon) = \tfrac12\sum_{(ia,jb)\in\mathcal M}K_{ia,jb}T_{ia,jb} . \label{eq:rpa-drpa-local-mask} $$

$P_{\mathcal M}$ is the `apply_pattern` call in the loop of @sec:drpa-three-ways-ring-amplitudes. It
zeroes both the residual and the update outside the pattern, so the dropped
amplitudes are exactly zero and the Riccati equation holds exactly on the
kept ones. The ring products $\mathbf K\mathbf T$, $\mathbf T\mathbf K$ and
$\mathbf T\mathbf K\mathbf T$ run over a ragged (per-pair) storage of the
kept blocks. An integral-free distance gate can drop whole occupied pairs
before assembly (`pair_gate_cal`).

Check the trivial limit. At $\varepsilon = 0$ the code keeps every element,
zeros included (it tests `cfg.eps == 0.0 || x.abs() > cfg.eps`), so $P_{\mathcal M}$
does nothing and (@eq:rpa-drpa-local-mask) is the unmasked iteration of (@eq:drpa-three-ways-2).

**Why the error is first order.** dRPA is not variational in the amplitudes.
The energy $\tfrac12\sum K\circ T$ is linear in $\mathbf T$ and has no
stationary point to expand about. Let $\mathbf T_0$ be the unmasked solution
and $\mathbf T_\varepsilon$ the masked one. Split the sum over all pairs
into the kept and the dropped part:

$$ E_c(\varepsilon) - E_c(0) = \tfrac12\sum_{\mathcal M}K\,(T_\varepsilon - T_0) - \tfrac12\sum_{\notin\mathcal M}K\,T_0 . $$

Both terms are linear in an amplitude difference. Compare local MP2 (@sec:amplitude-lmp2-amplitude-equations-energy),
where the Hylleraas energy is stationary at the solution, the term linear in
the amplitude error cancels, and the truncation error is second order with a
proven sign. dRPA has no such cancellation, so the error is first order in
what is dropped, and nothing fixes its sign.

The energy sum of the code runs over the pattern only. From
`ferric-mp2/src/drpa_amplitude.rs::riccati_masked_solve`, unchanged:

<!-- excerpt: ferric-mp2/src/drpa_amplitude.rs::riccati_masked_solve -->
```rust
let mut e_corr = 0.0;
for (p, pb) in rg.pairs.iter().enumerate() {
    let nbb = pb.db.len();
    for r in 0..pb.da.len() {
        for c in 0..nbb {
            if pb.pat[r * nbb + c] {
                e_corr += 0.5 * b_blocks[p][(r, c)] * t[p][(r, c)];
            }
        }
    }
}
```

| Code | Equation |
|---|---|
| `p`, `pb` | occupied pair $(i, j)$ and its block |
| `r`, `c` | $a$ and $b$ in the pair's virtual domains `pb.da`, `pb.db` |
| `pb.pat[r * nbb + c]` | $(ia, jb) \in \mathcal M$ |
| `b_blocks[p]`, `t[p]` | $K_{ia,jb}$, $T_{ia,jb}$ |

> **Unverified equation @eq:rpa-drpa-local-mask** · the masked Riccati energy
>
> The exact limit is covered: at $\varepsilon = 0$ the mask is vacuous and
> (@eq:rpa-drpa-local-mask) is (@eq:drpa-three-ways-2), whose rotation invariance and equality with the plasmon
> formula are verified in section @sec:drpa-three-ways. Nothing proves a bound on
> $E_c(\varepsilon) - E_c(0)$ at finite $\varepsilon$.
>
> | | |
> |---|---|
> | Proof | the $\varepsilon = 0$ limit only: `scripts/rpa_drpa_pdep_sympy.py`, check `riccati-vs-definition` (in a randomly rotated basis) |
> | Status | verify: the finite-$\varepsilon$ error is a measured map on one molecule |
>
> <!-- unverified: id=rpa.drpa.local-mask; todo=state and test an error model for E(eps)-E(0) on more than one molecule; no proof exists -->

## The algorithm {#local-drpa-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | RHF; Boys-localized occupied orbitals; projected valence virtuals and hard virtuals | shared with local MP2 (`ferric-mp2/src/lmp2_amplitude.rs::build_vvhv`) |
| 2 | $K$ blocks in the localized basis, assembled from the global three-index tensor (`amplitude_drpa`) or integral-direct (`amplitude_drpa_direct`) | `assemble_ragged_direct`, `assemble_ragged_direct_local` |
| 3 | the pattern $\mathcal M$ | in `riccati_masked_solve` |
| 4 | damped fixed point, optional DIIS on the increment, stop at $\lVert\mathbf R\rVert/\lVert\mathbf K\rVert < 10^{-12}$, error after 500 iterations | `riccati_masked_solve` |
| 5 | optionally the canonical plasmon energy, for the error printout | `canonical_plasmon_drpa` |

The CLI uses the global-tensor assembly (`amplitude_drpa`). The
integral-direct assembly is a library entry point for this method; both feed
the same `riccati_masked_solve`. `eps_sweep` and `run_drpa_scan` reuse one
SCF and one localized assembly across thresholds.

With DIIS on, the solver checks the residual a second time on the
extrapolated amplitudes it returns, and stops with an error if they fail the
tolerance that the pre-extrapolation iterate passed.

## Options {#local-drpa-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[local] eps`, `eps=` | $\ge 0$ | none, required | the threshold; 0 is exact |
| `[local] eps_sweep`, `run_drpa_scan(eps_list=)` | list | | several thresholds on one SCF |
| `[local] reference`, `compute_reference=` | bool | false | also compute the canonical plasmon energy and print the error |
| `diis=` | integer, 0 off | 8 (Python) | DIIS subspace |
| `eps_rtol_factor=` | float | off | stopping tolerance $\max(10^{-12}, c\,\varepsilon)$ |

`[local]` on a kind other than `rimp2`, `drpa` and `linlccd`, or a stray key,
is an error. `eps` has no default because it is part of the model, like the
basis set.

## Numerical anchors {#local-drpa-numerical-anchors}

From `site/src/reference/validation.md` (the `drpa` row and its anchors) and
`crates/ferric-mp2/tests/validation_drpa_amplitude.rs`.

**Exact limit.** At $\varepsilon = 0$ the Riccati solve, the integral-direct
route and the in-crate plasmon reference match an independent numpy plasmon
dRPA on PySCF integrals to $\le 3.0\times10^{-11}$ Ha (H2, H2O, n-butane,
n-octane; frozen cores 0, 1 and 4).

**Finite threshold.** n-octane / 6-31G (cc-pVDZ-RI), carbon cores frozen,
against the external $\varepsilon = 0$ reference. There is no external
finite-$\varepsilon$ reference.

| $\varepsilon$ | $E_c(\varepsilon) - E_{\mathrm{ref}}$ (Ha) | amplitudes kept |
|---|---|---|
| $10^{-6}$ | $+6.7\times10^{-7}$ | 82% |
| $10^{-5}$ | $+3.9\times10^{-5}$ | 50% |
| $10^{-4}$ | $+5.5\times10^{-4}$ | 18% |
| $10^{-3}$ | $+1.0\times10^{-2}$ | 2.8% |

The error is positive (less correlation) at every threshold measured and
grows faster than linearly in $\varepsilon$ over this range: each tenfold
step in $\varepsilon$ multiplies it by 14 to 58. The kept fraction falls
quickly over the same range, so each step drops more amplitudes as well as
larger ones. That is one molecule in one basis; the test asserts only the
sign and the monotonic growth. The positive sign is a measurement, not a
consequence of @sec:local-drpa-masked-riccati-equation, which allows either sign.

## Sources {#local-drpa-sources}

| Source | What it gives |
|---|---|
| `site/src/methods/rpa-gw.md`, "Exact and local dRPA" | the user guide |
| `crates/ferric-mp2/tests/validation_drpa_amplitude.rs` | the anchors and the threshold sweep |
| module documentation of `crates/ferric-mp2/src/drpa_amplitude.rs` | the mask, the assembly paths |

**Known limits (2026-11-15).** Closed shell, HF reference, energy only. No
scaling claim: the CLI assembly builds the global three-index tensor.
