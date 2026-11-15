# 18.6 Ring amplitudes with a threshold: local dRPA

<!-- section: 18.6; status: verify; registry: drpa;
     theory: crates/ferric-rpa/theory.md#"RPA Correlation Energy" -->

The ring-amplitude form of dRPA (section 18.1, equation 18.2) is solved in a
localized basis, where most coupling elements are small. The local
approximation drops the amplitudes whose coupling element is below a
threshold $\varepsilon$ and solves the Riccati equation on the pattern that
is left. At $\varepsilon = 0$ nothing is dropped and the energy is exact dRPA;
at finite $\varepsilon$ it is an approximation with a measured, not a
validated, error.

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

## 18.6.1 The masked Riccati equation

In the localized basis $K_{ia,jb} = 2(ia|jb)$ is symmetric under
$(ia) \leftrightarrow (jb)$. The kept pattern is

$$ \mathcal M = \{(ia, jb) : |K_{ia,jb}| > \varepsilon\}, $$

closed under the swap because $\mathbf K$ is symmetric. The amplitudes are
restricted to $\mathcal M$, and the damped fixed point of section 18.1 is
projected onto $\mathcal M$ after each step:

$$ \mathbf T \leftarrow P_{\mathcal M}\Big[\mathbf T - \mathbf R(\mathbf T)/\mathbf D\Big], \qquad E_c(\varepsilon) = \tfrac12\sum_{(ia,jb)\in\mathcal M}K_{ia,jb}T_{ia,jb} . \tag{18.16} $$

The ring products $\mathbf K\mathbf T$, $\mathbf T\mathbf K$ and
$\mathbf T\mathbf K\mathbf T$ run over a ragged (per-pair) storage of the kept
blocks. An integral-free distance gate can drop whole occupied pairs before
assembly (`pair_gate_cal`).

dRPA is not variational in the amplitudes, so the energy error of a masked
solve is first order in what is dropped, not second order as in local MP2.

> **Unverified equation 18.16** · the masked Riccati energy
>
> The exact limit is covered: at $\varepsilon = 0$ the mask is vacuous and
> (18.16) is (18.2), whose rotation invariance and equality with the plasmon
> formula are verified in section 18.1. Nothing proves a bound on
> $E_c(\varepsilon) - E_c(0)$ at finite $\varepsilon$.
>
> | | |
> |---|---|
> | Proof | the $\varepsilon = 0$ limit only: `scripts/rpa_drpa_pdep_sympy.py`, check `riccati-vs-definition` (in a randomly rotated basis) |
> | Status | verify: the finite-$\varepsilon$ error is a measured map on one molecule |
>
> <!-- unverified: id=rpa.drpa.local-mask; todo=state and test an error model for E(eps)-E(0) on more than one molecule; no proof exists -->

## 18.6.2 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | RHF; Boys-localized occupied orbitals; projected valence virtuals and hard virtuals | shared with local MP2 (`ferric-mp2/src/lmp2_amplitude.rs::build_vvhv`) |
| 2 | $K$ blocks in the localized basis, assembled from the global three-index tensor (`amplitude_drpa`) or integral-direct (`amplitude_drpa_direct`) | `assemble_ragged_direct`, `assemble_ragged_direct_local` |
| 3 | the pattern $\mathcal M$ | in `riccati_masked_solve` |
| 4 | damped fixed point, optional DIIS on the increment, stop at $\lVert\mathbf R\rVert/\lVert\mathbf K\rVert < 10^{-12}$, error after 500 iterations | `riccati_masked_solve` |
| 5 | optionally the canonical plasmon energy, for the error printout | `canonical_plasmon_drpa` |

The CLI uses the global-tensor assembly (`amplitude_drpa`); the integral-direct
assembly is a library entry point for this method. `eps_sweep` and
`run_drpa_scan` reuse one SCF and one localized assembly across thresholds.

## 18.6.3 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[local] eps`, `eps=` | $\ge 0$ | none, required | the threshold; 0 is exact |
| `[local] eps_sweep`, `run_drpa_scan(eps_list=)` | list | | several thresholds on one SCF |
| `[local] reference`, `compute_reference=` | bool | false | also compute the canonical plasmon energy and print the error |
| `diis=` | integer, 0 off | 8 (Python) | DIIS subspace |
| `eps_rtol_factor=` | float | off | stopping tolerance $\max(10^{-12}, c\,\varepsilon)$ |

`[local]` on a kind other than `rimp2`, `drpa` and `linlccd`, or a stray key,
is an error.

## 18.6.4 Numerical anchors

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
grows faster than linearly in $\varepsilon$ over this range. That is one
molecule in one basis; the test asserts only the sign and the monotonic
growth.

## 18.6.5 Sources

| Source | What it gives |
|---|---|
| `site/src/methods/rpa-gw.md`, "Exact and local dRPA" | the user guide |
| `crates/ferric-mp2/tests/validation_drpa_amplitude.rs` | the anchors and the threshold sweep |
| module documentation of `crates/ferric-mp2/src/drpa_amplitude.rs` | the mask, the assembly paths |

**Known limits (2026-11-15).** Closed shell, HF reference, energy only. No
scaling claim: the CLI assembly builds the global three-index tensor.
`examples/water-drpa-local.toml` describes the finite-$\varepsilon$ error as
"~linear in eps"; the only measurement (above) grows faster than linearly.
