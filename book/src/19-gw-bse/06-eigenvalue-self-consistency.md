# 19.6 Eigenvalue self-consistency

<!-- section: 19.6; status: ready; registry: gw;
     theory: crates/ferric-gw/theory.md#"8. Eigenvalue self-consistency" -->

G0W0 builds $G$ and $W$ from the mean-field spectrum, so its answer depends on
the reference. The eigenvalue self-consistent variants feed the quasiparticle
energies back into the propagator, and in the full variant into $W$ too,
until the energies stop moving. The orbitals never change; only the energies
in the denominators do.

| Variant | Propagator energies $\epsilon_q$ in $\Sigma^c$ | $W$ |
|---|---|---|
| G0W0 | mean-field | mean-field |
| evGW$_0$ | previous sweep's QP energies | frozen at the mean-field PDEP result |
| evGW | previous sweep's QP energies | rebuilt every iteration after the first |

| Symbol | Meaning | Code name |
|---|---|---|
| $\epsilon^{\rm QP,(k)}_p$ | QP energy after sweep $k$ | `eps_qp` |
| $\epsilon_q^{\rm prop}$ | propagator energies in the kernel of section 19.3 | `eps_prop` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/sigma.rs::{run_evgw0,run_evgw}` |
| CLI | `[gw] method = "evgw0"` or `"evgw"`, with `max_ev_iter` and `ev_conv_thresh` |
| Python | `run_gw(method="evgw0" \| "evgw", max_ev_iter=, ev_conv_thresh=)` |

## 19.6.1 The update

<!-- include: crates/ferric-gw/theory.md#"8. Eigenvalue self-consistency" -->

The iteration is a Jacobi update: within one sweep every state in the QP
window is solved against the same frozen snapshot of the propagator energies,
so the per-state solves are independent and run in parallel. The first sweep
is G0W0, because the starting QP energies are the mean-field ones. In evGW$_0$
the quasiparticle root $E$ still solves
$E = \epsilon^{\rm MF}_p + \delta_p + \Sigma^c_p(E;\,\epsilon^{\rm prop})$:
the left-hand $\epsilon_p$, the static shift $\delta_p$ and the Fermi level of
the continuation line stay at their mean-field values for the whole run.

In evGW the second and later sweeps copy the reference, overwrite the orbital
energies of the QP window with the current QP energies, rerun the PDEP-RPA
step on that copy, redress the new eigenpotentials into the unchanged $B$
basis, and project $B$ again. This is the cost driver: the projection of
section 19.2 (the largest GEMM in the GW code) and the whole frequency stack
are rebuilt each sweep. Convergence is declared only after at least one
rebuilt $W$ (`it > 0`).

| Step | Work per sweep | Function |
|---|---|---|
| set propagator energies to the last QP energies | $O(n_{\rm act})$ | `run_evgw0`, `run_evgw` |
| evGW only, after sweep 0: PDEP-RPA on the shifted reference | chapter 18 | `ferric_rpa::run_pdep_rpa` |
| evGW only: redress and project $B$ | section 19.2 | `redress_eigenpotentials`, `project_b_into_pdep` |
| per-state QP solve against the snapshot | section 19.5 | `solve_qp_for_mo` |
| $\max_p\lvert\Delta\epsilon^{\rm QP}_p\rvert <$ `ev_conv_thresh` | $O(n_{\rm win})$ | the loop test |

There is no identity to prove here: the update rule is a definition. The
evidence is the validation below.

## 19.6.2 Options and limits

| Option | Default | Effect |
|---|---|---|
| `max_ev_iter` | 20 | outer iteration cap |
| `ev_conv_thresh` | $10^{-4}$ Ha | convergence on $\max_p\lvert\Delta\epsilon^{\rm QP}_p\rvert$ |

`GwResult.outer_converged` is false when the cap is hit, and `run_gw` warns;
the returned energies (and, for evGW, $W$) are then the last iterate. For a KS
reference the shift $\Sigma^x - v^{xc}$ is computed once from the starting
orbitals and held fixed across the loop. States outside the window, the Fermi
level and $\epsilon_p$ in the QP equation are never updated.

**Known limits (2026-11-15).** No external reference exists for evGW@KS, so
`crates/ferric-gw/tests/evgw_pbe_h2o.rs` checks convergence and the KS shift
only.

## 19.6.3 Numerical anchors

From `site/src/reference/validation.md` (the COHSEX, evGW$_0$, evGW row):

| Quantity | Reference | Error | Test |
|---|---|---|---|
| evGW$_0$, evGW at HF, H2O / cc-pVDZ | PySCF `get_sigma` iterated with ferric's update rule | $5.7\times10^{-7}$ Ha | `evgw0_evgw_hf_h2o_vs_iterated_pyscf` in `crates/ferric-gw/tests/validation_gw.rs` |
| HOMO and LUMO, defaults against matched settings (`ev_conv_thresh` $10^{-4}$ against $10^{-7}$) | ferric at matched settings | $\le 0.11$ meV per knob | `crates/ferric-gw/tests/validation_gw_defaults.rs` |

## 19.6.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, "Eigenvalue self-consistency" | the update rules |
