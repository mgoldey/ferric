# Eigenvalue self-consistency {#eigenvalue-self-consistency}

<!-- status: ready; registry: gw -->

G0W0 builds $G$ and $W$ from the mean-field spectrum, so its answer depends
on the reference. A Hartree–Fock gap is too large, a semilocal Kohn–Sham gap
too small, and the energy denominators of $\Sigma^c$ and of $\Pi$ inherit the
error. The eigenvalue self-consistent variants feed the quasiparticle
energies back into the propagator, and in the full variant into $W$ as well,
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
| $\epsilon_q^{\rm prop}$ | propagator energies in the kernel of section @sec:correlation-self-energy | `eps_prop` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/sigma.rs::{run_evgw0,run_evgw}` |
| CLI | `[gw] method = "evgw0"` or `"evgw"`, with `max_ev_iter` and `ev_conv_thresh` |
| Python | `run_gw(method="evgw0" \| "evgw", max_ev_iter=, ev_conv_thresh=)` |

## The update {#eigenvalue-self-consistency-update}

**What changes.** The propagator energies $\epsilon_q$ in the kernel
$(z - \epsilon_q)/[(z-\epsilon_q)^2 + u'^2]$ of section @sec:correlation-self-energy are the only
input the loop updates. Sweep $k$ uses

$$ \epsilon_q^{\rm prop} = \begin{cases} \epsilon^{\rm QP,(k-1)}_q & q \text{ in the QP window} \\ \epsilon_q & \text{otherwise} \end{cases}, \qquad \epsilon^{\rm QP,(-1)}_q = \epsilon_q , $$

and solves, for every state $p$ in the window,

$$ E = \epsilon_p + \delta_p + \Sigma^c_p\big(E;\,\epsilon^{\rm prop}\big) $$

by the continuation and Newton solve of section @sec:continuation-and-qp-equation. The left-hand
$\epsilon_p$, the static shift $\delta_p$ and the Fermi level $\epsilon_F$ of
the continuation line stay at their mean-field values for the whole run, and
so do the energies of states outside the window.

**Check the first sweep.** The starting QP energies are the mean-field ones,
so sweep 0 is G0W0.

**Jacobi, not Gauss–Seidel.** Within one sweep every state in the window is
solved against the same frozen snapshot of the propagator energies. The
per-state solves are therefore independent and run in parallel. The loop in
`ferric-gw/src/sigma.rs::run_evgw0` first copies the last sweep's energies
into the snapshot:

<!-- excerpt: ferric-gw/src/sigma.rs::run_evgw0 -->
```rust
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            let m_loc = mo_abs - first_act;
            eps_prop[m_loc] = eps_qp[idx];
        }
```

then maps `solve_qp_for_mo` over the window with `eps_prop` as the
propagator energies, records $\max_p|\Delta\epsilon^{\rm QP}_p|$, and stops
when it falls below `ev_conv_thresh`.

**evGW: rebuild $W$ too.** In `ferric-gw/src/sigma.rs::run_evgw` the second and
later sweeps copy the reference, overwrite the orbital energies of the QP
window with the current QP energies, rerun the PDEP-RPA step on that copy
(the $\Delta_i^a$ of $\Pi$ in section @sec:screened-interaction-dielectric-matrix-single-pole change), redress the new
eigenpotentials into the unchanged $B$ basis, and project $B$ again. The
orbitals and $B$ are unchanged. The rebuild drives the cost: the projection of
section @sec:screened-interaction (the largest GEMM in the GW code) and the whole frequency stack
are rebuilt each sweep. Convergence is declared only after at least one
rebuilt $W$ (`it > 0`).

| Step | Work per sweep | Function |
|---|---|---|
| set propagator energies to the last QP energies | $O(n_{\rm act})$ | `run_evgw0`, `run_evgw` |
| evGW only, after sweep 0: PDEP-RPA on the shifted reference | chapter @sec:rpa | `ferric_rpa::run_pdep_rpa` |
| evGW only: redress and project $B$ | section @sec:screened-interaction | `redress_eigenpotentials`, `project_b_into_pdep` |
| per-state QP solve against the snapshot | section @sec:continuation-and-qp-equation | `solve_qp_for_mo` |
| $\max_p\lvert\Delta\epsilon^{\rm QP}_p\rvert <$ `ev_conv_thresh` | $O(n_{\rm win})$ | the loop test |

**Check the fixed point on one pair.** The update rule is a definition, but
its consequences can be checked on the two-state model of section
@sec:screened-interaction-dielectric-matrix-single-pole: one occupied state
$i$, one virtual $a$, one auxiliary function. There $\Sigma^c$ is the
single-pole closed form of section
@sec:correlation-self-energy-frequency-integral, and the evGW$_0$ fixed point
is a quadratic in the quasiparticle gap $g = E_a - E_i$. For evGW the gap in
$\Pi$ is the quasiparticle gap $e'$, so the pole of $W$ moves from
$\Omega^2 = e(e + 4b^2)$ to $\Omega'^2 = e'(e' + 4b^2)$. The script runs a
transcription of the outer loops of `run_evgw0` and `run_evgw` on this model
and reaches both fixed points.

> **Verified equation {#eq:gw-evgw-fixed-point}** · the eigenvalue-self-consistent updates and their fixed points
>
> Sweep 0 of evGW$_0$ and evGW, with $\epsilon^{{\rm QP},(-1)} = \epsilon$, is G0W0. On the one-pair, two-state model the evGW$_0$ gap $g = E_a - E_i$ at the fixed point solves $(g - e_0)(g + \Omega) = 2B_{ia}^2 s$ with $e_0 = \epsilon_a + \delta_a - \epsilon_i - \delta_i - (B_{ii}^2 + B_{aa}^2)s/\Omega$; for evGW the pole moves to $\Omega'^2 = e'(e' + 4b^2)$ with the QP gap $e'$.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_evgw_sympy.py`, claims EV1, EV2, EV3 |
> | Negative control | the G0W0 roots are not the evGW$_0$ fixed point, and the evGW$_0$ fixed point is not the evGW one |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/sigma.rs::{run_evgw0,run_evgw}` |
> | Code test | `evgw0_evgw_hf_h2o_vs_iterated_pyscf` in `crates/ferric-gw/tests/validation_gw.rs` |
>
> <!-- verified: id=gw.evgw.fixed-point; script=scripts/gw_evgw_sympy.py; claim=EV2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/sigma.rs::run_evgw0; rust_test=ferric-gw/tests/validation_gw.rs::evgw0_evgw_hf_h2o_vs_iterated_pyscf; registry=gw -->

## Options and limits {#eigenvalue-self-consistency-options-limits}

| Option | Default | Effect |
|---|---|---|
| `max_ev_iter` | 20 | outer iteration cap |
| `ev_conv_thresh` | $10^{-4}$ Ha | convergence on $\max_p\lvert\Delta\epsilon^{\rm QP}_p\rvert$ |

`GwResult.outer_converged` is false when the cap is hit, and `run_gw` warns.
The returned energies (and, for evGW, $W$) are then the last iterate. For a KS
reference the shift $\Sigma^x - v^{xc}$ is computed once from the starting
orbitals and held fixed across the loop: $v^{xc}$ is a property of the
starting orbitals, which the loop does not change. States outside the
window, the Fermi level and $\epsilon_p$ in the QP equation are never
updated.

**Known limits (2026-11-19).** No external reference exists for evGW@KS, so
`crates/ferric-gw/tests/evgw_pbe_h2o.rs` checks convergence and the KS shift
only. Only the states in `qp_mos` are updated, so with the default window
(HOMO−2 … LUMO+2) the propagator, and for evGW also $W$, sees a spectrum
that is part quasiparticle and part mean-field. $\epsilon_F$ is computed
once from the mean-field spectrum (`fermi_level` in `run_evgw0`), and the
occupation of state $q$ in the kernel is
$\mathrm{sgn}(\epsilon_F - \epsilon_q^{\rm prop})$ (section
@sec:correlation-self-energy-frequency-integral). The method assumes that no
updated energy crosses that fixed $\epsilon_F$; the code does not check it,
and a state that crossed would change side without an error.

## Numerical anchors {#eigenvalue-self-consistency-numerical-anchors}

From `site/src/reference/validation.md` (the COHSEX, evGW$_0$, evGW row):

| Quantity | Reference | Error | Test |
|---|---|---|---|
| evGW$_0$, evGW at HF, H2O / cc-pVDZ | PySCF `get_sigma` iterated with ferric's update rule | $5.7\times10^{-7}$ Ha | `evgw0_evgw_hf_h2o_vs_iterated_pyscf` in `crates/ferric-gw/tests/validation_gw.rs` |
| HOMO and LUMO, defaults against matched settings (`ev_conv_thresh` $10^{-4}$ against $10^{-7}$) | ferric at matched settings | $\le 0.11$ meV per knob | `crates/ferric-gw/tests/validation_gw_defaults.rs` |

`crates/ferric-gw/tests/h2o_g0w0_cohsex.rs` holds smoke tests of G0W0,
COHSEX, evGW$_0$ and evGW on water at cc-pVDZ.

## Sources {#eigenvalue-self-consistency-sources}

| Source | What it gives |
|---|---|
| Shishkin and Kresse, *Phys. Rev. B* **75**, 235102 (2007) | eigenvalue self-consistency in $G$ alone and in $G$ and $W$ |
| Blase, Attaccalite and Olevano, *Phys. Rev. B* **83**, 115103 (2011) | evGW for molecules |
