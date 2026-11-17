# Open-shell GW {#open-shell-gw}

<!-- status: ready; registry: u-gw, gw -->

In a radical the $\alpha$ and $\beta$ electrons occupy different orbitals with
different energies, so each spin has its own ionization energies and its own
gap. The screening, though, is shared. The Coulomb interaction does not care
about spin, so a test charge polarizes the $\alpha$ and the $\beta$ density
together, and a single $W$ built from the response of both spins screens
both. Everything else is per spin: each spin has its own $B$, its own
exchange and correlation self-energy summed over that spin's states, its own
continuation line, and its own static shift.

| Index | Range | Code name |
|---|---|---|
| $\sigma$ | spin, $\alpha$ or $\beta$ | `mo_b_a`, `mo_b_b` |
| $i_\sigma, a_\sigma, p, q$ | that spin's occupied, virtual, any active MO | local indices |
| $\xi$ | PDEP modes of the spin-summed dielectric | `m_modes` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/lib.rs::run_u_gw`; `ferric-gw/src/u_sigma.rs::{run_u_g0w0,run_u_evgw0,run_u_evgw}`; `ferric-gw/src/u_cohsex.rs::run_u_cohsex` |
| CLI | `method.kind = "gw"` with multiplicity > 1; `[gw] reference = "uhf" \| "rohf"` |
| Python | `run_u_gw(reference=)` |

## The spin-summed screening and the per-spin self-energy {#open-shell-gw-spin-summed-screening-per}

**The reference.** `run_u_gw` accepts a UHF or UKS reference as it is. A ROHF
or ROKS reference is first semi-canonicalized per spin
(`ferric_scf::semicanonical::unrestricted_reference`): each spin's Fock
matrix is diagonalized within its occupied block and within its virtual
block, which gives each spin a set of orbitals and orbital energies.

**The spin-summed dielectric.** Each spin contributes its own particle–hole
pairs to the response:

$$ \tilde{\boldsymbol\varepsilon}_U(iu) = \mathbf 1 + \Pi_\alpha(iu) + \Pi_\beta(iu), \qquad \Pi_\sigma = 2\sum_{i_\sigma a_\sigma} B_\sigma B_\sigma^{\mathsf T}\frac{\Delta_{i\sigma}^{a}}{u^2+(\Delta_{i\sigma}^{a})^2} $$

(`ferric-rpa/src/energy.rs`; section @sec:open-shell-dielectric). Where does the prefactor come
from? The closed-shell 4 of section @sec:screened-interaction-dielectric-matrix-single-pole is 2 for spin times 2 for the
resonant and antiresonant terms. Here the spin sum is explicit, so each spin
keeps only the 2.

**Check the closed-shell limit.** When the two spins have
the same orbitals and energies, $\Pi_\alpha = \Pi_\beta$, and their sum has
the prefactor 4 of the closed-shell $\Pi$. One spin alone has half of it.

> **Verified equation {#eq:gw-u-spin-summed-dielectric}** · the spin-summed dielectric with prefactor 2 per spin
>
> $\tilde\varepsilon_U(iu) = \mathbf 1 + \Pi_\alpha(iu) + \Pi_\beta(iu)$ with
> $\Pi_\sigma = 2\sum_{i_\sigma a_\sigma}B_\sigma B_\sigma^{\mathsf T}\Delta/(u^2+\Delta^2)$
> equals the closed-shell $\mathbf 1 + \Pi$ (prefactor 4) when the two spins
> coincide.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_self_energy_sympy.py`, claim 1; `scripts/gw_ucohsex_sympy.py`, claim U1b |
> | Negative control | one spin's prefactor 2 alone is not the closed-shell pole (claim 1); the runner turns the first spin's prefactor into 1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/sternheimer.rs::build_scale_factors_with_prefactor` |
> | Code test | `urpa_closed_shell_limit_h2o_vs_pyscf_rpa` in `crates/ferric-rpa/tests/validation_urpa.rs` |
>
> <!-- verified: id=gw.u.spin-summed-dielectric; script=scripts/gw_ucohsex_sympy.py; claim=U1b; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/sternheimer.rs::build_scale_factors_with_prefactor; rust_test=ferric-rpa/tests/validation_urpa.rs::urpa_closed_shell_limit_h2o_vs_pyscf_rpa; registry=u-gw -->

**The per-spin self-energy.** In a spin-orbital basis $G$ is diagonal in spin
and $W$ does not depend on spin. A state of spin $\sigma$ therefore sums the
exchange and screened-exchange terms over the occupied orbitals of its own
spin only. Each spin has its own projection
$B^\xi_\sigma = \hat U^{\mathsf T}B_\sigma$ on the shared modes, and the
static pieces of section @sec:cohsex become

$$ \Sigma^x_{\sigma p} = -\sum_{i\in\sigma}\sum_P (B^P_{\sigma,pi})^2, \qquad \Delta\Sigma^{\rm SEX}_{\sigma p} = -\sum_{i\in\sigma}\sum_\xi w_\xi (B^\xi_{\sigma,pi})^2, \qquad \Sigma^{\rm COH}_{\sigma p} = \tfrac12\sum_{q\in\sigma}\sum_\xi w_\xi (B^\xi_{\sigma,pq})^2, $$

with the weights $w_\xi = 1/\lambda_\xi(0) - 1$ of the spin-summed
$\tilde{\boldsymbol\varepsilon}_U(0)$. `ferric-gw/src/u_cohsex.rs::run_u_cohsex`
calls the closed-shell `cohsex_pieces` once per spin with the shared
weights. The Coulomb-hole sum runs over the complete set of that spin, and
its closure turns it into the local $\tfrac12 W_d(\mathbf r,\mathbf r)$, as
in section @sec:cohsex. The dynamic $\Sigma^c_\sigma$ sums over that spin's MOs in
the same way, and the static shift is per spin,
$\delta_{\sigma p} = \Sigma^x_{\sigma p} - v^{xc}_{\sigma p}$
(`ferric-gw/src/u_sigma.rs::static_shifts`).

**How the per-spin pieces are checked.** Claim U1 of the script builds the
same three quantities with no auxiliary or PDEP basis: a 6-site lattice with
different $\alpha$ and $\beta$ orbitals and 2 $\alpha$ and 1 $\beta$ electron,
$\chi_0 = \chi_{0\alpha} + \chi_{0\beta}$, $W = (1 - v\chi_0)^{-1}v$, and
$\Delta\Sigma^{\rm SEX}_{\sigma p} = -\sum_{i\in\sigma}\langle pi|W_d|ip\rangle$.
Summing over the other spin's occupied orbitals, or building $W$ from one
spin's $\Pi$ alone, are the negative controls. Claim U1b shows that with
identical channels the two spins' self-energies coincide with the closed-shell
ones.

> **Verified equation {#eq:gw-u-cohsex-per-spin}** · the per-spin static self-energy
>
> On a lattice model with different $\alpha$ and $\beta$ orbitals (2 $\alpha$
> and 1 $\beta$ electron), the code's $\Sigma^x_\sigma$,
> $\Delta\Sigma^{\rm SEX}_\sigma$ and $\Sigma^{\rm COH}_\sigma$, built from the
> spin-summed dielectric and its PDEP modes, equal a real-space spin-orbital
> construction with no auxiliary basis: $\chi_0 = \chi_{0\alpha}+\chi_{0\beta}$,
> $W = (1-v\chi_0)^{-1}v$, and sums over that spin's states only.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_ucohsex_sympy.py`, claim U1 |
> | Negative control | SEX summed over the other spin's occupied orbitals, and $W$ built from one spin's $\Pi$ alone, must both miss |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/u_cohsex.rs::run_u_cohsex`, `ferric-gw/src/cohsex.rs::cohsex_pieces` |
> | Code test | `u_cohsex_singlet_h2o_equals_closed_shell_cohsex` in `crates/ferric-gw/tests/validation_u_cohsex.rs` |
>
> <!-- verified: id=gw.u.cohsex-per-spin; script=scripts/gw_ucohsex_sympy.py; claim=U1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/u_cohsex.rs::run_u_cohsex; rust_test=ferric-gw/tests/validation_u_cohsex.rs::u_cohsex_singlet_h2o_equals_closed_shell_cohsex; registry=u-gw -->

**The per-spin Fermi level.** The continuation line $z = \epsilon_{F,\sigma} + iu$
assigns each state to the occupied or the virtual side by the sign of
$\mathrm{Re}(z - \epsilon_q)$ (section @sec:correlation-self-energy-frequency-integral). So it has to pass through the
gap of the spin whose self-energy it samples. Any $\epsilon_{F,\sigma}$
inside that gap gives the same continued function of $E$. The mid-gap is a
numerical choice: it keeps the nodes far from both edges. A radical's
$\alpha$ and $\beta$ gaps can sit at different energies, and a line
through the other spin's mid-gap can lie outside this spin's gap, mis-assign
a state, and continue to a different function. Claim U2 shows both
statements on the single-pole $\Sigma^c$ of section @sec:correlation-self-energy with Thiele
continuation in 80-digit arithmetic. The code takes a mid-gap per spin in
`ferric-gw/src/u_sigma.rs::qp_per_spin_g0w0`.

> **Verified equation {#eq:gw-u-fermi-level}** · the per-spin Fermi level
>
> The Thiele continuation of a single-pole $\Sigma^c$ sampled on
> $z = \epsilon_{F,\sigma} + iu$ is the same function of $E$ for every
> $\epsilon_{F,\sigma}$ inside that spin's gap, and a line through the other
> spin's mid-gap, outside this spin's gap, continues to a different function.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_ucohsex_sympy.py`, claim U2 |
> | Negative control | the other spin's mid-gap line must miss by more than $10^{-3}$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/u_sigma.rs::qp_per_spin_g0w0`, `ferric-gw/src/sigma.rs::fermi_level` |
> | Code test | `u_g0w0_oh_vs_pyscf_ugw_ac` in `crates/ferric-gw/tests/validation_gw.rs` |
>
> <!-- verified: id=gw.u.fermi-level; script=scripts/gw_ucohsex_sympy.py; claim=U2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/u_sigma.rs::qp_per_spin_g0w0; rust_test=ferric-gw/tests/validation_gw.rs::u_g0w0_oh_vs_pyscf_ugw_ac; registry=u-gw -->

## The algorithm {#open-shell-gw-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | reference: UHF/UKS as is; ROHF/ROKS semi-canonicalized per spin (each spin's Fock diagonalized in its occupied and virtual blocks) | `ferric_scf::semicanonical::unrestricted_reference` |
| 2 | spin-summed U-PDEP-RPA (section @sec:open-shell-dielectric): modes of $\tilde\varepsilon_U(0)$, the $\tilde W_d(iu_k)$ stack | `ferric_rpa::run_u_pdep_rpa` |
| 3 | $B_\alpha$, $B_\beta$ in one auxiliary basis | `build_full_b_both_spins` |
| 4 | redress the shared eigenpotentials once | `redress_with_check` |
| 5 | per spin: $B^\xi_\sigma = \hat U^{\mathsf T}B_\sigma$, $\Sigma^x_\sigma$ | `project_b_into_pdep`, `sigma_x_diag` |
| 6 | per spin, per state: the continuation and the root of sections @sec:correlation-self-energy and @sec:continuation-and-qp-equation, with $\epsilon_{F,\sigma}$ and $\delta_{\sigma p}$ | `qp_per_spin_g0w0` |
| 7 | COHSEX instead of 6 for the static method | `run_u_cohsex` |

The drivers are `run_u_g0w0`, `run_u_evgw0` and `run_u_evgw`, and
`run_u_cohsex` for the static method. Both spins' projected tensors are live
together, so the pool check sums them before either is built
(`guard_m_proj_both_spins`).

**Why the product rule matters most here.** The product-integration rule of
section @sec:correlation-self-energy-product-integration is built once and used by both spins. The plain quadrature at
the same points misses the open-shell $\Sigma^c$ at the continuation nodes by
0.11 Ha at 100 points for OH/cc-pVDZ at PBE, because the $\beta$ HOMO and LUMO
lie 0.015 Ha either side of $\epsilon_{F,\beta}$.

The default QP window is HOMO−2 … LUMO+2 around $\max(n_\alpha, n_\beta)$
(`default_u_qp_range`). A closed-shell result passed to `run_u_gw` is an
error. U-sc-COHSEX is not implemented and returns an error.

## Options and limits {#open-shell-gw-options-limits}

`[gw] reference = "rohf"` selects ROHF (ROKS with `[rpa] xc`); the default is
UHF (UKS). The CLI solves UHF with maximum overlap after five DIIS iterations
when the multiplicity is above 1, for `task = "energy"` only.

**Known limits (2026-11-15).** Energy only; no gradient. The reference of a
ROHF or ROKS run is the semi-canonical per-spin view, and the validation
compares at that convention.

## Numerical anchors {#open-shell-gw-numerical-anchors}

From `site/src/reference/validation.md` (the U-G0W0, U-COHSEX and ROHF rows):

| Quantity | Systems | Error | Test |
|---|---|---|---|
| U-G0W0@UHF against the grid-free U-dRPA pole sum at the same 18 nodes | OH, CH3, NH2 / cc-pVDZ and aug-cc-pVDZ; O2, CH2 triplets | $\le 3.8\times10^{-8}$ Ha; OH/cc-pVDZ $\beta$ LUMO+2 $1.2\times10^{-5}$ Ha | `u_g0w0_{oh,ch3,nh2}_vs_pyscf_ugw_ac`, `u_g0w0_triplets_o2_ch2_vs_pyscf_ugw_ac` in `crates/ferric-gw/tests/validation_gw.rs` |
| U-G0W0@UKS/PBE, product-rule $\Sigma^c$ against the pole sum | OH, CH3, NH2 / cc-pVDZ | $\le 5.2\times10^{-10}$ Ha | `u_g0w0_uks_pbe_*` in `crates/ferric-gw/tests/validation_gw.rs` |
| U-G0W0@UKS/PBE quasiparticle energies against the pole sum, every orbital | OH, CH3, NH2 / cc-pVDZ | $\le 2.7\times10^{-8}$ Ha | the same |
| U-COHSEX@UHF against numpy on PySCF density-fitted integrals | OH, CH3, NH2 / cc-pVDZ and aug-cc-pVDZ | $\varepsilon^{\rm QP}$ $\le 3.3\times10^{-9}$ Ha | `crates/ferric-gw/tests/validation_u_cohsex.rs` |
| singlet H2O UHF through `run_u_gw` against closed-shell COHSEX | H2O / cc-pVDZ | $8.1\times10^{-10}$ Ha | `u_cohsex_singlet_h2o_equals_closed_shell_cohsex` |
| U-G0W0 on a ROHF reference, semi-canonicalized | OH, CH3 / cc-pVDZ | $\le 1.5\times10^{-5}$ Ha (OH $\beta$), $\le 1.3\times10^{-6}$ Ha (CH3) | `crates/ferric-gw/tests/validation_rohf_reference.rs` |

Further tests: `crates/ferric-gw/tests/u_cohsex_evgw_closed_shell_limit.rs`
(the closed-shell limit of U-COHSEX and U-evGW) and
`crates/ferric-gw/tests/u_gw_ks_shift.rs` (the per-spin KS shift).

## Sources {#open-shell-gw-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, "Open shell" | the per-spin construction (boxes @eq:gw-u-spin-summed-dielectric to @eq:gw-u-fermi-level) |
| `scripts/gw_ucohsex_sympy.py`, claims U1, U1b, U2 | the proofs |
| `scripts/gw_self_energy_sympy.py`, claim 1 | the closed-shell limit of the dielectric |
