# 19.7 Open-shell GW

<!-- section: 19.7; status: ready; registry: u-gw, gw;
     theory: crates/ferric-gw/theory.md#"9. Open shell" -->

For an open-shell reference the $\alpha$ and $\beta$ orbitals differ, but the
screening does not: the bare interaction is spin-blind, so a single $W$ built
from the density response of both spins screens both. Everything else is
per spin: each spin has its own $B$, its own exchange and correlation
self-energy summed over that spin's states, its own continuation line, and
its own static shift. A radical's HOMO-LUMO gap can be very different in the
two spins, which is what makes the continuation line per spin.

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

## 19.7.1 The spin-summed screening and the per-spin self-energy

<!-- include: crates/ferric-gw/theory.md#"9. Open shell" -->

> **Verified equation 19.10** · the spin-summed dielectric with prefactor 2 per spin
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

> **Verified equation 19.11** · the per-spin static self-energy
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

> **Verified equation 19.12** · the per-spin Fermi level
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

## 19.7.2 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | reference: UHF/UKS as is; ROHF/ROKS semi-canonicalized per spin (each spin's Fock diagonalized in its occupied and virtual blocks) | `ferric_scf::semicanonical::unrestricted_reference` |
| 2 | spin-summed U-PDEP-RPA (section 18.4): modes of $\tilde\varepsilon_U(0)$, the $\tilde W_d(iu_k)$ stack | `ferric_rpa::run_u_pdep_rpa` |
| 3 | $B_\alpha$, $B_\beta$ in one auxiliary basis | `build_full_b_both_spins` |
| 4 | redress the shared eigenpotentials once | `redress_with_check` |
| 5 | per spin: $B^\xi_\sigma = \hat U^{\mathsf T}B_\sigma$, $\Sigma^x_\sigma$ | `project_b_into_pdep`, `sigma_x_diag` |
| 6 | per spin, per state: the continuation and the root of sections 19.3 and 19.5, with $\epsilon_{F,\sigma}$ and $\delta_{\sigma p}$ | `qp_per_spin_g0w0` |
| 7 | COHSEX instead of 6 for the static method | `run_u_cohsex` |

Both spins' projected tensors are live together, so the pool check sums them
before either is built (`guard_m_proj_both_spins`). The product-integration
rule of section 19.3 is built once and used by both spins; the plain
quadrature at the same points misses the open-shell $\Sigma^c$ at the
continuation nodes by 0.11 Ha at 100 points for OH/cc-pVDZ at PBE, because the
$\beta$ HOMO and LUMO lie 0.015 Ha either side of $\epsilon_{F,\beta}$.

The default QP window is HOMO−2 … LUMO+2 around $\max(n_\alpha, n_\beta)$
(`default_u_qp_range`). A closed-shell result passed to `run_u_gw` is an
error. U-sc-COHSEX is not implemented and returns an error.

## 19.7.3 Options and limits

`[gw] reference = "rohf"` selects ROHF (ROKS with `[rpa] xc`); the default is
UHF (UKS). The CLI solves UHF with maximum overlap after five DIIS iterations
when the multiplicity is above 1, for `task = "energy"` only.

**Known limits (2026-11-15).** Energy only; no gradient. The reference of a ROHF or ROKS run is the semi-canonical per-spin view, and the validation compares at that convention.

## 19.7.4 Numerical anchors

From `site/src/reference/validation.md` (the U-G0W0, U-COHSEX and ROHF rows):

| Quantity | Systems | Error | Test |
|---|---|---|---|
| U-G0W0@UHF against the grid-free U-dRPA pole sum at the same 18 nodes | OH, CH3, NH2 / cc-pVDZ and aug-cc-pVDZ; O2, CH2 triplets | $\le 3.8\times10^{-8}$ Ha; OH/cc-pVDZ $\beta$ LUMO+2 $1.2\times10^{-5}$ Ha | `u_g0w0_{oh,ch3,nh2}_vs_pyscf_ugw_ac`, `u_g0w0_triplets_o2_ch2_vs_pyscf_ugw_ac` in `crates/ferric-gw/tests/validation_gw.rs` |
| U-G0W0@UKS/PBE, product-rule $\Sigma^c$ against the pole sum | OH, CH3, NH2 / cc-pVDZ | $\le 5.2\times10^{-10}$ Ha | `u_g0w0_uks_pbe_*` in `crates/ferric-gw/tests/validation_gw.rs` |
| U-COHSEX@UHF against numpy on PySCF density-fitted integrals | OH, CH3, NH2 / cc-pVDZ and aug-cc-pVDZ | $\varepsilon^{\rm QP}$ $\le 3.3\times10^{-9}$ Ha | `crates/ferric-gw/tests/validation_u_cohsex.rs` |
| singlet H2O UHF through `run_u_gw` against closed-shell COHSEX | H2O / cc-pVDZ | $8.1\times10^{-10}$ Ha | `u_cohsex_singlet_h2o_equals_closed_shell_cohsex` |
| U-G0W0 on a ROHF reference, semi-canonicalized | OH, CH3 / cc-pVDZ | $\le 1.5\times10^{-5}$ Ha (OH $\beta$), $\le 1.3\times10^{-6}$ Ha (CH3) | `crates/ferric-gw/tests/validation_rohf_reference.rs` |

## 19.7.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, "Open shell" | the per-spin construction (boxes 19.10 to 19.12) |
| `scripts/gw_ucohsex_sympy.py`, claims U1, U1b, U2 | the proofs |
| `scripts/gw_self_energy_sympy.py`, claim 1 | the closed-shell limit of the dielectric |
