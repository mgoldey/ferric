# The two formulations in the code {#two-formulations}

<!-- status: ready; registry: rs-mp2-rpa, u-rs-mp2-rpa;
     theory: crates/ferric-rpa/theory.md#"Range-separated MP2 + RPA", "Open-shell SR-MP2 + LR-RPA" -->

`rs_mp2_lr_rpa` evaluates one of two energies. Both keep full-Coulomb MP2
and add dRPA brackets that start at third order (section @sec:cross-term-rule). They differ
in which ring diagrams beyond second order they keep.

| Entry point | Location |
|---|---|
| Rust | `ferric-rpa/src/rs_mp2_rpa.rs::{rs_mp2_lr_rpa,RsMp2RpaFormulation,RsMp2RpaConfig,RsMp2RpaResult}` |
| CLI | `method.kind = "rs-mp2-rpa"` with `[mp2] formulation`, `omega`, `attenuator`, `r0`, `terf_omega`, `frozen_core`, `auxbasis` (`examples/water-rs-mp2-rpa.toml`) |
| Python | `run_rs_mp2_rpa(mol, basis_set, auxbasis, omega=, formulation=, attenuator=, r0=, terf_omega=, frozen_core=)` |

## The energies {#two-formulations-energies}

With the components (@eq:cross-term-1) and $\Delta E_{\mathrm{dRPA}}[v] = E_{\mathrm{dRPA}}[v] - 2E_{\mathrm{OS}}[v]$,

$$ E^{\mathrm B}_c = E_{\mathrm{MP2}}[\mathrm C] + \Delta E_{\mathrm{dRPA}}[\mathrm{lr}] \qquad (\texttt{delta-lr}, \text{ default}), \label{eq:two-formulations-1} $$

$$ E^{\mathrm T}_c = E_{\mathrm{MP2}}[\mathrm C] + \Delta E_{\mathrm{dRPA}}[\mathrm C] - \Delta E_{\mathrm{dRPA}}[\mathrm{sr}] \qquad (\texttt{coupled-rings}). \label{eq:two-formulations-2} $$

B replaces the long-range direct rings of MP2 with their dRPA resummation.
T screens every ring with the full operator and then removes the screening
from the purely short-range rings. Each $E_{\mathrm{dRPA}}$ is a full-rank
PDEP evaluation (section @sec:pdep, `trunc_thresh` @sec:notation by default) on the same
fitted integrals as the MP2 components of the same operator.

Scaling every integral by $\lambda$, both energies have second-order term
$E_{\mathrm{MP2}}[\mathrm C]$ (exchange and the cross term $X$ included). At
third order, from (@eq:cross-term-3),

$$ E^{\mathrm B\,(3)}_c = E^{(3)}_{\mathrm{dRPA}}[\mathrm{lr}], \qquad E^{\mathrm T\,(3)}_c = E^{(3)}_{\mathrm{dRPA}}[\mathrm C] - E^{(3)}_{\mathrm{dRPA}}[\mathrm{sr}] . \label{eq:rsrpa-forms-orders} $$

In one mode, $k = k_s + k_l$ and (@eq:rsrpa-one-mode-series),

$$ E^{\mathrm T\,(3)}_c - E^{\mathrm B\,(3)}_c = \frac{(k_s + k_l)^3 - k_s^3 - k_l^3}{4\Delta^2} = \frac{3k_sk_l(k_s + k_l)}{4\Delta^2} : \label{eq:rsrpa-forms-one-mode} $$

the third-order rings with both short- and long-range interactions, which T
keeps and B drops. Neither formulation keeps the purely short-range rings
beyond second order.

> **Verified equation @eq:rsrpa-forms-orders** · second order is full MP2; the third-order rings of each formulation
>
> On the many-mode model of section @sec:cross-term with $\mathbf K_{\mathrm C} = \mathbf K_{\mathrm{sr}} + \mathbf K_{\mathrm{lr}}$,
> the $\lambda^2$ Taylor coefficients of the transcribed (@eq:two-formulations-1) and (@eq:two-formulations-2),
> taken by a Cauchy integral of the plasmon energy, equal
> $E_{\mathrm{MP2}}[\mathrm C]$ to $10^{-12}$ relative; the $\lambda^3$
> coefficients equal $E^{(3)}[\mathrm{lr}]$ and $E^{(3)}[\mathrm C] - E^{(3)}[\mathrm{sr}]$
> from the ring recursion (@eq:cross-term-3) to $10^{-11}$, and their difference is not
> zero.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_rs_mp2_rpa_sympy.py`, checks `b-form-second-order`, `t-form-second-order`, `b-form-third-order`, `t-form-third-order`, `mixed-rings-nonzero` |
> | Negative control | the runner subtracts $E_{\mathrm{OS}}[\mathrm{lr}]$ once instead of twice in B, and drops the $2E_{\mathrm{OS}}[\mathrm{sr}]$ that T adds back; each turns its second-order check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/rs_mp2_rpa.rs::rs_mp2_lr_rpa` |
> | Code test | `rs_mp2_rpa_h2o_cc_pvdz_vs_pyscf` in `crates/ferric-rpa/tests/validation_rs_mp2_rpa_pyscf.rs` |
>
> <!-- verified: id=rsrpa.forms.orders; script=scripts/rpa_rs_mp2_rpa_sympy.py; claim=b-form-second-order; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/rs_mp2_rpa.rs::rs_mp2_lr_rpa; rust_test=ferric-rpa/tests/validation_rs_mp2_rpa_pyscf.rs::rs_mp2_rpa_h2o_cc_pvdz_vs_pyscf; registry=rs-mp2-rpa -->

> **Verified equation @eq:rsrpa-forms-one-mode** · the mixed rings in one mode
>
> With $E_{\mathrm{dRPA}}(k)$ of (@eq:rsrpa-one-mode-series), the $\lambda^3$ term of B is
> $k_l^3/(4\Delta^2)$, that of T is $\big((k_s + k_l)^3 - k_s^3\big)/(4\Delta^2)$,
> and T − B is $3k_sk_l(k_s + k_l)/(4\Delta^2)$, symbolically.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_rs_mp2_rpa_sympy.py`, checks `one-mode-b-second`, `one-mode-b-third`, `one-mode-t-third`, `one-mode-t-minus-b` |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/rs_mp2_rpa.rs::RsMp2RpaFormulation` |
>
> <!-- verified: id=rsrpa.forms.one-mode; script=scripts/rpa_rs_mp2_rpa_sympy.py; claim=one-mode-t-minus-b; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/rs_mp2_rpa.rs::RsMp2RpaFormulation; registry=rs-mp2-rpa -->

## The limits {#two-formulations-limits}

As $\omega \to 0$ the erf operator vanishes: $v_{\mathrm{lr}} = 0$ and
$v_{\mathrm{sr}} = 1/r$. Then $\Delta E_{\mathrm{dRPA}}[\mathrm{lr}] = 0$ in
B, the two brackets of T cancel, and both energies are
$E_{\mathrm{MP2}}[\mathrm C]$. As $\omega \to \infty$, $v_{\mathrm{sr}} = 0$
and both are $E_{\mathrm{MP2}}[\mathrm C] + \Delta E_{\mathrm{dRPA}}[\mathrm C]$,
MP2 with every ring beyond second order screened.

The terf attenuator satisfies the same split identity, so it has the same
limits, with the direction reversed: $\omega = 1/(r_0\sqrt2)$, so a large
$r_0$ gives MP2 and a small $r_0$ gives MP2 + ΔdRPA.

> **Verified limits** · $\omega \to 0$ and $\omega \to \infty$
>
> On the model, removing the long-range factor gives $E_{\mathrm{MP2}}[\mathrm C]$
> from both transcriptions, and removing the short-range factor gives
> $E_{\mathrm{MP2}}[\mathrm C] + E_{\mathrm{dRPA}}[\mathrm C] - 2E_{\mathrm{OS}}[\mathrm C]$,
> to $10^{-14}$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_rs_mp2_rpa_sympy.py`, checks `omega-to-zero`, `omega-to-infinity` |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/rs_mp2_rpa.rs::rs_mp2_lr_rpa` |
> | Code test | `omega_to_zero_reduces_to_mp2`, `omega_to_infinity_is_mp2_plus_delta_drpa`, `coupled_rings_omega_to_zero_reduces_to_mp2`, `coupled_rings_omega_to_infinity_is_mp2_plus_delta_drpa`, `terf_large_r0_reduces_to_mp2`, `terf_small_r0_is_mp2_plus_delta_drpa` in `crates/ferric-rpa/src/rs_mp2_rpa.rs` (H2; the terf tests skip without the interpolation tables) |
>
> <!-- verified: id=rsrpa.forms.limits; script=scripts/rpa_rs_mp2_rpa_sympy.py; claim=omega-to-infinity; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/rs_mp2_rpa.rs::rs_mp2_lr_rpa; rust_test=ferric-rpa/src/rs_mp2_rpa.rs::omega_to_infinity_is_mp2_plus_delta_drpa; registry=rs-mp2-rpa -->

## In the code {#two-formulations-code}

The `match cfg.formulation` of `rs_mp2_lr_rpa`, from the
`coupled-rings` arm (quoted, not a checked excerpt):

```rust
let delta_full = drpa_coul.e_rpa - 2.0 * sc_full.e_os;
let delta_sr = drpa_erfc.e_rpa - 2.0 * sc_sr.e_os;
// T: E_MP2[Coulomb] + ΔdRPA[Coulomb] − ΔdRPA[erfc]
let e_corr = sc_full.e_total + delta_full - delta_sr;
```

and from the `delta-lr` arm, with `e_dmp2_lr = 2.0 * sc_lr.e_os`:

```rust
let e_corr = sc_full.e_total + drpa_lr.e_rpa - e_dmp2_lr;
```

| Symbol | Code |
|---|---|
| $E_{\mathrm{MP2}}[v]$, $E_{\mathrm{OS}}[v]$ | `sc_full`, `sc_sr`, `sc_lr` `.e_total`, `.e_os` |
| $E_{\mathrm{dRPA}}[\mathrm{lr}]$, $[\mathrm C]$, $[\mathrm{sr}]$ | `drpa_lr`, `drpa_coul`, `drpa_erfc` `.e_rpa` |
| $v_{\mathrm{lr}}$, $v_{\mathrm{sr}}$ | `op_lr`, `op_sr`: `Operator::erf(ω)`, `Operator::erfc(ω)`, or `terf(r0)`, `terfc(r0)` (with `terf_omega`: `terf_with_omega`, `terfc_with_omega`) |

Every call computes the MP2 components of all three operators, so the
result always carries `e_mp2_full`, `e_sr_mp2`, `e_lr_mp2` and
`e_dmp2_lr` $= 2E_{\mathrm{OS}}[\mathrm{lr}]$. B adds `e_drpa_lr` and the
diagnostic `e_corr_naive` (@eq:cross-term-6); T adds `e_delta_drpa_full` and
`e_delta_drpa_sr`. B makes one dRPA evaluation, T two.

**RI.** Each operator is fitted in its own metric: Cholesky for Coulomb and
erfc (and terfc), a symmetric $V^{-1/2}$ by eigendecomposition with modes
below $10^{-10}$ dropped for erf and terf (`ferric-mp2/src/rimp2.rs::metric_inverse_sqrt`).
The fitted $(ia|jb)_{\mathrm C}$ is therefore not exactly
$(ia|jb)_{\mathrm{sr}} + (ia|jb)_{\mathrm{lr}}$. Second order is unaffected:
each bracket of (@eq:two-formulations-1) and (@eq:two-formulations-2) uses one operator's integrals, so it
starts at third order for any fit. The decomposition (@eq:rsrpa-forms-orders)–(@eq:rsrpa-forms-one-mode), which
compares operators, holds up to the fitting differences.

## Options {#two-formulations-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `formulation` | `"delta-lr"`, `"coupled-rings"` | `"delta-lr"` | (@eq:two-formulations-1) or (@eq:two-formulations-2) |
| `omega` | positive, Å⁻¹ at the CLI and Python (Bohr⁻¹ in Rust) | 0.420 Å⁻¹ = 0.222 Bohr⁻¹ | erf/erfc split; ignored with terf |
| `attenuator` | `"erf"`, `"terf"` | `"erf"` | operator shape; terf needs the interpolation tables (`FERRIC_TERF_TABLE_DIR`) |
| `r0` | positive, Å (Bohr in Rust) | 3.18 Bohr, so that $1/(r_0\sqrt2) \approx 0.222$ Bohr⁻¹ | terf length |
| `terf_omega` | positive, Å⁻¹ | unset: $\omega = 1/(r_0\sqrt2)$ | terf sharpness decoupled from $r_0$ |
| `frozen_core` | integer | 0 | applied to every MP2 and dRPA call |

The dRPA solver and quadrature knobs are those of `pdep-rpa` (section @sec:pdep);
the default here is full rank. The choice of $\omega$ is not benchmarked, and
the CLI prints a notice saying so.

## Anchors {#two-formulations-anchors}

From `site/src/reference/validation.md` (rs-mp2-rpa is graded Proven):

| Check | Systems | Result |
|---|---|---|
| B and T `e_corr` vs PySCF's own `DFMP2` and `gw.rpa.RPA` per operator, default $\omega$ | H2O, C2H4, water dimer / cc-pVDZ, aug-cc-pVDZ; 20 and 40 points | $\le 2.0\times10^{-12}$ Ha; components $\le 5.5\times10^{-11}$ Ha, the erf metric shift |
| every component vs a numpy assembly on the same fitted integrals | H2O, NH3 / cc-pVDZ; H2O / aug-cc-pVDZ; $\omega$ = 0.222, 0.42 Bohr⁻¹ | $\le 4.7\times10^{-11}$ Ha |
| $\omega \to 0$ and $\omega \to \infty$ limits | same | to $4.3\times10^{-11}$ Ha |

The grade covers an RHF reference with exact J and K, the erf attenuator at
the default $\omega$, either formulation, frozen core 0, `trunc_thresh` 0 or
below, and Gauss–Legendre quadrature with $u_0 = 0.5$ and 20 or 40 points.
The terf split is checked by its limits only.

## Known limits (2026-11-15)

- Energy only, no gradient. The closed-shell RHF path is the one the
  sections above derive; the open-shell path is below.
- No benchmark selects $\omega$ or $r_0$; the terf arm has no reference
  beyond its limits.
- The fusion test of section @sec:shared-intermediates covers `coupled-rings` only.

## Open shell

With `multiplicity > 1`, `kind = "rs-mp2-rpa"` and `run_rs_mp2_rpa` take a
UHF reference (energies only), or a ROHF one semi-canonicalized per spin.
`rs_mp2_lr_rpa` then takes `scan_impl_unrestricted`. Each operator builds one pair of spin
intermediates in one aux basis. The pair feeds the unrestricted MP2 blocks
and the spin-summed dRPA, $\boldsymbol\epsilon = \mathbf 1 + \boldsymbol\Pi_\alpha + \boldsymbol\Pi_\beta$.
The subtracted second-order term is not $2E_{\rm OS}$. It is

$$ d = E_{\alpha\beta} + \tfrac12\big(D_{\alpha\alpha} + D_{\beta\beta}\big), \qquad D_{\sigma\sigma} = \sum_{ijab\in\sigma}\frac{(ia|jb)^2}{D_{ij}^{ab}}, $$

the direct part of spin-orbital MP2. It equals $2E_{\rm OS}$ only when the two
spins have the same orbitals. With $d$ in place of $2E_{\rm OS}$, both forms
again equal $E_{\rm MP2}[\mathrm C]$ through second order.

The code names: per operator $v$, `ferric-mp2/src/u_rimp2.rs::u_ri_mp2_from_parts`
gives the U-MP2 blocks $E_{\alpha\alpha}$, $E_{\beta\beta}$, $E_{\alpha\beta}$
from the spin pair $B_\sigma[v]$, and `ferric-rpa/src/lib.rs::u_pdep_rpa_from_intermediates`
gives the spin-summed dRPA of section @sec:open-shell-dielectric.
$D_{\sigma\sigma}$ is the $E_{\rm OS}$ of the closed-shell kernel
`ferric-mp2/src/rimp2.rs::spin_components_from_b_ov` run on spin $\sigma$'s
$B$ and orbital energies. Here $D_{ij}^{ab} = -\Delta_{ij}^{ab}$ is negative.
The two energies are the closed-shell ones with $2E_{\rm OS}$ replaced by $d$:

$$ E^{\rm B}_c = E_{\rm MP2}[\mathrm C] + E_{\rm dRPA}[\mathrm{lr}] - d[\mathrm{lr}], \qquad
E^{\rm T}_c = E_{\rm MP2}[\mathrm C] + \big(E_{\rm dRPA}[\mathrm C] - d[\mathrm C]\big) - \big(E_{\rm dRPA}[\mathrm{sr}] - d[\mathrm{sr}]\big), $$

with $d[v] = E_{\alpha\beta}[v] + \tfrac12\big(D_{\alpha\alpha}[v] + D_{\beta\beta}[v]\big)$.

Why $d$? Expand the dRPA integrand to second order in the integrals:
$\ln(1 + x) - x = -x^2/2 + O(x^3)$, so
$E^{(2)} = -\frac{1}{4\pi}\int_0^\infty\operatorname{tr}\boldsymbol\Pi^2\,du$.
With $\boldsymbol\Pi = \boldsymbol\Pi_\alpha + \boldsymbol\Pi_\beta$ and the
prefactor 2 per spin,

$$ \operatorname{tr}\boldsymbol\Pi^2 = \sum_{\sigma\tau}\sum_{ia\in\sigma}\sum_{jb\in\tau}\frac{4\,\Delta_{ia}\Delta_{jb}\,(ia|jb)^2}{(u^2 + \Delta_{ia}^2)(u^2 + \Delta_{jb}^2)}, \qquad \int_0^\infty\frac{\Delta_1\Delta_2\,du}{(u^2 + \Delta_1^2)(u^2 + \Delta_2^2)} = \frac{\pi}{2(\Delta_1 + \Delta_2)} . $$

So $E^{(2)} = \tfrac12\sum_{\sigma\tau}\sum(ia|jb)^2/D$ with
$D = -(\Delta_{ia} + \Delta_{jb})$. The $\alpha\alpha$ and $\beta\beta$ terms
give $\tfrac12D_{\alpha\alpha} + \tfrac12D_{\beta\beta}$; the $\alpha\beta$ and
$\beta\alpha$ terms together give $E_{\alpha\beta}$. That sum is $d$, the
direct (no-exchange) part of the spin-orbital MP2 energy,
$\tfrac12\sum_{ijab}\langle ij|ab\rangle^2/D$.

$2E_{\alpha\beta}$ equals $d$ only when
$D_{\alpha\alpha} = D_{\beta\beta} = E_{\alpha\beta}$, which holds when the
two spins have the same orbitals. Check the singlet limit: run through the
UHF path, $d = 2E_{\rm OS}$, the U-dRPA equals the closed-shell dRPA
(prefactors $2 + 2 = 4$), and every open-shell result equals the restricted
one. Every bracket again starts at third order, so both forms equal the
open-shell $E_{\rm MP2}[\mathrm C] = E_{\alpha\alpha} + E_{\beta\beta} + E_{\alpha\beta}$
through second order, and the limits are the closed-shell ones:
$v_{\rm lr} = 0$ gives $E_{\rm MP2}[\mathrm C]$, and $v_{\rm sr} = 0$ gives
$E_{\rm MP2}[\mathrm C] + E_{\rm dRPA}[\mathrm C] - d[\mathrm C]$.

The two spin sets of one operator are co-resident, so the pre-flight charges
both (`ferric-rpa/src/budget.rs::scan_charge_unrestricted`).
`scripts/u_attenuated_sympy.py` checks $d$ symbolically and against the
Cauchy-integral $\lambda^2$ coefficient of the spin-orbital plasmon energy,
the second-order and limit statements, and the singlet limit. On OH, NH2 and O2
at cc-pVDZ the T form matches PySCF (`DFUMP2` + `URPA`) to ≤ 1.5e-13 Ha.
Using $2E_{\alpha\beta}$ instead moves T by 1.9e-4 to 3.4e-3 Ha. The B form
needs the erf fit, whose metric PySCF factors differently, so it is checked
against numpy on ferric's integrals (≤ 7e-14 Ha). The Boys-screened $\chi_0$
is closed shell only.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-rpa/theory.md`, "Range-separated MP2 + RPA" | (@eq:two-formulations-1)–(@eq:rsrpa-forms-one-mode), the limits |
| `crates/ferric-rpa/theory.md`, "Open-shell SR-MP2 + LR-RPA" | the open-shell $d$ |
| `crates/ferric-rpa/src/rs_mp2_rpa.rs` (the `match` arms, `RsMp2RpaConfig`, `RsMp2RpaResult`) | @sec:two-formulations-code, @sec:two-formulations-options |
| `crates/ferric-mp2/src/rimp2.rs::metric_inverse_sqrt` | the RI paragraph |
| `scripts/rpa_rs_mp2_rpa_sympy.py` | the verified boxes |
| `site/src/reference/validation.md` | @sec:two-formulations-anchors |

Changes from the sources: `theory.md` writes the denominator as the negative
$D$; this section uses $\Delta$ (section @sec:cross-term).
