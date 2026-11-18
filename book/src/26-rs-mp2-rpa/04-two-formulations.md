# The two formulations in the code {#two-formulations}

<!-- status: ready; registry: rs-mp2-rpa, u-rs-mp2-rpa -->

MP2 lets every pair of electrons interact once through a ring of two
particle–hole excitations, and it does so without screening. Two
polarizable fragments far apart then correlate too strongly, because the
medium between them never weakens the interaction. The dRPA screens the
interaction by summing rings of every length. `rs_mp2_lr_rpa` keeps
full-Coulomb MP2, exchange and cross term included, and adds dRPA rings
beyond second order. Its two formulations differ in which of those rings
they keep.

The plan:

1. write both energies and check their second and third orders
   (@sec:two-formulations-energies) and their limits (@sec:two-formulations-limits);
2. explain the terf long range, which has no positive definite kernel
   (@sec:two-formulations-terf);
3. read the code, a run on water, the options and the anchors.

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
from the purely short-range rings. Each bracket uses one operator's fitted
integrals for both of its terms. With erf every $E_{\mathrm{dRPA}}$ is a
full-rank PDEP evaluation (section @sec:pdep; `trunc_thresh` is 0 by
default). With terf, $E_{\mathrm{dRPA}}[\mathrm{lr}]$ of B is the signed
log-determinant of section @sec:two-formulations-terf, and T's two dRPA
energies are PDEP evaluations with Coulomb and terfc.

Scale every integral by $\lambda$. Both energies then have the
second-order term $E_{\mathrm{MP2}}[\mathrm C]$, exchange and the cross term
$X$ included, because each bracket starts at third order (section
@sec:cross-term-rule). At third order, from (@eq:cross-term-3),

$$ E^{\mathrm B\,(3)}_c = E^{(3)}_{\mathrm{dRPA}}[\mathrm{lr}], \qquad E^{\mathrm T\,(3)}_c = E^{(3)}_{\mathrm{dRPA}}[\mathrm C] - E^{(3)}_{\mathrm{dRPA}}[\mathrm{sr}] . \label{eq:rsrpa-forms-orders} $$

In one mode, write $k = k_s + k_l$ and use (@eq:rsrpa-one-mode-series):

$$ E^{\mathrm T\,(3)}_c - E^{\mathrm B\,(3)}_c = \frac{(k_s + k_l)^3 - k_s^3 - k_l^3}{4\Delta^2} = \frac{3k_sk_l(k_s + k_l)}{4\Delta^2} : \label{eq:rsrpa-forms-one-mode} $$

these are the third-order rings with both short- and long-range
interactions. T keeps them and B drops them. Neither formulation keeps the
purely short-range rings beyond second order. The direct part of the cross term $X$ of (@eq:cross-term-2)
is the second-order ring with one short-range and one long-range
interaction; the mixed rings of T are that coloring at third order and
beyond.

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
limits with the direction reversed: $\omega = 1/(r_0\sqrt2)$, so a large
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

The terf tests reach their limits at different distances. On H2 / cc-pVDZ,
built from the `1fa635be6` sources, B equals RI-MP2 to $8.2\times10^{-13}$ Ha
and T to $2.1\times10^{-9}$ Ha at $r_0 = 20$ Bohr. At $r_0 = 0.05$ Bohr, the
smallest the tables reach, B is $-1.79\times10^{-4}$ Ha and T
$-7.1\times10^{-9}$ Ha from MP2 + ΔdRPA[C], inside the test's
$5\times10^{-4}$ Ha bar. The residual is the finite range of terf at
$\omega \approx 14$ Bohr⁻¹, the same truncation the erf test removes with
$\omega = 200$ Bohr⁻¹.

## The terf long range {#two-formulations-terf}

### Why the terf kernel is not positive definite {#two-formulations-terf-indefinite}

A Coulomb-like kernel is positive definite when every nonzero charge
distribution has positive self-interaction through it. Coulomb, erf, erfc
and terfc pass. terf fails, and a two-charge picture shows why. Put $+q$ at
the origin and $-q$ at distance $r$. With a kernel $w$ that is finite at
zero separation, the self-interaction is

$$ (\rho|w|\rho) = q^2\big[2w(0) - 2w(r)\big], $$

negative as soon as $w(r) > w(0)$. The erf kernel $\operatorname{erf}(\omega r)/r$
is largest at $r = 0$. The terf kernel switches on near $r = r_0$ instead
(@eq:erf-terf-2): for $r_0\omega > 1/\sqrt2$ its curvature at the origin,
$\tau_3$ of (@eq:op-terf-curvature-constraint), is positive, and it rises to
a maximum at $1.26$ to $1.39\,r_0$ for $r_0\omega = 1$, 2 and 4. At $r_0\omega = 2$, $w(0) = 0.041/r_0$ and
$w(r_0) = 0.500/r_0$, so the pair at separation $r_0$ has interaction
$-0.917\,q^2/r_0$. At $r_0\omega = 1/\sqrt2$ the maximum sits at the origin
and the two-charge argument does not apply, but the Fourier transform
$\cos(kr_0)\,e^{-k^2/4\omega^2}$ is still negative at $kr_0 = \pi$ (section
@sec:erf-terf-properties-decide-integrals-used), so the kernel is
indefinite for every finite $r_0$ and $\omega$. A density modulated at
wavelength near $2r_0$ lowers its energy through terf.

> **Verified claim** · the two-charge interaction of the terf kernel
>
> With $W(r) = \operatorname{terf}(r)/r$, $W(0) = (2\omega/\sqrt\pi)\,e^{-\omega^2r_0^2}$
> and $W(r_0) = \operatorname{erf}(2\omega r_0)/(2r_0)$ symbolically. The pair
> interaction $2W(0) - 2W(r_0)$ is $-0.165$, $-0.917$ and $-1.000\,q^2/r_0$ at
> $r_0\omega = 1$, 2 and 4, and $+0.013\,q^2/r_0$ at $1/\sqrt2$. The
> maximum of $W$ lies at $1.32$, $1.39$ and $1.26\,r_0$ for $r_0\omega = 1$,
> 2 and 4. The momentum-space integral of
> $\cos(kr_0)e^{-k^2/4\omega^2}$, which uses no real-space kernel, reproduces
> the pair interaction to $3\times10^{-22}$, and the transform is negative at
> $kr_0 = \pi$ for every $\omega$ and $r_0$ tried.
>
> | | |
> |---|---|
> | Proof | `scripts/terf_kernel_sign_sympy.py`, checks `two-charge`, `two-charge-fourier` |
> | Negative control | the first error function takes $r + r_0$; the transform loses its cosine; each turns its check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/rs_mp2_rpa.rs::rs_mp2_lr_rpa` |
>
> <!-- verified: id=rsrpa.terf-two-charge; script=scripts/terf_kernel_sign_sympy.py; claim=two-charge; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/rs_mp2_rpa.rs::rs_mp2_lr_rpa; registry=rs-mp2-rpa -->

What does that do to the fit? The metric $V_{\mathrm t} = (P|\operatorname{terf}|Q)$
inherits the sign. On Ne$_2$ at 3.6 Å with aug-cc-pVDZ-RIFIT and
$r_0 = 1.35$ Å, 68, 78 and 70 of its 144 eigenvalues are negative at
$r_0\omega = 1/\sqrt2$, 2 and 4 (`testdata/reference/ne2_terf_signed_fit.json`).
An indefinite matrix has no real $M$ with $M^{\mathsf T}M = V_{\mathrm t}^{-1}$,
so the factorization (@eq:factorization-and-metric-1) does not exist. A
factor built from the positive eigenvalues alone describes a different,
positive definite kernel (section @sec:factorization-and-metric-signed).

### The signed factor {#two-formulations-terf-signed}

ferric never fits terf. It fits Coulomb and terfc, both positive definite,
each by Cholesky in its own metric, and takes the difference. That is
(@eq:factorization-and-metric-signed): a factor with $2N_{\mathrm{aux}}$ rows and
signature $\boldsymbol\Sigma = \operatorname{diag}(\mathbf 1, -\mathbf 1)$.
Three properties make it the right construction.

- Each fit is the stationary fit of its own positive definite kernel, so
  each error is second order in its own residual (section
  @sec:factorization-and-metric-factorization).
- For a density in the auxiliary span it reproduces $d^{\mathsf T}V_{\mathrm t}d$
  exactly, negative modes included.
- The split is exact on the fitted integrals,
  $(ia|jb)_{\mathrm{lr}} = (ia|jb)_{\mathrm C} - (ia|jb)_{\mathrm{sr}}$, so
  (@eq:cross-term-2) holds exactly for them, and the long-range MP2
  components and B's long-range dRPA share one set of integrals with the
  Coulomb and terfc ones.

MP2 needs only the integrals: `ferric-mp2/src/rimp2.rs::signed_pair_block`
forms each pair block with two GEMMs. The dRPA energy needs a determinant.
Sylvester's identity (@eq:factorization-and-metric-signed-sylvester) moves it
from the excitation space to the $2N_{\mathrm{aux}}$-dimensional space,
with $\boldsymbol\Sigma$ inside. $\boldsymbol\Sigma\mathbf B\chi_0\mathbf B^{\mathsf T}$
is not symmetric, so `ferric-rpa/src/energy.rs::eval_signed_trace_log_summands`
takes its log-determinant by LU. The frequency integral equals the plasmon
energy of the indefinite kernel while the determinant stays positive at
every frequency. In one mode that is the condition under which the square
root of (@eq:rsrpa-one-mode-series) is real, $k > -\Delta/2$: a negative
coupling stronger than that makes the reference unstable, and the LU routine
refuses the nonpositive determinant.

### The sign of B's correction {#two-formulations-terf-sign}

In one mode the bracket $\Delta E_{\mathrm{dRPA}}(k) = E_{\mathrm{dRPA}}(k) + k^2/(4\Delta)$
has the sign of $k$ for every $k > -\Delta/2$. With
$s = \sqrt{1 + 2k/\Delta}$, which is positive exactly on that range,
$k = \Delta(s-1)(s+1)/2$ and the bracket is
$\Delta\,(s-1)^3(s+3)/16$, so its sign is that of $s - 1$, which is that of $k$.
A positive coupling, the only kind a positive definite kernel has, gives a
positive bracket: the higher rings screen and weaken the correlation. A
negative mode gives a negative bracket: the higher rings deepen it. The
terf kernel has negative modes, so B's correction with terf need not have
the screening sign. The one-mode model shows how the sign can flip; it does
not predict the many-mode sign.

> **Verified claim** · the sign of the one-mode bracket
>
> $\tfrac12\big(\sqrt{\Delta(\Delta+2k)} - \Delta - k\big) + k^2/(4\Delta)$
> equals $\Delta(s-1)^3(s+3)/16$ symbolically, and a scan over
> $-\Delta/2 < k \le 5\Delta$ finds the sign of $k$. The frequency integral
> $\frac1{2\pi}\int_0^\infty[\ln(1+p) - p]\,d\omega$ with
> $p = 2\Delta k/(\Delta^2 + \omega^2)$ equals the closed form to $10^{-19}$ at
> seven couplings from $-0.45\Delta$ to $5\Delta$. At $k \le -\Delta/2$ the
> determinant $1 + p(0)$ is nonpositive.
>
> | | |
> |---|---|
> | Proof | `scripts/terf_kernel_sign_sympy.py`, checks `one-mode-sign`, `one-mode-integral` |
> | Negative control | the shift $k^2/(2\Delta)$ replaces $k^2/(4\Delta)$; the check turns red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/energy.rs::eval_signed_trace_log_summands` |
>
> <!-- verified: id=rsrpa.terf-one-mode-sign; script=scripts/terf_kernel_sign_sympy.py; claim=one-mode-sign; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/energy.rs::eval_signed_trace_log_summands; registry=rs-mp2-rpa -->

The measurement is the counterpoise interaction energy of $\Delta E_{\mathrm{dRPA}}[\mathrm{lr}]$
on Ne$_2$ at 3.6 Å, aug-cc-pVDZ / aug-cc-pVDZ-RIFIT, $r_0 = 1.35$ Å, in µHa.
Every column comes from one run of
`terf_lr_matches_independent_constructions_ne2`
(`crates/ferric-rpa/tests/terf_signed_fit.rs`), built from the `1fa635be6`
sources. The ferric column is `rs_mp2_lr_rpa`; the other four are the
constructions the test builds from the raw integrals
(`scripts/gen_terf_signed_fit_refs.py` records them in
`testdata/reference/ne2_terf_signed_fit.json`):

| $r_0\omega$ | ferric | Coulomb RI − terfc RI | signed pseudo-inverse of $V_{\mathrm t}$ | Coulomb-fitted densities, $V_{\mathrm t}$ between | positive eigenvalues of $V_{\mathrm t}$ only |
|---|---|---|---|---|---|
| 2 | −3.1143 | −3.1141 | −3.1141 | −3.1965 | +23.6172 |
| 4 | −5.3226 | −5.3226 | −5.3217 | −5.5193 | +82.9808 |

The same quantity with the full Coulomb operator, from the JSON file, is
$+9.56$ µHa. ferric and the next three constructions agree to within
0.20 µHa, inside the test's 0.5 µHa bar; ferric's dimer and monomer
$E_{\mathrm{MP2}}$ pieces equal Coulomb RI − terfc RI to $2\times10^{-16}$ Ha. The positive-subspace fit misses by 27 and 88 µHa
and has the opposite sign: it computes the interaction under the positive
part of the kernel alone.

Interpretation, on this one system: with terf, B's long-range correction
binds Ne$_2$ more strongly, where the Coulomb ring correction binds it less.
That sign belongs to the kernel, since three independent fits give it. No
benchmark yet says whether it helps or hurts accuracy.

### What the code refuses, and why {#two-formulations-terf-refusals}

- **A terf metric factor.** `ferric-mp2/src/rimp2.rs::metric_inverse_sqrt`
  returns an error for terf: an indefinite metric has no inverse square root.
- **PDEP truncation with B and terf** (`trunc_thresh > 0`). Truncation keeps
  the leading modes of a symmetric static dielectric matrix. The signed
  log-determinant forms no symmetric matrix and no eigenvalues, so there is
  nothing to truncate.
- **The Laplace $\chi_0$ backend with B and terf.** The signed route
  implements one $\chi_0$: dense, with the exact energy denominator at each
  frequency.
- **The Boys-screened $\chi_0$ with B and terf**, including an `Auto`
  sparsity that resolves to it for a molecule with at least the cutoff
  number of atoms. The signed route has no screened build.
- **The inverse dielectric with B and terf** (`need_inv_dielectric_freq`).
  The signed route returns only the per-frequency energy summands.
- **A nonpositive determinant** at any frequency, in
  `eval_signed_trace_log_summands`: its logarithm is not real and the
  plasmon formula does not hold there.
- **Schwarz screening of terf integrals** (section
  @sec:erf-terf-properties-decide-integrals-used): a kernel that is not
  positive definite has no Cauchy–Schwarz bound.

The first five refusals come from `refuse_unsupported_signed_drpa`, which
runs only when B is requested with terf, before any integral is built. Four
of them are option refusals (`trunc_thresh`, the Laplace $\chi_0$, the
Boys-screened $\chi_0$, `need_inv_dielectric_freq`) and return
`FerricError::Unsupported { feature, reason }`: `feature` names the option as
the caller set it, `reason` gives the cause and the supported alternative
(`trunc_thresh = 0`, dense $\chi_0$). `crates/ferric-rpa/tests/terf_refusals.rs`
triggers each, closed shell and for truncation open shell, and shows that erf
with B and terf with T take the same options. T with terf needs none of them:
its dRPA energies use Coulomb and terfc. The CLI reads only `trunc_thresh` from
`[rpa]` for `rs-mp2-rpa`, and `run_rs_mp2_rpa` exposes no dRPA option, so the
others are reachable from Rust through `RsMp2RpaConfig::drpa` only. The Python
binding maps `Unsupported` to `NotImplementedError`.

## In the code {#two-formulations-code}

`Rings::assemble` turns one operator set's pieces into either energy. The
`match` on the formulation, unchanged except that the two `Option` fields
of each arm's tuple that the arm leaves `None` are cut:

<!-- excerpt: ferric-rpa/src/rs_mp2_rpa.rs::assemble -->
```rust
RsMp2RpaFormulation::DeltaLr => {
    let e_drpa = self.drpa_lr.expect("B requested => LR dRPA computed");
    (
        self.full.e_total + e_drpa - self.lr.dmp2,
```

<!-- excerpt: ferric-rpa/src/rs_mp2_rpa.rs::assemble -->
```rust
RsMp2RpaFormulation::CoupledRings => {
    let e_coul = self
        .drpa_coulomb
        .expect("T requested => Coulomb dRPA computed");
    let e_sr = self.drpa_sr.expect("T requested => SR dRPA computed");
    let delta_full = e_coul - self.full.dmp2;
    let delta_sr = e_sr - self.sr.dmp2;
    (
        self.full.e_total + delta_full - delta_sr,
```

- The B arm is (@eq:two-formulations-1): `self.full.e_total` is
  $E_{\mathrm{MP2}}[\mathrm C]$, `e_drpa - self.lr.dmp2` is
  $\Delta E_{\mathrm{dRPA}}[\mathrm{lr}]$. In the T arm `delta_full` and
  `delta_sr` are $\Delta E_{\mathrm{dRPA}}[\mathrm C]$ and
  $\Delta E_{\mathrm{dRPA}}[\mathrm{sr}]$ of (@eq:two-formulations-2).
- `dmp2` is $2E_{\mathrm{OS}}$ for a closed-shell reference and $d$ of the
  open-shell section below for an open-shell one; `assemble` does not know
  which.
- The B arm also returns `e_drpa_lr` and the diagnostic
  `e_corr_naive` $= E_{\mathrm{MP2}}[\mathrm{sr}] + E_{\mathrm{dRPA}}[\mathrm{lr}]$
  (@eq:cross-term-6); the T arm returns the two brackets.
- With terf, `self.lr` comes from the signed MP2 kernel
  `ferric-mp2/src/rimp2.rs::spin_components_from_signed_b_ov` and
  `self.drpa_lr` from `ferric-rpa/src/energy.rs::signed_drpa_energy`; the
  assembly is the same.

| Symbol | Code |
|---|---|
| $E_{\mathrm{MP2}}[v]$, $2E_{\mathrm{OS}}[v]$ | `self.full`, `self.sr`, `self.lr` `.e_total`, `.dmp2` |
| $E_{\mathrm{dRPA}}[\mathrm{lr}]$, $[\mathrm C]$, $[\mathrm{sr}]$ | `self.drpa_lr`, `self.drpa_coulomb`, `self.drpa_sr` |
| $v_{\mathrm{lr}}$, $v_{\mathrm{sr}}$ | `Operator::erf(ω)`, `Operator::erfc(ω)`, or `terf(r0)`, `terfc(r0)` (with `terf_omega`: `terf_with_omega`, `terfc_with_omega`) |

Every call computes the MP2 components of all three operators, so the
result always carries `e_mp2_full`, `e_sr_mp2`, `e_lr_mp2` and
`e_dmp2_lr` $= 2E_{\mathrm{OS}}[\mathrm{lr}]$. B makes one dRPA evaluation,
T two.

**RI.** Coulomb, erfc and terfc are each fitted in their own metric by
Cholesky; erf by a symmetric $V^{-1/2}$ from an eigendecomposition with
modes below $10^{-10}$ dropped (`ferric-mp2/src/rimp2.rs::metric_inverse_sqrt`).
For erf the fitted $(ia|jb)_{\mathrm C}$ is therefore not exactly
$(ia|jb)_{\mathrm{sr}} + (ia|jb)_{\mathrm{lr}}$, and the decomposition
(@eq:rsrpa-forms-orders)–(@eq:rsrpa-forms-one-mode), which compares
operators, holds up to the fitting differences. For terf the split is exact
on the fitted integrals (section @sec:two-formulations-terf-signed). Second
order is unaffected either way: each bracket of (@eq:two-formulations-1) and
(@eq:two-formulations-2) uses one operator's integrals, so it starts at
third order for any fit.

## A run on water {#two-formulations-run}

Water in cc-pVDZ, both formulations and both attenuators at their defaults,
from the `18375bead` build. `run_rs_mp2_rpa` uses RI-JK for the reference by
default; `scf_jk="exact"` gives the exact-J/K reference the anchors use. The
`os.environ` line points at the repository's `terf-tables/`:

```python
import os

os.environ.setdefault("FERRIC_TERF_TABLE_DIR", "terf-tables")  # the terf and terfc interpolation tables
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

for att in ("erf", "terf"):                    # erf at omega = 0.420 A^-1; terf at r0 = 1.6828 A
    b, t = (ferric.run_rs_mp2_rpa(mol, bs, aux, attenuator=att, formulation=f, scf_jk="exact")
            for f in ("delta-lr", "coupled-rings"))
    print(f"{att:4s}  E_MP2[C] = {b.e_mp2_full:.8f}  E_MP2[lr] = {b.e_lr_mp2:.8f}  2E_OS[lr] = {b.e_dmp2_lr:.8f}")
    print(f"      B: E_dRPA[lr] = {b.e_drpa_lr:.8f}  E_c = {b.e_corr:.8f}  naive A = {b.e_corr_naive:.8f}")
    print(f"      T: dRPA[C] - 2E_OS[C] = {t.e_delta_drpa_full:.8f}  dRPA[sr] - 2E_OS[sr] = {t.e_delta_drpa_sr:.8f}  E_c = {t.e_corr:.8f}")
```

```text
erf   E_MP2[C] = -0.20403346  E_MP2[lr] = -0.00014733  2E_OS[lr] = -0.00019046
      B: E_dRPA[lr] = -0.00018798  E_c = -0.20403098  naive A = -0.19883623
      T: dRPA[C] - 2E_OS[C] = 0.07365746  dRPA[sr] - 2E_OS[sr] = 0.07149869  E_c = -0.20187469
terf  E_MP2[C] = -0.20403346  E_MP2[lr] = -0.00000244  2E_OS[lr] = -0.00000318
      B: E_dRPA[lr] = -0.00000318  E_c = -0.20403345  naive A = -0.20407882
      T: dRPA[C] - 2E_OS[C] = 0.07365746  dRPA[sr] - 2E_OS[sr] = 0.07373603  E_c = -0.20411203
```

With erf, B lies 2.48 µHa above MP2 and T 2.16 mEh above it; the naive
sum A of (@eq:cross-term-6) lies 5.20 mEh above, the cross term and the
long-range exchange it drops. With terf at the default $r_0 = 3.18$ Bohr
the long-range MP2 energy is only 2.4 µHa, so B is within 0.01 µHa of MP2.
T is 78.6 µHa below MP2 there: its short-range bracket is larger than its
Coulomb one.

## Options {#two-formulations-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `formulation` | `"delta-lr"`, `"coupled-rings"` | `"delta-lr"` | (@eq:two-formulations-1) or (@eq:two-formulations-2) |
| `omega` | positive, Å⁻¹ at the CLI and Python (Bohr⁻¹ in Rust) | 0.420 Å⁻¹ = 0.222 Bohr⁻¹ | erf/erfc split; ignored with terf |
| `attenuator` | `"erf"`, `"terf"` | `"erf"` | operator shape; terf needs the interpolation tables (`FERRIC_TERF_TABLE_DIR`) |
| `r0` | positive, Å (Bohr in Rust) | 3.18 Bohr, so that $1/(r_0\sqrt2) \approx 0.222$ Bohr⁻¹ | terf length |
| `terf_omega` | positive, Å⁻¹ | unset: $\omega = 1/(r_0\sqrt2)$ | terf sharpness decoupled from $r_0$ |
| `frozen_core` | integer | 0 | applied to every MP2 and dRPA call |

The dRPA runs at full rank by default (`RsMp2RpaConfig::drpa`, a
`PdepRpaConfig` with `trunc_thresh = 0`). The CLI reads only `trunc_thresh`
from `[rpa]`; the Python entry point exposes no dRPA option. The choice of
$\omega$ is not benchmarked, and the CLI prints a notice saying so.

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
The terf split is checked by its limits and, for B's long range, by the
Ne$_2$ comparison of section @sec:two-formulations-terf-sign; the grade does
not cover it.

## Known limits (2026-11-17)

- Energy only, no gradient. The closed-shell RHF path is the one the
  sections above derive; the open-shell path is below.
- No benchmark selects $\omega$ or $r_0$.
- B with terf is compared against independent fits on one system, Ne$_2$
  at one geometry, $r_0 = 1.35$ Å and $r_0\omega \in \{2, 4\}$. The
  open-shell terf B path (`open_shell_signed_parts`) has no molecular test;
  its signed log-determinant is checked on a random model
  (`signed_logdet_matches_plasmon_open_shell` in
  `crates/ferric-rpa/src/energy.rs`).
- No test triggers the refusals of section @sec:two-formulations-terf-refusals.
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
With terf the long range uses the signed factor per spin:
`ferric-mp2/src/u_rimp2.rs::u_ri_mp2_signed_from_parts` for the U-MP2 blocks,
the signed closed-shell kernel for $D_{\sigma\sigma}$, and
`signed_drpa_energy` with one channel per spin (prefactor 2 each) for the
dRPA. The two energies are the closed-shell ones with $2E_{\rm OS}$ replaced by $d$:

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
| section @sec:cross-term-split-energy | (@eq:two-formulations-1)–(@eq:rsrpa-forms-one-mode), the limits |
| `crates/ferric-rpa/src/rs_mp2_rpa.rs` (`Rings::assemble`, `refuse_unsupported_signed_drpa`, `RsMp2RpaConfig`, `RsMp2RpaResult`) | @sec:two-formulations-code, @sec:two-formulations-terf-refusals, @sec:two-formulations-options |
| `crates/ferric-mp2/src/rimp2.rs::metric_inverse_sqrt` | the RI paragraph |
| section @sec:factorization-and-metric-signed, `scripts/rpa_terf_signed_fit_sympy.py` | @sec:two-formulations-terf-signed |
| `crates/ferric-rpa/tests/terf_signed_fit.rs`, `testdata/reference/ne2_terf_signed_fit.json` | the Ne$_2$ table |
| `scripts/rpa_rs_mp2_rpa_sympy.py` | the verified boxes |
| `scripts/terf_kernel_sign_sympy.py` | the two verified claims of @sec:two-formulations-terf-indefinite and @sec:two-formulations-terf-sign |
| `crates/ferric-rpa/tests/terf_refusals.rs` | @sec:two-formulations-terf-refusals |
| `site/src/reference/validation.md` | @sec:two-formulations-anchors |

The script writes the denominator as the negative $D$; this section uses
$\Delta$ (section @sec:cross-term). `scripts/terf_kernel_sign_sympy.py` checks the two-charge interaction and the
sign rule of the one-mode bracket with this section's $\Delta$ and $k$.
