# COHSEX {#cohsex}

<!-- status: ready; registry: gw -->

Suppose the electrons respond instantly to an added or removed charge. Then
$W$ does not depend on frequency, the integral of section @sec:correlation-self-energy is
elementary, and the self-energy splits into two pieces with a plain physical
meaning. The screened exchange (SEX) is Hartree–Fock exchange with $v$
replaced by $W$. The Coulomb hole (COH) is the energy of the electron in the
polarization it induces around itself. COHSEX is that static limit: no
frequency grid for the integral, no continuation and no root search, at the
cost of treating the screening as instantaneous.

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

## The static limit {#cohsex-static-limit}

**Do the integral.** Put a frequency-independent $W_{pq}$ into the integral
of section @sec:correlation-self-energy. $W$ comes out of the integral, and the kernel integral is the
one evaluated there, $\int_0^\infty a/(a^2+u^2)\,du = \tfrac\pi2\,\mathrm{sgn}(a)$:

$$ -\frac1\pi\int_0^\infty W\,\frac{a}{a^2+u^2}\,du = -\tfrac12\, W\,\mathrm{sgn}(a), \qquad a = z - \epsilon_q . $$

**Sort by occupation.** Take $z$ in the gap. Every occupied $i$ has
$a > 0$ and every virtual $a$ has $a < 0$, so

$$ \Sigma^c_p = -\tfrac12\sum_i W_{pi} + \tfrac12\sum_a W_{pa} . $$

Add and subtract $\tfrac12\sum_i W_{pi}$ to turn the virtual sum into a sum
over all states:

$$ \Sigma^c_p = \underbrace{-\sum_i W_{pi}}_{\Delta\Sigma^{\rm SEX}_p} + \underbrace{\tfrac12 \sum_q W_{pq}}_{\Sigma^{\rm COH}_p} . $$

The first term has the form of exchange: a sum over occupied orbitals with a
minus sign. The second sums over the complete set of states, occupied and
virtual.

**Check the plasmon limit.** Section @sec:correlation-self-energy gives the single-pole
self-energy in closed form. Hold the static value $W(0) = -2s/\Omega$ fixed and
let $\Omega \to \infty$, so that $s = -W(0)\,\Omega/2$. An occupied term
$s/(z-\epsilon_i+\Omega)$ tends to $s/\Omega = -\tfrac12W(0)$, and a virtual
term $s/(z-\epsilon_a-\Omega)$ tends to $-s/\Omega = +\tfrac12W(0)$. These are
the two terms above: a response infinitely fast compared with the electronic
energies is a static one.

> **Verified equation {#eq:gw-cohsex-static-limit}** · the static-$W$ limit of the integral is COHSEX
>
> A frequency-independent $W_{pq}$ in the integral of box @eq:gw-sigma-c-closed-form gives
> $-\tfrac12W\,\mathrm{sgn}(\mathrm{Re}(z-\epsilon_q))$, which at $z$ in the
> gap is $-\sum_iW_{pi} + \tfrac12\sum_qW_{pq}$. The plasmon model of
> section @sec:screened-interaction reaches the same limit as $\Omega\to\infty$ at fixed $W(0)$.
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

**In the PDEP basis.** At $u = 0$ the dynamic matrix is diagonal with weights
$w_\xi$ (section @sec:screened-interaction-pdep-basis), so $W_{pq}(0) = \sum_\xi w_\xi (B^\xi_{pq})^2$.
Substitute:

$$ \Delta\Sigma^{\rm SEX}_p = -\sum_i\sum_\xi w_\xi (B^\xi_{pi})^2, \qquad \Sigma^{\rm COH}_p = \tfrac12\sum_q\sum_\xi w_\xi (B^\xi_{pq})^2, \qquad w_\xi = \frac1{\lambda_\xi(0)} - 1 . $$

Add the bare exchange of section @sec:ri-factors-and-exchange. For a complete PDEP basis
$\sum_\xi(B^\xi_{pi})^2 = (pi|ip)$ (box @eq:gw-pdep-complete), and $1 + w_\xi = \lambda_\xi(0)^{-1}$,
so the full screened exchange is

$$ \Sigma^x_p + \Delta\Sigma^{\rm SEX}_p = -\sum_i\sum_\xi \lambda_\xi(0)^{-1}\,(B^\xi_{pi})^2 . $$

This is Hartree–Fock exchange with every mode weakened by its own dielectric
eigenvalue.

**What the Coulomb hole means.** The sum in $\Sigma^{\rm COH}_p$ runs over a
complete set of states. Its closure,
$\sum_q\phi_q(\mathbf r)\phi_q(\mathbf r') = \delta(\mathbf r - \mathbf r')$,
turns it into $\tfrac12\int|\phi_p(\mathbf r)|^2 W_d(\mathbf r, \mathbf r)\,d\mathbf r$:
half the interaction of a charge with the polarization it induces at its own
position. $W_d$ is negative, so the Coulomb hole lowers the energy of both
occupied and virtual states.

**The quasiparticle energy.** The self-energy does not depend on energy, so
there is no equation to solve:

$$ \epsilon^{\rm QP}_p = \epsilon_p + \Delta\Sigma^{\rm SEX}_p + \Sigma^{\rm COH}_p \;[+\;\Sigma^x_p - v^{xc}_p \text{ at a KS reference}], \qquad Z = 1 . $$

`ferric-gw/src/cohsex.rs::run_cohsex` assembles it, adding the KS shift as a
plain additive term, and `GwResult.z_factor` is all ones.

COHSEX next to G0W0 for water / cc-pVDZ at HF, from the `589b50832` build,
for the six states around the gap. The $Z$ column is 1 for every state.
COHSEX lowers the occupied energies below G0W0 by about 1 eV here and
lowers the virtual ones by about 0.1 eV.

```python
# Water / cc-pVDZ at HF: COHSEX against G0W0 for HOMO-2 ... LUMO+2.
import ferric

HA_EV = 27.211386245988
mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

kw = dict(n_quad=100, trunc_thresh=0.0)
cx = ferric.run_gw(mol, bs, aux, method="cohsex", **kw)
g0 = ferric.run_gw(mol, bs, aux, method="g0w0", **kw)
print(" MO   eps_HF(eV)  COHSEX(eV)  G0W0(eV)  Z_cohsex")
for k, mo in enumerate(cx.mo_indices):
    print(f"{mo:3d} {cx.eps_mf[k]*HA_EV:11.4f} {cx.eps_qp[k]*HA_EV:11.4f} "
          f"{g0.eps_qp[k]*HA_EV:10.4f} {cx.z_factor[k]:6.3f}")
```

```text
 MO   eps_HF(eV)  COHSEX(eV)  G0W0(eV)  Z_cohsex
  2    -18.9990    -19.7179   -18.5312  1.000
  3    -15.4359    -15.5074   -14.4566  1.000
  4    -13.4218    -13.1360   -12.1600  1.000
  5      5.0444      4.5861     4.7034  1.000
  6      6.9717      6.5239     6.6555  1.000
  7     21.4290     20.2017    20.3167  1.000
```

The two sums for one state, from `ferric-gw/src/cohsex.rs::cohsex_pieces`
(`alpha` is the mode $\xi$, `m_idx` the state $p$):

<!-- excerpt: ferric-gw/src/cohsex.rs::cohsex_pieces -->
```rust
        for alpha in 0..m_modes {
            let w_a = w_static[alpha];
            // SEX: sum over occupied i.
            let mut sex_acc = 0.0;
            for i in 0..n_occ {
                let v = m_proj[(alpha, m_idx, i)];
                sex_acc += v * v;
            }
            delta_sex -= w_a * sex_acc;
            // COH: sum over ALL p (occ + vir).
            let mut coh_acc = 0.0;
            for p in 0..n_act {
                let v = m_proj[(alpha, m_idx, p)];
                coh_acc += v * v;
            }
            coh += 0.5 * w_a * coh_acc;
        }
```

| Code | Equation |
|---|---|
| `w_a` | $w_\xi$ |
| `m_proj[(alpha, m_idx, i)]` | $B^\xi_{pi}$ |
| `delta_sex -= w_a * sex_acc` | $-\sum_i w_\xi(B^\xi_{pi})^2$ |
| `coh += 0.5 * w_a * coh_acc` | $\tfrac12\sum_q w_\xi(B^\xi_{pq})^2$ |

The Rust test `sigma::tests::sigma_c_static_w_in_gap_equals_cohsex_pieces`
puts a constant $W$ through the frequency integral `sigma_c_at_z` at
$z = \epsilon_F$ and asserts `cohsex_pieces` to $10^{-9}$ Ha. The KS shift
has its own test, `crates/ferric-gw/tests/cohsex_ks_shift.rs`.

## The algorithm {#cohsex-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $B$, $\Sigma^x$ | section @sec:ri-factors-and-exchange | `build_full_b`, `sigma_x_diag` |
| 2 | $w_\xi$ from the static PDEP eigenvalues, $B^\xi$ | $N_{\rm pdep}N_{\rm aux}n_{\rm act}^2$ | `static_weights`, `project_b_into_pdep` |
| 3 | $\Delta\Sigma^{\rm SEX}_p$, $\Sigma^{\rm COH}_p$ for every state, parallel over $p$ | $N_{\rm pdep}n_{\rm act}^2$ per state | `cohsex_pieces` |
| 4 | $\epsilon^{\rm QP}_p$, KS shift if any | $O(n_{\rm act})$ | `run_cohsex` |

The two sums are independent per state: each $(\Delta\Sigma^{\rm SEX}_p, \Sigma^{\rm COH}_p)$
pair is written once, so the parallel loop is bit-identical to the serial one.

## Options and limits {#cohsex-options-limits}

The static $W$ of this section is also the kernel of the Bethe–Salpeter
equation in section @sec:bse-tda and of the polarizability in section @sec:rpax-static-polarizability.

`sc-COHSEX`, the self-consistent variant, is not implemented: `GwMethod::ScCohsex`
returns an error from both `run_gw` and the open-shell driver.

## Numerical anchors {#cohsex-numerical-anchors}

From `site/src/reference/validation.md` (the COHSEX row):

| Quantity | Systems | Error | Test |
|---|---|---|---|
| COHSEX@HF against numpy on PySCF density-fitted integrals | H2O, N2 / cc-pVDZ | $9.4\times10^{-10}$ Ha | `cohsex_hf_h2o_n2_vs_numpy` in `crates/ferric-gw/tests/validation_gw.rs` |
| COHSEX@PBE, same, with the static $\Sigma^x - v^{xc}$ | H2O, N2 / cc-pVDZ | $1.2\times10^{-9}$ Ha | `cohsex_pbe_h2o_n2_vs_numpy` in `crates/ferric-gw/tests/validation_gw.rs` |

## Sources {#cohsex-sources}

| Source | What it gives |
|---|---|
| `scripts/gw_self_energy_sympy.py`, claim 3 | the proof |
| Hedin, *Phys. Rev.* **139**, A796 (1965) | GW and COHSEX |
