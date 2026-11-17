# Global and range-separated hybrids; tuning $\omega$ {#hybrids}

<!-- status: ready; registry: ks-hybrid-rsh;
     theory: crates/ferric-dft/theory.md#"Exact exchange in hybrid and range-separated hybrid functionals" -->

Exact exchange is the Hartree–Fock exchange energy evaluated with the
Kohn–Sham orbitals. It is not a function of the density at one point: it
couples every pair of points through the Coulomb kernel. A hybrid functional
replaces a fraction of the semilocal exchange by exact exchange. A
range-separated hybrid (RSH) uses one fraction when two electrons are close
and another when they are far apart, with the switch set by a parameter
$\omega$.

The plan: in @sec:hybrids-exact-exchange, split the Coulomb operator by distance, map libxc's
parameters onto ferric's two fractions, and show that the exchange part of
the Fock matrix is the derivative of the exchange energy; in @sec:hybrids-tuning, choose
$\omega$ per molecule by the ionization-potential condition. libxc
supplies the semilocal part and the fractions; the SCF builds the exact
exchange itself.

| Symbol | Meaning | Code name |
|---|---|---|
| $a$ | global exact-exchange fraction | `b3lyp_mix` |
| $c_{\rm sr}, c_{\rm lr}, \omega$ | short- and long-range fractions, range separation (Bohr$^{-1}$) | `CamCoeffs { c_sr, c_lr, omega }`, `KMix` |
| $K[\mathrm{erfc}]$, $K[\mathrm{erf}]$ | exchange matrices of the two attenuated kernels | `k_sr`, `k_lr` |
| $J(\omega)$ | tuning objective | `OmegaEval::j` |

| Entry point | Location |
|---|---|
| Rust | `ferric-dft/src/libxc.rs::cam_coefficients`, `ferric-scf/src/fock_assembly.rs::{subtract_rsh_exchange,build_rsh_dfk_pair}`, `ferric-scf/src/rsh_response.rs::exchange_response`, `ferric-scf/src/omega_tuning.rs::tune_omega` |
| CLI | `[dft] functional`, `xc_omega` |
| Python | `run_dft(xc=..., xc_omega=...)`, `tune_omega` |

## Exact exchange {#hybrids-exact-exchange}

**Splitting the operator.** Write the Coulomb operator as a Gaussian
integral,

$$ \frac1r = \frac{2}{\sqrt\pi}\int_0^\infty e^{-t^2r^2}\,dt , $$

and cut the integral at $t = \omega$. Substitute $u = tr$ in the lower part:
$\tfrac{2}{\sqrt\pi}\int_0^\omega e^{-t^2r^2}\,dt = \tfrac1r\,\tfrac{2}{\sqrt\pi}\int_0^{\omega r}e^{-u^2}\,du = \mathrm{erf}(\omega r)/r$.
The upper part is the rest, $\mathrm{erfc}(\omega r)/r$, since
$\mathrm{erf} + \mathrm{erfc} = 1$:

$$ \frac1r = \frac{\mathrm{erfc}(\omega r)}{r} + \frac{\mathrm{erf}(\omega r)}{r} . $$

The two pieces have different ranges. $\mathrm{erf}(\omega r)/r$ is finite at
$r = 0$ (it tends to $2\omega/\sqrt\pi$) and becomes $1/r$ at large $r$: it is
the long-range part. $\mathrm{erfc}(\omega r)/r$ is $1/r$ at short range and
decays like a Gaussian beyond $r \approx 1/\omega$: it is the short-range
part. $\omega$ is in Bohr$^{-1}$.

**ferric's form.** The CAM form of Yanai, Tew and Handy writes the
exact-exchange kernel as $[a + b\,\mathrm{erf}(\omega r)]/r$ and gives the
rest, $[1 - a - b\,\mathrm{erf}(\omega r)]/r$, to the semilocal exchange; the
two sum to $1/r$. ferric writes every exact-exchange kernel as

$$ v_x^{\rm HF}(r) = c_{\rm sr}\,\frac{\mathrm{erfc}(\omega r)}{r} + c_{\rm lr}\,\frac{\mathrm{erf}(\omega r)}{r} . $$

As $r \to 0$ the kernel is $c_{\rm sr}/r$, so $c_{\rm sr}$ is the
exact-exchange fraction at short range; as $r \to \infty$ it is
$c_{\rm lr}/r$. Matching the Yanai form at the two ends gives
$c_{\rm sr} = a$ and $c_{\rm lr} = a + b$.

Check the limits in $\omega$. As $\omega \to 0$, $\mathrm{erf}(\omega r) \to 0$
and the kernel is $c_{\rm sr}/r$ at every distance: a global hybrid with
fraction $c_{\rm sr}$. As $\omega \to \infty$ the kernel is $c_{\rm lr}/r$. A
global hybrid is $c_{\rm sr} = c_{\rm lr} = a$, for which the kernel is
$a/r$ whatever $\omega$ is.

**libxc's convention.** libxc's `xc_hyb_cam_coef` returns $(\omega, \alpha, \beta)$
for the kernel $\alpha/r + \beta\,\mathrm{erfc}(\omega r)/r$: `cam_alpha`
multiplies the full-range operator and `cam_beta` the short-range one.
Substitute $1/r = \mathrm{erfc}/r + \mathrm{erf}/r$ into the $\alpha$ term
and collect:

$$ \frac{\alpha}{r} + \beta\,\frac{\mathrm{erfc}(\omega r)}{r} = (\alpha + \beta)\,\frac{\mathrm{erfc}(\omega r)}{r} + \alpha\,\frac{\mathrm{erf}(\omega r)}{r}, \qquad c_{\rm sr} = \alpha + \beta,\qquad c_{\rm lr} = \alpha . $$

That is the whole of `ferric-dft/src/libxc.rs::cam_coefficients` after the
FFI call:

<!-- excerpt: ferric-dft/src/libxc.rs::cam_coefficients -->
```rust
Some(CamCoeffs {
    omega,
    c_sr: alpha + beta,
    c_lr: alpha,
})
```

A functional with $\omega = 0$ has no `CamCoeffs`. Its global fraction $a$
is `xc_hyb_exx_coef` (`XcFunctional::exact_exchange_mix`, stored in
`XcDef::b3lyp_mix`). PySCF's `dft.libxc.rsh_coeff` returns
$(\omega, c_{\rm lr}, c_{\rm sr} - c_{\rm lr})$, the same split. The sign of
`cam_beta` tells the families apart: $\beta < 0$ means more exact exchange at
long range, $\beta > 0$ more at short range.

| Functional | libxc $(\alpha, \beta, \omega)$ | $c_{\rm sr}$ | $c_{\rm lr}$ | Kind |
|---|---|---|---|---|
| ωB97X | $(1, -0.842294, 0.3)$ | 0.157706 | 1 | long-range corrected |
| ωB97X-V | $(1, -0.833, 0.3)$ | 0.167 | 1 | long-range corrected |
| CAM-B3LYP | $(0.65, -0.46, 0.33)$ | 0.19 | 0.65 | long-range corrected |
| LC-ωPBE | $(1, -1, 0.4)$ | 0 | 1 | long-range only |
| HSE06 | $(0, 0.25, 0.11)$ | 0.25 | 0 | screened (short-range only) |
| B3LYP, PBE0 | $\omega = 0$, $a = 0.2$, $0.25$ | $a$ | $a$ | global hybrid |

libxc's semilocal energy and potential for these functionals already carry
the complementary semilocal exchange, scaled, and for an RSH attenuated by
the matching short- or long-range factor. So ferric adds only the exact
exchange.

> **Verified equation {#eq:ks-rsh-libxc-map}** · the libxc map and the operator split
>
> The CAM split sums to $1/r$; libxc's $(\alpha, \beta)$ give
> $c_{\rm sr} = \alpha + \beta$, $c_{\rm lr} = \alpha$, and Yanai's $(a, b)$
> give $c_{\rm sr} = a$, $c_{\rm lr} = a + b$; numerically, on four s-type
> Gaussians the closed-form erf integrals equal quadrature of
> $1/r = (2/\sqrt\pi)\int_0^\infty e^{-t^2r^2}dt$ on $[0, \omega]$, and
> $K_{\rm tot}$ equals a K built on the CAM kernel directly.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_hybrid_rsh_sympy.py`, claims A1, A2, B1, B3 |
> | Negative control | the runner swaps the libxc map to $(\alpha, \alpha + \beta)$ (C1, the published-fraction table, fails) and swaps the erf and erfc exchange matrices (B3 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-dft/src/libxc.rs::cam_coefficients`, `ferric-scf/src/fock_assembly.rs::subtract_rsh_exchange` |
> | Code test | `crates/ferric-scf/tests/validation_ks_energies.rs` (HSE06 and ωB97X-V against PySCF) |
>
> <!-- verified: id=ks.rsh.libxc-map; script=scripts/dft_hybrid_rsh_sympy.py; claim=A2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/libxc.rs::cam_coefficients; registry=ks-hybrid-rsh -->

The script also checks the limits $\omega \to \infty$ (the erf integrals
become Coulomb integrals) and $\omega \to 0$ (they vanish), and that the
libxc values in the table map to the published short- and long-range
fractions.

**Mixing coefficients.** The SCF needs three numbers: the two fractions and
$\omega$. `ferric-dft/src/libxc.rs::k_mix_from_xc_def` (and `KsXc::k_mix`,
`KsXcUks::k_mix` in `ferric-dft/src/ks.rs`) turns a definition into
`ferric-dft/src/xc_trait.rs::KMix`:

<!-- excerpt: ferric-dft/src/libxc.rs::k_mix_from_xc_def -->
```rust
if let Some(cam) = xc.cam {
    KMix {
        sr: cam.c_sr,
        lr: cam.c_lr,
        omega: cam.omega,
    }
} else if let Some(mix) = xc.b3lyp_mix {
    KMix {
        sr: mix,
        lr: mix,
        omega: 0.0,
    }
} else {
    KMix {
        sr: 0.0,
        lr: 0.0,
        omega: 0.0,
    }
}
```

That is $(c_{\rm sr}, c_{\rm lr}, \omega)$ for an RSH, $(a, a, 0)$ for a
global hybrid and $(0, 0, 0)$ for a pure functional. Hartree–Fock, with no
functional, is $(1, 1, 0)$. A meta-GGA built from several libxc components
(`metagga_def_from_handles`) sums the components' `xc_hyb_exx_coef` into one
global $a$ (M06-2X: 0.54 from the exchange component, 0 from correlation),
takes `CamCoeffs` from at most one component, and refuses a definition that
has both. `xc_def_from_name_nspin_omega` overrides $\omega$ in every
component's `_omega` parameter and in `CamCoeffs.omega` together, and leaves
$\alpha$ and $\beta$ unchanged.

**Exchange matrices.** For an AO density $D$ and a kernel $v$,

$$ K[v](D)_{\mu\nu} = \sum_{\lambda\sigma}(\mu\lambda|v|\nu\sigma)D_{\lambda\sigma}, \qquad J(D)_{\mu\nu} = \sum_{\lambda\sigma}(\mu\nu|\lambda\sigma)D_{\lambda\sigma} . $$

The exchange matrix the SCF subtracts is

$$ K_{\rm tot}(D) = c_{\rm sr}\,K[\mathrm{erfc}](D) + c_{\rm lr}\,K[\mathrm{erf}](D) . $$

For $\omega > 0$ the two matrices come from two density-fitting builders over
the same auxiliary basis (`config.df_k_aux`, default `def2-universal-jkfit`),
one per operator: `ferric-scf/src/fock_assembly.rs::build_rsh_dfk_pair`
builds them with `Operator::erfc(ω)` and `Operator::erf(ω)`. For
$\omega = 0$, $K_{\rm tot} = a\,K[1/r]$ from whichever K builder the run
selected.

**The closed-shell energy and its derivative.** With $D$ the total density,
$D = 2C_{\rm occ}C_{\rm occ}^{\mathsf T}$,

$$ E_x^{\rm HF} = -\tfrac14\,\mathrm{Tr}[D\,K_{\rm tot}(D)] = -\tfrac14\sum_{\mu\nu\lambda\sigma}D_{\mu\nu}(\mu\lambda|v|\nu\sigma)D_{\lambda\sigma} . $$

Differentiate with respect to $D_{\mu\nu}$. Each $D$ gives one term. For real
orbitals $(\mu\lambda|v|\nu\sigma) = (\lambda\mu|v|\sigma\nu)$, which is the
swap $(\mu\nu)\leftrightarrow(\lambda\sigma)$, so the two terms are equal:

$$ \frac{\mathrm dE_x^{\rm HF}}{\mathrm dD} = -\tfrac12K_{\rm tot}(D), \qquad F = h + J(D) - \tfrac12 K_{\rm tot}(D) + V_{xc} . $$

The exchange part of $F$ is the derivative of the exchange energy, for both
kernels at once, because $K_{\rm tot}$ is linear in $D$ and each kernel has
the symmetry.

**Open shell.** With per-spin densities
$D^\sigma = C^\sigma_{\rm occ}C^{\sigma\,\mathsf T}_{\rm occ}$ (`uhf.rs`
`solve_uhf`, `rohf.rs` `solve_rohf`),

$$ E_x^{\rm HF} = -\tfrac12\sum_\sigma \mathrm{Tr}[D^\sigma K_{\rm tot}(D^\sigma)], \qquad F_\sigma = h + J(D^\alpha + D^\beta) - K_{\rm tot}(D^\sigma) + V_{xc}^\sigma . $$

The same argument gives $-K_{\rm tot}(D^\sigma)$: the $\tfrac12$ doubles to 1.
ROHF assembles the same $F_\alpha$ and $F_\beta$ and then couples them into its
Roothaan operator. With $D^\alpha = D^\beta = D/2$, both the energy and
$F_\sigma$ equal the closed-shell ones.

> **Verified equation {#eq:ks-rsh-exchange-derivative}** · the exchange part of $F$ is the derivative of $E_x$
>
> $\mathrm dE_x/\mathrm dD = -\tfrac12K_{\rm tot}(D)$ (closed shell) and
> $-K_{\rm tot}(D^\sigma)$ per spin (open shell), symbolically with generic
> 8-fold-symmetric integrals for both kernels; numerically by central
> differences of the total electronic energy for RHF with an RSH, a global
> hybrid and HF, and for UHF with an RSH; the open-shell form reduces to the
> closed-shell one at $D^\alpha = D^\beta = D/2$.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_hybrid_rsh_sympy.py`, claims A3, B4, B5 |
> | Negative control | the runner doubles the energy trace (B4 fails), and changes the exchange factor of $F$ to $\tfrac14$ (closed shell) or $\tfrac12$ (per spin), each failing B5 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/fock_assembly.rs::subtract_rsh_exchange` |
>
> <!-- verified: id=ks.rsh.exchange-derivative; script=scripts/dft_hybrid_rsh_sympy.py; claim=A3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/fock_assembly.rs::subtract_rsh_exchange; registry=ks-hybrid-rsh -->

The total energy follows section @sec:ks-equations-scf-iteration: one- and two-electron energy from
the Fock matrix before $V_{xc}$ is added, then $E_{xc}$ from the grid,

$$ E = \tfrac12\mathrm{Tr}[D(h + F_{\rm no\,xc})] + E_{xc} + E_{nn} = \mathrm{Tr}[Dh] + \tfrac12\mathrm{Tr}[DJ] - \tfrac14\mathrm{Tr}[D K_{\rm tot}] + E_{xc} + E_{nn}, $$

with $F_{\rm no\,xc} = h + J - \tfrac12 K_{\rm tot}$; open shell,
$\tfrac12\sum_\sigma\mathrm{Tr}[D^\sigma(h + F_{\sigma,\rm no\,xc})]$.

In the code, the last three lines of
`ferric-scf/src/fock_assembly.rs::subtract_rsh_exchange` form $K_{\rm tot}$
in place and subtract it:

<!-- excerpt: ferric-scf/src/fock_assembly.rs::subtract_rsh_exchange -->
```rust
k_sr *= c_sr;
k_sr.scaled_add(c_lr, &k_lr);
f.scaled_add(-eff_scale, &k_sr);
```

| Symbol | Code |
|---|---|
| $c_{\rm sr}K[\mathrm{erfc}] + c_{\rm lr}K[\mathrm{erf}]$ | `k_sr` after the first two lines |
| $\tfrac12$ (closed shell) or 1 (per spin) | `eff_scale` |

Two points about the function:

- **The occupation factor goes on the scale.** When the occupied MO
  coefficients are at hand, both K matrices are built from $C_{\rm occ}$ by
  the cheaper DF-K half-transform, which returns K for
  $C_{\rm occ}C_{\rm occ}^{\mathsf T}$. K is linear in $D$, so the factor 2 of
  the closed-shell density goes into `eff_scale` instead of into $C$. Scaling
  by an exact power of two avoids a $\sqrt2$ rounding on the coefficients.
- **No temporary for $K_{\rm tot}$.** The three in-place lines have the same
  element-wise association as forming $K_{\rm tot}$ and then subtracting it.

The response methods (stability, Newton, Z-vector, TDDFT) build the same
combination from the same pair of fitters for a density perturbation
(`exchange_response`), with the Fock's own sign and factor applied by the
caller.

With `DERIV_CROSSCHECK=1` and PySCF and ferric importable, the proof script
also checks PySCF's `rsh_coeff`, ferric's `XcFunctional.exchange_mix`, and
ferric's `AoIntegrals.veff` energy against its potential by central
differences (ωB97X-V, CAM-B3LYP, HSE06 on H$_2$/STO-3G).

## Tuning $\omega$ {#hybrids-tuning}

A range-separated functional ships with one published $\omega$. Tuning
picks $\omega$ per molecule instead, by asking that Koopmans' theorem hold
for the HOMO: the HOMO energy should equal minus the ionization potential
computed from two total energies. For a closed-shell neutral molecule and
its doublet cation at the same geometry, basis and grid,

$$ J(\omega) = \varepsilon_{\rm HOMO}(N;\omega) + E(N-1;\omega) - E(N;\omega), \label{eq:hybrids-1} $$

and `ferric-scf/src/omega_tuning.rs::tune_omega` minimizes $\lvert J\rvert$
by golden-section search. $E(N-1) - E(N)$ is the ΔSCF ionization potential,
so $J = 0$ means $\varepsilon_{\rm HOMO} = -\mathrm{IP}$.

Equation (@eq:hybrids-1) is the ionization-potential condition of Stein, Kronik and
Baer. It is a definition, not an identity, so it has no proof script; the
evidence is the comparison of $J$, its parts and $\omega^*$ with PySCF in
the anchors below.

| Step | What | Function |
|---|---|---|
| 1 | bracket $[\omega_{\rm lo}, \omega_{\rm hi}] = [0.1, 1.0]$ Bohr$^{-1}$ | `OmegaTuneConfig::default` |
| 2 | at each $\omega$: RKS of the neutral, UKS of the doublet cation, both with `xc_omega` set (a functional without an $\omega$ parameter is an error) | `tune_omega` |
| 3 | exact $J$ in both states unless an RI-J auxiliary set is given | |
| 4 | cation internal stability (on by default): a saddle at the returned $\omega^*$ is an error naming $\omega$ and $\lambda_{\min}$ | `check_cation_stability` |
| 5 | branch check: the cation's spin-population asymmetry against the nearest evaluated $\omega$ (tolerance $10^{-7}$) | `branch_tol` |
| 6 | golden section to $\omega$ tolerance $5\times10^{-3}$, at most 24 evaluations | |

Golden section keeps a bracket and two interior points at the fractions
$1 - \varphi^{-1}$ and $\varphi^{-1}$ of it, $\varphi^{-1} = 0.618\ldots$
(`INVPHI`). Each step discards the end next to the worse point and reuses
the other interior point, so one new evaluation shrinks the bracket by
$\varphi^{-1}$. From the default bracket of width 0.9 to the tolerance
$5\times10^{-3}$ takes 11 steps after the first two evaluations, 13
evaluations in all, well inside the cap of 24. The function returns the
evaluated $\omega$ with the smallest $\lvert J\rvert$, not the midpoint of the
final bracket. This is one step of the loop in `tune_omega`, unchanged:

<!-- excerpt: ferric-scf/src/omega_tuning.rs::tune_omega -->
```rust
if fc < fd {
    b = d;
    d = c;
    fd = fc;
    c = b - (b - a) * INVPHI;
    fc = f(c, &mut evals, &mut seeds, &mut warnings)?;
} else {
    a = c;
    c = d;
    fc = fd;
    d = a + (b - a) * INVPHI;
    fd = f(d, &mut evals, &mut seeds, &mut warnings)?;
}
```

Here `f` runs the two SCFs at one $\omega$ and returns $\lvert J\rvert$; `c`
and `d` are the interior points, and `fc`, `fd` their values. When
$\lvert J(c)\rvert < \lvert J(d)\rvert$ the minimum lies in $[a, d]$, so $b$
moves to $d$, the old $c$ becomes the new $d$, and only the new $c$ costs an
evaluation.

**Why the stability and branch checks.** $J(\omega)$ is a curve only if
every point is the same electronic state. The doublet cation of a symmetric
molecule can have two: the hole shared over symmetry-equivalent centres, or
localized on one of them. Which one an independent SCF reaches depends on
its guess, and golden section visits $\omega$ out of order, so independent
solves can assemble $J$ from two branches. Step 5 flags a jump in the
spin-population asymmetry. Step 4 runs the UKS orbital Hessian on every
cation (section @sec:stability), because a cation that is an internal saddle can have a
smooth $J$ and $\langle S^2\rangle$; only the Hessian's lowest eigenvalue
changes sign.

Continuation (seeding each solve from the nearest evaluated $\omega$) is
off by default: it leaves $\omega^*$ bit-identical on H2O and NH3, and moves
$J$ only by SCF stopping noise. For a VV10 functional the cation stability
cannot be analysed (no VV10 response kernel), and the result says so.

## Options {#hybrids-options}

| Option | Effect |
|---|---|
| `xc_omega` | overrides $\omega$ in every component's `_omega` parameter and in `CamCoeffs` together; $\alpha$, $\beta$ unchanged |
| `[scf] k_builder` | for $\omega > 0$ the two K matrices come from DF-K over `df_k_aux` (default `def2-universal-jkfit`) |
| `tune_omega(continuation=, branch_tol=, check_cation_stability=)` | as above |

## Numerical anchors {#hybrids-numerical-anchors}

| Quantity | Value | Source |
|---|---|---|
| HSE06 RKS energy vs PySCF | $4.2\times10^{-12}$ Ha | `crates/ferric-scf/tests/validation_ks_energies.rs` |
| ωB97X-V UKS energy vs PySCF | $6\times10^{-6}$ Ha (DF-K metrics differ) | the same |
| tuning, H2O and NH3 / def2-SVP, ωB97X-V: $\varepsilon_{\rm HOMO}$, IP, $J$, $\omega^*$ vs PySCF | $\le 6.7\times10^{-6}$ Ha, $\le 8.8\times10^{-7}$ Ha, $\le 6.3\times10^{-6}$ Ha, $\le 2.8\times10^{-5}$ Bohr$^{-1}$ | `crates/ferric-scf/tests/validation_rsh_omega.rs` |
| cation $\lambda_{\min}$ vs PySCF | $\le 2.3\times10^{-8}$ Ha/rad², sign matches at all 13 points | `crates/ferric-scf/tests/validation_rsh_cation_stability.rs` |

## Sources {#hybrids-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-dft/theory.md`, "Exact exchange in hybrid and range-separated hybrid functionals" | the reference statement of the equations and the libxc table |
| `crates/ferric-scf/src/omega_tuning.rs` module documentation | the cation-branch problem and the measurements behind the defaults |
| Yanai, Tew and Handy, Chem. Phys. Lett. 393, 51 (2004) | the CAM form |
| Stein, Kronik and Baer, J. Am. Chem. Soc. 131, 2818 (2009) | the tuning condition |
