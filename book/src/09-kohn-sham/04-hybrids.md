# 9.4 Global and range-separated hybrids; tuning $\omega$

<!-- section: 9.4; status: ready; registry: ks-hybrid-rsh;
     theory: crates/ferric-dft/theory.md#"Exact exchange in hybrid and range-separated hybrid functionals" -->

A hybrid adds a fraction of exact exchange to a semilocal functional; a
range-separated hybrid (RSH) uses one fraction at short and another at long
electron–electron distance, split by the error function at $\omega$. libxc
supplies the semilocal part and the fractions. The SCF builds the exact
exchange. Tuning picks $\omega$ per molecule so that Koopmans' theorem holds
for the HOMO.

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

## 9.4.1 Exact exchange

<!-- include: crates/ferric-dft/theory.md#"Exact exchange in hybrid and range-separated hybrid functionals" -->

> **Verified equation 9.5** · the libxc map and the operator split
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

> **Verified equation 9.6** · the exchange part of $F$ is the derivative of $E_x$
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

The response methods (stability, Newton, Z-vector, TDDFT) build the same
combination from the same pair of fitters for a density perturbation
(`exchange_response`), with the Fock's own sign and factor applied by the
caller.

## 9.4.2 Tuning $\omega$

For a closed-shell neutral molecule and its doublet cation at the same
geometry, basis and grid,

$$ J(\omega) = \varepsilon_{\rm HOMO}(N;\omega) + E(N-1;\omega) - E(N;\omega), \tag{9.7} $$

and `tune_omega` minimizes $\lvert J\rvert$ by golden-section search.

Equation (9.7) is the ionization-potential condition of Stein, Kronik and
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

Continuation (seeding each solve from the nearest evaluated $\omega$) is
off by default: it leaves $\omega^*$ bit-identical on H2O and NH3, and moves
$J$ only by SCF stopping noise. For a VV10 functional the cation stability
cannot be analysed (no VV10 response kernel), and the result says so.

## 9.4.3 Options

| Option | Effect |
|---|---|
| `xc_omega` | overrides $\omega$ in every component's `_omega` parameter and in `CamCoeffs` together; $\alpha$, $\beta$ unchanged |
| `[scf] k_builder` | for $\omega > 0$ the two K matrices come from DF-K over `df_k_aux` (default `def2-universal-jkfit`) |
| `tune_omega(continuation=, branch_tol=, check_cation_stability=)` | as above |

## 9.4.4 Numerical anchors

| Quantity | Value | Source |
|---|---|---|
| HSE06 RKS energy vs PySCF | $4.2\times10^{-12}$ Ha | `crates/ferric-scf/tests/validation_ks_energies.rs` |
| ωB97X-V UKS energy vs PySCF | $6\times10^{-6}$ Ha (DF-K metrics differ) | the same |
| tuning, H2O and NH3 / def2-SVP, ωB97X-V: $\varepsilon_{\rm HOMO}$, IP, $J$, $\omega^*$ vs PySCF | $\le 6.7\times10^{-6}$ Ha, $\le 8.8\times10^{-7}$ Ha, $\le 6.3\times10^{-6}$ Ha, $\le 2.8\times10^{-5}$ Bohr$^{-1}$ | `crates/ferric-scf/tests/validation_rsh_omega.rs` |
| cation $\lambda_{\min}$ vs PySCF | $\le 2.3\times10^{-8}$ Ha/rad², sign matches at all 13 points | `crates/ferric-scf/tests/validation_rsh_cation_stability.rs` |

## 9.4.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-dft/theory.md`, "Exact exchange in hybrid and range-separated hybrid functionals" | the equations and the libxc table |
| `crates/ferric-scf/src/omega_tuning.rs` module documentation | the cation-branch problem and the measurements behind the defaults |
| Yanai, Tew and Handy, Chem. Phys. Lett. 393, 51 (2004) | the CAM form |
| Stein, Kronik and Baer, J. Am. Chem. Soc. 131, 2818 (2009) | the tuning condition |
