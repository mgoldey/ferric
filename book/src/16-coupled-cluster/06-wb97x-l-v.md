# The ωB97X-L-V double hybrid {#wb97x-l-v}

<!-- status: ready; registry: double-hybrid;
     theory: crates/ferric-cc/theory.md#"ωB97X-L-V correlation" -->

A double hybrid adds a fraction of wavefunction correlation to a hybrid
density functional, usually MP2, and inherits MP2's divergence as the
HOMO–LUMO gap closes. ωB97X-L-V uses the short-range LinLCCD(hh)
correlation of section @sec:linlccd instead, whose hole–hole ladder keeps the
effective gap open. Long-range correlation is left to the VV10 term of the
density functional, so the wavefunction part sees only
$\mathrm{erfc}(\omega r)/r$ (`DoubleHybridConfig::omega`). The functional is
evaluated post-SCF: the ωB97X-L Kohn–Sham density with VV10 is converged
first, and the correlation is computed once on its orbitals.

$$E = E_{\rm KS}[\omega{\rm B97X\text{-}L}] + E_{c,{\rm VV10}}
+ \lambda\,E_{c,{\rm LinLCCD(hh)}}^{sr,\omega,\lambda} ,\qquad
\lambda = 0.6,\ \omega = 0.1\ {\rm Bohr}^{-1}. \label{eq:wb97xlv-lambda}$$

This is eqn 27 of Ransford and Carter-Fenk, Phys. Chem. Chem. Phys. 28,
14428 (2026).

| Symbol | Meaning | Code name |
|---|---|---|
| $\lambda$ | adiabatic-connection parameter | `DoubleHybridConfig::lambda`, `WB97X_L_V_LAMBDA` |
| $\omega$ | range separation (Bohr⁻¹) | `DoubleHybridConfig::omega`, `WB97X_L_V_OMEGA` |
| $E_c^{sr,\omega,\lambda}$ | LinLCCD(hh) energy with amplitudes under $\lambda\,{\rm erfc}(\omega r)/r$ | `DoubleHybridResult::e_c_wft` |
| $\lambda E_c^{sr,\omega,\lambda}$ | the term added | `DoubleHybridResult::e_c_scaled` |

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/double_hybrid.rs::{run_wb97x_l_v,solve_wb97x_l_v,u_solve_wb97x_l_v}` |
| CLI | `method.kind = "wb97x-l-v"` (`examples/water-wb97xlv.toml`) |
| Python | none |

## Where λ goes {#wb97x-l-v-goes}

**The amplitude equations.** Eqn 22 of the paper puts $\lambda$ on both the
driver and the hole–hole ladder of the LinLCCD(hh) equations, and leaves the
Fock terms unscaled:

$$ \Delta_{ij}^{ab}\,t_{ij}^{ab} = -\lambda\Big(\langle ij\|ab\rangle^{sr} + \tfrac12\sum_{kl}\langle kl\|ij\rangle^{sr}\,t_{kl}^{ab}\Big) . $$

These are the LinLCCD(hh) equations of section @sec:linlccd for the operator
$\lambda\,\mathrm{erfc}(\omega r)/r$. The added energy is

$$ \lambda E_c^{sr,\omega,\lambda} = \tfrac14\sum_{ijab}\big(\lambda\langle ij\|ab\rangle^{sr}\big)\,t_{ij}^{ab} . $$

**The MP2 limit.** Drop the ladder. Then
$t = \lambda\langle ij\|ab\rangle^{sr}/(-\Delta)$, and substitution gives
$\tfrac14\sum\lambda^2|\langle ij\|ab\rangle^{sr}|^2/(-\Delta) = \lambda^2E^{sr}_{c,\mathrm{MP2}}$.
The term is quadratic in $\lambda$ at leading order ("actually quadratic in
λ", the paper below eqn 23). Solving at $\lambda = 1$ and multiplying by
$\lambda$ afterwards would give $\lambda E^{sr}_{c,\mathrm{MP2}}$ in this
limit, too large by $1/\lambda$ (1.67 at $\lambda = 0.6$).

**One solve under a scaled operator.** The code runs LinLCCD(hh) once under
the composite operator $\lambda\,\mathrm{erfc}(\omega r)/r$:

<!-- excerpt: ferric-cc/src/double_hybrid.rs::lambda_scaled_sr_operator -->
```rust
(cfg.lambda > 0.0)
    .then(|| Operator::composite(&[(cfg.lambda, OperatorKind::ErfcCoulomb, cfg.omega)]))
```

Follow the scaling through the RI construction of section @sec:factorization-and-metric. The
three-centre integrals $(P|pq)$ and the metric $V_{PQ}$ are both integrals
of the operator, so both scale by $\lambda$. The factor
$B = V^{-1/2}(P|pq)$ then scales by $\lambda^{-1/2}\cdot\lambda = \sqrt\lambda$,
and every four-index integral $\sum_P B^P_{pq}B^P_{rs}$ by $\lambda$. The
LinLCCD(hh) solver of section @sec:linlccd, unchanged, therefore solves the eqn 22
amplitudes, and the energy it returns,
$\tfrac14\sum(\lambda\langle ij\Vert ab\rangle^{sr})\,t$, is already
$\lambda E_c^{sr,\omega,\lambda}$. `assemble` recovers $E_c^{sr,\omega,\lambda}$
(`e_c_wft`) by dividing by $\lambda$.

At $\lambda = 0$ the operator vanishes and its RI metric is singular, so no
correlation is computed (`None` above).

> **Verified equation @eq:wb97xlv-lambda** · λ placement and parameters
>
> | | |
> |---|---|
> | Claim | LinLCCD(hh) on λ-scaled three-centre integrals and metric equals $\lambda E_c$ of eqn 22 (λ on driver and hh ladder); with the ladder off it is $\lambda^2 E_{\mathrm{MP2}}$; the linear form $\lambda E_c(\lambda = 1)$ misses it by more than 10 %; at λ = 1 with the ladder off it is the brute-force MP2 doubles sum. The Rust ωB97X-L-V tables equal the paper's Table 2, with $c_{\mathrm{sr}} = \lambda$, $c_{x,0} = 1-\lambda$, $c_{ss,0} = c_{os,0} = 1-\lambda^2$ |
> | Proof | claim L (and claim P) of `scripts/double_hybrid_sympy.py` |
> | Code | `ferric-cc/src/double_hybrid.rs::{lambda_scaled_sr_operator,assemble}`, `ferric-dft/src/libxc.rs::{wb97x_l_v_def,WB97X_L_V_EXT_PARAMS}` |
> | Code test | `wb97xlv_h2o_def2svp_components_vs_pyscf` in `crates/ferric-cc/tests/validation_wb97xlv.rs` |
>
> <!-- verified: id=wb97xlv.lambda; script=scripts/double_hybrid_sympy.py; claim=L; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/double_hybrid.rs::{lambda_scaled_sr_operator,assemble}; rust_test=ferric-cc/tests/validation_wb97xlv.rs::wb97xlv_h2o_def2svp_components_vs_pyscf; registry=double-hybrid -->

**The density-functional half.** The KS half (`ferric-dft/src/libxc.rs::wb97x_l_v_def`) is libxc's
`HYB_GGA_XC_WB97X_V` form with the paper's Table 2 coefficients set through
the external-parameter interface (`WB97X_L_V_EXT_PARAMS`, checked by name
against libxc's order), and its own VV10 ($b = 10.0$, $C = 0.01$,
`WB97X_L_V_VV10`).

Exact exchange is full at long range and $\lambda$-scaled at short range. In
libxc's CAM convention $\alpha = 1$ and $\beta = -0.4$, so
$c_{\mathrm{lr}} = \alpha = 1$ and $c_{\mathrm{sr}} = \alpha + \beta = 0.6 = \lambda$,
with $\omega = 0.1$. The complementary semilocal exchange carries
$c_{x,0} = 1 - \lambda = 0.4$, and the same- and opposite-spin correlation
$c_{ss,0} = c_{os,0} = 1 - \lambda^2 = 0.64$; these are the
uniform-electron-gas constraints. The other twelve B97 coefficients are
fitted:

| | $c_{\cdot,0}$ | $c_{\cdot,1}$ | $c_{\cdot,2}$ | $c_{\cdot,3}$ | $c_{\cdot,4}$ |
|---|---|---|---|---|---|
| exchange $c_x$ | 0.4 | 0.154 | 3.884 | −11.300 | 8.425 |
| same-spin $c_{ss}$ | 0.64 | −1.417 | 4.716 | −2.956 | −0.861 |
| opposite-spin $c_{os}$ | 0.64 | −1.194 | −1.348 | 11.869 | −9.571 |

Claim P of `scripts/double_hybrid_sympy.py` proves that the Rust tables
equal Table 2 and the constraints above, and that `wb97x_l_v_def` takes only
`nspin`.

## The algorithm {#wb97x-l-v-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | ωB97X-L KS SCF with VV10 (libxc ωB97X-V form, the paper's parameters), through the KS convergence ladder; an error if every rung fails | `run_wb97x_l_v` |
| 2 | operator $\lambda\,{\rm erfc}(\omega r)/r$; $B^P_{pq}$ on the KS orbitals | `lambda_scaled_sr_operator` |
| 3 | LinLCCD(hh) (section @sec:linlccd) under that operator | `solve_wb97x_l_v` (`u_solve_wb97x_l_v` for UKS / semicanonical ROKS) |
| 4 | $E = E_{\rm KS} + \lambda E_c$, components kept separate | `assemble` |

The SCF converges the KS half alone, with RI-J and RI-K on
def2-universal-jkfit unless the input names another fit. The LinLCCD(hh)
term is evaluated once on its orbitals and orbital energies, with the RI
auxiliary basis of `[mp2] auxbasis`. The components stay separate in
`DoubleHybridResult` so that a bad SCF and a bad amplitude solve can be told
apart.

**Overrides.** The CLI kind converges its own ωB97X-L-V reference before the
shared SCF path. `[dft] lambda` and `[dft] omega` set
`DoubleHybridConfig::lambda` and `omega`, which enter only the correlation
term; an omitted key is the published value. `wb97x_l_v_def` takes no λ or
ω, so the KS half keeps the published $c_{\rm sr} = 0.6$, $\omega = 0.1$ and
the constrained $c_{x,0} = 0.4$, $c_{ss,0} = c_{os,0} = 0.64$ whatever the
override. An override therefore changes the correlation term on the
published KS reference; it does not change the functional as a whole. λ
must lie in $[0, 1]$ and ω must be positive.

## Numerical anchors {#wb97x-l-v-numerical-anchors}

| Quantity | Systems | Error | Test |
|---|---|---|---|
| $E_{\rm KS}$ vs PySCF with the paper's parameters | H2O (RKS, exact J and RI-J), OH (ROKS → semicanonical → unrestricted) / def2-SVP | $\le 1.05\times10^{-9}$ Ha | `crates/ferric-cc/tests/validation_wb97xlv.rs` |
| $\lambda E_c$ on the same orbitals vs numpy LinLCCD(hh) on PySCF DF $\lambda\,$erfc integrals | same | $7.2\times10^{-14}$ Ha | `crates/ferric-cc/tests/validation_wb97xlv.rs` |
| total on ferric's orbitals | same | $1.1\times10^{-7}$ Ha | `crates/ferric-cc/tests/validation_wb97xlv.rs` |
| Be₂ bond energy vs the paper's Table 4 (2.3 kcal/mol) | Be₂ / def2-QZVPPD | 2.299 kcal/mol | `crates/ferric-cc/tests/wb97x_l_v_be2_bde.rs` |

## Sources {#wb97x-l-v-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "ωB97X-L-V correlation" | the λ placement, the KS half, the override semantics |
| `scripts/double_hybrid_sympy.py` | claims L and P |
| Ransford and Carter-Fenk, PCCP 28, 14428 (2026), eqns 22 and 27, Tables 2 and 4 | the functional, parameters and the Be₂ reference |
| `ferric-cc/src/double_hybrid.rs`, module documentation | the post-SCF argument and the λ subtlety |

**Known limits (2026-11-15).** The CLI refuses an open-shell molecule; the
open-shell route is library-only. No Python binding. No gradient.
