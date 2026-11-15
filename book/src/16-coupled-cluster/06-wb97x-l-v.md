# 16.6 The ωB97X-L-V double hybrid

<!-- section: 16.6; status: gap; registry: double-hybrid;
     theory: crates/ferric-cc/theory.md#"ωB97X-L-V correlation" -->

ωB97X-L-V replaces the MP2 correlation of a double hybrid with the
short-range LinLCCD(hh) correlation of section 16.5. MP2 diverges as the
HOMO–LUMO gap closes; the hole–hole ladder does not, and that is the reason
for the choice. The functional is evaluated post-SCF: the ωB97X-L Kohn–Sham
density with VV10 is converged first, and the correlation is computed on
its orbitals.

$$E = E_{\rm KS}[\omega{\rm B97X\text{-}L}] + E_{c,{\rm VV10}}
+ \lambda\,E_{c,{\rm LinLCCD(hh)}}^{sr,\omega,\lambda} ,\qquad
\lambda = 0.6,\ \omega = 0.1\ {\rm Bohr}^{-1}. \tag{16.9}$$

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

## 16.6.1 Where λ goes

<!-- include: crates/ferric-cc/theory.md#"ωB97X-L-V correlation" -->

Solving at $\lambda = 1$ and multiplying by $\lambda$ afterwards would be
linear in $\lambda$ and, in the MP2 limit, overstate the correlation by
$1/\lambda$ (1.67 at $\lambda = 0.6$). The code avoids this by one solve
under the composite operator:

<!-- excerpt: ferric-cc/src/double_hybrid.rs::lambda_scaled_sr_operator -->
```rust
(cfg.lambda > 0.0)
    .then(|| Operator::composite(&[(cfg.lambda, OperatorKind::ErfcCoulomb, cfg.omega)]))
```

At $\lambda = 0$ the operator vanishes and its RI metric is singular, so no
correlation is computed.

This section has no verified box. The derivation and sympy proof of the
double hybrid belong to the double-hybrid grading work, and the registry
lists `double-hybrid` as a gap until they land.

## 16.6.2 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | ωB97X-L KS SCF with VV10 (libxc ωB97X-V form, the paper's parameters), through the KS convergence ladder; an error if every rung fails | `run_wb97x_l_v` |
| 2 | operator $\lambda\,{\rm erfc}(\omega r)/r$; $B^P_{pq}$ on the KS orbitals | `lambda_scaled_sr_operator` |
| 3 | LinLCCD(hh) (section 16.5) under that operator | `solve_wb97x_l_v` (`u_solve_wb97x_l_v` for UKS / semicanonical ROKS) |
| 4 | $E = E_{\rm KS} + \lambda E_c$, components kept separate | `assemble` |

The CLI kind converges its own ωB97X-L-V reference before the shared SCF
path, and takes `[dft] lambda` and `omega` as overrides of the published
0.6 and 0.1; an omitted key is the published value.

## 16.6.3 Numerical anchors

| Quantity | Systems | Error | Test |
|---|---|---|---|
| $E_{\rm KS}$ vs PySCF with the paper's parameters | H2O (RKS, exact J and RI-J), OH (ROKS → semicanonical → unrestricted) / def2-SVP | $\le 1.05\times10^{-9}$ Ha | `crates/ferric-cc/tests/validation_wb97xlv.rs` |
| $\lambda E_c$ on the same orbitals vs numpy LinLCCD(hh) on PySCF DF $\lambda\,$erfc integrals | same | $7.2\times10^{-14}$ Ha | `crates/ferric-cc/tests/validation_wb97xlv.rs` |
| total on ferric's orbitals | same | $1.1\times10^{-7}$ Ha | `crates/ferric-cc/tests/validation_wb97xlv.rs` |
| Be₂ bond energy vs the paper's Table 4 (2.3 kcal/mol) | Be₂ / def2-QZVPPD | 2.299 kcal/mol | `crates/ferric-cc/tests/wb97x_l_v_be2_bde.rs` |

## 16.6.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "ωB97X-L-V correlation" | the λ placement |
| Ransford and Carter-Fenk, PCCP 28, 14428 (2026), eqns 22 and 27, Tables 2 and 4 | the functional, parameters and the Be₂ reference |
| `ferric-cc/src/double_hybrid.rs`, module documentation | the post-SCF argument and the λ subtlety |

**Known limits (2026-11-15).** The CLI refuses an open-shell molecule; the
open-shell route is library-only. No Python binding. No gradient.
