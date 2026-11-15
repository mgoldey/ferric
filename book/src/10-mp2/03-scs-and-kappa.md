# 10.3 Spin-component scaling and κ-regularization

<!-- section: 10.3; status: ready; registry: scs-mp2;
     theory: crates/ferric-mp2/theory.md#"4. Spin-component scaling" -->

Two one-line changes to the MP2 energy reuse the pair loop of section 10.1
unchanged. Spin-component scaling (SCS) reweights the opposite- and
same-spin parts. κ-regularization damps each amplitude by a factor that
depends on its denominator. Both cost the same as RI-MP2.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/scs.rs::scs_mp2`; κ through `RiMp2Config::kappa` in `ferric-mp2/src/rimp2.rs` |
| CLI | `method.kind = "scs-mp2"` (`[mp2] c_os`, `c_ss`); κ: `method.kind = "rimp2"` with `[mp2] kappa` |
| Python | `run_scs_mp2(c_os=, c_ss=)`; `run_rimp2(kappa=)`; `run_optimize(method="scs-mp2")` for geometries |

## 10.3.1 SCS-MP2

$$ E_{\mathrm{SCS}} = c_{\mathrm{OS}}\,E_{\mathrm{OS}} + c_{\mathrm{SS}}\,E_{\mathrm{SS}} , \tag{10.9} $$

with $E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$ from (10.2) in a closed shell or
(10.8) in an open shell. The defaults are Grimme's $c_{\mathrm{OS}} = 6/5$
and $c_{\mathrm{SS}} = 1/3$ (Grimme, J. Chem. Phys. 118, 9095 (2003)).

The coefficients correct a known imbalance. MP2 overestimates same-spin
correlation and underestimates opposite-spin correlation. Raising
$c_{\mathrm{OS}}$ and lowering $c_{\mathrm{SS}}$ improves reaction energies.
For intermolecular interactions the best coefficients go the other way;
SCS-MP2(MI) uses $c_{\mathrm{OS}} = 0.40$, $c_{\mathrm{SS}} = 1.29$
(Goldey dissertation, chapter 5). No single pair serves both, which is the
motivation for section 10.5.

(10.9) is linear in the two components, so it needs no new proof beyond the
spin adaptation of section 10.1. With $c_{\mathrm{OS}} = c_{\mathrm{SS}} = 1$
it equals RI-MP2 exactly; the anchor below checks that.

SCS-MP2 has an analytic gradient (`run_optimize(method="scs-mp2")`): the
relaxed density weights the opposite-spin part by $c_{\mathrm{OS}}$ and the
rest by $c_{\mathrm{SS}}$. Its derivation is chapter 9's gradient section
(`crates/ferric-mp2/theory.md`, "9.2 Spin-component-scaled (SCS-MP2)
gradient").

## 10.3.2 κ-regularized MP2

κ-MP2 (Lee and Head-Gordon, J. Chem. Theory Comput. 14, 5203 (2018)) damps
every pair term of (10.3) by a factor that removes small-denominator
divergence:

$$ E_{\kappa_{\mathrm{reg}}} = -\sum_{ijab} \frac{(ia|jb)\,\big[\,2(ia|jb) - (ib|ja)\,\big]}{\Delta_{ij}^{ab}}\,\Big(1 - e^{-\kappa_{\mathrm{reg}}\Delta_{ij}^{ab}}\Big)^2 . \tag{10.10} $$

| $\kappa_{\mathrm{reg}}$ | Behaviour |
|---|---|
| $\kappa_{\mathrm{reg}} \to \infty$ | the factor is 1; (10.10) is RI-MP2 |
| $\kappa_{\mathrm{reg}} \to 0$ | the factor is $\kappa_{\mathrm{reg}}^2\Delta^2$; the energy goes to zero |
| finite | each term is bounded by $\kappa_{\mathrm{reg}}^2\Delta\,(ia|jb)[\ldots]$ as $\Delta \to 0$, so a near-degenerate pair cannot diverge |

$\kappa_{\mathrm{reg}}$ (`kappa` in the code and on every surface; $\kappa$ alone is reserved for orbital rotations) is in inverse Hartree, must be finite and positive, and has no default:
leaving the key out runs plain RI-MP2 on the unmodified code path. The
paper recommends a value near 1.45 Eh⁻¹. The same factor multiplies the
opposite- and same-spin parts, so `E_OS` and `E_SS` are reported regularized.

> **Unverified equation 10.10** · κ-regularized pair factor
>
> No committed sympy proof covers (10.10). The code applies
> $(1 - e^{\kappa_{\mathrm{reg}} \cdot \texttt{denom}})^2$ with
> `denom` $= -\Delta$ in `ferric-mp2/src/rimp2.rs::pair_energy`; the limits
> are tested numerically in `ferric-mp2/tests/kappa_mp2.rs::kappa_limits_recover_mp2_and_zero`.
>
> <!-- unverified: id=mp2.kappa.factor; todo=add a claim to scripts/mp2_variant_identities.py: kappa factor limits (kappa->inf gives MP2, kappa->0 gives 0, bounded as Delta->0) against ferric-mp2/src/rimp2.rs::pair_energy -->

## 10.3.3 What is anchored

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| SCS-MP2 at 6/5, 1/3 vs numpy RI-MP2 spin components on PySCF integrals | H2O, NH3, CH4 / aug-cc-pVDZ | $E_{\mathrm{SCS}}$ ≤ 3.5e-12 Ha; $E_{\mathrm{OS}}$, $E_{\mathrm{SS}}$ ≤ 3.0e-12 Ha | 1.5e-10 Ha; 1e-10 Ha | `crates/ferric-mp2/tests/validation_scs_mp2.rs` |
| κ-MP2 at κ = 0.5, 1.1, 2.0 Eh⁻¹ vs a numpy $(1-e^{-\kappa_{\mathrm{reg}}\Delta})^2$ sum on PySCF integrals | H2O, CH4 / cc-pVDZ | OS, SS, total ≤ 3.3e-12 Ha; κ → ∞ equals RI-MP2 exactly | 3e-11 Ha | `crates/ferric-mp2/tests/validation_kappa_mp2.rs` |

Figures from `site/src/reference/validation.md`.

## Known limits (2026-11-15)

- κ is closed-shell only on the CLI and in Python, and has no gradient;
  `run_rimp2(with_gradient=True, kappa=...)` raises.
- κ is refused by the local (amplitude-threshold) scheme.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "4. Spin-component scaling" | equation 10.9 |
| `crates/ferric-mp2/src/scs.rs`, `crates/ferric-mp2/src/rimp2.rs` | defaults, the κ factor and its validation |
| Goldey dissertation, chapter 5, Eqs. 5.1–5.3 | SCS and SCS(MI) coefficients |
| `site/src/reference/validation.md` | anchors |
