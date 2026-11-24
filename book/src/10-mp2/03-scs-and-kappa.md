# Spin-component scaling and κ-regularization {#scs-and-kappa}

<!-- status: ready; registry: scs-mp2, kappa-mp2 -->

MP2 has two systematic errors that a cheap change to the pair loop can
address. The first is an imbalance between the spin components: for bond
energies MP2 gets too little opposite-spin correlation and too much
same-spin correlation. Spin-component scaling (SCS) reweights the two parts
of (@eq:mp2-cs-spin-adaptation). The second is the $1/\Delta$ in every term: when a gap closes,
as in a stretched bond, the energy diverges. κ-regularization damps each
term by a factor that depends on its denominator. Both reuse the pair loop
of section @sec:closed-shell-ri-mp2 and cost the same as RI-MP2.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/scs.rs::scs_mp2`; κ through `RiMp2Config::kappa` in `ferric-mp2/src/rimp2.rs` |
| CLI | `method.kind = "scs-mp2"` (`[mp2] c_os`, `c_ss`); κ: `method.kind = "rimp2"` with `[mp2] kappa` |
| Python | `run_scs_mp2(c_os=, c_ss=)`; `run_rimp2(kappa=)`; `run_optimize(method="scs-mp2")` for geometries |

## SCS-MP2 {#scs-and-kappa-scs-mp2}

Section @sec:closed-shell-ri-mp2 showed that the MP2 energy is the sum of two physically
different pieces. Opposite-spin pairs get no help from Hartree–Fock in
avoiding each other, and carry about three quarters of the correlation
energy of water. Same-spin pairs already sit in an exchange hole. MP2
makes different errors in the two parts, so SCS scales them separately:

$$ E_{\mathrm{SCS}} = c_{\mathrm{OS}}\,E_{\mathrm{OS}} + c_{\mathrm{SS}}\,E_{\mathrm{SS}} , \label{eq:scs-and-kappa-1} $$

with $E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$ from (@eq:mp2-cs-spin-adaptation) in a closed shell or
(@eq:mp2-u-spin-blocks) in an open shell. The defaults are Grimme's $c_{\mathrm{OS}} = 6/5$
and $c_{\mathrm{SS}} = 1/3$ (Grimme, J. Chem. Phys. 118, 9095 (2003)).

Check the limits. With $c_{\mathrm{OS}} = c_{\mathrm{SS}} = 1$, (@eq:scs-and-kappa-1) is
(@eq:closed-shell-ri-mp2-3), plain RI-MP2; the anchor below checks that. With
$c_{\mathrm{SS}} = 0$ only the opposite-spin part is left. That is SOS-MP2,
whose energy contains no exchange product $(ia|jb)(ib|ja)$; section @sec:laplace-mp2 shows why
that admits a cheaper algorithm.

Which way should the coefficients go? It depends on the property. The
dissertation (chapter 5, section 5.1) quotes the SCS(MI) work: "the MP2
description of bond energies contains a systematically underestimated
opposite spin-component and a simultaneously overestimated same
spin-component, while the reverse appears generally true for
intermolecular interactions." Grimme's values raise $c_{\mathrm{OS}}$ and
lower $c_{\mathrm{SS}}$, which improves reaction energies. For
intermolecular interactions the best coefficients go the other way:
SCS-MP2(MI) uses $c_{\mathrm{OS}} = 0.40$, $c_{\mathrm{SS}} = 1.29$
(DiStasio and Head-Gordon, Mol. Phys. 105, 1073 (2007); quoted in the
dissertation, chapter 5). No single pair serves both, which is the
motivation for section @sec:scs-mp2-2terfc.

`ferric-mp2/src/scs.rs::scs_mp2` evaluates (@eq:scs-and-kappa-1) in two lines, copied
here with a comment added:

```rust
let sc = ri_mp2_spin_components_any(mol, obs, dfbs, Operator::coulomb(), rhf, &ri_config)?;
let scs_corr = config.c_os * sc.e_os + config.c_ss * sc.e_ss;    // E_SCS
```

`ferric-mp2/src/u_rimp2.rs::ri_mp2_spin_components_any` returns (@eq:mp2-cs-spin-adaptation) on a
restricted reference and the grouping $E_{\mathrm{OS}} = E_{\alpha\beta}$,
$E_{\mathrm{SS}} = E_{\alpha\alpha} + E_{\beta\beta}$ of section @sec:unrestricted-ri-mp2 on an
unrestricted one. So one line serves both.

For water in cc-pVDZ (cc-pVDZ-RI fitting basis, from the `589b50832` build),
Grimme's weights move the correlation energy by about 4 mEh, and dropping the
same-spin part leaves $E_{\mathrm{OS}}$ alone:

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

mp2 = ferric.run_scs_mp2(mol, bs, aux, c_os=1.0, c_ss=1.0)
scs = ferric.run_scs_mp2(mol, bs, aux)  # defaults 6/5, 1/3
sos = ferric.run_scs_mp2(mol, bs, aux, c_os=1.0, c_ss=0.0)
print("E_OS = %.6f  E_SS = %.6f" % (mp2.e_os, mp2.e_ss))
print("MP2     E_c = %.6f" % mp2.scs_corr)
print("SCS-MP2 E_c = %.6f  (6/5, 1/3)" % scs.scs_corr)
print("c_ss=0  E_c = %.6f  (E_OS only)" % sos.scs_corr)
```

```text
E_OS = -0.152440  E_SS = -0.051593
MP2     E_c = -0.204033
SCS-MP2 E_c = -0.200126  (6/5, 1/3)
c_ss=0  E_c = -0.152440  (E_OS only)
```

(@eq:scs-and-kappa-1) is linear in the two components, so the spin
adaptation of section @sec:closed-shell-ri-mp2 is its proof. Part C of
`scripts/mp2_family_sympy.py` checks the assembly itself: at
$c_{\mathrm{OS}} = c_{\mathrm{SS}} = 1$ it is the MP2 energy, and its
operator limits are those of section @sec:scs-mp2-2terfc.

SCS-MP2 has an analytic gradient (`run_optimize(method="scs-mp2")`): the
relaxed density weights the opposite-spin part by $c_{\mathrm{OS}}$ and the
rest by $c_{\mathrm{SS}}$. Its derivation is section @sec:rimp2-gradient-scs-mp2.

## κ-regularized MP2 {#scs-and-kappa-kappa-regularized-mp2}

Every term of (@eq:closed-shell-ri-mp2-3) has $\Delta_{ij}^{ab}$ in the denominator. When the
HOMO–LUMO gap closes, the smallest $\Delta$ goes to zero and the energy
diverges. κ-MP2 (Lee
and Head-Gordon, J. Chem. Theory Comput. 14, 5203 (2018)) multiplies every
pair term by a factor that vanishes with $\Delta$:

$$ E_{\kappa_{\mathrm{reg}}} = -\sum_{ijab} \frac{(ia|jb)\,\big[\,2(ia|jb) - (ib|ja)\,\big]}{\Delta_{ij}^{ab}}\,\Big(1 - e^{-\kappa_{\mathrm{reg}}\Delta_{ij}^{ab}}\Big)^2 . \label{eq:mp2-kappa-factor} $$

Take the three limits of the factor $(1 - e^{-\kappa_{\mathrm{reg}}\Delta})^2$
in turn.

- $\kappa_{\mathrm{reg}} \to \infty$ at fixed $\Delta > 0$:
  $e^{-\kappa_{\mathrm{reg}}\Delta} \to 0$, the factor is 1, and (@eq:mp2-kappa-factor) is
  RI-MP2.
- $\kappa_{\mathrm{reg}} \to 0$: expand
  $1 - e^{-x} = x - x^2/2 + \dots$ with $x = \kappa_{\mathrm{reg}}\Delta$.
  The factor is $\kappa_{\mathrm{reg}}^2\Delta^2$ to leading order, and the
  energy goes to zero.
- $\Delta \to 0$ at fixed $\kappa_{\mathrm{reg}}$: use $1 - e^{-x} \le x$
  for $x \ge 0$. (The difference $x - 1 + e^{-x}$ is 0 at $x = 0$ and its
  derivative $1 - e^{-x}$ is not negative.) Square it and divide by $\Delta$:

  $$ \frac{\big(1 - e^{-\kappa_{\mathrm{reg}}\Delta}\big)^2}{\Delta} \le \kappa_{\mathrm{reg}}^2\,\Delta . $$

  So the magnitude of each term of (@eq:mp2-kappa-factor) is at most
  $\kappa_{\mathrm{reg}}^2\Delta\,|(ia|jb)[\ldots]|$, which goes to zero
  with $\Delta$. A near-degenerate pair cannot diverge.

What does a finite $\kappa_{\mathrm{reg}}$ do? It sets an energy scale
$1/\kappa_{\mathrm{reg}}$. Excitations with $\Delta \gg 1/\kappa_{\mathrm{reg}}$
keep their full MP2 weight; excitations with $\Delta \ll 1/\kappa_{\mathrm{reg}}$
are suppressed as $\kappa_{\mathrm{reg}}^2\Delta^2$. The switch is gradual.
At $\kappa_{\mathrm{reg}} = 1.45$ Eh⁻¹ the factor is

| $\Delta$ (Eh) | 0.1 | 0.5 | 1 | 2 | 5 |
|---|---|---|---|---|---|
| $(1 - e^{-1.45\Delta})^2$ | 0.018 | 0.266 | 0.586 | 0.893 | 0.9986 |

so a pair with a denominator of 1 Eh keeps 59% of its MP2 weight.

The same water run, from the `589b50832` build, shows the factor at work. At
small $\kappa_{\mathrm{reg}}$ most of the correlation is gone; at 1.45 Eh⁻¹ about
1.3% is removed; by $\kappa_{\mathrm{reg}} = 20$ Eh⁻¹ the energy is RI-MP2 to the
printed digits:

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

ref = ferric.run_rimp2(mol, bs, aux)
print("RI-MP2         E_c = %.8f" % ref.mp2_corr)
for kappa in (0.1, 0.5, 1.45, 4.0, 20.0, 100.0):
    r = ferric.run_rimp2(mol, bs, aux, kappa=kappa)
    print("kappa = %6.2f  E_c = %.8f  ratio to MP2 = %.4f"
          % (kappa, r.mp2_corr, r.mp2_corr / ref.mp2_corr))
```

```text
RI-MP2         E_c = -0.20403346
kappa =   0.10  E_c = -0.03041023  ratio to MP2 = 0.1490
kappa =   0.50  E_c = -0.15495177  ratio to MP2 = 0.7594
kappa =   1.45  E_c = -0.20133264  ratio to MP2 = 0.9868
kappa =   4.00  E_c = -0.20402549  ratio to MP2 = 1.0000
kappa =  20.00  E_c = -0.20403346  ratio to MP2 = 1.0000
kappa = 100.00  E_c = -0.20403346  ratio to MP2 = 1.0000
```

$\kappa_{\mathrm{reg}}$ (`kappa` in the code and on every surface; $\kappa$
alone is reserved for orbital rotations) is in inverse Hartree, must be
finite and positive, and has no default: leaving the key out runs plain
RI-MP2 on the unmodified code path. The paper recommends a value near
1.45 Eh⁻¹. The same factor multiplies the opposite- and same-spin parts, so
`E_OS` and `E_SS` are reported regularized.

The code is the κ branch of `ferric-mp2/src/rimp2.rs::pair_energy`, next
to the plain loop of @sec:closed-shell-ri-mp2-algorithm; only the comments are changed:

```rust
Some(k) => {
    for a in 0..nvir {
        for b in 0..nvir {
            let g_ab = g_i[(a, jcol + b)]; // (ia|jb)
            let g_ba = g_i[(b, jcol + a)]; // (ib|ja)
            let denom = e_ij - eps[nocc_total + a] - eps[nocc_total + b];   // -Delta
            let d1 = 1.0 - (k * denom).exp();         // 1 - e^{-kappa Delta}
            let damp = d1 * d1;                       // the factor of E_kappa
            e_os_ij += damp * g_ab * g_ab / denom;
            e_ss_ij += damp * g_ab * (g_ab - g_ba) / denom;
        }
    }
}
```

Two points about it:

- **The sign is in `denom` again.** `denom` is $-\Delta$, so
  `(k * denom).exp()` is $e^{-\kappa_{\mathrm{reg}}\Delta}$. A port that
  stores $+\Delta$ has to flip the sign inside the exponential as well as in
  the division.
- **The branch is separate on purpose.** With `kappa` unset the `None` arm
  runs: the loop of @sec:closed-shell-ri-mp2-algorithm with no factor, so no multiplication by a number
  close to 1 enters the plain RI-MP2 energy.

> **Verified equation @eq:mp2-kappa-factor** · κ-regularized pair factor
>
> The factor $(1 - e^{\kappa_{\mathrm{reg}}\cdot\texttt{denom}})^2$ with
> `denom` $= -\Delta$ tends to 1 as $\kappa_{\mathrm{reg}} \to \infty$, is
> $\kappa_{\mathrm{reg}}^2\Delta^2$ to leading order as
> $\kappa_{\mathrm{reg}} \to 0$, and satisfies
> $(1 - e^{-\kappa_{\mathrm{reg}}\Delta})^2/\Delta \le \kappa_{\mathrm{reg}}^2\Delta$.
> The spatial kernel with this factor on both $E_{\mathrm{OS}}$ and
> $E_{\mathrm{SS}}$, summed over $j \ge i$ with weight 2 off the diagonal,
> equals the regularized spin-orbital sum component by component.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_family_sympy.py`, part G (symbolic limits and bound; the factor table above; a brute-force spin-orbital sum as the independent route) |
> | Negative control | the runner flips the sign of the exponent, $e^{\kappa_{\mathrm{reg}}\texttt{denom}} \to e^{-\kappa_{\mathrm{reg}}\texttt{denom}}$, and requires part G to fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::pair_energy` |
> | Code test | `kappa_limits_recover_mp2_and_zero` in `crates/ferric-mp2/tests/kappa_mp2.rs` |
>
> <!-- verified: id=mp2.kappa.factor; script=scripts/mp2_family_sympy.py; claim=G; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::pair_energy; rust_test=ferric-mp2/tests/kappa_mp2.rs::kappa_limits_recover_mp2_and_zero; registry=kappa-mp2 -->

## What is anchored {#scs-and-kappa-anchored}

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| SCS-MP2 at 6/5, 1/3 vs numpy RI-MP2 spin components on PySCF integrals | H2O, NH3, CH4 / aug-cc-pVDZ | $E_{\mathrm{SCS}}$ ≤ 3.5e-12 Ha; $E_{\mathrm{OS}}$, $E_{\mathrm{SS}}$ ≤ 3.0e-12 Ha | 1.5e-10 Ha; 1e-10 Ha | `crates/ferric-mp2/tests/validation_scs_mp2.rs` |
| κ-MP2 at κ = 0.5, 1.1, 2.0 Eh⁻¹ vs a numpy $(1-e^{-\kappa_{\mathrm{reg}}\Delta})^2$ sum on PySCF integrals | H2O, CH4 / cc-pVDZ | OS, SS, total ≤ 3.3e-12 Ha; κ → ∞ equals RI-MP2 exactly | 3e-11 Ha | `crates/ferric-mp2/tests/validation_kappa_mp2.rs` |

Figures from `site/src/reference/validation.md`.

## Known limits {#scs-and-kappa-limits}

- κ is closed-shell only on the CLI and in Python, and has no gradient;
  `run_rimp2(with_gradient=True, kappa=...)` raises.
- κ is refused by the local (amplitude-threshold) scheme.

## Sources {#scs-and-kappa-sources}

| Source | Used for |
|---|---|
| Lee and Head-Gordon, J. Chem. Theory Comput. 14, 5203 (2018) | the κ factor (@eq:mp2-kappa-factor) |
| DiStasio and Head-Gordon, Mol. Phys. 105, 1073 (2007) | the SCS-MP2(MI) coefficients |
| `crates/ferric-mp2/src/scs.rs`, `crates/ferric-mp2/src/rimp2.rs` | defaults, the κ factor and its validation |
| `scripts/mp2_family_sympy.py`, parts C and G | the SCS assembly; the κ limits, bound, factor table and spin adaptation |
| Goldey dissertation, chapter 5, section 5.1, Eqs. 5.1–5.3 | SCS and SCS(MI) coefficients, the quoted imbalance |
| `site/src/reference/validation.md` | anchors |
