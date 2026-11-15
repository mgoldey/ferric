# 22.4 The nuclear gradient of the PCM energy

<!-- section: 22.4; status: ready; registry: pcm-gradient;
     theory: crates/ferric-pcm/theory.md#"8. Nuclear gradient" -->

A geometry optimization in a continuum needs the force on every nucleus,
including the part that comes from the cavity moving with the atoms. Because
the SCF energy is variational in the density, the solvated gradient is the
vacuum SCF gradient evaluated with the solvated orbitals, plus one term: the
derivative of $E_{\mathrm{pcm}}$ at fixed AO density. This section is that
term.

The derivative has two parts. The first is the force on the fixed charges
$q_k$ from the solute's nuclei and electrons. The second is the change of the
operator $K^{-1}R$ itself, because $S$, $D$ and the areas depend on where the
tesserae are, and the tesserae ride on their atoms.

| Symbol | Meaning | Code name |
|---|---|---|
| $x$, $y$ | $K^{-1}Rv$ and $K^{-\mathsf T}v$ | `x`, `y` |
| $q$ | $\tfrac12(x + R^{\mathsf T}y)$ | `q` |
| $w$, $u$, $z$ | $Sx+v$, $Aw$, $y - \tfrac{f}{2\pi}AD^{\mathsf T}y$ | `w`, `u`, `z` |
| $p$ | $f/2\pi$ | `pref` |
| $A(k)$ | atom that carries tessera $k$ | `Tessera::atom_index` |
| $h_{\mathrm{sw}}$ | the quintic switch of section 22.1 | `switch_h`, `switch_h_deriv` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pcm/src/gradient.rs::{pcm_gradient,pcm_gradient_parts}`, summed into the SCF gradient by `ferric-scf/src/gradient.rs::with_pcm_gradient` |
| CLI | `[pcm]` with `task = "optimize"` on `rhf`, `uhf`, `rohf` and `ksdft` |
| Python | `run_optimize(solvent=...)` |

## 22.4.1 The derivative

<!-- include: crates/ferric-pcm/theory.md#"8. Nuclear gradient" -->

The response part is the heart of `ferric-pcm/src/gradient.rs::response_term`.
The four vectors are built once, and the double loop over tessera pairs
contracts them with the kernel derivatives:

<!-- excerpt: ferric-pcm/src/gradient.rs::response_term -->
```rust
let w = s.dot(x) + v;
let u = &area * &w;
let dty = d.t().dot(y);
let z = y - &(pref * &area * &dty);
```

The first loop weights each ordered pair $(i,j)$ by `cs = -0.5 * z[i] * x[j]`
for $S$ and `cd = 0.5 * pref * y[i] * u[j]` for $D$, which are the first two
terms of the response formula. Pairs on one sphere are skipped, because both
tesserae move with the same atom. The third term and the diagonal terms all
depend on the geometry through $\partial\ln\mathrm{swf}_j$, and the second
loop handles them together.

The assembled gradient is added to the SCF gradient of the solvated orbitals:

<!-- excerpt: ferric-scf/src/gradient.rs::with_pcm_gradient -->
```rust
grad += &ferric_pcm::pcm_gradient(mol, prep, result.density_total(), pcm)?;
```

> **Verified equation 22.7** · the response derivative of $\tfrac12v^{\mathsf T}K^{-1}Rv$
>
> For $S(t)$, $D(t)$ and $A(t)$ depending on a nuclear coordinate $t$ at
> fixed $v$, $E = \tfrac12v^{\mathsf T}K^{-1}Rv$ has derivative
> $\tfrac12\big[-z^{\mathsf T}S'x + p\,y^{\mathsf T}D'u + p\sum_j(D^{\mathsf T}y)_jw_jA'_{jj}\big]$,
> with $x$, $y$, $w$, $u$, $z$ as in the table above. The check is exact
> ($n = 3$) against the $t$-derivative of the energy.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 8 |
> | Negative control | dropping the area term, or using $q_{\mathrm{sym}}$ for $x$, breaks it; the runner flips the sign in $z$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/gradient.rs::response_term` |
> | Code test | `fixed_density_gradient_matches_fd_water_default` in `crates/ferric-pcm/tests/pcm_gradient.rs` |
>
> <!-- verified: id=pcm.gradient.response; script=scripts/pcm_ief_sympy.py; claim=8; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/gradient.rs::response_term; rust_test=ferric-pcm/tests/pcm_gradient.rs::fixed_density_gradient_matches_fd_water_default; registry=pcm-gradient -->

The script proves the formula on a model with $S$, $D$ and $A$ as functions of
one coordinate. It does not prove the kernel derivatives $S'(r)$, $h(r)$,
$h'(r)$ or the switching derivative, which are transcribed from the code;
those are covered by the finite-difference tests in the anchors table, and by
nine single-term mutants of the Rust code that each fail them (section
22.4.4).

## 22.4.2 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | cavity, $S$, $D$, $K$, $R$, $v$ at the current geometry and density | as one SCF iteration | `pcm_gradient_parts` |
| 2 | $x$, $y$, $q$ by two LU solves | $N^3$ | `pcm_gradient_parts` |
| 3 | potential term $q^{\mathsf T}\partial v$: nuclei, AO centres, tessera positions | $N\,N^2_{\mathrm{bf}}$ with libint derivative engines | `potential_term_point`, `potential_term_smeared`, `nuclear_tessera_term` |
| 4 | response term: kernel derivatives over all off-sphere ordered pairs | $N^2$ | `response_term` |
| 5 | switching derivative $\nabla\ln\mathrm{swf}_j$ over tesserae and atoms | $N\,N_{\mathrm{at}}$ | `response_term`, `switch_h_deriv` |
| 6 | add to the solvated SCF gradient | | `with_pcm_gradient` |

`pcm_gradient_parts` rebuilds the cavity from the molecule exactly as
`PcmContext::new` does, and re-solves the charges at the converged density.
It does not reuse the SCF's factorization.

## 22.4.3 Options and refusals

| Case | Behaviour |
|---|---|
| `PcmConfig::cavity` set (an injected tessera list) | error: the list carries no record of how it moves |
| `sd_kind`, `probe` | both `PointCharge` and `GaussianSmeared` $S$/$D$, both point and Gaussian probes |
| restricted, unrestricted and restricted-open SCF, and KS | the same term is added by `restricted_scf_gradient`, `unrestricted_scf_gradient` and `restricted_open_scf_gradient` |
| COSMO | `ferric-scf/src/gradient.rs::refuse_cosmo_gradient` refuses (section 22.5) |
| frequencies with `[pcm]` | refused by the CLI; there is no analytic Hessian |

## 22.4.4 Numerical anchors

| Check | System | Max error (Ha/Bohr) | Source |
|---|---|---|---|
| PySCF 2.13 `nuc_grad_method()`, IEF-PCM, 302 points, Gaussian probes | water, methanol / STO-3G | $4.2\times10^{-11}$, $7.2\times10^{-11}$ | `site/src/reference/validation.md`, IEF-PCM gradient row |
| central FD at fixed density | water, C1 methanol / STO-3G | $\le 2.0\times10^{-7}$ | same |
| central FD of the solvated SCF energy | same, RHF, RKS, UHF, ROHF | $\le 2.9\times10^{-7}$ | same |
| torque against FD of a rigid rotation | C1 methanol / STO-3G | $8\times10^{-9}$ (torque $1.36\times10^{-4}$) | same |
| $\varepsilon = 1 + 2\times10^{-9}$ against the vacuum gradient | same | $3.3\times10^{-12}$ | same |

Each part of the gradient also sums to zero over the atoms, cavity included.
The PySCF comparison runs on ferric's own cavity at Lebedev order 302, which
reproduces PySCF's `gen_surface` (section 22.1). Nine single-term mutants
(the switching, area, $S$ and $D$ diagonal and off-diagonal, tessera-site and
nuclear-tessera terms) each fail the FD tests.

## 22.4.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pcm/theory.md`, section 8 | the three parts of the derivative and their kernels |
| `scripts/pcm_ief_sympy.py`, claim 8 | the response formula on a model |
| `crates/ferric-pcm/tests/pcm_gradient.rs`, `crates/ferric-scf/tests/pcm_gradient.rs` | FD at fixed density and at SCF level |

**Known limits (2026-11-15).**

- Analytic first derivatives only: no analytic nuclear Hessian, and the CLI
  refuses frequencies with `[pcm]`.
- `PointCharge` $S$/$D$ energies are far rougher in the switching zone than
  the default, so central differences are not yet in their $O(h^2)$ regime at
  $h = 2.5\times10^{-4}$ on a coarse C1 methanol cavity
  (`crates/ferric-pcm/theory.md`, section 8.3).
