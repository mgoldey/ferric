# 22.2 The IEF-PCM operators, the symmetrized solve and the sphere limits

<!-- section: 22.2; status: ready; registry: pcm;
     theory: crates/ferric-pcm/theory.md#"3. The IEF-PCM operators $K$ and $R$", "4. Symmetrized charge solve and energy", "5. Sphere limits: Born, Kirkwood, Gauss" -->

Given the electrostatic potential $v_k$ of the solute at every tessera, the
isotropic integral equation formalism (IEF-PCM) finds the apparent surface
charges $q_k$ that represent the polarized dielectric. The boundary condition
at the cavity surface is written once as a linear system over the tesserae
and solved directly; there is no iteration in the solve. The self-consistency
between the charges and the solute density belongs to the SCF loop (section
22.3).

| Symbol | Meaning | Code name |
|---|---|---|
| $\varepsilon$ | solvent dielectric constant | `PcmConfig::epsilon` |
| $f(\varepsilon)$ | $(\varepsilon-1)/(\varepsilon+1)$ | `f_eps` |
| $A$ | $\operatorname{diag}(a_k)$ | the column scaling of `d` in `da` |
| $K$, $R$ | the two IEF operators, $Kq = Rv$ | `k`, `r` |
| $Q$ | the response $K^{-1}R$, not symmetric | implicit |
| $v$, $q$ | tessera potential, apparent charges | `v`, `q` |
| $E_{\mathrm{pcm}}$ | solvation energy $\tfrac12 q\cdot v$ | `e_pcm` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pcm/src/matrices.rs::build_k_r`, `ferric-pcm/src/solver.rs::solve_pcm_charges` |
| CLI | `[pcm]`; the charges and energy are inside every SCF iteration |
| Python | `solvent=` on the SCF drivers |

## 22.2.1 The operators

<!-- include: crates/ferric-pcm/theory.md#"3. The IEF-PCM operators $K$ and $R$" -->

The operators are two lines of `ferric-pcm/src/matrices.rs::build_k_r`, after
$DA$ is formed by scaling the columns of $D$ by the areas:

<!-- excerpt: ferric-pcm/src/matrices.rs::build_k_r -->
```rust
let k = s - &(f_eps / two_pi * &das);
```

<!-- excerpt: ferric-pcm/src/matrices.rs::build_k_r -->
```rust
let eye = Array2::<f64>::eye(n);
let r = -f_eps * (&eye - &(&da / two_pi));
```

| Symbol | Code |
|---|---|
| $f$ | `f_eps` |
| $DAS$ | `das`, from `da.dot(s)` |
| $2\pi$ | `two_pi` |
| $K$ | `k` |
| $R$ | `r` |

## 22.2.2 The symmetrized solve

<!-- include: crates/ferric-pcm/theory.md#"4. Symmetrized charge solve and energy" -->

The solver is the three lines of equation (the two LU solves are above them in
`ferric-pcm/src/solver.rs::solve_pcm_charges`):

<!-- excerpt: ferric-pcm/src/solver.rs::solve_pcm_charges -->
```rust
let q2 = r.t().dot(&kt_inv_v);

let q = 0.5 * (&q1 + &q2);
let e_pcm = 0.5 * q.dot(v);
```

| Symbol | Code |
|---|---|
| $K^{-1}Rv$ | `q1`, from `k.solve(&rv)` |
| $(K^{\mathsf T})^{-1}v$ | `kt_inv_v`, from `k_t.solve(v)` |
| $R^{\mathsf T}(K^{\mathsf T})^{-1}v$ | `q2` |
| $q$ | `q` |
| $E_{\mathrm{pcm}}$ | `e_pcm` |

Why symmetrize at all, when the energy is the same? The charges also set the
reaction-field operator, and the SCF needs $V^{\mathrm{pcm}} = \partial
E_{\mathrm{pcm}}/\partial P$ for the energy and the Fock matrix to describe one
functional. That holds for the symmetric response and not for the raw
$Qv$ (section 22.3, equation 22.4).

> **Verified equation 22.3** · the symmetrized charges keep the energy
>
> With $Q = K^{-1}R$ and a generic non-symmetric $D$, the energy of
> $q_{\mathrm{sym}} = \tfrac12[K^{-1}Rv + R^{\mathsf T}(K^{\mathsf T})^{-1}v]$
> equals the energy of the unsymmetrized $Qv$. The two charge vectors differ
> unless $DAS$ is symmetric, in which case they coincide exactly.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 5 (exact rational arithmetic, $n = 4$) |
> | Negative control | the reversed order $(K^{\mathsf T})^{-1}R^{\mathsf T}v$ changes the energy; a generic $D$ gives $q_{\mathrm{sym}} \ne Qv$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/solver.rs::solve_pcm_charges` |
> | Code test | `symmetrized_energy_equals_unsymmetrized_quadratic_form` in `crates/ferric-pcm/src/solver.rs` |
>
> <!-- verified: id=pcm.solve.symmetrized; script=scripts/pcm_ief_sympy.py; claim=5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/solver.rs::solve_pcm_charges; rust_test=ferric-pcm/src/solver.rs::symmetrized_energy_equals_unsymmetrized_quadratic_form; registry=pcm -->

## 22.2.3 The sphere limits

A single sphere has an exact solution, so it tests the operators without any
other code. In each spherical-harmonic channel the operators are scalars, and
the channel factor is Kirkwood's reaction-field factor for a dielectric
sphere.

<!-- include: crates/ferric-pcm/theory.md#"5. Sphere limits: Born, Kirkwood, Gauss" -->

> **Verified equation 22.4** · IEF-PCM is Kirkwood's solution on a sphere
>
> With $S_l = 4\pi R/(2l+1)$ and $D_l = -2\pi/(2l+1)$ the channel factor of
> $K$ and $R$ is $k_l = (\varepsilon-1)(l+1)/((l+1)\varepsilon+l)$ for every
> $l$. At $l = 0$ this is Gauss's law, total induced charge
> $-(\varepsilon-1)/\varepsilon$, and the Born energy
> $-(\varepsilon-1)Q^2/(2\varepsilon R)$.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 3 |
> | Negative control | $f = (\varepsilon-1)/\varepsilon$ is not Kirkwood; the C-PCM factor $(\varepsilon-1)/\varepsilon$ and the COSMO factor $(\varepsilon-1)/(\varepsilon+\tfrac12)$ each match at one $l$ only |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/matrices.rs::build_k_r` |
> | Code test | `point_charge_in_sphere_matches_born_kirkwood_and_gauss_law` in `crates/ferric-pcm/tests/ief_limits.rs`, the discrete counterpart (equation 22.5) |
>
> <!-- verified: id=pcm.sphere.kirkwood; script=scripts/pcm_ief_sympy.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/matrices.rs::build_k_r; rust_test=ferric-pcm/tests/ief_limits.rs::point_charge_in_sphere_matches_born_kirkwood_and_gauss_law; registry=pcm -->

The channel algebra is exact; the discrete Lebedev sphere reproduces it only
as the grid is refined. The script solves the discrete system with numpy on a
scipy Lebedev sphere, and the Rust test does the same through
`build_cavity`, `build_s_d_kind`, `build_k_r` and `solve_pcm_charges`.

> **Verified equation 22.5** · the discrete sphere converges to Born, Kirkwood and Gauss
>
> A unit charge at $d/R = 0, \tfrac14, \tfrac12$ inside a Lebedev sphere of
> 50, 110 and 302 points gives an energy that converges toward the Kirkwood
> series and a total charge that converges toward $-(\varepsilon-1)/\varepsilon$,
> with the 302-point error inside the a-posteriori estimate
> $|E_{302}-E_{110}|$.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 7 (numpy replica of `build_cavity`, `build_s_d_kind`, `build_k_r`, `solve_pcm_charges`; needs scipy $\ge 1.15$) |
> | Negative control | the transcribed diagonal $D_{ii} = -S_{ii}/R$ instead of $-S_{ii}/(2R)$ turns the script red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/solver.rs::solve_pcm_charges`, `ferric-pcm/src/matrices.rs::build_s_d_kind` |
> | Code test | `point_charge_in_sphere_matches_born_kirkwood_and_gauss_law` in `crates/ferric-pcm/tests/ief_limits.rs` |
>
> <!-- verified: id=pcm.sphere.discrete; script=scripts/pcm_ief_sympy.py; claim=7; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/solver.rs::solve_pcm_charges; rust_test=ferric-pcm/tests/ief_limits.rs::point_charge_in_sphere_matches_born_kirkwood_and_gauss_law; registry=pcm -->

## 22.2.4 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $S$, $D$ (section 22.1) | $N^2$ | `build_s_d_kind` |
| 2 | $DA$, $DAS$, $K$, $R$ | $N^3$ (one GEMM) | `build_k_r` |
| 3 | $Rv$ | $N^2$ | `solve_pcm_charges` |
| 4 | LU solve of $Kx = Rv$ and of $K^{\mathsf T}y = v$ | $N^3$ per SCF iteration | `solve_pcm_charges`, `ndarray_linalg::Solve` |
| 5 | $q = \tfrac12(x + R^{\mathsf T}y)$, $E_{\mathrm{pcm}} = \tfrac12 q\cdot v$ | $N^2$ | `solve_pcm_charges` |

Steps 1 and 2 run once per SCF in `PcmContext::from_tesserae`; steps 3 to 5
run every iteration. `solve_pcm_charges` factorizes $K$ and $K^{\mathsf T}$ on
each call and keeps no factorization between iterations, so the per-iteration
cost is two dense $N^3$ factorizations.

## 22.2.5 Reproducibility and failures

The solve is LAPACK with no iteration, so it carries no convergence
threshold. A singular $K$ (two coincident tesserae, a degenerate cavity)
returns an error from the solve, never a NaN charge vector. `build_k_r`
rejects $\varepsilon \le 1$ and a non-finite $\varepsilon$: vacuum has no
reaction field, and a run with no solvent omits `[pcm]` entirely.

## 22.2.6 Numerical anchors

| Quantity | System | Result | Source |
|---|---|---|---|
| charges, on PySCF's injected cavity | water, NH3 / STO-3G, cc-pVDZ; $\varepsilon = 78.4, 4.7$ | $1.3\times10^{-17}$ | `site/src/reference/validation.md`, IEF-PCM row |
| $E_{\mathrm{pcm}}$, same | same | $3.5\times10^{-18}$ Ha | same |
| total SCF energy on PySCF's cavity | same | $4.8\times10^{-12}$ Ha | same |
| Born energy at $d/R = 0$, 302 points | Ne sphere, $\varepsilon = 78.4$ and $4$ | $1.1\times10^{-8}$ and $1.7\times10^{-7}$ relative | same |
| Kirkwood energy at $d/R = 0.5$, 302 points | same | $\le 1.8\times10^{-4}$ relative | same |
| total induced charge against Gauss's law | same | within $1.3\times10^{-7}$ | same |
| conductor limit, water's cavity | $\varepsilon = 10^6$ and $10^8$ | gap scales as $1-f(\varepsilon)$, ratio 100.000 | same |

The solver comparison feeds PySCF's own cavity to ferric's solver, so the
first three rows isolate $K$, $R$ and the solve from the cavity construction.

## 22.2.7 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pcm/theory.md`, sections 3 to 5 | $K$, $R$, the symmetrized solve, the sphere limits |
| `scripts/pcm_ief_sympy.py`, claims 3, 5 and 7 | channel factors, symmetrization, the discrete sphere |
| `crates/ferric-pcm/tests/ief_limits.rs` | the Rust twin of claim 7, and the conductor limit |
| `crates/ferric-scf/tests/validation_pcm.rs` | the PySCF comparison |

**Known limits (2026-11-15).**

- The integral equation is isotropic: no anisotropic or ionic-strength
  variants.
- `PointCharge` $S$ and $D$ converge far more slowly than the default on the
  sphere tests (`crates/ferric-pcm/theory.md`, section 5): the central-charge
  error stays at the $10^{-3}$ level at 302 points, from the self-term on a
  grid with unequal weights.
- The sphere tests cannot see a swap of the source and field normals, because
  $D$ is symmetric on a sphere. The molecular-cavity PySCF comparison is the
  test for that.
