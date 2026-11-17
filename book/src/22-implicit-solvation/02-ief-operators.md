# 22.2 The IEF-PCM operators, the symmetrized solve and the sphere limits

<!-- section: 22.2; status: ready; registry: pcm;
     theory: crates/ferric-pcm/theory.md#"3. The IEF-PCM operators $K$ and $R$", "4. Symmetrized charge solve and energy", "5. Sphere limits: Born, Kirkwood, Gauss" -->

Given the electrostatic potential $v_k$ of the solute at every tessera, which
surface charges $q_k$ reproduce the polarized dielectric? The isotropic
integral equation formalism (IEF-PCM) answers with one linear system over
the tesserae, solved directly with no iteration. The self-consistency
between the charges and the solute density belongs to the SCF loop
(section 22.3). The plan: build the operators $K$ and $R$ (22.2.1), solve
for a symmetrized charge vector (22.2.2), and check the result against the
exact solution for a sphere (22.2.3).

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

Start from the isotropic IEF equation of Cancès, Mennucci and Tomasi
(JCP 107, 3032 (1997)), written on the tesserae with the matrices of
section 22.1:

$$ \left[ 2\pi\frac{\varepsilon+1}{\varepsilon-1} I - DA \right] S\,q = -\left( 2\pi I - DA\right) v . $$

Multiply both sides by $f/2\pi$, with

$$ f(\varepsilon) = \frac{\varepsilon-1}{\varepsilon+1} . $$

The bracket on the left becomes $I - \tfrac{f}{2\pi}DA$, because
$\tfrac{f}{2\pi}\cdot 2\pi\tfrac{\varepsilon+1}{\varepsilon-1} = 1$. The right
side becomes $-f(I - DA/2\pi)v$. Name the two matrices:

$$ K = S - \frac{f}{2\pi}\, DA\, S = \left(I - \tfrac{f}{2\pi}DA\right) S, \qquad
   R = -f\left(I - \frac{DA}{2\pi}\right), \qquad K q = R v . $$

$DA$ is $D$ with column $j$ scaled by $a_j$.

Check the two ends of the dielectric scale. In vacuum, $\varepsilon = 1$,
$f = 0$ and $R = 0$, so $q = 0$: no reaction field. `build_k_r` rejects
$\varepsilon \le 1$ and a non-finite $\varepsilon$ (22.2.5). As $\varepsilon\to\infty$,
$f\to 1$, and section 22.5 shows that the system becomes the conductor
equation $Sq = -v$.

$K$ and $R$ depend on the geometry and $\varepsilon$ only. They are built
once per SCF in `ferric-pcm/src/lib.rs::PcmContext::from_tesserae`. The
whole construction is the end of `build_k_r`, unchanged:

<!-- excerpt: ferric-pcm/src/matrices.rs::build_k_r -->
```rust
let mut da = d.clone();
for j in 0..n {
    let a_j = tess[j].area;
    for i in 0..n {
        da[(i, j)] *= a_j;
    }
}

let das = da.dot(s);
let two_pi = 2.0 * std::f64::consts::PI;

// K = S - f(eps)/(2*pi) * DA * S
let k = s - &(f_eps / two_pi * &das);

// R = -f(eps) * (I - DA/(2*pi))
let eye = Array2::<f64>::eye(n);
let r = -f_eps * (&eye - &(&da / two_pi));
```

| Symbol | Code |
|---|---|
| $f$ | `f_eps`, $(\varepsilon-1)/(\varepsilon+1)$ |
| $DA$ | `da` |
| $DAS$ | `das`, one GEMM |
| $2\pi$ | `two_pi` |
| $K$ | `k` |
| $R$ | `r` |

## 22.2.2 The symmetrized solve

The response $Q = K^{-1}R$ is not symmetric, because $D$ is not. The code
applies its symmetric part:

$$ q = \tfrac12\left[ K^{-1} R v + R^{\mathsf T} (K^{\mathsf T})^{-1} v \right] = \tfrac12 (Q + Q^{\mathsf T})\, v . $$

The second equality uses two rules for transposes: the transpose of a
product reverses the order, and the inverse of a transpose is the transpose
of the inverse. So $Q^{\mathsf T} = (K^{-1}R)^{\mathsf T} = R^{\mathsf T}(K^{\mathsf T})^{-1}$.
The vector needs two LU solves, $Kx = Rv$ and $K^{\mathsf T}y = v$, and then
$q = \tfrac12(x + R^{\mathsf T}y)$.

The energy is

$$ E_{\mathrm{pcm}} = \tfrac12\, q\cdot v = \tfrac12\, v^{\mathsf T} Q v . $$

Why does symmetrizing leave it unchanged? $v^{\mathsf T}Q^{\mathsf T}v$ is a
number, so it equals its own transpose $v^{\mathsf T}Qv$. Substitute into
$\tfrac12 v^{\mathsf T}\tfrac12(Q+Q^{\mathsf T})v$ and the two halves add up
to $\tfrac12 v^{\mathsf T}Qv$.

The order of the factors in the second term matters.
$(K^{\mathsf T})^{-1}R^{\mathsf T}v$ is $(RK^{-1})^{\mathsf T}v$, whose
quadratic form $v^{\mathsf T}RK^{-1}v$ differs from $v^{\mathsf T}K^{-1}Rv$:
$K$ and $R$ do not commute, because $S$ does not commute with $DA$. On
water/STO-3G at $\varepsilon = 78.4$ the reversed order shifts $E_{\mathrm{pcm}}$
by $3.2\times10^{-4}$ Ha, 5 % of it. The test
`symmetrized_energy_equals_unsymmetrized_quadratic_form` in
`ferric-pcm/src/solver.rs` pins this.

When are the symmetrized charges equal to the raw ones? Exactly when $DAS$ is
symmetric, $SAD^{\mathsf T} = DAS$. That is the discrete form of the
continuum Calderón identity $\hat S\hat D^{\dagger} = \hat D\hat S$. A Lebedev
sphere satisfies it only approximately, and a molecular cavity not at all.
So in practice $q \ne Qv$, while $E_{\mathrm{pcm}}$ is unaffected.

Why symmetrize at all, when the energy is the same? The charges also set the
reaction-field operator. The SCF needs $V^{\mathrm{pcm}} = \partial
E_{\mathrm{pcm}}/\partial P$ for the energy and the Fock matrix to describe
one functional. That holds for the symmetric response and not for the raw
$Qv$ (section 22.3, equation 22.6).

The body of `solve_pcm_charges` after its shape check, unchanged:

<!-- excerpt: ferric-pcm/src/solver.rs::solve_pcm_charges -->
```rust
let rv = r.dot(v);
let q1 = k.solve(&rv).map_err(|e| {
    FerricError::Lapack(format!(
        "IEF-PCM charge solve (K q = R v) failed — singular/ill-conditioned cavity: {e}"
    ))
})?;

let k_t = k.t().to_owned();
let kt_inv_v = k_t.solve(v).map_err(|e| {
    FerricError::Lapack(format!(
        "IEF-PCM transpose solve (Kᵀ x = v) failed — singular/ill-conditioned cavity: {e}"
    ))
})?;
let q2 = r.t().dot(&kt_inv_v);

let q = 0.5 * (&q1 + &q2);
let e_pcm = 0.5 * q.dot(v);
```

| Symbol | Code |
|---|---|
| $Rv$ | `rv` |
| $x = K^{-1}Rv$ | `q1` |
| $y = (K^{\mathsf T})^{-1}v$ | `kt_inv_v`, from `k_t.solve(v)` |
| $R^{\mathsf T}y$ | `q2` |
| $q$ | `q` |
| $E_{\mathrm{pcm}}$ | `e_pcm` |

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

On PySCF's own cavity these charges match PySCF's `pcm.py` to $10^{-17}$ and
the SCF energy to $5\times10^{-12}$ Ha (`ferric-scf/tests/validation_pcm.rs`;
the exact figures are in 22.2.6).

## 22.2.3 The sphere limits

A single sphere has an exact solution, so it tests the operators without
any other code. On a sphere the continuum operators are diagonal in the
spherical harmonics (box 22.2): $SA \to S_l = 4\pi R/(2l+1)$ and
$DA \to D_l = -2\pi/(2l+1)$ in channel $l$. Substitute into $K$ and $R$.
Since $-\tfrac{f}{2\pi}D_l = \tfrac{f}{2l+1}$,

$$ K_l = S_l\left(1 + \frac{f}{2l+1}\right), \qquad R_l = -f\left(1 + \frac{1}{2l+1}\right) . $$

Divide to get the charge density in channel $l$, $q_l = (R_l/K_l)\,v_l$.
Write it as $q_l = -k_l v_l/S_l$ and collect the scalars:

$$ k_l = \frac{2(l+1) f}{2l+1+f} = \frac{(\varepsilon-1)(l+1)}{(l+1)\varepsilon + l} . $$

The second form follows by substituting $f = (\varepsilon-1)/(\varepsilon+1)$
and multiplying the numerator and denominator by $\varepsilon+1$. It is
Kirkwood's reaction-field factor for a dielectric sphere, for every $l$.
Check it: $\varepsilon = 1$ gives $k_l = 0$, $\varepsilon\to\infty$ gives
$k_l = 1$ in every channel (the conductor), and $l\to\infty$ gives
$k_l \to f$. The last is the image-charge factor of a flat interface
between vacuum and the dielectric: a high multipole's field stays close to the surface, where the
sphere looks flat.

Now put a unit charge at distance $d$ from the centre of a sphere of radius
$R$. Expand its potential on the surface in Legendre polynomials,
$v = \sum_l (d^l/R^{l+1})P_l(\cos\theta)$. Each channel contributes
$\tfrac12\int q_l v_l\,dS = -\tfrac{k_l}{2S_l}\int v_l^2\,dS$. With
$\int P_l^2\,d\Omega = 4\pi/(2l+1)$ the integral is
$\int v_l^2\,dS = 4\pi d^{2l}/((2l+1)R^{2l})$, and dividing by $2S_l$ leaves

$$ E = -\frac{1}{2R}\sum_{l\ge0} \frac{(l+1)(\varepsilon-1)}{(l+1)\varepsilon+l}\left(\frac dR\right)^{2l} . $$

Only $l = 0$ carries net charge. At the centre, $d = 0$, only that term
survives, and it is the Born energy $-(\varepsilon-1)/(2\varepsilon R)$. The
total induced charge is $4\pi R^2$ times the $l = 0$ density,
$-k_0 = -(\varepsilon-1)/\varepsilon$, for any position inside the sphere.
That is Gauss's law: the dielectric screens the charge down to
$1/\varepsilon$ of its value. Replacing $f$ by $(\varepsilon-1)/\varepsilon$
breaks every channel (the control of box 22.4).

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

The channel algebra is exact. The discrete Lebedev sphere reproduces it only
as the grid is refined. The Rust test builds a one-atom Ne cavity (no
neighbour, so every $\mathrm{swf}_k = 1$) and runs the production chain
`build_cavity`, `build_s_d_kind`, `build_k_r`, `solve_pcm_charges` for a unit
charge at $d/R = 0, \tfrac14, \tfrac12$. The off-centre charges sit on an
off-axis direction so that no Lebedev symmetry element maps them to
themselves. They put weight on the $l > 0$ channels, and so on the
off-diagonal of $D$ and on the order of $R^{\mathsf T}$ and
$(K^{\mathsf T})^{-1}$ in the solve. The relative energy error of the
default `GaussianSmeared` $S$ and $D$:

| ε | d/R | 50 | 110 | 302 | $\lvert E_{302}-E_{110}\rvert/\lvert E\rvert$ |
|---|---|---:|---:|---:|---:|
| 78.4 | 0 | 3.1e-8 | 2.7e-8 | 1.1e-8 | 1.6e-8 |
| 78.4 | 0.25 | 2.8e-4 | 8.7e-5 | 1.9e-5 | 6.8e-5 |
| 78.4 | 0.5 | 2.8e-3 | 8.2e-4 | 1.8e-4 | 6.4e-4 |
| 4 | 0 | 5.4e-7 | 4.1e-7 | 1.7e-7 | 2.4e-7 |
| 4 | 0.25 | 2.4e-4 | 7.3e-5 | 1.6e-5 | 5.7e-5 |
| 4 | 0.5 | 2.4e-3 | 6.9e-4 | 1.5e-4 | 5.4e-4 |

The test asserts that the error falls monotonically from 50 to 110 to 302
points, and that the 302-point error is below the a-posteriori estimate in
the last column. The total charge must lie within $|Q_{302}-Q_{110}|$ of
Gauss's $-(\varepsilon-1)/\varepsilon$. The test derives all its
tolerances from these discretization estimates. `scripts/pcm_ief_sympy.py`
claim 7 does the same in numpy on a scipy Lebedev sphere.

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

What the table says: the central charge, which lives in $l = 0$ alone, is
within $5.4\times10^{-7}$ of Born at every order. The error grows with $d/R$,
because a charge nearer the surface puts weight on high $l$, and a Lebedev
grid of $N_{\mathrm{Leb}}$ points resolves only harmonics up to a finite
degree.

The `PointCharge` $S$ and $D$ on the same spheres (numpy) converge far more
slowly. The central-charge relative error is $4.5\times10^{-3}$,
$3.2\times10^{-3}$ and $2.0\times10^{-3}$ at 50, 110 and 302 points. The
cause is the self-term $1.0694\sqrt{4\pi/a_i}$ on a grid with unequal
weights. This is why `GaussianSmeared` is the default.

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
each call and keeps no factorization between iterations, so the
per-iteration cost is two dense $N^3$ factorizations.

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
first three rows isolate $K$, $R$ and the solve from the cavity
construction.

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
- The sphere tests cannot see a swap of the source and field normals,
  because $D$ is symmetric on a sphere. The molecular-cavity PySCF
  comparison is the test for that.
