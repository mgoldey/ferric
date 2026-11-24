# Second-order steps: Newton, augmented Hessian, trust region {#second-order}

<!-- status: ready; registry: rhf, uhf, rohf -->

DIIS sees only errors, which are first derivatives of the energy. Near a
point where the energy is flat along some rotation, or curves downward
along it, DIIS converges slowly or settles on a saddle point that satisfies the SCF
equations. The orbital Hessian, the second derivative of the energy with
respect to orbital rotations, tells the solver how far to go and which way
is downhill. All three methods here solve a shifted Newton equation
$(\mathbf H - \mu)\kappa = -g$ and differ in how they choose $\mu$; each
Hessian product $\mathbf H\kappa$ costs one $J$/$K$ build.

The plan: derive the gradient and the Hessian product for each reference,
show the rotation that applies a step, then derive the three steps from the
quadratic model of the energy.

| Symbol | Meaning | Code name |
|---|---|---|
| $\kappa_{ai}$ | occupied–virtual rotation, $C \leftarrow CU(\kappa)$ | `k`, `kappa` |
| $U(\kappa)$ | Cayley unitary $(1 - \kappa/2)^{-1}(1 + \kappa/2)$ | `cayley_unitary` |
| $g$ | the code's gradient, $f_{ai}$: a quarter of $\partial E/\partial\kappa_{ai}$ (RHF), half (UHF), not the $g$ of the notation page | `g` |
| $\mathbf H\kappa$ | the code's Hessian product | `hessian_matvec_jk` (RHF), `hessian_matvec` (UHF) |
| $\Delta_{\rm tr}$ | trust radius | the TRAH radius |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/rhf_newton.rs::{rhf_newton_step,hessian_matvec_jk}`, `ferric-scf/src/uhf_newton.rs::{uhf_newton_step,hessian_matvec}`, `ferric-scf/src/rohf_newton.rs::rohf_newton_step`, `ferric-scf/src/rohf_ah.rs::rohf_ah_step`, `ferric-scf/src/trah.rs::solve_trust_region`, `ferric-scf/src/cayley.rs::cayley_unitary` |
| CLI | `[scf] soscf = true` (sets `newton_trigger` to $10^{-3}$), `[scf] trah_trigger` |
| Python | `soscf=`, `trah_trigger=` on the SCF entry points |
| Rust only | `RhfConfig::newton_trigger` at other values, `RhfConfig::ah_trigger` (ROHF) |

## The orbital gradient and Hessian {#second-order-orbital-gradient-hessian}

### The quadratic model

Rotate the orbitals by $t\kappa$ and expand the energy to second order in $t$:

$$ E(t) = E(0) + t\,\frac{dE}{dt} + \frac{t^2}{2}\frac{d^2E}{dt^2} + O(t^3) . $$

Each method below writes the first derivative as $g^{\mathsf T}\kappa$ and the
second as $\kappa^{\mathsf T}\mathbf H\kappa$, up to a constant factor, and works
with the model $m(\kappa) = g^{\mathsf T}\kappa + \tfrac12\kappa^{\mathsf T}\mathbf H\kappa$.
Its stationary point solves $\mathbf H\kappa = -g$, the Newton step.

The code stores a fixed fraction of the true derivatives: a quarter for
RHF, a half for UHF. Scaling $g$ and $\mathbf H$ by the same factor leaves
$\kappa = -\mathbf H^{-1}g$ unchanged, so the Newton step does not see it. A
predicted energy change does, and so does a threshold on an eigenvalue
(section @sec:stability).

### The rotation

$C \leftarrow CU(\kappa)$ with $\kappa$ antisymmetric, stored as $\kappa_{ai}$,
and the Cayley unitary $U = (1 - \kappa/2)^{-1}(1 + \kappa/2)$
(`cayley_unitary`). Two properties make it a valid parametrization.

It is orthogonal. Transpose $U$ and use $\kappa^{\mathsf T} = -\kappa$:
$U^{\mathsf T} = (1 - \kappa/2)(1 + \kappa/2)^{-1}$. The factors are
polynomials in the same matrix $\kappa$, so they commute, and
$U^{\mathsf T}U = (1 + \kappa/2)^{-1}(1 - \kappa/2)(1 - \kappa/2)^{-1}(1 + \kappa/2) = 1$.

It equals $e^{\kappa}$ through second order. Expand the inverse as a
geometric series,
$(1 - t\kappa/2)^{-1} = 1 + t\kappa/2 + t^2\kappa^2/4 + t^3\kappa^3/8 + \dots$,
and multiply by $(1 + t\kappa/2)$:

$$ U(t\kappa) = 1 + t\kappa + \frac{t^2\kappa^2}{2} + \frac{t^3\kappa^3}{4} + O(t^4), \qquad e^{t\kappa} = 1 + t\kappa + \frac{t^2\kappa^2}{2} + \frac{t^3\kappa^3}{6} + O(t^4) . $$

The two first differ at $t^3$. So the first and second derivatives of
$E(CU(t\kappa))$ at $t = 0$ are those of the exponential map, and a converged
Newton iteration does not depend on which map is used.

### RHF

$D = 2C_oC_o^{\mathsf T}$ moves by $\delta D = 2C(\kappa + \kappa^{\mathsf T})C^{\mathsf T}$
at first order, where $\kappa + \kappa^{\mathsf T}$ here means the symmetric
matrix with $\kappa_{ai}$ in both corners (section @sec:diis-error-vector-extrapolation derives this). The
derivatives of $E(CU(t\kappa))$ are

$$\frac{dE}{dt} = 4\sum_{ai}f_{ai}\kappa_{ai},\qquad
\frac{d^2E}{dt^2} = 4\,\kappa\cdot(\mathbf H\kappa),$$

$$(\mathbf H\kappa)_{ai} = \sum_b f_{ab}\kappa_{bi} - \sum_j\kappa_{aj}f_{ji}
+ \sum_{bj}\big[4(ai|bj) - c_x\big((ab|ij) + (aj|bi)\big)\big]\kappa_{bj}
\quad(+\,\text{the } f_{xc} \text{ term for KS}).$$

The first two terms are the one-electron part: the Fock operator acting on
the rotated occupied and virtual orbitals. The bracket is the response of
the Fock matrix to the density change. The code builds it in the AO basis:
form $\delta D$, build $\delta F = J[\delta D] - \tfrac12c_xK[\delta D]$, and
take the virtual–occupied block of $C^{\mathsf T}\delta FC$. Contract the AO
integrals with $\delta D$ to see that the two forms agree:

$$ J[\delta D]_{ai} = \sum_{bj}(ai|bj)\,(2\kappa_{bj} + 2\kappa_{bj}) = 4\sum_{bj}(ai|bj)\kappa_{bj}, \qquad \tfrac12K[\delta D]_{ai} = \sum_{bj}\big[(ab|ij) + (aj|bi)\big]\kappa_{bj} . $$

The two corners of $\delta D$ give the two terms in each. The code's
gradient is $g = f_{ai}$ and its product is this $\mathbf H\kappa$: a quarter
of the true derivatives.

For the one-electron part the code adds only the diagonal gap
$(f_{aa} - f_{ii})\kappa_{ai}$. That equals $f_{vv}\kappa - \kappa f_{oo}$ only when
$f_{oo}$ and $f_{vv}$ are diagonal, that is at canonical orbitals.
`offdiag_gap_correction` (in `crates/ferric-scf/src/trah.rs`) adds
$f_{vv}\kappa - \kappa f_{oo} - (f_{aa}-f_{ii})\kappa$ for the general case. At
canonical orbitals $\mathbf H$ is the singlet $\mathbf A + \mathbf B$ of linear
response.

### UHF

Each spin has its own rotation. With
$\delta D^\sigma = C^\sigma(\kappa^\sigma + \kappa^{\sigma\mathsf T})C^{\sigma\mathsf T}$
(single occupation, so no factor 2) and
$\delta F^\sigma = J[\delta D^\alpha + \delta D^\beta] - c_xK[\delta D^\sigma]$,

$$\frac{dE}{dt} = 2\sum_\sigma\sum_{ai}f^\sigma_{ai}\kappa^\sigma_{ai},\qquad\frac{d^2E}{dt^2} = 2\sum_\sigma\kappa^\sigma\cdot(\mathbf H\kappa)^\sigma,$$

$$(\mathbf H\kappa)^\sigma_{ai} = (f^\sigma_{aa} - f^\sigma_{ii})\kappa^\sigma_{ai} + \sum_{\tau,bj}\big[2(a_\sigma i_\sigma|b_\tau j_\tau) - \delta_{\sigma\tau}c_x\big((ab|ij)+(aj|bi)\big)\big]\kappa^\tau_{bj} .$$

The Coulomb term couples the spins because $J$ sees the total density;
exchange acts within one spin. This is `hessian_matvec` in
`crates/ferric-scf/src/uhf_newton.rs`: half the true derivatives, exact at
canonical orbitals.

### ROHF

ROHF has one orbital set with occupations $n_\alpha$, $n_\beta$ and three
rotation blocks: virtual–closed $vc$, virtual–open $vo$ and open–closed
$oc$. Let $X$ be the unpacked antisymmetric generator, and define per spin
$X_\sigma = XN_\sigma - N_\sigma X$, so $(X_\sigma)_{pq} = X_{pq}(n_{\sigma q} - n_{\sigma p})$:
the rotation weighted by how much it changes that spin's occupation. With
$\delta D^\sigma = CX_\sigma C^{\mathsf T}$,
$\delta f^\sigma = C^{\mathsf T}(J[\delta D^\alpha + \delta D^\beta] - c_xK[\delta D^\sigma])C$,
$A_\sigma = f^\sigma X - Xf^\sigma$ and $\Delta n^\sigma_{pq} = n_{\sigma p} - n_{\sigma q}$,

$$G = \sum_\sigma\Big[\tfrac12\Delta n^\sigma\circ A_\sigma + \tfrac12(X_\sigma f^\sigma - f^\sigma X_\sigma) + \Delta n^\sigma\circ\delta f^\sigma\Big],\qquad (\mathbf H x)_{pq} = G_{qp} - G_{pq}$$

is the exact Hessian product of $E(CU(x))$ at any orbitals, not only at a
stationary point (`ferric-scf/src/rohf_trah.rs::rohf_exact_hessian_matvec`; the
gradient is $2g$ with the blocks of the ROHF energy functional,
`rohf_exact_gradient`); `rohf_trah` is the second-order solver of the
periodic injected ROHF path (chapter @sec:periodic). The Newton and AH steps
of this section use the product in `crates/ferric-scf/src/rohf_newton.rs`,
which is half of it.

> **Verified equation {#eq:scf-orbital-hessian}** · orbital gradients and Hessian products
>
> For RHF, UHF and ROHF the code's gradient and Hessian product equal
> fourth-order central differences of the energy $E(CU(t\kappa))$, at the
> scales stated in the included derivation: a quarter of the RHF derivatives,
> half of the UHF ones, the exact ROHF ones from `rohf_trah`. The RHF product
> is exact only at canonical orbitals unless the off-diagonal gap correction
> is added.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_orbital_hessian_sympy.py`, claims C1, H1 to H6 |
> | Negative control | the response density factor 2 → 1 or the exchange weight $-\tfrac12 \to -1$ must fail H2; $J[\delta D^\alpha + \delta D^\beta] \to J[\delta D^\sigma]$ must fail H4; the ROHF $\tfrac12 b \to b$ must fail H5 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/rhf_newton.rs::hessian_matvec_jk`, `ferric-scf/src/uhf_newton.rs::hessian_matvec`, `ferric-scf/src/rohf_trah.rs::rohf_exact_hessian_matvec` |
>
> <!-- verified: id=scf.orbital-hessian; script=scripts/scf_orbital_hessian_sympy.py; claim=H2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/rhf_newton.rs::hessian_matvec_jk, ferric-scf/src/uhf_newton.rs::hessian_matvec, ferric-scf/src/rohf_trah.rs::rohf_exact_hessian_matvec; registry=rhf -->

### The RHF product in code

`ferric-scf/src/rhf_newton.rs::hessian_matvec_jk`, with the $f_{xc}$ branch
replaced by a comment and the comments replaced by the terms:

```rust
let dd_single_ao = ao_from_ov(inp.c, k, no, n);          // C (k + k^T) C^T
let dd_ao = 2.0 * &dd_single_ao;                         // delta D = 2 C (k + k^T) C^T
let (dj, dk) = response_jk(ctx, inp, k, pool, jk, &dd_ao)?; // J[dD], K[dD]
let mut df: Array2<f64> = match (inp.rsh, dk) {
    (Some(rsh), _) => &dj - &(0.5 * &rsh.exchange_response(&dd_ao)?), // range-separated K
    (None, Some(dk)) => &dj - &(0.5 * inp.k_mix_sr * &dk),          // J - c_x K / 2
    (None, None) => dj,                                              // no exact exchange
};
// ... for KS, + f_xc response of the per-spin delta D ...
let df_mo = inp.c.t().dot(&df).dot(inp.c);              // C^T dF C
let mut h = occ_virt_block(&df_mo, no, n);               // (ai) block
let f_diag: Vec<f64> = (0..n).map(|i| inp.f_mo[(i, i)]).collect();
for (ir, a) in (no..n).enumerate() {
    for (ic, i) in (0..no).enumerate() {
        h[(ir, ic)] += (f_diag[a] - f_diag[i]) * k[(ir, ic)]; // (f_aa - f_ii) k_ai
    }
}
```

Three points about it:

- **One build per product.** `response_jk` builds $\delta J$ and $\delta K$
  with the builders the SCF lends it (density-fitted where the SCF fits,
  direct otherwise), so the response matches the Fock matrix the SCF
  converged.
- **The KS response is per spin.** For a restricted point
  $\delta D_\alpha = \delta D_\beta = \tfrac12\delta D$, so the $f_{xc}$ closure
  receives `dd_single_ao`, and one spin's output is added.
- **The gap is diagonal.** The last loop is the canonical-orbital form of
  the one-electron part, as discussed above.

## The rotation in code {#second-order-rotation}

The Cayley unitary needs one matrix inverse. The body of
`ferric-scf/src/cayley.rs::cayley_unitary`, shown whole:

```rust
let mut a = Array2::<f64>::eye(n);
a.scaled_add(-0.5, kappa);                                // 1 - k/2
let mut u = with_blas_threads(opt_in_blas_threads(), || a.inv())
    .map_err(|e| FerricError::Lapack(format!("Cayley inverse: {e}")))?; // (1 - k/2)^-1
u *= 2.0;
for i in 0..n {
    u[(i, i)] -= 1.0;                                     // 2 (1 - k/2)^-1 - 1
}
```

Why does the last line give $U$? Factor out the inverse:
$2(1 - \kappa/2)^{-1} - 1 = (1 - \kappa/2)^{-1}\big(2 - (1 - \kappa/2)\big) = (1 - \kappa/2)^{-1}(1 + \kappa/2)$.
One inverse and a diagonal update replace an inverse and a product.
$1 - \kappa/2$ is never singular for antisymmetric $\kappa$: the eigenvalues of
$\kappa$ are imaginary, so those of $1 - \kappa/2$ have real part 1. The
exponential map would need an eigendecomposition; the Cayley map needs one
matrix inverse and is exactly orthogonal.

## The three steps {#second-order-three-steps}

### Newton

`rhf_newton_step`, `uhf_newton_step` and `rohf_newton_step` solve
$\mathbf H\kappa = -g$ by preconditioned conjugate gradient (PCG). $\mathbf H$ is
symmetric and positive definite near a minimum, and PCG needs only
products $\mathbf H\kappa$; forming $\mathbf H$ itself would cost one $J$/$K$
build per rotation. The diagonal
gap plus a shift $\mu$, $\Delta + \mu$, is the preconditioner, and the start
is the diagonal solve $\kappa^0 = -g/(\Delta + \mu)$, the level-shifted
Roothaan step of section @sec:level-shift-mom-smearing-level-shift-damps-step. The core of
`ferric-scf/src/rhf_newton.rs::rhf_newton_step`, with the engine-pool setup
removed:

```rust
let g = occ_virt_block(inp.f_mo, no, n);                 // g_ai = f_ai
let f_diag: Vec<f64> = (0..n).map(|i| inp.f_mo[(i, i)]).collect();
let diag = build_gap(&f_diag, no, n, level_shift);       // Delta + mu, floored at 1e-6
let mut k = neg_div(&g, &diag);                          // k0 = -g / (Delta + mu)
let gmax = max_abs(&g);
if gmax >= cg_conv {
    pcg_refine(ctx, inp, jk, &pool, &g, &diag, &mut k, cg_max_iter, cg_conv)?; // H k = -g
}
let kmax = max_abs(&k);
if kmax > max_step {
    let s = max_step / kmax;
    k.mapv_inplace(|x| x * s);                           // clip max |k| to max_step
}
let c_new = crate::cayley::apply_cayley_ov(inp.c, &k, no, n)?; // C U(k)
```

Points about it:

- **The shift enters the preconditioner only.** PCG's residual is
  $-g - \mathbf H\kappa$, so a converged solve is the unshifted Newton step;
  $\mu$ changes the start and the convergence rate. The SCF passes
  $\mu = \max(\texttt{level\_shift}, 10^{-6})$.
- **The limits.** At most 20 PCG iterations; the solve stops when
  $\max\lvert r\rvert < 10^{-7}$ (`cg_conv`; ROHF uses
  $\min(\texttt{density\_conv}, 10^{-7})$). If the gradient is already below
  that, the step is $\kappa^0$ and no product is built.
- **The clip.** The whole step is scaled so $\max\lvert\kappa\rvert \le 0.2$
  (0.1 for ROHF, `ROHF_NEWTON_MAX_STEP`). This bounds the step without
  changing its direction, and it is not a trust region: no ratio of actual
  to predicted change is formed, and an uphill step is kept.
- **Negative curvature.** ROHF truncates PCG at negative curvature
  (Steihaug): along a direction with $p^{\mathsf T}\mathbf Hp < 0$ the CG step
  would move to the maximum of the model there, so it steps along $+p$ to
  the clip instead. RHF and UHF do not truncate.
- **When it runs.** RHF takes a Newton step once `iter > 3`, when
  $\max\lvert e\rvert <$ `newton_trigger` and the functional has no range
  separation ($\omega = 0$). The range-separated arm of the product above
  serves TRAH and the stability analysis; the RHF Newton step itself is
  gated to $\omega = 0$.

On water / cc-pVDZ (the `589b50832` build) all three routes reach the same
energy. Newton steps (`soscf=True`) save two iterations over DIIS alone, and
TRAH engaged at $\max\lvert e\rvert < 10^{-2}$ saves three:

```python
# Run from the repository root. DIIS alone, DIIS with Newton steps (soscf), and
# DIIS with TRAH engaged below max|e| = 1e-2, on water / cc-pVDZ.
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")

runs = {
    "DIIS": {},
    "soscf=True": {"soscf": True},
    "trah_trigger=1e-2": {"trah_trigger": 1e-2},
}
for label, kw in runs.items():
    r = ferric.run_rhf(mol, bs, max_iter=200, **kw)
    print(f"{label:18s} E = {r.energy:.10f}  converged={r.converged}  iterations={r.iterations}")
```

```text
DIIS               E = -76.0267679974  converged=True  iterations=10
soscf=True         E = -76.0267679974  converged=True  iterations=8
trah_trigger=1e-2  E = -76.0267679974  converged=True  iterations=7
```

### Augmented Hessian

Newton fails where $\mathbf H$ has a negative eigenvalue: $\kappa = -\mathbf H^{-1}g$
then goes uphill along that mode. The augmented Hessian adds a shift chosen
so that $\mathbf H - \mu$ is never indefinite. Take the lowest eigenpair
$(\mu, (v_0, v))$ of

$$\begin{pmatrix}0 & \alpha g^{\mathsf T}\\ \alpha g & \mathbf H\end{pmatrix}$$

and set $\kappa = v/(\alpha v_0)$. The lower rows read
$\mathbf Hv + \alpha gv_0 = \mu v$. Divide by $\alpha v_0$:

$$(\mathbf H - \mu)\kappa = -g .$$

This is a Newton step with the level shift $-\mu$. The top row reads
$\alpha g^{\mathsf T}v = \mu v_0$; divide by $\alpha^2v_0$ to get
$g^{\mathsf T}\kappa = \mu/\alpha^2$.

Why is $-\mu \ge 0$, and why is the step downhill? $\mathbf H$ is a principal
submatrix of the augmented matrix, so by Cauchy interlacing the lowest
eigenvalue of the augmented matrix is at most the lowest eigenvalue of
$\mathbf H$: $\mu \le \lambda_{\min}(\mathbf H)$. The Rayleigh quotient of the
first unit vector is the corner element 0, so also $\mu \le 0$. Then
$\mathbf H - \mu$ is positive semidefinite whatever the sign of
$\lambda_{\min}$, and $g^{\mathsf T}\kappa = \mu/\alpha^2 \le 0$: the step points
downhill.

The model predicts the change $g^{\mathsf T}\kappa + \tfrac12\kappa^{\mathsf T}\mathbf H\kappa$.
Substitute $\mathbf H\kappa = \mu\kappa - g$ in the second term:
$\kappa^{\mathsf T}\mathbf H\kappa = \mu\lvert\kappa\rvert^2 - g^{\mathsf T}\kappa$, and

$$ g^{\mathsf T}\kappa + \tfrac12\kappa^{\mathsf T}\mathbf H\kappa = \tfrac12\big(g^{\mathsf T}\kappa + \mu\lvert\kappa\rvert^2\big) , $$

which needs no further Hessian product.

Check the limit. If $\mathbf H$ is positive definite and $g$ is small, $\mu$
is second order in $g$ (to leading order $\mu = -\alpha^2g^{\mathsf T}\mathbf H^{-1}g$),
and $\kappa$ tends to the Newton step $-\mathbf H^{-1}g$. Near a minimum the
augmented Hessian becomes Newton.

> **Verified equation {#eq:scf-ah-step}** · the augmented-Hessian step
>
> The lowest eigenpair of the augmented matrix gives
> $(\mathbf H - \mu)\kappa = -g$ and the predicted change
> $\tfrac12(g^{\mathsf T}\kappa + \mu\lvert\kappa\rvert^2)$.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_orbital_hessian_sympy.py`, claim T1 |
> | Negative control | the sign of $\mu$ in the predicted change flipped must fail T1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/trah.rs::solve_trust_region`, `ferric-scf/src/rohf_ah.rs::rohf_ah_step` |
>
> <!-- verified: id=scf.ah-step; script=scripts/scf_orbital_hessian_sympy.py; claim=T1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/trah.rs::solve_trust_region, ferric-scf/src/rohf_ah.rs::rohf_ah_step; registry=rhf -->

Claim T1 covers the step equation and the predicted change. The bound
$\mu \le \lambda_{\min}$, the sign of $g^{\mathsf T}\kappa$ and the small-$g$
limit argued above are not part of any claim.

ROHF's `rohf_ah_step` solves the eigenproblem at $\alpha = 1$ with block
Davidson and then scales the step so $\max\lvert\kappa\rvert \le 0.2$. A
clipped step is a shortened AH step, which differs in direction from the
solution of the shifted equations at a smaller length. If the Davidson
solve does not converge, it takes the PCG Newton step instead.

### Trust region (TRAH)

The trust-region augmented Hessian (`solve_trust_region`, RHF and UHF) uses
$\alpha$ to control the length. A larger $\alpha$ makes $\mu$ more negative,
which is a harder shift and a shorter step: $\lVert\kappa(\alpha)\rVert$
decreases monotonically in $\alpha$ (the test
`larger_alpha_shortens_the_step_monotonically` in `crates/ferric-scf/src/trah.rs`
pins it). The search tries $\alpha = 1$ first. If $\lVert\kappa\rVert_2$ is
already inside the radius $\Delta_{\rm tr}$, that is the step. Otherwise it
grows $\alpha$ by factors of 8 until the step fits, then bisects in
$\log\alpha$, within $1 \le \alpha \le 1000$, until $\lVert\kappa\rVert_2$
lies within 5% below the radius. A step still too long at $\alpha = 1000$ is
scaled to the radius.

The step records both forms of the predicted change; their difference
checks the Davidson solve:

```rust
let hk = matvec(&step.kappa)?;                            // H k
let gk: f64 = g.iter().zip(step.kappa.iter()).map(|(a, b)| a * b).sum();   // g.k
let khk: f64 = step.kappa.iter().zip(hk.iter()).map(|(a, b)| a * b).sum(); // k.Hk
step.predicted = gk + 0.5 * khk;                          // g.k + k.Hk / 2
step.predicted_cheap = 0.5 * (gk + step.level_shift * step.norm * step.norm); // (g.k + mu |k|^2) / 2
let scale = step.predicted.abs().max(1e-300);
step.predicted_residual = (step.predicted - step.predicted_cheap).abs() / scale;
```

The two agree for an exact eigenpair, so a large `predicted_residual` means
Davidson did not converge it. The RHF and UHF wrappers multiply the
prediction by 4 and 2 (`RHF_ENERGY_SCALE`, `UHF_ENERGY_SCALE`) to turn it
into a true energy change. After the next Fock build the ratio
$\rho = \Delta E/\Delta E_{\rm pred}$ decides the radius:

| $\rho = \Delta E/\Delta E_{\rm pred}$ | Step | Radius |
|---|---|---|
| $\le 0$ | rejected | $\times 0.7$ |
| $(0, 0.25]$ | accepted | $\times 0.7$ |
| $(0.25, 0.75]$ | accepted | unchanged |
| $> 0.75$ | accepted | $\times 1.2$, at most 2.0 |

The radius starts at 0.4. Below $10^{-4}$ the region has collapsed and DIIS
takes over.

One case breaks the division by $v_0$. If the lowest eigenvector $v$ of
$\mathbf H$ is orthogonal to $g$, then $(0, v)$ is an eigenvector of the
augmented matrix with eigenvalue $\lambda_{\min}$. It is the lowest root when
$\lambda_{\min}$ lies below the lowest root of the problem with $v$ projected
out, which requires $\lambda_{\min} < 0$. In an SCF this typically happens
when the instability breaks a symmetry the density still has: the gradient
of a symmetric density is totally symmetric, and a symmetry-breaking mode is
not, so $g\cdot v = 0$. This
hard case is deflated, and the step is completed along the negative mode to
the boundary.

### The three side by side

| Method | $\mu$ chosen by | Step limit | Solver | Who has it |
|---|---|---|---|---|
| Newton | 0 (PCG on $\mathbf H\kappa = -g$), ROHF truncates at negative curvature | $\max\lvert\kappa\rvert \le 0.2$ (0.1 ROHF), at most 20 PCG iterations | `rhf_newton_step`, `uhf_newton_step`, `rohf_newton_step` | RHF, UHF, ROHF via `newton_trigger` (`soscf`) |
| augmented Hessian | the lowest eigenvalue of the augmented matrix, $\alpha = 1$ | $\max\lvert\kappa\rvert \le 0.2$, a clip after the eigenproblem | block Davidson | ROHF via `ah_trigger` (library only) |
| TRAH | the augmented eigenproblem, $\alpha$ adjusted until $\lvert\kappa\rvert$ fits the radius | $\Delta_{\rm tr}$, from 0.4, at most 2.0 | `solve_trust_region` | RHF, UHF via `trah_trigger` |

TRAH takes precedence over `newton_trigger` where both would fire. TRAH is
graded Smoke in the validation matrix: its convergence is measured on test
systems, and it has no reference-code comparison.

## Numerical anchors {#second-order-numerical-anchors}

The second-order solvers are tested by the energies they converge to and by
their smoke and anchor tests in `crates/ferric-scf/tests/`
(`rhf_newton_smoke.rs`, `uhf_newton_smoke.rs`, `rohf_newton_smoke.rs`,
`trah_converges.rs`, `trah_hard_case.rs`). The Hessian products themselves are
checked against finite differences of the energy by the proof above, which
is the anchor that a scale or sign error in the product would fail.

## Sources

| Source | Used for |
|---|---|
| `scripts/scf_orbital_hessian_sympy.py` | verified equations @eq:scf-orbital-hessian and @eq:scf-ah-step |
| `crates/ferric-scf/src/trah.rs` | the radius table and the $\alpha$ search |
| T. Steihaug, SIAM J. Numer. Anal. 20, 626 (1983) | truncated CG at negative curvature |
| B. Helmich-Paris, J. Chem. Phys. 154, 164104 (2021) | TRAH |
