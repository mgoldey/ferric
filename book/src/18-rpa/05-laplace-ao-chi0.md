# The Laplace form of $\chi_0$ {#laplace-ao-chi0}

<!-- status: ready; registry: pdep-rpa -->

The response kernel $\Delta/(u^2 + \Delta^2)$ in (@eq:pdep-1) couples every
occupied-virtual pair to the frequency through the gap
$\Delta_i^a = \varepsilon_a - \varepsilon_i$, which depends on $i$ and $a$
together. Written as an integral over imaginary time, the kernel factors
into a piece for $i$ times a piece for $a$. That separation is the first step
towards an atomic-orbital, reduced-scaling $\chi_0$, the same step Laplace
MP2 takes for its denominator (section @sec:laplace-mp2). ferric has the
molecular-orbital half of that construction, as library code. The AO half,
where the scaling would change, is not built. The backend has no row in the
validation table and is not graded: its anchors (section
@sec:laplace-ao-chi0-numerical-anchors) are test tolerances of
$5\times10^{-3}$ relative and $5\times10^{-4}$ Ha.

The plan: the Laplace form of the kernel and the dielectric built from it
(@sec:laplace-ao-chi0-laplace-form-kernel), the choice of the frequency grid for it
(@sec:laplace-ao-chi0-grid), the requirements the code enforces
(@sec:laplace-ao-chi0-requirements-code-enforces), then anchors and sources.

| Symbol | Meaning | Code name |
|---|---|---|
| $t_l, w_l$ | minimax-Laplace nodes and weights for $1/x$ on the gap range | `LaplaceQuadrature` |
| $t_{\max}$ | largest Laplace node | `t_max` |
| $n_{\mathrm L}$ | number of Laplace nodes | |
| $u$ | imaginary frequency | `omega` |
| $n$ | number of imaginary-frequency nodes (section @sec:quadratures) | `n_points` |

| Entry point | Location |
|---|---|
| Rust, Laplace $\chi_0$ | `ferric-rpa/src/laplace_chi0.rs::{dielectric_matrix_laplace,dielectric_matrix_laplace_into,build_laplace_for_gaps}`, selected by `Chi0Backend::Laplace` |
| CLI, Python | none; both always use the dense $\chi_0$ |

## The Laplace form of the kernel {#laplace-ao-chi0-laplace-form-kernel}

For $\Delta > 0$, integrate $e^{-(\Delta + iu)t}$ over $t \in [0, \infty)$:
the result is $1/(\Delta + iu)$. Its real part is
$\Delta/(u^2 + \Delta^2)$, and the real part of $e^{-iut}$ is $\cos(ut)$. So

$$ \frac{\Delta}{u^2 + \Delta^2} = \int_0^\infty e^{-\Delta t}\cos(ut)\,dt \approx \sum_l w_l\cos(ut_l)\,e^{-t_l\Delta} . \label{eq:laplace-ao-chi0-1} $$

Split each exponential in half, $e^{-t\Delta} = (e^{-t\Delta/2})^2$, and give
one half to each side of the product. The projected dielectric matrix of
(@eq:pdep-4) becomes a sum of symmetric terms, one per Laplace node:

$$ \tilde{\boldsymbol\varepsilon}_U(iu) - \mathbf 1 \approx \sum_l 4w_l\cos(ut_l)\;\mathbf X^{(l)}\mathbf X^{(l)\mathsf T}, \qquad X^{(l)}_{\xi,ia} = Y_{\xi,ia}\,e^{-t_l\Delta_i^a/2} . \label{eq:rpa-chi0-laplace} $$

**The nodes.** They are the minimax-Laplace nodes for $1/x$ on
$[\Delta_{\min}, \Delta_{\max}]$, the same tables as Laplace MP2 scaled by
$1/\Delta_{\min}$, with $n_{\mathrm L} \in \{3, 5, 7\}$. Why the scaling works: the table
gives $r^{\mathrm{tab}}(y) \approx 1/y$ on $[1, R]$. Put
$r(x) = r^{\mathrm{tab}}(x/\Delta_{\min})/\Delta_{\min}$, which is the table
with $t_l = t_l^{\mathrm{tab}}/\Delta_{\min}$ and
$w_l = w_l^{\mathrm{tab}}/\Delta_{\min}$. Then
$r(x) \approx (\Delta_{\min}/x)/\Delta_{\min} = 1/x$ on
$[\Delta_{\min}, R\,\Delta_{\min}]$, with the table's relative error. Here
$R = \Delta_{\max}/\Delta_{\min}$ and there is one gap per pair, so no
factor 2 as in Laplace MP2.

`build_laplace_for_gaps` takes the gap range from the orbital energies,

$$ \Delta_{\min} = \varepsilon_{\mathrm{LUMO}} - \varepsilon_{\mathrm{HOMO}}, \qquad \Delta_{\max} = \max_a\varepsilon_a - \min_i\varepsilon_i , $$

and `LaplaceQuadrature::new` picks the table entry with
`ferric-quadrature/src/minimax.rs::select_minimax_points`: the first entry
whose range satisfies $R_{\mathrm{tab}} \ge 0.99R$, a 1% slack. An $R$
beyond the last entry of the table is an error, as is an untabulated
$n_{\mathrm L}$.

**What the code computes.** The coded sum is
$\operatorname{Re}r(\Delta + iu)$ with $r(x) = \sum_lw_le^{-t_lx} \approx 1/x$:
the $1/x$ approximant at a complex argument, not a quadrature of the cosine
integral. Expand $e^{-t_l(\Delta + iu)} = e^{-t_l\Delta}\big(\cos(ut_l) - i\sin(ut_l)\big)$
and take the real part: the result is the sum in (@eq:laplace-ao-chi0-1). It is the
minimax approximant at $u = 0$ and is not controlled at large $u\,t_l$, where
the cosine oscillates across the nodes. So the code uses the dense kernel at
any frequency node with $u\,t_{\max} > \pi/2$. On the Laplace side every
$u\,t_l \le \pi/2$, so every coefficient $4w_l\cos(ut_l)$ is non-negative.

The core of `ferric-rpa/src/laplace_chi0.rs::dielectric_matrix_laplace_into`,
with the shape assertions and the code's comments removed, and the setup of
$\mathbf Y$, the gaps and the zeroed output replaced by `// ...`:

```rust
let t_max = laplace.points.iter().cloned().fold(0.0_f64, f64::max);
if omega * t_max > std::f64::consts::FRAC_PI_2 {             // u t_max > π/2
    let dense = crate::sternheimer::dielectric_matrix(v_mat, b_ov, eps_occ, eps_vir, omega);
    out.assign(&dense);                                      // exact Δ/(u² + Δ²)
    return;
}
// ...
for (&t_l, &w_l) in laplace.points.iter().zip(laplace.weights.iter()) {
    rhs_scaled.assign(&y_base);                              // Y
    let factor: Array1<f64> = e_ia.iter().map(|&e| (-0.5 * t_l * e).exp()).collect(); // e^{−t_l Δ/2}
    let factor_row = factor.view().insert_axis(Axis(0));
    Zip::from(&mut *rhs_scaled)
        .and_broadcast(factor_row)
        .for_each(|x, &f| *x *= f);                          // X^(l)
    let coeff = 4.0 * w_l * (omega * t_l).cos();             // 4 w_l cos(u t_l)
    let x_view = rhs_scaled.view();
    let xt = rhs_scaled.t();
    general_mat_mul(coeff, &x_view, &xt, 1.0, out);          // += coeff X Xᵀ
}
```

The identity matrix of (@eq:rpa-chi0-laplace) is added after the loop.

**Open shell** (`dielectric_matrix_laplace_unrestricted_into`, its per-spin
helper `accumulate_one_spin`). Each spin gets its own Laplace quadrature on
its own gap range, built in the open-shell driver `u_pdep_rpa_core`. The per-spin coefficient is
$2w_l\cos(ut_l)$, the pole pair alone, as in (@eq:open-shell-dielectric-1):

$$ \mathbf V^{\mathsf T}\tilde{\boldsymbol\varepsilon}^{\mathrm u}(iu)\mathbf V = \mathbf 1 + \sum_\sigma\sum_{l\in\sigma}2w_l\cos(ut_l)\,\mathbf X^{\sigma}_l\mathbf X^{\sigma\mathsf T}_l . $$

An empty $\beta$ channel (no $\beta$ occupied orbital) reuses the
$\alpha$ quadrature and contributes nothing. The fallback threshold uses the
larger $t_{\max}$ of the two spins, and both spins fall back to the dense
kernel together. With identical channels the two prefactors 2 sum to the
closed-shell 4.

**Where it is used.** In `run_pdep_rpa` the Laplace kernel enters the
frequency loop (`ferric-rpa/src/energy.rs::eval_eigenvalues_at_frequencies_laplace`,
the eigenvalues of the projected matrix at each node) and the Davidson
matrix-vector product of the static eigensolve. The default Lanczos static
eigensolve and the Boys-screened arms build $\tilde{\boldsymbol\varepsilon}(0)$
with the dense kernel. In `run_u_pdep_rpa` the Laplace kernel enters both the
static eigensolve and the frequency loop. The closed-shell inverse-dielectric
stack is built only on the dense backend.

**Where the factorization leads.** In this molecular-orbital form nothing is
saved: each frequency node does $n_{\mathrm L}$ products of the size the dense path does
once. The point of the form is
that $e^{-t\Delta_i^a} = e^{t\varepsilon_i}e^{-t\varepsilon_a}$ factorizes, so
a sum over $ia$ can be carried as two pseudo-densities
$\underline P_{\mu\nu}(t) = \sum_i C_{\mu i}e^{t\varepsilon_i}C_{\nu i}$ and
$\overline P_{\mu\nu}(t) = \sum_a C_{\mu a}e^{-t\varepsilon_a}C_{\nu a}$, which
are sparse for large molecules (Kaltak, Klimeš and Kresse, J. Chem. Theory
Comput. 10, 2498 (2014)). The formulation reaches $\mathcal O(N^3)$ only
with that AO sparsity (atom-pair cutoffs on the orbital coefficients through
local orbitals), which is not built.

> **Verified equation @eq:rpa-chi0-laplace** · the Laplace dielectric
>
> (@eq:laplace-ao-chi0-1) symbolically; the coded dielectric equals
> $\mathbf 1 + 4Y\operatorname{diag}(\operatorname{Re}r(\Delta + iu))Y^{\mathsf T}$
> to rounding, and at $u = 0$ it differs from the dense dielectric by exactly
> $4Y\operatorname{diag}(r(\Delta) - 1/\Delta)Y^{\mathsf T}$; the kernel error is
> bounded by the minimax table's on $[1, R]$; above $u\,t_{\max} = \pi/2$ the
> result is the dense dielectric, closed and open shell. The proof does not
> bound the energy error of the fallback rule; that is the Rust test below.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_laplace_chi0_sympy.py`, claims L1 (identity), L2 (gap scaling), L3 (dielectric), L4 (kernel error), L5 (fallback), L6 (open-shell prefactor) |
> | Negative control | the runner halves the coefficient $4w_l$, drops the $\tfrac12$ of the exponent split, scales the nodes by $\Delta_{\min}$ instead of $1/\Delta_{\min}$, moves the threshold to $\pi$, and doubles the open-shell prefactor |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/laplace_chi0.rs::{dielectric_matrix_laplace_into,dielectric_matrix_laplace_unrestricted_into,build_laplace_for_gaps}` |
> | Code test | `h2o_ccpvdz_laplace_matches_dense_at_static` in `crates/ferric-rpa/tests/laplace_chi0.rs` (elementwise, relative $5\times10^{-3}$); `closed_shell_rpa_laplace_chebyshev_matches_dense_h2o` in `crates/ferric-rpa/tests/u_pdep_rpa.rs` ($5\times10^{-4}$ Ha) |
>
> <!-- verified: id=rpa.chi0.laplace; script=scripts/rpa_laplace_chi0_sympy.py; claim=L3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/laplace_chi0.rs::{dielectric_matrix_laplace_into,dielectric_matrix_laplace_unrestricted_into,build_laplace_for_gaps}; rust_test=ferric-rpa/tests/laplace_chi0.rs::h2o_ccpvdz_laplace_matches_dense_at_static; registry=pdep-rpa -->

## Choosing the frequency grid {#laplace-ao-chi0-grid}

The backend falls back to the dense kernel at any frequency node with
$u\,t_{\max} > \pi/2$, so the imaginary-frequency grid of section
@sec:quadratures decides how many nodes take the Laplace path. Only
$u_0 = 0.5$ has been calibrated on the dense path. A smaller $u_0$ moves nodes to smaller $u$,
so it keeps more of them on the Laplace path.
`CHEBYSHEV_TAN_RECOMMENDED_U0 = 0.2` is calibrated for that backend only, and
costs a coarser high-frequency tail.

At equal $(n, u_0)$, Chebyshev–Tan and Gauss–Legendre send about the same
number of nodes to the fallback, and which sends more depends on $t_{\max}$.
At $n = 20$:

| $t_{\max}$ | $u_0$ | Nodes on the Laplace path, Chebyshev–Tan | Gauss–Legendre |
|---|---|---|---|
| $\approx 1.5$ (H2O / cc-pVDZ) | 0.2 | 16 | 15 |
| $\approx 1.5$ (H2O / cc-pVDZ) | 0.5 | 13 | 12 |

At $t_{\max} = 2\pi$ the order reverses: Chebyshev–Tan sends one node more to
the fallback, 13 against 12 at $u_0 = 0.5$. Lowering $u_0$ moves more nodes
onto the Laplace path than the choice of scheme does. The CLI and the Python
bindings always use `Chi0Backend::Dense`; `Chi0Backend::Laplace`
(`crates/ferric-rpa/src/laplace_chi0.rs`) is a library option.

## Requirements the code enforces {#laplace-ao-chi0-requirements-code-enforces}

| Requirement | Why | Where |
|---|---|---|
| canonical orbital energies | $\Delta_i^a = \varepsilon_a - \varepsilon_i$ is a per-pair scalar only in a Fock-diagonal basis | debug assertion in `build_e_ia` (`crates/ferric-rpa/src/laplace_chi0.rs`) |
| $n_{\mathrm L} \in \{3, 5, 7\}$ | only those minimax tables exist; other sizes are an error, not a silent fallback | `LaplaceQuadrature::new` |
| no aux-basis route | the energy-only route of section @sec:drpa-three-ways has no Laplace evaluator, so the Laplace backend always runs the static solve | `static_route_required` |

## Numerical anchors {#laplace-ao-chi0-numerical-anchors}

| Quantity | Value | Source |
|---|---|---|
| Laplace vs dense $\tilde{\boldsymbol\varepsilon}(0)$, H2O / cc-pVDZ | relative $< 5\times10^{-3}$ elementwise | `crates/ferric-rpa/tests/laplace_chi0.rs` |
| RPA energy, Chebyshev–Tan + Laplace vs Gauss–Legendre + dense, H2O / cc-pVDZ | $< 5\times10^{-4}$ Ha | `crates/ferric-rpa/tests/u_pdep_rpa.rs` |

These are tolerances of tests, not graded accuracy; the backend has no
validation row.

## Sources {#laplace-ao-chi0-sources}

| Source | What it gives |
|---|---|
| `scripts/rpa_laplace_chi0_sympy.py`, claims L1 to L6 | the proof |
| section @sec:laplace-mp2-quadrature | the node tables |
| module documentation of `crates/ferric-rpa/src/laplace_chi0.rs` | the Laplace form and its scope |
| Kaltak, Klimeš and Kresse, J. Chem. Theory Comput. 10, 2498 (2014) | imaginary-time and Laplace transformations for low-scaling RPA |

**Known limits (2026-11-15).** Library only. No reduced-scaling claim: the
Laplace form is dense. `ferric-rpa/src/ao_rpa.rs::ao_rpa_correlation_energy`
builds the occupied and virtual pseudo-densities and an imaginary-time
$\chi_0$ from them, but contracts them densely; no test checks its energy
against the dense route, so this page does not derive it.
