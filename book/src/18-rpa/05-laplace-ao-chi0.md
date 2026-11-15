# 18.5 Imaginary-time and AO forms of $\chi_0$

<!-- section: 18.5; status: gap; registry: none;
     theory: crates/ferric-quadrature/theory.md#"1. Minimax Laplace Quadrature", "2. Imaginary-Frequency Grids for RPA" -->

The response kernel $\Delta/(u^2 + \Delta^2)$ in (18.7) couples every
occupied-virtual pair to the frequency. Writing it as an integral over
imaginary time separates the occupied and virtual indices, which is the
first step towards an atomic-orbital, reduced-scaling $\chi_0$. ferric has
the molecular-orbital half of that construction, as library code. The AO
half, where the scaling would change, is not built. This section states what
exists and what does not; it has no derivation in a `theory.md` file yet.

| Symbol | Meaning | Code name |
|---|---|---|
| $t_l, w_l$ | minimax-Laplace nodes and weights for $1/x$ on the gap range | `LaplaceQuadrature` |
| $t_{\max}$ | largest Laplace node | `t_max` |
| $u$ | imaginary frequency | `omega` |

| Entry point | Location |
|---|---|
| Rust, Laplace $\chi_0$ | `ferric-rpa/src/laplace_chi0.rs::{dielectric_matrix_laplace,dielectric_matrix_laplace_into,build_laplace_for_gaps}`, selected by `Chi0Backend::Laplace` |
| Rust, imaginary time | `ferric-rpa/src/ao_rpa.rs::{ao_rpa_correlation_energy,build_tau_quadrature,pseudo_density_occ,pseudo_density_vir}` |
| CLI, Python | none; both always use the dense $\chi_0$ |

## 18.5.1 The Laplace form of the kernel

For $\Delta > 0$,

$$ \frac{\Delta}{u^2 + \Delta^2} = \int_0^\infty e^{-\Delta t}\cos(ut)\,dt \approx \sum_l w_l\cos(ut_l)\,e^{-t_l\Delta}, \tag{18.14} $$

so the projected dielectric matrix of (18.10) becomes

$$ \tilde{\boldsymbol\varepsilon}_U(iu) - \mathbf 1 \approx \sum_l 4w_l\cos(ut_l)\;\mathbf X^{(l)}\mathbf X^{(l)\mathsf T}, \qquad X^{(l)}_{\xi,ia} = Y_{\xi,ia}\,e^{-t_l\Delta_i^a/2} . \tag{18.15} $$

The nodes are the minimax-Laplace nodes for $1/x$ on
$[\Delta_{\min}, \Delta_{\max}]$, the same tables as Laplace MP2, with
$n \in \{3, 5, 7\}$. They are optimal for $1/x$ at $u = 0$ and not for the
oscillating integrand at $u > 0$, so the code uses the dense kernel at any
frequency node with $u\,t_{\max} > \pi/2$. The coefficient
$4w_l\cos(ut_l)$ can be negative, so each term is a general matrix product,
not a symmetric rank update.

In this molecular-orbital form the cost equals the dense path's. The point
of the form is that $e^{-t\Delta_i^a} = e^{t\varepsilon_i}e^{-t\varepsilon_a}$
factorizes, so a sum over $ia$ can be carried as two pseudo-densities
$\underline P_{\mu\nu}(t) = \sum_i C_{\mu i}e^{t\varepsilon_i}C_{\nu i}$ and
$\overline P_{\mu\nu}(t) = \sum_a C_{\mu a}e^{-t\varepsilon_a}C_{\nu a}$, which
are sparse for large molecules. `ao_rpa.rs` builds those pseudo-densities and
an imaginary-time $\chi_0$ from them (after Kaltak, Klimeš and Kresse 2014),
but evaluates the contraction densely; the sparse AO contraction is not
implemented.

> **Unverified equation 18.15** · the Laplace dielectric
>
> (18.14) and (18.15), and the claim that the fallback rule
> $u\,t_{\max} > \pi/2$ keeps the energy within the quadrature error.
>
> | | |
> |---|---|
> | Proof | none committed. The Rust tests compare against the dense kernel: `h2o_ccpvdz_laplace_matches_dense_at_static` in `crates/ferric-rpa/tests/laplace_chi0.rs` (elementwise, relative $5\times10^{-3}$), and `closed_shell_rpa_laplace_chebyshev_matches_dense_h2o` in `crates/ferric-rpa/tests/u_pdep_rpa.rs` ($5\times10^{-4}$ Ha). |
> | Status | gap: no `theory.md` derivation, no script, library only |
>
> <!-- unverified: id=rpa.chi0.laplace; todo=derive (18.14)-(18.15) in crates/ferric-rpa/theory.md and add a sympy check of the cosine-Laplace identity and the fallback rule -->

## 18.5.2 Requirements the code enforces

| Requirement | Why | Where |
|---|---|---|
| canonical orbital energies | $\Delta_i^a = \varepsilon_a - \varepsilon_i$ is a per-pair scalar only in a Fock-diagonal basis | debug assertion in `build_e_ia` (`crates/ferric-rpa/src/laplace_chi0.rs`) |
| $n \in \{3, 5, 7\}$ | only those minimax tables exist; other sizes are an error, not a silent fallback | `LaplaceQuadrature::new` |
| no aux-basis route | the energy-only route of section 18.1 has no Laplace evaluator, so the Laplace backend always runs the static solve | `static_route_required` |

## 18.5.3 Numerical anchors

| Quantity | Value | Source |
|---|---|---|
| Laplace vs dense $\tilde{\boldsymbol\varepsilon}(0)$, H2O / cc-pVDZ | relative $< 5\times10^{-3}$ elementwise | `crates/ferric-rpa/tests/laplace_chi0.rs` |
| RPA energy, Chebyshev–Tan + Laplace vs Gauss–Legendre + dense, H2O / cc-pVDZ | $< 5\times10^{-4}$ Ha | `crates/ferric-rpa/tests/u_pdep_rpa.rs` |

These are tolerances of tests, not graded accuracy; the backend has no
validation row.

## 18.5.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-quadrature/theory.md`, "1. Minimax Laplace Quadrature" | the node tables |
| `crates/ferric-quadrature/theory.md`, "2. Imaginary-Frequency Grids for RPA" | the fallback counts for Gauss–Legendre and Chebyshev–Tan |
| module documentation of `crates/ferric-rpa/src/laplace_chi0.rs` and `crates/ferric-rpa/src/ao_rpa.rs` | the two forms and their scope |

**Known limits (2026-11-15).** Library only. No reduced-scaling claim: both
forms are dense.
