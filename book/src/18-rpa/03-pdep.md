# 18.3 Projective dielectric eigenpotentials

<!-- section: 18.3; status: ready; registry: pdep-rpa;
     theory: crates/ferric-rpa/theory.md#"The PDEP Eigenpotential Approach" -->

PDEP evaluates the frequency form of the RPA energy (section 18.1, equation
18.4) in a basis taken from the static dielectric matrix instead of the full
auxiliary basis. The static matrix is diagonalized once; the modes that
differ from the identity by more than a threshold are kept; and at every
frequency the dielectric matrix is projected onto the kept modes and
diagonalized there. With every mode kept the result is the auxiliary-basis
energy exactly. With truncation it is the energy of the projected dielectric
matrix.

This section describes the loop the code runs. It differs in two places from
the description in `crates/ferric-rpa/theory.md`: the default static solve is
one dense diagonalization, not an iterative one, and the frequency loop
diagonalizes the projected matrix (both are listed under "Known limits").

| Index | Range | Code name |
|---|---|---|
| $i, a$ | active occupied, virtual; $ia$ flattened as $i\,n_v + a$ | `nocc`, `nvir` |
| $P, Q$ | auxiliary functions in the dressed basis | `naux` |
| $\xi, \eta$ | PDEP modes, kept count $N_{\mathrm{pdep}}$ | `n_keep` |
| $k$ | frequency node $u_k$, weight $w_k$ | `quad_freqs`, `quad_weights` |

| Entry point | Location |
|---|---|
| Rust | `ferric-rpa/src/lib.rs::{run_pdep_rpa,run_pdep_rpa_from_intermediates}`, `ferric-rpa/src/energy_route.rs::run_pdep_rpa_energy` |
| Static solve | `ferric-rpa/src/lanczos.rs::run_lanczos_full_rank_budgeted` (default), `ferric-rpa/src/davidson.rs::{run_davidson_static,run_davidson_seeded}`, `ferric-rpa/src/lanczos.rs::run_lanczos_seeded` |
| Truncation | `ferric-rpa/src/config.rs::n_kept_modes` |
| Frequency loop | `ferric-rpa/src/energy.rs::{eval_eigenvalues_at_frequencies_budgeted,eval_trace_log_summands_budgeted}`, `ferric-rpa/src/sternheimer.rs::{dielectric_apply,dielectric_matrix_from_projection_into}` |
| CLI | `method.kind = "pdep-rpa"`, section `[rpa]` (`examples/water-pdep-rpa.toml`) |
| Python | `run_pdep_rpa` (closed shell) |

## 18.3.1 The dielectric matrix and its static modes

In the dressed auxiliary basis of section 6.1 the symmetrized dielectric
matrix is

$$ \tilde{\boldsymbol\varepsilon}(iu) = \mathbf 1 + \boldsymbol\Pi(iu), \qquad \Pi_{PQ}(iu) = 4\sum_{ia} B^P_{ia}\,\frac{\Delta_i^a}{u^2 + (\Delta_i^a)^2}\,B^Q_{ia} . \tag{18.7} $$

$\boldsymbol\Pi(iu)$ is positive semidefinite, so every eigenvalue of
$\tilde{\boldsymbol\varepsilon}(iu)$ is at least 1. The static modes solve

$$ \tilde{\boldsymbol\varepsilon}(0)\,\mathbf v_\xi = \lambda_\xi(0)\,\mathbf v_\xi, \qquad \lambda_\xi(0) \ge 1 . \tag{18.8} $$

A mode with $\lambda_\xi(0)$ close to 1 is a potential the molecule barely
screens. The kept set is

$$ \mathcal K = \{\xi : |\lambda_\xi(0) - 1| > \tau\}, \qquad \tau = \texttt{trunc\_thresh}, \tag{18.9} $$

with $\tau \le 0$ keeping every mode, and at least one mode always kept.

![Left: the 84 static eigenvalues of water / cc-pVDZ (cc-pvdz-ri), as $|\lambda_\xi(0) - 1|$ sorted, with the default threshold. Right: the error of the truncated energy against the full-rank one, against the number of modes kept. Script: `book/src/figures/18-rpa/pdep_spectrum.py`.](../figures/18-rpa/pdep_spectrum.svg)

The spectrum falls smoothly over several decades, with no gap to cut at. The
truncation error is positive (the truncated energy is less negative) and
falls with the number of modes kept, faster once the threshold passes into
the steep tail of the spectrum.

## 18.3.2 The frequency loop

Let $\mathbf U$ be the $N_{\mathrm{aux}} \times N_{\mathrm{pdep}}$ matrix of
kept static modes. At each node the projected dielectric matrix is

$$ \tilde{\boldsymbol\varepsilon}_U(iu_k) = \mathbf U^{\mathsf T}\tilde{\boldsymbol\varepsilon}(iu_k)\,\mathbf U = \mathbf 1 + \mathbf Y\,\mathbf G(u_k)\,\mathbf Y^{\mathsf T}, \qquad \mathbf Y = \mathbf U^{\mathsf T}\mathbf B, \qquad G_{ia}(u) = \frac{4\Delta_i^a}{u^2 + (\Delta_i^a)^2} , \tag{18.10} $$

and the energy is

$$ E_c = \frac{1}{2\pi}\sum_k w_k \sum_{\eta} s\big(\lambda_\eta(iu_k)\big), \qquad s(\lambda) = \ln\lambda + 1 - \lambda , \tag{18.11} $$

where $\lambda_\eta(iu_k)$ are the eigenvalues of
$\tilde{\boldsymbol\varepsilon}_U(iu_k)$. The static modes do not diagonalize
$\tilde{\boldsymbol\varepsilon}(iu)$ at $u > 0$, so the eigenvalues are not the
diagonal elements $\mathbf v_\xi^{\mathsf T}\tilde{\boldsymbol\varepsilon}(iu_k)\mathbf v_\xi$.
Using the diagonal would drop the coupling between modes.

The sum in (18.11) is a trace of a matrix function, so it equals
$\ln\det\tilde{\boldsymbol\varepsilon}_U + \operatorname{tr}(\mathbf 1 - \tilde{\boldsymbol\varepsilon}_U)$,
and the code takes that route when nothing downstream needs the eigenvalues.
With every mode kept, $\mathbf U$ is orthogonal and (18.11) is the
auxiliary-basis energy of section 18.1; that is why the full-rank energy-only
configuration skips the static solve altogether.

> **Verified equation 18.11** · projection, diagonalization and truncation
>
> With the full static eigenbasis, the eigenvalues of
> $\mathbf U^{\mathsf T}\tilde{\boldsymbol\varepsilon}(iu)\mathbf U$ give the
> auxiliary-basis log-det energy exactly; the diagonal alone does not; and
> the energy of the modes kept by (18.9) converges to the full one as
> $\tau \to 0$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_drpa_pdep_sympy.py`, checks `pdep-full-rank`, `pdep-diagonal-control`, `pdep-truncation` |
> | Negative control | `pdep-diagonal-control`: the diagonal must miss |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/energy.rs::eval_eigenvalues_at_frequencies_budgeted`, `ferric-rpa/src/config.rs::n_kept_modes` |
> | Code test | `water_ccpvdz_aux_route_matches_the_eigensolve_route` in `crates/ferric-rpa/tests/rpa_energy_route.rs` |
>
> <!-- verified: id=rpa.pdep.projection; script=scripts/rpa_drpa_pdep_sympy.py; claim=pdep-full-rank; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/energy.rs::eval_eigenvalues_at_frequencies_budgeted; rust_test=ferric-rpa/tests/rpa_energy_route.rs::water_ccpvdz_aux_route_matches_the_eigensolve_route; registry=pdep-rpa -->

## 18.3.3 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF, $B^P_{ia}$, metric factor $M$ | as RI-MP2 | `compute_rpa_intermediates` (ferric-mp2) |
| 2 | $\tilde{\boldsymbol\varepsilon}(0)$ assembled in column panels, symmetrized, one dense eigendecomposition; pairs ordered by $\lvert\lambda - 1\rvert$, largest first | $N_{\mathrm{aux}}^2 n_on_v + N_{\mathrm{aux}}^3$ | `run_lanczos_full_rank_budgeted` |
| 3 | keep the first $N_{\mathrm{pdep}}$ pairs by (18.9) | | `n_kept_modes` |
| 4 | eigenpotentials in raw auxiliary coefficients, $\mathbf c_\xi = M^{\mathsf T}\mathbf v_\xi$ | $N_{\mathrm{aux}}^2N_{\mathrm{pdep}}$ | in `run_pdep_rpa` |
| 5 | $\mathbf Y = \mathbf U^{\mathsf T}\mathbf B$, once | $N_{\mathrm{pdep}}N_{\mathrm{aux}}n_on_v$ | frequency loop |
| 6 | per node: scale the columns of $\mathbf Y$ by $\sqrt{G_{ia}(u_k)}$, symmetric rank-$n_on_v$ product, add $\mathbf 1$ | $N_{\mathrm{pdep}}^2n_on_v$ | `dielectric_matrix_from_projection_into` |
| 7 | per node: eigenvalues of $\tilde{\boldsymbol\varepsilon}_U$ and (18.11), or LU and the log-det | $N_{\mathrm{pdep}}^3$ | `eval_eigenvalues_at_frequencies_budgeted`, `eval_trace_log_summands_budgeted` |

**Step 2 is not iterative by default.** The default eigensolver is called
`Lanczos`, and with the dense $\chi_0$ it seeds a block Krylov space with the
whole identity, which collapses to one dense diagonalization. The code
assembles $\tilde{\boldsymbol\varepsilon}(0)$ by applying it to panels of
identity columns (`dielectric_apply` at $u = 0$), so the memory per panel
follows the budget. The comment beside that arm records why the targeted
solve lost: RI auxiliary sets have 20 to 30 functions per atom, so a physical
threshold keeps most modes, and a dense diagonalization beats a narrow
Krylov iteration at that fraction. The savings of truncation are in steps 5
to 7, not in step 2.

The other static solvers are library options (`PdepRpaConfig::eigensolver`,
not exposed in the CLI or Python): Davidson, seeded
with four vectors per atom when $\tau > 0$ and $N_{\mathrm{aux}} > 4N_{\mathrm{at}}$,
and a seeded block Lanczos on the Boys-screened $\chi_0$, which
`chi0_sparsity = "boys"` selects from the CLI and Python. The Davidson eigenvalues are also returned in
descending order, so step 3 is the same for every solver.

**Step 4 needs the transpose.** A dressed vector $\mathbf v$ corresponds to
the auxiliary function $\sum_P (M^{\mathsf T}\mathbf v)_P\,\chi_P$. On the
Coulomb path $M = L^{-1}$ is lower triangular, and $M\mathbf v$ in place of
$M^{\mathsf T}\mathbf v$ gives potentials that are neither $V$-orthonormal
nor eigenvectors of $V + \Pi$ (section 6.1). The energy does not use
$\mathbf c_\xi$; the cube and NPZ exports and the polarizability paths do.

**Step 6, the kernel.** The scale factor per pair is the square root of
$G_{ia}(u)$:

<!-- excerpt: ferric-rpa/src/sternheimer.rs::build_scale_factors_with_prefactor -->
```rust
let e_ia = eps_a - eps_i;
s[i * nvir + a] = (prefactor * e_ia / (omega2 + e_ia * e_ia)).sqrt();
```

| Symbol | Code |
|---|---|
| $\Delta_i^a$ | `e_ia` |
| $u^2$ | `omega2` |
| 4 (closed shell) | `prefactor` |
| $\sqrt{G_{ia}(u)}$ | `s[i * nvir + a]` |

**Step 3, the truncation**, is (18.9) as written:

<!-- excerpt: ferric-rpa/src/config.rs::n_kept_modes -->
```rust
let n = if self.trunc_thresh <= 0.0 {
    eigenvalues.len()
} else {
    eigenvalues
        .iter()
        .filter(|&&lam| (lam - 1.0).abs() > self.trunc_thresh)
        .count()
};
n.max(1)
```

It counts the modes above the threshold and keeps that many from the front of
the ordered list, which is the kept set because the list is ordered by
$|\lambda - 1|$.

**Step 7, which route.** The Python `run_pdep_rpa` asks for the eigenvalues
at each node (`need_eigenvalues_freq = true`). The CLI energy run does not, and
takes the LU log-det of $\tilde{\boldsymbol\varepsilon}_U$. Both give (18.11).
With $\tau = 0$ and no export requested, the CLI goes through
`run_pdep_rpa_energy`, which skips steps 2 to 5 and evaluates the log-det in
the full dressed auxiliary basis (section 18.1).

## 18.3.4 Reproducibility and memory

The eigenvector signs and the order within degenerate blocks are arbitrary;
the energy is a trace and does not depend on them. Exported eigenpotentials
can change sign between builds.

The preflight estimate charges the panel-assembled $N_{\mathrm{aux}}^2$
dielectric and its eigenvectors, the $B$ tensor, and one projected matrix per
worker thread in the frequency loop. `[memory] budget_gb` sets the panel width.

## 18.3.5 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[rpa] trunc_thresh`, `trunc_thresh=` | $\ge 0$ | $10^{-4}$ | $\tau$ in (18.9); 0 keeps every mode |
| `[rpa] eigensolver_conv_thresh` | float | | residual tolerance of the iterative solvers; ignored, with a warning, by the default dense solve |
| `[rpa] chi0_sparsity` | `dense`, `boys`, `auto` | `dense` | the Boys-screened static solve |
| `[rpa] quadrature`, `n_quad`, `u0` | section 18.2 | Gauss–Legendre, 20, 0.5 | frequency grid |
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | frozen occupied orbitals |

## 18.3.6 Numerical anchors

| Quantity | Systems | Error | Source |
|---|---|---|---|
| Full-rank $E_c$ vs PySCF `gw.rpa.RPA`, 40 Gauss–Legendre nodes | distorted H2O, NH3, CH4 / cc-pVDZ | $\le 9.2\times10^{-13}$ Ha | `crates/ferric-rpa/tests/validation_rpa_gradient.rs` |
| Full-rank $E_c$, Coulomb, vs numpy dRPA on PySCF integrals | H2O, NH3 / cc-pVDZ; H2O / aug-cc-pVDZ | $\le 4.5\times10^{-11}$ Ha | `crates/ferric-rpa/tests/validation_attenuated_rpa.rs` |
| Eigenpotentials vs PySCF's generalized dielectric eigenproblem (NPZ export) | water / cc-pVDZ | metric $4.3\times10^{-12}$, projector $2.6\times10^{-7}$ | `crates/ferric-cli/tests/validation_npz.rs` |
| Aux-basis route vs the eigensolve route | water, n-octane, n-hexadecane / cc-pVDZ | asserted equal | `crates/ferric-rpa/tests/rpa_energy_route.rs` |

All from `site/src/reference/validation.md`, except the last row, which is
the test's own assertion.

## 18.3.7 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-rpa/theory.md`, "The PDEP Eigenpotential Approach" | (18.7)–(18.11) |
| `scripts/rpa_drpa_pdep_sympy.py`, check group 7 | projection and truncation |
| `crates/ferric-rpa/tests/pdep_eigenpotentials_are_physical.rs` | the back-transform of step 4 |

**Known limits (2026-11-15).**

- `crates/ferric-rpa/theory.md` says the static matrix is "diagonalized
  iteratively (block Lanczos by default)". The default is one dense
  diagonalization (step 2).
- In `crates/ferric-rpa/src/energy.rs` an orphaned doc block written for
  `eval_eigenvalues_at_frequencies` sits on top of
  `eval_eigenvalues_at_frequencies_laplace` and says it "uses the diagonal of
  the projected dielectric matrix". Both functions diagonalize it.
- The analytic RPA gradient (section 15.5) requires $\tau = 0$.
- Python `run_pdep_rpa` is closed shell only; the open-shell path is CLI-only
  (section 18.4).
