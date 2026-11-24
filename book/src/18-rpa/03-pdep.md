# Projective dielectric eigenpotentials {#pdep}

<!-- status: ready; registry: pdep-rpa -->

Apply a small external potential to a molecule and its electrons
rearrange to screen it. Some potentials, shaped like the molecule's
polarizable bonds and lone pairs, are screened strongly. Most of the
$N_{\mathrm{aux}}$ directions in the auxiliary basis are potentials the
electrons barely respond to, and they add almost nothing to the RPA energy.
Projective dielectric eigenpotentials (PDEP) find the strongly screened
potentials once, from the static response, and run the frequency loop in
that smaller basis.

The plan: the dielectric matrix and its static modes (@sec:pdep-dielectric-matrix-static-modes), the
frequency loop in the kept modes (@sec:pdep-frequency-loop), the algorithm and its code
(@sec:pdep-algorithm), then reproducibility, options and anchors.

PDEP evaluates the frequency form of the RPA energy (section @sec:drpa-three-ways, equation
@eq:drpa-three-ways-4) in a basis taken from the static dielectric matrix instead of the full
auxiliary basis. The static matrix is diagonalized once. The modes that
differ from the identity by more than a threshold are kept. At every
frequency the dielectric matrix is projected onto the kept modes and
diagonalized there. With every mode kept the result is the auxiliary-basis
energy exactly. With truncation it is the energy of the projected dielectric
matrix.

By default the static solve is one dense diagonalization. The iterative
solvers are library options (section @sec:pdep-algorithm).

| Index | Range | Code name |
|---|---|---|
| $i, a$ | active occupied, virtual; $ia$ flattened as $i\,n_v + a$ | `nocc`, `nvir` |
| $P, Q$ | auxiliary functions in the dressed basis | `naux` |
| $\xi, \eta$ | PDEP modes, kept count $N_{\mathrm{pdep}}$ | `n_keep` |
| $k$ | frequency node $u_k$, weight $w_k$ | `quad_freqs`, `quad_weights` |

| Entry point | Location |
|---|---|
| Rust | `ferric-rpa/src/lib.rs::{run_pdep_rpa,run_pdep_rpa_from_intermediates}`, `ferric-rpa/src/energy_route.rs::run_pdep_rpa_energy` |
| Static solve | `ferric-rpa/src/lib.rs::run_pdep_rpa_eigensolve` (dispatch), `ferric-rpa/src/lanczos.rs::run_lanczos_full_rank_budgeted` (default), `ferric-rpa/src/davidson.rs::{run_davidson_static,run_davidson_seeded}`, `ferric-rpa/src/lanczos.rs::run_lanczos_seeded` |
| Truncation | `ferric-rpa/src/config.rs::n_kept_modes` |
| Frequency loop | `ferric-rpa/src/energy.rs::{eval_eigenvalues_at_frequencies_budgeted,eval_trace_log_summands_budgeted}`, `ferric-rpa/src/sternheimer.rs::{dielectric_apply,dielectric_matrix_from_projection_into}` |
| CLI | `method.kind = "pdep-rpa"`, section `[rpa]` (`examples/water-pdep-rpa.toml`) |
| Python | `run_pdep_rpa` (closed shell) |

## The dielectric matrix and its static modes {#pdep-dielectric-matrix-static-modes}

In the dressed auxiliary basis of section @sec:factorization-and-metric the symmetrized dielectric
matrix is

$$ \tilde{\boldsymbol\varepsilon}(iu) = \mathbf 1 + \boldsymbol\Pi(iu), \qquad \Pi_{PQ}(iu) = 4\sum_{ia} B^P_{ia}\,\frac{\Delta_i^a}{u^2 + (\Delta_i^a)^2}\,B^Q_{ia} . \label{eq:pdep-1} $$

$\boldsymbol\Pi(iu)$ is positive semidefinite. For any vector $\mathbf v$,
$\mathbf v^{\mathsf T}\boldsymbol\Pi\mathbf v = \sum_{ia} g_{ia}\big(\sum_P v_PB^P_{ia}\big)^2$
with $g_{ia} = 4\Delta_i^a/(u^2 + (\Delta_i^a)^2) > 0$, a sum of
non-negative terms. So every eigenvalue of $\tilde{\boldsymbol\varepsilon}(iu)$
is at least 1. A plane-wave code applies $\boldsymbol\Pi$ to a trial
potential by a Sternheimer solve per occupied orbital. With every virtual
orbital in hand ferric sums over the occupied-virtual pairs instead, which
is (@eq:pdep-1) applied to a vector.

The static modes solve

$$ \tilde{\boldsymbol\varepsilon}(0)\,\mathbf v_\xi = \lambda_\xi(0)\,\mathbf v_\xi, \qquad \lambda_\xi(0) \ge 1 . \label{eq:pdep-2} $$

$\lambda_\xi(0) - 1$ measures how strongly the molecule screens the
potential $\mathbf v_\xi$. A mode with $\lambda_\xi(0)$ close to 1 is a
potential the molecule barely screens. The kept set is

$$ \mathcal K = \{\xi : |\lambda_\xi(0) - 1| > \tau\}, \qquad \tau = \texttt{trunc\_thresh}, \label{eq:pdep-3} $$

with $\tau \le 0$ keeping every mode, and at least one mode always kept.

How much does a dropped mode cost? At $u = 0$ the static modes diagonalize
$\tilde{\boldsymbol\varepsilon}(0)$, so the summand of (@eq:drpa-three-ways-4) there is
$\sum_\xi s(\lambda_\xi(0))$ with $s(\lambda) = \ln\lambda + 1 - \lambda$, and
dropping a mode removes its term. For $x = \lambda - 1 \ge 0$,
$\ln(1 + x) \ge x - x^2/2$, so $|s(\lambda)| \le x^2/2$. A dropped mode
therefore changes the $u = 0$ summand by at most $\tau^2/2$, which is
$5\times10^{-9}$ at the default $\tau = 10^{-4}$. This bounds the static
summand only. At $u > 0$ the static modes no longer diagonalize the matrix
(section @sec:pdep-frequency-loop), so the energy needs the sign argument
given there. The bound itself, $\ln(1 + x) - x + x^2/2 \ge 0$, follows from
its value 0 at $x = 0$ and its derivative $x^2/(1 + x) \ge 0$; check
`static-summand-bound` of the proof script.

![Left: the 84 static eigenvalues of water / cc-pVDZ (cc-pvdz-ri), as $|\lambda_\xi(0) - 1|$ sorted, with the default threshold. Right: the error of the truncated energy against the full-rank one, against the number of modes kept; it cannot be negative (section @sec:pdep-frequency-loop). Script: `book/src/figures/18-rpa/pdep_spectrum.py`.](../figures/18-rpa/pdep_spectrum.svg)

The spectrum falls smoothly over several decades, with no gap to cut at. The
truncation error is positive (the truncated energy is less negative), as
section @sec:pdep-frequency-loop proves it must be, and falls with the number
of modes kept, faster once the threshold passes into the steep tail of the
spectrum.

The same numbers from Python, from the `589b50832` build: the modes kept and
the energy error at five thresholds, against a full-rank run on the same
default 20-node grid (water / cc-pVDZ). That grid itself misses the integral
by $3.4\times10^{-6}$ Ha (section @sec:quadratures-defaults), more than the
truncation error at the default $\tau = 10^{-4}$.

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

full = ferric.run_pdep_rpa(mol, bs, aux, trunc_thresh=0.0)
print(f"full rank: {full.n_eigenpotentials} modes, E_c = {full.e_rpa:.10f}")
for tau in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    r = ferric.run_pdep_rpa(mol, bs, aux, trunc_thresh=tau)
    print(f"tau={tau:.0e}: {r.n_eigenpotentials:3d} modes, E_c - E_c(full) = {r.e_rpa - full.e_rpa:+.2e}")
```

```text
full rank: 84 modes, E_c = -0.2312229690
tau=1e-01:  15 modes, E_c - E_c(full) = +1.04e-02
tau=1e-02:  36 modes, E_c - E_c(full) = +5.25e-04
tau=1e-03:  51 modes, E_c - E_c(full) = +1.49e-05
tau=1e-04:  60 modes, E_c - E_c(full) = +1.95e-07
tau=1e-05:  67 modes, E_c - E_c(full) = +2.03e-08
```

## The frequency loop {#pdep-frequency-loop}

Let $\mathbf U$ be the $N_{\mathrm{aux}} \times N_{\mathrm{pdep}}$ matrix of
kept static modes. Project (@eq:pdep-1) onto them: $\mathbf U^{\mathsf T}\mathbf 1\mathbf U = \mathbf 1$
because the modes are orthonormal, and $\mathbf U^{\mathsf T}\boldsymbol\Pi\mathbf U$
needs only the projected factor $\mathbf Y = \mathbf U^{\mathsf T}\mathbf B$,
which does not depend on $u$. At each node the projected dielectric matrix
is

$$ \tilde{\boldsymbol\varepsilon}_U(iu_k) = \mathbf U^{\mathsf T}\tilde{\boldsymbol\varepsilon}(iu_k)\,\mathbf U = \mathbf 1 + \mathbf Y\,\mathbf G(u_k)\,\mathbf Y^{\mathsf T}, \qquad \mathbf Y = \mathbf U^{\mathsf T}\mathbf B, \qquad G_{ia}(u) = \frac{4\Delta_i^a}{u^2 + (\Delta_i^a)^2} , \label{eq:pdep-4} $$

and the energy is

$$ E_c = \frac{1}{2\pi}\sum_k w_k \sum_{\eta} s\big(\lambda_\eta(iu_k)\big), \qquad s(\lambda) = \ln\lambda + 1 - \lambda , \label{eq:rpa-pdep-projection} $$

where $\lambda_\eta(iu_k)$ are the eigenvalues of
$\tilde{\boldsymbol\varepsilon}_U(iu_k)$.

Why diagonalize at every node? The static modes do not diagonalize
$\tilde{\boldsymbol\varepsilon}(iu)$ at $u > 0$, because $G_{ia}(u)$ falls
off at a different rate for each gap. So the eigenvalues are not the
diagonal elements $\mathbf v_\xi^{\mathsf T}\tilde{\boldsymbol\varepsilon}(iu_k)\mathbf v_\xi$.
Using the diagonal would drop the coupling between modes: 2.8e-2 Ha on the
model of the proof script.

The sum in (@eq:rpa-pdep-projection) is a trace of a matrix function, so it equals
$\ln\det\tilde{\boldsymbol\varepsilon}_U + \operatorname{tr}(\mathbf 1 - \tilde{\boldsymbol\varepsilon}_U)$,
and the code takes that route when nothing downstream needs the eigenvalues.
Check the full-rank limit. With every mode kept, $\mathbf U$ is orthogonal,
and both the determinant and the trace are unchanged by
$\tilde{\boldsymbol\varepsilon} \to \mathbf U^{\mathsf T}\tilde{\boldsymbol\varepsilon}\mathbf U$.
So (@eq:rpa-pdep-projection) is the auxiliary-basis energy of section @sec:drpa-three-ways. That is why the
full-rank energy-only configuration skips the static solve altogether.

Truncation has a sign. Let $\lambda_1 \ge \lambda_2 \ge \dots$ be the
eigenvalues of $\tilde{\boldsymbol\varepsilon}(iu_k)$ and
$\mu_1 \ge \dots \ge \mu_m$ those of
$\mathbf U^{\mathsf T}\tilde{\boldsymbol\varepsilon}(iu_k)\mathbf U$ for any
$N_{\mathrm{aux}} \times m$ matrix $\mathbf U$ with orthonormal columns. By
Cauchy interlacing $\mu_i \le \lambda_i$, and $\mu_i \ge 1$ because
$\boldsymbol\Pi$ is positive semidefinite. $s(\lambda) = \ln\lambda + 1 - \lambda$
is non-positive and decreasing on $\lambda \ge 1$. So
$\sum_{i\le m}s(\mu_i) \ge \sum_{i\le m}s(\lambda_i) \ge \sum_{\text{all}}s(\lambda_i)$
at every node. The weights $w_k$ are positive, so the truncated energy is
never below the full-rank one on the same grid: $E_c(\tau) \ge E_c(0)$.
The proof script checks the interlacing and the summand inequality on random
projections (`pdep-truncation-sign`). The amplitude mask of local dRPA
(section @sec:local-drpa) has no such sign.

> **Verified equation @eq:rpa-pdep-projection** · projection, diagonalization and truncation
>
> With the full static eigenbasis, the eigenvalues of
> $\mathbf U^{\mathsf T}\tilde{\boldsymbol\varepsilon}(iu)\mathbf U$ give the
> auxiliary-basis log-det energy exactly; the diagonal alone does not; and
> the energy of the modes kept by (@eq:pdep-3) converges to the full one as
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

The same script covers the static-summand bound of section
@sec:pdep-dielectric-matrix-static-modes (`static-summand-bound`).

## The algorithm {#pdep-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF, $B^P_{ia}$, metric factor $M$ | as RI-MP2 | `compute_rpa_intermediates` (ferric-mp2) |
| 2 | $\tilde{\boldsymbol\varepsilon}(0)$ assembled in column panels, symmetrized, one dense eigendecomposition; pairs ordered by $\lvert\lambda - 1\rvert$, largest first | $N_{\mathrm{aux}}^2 n_on_v + N_{\mathrm{aux}}^3$ | `run_lanczos_full_rank_budgeted` |
| 3 | keep the first $N_{\mathrm{pdep}}$ pairs by (@eq:pdep-3) | | `n_kept_modes` |
| 4 | eigenpotentials in raw auxiliary coefficients, $\mathbf c_\xi = M^{\mathsf T}\mathbf v_\xi$ | $N_{\mathrm{aux}}^2N_{\mathrm{pdep}}$ | in `run_pdep_rpa` |
| 5 | $\mathbf Y = \mathbf U^{\mathsf T}\mathbf B$, once | $N_{\mathrm{pdep}}N_{\mathrm{aux}}n_on_v$ | frequency loop |
| 6 | per node: scale the columns of $\mathbf Y$ by $\sqrt{G_{ia}(u_k)}$, symmetric rank-$n_on_v$ product, add $\mathbf 1$ | $N_{\mathrm{pdep}}^2n_on_v$ | `dielectric_matrix_from_projection_into` |
| 7 | per node: eigenvalues of $\tilde{\boldsymbol\varepsilon}_U$ and (@eq:rpa-pdep-projection), or LU and the log-det | $N_{\mathrm{pdep}}^3$ | `eval_eigenvalues_at_frequencies_budgeted`, `eval_trace_log_summands_budgeted` |

**Step 2 is not iterative by default.** The default eigensolver is called
`Lanczos`. With the dense $\chi_0$ it seeds a block Krylov space with the
whole identity, which collapses to one dense diagonalization. The code
assembles $\tilde{\boldsymbol\varepsilon}(0)$ by applying it to panels of
identity columns (`dielectric_apply` at $u = 0$), so the memory per panel
follows the budget. The comment beside that arm in
`crates/ferric-rpa/src/lib.rs` records why the targeted solve lost. RI
auxiliary sets have 20 to 30 functions per atom, so a physical threshold
keeps most modes (60 of 84, 71%, for water at the default threshold in the
example above), and a dense diagonalization beats a narrow Krylov iteration
at that fraction. The
savings of truncation are in steps 5 to 7, not in step 2.

The other static solvers are library options (the choice
`PdepRpaConfig::eigensolver` is not exposed in the CLI or Python); `run_pdep_rpa_eigensolve` dispatches
between them. Davidson is seeded with four vectors per atom (an
atom-localized seed) when $\tau > 0$ and $N_{\mathrm{aux}} > 4N_{\mathrm{at}}$,
and from the identity otherwise. It is the only solver that reads
`eigensolver_conv_thresh` and `eigensolver_max_vecs`. A block Lanczos
iteration (`run_lanczos_seeded`) runs only on the Boys-screened $\chi_0$,
which `chi0_sparsity = "boys"` selects from the CLI and Python, started from
a Boys-orbital seed. With the Boys-screened $\chi_0$ only the static
eigensolve uses the screened tiles; the frequency loop uses the dense $B$.
The Davidson eigenvalues are also returned
in descending order, so step 3 is the same for every solver.

**Step 4 needs the transpose.** A dressed vector $\mathbf v$ corresponds to
the auxiliary function $\sum_P (M^{\mathsf T}\mathbf v)_P\,\chi_P$. With
$M^{\mathsf T}M = V^{-1}$ and orthonormal $\mathbf v_\xi$, the coefficients
are $V$-orthonormal:
$\mathbf c_\xi^{\mathsf T}V\mathbf c_\eta = \mathbf v_\xi^{\mathsf T}MVM^{\mathsf T}\mathbf v_\eta = \delta_{\xi\eta}$,
because $MVM^{\mathsf T} = \mathbf 1$ ($V = LL^{\mathsf T}$, $M = L^{-1}$). On the
Coulomb path $M = L^{-1}$ is lower triangular, and $M\mathbf v$ in place of
$M^{\mathsf T}\mathbf v$ gives potentials that are neither $V$-orthonormal
nor solutions of $(V + \Pi^{\mathrm{raw}})\mathbf c = \lambda V\mathbf c$
(section @sec:factorization-and-metric). The energy does not use
$\mathbf c_\xi$; the cube and NPZ exports and the polarizability paths do.

**Step 6, the kernel.** The scale factor per pair is the square root of
$G_{ia}(u)$. From `ferric-rpa/src/sternheimer.rs::build_scale_factors_with_prefactor`,
the body of the double loop over $i$ and $a$:

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

The square root lets the code scale the columns of $\mathbf Y$ once and form
$\mathbf Y\mathbf G\mathbf Y^{\mathsf T}$ as $(\mathbf Y\mathbf G^{1/2})(\mathbf Y\mathbf G^{1/2})^{\mathsf T}$,
a symmetric rank-$k$ update. `ferric-rpa/src/sternheimer.rs::dielectric_matrix_from_projection_into`,
from the line after its shape assertions to the end:

<!-- excerpt: ferric-rpa/src/sternheimer.rs::dielectric_matrix_from_projection_into -->
```rust
rhs_scaled.assign(y);
let scale_row = scale.view().insert_axis(Axis(0));
Zip::from(&mut *rhs_scaled)
    .and_broadcast(scale_row)
    .for_each(|x, &s| *x *= s);

syrk_aat_into(rhs_scaled, out);
for alpha in 0..m {
    out[(alpha, alpha)] += 1.0;
}
```

| Code | Equation |
|---|---|
| `y` | $\mathbf Y = \mathbf U^{\mathsf T}\mathbf B$ |
| `rhs_scaled` after the `Zip` | $\mathbf Y\mathbf G^{1/2}$ |
| `syrk_aat_into` | $\mathbf Y\mathbf G\mathbf Y^{\mathsf T}$ |
| the diagonal loop | $\mathbf 1 + \mathbf Y\mathbf G\mathbf Y^{\mathsf T}$, (@eq:pdep-4) |

`rhs_scaled` and `out` are scratch buffers owned by each worker thread and
reused across the frequencies it handles, so the scratch held at once is
one pair per thread, not one per frequency.

**Step 3, the truncation**, is (@eq:pdep-3) as written:

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
the ordered list. That is the kept set because the list is ordered by
$|\lambda - 1|$.

**Step 7, which route.** The Python `run_pdep_rpa` asks for the eigenvalues
at each node (`need_eigenvalues_freq = true`). The CLI energy run does not,
and takes the LU log-det of $\tilde{\boldsymbol\varepsilon}_U$. Both give
(@eq:rpa-pdep-projection). With $\tau = 0$ and no export requested, the CLI goes through
`run_pdep_rpa_energy`, which skips steps 2 to 5 and evaluates the log-det in
the full dressed auxiliary basis (section @sec:drpa-three-ways,
`RpaEnergyRoute::AuxBasisLogDet`). It takes `run_pdep_rpa`
(`RpaEnergyRoute::StaticEigenbasis`) instead when
`ferric-rpa/src/energy_route.rs::static_route_required` returns a reason, and
prints that reason. The reasons, checked in this order:

1. the caller asked for the per-frequency eigenvalues (`need_eigenvalues_freq`);
2. the caller asked for the inverse-dielectric stack (`need_inv_dielectric_freq`);
3. the $\chi_0$ backend is not `Chi0Backend::Dense` (the Laplace backend has no aux-basis evaluator);
4. `chi0_sparsity` resolves to anything but dense (the Boys-screened eigensolve);
5. `trunc_thresh` $> 0$.

The CLI `drpa` kind takes the aux-basis route; a `pdep-rpa` run with the
default `trunc_thresh` takes the static route.

## Reproducibility and memory {#pdep-reproducibility-memory}

The eigenvector signs and the order within degenerate blocks are arbitrary.
The energy is a trace and does not depend on them. Exported eigenpotentials
can change sign between builds.

The preflight estimate charges the panel-assembled $N_{\mathrm{aux}}^2$
dielectric and its eigenvectors, the $B$ tensor, and one projected matrix per
worker thread in the frequency loop. `[memory] budget_gb` sets the panel width.

## Options {#pdep-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[rpa] trunc_thresh`, `trunc_thresh=` | $\ge 0$ | $10^{-4}$ | $\tau$ in (@eq:pdep-3); 0 keeps every mode |
| `[rpa] eigensolver_conv_thresh`, `eigensolver_conv_thresh=` | float | $10^{-6}$ | residual tolerance of the iterative library solvers; the default dense solve ignores it and warns when it is set |
| `[rpa] chi0_sparsity` | `dense`, `boys`, `auto` | `dense` | the Boys-screened static solve |
| `[rpa] quadrature`, `n_quad`, `u0` | section @sec:quadratures | Gauss–Legendre, 20, 0.5 | frequency grid |
| `[rpa] frozen_core`, `frozen_core=` | integer | 0 | frozen occupied orbitals (the `drpa` kind reads `[mp2] frozen_core` instead) |

## Numerical anchors {#pdep-numerical-anchors}

| Quantity | Systems | Error | Source |
|---|---|---|---|
| Full-rank $E_c$ vs PySCF `gw.rpa.RPA`, 40 Gauss–Legendre nodes | distorted H2O, NH3, CH4 / cc-pVDZ | $\le 9.2\times10^{-13}$ Ha | `crates/ferric-rpa/tests/validation_rpa_gradient.rs` |
| Full-rank $E_c$, Coulomb, vs numpy dRPA on PySCF integrals | H2O, NH3 / cc-pVDZ; H2O / aug-cc-pVDZ | $\le 4.5\times10^{-11}$ Ha | `crates/ferric-rpa/tests/validation_attenuated_rpa.rs` |
| Eigenpotentials vs PySCF's generalized dielectric eigenproblem (NPZ export) | water / cc-pVDZ | metric $4.3\times10^{-12}$, projector $2.6\times10^{-7}$ | `crates/ferric-cli/tests/validation_npz.rs` |
| Aux-basis route vs the eigensolve route | water, n-octane, n-hexadecane / cc-pVDZ | asserted equal | `crates/ferric-rpa/tests/rpa_energy_route.rs` |

All from `site/src/reference/validation.md`, except the last row, which is
the test's own assertion. The kernel is also checked one vector at a time:
the test helper `ferric-rpa/src/sternheimer.rs::chi_from_trial_potential`
returns the single-spin $-2\sum_{ia}\Delta_i^a(\sum_PV_PB^P_{ia})^2/(u^2 + (\Delta_i^a)^2)$,
half the closed-shell $-\mathbf V^{\mathsf T}\boldsymbol\Pi\mathbf V$.

## Sources {#pdep-sources}

| Source | What it gives |
|---|---|
| `scripts/rpa_drpa_pdep_sympy.py`, check group 7 | projection and truncation |
| `crates/ferric-rpa/tests/pdep_eigenpotentials_are_physical.rs` | the back-transform of step 4 |
| Wilson, Gygi and Galli, Phys. Rev. B 78, 113303 (2008) | dielectric eigenpotentials as a basis for the response |
| Nguyen and de Gironcoli, Phys. Rev. B 79, 205114 (2009) | the RPA correlation energy from a few eigenmodes of the dielectric matrix |

**Known limits (2026-11-15).**

- The analytic RPA gradient (section @sec:rpa-gradient) requires $\tau = 0$.
- Python `run_pdep_rpa` is closed shell only; the open-shell path is CLI-only
  (section @sec:open-shell-dielectric).
