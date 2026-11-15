# 18.1 dRPA three ways: frequency integral, plasmon formula, ring amplitudes

<!-- section: 18.1; status: ready; registry: drpa;
     theory: crates/ferric-rpa/theory.md#"RPA Correlation Energy" -->

The direct random phase approximation (dRPA) correlation energy has three
standard expressions: a sum over plasmon excitation energies, the energy of
the direct-ring coupled-cluster doubles (drCCD) amplitudes, and an integral of
the dielectric function over imaginary frequency. They are one function of the
orbitals and integrals. ferric implements all three, uses the frequency form
for production energies and the PDEP method (section 18.3), the amplitude form
for the local approximation (section 18.6), and the plasmon form as the
reference that both are tested against.

This section covers a closed-shell RHF reference and the Coulomb operator.
The open-shell dielectric is section 18.4; attenuated operators enter only
through $B$ (section 6.1).

| Index | Range | Code name |
|---|---|---|
| $i, j$ | active occupied | `0..nocc`, after `frozen_core` |
| $a, b$ | virtual | `0..nvir` |
| $P, Q$ | auxiliary | `0..naux` |
| $k$ | frequency node, weight $w_k$ | `quad_freqs`, `quad_weights` |

| Entry point | Location |
|---|---|
| Rust, frequency | `ferric-rpa/src/energy_route.rs::{exact_drpa_config,run_pdep_rpa_energy}`, `ferric-rpa/src/energy.rs::rpa_correlation_energy` |
| Rust, plasmon | `ferric-mp2/src/drpa_amplitude.rs::canonical_plasmon_drpa`, `ferric-rpa/src/diagnostics.rs::ri_drpa_energy` |
| Rust, amplitudes | `ferric-mp2/src/drpa_amplitude.rs::{amplitude_drpa,amplitude_drpa_dense,riccati_masked_solve}` |
| CLI | `method.kind = "drpa"` (`examples/water-drpa.toml`): the frequency route |
| Python | `run_drpa`: the amplitude route at $\varepsilon = 0$; `run_pdep_rpa(trunc_thresh=0)`: the frequency route |

## 18.1.1 The plasmon formula

With $K_{ia,jb} = 2(ia|jb)$ (chapter 0, section 6) and the spin-adapted
singlet response matrices

$$ \mathbf A = \boldsymbol\Delta + \mathbf K, \qquad \mathbf B = \mathbf K, \qquad \Delta_{ia,jb} = \Delta_i^a\,\delta_{ij}\delta_{ab}, $$

the excitation energies $\Omega_n$ are the square roots of the eigenvalues of
$(\mathbf A - \mathbf B)(\mathbf A + \mathbf B) = \boldsymbol\Delta(\boldsymbol\Delta + 2\mathbf K)$, and

$$ E_c^{\mathrm{dRPA}} = \tfrac12\Big(\sum_n \Omega_n - \operatorname{tr}\mathbf A\Big) . \tag{18.1} $$

`canonical_plasmon_drpa` builds $\mathbf A$ and $\mathbf B$ from a global RI
factor on Fock blocks re-diagonalized in the occupied and virtual spaces,
diagonalizes the non-symmetric product with a general eigensolver, and stops
with an error if any $\Omega_n^2$ is negative or not real (an RPA
instability). The dense $(n_on_v)^2$ matrix makes it a reference, not a
production route.

## 18.1.2 Ring amplitudes

The drCCD amplitudes $T_{ia,jb}$ solve the Riccati equation

$$ \mathbf R(\mathbf T) = \mathbf K + \mathcal F(\mathbf T) + \mathbf K\mathbf T + \mathbf T\mathbf K + \mathbf T\mathbf K\mathbf T = 0, \qquad E_c^{\mathrm{dRPA}} = \tfrac12\sum_{ia,jb} K_{ia,jb}T_{ia,jb}, \tag{18.2} $$

with the Fock superoperator of a possibly non-canonical (localized) basis

$$ \mathcal F(\mathbf T)_{ia,jb} = \sum_c f_{ac}T_{ic,jb} + \sum_c T_{ia,jc}f_{cb} - \sum_k f_{ik}T_{ka,jb} - \sum_k T_{ia,kb}f_{kj} . $$

In canonical orbitals $\mathcal F(\mathbf T) = \Delta^{ab}_{ij}\,T_{ia,jb}$, and
dropping the three ring terms gives $T = -K/\Delta$ and the second-order
energy $-\sum 2(ia|jb)^2/\Delta_{ij}^{ab} = 2E_{\mathrm{OS}}$.

The code (`fock_super`, `riccati_masked_solve`) iterates the damped fixed
point $\mathbf T \leftarrow \mathbf T - \mathbf R(\mathbf T)/\mathbf D$, where
$\mathbf D$ is built from the diagonals of the localized occupied and virtual
Fock blocks, optionally with DIIS on the increment. It stops at
$\lVert\mathbf R\rVert/\lVert\mathbf K\rVert < 10^{-12}$ and returns an error
after 500 iterations. The energy is invariant under orthogonal rotations
within the occupied and within the virtual space, which is what licenses the
Boys-localized basis the local method needs (section 18.6).

## 18.1.3 Imaginary frequency

With the RI factor $B^P_{ia}$ of section 6.1 and a real imaginary frequency
$u$,

$$ \Pi_{PQ}(iu) = 4\sum_{ia} B^P_{ia}\,\frac{\Delta_i^a}{u^2 + (\Delta_i^a)^2}\,B^Q_{ia}, \tag{18.3} $$

$$ E_c^{\mathrm{dRPA}} = \frac{1}{2\pi}\int_0^\infty \Big[\ln\det\big(\mathbf 1 + \boldsymbol\Pi(iu)\big) - \operatorname{tr}\boldsymbol\Pi(iu)\Big]\,du \approx \frac{1}{2\pi}\sum_k w_k\,S(u_k) . \tag{18.4} $$

$\mathbf 1 + \boldsymbol\Pi$ is the symmetrized dielectric matrix
$\tilde{\boldsymbol\varepsilon}(iu)$ in the dressed auxiliary basis. The
prefactor 4 is 2 for the spin sum times 2 for the pair of poles at
$\pm\Delta_i^a$. The integrand is negative and falls as $u^{-2}$ at large $u$.
The quadrature is section 18.2.

The summand $S$ is evaluated in one of two ways, which agree per frequency:

| Route | $S(u_k)$ | Used by |
|---|---|---|
| Log-determinant | $\ln\det(\mathbf 1 + \boldsymbol\Pi) - \operatorname{tr}\boldsymbol\Pi$ by an LU factorization | the aux-basis route (`RpaEnergyRoute::AuxBasisLogDet`): CLI `drpa`, and PDEP with every mode kept and no eigenvalues requested |
| Eigenvalues | $\sum_\xi s(\lambda_\xi)$, $s(\lambda) = \ln\lambda + 1 - \lambda$ | PDEP with truncation, U-PDEP, and any caller that asks for eigenvalues |

$s(\lambda)$ is a difference of nearly equal numbers when $\lambda$ is close
to 1, so `trace_log_summand_term` evaluates it from a series in
$x = \lambda - 1$ when $|x| < 0.25$:

<!-- excerpt: ferric-rpa/src/energy.rs::trace_log_summand_term -->
```rust
let x = lam - 1.0;
if x.abs() < TRACE_LOG_SERIES_CUTOFF {
    // f = x² · Σ_{k=0}^{N−2} (−1)^{k+1} x^k / (k+2), Horner from the top.
    let mut acc = 0.0f64;
    for k in (0..=TRACE_LOG_SERIES_ORDER - 2).rev() {
        let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
        acc = acc.mul_add(x, sign / f64::from(k + 2));
    }
    acc * (x * x)
} else {
    lam.ln() - x
}
```

| Symbol | Code |
|---|---|
| $\lambda$ | `lam` |
| $x = \lambda - 1$ | `x` |
| $s(\lambda)$ | the return value |

## 18.1.4 One function, three algorithms

> **Verified equation 18.1** · plasmon, Riccati and frequency forms agree
>
> For a single mode with gap $\Delta$ and coupling $b$, the plasmon energy
> $\tfrac12(\sqrt{\Delta(\Delta + 2b)} - \Delta - b)$, the Riccati root and
> the frequency integral are one function, with second-order term
> $-b^2/(4\Delta)$. On random closed-shell models each of (18.1), (18.2) in a
> randomly rotated basis, and (18.4) equals the spin-orbital definition
> $\tfrac12\sum_n(\Omega_n - A_{nn})$ over both spins.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_drpa_pdep_sympy.py`, checks `one-mode-riccati`, `one-mode-frequency`, `plasmon-vs-definition`, `riccati-vs-definition`, `freq-vs-definition` |
> | Negative control | the runner's mutations: closed-shell $\Pi$ prefactor $4 \to 2$, spin-adapted $\mathbf B$ scaled by $\tfrac12$, the Riccati residual without $\mathbf T\mathbf K$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/drpa_amplitude.rs::{canonical_plasmon_drpa,riccati_masked_solve,fock_super}`, `ferric-rpa/src/sternheimer.rs::build_scale_factors` |
> | Code test | `three_routes_agree_water_ccpvdz` in `crates/ferric-rpa/tests/validation_drpa_frequency.rs` |
>
> <!-- verified: id=rpa.drpa.three-routes; script=scripts/rpa_drpa_pdep_sympy.py; claim=freq-vs-definition; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/drpa_amplitude.rs::{canonical_plasmon_drpa,riccati_masked_solve,fock_super}; rust_test=ferric-rpa/tests/validation_drpa_frequency.rs::three_routes_agree_water_ccpvdz; registry=drpa -->

> **Verified equation 18.5** · the series summand
>
> The series branch of `trace_log_summand_term` equals
> $\ln\lambda + 1 - \lambda$, and the eigenvalue route equals the log-det
> route at each frequency.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_drpa_pdep_sympy.py`, checks `series-summand` and `eig-vs-logdet` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/energy.rs::trace_log_summand_term` |
> | Code test | `water_ccpvdz_aux_route_matches_the_eigensolve_route` in `crates/ferric-rpa/tests/rpa_energy_route.rs` |
>
> <!-- verified: id=rpa.drpa.series-summand; script=scripts/rpa_drpa_pdep_sympy.py; claim=series-summand; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/energy.rs::trace_log_summand_term; rust_test=ferric-rpa/tests/rpa_energy_route.rs::water_ccpvdz_aux_route_matches_the_eigensolve_route; registry=drpa -->

## 18.1.5 The algorithms

**Frequency, aux basis** (`run_pdep_rpa_energy` with `exact_drpa_config`).

| Step | Quantity | Cost |
|---|---|---|
| 1 | RHF; $B^P_{ia}$ (section 6.1) | as RI-MP2 |
| 2 | per node $u_k$: scale $B$ columns by $\sqrt{4\Delta_i^a/(u_k^2 + (\Delta_i^a)^2)}$, symmetric rank-$n_on_v$ update into $\mathbf 1 + \boldsymbol\Pi$ | $N_{\mathrm{aux}}^2n_on_v$ |
| 3 | LU of $\mathbf 1 + \boldsymbol\Pi$, $S(u_k)$ | $N_{\mathrm{aux}}^3$ |
| 4 | $E = \frac{1}{2\pi}\sum_k w_kS(u_k)$ | |

`exact_drpa_config` fixes `trunc_thresh = 0`, Gauss–Legendre with 48 nodes at
$u_0 = 0.5$, the dense $\chi_0$ and no eigenvalue requests, which is exactly
the condition under which `run_pdep_rpa_energy` skips the static eigensolve.
Any other setting falls back to the PDEP route of section 18.3, and the reason
is printed. The log line and the certificate name the route taken.

**Ring amplitudes** (`amplitude_drpa` at $\varepsilon = 0$). Boys-localized
occupied orbitals and projected virtuals (the same localized spaces as local
MP2), $K$ assembled in that basis, the damped fixed point above. The memory
grows as $n_o^3n_v^2$ in the ring products; Python `run_drpa` refuses a run
that cannot fit before the SCF and names `run_pdep_rpa(trunc_thresh=0)` as
the low-memory route to the same energy.

**Plasmon.** A test reference only (section 18.1.1).

## 18.1.6 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | frozen occupied orbitals in all three routes |
| `[local]` | section | absent | the amplitude threshold of section 18.6 |
| `run_drpa(diis=)` | integer, 0 off | 8 | DIIS subspace of the Riccati solve |
| `[memory] budget_gb`, `memory_budget_gb=` | GB | auto | preflight of either route |

## 18.1.7 Numerical anchors

From `site/src/reference/validation.md` (row `drpa` and its anchors) and the
committed tests named there.

| Quantity | Systems | Error | Test |
|---|---|---|---|
| Frequency route vs an independent numpy plasmon dRPA on PySCF integrals | H2, water, n-butane at 6-31G and cc-pVDZ, n-octane at 6-31G, every frozen core | $\le 2.7\times10^{-10}$ Ha | `crates/ferric-rpa/tests/validation_drpa_frequency.rs` |
| Riccati route vs the same references | same | $\le 3.0\times10^{-11}$ Ha | `crates/ferric-mp2/tests/validation_drpa_amplitude.rs` |
| Riccati vs frequency route | water and n-hexane / cc-pVDZ | $\le 4.3\times10^{-11}$ Ha | `crates/ferric-rpa/tests/validation_drpa_frequency.rs` |

## 18.1.8 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-rpa/theory.md`, "RPA Correlation Energy" | (18.1)–(18.4) with their derivation |
| `scripts/rpa_drpa_pdep_sympy.py` | the equivalence of the three forms |
| `site/src/methods/rpa-gw.md`, "Exact and local dRPA" | the user guide |

**Known limits (2026-11-15).** Closed shell and HF reference only for `drpa`.
The plasmon route is $O((n_on_v)^3)$ and exists as a reference. The Riccati
route's $n_o^3n_v^2$ memory limits it to small molecules without `[local]`.
