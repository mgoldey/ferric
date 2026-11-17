# dRPA three ways: frequency integral, plasmon formula, ring amplitudes {#drpa-three-ways}

<!-- status: ready; registry: drpa -->

MP2 lets two electrons interact through the bare Coulomb interaction, as if
nothing else in the molecule moved. The other electrons do move: they
polarize in response to the pair's charge fluctuation and screen it, and at
long range the correlated motion of these charge fluctuations is
dispersion. The direct random phase approximation (dRPA) builds the
screening in by summing the ring diagrams, in which a particle-hole
fluctuation is handed from pair to pair, to infinite order. Its first term,
at second order, is direct MP2: twice the opposite-spin energy
$E_{\mathrm{OS}}$ of chapter @sec:mp2.

The plan: three expressions for the same energy (@sec:drpa-three-ways-plasmon-formula to @sec:drpa-three-ways-imaginary-frequency), why they
agree (@sec:drpa-three-ways-function-three-algorithms), then the algorithms, options and anchors.

The three expressions are a sum over plasmon excitation energies, the energy
of the direct-ring coupled-cluster doubles (drCCD) amplitudes, and an
integral of the dielectric function over imaginary frequency. They are one
function of the orbitals and integrals. ferric implements all three. It uses
the frequency form for production energies and for the PDEP method (section
@sec:pdep), the amplitude form for the local approximation (section @sec:local-drpa), and
the plasmon form as the reference both are tested against.

This section covers a closed-shell RHF reference and the Coulomb operator.
The open-shell dielectric is section @sec:open-shell-dielectric. Attenuated operators enter only
through the fitted factor $B$ (section @sec:factorization-and-metric).

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

## The plasmon formula {#drpa-three-ways-plasmon-formula}

The RPA excitation energies $\Omega_n$ solve the linear-response eigenproblem

$$ \begin{pmatrix}\mathbf A & \mathbf B\\ -\mathbf B & -\mathbf A\end{pmatrix}\begin{pmatrix}\mathbf X\\ \mathbf Y\end{pmatrix} = \Omega\begin{pmatrix}\mathbf X\\ \mathbf Y\end{pmatrix} . $$

Add the two block rows and subtract them. That gives
$(\mathbf A + \mathbf B)(\mathbf X + \mathbf Y) = \Omega(\mathbf X - \mathbf Y)$ and
$(\mathbf A - \mathbf B)(\mathbf X - \mathbf Y) = \Omega(\mathbf X + \mathbf Y)$.
Substitute the first into the second:

$$ (\mathbf A - \mathbf B)(\mathbf A + \mathbf B)(\mathbf X + \mathbf Y) = \Omega^2(\mathbf X + \mathbf Y) . $$

So the $\Omega_n$ are the square roots of the eigenvalues of a matrix of half
the dimension.

Direct RPA has no exchange. The coupling between the excitations
$i\sigma \to a\sigma$ and $j\tau \to b\tau$ is $(ia|jb)$ for every pair of
spins $\sigma, \tau$, since the Coulomb operator does not see spin. Change to
the sum and the difference of the $\alpha$ and $\beta$ excitations. The
coupling block $\begin{pmatrix}(ia|jb) & (ia|jb)\\ (ia|jb) & (ia|jb)\end{pmatrix}$
becomes $2(ia|jb)$ on the singlet (sum) and zero on the triplet
(difference). The triplet excitations keep $\Omega = \Delta$ and add nothing
to the energy below. With $K_{ia,jb} = 2(ia|jb)$ (chapter @sec:notation, section @sec:notation-orbital-rotations) the
singlet response matrices are

$$ \mathbf A = \boldsymbol\Delta + \mathbf K, \qquad \mathbf B = \mathbf K, \qquad \Delta_{ia,jb} = \Delta_i^a\,\delta_{ij}\delta_{ab}, $$

the excitation energies $\Omega_n$ are the square roots of the eigenvalues of
$(\mathbf A - \mathbf B)(\mathbf A + \mathbf B) = \boldsymbol\Delta(\boldsymbol\Delta + 2\mathbf K)$, and

$$ E_c^{\mathrm{dRPA}} = \tfrac12\Big(\sum_n \Omega_n - \operatorname{tr}\mathbf A\Big) . \label{eq:rpa-drpa-three-routes} $$

$\tfrac12\sum_n\Omega_n$ is the zero-point energy of the coupled
excitations. Subtracting $\tfrac12\operatorname{tr}\mathbf A = \tfrac12\sum_n\Delta_n + \tfrac12\operatorname{tr}\mathbf K$
removes the zero-point energy of the uncoupled ones and the term linear in
$\mathbf K$, so the energy starts at second order. Check it on one mode,
with gap $\Delta$ and coupling $b$:
$\Omega = \sqrt{\Delta(\Delta + 2b)}$, and

$$ \tfrac12\big(\sqrt{\Delta(\Delta + 2b)} - \Delta - b\big) = -\frac{b^2}{4\Delta} + \frac{b^3}{4\Delta^2} - \dots $$

The series alternates. In this one-mode model the third-order ring term
cancels part of the second-order one: that cancellation is the screening.

`canonical_plasmon_drpa` builds $\mathbf A$ and $\mathbf B$ from a global RI
factor. It first re-diagonalizes the occupied and virtual blocks of the
converged Fock matrix, because the energy is first order in any Fock
inconsistency. It then diagonalizes the non-symmetric product with a general
eigensolver. The end of the function, unchanged:

<!-- excerpt: ferric-mp2/src/drpa_amplitude.rs::canonical_plasmon_drpa -->
```rust
let m = (&a_mat - &b_mat).dot(&(&a_mat + &b_mat));
let (w, _) = m
    .eig()
    .map_err(|e| FerricError::General(format!("drpa plasmon eig: {e}")))?;
let mut sum_omega = 0.0;
for lam in w.iter() {
    if lam.im.abs() > 1e-8 * lam.re.abs().max(1.0) || lam.re < 0.0 {
        return Err(FerricError::General(format!(
            "drpa plasmon: non-real/negative Omega^2 = {lam} (RPA instability?)"
        )));
    }
    sum_omega += lam.re.sqrt();
}
let tr_a: f64 = (0..n).map(|k| a_mat[(k, k)]).sum();
Ok(0.5 * (sum_omega - tr_a))
```

| Code | Equation |
|---|---|
| `m` | $(\mathbf A - \mathbf B)(\mathbf A + \mathbf B)$ |
| `w` | the eigenvalues $\Omega_n^2$, from a general (non-symmetric) solver |
| `sum_omega` | $\sum_n\Omega_n$ |
| `tr_a` | $\operatorname{tr}\mathbf A$ |
| the return value | (@eq:rpa-drpa-three-routes) |

An $\Omega_n^2$ that is negative or not real is an RPA instability, and the
function stops with an error instead of taking a square root. The dense
$(n_on_v)^2$ matrix makes this a reference, not a production route.

## Ring amplitudes {#drpa-three-ways-ring-amplitudes}

The same eigenvectors give a second form. Collect the eigenvectors with
$\Omega_n > 0$ as the columns of $\mathbf X$ and $\mathbf Y$, put the
$\Omega_n$ in a diagonal matrix $\boldsymbol\Omega$, and define
$\mathbf T = \mathbf Y\mathbf X^{-1}$.
The first block row of the eigenproblem, $\mathbf A\mathbf X + \mathbf B\mathbf Y = \mathbf X\boldsymbol\Omega$,
becomes $\mathbf X\boldsymbol\Omega\mathbf X^{-1} = \mathbf A + \mathbf B\mathbf T$.
The second, $-\mathbf B\mathbf X - \mathbf A\mathbf Y = \mathbf Y\boldsymbol\Omega$,
becomes $-\mathbf B - \mathbf A\mathbf T = \mathbf T(\mathbf A + \mathbf B\mathbf T)$
after multiplying by $\mathbf X^{-1}$ on the right and substituting the
first. Rearranged:

$$ \mathbf B + \mathbf A\mathbf T + \mathbf T\mathbf A + \mathbf T\mathbf B\mathbf T = 0 . $$

Take the trace of $\mathbf X\boldsymbol\Omega\mathbf X^{-1} = \mathbf A + \mathbf B\mathbf T$:
$\sum_n\Omega_n = \operatorname{tr}\mathbf A + \operatorname{tr}(\mathbf B\mathbf T)$,
so (@eq:rpa-drpa-three-routes) is $\tfrac12\operatorname{tr}(\mathbf B\mathbf T)$, which is
$\tfrac12\sum\mathbf B\circ\mathbf T$ because $\mathbf T$ is symmetric. Insert
$\mathbf A = \boldsymbol\Delta + \mathbf K$, $\mathbf B = \mathbf K$, and
write $\boldsymbol\Delta\mathbf T + \mathbf T\boldsymbol\Delta$ in a form that
holds in any orbital basis. The drCCD amplitudes $T_{ia,jb}$ solve the Riccati
equation

$$ \mathbf R(\mathbf T) = \mathbf K + \mathcal F(\mathbf T) + \mathbf K\mathbf T + \mathbf T\mathbf K + \mathbf T\mathbf K\mathbf T = 0, \qquad E_c^{\mathrm{dRPA}} = \tfrac12\sum_{ia,jb} K_{ia,jb}T_{ia,jb}, \label{eq:drpa-three-ways-2} $$

with the Fock superoperator of a possibly non-canonical (localized) basis

$$ \mathcal F(\mathbf T)_{ia,jb} = \sum_c f_{ac}T_{ic,jb} + \sum_c T_{ia,jc}f_{cb} - \sum_k f_{ik}T_{ka,jb} - \sum_k T_{ia,kb}f_{kj} . $$

Read the terms as diagrams. $\mathbf K$ creates a pair of particle-hole
fluctuations. $\mathcal F$ carries the orbital energies, the same operator
as the local-MP2 residual, equation @eq:amplitude-lmp2-1. $\mathbf K\mathbf T$ and $\mathbf T\mathbf K$
attach one more ring to an existing one, and $\mathbf T\mathbf K\mathbf T$
joins two existing rings through one interaction. Exchange terms are absent,
which is the "direct" in the name.

In canonical orbitals $\mathcal F(\mathbf T) = \Delta^{ab}_{ij}\,T_{ia,jb}$.
Drop the three ring terms. That leaves $T = -K/\Delta$ and the second-order
energy $-\sum 2(ia|jb)^2/\Delta_{ij}^{ab} = 2E_{\mathrm{OS}}$. Compare (@eq:closed-shell-ri-mp2-3):
delete the exchange integral $(ib|ja)$ there and the result is
$2E_{\mathrm{OS}}$. Second-order dRPA is MP2 without exchange, with the
same-spin pairs counted as if they were opposite-spin pairs.
The MP2 code reaches the same number with the opposite sign convention:
`ferric-mp2/src/rimp2.rs::pair_energy` divides $(ia|jb)^2$ by the negative
`denom` $= \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b$, so
$E_{\mathrm{OS}} = -\sum(ia|jb)^2/\Delta_{ij}^{ab}$ comes out negative
without a minus sign in the loop.

Check the Riccati equation on one mode, $\mathbf A = \Delta + b$,
$\mathbf B = b$, $\Delta > 0$. Then $\mathcal F(T) = 2\Delta t$ and
$\mathbf R = 0$ is the quadratic

$$ b + 2(\Delta + b)t + bt^2 = 0 . $$

Of its two roots, take the one that stays finite as $b \to 0$:
$t = \big(\sqrt{\Delta(\Delta + 2b)} - \Delta - b\big)/b$, which goes to 0
with $b$. Then $\tfrac12 bt = \tfrac12\big(\sqrt{\Delta(\Delta + 2b)} - \Delta - b\big)$,
the plasmon value of @sec:drpa-three-ways-plasmon-formula.

The code (`fock_super`, `riccati_masked_solve`) iterates the damped fixed
point $\mathbf T \leftarrow \mathbf T - \mathbf R(\mathbf T)/\bar{\mathbf D}$, where
$\bar{\mathbf D}$ is built from the diagonals of the localized occupied and virtual
Fock blocks,

$$ \bar D_{ia,jb} = F_{aa} + F_{bb} - F_{ii} - F_{jj}, $$

optionally with DIIS on the increment. In canonical orbitals $\bar D$ is
$\Delta_{ij}^{ab}$. `amplitude_drpa_dense` stops with "non-positive
denominator (not a gapped system?)" if any $\bar D_{ia,jb} \le 0$. One iteration of
`ferric-mp2/src/drpa_amplitude.rs::riccati_masked_solve`, with the DIIS step,
the declarations of `r2` and `new_t` and the code's own comments removed:

```rust
let f_t = matvec_indexed(rg, f_oo, &t, &mut flops);       // F(T)
let bt = ring_product_planned(&b_ring_plan, &t);          // K T
let u: Vec<Array2<f64>> = b_blocks.iter().zip(&bt).map(|(b, c)| b + c).collect(); // K + K T
let tu = ring_product(rg, &t, &u);                        // T (K + K T) = T K + T K T
for (p, pb) in rg.pairs.iter().enumerate() {
    let mut r = &f_t[p] + &u[p];                          // F(T) + K + K T
    r += &tu[p];                                          //   + T K + T K T = R(T)
    apply_pattern(&mut r, &pb.pat, pb.db.len());          // the mask of 18.6
    r2 += r.iter().map(|x| x * x).sum::<f64>();           // ‖R‖²
    let mut tn = t[p].clone();
    tn -= &(&r / &pb.denom);                              // T − R(T)/D
    apply_pattern(&mut tn, &pb.pat, pb.db.len());
    new_t.push(tn);
}
```

- **Two ring products, not three.** $\mathbf T\mathbf K + \mathbf T\mathbf K\mathbf T = \mathbf T(\mathbf K + \mathbf K\mathbf T)$,
  so the code forms $\mathbf K\mathbf T$ once and reuses it inside the
  second product, one ring product fewer per iteration.
- **The code calls $\mathbf K$ `b_blocks`.** The module writes
  $B_{iajb} = 2(ia|jb)$ for the same matrix.
- **`apply_pattern` is a no-op at $\varepsilon = 0$.** The same loop runs
  the local approximation of section @sec:local-drpa, where it zeroes the dropped
  amplitudes.

The loop stops at $\lVert\mathbf R\rVert/\lVert\mathbf K\rVert < 10^{-12}$ and
returns an error after 500 iterations. The energy is invariant under
orthogonal rotations within the occupied and within the virtual space, which
is what licenses the Boys-localized basis the local method needs (section
@sec:local-drpa).

## Imaginary frequency {#drpa-three-ways-imaginary-frequency}

The third form works in the auxiliary basis. With the RI factor $B^P_{ia}$ of
section @sec:factorization-and-metric and a real imaginary frequency $u$,

$$ \Pi_{PQ}(iu) = 4\sum_{ia} B^P_{ia}\,\frac{\Delta_i^a}{u^2 + (\Delta_i^a)^2}\,B^Q_{ia}, \label{eq:drpa-three-ways-3} $$

$$ E_c^{\mathrm{dRPA}} = \frac{1}{2\pi}\int_0^\infty \Big[\ln\det\big(\mathbf 1 + \boldsymbol\Pi(iu)\big) - \operatorname{tr}\boldsymbol\Pi(iu)\Big]\,du \approx \frac{1}{2\pi}\sum_k w_k\,S(u_k) . \label{eq:drpa-three-ways-4} $$

$\boldsymbol\Pi = -\tilde{\boldsymbol\chi}_0\mathbf v$ is the
non-interacting response times the Coulomb interaction, written in the
dressed auxiliary basis, and $\mathbf 1 + \boldsymbol\Pi$ is the symmetrized
dielectric matrix $\tilde{\boldsymbol\varepsilon}(iu)$. The prefactor 4 is 2
for the spin sum times 2 for the pair of poles at $\pm\Delta_i^a$.

Why does (@eq:drpa-three-ways-4) equal (@eq:rpa-drpa-three-routes)? Follow it in four steps. Write
$g_{ia} = 4\Delta_i^a/(u^2 + (\Delta_i^a)^2)$, so $\boldsymbol\Pi = B\,\mathrm{diag}(g)\,B^{\mathsf T}$.

1. Move to the $ov$ space with Sylvester's identity
   $\det(\mathbf 1 + \mathbf M\mathbf N) = \det(\mathbf 1 + \mathbf N\mathbf M)$:
   $\det(\mathbf 1 + \boldsymbol\Pi) = \det(\mathbf 1 + g^{1/2}B^{\mathsf T}Bg^{1/2})$,
   and $B^{\mathsf T}B = (ia|jb) = \tfrac12\mathbf K$.
2. Factor out $(u^2 + \boldsymbol\Delta^2)^{-1/2}$ on both sides:
   $\mathbf 1 + g^{1/2}\tfrac12\mathbf Kg^{1/2} = (u^2 + \boldsymbol\Delta^2)^{-1/2}\big[u^2 + \boldsymbol\Delta^{1/2}(\boldsymbol\Delta + 2\mathbf K)\boldsymbol\Delta^{1/2}\big](u^2 + \boldsymbol\Delta^2)^{-1/2}$.
   The bracket has the eigenvalues of $u^2 + \boldsymbol\Delta(\boldsymbol\Delta + 2\mathbf K)$,
   which are $u^2 + \Omega_n^2$. So
   $$ \det(\mathbf 1 + \boldsymbol\Pi(iu)) = \prod_n\frac{u^2 + \Omega_n^2}{u^2 + \Delta_n^2} . $$
3. Integrate each factor:
   $\int_0^\infty\ln\frac{u^2 + \Omega^2}{u^2 + \Delta^2}\,du = \pi(\Omega - \Delta)$.
4. Integrate the trace: $\int_0^\infty \Delta/(u^2 + \Delta^2)\,du = \pi/2$,
   so $\int_0^\infty\operatorname{tr}\boldsymbol\Pi\,du = 2\pi\sum_{ia}(ia|ia) = \pi\operatorname{tr}\mathbf K$.

Divide by $2\pi$ and subtract: $\tfrac12\big(\sum_n\Omega_n - \sum_n\Delta_n - \operatorname{tr}\mathbf K\big)$,
which is (@eq:rpa-drpa-three-routes).

Expand the logarithm and the ring diagrams reappear. Where the series
converges, $\ln\det(\mathbf 1 + \boldsymbol\Pi) - \operatorname{tr}\boldsymbol\Pi = \sum_{n\ge2}(-1)^{n+1}\operatorname{tr}\boldsymbol\Pi^n/n$.
The term with $n$ factors of $\boldsymbol\Pi$ is the ring with $n$ bubbles.
The $n = 2$ term, $-\tfrac12\operatorname{tr}\boldsymbol\Pi^2$, integrates to
$2E_{\mathrm{OS}}$, the second-order term of @sec:drpa-three-ways-ring-amplitudes. Subtracting
$\operatorname{tr}\boldsymbol\Pi$ removes the $n = 1$ term.

The integrand is never positive, since $\ln(1 + x) - x \le 0$ for every
eigenvalue $x \ge 0$ of $\boldsymbol\Pi$. At large $u$, $\boldsymbol\Pi$ falls as $u^{-2}$,
so the integrand, led by $-\tfrac12\operatorname{tr}\boldsymbol\Pi^2$, falls
as $u^{-4}$. The quadrature is section @sec:quadratures.

The summand $S$ is evaluated in one of two ways, which agree per frequency:

| Route | $S(u_k)$ | Used by |
|---|---|---|
| Log-determinant (`ferric-rpa/src/energy.rs::trace_log_summands_of_projection`) | $\ln\det\tilde{\boldsymbol\varepsilon} + \operatorname{tr}(\mathbf 1 - \tilde{\boldsymbol\varepsilon})$, $\tilde{\boldsymbol\varepsilon} = \mathbf 1 + \boldsymbol\Pi$, by an LU factorization | the aux-basis route (`RpaEnergyRoute::AuxBasisLogDet`): CLI `drpa`, and PDEP with every mode kept and no eigenvalues requested |
| Eigenvalues (`ferric-rpa/src/energy.rs::{rpa_correlation_energy,trace_log_sum}`) | $\sum_\xi s(\lambda_\xi)$, $s(\lambda) = \ln\lambda + 1 - \lambda$ | PDEP with truncation, U-PDEP, and any caller that asks for eigenvalues |

$s(\lambda)$ is a difference of nearly equal numbers when $\lambda$ is close
to 1. $\ln\lambda$ and $x = \lambda - 1$ agree to first order, and their
difference, about $-x^2/2$, is formed from two numbers of size $|x|$. At
$x = 10^{-9}$ the literal form has no correct digits. So
`ferric-rpa/src/energy.rs::trace_log_summand_term` switches to the series
$s = -\sum_{k\ge2}(-x)^k/k$ when $|x| < 0.25$, summed to order 30:

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
| 0.25, 30 | `TRACE_LOG_SERIES_CUTOFF`, `TRACE_LOG_SERIES_ORDER` |
| $s(\lambda)$ | the return value |

The factor $x^2$ is pulled out of the sum, so the sum itself starts at
$-\tfrac12$ and no cancellation happens inside it.

## One function, three algorithms {#drpa-three-ways-function-three-algorithms}

The derivations above show the three forms are one function: @sec:drpa-three-ways-ring-amplitudes derives
the Riccati equation and its energy from the plasmon eigenvectors, and
@sec:drpa-three-ways-imaginary-frequency derives the frequency integral from the plasmon eigenvalues. The
proofs below check this on models, and the Rust test runs all three routes
on water.

> **Verified equation @eq:rpa-drpa-three-routes** · plasmon, Riccati and frequency forms agree
>
> For a single mode with gap $\Delta$ and coupling $b$, the plasmon energy
> $\tfrac12(\sqrt{\Delta(\Delta + 2b)} - \Delta - b)$, the Riccati root and
> the frequency integral are one function, with second-order term
> $-b^2/(4\Delta)$. On random closed-shell models each of (@eq:rpa-drpa-three-routes), (@eq:drpa-three-ways-2) in a
> randomly rotated basis, and (@eq:drpa-three-ways-4) equals the spin-orbital definition
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

> **Verified equation {#eq:rpa-drpa-series-summand}** · the series summand
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

The script checks the conclusions of @sec:drpa-three-ways-plasmon-formula to @sec:drpa-three-ways-imaginary-frequency, not the intermediate
steps written out there (the eigenvector form of $\mathbf T$, the determinant
product).

## The algorithms {#drpa-three-ways-algorithms}

Each form costs something different, which is why each has one job.

**Frequency, aux basis** (`run_pdep_rpa_energy` with `exact_drpa_config`).

| Step | Quantity | Cost |
|---|---|---|
| 1 | RHF; $B^P_{ia}$ (section @sec:factorization-and-metric) | as RI-MP2 |
| 2 | per node $u_k$: scale $B$ columns by $\sqrt{4\Delta_i^a/(u_k^2 + (\Delta_i^a)^2)}$, symmetric rank-$n_on_v$ update into $\mathbf 1 + \boldsymbol\Pi$ | $N_{\mathrm{aux}}^2n_on_v$ |
| 3 | LU of $\mathbf 1 + \boldsymbol\Pi$, $S(u_k)$ | $N_{\mathrm{aux}}^3$ |
| 4 | $E = \frac{1}{2\pi}\sum_k w_kS(u_k)$ | |

Step 2 is one symmetric rank-$k$ update (SYRK) of the scaled factor, a BLAS3
operation.

`exact_drpa_config` fixes `trunc_thresh = 0`, Gauss–Legendre with 48 nodes at
$u_0 = 0.5$, the dense $\chi_0$ and no eigenvalue requests. That is exactly
the condition under which `run_pdep_rpa_energy` skips the static eigensolve.
Any other setting falls back to the PDEP route of section @sec:pdep, and the
reason is printed. The log line and the certificate name the route taken.

**Ring amplitudes** (`amplitude_drpa` at $\varepsilon = 0$). Boys-localized
occupied orbitals and projected virtuals (the same localized spaces as local
MP2, section @sec:amplitude-lmp2), $\mathbf K$ assembled in that basis, and the damped fixed
point above. The memory grows as $n_o^3n_v^2$ in the ring products. Python
`run_drpa` refuses a run that cannot fit before the SCF and names
`run_pdep_rpa(trunc_thresh=0)` as the low-memory route to the same energy.

**Plasmon.** A test reference only (section @sec:drpa-three-ways-plasmon-formula).

## Options {#drpa-three-ways-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | frozen occupied orbitals in all three routes |
| `[local]` | section | absent | the amplitude threshold of section @sec:local-drpa |
| `run_drpa(diis=)` | integer, 0 off | 8 | DIIS subspace of the Riccati solve |
| `[memory] budget_gb`, `memory_budget_gb=` | GB | auto | preflight of either route |

## Numerical anchors {#drpa-three-ways-numerical-anchors}

From `site/src/reference/validation.md` (row `drpa` and its anchors) and the
committed tests named there.

| Quantity | Systems | Error | Test |
|---|---|---|---|
| Frequency route vs an independent numpy plasmon dRPA on PySCF integrals | H2, water, n-butane at 6-31G and cc-pVDZ, n-octane at 6-31G, every frozen core | $\le 2.7\times10^{-10}$ Ha | `crates/ferric-rpa/tests/validation_drpa_frequency.rs` |
| Riccati route vs the same references | same | $\le 3.0\times10^{-11}$ Ha | `crates/ferric-mp2/tests/validation_drpa_amplitude.rs` |
| Riccati vs frequency route | water and n-hexane / cc-pVDZ | $\le 4.3\times10^{-11}$ Ha | `crates/ferric-rpa/tests/validation_drpa_frequency.rs` |

The frequency-route bar, $2.7\times10^{-10}$ Ha, is the number the
quadrature convergence table gives at 48 nodes (section @sec:quadratures). The Riccati
route has no quadrature.

## Sources {#drpa-three-ways-sources}

| Source | What it gives |
|---|---|
| `scripts/rpa_drpa_pdep_sympy.py` | the equivalence of the three forms |
| `site/src/methods/rpa-gw.md`, "Exact and local dRPA" | the user guide |
| Eshuis, Bates and Furche, Theor. Chem. Acc. 131, 1084 (2012) | review of RI-RPA: the frequency form in the auxiliary basis |

**Known limits (2026-11-15).** Closed shell and HF reference only for `drpa`.
The plasmon route is $O((n_on_v)^3)$ and exists as a reference. The Riccati
route's $n_o^3n_v^2$ memory limits it to small molecules without `[local]`.
