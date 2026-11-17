# EOM-CCSD singlet excitation energies {#eom-ccsd}

<!-- status: ready; registry: eom-ccsd;
     theory: crates/ferric-cc/theory.md#"EE-EOM-CCSD singlets, closed shell" -->

An excited state differs from the ground state mostly by one electron
promoted, but the other electrons stay correlated around it. Equation-of-motion
CCSD keeps the converged ground-state correlation $e^{\hat T}$ of section
@sec:closed-shell-ccsd and applies a linear excitation operator on top, so the excitation
energies are eigenvalues of the similarity-transformed Hamiltonian
$\bar H = e^{-\hat T}\hat He^{\hat T}$ in the space of singles and doubles.
That matrix is also the derivative of the CC residual, so EOM-CCSD is the
linear response of the CC state. $\bar H$ is not symmetric, so the
eigensolver is a non-symmetric Davidson, and ferric computes singlet roots
of a closed-shell reference, right eigenvectors only.

The section has three parts: the eigenvalue problem and its two limits
(@sec:eom-ccsd-equations), the solver (@sec:eom-ccsd-algorithm) and its options (@sec:eom-ccsd-options).

| Index | Range | Code name |
|---|---|---|
| $i, j$ / $a, b$ | active occupied / virtual spatial orbitals | `0..no`, `0..nv` |
| $r_i^a$, $r_{ij}^{ab} = r_{ji}^{ba}$ | singlet right vector | `r1[i,a]`, `r2[i,j,a,b]` |
| $\omega$ | excitation energy (Hartree) | `excitation_energies` |

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/eom_ccsd.rs::{eom_ccsd_singlets,EomCcsdOperator}`, `ferric-cc/src/eom_davidson.rs::davidson_nonsymmetric` |
| CLI | `method.kind = "eom-ccsd"` with `[eom]` (`examples/water-eom-ccsd.toml`) |
| Python | `run_eom_ccsd` |

## The equations {#eom-ccsd-equations}

**The eigenvalue problem.** Write the excited state as
$|\Psi_k\rangle = \hat R_k e^{\hat T}|\Phi_0\rangle$, with $\hat R_k$ a
linear combination of the reference, singles and doubles. $\hat R_k$ and
$\hat T$ are both excitation operators, so they commute. Insert into
$\hat H|\Psi_k\rangle = E_k|\Psi_k\rangle$, move $e^{\hat T}$ to the left of
$\hat R_k$, and multiply by $e^{-\hat T}$:

$$ \bar H\hat R_k|\Phi_0\rangle = E_k\,\hat R_k|\Phi_0\rangle . $$

Project onto the singles and doubles $\langle\mu|$ and subtract
$\hat R_k\bar H|\Phi_0\rangle$. On that space $\bar H|\Phi_0\rangle$ has
only its $E_{CC}$ component, since the CC equations zero its singles and
doubles projections and an excitation operator cannot lower a higher
excitation into the space. What remains is an eigenvalue problem for the
excitation energy $\omega_k = E_k - E_{CC}$:
the excitation energies are the eigenvalues of $\bar H - E_{CC}$ on singlet
singles and doubles $r_i^a$, $r_{ij}^{ab} = r_{ji}^{ba}$
(`ferric-cc/src/eom_ccsd.rs`; Stanton and Bartlett 1993; closed-shell
intermediates and product of PySCF `eom_rccsd._IMDS.make_ee` and
`eeccsd_matvec_singlet`).

**The CC Jacobian.** At converged amplitudes that matrix is the derivative
of the CC residual:

$$(\bar H - E_{CC})\, r = \frac{\partial \Omega(t)}{\partial t}\, r,$$

where $\Omega$ is the closed-shell CCSD residual. Expand the residual at a
displaced amplitude:
$\Omega_\mu(t + \delta) = \langle\mu|e^{-\hat T}e^{-\hat\delta}\hat He^{\hat\delta}e^{\hat T}|0\rangle$,
whose first-order part is
$\langle\mu|[\bar H, \hat\delta]|0\rangle = \langle\mu|\bar H\hat\delta|0\rangle - \delta_\mu E_{CC}$
once $\langle\mu|\bar H|0\rangle = 0$. So the excitation energies measure how
the ground-state equations respond to a change of the amplitudes, and the
identity makes the method checkable without diagrams. The prototype
(`scripts/prototypes/eom_ccsd_proto.py`) checks the product against this
Jacobian by a 5-point central difference, which is exact for the quartic
$\Omega$.

> **Verified equation {#eq:cc-eom-jacobian}** · $\bar H - E_{CC}$ is the CC Jacobian
>
> On a ten-spin-orbital model, CCSD is converged with the residual of
> section @sec:ccd-ccsd-spin-orbital, $e^{-\hat T}\hat He^{\hat T}$ is built as a $1024\times1024$
> matrix from Jordan–Wigner operators, and its block on the singles and
> doubles determinants minus $E_{CC}$ equals the 5-point central-difference
> Jacobian of the residual. The stencil is exact for a quartic residual, so
> the agreement is to round-off.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_ccsd_eom_brute_force_sympy.py`, check `eom-jacobian` |
> | Negative control | the runner's mutation: $+E_{CC}$ for $-E_{CC}$ on the diagonal |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/eom_ccsd.rs::{sigma,intermediates}` |
> | Code test | `validation_eom_ccsd_h2o_ccpvdz` in `crates/ferric-cc/tests/validation_eom_ccsd.rs` |
>
> <!-- verified: id=cc.eom.jacobian; script=scripts/cc_ccsd_eom_brute_force_sympy.py; claim=eom-jacobian; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/eom_ccsd.rs::{sigma,intermediates}; rust_test=ferric-cc/tests/validation_eom_ccsd.rs::validation_eom_ccsd_h2o_ccpvdz; registry=eom-ccsd -->

**The product.** `sigma` forms $(\bar H - E_{CC})R$ without building the
matrix, from the dressed intermediates of `intermediates`: $F_{oo}$,
$F_{vv}$, $F_{ov}$, $W_{mnij}$, $W_{mbij}$, $W_{ejab}$ and the two $W_{mbej}$
couplings. $\tau$ with a right vector is `make_tau`,
$\theta = 2r_{ij}^{ab} - r_{ij}^{ba}$ is `theta_of`, and the doubles part is
symmetrized at the end, $\sigma_{ijab} \leftarrow \sigma_{ijab} + \sigma_{jiba}$.
The particle–particle ladder takes the dense $(ae|bf)$ block or RI panels,
the same choice as the CCSD ladder.

**The CIS limit.** At $t = 0$, $\bar H = \hat H$, and the singles–singles
block is the CIS matrix

$$ \delta_{ij}\delta_{ab}(\varepsilon_a - \varepsilon_i) + 2(ia|jb) - (ij|ab) . $$

The factor 2 is the singlet spin coupling. The singlet combines the α and β
singles in phase. $(ia|jb)$ couples an α single to both the α and the β
singles, while $(ij|ab)$ couples only singles of the same spin, so the
singlet block has $2(ia|jb) - (ij|ab)$.

**The norm.** The packed vector (`pack`) is $r_1$ then the lower triangle of
$M_{ia,jb} = r_{ij}^{ab}$. The singles weight needs the norm of the state,
which is the norm of its spin-orbital expansion over distinct determinants.
Count them:

- each singles $r_i^a$ appears on an α and a β determinant: $2\|r_1\|^2$;
- each opposite-spin double ($i\alpha\to a\alpha$, $j\beta\to b\beta$) is a
  distinct determinant with coefficient $r_{ij}^{ab}$: $\|r_2\|^2$;
- a same-spin double has coefficient $r_{ij}^{ab} - r_{ij}^{ba}$,
  antisymmetric in $ij$ and in $ab$, so the distinct determinants
  $i<j$, $a<b$ carry a quarter of the full sum, and the two spins double
  it: $\tfrac12\|r_2 - r_2^{(ab)}\|^2$.

Adding them (`spin_norm`):

$$\|R\|^2 = 2\|r_1\|^2 + \|r_2\|^2 + \tfrac12 \|r_2 - r_2^{(ab)}\|^2,$$

with $r_2^{(ab)}$ the vector with $a,b$ swapped.

<!-- excerpt: ferric-cc/src/eom_ccsd.rs::spin_norm -->
```rust
    let s1 = 2.0 * r1.iter().map(|x| x * x).sum::<f64>();
    let ab = r2.iter().map(|x| x * x).sum::<f64>();
    let swapped = r2.view().permuted_axes([0, 1, 3, 2]);
    let same: f64 = r2
        .iter()
        .zip(swapped.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    let total = s1 + ab + 0.5 * same;
    (s1 / total, 1.0 / total.sqrt())
```

| Symbol | Code |
|---|---|
| $2\|r_1\|^2$ | `s1` |
| $\|r_2\|^2$ | `ab` |
| $r_2^{(ab)}$ | `swapped`, a view with the last two axes exchanged |
| $\|r_2 - r_2^{(ab)}\|^2$ | `same` |

The function returns the singles weight $2\|r_1\|^2/\|R\|^2$ and the factor
that scales $R$ to unit norm.

> **Verified equation {#eq:cc-eom-cis-limit}** · the CIS limit and the spin-summed norm
>
> At $t = 0$ the singlet singles block of the brute-force $\bar H$ is
> $\delta_{ij}\delta_{ab}(\varepsilon_a - \varepsilon_i) + 2(ia|jb) - (ij|ab)$.
> For random singlet $(r_1, r_2)$, the squared norm of the spin-orbital
> expansion over distinct determinants is
> $2\|r_1\|^2 + \|r_2\|^2 + \tfrac12\|r_2 - r_2^{(ab)}\|^2$.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_ccsd_eom_brute_force_sympy.py`, checks `cis-limit`, `spin-norm` |
> | Negative control | the runner's mutations: Coulomb weight 1 instead of 2; $\tfrac12$ to 1 in the norm |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/eom_ccsd.rs::{EomCcsdOperator,spin_norm}` |
> | Code test | `t_zero_singles_block_is_cis` in `crates/ferric-cc/tests/eom_ccsd_anchors.rs` |
>
> <!-- verified: id=cc.eom.cis-limit; script=scripts/cc_ccsd_eom_brute_force_sympy.py; claim=cis-limit; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/eom_ccsd.rs::{EomCcsdOperator,spin_norm}; rust_test=ferric-cc/tests/eom_ccsd_anchors.rs::t_zero_singles_block_is_cis; registry=eom-ccsd -->

The boxes prove the matrix the code must represent and the CIS and norm
pieces of the product. The closed-shell intermediates of `sigma` are a
transcription of PySCF `eom_rccsd`; the comparison with PySCF in the anchors
below (`tests/validation_eom_ccsd.rs`) and the CIS block at $t=0$
(`tests/eom_ccsd_anchors.rs`) are their evidence.

## The algorithm {#eom-ccsd-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | converged spin-adapted CCSD (section @sec:closed-shell-ccsd) | | `ccsd_closed_shell` |
| 2 | dressed intermediates $F$, $W$ from $t_1, t_2$, once | $n_o^2n_v^4$ (ladder) | `intermediates` |
| 3 | memory plan for the subspace and the ladder route | | `eom_memory_plan` |
| 4 | per product: $\sigma = (\bar H - E_{CC})R$, doubles symmetrized | $n_o^2n_v^4$ | `sigma` |
| 5 | project, `dgeev`, lowest $n$ roots with real part above $10^{-10}$ | $m^3$ | `davidson_nonsymmetric` |
| 6 | Olsen correction, orthogonalize, collapse at `max_space` | | `olsen_correction` |
| 7 | stop when every residual norm is below `conv_tol`; otherwise error at `max_iter` | | |

**The correction vector.** For a Ritz pair $(\theta, x)$ with residual $r$,
`ferric-cc/src/eom_davidson.rs::olsen_correction` adds

$$\delta = -(\Delta - \theta)^{-1} r + \epsilon (\Delta - \theta)^{-1} x,\qquad
\epsilon = \frac{x^\top (\Delta-\theta)^{-1} r}{x^\top (\Delta-\theta)^{-1} x},$$

where $\Delta$ is the diagonal $F_{vv,aa} - F_{oo,ii}$ (doubles: the sum over
both pairs). Multiply by $x^\top$: the two terms cancel by the choice of
$\epsilon$, so $x^\top\delta = 0$. The plain preconditioned residual
$-(\Delta-\theta)^{-1}r$ has a component along $x$, the vector already in
the subspace; the second term removes it. The code forms
$1/(\theta - \Delta)$ once, with the denominator floored at
$10^{-8}$ in magnitude (`PRECOND_FLOOR`), and falls back to the plain
preconditioned residual when $x^\top(\Delta-\theta)^{-1}x$ is zero or not
finite.

The particle–particle ladder of the product takes the same dense or RI route
as the CCSD ladder (`EomCcsdResult::ladder_route`). Each returned vector is
scaled to unit spin-summed norm, and `singles_weight` is $2\|r_1\|^2/\|R\|^2$.

## Options {#eom-ccsd-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[eom] n_roots`, `n_roots=` | integer $\ge 1$ | 5 | lowest singlet roots |
| `[eom] conv_tol`, `conv_tol=` | Hartree, $>0$ | $10^{-6}$ | Davidson residual-norm threshold |
| `[eom] max_iter`, `max_iter=` | integer | 100 | iteration cap; an unconverged root is an error |
| `[eom] max_space`, `max_space=` | integer | $\max(n_{\rm guess} + 10n_{\rm roots}, 50)$ | subspace size before a collapse |
| `[cc] ladder`, `ladder=` | `auto`, `dense`, `ri` | `auto` | ladder route of CCSD and of the product |

The residual threshold bounds the eigenvalue error only to first order in
the residual, since $\bar H$ is not symmetric. For a symmetric matrix the
eigenvalue error would be second order in $\|r\|$.

## Numerical anchors {#eom-ccsd-numerical-anchors}

| Quantity | Systems | Error | Test |
|---|---|---|---|
| lowest five excitation energies vs PySCF `EOMEESinglet` on the same DF integrals | H2O / cc-pVDZ | $\le 2.9\times10^{-10}$ Ha | `crates/ferric-cc/tests/validation_eom_ccsd.rs` |
| the same | C2H4 / cc-pVDZ | $\le 7.4\times10^{-11}$ Ha | `crates/ferric-cc/tests/validation_eom_ccsd.rs` |
| $t = 0$ singles block vs CIS | | $3.6\times10^{-14}$ Ha | `crates/ferric-cc/tests/eom_ccsd_anchors.rs` |
| dense $\bar H$ vs Davidson | | $\le 1.0\times10^{-12}$ Ha | `crates/ferric-cc/tests/eom_ccsd_anchors.rs` |

## Sources {#eom-ccsd-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "EE-EOM-CCSD singlets, closed shell" | the operator, the norm and the solver |
| Stanton and Bartlett, JCP 98, 7029 (1993); PySCF `cc/eom_rccsd.py` | the method and the closed-shell intermediates |
| `scripts/cc_ccsd_eom_brute_force_sympy.py` | the Jacobian, CIS and norm checks |

**Known limits (2026-11-15).** Singlets of a closed-shell reference only;
right eigenvectors only, so no left vectors, transition densities or
oscillator strengths. The anchors are all-electron; a frozen-core reference
is accepted but not compared.
