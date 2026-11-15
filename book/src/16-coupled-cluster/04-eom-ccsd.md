# 16.4 EOM-CCSD singlet excitation energies

<!-- section: 16.4; status: ready; registry: eom-ccsd;
     theory: crates/ferric-cc/theory.md#"EE-EOM-CCSD singlets, closed shell" -->

Equation-of-motion CCSD finds excited states as eigenvectors of the
similarity-transformed Hamiltonian $\bar H = e^{-\hat T}\hat He^{\hat T}$ in
the space of singles and doubles, with the converged ground-state $\hat T$ of
section 16.2. $\bar H$ is not symmetric, so the eigensolver is a
non-symmetric Davidson. ferric computes singlet roots of a closed-shell
reference, right eigenvectors only.

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

## 16.4.1 The equations

<!-- include: crates/ferric-cc/theory.md#"EE-EOM-CCSD singlets, closed shell" -->

The Jacobian identity is what makes the method checkable without diagrams:
at $\Omega(t) = 0$, $\partial\Omega/\partial t$ is the singles-and-doubles
block of $\bar H - E_{CC}$, because
$\Omega_\mu(t + \delta) = \langle\mu|e^{-\hat T}e^{-\hat\delta}\hat He^{\hat\delta}e^{\hat T}|0\rangle$
and the first-order part of that is
$\langle\mu|[\bar H, \hat\delta]|0\rangle = \langle\mu|\bar H\hat\delta|0\rangle - \delta_\mu E_{CC}$
once $\langle\mu|\bar H|0\rangle = 0$.

> **Verified equation 16.6** · $\bar H - E_{CC}$ is the CC Jacobian
>
> On a ten-spin-orbital model, CCSD is converged with the residual of
> section 16.1, $e^{-\hat T}\hat He^{\hat T}$ is built as a $1024\times1024$
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

> **Verified equation 16.7** · the CIS limit and the spin-summed norm
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
below is their evidence.

## 16.4.2 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | converged spin-adapted CCSD (section 16.2) | | `ccsd_closed_shell` |
| 2 | dressed intermediates $F$, $W$ from $t_1, t_2$, once | $n_o^2n_v^4$ (ladder) | `intermediates` |
| 3 | memory plan for the subspace and the ladder route | | `eom_memory_plan` |
| 4 | per product: $\sigma = (\bar H - E_{CC})R$, doubles symmetrized | $n_o^2n_v^4$ | `sigma` |
| 5 | project, `dgeev`, lowest $n$ roots with real part above $10^{-10}$ | $m^3$ | `davidson_nonsymmetric` |
| 6 | Olsen correction, orthogonalize, collapse at `max_space` | | `olsen_correction` |
| 7 | stop when every residual norm is below `conv_tol`; otherwise error at `max_iter` | | |

The particle–particle ladder of the product takes the same dense or RI route
as the CCSD ladder (`EomCcsdResult::ladder_route`). Each returned vector is
scaled to unit spin-summed norm, and `singles_weight` is $2\|r_1\|^2/\|R\|^2$.

## 16.4.3 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[eom] n_roots`, `n_roots=` | integer $\ge 1$ | 5 | lowest singlet roots |
| `[eom] conv_tol`, `conv_tol=` | Hartree, $>0$ | $10^{-6}$ | Davidson residual-norm threshold |
| `[eom] max_iter`, `max_iter=` | integer | 100 | iteration cap; an unconverged root is an error |
| `[eom] max_space`, `max_space=` | integer | $\max(n_{\rm guess} + 10n_{\rm roots}, 50)$ | subspace size before a collapse |
| `[cc] ladder`, `ladder=` | `auto`, `dense`, `ri` | `auto` | ladder route of CCSD and of the product |

The residual threshold bounds the eigenvalue error only to first order in
the residual, since $\bar H$ is not symmetric.

## 16.4.4 Numerical anchors

| Quantity | Systems | Error | Test |
|---|---|---|---|
| lowest five excitation energies vs PySCF `EOMEESinglet` on the same DF integrals | H2O / cc-pVDZ | $\le 2.9\times10^{-10}$ Ha | `crates/ferric-cc/tests/validation_eom_ccsd.rs` |
| the same | C2H4 / cc-pVDZ | $\le 7.4\times10^{-11}$ Ha | `crates/ferric-cc/tests/validation_eom_ccsd.rs` |
| $t = 0$ singles block vs CIS | | $3.6\times10^{-14}$ Ha | `crates/ferric-cc/tests/eom_ccsd_anchors.rs` |
| dense $\bar H$ vs Davidson | | $\le 1.0\times10^{-12}$ Ha | `crates/ferric-cc/tests/eom_ccsd_anchors.rs` |

## 16.4.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "EE-EOM-CCSD singlets, closed shell" | the operator, the norm and the solver |
| Stanton and Bartlett, JCP 98, 7029 (1993); PySCF `cc/eom_rccsd.py` | the method and the closed-shell intermediates |
| `scripts/cc_ccsd_eom_brute_force_sympy.py` | the Jacobian, CIS and norm checks |

**Known limits (2026-11-15).** Singlets of a closed-shell reference only;
right eigenvectors only, so no left vectors, transition densities or
oscillator strengths. The anchors are all-electron; a frozen-core reference
is accepted but not compared.
