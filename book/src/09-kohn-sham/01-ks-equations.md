# The Kohn–Sham equations and functional families {#ks-equations}

<!-- status: ready; registry: ks-semilocal, ks-mgga -->

Kohn and Sham replace the interacting electrons by non-interacting ones that
move in one effective potential, chosen so that the two systems have the
same density. The non-interacting system is a single determinant, with the
orbitals, kinetic energy and Coulomb energy of a Hartree–Fock-like
calculation. Exchange, correlation and the difference between the
interacting and non-interacting kinetic energies go into one term, the
exchange-correlation energy $E_{xc}$, a functional of the density. The
approximate functionals for $E_{xc}$ are what the rest of this chapter
evaluates.

The plan: derive the Fock matrix from the energy (@sec:ks-equations-energy-fock-matrix), follow one SCF
iteration through the code (@sec:ks-equations-scf-iteration), and list the functional families
(@sec:ks-equations-functional-families).

For the code, Kohn–Sham is Hartree–Fock with one extra term. ferric has no
separate Kohn–Sham solver. The RHF, UHF and ROHF drivers run RKS, UKS and
ROKS when the configuration names a functional. Two things change: an
exchange-correlation energy and matrix are added once per iteration, and the
exact-exchange matrix is scaled by the functional's fractions.

| Symbol | Meaning | Code name |
|---|---|---|
| $D$ | closed-shell total density, $2\mathbf C_o\mathbf C_o^{\mathsf T}$ | `d` |
| $D^\sigma$, $D^{\rm t}$ | spin densities, their sum | `d_a`, `d_b` |
| $h$ | one-electron operator (kinetic, nuclear, ECP, external) | `h` |
| $J$, $K_{\rm tot}$ | Coulomb matrix, exact exchange with the functional's fractions | `j`, `k` |
| $E_{xc}$, $V_{xc}$ | XC energy and matrix on the grid | `e_xc`, `vxc` |
| $F_{\rm no\,xc}$ | Fock matrix before $V_{xc}$ is added | `f` at the energy trace |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/rhf.rs::solve_rhf`, `ferric-scf/src/uhf.rs::solve_uhf`, `ferric-scf/src/rohf.rs::solve_rohf` with `RhfConfig.xc`; `ferric-dft/src/ks.rs::{KsXc,KsXcUks}`; `ferric-scf/src/rhf.rs::ks_vxc_closed` outside the loop |
| CLI | `method.kind = "ksdft"` (RKS, UKS above multiplicity 1), or `rhf` / `uhf` / `rohf` with `[dft] functional` |
| Python | `run_dft`, `run_ksdft`; `run_rhf`, `run_uhf`, `run_rohf` with `xc=` |

## Energy and Fock matrix {#ks-equations-energy-fock-matrix}

Take a closed shell with total density matrix
$D = 2\mathbf C_o\mathbf C_o^{\mathsf T}$. The Kohn–Sham energy is

$$ E_{\rm KS}[D] = \mathrm{Tr}[Dh] + \tfrac12\mathrm{Tr}[DJ(D)] - \tfrac14\,\mathrm{Tr}[D\,K_{\rm tot}(D)] + E_{xc}[\rho_D] + E_{nn} . $$

Term by term:

- $h$ is the one-electron operator: kinetic energy plus nuclear attraction,
  and the ECP and external-field terms when a run has them.
- $J(D)_{\mu\nu} = \sum_{\lambda\sigma}(\mu\nu|\lambda\sigma)D_{\lambda\sigma}$
  is the Coulomb matrix, so $\tfrac12\mathrm{Tr}[DJ]$ is the classical
  repulsion of the density with itself.
- $K_{\rm tot}$ is the exact-exchange matrix of a hybrid or range-separated
  hybrid, with the functional's fractions built in (section @sec:hybrids). It is zero
  for a pure functional.
- $E_{xc}$ is the grid quadrature of section @sec:vxc, a function of $D$ through
  the density $\rho_D$ and its derivatives on the grid.

The Fock matrix is the derivative of this energy with respect to $D$.
Differentiate term by term:

- $\partial\,\mathrm{Tr}[Dh]/\partial D_{\mu\nu} = h_{\mu\nu}$.
- $\tfrac12\mathrm{Tr}[DJ(D)] = \tfrac12\sum D_{\mu\nu}(\mu\nu|\lambda\sigma)D_{\lambda\sigma}$
  is quadratic in $D$. Its derivative has two terms, one from each $D$, and
  they are equal because $(\mu\nu|\lambda\sigma) = (\lambda\sigma|\mu\nu)$.
  The $\tfrac12$ cancels and the result is $J(D)_{\mu\nu}$.
- The exchange term is quadratic in the same way, so its $\tfrac14$ becomes
  $\tfrac12$: the derivative is $-\tfrac12K_{\rm tot}(D)$ (verified equation
  @eq:ks-rsh-exchange-derivative proves it for both range-separated kernels).
- The derivative of $E_{xc}$ defines the XC matrix,
  $(V_{xc})_{\mu\nu} = \partial E_{xc}/\partial D_{\mu\nu}$. Section @sec:vxc
  computes it.

So

$$ F = h + J(D) - \tfrac12K_{\rm tot}(D) + V_{xc} . $$

Now make $E_{\rm KS}$ stationary under variations of $\mathbf C_o$ that keep
the orbitals orthonormal, $\mathbf C_o^{\mathsf T}S\mathbf C_o = 1$. Add a
symmetric matrix of Lagrange multipliers for the constraint and set the
derivative with respect to $\mathbf C_o$ to zero. Because
$\mathrm dE_{\rm KS} = \mathrm{Tr}[F\,\mathrm dD]$, the result is
$F\mathbf C_o = S\mathbf C_o\boldsymbol\varepsilon_o$, the same condition as
for Hartree–Fock (section @sec:energy-functionals) with this $F$. The occupied rotation that
diagonalizes $\boldsymbol\varepsilon_o$ gives the canonical Kohn–Sham
equations, written for all orbitals as

$$ F\mathbf C = S\mathbf C\boldsymbol\varepsilon . $$

Check the two ends. With no functional, $E_{xc} = 0$ and $K_{\rm tot}$ is the
full exchange matrix (section @sec:hybrids gives it the fractions $(1, 1, 0)$), so
$F$ is the Hartree–Fock Fock matrix. With a pure functional,
$K_{\rm tot} = 0$, and all of exchange comes from $V_{xc}$.

**Open shell (UKS, ROKS).** Each spin has its own density,
$D^\sigma = \mathbf C^\sigma_o\mathbf C^{\sigma\,\mathsf T}_o$, and
$D^{\rm t} = D^\alpha + D^\beta$. Coulomb repulsion sees the total density.
Exchange acts within one spin:

$$ E_{\rm KS} = \sum_\sigma\mathrm{Tr}[D^\sigma h] + \tfrac12\mathrm{Tr}[D^{\rm t}J(D^{\rm t})] - \tfrac12\sum_\sigma\mathrm{Tr}[D^\sigma K_{\rm tot}(D^\sigma)] + E_{xc}[\rho_\alpha,\rho_\beta] + E_{nn} . $$

Differentiate with respect to $D^\sigma$. The Coulomb term gives
$J(D^{\rm t})$, since $\partial D^{\rm t}/\partial D^\sigma = 1$. The
exchange term involves only $D^\sigma$ and gives $-K_{\rm tot}(D^\sigma)$:

$$ F_\sigma = h + J(D^{\rm t}) - K_{\rm tot}(D^\sigma) + V^\sigma_{xc} . $$

At $D^\alpha = D^\beta = D/2$ this is the closed-shell result:
$-K_{\rm tot}(D/2) = -\tfrac12K_{\rm tot}(D)$, and the exchange energy
$-\tfrac12\cdot 2\cdot\mathrm{Tr}[\tfrac D2K_{\rm tot}(\tfrac D2)] = -\tfrac14\mathrm{Tr}[DK_{\rm tot}(D)]$.

> **Verified equation {#eq:ks-fock-is-de-dd}** · the Fock matrix is the derivative of the energy
>
> $F = h + J(D) - \tfrac12K_{\rm tot}(D) + V_{xc}$ with
> $V_{xc} = \partial E_{xc}/\partial D$ is $\partial E_{\rm KS}/\partial D$:
> for the exchange part with generic 8-fold-symmetric integrals of both
> range-separated kernels, symbolically (closed and open shell), and
> numerically by central differences of the total electronic energy against
> $\mathrm{Tr}[F\,\delta D]$ for RHF with a range-separated hybrid, a global
> hybrid and HF; for the XC part on a meta-GGA-like model (section @sec:vxc).
>
> | | |
> |---|---|
> | Proof | `scripts/dft_hybrid_rsh_sympy.py`, claims A3 and B4; `scripts/dft_vxc_matrix_elements_sympy.py`, claim numeric |
> | Negative control | the runner's mutation of the energy trace, $\tfrac12\mathrm{Tr}[D(h+F)] \to \mathrm{Tr}[D(h+F)]$, fails B4 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/rhf.rs::RhfState::energy`, `ferric-scf/src/fock_assembly.rs::subtract_rsh_exchange` |
> | Code test | `ks_vxc_closed_is_the_scf_fock_xc_term_pbe0` in `crates/ferric-scf/src/ks_vxc_tests.rs` |
>
> <!-- verified: id=ks.fock-is-dE-dD; script=scripts/dft_hybrid_rsh_sympy.py; claim=B4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/rhf.rs::RhfState::energy; rust_test=ferric-scf/src/ks_vxc_tests.rs::ks_vxc_closed_is_the_scf_fock_xc_term_pbe0; registry=ks-semilocal -->

In the code, `solve_rhf`, `solve_uhf` and `solve_rohf` run Kohn–Sham when
`RhfConfig.xc` names a functional. They add $(E_{xc}, V_{xc})$ once per
iteration through `ferric-dft/src/ks.rs::KsXc` (closed shell, `add_xc`) or
`KsXcUks` (`add_xc_uks`). Both route to the batched path,
`ferric-dft/src/xc_batch.rs` (section @sec:vxc-batched-algorithm).
`ferric-scf/src/rhf.rs::ks_vxc_closed` returns the same closed-shell pair
outside the loop. The test `ks_vxc_tests` pins it against the SCF's own Fock
matrix.

## One SCF iteration {#ks-equations-scf-iteration}

| Step | What | Function |
|---|---|---|
| 1 | $J(D)$ and, for a hybrid, $K$ (or the short- and long-range pair) | the run's J and K builders; `subtract_rsh_exchange` |
| 2 | $F_{\rm no\,xc} = h + J - \tfrac12K_{\rm tot}$ | `RhfJk::build_fock` |
| 3 | $e_{\rm no\,xc} = \tfrac12\mathrm{Tr}[D(h + F_{\rm no\,xc})]$ | `RhfState::energy` |
| 4 | $(E_{xc}, V_{xc})$ on the grid, $F \mathrel{+}= V_{xc}$ | `KsXc::add_xc` (section @sec:vxc) |
| 5 | $E = e_{\rm no\,xc} + E_{xc} + E_{nn}$ (+ solvation, embedding, hook) | `RhfState::energy` |
| 6 | DIIS, diagonalize, new $D$ | as for HF |

Step 3 reads the one- and two-electron energy off the Fock matrix. Expand
it: $\tfrac12\mathrm{Tr}[D(h + h + J - \tfrac12K_{\rm tot})] = \mathrm{Tr}[Dh] + \tfrac12\mathrm{Tr}[DJ] - \tfrac14\mathrm{Tr}[DK_{\rm tot}]$.
The $\tfrac12$ appears once, on the trace, and undoes the doubling of the
quadratic terms in $F$. Here is that line of
`ferric-scf/src/rhf.rs::RhfState::energy`:

<!-- excerpt: ferric-scf/src/rhf.rs::RhfState::energy -->
```rust
let e_elec_no_xc: f64 = 0.5 * (d * &(&env.h + &*f)).sum();
```

and the return, after $V_{xc}$ has gone into `f`:

<!-- excerpt: ferric-scf/src/rhf.rs::RhfState::energy -->
```rust
Ok(e_elec_no_xc + e_xc + e_hook + e_cosmo + e_pcm + e_pol + env.vnn)
```

| Symbol | Code |
|---|---|
| $D$, $h$, $F_{\rm no\,xc}$ | `d`, `env.h`, `f` |
| $E_{xc}$, $E_{nn}$ | `e_xc`, `env.vnn` |

**Why add $V_{xc}$ after the trace?** If $V_{xc}$ were already in $F$, the
trace would contribute $\tfrac12\mathrm{Tr}[DV_{xc}]$ in place of
$E_{xc}$. The two are different numbers. On a grid,
$\mathrm{Tr}[DV_{xc}] = \sum_g w_g\,\rho_g v_{\rho,g}$ for an LDA (section
@sec:vxc), and that equals $E_{xc} = \sum_g w_g f(\rho_g)$ only when
$\rho\,\partial f/\partial\rho = f$, that is, when $f$ is homogeneous of
degree one in $\rho$. Slater exchange, $f = -C_x\rho^{4/3}$, is homogeneous
of degree $\tfrac43$, so $\mathrm{Tr}[DV_x] = \tfrac43E_x$. The energy
therefore takes $E_{xc}$ from the grid as its own term, and $V_{xc}$ is
added to $F$ after the trace. Solvation, embedding and the Fock hook follow
the same rule: their potentials go into $F$, their energies are separate
terms.

**The density floor.** $E_{xc} = \sum_g w_g\rho_g\epsilon_g$ is summed over
every grid point; libxc returns $\epsilon = 0$ below its own per-functional
threshold. $V_{xc}$, the XC gradient and the response kernels skip points
whose $\rho$ (per spin, $\rho_\sigma$) is at or below
`DensityFloor::for_xc`: $10^{-14}$ for LDA, GGA and their hybrids,
$10^{-10}$ when any component is a meta-GGA. The energy and the matrix
therefore differ by at most the floored points' contribution. The tests
bound that difference through $\mathrm{Tr}[V_{xc}X] = \mathrm dE_{xc}/\mathrm dt$
at $10^{-9}$ relative (section @sec:vxc-matrix-elements).

**Reproducibility.** $E_{xc}$ is a grouped, fixed-order sum over points
(`deterministic_point_sum`), so it is bit-identical across thread counts. A
meta-GGA SCF gets a default virtual-block level shift of 0.5 when none is
set, ramped to zero at convergence.

## Functional families {#ks-equations-functional-families}

ferric evaluates every functional through libxc: LDA $\epsilon(\rho)$, GGA
$\epsilon(\rho, \sigma)$, hybrids and range-separated hybrids (the SCF builds
the exact-exchange part), $\tau$-only meta-GGAs (SCAN, r2SCAN, TPSS) and VV10
nonlocal correlation. Each family adds one ingredient to the one before.
LDA sees only the density at a point. GGA adds the gradient through
$\sigma = \lvert\nabla\rho\rvert^2$. A meta-GGA adds the kinetic-energy
density $\tau$ (section @sec:vxc-grid-quantities). A hybrid mixes in exact exchange, which is
not a local function of the density at all.

A definition, `ferric-dft/src/libxc.rs::XcDef`, is one or more libxc
components and a family, `ferric-dft/src/libxc.rs::FunctionalFamily`. The
family fixes which variables the grid evaluates and which terms of $V_{xc}$
exist:

| Family | Variables per point | $V_{xc}$ terms | Exact exchange | Friendly names (`FRIENDLY_NAMES`) |
|---|---|---|---|---|
| `Lda` | $\rho$ | $v_\rho$ | none | LDA, SVWN |
| `Gga` | $\rho,\sigma$ | $v_\rho, v_\sigma$ | none | PBE, PBEsol, BLYP, BP86 |
| `HybridGga` | $\rho,\sigma$ | $v_\rho, v_\sigma$ | global $a$ | PBE0, B3LYP |
| `RangeSepGga` | $\rho,\sigma$ | $v_\rho, v_\sigma$ | $c_{\rm sr}, c_{\rm lr}, \omega$ | HSE06, ωB97X-V (+VV10) |
| `MetaGga` | $\rho,\sigma,\tau$ | $v_\rho, v_\sigma, v_\tau$ | none, global or CAM | SCAN, r2SCAN, TPSS, TPSSh, M06-2X, SCAN0, r2SCAN0, ωB97M-V |

Any libxc name is also accepted directly. A meta-GGA definition
(`ferric-dft/src/libxc.rs::metagga_def_from_handles`) refuses, with an error
in place of a silent approximation:

- a component libxc flags as needing the density Laplacian
  (`XC_FLAGS_NEEDS_LAPLACIAN`);
- a component with no energy density;
- more than one range-separated component, or more than one VV10 component;
- a range-separated component combined with a separate global
  exact-exchange fraction.

The family fixes what the grid evaluates (section @sec:vxc) and which response
kernel exists (section @sec:fxc). The exact-exchange fractions are read from
libxc and turned into the SCF's mixing coefficients in section @sec:hybrids.

## Options {#ks-equations-options}

| Option (CLI `[dft]`, Python) | Effect |
|---|---|
| `functional` / `xc=` | a friendly name from the table or any libxc name |
| `xc_omega` | overrides $\omega$ of a range-separated functional (section @sec:hybrids) |
| `grid_radial`, `grid_angular`, `grid_prune`, `grid_weight_cut` | the XC grid (section @sec:grids) |
| `nlc_grid` | the VV10 grid (section @sec:vv10) |
| `[scf] jk`, `k_builder` | how $J$ and $K$ are built; KS defaults to RI-J |

## Numerical anchors {#ks-equations-numerical-anchors}

From `site/src/reference/validation.md`, rows "KS-DFT energies" and
"Meta-GGA energies".

| Quantity | Systems | Agreement | Test |
|---|---|---|---|
| RKS energy, PBE and B3LYP | H2S, HCl, SiH4, CH4, NH3, H2O, HF / def2-SVP, def2-TZVP | $\le 3.1\times10^{-12}$ Ha vs PySCF, same grid and fitting | `crates/ferric-scf/tests/validation_ks_energies.rs` |
| RKS energy, HSE06 | the same | $4.2\times10^{-12}$ Ha | the same |
| UKS energy, ωB97X-V | NH2, CH3, HO2, O2 / 6-31G, def2-SVP | $6\times10^{-6}$ Ha (the DF-K metrics differ) | the same |
| ROKS energy | NH2, CH3, HO2, OH × PBE, B3LYP | $8.0\times10^{-13}$ Ha; OH $8.0\times10^{-7}$ Ha | `crates/ferric-scf/tests/validation_roks_energies.rs` |
| SCAN, r2SCAN energy | H2O, NH3, H2S, CH4 (RKS), NH2 (UKS) / def2-SVP, def2-TZVP | RKS $\le 4.0\times10^{-12}$, UKS $\le 2.4\times10^{-13}$ Ha | `crates/ferric-scf/tests/validation_mgga_energies.rs` |

## Sources {#ks-equations-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-dft/theory.md`, "Kohn-Sham Equations", "Functionals" | the reference statement of the equations above |
| `site/src/methods/scf.md` | how to run KS, the J and K choices, measured costs |
| Kohn and Sham, Phys. Rev. 140, A1133 (1965) | the equations |
