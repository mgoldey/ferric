# 9.1 The Kohn–Sham equations and functional families

<!-- section: 9.1; status: ready; registry: ks-semilocal, ks-mgga;
     theory: crates/ferric-dft/theory.md#"Kohn-Sham Equations" "Functionals" -->

ferric has no separate Kohn–Sham solver. The RHF, UHF and ROHF drivers run
RKS, UKS and ROKS when the configuration names a functional; the only
difference is an exchange-correlation energy and matrix added once per
iteration, and an exact-exchange matrix scaled by the functional's
fractions.

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

## 9.1.1 Energy and Fock matrix

<!-- include: crates/ferric-dft/theory.md#"Kohn-Sham Equations" -->

> **Verified equation 9.1** · the Fock matrix is the derivative of the energy
>
> $F = h + J(D) - \tfrac12K_{\rm tot}(D) + V_{xc}$ with
> $V_{xc} = \partial E_{xc}/\partial D$ is $\partial E_{\rm KS}/\partial D$:
> for the exchange part with generic 8-fold-symmetric integrals of both
> range-separated kernels, symbolically (closed and open shell), and
> numerically by central differences of the total electronic energy against
> $\mathrm{Tr}[F\,\delta D]$ for RHF with a range-separated hybrid, a global
> hybrid and HF; for the XC part on a meta-GGA-like model (section 9.3).
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

## 9.1.2 One SCF iteration

| Step | What | Function |
|---|---|---|
| 1 | $J(D)$ and, for a hybrid, $K$ (or the short- and long-range pair) | the run's J and K builders; `subtract_rsh_exchange` |
| 2 | $F_{\rm no\,xc} = h + J - \tfrac12K_{\rm tot}$ | `RhfJk::build_fock` |
| 3 | $e_{\rm no\,xc} = \tfrac12\mathrm{Tr}[D(h + F_{\rm no\,xc})]$ | `RhfState::energy` |
| 4 | $(E_{xc}, V_{xc})$ on the grid, $F \mathrel{+}= V_{xc}$ | `KsXc::add_xc` (section 9.3) |
| 5 | $E = e_{\rm no\,xc} + E_{xc} + E_{nn}$ (+ solvation, embedding, hook) | `RhfState::energy` |
| 6 | DIIS, diagonalize, new $D$ | as for HF |

<!-- excerpt: ferric-scf/src/rhf.rs::RhfState::energy -->
```rust
let e_elec_no_xc: f64 = 0.5 * (d * &(&env.h + &*f)).sum();
```

<!-- excerpt: ferric-scf/src/rhf.rs::RhfState::energy -->
```rust
Ok(e_elec_no_xc + e_xc + e_hook + e_cosmo + e_pcm + e_pol + env.vnn)
```

| Symbol | Code |
|---|---|
| $D$, $h$, $F_{\rm no\,xc}$ | `d`, `env.h`, `f` |
| $E_{xc}$, $E_{nn}$ | `e_xc`, `env.vnn` |

**Reproducibility.** $E_{xc}$ is a grouped, fixed-order sum over points
(`deterministic_point_sum`), so it is bit-identical across thread counts.
$V_{xc}$ skips points below the density floor of `DensityFloor::for_xc`;
$E_{xc}$ does not (theory section above). A meta-GGA SCF gets a default
virtual-block level shift of 0.5 when none is set, ramped to zero at
convergence.

## 9.1.3 Functional families

<!-- include: crates/ferric-dft/theory.md#"Functionals" -->

The family fixes what the grid evaluates (section 9.3) and which response
kernel exists (section 9.6). The exact-exchange fractions are read from
libxc and turned into the SCF's mixing coefficients in section 9.4.

## 9.1.4 Options

| Option (CLI `[dft]`, Python) | Effect |
|---|---|
| `functional` / `xc=` | a friendly name from the table or any libxc name |
| `xc_omega` | overrides $\omega$ of a range-separated functional (section 9.4) |
| `grid_radial`, `grid_angular`, `grid_prune`, `grid_weight_cut` | the XC grid (section 9.2) |
| `nlc_grid` | the VV10 grid (section 9.5) |
| `[scf] jk`, `k_builder` | how $J$ and $K$ are built; KS defaults to RI-J |

## 9.1.5 Numerical anchors

From `site/src/reference/validation.md`, rows "KS-DFT energies" and
"Meta-GGA energies".

| Quantity | Systems | Agreement | Test |
|---|---|---|---|
| RKS energy, PBE and B3LYP | H2S, HCl, SiH4, CH4, NH3, H2O, HF / def2-SVP, def2-TZVP | $\le 3.1\times10^{-12}$ Ha vs PySCF, same grid and fitting | `crates/ferric-scf/tests/validation_ks_energies.rs` |
| RKS energy, HSE06 | the same | $4.2\times10^{-12}$ Ha | the same |
| UKS energy, ωB97X-V | NH2, CH3, HO2, O2 / 6-31G, def2-SVP | $6\times10^{-6}$ Ha (the DF-K metrics differ) | the same |
| ROKS energy | NH2, CH3, HO2, OH × PBE, B3LYP | $8.0\times10^{-13}$ Ha; OH $8.0\times10^{-7}$ Ha | `crates/ferric-scf/tests/validation_roks_energies.rs` |
| SCAN, r2SCAN energy | H2O, NH3, H2S, CH4 (RKS), NH2 (UKS) / def2-SVP, def2-TZVP | RKS $\le 4.0\times10^{-12}$, UKS $\le 2.4\times10^{-13}$ Ha | `crates/ferric-scf/tests/validation_mgga_energies.rs` |

## 9.1.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-dft/theory.md`, "Kohn-Sham Equations", "Functionals" | the equations above |
| `site/src/methods/scf.md` | how to run KS, the J and K choices, measured costs |
| Kohn and Sham, Phys. Rev. 140, A1133 (1965) | the equations |
