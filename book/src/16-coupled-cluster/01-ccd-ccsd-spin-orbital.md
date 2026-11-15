# 16.1 CCD and CCSD in spin orbitals

<!-- section: 16.1; status: ready; registry: ccd, ccsd;
     theory: crates/ferric-cc/theory.md#"CCD", "CCSD in spin orbitals" -->

Coupled cluster writes the wavefunction as $e^{\hat T}|\Phi_0\rangle$ and
fixes the amplitudes by projecting the similarity-transformed Hamiltonian
onto the excited determinants. The spin-orbital solvers transcribe the
textbook equations with antisymmetrized integrals. They are the reference
form: `ccd` is the production CCD, and the spin-orbital `ccsd` is the
cross-check of the closed-shell CCSD of section 16.2.

| Index | Range | Code name |
|---|---|---|
| $i, j, k, l, m, n$ | active occupied spin orbitals | `0..2*no` |
| $a, b, c, d, e, f$ | virtual spin orbitals | `0..2*nv` |
| $P$ | auxiliary | `0..naux` |

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/ccd.rs::{ccd,ccd_spinorbital}`, `ferric-cc/src/ccsd.rs::ccsd` |
| CLI | `method.kind = "ccd"` (`examples/water-ccd.toml`); the spin-orbital CCSD has no CLI kind |
| Python | `run_ccd` |

## 16.1.1 The equations

$$ R_\mu = \langle\Phi_\mu|e^{-\hat T}\hat He^{\hat T}|\Phi_0\rangle = 0,\qquad
E_c = \langle\Phi_0|e^{-\hat T}\hat He^{\hat T}|\Phi_0\rangle - E_{\rm HF} . \tag{16.1} $$

The similarity transform terminates: $\hat H$ has at most four operators to
contract with $\hat T$, so the doubles residual is at most quadratic in
$t_2$ (CCD) and quartic in $t_1$ (CCSD). With canonical orbitals the Fock
operator contributes only $\Delta^{ab}_{ij}t^{ab}_{ij}$ (and
$\Delta^a_i t^a_i$), and the code writes $R = \Delta t + r$ and iterates
$t \leftarrow r/(-\Delta)$.

**CCD.**

<!-- include: crates/ferric-cc/theory.md#"CCD" -->

**CCSD.** $R_i^a = \langle\Phi_i^a|e^{-\hat T}\hat He^{\hat T}|\Phi_0\rangle$ and
$R_{ij}^{ab}$ likewise, with $\hat T = \hat T_1 + \hat T_2$.

<!-- include: crates/ferric-cc/theory.md#"CCSD in spin orbitals" -->

The two energy terms are the only connected closed diagrams: $f_{ia} = 0$
removes $\sum f_{ia}t_i^a$.

> **Verified equation 16.1** · the CCD residual and energy
>
> On a ten-spin-orbital model with a diagonal Fock matrix by construction,
> the eight-term $r + \Delta t$ of `ccd_spinorbital` at random antisymmetric
> $t_2$ equals $\langle\Phi_{ij}^{ab}|e^{-\hat T_2}\hat He^{\hat T_2}|\Phi_0\rangle$
> built from Jordan–Wigner matrices, and $\tfrac14\langle ij\|ab\rangle t$
> equals the projected energy. The brute-force route has no diagrams, no
> intermediates and no Wick contractions.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_ccsd_eom_brute_force_sympy.py`, check `ccd-residual` |
> | Negative control | the runner's mutation: the $\tfrac12$ of the second quadratic term set to 1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/ccd.rs::ccd_spinorbital`, `ferric-cc/src/helpers.rs::{p_ij,p_ab,p_ij_ab}` |
> | Code test | `ccd_matches_df_reference` in `crates/ferric-cc/tests/validation_cc.rs` (against PySCF `CCD` on the same DF integrals) |
>
> <!-- verified: id=cc.ccd.residual; script=scripts/cc_ccsd_eom_brute_force_sympy.py; claim=ccd-residual; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/ccd.rs::ccd_spinorbital; rust_test=ferric-cc/tests/validation_cc.rs::ccd_matches_df_reference; registry=ccd -->

> **Verified equation 16.2** · the CCSD residuals and energy
>
> At random $t_1$ and $t_2$ on the same model, the Stanton–Gauss singles and
> doubles residuals of `ccsd` (intermediates $F_{ae}$, $F_{mi}$, $F_{me}$,
> $W_{mnij}$, $W_{abef}$, $W_{mbej}$, $\tau$, $\tilde\tau$) plus $\Delta t$
> equal the brute-force projections; the energy
> $\tfrac14\langle ij\|ab\rangle t_{ij}^{ab} + \tfrac12\langle ij\|ab\rangle t_i^at_j^b$
> equals the projected energy; and the doubles residual at $t_1 = 0$ is the
> CCD residual of (16.1).
>
> | | |
> |---|---|
> | Proof | `scripts/cc_ccsd_eom_brute_force_sympy.py`, checks `ccsd-residual`, `ccsd-energy`, `ccsd-ccd-limit` |
> | Negative control | the runner's mutations: $t_2$ instead of $\tfrac12t_2$ in $W_{mbej}$; the $t_1t_1$ energy weight $\tfrac14$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/ccsd.rs::ccsd` |
> | Code test | `ccsd_spin_orbital_matches_df_reference` in `crates/ferric-cc/tests/validation_cc.rs` |
>
> <!-- verified: id=cc.ccsd.spin-orbital; script=scripts/cc_ccsd_eom_brute_force_sympy.py; claim=ccsd-residual; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/ccsd.rs::ccsd; rust_test=ferric-cc/tests/validation_cc.rs::ccsd_spin_orbital_matches_df_reference; registry=ccsd -->

## 16.1.2 The algorithm

Both solvers share one setup and one loop shape.

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF; $B^P_{ov}, B^P_{oo}, B^P_{vv}$ (section 6.1) | $N_{\rm aux}N^2n$ | `transform_3center_*`, `build_b` |
| 2 | spatial chemist blocks, then antisymmetrized spin-orbital blocks $\langle pq\|rs\rangle$ | $N_{\rm aux}n_v^4$ for VVVV | `asym_oovv`, `asym_same`, `asym_ovvo` (CCD); `asym_phys`, 11 blocks (CCSD) |
| 3 | MP2 guess $t_2 = \langle ij\|ab\rangle/(-\Delta)$, $t_1 = 0$ | | |
| 4 | residual $r$ by `einsum!` GEMMs; the pp ladder dominates | $(2n_o)^2(2n_v)^4$ per iteration | |
| 5 | $t \leftarrow r/(-\Delta)$; DIIS on $t_2$, error = increment | | `ferric_scf::diis::Diis` |
| 6 | stop when the energy changes by less than $10^{-10}$ (CCD) or $\min(\texttt{energy\_conv}, 10^{-9})$ (CCSD) | | |

The ring term of CCD, with its permutation pair applied once to the
contracted block:

<!-- excerpt: ferric-cc/src/ccd.rs::ccd_spinorbital -->
```rust
let bjia: ArrayD<f64> = einsum!("kbcj,ikac->bjia", &ovvo_t, &t_t);
let x = permute_to_owned(bjia.view().permuted_axes(IxDyn(&[2, 1, 3, 0])));
r = r + p_ij_ab(&x);
```

| Symbol | Code |
|---|---|
| $\langle kb\|cj\rangle$ | `ovvo_t` |
| $t_{ik}^{ac}$ | `t_t` |
| $P(ij)P(ab)$ | `p_ij_ab` |
| $r_{ij}^{ab}$ | `r` |

**Memory.** Every spin-orbital tensor is $2^4 = 16$ times its spatial
counterpart, and the $(2n_v)^4$ VVVV block is held in core (CCSD holds it
twice: the block and the per-iteration $W_{abef}$). Both drivers commit a
`MemoryPlan` before the first allocation and refuse a job that cannot fit,
naming the dominant term.

## 16.1.3 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | frozen occupied spatial orbitals |
| `CcConfig::max_iter` | integer | 50 | iteration cap; non-convergence is an error (library field; the CLI uses the default) |
| `CcConfig::diis_subspace` | integer | 6 | DIIS history length (library field) |
| `[memory] budget_gb` | GB | auto | preflight of the plan |

## 16.1.4 Numerical anchors

From `site/src/reference/validation.md` and the tests it names.

| Quantity | Systems | Error | Test |
|---|---|---|---|
| RI-CCD vs PySCF `CCD` on the same DF integrals | H2O, NH3 / cc-pVDZ, def2-SVP | $\le 1.9\times10^{-11}$ Ha | `crates/ferric-cc/tests/validation_cc.rs` |
| RI-CCSD, spin-orbital and spin-adapted, vs PySCF `CCSD` | H2O, NH3, CH4, HF / cc-pVDZ, def2-SVP and others | $\le 4.7\times10^{-11}$ Ha | `crates/ferric-cc/tests/validation_cc.rs` |

## 16.1.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, "CCD", "CCSD in spin orbitals" | the working equations above |
| `scripts/cc_ccsd_eom_brute_force_sympy.py` | the brute-force check of both residuals |
| `site/src/methods/cc.md`, "CCSD and CCSD(T)" | the user guide |

**Known limits (2026-11-15).** Restricted orbitals only: every CC solver
refuses an unrestricted reference. Canonical HF orbitals are assumed
($f_{ia} = 0$, diagonal $f_{oo}$ and $f_{vv}$). The spin-orbital CCSD is a
library cross-check with no CLI kind.
