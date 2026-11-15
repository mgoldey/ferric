# 19.8 The Bethe–Salpeter equation in the Tamm–Dancoff approximation

<!-- section: 19.8; status: ready; registry: bse-tda;
     theory: crates/ferric-gw/theory.md#"11. BSE-TDA" -->

An excitation is a correlated electron–hole pair. The Bethe–Salpeter equation
treats the pair with two interactions: the electron and the hole attract each
other through the screened interaction $W$, and they exchange through the
bare interaction $v$. On top of GW quasiparticle energies, with a static $W$
and without the de-excitation block (Tamm–Dancoff), it is a symmetric
eigenproblem in the space of single excitations, the same shape as CIS or
TDA-TDDFT (section 20.1) with a different kernel.

| Symbol | Meaning | Code name |
|---|---|---|
| $ia$ | compound index, `i * nvir + a`; active occupied $i$, all virtual $a$ | `ia` |
| $\epsilon^{\rm QP}_p$ | G0W0@HF quasiparticle energies for every active MO | `eps_qp_full` |
| $(pq|rs)$ | bare integral from $B$ | `bare` |
| $(pq|W(0)|rs)$ | statically screened integral | `screened` |
| $\Omega_n$, $X_n$ | excitation energy and amplitudes | `BseResult::omega`, the columns of `evecs` |
| $f_n$ | oscillator strength | `BseResult::oscillator_strength` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/bse.rs::{run_bse_tda,run_cis_tda}` |
| CLI | `method.kind = "bse-tda"`, `[rpa]` and `[gw] frozen_core`; example `examples/water-bse-tda.toml` |
| Python | `run_bse_tda` |

## 19.8.1 The matrix

<!-- include: crates/ferric-gw/theory.md#"11. BSE-TDA" -->

The assembly is a plain double loop over $(ia, jb)$ with the two integral
closures; each row is written once, so the loop parallelizes over rows
without a reduction (serial below eight rows).

<!-- excerpt: ferric-gw/src/bse.rs::run_bse_tda -->
```rust
    let fill_row = |ia: usize, row: &mut [f64]| {
        let i = ia / nvir;
        let a = ia % nvir;
        let i_loc = i; // occupied-active local index
        let eps_i = eps_qp_full[first_act + i];
        let a_loc = nocc + a; // virtual local index in the active block
        let eps_a = eps_qp_full[nocc_total + a];
        for j in 0..nocc {
            let j_loc = j;
            for bb in 0..nvir {
                let b_loc = nocc + bb;
                let jb = j * nvir + bb;
                let coul = bare(i_loc, a_loc, j_loc, b_loc); // (ia|jb)
                let scr = screened(a_loc, b_loc, i_loc, j_loc); // (ab|W|ij)
                row[jb] = 2.0 * coul - scr;
            }
        }
        row[ia] += eps_a - eps_i;
    };
```

| Symbol | Variable |
|---|---|
| $(ia|jb)$ | `coul` |
| $(ab|W(0)|ij)$ | `scr` |
| $\epsilon^{\rm QP}_a - \epsilon^{\rm QP}_i$ | `eps_a - eps_i` |

> **Verified equation 19.13** · the singlet BSE-TDA matrix is the spin adaptation of the spin-orbital one
>
> On a random closed-shell model with a complete auxiliary basis, the code's
> matrix equals the singlet block of the spin-orbital matrix
> $\delta\Delta + (ia|jb) - \delta_{\sigma\tau}(ab|W|ij)$ built with
> $W = B^{\mathsf T}\tilde\varepsilon^{-1}B$, a matrix inverse that shares
> nothing with the PDEP eigenbasis; the triplet block is
> $\Delta - W(ab|ij)$, the singlet–triplet blocks vanish and the spin-orbital
> spectrum is the union of the two. With $W\to v$ the same matrix equals a
> brute-force Jordan–Wigner CIS over determinants.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_bse_rpax_sympy.py`, claims S1, B1, B2 |
> | Negative control | the singlet and triplet blocks must differ (the $2(ia|jb)$ Hartree term is live); the brute-force triplet block must not equal the singlet matrix; the runner sets the Hartree prefactor 2 to 1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/bse.rs::run_bse_tda` |
> | Code test | `cis_tda_anchor_vs_numpy_and_pyscf_tda` in `crates/ferric-gw/tests/validation_bse.rs` |
>
> <!-- verified: id=bse.tda.matrix; script=scripts/gw_bse_rpax_sympy.py; claim=B1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/bse.rs::run_bse_tda; rust_test=ferric-gw/tests/validation_bse.rs::cis_tda_anchor_vs_numpy_and_pyscf_tda; registry=bse-tda -->

> **Verified equation 19.14** · the screened kernel from the complete PDEP basis
>
> For a complete orthonormal basis, $v + \sum_\xi w_\xi B^\xi B^\xi = \sum_\xi\lambda_\xi(0)^{-1}B^\xi B^\xi$.
> The code uses the left side, which stays correct when modes are dropped.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_self_energy_sympy.py`, claim 8 (and claim B1 above for the matrix-inverse route) |
> | Negative control | a truncated basis must miss $(pq|rs)$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/bse.rs::run_bse_tda` |
>
> <!-- verified: id=bse.tda.kernel; script=scripts/gw_self_energy_sympy.py; claim=8; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/bse.rs::run_bse_tda; registry=bse-tda -->

## 19.8.2 Oscillator strengths

The singlet state is a spin-adapted combination of the $\alpha$ and $\beta$
excitations, so its transition dipole to the ground state is $\sqrt2$ times
the amplitude-weighted occupied–virtual dipole:
$f_n = \tfrac23\Omega_n\,\lvert\sqrt2\sum_{ia}X_{n,ia}\langle i|\mathbf r|a\rangle\rvert^2$.
The occupied–virtual dipole blocks do not depend on the origin, by
orbital orthogonality.

> **Verified equation 19.15** · the transition moment and the sum rule
>
> $\langle\Phi_0|\mathbf r|\Psi_n\rangle$ over Jordan–Wigner determinants,
> with $\Psi_n$ the spin-adapted singlet of the CIS roots, equals
> $\sqrt2\sum_{ia}X_{ia}r_{ia}$; and
> $\sum_nf_n = \tfrac43\sum_x\mathbf r_x^{\mathsf T}\mathbf A\,\mathbf r_x$ over all roots of the screened
> matrix, which fixes both the $\sqrt2$ and the factor $\Omega$.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_bse_rpax_sympy.py`, claims S2, B3, B4 |
> | Negative control | dropping the $\sqrt2$ halves the sum rule (B4); the runner drops it from the transition moment (B3) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/bse.rs::tda_oscillator_strengths` |
> | Code test | `cis_tda_oscillator_strengths_match_pyscf_df_kernel` in `crates/ferric-gw/tests/bse_oscillator_strength.rs` |
>
> <!-- verified: id=bse.tda.osc; script=scripts/gw_bse_rpax_sympy.py; claim=B3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/bse.rs::tda_oscillator_strengths; rust_test=ferric-gw/tests/bse_oscillator_strength.rs::cis_tda_oscillator_strengths_match_pyscf_df_kernel; registry=bse-tda -->

## 19.8.3 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | G0W0@HF for **all** active MOs (`qp_mos = 0..nmo`) | section 19.3 for every state | `run_gw` |
| 2 | flag the states whose QP solve is poorly determined, among those on the diagonal | $O(n_{\rm act})$ | `flag_suspect_qp` |
| 3 | $B$, redressed PDEP modes, $B^\xi$ | section 19.2 | `build_full_b`, `redress_with_check`, `project_b_into_pdep` |
| 4 | the dense matrix $\mathbf A$, $n = n_on_v$ | $n^2$ entries, each a contraction over $N_{\rm aux}$ and $N_{\rm pdep}$ | `run_bse_tda` |
| 5 | dense `eigh` | $n^3$ | `run_bse_tda` |
| 6 | $f_n$ from the dipole blocks | $O(n^2)$ | `tda_oscillator_strengths` |

**Why every QP energy is solved.** The G0W0 energies of core and high virtual
orbitals are individually ill-conditioned: the Thiele continuation of
$\Sigma^c$ far from the Fermi level moves them by up to $3.8\times10^{-3}$ Ha
between two runs of the reference generator. The lowest five $\Omega$ move by
at most $1.1\times10^{-7}$ Ha, because those states carry little of the
low-lying eigenvectors. A rigid scissor outside a $\text{HOMO}-k\ldots\text{LUMO}+k$
window instead moves the lowest five $\Omega$ by $1.5\times10^{-4}$ to
$9.4\times10^{-3}$ Ha at $k = 2$ (validation page). So the code solves every
state, and reports the poorly determined ones (a Newton solve that did not
converge, or a renormalization factor pinned at the clamp boundary 0 or 1.5)
in `BseResult::qp_suspect_mos`, with a warning, instead of replacing them.

**Memory.** The dense matrix and its eigenvectors are a hard pool charge
(`check_dense_response_alloc`), and a pre-flight (`check_bse_dense_alloc`)
refuses a system whose $(n_on_v)^2$ does not fit. There is no iterative
solver to fall back to.

## 19.8.4 Options and limits

| Option | Effect |
|---|---|
| `frozen_core` (`[gw]`) | removes the lowest occupied orbitals from the $(ia)$ space and from GW |
| `[rpa] n_quad`, `trunc_thresh` | the GW and PDEP settings of the QP energies and of $W(0)$ |

The CLI refuses multiplicity above 1 (closed-shell RHF only). With $W\to v$
and Hartree–Fock energies the same assembly is CIS (`run_cis_tda`), the
isolation test for the kernel layout.

**Known limits (2026-11-15).** Singlets only, static $W$, no dynamical
correction, dense. The quasiparticle energies come from the internal G0W0,
which matches PySCF only at `n_quad = 100`, `trunc_thresh = 0`; the defaults
are coarser. The grade is Smoke: the lowest five singlets match an
independent numpy BSE-TDA given the same QP energies, and the physics is not
benchmarked against experiment here.

## 19.8.5 Numerical anchors

From `site/src/reference/validation.md` (BSE-TDA row of the anchors table):

| Quantity | Systems | Error | Test |
|---|---|---|---|
| $\Omega$, lowest five singlets, against numpy BSE-TDA on PySCF density-fitted integrals, same QP energies | H2O / cc-pVDZ and aug-cc-pVDZ, NH3, CH2O / cc-pVDZ | $\le 1.8\times10^{-10}$ Ha | `bse_tda_{h2o,nh3,ch2o}_vs_numpy` in `crates/ferric-gw/tests/validation_bse.rs` |
| oscillator strengths, same | same | $\le 2.6\times10^{-9}$ | same |
| $\Omega$ with each side's own QP energies | same | $\le 2.2\times10^{-7}$ Ha | same |
| CIS limit against numpy DF-CIS | same | $\le 1.6\times10^{-9}$ Ha | `cis_tda_anchor_vs_numpy_and_pyscf_tda` |

## 19.8.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, "BSE-TDA" | the definition, the singlet reduction, the oscillator strength |
| `scripts/gw_bse_rpax_sympy.py`, claims S1, S2, B1 to B4 | the proofs |
| Blase, Duchemin and Jacquemin, *Chem. Soc. Rev.* **47**, 1022 (2018) | BSE for molecules |
