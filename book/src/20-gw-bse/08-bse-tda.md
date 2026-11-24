# The Bethe–Salpeter equation in the Tamm–Dancoff approximation {#bse-tda}

<!-- status: ready; registry: bse-tda -->

A quasiparticle gap $\epsilon^{\rm QP}_a - \epsilon^{\rm QP}_i$ is the cost of
removing an electron from $i$ and, separately, adding one to $a$. An optical
excitation does both at once: the excited electron stays in the molecule,
next to the hole it left. The two attract each other, and that attraction is
screened by the other electrons, so it is $W$. The Bethe–Salpeter equation
adds this electron–hole interaction, plus a bare exchange term, to the
quasiparticle gaps. With a static $W$ and without the de-excitation block
(Tamm–Dancoff), it is a symmetric eigenproblem in the space of single
excitations, the same shape as CIS or TDA-TDDFT (section @sec:response) with a
different kernel. This section takes the response matrices, the singlet
spin adaptation and the oscillator-strength formula from chapter @sec:tddft
and states what the BSE kernel changes.

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

## The matrix {#bse-tda-matrix}

**In spin orbitals.** The Bethe–Salpeter kernel for a particle–hole pair has
two terms. In the Hartree-type term the pair annihilates into the bare
interaction $v$ and is recreated as another pair. In the direct term the
particle and the hole exchange a screened interaction $W$. Take a static
$W(0)$, quasiparticle energies on the diagonal, and the Tamm–Dancoff
approximation (no de-excitation block). The matrix over single excitations
$i_\sigma\to a_\sigma$ is

$$ A_{i_\sigma a_\sigma,\,j_\tau b_\tau} = (\epsilon^{\rm QP}_a - \epsilon^{\rm QP}_i)\,\delta_{ij}\delta_{ab}\delta_{\sigma\tau} + (ia|jb) - \delta_{\sigma\tau}\,(ab|W(0)|ij) . $$

The Hartree term connects every spin pair: the transition density
$\phi_i\phi_a$ has no spin dependence beyond $\sigma$, and $v$ does not see
spin. The screened term connects equal spins only, because the electron and
hole it acts on must keep their spins.

**Spin-adapt.** The blocks are $A_{\alpha\alpha} = \Delta + (ia|jb) - (ab|W|ij)$
and $A_{\alpha\beta} = (ia|jb)$. The spin rotation of equation
@eq:td-matrices-spin-adaptation (section @sec:response-matrices) gives the
singlet block $A_{\alpha\alpha}+A_{\alpha\beta}$ and the triplet block
$A_{\alpha\alpha}-A_{\alpha\beta}$, exactly as for TDDFT with $-(ab|W|ij)$
in the place of $-c_{\rm HF}(ij|ab) + K$. The singlet block is the code's
matrix, with $(ia|jb)$ counted twice:

$$ \mathbf A_{ia,jb} = (\epsilon^{\rm QP}_a - \epsilon^{\rm QP}_i)\,\delta_{ij}\delta_{ab} + 2(ia|jb) - (ab|W(0)|ij), $$

$$ (pq|W(0)|rs) = (pq|rs) + \sum_\xi w_\xi B^\xi_{pq}B^\xi_{rs}, \qquad w_\xi = \frac1{\lambda_\xi(0)} - 1 . $$

The triplet block is $\Delta - (ab|W|ij)$, with no Hartree term. Claim B1
proves it, but the code does not compute it: `run_bse_tda` returns singlets
only.

**Check the limits.** Drop the kernel and the eigenvalues are the
quasiparticle gaps $\epsilon^{\rm QP}_a - \epsilon^{\rm QP}_i$. Replace $W$ by
$v$ and the quasiparticle energies by Hartree–Fock ones, and the matrix is
CIS (`ferric-gw/src/bse.rs::run_cis_tda`). On the diagonal the direct term
$-(aa|W|ii)$ is the electron–hole attraction. CIS uses the bare
$-(aa|ii)$; BSE uses the interaction screened by the other electrons.

**The screened kernel.** The bare part of $(pq|W(0)|rs)$ comes from $B$, and
the dynamic part from the PDEP modes. For a complete basis the kernel equals
$\sum_\xi \lambda_\xi(0)^{-1} B^\xi B^\xi$ (section @sec:screened-interaction-pdep-basis), and the split form
stays correct when modes are truncated. The Hartree term uses bare $v$; the
direct term uses the static full $W$. This is static-screening BSE-TDA as
defined here: ferric does not add a dynamical correction to $W$.

**How the matrix is checked.** On a random model with a complete auxiliary
basis, claim B1 compares the code's singlet matrix with the singlet block of
the spin-orbital matrix built with $W = B^{\mathsf T}\tilde\varepsilon^{-1}B$,
a matrix inverse, not the PDEP eigenbasis. The triplet block is
$\Delta - W(ab|ij)$, the singlet–triplet blocks vanish, and the spin-orbital
spectrum is the union of the two. Claim B2 sets $W\to v$ with a diagonal Fock
matrix and checks that the same matrix equals
$\langle\Phi_i^a|H - E_0|\Phi_j^b\rangle$ over Jordan–Wigner determinants of
an 8-spin-orbital Fock space, and that the Brillouin condition holds.

> **Verified equation {#eq:bse-tda-box}** · the singlet BSE-TDA matrix is the spin adaptation of the spin-orbital one
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

> **Verified equation {#eq:bse-tda-kernel}** · the screened kernel from the complete PDEP basis
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
> | Code test | `bse_tda_h2o_vs_numpy` in `crates/ferric-gw/tests/validation_bse.rs` |
>
> <!-- verified: id=bse.tda.kernel; script=scripts/gw_self_energy_sympy.py; claim=8; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/bse.rs::run_bse_tda; rust_test=ferric-gw/tests/validation_bse.rs::bse_tda_h2o_vs_numpy; registry=bse-tda -->

**The assembly.** `ferric-gw/src/bse.rs::run_bse_tda` fills the matrix with a
double loop over $(ia, jb)$ and two integral closures, `bare` and
`screened`, inside the row closure `fill_row`. Each row is written once, so
the loop parallelizes over rows without a reduction (serial below eight
rows):

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

Each closure call contracts over $N_{\rm aux}$ (and $N_{\rm pdep}$ for
`screened`) on the fly. The matrix is then diagonalized densely; LAPACK
returns the eigenvalues ascending with matching eigenvector columns.

## Oscillator strengths {#bse-tda-oscillator-strengths}

**The transition moment.** The transition dipole and the oscillator strength
are those of equation @eq:td-osc-sum-rule (section
@sec:oscillator-strengths-length-gauge-formula) with $Y = 0$:
$\langle 0|\mathbf r|n\rangle = \sqrt2\sum_{ia}X_{ia}\langle i|\mathbf r|a\rangle$
and $f_n = \tfrac23\,\Omega_n|\langle 0|\mathbf r|n\rangle|^2$. The
$\sqrt2$ is the same spin sum, and the occupied–virtual dipole blocks do not
depend on the origin for the same reason. BSE-TDA has its own copy of the
loop, in
`ferric-gw/src/bse.rs::tda_oscillator_strengths`:

<!-- excerpt: ferric-gw/src/bse.rs::tda_oscillator_strengths -->
```rust
            for i in 0..nocc {
                for a in 0..nvir {
                    acc += x[i * nvir + a] * dip[(i, a)];
                }
            }
            mu[d] = std::f64::consts::SQRT_2 * acc;
        }
        let mu2 = mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2];
        f.push((2.0 / 3.0) * evals[state] * mu2);
```

| Code | Equation |
|---|---|
| `x[i * nvir + a]` | $X_{n,ia}$ |
| `dip[(i, a)]` | $\langle i|r_d|a\rangle$ |
| `SQRT_2 * acc` | $\langle 0|r_d|n\rangle$ |
| `(2.0 / 3.0) * evals[state] * mu2` | $f_n$ |

**How the factors are checked.** Claim B3 evaluates
$\langle\Phi_0|\mathbf r|\Psi_n\rangle$ over the Jordan–Wigner determinants.
Claim B4 checks the sum rule over all roots. It is the TDA form of equation
@eq:td-osc-sum: with $Y = 0$ and $X_n$ orthonormal,
$\sum_n\Omega_nX_nX_n^{\mathsf T} = \mathbf A$, so $\mathbf A$ takes the
place of $\mathbf A - \mathbf B$ and
$\sum_n f_n = \tfrac43\sum_x \mathbf r_x^{\mathsf T}\mathbf A\,\mathbf r_x$.
The rule fixes both the $\sqrt2$ and the factor $\Omega$.

> **Verified equation {#eq:bse-tda-osc}** · the transition moment and the sum rule
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

The lowest five singlets of water / cc-pVDZ, from the `589b50832` build, with
their oscillator strengths. Every root lies below the G0W0@HF gap of the
first line: the screened electron–hole attraction of the direct term pulls
them down. The second root is dark ($f = 0$ to five places). The CLI input
is `examples/water-bse-tda.toml`.

```python
# Water / cc-pVDZ: lowest five BSE-TDA singlets, their oscillator strengths, and the G0W0 gap they sit below.
import ferric

HA_EV = 27.211386245988
mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

r = ferric.run_bse_tda(mol, bs, aux, n_quad=100, trunc_thresh=0.0)
homo = r.nocc - 1  # active indices; frozen_core = 0 here
gap = r.eps_qp[homo + 1] - r.eps_qp[homo]
print(f"G0W0@HF gap: {gap*HA_EV:.4f} eV")
for n in range(5):
    print(f"Omega_{n}: {r.omega[n]*HA_EV:9.4f} eV   f = {r.oscillator_strength[n]:.5f}")
print("certificate warnings:", r.certificate.get("warnings"))
```

```text
G0W0@HF gap: 16.8635 eV
Omega_0:    8.4576 eV   f = 0.02685
Omega_1:   10.5022 eV   f = 0.00000
Omega_2:   11.1690 eV   f = 0.10004
Omega_3:   13.2132 eV   f = 0.07737
Omega_4:   14.9967 eV   f = 0.29619
certificate warnings: []
```

## The algorithm {#bse-tda-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | G0W0@HF for **all** active MOs (`qp_mos = 0..nmo`) | section @sec:correlation-self-energy for every state | `run_gw` |
| 2 | flag the states whose QP solve is poorly determined, among those on the diagonal | $O(n_{\rm act})$ | `flag_suspect_qp` |
| 3 | $B$, redressed PDEP modes, $B^\xi$ | section @sec:screened-interaction | `build_full_b`, `redress_with_check`, `project_b_into_pdep` |
| 4 | the dense matrix $\mathbf A$, $n = n_on_v$ | $n^2$ entries, each a contraction over $N_{\rm aux}$ and $N_{\rm pdep}$ | `run_bse_tda` |
| 5 | dense `eigh` | $n^3$ | `run_bse_tda` |
| 6 | $f_n$ from the dipole blocks | $O(n^2)$ | `tda_oscillator_strengths` |

**Why every QP energy is solved.** The G0W0 energies of core and high virtual
orbitals are individually ill-conditioned: the Thiele continuation of
$\Sigma^c$ far from the Fermi level moves them by up to $3.8\times10^{-3}$ Ha
between two independent runs of the same reference calculation. The lowest five $\Omega$ move by
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

## Options and limits {#bse-tda-options-limits}

| Option | Effect |
|---|---|
| `frozen_core` (`[gw]`) | removes the lowest occupied orbitals from the $(ia)$ space and from GW |
| `[rpa] n_quad`, `trunc_thresh` | the GW and PDEP settings of the QP energies and of $W(0)$ |

The CLI refuses multiplicity above 1 (closed-shell RHF only). With $W\to v$
and Hartree–Fock energies the same assembly is CIS (`run_cis_tda`), the
isolation test for the kernel layout.

**Known limits (2026-11-19).** Singlets only, static $W$, no dynamical
correction, dense. HF starting point only: `run_bse_tda` takes an RHF
reference and the Python call has no `xc`. The quasiparticle energies come
from the internal G0W0, which matches PySCF only at `n_quad = 100`,
`trunc_thresh = 0`; the defaults are coarser. The grade is Smoke: the lowest
five singlets match an independent numpy BSE-TDA given the same QP energies.
The physics has one committed benchmark: BSE-TDA[G0W0@HF] / aug-cc-pVDZ on
a ten-molecule Thiel-set subset against the QUEST best estimates
(`testdata/reference/thiel_set_subset.json`, inputs
`examples/*-bse-tda-augdz.toml`, logs in `benchmarks/bse-tda-pilot/`) gives
a mean absolute error of 0.641 eV and a mean signed error of +0.598 eV over
17 assigned states. `corrected_statistic_is_worse_than_the_naive_one_and_both_are_pinned`
in `crates/ferric-gw/tests/bse_tda_phase2_assignment.rs` recomputes those
figures from the committed logs. It measures the method at that basis, not
the implementation, and does not set the grade.

## Numerical anchors {#bse-tda-numerical-anchors}

From `site/src/reference/validation.md` (BSE-TDA row of the anchors table):

| Quantity | Systems | Error | Test |
|---|---|---|---|
| $\Omega$, lowest five singlets, against numpy BSE-TDA on PySCF density-fitted integrals, same QP energies | H2O / cc-pVDZ and aug-cc-pVDZ, NH3, CH2O / cc-pVDZ | $\le 1.8\times10^{-10}$ Ha | `bse_tda_{h2o,nh3,ch2o}_vs_numpy` in `crates/ferric-gw/tests/validation_bse.rs` |
| oscillator strengths, same | same | $\le 2.6\times10^{-9}$ | same |
| $\Omega$ with each side's own QP energies | same | $\le 2.2\times10^{-7}$ Ha | same |
| CIS limit against numpy DF-CIS | same | $\le 1.6\times10^{-9}$ Ha | `cis_tda_anchor_vs_numpy_and_pyscf_tda` |

Further tests: `crates/ferric-gw/tests/bse_oscillator_strength.rs` and
`crates/ferric-gw/tests/h2o_bse_tda.rs`.

## Sources {#bse-tda-sources}

| Source | What it gives |
|---|---|
| `scripts/gw_bse_rpax_sympy.py`, claims S1, S2, B1 to B4 | the proofs |
| Blase, Duchemin and Jacquemin, *Chem. Soc. Rev.* **47**, 1022 (2018) | BSE for molecules |
