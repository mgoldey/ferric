# 19.1 RI factors and the exchange self-energy

<!-- section: 19.1; status: ready; registry: gw, u-gw;
     theory: crates/ferric-gw/theory.md#"1. RI factorization and exchange" -->

Take an electron out of orbital $p$. In Hartree–Fock its energy has two
interaction parts: the Coulomb repulsion from the average density, and the
exchange interaction with the electrons of the same spin. GW splits the
self-energy the same way. The bare exchange part, $\Sigma^x$, is the
Hartree–Fock exchange of state $p$ with every occupied orbital. The
correlation part, $\Sigma^c$ (sections 19.3 to 19.5), is what screening
adds on top.

Every GW quantity is a contraction of one object: the Coulomb integrals
$(pq|rs)$ over all pairs of active orbitals, held in the
resolution-of-identity form of chapter 6. RI-MP2 needs only the
occupied–virtual block $(ia|jb)$. GW needs every pair $pq$, because the
self-energy of a state $p$ sums over all states $q$. The exchange
self-energy is the first use.

| Symbol | Meaning | Code name |
|---|---|---|
| $i, j$ / $a, b$ / $p, q$ | active occupied / virtual / any active MO | local indices into `MoB` |
| $P, Q$ | auxiliary functions | `naux` |
| $M$ | metric factor with $M^{\mathsf T}M = V^{-1}$ | `v_inv_sqrt` |
| $B^P_{pq}$ | dressed three-index tensor over the active square | `MoB::b_full`, shape `(naux, n_act, n_act)` |
| $\Sigma^x_p$ | exchange self-energy diagonal | `sigma_x_diag` |

"Active" means MO indices `frozen_core..nmo`. A frozen core orbital is absent
from the sums on both sides of the self-energy.

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/lib.rs::{run_gw,run_u_gw}` |
| Tensor | `ferric-gw/src/mo_b.rs::{build_full_b,build_full_b_spin,build_full_b_both_spins}` |
| Exchange | `ferric-gw/src/cohsex.rs::sigma_x_diag` |
| CLI | `method.kind = "gw"`, with the `[gw]` section (`method = "g0w0" \| "cohsex" \| "evgw0" \| "evgw"`) |
| Python | `run_gw`, `run_u_gw` |

## 19.1.1 The factorization

**The tensor.** Start from the Coulomb metric $V_{PQ} = (P|Q)$ of the
auxiliary basis and any factor $M$ with $M^{\mathsf T}M = V^{-1}$. Section
6.1 shows that the choice of $M$ does not change any contraction of two $B$
factors. Fit each orbital-pair density $\phi_p\phi_q$ and dress it with $M$:

$$ B^P_{pq} = \sum_Q M_{PQ}\,(Q|pq), \qquad (pq|rs) \approx \sum_P B^P_{pq} B^P_{rs} . $$

The code holds $B$ over the full active square, shape
`(naux, n_act, n_act)`. `ferric-gw/src/mo_b.rs::build_full_b` builds it for a
closed-shell reference, and `build_full_b_spin` and
`build_full_b_both_spins` build one per spin.

**The exchange self-energy.** The exchange energy of state $p$ with the
occupied orbitals of its spin is the diagonal of the Hartree–Fock exchange
operator, $\Sigma^x_p = -\sum_i (pi|ip)$. Substitute the factorization. The
orbitals are real, so $B^P_{ip} = B^P_{pi}$, and the product of the two
factors is a square:

$$ \Sigma^x_p = -\sum_i \sum_P B^P_{pi}B^P_{ip} = -\sum_i \sum_P \left(B^P_{pi}\right)^2 . $$

The result is the closed-shell Hartree–Fock exchange diagonal in the RI basis. For
an open-shell reference the same formula runs on each spin's $B_\sigma$,
with $i$ over that spin's occupied orbitals.

**Check the Hartree–Fock limit.** The closed-shell Fock diagonal is
$F_{pp} = h_{pp} + \sum_i\big[2(pp|ii) - (pi|ip)\big]$. Its exchange part is
$\Sigma^x_p$. So on a Hartree–Fock reference, $\epsilon_p$ already contains
$\Sigma^x_p$, and the quasiparticle correction is $\Sigma^c$ alone. If the
screening vanishes ($W \to v$, section 19.2), $\Sigma^c = 0$ and the
quasiparticle energies are the Koopmans orbital energies. Section 19.1.3
gives the Kohn–Sham case.

$\Sigma^x$ is computed from the full $B$, never from the PDEP projection of
section 19.2. PDEP rank truncation therefore changes only the correlation
part.

The heart of `ferric-gw/src/cohsex.rs::sigma_x_diag` is one closure per
state $p$ (`m` in the code), mapped in parallel over states:

<!-- excerpt: ferric-gw/src/cohsex.rs::sigma_x_diag -->
```rust
    let compute_one = |m: usize| -> f64 {
        let mut acc = 0.0;
        for i in 0..n_occ {
            for p in 0..naux {
                let b = mo_b.b_full[(p, m, i)];
                acc += b * b;
            }
        }
        -acc
    };
```

| Code | Equation |
|---|---|
| `m` | the state $p$ of $\Sigma^x_p$ |
| `p` | the auxiliary index $P$ |
| `b * b` | $(B^P_{pi})^2$ |
| `-acc` | the minus sign of exchange |

Each `sx[m]` is written by one closure, so the parallel map over states has
no reduction and gives the same bits as the serial loop. Below eight states
the code runs it serially.

> **Verified equation 19.1** · the exchange self-energy is the closed-shell Fock exchange in the RI basis
>
> On a lattice model with spin-dependent orbitals, the code's
> $\Sigma^x_{\sigma p} = -\sum_{i\in\sigma}\sum_P (B^P_{\sigma,pi})^2$ equals
> the real-space construction $-\sum_{i\in\sigma}\langle pi|v|ip\rangle$ with
> no auxiliary basis.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_ucohsex_sympy.py`, claim U1 (the `sx` terms) |
> | Negative control | summing over the other spin's occupied orbitals must miss |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/cohsex.rs::sigma_x_diag` |
> | Code test | `g0w0_hf_h2o_vs_pyscf_gw_ac` in `crates/ferric-gw/tests/validation_gw.rs` (the $\Sigma^x$ anchor) |
>
> <!-- verified: id=gw.sigma-x; script=scripts/gw_ucohsex_sympy.py; claim=U1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/cohsex.rs::sigma_x_diag; rust_test=ferric-gw/tests/validation_gw.rs::g0w0_hf_h2o_vs_pyscf_gw_ac; registry=u-gw -->

**Where the factor comes from.** `build_full_b` takes the metric from the
Coulomb operator by Cholesky, $V = LL^{\mathsf T}$, and applies $M = L^{-1}$
to the three-index integrals in one pass over the active square. The
attenuated operators of chapter 4 use the symmetric eigendecomposition
factor $V^{-1/2}$ instead. The GW drivers carry the factor with the tensor
(`MoB::v_inv_sqrt`), because section 19.2 needs it to undress the dielectric
eigenvectors. The tensor is a hard memory charge of
$N_{\rm aux}\,n_{\rm act}^2$ doubles, accounted in the pool before it is
built (`guard_b_full`).

## 19.1.2 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF or KS reference, all orbitals | as the reference | the caller |
| 2 | $B^P_{pq}$ over the active square, metric applied | $N_{\rm aux}\,n_{\rm act}^2 N$ for the transform | `build_full_b` |
| 3 | $\Sigma^x_p = -\sum_{iP}(B^P_{pi})^2$ for every active $p$ | $N_{\rm aux}\,n_{\rm act}\,n_o$ | `sigma_x_diag` |
| 4 | the PDEP-RPA run that supplies $W$ (section 19.2) | chapter 18 | `ferric_rpa::run_pdep_rpa` |

`run_gw` forces the inverse-dielectric stack on in the RPA configuration
(`with_inv_dielectric`), because the correlation self-energy reads it. An
energy-only RPA run leaves it off to save memory. `run_gw` rejects a
`qp_mos` range past the number of orbitals before any expensive step.

## 19.1.3 What a KS reference changes

On a Hartree–Fock reference $\epsilon_p$ already contains $\Sigma^x_p$
(section 19.1.1). A Kohn–Sham orbital energy contains the
exchange-correlation potential $v^{xc}_p$ in its place. GW removes
$v^{xc}_p$ and puts in the exchange and correlation self-energy, so the
quasiparticle equation of section 19.5 carries the static shift

$$ \delta_p = \Sigma^x_p - v^{xc}_p , $$

with $v^{xc}_p$ the diagonal of the XC potential in the MO basis
(`ferric-gw/src/vxc_mo.rs::vxc_diagonal_mo`, the same XC code as the SCF's
Fock build). For a semilocal functional the Kohn–Sham Fock diagonal is
$h_{pp} + J_{pp} + v^{xc}_p$, so with $\Sigma^c = 0$ the corrected energy
$\epsilon_p + \delta_p$ is the Hartree–Fock Fock diagonal evaluated in the
Kohn–Sham orbitals. The shift is computed once, from the starting orbitals.

## 19.1.4 Numerical anchors

From the table in the module header of `crates/ferric-gw/tests/validation_gw.rs`:

| Quantity | Systems | Error | Bar | Test |
|---|---|---|---|---|
| $\Sigma^x$ against PySCF's density-fitted exchange diagonal, same auxiliary basis | the validation set; worst case CH2 triplet | $4.9\times10^{-9}$ Ha | $3\times10^{-8}$ Ha (`TOL_SX`) | `crates/ferric-gw/tests/validation_gw.rs` |

## 19.1.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, "RI factorization and exchange" | the factorization and $\Sigma^x$ |
| `scripts/gw_ucohsex_sympy.py`, claim U1 | the proof of the per-spin exchange sum |
