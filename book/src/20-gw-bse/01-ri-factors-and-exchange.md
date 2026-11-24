# RI factors and the exchange self-energy {#ri-factors-and-exchange}

<!-- status: ready; registry: gw, u-gw -->

Take an electron out of orbital $p$. In Hartree–Fock its energy has two
interaction parts: the Coulomb repulsion from the average density, and the
exchange interaction with the electrons of the same spin. GW splits the
self-energy the same way. The bare exchange part, $\Sigma^x$, is the
Hartree–Fock exchange of state $p$ with every occupied orbital. The
correlation part, $\Sigma^c$ (sections @sec:correlation-self-energy to @sec:continuation-and-qp-equation), is what screening
adds on top.

Every GW quantity is a contraction of one object: the Coulomb integrals
$(pq|rs)$ over all pairs of active orbitals, held in the
resolution-of-identity form of chapter @sec:density-fitting. RI-MP2 needs only the
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
| $\delta_p$ | static shift, $0$ (HF) or $\Sigma^x_p - v^{xc}_p$ (KS) | `static_shift` |
| $c_{\rm sr}, c_{\rm lr}, \omega$ | exact-exchange fractions and range-separation parameter of a hybrid | `KMix { sr, lr, omega }` |

"Active" means MO indices `frozen_core..nmo`. A frozen core orbital is absent
from the sums on both sides of the self-energy.

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/lib.rs::{run_gw,run_u_gw}` |
| Tensor | `ferric-gw/src/mo_b.rs::{build_full_b,build_full_b_spin,build_full_b_both_spins}` |
| Exchange | `ferric-gw/src/cohsex.rs::sigma_x_diag` |
| CLI | `method.kind = "gw"`, with the `[gw]` section (`method = "g0w0" \| "cohsex" \| "evgw0" \| "evgw"`) |
| Python | `run_gw`, `run_u_gw` |

## The factorization {#ri-factors-and-exchange-factorization}

**The tensor.** Start from the Coulomb metric $V_{PQ} = (P|Q)$ of the
auxiliary basis and any factor $M$ with $M^{\mathsf T}M = V^{-1}$. Section
@sec:factorization-and-metric shows that the choice of $M$ does not change any contraction of two $B$
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
screening vanishes ($W \to v$, section @sec:screened-interaction), $\Sigma^c = 0$ and the
quasiparticle energies are the Koopmans orbital energies. Section @sec:ri-factors-and-exchange-ks-reference-changes
gives the Kohn–Sham case.

ferric keeps only the diagonal of $\Sigma$ in the reference orbitals. On a
Hartree–Fock reference the off-diagonal exchange elements $\Sigma^x_{pq}$
vanish against the Fock matrix (Brillouin), so nothing is lost at the
exchange level. On a Kohn–Sham reference the off-diagonal elements of
$\Sigma - v^{xc}$ are not zero; they would mix the orbitals, and they are
dropped for every reference.

$\Sigma^x$ is computed from the full $B$, never from the PDEP projection of
section @sec:screened-interaction. PDEP rank truncation therefore changes only the correlation
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

> **Verified equation {#eq:gw-sigma-x}** · the exchange self-energy is the same-spin Fock exchange in the RI basis
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
attenuated operators of chapter @sec:operators use the symmetric eigendecomposition
factor $V^{-1/2}$ instead. The GW drivers carry the factor with the tensor
(`MoB::v_inv_sqrt`), because section @sec:screened-interaction needs it to undress the dielectric
eigenvectors. The tensor is a hard memory charge of
$N_{\rm aux}\,n_{\rm act}^2$ doubles, accounted in the pool before it is
built (`guard_b_full`).

## The algorithm {#ri-factors-and-exchange-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | RHF or KS reference, all orbitals | as the reference | the caller |
| 2 | $B^P_{pq}$ over the active square, metric applied | $N_{\rm aux}\,n_{\rm act}^2 N$ for the transform | `build_full_b` |
| 3 | $\Sigma^x_p = -\sum_{iP}(B^P_{pi})^2$ for every active $p$ | $N_{\rm aux}\,n_{\rm act}\,n_o$ | `sigma_x_diag` |
| 4 | the PDEP-RPA run that supplies $W$ (section @sec:screened-interaction) | chapter @sec:rpa | `ferric_rpa::run_pdep_rpa` |

`run_gw` forces the inverse-dielectric stack on in the RPA configuration
(`with_inv_dielectric`), because the correlation self-energy reads it. An
energy-only RPA run leaves it off to save memory. `run_gw` rejects a
`qp_mos` range past the number of orbitals before any expensive step.

## What a KS reference changes {#ri-factors-and-exchange-ks-reference-changes}

On a Hartree–Fock reference $\epsilon_p$ already contains $\Sigma^x_p$
(section @sec:ri-factors-and-exchange-factorization). A Kohn–Sham orbital energy contains the
exchange-correlation potential $v^{xc}_p$ in its place. GW removes
$v^{xc}_p$ and puts in the exchange and correlation self-energy, so the
quasiparticle equation of section @sec:continuation-and-qp-equation carries the static shift

$$ \delta_p = \Sigma^x_p - v^{xc}_p , $$

with $v^{xc}_p$ the diagonal of the XC potential in the MO basis
(`ferric-gw/src/vxc_mo.rs::vxc_diagonal_mo`, the same XC code as the SCF's
Fock build). For a semilocal functional the Kohn–Sham Fock diagonal is
$h_{pp} + J_{pp} + v^{xc}_p$, so with $\Sigma^c = 0$ the corrected energy
$\epsilon_p + \delta_p$ is the Hartree–Fock Fock diagonal evaluated in the
Kohn–Sham orbitals. The shift is computed once, from the starting orbitals.
A hybrid's Fock matrix also holds a fraction of exact exchange, which
$v^{xc}_p$ then includes (section @sec:continuation-and-qp-equation-hybrid-references).

## Hybrid references {#continuation-and-qp-equation-hybrid-references}

What does $v^{xc}_p$ mean when the reference is PBE0 or CAM-B3LYP? The shift
$\delta_p$ removes from $\epsilon_p$ everything the Kohn–Sham Fock matrix holds
besides $h + J$, and puts $\Sigma^x_p$ in its place. A hybrid's Fock matrix
holds a fraction of exact exchange next to the semilocal potential. $\Sigma^x_p$
is the full exchange, so a $v^{xc}_p$ without that fraction counts it twice.

**The Fock matrix.** Closed shell, with $D = 2D^\alpha$, the SCF assembles
(`ferric-scf/src/rhf.rs`, "F = H + J − ½ K_total")

$$ F = h + J[D] + V^{\rm sl} - \tfrac12\big(c_{\rm sr} K^{\rm erfc}_\omega[D] + c_{\rm lr} K^{\rm erf}_\omega[D]\big), \qquad K[D]_{\mu\nu} = \sum_{\lambda\sigma} (\mu\lambda|\nu\sigma) D_{\lambda\sigma} . $$

$V^{\rm sl}$ is the semilocal potential with VV10, and
$(c_{\rm sr}, c_{\rm lr}, \omega)$ is the functional's `KMix`
(`ferric-dft/src/ks.rs::KsXc`, method `k_mix`). A global hybrid has
$\omega = 0$ and $c_{\rm sr} = c_{\rm lr} = c_x$ on one Coulomb $K$; a pure
functional has all three zero. $K$ is linear in $D$, so
$\tfrac12 K[D] = K[D^\alpha]$. Subtract $h + J$ and write the result per spin:

$$ v^{\rm mf}_\sigma = V^{\rm sl}_\sigma - c_{\rm sr} K^{\rm erfc}_\omega[D^\sigma] - c_{\rm lr} K^{\rm erf}_\omega[D^\sigma], \qquad \delta_p = \Sigma^x_p - \big(C^{\sigma T} v^{\rm mf}_\sigma C^\sigma\big)_{pp} . \label{eq:gw-hybrid-vmf} $$

The unrestricted Fock matrix has $-K[D^\sigma]$ in $F^\sigma$
(`ferric-scf/src/uhf.rs`), so (@eq:gw-hybrid-vmf) is also the UKS and ROKS
form.

**The global hybrid.** Transform $K[D^\sigma]$ to the MO basis. With
$D^\sigma = \sum_i C_{\mu i}C_{\nu i}$ over occupied $i$,
$(K[D^\sigma])_{pp} = \sum_i (pi|ip) = -\Sigma^x_p$. Substitute into
(@eq:gw-hybrid-vmf) with $\omega = 0$:

$$ \delta_p = (1 - c_x)\,\Sigma^x_p - V^{\rm sl}_{pp} . \label{eq:gw-hybrid-shift} $$

Check the limits. At $c_x = 0$ the shift reduces to the semilocal one of
section @sec:ri-factors-and-exchange-ks-reference-changes. At $c_x = 1$ and
$V^{\rm sl} = 0$, Hartree–Fock, it is zero, which is the shift an HF
reference gets. For a range-separated hybrid, $\mathrm{erf} + \mathrm{erfc} = 1$
splits $\Sigma^x$ into short- and long-range parts, and
$c_{\rm sr} = c_{\rm lr}$ collapses the result to (@eq:gw-hybrid-shift).

**The size of the double count.** Leave the $K$ term out and $\delta_p$ is
off by $c_x \Sigma^x_p \le 0$, so every quasiparticle energy moves down by
about $Z_pc_x\lvert\Sigma^x_p\rvert$. For an occupied valence
state $\lvert\Sigma^x_p\rvert$ is of order an hartree, so at $c_x = 1/4$ the
error is several eV.

> **Verified equation @eq:gw-hybrid-shift** · the static shift on a hybrid reference
>
> The RKS Fock matrix's exchange term $-\tfrac12 c K[D]$ is $-cK[D^\alpha]$,
> whose MO diagonal is $c\,\Sigma^x_p$; hence
> $\Sigma^x_p - v^{\rm mf}_p = (1-c)\Sigma^x_p - V^{\rm sl}_{pp}$, equal to the
> semilocal shift at $c = 0$ and to zero for Hartree–Fock; the erf/erfc split
> reduces to it when $c_{\rm sr} = c_{\rm lr}$; and PySCF's
> `get_veff − get_j` minus its semilocal potential equals $-\tfrac12 cK$
> built separately (H2O/cc-pVDZ, PBE0 and CAM-B3LYP).
>
> | | |
> |---|---|
> | Proof | `scripts/gw_hybrid_vmf_sympy.py`, claims H1 to H5 |
> | Negative control | the Fock exchange without the $\tfrac12$, and a shift that subtracts $V^{\rm sl}$ alone, must fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/vxc_mo.rs::{vxc_diagonal_mo_on_grids,add_exact_exchange}` |
> | Code test | `g0w0_pbe0_h2o_vs_pyscf` in `crates/ferric-gw/tests/gw_hybrid_reference.rs` |
>
> <!-- verified: id=gw.hybrid-shift; script=scripts/gw_hybrid_vmf_sympy.py; claim=H2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/vxc_mo.rs::{vxc_diagonal_mo_on_grids,add_exact_exchange}; rust_test=ferric-gw/tests/gw_hybrid_reference.rs::g0w0_pbe0_h2o_vs_pyscf; registry=gw -->

**The code.** `ferric-gw/src/vxc_mo.rs::add_exact_exchange` adds the $K$ term
of (@eq:gw-hybrid-vmf) to the AO potential that `add_xc` built, before the MO
transform. The excerpt is the operator choice. Each term's $K[D^\sigma]$ comes
from `ferric-scf/src/rhf.rs::build_jk` and is subtracted with its coefficient:

<!-- excerpt: ferric-gw/src/vxc_mo.rs::add_exact_exchange -->
```rust
    let terms: Vec<(Operator, f64)> = if k_mix.omega > 0.0 {
        vec![
            (Operator::erfc(k_mix.omega), k_mix.sr), // c_sr K^erfc_ω
            (Operator::erf(k_mix.omega), k_mix.lr),  // c_lr K^erf_ω
        ]
    } else {
        vec![(Operator::coulomb(), k_mix.sr)] // c_x K
    };
```

- The closed-shell call passes $D^\alpha$ (`density_alpha`), which is
  $\tfrac12 D$; the factor $\tfrac12$ of the Fock matrix is in that choice.
- A term with a zero coefficient is dropped before any integral is computed.
  A pure functional therefore builds nothing, and its $v^{\rm mf}$ is the
  semilocal potential bit for bit (`pure_functional_has_no_k_term`).
- The $K$ term uses exact four-centre integrals on the operator the SCF's
  `k_mix` names. A reference SCF with RI-K, the default for a functional and
  the only route ferric has for $\omega > 0$, holds $K^{\rm RI}$ in its Fock
  matrix, so $\delta_p$ then carries $c\,(K^{\rm RI} - K)_{pp}$, the fitting
  error. $\Sigma^x$ is the GW auxiliary basis's RI exchange in every case,
  Hartree–Fock included.

## Numerical anchors {#ri-factors-and-exchange-numerical-anchors}

From the table in the module header of `crates/ferric-gw/tests/validation_gw.rs`:

| Quantity | Systems | Error | Bar | Test |
|---|---|---|---|---|
| $\Sigma^x$ against PySCF's density-fitted exchange diagonal, same auxiliary basis | the validation set; worst case CH2 triplet | $4.9\times10^{-9}$ Ha | $3\times10^{-8}$ Ha (`TOL_SX`) | `g0w0_hf_{h2o,nh3,n2}_vs_pyscf_gw_ac`, `u_g0w0_triplets_o2_ch2_vs_pyscf_ugw_ac` in `crates/ferric-gw/tests/validation_gw.rs` |

**Known limits (2026-11-19).** Diagonal $\Sigma$ only: off-diagonal
self-energy elements are dropped, so the quasiparticle orbitals are the
reference orbitals. $\Sigma^x$ is the RI exchange of the GW auxiliary basis,
on every reference. The full-square tensor $B$, $N_{\rm aux}\,n_{\rm act}^2$
doubles, is held in memory whole; `guard_b_full` refuses a run whose pool
cannot hold it.

## Sources {#ri-factors-and-exchange-sources}

| Source | What it gives |
|---|---|
| `scripts/gw_ucohsex_sympy.py`, claim U1 | the proof of the per-spin exchange sum |
| `scripts/gw_hybrid_vmf_sympy.py`, claims H1 to H5 | the hybrid static shift |
| Hedin, *Phys. Rev.* **139**, A796 (1965) | the split of $\Sigma$ into exchange and correlation |
| Ren, Rinke, Blum *et al.*, *New J. Phys.* **14**, 053020 (2012) | RI GW over the full MO square |
