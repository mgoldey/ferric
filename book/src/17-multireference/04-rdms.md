# 17.4 Reduced density matrices

<!-- section: 17.4; status: ready; registry: casci;
     theory: crates/ferric-ci/theory.md#"Reduced density matrices (`src/rdm.rs`)", "NEVPT RDMs (`src/rdm3.rs`)" -->

Everything after the CI solve reads the wavefunction through its reduced
density matrices: the CAS-CI energy can be rebuilt from $\gamma$ and
$\Gamma$, the CASSCF gradient (section 17.5) needs both, and SC-NEVPT2
(section 17.6) needs the 3-RDM, two contractions of the 4-RDM and the hole
RDMs. ferric uses PySCF's index conventions throughout, so that PySCF
formulas port without reindexing.

| Symbol | Meaning | Code name |
|---|---|---|
| $\gamma_{pq}$ | $\langle E_{pq}\rangle$ | `CasCiRdms::dm1[p,q]` |
| $\Gamma_{pqrs}$ | $\sum_{\sigma\tau}\langle a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}\rangle$ | `CasCiRdms::dm2[p,q,r,s]` |
| $D_{rs}$ | $E_{rs}C$, one CI-sized vector per pair | built per block of α rows |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/rdm.rs::{spin_summed_rdms,rdm_energy,spin_summed_transition_rdms}`, `ferric-ci/src/rdm3.rs::{nevpt_rdms,hole_rdms,e_product_rdms}` |
| Python | `run_casci(compute_rdms=True)` returns `dm1`, `dm2` |

## 17.4.1 The 1- and 2-RDM

<!-- include: crates/ferric-ci/theory.md#"Reduced density matrices (`src/rdm.rs`)" -->

> **Verified equation 17.4** · the RDM energy and the identities
>
> On Jordan–Wigner matrices of a three-orbital active space and random CI
> vectors in the (2,2) and (2,1) sectors, with $\gamma$ and $\Gamma$ taken
> as spin-summed expectation values:
> $E_\text{core} + \sum h\gamma + \tfrac12\sum(pq|rs)\Gamma = \langle C|H|C\rangle + E_\text{core}$;
> $\Gamma_{pqrs} = \langle E_{pq}E_{rs}\rangle - \delta_{qr}\gamma_{ps}$;
> ${\rm tr}\,\gamma = N$, $\sum\Gamma_{pprr} = N(N-1)$,
> $\sum_r\Gamma_{pqrr} = (N-1)\gamma_{pq}$ and
> $\Gamma_{pqrs} = \Gamma_{rspq} = \Gamma_{qpsr}$.
>
> | | |
> |---|---|
> | Proof | `scripts/ci_rdm_casscf_gradient_sympy.py`, checks `rdm-energy`, `rdm-identities` |
> | Negative control | the runner's mutation: $\Gamma$ built with $a^\dagger_pa^\dagger_ra_qa_s$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/rdm.rs::{spin_summed_rdms,rdm_energy}` |
> | Code test | `rdm_energy_equals_davidson_eigenvalue_h2o_cas44` in `crates/ferric-ci/tests/casci_rdms.rs` |
>
> <!-- verified: id=ci.rdm.energy; script=scripts/ci_rdm_casscf_gradient_sympy.py; claim=rdm-energy; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/rdm.rs::{spin_summed_rdms,rdm_energy}; rust_test=ferric-ci/tests/casci_rdms.rs::rdm_energy_equals_davidson_eigenvalue_h2o_cas44; registry=casci -->

The trace identities alone are blind to a wrong sign or index in the
cumulant, which vanishes for one determinant and is summed away by a trace.
The energy identity is not, and the Rust tests also compare with PySCF
`make_rdm12` element-wise.

## 17.4.2 The NEVPT RDMs

<!-- include: crates/ferric-ci/theory.md#"NEVPT RDMs (`src/rdm3.rs`)" -->

These definitions were fixed against PySCF by exhaustive index permutation
on dense Fock-space expectation values, and the Rust tests compare them with
PySCF on ferric's CI vectors. No proof script covers them yet; they are part
of the SC-NEVPT2 gap of section 17.6.

## 17.4.3 Checks

| What | Systems | Test |
|---|---|---|
| RDM energy = $c\cdot Hc$ = Davidson eigenvalue (to the residual) | H2O CAS(4,4), N2 CAS(6,6) | `crates/ferric-ci/tests/casci_rdms.rs` |
| traces, partial trace, symmetries | same | `crates/ferric-ci/tests/casci_rdms.rs` |
| `dm1`, `dm2` vs PySCF `make_rdm12` on its own ground state, rotated into ferric's orbitals | same | `crates/ferric-ci/tests/casci_rdms.rs` |
| 3-RDM, $f^\text{ca}$, $f^\text{ac}$ and hole RDMs vs PySCF on the same CI vector; partial traces; single determinants | H2O CAS(4,4), N2 CAS(6,6) at 1.10 and 2.20 Å | `crates/ferric-ci/tests/casci_rdm3.rs` |

## 17.4.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "Reduced density matrices", "NEVPT RDMs" | definitions, construction, cost and memory |
| PySCF `fci.direct_spin1.make_rdm12`, `fci.rdm.make_dm1234`, `mrpt.nevpt2` | the conventions matched |
| `scripts/ci_rdm_casscf_gradient_sympy.py` | `rdm-energy`, `rdm-identities` |

**Known limits (2026-11-15).** Spin-summed RDMs of one real root; the
transition RDMs (`spin_summed_transition_rdms`) serve the CASSCF stability
check. No 4-RDM is ever formed: NEVPT2 receives only its two contractions
with $(aa|aa)$.
