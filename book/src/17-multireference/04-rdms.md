# Reduced density matrices {#rdms}

<!-- status: ready; registry: casci;
     theory: crates/ferric-ci/theory.md#"Reduced density matrices (`src/rdm.rs`)", "NEVPT RDMs (`src/rdm3.rs`)" -->

The CI vector has $N_\text{det}$ coefficients, but $H$ has only one- and
two-electron terms, so its expectation value needs only the one- and
two-electron reduced density matrices $\gamma$ ($n^2$ numbers) and
$\Gamma$ ($n^4$). Everything after the CI solve reads the wavefunction
through them: the CAS-CI energy can be rebuilt from $\gamma$ and $\Gamma$,
the CASSCF gradient (section @sec:casscf) needs both, and SC-NEVPT2 (section @sec:nevpt2)
needs the 3-RDM, two contractions of the 4-RDM and the hole RDMs. ferric
uses PySCF's index conventions throughout, so that PySCF formulas port
without reindexing.

| Symbol | Meaning | Code name |
|---|---|---|
| $\gamma_{pq}$ | $\langle E_{pq}\rangle$ | `CasCiRdms::dm1[p,q]` |
| $\Gamma_{pqrs}$ | $\sum_{\sigma\tau}\langle a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}\rangle$ | `CasCiRdms::dm2[p,q,r,s]` |
| $D_{rs}$ | $E_{rs}C$, one CI-sized vector per pair | built per block of α rows |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/rdm.rs::{spin_summed_rdms,rdm_energy,spin_summed_transition_rdms}`, `ferric-ci/src/rdm3.rs::{nevpt_rdms,hole_rdms,e_product_rdms}` |
| Python | `run_casci(compute_rdms=True)` returns `dm1`, `dm2` |

## The 1- and 2-RDM {#rdms-1-2-rdm}

### Definitions and the energy

For the real, normalized CI vector $C$ over the active orbitals, with
$E_{pq} = E^\alpha_{pq} + E^\beta_{pq}$, the spin-summed RDMs in the index
convention of PySCF `fci.direct_spin1.make_rdm12` are

$$
\gamma_{pq} = \langle C|E_{pq}|C\rangle ,\qquad
\Gamma_{pqrs} = \sum_{\sigma\tau}\langle C|a^\dagger_{p\sigma} a^\dagger_{r\tau}
                 a_{s\tau} a_{q\sigma}|C\rangle
              = \langle C|E_{pq}E_{rs}|C\rangle - \delta_{qr}\gamma_{ps}.
$$

The second form of $\Gamma$ is the operator identity of section @sec:direct-sigma-davidson-split,
$E^\sigma_{pq}E^\tau_{rs} = \delta_{qr}\delta_{\sigma\tau}E^\sigma_{ps} + a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}$,
summed over $\sigma\tau$ and taken as an expectation value. In the code,
`dm1[p,q]` $=\gamma_{pq}$ and `dm2[p,q,r,s]` $=\Gamma_{pqrs}$.

The index order of $\Gamma$ follows the integral: $(p, q)$ belong to
electron 1 and $(r, s)$ to electron 2, as in $(pq|rs)$. Take the
expectation value of $H$ of section @sec:determinants-hamiltonian term by term and the energy is

$$
E = E_\text{core} + \sum_{pq} h_{pq}\gamma_{pq}
  + \tfrac12\sum_{pqrs} (pq|rs)\,\Gamma_{pqrs}.
$$

For an $N$-electron state: $\operatorname{tr}\gamma = N$,
$\sum_{pr}\Gamma_{pprr} = N(N-1)$, $\sum_r \Gamma_{pqrr} = (N-1)\gamma_{pq}$
and $\Gamma_{pqrs} = \Gamma_{rspq}$. For a real state also
$\gamma = \gamma^\mathsf{T}$ and $\Gamma_{pqrs} = \Gamma_{qpsr}$. The
eigenvalues of $\gamma$ lie in $[0, 2]$.

### Two limits

**One doubly occupied orbital.** $\gamma_{00} = 2$. In $\Gamma_{0000}$ the
terms with $\sigma = \tau$ try to annihilate the same spin orbital twice and
vanish; the two with $\sigma \ne \tau$ give 1 each, so $\Gamma_{0000} = 2 =
N(N-1)$. The energy is $2h_{00} + (00|00)$, as in section @sec:direct-sigma-davidson-split.

**The (2,2) wavefunction of section @sec:slater-condon.** For
$C = c_1|0\alpha\,0\beta\rangle + c_2|1\alpha\,1\beta\rangle$ the nonzero
elements are

$$
\gamma = \begin{pmatrix} 2c_1^2 & 0 \\ 0 & 2c_2^2\end{pmatrix},\qquad
\Gamma_{0000} = 2c_1^2,\quad \Gamma_{1111} = 2c_2^2,\quad
\Gamma_{0101} = \Gamma_{1010} = 2c_1c_2 .
$$

$\gamma_{01} = 0$ because $E_{01}$ moves one electron and leaves a
determinant that is not in $C$. $\Gamma_{0101}$ moves a pair from orbital 1
to orbital 0, so it connects the two configurations and is proportional to
the product of their coefficients. The energy
formula gives
$2c_1^2h_{00} + 2c_2^2h_{11} + c_1^2(00|00) + c_2^2(11|11) + 2c_1c_2(01|01)$,
the expectation value of the $2\times2$ matrix of section @sec:slater-condon. The
eigenvalues of $\gamma$, $2c_1^2$ and $2c_2^2$, are the natural orbital
occupations. Their distance from 2 and 0 measures the static correlation,
and at dissociation both approach 1. These elements and the energy were
checked on Jordan–Wigner matrices in scratch.

> **Verified equation {#eq:ci-rdm-energy}** · the RDM energy and the identities
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

### Construction

For each pair $rs$, form the CI-sized vector $D_{rs} = E_{rs}C$ from the
excitation tables of the direct sigma. A table entry $E_{pq}|I\rangle =
\epsilon|T\rangle$ says $\langle T|E_{pq}|I\rangle = \epsilon$, so its
transpose is $\langle I|E_{qp}|T\rangle = \epsilon$, and

$$
D_{qp}(I_\alpha,I_\beta) \mathrel{+}= \epsilon\, C(T,I_\beta)\ \ (\alpha\text{ entry of }I_\alpha),
\qquad
D_{qp}(I_\alpha,I_\beta) \mathrel{+}= \epsilon\, C(I_\alpha,T)\ \ (\beta\text{ entry of }I_\beta).
$$

A single $E^\beta$ carries $(-1)^{N_\alpha}$ twice, so the β phase is the
single-channel one. Now use $E_{pq}^\dagger = E_{qp}$:
$\langle C|E_{pq} = (E_{qp}C)^\mathsf{T} = D_{qp}^\mathsf{T}$, so

$$
\gamma_{rs} = \sum_I C(I)\,D_{rs}(I),\qquad
\langle C|E_{pq}E_{rs}|C\rangle = \sum_I D_{qp}(I)\,D_{rs}(I).
$$

The second sum, over all pairs at once, is one $n^2\times n^2$ GEMM
$DD^\mathsf{T}$. Write $D = D^\alpha + D^\beta$. The αα, ββ and αβ (cross)
parts of $\Gamma$ are the blocks of that product, and none is assembled
separately. `ferric-ci/src/rdm.rs::spin_summed_rdms` subtracts the
$\delta_{qr}$ term at the end:

```rust
let mut v = g[[q * n + p, r * n + s]];   // ⟨E_pq E_rs⟩ = Σ_I D_qp(I) D_rs(I)
if q == r {
    v -= dm1[[p, s]];                    // − δ_qr γ_ps
}
dm2[[p, q, r, s]] = v;                   // Γ_pqrs
```

The row index of `g` is $qp$, not $pq$: that is the transpose from
$\langle C|E_{pq} = D_{qp}^\mathsf{T}$.

The determinants are processed in fixed blocks of α rows, about
`CHUNK_DETS` = 4096 determinants each. Each block's $D$ is built,
contracted and dropped, and block results are summed in block order, so the
result does not depend on the thread count. The cost is $O(N_\text{det}\,n^4)$
for the GEMM plus $O(N_\text{det}(n^\alpha_\text{exc}+n^\beta_\text{exc}))$
for $D$. The scratch is one $n^2\times$block $D$ per worker and
`BATCH_CHUNKS` = 64 block results of $n^4+n^2$ doubles, charged to the
CAS-CI memory plan when `compute_rdms` is set.

## The NEVPT RDMs {#rdms-nevpt-rdms}

### What NEVPT2 needs

Strongly contracted NEVPT2 needs three more inputs: the normal-ordered
3-RDM, two contractions of the 4-RDM with the active $(aa|aa)$ block, and
hole RDMs. ferric uses PySCF's definitions, so that `pyscf.mrpt.nevpt2`
formulas port directly:

$$
\Gamma^{(3)}_{p_1q_1p_2q_2p_3q_3} = \sum_{\sigma\tau\upsilon}
  \langle a^\dagger_{p_1\sigma}a^\dagger_{p_2\tau}a^\dagger_{p_3\upsilon}
  a_{q_3\upsilon}a_{q_2\tau}a_{q_1\sigma}\rangle ,
$$

$$
f^\text{ca}_{abcdef} = \sum_{ijk}(ki|fj)\,\langle E_{ab}E_{cd}E_{ej}E_{ki}\rangle ,\qquad
f^\text{ac}_{abcdef} = \sum_{ijk}(ik|je)\,\langle E_{ab}E_{cd}E_{jf}E_{ik}\rangle ,
$$

$$
\eta^{(k)}_{p_1..p_k,a_1..a_k} = \sum_\text{spins}
  \langle a_{p_1}\cdots a_{p_k}\,a^\dagger_{a_k}\cdots a^\dagger_{a_1}\rangle .
$$

$\Gamma^{(3)}$ extends the pattern of $\Gamma$: electron $k$ owns the index
pair $(p_k, q_k)$. The hole RDMs $\eta^{(k)}$ put the annihilators first,
so they count the empty spin orbitals the way $\gamma$ counts the occupied
ones. The indices $i, j, k$ in $f^\text{ca}$ and $f^\text{ac}$ are active.

How each maps onto PySCF:

- $\Gamma^{(3)}$ is `fci.rdm.make_dm1234` followed by `reorder_dm1234`.
- $f^\text{ca}$ and $f^\text{ac}$ are `mrpt.nevpt2._contract4pdm` with the
  kernels `NEVPTkern_cedf_aedf` and `NEVPTkern_aedf_ecdf`. They contract
  the E-product 4-RDM, not the normal-ordered one. The einsum comments in
  PySCF's `make_a16` and `make_a22` that list the 4-RDM in the argument
  order above agree with the kernels to $4\times10^{-14}$ on random states.
  The comment `kbij,rpqjkiac` above the first `f3ca` use in `make_a16` does
  not (difference $\sim 40$).
- $\eta^{(k)}$ is `make_hdm1/2/3` fed with the E-product RDMs of
  `fci.rdm.make_dm123`, which is what PySCF's NEVPT2 driver passes. Fed
  with the reordered ones it matches no operator ordering.

The operator forms above were identified by exhaustive index permutation
against dense Fock-space expectation values.

### Construction

Everything is built from CI-sized vectors, as for the 2-RDM. Start from
$X_{rs} = E_{rs}C$, the $D$ of the 2-RDM build. As in @sec:rdms-1-2-rdm,
$\langle C|E_{ab}E_{cd} = (E_{dc}E_{ba}C)^\mathsf{T} = (E_{dc}X_{ba})^\mathsf{T}$,
so

$$
\langle E_{ab}E_{cd}E_{ef}\rangle = \langle E_{dc}X_{ba}|X_{ef}\rangle .
$$

For $f^\text{ca}$, the rightmost operators carry the integral. Sum them
into one vector before taking the overlap: with
$W_{fj} = \sum_{ki}(ki|fj)X_{ki}$ and $R_{ef} = \sum_j E_{ej}W_{fj}$,

$$
f^\text{ca}_{abcdef} = \langle E_{dc}X_{ba}|R_{ef}\rangle .
$$

For $f^\text{ac}$, use $(ik|je) = (ki|ej)$, so the same $W$ appears as
$W_{ej}$, and with $R'_{ef} = \sum_j E_{jf}W_{ej}$

$$
f^\text{ac}_{abcdef} = \langle E_{dc}X_{ba}|R'_{ef}\rangle .
$$

$X$, $W$, $R$ and $R'$ are $n^2$ vectors over all determinants; $W$ is one
GEMM. $L_{dc,ba} = E_{dc}X_{ba}$ is built for one block of α rows at a time
(about $2^{21}/n^4$ determinants) and contracted as $L^\mathsf{T}[X|R|R']$,
an $n^4\times3n^2$ product. Blocks are summed in block order. No 4-RDM is
formed.

The E-product 3-RDM becomes the normal-ordered one by the PySCF
`reorder_dm123` formula, which moves each annihilator to the right of the
creators that follow it and subtracts one term for every contraction this
produces:

$$
\Gamma^{(3)}_{pqrstu} = \langle E_{pq}E_{rs}E_{tu}\rangle
 - \delta_{qr}\Gamma_{pstu} - \delta_{st}\Gamma_{pqru} - \delta_{qt}\Gamma_{purs}
 - \delta_{qr}\delta_{st}\gamma_{up}.
$$

The cost is $O(3N_\text{det}n^6)$. The memory is $4n^2N_\text{det} + 6n^6$
resident plus one $L$ block and its $3n^6$ result per worker, reserved
through a `MemoryPlan` before any allocation (`nevpt_rdm_memory_elems`).

Identities: $\sum_x\Gamma^{(3)}_{pqrsxx} = (N-2)\Gamma_{pqrs}$,
$\sum_x\eta^{(3)}_{pqx,abx} = (2n-N-2)\eta^{(2)}_{pq,ab}$ and
$\sum_x\eta^{(2)}_{px,ax} = (2n-N-1)\eta^{(1)}_{pa}$. For one determinant
every RDM is the antisymmetrized product of the spin-resolved $\gamma$
(particles) or $\eta = 1-\gamma^\mathsf{T}$ (holes). Both checks are blind to
a wrong sign or index in the cumulant, which vanishes for one determinant
and is summed away by a trace.

These definitions were fixed against PySCF by exhaustive index permutation
on dense Fock-space expectation values, and the Rust tests compare them with
PySCF on ferric's CI vectors. Verified equation @eq:ci-nevpt2-rdms in section @sec:nevpt2 checks
each input by two routes on a model CI vector.

## Checks {#rdms-checks}

| What | Systems | Test |
|---|---|---|
| RDM energy = $c\cdot Hc$ = Davidson eigenvalue (to the residual) | H2O CAS(4,4), N2 CAS(6,6) | `crates/ferric-ci/tests/casci_rdms.rs` |
| traces, partial trace, symmetries | same | `crates/ferric-ci/tests/casci_rdms.rs` |
| `dm1`, `dm2` vs PySCF `make_rdm12` on its own ground state, rotated into ferric's orbitals | same | `crates/ferric-ci/tests/casci_rdms.rs` |
| 3-RDM, $f^\text{ca}$, $f^\text{ac}$ and hole RDMs vs PySCF on the same CI vector; partial traces; single determinants | H2O CAS(4,4), N2 CAS(6,6) at 1.10 and 2.20 Å | `crates/ferric-ci/tests/casci_rdm3.rs` |

### What is proven and what is not {#rdms-proven}

`crates/ferric-ci/tests/casci_rdms.rs` fails under each of three
mutations: a dropped β phase, the missing αβ cross terms, and a doubled
diagonal.

`crates/ferric-ci/tests/casci_rdm3.rs` runs on ferric's CI vectors for
H2O/6-31G CAS(4,4) and N2/cc-pVDZ CAS(6,6) at 1.10 and 2.20 Å. The
references come from `scripts/gen_casci_rdm3_refs.py`, which applies PySCF's
`make_dm1234` with `reorder_dm1234`, `_contract4pdm` and `make_hdm*` to the
same vector. The test runs its anchors first:

| Check | Measured | Bar |
|---|---|---|
| partial trace dm3 → dm2 | $\le 8.9\times10^{-15}$ | |
| partial trace hdm3 → hdm2 | $\le 1.7\times10^{-14}$ | |
| single determinants | exactly zero difference | |
| dm1 to dm3 and hdm1 to hdm3 vs PySCF, element-wise | $\le 6\times10^{-15}$ | $100N_\text{det}\varepsilon\max\lvert\cdot\rvert$, $1.3\times10^{-12}$ to $6.8\times10^{-11}$ |
| $f^\text{ca}$, $f^\text{ac}$ vs PySCF, element-wise (H2O) | $7.1\times10^{-15}$ | $1.8\times10^{-11}$ |
| $f^\text{ca}$, $f^\text{ac}$ through 32 random $\pm1$ projections | $\le 2.8\times10^{-12}$ | $7\times10^{-8}$ to $10^{-5}$ |
| an independent dense Fock-space construction (H2O): dm3, hdm3 | $5\times10^{-16}$, $3.5\times10^{-15}$ | |
| the same: $f^\text{ca}$, $f^\text{ac}$ as explicit contractions of a stored E-product 4-RDM | $1.1\times10^{-14}$ | |
| dense fallback: Davidson stopped at `max_iter` 1 (residual $3.9\times10^{-2}$), then `conv_thresh` $10^{-13}$ met at residual $6.1\times10^{-14}$ | | |
| dense vs Davidson root, N2 1.10 Å | $\lvert\Delta E\rvert = 1.4\times10^{-13}$ | $4.5\times10^{-10}$ |

Seven mutations fail it: the β Jordan–Wigner phase forced to $+1$ (dm3 off
by 2.32 on N2 at 2.20 Å); one index swap in the reorder (off by 0.82); the
same-spin block omitted (off by 2.84); a swapped destination index of $R'$
($f^\text{ac}$ projections off by $1.7\times10^{2}$ on N2 at 2.20 Å); the
hdm3 coefficient 2 changed to 1; an uncommitted `MemoryPlan` (a starved
budget is not refused); and the dense fallback disabled (the residual stays
at $3.9\times10^{-2}$).

## Sources {#rdms-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "Reduced density matrices", "NEVPT RDMs" | definitions, construction, cost and memory |
| PySCF `fci.direct_spin1.make_rdm12`, `fci.rdm.make_dm1234`, `mrpt.nevpt2` | the conventions matched |
| `scripts/ci_rdm_casscf_gradient_sympy.py` | `rdm-energy`, `rdm-identities` |

**Known limits (2026-11-15).** Spin-summed RDMs of one real root; the
transition RDMs (`spin_summed_transition_rdms`) serve the CASSCF stability
check. No 4-RDM is ever formed: NEVPT2 receives only its two contractions
with $(aa|aa)$.
