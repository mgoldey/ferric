# 10.1 Closed-shell RI-MP2 energy

<!-- section: 10.1; status: edit; registry: ri-mp2;
     theory: crates/ferric-mp2/theory.md#"1. Density fitting", "2. Closed-shell RI-MP2" -->

Hartree–Fock gives each electron the *average* field of all the others.
Real electrons do better than that: they see each other coming and step
aside. MP2 is the cheapest honest account of that sidestep. It is
second-order perturbation theory, with Hartree–Fock as the starting point.

This section builds closed-shell RI-MP2 from the textbook formula to the
loop that actually runs. Almost everything later in Part III starts here:
spin-component scaling, attenuation, orbital optimization, the Laplace
forms and local MP2 all reuse this pair loop and these fitted integrals.
Getting this one right pays off many times.

The plan has three steps:

1. Write the energy in spin orbitals, where it is short and obviously right.
2. Sum out spin, which leaves two physically different pieces.
3. Make the integrals cheap with density fitting, and look at the code that
   does it.

Here is the notation, once:

| Symbol | Meaning | In the code |
|---|---|---|
| $i, j$ | occupied spatial orbitals (active, i.e. after frozen core) | `0..nocc`, shifted by `first_occ` into `eps` |
| $a, b$ | virtual spatial orbitals | `0..nvir`, shifted by `nocc_total` into `eps` |
| $\mu, \nu$ | atomic orbitals | `0..nbas` |
| $P, Q$ | auxiliary (fitting) functions | `0..naux` |

Entry points: Rust `ferric_mp2::rimp2::ri_mp2`, CLI `method.kind = "rimp2"`,
Python `ferric.run_rimp2`.

## 10.1.1 The energy, and why it splits in two

Start with spin orbitals and canonical Hartree–Fock orbitals. Second-order
perturbation theory says to promote two electrons, $ij \to ab$, ask how
strongly the Hamiltonian connects that double excitation to the ground
state, square it, and divide by what the excitation costs:

$$ E^{(2)} = -\frac14 \sum_{ijab} \frac{|\langle ij\|ab\rangle|^2}{\Delta_{ij}^{ab}}, \qquad \Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j . \tag{10.1} $$

Every term is negative, because $\Delta > 0$ for a ground state with a gap.
So MP2 always lowers the energy, and small gaps make it lower *fast*. Keep
that in mind; it comes back when the method misbehaves.

The $\tfrac14$ is bookkeeping. The antisymmetrized integral
$\langle ij\|ab\rangle = \langle ij|ab\rangle - \langle ij|ba\rangle$ is the
same term counted four ways ($i \leftrightarrow j$, $a \leftrightarrow b$).

A closed-shell reference has every spatial orbital doubly occupied, so we
should not pay for spin. Give each of the four electrons a spin label and
see what survives.

**Opposite spins** ($i\alpha, j\beta \to a\alpha, b\beta$). Spin has to be
conserved electron by electron, so the exchange half
$\langle ij|ba\rangle$ would need $i\alpha \to b\beta$, which is zero. Only
the direct integral is left. In chemists' notation that is $(ia|jb)$.

**Same spins** ($i\alpha, j\alpha \to a\alpha, b\alpha$). Both halves
survive, so the integral is $(ia|jb) - (ib|ja)$.

Add up the spin cases. The $\tfrac14$ is cancelled by the ways of choosing
which electron is which. The result:

$$ E_{\mathrm{OS}} = -\sum_{ijab} \frac{(ia|jb)^2}{\Delta_{ij}^{ab}}, \qquad E_{\mathrm{SS}} = -\sum_{ijab} \frac{(ia|jb)\,\big[(ia|jb) - (ib|ja)\big]}{\Delta_{ij}^{ab}} . \tag{10.2} $$

$$ E^{(2)} = E_{\mathrm{OS}} + E_{\mathrm{SS}} = -\sum_{ijab} \frac{(ia|jb)\,\big[\,2(ia|jb) - (ib|ja)\,\big]}{\Delta_{ij}^{ab}} . \tag{10.3} $$

Why is $E_{\mathrm{SS}}$ smaller? The minus sign in
$(ia|jb) - (ib|ja)$ is the Pauli principle at work. Two same-spin electrons
already avoid each other in Hartree–Fock: that is the exchange hole. So
there is less left for correlation to fix. Opposite-spin electrons get no
such help from Hartree–Fock, so they carry most of the correlation energy.

Here is water in cc-pVDZ (cc-pVDZ-RI fitting basis, from the
`a410529d0` build):

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

# c_os = c_ss = 1 is plain MP2, split into its two pieces
r = ferric.run_scs_mp2(mol, bs, aux, c_os=1.0, c_ss=1.0)
print(r.e_os, r.e_ss, r.e_os + r.e_ss)
# -0.15244  -0.05159  -0.20403   (Hartree)
```

About three quarters of the correlation energy is opposite-spin. That
imbalance is the whole reason SCS-MP2 exists. If one piece is
systematically over- or under-counted relative to the other, scale them
separately: $c_{\mathrm{OS}}E_{\mathrm{OS}} + c_{\mathrm{SS}}E_{\mathrm{SS}}$
(section 10.3); the code returns both in `ferric-mp2/src/rimp2.rs::SpinComponents`. It also explains why SOS-MP2 (§12.3) can drop
$E_{\mathrm{SS}}$ altogether and buy a cheaper algorithm with it.

> **Verified equation 10.2** · spin adaptation of the MP2 energy
>
> The spatial sums $E_{\mathrm{OS}}$ and $E_{\mathrm{SS}}$ of (10.2) equal the
> opposite-spin and same-spin parts of the spin-orbital sum (10.1), and their
> sum equals (10.1).
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 1 |
> | Negative control | $E_{\mathrm{SS}}$ without the exchange term must differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::pair_energy` |
>
> <!-- verified: id=mp2.cs.spin-adaptation; script=scripts/mp2_variant_identities.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::pair_energy -->

### Reading the inner loop

Equation (10.2) is the code almost word for word. This is the heart of
`ferric-mp2/src/rimp2.rs::pair_energy` with the κ-regularization branch removed:

```rust
let jcol = (j - i) * nvir;              // where pair (i,j) starts in the block
let e_ij = eps[first_occ + i] + eps[first_occ + j];
for a in 0..nvir {
    for b in 0..nvir {
        let g_ab = g_i[(a, jcol + b)];  // (ia|jb)
        let g_ba = g_i[(b, jcol + a)];  // (ib|ja): same block, transposed
        let denom = e_ij - eps[nocc_total + a] - eps[nocc_total + b];
        e_os_ij += g_ab * g_ab / denom;
        e_ss_ij += g_ab * (g_ab - g_ba) / denom;
    }
}
```

Three things are worth noticing.

- **The sign is hiding in `denom`.** The code computes
  $\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b = -\Delta$,
  which is negative, and has no leading minus. The two minus signs cancel.
  It's the same number as (10.2), but if you port this loop, port the pair of
  signs together or not at all.
- **The exchange integral costs nothing.** $(ib|ja)$ sits in the same block
  as $(ia|jb)$, transposed in $a,b$. One block of integrals per pair feeds
  both spin cases.
- **The loop is per pair on purpose.** For a fixed $(i,j)$ everything is a
  dense $n_v \times n_v$ block. That is the unit later chapters parallelize,
  ship to a GPU (`rimp2_gpu` calls this same function, so the two paths
  differ only in how `g_i` was built), and in local MP2 *throw away* (§13).

## 10.1.2 Density fitting: making $(ia|jb)$ affordable

The loop above needs $(ia|jb)$ for every $i, j, a, b$. Building those from
four-center AO integrals means storing $N^4$ numbers and running an $O(N^5)$
transformation. For anything bigger than a small molecule, that is where
the time and memory go.

The fix rests on one observation. $(ia|jb)$ is the Coulomb interaction of
two *charge distributions*, $\rho_{ia} = \phi_i\phi_a$ and
$\rho_{jb} = \phi_j\phi_b$. Expand each distribution in an auxiliary basis
of atom-centered functions $\chi_P$, built for exactly this job, and the
four-index object factors into two three-index ones:

$$ (ia|jb) \approx \sum_P B^P_{ia} B^P_{jb}, \qquad B^P_{ia} = \sum_Q M_{PQ}\,(Q|ia), \qquad M^{\mathsf T}M = V^{-1}, \qquad V_{PQ} = (P|Q) . \tag{10.4} $$

Read it right to left:
- $(Q|ia)$ is how much $\rho_{ia}$ "looks like" $\chi_Q$, measured with the
  Coulomb operator;
- $V^{-1}$ undoes the double counting from the $\chi$'s overlapping each
  other;
- $B$ is the result, a "dressed" factor that you contract like a matrix.

Be honest about what this buys. It does not change the scaling. Matt's
dissertation puts it plainly: "the fundamental fifth-order scaling is not
ameliorated" (ch. 4, after Eq. 4.5). The last contraction is still
$O(n_o^2 n_v^2 N_{\mathrm{aux}})$. What changes is the prefactor and the
storage: no $N^4$ integrals, and the expensive step becomes a large matrix
multiply, which is the one thing computers are reliably fast at.

### Why the code uses Cholesky and not $V^{-1/2}$

Most papers write $M = V^{-1/2}$, the symmetric inverse square root. But look
at where $M$ enters: only through $B^{\mathsf T}B = (ia|P)\,M^{\mathsf T}M\,(Q|jb)$.
*Any* $M$ with $M^{\mathsf T}M = V^{-1}$ gives the same energy, to the last
digit you'd trust. So pick the cheapest one.

That's a Cholesky factor. Write $V = LL^{\mathsf T}$ and take $M = L^{-1}$:

$$ M^{\mathsf T}M = L^{-\mathsf T}L^{-1} = (LL^{\mathsf T})^{-1} = V^{-1}. \checkmark $$

One factorization ($N_{\mathrm{aux}}^3/3$ flops) replaces an
eigendecomposition. In the code it's `ferric-mp2/src/rimp2.rs::cholesky_inverse_sqrt`, and the comment
at the end says exactly this:

```rust
let l = v.cholesky(UPLO::Lower)?;        // V = L L^T
// ... forward substitution builds l_inv = L^{-1} ...
// V^{-1/2} = L^{-1} (so that B = L^{-1} (Q|ia) and B^T B = (ia|P) V^{-1} (Q|jb))
Ok(l_inv)
```

There's a bonus. Cholesky is also a *test*: it fails if $V$ is not
positive definite. For the Coulomb operator that can't happen with a sound
basis and correct integrals. So when it does fail, ferric stops instead of
quietly fitting with a broken metric.

The exception proves the rule. The long-range operators erf and terf
(Chapter 4) give metrics that genuinely lose rank in a Coulomb-fitted
auxiliary basis. For those, `ferric-mp2/src/rimp2.rs::metric_inverse_sqrt` switches to an
eigendecomposition that drops the tiny eigenvalues
(`ferric-mp2/src/rimp2.rs::eigh_inverse_sqrt`).
That isn't a workaround. It's the right tool when the matrix really is
singular, and the wrong one when it shouldn't be.

> **Unverified equation 10.5** · the fitting error is quadratic
>
> With the same operator in the metric and the three-index integrals, the
> error of (10.4) is the self-interaction of the fitting residuals:
>
> $$ (ia|jb) - \sum_P B^P_{ia}B^P_{jb} = (\delta_{ia}|\delta_{jb}), \qquad \delta_{ia} = \phi_i\phi_a - \sum_{PQ} \chi_P\,V^{-1}_{PQ}\,(Q|ia) . $$
>
> | | |
> |---|---|
> | Proof | none committed. To add: a claim in `scripts/mp2_variant_identities.py` that builds a model four-index tensor from a larger "exact" factor, fits it in a subspace and checks the residual identity. |
> | Status | derived in the text; no script, no test |
>
> <!-- unverified: id=df.quadratic-error; todo=add claim to scripts/mp2_variant_identities.py -->

Equation 10.5 follows because the fitted density is the projection of
$\phi_i\phi_a$ onto the auxiliary space in the $\hat w$ inner product, so the
residual is orthogonal to every auxiliary function. It is the reason the
metric and the three-index integrals share one operator. A different metric
operator (`RiMp2Config::metric_op`) breaks the orthogonality and makes the
error first order; the code keeps it as a measurement option, not a default.

## 10.1.3 The algorithm

The dissertation's shared-memory algorithm (chapter 4, Table 4.1) has four
steps. The code keeps the four steps and changes where each intermediate
lives. $n_o$, $n_v$, $N$, $N_{\mathrm{aux}}$ are $O$, $V$, $N$, $X$ in the
dissertation's table.

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | $V_{PQ} = (P\vert Q)$, then $M$ | $N_{\mathrm{aux}}^3$ | `ferric-integrals/src/threeindex.rs::coulomb_metric_2c`, `ferric-mp2/src/rimp2.rs::metric_inverse_sqrt` |
| 2 | $(P\vert ia) = \sum_{\mu\nu} C_{\mu i}\,(P\vert \mu\nu)\,C_{\nu a}$ | $N_{\mathrm{aux}} N^2 n_o + N_{\mathrm{aux}} N n_o n_v$ | `ferric-mp2/src/rimp2.rs::stream_dressed_mo_band_budgeted` |
| 3 | $B^P_{ia} = \sum_Q M_{PQ}\,(Q\vert ia)$ | $N_{\mathrm{aux}}^2 n_o n_v$ | same function, same pass |
| 4 | $(ia\vert jb) = \sum_P B^P_{ia}B^P_{jb}$ and the energy | $\tfrac12 n_o(n_o+1)\,n_v^2 N_{\mathrm{aux}}$ | `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu`, `ferric-mp2/src/rimp2.rs::pair_energy` |

The driver that calls them in order is
`ferric-mp2/src/rimp2.rs::ri_mp2_spin_components`.

**Steps 2 and 3 are one pass.** The raw AO integrals $(P|\mu\nu)$ arrive in
blocks of auxiliary functions from a
`ferric-integrals/src/three_index_source.rs::ThreeIndexSource`. For each chunk
of auxiliary rows the code does two matrix products per row, $C_o^{\mathsf T}
(P|\mu\nu)$ and then $\cdot\,C_v$, and adds the chunk's contribution to every
row of $B$ with one matrix product against the matching columns of $M$:

$$ B \mathrel{+}= M_{[:,\,\text{chunk}]}\;(\text{chunk}|ia) . $$

Only the output $B$, of $N_{\mathrm{aux}} n_o n_v$ doubles, and one chunk are
held. The undressed $(P|ia)$ tensor is never stored whole. The dissertation's
algorithm wrote $(ia|P)$ and then $B$ to disk in separate functions (its
Functions 2 and 3); here $B$ stays in memory and the raw AO tensor is the
quantity that goes to disk when it does not fit the memory budget.

**Step 4 is batched over $i$.** For each occupied $i$, one matrix product
forms every integral that $i$ needs:

$$ G^{(i)}_{a,\,jb} = \sum_P B^P_{ia} B^P_{jb}, \qquad j \ge i . $$

In the code `b_i` is the $n_v$ columns of `b_ov` belonging to $i$, `b_tail`
is every column from $i$ onward, and `g_i = b_i.t().dot(&b_tail)`. Batching
over the occupied index makes each product $n_v \times N_{\mathrm{aux}}$ times
$N_{\mathrm{aux}} \times (n_o - i)n_v$, which is the shape the dissertation
chose for the same reason: "the number of virtual orbitals, $V$, is
significantly larger than that of the occupied orbitals, $O$" (chapter 4,
section 4.2). The occupied index is also the parallel index. Each $i$ is one
task; BLAS runs single-threaded inside it.

**Only pairs $j \ge i$ are visited.** Off-diagonal pairs are weighted by 2.

> **Verified equation 10.6** · unique-pair sum
>
> $$ \sum_{ij} e_{ij} = \sum_{i} e_{ii} + 2\sum_{i<j} e_{ij}, \qquad e_{ij} = \sum_{ab}\frac{(ia|jb)\,[\,2(ia|jb) - (ib|ja)\,]}{\Delta_{ij}^{ab}}, $$
>
> and the same holds for the opposite-spin and same-spin parts separately,
> because each is invariant under the joint swap $(i,a) \leftrightarrow (j,b)$.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 1 (unique-pair checks) |
> | Negative control | off-diagonal weight 1 must differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` (on branch `docs/derivations-backfill`; no runner on this commit) |
> | Code | `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu` |
> | Code test | `spin_components_symmetry_matches_naive_double_loop` in `crates/ferric-mp2/src/rimp2.rs`: the production loop against a full $(i,j)$ double loop on H2O/cc-pVDZ, relative $10^{-12}$ |
>
> <!-- verified: id=mp2.cs.unique-pairs; script=scripts/mp2_variant_identities.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::spin_components_from_b_ov_kappa_cpu; rust_test=ferric-mp2/src/rimp2.rs::spin_components_symmetry_matches_naive_double_loop -->

**The pair kernel.** `ferric-mp2/src/rimp2.rs::pair_energy` is equation (10.2)
for one pair, read from the block $G^{(i)}$:

<!-- excerpt: ferric-mp2/src/rimp2.rs::pair_energy -->
```rust
let g_ab = g_i[(a, jcol + b)]; // (ia|jb)
let g_ba = g_i[(b, jcol + a)]; // (ib|ja)
let denom = e_ij - eps[nocc_total + a] - eps[nocc_total + b];
e_os_ij += g_ab * g_ab / denom;
e_ss_ij += g_ab * (g_ab - g_ba) / denom;
```

| Code | Equation |
|---|---|
| `g_ab` | $(ia\vert jb)$ |
| `g_ba` | $(ib\vert ja)$, the same block with $a$ and $b$ exchanged |
| `denom` | $-\Delta_{ij}^{ab}$ |
| `e_os_ij` | the $ij$ term of $E_{\mathrm{OS}}$ |
| `e_ss_ij` | the $ij$ term of $E_{\mathrm{SS}}$ |
| `jcol` | $(j - i)\,n_v$, the column offset of $j$ inside the tail block |

## 10.1.4 Reproducibility

Two choices make the energy independent of the thread count.

- Each $i$ task returns its partial sums, and the partials are added in
  order of $i$ on one thread. A parallel reduction would combine them in a
  tree whose shape depends on the number of workers.
- The auxiliary chunk width of the transform is a function of the memory
  budget and the problem size only
  (`ferric-mp2/src/rimp2.rs::mo_stream_chunk_for`). The dressing step
  accumulates into $B$ chunk by chunk, so the width sets the order of a
  floating-point sum. At an ample budget the width is the constant
  `MO_STREAM_CHUNK`. Two runs at different `[memory] budget_gb` can differ in
  the last digits; two runs at the same budget are bit-identical.

## 10.1.5 Memory

| Quantity | Size in doubles | Lifetime |
|---|---|---|
| Raw AO tensor $(P\vert \mu\nu)$ | $N_{\mathrm{aux}} N^2$ in core, or one block when spilled | until the transform finishes; then dropped |
| $B_{ov}$ | $N_{\mathrm{aux}} n_o n_v$ | whole energy evaluation |
| $G^{(i)}$ | at most $n_o n_v^2$ per worker thread | one $i$ task |
| $M$, $V$ | $N_{\mathrm{aux}}^2$ each | whole energy evaluation |

The budget is resolved once by `ferric-mp2/src/rimp2.rs::eri3_budget_bytes`.
`ferric-mp2/src/rimp2.rs::check_mo_side_alloc` refuses a run whose $B_{ov}$
cannot fit, before the AO tensor and $B_{ov}$ are allocated. The four-index integrals are never
stored.

## 10.1.6 Options that change the equation

| Option | Effect | Where |
|---|---|---|
| `frozen_core` | removes the lowest occupied orbitals from $i, j$. The active count comes from `ferric-mp2/src/rimp2.rs::active_occ`, which returns an error if nothing is left to correlate. | this section |
| `kappa` | multiplies each term by $(1 - e^{-\kappa_{\mathrm{reg}}\Delta_{ij}^{ab}})^2$ | section 10.3 |
| operator $\hat w$ | erfc or terfc in both the metric and the three-index integrals | section 10.4 |
| `metric_op` | a metric operator different from $\hat w$; a measurement option | section 6.3 |
| `eri3_screen_thresh` | drops distant three-index shell triples; an approximation, off by default | section 5.2 |

## 10.1.7 Numerical anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| H2O/cc-pVDZ, cc-pVDZ-RI, correlation energy $-0.2040334729$ Ha | PySCF (`testdata/reference/h2o_cc-pvdz_rimp2.json`) | $10^{-6}$ Ha asserted | `test_rimp2_h2o_ccpvdz` in `crates/ferric-mp2/src/rimp2.rs` |
| Every bundled RI and JK auxiliary set, all-electron correlation energy | PySCF fed the same sets from the Basis Set Exchange | worst case $2.4 \times 10^{-10}$ Ha (a PP-RIFIT set on Pd; the others $\le 5 \times 10^{-11}$ Ha), bar $10^{-8}$ Ha, as stated on the validation page | `crates/ferric-mp2/tests/validation_basis_coverage.rs` |
| Einsum implementation against the scalar loop | same integrals | $10^{-9}$ Ha asserted | `ri_mp2_einsum_matches_scalar` in `crates/ferric-mp2/src/rimp2.rs` |

<!-- figures: site/src/reference/validation.md "Bundled basis sets"; testdata/reference/h2o_cc-pvdz_rimp2.json -->

The second row is the method's registered energy anchor
(`crates/ferric-core/src/certificate/anchors.rs`, method `"RI-MP2"`). A run's
certificate carries that figure as its measured error.

## 10.1.8 Running it

```toml
[molecule]
xyz = "testdata/molecules/water.xyz"

[basis]
name = "cc-pvdz"

[method]
kind = "rimp2"

[mp2]
auxbasis = "cc-pvdz-ri"
```

```
cargo run --release -- examples/water-rimp2.toml
```

The auxiliary basis should be the `-RI` set matched to the orbital basis.
`frozen_core` defaults to 0 in `[mp2]`; `"auto"` selects the standard small
core.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "Conventions", "1. Density fitting", "2. Closed-shell RI-MP2" | equations 10.2 to 10.4, function names |
| Dissertation, chapter 4, sections 4.1 and 4.2 (Eqs. 4.3 to 4.5, Table 4.1, Functions 1 to 4) | the four-step algorithm, the choice of batch index, the scaling statement |
| `scripts/mp2_variant_identities.py`, claim 1 | verified equations 10.2 and 10.6 |
| `crates/ferric-mp2/src/rimp2.rs` | the pair kernel, the reproducibility and memory rules |
| `site/src/reference/validation.md` | the anchor figure |

Changes from the sources: the denominator is written as the positive
$\Delta_{ij}^{ab}$ with an explicit minus sign; the exchange partner is
written $(ib|ja)$; the dressing matrix is $M$.
