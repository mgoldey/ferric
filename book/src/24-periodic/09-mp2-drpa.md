# Periodic MP2 and dRPA {#mp2-drpa}

<!-- status: ready; registry: pbc-mp2, pbc-drpa -->

Periodic MP2 and direct RPA reuse the molecular kernels of chapters @sec:mp2, @sec:rimp2-algorithms
and @sec:rpa. At the Gamma point the orbitals are real, and the only new
ingredients are the integrals and the orbital energies. On a mesh the
integrals carry crystal momentum: an excitation $i \to a$ moves momentum
$k_a - k_i$, and only pairs of excitations with opposite momenta couple. Two questions
carry over from the SCF: does the dropped $\mathbf G = 0$ term need a
correction in correlation, and what does `exxdiv` do to the denominators?

Section @sec:mp2-drpa-carries-over-scf answers both; @sec:mp2-drpa-k-point-energies derives the $k$-point energies and their
supercell identity, @sec:mp2-drpa-complex-response-through-real the real form of the complex response and the
link between dRPA and MP2, and @sec:mp2-drpa-large-cell-limit the large-cell limit. The algorithm,
options, anchors, limits and sources follow.

| Symbol | Meaning | Code name |
|---|---|---|
| $B^P_{ia}$ | fitted integrals in the occupied-virtual block | `b_ov_from_ao_b` |
| $\Pi(q, iu)$ | density response at momentum $q$ | |
| $V^{k_ik_jk_a}_{iajb}$ | $(ik_i\,ak_a|jk_j\,bk_b)$, $k_b = k_i + k_j - k_a$ | |
| $\Delta_{ia} = \varepsilon_a - \varepsilon_i$ | excitation energy | |
| $u_0$ | quadrature scale, 0.5 | |

| Entry point | Location |
|---|---|
| Gamma MP2 | `ferric-pbc/src/mp2.rs::gamma_mp2` |
| Gamma dRPA | `ferric-pbc/src/drpa.rs::gamma_drpa` |
| $k$-point MP2, dRPA | `ferric-pbc/src/kcorr.rs::{kpoint_mp2,kpoint_drpa}` |
| CLI | `[cell]` with `method.kind = "rimp2"` or `"pdep-rpa"` |
| Python | `run_mp2_gamma`, `run_drpa_gamma`, `run_mp2_kpts`, `run_drpa_kpts` |

## What carries over from the SCF {#mp2-drpa-carries-over-scf}

**The energies at Gamma.** The orbitals are real, and the molecular
closed-shell formulas apply unchanged (`mp2.rs`, `drpa.rs`):

$$
E_\text{MP2} = \sum_{ijab}\frac{(ia|jb)\big[2(ia|jb) - (ib|ja)\big]}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b},\qquad
E_c^\text{dRPA} = \frac{1}{2\pi}\int_0^\infty\Big[\ln\det\big(1 + \Pi(iu)\big) - \operatorname{tr}\Pi(iu)\Big]du,
$$

$$
\Pi_{PQ}(iu) = 4\sum_{ia}B^P_{ia}\frac{\Delta_{ia}}{u^2 + \Delta_{ia}^2}B^Q_{ia} .
$$

The integrals $(ia|jb)$ are in the SCF's $\mathbf G = 0$-dropped kernel.

**No charge, no divergence.** The SCF needed the Madelung term because the
exchange pair density $\phi_i\phi_i$ carries one electron's charge, and its
$\mathbf G = 0$ term diverges (section @sec:exchange-divergence). An occupied-virtual pair
density carries none. Its $\mathbf G = 0$ component is
$\int\phi_i^*\phi_a = C_i^{\mathsf H}SC_a$, and the orbitals are
$S$-orthonormal, so it is zero at every $k$. There is no divergent term to
correct, and the SCF's kernel is used as is. What the dropped point does
leave out is finite: $P_{ia}(\mathbf K)$ vanishes at $\mathbf K = 0$ and grows
linearly for small $|\mathbf K|$, so $v(\mathbf K)|P_{ia}|^2$ has a finite,
direction-dependent limit. Adding that limit is the $q = 0$ head correction,
which is not implemented (@sec:mp2-drpa-limits).

> **Verified equation {#eq:pbc-corr-ov}** · occupied-virtual pair densities are neutral
>
> For $S$-orthonormal orbitals, $C_o^{\mathsf H}S(k)C_v = 0$ at every $k$,
> so $\int\phi_i^*\phi_a = 0$ and the $\mathbf G = 0$ component of every
> occupied-virtual pair density vanishes.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `ov-neutral` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/mp2.rs::b_ov_from_ao_b` |
> | Code test | `mp2_from_rsgdf_matches_dense_aft_in_the_trivial_aux_limit` in `crates/ferric-pbc/tests/pbc_mp2.rs` |
>
> <!-- verified: id=pbc.corr.ov; script=scripts/pbc_kpoint_corr_sympy.py; claim=ov-neutral; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/mp2.rs::b_ov_from_ao_b; rust_test=ferric-pbc/tests/pbc_mp2.rs::mp2_from_rsgdf_matches_dense_aft_in_the_trivial_aux_limit; registry=pbc-mp2 -->

**The integrals.** Production runs take the SCF's own RS-GDF factor $B$ of
section @sec:rsgdf and transform it to $B^P_{ia}$ (`b_ov_from_ao_b`). It goes to
the molecular kernels: `ferric_mp2::rimp2::spin_components_from_b_ov` for
MP2, and `ferric_rpa::run_pdep_rpa_from_parts` at full rank with the log-det
summand for dRPA. $B$ is already multiplied by the inverse square root of the
metric, so no further $V^{-1/2}$ is applied.

**The denominators.** The exchange divergence reaches correlation only
through the occupied orbital energies. Box @eq:pbc-exxdiv-level-shift of section @sec:exchange-divergence showed that
`ewald` lowers every occupied level by $v_M$ and leaves the orbitals as they
are. So two conventions exist (`Mp2Denominators`): `MadelungShifted` uses the
occupied $\varepsilon$ of the `ewald` Fock matrix, `Unshifted` those of the
`none` Fock matrix. The SCF result does not record which `exxdiv` produced
it, so the caller states it, and `occupied_shift` adds the difference: $0$
when the reference already has the requested convention, $-v_M$ to go from
`none` to `MadelungShifted`, $+v_M$ to go from `ewald` to `Unshifted`. The
conversion is exact because the orbitals are the same under both. Section
@sec:mp2-drpa-large-cell-limit shows which convention approaches the molecule.

For H2/STO-3G in an 8 Bohr cubic cell, from the `589b50832` build, an `ewald`
and a `none` reference give the same correlation energy for the same requested
denominators; only `occ_shift` differs:

```python
import ferric

BOHR = 0.529177210903
mol = ferric.Molecule.from_xyz_string("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
basis = ferric.BasisSet.bundled("sto-3g")
a = 8.0 * BOHR
lat = [[a, 0, 0], [0, a, 0], [0, 0, a]]
for den in ("shifted", "unshifted"):
    for exx in ("ewald", "none"):
        r = ferric.run_mp2_gamma(mol, lat, basis, exxdiv=exx, denominators=den)
        print(f"denominators={den:9s} reference exxdiv={exx:5s}  "
              f"E_corr={r.correlation_energy:.12f}  occ_shift={r.occ_shift:+.10f}  v_M={r.madelung:.10f}")
```

```text
denominators=shifted   reference exxdiv=ewald  E_corr=-0.011726901617  occ_shift=+0.0000000000  v_M=0.3546621849
denominators=shifted   reference exxdiv=none   E_corr=-0.011726901617  occ_shift=-0.3546621849  v_M=0.3546621849
denominators=unshifted reference exxdiv=ewald  E_corr=-0.016161580433  occ_shift=+0.3546621849  v_M=0.3546621849
denominators=unshifted reference exxdiv=none   E_corr=-0.016161580433  occ_shift=+0.0000000000  v_M=0.3546621849
```

## The k-point energies {#mp2-drpa-k-point-energies}

**Momentum conservation.** With per-cell normalized $C(k)$ (`kcorr.rs`), the
pair density $\phi^*_{ik_i}\phi_{ak_a}$ lives on $\mathbf K \in \mathbf G + q$,
$q = k_a - k_i$ (section @sec:kpoint-scf), and $\phi^*_{jk_j}\phi_{bk_b}$ carries
$k_b - k_j$. The Coulomb integral of the two is invariant under a common
lattice translation, so it vanishes unless the momenta add to zero modulo
the reciprocal lattice: $k_a - k_i + k_b - k_j \equiv 0$, which is
$k_b = k_i + k_j - k_a$. The second leg, written as $P^{vo}_{bj}$, then
carries $k_j - k_b = q$ as well. Then

$$
V^{k_ik_jk_a}_{iajb} = (ik_i\,ak_a|jk_j\,bk_b) = \frac{1}{\Omega}\sum_{\mathbf K\in\mathbf G+q}v(\mathbf K)\,
 P^{ov}_{ia}(\mathbf K)\,P^{vo}_{bj}(\mathbf K)^* ,
$$

and both legs lie in the class $q$ and share its auxiliary set. There are
$N_k^3$ independent momentum triples, not $N_k^4$.

**MP2.** Summing over the triples with a $1/N_k^3$ average:

$$
E_\text{MP2}/\text{cell} = \frac{1}{N_k^3}\sum_{k_ik_jk_a}\sum_{ijab}
 \frac{V_{iajb}^*\big[2V_{iajb} - V_{ibja}\big]}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b},
$$

where $V_{ibja}$ is the element of $V^{k_ik_jk_b}$. The second leg $B^{vo}$ is
transformed from $B(k_b, k_j)$ itself. It is not obtained by conjugating the
$-q$ class, which would require the orbitals at $-k$ to be the conjugates of
those at $k$, a time-reversal gauge that a caller-built SCF result need not
have. The cost is a second occupied-virtual transform per class, which
doubles that step; the contraction over the triples is unchanged.

**dRPA.** Each class $q$ has its own response, with $1/N_k$ averaging the
excitations that carry momentum $q$:

$$
\begin{aligned}
\Pi(q, iu) &= \frac{4}{N_k}\sum_{k'}\sum_{ia}B_{ia}(k' - q, k')\,B_{ia}(k' - q, k')^{\mathsf H}
 \frac{\Delta}{u^2 + \Delta^2},\qquad \Delta = \varepsilon_a(k') - \varepsilon_i(k' - q), \\
E_c^\text{dRPA}/\text{cell} &= \frac{1}{N_k}\sum_q\frac{1}{2\pi}\int_0^\infty
 \Big[\ln\det\big(1 + \Pi(q, iu)\big) - \operatorname{tr}\Pi(q, iu)\Big]du .
\end{aligned}
$$

**Check: the supercell.** With these normalizations, a Gamma-centred mesh
gives, term by term, the Gamma MP2 and dRPA of the $\operatorname{diag}(N)$
supercell per cell, the same identity as for the SCF (section @sec:kpoint-scf). The
code test compares a $1\times1\times3$ mesh with the tripled cell to
$10^{-11}$ Ha.

> **Verified equation {#eq:pbc-corr-kmp2}** · k-point MP2 is the supercell MP2
>
> With $k_b = k_i + k_j - k_a$, the kernel at $\mathbf G + q$,
> $q = k_a - k_i$, and the factor $1/N_k^3$, the $k$-point MP2 energy per
> cell equals the closed-shell MP2 of the $\operatorname{diag}(N)$
> supercell per cell.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `kmp2-supercell` |
> | Negative control | the runner's mutation: $k_b = k_i - k_j + k_a$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kcorr.rs::kpoint_mp2` |
> | Code test | `h2_1x1x3_mesh_is_the_gamma_supercell` in `crates/ferric-pbc/tests/pbc_kcorr.rs` |
>
> <!-- verified: id=pbc.corr.kmp2; script=scripts/pbc_kpoint_corr_sympy.py; claim=kmp2-supercell; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kcorr.rs::kpoint_mp2; rust_test=ferric-pbc/tests/pbc_kcorr.rs::h2_1x1x3_mesh_is_the_gamma_supercell; registry=pbc-mp2 -->

> **Verified equation {#eq:pbc-corr-kdrpa}** · k-point dRPA is the supercell dRPA
>
> $\Pi(q, iu)$ with the factor $4/N_k$ and
> $E_c = N_k^{-1}\sum_q(2\pi)^{-1}\int[\ln\det(1 + \Pi) - \operatorname{tr}\Pi]\,du$
> equals the supercell's plasmon-formula dRPA per cell.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `kdrpa-supercell` |
> | Negative control | the runner's mutation: $\Pi$ without the $1/N_k$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kcorr.rs::kpoint_drpa` |
> | Code test | `h2_1x1x3_mesh_is_the_gamma_supercell` in `crates/ferric-pbc/tests/pbc_kcorr.rs` |
>
> <!-- verified: id=pbc.corr.kdrpa; script=scripts/pbc_kpoint_corr_sympy.py; claim=kdrpa-supercell; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kcorr.rs::kpoint_drpa; rust_test=ferric-pbc/tests/pbc_kcorr.rs::h2_1x1x3_mesh_is_the_gamma_supercell; registry=pbc-drpa -->

## A complex response through a real pipeline {#mp2-drpa-complex-response-through-real}

**Realification.** $\Pi(q)$ is complex Hermitian, and the Gamma dRPA
pipeline is real. Write $B_q = X + iY$ and $F$ for the diagonal of
excitation factors. Then
$\Pi = BFB^{\mathsf H} = (XFX^{\mathsf T} + YFY^{\mathsf T}) + i(YFX^{\mathsf T} - XFY^{\mathsf T})$.
Build the real matrix with every excitation duplicated,

$$ B_R = \begin{pmatrix}X & -Y\\ Y & X\end{pmatrix},\qquad
B_R\begin{pmatrix}F & 0\\ 0 & F\end{pmatrix}B_R^{\mathsf T}
 = \begin{pmatrix}\operatorname{Re}\Pi & -\operatorname{Im}\Pi\\ \operatorname{Im}\Pi & \operatorname{Re}\Pi\end{pmatrix} . $$

That block matrix is the standard real form of a Hermitian matrix: it has
the eigenvalues of $\Pi$, each twice. The log-det and the trace both double,
so $E_q = \tfrac12E[\Pi_R]$, and the complex class goes through the unchanged
real Gamma pipeline.

> **Verified equation {#eq:pbc-corr-realify}** · realification
>
> For $B_q = X + iY$, the real
> $B_R = \begin{pmatrix}X & -Y\\ Y & X\end{pmatrix}$ with every excitation
> energy duplicated gives a real response whose spectrum is that of
> $\Pi(q)$ with each eigenvalue twice, so $E_q = \tfrac12E[\Pi_R]$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `realify` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kcorr.rs::drpa_q_realified` |
> | Code test | `h2_1x1x3_mesh_is_the_gamma_supercell` in `crates/ferric-pbc/tests/pbc_kcorr.rs` |
>
> <!-- verified: id=pbc.corr.realify; script=scripts/pbc_kpoint_corr_sympy.py; claim=realify; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kcorr.rs::drpa_q_realified; rust_test=ferric-pbc/tests/pbc_kcorr.rs::h2_1x1x3_mesh_is_the_gamma_supercell; registry=pbc-drpa -->

The construction is this part of `ferric-pbc/src/kcorr.rs::drpa_q_realified`:

<!-- excerpt: ferric-pbc/src/kcorr.rs::drpa_q_realified -->
```rust
    let mut br = Array2::<f64>::zeros((2 * nr, 2 * nc));
    let mut ed = vec![0.0; 2 * nc];
    for (j, &c) in order.iter().enumerate() {
        // Each energy appears twice (Re and Im halves); interleave them so
        // the doubled list stays ascending too.
        ed[2 * j] = e[c];
        ed[2 * j + 1] = e[c];
        for p in 0..nr {
            let z = bq[(p, c)];
            br[(p, 2 * j)] = z.re;
            br[(p, 2 * j + 1)] = -z.im;
            br[(nr + p, 2 * j)] = z.im;
            br[(nr + p, 2 * j + 1)] = z.re;
        }
    }
    let res = drpa_from_b_ov(br, None, &[0.0], &ed, quad_points, Some(budget))?;
    if !res.eigensolver_converged {
        return Err(FerricError::General(
            "kpoint_drpa: static dielectric eigensolve reported non-convergence".into(),
        ));
    }
    Ok(0.5 * res.e_rpa)
```

| Code | Equation |
|---|---|
| `bq` | $B_q$, rows auxiliary, columns excitations |
| `br` | $B_R$, columns interleaved: $(X; Y)$ then $(-Y; X)$ per excitation |
| `ed` | the duplicated excitation energies |
| `0.5 * res.e_rpa` | $E_q = \tfrac12E[\Pi_R]$ |

The columns are interleaved rather than stacked as two blocks; that is a
column permutation, under which $B_RFB_R^{\mathsf T}$ is unchanged. The
columns are also sorted by excitation energy before this loop: the energy
does not depend on the order, but the RPA library's guard reads an
out-of-order list as a non-canonical basis.

**The second-order term is direct MP2.** Expand
$\ln\det(1 + \Pi) - \operatorname{tr}\Pi = -\tfrac12\operatorname{tr}\Pi^2 + O(\Pi^3)$.
With $(ia|jb) = \sum_PB^P_{ia}B^P_{jb}$,

$$ \operatorname{tr}\Pi^2 = 16\sum_{ia,jb}(ia|jb)^2\frac{\Delta_{ia}\Delta_{jb}}{(u^2 + \Delta_{ia}^2)(u^2 + \Delta_{jb}^2)} . $$

The frequency integral is
$\int_0^\infty e_1e_2/((u^2 + e_1^2)(u^2 + e_2^2))\,du = \pi/(2(e_1 + e_2))$.
Multiply by $-\tfrac12\cdot\tfrac{1}{2\pi}$:

$$ E^{(2)} = -2\sum_{ia,jb}\frac{(ia|jb)^2}{\Delta_{ia} + \Delta_{jb}}
 = \sum_{ijab}\frac{2(ia|jb)^2}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}, $$

the direct (Coulomb) part of MP2, without the exchange term $-(ib|ja)$. The
code test computes it with the Gamma MP2 code and compares.

> **Verified equation {#eq:pbc-corr-second}** · the second-order dRPA term is direct MP2
>
> $\int_0^\infty e_1e_2/((u^2 + e_1^2)(u^2 + e_2^2))\,du = \pi/(2(e_1 + e_2))$,
> so the $O(\Pi^2)$ term of the dRPA energy is
> $\sum 2|V|^2/(\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b)$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `drpa-second-order` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/drpa.rs::gamma_drpa` |
> | Code test | `drpa_second_order_term_is_direct_mp2_from_gamma_mp2` in `crates/ferric-pbc/tests/pbc_drpa.rs` |
>
> <!-- verified: id=pbc.corr.second; script=scripts/pbc_kpoint_corr_sympy.py; claim=drpa-second-order; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/drpa.rs::gamma_drpa; rust_test=ferric-pbc/tests/pbc_drpa.rs::drpa_second_order_term_is_direct_mp2_from_gamma_mp2; registry=pbc-drpa -->

**The frequency grid.** The Gamma dRPA integral uses Gauss–Legendre points
$x \in (-1, 1)$ mapped by $u = u_0(1 + x)/(1 - x)$, $u_0 = 0.5$, which sends
$x \to 1$ to $u \to \infty$. The default is 40 points
(`DEFAULT_GAMMA_DRPA_QUAD_POINTS`), not the 20 of the molecular dRPA
(@sec:quadratures): with $u_0 = 0.5$ the 20-point grid misses H$_2$O/6-31G
by $2.3\times10^{-6}$ Ha, because the core excitations ($\Delta_{ia} \approx 21$ Ha)
sit far above $u_0$; 40 points give $7.4\times10^{-10}$ Ha
(`ferric-pbc/src/drpa.rs` module documentation).

## The large-cell limit {#mp2-drpa-large-cell-limit}

As the cell grows around one molecule, the periodic MP2 energy should
approach the molecular one. Which denominators make that happen? The test
`box_limit_shifted_is_a3_with_the_predicted_coefficient_unshifted_is_1_over_a`
in `crates/ferric-pbc/tests/pbc_mp2.rs` measures it for H2/STO-3G in cubic
boxes, against ferric's own canonical molecular MP2, with the dense AFT
integrals on the periodic side so that the difference is finite-size error
only:

| box edge $a$ (Bohr) | $E^\text{shifted} - E^\text{mol}$ (Ha) | $E^\text{unshifted} - E^\text{mol}$ (Ha) |
|---|---|---|
| 16 | $+1.646\times10^{-4}$ | $-1.980\times10^{-3}$ |
| 20 | $+8.419\times10^{-5}$ | $-1.589\times10^{-3}$ |
| 24 | $+4.868\times10^{-5}$ | $-1.321\times10^{-3}$ |

The Madelung-shifted difference falls as $a^{-3}$: the test requires a local
exponent between 2.8 and 3.3, and the fitted $a^{-3}$ coefficient agrees
within 1% with a prediction built from molecular quantities only. With one
occupied and one virtual level the molecular energy is
$E = (ia|ia)^2/D$, $D = 2(\varepsilon_i - \varepsilon_a)$. In the box,
$(ia|ia)$ gains the image dipole term $-\kappa|\mathbf d_{ia}|^2/a^3$,
$\kappa = 4\pi/3$, with $\mathbf d_{ia}$ the transition dipole, and the
shifted denominator moves by $-\kappa(2\sigma^2 + 2|\mathbf d_{ia}|^2)/a^3$,
with $\sigma^2$ the occupied orbital's second moment about its centre. To
first order in both,

$$ E^\text{shifted} - E^\text{mol} \approx \frac{c_3}{a^3}, \qquad c_3 = -\kappa\Big[\frac{2(ia|ia)|\mathbf d_{ia}|^2}{D} - \frac{(ia|ia)^2(2\sigma^2 + 2|\mathbf d_{ia}|^2)}{D^2}\Big] = 0.671\ \text{Ha·Bohr}^3 \label{eq:pbc-corr-box-c3} $$

(0.67109405 in the test, which evaluates it from ferric's molecular SCF). The
unshifted difference falls as $a^{-1}$. That is the size of $v_M$: the
`none` occupied levels sit $v_M$ above the `ewald` ones (box @eq:pbc-exxdiv-level-shift), so each
denominator $\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b$
is less negative by $2v_M$, and the correlation energy is too negative by an
amount that falls as $v_M$ does (the test requires a local exponent between
0.6 and 1.4). The
Madelung-shifted denominators are the ones that approach the molecule at
the rate of the integrals.

The $1/a$ part can be seen without a molecular reference. From the
`589b50832` build, for H2/STO-3G at 0.74 Å on the $z$ axis (not the test's
geometry, so the energies are not the table's), the unshifted minus shifted
correlation energy times $a$ falls from 0.0342 to 0.0328 Ha·Bohr between 16
and 24 Bohr, a 4% drift on top of the $1/a$ law:

```python
import ferric

BOHR = 0.529177210903
mol = ferric.Molecule.from_xyz_string("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
basis = ferric.BasisSet.bundled("sto-3g")
# The book's table subtracts ferric's canonical molecular MP2 (tests/pbc_mp2.rs);
# the difference unshifted - shifted printed here is the v_M-driven 1/a part.
for a_bohr in (16.0, 20.0, 24.0):
    a = a_bohr * BOHR
    lat = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    s = ferric.run_mp2_gamma(mol, lat, basis, exxdiv="ewald", denominators="shifted").correlation_energy
    u = ferric.run_mp2_gamma(mol, lat, basis, exxdiv="ewald", denominators="unshifted").correlation_energy
    print(f"a={a_bohr:5.1f}  E_shifted={s:.10f}  E_unshifted={u:.10f}  unshifted-shifted={u - s:+.6e}")
```

```text
a= 16.0  E_shifted=-0.0129738471  E_unshifted=-0.0151125955  unshifted-shifted=-2.138748e-03
a= 20.0  E_shifted=-0.0130540831  E_unshifted=-0.0147231652  unshifted-shifted=-1.669082e-03
a= 24.0  E_shifted=-0.0130895087  E_unshifted=-0.0144559536  unshifted-shifted=-1.366445e-03
```

## The algorithm {#mp2-drpa-algorithm}

| Step | Gamma | $k$-points |
|---|---|---|
| 1 | the SCF's RS-GDF $B$, transformed to $B^P_{ia}$ | the pair tensors of each class $q$ |
| 2 | occupied energies shifted by `occupied_shift` | the same |
| 3 | MP2: `ferric_mp2::rimp2::spin_components_from_b_ov` | MP2: loop over $(k_i, k_j, k_a)$, $k_b$ by momentum conservation |
| 4 | dRPA: `ferric_rpa::run_pdep_rpa_from_parts`, full rank | dRPA: per $q$, realified, through the Gamma pipeline |

The denominator conversion is one match:

<!-- excerpt: ferric-pbc/src/mp2.rs::occupied_shift -->
```rust
    match (reference_exxdiv, denominators) {
        (ExxDiv::Ewald, Mp2Denominators::MadelungShifted) => 0.0,
        (ExxDiv::None, Mp2Denominators::MadelungShifted) => -madelung,
        (ExxDiv::None, Mp2Denominators::Unshifted) => 0.0,
        (ExxDiv::Ewald, Mp2Denominators::Unshifted) => madelung,
    }
```

| Symbol | Code |
|---|---|
| the SCF's `exxdiv` | `reference_exxdiv` |
| the requested convention | `denominators` |
| $v_M$ | `madelung` |

## Options {#mp2-drpa-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cell] denominators`, `denominators=` | `"shifted"` (`MadelungShifted`), `"unshifted"` | none (required) | the occupied orbital energies in the denominators |
| `frozen_core` | integer | 0 | core orbitals excluded from correlation |
| `[cell] quad_points` | positive integer | 40 | dRPA Gauss-Legendre points |
| `[cell] drpa_energy` | `"quadrature"`, `"plasmon"`, `"second-order"` | `"quadrature"` | the dRPA energy route, `kmesh` only: the frequency integral, the plasmon formula of @sec:drpa-three-ways-plasmon-formula, or the second-order term alone |

## Numerical anchors {#mp2-drpa-numerical-anchors}

| Quantity | Bar | Test |
|---|---|---|
| $k$-point MP2 and plasmon dRPA against the explicit supercell | $10^{-11}$ Ha | `crates/ferric-pbc/tests/pbc_kcorr.rs` `h2_1x1x3_mesh_is_the_gamma_supercell` |
| momentum, conjugation, Madelung, occupied-$k$ and normalization mutants | each detected | `supercell_anchor_catches_momentum_conj_madelung_occupied_k_and_normalisation_mutants` |
| Gamma MP2 against PySCF | | `crates/ferric-pbc/tests/pbc_mp2.rs` `h2_sto3g_a4_matches_pinned_pyscf_rmp2` |
| $k$-point MP2 against PySCF `KMP2`, both `exxdiv` | | `kmp2_h2_1x1x2_matches_pinned_pyscf_kmp2_both_exxdiv` |
| Gamma dRPA from RS-GDF with a complete auxiliary basis (frequency quadrature, 80 points) against dense AFT with the plasmon formula, shifted and unshifted denominators, same and own SCF orbitals; the default 40-point grid; $\tilde\epsilon(0)$ eigenvalues $\ge 1$; dRPA differs from direct MP2 by more than $10^{-4}$ | $10^{-11}$ Ha; $10^{-10}$ Ha (40 points) | `crates/ferric-pbc/tests/pbc_drpa.rs` `drpa_from_rsgdf_matches_dense_plasmon_in_the_trivial_aux_limit` |
| large-cell limit: shifted $\propto a^{-3}$ with the predicted coefficient, unshifted $\propto a^{-1}$ | coefficient within 1% | `box_limit_shifted_is_a3_with_the_predicted_coefficient_unshifted_is_1_over_a` |

## Limits {#mp2-drpa-limits}

- Closed shell only on these routes. Open-shell UMP2 and URPA at Gamma are
  library-only.
- Every $k$ must hold the same number of occupied levels; the global aufbau
  of section @sec:kpoint-scf can violate this, and the run is then refused.
- The $q = 0$ head correction is not implemented, so the energy converges
  with the mesh at the rate of the uncorrected sum.

## Sources {#mp2-drpa-sources}

| Source | What it gives |
|---|---|
| `scripts/pbc_kpoint_corr_sympy.py` | the proof |
| PySCF `pyscf.pbc.mp.kmp2` | the $k$-point MP2 conventions |
