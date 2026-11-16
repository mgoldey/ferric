# 27.3 Polarizabilities, PDEP modes and $C_6$

<!-- section: 27.3; status: ready; registry: rpa-polarizability-c6, pdep-rpa;
     theory: crates/ferric-rpa/theory.md#"Dispersion: Casimir-Polder C6" -->

The response arrays are the part of the bundle a cheaper model cannot
reproduce from the density alone: the molecular polarizability, its split
into atoms plus a charge-transfer remainder, the same split along the
imaginary frequency axis, and the Casimir–Polder $C_6$ coefficients built on
it. This section gives each definition as the code computes it and marks
which identities a proof script checks. Section 21.1 derives the dielectric
machinery; here the interest is what the arrays mean.

| Index | Range | Code name |
|---|---|---|
| $i$ / $a$ | occupied / virtual spatial orbitals of the closed-shell reference | `nocc`, `nvir`; frozen core is 0 on this path |
| $P$ | auxiliary functions | `naux` |
| $A, B$ | atoms | `natoms` |
| $c, d, j$ | Cartesian directions | `0..3` |
| $u_k$, $w_k$ | imaginary-frequency nodes and weights | `c6_freqs`, `c6_weights` |
| $\xi$ | PDEP modes | columns of `pdep_eigenvectors` |

| Entry point | Location |
|---|---|
| Molecular static α | `ferric-rpa/src/properties.rs::pdep_polarizability_static` |
| Molecular α(iu) | `ferric-rpa/src/properties.rs::molecular_dynamic_polarizability` |
| Per-atom α, Becke | `ferric-rpa/src/properties/per_atom.rs::{pdep_polarizability_becke,pdep_polarizability_becke_dynamic}` |
| Per-atom α, Hirshfeld | `ferric-rpa/src/properties/per_atom.rs::{pdep_polarizability_hirshfeld,pdep_polarizability_hirshfeld_dynamic}` |
| CT remainder | `ferric-rpa/src/properties/per_atom.rs::{charge_transfer_remainder,charge_transfer_remainder_dynamic}` |
| $C_6$ | `ferric-rpa/src/dispersion.rs::{casimir_polder_c6,ts_dynamic_polarizability,pdep_dynamic_polarizability}` |

## 27.3.1 The molecular polarizability

Let $\mu^j_{ia} = \langle\phi_i\lvert r_j\rvert\phi_a\rangle$, let
$\Delta_{ia} = \varepsilon_a - \varepsilon_i > 0$, and let $\tilde B_{P,ia}$
be the density-fitted three-index factor of the auxiliary basis. With
$g_{ia}(u) = \Delta_{ia}/(u^2 + \Delta_{ia}^2)$, which is $1/\Delta_{ia}$ at
$u = 0$, the code evaluates

$$ \alpha_{cj}(iu) = 4\,\mu^{c\,\mathsf T}(g\circ\mu^j) - 16\,\big(\tilde B(g\circ\mu^c)\big)^{\mathsf T}\,\tilde\varepsilon(u)^{-1}\,\big(\tilde B(g\circ\mu^j)\big), \qquad \tilde\varepsilon(u) = I + 4\,\tilde B\,\mathrm{diag}(g)\,\tilde B^{\mathsf T} . \tag{27.8} $$

The first term is the uncoupled sum over orbital pairs. The second screens it
with the direct-RPA dielectric matrix in the auxiliary basis. It has the
density-fitted Coulomb kernel only, no exchange kernel, so it is not a
coupled-perturbed Hartree–Fock or Kohn–Sham polarizability. The solve uses
every auxiliary function: it is not truncated to the PDEP modes. The result
is symmetrized. Shifting the origin by $\mathbf t$ adds
$\mathbf t\langle i\vert a\rangle = 0$ to every $\mu_{ia}$, so (27.8) does
not depend on the origin.

## 27.3.2 The per-atom split and the charge-transfer remainder

The per-atom intrinsic polarizability of Krishtal, Senet and Van Alsenoy
(J. Chem. Phys. 125, 034312 (2006)) replaces the molecular dipole on one side
of (27.8) by an atom-centred, partition-weighted dipole:

$$ m^{A,d}_{ia} = \sum_{g\in A} w_g\,(\mathbf r_g-\mathbf R_A)_d\,\phi_i(\mathbf r_g)\phi_a(\mathbf r_g), \qquad \alpha^A_{dj}(iu) = 4\,m^{A,d\,\mathsf T}(g\circ\mu^j) - 16\,\big(\tilde B(g\circ m^{A,d})\big)^{\mathsf T}\tilde\varepsilon^{-1}\big(\tilde B(g\circ\mu^j)\big) . \tag{27.9} $$

For the Becke partition, the sum runs over the points of the flat 75 × 110
Becke–Lebedev grid whose home atom is $A$, with the Becke weight folded into
$w_g$. For the Hirshfeld partition, $w_g$ is the Hirshfeld weight on a cubic
lattice of 0.20 Bohr spacing and 6 Bohr margin
(`FERRIC_HIRSHFELD_SPACING`, `FERRIC_HIRSHFELD_MARGIN`). Each $\alpha^A$ is
symmetrized.

Because the partition weights sum to one at every point, the atom-centred
dipoles miss exactly the part of the response that moves charge between
atoms. The code keeps that part as a separate array:

$$ \alpha_{\mathrm{CT}} = \alpha - \sum_A \alpha^A . \tag{27.10} $$

> **Verified equation 27.9** · the per-atom kernel equals a sum over excitations
>
> The coded per-atom kernel (27.9), static or at any $u$, equals the
> independent sum over closed-shell direct-RPA (Casida) excitations
> $\sum_n f_n/(\Omega_n^2 + u^2)$ with $\Omega_n^2$ the eigenvalues of
> $\Delta^{1/2}(\Delta + 4K)\Delta^{1/2}$, and its static form equals the
> $u \to 0$ limit. With $m^A$ replaced by $\mu$ it is (27.8).
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_c6_polarizability_sympy.py`, claim 3 |
> | Negative control | the screening prefactor 16 changed to 4 turns the check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha` |
> | Rust test | `ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_per_atom_becke_h2o_aug_cc_pvdz_vs_numpy` |
>
> <!-- verified: id=ml.alpha.per-atom-kernel; script=scripts/rpa_c6_polarizability_sympy.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha; rust_test=ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_per_atom_becke_h2o_aug_cc_pvdz_vs_numpy; registry=rpa-polarizability-c6 -->

> **Verified equation 27.10** · the charge-transfer sum rule
>
> For partition weights that sum to one, so that
> $\sum_A\langle i\lvert w_A\rvert a\rangle = 0$, the remainder
> $\alpha - \sum_A\mathrm{sym}(\alpha^A)$ equals
> $\mathrm{sym}\,\langle\sum_A\mathbf R_A w_A\lvert R\rvert\mu\rangle$, a
> charge-flow term, where sym is the code's symmetrization.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_c6_polarizability_sympy.py`, claim 5 |
> | Negative control | none in the script's mutation list; the Rust test checks the array identity exactly |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties/per_atom.rs::charge_transfer_remainder` |
> | Rust test | `ferric-rpa/tests/property_bundle.rs::alpha_ct_is_the_exact_remainder` |
>
> <!-- verified: id=ml.alpha.ct-sum-rule; script=scripts/rpa_c6_polarizability_sympy.py; claim=5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties/per_atom.rs::charge_transfer_remainder; rust_test=ferric-rpa/tests/property_bundle.rs::alpha_ct_is_the_exact_remainder; registry=rpa-polarizability-c6 -->

The remainder is not small. On the validation set the isotropic
$\alpha_{\mathrm{CT}}(0)$ is 27.0% of $\alpha_{\mathrm{iso}}$ for H2O and
22.7% for N2 at aug-cc-pVDZ (`site/src/reference/validation.md`, row "Dynamic
α(iω) and Casimir–Polder C6"). A model that predicts per-atom α and sums them
will therefore miss about a quarter of the molecular polarizability unless it
also predicts $\alpha_{\mathrm{CT}}$, or predicts $\alpha$ directly. Figure
27.1(b) shows the split for water/cc-pVDZ.

**Frame behaviour.** $\alpha$ is a rank-2 tensor and origin-independent. Each
$\alpha^A$ is centred on its own nucleus, so it is translation-invariant; the
NPZ validation holds both to 3.4e-13 relative under translation. Under
rotation they are tensors up to the integration error of a grid whose
Lebedev spheres (Becke) or lattice axes (Hirshfeld) are fixed in the lab
frame. The Hirshfeld lattice is an axis-aligned box around the molecule: it
moves with the molecule under translation, but its axes, and its number of
points, change under rotation.

## 27.3.3 Dynamic polarizabilities: three sources

`c6_source` selects where $\alpha^A(iu_k)$ comes from:

| `c6_source` | $\alpha^A(iu)$ | Molecular $\alpha(iu)$ | Default partition |
|---|---|---|---|
| `ts` (default) | Tkatchenko–Scheffler single pole, shaped by the static $\alpha^A$ | sum of the atoms | `becke` |
| `pdep` | (27.9) at each node | (27.8) at each node | `hirshfeld` |
| `mbd` | the MBD screened model | the MBD model | `becke` |

The `ts` model, for atom $A$ with free-atom values $\alpha^{\mathrm{free}}$ and
$C_6^{\mathrm{free}}$ and Hirshfeld volume ratio $v = V_A/V_A^{\mathrm{free}}$,
is

$$ \bar\alpha_A = v\,\alpha^{\mathrm{free}},\quad C_{6,A} = v^2 C_6^{\mathrm{free}},\quad \omega_A = \frac{4}{3}\frac{C_{6,A}}{\bar\alpha_A^2},\qquad \alpha^A_{ij}(iu) = \frac{\bar\alpha_A}{1+(u/\omega_A)^2}\,\frac{\alpha^{A,\mathrm{static}}_{ij}}{\tfrac13\mathrm{tr}\,\alpha^{A,\mathrm{static}}} . \tag{27.11} $$

Only the shape of the static per-atom tensor enters. The volume ratios always
come from Hirshfeld volumes, whatever `c6_partition` says; the free-atom
volume comes from a live free-atom SCF, and if that SCF fails the C6 block is
skipped with a warning. Elements beyond Z = 54 have no TS reference and are
an error. TS has no charge transfer, so `alpha_ct_dynamic` is written only for
`pdep`.

> **Verified equation 27.11** · the single-pole frequency
>
> With $\alpha(iu) = \bar\alpha/(1+(u/\omega)^2)$ the Casimir–Polder integral
> gives the London formula $\tfrac32\bar\alpha_A\bar\alpha_B\omega_A\omega_B/(\omega_A+\omega_B)$,
> which at $A = B$ is $\tfrac34\bar\alpha^2\omega$, so
> $\omega_A = \tfrac43 C_{6,A}/\bar\alpha_A^2$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_c6_polarizability_sympy.py`, claim 1 |
> | Negative control | $\omega_A$ without the factor 4/3 turns the check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/dispersion/mbd.rs::ts_atom_params` |
>
> <!-- verified: id=ml.c6.ts-pole; script=scripts/rpa_c6_polarizability_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/dispersion/mbd.rs::ts_atom_params; registry=rpa-polarizability-c6 -->

## 27.3.4 Casimir–Polder $C_6$

The nodes $u_k$ and weights $w_k$ are the run's RPA frequency quadrature, by
default 20-point Gauss–Legendre with $u_0 = 0.5$; changing `[rpa] n_quad`
changes `c6_freqs`. The pair and molecular coefficients are

$$ C_6^{AB} = \frac{3}{\pi}\sum_k w_k\,\bar\alpha^A(iu_k)\,\bar\alpha^B(iu_k),\qquad C_{6,\mathrm{mol}} = \frac{3}{\pi}\sum_k w_k\,\bar\alpha(iu_k)^2, \tag{27.12} $$

with $\bar\alpha = \tfrac13\mathrm{tr}\,\alpha$.

> **Verified equation 27.12** · the Casimir–Polder quadrature
>
> The coded sum over the default 20-point, $u_0 = 0.5$ grid reproduces the
> London value of the single-pole integral, and converges below $10^{-10}$ at
> 200 points. The anisotropic coefficient is formed element by element and the
> isotropic one from $\tfrac13\mathrm{tr}\,\alpha$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_c6_polarizability_sympy.py`, claim 2 |
> | Negative control | the prefactor $3/\pi$ changed to $3/(2\pi)$ turns the check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/dispersion.rs::casimir_polder_c6` |
> | Rust test | `ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_molecular_h2o_aug_cc_pvdz_vs_numpy` |
>
> <!-- verified: id=ml.c6.cp-quadrature; script=scripts/rpa_c6_polarizability_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/dispersion.rs::casimir_polder_c6; rust_test=ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_molecular_h2o_aug_cc_pvdz_vs_numpy; registry=rpa-polarizability-c6 -->

Three things about these arrays matter to a consumer.

1. **`c6_iso.sum()` is not the molecular $C_6$.** The square of a sum is not
   the sum over pairs once $\alpha_{\mathrm{CT}} \ne 0$. Use
   `c6_molecular_iso`. A committed test bounds how far the pair sum strays
   from the molecular value for water
   (`ferric-rpa/tests/s9_per_atom_c6_consistency.rs::bounded_divergence_pair_sum_vs_molecular_c6_water`).
2. **`c6_aniso` is a convention, not a tensor.** The entry
   $\tfrac3\pi\sum_k w_k\alpha^A_{ij}\alpha^B_{ij}$ is an element-wise product.
   It does not transform as a rank-2 tensor under rotation, so an equivariant
   model should be fed the dynamic tensors `alpha_atomic_dynamic` instead.
3. **Pair $C_6$ depends on the partition.** Becke and Hirshfeld give
   different `c6_iso`; record `c6_partition` and `c6_source` with the data.

## 27.3.5 PDEP eigenvectors

`pdep_eigenvectors` holds $M$ columns $c_\xi$ over the auxiliary basis. Each
is the coefficient vector of a mode charge density
$\rho_\xi(\mathbf r) = \sum_P c_{P\xi}\chi_P(\mathbf r)$, whose Coulomb
potential is the PDEP eigenpotential. With $V$ the auxiliary Coulomb metric
and $\Pi$ the static response matrix in the same basis, the columns satisfy

$$ c^{\mathsf T}Vc = I, \qquad c^{\mathsf T}(V+\Pi)\,c \ \text{diagonal}. $$

The eigenvalues are on the Python result, not in the file. $M$ is set by the
`[rpa] trunc_thresh` cut on the spectrum, so it varies with the molecule and
the setting. Each column's sign is arbitrary, and so is the basis inside a
degenerate eigenspace. This is why the validation compares the projector
$cc^{\mathsf T}$ with PySCF's (2.6e-7) and the metric condition (4.3e-12),
not the columns. A model should consume the projector, or a sign- and
rotation-invariant function of the columns, never the raw columns.

## 27.3.6 How well these are checked

| Quantity | Agreement (same orbitals; own SCF) | Test |
|---|---|---|
| static α | 4.8e-14 rel.; 5.3e-10 rel. | `ferric-rpa/tests/validation_static_alpha.rs::static_alpha_h2o_aug_cc_pvdz_vs_numpy` |
| α(iu), molecular $C_6$ | 4.9e-14, 6.2e-14 rel. | `ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_molecular_h2o_aug_cc_pvdz_vs_numpy` |
| per-atom α (Becke) | 3.4e-14 rel. | `ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_per_atom_becke_h2o_aug_cc_pvdz_vs_numpy` |
| NPZ α against a numpy dRPA solve | 9.2e-11 rel. | `crates/ferric-cli/tests/validation_npz.rs` |

The references are numpy sums over Casida excitations on PySCF integrals in
the same auxiliary basis (`site/src/reference/validation.md`, "Anchors").
These check that ferric computes (27.8) and (27.9). They do not say how close
a direct-RPA polarizability is to experiment.

## Known limits (2026-11-15)

- The Hirshfeld per-atom α still integrates on the axis-aligned lattice, so
  it depends on the molecule's orientation. Hirshfeld charges moved off such a
  lattice to an atom-centred grid; the per-atom α has not.
- On an open-shell reference, `pdep_dynamic_polarizability` falls back from
  the Hirshfeld to the Becke partition while the label still says
  `hirshfeld` (section 27.1).

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-rpa/theory.md`, "Dispersion: Casimir-Polder C6" | the notation of (27.8)–(27.12) |
| `scripts/rpa_c6_polarizability_sympy.py`, claims 1, 2, 3, 5 | verified equations 27.9–27.12 |
| `crates/ferric-rpa/src/properties/per_atom.rs`, `crates/ferric-rpa/src/dispersion.rs`, `crates/ferric-rpa/src/property_bundle.rs` | grids, defaults, fallbacks, the quadrature source |
| `crates/ferric-cli/tests/validation_npz.rs` | the PDEP eigenvector conditions |
| `site/src/reference/validation.md` | the figures in 27.3.2 and 27.3.6 |

Changes from the sources: the theory text writes the imaginary frequency as
$\omega$ and the PDEP mode index as $\alpha$; this section uses $u$ and $\xi$
(notation).
