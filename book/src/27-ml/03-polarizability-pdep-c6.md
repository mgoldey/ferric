# Polarizabilities, PDEP modes and $C_6$ {#polarizability-pdep-c6}

<!-- status: ready; registry: rpa-polarizability-c6, pdep-rpa -->

A polarizability says how far the density moves when a field is switched on.
That cannot be read off the ground-state density, so a model that predicts it
needs a response calculation as its label. The bundle carries four kinds of
response array: the molecular polarizability, its split into atoms plus a
charge-transfer remainder, the same split along the imaginary frequency axis,
and the Casimir–Polder $C_6$ coefficients built on it.

Section @sec:polarizabilities-from-dielectric-modes derives the dielectric response and section @sec:atomic-partitions-partition-unity-buys-polarizability the per-atom
split; section @sec:casimir-polder-c6 derives the $C_6$ quadrature. This section restates each
result only as far as a consumer needs it: what the array is, which
convention it depends on, and which identity a proof script checks. The plan
follows the arrays: molecular α (@sec:polarizability-pdep-c6-molecular-polarizability), per-atom α and the remainder
(@sec:polarizability-pdep-c6-per-atom-split-charge), the three dynamic sources (@sec:polarizability-pdep-c6-dynamic-polarizabilities-three-sources), $C_6$ (@sec:polarizability-pdep-c6-casimir-polder), the PDEP
eigenvectors (@sec:polarizability-pdep-c6-pdep-eigenvectors) and the checks (@sec:polarizability-pdep-c6-well-these-checked).

| Index | Range | Code name |
|---|---|---|
| $i$ / $a$ | occupied / virtual spatial orbitals of the closed-shell reference | `nocc`, `nvir`; frozen core is 0 on this path |
| $P$ | auxiliary functions | `naux` |
| $A, B$ | atoms | `natoms` |
| $c, d, e$ | Cartesian directions | `0..3` |
| $p$ | grid points $\mathbf r_p$ with weights $w_p$ | grid arrays |
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

## The molecular polarizability {#polarizability-pdep-c6-molecular-polarizability}

Section @sec:polarizabilities-from-dielectric-modes-dielectric-form derives the polarizability from the dielectric matrix,
as (@eq:pol-kernel-vs-casida), and reads its uncoupled and screening terms;
here it is in the notation of this section. Let
$\mu^c_{ia} = \langle\phi_i\lvert r_c\rvert\phi_a\rangle$, let
$\Delta_{ia} = \varepsilon_a - \varepsilon_i > 0$, and let $\tilde B_{P,ia}$
be the density-fitted three-index factor of the auxiliary basis. The
frequency enters only through
$g_{ia}(u) = \Delta_{ia}/(u^2 + \Delta_{ia}^2)$:

$$ \alpha_{cd}(iu) = 4\,\mu^{c\,\mathsf T}(g\circ\mu^d) - 16\,\big(\tilde B(g\circ\mu^c)\big)^{\mathsf T}\,\tilde\varepsilon(u)^{-1}\,\big(\tilde B(g\circ\mu^d)\big), \qquad \tilde\varepsilon(u) = I + 4\,\tilde B\,\mathrm{diag}(g)\,\tilde B^{\mathsf T} . \label{eq:polarizability-pdep-c6-1} $$

For a label, three facts follow from the form of (@eq:polarizability-pdep-c6-1).

- **The kernel is Coulomb only.** It has the density-fitted Coulomb kernel
  and no exchange kernel, so it is not a coupled-perturbed Hartree–Fock or
  Kohn–Sham polarizability. Mixing it with CPHF values from another code
  mixes definitions (section @sec:training-labels-response-properties).
- **No mode truncation.** The solve uses every auxiliary function. It is not
  truncated to the PDEP modes, so `trunc_thresh` does not change α.
- **No origin.** Shifting the origin by $\mathbf t$ adds
  $\mathbf t\langle i\vert a\rangle = 0$ to every $\mu_{ia}$, so (@eq:polarizability-pdep-c6-1) does
  not depend on the origin.

The result is symmetrized.

## The per-atom split and the charge-transfer remainder {#polarizability-pdep-c6-per-atom-split-charge}

A per-atom polarizability is what a transferable model wants: one tensor per
atom that it can predict from the atom's neighbourhood. Section
@sec:atomic-partitions-partition-unity-buys-polarizability defines it, after Krishtal, Senet and Van Alsenoy
(J. Chem. Phys. 125, 034312 (2006)): the molecular dipole on one side of
(@eq:polarizability-pdep-c6-1) is replaced by an atom-centred,
partition-weighted dipole,

$$ m^{A,d}_{ia} = \sum_{p\in A} w_p\,(\mathbf r_p-\mathbf R_A)_d\,\phi_i(\mathbf r_p)\phi_a(\mathbf r_p), \qquad \alpha^A_{de}(iu) = 4\,m^{A,d\,\mathsf T}(g\circ\mu^e) - 16\,\big(\tilde B(g\circ m^{A,d})\big)^{\mathsf T}\tilde\varepsilon^{-1}\big(\tilde B(g\circ\mu^e)\big) . \label{eq:ml-alpha-per-atom-kernel} $$

For the Becke partition, $w_p$ is the Becke weight on the flat 75 × 110
grid points whose home atom is $A$; for the Hirshfeld partition, it is the
Hirshfeld weight on the lattice of section @sec:atomic-partitions-hirshfeld-l-wdin-mulliken. Each $\alpha^A$ is
symmetrized.

The atoms do not add up to $\alpha$. The weights sum to one, so
$\sum_A m^A = \mu - \sum_A \mathbf R_A w_A$, and the second part is the
dipole of charge that the field moves between cells (@eq:part-ct-sum-rule).
The code exports that part as a separate array:

$$ \alpha_{\mathrm{CT}} = \alpha - \sum_A \alpha^A . \label{eq:ml-alpha-ct-sum-rule} $$

In the second box below, $\langle a\lvert R\rvert b\rangle$ is the
bilinear form of (@eq:polarizability-pdep-c6-1) between two dipole vectors
$a$ and $b$: $4\,a^{\mathsf T}(g\circ b) - 16\,\big(\tilde B(g\circ a)\big)^{\mathsf T}\tilde\varepsilon^{-1}\big(\tilde B(g\circ b)\big)$,
so that $\alpha_{cd} = \langle\mu^c\lvert R\rvert\mu^d\rangle$.

> **Verified equation @eq:ml-alpha-per-atom-kernel** · the per-atom kernel equals a sum over excitations
>
> The coded per-atom kernel (@eq:ml-alpha-per-atom-kernel), static or at any $u$, equals the
> independent sum over closed-shell direct-RPA (Casida) excitations
> $\sum_n f_n/(\Omega_n^2 + u^2)$ with $\Omega_n^2$ the eigenvalues of
> $\Delta^{1/2}(\Delta + 4K)\Delta^{1/2}$, and its static form equals the
> $u \to 0$ limit. With $m^A$ replaced by $\mu$ it is (@eq:polarizability-pdep-c6-1).
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

> **Verified equation @eq:ml-alpha-ct-sum-rule** · the charge-transfer sum rule
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

How large is the remainder? On the validation set the isotropic
$\alpha_{\mathrm{CT}}(0)$ is 27.0% of $\alpha_{\mathrm{iso}}$ for H2O and
22.7% for N2 at aug-cc-pVDZ (`site/src/reference/validation.md`, row "Dynamic
α(iω) and Casimir–Polder C6"). A model that predicts per-atom α and sums them
therefore misses about a quarter of the molecular polarizability, unless it
also predicts $\alpha_{\mathrm{CT}}$ or predicts $\alpha$ directly. Panel
(b) of the figure in section
@sec:electrostatics-and-charges-five-charge-schemes shows the split for
water/cc-pVDZ.

**Frame behaviour.** $\alpha$ is a rank-2 tensor and origin-independent. Each
$\alpha^A$ is centred on its own nucleus, so it is translation-invariant; the
NPZ validation holds both to 3.4e-13 relative under translation. Under
rotation they are tensors up to the integration error of a grid whose
Lebedev spheres (Becke) or lattice axes (Hirshfeld) are fixed in the lab
frame. The Hirshfeld lattice is an axis-aligned box around the molecule. It
moves with the molecule under translation, but its axes, and its number of
points, change under rotation.

## Dynamic polarizabilities: three sources {#polarizability-pdep-c6-dynamic-polarizabilities-three-sources}

A $C_6$ needs the polarizability along the whole imaginary frequency axis,
from $u = 0$ outward. ferric has three ways to get $\alpha^A(iu_k)$, which
differ in cost and in what they assume, and `c6_source` selects one. Section @sec:casimir-polder-c6-three-sources
compares them as dispersion models; for a dataset, the point is that the
three give different arrays under the same key names, so `c6_source` must
travel with the data.

| `c6_source` | $\alpha^A(iu)$ | Molecular $\alpha(iu)$ | Default partition |
|---|---|---|---|
| `ts` (default) | Tkatchenko–Scheffler single pole, shaped by the static $\alpha^A$ | sum of the atoms | `becke` |
| `pdep` | (@eq:ml-alpha-per-atom-kernel) at each node | (@eq:polarizability-pdep-c6-1) at each node | `hirshfeld` |
| `mbd` | the MBD screened model | the MBD model | `becke` |

The `ts` model, for atom $A$ with free-atom values $\alpha^{\mathrm{free}}$ and
$C_6^{\mathrm{free}}$ and Hirshfeld volume ratio $v = V_A/V_A^{\mathrm{free}}$,
is

$$ \bar\alpha_A = v\,\alpha^{\mathrm{free}},\quad C_{6,A} = v^2 C_6^{\mathrm{free}},\quad \omega_A = \frac{4}{3}\frac{C_{6,A}}{\bar\alpha_A^2},\qquad \alpha^A_{cd}(iu) = \frac{\bar\alpha_A}{1+(u/\omega_A)^2}\,\frac{\alpha^{A,\mathrm{static}}_{cd}}{\tfrac13\mathrm{tr}\,\alpha^{A,\mathrm{static}}} . \label{eq:ml-c6-ts-pole} $$

The magnitude $\bar\alpha_A$ and the pole $\omega_A$ come from the free atom
scaled by the volume ratio; the computed static per-atom tensor contributes
only its shape, normalized to unit trace over three. At $u = 0$ the trace
over three of $\alpha^A$ is therefore $\bar\alpha_A$, not the computed
static value.

Four rules govern the inputs. The volume ratios always come from Hirshfeld
volumes, whatever `c6_partition` says; under `ts` the partition controls
only the shape tensor $\alpha^{A,\mathrm{static}}$. The free-atom volume comes from a live
free-atom SCF, and if that SCF fails the C6 block is skipped with a warning.
Elements beyond Z = 54 have no TS reference and are an error. TS has no
charge transfer, so `alpha_ct_dynamic` is written only for `pdep`.

> **Verified equation @eq:ml-c6-ts-pole** · the single-pole frequency
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

## Casimir–Polder $C_6$ {#polarizability-pdep-c6-casimir-polder}

Section @sec:casimir-polder-c6-casimir-polder-integral derives the Casimir–Polder integral; the code replaces it by a
sum over the run's RPA frequency quadrature. The nodes $u_k$ and weights
$w_k$ are by default 20-point Gauss–Legendre with $u_0 = 0.5$, so changing
`[rpa] n_quad` changes `c6_freqs`, and two datasets with different `n_quad`
have arrays of different length under the same key. The pair and molecular
coefficients are

$$ C_6^{AB} = \frac{3}{\pi}\sum_k w_k\,\bar\alpha^A(iu_k)\,\bar\alpha^B(iu_k),\qquad C_{6,\mathrm{mol}} = \frac{3}{\pi}\sum_k w_k\,\bar\alpha(iu_k)^2, \label{eq:ml-c6-cp-quadrature} $$

with $\bar\alpha = \tfrac13\mathrm{tr}\,\alpha$.

> **Verified equation @eq:ml-c6-cp-quadrature** · the Casimir–Polder quadrature
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

1. **`c6_iso.sum()` is not the molecular $C_6$.** Sum (@eq:ml-c6-cp-quadrature) over all
   $N^2$ entries and pull the sums inside:
   $\sum_{AB}C_6^{AB} = \tfrac3\pi\sum_k w_k\big(\sum_A\bar\alpha^A(iu_k)\big)^2$.
   That is the $C_6$ of $\sum_A\alpha^A$, which equals the molecular value
   only when $\bar\alpha_{\mathrm{CT}}(iu_k) = 0$ at every node. Use
   `c6_molecular_iso`. A committed test bounds how far the pair sum strays
   from the molecular value for water
   (`ferric-rpa/tests/s9_per_atom_c6_consistency.rs::bounded_divergence_pair_sum_vs_molecular_c6_water`).
2. **`c6_aniso` is a convention, not a tensor.** The entry
   $\tfrac3\pi\sum_k w_k\alpha^A_{cd}\alpha^B_{cd}$ is an element-wise product.
   It does not transform as a rank-2 tensor under rotation, so an equivariant
   model should be fed the dynamic tensors `alpha_atomic_dynamic` instead.
3. **Pair $C_6$ depends on the partition.** Becke and Hirshfeld give
   different `c6_iso`; record `c6_partition` and `c6_source` with the data.

## PDEP eigenvectors {#polarizability-pdep-c6-pdep-eigenvectors}

The PDEP modes are the eigenvectors of the static dielectric matrix, ordered
by eigenvalue, which measures how strongly each mode is screened (section
@sec:polarizabilities-from-dielectric-modes-mode-form). The file keeps only the leading modes, which makes them a compact
feature. They also carry the gauge freedom of any eigenvector, and a
consumer has to remove it.

`pdep_eigenvectors` holds $M$ columns $c_\xi$ over the auxiliary basis. Each
is the coefficient vector of a mode charge density
$\rho_\xi(\mathbf r) = \sum_P c_{P\xi}\chi_P(\mathbf r)$, whose Coulomb
potential is the PDEP eigenpotential. With $V$ the auxiliary Coulomb metric
and $\Pi$ the static response matrix in the same basis, the columns satisfy

$$ c^{\mathsf T}Vc = I, \qquad c^{\mathsf T}(V+\Pi)\,c \ \text{diagonal}. $$

The eigenvalues are on the Python result, not in the file. $M$ is set by the
`[rpa] trunc_thresh` cut on the spectrum, so it varies with the molecule and
the setting. Each column's sign is arbitrary, and so is the basis inside a
degenerate eigenspace: $c \to cU$ with $U$ orthogonal inside such a space
leaves both conditions above unchanged. The projector $cc^{\mathsf T}$ does
not change under $c \to cU$, which is why the validation compares the
projector with PySCF's (2.6e-7) and checks the metric condition (4.3e-12),
and does not compare columns. A model should consume the projector, or a
sign- and rotation-invariant function of the columns, never the raw columns.

## How well these are checked {#polarizability-pdep-c6-well-these-checked}

| Quantity | Agreement (same orbitals; own SCF) | Test |
|---|---|---|
| static α | 4.8e-14 rel.; 5.3e-10 rel. | `ferric-rpa/tests/validation_static_alpha.rs::static_alpha_h2o_aug_cc_pvdz_vs_numpy` |
| α(iu), molecular $C_6$ | 4.9e-14, 6.2e-14 rel. | `ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_molecular_h2o_aug_cc_pvdz_vs_numpy` |
| per-atom α (Becke) | 3.4e-14 rel. | `ferric-rpa/tests/validation_pdep_c6.rs::pdep_c6_per_atom_becke_h2o_aug_cc_pvdz_vs_numpy` |
| NPZ α against a numpy dRPA solve | 9.2e-11 rel. | `crates/ferric-cli/tests/validation_npz.rs` |

The references are numpy sums over Casida excitations on PySCF integrals in
the same auxiliary basis (`site/src/reference/validation.md`, "Anchors").
That is an independent route: it diagonalizes the response instead of
solving the dielectric system. These checks say that ferric computes (@eq:polarizability-pdep-c6-1)
and (@eq:ml-alpha-per-atom-kernel). They do not say how close a direct-RPA polarizability is to
experiment.

## Known limits (2026-11-15)

- The Hirshfeld per-atom α still integrates on the axis-aligned lattice, so
  it depends on the molecule's orientation. Hirshfeld charges moved off such a
  lattice to an atom-centred grid; the per-atom α has not.
- On an open-shell reference, `pdep_dynamic_polarizability` falls back from
  the Hirshfeld to the Becke partition while the label still says
  `hirshfeld` (section @sec:npz-bundle).

## Sources

| Source | Used for |
|---|---|
| section @sec:polarizabilities-from-dielectric-modes-dielectric-form | the notation of (@eq:polarizability-pdep-c6-1)–(@eq:ml-c6-cp-quadrature) |
| `scripts/rpa_c6_polarizability_sympy.py`, claims 1, 2, 3, 5 | verified equations @eq:ml-alpha-per-atom-kernel–@eq:ml-c6-cp-quadrature |
| `crates/ferric-rpa/src/properties/per_atom.rs`, `crates/ferric-rpa/src/dispersion.rs`, `crates/ferric-rpa/src/property_bundle.rs` | grids, defaults, fallbacks, the quadrature source |
| `crates/ferric-cli/tests/validation_npz.rs` | the PDEP eigenvector conditions |
| `site/src/reference/validation.md` | the figures in @sec:polarizability-pdep-c6-per-atom-split-charge and @sec:polarizability-pdep-c6-well-these-checked |

Changes from the sources: the theory text writes the imaginary frequency as
$\omega$ and the PDEP mode index as $\alpha$; this section uses $u$ and $\xi$
(notation). The pair-sum identity in @sec:polarizability-pdep-c6-casimir-polder and the invariance of the
projector in @sec:polarizability-pdep-c6-pdep-eigenvectors are new here; the first was checked with numpy on random
inputs, the second follows from $UU^{\mathsf T} = I$.
